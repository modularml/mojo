# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""TFLOPS benchmark for the Apple M5 MMA flash-attention prefill kernel.

Warmup + hot loops with a per-iteration `ctx.synchronize()` and a host
`perf_counter`, so each launch is measured in isolation (single-kernel latency),
not pipelined throughput. This matches TTFT, a single-launch latency: an
unsynchronized timer lets Metal overlap consecutive launches, which rewards
under-occupied configs and inverts the true per-launch ranking.

Runtime knob honored by the launcher (env, no recompile):
  `MODULAR_APPLE_FA_PREFILL_NUM_SIMDGROUPS`.
Shape knobs via `-D` (place before the filename): `seq`, `causal`, `batch`,
`depth`, `heads`, `kv_heads`, `num_simdgroups`. With `seq` unset the default
sweep runs; with `seq` set, one (batch, seq) shape runs.

Run:
  mojo -D seq=8192 -D causal=true \
    max/kernels/benchmarks/gpu/nn/bench_apple_fa_prefill.mojo
"""

from std.collections import OptionalReg
from std.os import getenv
from std.sys import get_defined_bool, get_defined_int
from std.sys.info import _accelerator_arch
from std.time import perf_counter

from max.gpu.host import DeviceContext
from std.math import sqrt

from layout import (
    UNKNOWN_VALUE,
    Idx,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)

from nn.attention.mha_mask import CausalMask, NullMask, MHAMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.apple.fa_prefill import fa_prefill_apple
from nn.attention.gpu.mha import mha_gpu_naive


def _mask_label[mask_t: MHAMask]() -> String:
    comptime if mask_t == CausalMask:
        return "causal"
    else:
        return "null"


# ===-------------------------------------------------------------------=== #
# One shape, warmup + hot timing (per-iter synchronize -> isolated latency).
# ===-------------------------------------------------------------------=== #
def _bench_prefill[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    kv_heads: Int,
    mask_t: MHAMask,
    naive: Bool = False,
    num_simdgroups: Int = 4,
](
    mask: mask_t,
    batch: Int,
    seq: Int,
    num_keys: Int,
    ctx: DeviceContext,
    warmup: Int = 10,
    hot: Int = 20,
) raises -> Float64:
    comptime group = num_heads // kv_heads
    var scale = Float32(1) / sqrt(Float32(depth))

    var q_n = batch * seq * num_heads * depth
    var k_n = batch * num_keys * kv_heads * depth
    var o_n = q_n

    # Reused device buffers (no cache-busting): the per-iteration synchronize
    # isolates each launch.
    var q_d = ctx.enqueue_create_buffer[qkv_type](q_n)
    var k_d = ctx.enqueue_create_buffer[qkv_type](k_n)
    var v_d = ctx.enqueue_create_buffer[qkv_type](k_n)
    var o_d = ctx.enqueue_create_buffer[qkv_type](o_n)
    q_d.enqueue_fill(Scalar[qkv_type](0.1))
    k_d.enqueue_fill(Scalar[qkv_type](0.1))
    v_d.enqueue_fill(Scalar[qkv_type](0.1))

    var q_t = TileTensor(q_d, row_major(batch, seq, Idx[num_heads], Idx[depth]))
    var k_t = TileTensor(
        k_d, row_major(batch, num_keys, Idx[kv_heads], Idx[depth])
    )
    var v_t = TileTensor(
        v_d, row_major(batch, num_keys, Idx[kv_heads], Idx[depth])
    )
    var o_t = TileTensor(o_d, row_major(batch, seq, Idx[num_heads], Idx[depth]))
    var k_op = LayoutTensorMHAOperand(k_t)
    var v_op = LayoutTensorMHAOperand(v_t)

    # Dummy valid_length (dense path doesn't read it).
    var vl_d = ctx.enqueue_create_buffer[.uint32](batch + 1)
    var vl_t = TileTensor(vl_d, row_major(batch + 1))

    comptime SinkOpt = OptionalReg[
        LayoutTensor[qkv_type, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ]
    var sink_opt = SinkOpt(None)

    @__parameter
    @always_inline
    @__copy_capture(
        q_t, k_t, v_t, o_t, k_op, v_op, vl_t, sink_opt, scale, seq, num_keys
    )
    def _launch() raises:
        comptime if naive:
            mha_gpu_naive[
                ragged=False,
                _use_valid_length=False,
                _is_cache_length_accurate=True,
            ](
                q_t.to_layout_tensor(),
                k_op,
                v_op,
                mask,
                o_t.to_layout_tensor(),
                vl_t.to_layout_tensor(),
                scale,
                batch,
                seq,
                num_keys,
                num_heads,
                depth,
                group,
                ctx,
                sink_opt,
            )
        else:
            fa_prefill_apple[
                ragged=False,
                _use_valid_length=False,
                _is_cache_length_accurate=True,
                num_simdgroups=num_simdgroups,
            ](
                q_t.to_layout_tensor(),
                k_op,
                v_op,
                mask,
                o_t.to_layout_tensor(),
                vl_t.to_layout_tensor(),
                scale,
                batch,
                seq,
                num_keys,
                num_heads,
                depth,
                group,
                ctx,
                sink_opt,
            )

    # Warmup runs (untimed).
    for _ in range(warmup):
        _launch()
        ctx.synchronize()

    # Hot runs (timed).
    var start = perf_counter()
    for _ in range(hot):
        _launch()
        ctx.synchronize()
    var elapsed = perf_counter() - start

    var avg_sec = elapsed / Float64(hot)
    # FLOP count 2*B*H*seq*num_keys*depth. For causal the tile-skip makes this
    # EFFECTIVE throughput (~2x the actual MMA rate); NullMask is the true rate.
    var flops = (
        2.0
        * Float64(batch)
        * Float64(num_heads)
        * Float64(seq)
        * Float64(num_keys)
        * Float64(depth)
    )
    var tflops = flops / (avg_sec * 1e12)

    # Print the env-effective simdgroup count (the launcher resolves it at
    # dispatch), so each line is honest about what actually ran.
    var sg_env = getenv("MODULAR_APPLE_FA_PREFILL_NUM_SIMDGROUPS", "")
    var eff_sg = sg_env if sg_env != "" else String(num_simdgroups)

    comptime label = "mha_gpu_naive" if naive else "fa_prefill_apple"
    print(
        " ",
        label,
        _mask_label[mask_t](),
        " b=",
        batch,
        "seq=",
        seq,
        "d=",
        depth,
        "sg=",
        eff_sg,
        ": avg",
        avg_sec * 1000.0,
        "ms,",
        tflops,
        "TFLOPS",
    )

    # Keep buffers alive until timing is done.
    _ = q_d^
    _ = k_d^
    _ = v_d^
    _ = o_d^
    _ = vl_d^
    return avg_sec


def main() raises:
    comptime if "metal" not in _accelerator_arch():
        print("SKIP: Apple GPU required")
        return
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 required (compute_capability == 5)")
        return

    comptime qkv = DType.bfloat16
    comptime d = get_defined_int["depth", 128]()
    comptime nh = get_defined_int["heads", 32]()
    comptime kvh = get_defined_int["kv_heads", 8]()
    comptime user_seq = get_defined_int["seq", 0]()
    comptime user_batch = get_defined_int["batch", 1]()
    comptime causal = get_defined_bool["causal", True]()
    comptime nsg = get_defined_int["num_simdgroups", 4]()

    print(
        (
            "== bench_apple_fa_prefill (warmup=10, hot=20, per-iter sync;"
            " sg default="
        ),
        nsg,
        "; env overrides honored)",
    )

    comptime if user_seq > 0:
        # One user-specified shape (mask selected by `causal`, default True).
        comptime if causal:
            _ = _bench_prefill[qkv, d, nh, kvh, CausalMask, num_simdgroups=nsg](
                CausalMask(), user_batch, user_seq, user_seq, ctx
            )
        else:
            _ = _bench_prefill[qkv, d, nh, kvh, NullMask, num_simdgroups=nsg](
                NullMask(), user_batch, user_seq, user_seq, ctx
            )
    else:
        # Default sweep: fa_prefill_apple vs mha_gpu_naive (causal) at small
        # seqs (naive is O(seq^2), so cap it short), then fa_prefill_apple
        # throughput across the seq range, causal + NullMask, b1.
        var small = [512, 1024, 2048]
        for i in range(len(small)):
            _ = _bench_prefill[qkv, d, nh, kvh, CausalMask, num_simdgroups=nsg](
                CausalMask(), 1, small[i], small[i], ctx
            )
            _ = _bench_prefill[
                qkv, d, nh, kvh, CausalMask, naive=True, num_simdgroups=nsg
            ](CausalMask(), 1, small[i], small[i], ctx)
        var seqs = [512, 1024, 2048, 4096, 8192, 16384]
        for i in range(len(seqs)):
            _ = _bench_prefill[qkv, d, nh, kvh, CausalMask, num_simdgroups=nsg](
                CausalMask(), 1, seqs[i], seqs[i], ctx
            )
            _ = _bench_prefill[qkv, d, nh, kvh, NullMask, num_simdgroups=nsg](
                NullMask(), 1, seqs[i], seqs[i], ctx
            )
