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
"""TFLOPS bench for `MhaPrefillV2` against contiguous (LayoutTensor) K/V.

Measures `mha_prefill_v2` at canonical BF16 causal shapes. Q/K/V/O are
oversized via `CacheBustingBuffer` and offset per iteration to defeat
L2 reuse. FLOP count uses the causal half-tile budget
(`2 * B * H * N * NK * D`).

Run:
  ./bazelw run //max/kernels/benchmarks:gpu/nn/bench_mha_prefill_v2 \
      -- --seq_len=8192 --num_keys=8192 --batch_size=1 --verify=False
"""

from std.math import isclose
from std.random import seed
from std.sys import get_defined_bool, get_defined_dtype, get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import *
from max.gpu.host import DeviceContext
from std.utils import StaticTuple
from std.utils.numerics import min_or_neg_inf

from internal_utils import CacheBustingBuffer, arg_parse
from internal_utils._utils import InitializationType
from layout import Idx, LayoutTensor, TileTensor, row_major
from layout.coord import Coord
from layout.runtime_layout import RuntimeLayout

from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand

from nn.attention.gpu.amd_structured.mha_prefill_v2 import (
    MhaConfigV2,
    mha_prefill_v2,
)


comptime _Q_BLOCK_SIZE = 32
comptime _NUM_WARPS = 8
comptime _BM = _NUM_WARPS * _Q_BLOCK_SIZE  # 256


def run_mha_prefill_v2[
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    kv_block: Int,
    group: Int = 1,
    cache_busting: Bool = True,
    sink: Bool = False,
](
    mut m: Bench,
    seq_len: Int,
    num_keys: Int,
    batch_size: Int,
    bench: Bool,
    verify: Bool,
    ctx: DeviceContext,
) raises:
    comptime scale = Float32(0.125)  # ~rsqrt(64)
    comptime kv_num_heads = num_heads // group
    comptime _num_threads = _NUM_WARPS * 64

    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size

    # Cache busting: oversize allocation, defeat L2 reuse across iters.
    comptime simd_size = 4
    var cb_q = CacheBustingBuffer[qkv_type](
        q_size, simd_size, ctx, cache_busting
    )
    var cb_k = CacheBustingBuffer[qkv_type](
        k_size, simd_size, ctx, cache_busting
    )
    var cb_v = CacheBustingBuffer[qkv_type](
        v_size, simd_size, ctx, cache_busting
    )
    # Output is FP32 per MhaPrefillV2.attend_ker signature.
    var cb_o = CacheBustingBuffer[.float32](
        o_size, simd_size, ctx, cache_busting
    )

    # Phase-5b sink path: per-q-head scalar weight buffer. Read only
    # when `sink=True` is comptime; otherwise the kernel receives a
    # dangling pointer and the comptime branch elides the load.
    var sw_buf = ctx.enqueue_create_buffer[qkv_type](num_heads)
    comptime if sink:
        ctx.enqueue_memset(sw_buf, 0.05)

    comptime random_distribution = InitializationType.uniform_distribution
    cb_q.init_on_device(random_distribution, ctx)
    cb_k.init_on_device(random_distribution, ctx)
    cb_v.init_on_device(random_distribution, ctx)

    comptime assert qkv_type == .bfloat16, "MhaPrefillV2 is BF16-only"

    comptime _config = MhaConfigV2(
        q_block_size=_Q_BLOCK_SIZE,
        kv_block=kv_block,
        depth=depth,
        num_heads=num_heads,
        num_kv_heads=kv_num_heads,
        num_warps=_NUM_WARPS,
    )

    # Per-kernel LLVM tuning forwarded as `-mllvm` flags via the
    # launcher's `compile_options` param:
    # - `amdgpu-igrouplp-exact-solver=true`: enable exponential solver.
    # - `...-max-branches=10000`: cap branches (avoid exponential blowup).
    # - `...-cost-heur=false`: node-order priority (over-declared hints fit better).
    comptime _PREFILL_IGLP_OPTS: StaticString = (
        "amdgpu-igrouplp-exact-solver=true,"
        "amdgpu-igrouplp-exact-solver-max-branches=10000,"
        "amdgpu-igrouplp-exact-solver-cost-heur=false"
    )

    if bench:

        @always_inline
        def bench_func(
            mut b: Bencher,
        ) raises {var cb_q, var cb_k, var cb_v, var cb_o, imm}:
            @always_inline
            def _kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
                var q_ptr = (
                    cb_q.offset_ptr(iteration)
                    .bitcast[BFloat16]()
                    .as_imm()
                    .as_unsafe_any_origin()
                )
                var k_ptr = (
                    cb_k.offset_ptr(iteration)
                    .bitcast[BFloat16]()
                    .as_imm()
                    .as_unsafe_any_origin()
                )
                var v_ptr = (
                    cb_v.offset_ptr(iteration)
                    .bitcast[BFloat16]()
                    .as_imm()
                    .as_unsafe_any_origin()
                )
                var q_tt = TileTensor(
                    q_ptr,
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(seq_len),
                            Idx[num_heads],
                            Idx[depth],
                        )
                    ),
                )
                var k_tt = TileTensor(
                    k_ptr,
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(num_keys),
                            Idx[kv_num_heads],
                            Idx[depth],
                        )
                    ),
                )
                var v_tt = TileTensor(
                    v_ptr,
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(num_keys),
                            Idx[kv_num_heads],
                            Idx[depth],
                        )
                    ),
                )
                var o_tt = TileTensor(
                    cb_o.offset_ptr(iteration).bitcast[Float32](),
                    row_major(
                        Coord(
                            Int32(batch_size),
                            Int32(seq_len),
                            Idx[num_heads],
                            Idx[depth],
                        )
                    ),
                )
                var k_op = LayoutTensorMHAOperand(
                    k_tt.as_immut().as_unsafe_any_origin()
                )
                var v_op = LayoutTensorMHAOperand(
                    v_tt.as_immut().as_unsafe_any_origin()
                )
                comptime if sink:
                    # Launcher infers `sink_weights_ptr`'s dtype from
                    # `q.dtype` (literal BF16 here), so the cast must
                    # land on `BFloat16` — generic
                    # `Scalar[qkv_type]` won't unify even when equal.
                    var sw_ptr = (
                        sw_buf.unsafe_ptr()
                        .bitcast[BFloat16]()
                        .as_unsafe_any_origin()
                        .as_imm()
                    )
                    mha_prefill_v2[
                        _config,
                        sink=True,
                        compile_options=_PREFILL_IGLP_OPTS,
                    ](
                        q_tt,
                        k_op,
                        v_op,
                        o_tt,
                        CausalMask(),
                        scale,
                        num_keys,
                        0,  # start_pos
                        ctx,
                        sw_ptr,
                    )
                else:
                    mha_prefill_v2[_config, compile_options=_PREFILL_IGLP_OPTS](
                        q_tt,
                        k_op,
                        v_op,
                        o_tt,
                        CausalMask(),
                        scale,
                        num_keys,
                        0,  # start_pos
                        ctx,
                    )

            bencher_iter_custom(b, _kernel_launch, ctx)

        def compute_flops() {imm} -> Int:
            # Causal-mask: half the tiles. Matches bench_hk_mha_exact's
            # formula (`2 * B * H * N * NK * D`).
            return 2 * batch_size * num_heads * seq_len * num_keys * depth

        m.bench_function(
            bench_func,
            BenchId(
                "mha_prefill_v2",
                # fmt: off
                input_id=String(
                    "qkv_type=", qkv_type,
                    "/depth=", depth,
                    "/num_heads=", num_heads,
                    "/group=", group,
                    "/seq_len=", seq_len,
                    "/num_keys=", num_keys,
                    "/batch_size=", batch_size,
                    "/cache_busting=", cache_busting,
                ),
                # fmt: on
            ),
            [ThroughputMeasure(BenchMetric.flops, compute_flops())],
        )
        ctx.synchronize()

    # Verification: compare a single launch (zero offset) against
    # NOTE: the previous `verify` path compared against `flash_attention_hk_exact`
    # (v1) but that dependency is intentionally out of scope here. For
    # correctness use `test_mha_prefill_v2*` in tests/gpu/structured_kernels.
    _ = verify

    _ = cb_q
    _ = cb_k
    _ = cb_v
    _ = cb_o


@fieldwise_init
struct MhaPrefillV2Cfg(ImplicitlyCopyable, Writable):
    var qkv_type: DType
    var mask_type: DType
    var depth: Int
    var num_heads: Int
    var kv_block: Int
    var group: Int
    var cache_busting: Bool


def main() raises:
    # Fixed seed so random Q/K/V fills are reproducible across runs.
    seed(0)

    comptime qkv_type = get_defined_dtype["qkv_type", .bfloat16]()
    comptime mask_type = get_defined_dtype["mask_type", .bfloat16]()
    comptime depth = get_defined_int["depth", 128]()
    comptime num_heads = get_defined_int["num_heads", 16]()
    comptime kv_block = get_defined_int["kv_block", 64]()
    comptime group = get_defined_int["group", 1]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()
    comptime sink = get_defined_bool["sink", False]()

    var seq_len = Int(arg_parse("seq_len", 8192))
    var num_keys = Int(arg_parse("num_keys", 8192))
    var batch_size = Int(arg_parse("batch_size", 1))
    var bench = arg_parse("benchmark", True)
    var verify = arg_parse("verify", False)

    comptime cfg = MhaPrefillV2Cfg(
        qkv_type=qkv_type,
        mask_type=mask_type,
        depth=depth,
        num_heads=num_heads,
        kv_block=kv_block,
        group=group,
        cache_busting=cache_busting,
    )

    print("Running MhaPrefillV2 benchmark with config:")
    print("  qkv_type:", cfg.qkv_type)
    print("  depth:", cfg.depth)
    print("  num_heads:", cfg.num_heads)
    print("  kv_block:", cfg.kv_block)
    print("  group:", cfg.group)
    print("  cache_busting:", cfg.cache_busting)
    print("  seq_len:", seq_len, " num_keys:", num_keys)
    print("  batch_size:", batch_size, " verify:", verify)

    var m = Bench()
    with DeviceContext() as ctx:
        run_mha_prefill_v2[
            cfg.qkv_type,
            cfg.mask_type,
            cfg.depth,
            cfg.num_heads,
            cfg.kv_block,
            cfg.group,
            cfg.cache_busting,
            sink=sink,
        ](
            m,
            seq_len,
            num_keys,
            batch_size,
            bench,
            verify,
            ctx,
        )
    m.dump_report()
