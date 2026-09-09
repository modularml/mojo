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
"""End-to-end correctness test for the Apple M5 MMA-based FA2-prefill kernels.

Mirrors `test/gpu/nn/test_flash_attention.mojo` (the e2e host-reference pattern)
for the Apple prefill path: build dense BSHD Q/K/V, run a prefill kernel directly
against a `LayoutTensorMHAOperand`, and compare against an fp32 host attention
reference. Covers `NullMask` + `CausalMask` + `SlidingWindowCausalMask`, fp16/bf16
in, head dims that are multiples of 16 up to 256, and ragged M/N (seq_len /
num_keys not multiples of the tile size, to exercise the bounded edges). PASS on
M5 gates increment 3 of the prefill bring-up (DESIGN.md).

`_run` drives the single `fa_prefill_apple` kernel (the wide-threadgroup no-SMEM
prefill, `num_simdgroups=16`) against the fp32 reference for every shape/mask.
"""

from std.collections import OptionalReg
from max.gpu.host import DeviceContext
from std.math import exp, isinf, isnan, sqrt
from std.random import rand, seed
from std.sys.info import _accelerator_arch

from layout import (
    UNKNOWN_VALUE,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
)
from layout.tile_layout import row_major

from nn.attention.gpu.apple.fa_prefill import fa_prefill_apple
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    NullMask,
    SlidingWindowCausalMask,
)
from nn.attention.mha_operand import LayoutTensorMHAOperand

from std.utils.index import Index


# ===-------------------------------------------------------------------=== #
# fp32 host attention reference (BSHD, batch=1 path generalized over batch).
# ===-------------------------------------------------------------------=== #
def _host_attention[
    mask_kind: Int,  # 0 = null, 1 = causal, 2 = sliding-window
    window: Int = 0,
    use_sink: Bool = False,
](
    q: List[Float32],  # [batch, seq, num_heads, depth]
    k: List[Float32],  # [batch, num_keys, kv_heads, depth]
    v: List[Float32],
    out_zero: List[Float32],  # written: [batch, seq, num_heads, depth]
    sink_w: List[Float32],  # per-head raw sink weights (used iff use_sink)
    batch: Int,
    seq: Int,
    num_keys: Int,
    num_heads: Int,
    kv_heads: Int,
    depth: Int,
    scale: Float32,
) -> List[Float32]:
    var result = out_zero.copy()
    var group = num_heads // kv_heads
    for b in range(batch):
        for h in range(num_heads):
            var kvh = h // group
            for qi in range(seq):
                # Score row (no prior cache here: cache_len == seq).
                var scores = List[Float32](length=num_keys, fill=0)
                var m = Float32(-3.0e38)
                for ki in range(num_keys):
                    var dot = Float32(0)
                    for d in range(depth):
                        var qv = q[((b * seq + qi) * num_heads + h) * depth + d]
                        var kv = k[
                            ((b * num_keys + ki) * kv_heads + kvh) * depth + d
                        ]
                        dot += qv * kv
                    var s = dot * scale
                    # Mask: causal -> qi (absolute) >= ki visible; sliding ->
                    # additionally ki > qi - window.
                    var visible = True
                    comptime if mask_kind == 1:
                        visible = ki <= qi
                    elif mask_kind == 2:
                        visible = ki <= qi and ki > qi - window
                    if not visible:
                        s = Float32(-3.0e38)
                    scores[ki] = s
                    m = max(m, s)
                # Sink: raw (unscaled) sink weight participates in row_max and
                # the denominator (mirrors nn/softmax.mojo); it adds no value.
                comptime if use_sink:
                    m = max(m, sink_w[h])
                var l = Float32(0)
                for ki in range(num_keys):
                    scores[ki] = exp(scores[ki] - m)
                    l += scores[ki]
                comptime if use_sink:
                    l += exp(sink_w[h] - m)
                for d in range(depth):
                    var acc = Float32(0)
                    for ki in range(num_keys):
                        acc += (
                            scores[ki]
                            * v[
                                ((b * num_keys + ki) * kv_heads + kvh) * depth
                                + d
                            ]
                        )
                    # A fully-masked row (l==0) cannot occur for causal because
                    # ki==qi is always visible; guard anyway.
                    var denom = l if l > 0 else Float32(1)
                    result[((b * seq + qi) * num_heads + h) * depth + d] = (
                        acc / denom
                    )
    return result^


def _run[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    kv_heads: Int,
    mask_t: MHAMask,
    mask_kind: Int,
    window: Int = 0,
    use_sink: Bool = False,
    # When True, allocate `OOB_PAD` extra K/V rows past `num_keys` and fill them
    # with NaN, but present only `num_keys` rows to the kernel/reference. This
    # deterministically reproduces the OOB-V-poison bug: the PV path read V rows
    # past num_keys, and `P_oob(0) * V_oob(NaN) = NaN` poisoned the fp32
    # accumulator. The fix bounds the V fragment load to zero-fill OOB rows.
    # Without the explicit NaN poison the bug is allocation-layout-dependent (the
    # adjacent allocation may happen to be finite), so the regression must poison
    # the tail to be reliable. batch=1 only.
    oob_poison: Bool = False,
](
    mask: mask_t,
    batch: Int,
    seq: Int,
    num_keys: Int,
    ctx: DeviceContext,
    scale_override: Float32 = -1.0,
) raises:
    print(
        "  prefill",
        "dtype:",
        qkv_type,
        "depth:",
        depth,
        "heads:",
        num_heads,
        "kv_heads:",
        kv_heads,
        "batch:",
        batch,
        "seq:",
        seq,
        "num_keys:",
        num_keys,
        "mask_kind:",
        mask_kind,
        "sink:",
        use_sink,
    )
    var group = num_heads // kv_heads
    var scale = scale_override if scale_override >= Float32(0.0) else Float32(
        1.0
    ) / sqrt(Float32(depth))

    # Per-head raw sink weights (only used when use_sink). Large values so a
    # dropped sink seed clearly fails (the shared sink test uses 5.0 / 3.0).
    var sink_w = List([Float32(0)]) * num_heads
    for h in range(num_heads):
        sink_w[h] = Float32(5.0) - Float32(h) * 0.7

    var q_n = batch * seq * num_heads * depth
    var k_n = batch * num_keys * kv_heads * depth
    var o_n = q_n

    # Host fp32 master data (so the reference and the device input agree).
    var q_f = List([Float32(0)]) * q_n
    var k_f = List([Float32(0)]) * k_n
    var v_f = List([Float32(0)]) * k_n
    var o_zero = List([Float32(0)]) * o_n
    seed(123)
    for i in range(q_n):
        q_f[i] = Float32(((i * 53 + 17) % 197) - 98) * 0.01
    for i in range(k_n):
        k_f[i] = Float32(((i * 31 + 7) % 211) - 105) * 0.01
        v_f[i] = Float32(((i * 91 + 13) % 173) - 86) * 0.01

    var ref_out = _host_attention[mask_kind, window, use_sink](
        q_f,
        k_f,
        v_f,
        o_zero,
        sink_w,
        batch,
        seq,
        num_keys,
        num_heads,
        kv_heads,
        depth,
        scale,
    )

    # Device buffers (cast to qkv_type for the kernel).
    var q_h = ctx.enqueue_create_host_buffer[qkv_type](q_n)
    var k_h = ctx.enqueue_create_host_buffer[qkv_type](k_n)
    var v_h = ctx.enqueue_create_host_buffer[qkv_type](k_n)
    for i in range(q_n):
        q_h[i] = Scalar[qkv_type](q_f[i])
    for i in range(k_n):
        k_h[i] = Scalar[qkv_type](k_f[i])
        v_h[i] = Scalar[qkv_type](v_f[i])

    # OOB poison: append OOB_PAD NaN-filled key rows after the logical num_keys
    # rows so the kernel's read past num_keys lands on NaN (deterministic
    # reproduction of the OOB-V-poison bug). Poison cases are batch=1 so the
    # pad sits immediately after key num_keys in the row-major layout.
    comptime OOB_PAD = 64
    var pad_n = (OOB_PAD * kv_heads * depth) if oob_poison else 0
    if oob_poison and batch != 1:
        raise Error("oob_poison regression cases must use batch=1")

    var q_d = ctx.enqueue_create_buffer[qkv_type](q_n)
    var k_d = ctx.enqueue_create_buffer[qkv_type](k_n + pad_n)
    var v_d = ctx.enqueue_create_buffer[qkv_type](k_n + pad_n)
    var o_d = ctx.enqueue_create_buffer[qkv_type](o_n)
    ctx.enqueue_copy(q_d, q_h)
    # Copy the logical k_n elements into the front of the (possibly padded) KV
    # device buffers. With a pad we must copy into a k_n-sized sub-buffer so the
    # host->device sizes match.
    comptime if oob_poison:
        ctx.enqueue_copy(k_d.create_sub_buffer[qkv_type](0, k_n), k_h)
        ctx.enqueue_copy(v_d.create_sub_buffer[qkv_type](0, k_n), v_h)
        var nan_pad = ctx.enqueue_create_host_buffer[qkv_type](pad_n)
        var nan_val = Scalar[qkv_type](0) / Scalar[qkv_type](0)
        for i in range(pad_n):
            nan_pad[i] = nan_val
        # Write the NaN pad into [k_n, k_n + pad_n) of the K and V device
        # buffers (keys num_keys..num_keys+OOB_PAD for batch 0).
        ctx.enqueue_copy(k_d.create_sub_buffer[qkv_type](k_n, pad_n), nan_pad)
        ctx.enqueue_copy(v_d.create_sub_buffer[qkv_type](k_n, pad_n), nan_pad)
        ctx.synchronize()
        _ = nan_pad^
    else:
        ctx.enqueue_copy(k_d, k_h)
        ctx.enqueue_copy(v_d, v_h)

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

    # Sink weights device tensor (only consumed when use_sink).
    var sink_d = ctx.enqueue_create_buffer[qkv_type](num_heads)
    var sink_h = ctx.enqueue_create_host_buffer[qkv_type](num_heads)
    for h in range(num_heads):
        sink_h[h] = Scalar[qkv_type](sink_w[h])
    ctx.enqueue_copy(sink_d, sink_h)
    comptime sinks_layout = Layout.row_major(UNKNOWN_VALUE)
    comptime SinkOpt = OptionalReg[
        LayoutTensor[qkv_type, sinks_layout, ImmutAnyOrigin]
    ]
    var sink_opt: SinkOpt
    comptime if use_sink:
        sink_opt = SinkOpt(
            LayoutTensor[qkv_type, sinks_layout](
                sink_d.unsafe_ptr(),
                RuntimeLayout[sinks_layout].row_major(Index(num_heads)),
            )
            .as_imm()
            .as_unsafe_any_origin()
        )
    else:
        sink_opt = SinkOpt(None)

    fa_prefill_apple[
        ragged=False,
        sink=use_sink,
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
        seq,  # max_prompt_len
        num_keys,  # max_cache_size
        num_heads,
        depth,
        group,
        ctx,
        sink_opt,
    )
    _ = sink_d^

    var o_h = ctx.enqueue_create_host_buffer[qkv_type](o_n)
    ctx.enqueue_copy(o_h, o_d)
    ctx.synchronize()
    _ = q_d^
    _ = k_d^
    _ = v_d^
    _ = o_d^
    _ = vl_d^

    # Tolerance: bf16/fp16 inputs with fp32 accum; loosen for bf16.
    var atol = Float32(2e-2) if qkv_type == DType.bfloat16 else Float32(8e-3)
    var pass_ = True
    var max_err = Float32(0)
    for i in range(o_n):
        var got = Float32(o_h[i])
        var exp_v = ref_out[i]
        # Explicit NaN/Inf check: `abs(NaN - x) > atol` is FALSE (every NaN
        # comparison is), so a NaN output would otherwise slip through as a
        # silent PASS. The OOB-V-poison bug produced exactly such NaN outputs.
        if isnan(got) or isinf(got):
            if pass_:
                print("    FAIL idx", i, "got NaN/Inf", got, "exp", exp_v)
            pass_ = False
            continue
        var err = abs(got - exp_v)
        max_err = max(max_err, err)
        if err > atol * (1.0 + abs(exp_v)):
            if pass_:  # print only the first few
                print(
                    "    FAIL idx",
                    i,
                    "got",
                    got,
                    "exp",
                    exp_v,
                    "err",
                    err,
                )
            pass_ = False
    if not pass_:
        raise Error("FAILED (max_err=", max_err, ")")
    print("    PASS (max_err=", max_err, ")")


def _cases(ctx: DeviceContext) raises:
    """All `fa_prefill_apple` correctness cases vs the fp32 host reference."""
    print("== test_apple_fa_prefill (MMA prefill vs fp32 host attention)")

    # --- NullMask (full attention): aligned + ragged M/N, fp16 + bf16. ---
    _run[.float16, 64, 4, 4, NullMask, 0](NullMask(), 1, 32, 32, ctx)
    _run[.bfloat16, 64, 4, 4, NullMask, 0](NullMask(), 1, 32, 32, ctx)
    # ragged M (seq=20) and N (num_keys=37): exercises bounded edges.
    _run[.float16, 64, 4, 4, NullMask, 0](NullMask(), 1, 20, 37, ctx)
    _run[.bfloat16, 32, 2, 2, NullMask, 0](NullMask(), 1, 17, 29, ctx)

    # --- depth sweep: multiples of 16 up to 256. ---
    _run[.float16, 16, 2, 2, NullMask, 0](NullMask(), 1, 32, 48, ctx)
    _run[.float16, 128, 2, 2, NullMask, 0](NullMask(), 1, 33, 33, ctx)
    _run[.float16, 256, 2, 2, NullMask, 0](NullMask(), 1, 18, 50, ctx)

    # --- OOB-V-poison regression (MOCO): NullMask cross-attention with
    # num_keys < seq, the last KV sub-tile reading V rows past num_keys. The
    # `oob_poison=True` cases fill those OOB rows with NaN, so without the
    # bounded V fragment load the PV `0 * NaN = NaN` poisons every output
    # element. num_keys deliberately NOT a multiple of 16 (so the last sub-tile
    # is partial). Group ratios 1/3/4, depths 64/128, fp16 + bf16. Mirrors the
    # `test_flash_attention.mojo` configs that previously NaN'd
    # (seq=1024/keys=100 group=3 d=128; seq=512/keys=37). ---
    _run[.bfloat16, 128, 24, 8, NullMask, 0, oob_poison=True](
        NullMask(), 1, 1024, 100, ctx
    )
    _run[.float16, 128, 24, 8, NullMask, 0, oob_poison=True](
        NullMask(), 1, 512, 37, ctx
    )
    # group=1 (num_heads == kv_heads).
    _run[.bfloat16, 64, 4, 4, NullMask, 0, oob_poison=True](
        NullMask(), 1, 256, 100, ctx
    )
    # group=4 (num_heads=4, kv_heads=1).
    _run[.float16, 64, 4, 1, NullMask, 0, oob_poison=True](
        NullMask(), 1, 256, 37, ctx
    )
    # Tiny key set (num_keys=5 << seq): entire valid KV in the first sub-tile,
    # rows 5..15 of sub-tile 0 plus sub-tiles 1..7 all OOB / NaN.
    _run[.bfloat16, 128, 2, 2, NullMask, 0, oob_poison=True](
        NullMask(), 1, 320, 5, ctx
    )

    # --- CausalMask: aligned + ragged. ---
    _run[.float16, 64, 4, 4, CausalMask, 1](CausalMask(), 1, 32, 32, ctx)
    _run[.bfloat16, 64, 4, 4, CausalMask, 1](CausalMask(), 1, 48, 48, ctx)
    _run[.float16, 128, 2, 2, CausalMask, 1](CausalMask(), 1, 35, 35, ctx)

    # --- CausalMask spanning MULTIPLE SK tiles (SK = NumNMmas*16 = 128 with
    # the default 1x8 tile shape): exercises the upper-triangle tile-skip /
    # monotonic early-exit. seq=300 => 3 KV tiles, so the lowest query tiles
    # skip tiles 1 and 2 (kv0=128, 256) entirely, and a ragged tail
    # (seq=200 => 2 tiles, num_keys ragged) checks the partial diagonal tile
    # plus the bounded edge under a skip. ---
    _run[.float16, 64, 4, 4, CausalMask, 1](CausalMask(), 1, 300, 300, ctx)
    _run[.bfloat16, 64, 2, 2, CausalMask, 1](CausalMask(), 1, 200, 200, ctx)
    _run[.float16, 64, 2, 2, CausalMask, 1](CausalMask(), 1, 257, 257, ctx)

    # --- SlidingWindowCausalMask. ---
    _run[.float16, 64, 2, 2, SlidingWindowCausalMask[16], 2, 16](
        SlidingWindowCausalMask[16](), 1, 48, 48, ctx
    )
    # Small window spanning a score-tile boundary (SK=128): the upper rows of
    # the q-tile straddling key 128 are fully masked in the PARTIAL (processed,
    # not skipped) tile [0,128) and only attend keys once their window opens in
    # the next tile -- exercises a row reaching the softmax fully masked with
    # its running max still at the NEG_INF floor.
    _run[.bfloat16, 64, 2, 2, SlidingWindowCausalMask[8], 2, 8](
        SlidingWindowCausalMask[8](), 1, 256, 256, ctx
    )

    # --- Grouped (GQA: num_heads > kv_heads). ---
    _run[.float16, 64, 8, 2, CausalMask, 1](CausalMask(), 1, 40, 40, ctx)

    # --- Multi-batch. ---
    _run[.float16, 64, 4, 4, CausalMask, 1](CausalMask(), 2, 32, 32, ctx)

    # --- Sink (attention sink as init-state): NullMask + CausalMask. ---
    _run[.float16, 64, 2, 2, NullMask, 0, 0, use_sink=True](
        NullMask(), 1, 32, 32, ctx
    )
    _run[.float16, 64, 4, 4, CausalMask, 1, 0, use_sink=True](
        CausalMask(), 1, 48, 48, ctx
    )
    _run[.bfloat16, 128, 2, 2, NullMask, 0, 0, use_sink=True](
        NullMask(), 1, 33, 40, ctx
    )
    # scale=0 sink case (mirrors test_flash_attention_sink_kernel exactly:
    # all QK logits 0, output == num_keys/(num_keys + exp(sink))).
    _run[.bfloat16, 128, 2, 2, NullMask, 0, 0, use_sink=True](
        NullMask(), 1, 8, 64, ctx, scale_override=0.0
    )

    print("== all prefill cases PASS")


def test_apple_fa_prefill(ctx: DeviceContext) raises:
    _cases(ctx)


def main() raises:
    comptime if "metal" not in _accelerator_arch():
        print("SKIP: Apple GPU required")
        return
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 required (16x16 simdgroup MMA)")
        return
    test_apple_fa_prefill(ctx)
