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
"""KS64: `fused_silu_mx` direct `scale_4d` emission.

The MXFP4 down-projection grouped matmul consumes the activation E8M0 scale in
the per-expert fixed-stride `scale_4d` slot layout (`Shuffler.scale_4d_byte_off`).
Today that layout is produced by a standalone kernel
(`preshuffle_grouped_scale_4d_gpu`, traced
`block_scaled_preshuffle_grouped_scale_4d_kernel_KS64`) that runs *serially*
after `fused_silu_mx` writes the scale row-major. The
fusion (KS64/down-proj) has `fused_silu` write the scale DIRECTLY in
the slot layout behind a `comptime fuse_a_scale_preshuffle` flag, deleting the
separate kernel from the critical path.

This test drives the REAL `fused_silu_mx_kernel` two ways and asserts the
produced slot buffers are byte-for-byte equal:

  reference (fuse_a_scale_preshuffle=False): fused_silu writes raw [tokens, K/32]
      --preshuffle_grouped_scale_4d_gpu-->  per-expert scale_4d slots
  fused     (fuse_a_scale_preshuffle=True):  fused_silu writes the slot layout
      directly.

Both slot buffers are memset to 0 first; the comparison spans the full
`num_active * max_padded_M * K_SCALES` slot region (trailing pad rows the
producer never visits stay zero in both, exactly as the matmul's tight
per-expert V# bound clamps OOB reads to 0).

This is a *local* quantize kernel: a synthesized ragged intermediate
activation + `row_offsets` (the per-expert prefix sum) drives it on ONE GPU.
No EP comm / no multi-GPU is needed (the scale write reads a local buffer and
writes a local buffer; nothing crosses ranks).

Currently MI355X-only.

Usage (after editing ep_comm.mojo, plain `mojo` is shadowed by the installed
shmem package — use bmojo / bazel):
  ./bazelw run //Mojo/tools/mojo -- \\
      max/kernels/test/gpu/shmem/test_mxfp4_fused_silu_scale_fusion.mojo
"""

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from std.math import align_up, exp, isfinite
from std.random import random_float64, seed

from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import E2M1_TO_FLOAT32, MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import Shuffler

from shmem.ep_comm import fused_silu_mx_kernel

from std.testing import assert_equal


# ===----------------------------------------------------------------------=== #
# Input helpers
# ===----------------------------------------------------------------------=== #


def _fill_random_bf16(
    buf: HostBuffer[.bfloat16],
    n: Int,
    lo: Float64 = -4.0,
    hi: Float64 = 4.0,
):
    # Deterministic activations in [lo, hi] so ref and fused paths quantize
    # identically. Default [-4,4] stays under the clamp (limit=7); the probe
    # widens it so the OAI clamps fire.
    for i in range(n):
        buf[i] = BFloat16(random_float64(lo, hi))


def _build_routing(
    a_offsets_host: HostBuffer[.uint32],
    num_tokens_by_expert: List[Int],
):
    """Ragged per-expert prefix sums: row_offsets[0]=0, row_offsets[e+1]=sum."""
    a_offsets_host[0] = UInt32(0)
    for i in range(len(num_tokens_by_expert)):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )


# ===----------------------------------------------------------------------=== #
# The check: fused slots == reference (separate-kernel) slots, byte for byte.
# ===----------------------------------------------------------------------=== #


def _run_fusion_check[
    hidden_size: Int,
    NUM_ACTIVE: Int,
    clamp_activation: Bool = False,
](
    name: String,
    num_tokens_by_expert: List[Int],
    inflate_padded_M: Int,
    ctx: DeviceContext,
    alpha: Float32 = 0.0,
    limit: Float32 = 0.0,
) raises:
    comptime assert hidden_size % MXFP4_SF_VECTOR_SIZE == 0
    # fused_silu input is gate+up concatenated: input_dim == hidden_size * 2.
    comptime input_dim = hidden_size * 2
    # FP4 packs 2 elements/byte → packed output dim is hidden_size // 2.
    comptime output_dim = hidden_size // 2
    comptime scale_K = hidden_size // MXFP4_SF_VECTOR_SIZE

    # `row_offsets` MUST have a STATIC outer dim: the kernel reads
    # `row_offsets[static_shape[0] - 1]` for num_tokens (and the fused path
    # scans `static_shape[0] - 1` experts). A runtime dim makes static_shape[0]
    # == -1, so the kernel reads OOB garbage. Hence NUM_ACTIVE is comptime.
    comptime n_off = NUM_ACTIVE + 1
    var num_active = NUM_ACTIVE
    debug_assert(len(num_tokens_by_expert) == NUM_ACTIVE)
    var total_tokens = 0
    var max_tokens = 0
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)
    # The slot stride the matmul reads with. `inflate_padded_M` lets a test
    # drive a build-time bound strictly larger than the runtime max (the common
    # decode case) to guard the producer/consumer single-source-of-truth.
    var max_padded_M = max(align_up(max_tokens, 32), inflate_padded_M)
    var slot_bytes = num_active * max_padded_M * scale_K

    print(
        "  ",
        name,
        " active=",
        num_active,
        " total_tokens=",
        total_tokens,
        " hidden=",
        hidden_size,
        " scale_K=",
        scale_K,
        " max_padded_M=",
        max_padded_M,
    )

    comptime hw = ctx.default_device_info

    # --- Host inputs: synthesized ragged BF16 activation + ragged routing. ---
    var input_h = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * input_dim
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](n_off)
    ctx.synchronize()
    _fill_random_bf16(input_h, total_tokens * input_dim)
    _build_routing(a_off_h, num_tokens_by_expert)

    # --- Device buffers. ---
    var input_d = ctx.enqueue_create_buffer[.bfloat16](total_tokens * input_dim)
    var a_off_d = ctx.enqueue_create_buffer[.uint32](n_off)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(a_off_d, a_off_h)

    var input_tt = TileTensor[origin=ImmutAnyOrigin](
        input_d, row_major(Coord(total_tokens, Idx[input_dim]))
    )
    var a_off_tt = TileTensor[origin=ImmutAnyOrigin](
        a_off_d, row_major[n_off]()
    )

    # The fused_silu_mx_kernel is monomorphic over layout types; bind the
    # comptime params (dtypes, layouts, threads/SMs) and `fuse_a_scale_preshuffle`.
    comptime out_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(Coord(total_tokens, Idx[output_dim]))
        )
    ).LayoutType
    comptime raw_scales_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(Coord(total_tokens, Idx[scale_K]))
        )
    ).LayoutType
    comptime slot_scales_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(Coord(num_active * max_padded_M, Idx[scale_K]))
        )
    ).LayoutType

    comptime kernel_ref = fused_silu_mx_kernel[
        DType.uint8,
        DType.float8_e8m0fnu,
        DType.bfloat16,
        out_layout,
        raw_scales_layout,
        input_tt.LayoutType,
        a_off_tt.LayoutType,
        hw.max_thread_block_size,
        hw.sm_count,
        fuse_a_scale_preshuffle=False,
        clamp_activation=clamp_activation,
    ]
    comptime kernel_fused = fused_silu_mx_kernel[
        DType.uint8,
        DType.float8_e8m0fnu,
        DType.bfloat16,
        out_layout,
        slot_scales_layout,
        input_tt.LayoutType,
        a_off_tt.LayoutType,
        hw.max_thread_block_size,
        hw.sm_count,
        fuse_a_scale_preshuffle=True,
        clamp_activation=clamp_activation,
    ]

    # ---- Path A (reference): fused_silu writes raw [tokens, scale_K], then
    #      the standalone preshuffle kernel rearranges into slots. ----
    var raw_out_d = ctx.enqueue_create_buffer[.uint8](total_tokens * output_dim)
    var raw_scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var ref_d = ctx.enqueue_create_buffer[.uint8](slot_bytes)
    ref_d.enqueue_fill(UInt8(0))

    ctx.enqueue_function[kernel_ref](
        TileTensor[origin=MutAnyOrigin](
            raw_out_d, row_major(Coord(total_tokens, Idx[output_dim]))
        ),
        TileTensor[origin=MutAnyOrigin](
            raw_scales_d, row_major(Coord(total_tokens, Idx[scale_K]))
        ),
        input_tt,
        a_off_tt,
        Int32(0),  # max_padded_M unused when fuse_a_scale_preshuffle=False
        alpha,
        limit,
        grid_dim=hw.sm_count,
        block_dim=hw.max_thread_block_size,
    )

    var raw_scales_bytes = TileTensor[origin=ImmutAnyOrigin](
        raw_scales_d.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        row_major(Coord(total_tokens, Idx[scale_K])),
    )
    var ref_slots_tt = TileTensor[origin=MutAnyOrigin](
        ref_d, row_major(Coord(num_active * max_padded_M, Idx[scale_K]))
    )
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        raw_scales_bytes,
        ref_slots_tt,
        a_off_tt,
        num_active,
        # IMPORTANT: the reference standalone preshuffle must write with the
        # SAME slot stride the fused path uses (`max_padded_M`), not the runtime
        # `max_tokens`. Passing the (possibly inflated) bound keeps both paths'
        # slot strides identical so the whole-buffer byte compare is valid
        # (producer/consumer single-source-of-truth, in test form).
        max_padded_M,
        hw.sm_count * 2,
        ctx,
    )

    # ---- Path B (fused): fused_silu writes the slot layout directly. ----
    var fused_out_d = ctx.enqueue_create_buffer[.uint8](
        total_tokens * output_dim
    )
    var fused_scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slot_bytes)
    fused_scales_d.enqueue_fill(Float8_e8m0fnu(0))

    ctx.enqueue_function[kernel_fused](
        TileTensor[origin=MutAnyOrigin](
            fused_out_d, row_major(Coord(total_tokens, Idx[output_dim]))
        ),
        # When preshuffled the kernel treats `scales` as a flat slot byte
        # buffer via `scale_4d_byte_off`; the [slot_rows, scale_K] 2D wrapper
        # just supplies the base pointer and total element count.
        TileTensor[origin=MutAnyOrigin](
            fused_scales_d,
            row_major(Coord(num_active * max_padded_M, Idx[scale_K])),
        ),
        input_tt,
        a_off_tt,
        Int32(max_padded_M),
        alpha,
        limit,
        grid_dim=hw.sm_count,
        block_dim=hw.max_thread_block_size,
    )

    # --- Compare byte for byte over the full slot region. ---
    var ref_host = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    var fused_host = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    ctx.enqueue_copy(ref_host, ref_d)
    ctx.enqueue_copy(fused_host, fused_scales_d.unsafe_ptr().bitcast[UInt8]())
    ctx.synchronize()

    var mismatches = 0
    for i in range(slot_bytes):
        if ref_host[i] != fused_host[i]:
            mismatches += 1
    assert_equal(mismatches, 0)
    print("    OK: ", slot_bytes, " bytes match")


# ===----------------------------------------------------------------------=== #
# Numerical activation probe: the clamp-math correctness gate.
# ===----------------------------------------------------------------------=== #


def _run_activation_probe[
    hidden_size: Int,
](
    name: String,
    num_tokens: Int,
    alpha: Float32,
    limit: Float32,
    ctx: DeviceContext,
) raises:
    """Numerical correctness gate for the OAI-clamped SwiGLU activation.

    `_run_fusion_check`'s byte-compare shares the activation on both sides, so a
    clamp bug is invisible to it. This dequantizes the FP4 output and checks
    each element against a host fp32 reference (`_swigluoai_activation`):

        g' = min(g, L);  u' = clamp(u, -L, L);  z = (u'+1)*g'*sigmoid(g'*alpha)

    Tolerance ~1 MXFP4 quant step, so a wrong clamp fails. Inputs span [-12,12]
    (> limit) so both clamps fire; `fuse_a_scale_preshuffle=False` gives
    row-major scales for a direct [token, block] lookup.
    """
    comptime input_dim = hidden_size * 2
    comptime output_dim = hidden_size // 2
    comptime scale_K = hidden_size // MXFP4_SF_VECTOR_SIZE
    comptime NUM_ACTIVE = 1
    comptime n_off = NUM_ACTIVE + 1

    print(
        "  ",
        name,
        " tokens=",
        num_tokens,
        " hidden=",
        hidden_size,
        " scale_K=",
        scale_K,
        " alpha=",
        alpha,
        " limit=",
        limit,
    )

    comptime hw = ctx.default_device_info

    # One expert over all tokens (activation is per-element).
    var input_h = ctx.enqueue_create_host_buffer[.bfloat16](
        num_tokens * input_dim
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](n_off)
    ctx.synchronize()
    # > limit so both clamps fire.
    _fill_random_bf16(input_h, num_tokens * input_dim, -12.0, 12.0)
    for h in range(MXFP4_SF_VECTOR_SIZE):
        input_h[h] = BFloat16(-3.0e38)
        input_h[h + hidden_size] = BFloat16(7.0)
    a_off_h[0] = UInt32(0)
    a_off_h[1] = UInt32(num_tokens)

    var input_d = ctx.enqueue_create_buffer[.bfloat16](num_tokens * input_dim)
    var a_off_d = ctx.enqueue_create_buffer[.uint32](n_off)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(a_off_d, a_off_h)

    var input_tt = TileTensor[origin=ImmutAnyOrigin](
        input_d, row_major(Coord(num_tokens, Idx[input_dim]))
    )
    var a_off_tt = TileTensor[origin=ImmutAnyOrigin](
        a_off_d, row_major[n_off]()
    )

    comptime out_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(Coord(num_tokens, Idx[output_dim]))
        )
    ).LayoutType
    comptime scales_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(Coord(num_tokens, Idx[scale_K]))
        )
    ).LayoutType

    comptime kernel = fused_silu_mx_kernel[
        DType.uint8,
        DType.float8_e8m0fnu,
        DType.bfloat16,
        out_layout,
        scales_layout,
        input_tt.LayoutType,
        a_off_tt.LayoutType,
        hw.max_thread_block_size,
        hw.sm_count,
        fuse_a_scale_preshuffle=False,
        clamp_activation=True,
    ]

    var out_d = ctx.enqueue_create_buffer[.uint8](num_tokens * output_dim)
    var scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        num_tokens * scale_K
    )
    out_d.enqueue_fill(UInt8(0))
    scales_d.enqueue_fill(Float8_e8m0fnu(0))

    ctx.enqueue_function[kernel](
        TileTensor[origin=MutAnyOrigin](
            out_d, row_major(Coord(num_tokens, Idx[output_dim]))
        ),
        TileTensor[origin=MutAnyOrigin](
            scales_d, row_major(Coord(num_tokens, Idx[scale_K]))
        ),
        input_tt,
        a_off_tt,
        Int32(0),  # max_padded_M unused when fuse_a_scale_preshuffle=False
        alpha,
        limit,
        grid_dim=hw.sm_count,
        block_dim=hw.max_thread_block_size,
    )

    var out_host = ctx.enqueue_create_host_buffer[.uint8](
        num_tokens * output_dim
    )
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        num_tokens * scale_K
    )
    ctx.enqueue_copy(out_host, out_d)
    ctx.enqueue_copy(scales_host, scales_d)
    ctx.synchronize()

    var violations = 0
    var max_abs_err = Float32(0.0)
    var max_ratio = Float32(0.0)
    for m in range(num_tokens):
        for h in range(hidden_size):
            var gate = input_h[m * input_dim + h].cast[.float32]()
            var up = input_h[m * input_dim + h + hidden_size].cast[
                DType.float32
            ]()
            # OAI-clamped SwiGLU reference (matches `_swigluoai_activation`).
            var g_c = min(gate, limit)
            var u_c = min(max(up, -limit), limit)
            var sig = 1.0 / (1.0 + exp(-(g_c * alpha)))
            var expected = g_c * sig * (u_c + 1.0)

            # Dequant: low nibble=even elem, high=odd; E2M1 value * block scale
            # (E8M0->f32 = 2^(byte-127), the scale fed to cvt.scalef32.pk.fp4).
            var byte = out_host[m * output_dim + (h // 2)]
            var nibble = (byte & 0xF) if (h % 2 == 0) else ((byte >> 4) & 0xF)
            var fp4 = E2M1_TO_FLOAT32[Int(nibble)]
            var s = scales_host[m * scale_K + (h // MXFP4_SF_VECTOR_SIZE)].cast[
                DType.float32
            ]()
            var dequant = fp4 * s

            var err = abs(dequant - expected)
            if not isfinite(dequant):
                violations += 1
            # ~1 quant step (= block scale). 1.10x covers the 4->6 midpoint
            # (err==1.0*S) + transcendental jitter, still fails a wrong clamp.
            var tol = 1.10 * s
            if err > tol:
                violations += 1
            if err > max_abs_err:
                max_abs_err = err
            if s > 0.0 and (err / s) > max_ratio:
                max_ratio = err / s

    print(
        "    max_abs_err=",
        max_abs_err,
        " worst err/S=",
        max_ratio,
        " violations=",
        violations,
    )
    assert_equal(violations, 0)
    print(
        "    OK: activation probe passed (", num_tokens * hidden_size, " elems)"
    )


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp4_fused_silu_scale_fusion currently requires MI355X"

    # KS64 = down proj: intermediate (hidden) = 2048 → scale_K = 64.
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=1](
        "down-proj-single-tiny", [1], 0, ctx
    )
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=4](
        "down-proj-decode", [1, 2, 0, 4], 0, ctx
    )
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=5](
        "down-proj-mixed", [3, 33, 0, 17, 64], 0, ctx
    )
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=3](
        "down-proj-prefill", [128, 33, 96], 0, ctx
    )
    # Inflated max_padded_M (> runtime max) with >1 expert: guards that the
    # producer places each expert at `expert_slot * max_padded_M * scale_K`
    # using the SAME build-time slot stride as the matmul reader — the exact
    # single-source-of-truth class the matmul A/B (in `amd_tests`, currently
    # `manual`-tagged) would otherwise be the only thing covering.
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=3](
        "down-proj-inflated-stride", [5, 20, 12], 128, ctx
    )

    # KS96 = M3 down proj (hidden 3072 -> scale_K 96) with clamped SwiGLU.
    # Fold-off and fold-on run the same clamp, so slot byte-equivalence holds.
    _run_fusion_check[hidden_size=3072, NUM_ACTIVE=1, clamp_activation=True](
        "ks96-clamp-single-tiny", [1], 0, ctx, alpha=1.702, limit=7.0
    )
    _run_fusion_check[hidden_size=3072, NUM_ACTIVE=4, clamp_activation=True](
        "ks96-clamp-decode", [1, 2, 0, 4], 0, ctx, alpha=1.702, limit=7.0
    )
    _run_fusion_check[hidden_size=3072, NUM_ACTIVE=3, clamp_activation=True](
        "ks96-clamp-prefill", [128, 33, 96], 0, ctx, alpha=1.702, limit=7.0
    )
    # Inflated max_padded_M (> runtime max), clamp on: slot-stride guard.
    _run_fusion_check[hidden_size=3072, NUM_ACTIVE=3, clamp_activation=True](
        "ks96-clamp-inflated-stride",
        [5, 20, 12],
        128,
        ctx,
        alpha=1.702,
        limit=7.0,
    )
    # Cross-coverage: KS64 with the clamp on, and KS96 with plain SiLU.
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=4, clamp_activation=True](
        "ks64-clamp-decode", [1, 2, 0, 4], 0, ctx, alpha=1.702, limit=7.0
    )
    _run_fusion_check[hidden_size=3072, NUM_ACTIVE=5](
        "ks96-plainsilu-mixed", [3, 33, 0, 17, 64], 0, ctx
    )
    print("All fused_silu_mx scale-fusion checks passed.")

    # Clamp-math correctness gate (byte-equivalence can't catch it).
    # M3 constants: alpha=1.702, limit=7.0.
    _run_activation_probe[hidden_size=3072](
        "ks96-activation-probe", 128, 1.702, 7.0, ctx
    )
    _run_activation_probe[hidden_size=2048](
        "ks64-activation-probe", 96, 1.702, 7.0, ctx
    )
    print("All fused_silu_mx activation probes passed.")
