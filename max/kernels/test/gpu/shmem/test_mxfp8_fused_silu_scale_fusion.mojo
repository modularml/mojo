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
"""`fused_silu_mxfp8` direct `scale_4d` emission — MXFP8 sibling of
`test_mxfp4_fused_silu_scale_fusion.mojo`.

Two gates per case, because neither alone is sufficient:

1. Byte-compare the slot region between the fold OFF (raw scales + the standalone
   `preshuffle_grouped_scale_4d_gpu`) and ON (direct slot write). Validates the fold.
2. Dequantize the FP8 output against a host fp32 SwiGLU reference. Validates the
   element-to-scale-block pairing, which gate 1 cannot see: both its paths share
   the kernel's `k_scale`, so a wrong one lands at the same wrong slot on each
   side and cancels. MXFP8 has `hidden_size == output_dim` where MXFP4 has
   `output_dim * 2`, so that is the mistake worth guarding. Verified by injecting
   an off-by-one: gate 1 stayed green, gate 2 caught it.

Usage (plain `mojo` is shadowed by the installed shmem package — use bazel):
  ./bazelw run //Mojo/tools/mojo -- \
      max/kernels/test/gpu/shmem/test_mxfp8_fused_silu_scale_fusion.mojo
"""

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from std.math import align_up, exp, isfinite
from std.random import random_float64, seed
from std.testing import assert_equal, assert_true

from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import Shuffler
from shmem.ep_comm import fused_silu_mx_kernel


def _fill_random_bf16(
    mut buf: HostBuffer[.bfloat16],
    n: Int,
    lo: Float64 = -4.0,
    hi: Float64 = 4.0,
):
    for i in range(n):
        buf[i] = BFloat16(random_float64(lo, hi))


def _build_routing(
    mut a_off: HostBuffer[.uint32], num_tokens_by_expert: List[Int]
):
    a_off[0] = UInt32(0)
    for i in range(len(num_tokens_by_expert)):
        a_off[i + 1] = a_off[i] + UInt32(num_tokens_by_expert[i])


def _run_check[
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
    comptime assert hidden_size % MXFP8_SF_VECTOR_SIZE == 0
    comptime input_dim = hidden_size * 2
    # MXFP8: one byte per element, so the output keeps the full hidden dim.
    comptime output_dim = hidden_size
    comptime scale_K = hidden_size // MXFP8_SF_VECTOR_SIZE

    # `row_offsets` needs a STATIC outer dim: the kernel reads
    # `row_offsets[static_shape[0] - 1]` for num_tokens.
    comptime n_off = NUM_ACTIVE + 1
    var num_active = NUM_ACTIVE
    debug_assert(len(num_tokens_by_expert) == NUM_ACTIVE)
    var total_tokens = 0
    var max_tokens = 0
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)
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

    var input_h = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * input_dim
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](n_off)
    ctx.synchronize()
    _fill_random_bf16(input_h, total_tokens * input_dim)
    _build_routing(a_off_h, num_tokens_by_expert)

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
        DType.float8_e4m3fn,
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
        DType.float8_e4m3fn,
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

    # ---- Path A: raw scales, then the standalone preshuffle. ----
    var raw_out_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
        total_tokens * output_dim
    )
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
        Int32(0),  # max_padded_M unused when the fold is off
        alpha,
        limit,
        grid_dim=hw.sm_count,
        block_dim=hw.max_thread_block_size,
    )

    var raw_scales_bytes = TileTensor[origin=ImmutAnyOrigin](
        raw_scales_d.unsafe_ptr()
        .unsafe_bitcast[UInt8]()
        .as_unsafe_any_origin(),
        row_major(Coord(total_tokens, Idx[scale_K])),
    )
    var ref_slots_tt = TileTensor[origin=MutAnyOrigin](
        ref_d, row_major(Coord(num_active * max_padded_M, Idx[scale_K]))
    )
    # Same slot stride as the fused path, so the whole-buffer compare is valid.
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        raw_scales_bytes,
        ref_slots_tt,
        a_off_tt,
        num_active,
        max_padded_M,
        hw.sm_count * 2,
        ctx,
    )

    # ---- Path B: direct slot write. ----
    var fused_out_d = ctx.enqueue_create_buffer[.float8_e4m3fn](
        total_tokens * output_dim
    )
    var fused_scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slot_bytes)
    fused_scales_d.enqueue_fill(Float8_e8m0fnu(0))

    ctx.enqueue_function[kernel_fused](
        TileTensor[origin=MutAnyOrigin](
            fused_out_d, row_major(Coord(total_tokens, Idx[output_dim]))
        ),
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

    var ref_host = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    var fused_host = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    ctx.enqueue_copy(ref_host, ref_d)
    ctx.enqueue_copy(
        fused_host, fused_scales_d.unsafe_ptr().unsafe_bitcast[UInt8]()
    )
    ctx.synchronize()

    var mismatches = 0
    for i in range(slot_bytes):
        if ref_host[i] != fused_host[i]:
            mismatches += 1
    assert_equal(mismatches, 0)

    # Gate 2: dequantize Path A's output (row-major scales) against a host fp32
    # SwiGLU reference. Independent of the fold, so it catches a `k_scale`
    # derived from the wrong hidden size — which gate 1 cancels out.
    comptime if not clamp_activation:
        var out_h = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
            total_tokens * output_dim
        )
        var sc_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
            total_tokens * scale_K
        )
        ctx.enqueue_copy(out_h, raw_out_d)
        ctx.enqueue_copy(sc_h, raw_scales_d)
        ctx.synchronize()

        for m in range(total_tokens):
            for k in range(output_dim):
                var g = input_h[m * input_dim + k].cast[.float32]()
                var u = input_h[m * input_dim + hidden_size + k].cast[
                    DType.float32
                ]()
                var want = (g / (1.0 + exp(-g))) * u
                var scale = sc_h[m * scale_K + k // MXFP8_SF_VECTOR_SIZE].cast[
                    DType.float32
                ]()
                var got = out_h[m * output_dim + k].cast[.float32]() * scale
                assert_true(isfinite(got), "dequantized output must be finite")
                # One E4M3 step at block scale; only a real mis-pairing fails.
                var tol = max(Float32(0.35) * abs(want), scale * Float32(0.75))
                assert_true(
                    abs(got - want) <= tol,
                    String(
                        "dequant mismatch at (",
                        m,
                        ",",
                        k,
                        "): got ",
                        got,
                        " want ",
                        want,
                        " scale ",
                        scale,
                    ),
                )

    comptime dq = "; dequant verified" if not clamp_activation else ""
    print("    OK: ", slot_bytes, " scale bytes match", dq)


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp8_fused_silu_scale_fusion currently requires MI355X"

    print("===> fused_silu_mxfp8: fold equivalence + dequant")
    # M3 down proj: intermediate 3072 -> scale_K 96.
    _run_check[hidden_size=3072, NUM_ACTIVE=1](
        "down-proj-single-tiny", [1], 0, ctx
    )
    _run_check[hidden_size=3072, NUM_ACTIVE=4](
        "down-proj-decode", [1, 2, 0, 4], 0, ctx
    )
    _run_check[hidden_size=3072, NUM_ACTIVE=5](
        "down-proj-mixed", [3, 33, 0, 17, 64], 0, ctx
    )
    _run_check[hidden_size=3072, NUM_ACTIVE=3](
        "down-proj-prefill", [128, 33, 96], 0, ctx
    )
    # Inflated slot stride: guards producer/consumer agreement on max_padded_M.
    _run_check[hidden_size=3072, NUM_ACTIVE=3](
        "down-proj-inflated-stride", [5, 20, 12], 128, ctx
    )
    # Second scale_K (2048 -> 64) covers the K_SCALES derivation at another size.
    _run_check[hidden_size=2048, NUM_ACTIVE=2](
        "down-proj-ks64", [9, 40], 0, ctx
    )
    # Clamped SwiGLU-OAI variant.
    _run_check[hidden_size=3072, NUM_ACTIVE=2, clamp_activation=True](
        "down-proj-clamped", [7, 40], 0, ctx, Float32(1.702), Float32(7.0)
    )
