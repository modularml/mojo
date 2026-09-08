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
"""`fused_silu_mxfp6_kernel` against a host fp32 SwiGLU reference.

The MXFP4/MXFP8 siblings pair a fold-equivalence gate with a dequant gate;
MXFP6 has no A-scale fold to compare against, so the dequant gate carries the
whole test. It is the one that matters here anyway: it is the only check that
can see a wrong element-to-scale-block pairing or a mis-sized packed row.

FP6's byte width is three quarters of the hidden size, where MXFP8 is 1x and
MXFP4 is 1/2. A kernel that reused either ratio would write a correctly-shaped
tensor full of misplaced bytes, so the reference is compared element by
element, not just in aggregate.

Usage (plain `mojo` is shadowed by the installed shmem package -- use bazel):
  ./bazelw run //Mojo/tools/mojo -- \
      max/kernels/test/gpu/shmem/test_mxfp6_fused_silu.mojo
"""

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from std.math import exp, isfinite
from std.random import random_float64, seed
from std.testing import assert_almost_equal, assert_true

from layout import Coord, Idx, TileTensor, row_major
from linalg.fp6_utils import (
    MXFP6_SF_VECTOR_SIZE,
    FP6Format,
    fp6_reference_table,
)
from shmem.ep_comm import fused_silu_mxfp6_kernel


def _decode_fp6_host[fmt: FP6Format](code: UInt8) -> Float32:
    """Decodes one six-bit code via the OCP reference table."""
    return fp6_reference_table[fmt]()[Int(code)]


def _unpack_code(packed: HostBuffer[.uint8], base: Int, i: Int) -> UInt8:
    """Extracts element `i` of a packed FP6 row starting at byte `base`.

    Element `i` occupies bits `[6i+5 : 6i]` of a little-endian 24-bit word
    covering four elements, so a group -- not a byte -- is the smallest
    addressable unit.
    """
    var group = i // 4
    var within = i % 4
    var b0 = UInt32(Int(packed[base + group * 3 + 0]))
    var b1 = UInt32(Int(packed[base + group * 3 + 1]))
    var b2 = UInt32(Int(packed[base + group * 3 + 2]))
    var word = b0 | (b1 << 8) | (b2 << 16)
    return UInt8(Int((word >> UInt32(6 * within)) & UInt32(0x3F)))


def _run_case[
    hidden_size: Int, NUM_EXPERTS: Int, clamp_activation: Bool
](ctx: DeviceContext, num_tokens_by_expert: List[Int]) raises:
    comptime fmt = FP6Format.E2M3
    comptime out_bytes = (hidden_size * 6) // 8
    comptime scale_k = hidden_size // MXFP6_SF_VECTOR_SIZE
    comptime input_dim = hidden_size * 2
    comptime hw = MI355X

    comptime n_experts = NUM_EXPERTS
    comptime n_off = NUM_EXPERTS + 1
    debug_assert(len(num_tokens_by_expert) == n_experts)
    var total_tokens = 0
    for i in range(n_experts):
        total_tokens += num_tokens_by_expert[i]

    var alpha = Float32(1.702)
    var limit = Float32(7.0)

    var input_h = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * input_dim
    )
    ctx.synchronize()
    for i in range(total_tokens * input_dim):
        input_h[i] = BFloat16(random_float64(-4.0, 4.0))

    var off_h = ctx.enqueue_create_host_buffer[.uint32](n_off)
    ctx.synchronize()
    off_h[0] = UInt32(0)
    for i in range(n_experts):
        off_h[i + 1] = off_h[i] + UInt32(num_tokens_by_expert[i])

    var input_d = ctx.enqueue_create_buffer[.bfloat16](total_tokens * input_dim)
    var off_d = ctx.enqueue_create_buffer[.uint32](n_off)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(off_d, off_h)

    var out_d = ctx.enqueue_create_buffer[.uint8](total_tokens * out_bytes)
    var scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        total_tokens * scale_k
    )
    out_d.enqueue_fill(UInt8(0))

    var input_tt = TileTensor[origin=ImmutAnyOrigin](
        input_d, row_major(Coord(total_tokens, Idx[input_dim]))
    )
    var off_tt = TileTensor[origin=ImmutAnyOrigin](off_d, row_major[n_off]())
    var out_tt = TileTensor[origin=MutAnyOrigin](
        out_d, row_major(Coord(total_tokens, Idx[out_bytes]))
    )
    var scales_tt = TileTensor[origin=MutAnyOrigin](
        scales_d, row_major(Coord(total_tokens, Idx[scale_k]))
    )

    comptime kernel = fused_silu_mxfp6_kernel[
        DType.float8_e8m0fnu,
        DType.bfloat16,
        out_tt.LayoutType,
        scales_tt.LayoutType,
        input_tt.LayoutType,
        off_tt.LayoutType,
        hw.max_thread_block_size,
        hw.sm_count,
        fp6_format=fmt,
        clamp_activation=clamp_activation,
    ]

    ctx.enqueue_function[kernel](
        out_tt,
        scales_tt,
        input_tt,
        off_tt,
        Int32(0),  # max_padded_M unused when the fold is off
        alpha,
        limit,
        grid_dim=hw.sm_count,
        block_dim=hw.max_thread_block_size,
    )

    var out_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * out_bytes)
    var scales_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        total_tokens * scale_k
    )
    ctx.enqueue_copy(out_h, out_d)
    ctx.enqueue_copy(scales_h, scales_d)
    ctx.synchronize()

    var worst = Float32(0.0)
    for m in range(total_tokens):
        for k in range(hidden_size):
            var g = Float32(input_h[m * input_dim + k])
            var u = Float32(input_h[m * input_dim + hidden_size + k])

            var want: Float32
            comptime if clamp_activation:
                var g_c = min(g, limit)
                var u_c = max(min(u, limit), -limit)
                want = (g_c / (1.0 + exp(-(g_c * alpha)))) * (u_c + 1.0)
            else:
                want = (g / (1.0 + exp(-g))) * u

            var scale = Float32(
                scales_h[m * scale_k + k // MXFP6_SF_VECTOR_SIZE]
            )
            var code = _unpack_code(out_h, m * out_bytes, k)
            var got = _decode_fp6_host[fmt](code) * scale

            # E2M3 carries three mantissa bits, so a block-scaled value lands
            # within ~1/16 of its block maximum. Compare against the block
            # scale rather than the value: a near-zero element in a block with
            # a large maximum is legitimately coarse.
            if isfinite(want):
                assert_true(
                    isfinite(got),
                    String(
                        (
                            "MXFP6 fused SwiGLU produced non-finite output for"
                            " a finite reference at token "
                        ),
                        m,
                        " elem ",
                        k,
                    ),
                )
                var err = abs(want - got)
                if scale > Float32(0.0):
                    err = err / scale
                worst = max(worst, err)

    print(
        "  hidden=",
        hidden_size,
        " clamp=",
        clamp_activation,
        " tokens=",
        total_tokens,
        " worst scaled err=",
        worst,
    )
    # One E2M3 quantum at the top of a block is 0.5 (max finite 7.5 with three
    # mantissa bits), so half a quantum plus bf16 input rounding sits well
    # under 1.0 scale units. A mis-paired scale or a mis-sized row blows far
    # past this rather than nudging it.
    assert_true(
        worst < Float32(1.0),
        String("MXFP6 fused SwiGLU diverged from the reference: ", worst),
    )


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp6_fused_silu requires MI355X (CDNA4)"

    print("===> fused_silu_mxfp6: dequant vs host SwiGLU reference")
    _run_case[256, 2, False](ctx, [3, 5])
    _run_case[256, 2, True](ctx, [3, 5])
    # More tokens than one wave of blocks, and an expert with none routed to
    # it -- an empty slot must not shift the rows that follow.
    _run_case[512, 3, False](ctx, [17, 0, 40])
