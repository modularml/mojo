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
"""Tests for native MXFP6 grouped matmul on AMD CDNA4.

The MXFP6 twin of `test_block_scaled_grouped_matmul_amd.mojo`, running that file's
case list -- same expert counts, shapes and routing -- once per FP6 encoding.
This is the non-preshuffled grouped kernel (`block_scaled_grouped_matmul_amd`, raw
rank-3 B); the preshuffled-B kernels live in
`test_mxfp6_grouped_matmul_amd_kernels.mojo`.

Both tile paths are exercised: K=512 with `max_tokens <= 64` takes the
BK_ELEMS=512 branch (a 384-byte tile in FP6), everything else falls back to
BK_ELEMS=128 (96 bytes).

The reference is a per-element software decode rather than the per-expert
ungrouped MFMA kernel the MXFP4 file uses. It also emits `sum |a*b|`, so the
tolerance is K-scaled against the accumulated magnitude instead of against a
result that may have cancelled to near zero -- and it shares no code with the
path under test, which a same-MFMA-family reference cannot claim.

Usage:
  br test_mxfp6_grouped_matmul_amd.mojo.test
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64, seed
from std.testing import assert_true
from std.utils import StaticTuple

from layout import Coord, Idx, TileTensor, row_major
from linalg.arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat
from linalg.fp6_utils import (
    FP6Format,
    MXFP6_SF_VECTOR_SIZE,
    decode_fp6_to_f32,
    unpack_fp6_x32,
)
from linalg.matmul.gpu.amd import block_scaled_grouped_matmul_amd


def _mfma_format[fmt: FP6Format]() -> CDNA4F8F6F4MatrixFormat:
    comptime if fmt == FP6Format.E2M3:
        return CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3
    return CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2


# ===----------------------------------------------------------------------=== #
# Reference — one thread per (m, n), decoding in software.
#
# Independent of the path under test: the kernel never software-decodes, it
# hands raw bits to the MFMA and the hardware decodes them. `test_fp6_utils`
# pins this decoder against hand-written tables
# pins the hardware against the same convention.
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
def _mxfp6_matmul_ref[
    fmt: FP6Format
](
    a_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    b_ptr: ImmPointer[UInt8, ImmutAnyOrigin],
    a_sf_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    b_sf_ptr: ImmPointer[Float8_e8m0fnu, ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    mag_ptr: MutPointer[Float32, MutAnyOrigin],
    M_dev: Int32,
    N_dev: Int32,
    K_dev: Int32,
    c_row_stride: Int32,
):
    """Computes one expert's tile; `c_row_stride` lets the caller write into a
    row range of the grouped output without copying."""
    var M = Int(M_dev)
    var N = Int(N_dev)
    var K = Int(K_dev)

    var m = Int(global_idx.x)
    var n = Int(global_idx.y)
    if m >= M or n >= N:
        return

    var k_groups = K // 32
    var k_bytes = (K * 6) // 8

    var accum = Float32(0)
    var magnitude = Float32(0)

    for ko in range(k_groups):
        var a_scale = a_sf_ptr[unsafe_offset=m * k_groups + ko].cast[.float32]()
        var b_scale = b_sf_ptr[unsafe_offset=n * k_groups + ko].cast[.float32]()

        # 32 elements is one MX block = 24 packed bytes, always 8-byte aligned
        # because k_bytes is a multiple of 24 whenever K is a multiple of 32.
        var a_base = m * k_bytes + ko * 24
        var b_base = n * k_bytes + ko * 24

        var fa = SIMD[.uint8, 32](0)
        var fb = SIMD[.uint8, 32](0)
        comptime for chunk in range(3):
            fa = fa.insert[offset=chunk * 8](
                a_ptr.load[width=8](a_base + chunk * 8)
            )
            fb = fb.insert[offset=chunk * 8](
                b_ptr.load[width=8](b_base + chunk * 8)
            )

        var av = decode_fp6_to_f32[fmt](unpack_fp6_x32(fa)) * a_scale
        var bv = decode_fp6_to_f32[fmt](unpack_fp6_x32(fb)) * b_scale
        var prod = av * bv
        accum += prod.reduce_add()
        magnitude += abs(prod).reduce_add()

    c_ptr[unsafe_offset=m * Int(c_row_stride) + n] = accum
    mag_ptr[unsafe_offset=m * Int(c_row_stride) + n] = magnitude


# ===----------------------------------------------------------------------=== #
# Test harness
# ===----------------------------------------------------------------------=== #


def test_mxfp6_grouped_matmul[
    fmt: FP6Format, num_experts: Int, N: Int, K: Int
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises -> Bool:
    """Tests `block_scaled_grouped_matmul_amd` in FP6 mode against a software decode.

    Parameters:
        fmt: The FP6 encoding under test (E2M3 or E3M2).
        num_experts: Total number of expert weight matrices.
        N: Output columns / weight rows.
        K: Logical K dimension (FP6 elements, must be a multiple of 128).

    Args:
        num_active_experts: Number of routed expert slots.
        num_tokens_by_expert: Token count per active slot.
        expert_ids_list: Expert index per active slot.
        ctx: Device context.

    Returns:
        True if every output element is within tolerance.
    """
    comptime assert (
        K % 128 == 0
    ), "K must be a multiple of 128 (MFMA K dimension)"

    comptime K_BYTES = (K * 6) // 8
    comptime scale_K = K // MXFP6_SF_VECTOR_SIZE
    comptime fmt_name = "E2M3" if fmt == FP6Format.E2M3 else "E3M2"

    var total_tokens = 0
    var max_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_tokens += num_tokens_by_expert[i]
        max_tokens = max(max_tokens, num_tokens_by_expert[i])

    print(
        "  ",
        fmt_name,
        " grouped matmul: num_experts=",
        num_experts,
        " N=",
        N,
        " K=",
        K,
        " num_active=",
        num_active_experts,
        " total_tokens=",
        total_tokens,
    )

    var a_host = ctx.enqueue_create_host_buffer[.uint8](total_tokens * K_BYTES)
    var b_host = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * K_BYTES
    )
    var a_scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_host = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )
    ctx.synchronize()

    # Every 6-bit code is a finite number in both encodings, so random bytes
    # need no NaN/Inf filtering.
    for i in range(total_tokens * K_BYTES):
        a_host[i] = UInt8(random_ui64(0, 255))
    for i in range(num_experts * N * K_BYTES):
        b_host[i] = UInt8(random_ui64(0, 255))

    # Scales: exponent range [125..129] for reasonable magnitudes.
    for i in range(total_tokens * scale_K):
        a_scales_host[i] = bitcast[.float8_e8m0fnu](
            UInt8(random_ui64(125, 129))
        )
    for i in range(num_experts * N * scale_K):
        b_scales_host[i] = bitcast[.float8_e8m0fnu](
            UInt8(random_ui64(125, 129))
        )

    a_offsets_host[0] = UInt32(0)
    for i in range(num_active_experts):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host[i] = Int32(expert_ids_list[i])

    var a_dev = ctx.enqueue_create_buffer[.uint8](total_tokens * K_BYTES)
    var b_dev = ctx.enqueue_create_buffer[.uint8](num_experts * N * K_BYTES)
    var a_scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[.int32](num_active_experts)
    var c_dev = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    var c_ref_dev = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    var mag_dev = ctx.enqueue_create_buffer[.float32](total_tokens * N)

    # Inactive slots are written by neither side, so both need a known value.
    c_dev.enqueue_fill(Float32(0.0))
    c_ref_dev.enqueue_fill(Float32(0.0))
    mag_dev.enqueue_fill(Float32(0.0))

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)
    ctx.enqueue_copy(a_offsets_dev, a_offsets_host)
    ctx.enqueue_copy(expert_ids_dev, expert_ids_host)

    # Reference: one software-decode launch per active expert, writing its
    # token range of the grouped output in place.
    comptime REF_BLOCK = 16
    for i in range(num_active_experts):
        var token_start = Int(a_offsets_host[i])
        var num_tokens = Int(a_offsets_host[i + 1]) - token_start
        var expert_id = Int(expert_ids_host[i])
        if num_tokens <= 0 or expert_id < 0:
            continue

        ctx.enqueue_function[_mxfp6_matmul_ref[fmt]](
            a_dev.unsafe_ptr() + token_start * K_BYTES,
            b_dev.unsafe_ptr() + expert_id * N * K_BYTES,
            a_scales_dev.unsafe_ptr() + token_start * scale_K,
            b_scales_dev.unsafe_ptr() + expert_id * N * scale_K,
            c_ref_dev.unsafe_ptr() + token_start * N,
            mag_dev.unsafe_ptr() + token_start * N,
            Int32(num_tokens),
            Int32(N),
            Int32(K),
            Int32(N),
            grid_dim=(ceildiv(num_tokens, REF_BLOCK), ceildiv(N, REF_BLOCK)),
            block_dim=(REF_BLOCK, REF_BLOCK),
        )

    var a_tt = TileTensor(
        a_dev, row_major(Coord(total_tokens, Idx[K_BYTES]))
    ).as_immut()
    var b_tt = TileTensor(
        b_dev, row_major[num_experts, N, K_BYTES]()
    ).as_immut()
    var a_scales_tt = TileTensor(
        a_scales_dev, row_major(Coord(total_tokens, Idx[scale_K]))
    ).as_immut()
    var b_scales_tt = TileTensor(
        b_scales_dev, row_major[num_experts, N, scale_K]()
    ).as_immut()
    var a_offsets_tt = TileTensor(
        a_offsets_dev, row_major(Coord(num_active_experts + 1))
    )
    var expert_ids_tt = TileTensor(
        expert_ids_dev, row_major(Coord(num_active_experts))
    )
    var c_tt = TileTensor(c_dev, row_major(Coord(total_tokens, Idx[N])))

    block_scaled_grouped_matmul_amd[matrix_format=_mfma_format[fmt]()](
        c_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        a_offsets_tt,
        expert_ids_tt,
        max_tokens,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    var c_host = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var mag_host = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.enqueue_copy(mag_host, mag_dev)
    ctx.synchronize()

    # float32 accumulation error grows with the number of summed terms, so the
    # bound is K-scaled and applied to sum |a*b| -- not to a result that may
    # have cancelled to near zero, where any relative bound is meaningless.
    comptime ULP_F32 = 5.9604644775390625e-8
    var rel_tol = Float64(16 * K) * ULP_F32

    var mismatches = 0
    var saw_nonzero = False
    for i in range(total_tokens * N):
        var want = Float64(c_ref_host[i])
        var got = Float64(c_host[i])
        if want != Float64(0.0):
            saw_nonzero = True
        if abs(got - want) > rel_tol * Float64(mag_host[i]):
            if mismatches < 3:
                print(
                    "      [",
                    i // N,
                    ",",
                    i % N,
                    "] got=",
                    got,
                    " want=",
                    want,
                    " mag=",
                    Float64(mag_host[i]),
                )
            mismatches += 1

    _ = a_dev^
    _ = b_dev^
    _ = a_scales_dev^
    _ = b_scales_dev^
    _ = a_offsets_dev^
    _ = expert_ids_dev^
    _ = c_dev^
    _ = c_ref_dev^
    _ = mag_dev^

    # An all-zero reference would make the comparison vacuous: it would pass
    # against a kernel that wrote nothing.
    if not saw_nonzero:
        print("    FAIL  reference is all zero; comparison proves nothing")
        return False
    if mismatches > 0:
        print("    FAIL ", mismatches, " wrong of ", total_tokens * N)
        return False
    print("    PASS")
    return True


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        print("===> MXFP6 grouped matmul (native CDNA4 MFMA)")

        var ok = True

        comptime for i in range(2):
            comptime fmt = FP6Format.E2M3 if i == 0 else FP6Format.E3M2

            # Single expert (degenerates to regular matmul)
            print("-- Single expert --")
            ok &= test_mxfp6_grouped_matmul[fmt, 1, 128, 128](
                1, [128], [0], ctx
            )
            ok &= test_mxfp6_grouped_matmul[fmt, 1, 256, 256](
                1, [128], [0], ctx
            )

            # Multiple experts, simple routing
            print("-- Multiple experts --")
            ok &= test_mxfp6_grouped_matmul[fmt, 4, 128, 128](
                2, [128, 128], [0, 2], ctx
            )
            ok &= test_mxfp6_grouped_matmul[fmt, 4, 256, 256](
                3, [128, 128, 128], [0, 1, 3], ctx
            )

            # Unequal token counts
            print("-- Unequal token counts --")
            ok &= test_mxfp6_grouped_matmul[fmt, 4, 128, 256](
                2, [128, 256], [0, 2], ctx
            )

            # Larger dimensions
            print("-- Larger dimensions --")
            ok &= test_mxfp6_grouped_matmul[fmt, 4, 256, 512](
                2, [128, 256], [1, 3], ctx
            )

            # Decode tile path: max_tokens_per_expert <= 64 with a whole
            # number of BK_ELEMS=512 tiles (384 bytes each in FP6).
            print("-- Decode tile (small max tokens, large K) --")
            ok &= test_mxfp6_grouped_matmul[fmt, 1, 128, 512](1, [32], [0], ctx)
            ok &= test_mxfp6_grouped_matmul[fmt, 4, 128, 512](
                2, [16, 64], [0, 2], ctx
            )

        assert_true(ok, "one or more MXFP6 grouped matmul cases failed")
        print("==== All MXFP6 grouped matmul tests passed ====")
