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
"""AMD RDNA3/4 WMMA implementation for matrix multiply-accumulate operations.

This module provides MMA implementations for AMD RDNA3 and RDNA4 consumer GPUs
using the WMMA (Wave Matrix Multiply Accumulate) instructions.

Reference: https://gpuopen.com/learn/wmma_on_rdna3/
"""

from std.sys import llvm_intrinsic
from std.sys.info import _is_amd_rdna3, _is_amd_rdna4
from std.memory import bitcast
from std._gpu import lane_id

# Import helper functions from parent module
from ..mma import _has_type, _has_shape, _unsupported_mma_op


# ===----------------------------------------------------------------------=== #
# RDNA Matrix Loaders
# ===----------------------------------------------------------------------=== #


@always_inline
def _load_matrix_a_amd_rdna[
    m: Int, n: Int, k: Int
](
    a_ptr: Pointer[Float16, _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[
    DType.float16, 16
]:
    """RDNA-specific implementation: loads 16 FP16 elements per thread."""
    comptime assert m == 16 and n == 16 and k == 16
    var lane = lane_id()
    var thread_x = lane & 15
    var a = SIMD[.float16, 16]()

    comptime for i in range(16):
        var a_idx = ldm * (tile_row + thread_x) + tile_col + i
        a[i] = a_ptr[unsafe_offset=a_idx]

    return a


@always_inline
def _load_matrix_a_amd_rdna[
    m: Int, n: Int, k: Int
](
    a_ptr: Pointer[BFloat16, _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[
    DType.bfloat16, 16
]:
    """RDNA-specific implementation: loads 16 BF16 elements per thread."""
    comptime assert m == 16 and n == 16 and k == 16
    var lane = lane_id()
    var thread_x = lane & 15
    var a = SIMD[.bfloat16, 16]()

    comptime for i in range(16):
        var a_idx = ldm * (tile_row + thread_x) + tile_col + i
        a[i] = a_ptr[unsafe_offset=a_idx]

    return a


@always_inline
def _load_matrix_b_amd_rdna[
    m: Int, n: Int, k: Int
](
    b_ptr: Pointer[Float16, _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[
    DType.float16, 16
]:
    """RDNA-specific implementation: loads 16 FP16 elements per thread."""
    comptime assert m == 16 and n == 16 and k == 16
    var lane = lane_id()
    var thread_y = lane & 15
    var b = SIMD[.float16, 16]()

    comptime for i in range(16):
        var b_idx = ldm * (tile_row + i) + tile_col + thread_y
        b[i] = b_ptr[unsafe_offset=b_idx]

    return b


@always_inline
def _load_matrix_b_amd_rdna[
    m: Int, n: Int, k: Int
](
    b_ptr: Pointer[BFloat16, _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[
    DType.bfloat16, 16
]:
    """RDNA-specific implementation: loads 16 BF16 elements per thread."""
    comptime assert m == 16 and n == 16 and k == 16
    var lane = lane_id()
    var thread_y = lane & 15
    var b = SIMD[.bfloat16, 16]()

    comptime for i in range(16):
        var b_idx = ldm * (tile_row + i) + tile_col + thread_y
        b[i] = b_ptr[unsafe_offset=b_idx]

    return b


@always_inline
def load_matrix_a_amd_rdna16x16x16(
    a_ptr: Pointer[Float16, _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[.float16, 16]:
    """Loads 16×16×16 matrix A tile for RDNA (Wave32) architecture.

    This function is optimized for AMD RDNA GPUs (Radeon RX 7000 series)
    which use Wave32 execution mode. Each thread loads 16 contiguous FP16
    elements using an access pattern appropriate for WMMA instructions.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing 16 FP16 values for this thread.

    Notes:
        The concrete return type (SIMD[16]) avoids type ambiguity and padding overhead.
        This function is architecture-specific for RDNA - for CDNA, use
        load_matrix_a_amd_cdna16x16x16() which returns SIMD[4].
    """
    return _load_matrix_a_amd_rdna[16, 16, 16](a_ptr, tile_row, tile_col, ldm)


@always_inline
def load_matrix_a_amd_rdna16x16x16(
    a_ptr: Pointer[BFloat16, _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[.bfloat16, 16]:
    """Loads 16×16×16 matrix A tile for RDNA (Wave32) architecture.

    This function is optimized for AMD RDNA GPUs (Radeon RX 7000 series)
    which use Wave32 execution mode. Each thread loads 16 contiguous BF16
    elements using an access pattern appropriate for WMMA instructions.

    Args:
        a_ptr: Pointer to matrix A data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix A (stride between rows).

    Returns:
        SIMD vector containing 16 BF16 values for this thread.

    Notes:
        The concrete return type (SIMD[16]) avoids type ambiguity and padding overhead.
        This function is architecture-specific for RDNA - for CDNA, use
        load_matrix_a_amd_cdna16x16x16() which returns SIMD[4].
    """
    return _load_matrix_a_amd_rdna[16, 16, 16](a_ptr, tile_row, tile_col, ldm)


@always_inline
def load_matrix_b_amd_rdna16x16x16(
    b_ptr: Pointer[Float16, _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[.float16, 16]:
    """Loads 16×16×16 matrix B tile for RDNA (Wave32) architecture.

    This function is optimized for AMD RDNA GPUs (Radeon RX 7000 series)
    which use Wave32 execution mode. Each thread loads 16 contiguous FP16
    elements using an access pattern appropriate for WMMA instructions.

    Args:
        b_ptr: Pointer to matrix B data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix B (stride between rows).

    Returns:
        SIMD vector containing 16 FP16 values for this thread.

    Notes:
        The concrete return type (SIMD[16]) avoids type ambiguity and padding overhead.
        This function is architecture-specific for RDNA - for CDNA, use
        load_matrix_b_amd_cdna16x16x16() which returns SIMD[4].
    """
    return _load_matrix_b_amd_rdna[16, 16, 16](b_ptr, tile_row, tile_col, ldm)


@always_inline
def load_matrix_b_amd_rdna16x16x16(
    b_ptr: Pointer[BFloat16, _],
    tile_row: Int,
    tile_col: Int,
    ldm: Int,
) -> SIMD[.bfloat16, 16]:
    """Loads 16×16×16 matrix B tile for RDNA (Wave32) architecture.

    This function is optimized for AMD RDNA GPUs (Radeon RX 7000 series)
    which use Wave32 execution mode. Each thread loads 16 contiguous BF16
    elements using an access pattern appropriate for WMMA instructions.

    Args:
        b_ptr: Pointer to matrix B data in memory.
        tile_row: Starting row index of the tile.
        tile_col: Starting column index of the tile.
        ldm: Leading dimension of matrix B (stride between rows).

    Returns:
        SIMD vector containing 16 BF16 values for this thread.

    Notes:
        The concrete return type (SIMD[16]) avoids type ambiguity and padding overhead.
        This function is architecture-specific for RDNA - for CDNA, use
        load_matrix_b_amd_cdna16x16x16() which returns SIMD[4].
    """
    return _load_matrix_b_amd_rdna[16, 16, 16](b_ptr, tile_row, tile_col, ldm)


# ===----------------------------------------------------------------------=== #
# RDNA WMMA Intrinsics
# ===----------------------------------------------------------------------=== #


@always_inline
def _mma_wmma_rdna(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """Performs AMD RDNA3+ WMMA (Wave Matrix Multiply-Accumulate) operations.

    This function implements matrix multiply-accumulate operations for AMD RDNA3+
    consumer GPUs using WMMA instructions. WMMA was introduced in RDNA3 and is not
    available on RDNA1/2 hardware.

    Supported operations by RDNA generation:

    RDNA3+ (all operations):
        - F32 = F16 * F16 + F32 (16x16x16 shape)
        - F32 = BF16 * BF16 + F32 (16x16x16 shape)

    RDNA4 additional operations:
        - F32 = FP8 * FP8 + F32 (16x16x32 shape, native hardware support)

    FP8 support by generation:
        - RDNA4: Native FP8/BF8 via llvm.amdgcn.wmma.f32.16x16x32.fp8
          - Supports E4M3 (float8_e4m3fn) and E5M2 (float8_e5m2) formats
          - Hardware V_DOT4 instructions for 4-element dot products
          - NEG must be zero for A/B matrices in WMMA operations

    RDNA4 FP8/BF8 native support:
        - Native hardware support for 8-bit float operations
        - E4M3 formats (float8_e4m3fn, float8_e4m3fnuz) map to FP8
        - E5M2 formats (float8_e5m2, float8_e5m2fnuz) map to BF8
        - Supports all combinations: FP8×FP8, BF8×BF8, FP8×BF8, BF8×FP8
        - Each combination has a specific WMMA intrinsic

    Quantized integer operations (RDNA3+):
        - I32 = I8/U8 * I8/U8 + I32 (16x16x16 shape via iu8 intrinsic)
        - I32 = U4 * U4 + I32 (16x16x16 shape via iu4 intrinsic)
        - Inputs are bitcast to int32 before passing to WMMA intrinsics

    Hardware intrinsics used:
        - llvm.amdgcn.wmma.f32.16x16x16.f16 (FP16)
        - llvm.amdgcn.wmma.f32.16x16x16.bf16 (BF16)
        - llvm.amdgcn.wmma.i32.16x16x16.iu8 (INT8/UINT8, RDNA3+)
        - llvm.amdgcn.wmma.i32.16x16x16.iu4 (UINT4, RDNA3+)
        - llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8 (E4M3×E4M3, RDNA4 only)
        - llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8 (E5M2×E5M2, RDNA4 only)
        - llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8 (E4M3×E5M2, RDNA4 only)
        - llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8 (E5M2×E4M3, RDNA4 only)

    RDNA3 FP8 emulation details:
        - RDNA3: FP8 is supported through FP16 emulation:
          - FP8 emulation via FP16 or BF16 conversion on RDNA3
            - We use BF16 for e5m2 variants (wider exponent range),
              FP16 for e4m3 variants
          - We support size 16 (split into 4x size 4) and size 4
          - For size 8, we need to split into two WMMA operations
            - This is a packed format where we do 2x 16x16x16 operations
            - But the intrinsic itself only supports size 4
          - For size 16 FP8 - will split into 4x size 4 operations
            - We convert each FP8 chunk to target dtype and run WMMA
        - RDNA1/2: FP8 not supported

    Args:
        d: Output accumulator SIMD vector (modified in-place).
        a: First input matrix as SIMD vector.
        b: Second input matrix as SIMD vector.
        c: Accumulator matrix as SIMD vector.

    RDNA WMMA Fragment Requirements:
        RDNA WMMA is a wave-cooperative operation where each lane holds a fragment
        of the full matrix. For the 16×16×16 WMMA operation (M×N×K dimensions):

        Matrix Dimensions:
            - Matrix A: 16×16 (M×K) = 256 fp16/bf16 elements total
            - Matrix B: 16×16 (K×N) = 256 fp16/bf16 elements total
            - Matrix C/D: 16×16 (M×N) = 256 fp32 elements total

        Per-Lane Fragment Sizes (wave32 mode):
            - A fragment: 16 fp16/bf16 elements (full K=16 dimension per lane)
            - B fragment: 16 fp16/bf16 elements (full K=16 dimension per lane)
            - C/D fragment: 8 fp32 elements (M×N=256 distributed: 256/32 lanes = 8)

        This means the SIMD sizes passed to mma() for wave32 must be:
            - a.length = 16, b.length = 16, c.length = 8, d.length = 8

        LLVM Intrinsic Signatures (RDNA3/gfx11):
            - FP16: llvm.amdgcn.wmma.f32.16x16x16.f16(<16 x half>, <16 x half>, <8 x float>)
            - BF16: llvm.amdgcn.wmma.f32.16x16x16.bf16(<16 x i16>, <16 x i16>, <8 x float>)

            Note: BF16 fragments must be bitcast to <16 x i16> (packed BF16 as int16)
            before calling the intrinsic, not passed as <16 x bfloat>.

            RDNA4/gfx12 dropped the half-wave replication, so every A/B
            fragment is half as wide as on gfx11 (f16 <8 x half>, bf16
            <8 x i16>, iu8 <2 x i32>, iu4 i32, fp8 <2 x i32>): the K dimension
            is split across the wave (lanes 0-15 hold the low half, lanes 16-31
            the high half) instead of every lane holding the full row.

        Wave-Cooperative Distribution:
            - All 32 lanes in wave32 participate in the computation
            - A/B fragments: gfx11 lane holds the full row (16 elements) and
              lanes 16-31 mirror lanes 0-15; gfx12 lane holds half the row
              (8 elements) with no replication
            - C/D fragments: Output distributed across all lanes (8 elements each)

        Hardware Register Usage:
            - A/B: 8 VGPRs per lane (2 fp16/bf16 packed per 32-bit VGPR)
            - C/D: 8 VGPRs per lane (1 fp32 per 32-bit VGPR)

        Type and shape validation is performed by get_intrinsic_name() which calls
        _unsupported_mma_op() for invalid combinations.

    References:
        - RDNA3 WMMA: https://gpuopen.com/learn/wmma_on_rdna3/
        - RDNA3 ISA: https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf
        - RDNA4 ISA: https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf
          - Section 7.5 (8-bit Math) for FP8/BF8 details
    """

    @__parameter
    def get_intrinsic_name() -> String:
        # ===------------------------------------------------------------------===#
        # F32 = F16 * F16 + F32 (16x16x16)
        # Or
        # F32 = BF16 * BF16 + F32 (16x16x16)
        # ===------------------------------------------------------------------===#
        comptime if _has_type[
            (DType.float16, DType.float16, DType.float32, DType.float32)
        ](a.dtype, b.dtype, c.dtype, d.dtype) or _has_type[
            (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
        ](
            a.dtype, b.dtype, c.dtype, d.dtype
        ):
            comptime if _has_shape[(16, 16, 8, 8)](
                a.length, b.length, c.length, d.length
            ):
                comptime type_name = "f16" if a.dtype == DType.float16 else "bf16"
                return "llvm.amdgcn.wmma.f32.16x16x16." + type_name
            else:
                _unsupported_mma_op(d, a, b, c)
                return ""
        elif (
            a.dtype.is_float8()
            and b.dtype.is_float8()
            and c.dtype == .float32
            and d.dtype == .float32
        ):
            comptime if _is_amd_rdna4():
                # E4M3 formats map to fp8, E5M2 formats map to bf8
                comptime a_is_e4m3 = a.dtype in [
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fnuz,
                ]
                comptime a_is_e5m2 = a.dtype in [
                    DType.float8_e5m2,
                    DType.float8_e5m2fnuz,
                ]
                comptime b_is_e4m3 = b.dtype in [
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fnuz,
                ]
                comptime b_is_e5m2 = b.dtype in [
                    DType.float8_e5m2,
                    DType.float8_e5m2fnuz,
                ]

                comptime if a_is_e4m3 and b_is_e4m3:
                    return "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8"
                elif a_is_e5m2 and b_is_e5m2:
                    return "llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8"
                elif a_is_e4m3 and b_is_e5m2:
                    return "llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8"
                elif a_is_e5m2 and b_is_e4m3:
                    return "llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8"
                else:
                    _unsupported_mma_op(d, a, b, c)
                    return ""
            else:
                _unsupported_mma_op(d, a, b, c)
                return ""
        elif (
            _is_amd_rdna3()
            and a.dtype.is_float8()
            and b.dtype.is_float8()
            and c.dtype == .float32
            and d.dtype == .float32
        ):
            comptime if _has_shape[4](a.length, b.length, c.length, d.length):
                return "llvm.amdgcn.wmma.f32.16x16x16.f16"
            elif (
                a.length == 16
                and b.length == 16
                and c.length == 32
                and d.length == 32
            ):
                return "llvm.amdgcn.wmma.f32.16x16x16.f16"
            else:
                _unsupported_mma_op(d, a, b, c)
                return ""
        elif (
            (a.dtype == .int8 or a.dtype == .uint8)
            and (b.dtype == .int8 or b.dtype == .uint8)
            and c.dtype == .int32
            and d.dtype == .int32
        ):
            comptime if _is_amd_rdna3() or _is_amd_rdna4():
                comptime if _has_shape[(16, 16, 8, 8)](
                    a.length, b.length, c.length, d.length
                ):
                    return "llvm.amdgcn.wmma.i32.16x16x16.iu8"
                else:
                    _unsupported_mma_op(d, a, b, c)
                    return ""
            else:
                _unsupported_mma_op(d, a, b, c)
                return ""
        elif (
            a.dtype == ._uint4
            and b.dtype == ._uint4
            and c.dtype == .int32
            and d.dtype == .int32
        ):
            comptime if _is_amd_rdna3() or _is_amd_rdna4():
                comptime if _has_shape[(16, 16, 8, 8)](
                    a.length, b.length, c.length, d.length
                ):
                    return "llvm.amdgcn.wmma.i32.16x16x16.iu4"
                else:
                    _unsupported_mma_op(d, a, b, c)
                    return ""
            else:
                _unsupported_mma_op(d, a, b, c)
                return ""
        else:
            _unsupported_mma_op(d, a, b, c)
            return ""

    comptime if _is_amd_rdna3() and a.dtype.is_float8():
        comptime target_dtype = DType.bfloat16 if (
            a.dtype == .float8_e5m2 or a.dtype == .float8_e5m2fnuz
        ) else DType.float16
        comptime intrinsic_suffix = "bf16" if (
            a.dtype == .float8_e5m2 or a.dtype == .float8_e5m2fnuz
        ) else "f16"

        comptime if a.length == 16 and b.length == 16:
            var result = c.cast[.float32]()
            comptime intrinsic_name = "llvm.amdgcn.wmma.f32.16x16x16." + intrinsic_suffix

            comptime for i in range(4):
                comptime offset = i * 4
                var a_chunk = a.slice[4, offset=offset]()
                var b_chunk = b.slice[4, offset=offset]()
                var c_chunk = result.slice[4, offset=offset]()
                var a_converted = a_chunk.cast[target_dtype]()
                var b_converted = b_chunk.cast[target_dtype]()
                var r_chunk = llvm_intrinsic[intrinsic_name, SIMD[.float32, 4]](
                    a_converted, b_converted, c_chunk
                )
                result = result.insert[offset=offset](r_chunk)

            d = rebind[type_of(d)](result)
            return
        elif a.length == 4 and b.length == 4:
            var a_converted = a.cast[target_dtype]()
            var b_converted = b.cast[target_dtype]()

            var r = llvm_intrinsic[
                "llvm.amdgcn.wmma.f32.16x16x16." + intrinsic_suffix,
                SIMD[c.dtype, c.length],
            ](a_converted, b_converted, c)
            d = rebind[type_of(d)](r)
            return

    comptime if (
        a.length == 16 and b.length == 16 and c.length == 8 and d.length == 8
    ):
        comptime intrinsic_name = get_intrinsic_name()

        # gfx12 halves the A/B width vs gfx11 (see the struct docstring), so on
        # gfx12 pick this lane's 8-wide half of the replicated 16-wide fragment.
        var second_half = _is_amd_rdna4() and (lane_id() & 16) != 0

        comptime if a.dtype == .float16:
            comptime if _is_amd_rdna4():
                var a8 = a.slice[8, offset=8]() if second_half else a.slice[
                    8, offset=0
                ]()
                var b8 = b.slice[8, offset=8]() if second_half else b.slice[
                    8, offset=0
                ]()
                var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                    a8, b8, c
                )
                d = rebind[type_of(d)](r)
            else:
                var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                    a, b, c
                )
                d = rebind[type_of(d)](r)
        elif a.dtype == .bfloat16:
            # BF16 fragments are passed to the intrinsic as packed int16.
            comptime if _is_amd_rdna4():
                var ai = bitcast[.int16, 16](a)
                var bi = bitcast[.int16, 16](b)
                var a8 = ai.slice[8, offset=8]() if second_half else ai.slice[
                    8, offset=0
                ]()
                var b8 = bi.slice[8, offset=8]() if second_half else bi.slice[
                    8, offset=0
                ]()
                var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                    a8, b8, c
                )
                d = rebind[type_of(d)](r)
            else:
                var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                    bitcast[.int16, 16](a), bitcast[.int16, 16](b), c
                )
                d = rebind[type_of(d)](r)
        elif a.dtype.is_float8():
            # RDNA4-only native fp8/bf8: 8 fp8 per lane packed as `<2 x i32>`.
            var a8 = a.slice[8, offset=8]() if second_half else a.slice[
                8, offset=0
            ]()
            var b8 = b.slice[8, offset=8]() if second_half else b.slice[
                8, offset=0
            ]()
            var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                bitcast[.int32, 2](a8), bitcast[.int32, 2](b8), c
            )
            d = rebind[type_of(d)](r)
        elif a.dtype == .int8 or a.dtype == .uint8:
            # iu8: the interleaved `i1` signedness and trailing `i1` clamp are
            # immediates; A/B pack to `<4 x i32>` (gfx11) or `<2 x i32>` (gfx12).
            comptime a_signed = a.dtype == DType.int8
            comptime b_signed = b.dtype == DType.int8
            comptime if _is_amd_rdna4():
                var a8 = a.slice[8, offset=8]() if second_half else a.slice[
                    8, offset=0
                ]()
                var b8 = b.slice[8, offset=8]() if second_half else b.slice[
                    8, offset=0
                ]()
                var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                    a_signed,
                    bitcast[.int32, 2](a8),
                    b_signed,
                    bitcast[.int32, 2](b8),
                    c,
                    False,
                )
                d = rebind[type_of(d)](r)
            else:
                var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                    a_signed,
                    bitcast[.int32, 4](a),
                    b_signed,
                    bitcast[.int32, 4](b),
                    c,
                    False,
                )
                d = rebind[type_of(d)](r)
        else:
            # iu4 (uint4): unsigned; A/B pack to `<2 x i32>` (gfx11) or `i32`.
            comptime if _is_amd_rdna4():
                var a8 = a.slice[8, offset=8]() if second_half else a.slice[
                    8, offset=0
                ]()
                var b8 = b.slice[8, offset=8]() if second_half else b.slice[
                    8, offset=0
                ]()
                var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                    False,
                    bitcast[.int32, 1](a8),
                    False,
                    bitcast[.int32, 1](b8),
                    c,
                    False,
                )
                d = rebind[type_of(d)](r)
            else:
                var r = llvm_intrinsic[intrinsic_name, SIMD[c.dtype, 8]](
                    False,
                    bitcast[.int32, 2](a),
                    False,
                    bitcast[.int32, 2](b),
                    c,
                    False,
                )
                d = rebind[type_of(d)](r)
    else:
        _unsupported_mma_op(d, a, b, c)
