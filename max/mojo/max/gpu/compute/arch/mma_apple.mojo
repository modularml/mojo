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
"""Apple Silicon MMA implementation for matrix multiply-accumulate operations.

This module provides two simdgroup_matrix MMA shapes:

- 16x16x16 (`_mma_apple`): Apple M5 only (Metal 4.0 / AIR 2.8.0), float and
  integer-widening.
- 8x8 (`_mma_apple_8x8`): all Apple GPU generations (M1-M5), float-only.

Supported operations:
- Float multiply-accumulate (16x16): {F16, BF16, F32, E4M3, E5M2} inputs, F32
  accumulator
- Integer widening multiply-accumulate (16x16): {I8, U8} inputs, I32/U32 accumulator
- Float multiply-accumulate (8x8): {F16, BF16, F32} inputs, F32
  accumulator
"""

from std.sys import llvm_intrinsic

from std._gpu import lane_id

# Import helper functions from parent module
from ..mma import _has_shape, _unsupported_mma_op


@always_inline
def _apple_frag_layout(tid: Int) -> Tuple[Int, Int]:
    """Returns (row_lo, col_base) for the given simdgroup thread."""
    return (
        ((tid & 7) // 2) + ((tid & 16) >> 2),
        ((tid & 1) << 2) + (tid & 8),
    )


@always_inline
def _apple_frag_layout_8x8(tid: Int) -> Tuple[Int, Int]:
    """Returns (row_lo, col_base) for the given simdgroup thread (8x8).

    The lane owns (row_lo, col_base) and (row_lo, col_base + 1) -- two
    consecutive columns in one row. Ground-truthed via Metal
    `thread_elements()`.
    """
    return (
        ((tid & 6) >> 1) + ((tid & 16) >> 2),
        ((tid & 1) << 1) + ((tid & 8) >> 1),
    )


@always_inline
def apple_mma_load[
    dtype: DType,
](
    ptr: Pointer[Scalar[dtype], ...],
    row_stride: Int,
    col_stride: Int = 1,
) -> SIMD[dtype, 8]:
    """Loads a 16x16 matrix fragment for the current simdgroup thread.

    Parameters:
        dtype: Element type of the matrix.

    Args:
        ptr: Pointer to the top-left corner of the 16x16 tile.
        row_stride: Distance between consecutive rows in the buffer.
        col_stride: Distance between consecutive columns within a row.

    Returns:
        SIMD vector of 8 elements for this thread's fragment.
    """
    var tid = Int(lane_id())
    var layout = _apple_frag_layout(tid)
    var row_lo = layout[0]
    var col_base = layout[1]

    # If row-elems are contiguous, vectorize load
    if col_stride == 1:
        var lo = ptr.unsafe_offset(row_lo * row_stride + col_base).unsafe_load[
            width=4
        ]()
        var hi = ptr.unsafe_offset(
            (row_lo + 8) * row_stride + col_base
        ).unsafe_load[width=4]()
        return lo.join(hi)
    else:
        var frag = SIMD[dtype, 8]()
        for el in range(8):
            var row = row_lo + (Int(el > 3) * 8)
            var col = col_base + (el & 3)
            frag[el] = ptr[unsafe_offset=row * row_stride + col * col_stride]
        return frag


@always_inline
def apple_mma_store[
    dtype: DType,
](
    ptr: Pointer[mut=True, Scalar[dtype], ...],
    row_stride: Int,
    frag: SIMD[dtype, 8],
    col_stride: Int = 1,
):
    """Stores a 16x16 matrix fragment from the current simdgroup thread.

    Parameters:
        dtype: Element type of the matrix.

    Args:
        ptr: Pointer to the top-left corner of the 16x16 tile.
        row_stride: Distance between consecutive rows in the buffer.
        frag: SIMD vector of 8 elements to store.
        col_stride: Distance between consecutive columns within a row.
    """
    var tid = Int(lane_id())
    var layout = _apple_frag_layout(tid)
    var row_lo = layout[0]
    var col_base = layout[1]

    # If row-elems are contiguous, vectorize store
    if col_stride == 1:
        ptr.unsafe_offset(row_lo * row_stride + col_base).unsafe_store(
            frag.slice[4, offset=0]()
        )
        ptr.unsafe_offset((row_lo + 8) * row_stride + col_base).unsafe_store(
            frag.slice[4, offset=4]()
        )
    else:
        for el in range(8):
            var row = row_lo + (Int(el > 3) * 8)
            var col = col_base + (el & 3)
            ptr[unsafe_offset=row * row_stride + col * col_stride] = frag[el]


@always_inline
def apple_mma_load_8x8[
    dtype: DType,
](
    ptr: Pointer[Scalar[dtype], ...],
    row_stride: Int,
    col_stride: Int = 1,
) -> SIMD[dtype, 2]:
    """Loads an 8x8 matrix fragment for the current simdgroup thread.

    Parameters:
        dtype: Element type of the matrix.

    Args:
        ptr: Pointer to the top-left corner of the 8x8 tile.
        row_stride: Distance between consecutive rows in the buffer.
        col_stride: Distance between consecutive columns within a row.

    Returns:
        SIMD vector of 2 elements for this thread's fragment.
    """
    var tid = Int(lane_id())
    var layout = _apple_frag_layout_8x8(tid)
    var row_lo = layout[0]
    var col_base = layout[1]

    if col_stride == 1:
        return ptr.unsafe_offset(row_lo * row_stride + col_base).unsafe_load[
            width=2
        ]()
    else:
        var frag = SIMD[dtype, 2]()
        for el in range(2):
            var col = col_base + el
            frag[el] = ptr[unsafe_offset=row_lo * row_stride + col * col_stride]
        return frag


@always_inline
def apple_mma_store_8x8[
    dtype: DType,
](
    ptr: Pointer[mut=True, Scalar[dtype], ...],
    row_stride: Int,
    frag: SIMD[dtype, 2],
    col_stride: Int = 1,
):
    """Stores an 8x8 matrix fragment from the current simdgroup thread.

    Parameters:
        dtype: Element type of the matrix.

    Args:
        ptr: Pointer to the top-left corner of the 8x8 tile.
        row_stride: Distance between consecutive rows in the buffer.
        frag: SIMD vector of 2 elements to store.
        col_stride: Distance between consecutive columns within a row.
    """
    var tid = Int(lane_id())
    var layout = _apple_frag_layout_8x8(tid)
    var row_lo = layout[0]
    var col_base = layout[1]

    if col_stride == 1:
        ptr.unsafe_offset(row_lo * row_stride + col_base).unsafe_store(frag)
    else:
        for el in range(2):
            var col = col_base + el
            ptr[unsafe_offset=row_lo * row_stride + col * col_stride] = frag[el]


@always_inline
def _mma_apple(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    _mma_apple_transposable(d, a, b, c, False, False)


@always_inline
def _mma_apple_transposable(
    mut d: SIMD,
    a: SIMD,
    b: SIMD,
    c: SIMD,
    transpose_a: Bool,
    transpose_b: Bool,
):
    """Variant of `_mma_apple` with runtime transpose flags."""
    comptime assert _has_shape[8](
        a.length, b.length, c.length, d.length
    ), "Apple MMA requires 8-element fragments (16x16 / 32 threads)"

    comptime assert d.dtype in (
        DType.float32,
        DType.int32,
        DType.uint32,
    ) and c.dtype in (
        DType.float32,
        DType.int32,
        DType.uint32,
    ), "Apple MMA accumulator (C and D) must be 32-bit"

    comptime assert c.dtype == d.dtype, "Apple MMA C and D types must match"

    # fp8 (E4M3/E5M2) reuses the same float multiply-accumulate intrinsic;
    # KGEN lowers the native fp8 fragment to AIR's `<8 x i8>` storage form.
    comptime _valid_float_dtypes = (
        DType.float16,
        DType.bfloat16,
        DType.float32,
        DType.float8_e4m3fn,
        DType.float8_e5m2,
    )
    comptime _valid_float_input = (
        a.dtype in _valid_float_dtypes and b.dtype in _valid_float_dtypes
    )

    comptime _valid_int_input = (
        a.dtype in (DType.int8, DType.uint8)
        and b.dtype in (DType.int8, DType.uint8)
    )

    comptime if _valid_float_input and d.dtype == .float32:
        d = rebind[type_of(d)](
            llvm_intrinsic[
                "llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate",
                SIMD[.float32, 8],
            ](a, transpose_a, b, transpose_b, c)
        )

    elif _valid_int_input and d.dtype in (DType.int32, DType.uint32):
        d = rebind[type_of(d)](
            llvm_intrinsic[
                "llvm.air.simdgroup_matrix_16x16x16_widening_multiply_accumulate",
                SIMD[d.dtype, 8],
            ](a, transpose_a, b, transpose_b, c)
        )

    else:
        _unsupported_mma_op(d, a, b, c)


@always_inline
def _mma_apple_8x8(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """Performs an 8x8 simdgroup_matrix multiply-accumulate: D = A @ B + C.

    Available on all Apple GPU generations (M1-M5). Float-only: inputs in
    {F16, BF16, F32}, accumulator (C and D) is F32. The intrinsic takes only
    (a, b, c) -- there is no transpose variant.
    """
    comptime assert _has_shape[2](
        a.length, b.length, c.length, d.length
    ), "Apple 8x8 MMA requires 2-element fragments (8x8 / 32 threads)"

    comptime assert (
        c.dtype == .float32 and d.dtype == .float32
    ), "Apple 8x8 MMA accumulator (C and D) must be F32"

    comptime _valid_float_input = (
        a.dtype in (DType.float16, DType.bfloat16, DType.float32)
        and b.dtype in (DType.float16, DType.bfloat16, DType.float32)
    )

    comptime if _valid_float_input:
        # KGEN selects the AIR MMA variant by operand width, not name. The
        # vec2 fragments emit a `.v2bf16` form that M1/M2 mishandle (bf16 read
        # as fp16, corrupting results); widening to 64 forces the `.v64bf16`
        # form MSL emits, which every Apple GPU handles. air-lld collapses the
        # sparse `<64x>` back to the per-lane fragment, so this is perf-neutral.
        var a_wide = SIMD[a.dtype, 64](0)
        var b_wide = SIMD[b.dtype, 64](0)
        var c_wide = SIMD[.float32, 64](0)

        comptime for s in range(2):
            a_wide[s] = a[s]
            b_wide[s] = b[s]
            c_wide[s] = rebind[Float32](c[s])

        var d_wide = llvm_intrinsic[
            "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
            SIMD[.float32, 64],
        ](a_wide, b_wide, c_wide)

        comptime for s in range(2):
            d[s] = rebind[Scalar[d.dtype]](d_wide[s])
    else:
        _unsupported_mma_op(d, a, b, c)
