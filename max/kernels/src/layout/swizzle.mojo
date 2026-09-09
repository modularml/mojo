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
"""Defines swizzle layouts for optimizing memory access patterns.

This module is designed for use in shared memory, especially in GPU
kernels, to reduce bank conflicts.  It provides tools to create and
apply swizzle transformations to memory indices.  Swizzling
rearranges memory access order to distribute accesses across
different memory banks.  This mitigates bank contention and improves
memory access efficiency.

Module components:
  - `Swizzle` struct: Represents a swizzle transformation with
    configurable bits, base, and shift parameters.
  - Helper functions: `make_ldmatrix_swizzle`, `make_swizzle` create
    predefined swizzle patterns. These are optimized for scenarios
    like `ldmatrix` instructions and general 2D memory access.
  - `ComposedLayout` struct: Combines a base layout with a swizzle
    layout for complex memory access optimizations.

Primary use case: GPU kernel development where shared memory bank
conflicts can degrade performance.  Applying swizzle layouts
optimizes memory access patterns for higher throughput.
"""

from std.sys import simd_width_of, size_of
from std.math.uutils import udivmod

from std.bit import log2_floor
from max.gpu.host.nvidia.tma import TensorMapSwizzle

from .int_tuple import flatten, IntTuple
from .layout import Layout

# ===-----------------------------------------------------------------------===#
# Motivation of thread swizzling                                               #
# ===-----------------------------------------------------------------------===#

# Memory access pattern without swizzling can lead to bank conflicts
# in shared memory, especially in GPU kernels.  Consider a vector of
# 4 FP32 elements (|--------|).

# Assumptions for the following example:
# * Matrix dimensions BM x BK = 64 x 16
# * Matrix layout is row-major

# A BM x BN tile in global memory (8 rows shown):
#
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ... |-------|--------|--------|--------| ...
#     ...

# This tile is copied to shared memory. How to map threads to vectors?

# Naive thread mapping method:
# Global memory view:
#
#     ... |-- T0 --|-- T1 --|-- T2 --|-- T3 --| ...
#     ... |-- T4 --|-- T5 --|---T6 --|-- T7 --| ...
#     ... |-- T8 --|-- T9 --|--------|--------| ...
#     ... |--------|--------|--------|--------| ...
#     ... |--------|--------|--------|--------| ...
#     ... |--------|--------|--------|--------| ...
#     ... |--------|--------|--------|--------| ...
#     ... |--------|--------|--------|-- T31--| ...

# Shared memory view (row_major(BM, BK), 8 vectors/row for bank conflict):
# (Each row covers all 32 banks, showing potential bank conflicts)
#
#     |-- T0 --|-- T1 --|-- T2 --|-- T3 --|-- T4 --|-- T5 --|---T6 --|-- T7 --|
#     |-- T8 --|-- T9 --|--------|--------|--------|--------|--------|-- T15--|
#     |--------|--------|--------|--------|--------|--------|--------|--------|
#     |--------|--------|--------|--------|--------|--------|--------|-- T31--|

# Memory access phases (wavefronts):
# * 128B access from 32 banks is a wavefront.
# * Wavefronts are processed in phases/cycles.
# * No bank conflict assumption:
#   - Scalar load/store: warp requests in 1 phase/wavefront.
#   - V2 load/store: 16 threads/phase/wavefront.
#   - V4 load/store: 8 threads/phase/wavefront.
#   - Bank conflict within a phase degrades performance.

# Loads complete in 4 phases: T0-T7, T8-T15, T16-T23, T24-T31.
# T0 and T8 map to same banks, but loads are in separate phases.
# No bank conflict in this naive case.

# ==============================================================================
#
# TF32 MMA (Matrix Multiply Accumulate) example:
#
# M x N x K = 16 x 8 x 8 matrix multiplication
#
# Thread map for the 16 x 8 matrix:
# | Ti |: a scalar mapped to thread Ti
#
#     | T0 | T1 | T2 | T3 || T0 | T1 | T2 | T3 |
#     | T4 | T5 | T6 | T7 || T4 | T5 | T6 | T7 |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | T31|| -- | -- | -- | T31|
#     +----------------------------------------+
#     | T0 | T1 | T2 | T3 || T0 | T1 | T2 | T3 |
#     | T4 | T5 | T6 | T7 || T4 | T5 | T6 | T7 |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | -- || -- | -- | -- | -- |
#     | -- | -- | -- | T31|| -- | -- | -- | T31|
#
#     mma(
#         d0, d1, d2, d3,
#         a0, a1, a2, a3,       # Four scalars mapped to T0
#         b0, b1,
#         c0, c1, c2, c3,
#     )

# How can T0 load its four scalars efficiently?
# Four scalar loads are inefficient.

# `ldmatrix`: vector load + warp shuffle conceptually.

# `ldmatrix.x1` (8x4 matrix, 32 bits/element):
#
#     lane_addr     8 x 4 matrix (32-bit elements)
#
#       T0    ->    | T0 | T1 | T2 | T3 |
#       T1    ->    | T4 | T5 | T6 | T7 |
#       T2    ->    | -- | -- | -- | -- |
#       T3    ->    | -- | -- | -- | -- |
#       T4    ->    | -- | -- | -- | -- |
#       T5    ->    | -- | -- | -- | -- |
#       T6    ->    | -- | -- | -- | -- |
#       T7    ->    | -- | -- | -- | T31|
#
#     instruction:  `ld_matrix...x1... reg, Ti_address`
#
#     * T0-T7 lane addresses used in vector loads. Rest ignored.
#     * T0-T7 perform vector loads.
#     * `reg` gets scalar value for Ti.

# For 16 x 8 matrix in tf32 mma:
#
#     instruction:  `ld_matrix...x4... reg0, reg1, reg2, reg3, Ti_address`
#
#     * All threads' lane addresses used in vector loads.
#     * Outputs four registers.

# Problem: How does naive thread map work with `ldmatrix`?

# Recap: Naive Thread map for shared memory store:
# Each row covers 32 banks.
#
#     |-- T0 --|-- T1 --|-- T2 --|-- T3 --|-- T4 --|-- T5 --|---T6 --|-- T7 --|
#     |-- T8 --|-- T9 --|--------|--------|--------|--------|--------|-- T15--|
#     |--------|--------|--------|--------|--------|--------|--------|--------|
#     |--------|--------|--------|--------|--------|--------|--------|-- T31--|

# Assign lane address for `ld_matrix`. First 8x4 matrix in 16x8.
# T1's lane_addr points to 2nd row of BMxBK = 64x16 tile.
#
#     |-- T0 --|--------|--------|--------|-- T1 --|--------|--------|--------|
#     |-- T2 --|--------|--------|--------|-- T3 --|--------|--------|--------|
#     |-- T4 --|--------|--------|--------|-- T5 --|--------|--------|--------|
#     |-- T6 --|--------|--------|--------|-- T7 --|--------|--------|--------|
#
#                !!!!!!!! 4-way bank conflicts per phase !!!!!!!!

# ==============================================================================
#
# Thread Swizzling:
#
# References:
# https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
# https://github.com/NVIDIA/cutlass/blob/5c447dd84f8ae0e1d48ff9a2eae26ce8c4958101/python/pycute/swizzle.py#L48-L59

# `Swizzle[bits, base, shift](index)` functor:
#
#     Generic swizzle functor.
#     0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
#                                   ^--^  Base: # least-sig bits constant
#                      ^-^       ^-^      Bits: # bits in mask (YYY)
#                        ^---------^      Shift: shift distance for YYY mask
#                                           (pos: right shift, neg: left)
#
#     e.g., `Swizzle[2, 0, shift]`
#     0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZ
#     Result:
#     0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAA where AA = ZZ xor YY

# XOR refresh:
#   [0|1] ^ 0 = [0|1] (no change)
#   [0|1] ^ 1 = [1|0] (flip)
#
#   A   ^ 1    : swap odd and even indices.
#   Ax  ^ 10   : swap two elements each time.
#   Axx ^ 100  : swap 2**2 elements each time.
#   ...

# `Swizzle[2, 0, 3]` for shared memory store:
#
#     lane_id: xxxxx
#              ^^    (shift 3 bits, extract 2-bit mask)
#
#  00xxx ^ 00  |-- T0 --|-- T1 --|-- T2 --|-- T3 --|-- T4 --|-- T5 --|-- T6 --|-- T7 --|
#  01xxx ^ 01  |-- T9 --|-- T8 --|-- T11--|-- T10--|-- T13--|-- T12--|-- T15--|-- T14--|
#  10xxx ^ 10  |-- T18--|-- T19--|-- T16--|-- T17--|-- T22--|-- T23--|-- T20--|-- T21--|
#  11xxx ^ 11  |-- T27--|-- T26--|-- T25--|-- T24--|-- T31--|-- T30--|-- T29--|-- T28--|

# ===-----------------------------------------------------------------------===#
# Helpers                                                                      #
# ===-----------------------------------------------------------------------===#


@always_inline
def shiftr(a: Scalar, s: Scalar[a.dtype]) -> Scalar[a.dtype]:
    """Shift right/left based on sign of shift for scalars.

    Scalar version of `shiftr`.  Right shift if `s` is positive,
    left shift if `s` is negative.

    Args:
        a: The scalar value to shift.
        s: The scalar shift amount. Positive for right, negative left.

    Returns:
        The shifted scalar value.
    """
    return a >> s if s > 0 else a << -s


@always_inline
def shiftl(a: Scalar, s: Scalar[a.dtype]) -> Scalar[a.dtype]:
    """Shift left/right based on sign of shift for scalars.

    Scalar version of `shiftl`.  Left shift if `s` is positive,
    right shift if `s` is negative.

    Args:
        a: The scalar value to shift.
        s: The scalar shift amount. Positive for left, negative right.

    Returns:
        The shifted scalar value.
    """
    return a << s if s > 0 else a >> -s


# ===-----------------------------------------------------------------------===#
# Swizzle                                                                      #
# ===-----------------------------------------------------------------------===#


struct Swizzle(Copyable, Deinitable, TrivialRegisterPassable, Writable):
    """Swizzle functor for memory access pattern optimization.

    Implements a swizzling pattern to reduce bank conflicts in shared
    memory accesses.  It XORs specific bits of memory indices based
    on configurable parameters.

    Swizzle operation:
    Given index `i`, and Swizzle[bits, base, shift]:

    1. Extract `bits` number of bits from `i` starting from position
       `base + max(shift, 0)`. Let's call this `YYY`.
    2. Extract `bits` number of bits from `i` starting from position
       `base - min(shift, 0)`. Let's call this `ZZZ`.
    3. Result is `i ^ (YYY shifted by 'shift' positions)`.

    Example (Swizzle[2, 0, 3]):
    Input index bits:  `xxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxxx`
    Output index bits: `xxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxxx`
    where `AA = ZZ ^ YY`.

    Attributes:
        bits (Int): Number of bits in the mask (YYY).
        base (Int): Number of least significant bits to keep constant.
        shift (Int): Shift distance for the mask (positive: right,
                     negative: left).
        yyy_mask (Int): Mask for the bits to be shifted (YYY).
        zzz_mask (Int): Mask for the target bits (ZZZ).
    """

    comptime has_shape = False
    """Indicates if layout has shape. Swizzle always False."""

    var bits: Int
    """Number of bits in the mask."""
    var base: Int
    """Number of least significant bits to keep constant."""
    var shift: Int
    """Distance to shift the mask (pos right, neg left)."""
    var yyy_mask: Int
    """Mask for the bits to be shifted."""
    var zzz_mask: Int
    """Mask for the target bits."""

    @always_inline
    def __init__(out self, bits: Int, base: Int, shift: Int):
        """Initialize a Swizzle object.

        Configures the swizzle operation based on bits, base, and
        shift parameters.

        Args:
            bits: Number of bits in the mask.
            base: Least significant bits to keep constant.
            shift: Distance to shift the mask.
        """
        if not __is_run_in_comptime_interpreter:
            assert (
                bits >= 0 and base >= 0
            ), "Require non-negative mask bits and base"

            assert abs(shift) >= bits, "shift should be less than bits"

        self.bits = bits
        self.base = base
        self.shift = shift
        self.yyy_mask = ((1 << self.bits) - 1) << (
            self.base + max(self.shift, 0)
        )
        self.zzz_mask = ((1 << self.bits) - 1) << (
            self.base - min(self.shift, 0)
        )

    @always_inline
    def __call__(self, index: IntTuple) -> Int:
        """Apply swizzle to an IntTuple index.

        Unwraps the IntTuple and applies the swizzle to the integer
        value.

        Args:
            index: The IntTuple index to swizzle.

        Returns:
            The swizzled index value.
        """
        return self.__call__(index.value())

    @always_inline
    def __call__(self, offset: Int) -> Int:
        """Apply swizzle to an integer offset.

        Performs the swizzle operation on an integer offset to
        rearrange memory access patterns.

        Args:
            offset: The integer offset to swizzle.

        Returns:
            The swizzled offset value.
        """
        return offset ^ shiftr(offset & self.yyy_mask, self.shift)

    @always_inline
    def __call__(self, offset: Scalar) -> Scalar[offset.dtype]:
        """Apply swizzle to a scalar offset.

        Scalar version of the swizzle operation.  Applies swizzle to
        a scalar offset.

        Args:
            offset: The scalar offset to swizzle.

        Returns:
            The swizzled scalar value.
        """
        return offset ^ shiftr(
            offset & Scalar[offset.dtype](self.yyy_mask),
            Scalar[offset.dtype](self.shift),
        )

    @always_inline
    def size(self) -> Int:
        """Get the size of the swizzle pattern.

        Calculates the size of the memory region affected by the
        swizzle pattern.

        Returns:
            The size of the swizzle pattern.
        """
        return 1 << (self.bits + self.base + abs(self.shift))

    @always_inline
    def cosize(self) -> Int:
        """Get the cosize of the swizzle pattern.

        Cosize is the same as size for swizzle layouts, representing
        the output size.

        Returns:
            The cosize of the swizzle pattern (same as size).
        """
        return self.size()

    def write_to(self, mut writer: Some[Writer]):
        """Write the swizzle parameters to a writer.

        Outputs the swizzle parameters (bits, base, shift) in a
        tuple format.

        Args:
            writer: The writer to write to.
        """
        writer.write("(", self.bits, ",", self.base, ",", self.shift, ")")


@always_inline
def make_ldmatrix_swizzle[
    dtype: DType, row_size: Int, log2_vector_width: Int = 0
]() -> Swizzle:
    """Make swizzle to avoid bank conflict for ldmatrix ops.

    Creates a swizzle pattern optimized for `ldmatrix` operations.
    Minimizes bank conflicts in shared memory for these operations.
    Calculates swizzle parameters based on data type and row size.

    Parameters:
        dtype: The data type of the elements.
        row_size: Size of each row in elements.
        log2_vector_width: Log2 of the vector width (default: 0).

    Returns:
        A `Swizzle` object configured for `ldmatrix`.
    """
    # For Nvidia GPU, 32 banks of 4B each.
    comptime bytes_32_banks = 128
    comptime type_size = size_of[dtype]()
    comptime bytes_row = row_size * type_size

    comptime assert (
        bytes_row % bytes_32_banks == 0 or bytes_32_banks % bytes_row == 0
    ), (
        "Row sizes should be multiples of 32 banks, or multiple"
        " rows should fit in 32 banks."
    )

    # `ldmatrix` loads 8x4 matrix (each row 4x4B=16B vector).
    # Stride between vectors is `row_size`.
    # Conflict ways: banks spanned by 8x4 matrix / 32.
    # E.g., fp32, row_size=16, rows 0, 2, 4, 6 in `ld_matrix` conflict.
    comptime conflict_ways = min(8 * row_size * type_size // bytes_32_banks, 8)
    comptime bits = log2_floor(conflict_ways)

    # Apply one swizzle bit pattern (^01) to same row if row > 32 banks
    # or multiple rows fit in 32 banks.
    comptime simd_size = simd_width_of[dtype]()
    comptime shifts = log2_floor(max(row_size // simd_size, 8))

    return Swizzle(bits, log2_vector_width, shifts)


@always_inline
def make_swizzle[num_rows: Int, row_size: Int, access_size: Int]() -> Swizzle:
    """Create a 2D swizzle to avoid bank conflicts.

    Generates a swizzle pattern for 2D memory layout to minimize
    bank conflicts in shared memory access.

    Parameters:
        num_rows: Number of rows in the minimum access pattern.
        row_size: Size of each row in elements.
        access_size: Number of elements accessed at once.

    Returns:
        A `Swizzle` object for 2D memory access.
    """
    comptime bits = log2_floor(num_rows)
    comptime base = log2_floor(access_size)
    comptime shifts = log2_floor(row_size) - base

    comptime assert shifts > 0, "Negative shifts in swizzling likely a bug."

    return Swizzle(bits, base, shifts)


@always_inline
def make_swizzle[dtype: DType, mode: TensorMapSwizzle]() -> Swizzle:
    """Create swizzle based on predefined swizzle modes.

    Returns a swizzle pattern based on standard modes (32B, 64B,
    128B, none), adjusted for data type.

    Parameters:
        dtype: The data type of the elements.
        mode: The swizzle mode to use (TensorMapSwizzle enum).

    Returns:
        A `Swizzle` object configured by the specified mode.
    """
    comptime type_size = size_of[dtype]()

    comptime assert mode in (
        TensorMapSwizzle.SWIZZLE_128B,
        TensorMapSwizzle.SWIZZLE_64B,
        TensorMapSwizzle.SWIZZLE_32B,
        TensorMapSwizzle.SWIZZLE_NONE,
    ), "Only support 32B, 64B, 128B, or no swizzle"
    return Swizzle(Int(mode), log2_floor(16 // type_size), 3)


# ===-----------------------------------------------------------------------===#
# Composed Layout                                                              #
# ===-----------------------------------------------------------------------===#


struct ComposedLayout[offset: Optional[Int] = 0](Copyable):
    """Layout composed of a Layout and a Swizzle applied sequentially.

    Combines a base layout with a swizzle. The output of the layout is
    input to the swizzle, with an optional offset in between.

    Parameters:
        offset: Optional offset between layouts (default: 0).
    """

    comptime has_shape = True
    """True because the base layout always has a shape."""

    var layout_a: Layout
    """The base layout to apply."""
    var layout_b: Swizzle
    """The swizzle to apply."""

    @always_inline
    def __init__(out self, var layout_a: Layout, layout_b: Swizzle):
        """Initialize ComposedLayout with a layout and swizzle.

        Args:
            layout_a: The base layout.
            layout_b: The swizzle.
        """
        comptime assert (
            not Self.offset or Self.offset.value() >= 0
        ), "Requires non-negative offset if present"
        self.layout_a = layout_a^
        self.layout_b = layout_b

    @always_inline
    def __call__(self, idx: IntTuple) -> Int:
        """Apply composed layout to an index.

        Applies the layout, then adds offset, then applies the swizzle.

        Args:
            idx: The index to transform.

        Returns:
            The transformed index.
        """
        var offset_val = Self.offset.value() if Self.offset else 0
        return self.layout_b(offset_val + self.layout_a(idx))

    @always_inline
    def __call__(self, idx: IntTuple, offset_val: Int) -> Int:
        """Apply composed layout with runtime offset.

        Applies the layout, then adds runtime `offset_val`, then the swizzle.
        Static offset must not be set when using runtime offset.

        Args:
            idx: The index to transform.
            offset_val: Runtime offset to apply.

        Returns:
            The transformed index.
        """
        comptime assert (
            not Self.offset
        ), "Static offset set; runtime offset not allowed."
        return self.layout_b(offset_val + self.layout_a(idx))

    @always_inline
    def size(self) -> Int:
        """Get the size of the composed layout.

        Returns the size of the base layout.

        Returns:
            The size of the base layout.
        """
        return self.layout_a.size()

    @always_inline
    def cosize(self) -> Int:
        """Get the cosize of the composed layout.

        Returns the cosize of the swizzle.

        Returns:
            The cosize of the swizzle.
        """
        return self.layout_b.cosize()


@always_inline
def eval_composed[
    composed_layout: ComposedLayout
](idx: Int, offset: Int = 0) -> Int:
    """Evaluate a composed layout with swizzle.

    Applies the base layout, adds an optional offset, and then applies
    the swizzle.

    Parameters:
        composed_layout: The composed layout to evaluate.

    Args:
        idx: The input index to transform.
        offset: Optional offset to apply between layouts (default: 0).

    Returns:
        The transformed index after applying both layouts.
    """
    var a_idx = idx
    var b_idx = 0

    comptime shape_a = flatten(composed_layout.layout_a.shape)
    comptime stride_a = flatten(composed_layout.layout_a.stride)

    var coord_i: Int
    comptime for i in range(len(stride_a)):
        comptime s = shape_a[i].value()
        comptime st = stride_a[i].value()
        a_idx, coord_i = udivmod(a_idx, s)
        b_idx += coord_i * st

    b_idx += offset

    comptime layout_b = composed_layout.layout_b
    return layout_b(b_idx)
