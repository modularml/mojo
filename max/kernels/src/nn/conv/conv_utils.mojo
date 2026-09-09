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

"""Utilities for direct convolution kernels on CPU.

Provides convolution shape description, L2 tiling heuristics, micro-kernel
shape selection, and work partitioning helpers used by the direct and
im2col convolution implementations.
"""

from std.math import align_down, ceildiv, sqrt
from std.sys._build import is_debug_build
from std.sys.info import CompilationTarget, simd_width_of, size_of

from layout import IntTuple, TileTensor, UNKNOWN_VALUE
from layout.coord import Coord, DynamicCoord, coord, coord_to_index_list
from linalg.utils import partition_work

from std.utils.index import Index, IndexList


# ===----------------------------------------------------------------------=== #
# Epilogue Helper                                                              #
# ===----------------------------------------------------------------------=== #


# Elementwise epilogue signature
comptime elementwise_epilogue_type = def[rank: Int](
    coords: IndexList[rank],
    f_size: Int,
) capturing -> None

comptime elementwise_simd_epilogue_type = def[
    dtype: DType, rank: Int, width: SIMDLength, alignment: Int = 1
](IndexList[rank], SIMD[dtype, width]) capturing -> None


# ===----------------------------------------------------------------------=== #
# Wrapper for  Convolution Shape                                               #
# ===----------------------------------------------------------------------=== #


struct ConvShape[rank: Int](TrivialRegisterPassable):
    """A shape struct describing the convolution dimensions.

    Parameters:
        rank: Spatial rank of the convolution (1, 2, or 3).
    """

    var n: Int  # Input batch size.

    var input_dims: DynamicCoord[.int64, Self.rank]  # Ex H and W for 2D
    var output_dims: DynamicCoord[
        DType.int64, Self.rank
    ]  # Ex HO and WO for 2D.
    var filter_dims: DynamicCoord[.int64, Self.rank]  # Ex R and S for 2D.

    var c: Int  # Input channel.
    var f: Int  # Output channel.

    var stride: DynamicCoord[.int64, Self.rank]

    var dilation: DynamicCoord[.int64, Self.rank]

    # TODO: change paddings to
    # pad_lower: DynamicCoord[DType.int64, rank]
    # pad_upper: DynamicCoord[DType.int64, rank]
    var pad_d: DynamicCoord[.int64, 2]
    var pad_h: DynamicCoord[.int64, 2]
    var pad_w: DynamicCoord[.int64, 2]

    var num_groups: Int

    @always_inline
    def __init__(
        out self,
        n: Int,
        input_dims: DynamicCoord[.int64, Self.rank],
        output_dims: DynamicCoord[.int64, Self.rank],
        filter_dims: DynamicCoord[.int64, Self.rank],
        c: Int,
        f: Int,
        stride: DynamicCoord[.int64, Self.rank],
        dilation: DynamicCoord[.int64, Self.rank],
        pad_d: DynamicCoord[.int64, 2],
        pad_h: DynamicCoord[.int64, 2],
        pad_w: DynamicCoord[.int64, 2],
        num_groups: Int,
    ):
        self.n = n
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.filter_dims = filter_dims
        self.c = c
        self.f = f
        self.stride = stride
        self.dilation = dilation
        self.pad_d = pad_d
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.num_groups = num_groups

    @always_inline
    def d(self) -> Int:
        """Input depth."""

        comptime if Self.rank >= 3:
            return Int(self.input_dims[Self.rank - 3].value())
        else:
            return 1

    @always_inline
    def h(self) -> Int:
        """Input height."""

        comptime if Self.rank >= 2:
            return Int(self.input_dims[Self.rank - 2].value())
        else:
            return 1

    @always_inline
    def w(self) -> Int:
        """Input width."""
        return Int(self.input_dims[Self.rank - 1].value())

    @always_inline
    def do(self) -> Int:
        """Output depth."""

        comptime if Self.rank >= 3:
            return Int(self.output_dims[Self.rank - 3].value())
        else:
            return 1

    @always_inline
    def ho(self) -> Int:
        """Output height."""

        comptime if Self.rank >= 2:
            return Int(self.output_dims[Self.rank - 2].value())
        else:
            return 1

    @always_inline
    def wo(self) -> Int:
        """Output width."""
        return Int(self.output_dims[Self.rank - 1].value())

    @always_inline
    def q(self) -> Int:
        """Filter window depth."""

        comptime if Self.rank >= 3:
            return Int(self.filter_dims[Self.rank - 3].value())
        else:
            return 1

    @always_inline
    def r(self) -> Int:
        """Filter window height."""

        comptime if Self.rank >= 2:
            return Int(self.filter_dims[Self.rank - 2].value())
        else:
            return 1

    @always_inline
    def s(self) -> Int:
        """Filter window width."""
        return Int(self.filter_dims[Self.rank - 1].value())

    @always_inline
    def stride_at[axis: Int](self) -> Int:
        """Stride along `axis`.

        Parameters:
            axis: Spatial axis to read.
        """
        return Int(self.stride[axis].value())

    @always_inline
    def dilation_at[axis: Int](self) -> Int:
        """Dilation along `axis`.

        Parameters:
            axis: Spatial axis to read.
        """
        return Int(self.dilation[axis].value())

    @always_inline
    def pad_d_lower(self) -> Int:
        """Depth padding before the first input point."""
        return Int(self.pad_d[0].value())

    @always_inline
    def pad_h_lower(self) -> Int:
        """Height padding above the first input row."""
        return Int(self.pad_h[0].value())

    @always_inline
    def pad_w_lower(self) -> Int:
        """Width padding left of the first input column."""
        return Int(self.pad_w[0].value())

    @always_inline
    def filter_window_flat_size(self) -> Int:
        return Int(self.filter_dims.product())

    @always_inline
    def input_image_flat_size(self) -> Int:
        return Int(self.input_dims.product())

    @always_inline
    def output_image_flat_size(self) -> Int:
        return Int(self.output_dims.product())

    @always_inline
    def output_space_dims(self) -> IndexList[Self.rank]:
        # The compiler types the result as `IndexList[Int(len(tabulate(rank)))]`
        # and cannot prove that length equals `rank`, so reconcile the two.
        return rebind[IndexList[Self.rank]](
            coord_to_index_list(self.output_dims)
        )

    @always_inline
    def output_flat_coord_to_input_offset(
        self, n: Int, output_flat_coord: Int
    ) -> Int:
        comptime assert (
            Self.rank == 1 or Self.rank == 2 or Self.rank == 3
        ), "Only support 1d, 2d, and 3d convolution."

        comptime if Self.rank == 1:
            var w = output_flat_coord * self.stride_at[0]() - self.pad_w_lower()

            return self.c * w

        elif Self.rank == 2:
            # Unpack output coordinates
            var ho, wo = divmod(output_flat_coord, self.wo())

            # Input coordinates
            var h = ho * self.stride_at[0]() - self.pad_h_lower()
            var w = wo * self.stride_at[1]() - self.pad_w_lower()

            return self.c * (w + self.w() * (h + n * self.h()))

        elif Self.rank == 3:
            # Unpack output coordinates
            var doho, wo = divmod(output_flat_coord, self.wo())
            var do, ho = divmod(doho, self.ho())

            # Input coordinates
            var d = do * self.stride_at[0]() - self.pad_d_lower()
            var h = ho * self.stride_at[1]() - self.pad_h_lower()
            var w = wo * self.stride_at[2]() - self.pad_w_lower()

            return self.c * (
                w + self.w() * (h + self.h() * (d + self.d() * self.n))
            )

        else:
            # Pass compile.
            return -1

    @always_inline
    def matmul_M(self) -> Int:
        return self.n * self.output_image_flat_size() * self.num_groups

    @always_inline
    def matmul_N(self) -> Int:
        return self.f // self.num_groups

    @always_inline
    def matmul_K(self) -> Int:
        return self.c * self.filter_window_flat_size() // self.num_groups

    @always_inline
    def padded(self) -> Bool:
        return (
            self.pad_w != coord[0, 0]
            or self.pad_h != coord[0, 0]
            or self.pad_d != coord[0, 0]
        )

    @always_inline
    def c_per_group(self) -> Int:
        """Returns the number of channels per group. Channel count must be divisible by group size.
        """
        return self.c // self.num_groups

    @always_inline
    def f_per_group(self) -> Int:
        """Returns the number of filters per group. Filter count must be divisible by group size.
        """
        return self.f // self.num_groups

    @always_inline
    def f_to_group(self, f_idx: Int) -> Int:
        """Given a global filter idx, returns the group idx of the group the filter belongs to.

        Args:
            f_idx: Global filter index across all groups.
        """
        return f_idx // self.f_per_group()

    @always_inline
    def c_to_group(self, c_idx: Int) -> Int:
        """Given a global channel idx, returns the group idx of the group the channel belongs to.

        Args:
            c_idx: Global channel index across all groups.
        """
        return c_idx // self.c_per_group()

    @always_inline
    def f_in_group(self, f_idx: Int) -> Int:
        """Given a global filter idx, returns the offset of the filter in its group.

        Args:
            f_idx: Global filter index across all groups.
        """
        return f_idx % self.f_per_group()

    @always_inline
    def c_in_group(self, c_idx: Int) -> Int:
        """Given a global channel idx, returns the offset of the channel in its group.

        Args:
            c_idx: Global channel index across all groups.
        """
        return c_idx % self.c_per_group()


@always_inline
def get_conv_shape[
    rank: Int,
    filter_packed: Bool,
](
    output: TileTensor,
    input: TileTensor[mut=False, ...],
    filter: TileTensor[mut=False, ...],
    stride: IndexList[rank],
    dilation: IndexList[rank],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    num_groups: Int,
) -> ConvShape[rank]:
    """Builds a `ConvShape` from the output, input, and filter tile tensors.

    Parameters:
        rank: Spatial rank of the convolution (1, 2, or 3).
        filter_packed: Whether the filter tensor stores spatial dimensions
            starting at index 1 (packed) rather than index 0.

    Args:
        output: Output tile tensor in NHWC layout.
        input: Input tile tensor in NHWC layout.
        filter: Filter tile tensor.
        stride: Per-dimension stride values.
        dilation: Per-dimension dilation values.
        pad_d: Depth padding as `(lower, upper)`.
        pad_h: Height padding as `(lower, upper)`.
        pad_w: Width padding as `(lower, upper)`.
        num_groups: Number of convolution groups.

    Returns:
        A populated `ConvShape` describing the convolution dimensions.
    """
    var output_dims = DynamicCoord[.int64, rank]()
    var input_dims = DynamicCoord[.int64, rank]()
    var filter_dims = DynamicCoord[.int64, rank]()

    comptime for i in range(rank):
        output_dims[i] = rebind[output_dims.element_types[i]](
            Int64(output.dim[i + 1]())
        )
        input_dims[i] = rebind[input_dims.element_types[i]](
            Int64(input.dim[i + 1]())
        )

        comptime if filter_packed:
            filter_dims[i] = rebind[filter_dims.element_types[i]](
                Int64(filter.dim[i + 1]())
            )
        else:
            filter_dims[i] = rebind[filter_dims.element_types[i]](
                Int64(filter.dim[i]())
            )

    return ConvShape[rank](
        n=Int(input.dim[0]()),
        input_dims=input_dims,
        output_dims=output_dims,
        filter_dims=filter_dims,
        c=Int(input.dim[rank + 1]()),
        f=Int(output.dim[rank + 1]()),
        stride=Coord(stride),
        dilation=Coord(dilation),
        pad_d=Coord(pad_d),
        pad_h=Coord(pad_h),
        pad_w=Coord(pad_w),
        num_groups=num_groups,
    )


@always_inline
def get_conv_tile_size[dtype: DType]() -> Int:
    """Returns the convolution L2 cache tile size in elements for the target.

    The rule-of-thumb is one half of the L2 cache size, rounded up to a
    multiple of nine to suit common 3x3 filter windows.

    Parameters:
        dtype: Element type of the convolution operands, used to convert the
            target L2 cache byte budget into an element count.
    """
    comptime KB = 1024

    # See MatmulUtils for context on tile size for debug built and macos.
    comptime if is_debug_build():
        return 4 * KB // size_of[dtype]()

    comptime if CompilationTarget.is_macos():
        return 64 * KB // size_of[dtype]()

    comptime if CompilationTarget.has_neon() or CompilationTarget.has_avx512f():
        #  Graviton 2 and Skylake server
        # have a 1 MiB L2 cache
        return 576 * KB // size_of[dtype]()

    # AMD Rome has a 512 KiB L2 cache.
    return 288 * KB // size_of[dtype]()


@always_inline
def get_conv_tile_shape[
    dtype: DType,
](c: Int, filter_window_size: Int, micro_kernel_width: Int,) -> IndexList[2]:
    """Compute the (c, f) tile shape in L2.

    Assume NHWC layout, the tile shape is (R, S, c_tile, f_tile). R and S are
    by default fully covered. The heuristic tried to block in C as much as
    possible. If C is small, it would start to block F.

    Parameters:
        dtype: Element type of the convolution operands, used to derive the
            target SIMD width and L2 tile size.

    Args:
        c: Number of input channels, bounding the C tile size.
        filter_window_size: Flattened filter window size (the product of the
            spatial filter dimensions, for example `R * S` for 2D), divided
            into the L2 tile budget to size the C and F tiles.
        micro_kernel_width: Micro-kernel width in SIMD lanes; multiplied by
            the SIMD width to give the micro-kernel size in the F dimension,
            which sets the F tile granularity.
    """
    comptime simd_size = simd_width_of[dtype]()

    # Number of elements in tile.
    var tile_size = get_conv_tile_size[dtype]()
    # Number of elements in micro kernel's f dimension.
    var micro_kernel_f = micro_kernel_width * simd_size
    # Max C tile size, assuming R, S, and micro_kernel_f are covered.
    # Round up to multiple simd_size
    var CF_tile_size = tile_size // filter_window_size
    var max_c_tile_size = (
        max(CF_tile_size // micro_kernel_f // simd_size, 1) * simd_size
    )
    # C tile size is bounded by the input channels.
    var c_tile_size = min(max_c_tile_size, c)
    # F tile size is rounded up to multiple micro_kernel_f.
    var f_tile_size = (
        max(CF_tile_size // c_tile_size // micro_kernel_f, 1) * micro_kernel_f
    )

    return Index(c_tile_size, f_tile_size)


@always_inline
def extend_shape[
    rank: Int
](in_shape: IndexList[rank], first: Int, last: Int) -> IndexList[rank + 2]:
    """Extend input shape by inserting `first` and `last` at both ends.

    Parameters:
        rank: Spatial rank of the input shape.

    Args:
        in_shape: Input shape to extend.
        first: Value inserted at the front of the shape.
        last: Value inserted at the back of the shape.
    """
    var out_shape = IndexList[rank + 2](0)
    out_shape[0] = first
    out_shape[rank + 1] = last

    comptime for i in range(rank):
        out_shape[i + 1] = in_shape[i]

    return out_shape


@always_inline
def append_shape[
    rank: Int
](in_shape: IndexList[rank], last2nd: Int, last: Int) -> IndexList[rank + 2]:
    """Append input shape by inserting `last2nd` and `last` at the end.

    Parameters:
        rank: Spatial rank of the input shape.

    Args:
        in_shape: Input shape to extend.
        last2nd: Value inserted at the second-to-last position.
        last: Value inserted at the last position.
    """
    var out_shape = IndexList[rank + 2](0)
    out_shape[rank] = last2nd
    out_shape[rank + 1] = last

    comptime for i in range(rank):
        out_shape[i] = in_shape[i]

    return out_shape


@always_inline
def reorder_padding[rank: Int](pad: IntTuple) -> IntTuple:
    """Reorders padding entries into `(lower, upper)` pairs per dimension.

    Parameters:
        rank: Spatial rank of the convolution (1, 2, or 3), selecting the
            reordering pattern applied to the flattened padding tuple.

    Args:
        pad: Padding values in the original layout.

    Returns:
        Flattened padding tuple ordered as per-dimension `(lower, upper)` pairs.
    """
    comptime if rank == 1:
        return IntTuple(pad).flatten()
    elif rank == 2:
        return IntTuple(pad[0], pad[2], pad[1], pad[3])
    else:
        return IntTuple(pad[0], pad[2], pad[4], pad[1], pad[3], pad[5])


struct ConvInfoStatic[rank: Int](Defaultable):
    """Holds statically known convolution attributes (padding, stride, dilation, groups).

    Stores the attributes in flattened form and reports whether every field is
    known at compile time, enabling micro-kernel shape specialization.

    Parameters:
        rank: Spatial rank of the convolution (1, 2, or 3).
    """

    var pad: IntTuple
    var stride: IntTuple
    var dilation: IntTuple
    var num_groups: Int

    def __init__(
        out self,
        pad: IntTuple,
        stride: IntTuple,
        dilation: IntTuple,
        num_groups: Int,
    ):
        self.pad = IntTuple(pad).flatten()
        self.stride = IntTuple(stride).flatten()
        self.dilation = IntTuple(dilation).flatten()
        self.num_groups = num_groups

    @always_inline
    def __init__(out self):
        self.pad = IntTuple(num_elems=Self.rank * 2)
        _ = self.pad._fill(UNKNOWN_VALUE)
        self.stride = IntTuple(num_elems=Self.rank)
        _ = self.stride._fill(UNKNOWN_VALUE)
        self.dilation = IntTuple(num_elems=Self.rank)
        _ = self.dilation._fill(UNKNOWN_VALUE)
        self.num_groups = UNKNOWN_VALUE

    @always_inline
    def __init__(
        out self,
        pad: IntTuple,
        stride: IntTuple,
        dilation: IntTuple,
        input_c: Int,
        filter_c: Int,
    ):
        comptime assert (
            Self.rank == 3 or Self.rank == 2 or Self.rank == 1
        ), "Only support 1d/2d/3d/ conv attributes"

        var num_groups = UNKNOWN_VALUE
        if input_c != UNKNOWN_VALUE and filter_c != UNKNOWN_VALUE:
            num_groups = input_c // filter_c

        self.pad = reorder_padding[Self.rank](pad)
        self.stride = IntTuple(stride).flatten()
        self.dilation = IntTuple(dilation).flatten()
        self.num_groups = num_groups

    @always_inline
    def all_known(self) -> Bool:
        return (
            self.pad.all_known()
            and self.stride.all_known()
            and self.dilation.all_known()
            and self.num_groups != UNKNOWN_VALUE
        )

    @always_inline
    def pad_left(self) -> Int:
        # TODO: extend to 1d/3d.
        return Int(self.pad[1])

    @always_inline
    def pad_bottom(self) -> Int:
        # TODO: extend to 1d/3d.
        return Int(self.pad[0])

    @always_inline
    def strides(self) -> IndexList[2]:
        return Index(self.stride[0], self.stride[1])

    @always_inline
    def dilations(self) -> IndexList[2]:
        return Index(self.dilation[0], self.dilation[1])


def get_direct_conv_micro_kernel_height() -> Int:
    """Returns the micro-kernel height for direct convolution on the target CPU.
    """
    comptime if CompilationTarget.has_avx512f():
        return 6
    elif CompilationTarget.is_neoverse_n1():
        return 8
    elif CompilationTarget.has_neon():  # neon other than neoverse-N1
        return 6
    return 4


def get_direct_conv_micro_kernel_width() -> Int:
    """Returns the micro-kernel width for direct convolution on the target CPU.
    """
    comptime if CompilationTarget.has_avx512f():
        return 4
    elif CompilationTarget.is_neoverse_n1():
        return 2
    elif CompilationTarget.has_neon():  # neon other than neoverse-N1
        return 4
    return 3


def get_micro_kernel_shape[
    rank: Int, WO: Int, F: Int, conv_attr: ConvInfoStatic[rank], simd_size: Int
]() -> IndexList[2]:
    """Selects the `(height, width)` micro-kernel tile shape for the target.

    When the output width, filter count, and convolution attributes are all
    statically known, the shape is tuned to minimize remainder work given the
    available SIMD register budget; otherwise a default per-target shape is
    returned.

    Parameters:
        rank: Spatial rank of the convolution.
        WO: Static output width, or `UNKNOWN_VALUE`.
        F: Static filter count, or `UNKNOWN_VALUE`.
        conv_attr: Static convolution attributes.
        simd_size: SIMD vector width for the target dtype.

    Args:

    Returns:
        A two-element `IndexList` of `(micro_kernel_height, micro_kernel_width)`.
    """
    comptime optimize_static_shapes = WO != UNKNOWN_VALUE and F != UNKNOWN_VALUE and conv_attr.all_known()

    # Number of named simd registers for each architecture.
    # TODO: configure micro kernel shape are other architectures.
    comptime num_avx512_registers = 32
    comptime num_avx2_registers = 16

    comptime if optimize_static_shapes:
        # TODO: extend to 1d/3d.
        comptime pad_h_val = Index(conv_attr.pad[0], conv_attr.pad[2])
        comptime pad_w_val = Index(conv_attr.pad[1], conv_attr.pad[3])
        comptime has_padding = pad_h_val != Index(0, 0) or pad_w_val != Index(
            0, 0
        )

        comptime if CompilationTarget.has_avx512f():
            # The micro tile is m rows by n*simd_size columns.
            # The register usage in tiling for avx512/avx2:
            #   (1) load n registers in F dimension.
            #   (2) broadcast 1 element from each row into 1 register. The same
            #       is used for all rows. This doesn't serialize the accumulation
            #       because register renaming can resolve RAR dependence.
            #   (3) accumulate m * n registers.
            # There are in total m*n + n + 1 registers needed.
            # Iterating n from 2, we get possible (m, n) combinations including
            # (14, 2), (9, 3), (6, 4), and (5, 5).

            # Static shapes enable a better algorithm for padding, which can choose micro
            # kernel shape based on input and output sizes.
            if has_padding:
                # Traverse the possible combinations (14, 2), (9, 3), (6, 4), and (5, 5).
                for n in range(2, 6):
                    var m = (num_avx512_registers - 1) // n - 1
                    # Short circuit if the row fit in one micro kernel and F is divisible.
                    # E.x. for WO=7 and F=512, 7x2 can be a better micro kernel than 7x3
                    # for multi-threading due to partition granularity (kernel width) in F.
                    if F % (n * simd_size) == 0 and WO <= m:
                        return Index(WO, n)
            # Use 6x4 by default as it achieves the best performance for most shapes.
            return Index(6, 4)

        comptime if CompilationTarget.has_avx2():
            if has_padding:
                # Register usage formula is the same as avx512.
                # There are in total 16 named simd registers, the viable micro kernels
                # are (6, 2) and (4, 3).

                # The heuristic searches the micro kernel shape leading to the
                # least remainder. The following values will be overwritten since
                # the residual is at most 2 * WO * F.
                var min_num_residual = 3 * WO * F
                var micro_kernel_height = -1
                var micro_kernel_width = -1
                for n in range(2, 4):
                    var m = (num_avx2_registers - 1) // n - 1
                    var num_residual = WO * (F % (n * simd_size)) + (WO % m) * F
                    if num_residual < min_num_residual:
                        micro_kernel_height = m
                        micro_kernel_width = n
                        min_num_residual = num_residual
                return Index(micro_kernel_height, micro_kernel_width)
            return Index(4, 3)

        comptime if CompilationTarget.is_neoverse_n1():
            return Index(8, 2)
        elif CompilationTarget.has_neon():  # neon other than neoverse-N1
            return Index(6, 4)

        return Index(6, 2)

    else:  # Default options for dynamic shapes.
        comptime if CompilationTarget.has_avx512f():
            return Index(6, 4)
        elif CompilationTarget.is_neoverse_n1():
            return Index(8, 2)
        elif CompilationTarget.has_neon():  # neon other than neoverse-N1
            return Index(6, 4)
        # default, including AVX2
        else:
            return Index(4, 3)


# ===-----------------------------------------------------------------------===#
# Partition Heuristics
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ConvPartition(TrivialRegisterPassable):
    """Work range for a partition."""

    # Batch and group dims are merged into one.
    var ng_offset: Int
    var ng_size: Int

    # Output channel range.
    var f_offset: Int
    var f_size: Int

    # Input dim.
    # For point-wise conv, ho and wo dims are merged and partitioned.
    # For others, only ho is partitioned.
    var ho_or_howo_offset: Int
    var ho_or_howo_size: Int

    # Input Channel dim.
    var c_offset: Int
    var c_size: Int

    @always_inline
    def empty(self) -> Bool:
        # fmt: off
        return self.ng_size <= 0 or \
               self.f_size <= 0 or \
               self.ho_or_howo_size <= 0 or \
               self.c_size <= 0
        # fmt: on


@always_inline
def get_conv_num_tasks(num_threads: Int, conv_shape: ConvShape) -> Int:
    """Returns the number of tasks to partition the convolution into.

    Scales the matmul-equivalent complexity by a minimum task size and clamps
    the result to the available thread count.

    Args:
        num_threads: Number of worker threads available.
        conv_shape: Convolution shape describing the workload.

    Returns:
        The number of tasks, bounded by `num_threads`.
    """
    # Currently use matmul's min task size but the optimal value
    # for direct conv may be different.
    comptime min_task_size = 64 * 1024
    # fmt: off
    var complexity = conv_shape.matmul_M() * conv_shape.matmul_N() \
                   * conv_shape.matmul_K()
    # fmt: on
    # Ensure at most one task per thread.
    return min(ceildiv(complexity, min_task_size), num_threads)


def get_conv_num_partitions[
    micro_kernel_w: Int, micro_kernel_f: Int
](num_threads: Int, conv_shape: ConvShape) -> IndexList[4]:
    """Partition the workload in (batch, C, F, HOWO) dimensions.
    HOWO is the combination of HO and WO dimensions.
    The actual number of tasks are the product of return num_partitions.

    Parameters:
        micro_kernel_w: Micro-kernel width in the spatial dimension, used as
            the partition granularity for row tasks.
        micro_kernel_f: Micro-kernel size in the filter dimension, used as the
            partition granularity for column tasks.

    Args:
        num_threads: Number of worker threads available.
        conv_shape: Convolution shape describing the workload.
    """

    var max_num_tasks = get_conv_num_tasks(num_threads, conv_shape)

    # Heuristic parameters for partitioning
    # AVX512, partitioning channel can be beneficial for some shapes.
    comptime min_rows_per_task_avx512 = align_down(196, micro_kernel_w)
    comptime min_c_per_task_avx512 = 64
    # Otherwise, discourage partitioning channel.
    comptime min_rows_per_task = min_rows_per_task_avx512 if CompilationTarget.has_avx512f() else align_down(
        64, micro_kernel_w
    )
    comptime min_c_per_task = min_c_per_task_avx512 if CompilationTarget.has_avx512f() else 1024

    # alias min_rows_per_task = (196 // micro_kernel_w) * micro_kernel_w
    # alias min_c_per_task = 64

    var matmul_M = conv_shape.matmul_M()
    var matmul_N = conv_shape.matmul_N()
    # var matmul_K = conv_shape.matmul_K()

    # Accessing A is more expensive in im2col than accessing B.
    # Time a factor to M to var the heuristic bias on partitioning M.
    # TODO: make this bias factor part of function parameter/argument and
    # unifies interface with matmul partition, e.x. bias=1 for matmul.
    comptime bias = 0.25
    var matmul_M_biased = Int(max(Float64(matmul_M) * bias, 1))

    # The ideal partition in theory is to balance the cost of memory access in
    # M and N dimensions using square sub-matrix (after applying the bias).
    var ideal_num_col_tasks = sqrt(
        ceildiv(matmul_N * max_num_tasks, matmul_M_biased)
    )
    var num_row_tasks = max_num_tasks // ideal_num_col_tasks
    var num_col_tasks = ideal_num_col_tasks

    # There must at least have enough elements to support a micro kernel.
    # Do not partition F when num_groups > 1.
    var max_num_col_tasks = min(
        ceildiv(matmul_N, micro_kernel_f), max_num_tasks
    )
    if ideal_num_col_tasks > max_num_col_tasks:
        num_col_tasks = max_num_col_tasks
        num_row_tasks = max_num_tasks // num_col_tasks
    # In this branch, not all threads get used for ideal_num_col_tasks
    # Check for alternative factorizations use the most threads.
    elif max_num_tasks % ideal_num_col_tasks != 0:
        # Set 20% deviation.
        var eps = ceildiv(2 * ideal_num_col_tasks, 10)
        max_num_col_tasks = min(max_num_col_tasks, ideal_num_col_tasks + eps)
        var num_col_tasks_tmp = max(ideal_num_col_tasks - eps, 1)
        var num_threads_used = (
            max_num_tasks // ideal_num_col_tasks
        ) * ideal_num_col_tasks
        while num_col_tasks_tmp <= max_num_col_tasks:
            var num_row_tasks_tmp = max_num_tasks // num_col_tasks_tmp
            if num_row_tasks_tmp * num_col_tasks_tmp >= num_threads_used:
                num_col_tasks = num_col_tasks_tmp
                num_row_tasks = num_row_tasks_tmp
                num_threads_used = num_row_tasks_tmp * num_col_tasks_tmp
            num_col_tasks_tmp += 1

    var max_num_row_tasks = max(matmul_M // min_rows_per_task, 1)
    num_row_tasks = min(max_num_row_tasks, num_row_tasks)

    # Do not partition channels when num_groups > 1.
    var max_num_channel_tasks = (
        max(conv_shape.c // min_c_per_task, 1) if conv_shape.num_groups == 1
        and conv_shape.rank == 2 else 1
    )
    var num_channel_tasks = min(
        max_num_channel_tasks,
        max_num_tasks // (num_row_tasks * num_col_tasks),
    )

    var num_batch_group_tasks = min(
        conv_shape.n * conv_shape.num_groups, num_row_tasks
    )

    num_row_tasks = num_row_tasks // num_batch_group_tasks

    return Index(
        num_batch_group_tasks, num_channel_tasks, num_col_tasks, num_row_tasks
    )


@always_inline
def get_partition(
    task_id: Int,
    num_partitions: IndexList[4],
    conv_shape: ConvShape,
    micro_kernel_height: Int,
    micro_kernel_f_size: Int,
) -> ConvPartition:
    """Computes the `ConvPartition` work range for a given task id.

    Args:
        task_id: Linear task identifier.
        num_partitions: Per-dimension partition counts as
            `(batch_group, channel, filter, spatial)`.
        conv_shape: Convolution shape describing the workload.
        micro_kernel_height: Micro-kernel height, used as the spatial work unit
            when output loops are merged.
        micro_kernel_f_size: Micro-kernel size in the filter dimension.

    Returns:
        The `ConvPartition` describing the offset and size of this task's work
        across batch/group, channel, filter, and spatial dimensions.
    """
    var quotient, task_id_f = divmod(task_id, num_partitions[2])
    var quotient2, task_id_c = divmod(quotient, num_partitions[1])
    var task_id_ng, task_id_howo = divmod(quotient2, num_partitions[3])

    var ng_range = partition_work(
        task_id_ng, num_partitions[0], conv_shape.n * conv_shape.num_groups, 1
    )

    var c_range = partition_work(task_id_c, num_partitions[1], conv_shape.c, 1)

    var f_range = partition_work(
        task_id_f,
        num_partitions[2],
        conv_shape.f // conv_shape.num_groups,
        micro_kernel_f_size,
    )

    # Merge output space loops when there is no padding and 2D.
    # Otherwise the partition granularity is a row.
    # TODO: generalize to 1D and 3D.
    var merge_loop = not conv_shape.padded() and conv_shape.rank == 2
    var work_unit = micro_kernel_height if merge_loop else 1
    var work_load = (
        conv_shape.output_image_flat_size() if merge_loop else conv_shape.ho()
    )
    var howo_range = partition_work(
        task_id_howo, num_partitions[3], work_load, work_unit
    )

    return ConvPartition(
        ng_offset=ng_range[0],
        ng_size=ng_range[1],
        f_offset=f_range[0],
        f_size=f_range[1],
        ho_or_howo_offset=howo_range[0],
        ho_or_howo_size=howo_range[1],
        c_offset=c_range[0],
        c_size=c_range[1],
    )


# ===-----------------------------------------------------------------------===#
# Convolution Algorithms Selection
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ConvAlgorithm(TrivialRegisterPassable):
    """Tags the convolution algorithm selected for a given operation.

    `Default` indicates an unknown layout, `Im2Col` selects the im2col-based
    algorithm, and `Direct` selects the direct convolution algorithm.
    """

    var value: Int
    comptime Default = ConvAlgorithm(0)  # statically unknown layout.
    comptime Im2Col = ConvAlgorithm(1)  # channels first layout.
    comptime Direct = ConvAlgorithm(
        2
    )  # TF filter layout for channels last input.

    @always_inline("nodebug")
    def __eq__(self, rhs: ConvAlgorithm) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    def __ne__(self, rhs: ConvAlgorithm) -> Bool:
        return self.value != rhs.value


# ===----------------------------------------------------------------------=== #
# align_down_residual
# ===----------------------------------------------------------------------=== #


@always_inline
def align_down_residual(value: Int, alignment: Int) -> Int:
    """Returns the remainder after aligning down value to alignment.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        The remainder after aligning down value to the closest multiple of
        alignment. In other words, value - align_down(value, alignment).
    """
    return value - align_down(value, alignment)
