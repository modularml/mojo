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
"""ManagedTensorSlice and the types kernel authors compose against.

A custom kernel's entry-point signature uses these:

- `ManagedTensorSlice`: view of a tensor argument.
- `IOSpec` (and `IO`): input/output/mutability annotations.
- `StaticTensorSpec`: runtime + compile-time tensor
  metadata.
- Fusion traits (`InputFusion`, `OutputFusion`, ...) and their `_NoFusion*`
  sentinels.

The decorators that register a kernel (`register`, `register_internal`,
`view_kernel`) live next to this file in `register.mojo`.
"""
from max.algorithm.functional import elementwise

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import Optional
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu
from max.gpu.host.info import is_gpu as _is_gpu
from std.math import ceil, fma
from std.memory import AddressSpace
from std.sys import align_of, simd_width_of, size_of
from std.sys.info import CompilationTarget, is_gpu
from std.sys.intrinsics import strided_load, strided_store
from std.utils import IndexList, StaticTuple, product
from std.utils._serialize import _serialize

from max.runtime.tracing import trace_arg

from layout import (
    Coord,
    CoordLike,
    IntTuple,
    Layout,
    LayoutTensor,
    TileTensor,
    coord_to_index_list,
)
from layout.coord import (
    ComptimeInt,
    _IntToComptimeInt,
    crd2idx,
)
from layout.int_tuple import _IntTupleToCoordLike, coord_to_int_tuple
from layout.tile_io import TileCopier
from layout.tile_layout import Layout as TileLayout, TensorLayout, _RowMajor

from .decorators import register_internal


# ===----------------------------------------------------------------------=== #
# IO direction tag + IOSpec marker
# ===----------------------------------------------------------------------=== #


struct IO(Equatable, TrivialRegisterPassable):
    """Tags the direction and fusion kind of a tensor argument to a DPS kernel.

    An `IO` value distinguishes plain inputs, outputs, mutable inputs, and the
    fused variants (input, output, and compute-output) that the graph compiler
    wires into a custom kernel's fusion lambdas.
    """

    var value: SIMDLength

    # TODO: either rename or get rid of this
    comptime Unknown = IO(-1)

    comptime Output = IO(0)
    comptime Input = IO(1)

    # Represents the standard kind of fusion where we only make accesses
    # through the fusion lambda (e.g. any of the elementwise ops).
    comptime FusedInput = IO(2)
    comptime FusedOutput = IO(3)

    # Output fusion using a compute lambda.
    comptime _FusedComputeOutput = IO(31)

    # Output fusion using a tile-based compute lambda.
    comptime _FusedComputeOutputTile = IO(32)

    # Output fusion using a tile-based store lambda (the fusion owns the store).
    comptime _FusedOutputTile = IO(33)

    @always_inline("builtin")
    def __init__(out self, value: Int):
        self.value = value

    @always_inline("nodebug")
    def is_fused(self) -> Bool:
        """True when this IO represents any fused variant (input, output, or
        compute-output)."""
        return (
            self == IO.FusedInput
            or self == IO.FusedOutput
            or self == IO._FusedComputeOutput
            or self == IO._FusedComputeOutputTile
            or self == IO._FusedOutputTile
        )


@fieldwise_init
struct IOSpec[mut: Bool, input: IO](TrivialRegisterPassable):
    """
    Parameter used to encode whether a particular tensor argument to a DPS kernel
    is an output, input, or mutable input.

    ```mojo
    IOSpec.Input == IOSpec[False, IO.Input]()
    IOSpec.Output == IOSpec[True, IO.Output]()
    IOSpec.MutableInput == IOSpec[True, IO.Input]()
    IOSpec.FusedInput == IOSpec[False, IO.FusedInput]()
    IOSpec.FusedOutput == IOSpec[True, IO.FusedOutput]()
    ```

    These value aliases live as static members of `IOSpec` (rather than as
    module-level aliases) so that the bare names `Input`/`Output`/`MutableInput`
    are free for the tensor-argument traits in `tensor_arg_traits.mojo`.
    """

    comptime Unknown = IOSpec[True, IO.Unknown]()

    comptime Input = IOSpec[False, IO.Input]()
    comptime Output = IOSpec[True, IO.Output]()
    comptime MutableInput = IOSpec[True, IO.Input]()

    comptime FusedInput = IOSpec[False, IO.FusedInput]()
    comptime FusedOutput = IOSpec[True, IO.FusedOutput]()

    comptime _FusedComputeOutput = IOSpec[True, IO._FusedComputeOutput]()

    comptime _FusedComputeOutputTile = IOSpec[
        True, IO._FusedComputeOutputTile
    ]()

    comptime _FusedOutputTile = IOSpec[True, IO._FusedOutputTile]()


# ===----------------------------------------------------------------------=== #
# Indexing helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _dot_prod[rank: Int](x: IndexList[rank], y: IndexList[rank]) -> Int:
    var offset = 0

    comptime for i in range(rank):
        offset += x[i] * y[i]
    return offset


# ===----------------------------------------------------------------------=== #
# TileLayout helper aliases
# ===----------------------------------------------------------------------=== #

comptime _AllScalar[rank: Int] = TypeList.splat[
    Trait=CoordLike, count=rank, type=Int
]()
"""A variadic of `rank` Scalar types."""

comptime _UnknownTileLayout[rank: Int] = TileLayout[
    shape_types=_AllScalar[rank],
    stride_types=_AllScalar[rank],
]
"""A fully-dynamic TileLayout where all shape and stride dims are Scalar."""


comptime _RowMajorTileLayout[
    shape_types: TypeList[Trait=CoordLike, ...]
] = TileLayout[
    shape_types=shape_types,
    stride_types=_RowMajor[*shape_types],
]
"""A TileLayout with row-major strides derived from the given shape types."""

comptime _IndexListToCoordLikeTabulator[
    list: IndexList,
    idx: Int,
]: CoordLike = ComptimeInt[list[idx]] if list[idx] >= 0 else Int

"""Maps a single IndexList element to a CoordLike type.
Negative values (-1 = dynamic) become Scalar, others become ComptimeInt."""


comptime _IndexListToCoordLike[list: IndexList] = TypeList.tabulate[
    list.size, _IndexListToCoordLikeTabulator[list, _]
]()
"""Converts a compile-time IndexList to a variadic of CoordLike types.
Negative values become Scalar, non-negative become ComptimeInt."""


comptime _IndexListToTileLayout[
    shape: IndexList, strides: IndexList
] = TileLayout[
    shape_types=_IndexListToCoordLike[shape],
    stride_types=_IndexListToCoordLike[strides],
]
"""Convert a pair of compile-time IndexLists to a TileLayout.
Negative values (-1) become Scalar, non-negative become ComptimeInt."""


comptime _RowMajorIntTupleTileLayout[
    shape: IntTuple,
] = _RowMajorTileLayout[_IntTupleToCoordLike[.int, shape]]
"""A TileLayout with row-major strides derived from an IntTuple shape."""


comptime _IntTupleShapeIndexListStridesToTileLayout[
    shape: IntTuple, strides: IndexList
] = TileLayout[
    shape_types=_IntTupleToCoordLike[.int, shape],
    stride_types=_IndexListToCoordLike[strides],
]
"""Convert an IntTuple shape and IndexList strides to a TileLayout."""


def get_row_major_tensor_spec_static[
    dtype: DType, rank: Int, *shape_dims: Int
]() -> StaticTensorSpec[
    dtype,
    rank,
    static_layout=_RowMajorTileLayout[_IntToComptimeInt[*shape_dims]],
]:
    """Returns a row-major StaticTensorSpec from compile-time Int dimensions.

    All dimensions must be static (known at compile time).

    Parameters:
        dtype: The element data type.
        rank: The tensor rank (must match `len(shape_dims)`).
        shape_dims: Compile-time integer dimensions of the tensor shape.
    """
    return {align_of[dtype](), .GENERIC}


def _get_unknown_tensor_spec[
    dtype: DType, rank: Int
]() -> StaticTensorSpec[dtype, rank, static_layout=_UnknownTileLayout[rank]]:
    """
    Returns a StaticTensorSpec with the specified type and rank with all
    fields dynamic or defaulted.
    """
    return {1, .GENERIC}


# ===----------------------------------------------------------------------=== #
# Fusion traits and sentinel types
# ===----------------------------------------------------------------------=== #


trait InputFusion(TrivialRegisterPassable):
    """Trait for input fusion structs that provide custom load behavior."""

    def load[
        dtype: DType,
        rank: Int,
        simd_width: Int,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
        ...


trait OutputFusion(TrivialRegisterPassable):
    """Trait for output fusion structs that provide custom store behavior."""

    def store[
        dtype: DType,
        rank: Int,
        simd_width: SIMDLength,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]):
        ...


trait ComputeOutputFusion(TrivialRegisterPassable):
    """Trait for compute-output fusion structs that transform values before
    storing."""

    def compute[
        dtype: DType,
        rank: Int,
        simd_width: SIMDLength,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]) -> SIMD[
        dtype, simd_width
    ]:
        ...


trait ElementwiseFusion(TrivialRegisterPassable):
    """Trait for pure elementwise fusion structs emitted by the graph
    compiler."""

    def compute[
        dtype: DType,
        rank: Int,
        simd_width: Int,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
        ...


struct _NoFusionIn(InputFusion):
    """Sentinel type indicating no input fusion is active."""

    def __init__(out self):
        pass

    def load[
        dtype: DType,
        rank: Int,
        simd_width: Int,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank]) -> SIMD[dtype, simd_width]:
        comptime assert False, "load() not implemented for this InputFusion"


struct _NoFusionOut(OutputFusion):
    """Sentinel type indicating no output fusion is active."""

    def __init__(out self):
        pass

    def store[
        dtype: DType,
        rank: Int,
        simd_width: SIMDLength,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]):
        comptime assert False, "store() not implemented for this OutputFusion"


struct _NoComputeFusion(ComputeOutputFusion):
    """Sentinel type indicating no compute-output fusion is active."""

    def __init__(out self):
        pass

    def compute[
        dtype: DType,
        rank: Int,
        simd_width: SIMDLength,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]) -> SIMD[
        dtype, simd_width
    ]:
        comptime assert (
            False
        ), "compute() not implemented for this ComputeOutputFusion"


trait ComputeOutputFusionTile(TrivialRegisterPassable):
    """Trait for tile-based compute-output fusion structs that transform a
    TileTensor before storing."""

    def compute[
        dtype: DType,
        rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
    ](
        self,
        tile_coords: IndexList[rank],
        copier: Copier,
        val: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[dtype, LayoutType, MutAnyOrigin]:
        ...


struct _NoComputeFusionTile(ComputeOutputFusionTile):
    """Sentinel type indicating no tile-based compute-output fusion is active.
    """

    def __init__(out self):
        pass

    def compute[
        dtype: DType,
        rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
    ](
        self,
        tile_coords: IndexList[rank],
        copier: Copier,
        val: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[dtype, LayoutType, MutAnyOrigin]:
        comptime assert (
            False
        ), "compute() not implemented for this ComputeOutputFusionTile"


trait OutputFusionTile(TrivialRegisterPassable):
    """Trait for tile-based output fusion structs that provide custom store
    behavior for a `TileTensor`.

    The tile analog of `OutputFusion`: `store` is terminal (owns the store and
    returns nothing), where `ComputeOutputFusionTile.compute` transforms a tile
    and returns it for the primary kernel to store.

    Takes two copiers, both chosen by the calling kernel: `copier` loads any aux
    epilogue inputs (e.g. a broadcast bias) into local, and `store_copier`
    writes the transformed tile out to the output. Passing the store copier
    (rather than having the graph compiler synthesize one) keeps the copier
    choice with the kernel author.
    """

    def store[
        dtype: DType,
        rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
        StoreCopier: TileCopier,
    ](
        self,
        tile_coords: IndexList[rank],
        copier: Copier,
        store_copier: StoreCopier,
        val: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ):
        ...


struct _NoOutputFusionTile(OutputFusionTile):
    """Sentinel type indicating no tile-based output fusion is active."""

    def __init__(out self):
        pass

    def store[
        dtype: DType,
        rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
        StoreCopier: TileCopier,
    ](
        self,
        tile_coords: IndexList[rank],
        copier: Copier,
        store_copier: StoreCopier,
        val: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ):
        comptime assert (
            False
        ), "store() not implemented for this OutputFusionTile"


trait ElementwiseFusionTile(TrivialRegisterPassable):
    """The `TileTensor` variant of `ElementwiseFusion`: a tile-based pure
    elementwise fusion struct emitted by the graph compiler.

    `compute` fills the driver-provided `dst` tile (which carries the tile
    layout) and returns it.
    """

    def compute[
        dtype: DType,
        rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
    ](
        self,
        tile_coords: IndexList[rank],
        copier: Copier,
        dst: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[dtype, LayoutType, MutAnyOrigin]:
        ...


# Compile time Tensor information
struct StaticTensorSpec[
    dtype: DType,
    rank: Int,
    static_layout: TensorLayout,
    InFusion: InputFusion = _NoFusionIn,
    OutFusion: OutputFusion = _NoFusionOut,
    ComputeFusion: ComputeOutputFusion = _NoComputeFusion,
    ComputeFusionTile: ComputeOutputFusionTile = _NoComputeFusionTile,
    OutFusionTile: OutputFusionTile = _NoOutputFusionTile,
](ImplicitlyCopyable):
    """Carries the compile-time and runtime metadata describing a tensor argument.

    The compile-time parameters encode the element `dtype`, tensor `rank`,
    static layout, and optional fusion traits. The runtime fields store the
    alignment and address space of the backing memory. Custom kernels receive
    `ManagedTensorSlice` instances parameterized by a `StaticTensorSpec`.
    """

    # IntTuple aliases for static shape/strides.
    comptime shape_tuple = coord_to_int_tuple[
        *Self.static_layout._shape_types
    ]()
    comptime strides_tuple = coord_to_int_tuple[
        *Self.static_layout._stride_types
    ]()

    var alignment: Int
    var address_space: AddressSpace

    def __init__(
        out self,
        alignment: Int,
        address_space: AddressSpace,
    ):
        comptime assert Self.rank == Self.static_layout.rank, "rank mismatch"
        comptime _has_in = Self.InFusion != _NoFusionIn
        comptime _has_out = Self.OutFusion != _NoFusionOut
        comptime _has_compute = Self.ComputeFusion != _NoComputeFusion
        comptime _has_compute_tile = (
            Self.ComputeFusionTile != _NoComputeFusionTile
        )
        comptime _has_out_tile = Self.OutFusionTile != _NoOutputFusionTile
        comptime assert (
            Int(_has_in)
            + Int(_has_out)
            + Int(_has_compute)
            + Int(_has_compute_tile)
            + Int(_has_out_tile)
            <= 1
        ), "StaticTensorSpec can have at most one fusion type"
        self.alignment = alignment
        self.address_space = address_space

    def __init__(
        out self, internals: StaticTensorSpecInternal[Self.dtype, Self.rank]
    ):
        """
        Returns a StaticTensorSpec from a StaticTensorSpecInternal.
        """
        comptime _has_in = Self.InFusion != _NoFusionIn
        comptime _has_out = Self.OutFusion != _NoFusionOut
        comptime _has_compute = Self.ComputeFusion != _NoComputeFusion
        comptime _has_compute_tile = (
            Self.ComputeFusionTile != _NoComputeFusionTile
        )
        comptime _has_out_tile = Self.OutFusionTile != _NoOutputFusionTile
        comptime assert (
            Int(_has_in)
            + Int(_has_out)
            + Int(_has_compute)
            + Int(_has_compute_tile)
            + Int(_has_out_tile)
            <= 1
        ), "StaticTensorSpec can have at most one fusion type"
        self.alignment = internals.alignment
        self.address_space = internals.address_space

    @always_inline
    def to_unfused(
        self,
    ) -> StaticTensorSpec[
        Self.dtype, Self.rank, static_layout=Self.static_layout
    ]:
        """Returns a copy with sentinel (no-op) fusion types.

        The runtime fields (alignment, etc.) are identical;
        only the compile-time fusion type parameters change.
        """
        return {
            self.alignment,
            self.address_space,
        }

    # This indirect approach to providing get_unknown is necessary because the
    # we don't want clients to have to bind rank and shapes to use this. Aliases
    # only require the parameters they USE to be bound, whereas static methods
    # require all parameters to be bound.
    comptime get_unknown = _get_unknown_tensor_spec[Self.dtype, Self.rank]

    @always_inline
    def with_tile_layout[
        new_layout: TensorLayout,
    ](
        self,
    ) -> StaticTensorSpec[
        Self.dtype,
        new_layout.rank,
        static_layout=new_layout,
    ]:
        return {
            self.alignment,
            self.address_space,
        }

    @always_inline
    def with_tile_layout[
        new_rank: Int,
        new_layout: TensorLayout,
    ](
        self,
    ) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=new_layout,
    ]:
        comptime assert new_rank == new_layout.rank, "rank mismatch"
        return {
            self.alignment,
            self.address_space,
        }

    @always_inline
    def with_tile_layout_and_alignment[
        new_layout: TensorLayout,
    ](self, new_alignment: Int) -> StaticTensorSpec[
        Self.dtype,
        new_layout.rank,
        static_layout=new_layout,
    ]:
        return {
            new_alignment,
            self.address_space,
        }

    @always_inline
    def with_tile_layout_and_alignment[
        new_rank: Int,
        new_layout: TensorLayout,
    ](self, new_alignment: Int) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=new_layout,
    ]:
        comptime assert new_rank == new_layout.rank, "rank mismatch"
        return {
            new_alignment,
            self.address_space,
        }

    @always_inline
    def with_int_tuple_layout[
        new_rank: Int, new_shape: IntTuple, new_strides: IndexList
    ](
        self,
    ) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=_IntTupleShapeIndexListStridesToTileLayout[
            new_shape, new_strides
        ],
    ]:
        return {
            self.alignment,
            self.address_space,
        }

    @always_inline
    def with_int_tuple_layout_and_alignment[
        new_rank: Int, new_shape: IntTuple, new_strides: IndexList
    ](self, new_alignment: Int) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=_IntTupleShapeIndexListStridesToTileLayout[
            new_shape, new_strides
        ],
    ]:
        return {
            new_alignment,
            self.address_space,
        }

    @always_inline
    def with_row_major_int_tuple_layout[
        new_rank: Int, new_shape: IntTuple
    ](
        self,
    ) -> StaticTensorSpec[
        Self.dtype,
        new_rank,
        static_layout=_RowMajorIntTupleTileLayout[new_shape],
    ]:
        return {
            self.alignment,
            self.address_space,
        }

    @always_inline
    def with_input_fusion[
        F: InputFusion
    ](self) -> StaticTensorSpec[
        Self.dtype,
        Self.rank,
        Self.static_layout,
        F,
        Self.OutFusion,
        Self.ComputeFusion,
        Self.ComputeFusionTile,
        Self.OutFusionTile,
    ]:
        return {
            self.alignment,
            self.address_space,
        }

    @always_inline
    def with_output_fusion[
        F: OutputFusion
    ](self) -> StaticTensorSpec[
        Self.dtype,
        Self.rank,
        Self.static_layout,
        Self.InFusion,
        F,
        Self.ComputeFusion,
        Self.ComputeFusionTile,
        Self.OutFusionTile,
    ]:
        return {
            self.alignment,
            self.address_space,
        }

    @always_inline
    def with_compute_fusion[
        F: ComputeOutputFusion
    ](self) -> StaticTensorSpec[
        Self.dtype,
        Self.rank,
        Self.static_layout,
        Self.InFusion,
        Self.OutFusion,
        F,
        Self.ComputeFusionTile,
        Self.OutFusionTile,
    ]:
        return {
            self.alignment,
            self.address_space,
        }

    @always_inline
    def with_compute_fusion_tile[
        F: ComputeOutputFusionTile
    ](self) -> StaticTensorSpec[
        Self.dtype,
        Self.rank,
        Self.static_layout,
        Self.InFusion,
        Self.OutFusion,
        Self.ComputeFusion,
        F,
        Self.OutFusionTile,
    ]:
        return {
            self.alignment,
            self.address_space,
        }

    @always_inline
    def with_output_fusion_tile[
        F: OutputFusionTile
    ](self) -> StaticTensorSpec[
        Self.dtype,
        Self.rank,
        Self.static_layout,
        Self.InFusion,
        Self.OutFusion,
        Self.ComputeFusion,
        Self.ComputeFusionTile,
        F,
    ]:
        return {
            self.alignment,
            self.address_space,
        }

    @always_inline
    def to_layout(self) -> Layout:
        return Layout(
            coord_to_int_tuple[*Self.static_layout._shape_types](),
            coord_to_int_tuple[*Self.static_layout._stride_types](),
        )

    comptime static_size: Int = Layout(
        coord_to_int_tuple[*Self.static_layout._shape_types](),
        coord_to_int_tuple[*Self.static_layout._stride_types](),
    ).size()

    def get_internals(self) -> StaticTensorSpecInternal[Self.dtype, Self.rank]:
        """
        Returns a StaticTensorSpecInternal from a StaticTensorSpec.
        """
        return {
            self.alignment,
            self.address_space,
        }


@fieldwise_init
struct StaticTensorSpecInternal[dtype: DType, rank: Int](ImplicitlyCopyable):
    """Stores the runtime-only portion of a `StaticTensorSpec`.

    Holds the alignment and address space fields, providing a device-passable
    view that drops the compile-time layout and fusion type parameters.
    """

    var alignment: Int
    var address_space: AddressSpace


# ===----------------------------------------------------------------------=== #
# Load / Store Helper primitives
# ===----------------------------------------------------------------------=== #


@__parameter
@always_inline
def _gcd_pow2[a: Int, b: Int]() -> Int:
    # alignments should always be powers of 2
    comptime assert (
        a.is_power_of_two() and b.is_power_of_two()
    ), "a and b must be powers of 2"
    return min(a, b)


# TODO(GEX-1523): Consider moving these and other methods implementation into
# non-class member functions.
#
# TODO(GEX-1831): Remove redundant parameters present in the StaticTensorSpec
#
# Note: these methods are forced inline in the graph compiler. We keep the
# inlining at the whims of the automatic inliner for now since we want to
# predictably introspect and manipulate these particular functions.
#
# They are set to be inlined further down graph compiler stack.
@doc_hidden
@register_internal("simd_store_into_managed_tensor_slice")
@always_inline
def simd_store_into_managed_tensor_slice[
    dtype: DType,
    rank: Int,
    simd_width: SIMDLength,
    //,
    static_spec: StaticTensorSpec[dtype, rank, ...],
    element_alignment: Int = 1,
](
    tensor: ManagedTensorSlice[static_spec=static_spec, ...],
    indices: IndexList[rank],
    value: SIMD[dtype, simd_width],
):
    var flat_index = tensor._compute_offset(indices)

    # Store alignment cannot exceed the data type's alignment.
    comptime max_alignment = _gcd_pow2[
        tensor.alignment, element_alignment * align_of[dtype]()
    ]()

    comptime _last_stride_is_static = tensor.static_spec.static_layout._stride_types[
        rank - 1
    ].is_static_value
    comptime _last_stride_value = tensor.static_spec.static_layout._stride_types[
        rank - 1
    ].static_value

    # Stride = 1
    @__parameter
    @always_inline
    def store_stride1():
        comptime if dtype == .bool:
            var v = value.cast[.uint8]()
            tensor._ptr.unsafe_bitcast[UInt8]().unsafe_store(flat_index, v)
        else:
            tensor._ptr.unsafe_store[alignment=max_alignment](flat_index, value)

    # Stride > 1
    @__parameter
    @always_inline
    def store_strided(stride: Int):
        comptime if dtype == .bool:
            var v = value.cast[.uint8]()
            strided_store(
                v,
                tensor._ptr.unsafe_bitcast[UInt8]().unsafe_offset(flat_index),
                stride,
            )
        else:
            return strided_store(
                value, tensor._ptr.unsafe_offset(flat_index), stride
            )

    comptime if not _last_stride_is_static:
        var stride = tensor.stride_length[rank - 1]()
        # Dynamic stride
        if stride == 0:
            tensor._ptr.unsafe_store[alignment=max_alignment](0, value)
        elif stride == 1:
            store_stride1()
        else:
            store_strided(stride)
    else:
        # static stride
        comptime if _last_stride_value == 0:
            tensor._ptr.unsafe_store[alignment=max_alignment](0, value)
        elif _last_stride_value == 1:
            store_stride1()
        else:
            store_strided(_last_stride_value)


@doc_hidden
@register_internal("simd_store_into_tensor_pointer")
@always_inline
def simd_store_into_tensor_pointer[
    dtype: DType,
    rank: Int,
    //,
    static_spec: StaticTensorSpec[dtype, rank, ...],
    simd_width: SIMDLength,
    element_alignment: Int = 1,
](
    ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    shape: IndexList[rank],
    strides: IndexList[rank],
    indices: IndexList[rank],
    value: SIMD[dtype, simd_width],
):
    """Store a SIMD vector to raw tensor components.

    This function is GPU-safe because it only takes trivial types (pointer,
    IndexList) that can be properly captured in GPU kernel closures. Use this
    instead of simd_store_into_managed_tensor_slice when generating code for
    GPU kernels.

    Parameters:
        dtype: The data type of tensor elements.
        rank: The rank (number of dimensions) of the tensor.
        static_spec: The static specs of the tensor.
        simd_width: The SIMD width for the store operation.
        element_alignment: The element alignment for the store.

    Args:
        ptr: The raw pointer to tensor data.
        shape: The runtime shape of the tensor.
        strides: The runtime strides of the tensor.
        indices: The indices to store into.
        value: The value to store.
    """
    var tensor = OutputTensor[dtype=dtype, rank=rank, static_spec=static_spec](
        ptr, shape, strides
    )
    simd_store_into_managed_tensor_slice[element_alignment=element_alignment](
        tensor, indices, value
    )


# GPU-safe load function that takes raw components (pointer, strides) instead of
# ManagedTensorSlice. This avoids capturing ManagedTensorSlice in GPU kernels,
# which doesn't work correctly due to closure capture limitations.
@doc_hidden
@register_internal("simd_load_from_tensor_pointer")
@always_inline
def simd_load_from_tensor_pointer[
    dtype: DType,
    rank: Int,
    //,
    static_spec: StaticTensorSpec[dtype, rank, ...],
    simd_width: Int,
    element_alignment: Int = 1,
](
    ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    shape: IndexList[rank],
    strides: IndexList[rank],
    indices: IndexList[rank],
) -> SIMD[dtype, simd_width]:
    """Load a SIMD vector from raw tensor components.

    This function is GPU-safe because it only takes trivial types (pointer,
    IndexList) that can be properly captured in GPU kernel closures. Use this
    instead of simd_load_from_managed_tensor_slice when generating code for
    GPU kernels.

    Parameters:
        dtype: The data type of tensor elements.
        rank: The rank (number of dimensions) of the tensor.
        static_spec: The static specs of the tensor.
        simd_width: The SIMD width for the load operation.
        element_alignment: The element alignment for the load.

    Args:
        ptr: The raw pointer to tensor data.
        shape: The runtime shape of the tensor.
        strides: The runtime strides of the tensor.
        indices: The indices to load from.

    Returns:
        A SIMD vector with the loaded values.
    """
    var tensor = InputTensor[dtype=dtype, rank=rank, static_spec=static_spec](
        ptr, shape, strides
    )
    return simd_load_from_managed_tensor_slice[
        simd_width=simd_width, element_alignment=element_alignment
    ](tensor, indices)


@doc_hidden
@register_internal("simd_load_from_managed_tensor_slice")
@always_inline
def simd_load_from_managed_tensor_slice[
    dtype: DType,
    rank: Int,
    simd_width: Int,
    //,
    static_spec: StaticTensorSpec[dtype, rank, ...],
    element_alignment: Int = 1,
](
    tensor: ManagedTensorSlice[static_spec=static_spec, ...],
    indices: IndexList[rank],
) -> SIMD[dtype, simd_width]:
    var flat_index = tensor._compute_offset(indices)
    comptime _last_stride_is_static = tensor.static_spec.static_layout._stride_types[
        rank - 1
    ].is_static_value
    comptime _last_stride_value = tensor.static_spec.static_layout._stride_types[
        rank - 1
    ].static_value

    # Load alignment cannot exceed the data type's alignment.
    comptime max_alignment = _gcd_pow2[
        tensor.alignment, element_alignment * align_of[dtype]()
    ]()
    comptime invariant = not tensor.io_spec.mut

    # Stride = 1
    @__parameter
    @always_inline
    def load_stride1() -> SIMD[dtype, simd_width]:
        comptime if dtype == .bool:
            var v = tensor._ptr.unsafe_bitcast[UInt8]().unsafe_load[
                width=simd_width,
                invariant=invariant,
            ](flat_index)
            return v.cast[dtype]()
        else:
            return tensor._ptr.unsafe_load[
                width=simd_width, alignment=max_alignment, invariant=invariant
            ](flat_index)

    # Stride > 1
    @__parameter
    @always_inline
    def load_strided(stride: Int) -> SIMD[dtype, simd_width]:
        comptime if dtype == .bool:
            var v = strided_load[simd_width, invariant=invariant](
                tensor._ptr.unsafe_bitcast[UInt8]().unsafe_offset(flat_index),
                stride,
            )
            return v.cast[dtype]()
        else:
            return strided_load[simd_width, invariant=invariant](
                tensor._ptr.unsafe_offset(flat_index), stride
            )

    comptime if not _last_stride_is_static:
        var stride = tensor.stride_length[rank - 1]()
        # Dynamic stride
        if stride == 0:
            return tensor._ptr.unsafe_load[invariant=invariant](flat_index)
        elif stride == 1:
            return load_stride1()
        else:
            return load_strided(stride)
    else:
        # Static stride
        comptime if _last_stride_value == 0:
            return tensor._ptr.unsafe_load[invariant=invariant](flat_index)
        elif _last_stride_value == 1:
            return load_stride1()
        else:
            return load_strided(_last_stride_value)


# ===----------------------------------------------------------------------=== #
# ManagedTensorSlice class
# ===----------------------------------------------------------------------=== #

comptime OutputTensor = ManagedTensorSlice[io_spec=IOSpec.Output, ...]
comptime InputTensor = ManagedTensorSlice[io_spec=IOSpec.Input, ...]

comptime _MutableInputTensor = ManagedTensorSlice[
    io_spec=IOSpec.MutableInput, ...
]
comptime _FusedOutputTensor = ManagedTensorSlice[
    io_spec=IOSpec.FusedOutput, ...
]
comptime _FusedInputTensor = ManagedTensorSlice[io_spec=IOSpec.FusedInput, ...]

comptime _FusedComputeOutputTensor = ManagedTensorSlice[
    io_spec=IOSpec._FusedComputeOutput, ...
]

comptime _FusedComputeOutputTileTensor = ManagedTensorSlice[
    io_spec=IOSpec._FusedComputeOutputTile, ...
]

comptime _FusedOutputTileTensor = ManagedTensorSlice[
    io_spec=IOSpec._FusedOutputTile, ...
]

comptime DynamicTensor[dtype: DType, rank: Int] = ManagedTensorSlice[
    io_spec=IOSpec.Unknown,
    static_spec=StaticTensorSpec[dtype, rank, ...].get_unknown(),
]


@always_inline
def _index_list_to_static_coord[
    element_types: TypeList[Trait=CoordLike, ...],
](values: IndexList) -> Coord[*element_types]:
    """Builds a `Coord` of the given element types from a runtime `IndexList`.

    Static elements keep their compile-time values (from the default-constructed
    `Coord`), while dynamic elements are filled from `values`. This preserves
    the static-vs-dynamic structure encoded in `element_types`.

    Parameters:
        element_types: The `CoordLike` element types of the resulting `Coord`.

    Args:
        values: The runtime values used to fill the dynamic elements.

    Returns:
        A `Coord` with the given element types.
    """
    comptime assert values.size == element_types.length, "rank mismatch"
    var result = Coord[*element_types]()

    comptime for i in range(element_types.length):
        comptime if not result.element_types[i].is_static_value:
            result[i] = rebind[result.element_types[i]](
                Scalar[result.element_types[i].DTYPE](values[i])
            )

    return result


@fieldwise_init
struct ManagedTensorSlice[
    mut: Bool,
    input: IO,
    dtype: DType,
    rank: Int,
    InFusion: InputFusion,
    OutFusion: OutputFusion,
    ComputeFusion: ComputeOutputFusion,
    ComputeFusionTile: ComputeOutputFusionTile,
    OutFusionTile: OutputFusionTile,
    //,
    io_spec: IOSpec[mut, input],
    *,
    static_spec: StaticTensorSpec[
        dtype,
        rank,
        _,
        InFusion,
        OutFusion,
        ComputeFusion,
        ComputeFusionTile,
        OutFusionTile,
    ],
](DevicePassable, TrivialRegisterPassable, Writable):
    """A view of a tensor that does not own the underlying allocated pointer.
    When the object lifetime ends it does not free the underlying pointer.
    Conversely, if a `ManagedTensorSlice` is created, it will not extend the
    life of the underlying pointer.

    Therefore, the user must take care to keep the pointer alive until the last
    use of a `ManagedTensorSlice` instance. This class is useful for writing
    custom operations where memory is managed by an external runtime like in
    MAX's inference stack.
    """

    # `trait DevicePassable` implementation
    comptime device_type: AnyType = LayoutTensor[
        Self.dtype, Self.static_spec.to_layout(), MutAnyOrigin
    ]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.to_layout_tensor(), target)

    @staticmethod
    def get_type_name() -> String:
        return (
            "ManagedTensorSlice[mut = "
            + String(Self.mut)
            + ", dtype = "
            + String(Self.dtype)
            + ", rank = "
            + String(Self.rank)
            + ", static_spec (as Layout) = "
            + String(Self.static_spec.to_layout())
            + "]"
        )

    comptime address_space = Self.static_spec.address_space
    comptime alignment = Self.static_spec.alignment
    # IntTuple aliases for static shape/strides.
    comptime _static_shape_tuple = Self.static_spec.shape_tuple
    comptime _static_strides_tuple = Self.static_spec.strides_tuple

    # Fusion query aliases.
    # _is_unfused and _has_input_fusion are derived purely from IOSpec.
    # _has_output_store_fusion and _has_compute_fusion must inspect the
    # actual type parameters because _bind_to_fused_compute_output (OutputFusion
    # overload) produces the same IO (_FusedComputeOutput) as the
    # ComputeOutputFusion overload but
    # populates OutFusion instead of ComputeFusion.
    comptime _is_unfused: Bool = not Self.input.is_fused()
    comptime _has_input_fusion: Bool = (Self.input == IO.FusedInput)
    comptime _has_output_store_fusion: Bool = Self.OutFusion != _NoFusionOut
    comptime _has_compute_fusion: Bool = Self.ComputeFusion != _NoComputeFusion
    comptime _has_compute_fusion_tile: Bool = (
        Self.ComputeFusionTile != _NoComputeFusionTile
    )
    comptime _has_output_fusion_tile: Bool = (
        Self.OutFusionTile != _NoOutputFusionTile
    )

    comptime RuntimeLayout = TileLayout[
        shape_types=Self.static_spec.static_layout._shape_types,
        stride_types=Self.static_spec.static_layout._stride_types,
    ]

    var _ptr: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _runtime_layout: Self.RuntimeLayout
    var in_fusion: Self.InFusion
    var out_fusion: Self.OutFusion
    var compute_fusion: Self.ComputeFusion
    var compute_fusion_tile: Self.ComputeFusionTile
    var output_fusion_tile: Self.OutFusionTile

    @staticmethod
    @always_inline
    def _sentinel_in_fusion() -> Self.InFusion:
        """Return a sentinel InFusion value, or an uninitialized placeholder
        when the type parameter is a real fusion struct (never reached at
        runtime, but must compile for all instantiations)."""
        comptime if Self.InFusion == _NoFusionIn:
            return rebind[Self.InFusion](_NoFusionIn())
        else:
            var f: Self.InFusion
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(f)
            )
            return f

    @staticmethod
    @always_inline
    def _sentinel_out_fusion() -> Self.OutFusion:
        """Return a sentinel OutFusion value, or an uninitialized placeholder
        when the type parameter is a real fusion struct."""
        comptime if Self.OutFusion == _NoFusionOut:
            return rebind[Self.OutFusion](_NoFusionOut())
        else:
            var f: Self.OutFusion
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(f)
            )
            return f

    @staticmethod
    @always_inline
    def _sentinel_compute_fusion() -> Self.ComputeFusion:
        """Return a sentinel ComputeFusion value, or an uninitialized
        placeholder when the type parameter is a real fusion struct."""
        comptime if Self.ComputeFusion == _NoComputeFusion:
            return rebind[Self.ComputeFusion](_NoComputeFusion())
        else:
            var f: Self.ComputeFusion
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(f)
            )
            return f

    @staticmethod
    @always_inline
    def _sentinel_compute_fusion_tile() -> Self.ComputeFusionTile:
        """Return a sentinel ComputeFusionTile value, or an uninitialized
        placeholder when the type parameter is a real fusion struct."""
        comptime if Self.ComputeFusionTile == _NoComputeFusionTile:
            return rebind[Self.ComputeFusionTile](_NoComputeFusionTile())
        else:
            var f: Self.ComputeFusionTile
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(f)
            )
            return f

    @staticmethod
    @always_inline
    def _sentinel_output_fusion_tile() -> Self.OutFusionTile:
        """Return a sentinel OutFusionTile value, or an uninitialized
        placeholder when the type parameter is a real fusion struct."""
        comptime if Self.OutFusionTile == _NoOutputFusionTile:
            return rebind[Self.OutFusionTile](_NoOutputFusionTile())
        else:
            var f: Self.OutFusionTile
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(f)
            )
            return f

    @staticmethod
    @always_inline
    def _make_runtime_layout(
        shape: IndexList[Self.rank], strides: IndexList[Self.rank]
    ) -> Self.RuntimeLayout:
        """Builds the runtime layout from a runtime shape and strides.

        Static dimensions keep their compile-time values; dynamic dimensions are
        filled from `shape`/`strides`.
        """
        return Self.RuntimeLayout(
            _index_list_to_static_coord[
                Self.static_spec.static_layout._shape_types
            ](shape),
            _index_list_to_static_coord[
                Self.static_spec.static_layout._stride_types
            ](strides),
        )

    def __init__(
        out self,
        ptr: Pointer[mut=True, Scalar[Self.dtype], _],
        shape: IndexList[Self.rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer and shape.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self._ptr = ptr.unsafe_origin_cast[MutUntrackedOrigin]()
        self._runtime_layout = Self._make_runtime_layout(
            shape, shape.get_row_major_strides()
        )
        self.in_fusion = Self._sentinel_in_fusion()
        self.out_fusion = Self._sentinel_out_fusion()
        self.compute_fusion = Self._sentinel_compute_fusion()
        self.compute_fusion_tile = Self._sentinel_compute_fusion_tile()
        self.output_fusion_tile = Self._sentinel_output_fusion_tile()

    def __init__(
        out self,
        ptr: Pointer[mut=True, Scalar[Self.dtype], _],
        shape: IndexList[Self.rank],
        strides: IndexList[Self.rank],
    ):
        """Initializes a ManagedTensorSlice from a pointer, shape, and strides.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self._ptr = ptr.unsafe_origin_cast[MutUntrackedOrigin]()
        self._runtime_layout = Self._make_runtime_layout(shape, strides)
        self.in_fusion = Self._sentinel_in_fusion()
        self.out_fusion = Self._sentinel_out_fusion()
        self.compute_fusion = Self._sentinel_compute_fusion()
        self.compute_fusion_tile = Self._sentinel_compute_fusion_tile()
        self.output_fusion_tile = Self._sentinel_output_fusion_tile()

    def __init__(
        out self,
        ptr: Pointer[mut=True, Scalar[Self.dtype], _],
        shape: Coord[*Self.RuntimeLayout.shape_types],
        strides: Coord[*Self.RuntimeLayout.stride_types],
    ):
        """Initializes a ManagedTensorSlice from a pointer, shape, and strides.

        The shape and strides are provided as `Coord`s; their runtime values
        fill the dynamic dimensions of the tensor slice's layout.

        In general, custom operations should not create `ManagedTensorSlice`
        instances, but instead use the ones provided by the MAX inference
        engine.
        """
        self._ptr = ptr.unsafe_origin_cast[MutUntrackedOrigin]()
        self._runtime_layout = Self.RuntimeLayout(shape, strides)
        self.in_fusion = Self._sentinel_in_fusion()
        self.out_fusion = Self._sentinel_out_fusion()
        self.compute_fusion = Self._sentinel_compute_fusion()
        self.compute_fusion_tile = Self._sentinel_compute_fusion_tile()
        self.output_fusion_tile = Self._sentinel_output_fusion_tile()

    @always_inline
    def __getitem__(self, indices: IndexList[Self.rank]) -> Scalar[Self.dtype]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        comptime assert (
            not Self._has_input_fusion
        ), "Direct load on fused tensor is forbidden"
        var offset = self._compute_offset(indices)
        return self._ptr[unsafe_offset=offset]

    @always_inline
    def __getitem__(self, *indices: Int) -> Scalar[Self.dtype]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        comptime assert (
            not Self._has_input_fusion
        ), "Direct load on fused tensor is forbidden"
        assert (
            len(indices) == Self.rank
        ), "mismatch between requested index and rank"
        return self[IndexList[Self.rank](*indices)]

    @always_inline
    def __setitem__(self, *indices: Int, val: Scalar[Self.dtype]):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        comptime assert (
            not Self._has_output_store_fusion
        ), "Direct store on fused tensor is forbidden"
        assert (
            len(indices) == Self.rank
        ), "mismatch between requested index and rank"
        self[IndexList[Self.rank](*indices)] = val

    @always_inline
    def __setitem__(
        self, indices: IndexList[Self.rank], val: Scalar[Self.dtype]
    ):
        """Stores the value at the specified indices.

        Args:
          indices: The indices of the value to store.
          val: The value to store.

        """
        comptime assert (
            not Self._has_output_store_fusion
        ), "Direct store on fused tensor is forbidden"
        var offset = self._compute_offset(indices)
        self._ptr[unsafe_offset=offset] = val

    @always_inline
    def shape(self) -> IndexList[Self.rank]:
        """Gets the shape of this tensor slice, as an `IndexList`.

        Returns:
            The shape of this tensor slice.
        """
        var result = IndexList[Self.rank]()

        comptime for i in range(Self.rank):
            result[i] = Int(self._runtime_layout.shape[i]().value())

        return result

    @always_inline
    def shape_coord(
        self,
    ) -> Coord[*Self.RuntimeLayout.shape_types]:
        """Gets the shape of this tensor slice as a `Coord`.

        Unlike `shape`, which returns a runtime `IndexList`, the returned
        `Coord` preserves the static-vs-dynamic structure of the tensor's static
        layout: statically-known dimensions are encoded as compile-time values
        in the `Coord`'s type, while dynamic dimensions are filled from the
        runtime shape.

        Returns:
            The shape of this tensor slice as a `Coord`.
        """
        return self._runtime_layout.shape_coord()

    @always_inline
    def strides_coord(
        self,
    ) -> Coord[*Self.RuntimeLayout.stride_types]:
        """Gets the strides of this tensor slice as a `Coord`.

        Unlike `strides`, which returns a runtime `IndexList`, the returned
        `Coord` preserves the static-vs-dynamic structure of the tensor's static
        layout: statically-known strides are encoded as compile-time values in
        the `Coord`'s type, while dynamic strides are filled from the runtime
        strides.

        Returns:
            The strides of this tensor slice as a `Coord`.
        """
        return self._runtime_layout.stride_coord()

    @always_inline
    def runtime_layout(self) -> Self.RuntimeLayout:
        """Gets the runtime layout of this tensor slice.

        The layout bundles the shape and strides as `Coord`s, preserving the
        static-vs-dynamic structure of the tensor's static layout.

        Returns:
            The runtime layout of this tensor slice.
        """
        return self._runtime_layout

    @always_inline
    def dim_size(self, index: Int) -> Int:
        """Gets the size of a given dimension of this tensor slice using a run
        time value.

        Args:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """
        return self.shape()[index]

    @always_inline
    def dim_size[index: Int](self) -> Int:
        """Gets the size of a given dimension of this tensor slice using a
        compile time value.

        Parameters:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """

        comptime assert 0 <= index < Self.rank, String(
            t"dim_size index of {index} is out of bounds for tensor rank [0,"
            t" {Self.rank}]"
        )

        return Int(self._runtime_layout.shape[index]().value())

    @always_inline
    def strides(self) -> IndexList[Self.rank]:
        """Gets the strides of this tensor slice, as an `IndexList`.

        Returns:
            The strides of this tensor slice.
        """
        var result = IndexList[Self.rank]()

        comptime for i in range(Self.rank):
            result[i] = Int(self._runtime_layout.stride[i]().value())

        return result

    @always_inline
    def stride_length(self, index: Int) -> Int:
        """Gets the length of the stride of a given dimension of this tensor
        slice using a run time value.

        Args:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """
        return self.strides()[index]

    @always_inline
    def stride_length[index: Int](self) -> Int:
        """Gets the length of the stride of a given dimension of this tensor
        slice using a compile time value.

        Parameters:
            index: The zero-based index of the dimension.

        Returns:
            The size of the tensor slice in the given dimension.
        """

        comptime assert 0 <= index < Self.rank, String(
            t"stride_length index of {index} is out of bounds for tensor rank"
            t" [0, {Self.rank}]"
        )

        return Int(self._runtime_layout.stride[index]().value())

    @always_inline
    def size(self) -> Int:
        """Computes the tensor slice's number of elements.

        Returns:
            The total number of elements in the tensor slice.
        """
        return Int(self._runtime_layout.size())

    @always_inline
    def bytecount(self) -> Int:
        """Returns the size of the tensor slice in bytes.

        Returns:
            The total number of bytes in the tensor slice.
        """
        return self.size() * size_of[Self.dtype]()

    @always_inline
    def unsafe_ptr[
        _dtype: DType = Self.dtype
    ](self) -> Pointer[Scalar[_dtype], MutAnyOrigin]:
        """Get the pointer stored in this tensor slice.

        Since this method obtains the pointer stored in this tensor slice, it
        can modify the invariants of this tensor slice and lead to unexpected
        behavior. It should be used with caution.

        Parameters:
            _dtype: The type of the `Pointer` in this tensor slice.

        Returns:
            The `Pointer` which contains the data for this tensor slice.
        """
        return rebind[Pointer[Scalar[_dtype], MutAnyOrigin]](self._ptr)

    @always_inline
    def to_device_buffer(self, ctx: DeviceContext) -> DeviceBuffer[Self.dtype]:
        var size = self.size()
        if size > 0:
            return DeviceBuffer[Self.dtype](
                ctx,
                self.unsafe_ptr(),
                size,
                owning=False,
            )
        else:
            return DeviceBuffer[Self.dtype].empty(ctx)

    @always_inline
    def load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](self, index: IndexList[_rank]) -> SIMD[Self.dtype, width]:
        """Gets data from this tensor slice as a `SIMD`.

        Parameters:
            width: The width of the `SIMD` value. This must be large enough to contain the data from this tensor slice.
            _rank: The rank of the tensor slice.
            element_alignment: Indicate the alignment of the pointer stored to memory. This is needed to issue vector load for GPUs with strict alignment requirements.

        Args:
            index: An `IndexList` of size `_rank` to indicate the dimension of the tensor slice to obtain data from.

        Returns:
            Data from this tensor slice at dimension `index`.
        """
        comptime assert (
            Self.input == IO.Input or Self.input == IO.Unknown
        ), "loading not supported for output tensors"

        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)
        return simd_load_from_managed_tensor_slice[
            simd_width=width, element_alignment=element_alignment
        ](self, ridx)

    @always_inline
    def load[
        width: Int,
        element_alignment: Int = 1,
    ](self, index: Coord) -> SIMD[Self.dtype, width]:
        """Gets data from this tensor slice as a `SIMD`, indexed by a `Coord`.

        Parameters:
            width: The width of the `SIMD` value. This must be large enough to contain the data from this tensor slice.
            element_alignment: Indicate the alignment of the pointer stored to memory. This is needed to issue vector load for GPUs with strict alignment requirements.

        Args:
            index: A `Coord` indicating the dimension of the tensor slice to obtain data from.

        Returns:
            Data from this tensor slice at dimension `index`.
        """
        comptime assert index.rank == Self.rank
        return self.load[width, element_alignment=element_alignment](
            rebind[IndexList[Self.rank]](coord_to_index_list(index))
        )

    @always_inline
    def _fused_load[
        width: Int,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](self, index: IndexList[_rank]) -> SIMD[Self.dtype, width]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime if Self._has_input_fusion:
            return self.in_fusion.load[
                Self.dtype, Self.rank, width, element_alignment
            ](ridx)
        else:
            return simd_load_from_managed_tensor_slice[
                simd_width=width, element_alignment=element_alignment
            ](self, ridx)

    @always_inline
    def _fused_load[
        width: Int,
        element_alignment: Int = 1,
    ](self, index: Coord) -> SIMD[Self.dtype, width]:
        comptime assert index.rank == Self.rank
        return self._fused_load[width, element_alignment=element_alignment](
            rebind[IndexList[Self.rank]](coord_to_index_list(index))
        )

    @always_inline("nodebug")
    def _lambda_load[
        width: Int,
        _rank: Int,
        element_alignment: Int = 1,
    ](self, index: IndexList[_rank]) -> SIMD[Self.dtype, width]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime assert (
            Self._has_input_fusion
        ), "_lambda_load called on unfused tensor"
        return self.in_fusion.load[
            Self.dtype, Self.rank, width, element_alignment
        ](ridx)

    @always_inline
    def _lambda_load[
        width: Int,
        element_alignment: Int = 1,
    ](self, index: Coord) -> SIMD[Self.dtype, width]:
        comptime assert index.rank == Self.rank
        return self._lambda_load[width, element_alignment=element_alignment](
            rebind[IndexList[Self.rank]](coord_to_index_list(index))
        )

    @always_inline
    def _compute_offset(self, index: IndexList[Self.rank]) -> Int:
        comptime if Self.rank == 0:
            return 0

        # Special case for NVidia GPU on shared memory.
        # We can do the offset computation in int32 instead.
        comptime if is_gpu() and Self.address_space in (
            AddressSpace.SHARED,
            AddressSpace.LOCAL,
            AddressSpace.CONSTANT,
        ):
            var offset: Int32 = 0

            comptime for i in range(Self.rank):
                offset = fma(
                    Int32(index[i]),
                    Int32(self._runtime_layout.stride[i]().value()),
                    offset,
                )
            return Int(offset)

        var offset = 0

        comptime for i in range(Self.rank):
            offset = fma(
                Int(index[i]),
                Int(self.stride_length[i]()),
                offset,
            )

        return offset

    @always_inline
    def _compute_offset(self, index: Coord) -> Int:
        comptime assert index.rank == Self.rank

        comptime if Self.rank == 0:
            return 0

        var shape = self.shape_coord()
        var strides = self.strides_coord()

        comptime if is_gpu() and Self.address_space in (
            AddressSpace.SHARED,
            AddressSpace.LOCAL,
            AddressSpace.CONSTANT,
        ):
            return Int(crd2idx[out_type=DType.int32](index, shape, strides))

        return Int(crd2idx[out_type=DType.int](index, shape, strides))

    @__allow_legacy_custom_self_type
    @always_inline
    def store[
        width: SIMDLength,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ):
        """Sets data in this tensor slice from a `SIMD`.

        Parameters:
            width: The width of the `SIMD` value.
            _rank: The rank of the tensor slice.
            element_alignment: Indicate the alignment of the pointer stored to memory. This is needed to issue vector store for GPUs with strict alignment requirements.

        Args:
            index: An `IndexList` of size `_rank` to indicate the dimension of the tensor slice to set data in.
            val: The data to set into this tensor slice.
        """
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        simd_store_into_managed_tensor_slice[
            simd_width=width,
            element_alignment=element_alignment,
        ](self, ridx, val)

    @__allow_legacy_custom_self_type
    @always_inline
    def store[
        width: SIMDLength,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        index: Coord,
        val: SIMD[Self.dtype, width],
    ):
        """Sets data in this tensor slice from a `SIMD`, indexed by a `Coord`.

        Parameters:
            width: The width of the `SIMD` value.
            element_alignment: Indicate the alignment of the pointer stored to memory. This is needed to issue vector store for GPUs with strict alignment requirements.

        Args:
            index: A `Coord` indicating the dimension of the tensor slice to set data in.
            val: The data to set into this tensor slice.
        """
        comptime assert index.rank == Self.rank
        self.store[width, element_alignment=element_alignment](
            rebind[IndexList[Self.rank]](coord_to_index_list(index)), val
        )

    @__allow_legacy_custom_self_type
    @always_inline
    def _fused_store[
        width: SIMDLength,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ):
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime if Self._has_output_store_fusion:
            self.out_fusion.store[
                Self.dtype, Self.rank, width, element_alignment
            ](ridx, val)
        else:
            simd_store_into_managed_tensor_slice[
                simd_width=width,
                element_alignment=element_alignment,
            ](self, ridx, val)

    @__allow_legacy_custom_self_type
    @always_inline
    def _fused_store[
        width: SIMDLength,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        index: Coord,
        val: SIMD[Self.dtype, width],
    ):
        comptime assert index.rank == Self.rank
        self._fused_store[width, element_alignment=element_alignment](
            rebind[IndexList[Self.rank]](coord_to_index_list(index)), val
        )

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def _lambda_store[
        width: SIMDLength,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[
            io_spec=IOSpec[True, Self.input](),
            static_spec=Self.static_spec,
        ],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ):
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime assert (
            Self._has_output_store_fusion
        ), "_lambda_store called on unfused tensor"
        self.out_fusion.store[Self.dtype, Self.rank, width, element_alignment](
            ridx, val
        )

    @__allow_legacy_custom_self_type
    @always_inline
    def _lambda_store[
        width: SIMDLength,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[
            io_spec=IOSpec[True, Self.input](),
            static_spec=Self.static_spec,
        ],
        index: Coord,
        val: SIMD[Self.dtype, width],
    ):
        comptime assert index.rank == Self.rank
        self._lambda_store[width, element_alignment=element_alignment](
            rebind[IndexList[Self.rank]](coord_to_index_list(index)), val
        )

    @__allow_legacy_custom_self_type
    @always_inline
    def _fused_compute_output_lambda[
        width: SIMDLength,
        # Necessary to make it simpler on the call site.
        _rank: Int,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        index: IndexList[_rank],
        val: SIMD[Self.dtype, width],
    ) -> SIMD[Self.dtype, width]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](index)

        comptime if Self._has_compute_fusion:
            return self.compute_fusion.compute[
                Self.dtype, Self.rank, width, element_alignment
            ](ridx, val)
        else:
            return val

    @__allow_legacy_custom_self_type
    @always_inline
    def _fused_compute_output_lambda[
        width: SIMDLength,
        element_alignment: Int = 1,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        index: Coord,
        val: SIMD[Self.dtype, width],
    ) -> SIMD[Self.dtype, width]:
        comptime assert index.rank == Self.rank
        return self._fused_compute_output_lambda[
            width, element_alignment=element_alignment
        ](rebind[IndexList[Self.rank]](coord_to_index_list(index)), val)

    @__allow_legacy_custom_self_type
    @always_inline
    def _fused_compute_output_tile_lambda[
        _rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        tile_coords: IndexList[_rank],
        copier: Copier,
        val: TileTensor[Self.dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[Self.dtype, LayoutType, MutAnyOrigin]:
        comptime assert _rank == Self.rank
        var ridx = rebind[IndexList[Self.rank]](tile_coords)

        comptime if Self._has_compute_fusion_tile:
            return self.compute_fusion_tile.compute[
                Self.dtype, Self.rank, LayoutType, Copier
            ](ridx, copier, val)
        else:
            return val

    @__allow_legacy_custom_self_type
    @always_inline
    def _fused_compute_output_tile_lambda[
        LayoutType: TensorLayout,
        Copier: TileCopier,
    ](
        self: ManagedTensorSlice[mut=True, static_spec=Self.static_spec, ...],
        tile_coords: Coord,
        copier: Copier,
        val: TileTensor[Self.dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[Self.dtype, LayoutType, MutAnyOrigin]:
        comptime assert tile_coords.rank == Self.rank
        return self._fused_compute_output_tile_lambda(
            rebind[IndexList[Self.rank]](coord_to_index_list(tile_coords)),
            copier,
            val,
        )

    @always_inline
    def with_tile_layout[
        new_layout: TensorLayout,
    ](
        self,
        new_runtime_shape: IndexList[new_layout.rank],
        new_runtime_strides: IndexList[new_layout.rank],
        offset_ptr: Optional[Pointer[Scalar[Self.dtype], MutAnyOrigin]] = None,
        out result: ManagedTensorSlice[
            rank=new_layout.rank,
            io_spec=Self.io_spec,
            static_spec=Self.static_spec.with_tile_layout[new_layout](),
        ],
    ):
        return type_of(result)(
            offset_ptr.or_else(self._ptr.as_unsafe_any_origin()),
            new_runtime_shape,
            new_runtime_strides,
        )

    @doc_hidden
    @always_inline
    def _bind_to_fused_input[
        F: InputFusion
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=IOSpec.FusedInput,
            static_spec=Self.static_spec.with_input_fusion[F](),
        ],
    ):
        """Bind a trait-based input fusion struct to this tensor.

        The returned MTS dispatches loads through `fusion.load()` instead of
        reading from the underlying data pointer.
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        # rebind needed for unfused slots: _is_unfused guarantees the type
        # params equal _NoFusionOut/_NoComputeFusion, but the compiler can't
        # prove it statically.
        return {
            self._ptr,
            self._runtime_layout,
            fusion,
            rebind[type_of(result).OutFusion](_NoFusionOut()),
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
            rebind[type_of(result).ComputeFusionTile](_NoComputeFusionTile()),
            rebind[type_of(result).OutFusionTile](_NoOutputFusionTile()),
        }

    @doc_hidden
    @always_inline
    def _bind_to_fused_output[
        F: OutputFusion
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=IOSpec.FusedOutput,
            static_spec=Self.static_spec.with_output_fusion[F](),
        ],
    ):
        """Bind a trait-based output fusion struct to this tensor.

        The returned MTS dispatches stores through `fusion.store()` instead of
        writing to the underlying data pointer.
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        return {
            self._ptr,
            self._runtime_layout,
            rebind[type_of(result).InFusion](_NoFusionIn()),
            fusion,
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
            rebind[type_of(result).ComputeFusionTile](_NoComputeFusionTile()),
            rebind[type_of(result).OutFusionTile](_NoOutputFusionTile()),
        }

    @doc_hidden
    @always_inline
    def _bind_to_fused_compute_output[
        F: OutputFusion
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=IOSpec._FusedComputeOutput,
            static_spec=Self.static_spec.with_output_fusion[F](),
        ],
    ):
        """Bind an OutputFusion struct but with _FusedComputeOutput io_spec.

        Used for the OutputLegacyForCompute case: the kernel expects
        _FusedComputeOutput io_spec but the fusion performs a store.
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        return {
            self._ptr,
            self._runtime_layout,
            rebind[type_of(result).InFusion](_NoFusionIn()),
            fusion,
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
            rebind[type_of(result).ComputeFusionTile](_NoComputeFusionTile()),
            rebind[type_of(result).OutFusionTile](_NoOutputFusionTile()),
        }

    @doc_hidden
    @always_inline
    def _bind_to_fused_compute_output[
        F: ComputeOutputFusion
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=IOSpec._FusedComputeOutput,
            static_spec=Self.static_spec.with_compute_fusion[F](),
        ],
    ):
        """Bind a trait-based compute-output fusion struct to this tensor.

        The returned MTS dispatches compute-output through
        `fusion.compute()` to transform values before the final store.
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        return {
            self._ptr,
            self._runtime_layout,
            rebind[type_of(result).InFusion](_NoFusionIn()),
            rebind[type_of(result).OutFusion](_NoFusionOut()),
            fusion,
            rebind[type_of(result).ComputeFusionTile](_NoComputeFusionTile()),
            rebind[type_of(result).OutFusionTile](_NoOutputFusionTile()),
        }

    @doc_hidden
    @always_inline
    def _bind_to_fused_compute_output_tile[
        F: ComputeOutputFusionTile
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=IOSpec._FusedComputeOutputTile,
            static_spec=Self.static_spec.with_compute_fusion_tile[F](),
        ],
    ):
        """Bind a tile-based compute-output fusion struct to this tensor.

        The returned MTS dispatches tile-based compute-output through
        `fusion.compute()` to transform tile values before the final store.
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        return {
            self._ptr,
            self._runtime_layout,
            rebind[type_of(result).InFusion](_NoFusionIn()),
            rebind[type_of(result).OutFusion](_NoFusionOut()),
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
            fusion,
            rebind[type_of(result).OutFusionTile](_NoOutputFusionTile()),
        }

    @doc_hidden
    @always_inline
    def _bind_to_fused_output_tile[
        F: OutputFusionTile
    ](
        self,
        fusion: F,
        out result: ManagedTensorSlice[
            dtype=Self.dtype,
            rank=Self.rank,
            io_spec=IOSpec._FusedOutputTile,
            static_spec=Self.static_spec.with_output_fusion_tile[F](),
        ],
    ):
        """Bind a tile-based output fusion struct to this tensor.

        The returned MTS carries the fusion in `output_fusion_tile`; the
        fusion's `store()` owns the write, so the primary kernel skips its own
        store path (the tile analog of `_bind_to_fused_output`).
        """
        comptime assert (
            Self._is_unfused
        ), "The tensor is already bound to a fusion struct"
        return {
            self._ptr,
            self._runtime_layout,
            rebind[type_of(result).InFusion](_NoFusionIn()),
            rebind[type_of(result).OutFusion](_NoFusionOut()),
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
            rebind[type_of(result).ComputeFusionTile](_NoComputeFusionTile()),
            fusion,
        }

    @always_inline
    def to_layout_tensor(
        self,
        out result: LayoutTensor[
            Self.dtype, Self.static_spec.to_layout(), MutAnyOrigin
        ],
    ):
        comptime layout = Self.static_spec.to_layout()
        return type_of(result)(
            self.unsafe_ptr(),
            type_of(result.runtime_layout)(
                self.shape().cast[result.layout_int_type](),
                self.strides().cast[result.linear_idx_type](),
            ),
        )

    @always_inline
    def to_tile_tensor[
        coord_dtype: DType = .int64
    ](
        self,
        out result: TileTensor[
            Self.dtype,
            origin=MutUntrackedOrigin,
            LayoutType=Self.RuntimeLayout,
        ],
    ):
        return {
            self.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            self._runtime_layout,
        }

    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this buffer to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write("ManagedTensorSlice(")

        @__parameter
        def serialize[T: Writable](val: T):
            writer.write(val)

        var shape = List[Int]()
        for i in range(Self.rank):
            shape.append(self.shape()[i])

        # Pin the argument to a concrete-origin optional pointer so this call
        # stays origin-exact across the `_serialize` safe-Pointer flip.
        var serialize_ptr = OptionalPointer[Scalar[Self.dtype], ImmutAnyOrigin](
            self._ptr.as_imm().unsafe_origin_cast[ImmutAnyOrigin]()
        )
        # TODO(1937): make this work with all valid strides
        _serialize[serialize_fn=serialize, serialize_end_line=False](
            serialize_ptr, shape
        )

        writer.write("){")
        writer.write("static_shape = ", self._static_shape_tuple)
        writer.write(", static_strides = ", self._static_strides_tuple)
        writer.write(", dynamic_shape = ", self.shape())
        writer.write(", dynamic_strides = ", self.strides())
        writer.write(", alignment = ", self.alignment)
        writer.write(", address_space = ", self.address_space)
        writer.write("}")

    def write_repr_to(self, mut writer: Some[Writer]):
        """
        Formats this buffer to the provided Writer.

        Args:
            writer: The object to write to.
        """
        self.write_to(writer)


# TODO: Move to Mojo/stdlib/stdlib/runtime/tracing.mojo and
# rename to trace_arg
@always_inline
def trace_slice_arg(name: String, buf: ManagedTensorSlice) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.

    Args:
        name: The name of the argument.
        buf: The tensor to trace.

    Returns:
        A string representation of the buffer with its shape and data type.
    """
    return trace_arg(name, buf.strides(), buf.dtype)


# ===----------------------------------------------------------------------=== #
# VariadicTensors
# ===----------------------------------------------------------------------=== #

comptime InputVariadicTensors = VariadicTensors[io_spec=IOSpec.Input, ...]
comptime OutputVariadicTensors = VariadicTensors[io_spec=IOSpec.Output, ...]

comptime _MutableInputVariadicTensors = VariadicTensors[
    io_spec=IOSpec.MutableInput, ...
]


@fieldwise_init
struct StaticTensorSpecList[
    dtype: DType,
    rank: Int,
    //,
    internals_list: ParameterList[
        type=StaticTensorSpecInternal[dtype, rank], ...
    ],
    shapes_list: ParameterList[type=IndexList[rank], ...],
    strides_list: ParameterList[type=IndexList[rank], ...],
]:
    """A statically indexable list of data that can be assembled into a
    StaticTensorSpecList on demand. This handles the complexities that arise
    with heterogenous specs.
    """

    def __getitem_param__[
        index: Int
    ](
        self,
        out result: StaticTensorSpec[
            Self.dtype,
            Self.rank,
            static_layout=_IndexListToTileLayout[
                Self.shapes_list[index],
                Self.strides_list[index],
            ],
        ],
    ):
        return {Self.internals_list[index]}


@fieldwise_init
struct VariadicTensors[
    mut: Bool,
    input: IO,
    dtype: DType,
    rank: Int,
    //,
    size: Int,
    io_spec: IOSpec[mut, input],
    *,
    static_specs: StaticTensorSpecList[dtype=dtype, rank=rank, ...],
](Sized, TrivialRegisterPassable):
    """A tuple-like container of tensors representing variadic arguments from
    the graph compiler."""

    var _tensors: StaticTuple[DynamicTensor[Self.dtype, Self.rank], Self.size]

    def __init__(
        out self,
        ptrs: StaticTuple[
            Pointer[Scalar[Self.dtype], MutUntrackedOrigin], Self.size
        ],
        shapes: StaticTuple[IndexList[Self.rank], Self.size],
    ):
        """Initialize the variadic tensor from tuples of pointers and shapes.

        This is a bulk initialization of the VariadicTensors value from an
        array of pointers and an array of runtime shapes. This allows the graph
        compiler to avoid generating code to construct DynamicTensor values
        directly.
        """

        self._tensors = {}

        for i in range(Self.size):
            var tensor = DynamicTensor[Self.dtype, Self.rank](
                ptrs[i], shapes[i]
            )
            self._tensors._unsafe_ref(i) = tensor

    def __len__(self) -> Int:
        """Returns the number of variadic arguments in the pack.

        Returns:
            The number of variadic arguments.
        """
        return Self.size

    def __getitem_param__[
        index: Int
    ](
        self,
        out result: ManagedTensorSlice[
            io_spec=Self.io_spec, static_spec=Self.static_specs[index]
        ],
    ):
        """Returns the tensor at the given position in the variadic argument
        argument pack.

        Parameters:
            index: The index into the variadic tensor arguments.

        Returns:
            The tensor at the specified index.
        """
        comptime assert index < Self.size
        var tensor = self._tensors[index]
        return type_of(result)(
            tensor._ptr,
            tensor.shape(),
            tensor.strides(),
        )


# ===----------------------------------------------------------------------=== #
# New VariadicTensors (trait-based fusion, no capturing closures)
# ===----------------------------------------------------------------------=== #


struct _FusionPack[*Ts: TrivialRegisterPassable](TrivialRegisterPassable):
    """TrivialRegisterPassable heterogeneous pack for fusion structs.

    Unlike Tuple, this uses a value-based `!kgen.struct` (not reference-based),
    making it safe to pass across the host-device boundary via GPU closures.
    """

    comptime _mlir_type = __mlir_type[
        `!kgen.struct<`, ~Self.Ts.values, ` isParamPack>`
    ]
    var _mlir_value: Self._mlir_type

    @always_inline("nodebug")
    def __init__(out self, *args: *Self.Ts):
        self._mlir_value = __mlir_op.`kgen.rebind`[_type=Self._mlir_type](
            args.get_loaded_kgen_pack()
        )

    @always_inline("nodebug")
    def __getitem_param__[i: Int](self) -> Self.Ts[i]:
        return __mlir_op.`kgen.struct.extract`[index=i.__mlir_index__()](
            self._mlir_value
        )


struct _FusedInputVariadicTensors[
    dtype: DType,
    rank: Int,
    size: Int,
    //,
    *FusionTypes: InputFusion,
    static_specs: StaticTensorSpecList[dtype=dtype, rank=rank, ...],
](Sized, TrivialRegisterPassable):
    """Variadic input tensors with per-element heterogeneous fusion.

    Tensor data (ptr, shape, strides) is stored in a homogeneous StaticTuple.
    Per-element fusion structs are stored in a _FusionPack, where each
    element conforms to InputFusion. Every element must have a real fusion
    struct; use plain VariadicTensors for unfused variadics.
    """

    var _tensors: StaticTuple[DynamicTensor[Self.dtype, Self.rank], Self.size]
    var _fusions: _FusionPack[*Self.FusionTypes]

    def __init__(
        out self,
        ptrs: StaticTuple[
            Pointer[Scalar[Self.dtype], origin=MutUntrackedOrigin],
            Self.size,
        ],
        shapes: StaticTuple[IndexList[Self.rank], Self.size],
        fusions: _FusionPack[*Self.FusionTypes],
    ):
        comptime for i in range(Self.size):
            comptime assert Self.FusionTypes[i] != _NoFusionIn, (
                "_FusedInputVariadicTensors requires a real fusion struct"
                " for every element; use plain VariadicTensors for unfused"
                " inputs"
            )
        self._tensors = {}
        for i in range(Self.size):
            self._tensors._unsafe_ref(i) = DynamicTensor[Self.dtype, Self.rank](
                ptrs[i], shapes[i]
            )
        self._fusions = fusions

    def __len__(self) -> Int:
        return Self.size

    def __getitem_param__[
        index: Int
    ](
        self,
        out result: ManagedTensorSlice[
            io_spec=IOSpec.FusedInput,
            static_spec=Self.static_specs[index].with_input_fusion[
                Self.FusionTypes[index]
            ](),
        ],
    ):
        """Returns the fused tensor at the given index as a ManagedTensorSlice.

        The returned slice dispatches loads through the element's fusion
        struct, so callers can use `_fused_load` or `_lambda_load` to
        apply the fused computation.

        Parameters:
            index: The index into the variadic tensor arguments.

        Returns:
            The fused tensor at the specified index.
        """
        comptime assert index < Self.size
        var tensor = self._tensors[index]
        return {
            tensor._ptr,
            type_of(result)._make_runtime_layout(
                tensor.shape(), tensor.strides()
            ),
            self._fusions[index],
            rebind[type_of(result).OutFusion](_NoFusionOut()),
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
            rebind[type_of(result).ComputeFusionTile](_NoComputeFusionTile()),
            rebind[type_of(result).OutFusionTile](_NoOutputFusionTile()),
        }

    def shape[index: Int](self) -> IndexList[Self.rank]:
        """Returns the shape of the tensor at the given index."""
        comptime assert index < Self.size
        return self._tensors[index].shape()

    def get_fusion[index: Int](self) -> Self.FusionTypes[index]:
        """Returns the fusion struct at the given index."""
        return self._fusions[index]


struct _FusedOutputVariadicTensors[
    dtype: DType,
    rank: Int,
    size: Int,
    //,
    *FusionTypes: OutputFusion,
    static_specs: StaticTensorSpecList[dtype=dtype, rank=rank, ...],
](Sized, TrivialRegisterPassable):
    """Variadic output tensors with per-element heterogeneous fusion.

    Tensor data is stored in a homogeneous StaticTuple. Per-element fusion
    structs are stored in a _FusionPack, where each element conforms
    to OutputFusion. Every element must have a real fusion struct; use
    plain VariadicTensors for unfused variadics.
    """

    var _tensors: StaticTuple[DynamicTensor[Self.dtype, Self.rank], Self.size]
    var _fusions: _FusionPack[*Self.FusionTypes]

    def __init__(
        out self,
        ptrs: StaticTuple[
            Pointer[Scalar[Self.dtype], origin=MutUntrackedOrigin],
            Self.size,
        ],
        shapes: StaticTuple[IndexList[Self.rank], Self.size],
        fusions: _FusionPack[*Self.FusionTypes],
    ):
        comptime for i in range(Self.size):
            comptime assert Self.FusionTypes[i] != _NoFusionOut, (
                "_FusedOutputVariadicTensors requires a real fusion struct"
                " for every element; use plain VariadicTensors for unfused"
                " outputs"
            )
        self._tensors = {}
        for i in range(Self.size):
            self._tensors._unsafe_ref(i) = DynamicTensor[Self.dtype, Self.rank](
                ptrs[i], shapes[i]
            )
        self._fusions = fusions

    def __len__(self) -> Int:
        return Self.size

    def __getitem_param__[
        index: Int
    ](
        self,
        out result: ManagedTensorSlice[
            io_spec=IOSpec.FusedOutput,
            static_spec=Self.static_specs[index].with_output_fusion[
                Self.FusionTypes[index]
            ](),
        ],
    ):
        """Returns the fused tensor at the given index as a ManagedTensorSlice.

        The returned slice dispatches stores through the element's fusion
        struct, so callers can use `_lambda_store` to apply the fused
        computation.

        Parameters:
            index: The index into the variadic tensor arguments.

        Returns:
            The fused tensor at the specified index.
        """
        comptime assert index < Self.size
        var tensor = self._tensors[index]
        return {
            tensor._ptr,
            type_of(result)._make_runtime_layout(
                tensor.shape(), tensor.strides()
            ),
            rebind[type_of(result).InFusion](_NoFusionIn()),
            self._fusions[index],
            rebind[type_of(result).ComputeFusion](_NoComputeFusion()),
            rebind[type_of(result).ComputeFusionTile](_NoComputeFusionTile()),
            rebind[type_of(result).OutFusionTile](_NoOutputFusionTile()),
        }

    def shape[index: Int](self) -> IndexList[Self.rank]:
        """Returns the shape of the tensor at the given index."""
        comptime assert index < Self.size
        return self._tensors[index].shape()

    def get_fusion[index: Int](self) -> Self.FusionTypes[index]:
        """Returns the fusion struct at the given index."""
        return self._fusions[index]


# ===----------------------------------------------------------------------=== #
# ForEach / view copy primitives
# ===----------------------------------------------------------------------=== #


@doc_hidden
def get_kernel_simd_width[dtype: DType, target: StaticString]() -> Int:
    """Get the simd width used in lambda functions.

    For non-simd arch like GPU, this is the width in terms of number of elements
    used per load/store instruction.
    """

    comptime if _is_gpu[target]():
        # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
        # arch support 32B load/store to global memory, see KERN-2037.
        comptime if CompilationTarget[get_gpu_target()]._is_arch["sm_100a"]():
            return 32 // size_of[dtype]()

        return simd_width_of[dtype, target=get_gpu_target()]()

    return simd_width_of[dtype]()


@doc_hidden
def get_kernel_tile_shape[dtype: DType, target: StaticString]() -> IndexList[2]:
    """Get the 2D tile shape used by tile-programming-model fusion kernels.

    The tile analog of `get_kernel_simd_width`: returns the `(rows, cols)` of
    the register / on-chip tile a tile-based kernel (e.g. `Add.elementwise` on a
    `TileTensor`) operates on for `target`. It is returned as an `IndexList[2]`
    (the same lightweight, comptime-materializable shape descriptor used for
    tensor shapes elsewhere in this file) so a driver can splat it into e.g.
    `row_major[s[0], s[1]]()`.

    Currently hardcoded for CPU/GPU; support for additional target devices will
    be added in a future update.
    """
    comptime if _is_gpu[target]():
        return IndexList[2](16, 16)

    return IndexList[2](8, 8)


def foreach[
    dtype: DType,
    rank: Int,
    //,
    func: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises:
    """Apply the function `func` to each element of the tensor slice.

    The `func` body receives the element index as a `Coord`. Use
    `coord_to_index_list` to convert it to an `IndexList` if integer index
    arithmetic is needed.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        func: The function to apply to each element of the tensor slice.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target (usually leave this as its default value).
        _trace_name: Name of the executed operation displayed in the trace_description.

    Args:
        tensor: The output tensor slice which receives the return values from `func`.
        ctx: The call context (forward this from the custom operation).
    """

    @always_inline
    def elementwise_fn_wrapper[
        width: Int,
        alignment: Int = 1,
    ](index: Coord) {var}:
        var val = func[width](index)
        tensor._fused_store[element_alignment=alignment](index, val)

    elementwise[
        simd_width,
        target=target,
        _trace_description=_trace_name,
    ](elementwise_fn_wrapper, tensor.shape_coord(), ctx)


def _shape_types_compatible[
    x_types: TypeList[Trait=CoordLike, ...],
    y_types: TypeList[Trait=CoordLike, ...],
    rank: Int,
]() -> Bool:
    comptime for i in range(rank):
        comptime if x_types[i].is_static_value and y_types[i].is_static_value:
            comptime if x_types[i].static_value != y_types[i].static_value:
                return False
    return True
