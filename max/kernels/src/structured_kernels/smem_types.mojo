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
"""Shared memory type aliases for LayoutTensor-based GPU kernels.

This module defines the core SMEM type aliases used across SM90, SM100, and
other GPU kernel implementations. They depend only on `layout` and stdlib,
making them safe to import without pulling in higher-level kernel packages.

Types:
- SMemTile: Shared memory tile (LayoutTensor alias)
- RegTile: Register tile (LayoutTensor alias)
- SMemBarrier: Pointer to shared memory barrier
- PipelineBarrier: Array of pipeline barriers
- SMemTileIter: Iterator over shared memory tiles
- SMemTileArray: Array of shared memory tiles
- SMemArray: Generic shared memory array
- SMemPtr: Typed pointer into shared memory
- eval: Compile-time expression evaluator
"""

from std.sys import align_of, size_of

from layout import Layout, LayoutTensor, lt_to_tt
from layout.int_tuple import _get_index_type, _get_layout_type
from layout.layout_tensor import LayoutTensorIter
from layout.tma_async import SharedMemBarrier
from std.memory import unsafe_stack_allocation


comptime SMemTile[
    _dtype: DType,
    layout: Layout,
    /,
    *,
    element_layout: Layout = Layout(1, 1),
    layout_int_type: DType = _get_layout_type(layout, .SHARED),
    linear_idx_type: DType = _get_index_type(layout, .SHARED),
    masked: Bool = False,
    alignment: Int = align_of[_dtype](),
] = LayoutTensor[
    _dtype,
    layout,
    MutAnyOrigin,
    address_space=.SHARED,
    element_layout=element_layout,
    layout_int_type=layout_int_type,
    linear_idx_type=linear_idx_type,
    masked=masked,
    alignment=alignment,
]
"""Type alias for shared memory tile tensors."""

comptime RegTile[
    _dtype: DType,
    layout: Layout,
    /,
    *,
    element_layout: Layout = Layout(1, 1),
    layout_int_type: DType = _get_layout_type(layout, .LOCAL),
    linear_idx_type: DType = _get_index_type(layout, .LOCAL),
    masked: Bool = False,
    alignment: Int = align_of[_dtype](),
] = LayoutTensor[
    _dtype,
    layout,
    MutAnyOrigin,
    address_space=.LOCAL,
    element_layout=element_layout,
    layout_int_type=layout_int_type,
    linear_idx_type=linear_idx_type,
    masked=masked,
    alignment=alignment,
]
"""Type alias for register (local memory) tile tensors."""


@always_inline
def reg_tile_to_tile_tensor[
    dtype: DType,
    layout: Layout,
](tile: RegTile[dtype, layout, ...]) -> type_of(lt_to_tt(tile)):
    """Return a TileTensor view of a register tile.

    Kept here so kernels that have moved to TileTensor do not need to
    reference the legacy LayoutTensor conversion helper directly.
    """
    return lt_to_tt(tile)


comptime SMemBarrier = UnsafePointer[
    mut=True, SharedMemBarrier, _, address_space=.SHARED
]
"""Type alias for shared memory barrier pointer."""

comptime PipelineBarrier[num_pipeline_stages: Int] = SMemArray[
    SharedMemBarrier, num_pipeline_stages
]
"""Type alias for shared memory pipeline barrier array."""

comptime SMemTileIter[
    dtype: DType,
    layout: Layout,
] = LayoutTensorIter[
    dtype,
    layout,
    MutAnyOrigin,
    address_space=.SHARED,
    alignment=128,
]


# TODO: This type should correctly propagate mutability.
struct SMemTileArray[
    dtype: DType,
    layout: Layout,
    num_tiles: Int,
    alignment: Int,
](TrivialRegisterPassable):
    """Array of tiles in shared memory.

    Parameters:
        dtype: Tile data type.
        layout: Tile layout configuration.
        num_tiles: Number of tiles.
        alignment: Memory alignment.
    """

    comptime Tile = SMemTile[
        Self.dtype,
        Self.layout,
        alignment=Self.alignment,
    ]

    comptime num_elements = Self.layout.size() * Self.num_tiles

    comptime storage_size = Self.num_elements * size_of[Self.dtype]()

    comptime Storage = Array[Scalar[Self.dtype], Self.num_elements]

    var ptr: UnsafePointer[
        Scalar[Self.dtype], MutUntrackedOrigin, address_space=.SHARED
    ]

    def __init__(
        ref[AddressSpace.SHARED] storage: Self.Storage,
    ) -> Self:
        """Initialize with Storage.

        Args:
            storage: Storage.
        """
        return Self(storage.unsafe_ptr())

    def __init__(
        out self,
        # TODO: This should correctly propagate mutability.
        unsafe_ptr: UnsafePointer[Scalar[Self.dtype], _, address_space=.SHARED],
    ):
        """Initialize with shared memory pointer.

        Args:
            unsafe_ptr: Shared memory pointer.
        """
        comptime assert (
            Self.layout.all_dims_known()
        ), "Layout must be known at compile time."

        self.ptr = rebind[type_of(self.ptr)](unsafe_ptr)

    @always_inline
    def __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at index.

        Args:
            index: Tile index.

        Returns:
            Tile at index.
        """
        return Self.Tile(
            (
                self.ptr + eval[Self.layout.size()] * Int(index)
            ).as_unsafe_any_origin()
        )

    def slice[
        length: Int
    ](
        self,
        start: Int,
        out result: SMemTileArray[
            Self.dtype, Self.layout, length, Self.alignment
        ],
    ):
        return type_of(result)(self.ptr + eval[Self.layout.size()] * start)

    @always_inline
    @staticmethod
    def stack_allocation() -> Self:
        var ptr = unsafe_stack_allocation[
            Self.storage_size,
            Self.dtype,
            alignment=Self.alignment,
            address_space=.SHARED,
        ]()
        return Self(ptr)


struct SMemArray[type: TrivialRegisterPassable, size: Int](
    TrivialRegisterPassable
):
    """Shared memory array of fixed size.

    Parameters:
        type: Element type.
        size: Number of elements.
    """

    comptime ptr_type = UnsafePointer[
        Self.type, MutUntrackedOrigin, address_space=.SHARED
    ]
    comptime storage_size = Self.size * size_of[Self.type]()
    comptime Storage = Array[Self.type, Self.size]

    var ptr: Self.ptr_type

    @always_inline
    def __init__(
        out self,
        unsafe_ptr: Self.ptr_type,
    ):
        """Initialize with shared memory pointer.

        Args:
            unsafe_ptr: Shared memory pointer.
        """
        self.ptr = unsafe_ptr

    def __init__(ref[AddressSpace.SHARED] storage: Self.Storage) -> Self:
        """Initialize from Storage."""
        return Self(rebind[Self.ptr_type](storage.unsafe_ptr()))

    @always_inline
    def __getitem__[T: Intable](self, index: T) -> Self.ptr_type:
        """Get a pointer to the element at index.

        Args:
            index: Element index.

        Returns:
            Pointer to element.
        """
        return self.ptr + Int(index)

    @always_inline
    @staticmethod
    def len() -> Int:
        """Get array length in bytes.

        Returns:
            Total size in bytes.
        """
        return Self.size * size_of[Self.type]()

    @always_inline
    @staticmethod
    def stack_allocation[alignment: Int = align_of[Self.type]()]() -> Self:
        var ptr = unsafe_stack_allocation[
            Self.len(),
            Self.type,
            alignment=alignment,
            address_space=.SHARED,
        ]()
        return Self(ptr)


comptime eval[T: AnyType, //, val: T] = val
"""Helper alias to force evaluation of expressions at compile time."""

comptime SMemPtr[type: AnyType] = UnsafePointer[
    type, MutUntrackedOrigin, address_space=.SHARED
]
