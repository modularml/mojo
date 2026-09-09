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
"""Provides element-based access to memory using layout-driven vectorization.

This module implements efficient memory access patterns for multi-dimensional data
using the layout system. It provides abstractions for loading and storing data with
specific memory layouts, enabling vectorized operations that respect the underlying
memory organization.

Key components:
- `Element`: A wrapper around SIMD types that provides layout-driven vectorized
  operations
- `MemoryElement`: Represents data in memory organized according to a specific layout

These components enable efficient tensor operations by ensuring memory accesses
follow optimal patterns defined by the layout system.
"""

from std.sys import align_of

from layout.layout import coalesce, is_contiguous_dim

from . import Layout, RuntimeLayout, RuntimeTuple
from .int_tuple import IntTuple, UNKNOWN_VALUE, _get_index_type


@always_inline
def _get_offset[
    i: Int
](runtime_layout: RuntimeLayout) -> Scalar[runtime_layout.linear_idx_type]:
    """Returns the offset for a single index into the runtime layout.

    Parameters:
        i: The index to get the offset for.

    Args:
        runtime_layout: The runtime layout to get the offset from.

    Returns:
        The offset value for the given index.
    """

    comptime if runtime_layout.layout.all_dims_known():
        comptime offset = runtime_layout.layout(i)
        return Scalar[runtime_layout.linear_idx_type](offset)
    else:
        return runtime_layout(i)


@always_inline
def _get_offset[
    i: Int, j: Int
](runtime_layout: RuntimeLayout) -> Scalar[runtime_layout.linear_idx_type]:
    """Returns the offset for a 2D index into the runtime layout.

    Parameters:
        i: The first index to get the offset for.
        j: The second index to get the offset for.

    Args:
        runtime_layout: The runtime layout to get the offset from.

    Returns:
        The offset value for the given indices.
    """

    comptime if runtime_layout.layout.all_dims_known():
        comptime offset = runtime_layout.layout([i, j])
        return Scalar[runtime_layout.linear_idx_type](offset)
    else:
        return runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)](i, j)
        )


struct Element[
    dtype: DType,
    layout: Layout,
    /,
    index_type: DType = _get_index_type(layout),
](Writable):
    """A wrapper around SIMD types that provides layout-driven vectorized operations.

    The `Element` struct extends SIMD types with layout-aware load and store
    operations, enabling efficient vectorized access to multi-dimensional data.
    It maps between logical tensor coordinates and physical memory locations
    according to the specified layout.

    Parameters:
        dtype: The data type of the elements.
        layout: The memory layout describing how elements are organized.
        index_type: The integer type of the index pointing to each element.
    """

    comptime element_data_type = SIMD[Self.dtype, length=Self.layout.size()]
    """The SIMD type used to store and process the element data.

    This type alias defines a SIMD vector with the specified data type and size
    matching the layout's total element count, enabling efficient vectorized operations.
    """

    var element_data: Self.element_data_type
    """The actual SIMD data stored in this element.

    This field contains the vectorized data values that can be processed
    efficiently using SIMD operations.
    """

    var runtime_layout: RuntimeLayout[
        Self.layout,
        element_type=.int32,
        linear_idx_type=Self.index_type,
    ]
    """The runtime layout information for memory access patterns.

    This field stores the layout information needed to map between logical tensor
    coordinates and physical memory locations, supporting both compile-time and
    runtime-determined access patterns.
    """

    def __init__(out self, element_data: Self.element_data_type):
        """Initializes an Element with the given SIMD data.

        Args:
            element_data: The SIMD data to initialize the element with.
        """
        self.element_data = element_data
        self.runtime_layout = {}

    def __init__(
        out self,
        element_data: Self.element_data_type,
        runtime_layout: RuntimeLayout[
            Self.layout,
            element_type=.int32,
            linear_idx_type=Self.index_type,
        ],
    ):
        """Initializes an Element with the given SIMD data and runtime layout.

        Args:
            element_data: The SIMD data to initialize the element with.
            runtime_layout: The runtime layout to use for memory access.
        """
        self.element_data = element_data
        self.runtime_layout = runtime_layout

    @always_inline("nodebug")
    @staticmethod
    def load(
        # Bare `Pointer`, not `ImmPointer`: a `mut=True` source pointer (the
        # usual case, since `MemoryElement` wraps a mutable `LayoutTensor`'s
        # `.ptr`) passed to `ImmPointer[Scalar, ...]` would force a mut->imm
        # conversion whose `...`-elided `address_space` falls back to `.GENERIC`,
        # silently breaking `.LOCAL`/`.SHARED` loads on the GPU.
        ptr: Pointer[Scalar[Self.dtype], ...],
        runtime_layout: RuntimeLayout[
            Self.layout,
            element_type=.int32,
            linear_idx_type=Self.index_type,
        ] = RuntimeLayout[
            Self.layout,
            element_type=.int32,
            linear_idx_type=Self.index_type,
        ](),
    ) -> Self:
        """Loads data from memory according to the specified layout.

        This method loads data from memory using the layout information to determine
        the memory access pattern. It supports both rank-1 and rank-2 layouts with
        various stride patterns, optimizing for contiguous memory access when
        possible.

        Args:
            ptr: Pointer to the memory location to load from.
            runtime_layout: The runtime layout to use for memory access.

        Returns:
            A new `Element` containing the loaded data.
        """
        comptime flat_layout = coalesce(Self.layout)
        comptime assert flat_layout.rank() <= 2, "Only supports rank <= 2"

        var element_data = Self.element_data_type()

        comptime if flat_layout.rank() == 1:
            comptime size = flat_layout.size()

            comptime if is_contiguous_dim(flat_layout, 0):
                comptime alignment = align_of[Self.element_data_type]()
                return Self(
                    ptr.load[
                        width=Self.element_data_type.length, alignment=alignment
                    ]()
                )

            comptime for i in range(size):
                element_data[i] = ptr[_get_offset[i](runtime_layout)]
            return Element(element_data, runtime_layout)

        comptime if is_contiguous_dim(flat_layout, 0):
            comptime size = Int(flat_layout.shape[0])
            comptime elements = Int(flat_layout.shape[1])
            comptime vec_type = SIMD[Self.dtype, size]
            comptime alignment = align_of[vec_type]()

            comptime for i in range(elements):
                var vec_i = ptr.load[width=size, alignment=alignment](
                    _get_offset[0, i](runtime_layout)
                )
                element_data = element_data.insert[offset=i * size](vec_i)
            return Element(element_data, runtime_layout)

        elif is_contiguous_dim(flat_layout, 1):
            comptime size = Int(flat_layout.shape[1])
            comptime elements = Int(flat_layout.shape[0])
            comptime vec_type = SIMD[Self.dtype, size]
            comptime alignment = align_of[vec_type]()

            comptime for i in range(elements):
                var vec_i = ptr.load[width=size, alignment=alignment](
                    _get_offset[i, 0](runtime_layout)
                )
                element_data = element_data.insert[offset=i * size](vec_i)
            return Element(element_data, runtime_layout)

        comptime dim_0 = Int(flat_layout.shape[0])
        comptime dim_1 = Int(flat_layout.shape[1])

        comptime for i in range(dim_0):
            comptime for j in range(dim_1):
                element_data[i + j * dim_0] = ptr[
                    _get_offset[i, j](runtime_layout)
                ]
        return Element(element_data, runtime_layout)

    @always_inline("nodebug")
    @staticmethod
    def masked_load(
        ptr: Pointer[Scalar[Self.dtype], ...],
        runtime_layout: RuntimeLayout[
            Self.layout,
            element_type=.int32,
            linear_idx_type=Self.index_type,
        ] = RuntimeLayout[
            Self.layout,
            element_type=.int32,
            linear_idx_type=Self.index_type,
        ](),
    ) -> Self:
        """Loads data from memory with masking for partial loads.

        This method loads data from memory using the layout information, but also
        handles cases where the runtime dimensions are smaller than the static
        layout dimensions. It ensures that only valid memory locations are accessed.

        Args:
            ptr: Pointer to the memory location to load from.
            runtime_layout: The runtime layout to use for memory access.

        Returns:
            A new `Element` containing the loaded data, with zeros in positions
            beyond the runtime dimensions.
        """
        # TODO: Use partial_simd_load after closing KERN-729.
        comptime assert Self.layout.rank() <= 2, "Only supports rank <= 2"
        var element_data = Self.element_data_type()

        comptime if Self.layout.rank() == 1:
            comptime size = Self.layout.size()

            comptime if Self.layout.stride[0] == 1:
                comptime alignment = align_of[Self.element_data_type]()
                if runtime_layout.dim(0) < size:
                    comptime for i in range(size):
                        if i >= runtime_layout.dim(0):
                            break
                        element_data[i] = ptr[_get_offset[i](runtime_layout)]
                    return Element(element_data, runtime_layout)

                return Self(
                    ptr.load[
                        width=Self.element_data_type.length, alignment=alignment
                    ](0)
                )

            comptime for i in range(size):
                if i >= runtime_layout.dim(0):
                    break
                element_data[i] = ptr[_get_offset[i](runtime_layout)]
            return Element(element_data, runtime_layout)

        # rank-2 element.
        comptime if Self.layout.stride[0] == 1:
            comptime size = Int(Self.layout.shape[0])
            comptime elements = Int(Self.layout.shape[1])
            var element_data = Self.element_data_type()
            if runtime_layout.dim(0) < size:
                comptime dim_0 = Int(Self.layout.shape[0])
                comptime dim_1 = Int(Self.layout.shape[1])

                comptime for i in range(dim_0):
                    if i >= runtime_layout.dim(0):
                        break

                    comptime for j in range(dim_1):
                        if j >= runtime_layout.dim(1):
                            break
                        element_data[i + j * dim_0] = ptr[
                            _get_offset[i, j](runtime_layout)
                        ]
                return Element(element_data, runtime_layout)

            comptime for i in range(elements):
                if i >= runtime_layout.dim(0):
                    break
                var vec_i = ptr.load[width=size](
                    _get_offset[0, i](runtime_layout)
                )
                element_data = element_data.insert[offset=i * size](vec_i)
            return Element(element_data, runtime_layout)

        elif Self.layout.stride[1] == 1:
            comptime size = Int(Self.layout.shape[1])
            comptime elements = Int(Self.layout.shape[0])
            var element_data = Self.element_data_type()
            if runtime_layout.dim(1) < size:
                comptime dim_0 = Int(Self.layout.shape[0])
                comptime dim_1 = Int(Self.layout.shape[1])

                comptime for i in range(dim_0):
                    if i >= runtime_layout.dim(0):
                        break

                    comptime for j in range(dim_1):
                        if j >= runtime_layout.dim(1):
                            break
                        element_data[i + j * dim_0] = ptr[
                            _get_offset[i, j](runtime_layout)
                        ]
                return Element(element_data, runtime_layout)

            comptime for i in range(elements):
                if i >= runtime_layout.dim(0):
                    break
                var vec_i = ptr.load[width=size](
                    _get_offset[i, 0](runtime_layout)
                )
                element_data = element_data.insert[offset=i * size](vec_i)
            return Element(element_data, runtime_layout)

        comptime dim_0 = Int(Self.layout.shape[0])
        comptime dim_1 = Int(Self.layout.shape[1])

        comptime for i in range(dim_0):
            if i >= runtime_layout.dim(0):
                break

            comptime for j in range(dim_1):
                if j >= runtime_layout.dim(1):
                    break
                element_data[i + j * dim_0] = ptr[
                    _get_offset[i, j](runtime_layout)
                ]
        return Element(element_data, runtime_layout)

    @always_inline("nodebug")
    def store(self, ptr: MutPointer[Scalar[Self.dtype], ...]):
        """Stores element data to memory according to the specified layout.

        This method performs a layout-aware store operation, writing data to memory
        following the access patterns defined by the layout. It optimizes memory
        writes based on the layout's stride patterns to maximize performance.

        The method handles different memory layout patterns:
        - For rank-1 tensors with contiguous memory (stride=1), it uses vectorized stores
        - For rank-2 tensors with contiguous rows or columns, it uses optimized slice-based stores
        - For non-contiguous memory layouts, it performs element-by-element stores

        Unlike `masked_store()`, this method assumes the full static dimensions will be written
        and does not perform runtime dimension boundary checking.

        Args:
            ptr: Mutable pointer to the memory location where data will be stored.

        Note:
            This method is constrained to layouts with rank <= 2. For higher-rank
            tensors, consider decomposing the operation.
        """
        comptime assert Self.layout.rank() <= 2, "Only supports rank <= 2"

        comptime if Self.layout.rank() == 1:
            comptime size = Self.layout.size()

            comptime if Self.layout.stride[0] == 1:
                comptime alignment = align_of[Self.element_data_type]()
                ptr.store[alignment=alignment](self.element_data)
                return

            comptime for i in range(size):
                ptr[_get_offset[i](self.runtime_layout)] = self.element_data[i]
            return

        comptime if Self.layout.stride[0] == 1:
            comptime size = Int(Self.layout.shape[0])
            comptime elements = Int(Self.layout.shape[1])
            comptime vec_type = SIMD[Self.dtype, size]
            comptime alignment = align_of[vec_type]()

            comptime for i in range(elements):
                ptr.store[alignment=alignment](
                    _get_offset[0, i](self.runtime_layout),
                    self.element_data.slice[size, offset=i * size](),
                )
            return

        elif Self.layout.stride[1] == 1:
            comptime size = Int(Self.layout.shape[1])
            comptime elements = Int(Self.layout.shape[0])
            comptime vec_type = SIMD[Self.dtype, size]
            comptime alignment = align_of[vec_type]()

            comptime for i in range(elements):
                ptr.store[alignment=alignment](
                    _get_offset[i, 0](self.runtime_layout),
                    self.element_data.slice[size, offset=i * size](),
                )
            return

        comptime dim_0 = Int(Self.layout.shape[0])
        comptime dim_1 = Int(Self.layout.shape[1])

        comptime for i in range(dim_0):
            comptime for j in range(dim_1):
                (ptr + _get_offset[i, j](self.runtime_layout)).store(
                    self.element_data[i + j * dim_0]
                )

    @always_inline("nodebug")
    def masked_store(self, ptr: MutPointer[Scalar[Self.dtype], ...]):
        """Stores element data to memory with masking for partial stores.

        This method performs a layout-aware store operation with boundary checking.
        It ensures that only valid memory locations are written to when the runtime
        dimensions are smaller than the static layout dimensions, preventing out-of-bounds
        memory access.

        The method optimizes for different memory layouts:
        - For contiguous memory (stride=1), it uses vectorized stores when possible
        - For non-contiguous memory, it performs element-by-element stores
        - For all patterns, it respects runtime dimension bounds

        Args:
            ptr: Pointer to the memory location where data will be stored.

        Note:
            This method is constrained to layouts with rank <= 2. For higher-rank
            tensors, consider decomposing the operation.
        """
        comptime assert Self.layout.rank() <= 2, "Only supports rank <= 2"

        comptime if Self.layout.rank() == 1:
            comptime size = Self.layout.size()

            comptime if Self.layout.stride[0] == 1:
                if self.runtime_layout.dim(0) < size:
                    comptime for i in range(size):
                        if i >= self.runtime_layout.dim(0):
                            break
                        ptr[
                            _get_offset[i](self.runtime_layout)
                        ] = self.element_data[i]
                    return

                comptime alignment = align_of[Self.element_data_type]()
                ptr.store(self.element_data)
                return

            comptime for i in range(size):
                if i >= self.runtime_layout.dim(0):
                    break
                ptr[_get_offset[i](self.runtime_layout)] = self.element_data[i]
            return

        comptime if Self.layout.stride[0] == 1:
            comptime size = Int(Self.layout.shape[0])
            comptime elements = Int(Self.layout.shape[1])
            comptime vec_type = SIMD[Self.dtype, size]
            comptime alignment = align_of[vec_type]()
            if self.runtime_layout.dim(1) < size:
                comptime dim_0 = Int(Self.layout.shape[0])
                comptime dim_1 = Int(Self.layout.shape[1])

                comptime for i in range(dim_0):
                    if i >= self.runtime_layout.dim(0):
                        break

                    comptime for j in range(dim_1):
                        if j >= self.runtime_layout.dim(1):
                            break
                        (ptr + _get_offset[i, j](self.runtime_layout)).store(
                            self.element_data[i + j * dim_0],
                        )
                return

            comptime for i in range(elements):
                if i >= self.runtime_layout.dim(0):
                    break
                (ptr + _get_offset[i, 0](self.runtime_layout)).store[
                    alignment=alignment
                ](
                    self.element_data.slice[size, offset=i * size](),
                )
            return

        elif Self.layout.stride[1] == 1:
            comptime size = Int(Self.layout.shape[1])
            comptime elements = Int(Self.layout.shape[0])
            comptime vec_type = SIMD[Self.dtype, size]
            comptime alignment = align_of[vec_type]()
            if self.runtime_layout.dim(1) < size:
                comptime dim_0 = Int(Self.layout.shape[0])
                comptime dim_1 = Int(Self.layout.shape[1])

                comptime for i in range(dim_0):
                    if i >= self.runtime_layout.dim(0):
                        break

                    comptime for j in range(dim_1):
                        if j >= self.runtime_layout.dim(1):
                            break
                        (ptr + _get_offset[i, j](self.runtime_layout)).store(
                            self.element_data[i + j * dim_0],
                        )
                return

            comptime for i in range(elements):
                if i >= self.runtime_layout.dim(0):
                    break
                (ptr + _get_offset[i, 0](self.runtime_layout)).store[
                    alignment=alignment
                ](
                    self.element_data.slice[size, offset=i * size](),
                )
            return

        comptime dim_0 = Int(Self.layout.shape[0])
        comptime dim_1 = Int(Self.layout.shape[1])

        comptime for i in range(dim_0):
            if i >= self.runtime_layout.dim(0):
                break

            comptime for j in range(dim_1):
                if j >= self.runtime_layout.dim(1):
                    break
                (ptr + _get_offset[i, j](self.runtime_layout)).store(
                    self.element_data[i + j * dim_0]
                )

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the element to the specified writer.

        Args:
            writer: The writer to output the element representation to.
        """
        writer.write(self.element_data)


struct MemoryElement[
    mut: Bool,
    //,
    dtype: DType,
    layout: Layout,
    origin: Origin[mut=mut],
    /,
    address_space: AddressSpace,
    *,
    index_type: DType = _get_index_type(
        address_space=address_space, layout=layout
    ),
]:
    """Represents data in memory organized according to a specific layout.

    The `MemoryElement` struct provides a high-level interface for accessing data
    in memory with a specific layout. It encapsulates a pointer to the memory
    location and the runtime layout information needed to access the data correctly.

    This abstraction enables efficient memory operations that respect the underlying
    memory organization, supporting vectorized loads and stores while handling
    different memory layouts transparently.

    Parameters:
        mut: Whether the memory element is mutable.
        dtype: The data type of the elements.
        layout: The memory layout describing how elements are organized.
        origin: The origin of the memory element.
        address_space: The memory address space where the data is located.
        index_type: The integer type of the index pointing to each memory element.
    """

    comptime _AsMut[
        mut_origin: MutOrigin,
    ] = MemoryElement[
        Self.dtype,
        Self.layout,
        mut_origin,
        Self.address_space,
        index_type=Self.index_type,
    ]

    var ptr: Pointer[
        Scalar[Self.dtype], Self.origin, address_space=Self.address_space
    ]
    """Pointer to the memory location where the data is stored.

    This pointer provides access to the underlying memory with the specified
    address space and alignment requirements. It points to the first element
    of the data structure in memory.
    """

    var runtime_layout: RuntimeLayout[
        Self.layout,
        element_type=.int32,
        linear_idx_type=Self.index_type,
    ]
    """Runtime layout information used for memory access calculations.

    This field stores the runtime layout information needed to compute memory
    offsets for accessing elements according to the specified layout pattern.
    It handles both compile-time known dimensions and runtime-determined dimensions.
    """

    def __init__(
        out self,
        ptr: Pointer[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ],
        runtime_layout: RuntimeLayout[
            Self.layout,
            element_type=.int32,
            linear_idx_type=Self.index_type,
        ],
    ):
        """Initializes a `MemoryElement` with the given pointer and runtime layout.

        Args:
            ptr: Pointer to the memory location of the element.
            runtime_layout: The runtime layout to use for memory access.
        """
        self.ptr = ptr
        self.runtime_layout = runtime_layout

    @always_inline("nodebug")
    def load(
        self,
        out result: Element[
            Self.dtype, Self.layout, index_type=Self.index_type
        ],
    ):
        """Loads data from memory according to the specified layout.

        This method performs a layout-aware load operation, reading data from memory
        following the access patterns defined by the layout. It optimizes memory
        reads based on the layout's stride patterns to maximize performance.

        The method leverages the underlying `Element.load` implementation which handles
        different memory layout patterns including contiguous and strided access.

        Returns:
            An `Element` containing the loaded data organized according to the layout.
        """
        return type_of(result).load(self.ptr, self.runtime_layout)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def store(
        self: Self._AsMut,
        src: Element[Self.dtype, Self.layout, ...],
    ):
        """Stores element data to the memory location of this MemoryElement.

        This method performs a layout-aware store operation, writing data to memory
        following the access patterns defined by the layout. It optimizes memory
        writes based on the layout's stride patterns to maximize performance.

        The method delegates to the `Element.store` implementation which handles
        different memory layout patterns including vectorized stores for contiguous memory
        and element-by-element stores for non-contiguous layouts.

        Args:
            src: The `Element` containing the data to store.
        """
        return src.store(self.ptr)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def transfer(self: Self._AsMut, src: MemoryElement):
        """Transfers data from another `MemoryElement` to this one.

        This method efficiently transfers data between memory locations with potentially
        different layouts and data types. It performs the following operations:
        1. Loads data from the source `MemoryElement` using its layout
        2. Converts the data to the destination data type if necessary
        3. Stores the converted data to the destination memory location using its layout

        This provides a high-performance way to copy and convert data between different
        memory representations while respecting both source and destination memory layouts.

        Args:
            src: The source `MemoryElement` to transfer data from.
        """
        # Load source element and convert to destination dtype if needed
        var src_element = src.load()
        var converted_element = Element[
            Self.dtype, src.layout, index_type=src.index_type
        ](
            src_element.element_data.cast[Self.dtype](),
            src_element.runtime_layout,
        )
        self.store(
            rebind[
                Element[
                    Self.dtype, Self.layout, index_type=src_element.index_type
                ]
            ](converted_element)
        )
