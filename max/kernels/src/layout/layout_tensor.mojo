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
"""Provides the `LayoutTensor` type for representing multidimensional data.
"""
from std.math import align_up, ceildiv, exp
from std.math.math import _Expable
from std.math.uutils import umod, ufloordiv
from std.sys import (
    align_of,
    is_amd_gpu,
    is_nvidia_gpu,
    prefetch,
    simd_width_of,
    size_of,
)
from std.memory.unsafe_pointer import unsafe_cast
from std.sys.intrinsics import PrefetchOptions, readfirstlane

import max.gpu.memory as gpu_memory
from std.algorithm import vectorize
from std.bit import log2_floor
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.dtype import _unsigned_integral_type_of
from max.gpu.host import DeviceBuffer, HostBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import (
    block_dim,
    block_idx,
    lane_id,
    thread_idx,
)
from max.gpu.intrinsics import AMDBufferResource
from max.gpu.memory import CacheEviction, CacheOperation, Fill, async_copy
from layout._fillers import BATCH_SIZE
from layout._utils import make_amd_buffer_resource
from layout.element import Element, MemoryElement
from layout.tma_async import _tma_desc_tile_shape
from std.memory import unsafe_stack_allocation
from std.utils import IndexList, StaticTuple
from std.utils.index import Index
from .int_tuple import (
    _get_index_type,
    _get_layout_type,
    _get_unsigned_type,
    congruent,
    depth,
    fill_like,
    flatten,
    product,
    propagate_unknown,
    to_nest,
)
from .layout import *
from .runtime_layout import RuntimeLayout
from .runtime_layout import make_layout as make_runtime_layout
from .runtime_tuple import RuntimeTuple
from .swizzle import Swizzle, make_ldmatrix_swizzle

from std.builtin.debug_assert import ASSERT_MODE


def _compute_distribute_layout[
    data_layout: Layout,
    threads_layout: Layout,
    axis: Optional[Int] = None,
]() -> Layout:
    """Computes a layout for distributing threads across data.

    Distributes thread_layout into data_layout. If axis is provided, distributes
    into threads_layout projected into this axis.

    Parameters:
        data_layout: The layout of the data to be distributed.
        threads_layout: The layout of the threads.
        axis: Optional axis for projection-based distribution.

    Returns:
        A layout representing the distribution of threads across data.
    """
    var thread_tile = LayoutList()

    comptime if axis:
        return zipped_divide(
            materialize[data_layout](),
            Layout(threads_layout.shape[axis.value()]),
        )

    else:
        for dim in threads_layout.shape:
            thread_tile.append(Layout(dim))

        return zipped_divide(materialize[data_layout](), thread_tile)


def _project_on_axis[
    axis: Int, submode_axis: Optional[Int] = None
](t: IntTuple) -> IntTuple:
    """Projects an IntTuple onto a specific axis.

    Creates an IntTuple with zeros in all positions except the specified axis,
    which contains ones. When submode_axis is provided, the projection happens
    only on that specific submode.

    Parameters:
        axis: The axis to project onto.
        submode_axis: Optional submode axis for nested projection.

    Args:
        t: The input IntTuple to project.

    Returns:
        A projected IntTuple.
    """
    if not submode_axis:
        var p_t = fill_like(t, 0)
        # p_t[axis] = fill_like(t[axis], 1)
        p_t = p_t.replace_entry(axis, fill_like(t[axis], 1))
        return p_t
    var p_t = fill_like(t, 1)
    # p_t[axis] = fill_like(t[axis], 0)
    # p_t[axis][submode_axis.value()] = 1
    var filled = fill_like(t[axis], 0)
    filled = filled.replace_entry(submode_axis.value(), 1)
    p_t = p_t.replace_entry(axis, filled)
    return p_t


comptime _swizzle_signature = def[dtype: DType](Scalar[dtype]) thin -> Scalar[
    dtype
]


def _get_slice_size(layout: Layout, slc: Slice, dim: Int) -> Int:
    """Calculates the size of a slice in a specific layout dimension.

    Computes the number of elements in a slice for a given dimension of the
    layout. This function handles the conversion between slice notation and
    actual element counts.

    Args:
        layout: The layout containing the dimension information.
        slc: The slice specification (start:end:step).
        dim: The dimension index to slice.

    Returns:
        The number of elements in the slice for the specified dimension.
    """
    var start, end, _ = slc.indices(Int(layout.shape[dim]))
    return end - start


def _not_in_tuple[n: Int, size: Int, tuple: IndexList[size]]() -> Bool:
    """Checks if a value is *not* present in an `IndexList`.

    This utility function searches through an `IndexList` to determine if a
    specific value is absent. Used for dimension validation and filtering
    operations.

    Parameters:
        n: The value to check for in the `IndexList`.
        size: The size of the `IndexList`.
        tuple: The `IndexList` to search in.

    Returns:
        True if the value is not found in the `IndexList`, False if it is
        present.
    """

    comptime for i in range(size):
        comptime if tuple[i] == n:
            return False
    return True


def _tile_is_masked[layout: Layout, *tile_sizes: Int]() -> Bool:
    """Determines if a tiled layout requires masked access.

    When tiling a tensor, this function checks if any dimension of the layout is
    not evenly divisible by its corresponding tile size. If any dimension
    requires padding, masked access is needed to prevent out-of-bounds memory
    accesses.

    Parameters:
        layout: The layout to check for divisibility.
        tile_sizes: The tile sizes for each dimension of the layout.

    Returns:
        True if masked access is required (any dimension not evenly divisible),
        False if all dimensions are perfectly divisible by their tile sizes.
    """

    comptime if not layout.all_dims_known():
        return True

    comptime for axis in range(layout.rank()):
        comptime dim = product(layout.shape[axis])

        comptime if dim % tile_sizes[axis] != 0:
            return True
    return False


def _distribute_is_masked[
    layout: Layout, threads_layout: Layout, axis: Optional[Int] = None
]() -> Bool:
    """Determines if a distributed layout requires masked access.

    When distributing computation across threads, this function checks if the
    layout's dimensions are evenly divisible by the corresponding thread
    dimensions. Masked access is required when dimensions don't divide evenly to
    prevent out-of-bounds accesses.

    Parameters:
        layout: The layout to distribute across threads.
        threads_layout: The layout representing thread organization.
        axis: Optional axis for projection-based distribution. When specified,
              distribution occurs along this axis only.

    Returns:
        True if masked access is required (dimensions not evenly divisible),
        False if all dimensions are perfectly divisible by thread dimensions.
    """

    # TODO: relax this constraint
    comptime if depth(threads_layout.shape) > 1:
        return False

    comptime if axis:
        return False

    comptime if not layout.all_dims_known():
        return True

    comptime for i in range(layout.rank()):
        comptime layout_dim = product(layout.shape[i])
        comptime thread_dim = product(threads_layout.shape[i])

        comptime if layout_dim % thread_dim != 0:
            return True

    return False


struct LayoutTensor[
    mut: Bool,
    //,
    dtype: DType,
    layout: Layout,
    origin: Origin[mut=mut],
    /,
    *,
    address_space: AddressSpace = .GENERIC,
    element_layout: Layout = Layout(1, 1),
    layout_int_type: DType = _get_layout_type(layout, address_space),
    linear_idx_type: DType = _get_index_type(layout, address_space),
    masked: Bool = False,
    alignment: Int = align_of[dtype](),
](
    DevicePassable,
    TrivialRegisterPassable,
    Writable,
    _Expable,
):
    """A high-performance tensor with explicit memory layout and
    hardware-optimized access patterns.

    `LayoutTensor` provides a powerful abstraction for multi-dimensional data
    with precise control over memory organization. It supports various memory
    layouts (row-major, column-major, tiled), hardware-specific optimizations,
    and efficient parallel access patterns.

    Parameters:
        mut: The inferred mutability of the underlying pointer.
        dtype: The data type of the underlying pointer.
        layout: The memory layout of the tensor.
        origin: The origin of the underlying pointer.
        address_space: The address space of the underlying pointer.
        element_layout: The memory layout of each element in the tensor.
        layout_int_type: The integer type of each dimension of runtime layout.
        linear_idx_type: The integer type of the index pointing to memory
            locations.
        masked: If true the tensor is masked and runtime layouts determine the
            shape.
        alignment: Alignment of the data pointer.

    Example:

    ```mojo
    from layout import Layout, LayoutTensor

    # Create tensor on CPU using Array to allocate storage space.
    var storage = Array[Float32, 5 * 4](uninitialized=True)
    var tensor_5x4 = LayoutTensor[.float32, Layout.row_major(5, 4)](storage)
    ```
    """

    # `trait DevicePassable` implementation, to allow LayoutTensor to be passed directly to kernels
    comptime device_type: AnyType = Self
    """The device-side type representation."""

    @staticmethod
    def _is_convertible_to_device_type[T: AnyType]() -> Bool:
        comptime if Self.mut:
            return TypeList.of[
                Self,
                Self.OriginCastType[MutAnyOrigin],
                Self.OriginCastType[MutUntrackedOrigin],
                Self.OriginCastType[ImmutAnyOrigin],
                Self.OriginCastType[ImmUntrackedOrigin],
            ]().contains[T]()
        else:
            return TypeList.of[
                Self,
                Self.OriginCastType[ImmutAnyOrigin],
                Self.OriginCastType[ImmUntrackedOrigin],
            ]().contains[T]()

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """
        Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        return String(
            "LayoutTensor[mut = ",
            Self.mut,
            ", dtype = ",
            Self.dtype,
            ", layout = ",
            materialize[Self.layout](),
            ", address_space = ",
            Self.address_space,
            ", element_layout = ",
            materialize[Self.element_layout](),
            ", layout_int_type = ",
            Self.layout_int_type,
            ", linear_idx_type = ",
            Self.linear_idx_type,
            ", masked = ",
            Self.masked,
            ", alignment = ",
            Self.alignment,
            "]",
        )

    comptime rank = Self.layout.rank()
    """The number of dimensions in the tensor's layout."""

    var ptr: Pointer[
        Scalar[Self.dtype], address_space=Self.address_space, origin=Self.origin
    ]
    """Pointer to the underlying memory buffer containing the tensor data.

    This pointer respects the specified address space, alignment, mutability,
    and origin tracking for memory safety and performance optimization."""

    comptime storage_size = size_of[Self.dtype]() * Self.layout.size()
    """Total storage size in bytes for the tensor data."""

    comptime RuntimeLayoutType = RuntimeLayout[
        Self.layout,
        element_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for the runtime layout."""

    var runtime_layout: Self.RuntimeLayoutType
    """Runtime representation of the tensor's memory layout.

    Handles both compile-time and runtime-determined dimensions, enabling
    efficient mapping between logical tensor coordinates and physical memory
    locations."""

    comptime RuntimeElementLayoutType = RuntimeLayout[
        Self.element_layout,
        element_type=.int32,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for the runtime element layout."""

    var runtime_element_layout: Self.RuntimeElementLayoutType
    """Runtime representation of each element's internal layout.

    Used when elements themselves have structure, such as in blocked or tiled
    layouts."""

    comptime element_size = Self.element_layout.size()
    """The number of scalar values in each element of the tensor."""

    comptime element_type = SIMD[Self.dtype, Self.element_size]
    """The SIMD vector type used for vectorized operations on tensor elements."""

    comptime num_strides: Int = Self.RuntimeLayoutType.StrideType.scalar_length
    """Number of stride values in the layout."""
    comptime idx_list_t[rank: Int = Self.rank] = IndexList[
        rank, element_type=Self.linear_idx_type
    ]
    """Type alias for index lists of the tensor's rank.

    Parameters:
        rank: The number of dimensions in the index list.
    """

    comptime GenericAddressSpaceLayoutTensor = LayoutTensor[
        mut=Self.mut,
        Self.dtype,
        Self.layout,
        Self.origin,
        address_space=.GENERIC,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
        alignment=Self.alignment,
    ]
    """LayoutTensor variant using generic address space."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    def __init__(
        out self: Self.GenericAddressSpaceLayoutTensor,
        span: Span[Scalar[Self.dtype], Self.origin],
    ):
        """Create a `LayoutTensor` with a `Span`.

        Constraints:
            Layout must be fully static.

        Args:
            span: The `Span` pointing to the underlying data.
        """
        self = Self.GenericAddressSpaceLayoutTensor(span.unsafe_ptr())

    @always_inline
    def __init__(
        out self: Self.GenericAddressSpaceLayoutTensor,
        span: Span[Scalar[Self.dtype], Self.origin],
        runtime_layout: RuntimeLayout[Self.layout, ...],
    ):
        """Create a `LayoutTensor` with a `Span` and a runtime layout
        for the tensor. The runtime layout element type will be casted to the
        layout tensor layout integer type.

        Constraints:
            - Element layout must be fully static.

        Args:
            span: The `Span` pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
        """
        self = Self.GenericAddressSpaceLayoutTensor(
            span.unsafe_ptr(), runtime_layout
        )

    @always_inline
    def __init__(
        out self: Self.GenericAddressSpaceLayoutTensor,
        span: Span[Scalar[Self.dtype], Self.origin],
        runtime_layout: RuntimeLayout[Self.layout, ...],
        element_runtime_layout: RuntimeLayout[Self.element_layout, ...],
    ):
        """Create a `LayoutTensor` with a `Span`, a runtime layout of
        the tensor, and the runtime layout of each element. The runtime layout
        element type will be casted to the layout tensor layout integer type.

        Constraints:
            - Runtime layout and `LayoutTensor` must have the same bitwidth and
                index type.

        Args:
            span: The `Span` pointing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
            element_runtime_layout: The runtime layout of each element.
        """
        self = Self.GenericAddressSpaceLayoutTensor(
            span.unsafe_ptr(), runtime_layout, element_runtime_layout
        )

    @always_inline
    @doc_hidden
    def __init__(
        out self,
        unsafe_ptr: OptionalPointer[
            Scalar[Self.dtype],
            Self.origin,
            address_space=Self.address_space,
        ],
    ):
        self = Self(unsafe_ptr._unsafe_nullable())

    @always_inline
    def __init__(
        out self,
        unsafe_ptr: Pointer[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ],
    ):
        """Create a `LayoutTensor` with a `Pointer`.

        Constraints:
            Layout must be fully static.

        Args:
            unsafe_ptr: The `Pointer` pointing to the underlying data.
        """

        comptime assert (
            Self.layout.all_dims_known()
        ), "Layout must be fully static"

        comptime assert (
            Self.layout_int_type.is_signed()
            and Self.linear_idx_type.is_signed()
        ), "Layout integer type and linear index type must be signed."

        self.ptr = unsafe_ptr
        self.runtime_layout = {}
        self.runtime_element_layout = {}

    @always_inline
    @doc_hidden
    def __init__(
        out self,
        unsafe_ptr: OptionalPointer[
            Scalar[Self.dtype],
            Self.origin,
            address_space=Self.address_space,
        ],
        runtime_layout: RuntimeLayout[Self.layout, ...],
    ):
        self = Self(unsafe_ptr._unsafe_nullable(), runtime_layout)

    @always_inline
    def __init__(
        out self,
        unsafe_ptr: Pointer[
            Scalar[Self.dtype], Self.origin, address_space=Self.address_space
        ],
        runtime_layout: RuntimeLayout[Self.layout, ...],
    ):
        """Create a `LayoutTensor` with a `Pointer` and a runtime layout
        for the tensor. The runtime layout element type will be casted to the
        layout tensor layout integer type.

        Constraints:
            Element layout must be fully static.

        Args:
            unsafe_ptr: The `Pointer` pointing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
        """

        comptime assert (
            Self.element_layout.all_dims_known()
        ), "Layout must be fully static"

        self.ptr = unsafe_ptr
        self.runtime_layout = runtime_layout.cast[
            Self.layout_int_type, target_linear_idx_type=Self.linear_idx_type
        ]()
        self.runtime_element_layout = {}

    @always_inline
    @doc_hidden
    def __init__(
        out self,
        unsafe_ptr: OptionalPointer[
            Scalar[Self.dtype],
            Self.origin,
            address_space=Self.address_space,
        ],
        runtime_layout: RuntimeLayout[Self.layout, ...],
        element_runtime_layout: RuntimeLayout[Self.element_layout, ...],
    ):
        self = Self(
            unsafe_ptr._unsafe_nullable(),
            runtime_layout,
            element_runtime_layout,
        )

    @always_inline
    def __init__(
        out self,
        unsafe_ptr: Pointer[
            Scalar[Self.dtype],
            origin=Self.origin,
            address_space=Self.address_space,
        ],
        runtime_layout: RuntimeLayout[Self.layout, ...],
        element_runtime_layout: RuntimeLayout[Self.element_layout, ...],
    ):
        """Create a `LayoutTensor` with a `Pointer`, a runtime layout for
        the tensor, and the runtime layout of each element. The runtime layout
        element type will be casted to the layout tensor layout integer type.

        Args:
            unsafe_ptr: The `Pointer` pointing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
            element_runtime_layout: The runtime layout of each element.
        """

        self.ptr = unsafe_ptr
        self.runtime_layout = runtime_layout.cast[
            Self.layout_int_type, target_linear_idx_type=Self.linear_idx_type
        ]()
        self.runtime_element_layout = element_runtime_layout.cast[
            DType.int32, target_linear_idx_type=Self.linear_idx_type
        ]()

    comptime GenericLayoutTensorType = LayoutTensor[
        Self.dtype,
        Self.layout,
        Self.origin,
        address_space=.GENERIC,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
        alignment=Self.alignment,
    ]
    """LayoutTensor type with generic address space."""

    @always_inline
    def __init__(
        out self: Self.GenericLayoutTensorType,
        ref[Self.origin] device_buffer: DeviceBuffer[Self.dtype],
    ):
        """Create a `LayoutTensor` from a `DeviceBuffer`. The layout must have
        statically known dimensions.

        Note that the device buffer memory is on the accelerator device (GPU
        global memory). Code running on the CPU can use the
        [`DeviceContext`](/api/mojo/max/gpu/host/device_context/DeviceContext/) to
        allocate a `DeviceBuffer` and use that to construct a `LayoutTensor`
        that can be accessed on the GPU. You cannot directly access data in the
        `DeviceBuffer` or `LayoutTensor` from the CPU.

        The following example shows a typical pattern for using `DeviceBuffer`
        to construct a `LayoutTensor` that you can use on the GPU.

        ```mojo
        from max.gpu.host import DeviceContext, DeviceBuffer
        from layout import Layout, LayoutTensor

        comptime dtype = DType.float32

        var ctx = DeviceContext()
        # Allocate buffers
        var dev_buf = ctx.enqueue_create_buffer[dtype](16)
        var host_buf = ctx.enqueue_create_host_buffer[dtype](16)
        # Ensure buffers have been created
        ctx.synchronize()

        # Initialize host buffer and copy to device buffer
        for i in range(16):
            host_buf[i] = i
        ctx.enqueue_copy(dev_buf, host_buf)

        # Create LayoutTensor to use on device
        comptime layout = Layout.row_major(4, 4)
        var tensor = LayoutTensor[dtype, layout](dev_buf)
        ...
        ```

        Constraints:
            - Layout must be fully static.

        Args:
            device_buffer: Contains the underlying data to point to.
        """
        self = Self.GenericLayoutTensorType(
            device_buffer.unsafe_ptr()
            .unsafe_mut_cast[Self.mut]()
            .unsafe_origin_cast[Self.origin]()
        )

    @always_inline
    def __init__(
        out self: Self.GenericLayoutTensorType,
        ref[Self.origin] host_buffer: HostBuffer[Self.dtype],
    ):
        """Create a `LayoutTensor` from a `HostBuffer`. The layout must have
        statically known dimensions.

        The resulting tensor's data can only be accessed on the CPU.

        ```mojo
        from max.gpu.host import DeviceContext, HostBuffer
        from layout import Layout, LayoutTensor

        comptime dtype = DType.float32

        var ctx = DeviceContext()
        var dev_buf = ctx.enqueue_create_host_buffer[dtype](8)

        comptime layout = Layout.row_major(4, 4)
        var tensor = LayoutTensor[dtype, layout](dev_buf)
        ```

        Constraints:
            - Layout must be fully static.

        Args:
            host_buffer: Contains the underlying data to point to.
        """
        # TODO(MOCO-4435): remove this temporary variable.
        var host_ptr = Optional(host_buffer.unsafe_ptr())
        self = Self.GenericLayoutTensorType(
            unsafe_cast[origin=Self.origin](host_ptr),
        )

    @always_inline
    def __init__(
        out self: Self.GenericLayoutTensorType,
        ref[Self.origin] device_buffer: DeviceBuffer[Self.dtype],
        runtime_layout: RuntimeLayout[Self.layout, ...],
    ):
        """Create a `LayoutTensor` from a `DeviceBuffer` and a runtime layout.
        The runtime layout element type will be casted to the layout tensor layout
        integer type.

        The resulting tensor's data can only be accessed on the GPU.

        Constraints:
            - Element layout must be fully static.

        Args:
            device_buffer: The `DeviceBuffer` containing to the underlying data.
            runtime_layout: The runtime layout of the LayoutTensor.
        """
        self = Self.GenericLayoutTensorType(
            device_buffer.unsafe_ptr()
            .unsafe_mut_cast[Self.mut]()
            .unsafe_origin_cast[Self.origin](),
            runtime_layout,
        )

    @always_inline
    def __init__(
        out self: Self.GenericLayoutTensorType,
        ref[Self.origin] host_buffer: HostBuffer[Self.dtype],
        runtime_layout: RuntimeLayout[Self.layout, ...],
    ):
        """Create a `LayoutTensor` from a `HostBuffer` and a runtime layout.
        The runtime layout element type will be casted to the layout tensor layout
        integer type.

        The resulting tensor's data can only be accessed on the CPU.

        Constraints:
            - Element layout must be fully static.

        Args:
            host_buffer: The `HostBuffer` containing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
        """
        # TODO(MOCO-4435): remove this temporary variable.
        var host_ptr = Optional(host_buffer.unsafe_ptr())
        self = Self.GenericLayoutTensorType(
            unsafe_cast[origin=Self.origin](host_ptr),
            runtime_layout,
        )

    @always_inline
    def __init__(
        out self: Self.GenericLayoutTensorType,
        ref[Self.origin] device_buffer: DeviceBuffer[Self.dtype],
        runtime_layout: RuntimeLayout[Self.layout, ...],
        element_runtime_layout: RuntimeLayout[Self.element_layout, ...],
    ):
        """Create a `LayoutTensor` from a `DeviceBuffer`, a runtime layout for
        the tensor, and the runtime layout of each element. The runtime layout
        element type will be casted to the layout tensor layout integer type.

        The resulting tensor's data can only be accessed on the GPU.

        Args:
            device_buffer: The `DeviceBuffer` containing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
            element_runtime_layout: The runtime layout of each element.
        """
        self = Self.GenericLayoutTensorType(
            device_buffer.unsafe_ptr()
            .unsafe_mut_cast[Self.mut]()
            .unsafe_origin_cast[Self.origin](),
            runtime_layout,
            element_runtime_layout,
        )

    @always_inline
    def __init__(
        out self: Self.GenericLayoutTensorType,
        ref[Self.origin] host_buffer: HostBuffer[Self.dtype],
        runtime_layout: RuntimeLayout[Self.layout, ...],
        element_runtime_layout: RuntimeLayout[Self.element_layout, ...],
    ):
        """Create a `LayoutTensor` from a `HostBuffer`, a runtime layout for the
        tensor, and the runtime layout of each element. The runtime layout
        element type will be casted to the layout tensor layout integer type.

        The resulting tensor's data can only be accessed on the CPU.

        Args:
            host_buffer: The `HostBuffer` containing to the underlying data.
            runtime_layout: The runtime layout of the `LayoutTensor`.
            element_runtime_layout: The runtime layout of each element.
        """
        # TODO(MOCO-4435): remove this temporary variable.
        var host_ptr = Optional(host_buffer.unsafe_ptr())
        self = Self.GenericLayoutTensorType(
            unsafe_cast[origin=Self.origin](host_ptr),
            runtime_layout,
            element_runtime_layout,
        )

    @always_inline("builtin")
    @implicit
    def __init__(
        other: LayoutTensor,
        out self: type_of(other).Immut,
    ):
        """Implicitly cast a mutable LayoutTensor to immutable.

        Args:
            other: The mutable LayoutTensor to cast from.
        """
        self.ptr = other.ptr
        self.runtime_layout = other.runtime_layout
        self.runtime_element_layout = other.runtime_element_layout

    @always_inline("builtin")
    @implicit
    @doc_hidden
    def __init__[
        __disambig: Int = 0
    ](
        other: LayoutTensor[mut=True, ...],
        out self: type_of(other).OriginCastType[MutAnyOrigin],
    ):
        self.ptr = other.ptr.as_unsafe_any_origin()
        self.runtime_layout = other.runtime_layout
        self.runtime_element_layout = other.runtime_element_layout

    @always_inline("builtin")
    @implicit
    @doc_hidden
    def __init__[
        __disambig: Int = 0
    ](
        other: LayoutTensor,
        out self: type_of(other).OriginCastType[ImmutAnyOrigin],
    ):
        self.ptr = other.ptr.as_unsafe_any_origin()
        self.runtime_layout = other.runtime_layout
        self.runtime_element_layout = other.runtime_element_layout

    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(
            LayoutTensor[
                Self.dtype,
                Self.layout,
                _,
                address_space=Self.address_space,
                alignment=Self.alignment,
                element_layout=Self.element_layout,
                layout_int_type=Self.layout_int_type,
                linear_idx_type=Self.linear_idx_type,
                masked=Self.masked,
            ]
        ),
    ](
        self,
        out result: LayoutTensor[
            Self.dtype,
            Self.layout,
            origin_of(Self.origin, other_type.origin),
            alignment=Self.alignment,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            layout_int_type=Self.layout_int_type,
            linear_idx_type=Self.linear_idx_type,
            masked=Self.masked,
        ],
    ):
        """Returns a tensor merged with the specified `other_type`.

        Parameters:
            other_type: The type of the tensor to merge with.

        Returns:
            A tensor merged with the specified `other_type`.
        """
        return {
            self.ptr.unsafe_mut_cast[result.mut]().unsafe_origin_cast[
                result.origin
            ](),
            self.runtime_layout,
            self.runtime_element_layout,
        }

    comptime BitcastType[
        new_dtype: DType,
        /,
        address_space: AddressSpace = Self.address_space,
        element_layout: Layout = Self.element_layout,
    ] = LayoutTensor[
        new_dtype,
        Self.layout,
        Self.origin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
    ]
    """Type alias for bitcast result tensors.

    Parameters:
        new_dtype: The target data type to cast to.
        address_space: The address space for the result tensor.
        element_layout: The element layout for the result tensor.
    """

    @always_inline
    def bitcast[
        new_dtype: DType,
        /,
        target_address_space: AddressSpace = Self.address_space,
        _element_layout: Layout = Self.element_layout,
    ](self) -> Self.BitcastType[
        new_dtype, target_address_space, _element_layout
    ]:
        """Bitcast the underlying pointer to a new data type.

        Parameters:
            new_dtype: The new data type it is casting to.
            target_address_space: The address space of the returned `LayoutTensor`.
            _element_layout: The element layout of the returned `LayoutTensor`.

        Returns:
            A new `LayoutTensor` with the same memory location but with the
            specified data type, address space, and element layout.
        """
        return Self.BitcastType[
            new_dtype, target_address_space, _element_layout
        ](
            self.ptr.bitcast[Scalar[new_dtype]]().address_space_cast[
                target_address_space
            ](),
            self.runtime_layout,
        )

    comptime OriginCastType[
        mut: Bool,
        //,
        origin: Origin[mut=mut],
    ] = LayoutTensor[
        Self.dtype,
        Self.layout,
        origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
        alignment=Self.alignment,
    ]
    """Type alias for origin-cast result tensors.

    Parameters:
        mut: Whether the result tensor is mutable.
        origin: The origin for the result tensor.
    """

    comptime Immut = Self.OriginCastType[ImmOrigin(Self.origin)]
    """Type alias for an immutably-casted tensor."""

    comptime MutableAnyType = Self.OriginCastType[MutAnyOrigin]
    """Mutable LayoutTensor type with MutAnyOrigin."""
    comptime _AsMut = Self.OriginCastType[mut=True, _]

    @always_inline("nodebug")
    def as_unsafe_any_origin(
        self,
    ) -> type_of(self).OriginCastType[UnsafeAnyOrigin[mut=Self.mut]]:
        """Casts the origin of the `LayoutTensor` to `UnsafeAnyOrigin`.

        Returns:
            A tensor with the origin set to `UnsafeAnyOrigin`.

        Safety:

        It is **always** preferred to maintain a concrete origin values instead of
        using `UnsafeAnyOrigin`. Casting to `UnsafeAnyOrigin` is an inherently unsafe
        operation that will silently extend unrelated lifetimes and turn off
        exclusivity checking.
        """
        return {
            self.ptr.as_unsafe_any_origin(),
            self.runtime_layout,
            self.runtime_element_layout,
        }

    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=as_unsafe_any_origin)
    def as_any_origin(
        self,
    ) -> type_of(self).OriginCastType[AnyOrigin[mut=Self.mut]]:
        return self.as_unsafe_any_origin()

    comptime AddressSpaceCastType[
        address_space: AddressSpace = Self.address_space,
    ] = LayoutTensor[
        Self.dtype,
        Self.layout,
        Self.origin,
        address_space=address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
        alignment=Self.alignment,
    ]
    """Type alias for address-space-cast result tensors.

    Parameters:
        address_space: The target address space for the result tensor.
    """

    @always_inline("nodebug")
    def address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> Self.AddressSpaceCastType[target_address_space]:
        """Changes the address space of the `LayoutTensor`.

        Parameters:
            target_address_space: The new address space.

        Returns:
            A new `LayoutTensor` object with the same type and origin
            as the original `LayoutTensor`, and the new specified address_space.
        """
        return Self.AddressSpaceCastType[target_address_space](
            self.ptr.address_space_cast[target_address_space](),
            self.runtime_layout,
            self.runtime_element_layout,
        )

    @always_inline
    def as_imm(
        self,
    ) -> Self.OriginCastType[ImmOrigin(Self.origin)]:
        """
        Return an immutable version of this tensor.

        Returns:
            A `LayoutTensor` covering the same elements, but without mutability.
        """
        return {
            self.ptr.as_imm(),
            self.runtime_layout,
            self.runtime_element_layout,
        }

    @always_inline
    @deprecated(use=as_imm)
    def get_immutable(
        self,
    ) -> Self.OriginCastType[ImmOrigin(Self.origin)]:
        """
        Return an immutable version of this tensor.

        Returns:
            A `LayoutTensor` covering the same elements, but without mutability.
        """
        return self.as_imm()

    @always_inline
    def _offset(self, m: Int, n: Int) -> Int:
        """Calculate the memory offset for a 2D tensor element.

        Delegates to the IndexList overload for consistent behavior.

        Args:
            m: The row index (dimension 0).
            n: The column index (dimension 1).

        Returns:
            The calculated memory offset as an integer.
        """
        return self._offset(Index(m, n))

    @always_inline
    def _offset(self, coords: IndexList) -> Int:
        """Calculate the memory offset for a tensor element.

        Computes the linear memory offset for the given coordinates based on
        the tensor's stride configuration. Uses a per-dimension approach:
        for each dimension, if the compile-time stride is known (not
        UNKNOWN_VALUE), uses the static stride to enable constant folding;
        otherwise falls back to the runtime stride for that dimension.

        This approach allows tensors with partially known strides to benefit
        from constant folding where possible, while correctly handling view
        tensors where some strides depend on runtime values.

        Args:
            coords: The coordinates for the index. Must have the same size as
                the tensor's rank.

        Returns:
            The calculated memory offset as an integer.
        """
        comptime assert self.rank == coords.size

        # Use per-dimension approach: compile-time stride if known,
        # runtime stride if UNKNOWN_VALUE
        comptime static_strides = Self._to_static[
            Self.layout.stride, Self.linear_idx_type
        ]()
        var offset = 0

        comptime for i in range(Self.rank):
            comptime if static_strides[i] == UNKNOWN_VALUE:
                # Use runtime stride for unknown dimensions
                offset += self.runtime_layout.stride.value[i] * coords[i]
            else:
                # Use compile-time stride for known dimensions (enables
                # constant folding)
                offset += static_strides[i] * coords[i]

        return offset

    @always_inline("nodebug")
    def ptr_at_offset(
        self, coords: IndexList
    ) -> Pointer[
        Scalar[Self.dtype], address_space=Self.address_space, origin=self.origin
    ]:
        """Get a pointer offset at the given flattened coordinates.

        Args:
            coords: A flattened list of the offset coordinates.

        Returns:
           A pointer offset at the given flattened coordinates.
        """

        return self.ptr + self._offset(coords)

    @always_inline
    def _elementwise_unary[
        func: def(Self.element_type) capturing -> (Self.element_type),
    ](self) -> Self:
        """Apply an elementwise unary operation to all elements in the tensor.

        This is an internal method that applies the provided function to each
        element in the tensor. The operation is performed in-place and optimized
        for the tensor's memory layout.

        Parameters:
            func: A function that takes a single element and returns a
                transformed element. The function should be pure with no side
                effects for predictable results.

        Returns:
            Self: The modified tensor with the unary operation applied.

        Notes:

        This method requires the tensor to have a statically known layout
        for compile-time optimization.
        """
        comptime assert Self.layout.all_dims_known(), (
            "__elmentwise_unary must operates on tensors of statically know"
            " layouts"
        )

        comptime for i in range(self.layout.size()):
            comptime idx = self.layout(i)
            self.ptr.unsafe_mut_cast[True]().store(
                idx, func(self.ptr.load[width=Self.element_size](idx))
            )
        return self

    @always_inline
    def _elementwise_binary_with_broadcast[
        other_mut: Bool,
        //,
        func: def(Self.element_type, Self.element_type) capturing -> (
            Self.element_type
        ),
        other_layout: Layout,
        other_origin: Origin[mut=other_mut],
        other_masked: Bool,
        other_alignment: Int,
        other_layout_int_type: DType,
        other_linear_idx_type: DType,
    ](
        self,
        other: LayoutTensor[
            Self.dtype,
            other_layout,
            other_origin,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            layout_int_type=other_layout_int_type,
            linear_idx_type=other_linear_idx_type,
            masked=other_masked,
            alignment=other_alignment,
        ],
    ) -> Self:
        """Apply an elementwise binary operation with broadcasting support.

        This internal method applies a binary operation between elements of this
        tensor and another tensor, with support for limited broadcasting
        patterns. The operation is performed in-place on this tensor.

        Parameters:
            other_mut: Whether the other tensor is mutable.
            func: A binary function that takes two elements (one from each
                tensor) and returns a single element as the result of the
                operation.
            other_layout: The layout of the other tensor.
            other_origin: The origin type of the other tensor.
            other_masked: Whether the other tensor is masked.
            other_alignment: The memory alignment of the other tensor.
            other_layout_int_type: The dimension type of the other tensor.
            other_linear_idx_type: The linear idx type of the other tensor.

        Args:
            other: The second tensor operand for the binary operation.

        Returns:
            Self: The modified tensor with the binary operation applied.

        Notes:

        - Currently supports only rank-2 tensors or tensors of the same rank.
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
            match the corresponding dimension of the rank-2 tensor.
        - The operation is optimized based on the memory layout of both tensors.
        """

        comptime if Self.rank == other.rank:
            comptime for axis in range(Self.rank):
                comptime assert axis != UNKNOWN_VALUE
                comptime assert other.shape[axis]() == self.shape[axis](), (
                    "_elementwise_binary_with_broadcast requires shape to"
                    " be the same for tensors of the same rank"
                )

        comptime assert Self.layout.all_dims_known(), (
            "_elementwise_binary_with_broadcast must operates on tensors"
            " of statically know layouts"
        )
        comptime assert other.rank <= Self.rank, (
            "_elementwise_binary_with_broadcast must operates on tensor of"
            " equal of lower rank"
        )

        # TODO(KERN-812): Support numpy like broadcasting and relax rank-2
        # constrain.
        comptime assert (
            Self.rank == 2 or Self.rank == other.rank
        ), "Only supports rank-2 tensor, or same rank"

        comptime if other.rank == 1:
            comptime assert other.shape[0]() == self.shape[0](), (
                "_elementwise_binary_with_broadcast 1d tensor operand must"
                " have a dim that matches the tensors"
            )

            comptime for i in range(self.layout.size()):
                comptime other_size = other.layout.size()

                comptime lhs_idx = self.layout(i)
                comptime rhs_idx = other.layout(i % other_size)

                self.ptr.unsafe_mut_cast[True]().store(
                    lhs_idx,
                    func(
                        self.ptr.load[width=Self.element_size](lhs_idx),
                        other.ptr.load[width=Self.element_size](rhs_idx),
                    ),
                )
            return self

        comptime for i in range(self.layout.size()):
            comptime lhs_idx = self.layout(i)
            comptime rhs_idx = other.layout(i)
            self.ptr.unsafe_mut_cast[True]().store(
                lhs_idx,
                func(
                    self.ptr.load[width=Self.element_size](lhs_idx),
                    other.ptr.load[width=Self.element_size](rhs_idx),
                ),
            )
        return self

    @always_inline
    def __add__(
        self, other: Scalar[Self.dtype]
    ) -> Self.OriginCastType[MutAnyOrigin]:
        """Add a scalar value to each element of the tensor.

        Performs an elementwise addition operation, adding the scalar value to
        each element in the tensor. This operation creates a new tensor with the
        results.

        Args:
            other: The scalar value to add to each element.

        Returns:
            A new tensor containing the results of the addition operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            addition.
        - For in-place addition, use the `__iadd__` method instead (`+=`
            operator).
        """

        @__parameter
        def add_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) + val

        return self._stack_copy()._elementwise_unary[add_val]()

    @always_inline
    def __iadd__(self, other: Scalar[Self.dtype]):
        """Add a scalar value to each element of the tensor in-place.

        Performs an elementwise addition operation, adding the scalar value to
        each element in the tensor. This operation modifies the tensor in-place.

        Args:
            other: The scalar value to add to each element.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
        """

        @__parameter
        def add_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) + val

        _ = self._elementwise_unary[add_val]()

    @always_inline
    def __add__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            Self.dtype,
            other_layout,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            ...,
        ],
    ) -> Self.OriginCastType[MutAnyOrigin]:
        """Add another tensor to this tensor elementwise.

        Performs an elementwise addition between this tensor and another tensor.
        This operation creates a new tensor with the results.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to add to this tensor.

        Returns:
            A new tensor containing the results of the addition operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            addition.
        - For in-place addition, use the `__iadd__` method instead (`+=`
            operator).
        """

        def add_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs + rhs

        return self._stack_copy()._elementwise_binary_with_broadcast[add_val](
            other
        )

    @always_inline
    def __iadd__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            Self.dtype,
            other_layout,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            ...,
        ],
    ):
        """Add another tensor to this tensor elementwise in-place.

        Performs an elementwise addition between this tensor and another tensor.
        This operation modifies the tensor in-place.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to add to this tensor.

        Performance:

        - This operation modifies the tensor directly without creating a
            copy.
        """

        def add_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs + rhs

        _ = self._elementwise_binary_with_broadcast[add_val](other)

    @always_inline
    def __mul__(
        self, other: Scalar[Self.dtype]
    ) -> Self.OriginCastType[MutAnyOrigin]:
        """Multiply each element of the tensor by a scalar value.

        Performs an elementwise multiplication operation, multiplying each
        element in the tensor by the scalar value. This operation creates a new
        tensor with the results.

        Args:
            other: The scalar value to multiply with each element.

        Returns:
            A new tensor containing the results of the multiplication operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            multiplication.
        - For in-place multiplication, use the `__imul__` method instead
            (`*=` operator).
        """

        @__parameter
        def mul_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) * val

        return self._stack_copy()._elementwise_unary[mul_val]()

    @always_inline
    def __mul__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            Self.dtype,
            other_layout,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            ...,
        ],
    ) -> Self.OriginCastType[MutAnyOrigin]:
        """Multiply this tensor with another tensor elementwise.

        Performs an elementwise multiplication (Hadamard product) between this tensor
        and another tensor. This operation creates a new tensor with the results.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Note: This is NOT a matrix multiplication operation. For matrix
        multiplication, use the appropriate matmul function instead.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to multiply with this tensor.

        Returns:
            A new tensor containing the results of the elementwise
            multiplication.

        Performance:

        - This operation creates a copy of the tensor before performing the
            multiplication.
        - For in-place multiplication, use the `__imul__` method instead
            (`*=` operator).
        """

        def mul_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs * rhs

        return self._stack_copy()._elementwise_binary_with_broadcast[mul_val](
            other
        )

    @always_inline
    def __imul__(self, other: Scalar[Self.dtype]):
        """Multiply each element of the tensor by a scalar value in-place.

        Performs an elementwise multiplication operation, multiplying each
        element in the tensor by the scalar value. This operation modifies the
        tensor in-place.

        Args:
            other: The scalar value to multiply with each element.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
        """

        @__parameter
        def mul_val(val: Self.element_type) -> Self.element_type:
            return Self.element_type(other) * val

        _ = self._elementwise_unary[mul_val]()

    @always_inline
    def __imul__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            Self.dtype,
            other_layout,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            ...,
        ],
    ):
        """Multiply this tensor with another tensor elementwise in-place.

        Performs an elementwise multiplication (Hadamard product) between this
        tensor and another tensor. This operation modifies the tensor in-place.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Note: This is NOT a matrix multiplication operation. For matrix
        multiplication, use the appropriate matmul function instead.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to multiply with this tensor.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
        """

        def mul_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs * rhs

        _ = self._elementwise_binary_with_broadcast[mul_val](other)

    @always_inline
    def __sub__(
        self, other: Scalar[Self.dtype]
    ) -> Self.OriginCastType[MutAnyOrigin]:
        """Subtract a scalar value from each element of the tensor.

        Performs an elementwise subtraction operation, subtracting the scalar
        value from each element in the tensor. This operation creates a new
        tensor with the results.

        Args:
            other: The scalar value to subtract from each element.

        Returns:
            A new tensor containing the results of the subtraction operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            subtraction.
        - For in-place subtraction, use the `__isub__` method instead (`-=`
            operator).
        """

        @__parameter
        def sub_val(val: Self.element_type) -> Self.element_type:
            return val - Self.element_type(other)

        return self._stack_copy()._elementwise_unary[sub_val]()

    @always_inline
    def __sub__[
        other_layout: Layout,
    ](
        self,
        other: LayoutTensor[
            Self.dtype,
            other_layout,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            ...,
        ],
    ) -> Self.OriginCastType[MutAnyOrigin]:
        """Subtract another tensor from this tensor elementwise.

        Performs an elementwise subtraction between this tensor and another
        tensor. This operation creates a new tensor with the results.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to subtract from this tensor.

        Returns:
            A new tensor containing the results of the subtraction operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            subtraction.
        - For in-place subtraction, use the `__isub__` method instead (`-=`
            operator).
        """

        def sub_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs - rhs

        return self._stack_copy()._elementwise_binary_with_broadcast[sub_val](
            other
        )

    @always_inline
    def __isub__(self, other: Scalar[Self.dtype]):
        """Subtract a scalar value from each element of the tensor in-place.

        Performs an elementwise subtraction operation, subtracting the scalar
        value from each element in the tensor. This operation modifies the
        tensor in-place.

        Args:
            other: The scalar value to subtract from each element.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
        """

        @__parameter
        def sub_val(val: Self.element_type) -> Self.element_type:
            return val - Self.element_type(other)

        _ = self._elementwise_unary[sub_val]()

    @always_inline
    def __isub__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            Self.dtype,
            other_layout,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            ...,
        ],
    ):
        """Subtract another tensor from this tensor elementwise in-place.

        Performs an elementwise subtraction between this tensor and another
        tensor. This operation modifies the tensor in-place.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to subtract from this tensor.

        Performance:

        - This operation modifies the tensor directly without creating a copy.
        """

        def sub_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs - rhs

        _ = self._elementwise_binary_with_broadcast[sub_val](other)

    @always_inline
    def __truediv__(
        self, other: Scalar[Self.dtype]
    ) -> Self.OriginCastType[MutAnyOrigin]:
        """Divide each element of the tensor by a scalar value.

        Performs an elementwise division operation, dividing each element in the
        tensor by the scalar value. This operation creates a new tensor with the
        results.

        Args:
            other: The scalar value to divide each element by.

        Returns:
            A new tensor containing the results of the division operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            division.
        - For in-place division, use the `__itruediv__` method instead
            (`/=` operator).

        Notes:

        - Division by zero will result in undefined behavior or errors
            depending on the dtype.
        - For integer dtypes, this performs integer division.
        """

        @__parameter
        def div_val(val: Self.element_type) -> Self.element_type:
            return val / Self.element_type(other)

        return self._stack_copy()._elementwise_unary[div_val]()

    @always_inline
    def __truediv__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            Self.dtype,
            other_layout,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            ...,
        ],
    ) -> Self.OriginCastType[MutAnyOrigin]:
        """Divide this tensor by another tensor elementwise.

        Performs an elementwise division between this tensor and another tensor.
        This operation creates a new tensor with the results.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to divide this tensor by.

        Returns:
            A new tensor containing the results of the division operation.

        Performance:

        - This operation creates a copy of the tensor before performing the
            division.
        - For in-place division, use the `__itruediv__` method instead
            (`/=` operator).

        Notes:

        - Division by zero will result in undefined behavior or errors depending on the dtype.
        - For integer dtypes, this performs integer division.
        """

        def div_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs / rhs

        return self._stack_copy()._elementwise_binary_with_broadcast[div_val](
            other
        )

    def __itruediv__(self, other: Scalar[Self.dtype]):
        """Divide each element of the tensor by a scalar value in-place.

        Performs an elementwise division operation, dividing each element in the
        tensor by the scalar value. This operation modifies the tensor in-place.

        Args:
            other: The scalar value to divide each element by.

        Performance:

        - This operation modifies the tensor directly without creating a copy.

        Notes:

        - Division by zero will result in undefined behavior or errors depending on the dtype.
        - For integer dtypes, this performs integer division.
        """

        @__parameter
        def div_val(val: Self.element_type) -> Self.element_type:
            return val / Self.element_type(other)

        _ = self._elementwise_unary[div_val]()

    @always_inline
    def __itruediv__[
        other_layout: Layout
    ](
        self,
        other: LayoutTensor[
            Self.dtype,
            other_layout,
            address_space=Self.address_space,
            element_layout=Self.element_layout,
            ...,
        ],
    ):
        """Divide this tensor by another tensor elementwise in-place.

        Performs an elementwise division between this tensor and another tensor.
        This operation modifies the tensor in-place.

        Limited broadcasting is supported:
        - For tensors of the same rank, shapes must match exactly.
        - For rank-1 to rank-2 broadcasting, the rank-1 tensor's dimension must
          match the corresponding dimension of the rank-2 tensor.

        Parameters:
            other_layout: The layout of the other tensor.

        Args:
            other: The tensor to divide this tensor by.

        Performance:

        - This operation modifies the tensor directly without creating a copy.

        Notes:

        - Division by zero will result in undefined behavior or errors depending on the dtype.
        - For integer dtypes, this performs integer division.
        """

        def div_val(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return lhs / rhs

        _ = self._elementwise_binary_with_broadcast[div_val](other)

    @always_inline
    def __exp__(self) -> Self:
        """Computes element-wise exponential function.

        Returns a new tensor containing the
        [element-wise exponential](https://mojolang.org/docs/std/math/math/exp/) of the input tensor.

        Returns:
            A new tensor containing the element-wise exponential.
        """
        comptime assert (
            Self.dtype.is_floating_point()
        ), "dtype must be floating point"

        @__parameter
        def exp_func(val: Self.element_type) -> Self.element_type:
            return exp(val)

        return {
            self._stack_copy()
            ._elementwise_unary[exp_func]()
            .ptr.unsafe_mut_cast[Self.mut]()
            .unsafe_origin_cast[Self.origin](),
            self.runtime_layout,
            self.runtime_element_layout,
        }

    @always_inline("nodebug")
    def _load_offset(
        self, offset: Scalar[Self.linear_idx_type]
    ) -> Self.element_type:
        """Retrieves a single element from the tensor at the specified offset.

        This method provides array-like linear indexing for the tensor.

        Args:
            offset: The integer offset for array indexing.

        Returns:
            The element at the specified offset with the tensor's data type.
        """

        return (
            Element[index_type=Self.linear_idx_type]
            .load(self.ptr + offset, self.runtime_element_layout)
            .element_data
        )

    @always_inline("nodebug")
    def _load_scalar_offset(
        self, offset: Scalar[Self.linear_idx_type]
    ) -> Scalar[Self.dtype]:
        """Retrieves a single scalar from the tensor at the specified offset.

        This method loads the element at the given offset and returns the first
        scalar lane. For tensors with element_size == 1, this is the only value.
        For tiled/vectorized elements, this returns the 0th lane.

        Args:
            offset: The integer offset for array indexing.

        Returns:
            The scalar value at the specified offset.
        """
        return self._load_offset(offset)[0]

    @always_inline("nodebug")
    def __getitem__[*Tys: Indexer](self, *args: *Tys) -> Self.element_type:
        """Retrieves a single element from the tensor at the specified indices.

        This method provides array-like indexing for the tensor. The number of
        indices provided must match the rank of the tensor, otherwise an error
        will occur at runtime.

        Parameters:
            Tys: The type of the indices. Must implement the `Indexer` trait,
                and match the rank of the tensor.

        Args:
            args: The indices specifying the element's position in the tensor.

        Returns:
            The element at the specified position with the tensor's data type.
        """
        comptime arg_count = args.__len__()

        comptime assert (
            Self.rank == arg_count or Self.num_strides == arg_count
        ), (
            "Indexed with "
            + String(arg_count)
            + " dims, but Self.rank, Self.num_strides = "
            + String(Self.rank)
            + ", "
            + String(self.num_strides)
        )

        var index_list = Self.idx_list_t[arg_count](fill=0)

        comptime for arg_idx in range(arg_count):
            index_list[arg_idx] = index(args[arg_idx])

        # Bounds checking for each dimension
        # Note: We skip bounds checking for nested layouts because computing the logical
        # dimension size requires calling IntTuple.__getitem__ which cannot be evaluated
        # at compile-time when assertions are enabled (it would trigger llvm.memcpy).
        #
        # We use a simple static error message to minimize register pressure on GPU kernels.
        # Including runtime values (idx, dim_size) in the message requires passing them to
        # debug_assert and allocating buffer machinery, which increases register usage.
        comptime if ASSERT_MODE == "all" and depth(Self.layout.shape) <= 1:
            comptime for arg_idx in range(arg_count):
                var idx = index_list[arg_idx]
                var dim_size = self.dim[arg_idx]()
                assert 0 <= idx < dim_size, "LayoutTensor index out of bounds"

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset[rank=arg_count](strides, index_list)
        return self._load_offset(Scalar[Self.linear_idx_type](offset))

    @always_inline("nodebug")
    def __getitem__(self, crd: RuntimeTuple) -> Self.element_type:
        """Retrieves a single element from the tensor at the specified indices.

        This method provides array-like indexing for the tensor. The number of
        indices provided must match the rank of the tensor, otherwise an error
        will occur at runtime.

        Args:
            crd: The coordinate specifying the element's position in each dimension. For example, in a 3D tensor, you would use (i, j, k).

        Returns:
            The element at the specified position with the tensor's data type.
        """

        var offset = self.runtime_layout(crd)
        return self._load_offset(offset)

    @always_inline("nodebug")
    def load_scalar[*Tys: Indexer](self, *args: *Tys) -> Scalar[Self.dtype]:
        """Retrieves a single scalar from the tensor at the specified indices.

        This method provides scalar element access for the tensor, which is
        useful in generic contexts where `__getitem__` returns a SIMD vector
        of `element_size` elements. This method always returns a single scalar
        value (the 0th lane of the element).

        The number of indices provided must match the rank of the tensor,
        otherwise an error will occur at runtime.

        Parameters:
            Tys: The type of the indices. Must implement the `Indexer` trait,
                and match the rank of the tensor.

        Args:
            args: The indices specifying the element's position in the tensor.

        Returns:
            The scalar value at the specified position with the tensor's dtype.
        """
        comptime arg_count = args.__len__()

        comptime assert (
            Self.rank == arg_count or Self.num_strides == arg_count
        ), (
            "Indexed with "
            + String(arg_count)
            + " dims, but Self.rank, Self.num_strides = "
            + String(Self.rank)
            + ", "
            + String(self.num_strides)
        )

        var index_list = Self.idx_list_t[arg_count](fill=0)

        comptime for arg_idx in range(arg_count):
            index_list[arg_idx] = index(args[arg_idx])

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset[rank=arg_count](strides, index_list)
        return self._load_scalar_offset(Scalar[Self.linear_idx_type](offset))

    @always_inline("nodebug")
    def load_scalar(self, crd: RuntimeTuple) -> Scalar[Self.dtype]:
        """Retrieves a single scalar from the tensor at the specified coordinates.

        This method provides scalar element access for the tensor, which is
        useful in generic contexts where `__getitem__` returns a SIMD vector
        of `element_size` elements. This method always returns a single scalar
        value (the 0th lane of the element).

        Args:
            crd: The coordinate specifying the element's position in each
                dimension. For example, in a 3D tensor, you would use (i, j, k).

        Returns:
            The scalar value at the specified position with the tensor's dtype.
        """
        var offset = self.runtime_layout(crd)
        return self._load_scalar_offset(offset)

    @always_inline("nodebug")
    def __setitem__[
        *Tys: Indexer
    ](self, *args: *Tys, val: Self.element_type) where Self.mut:
        """Sets a single element in a tensor at the specified indices.

        This method provides array-like element assignment for tensors.

        Parameters:
            Tys: The type of the indices. Must implement the `Indexer` trait,
                and match the rank of the tensor.

        Args:
            args: The indices specifying the element's position in the tensor.
            val: The value to write to the tensor at the specified position.

        Notes:

        - Bounds checking is NOT currently supported for `__setitem__` due to
          complications with certain layout types and mutation contexts.
          Use `__getitem__`, `load`, or `store` methods for bounds-checked access.
          In the future, this restriction will be lifted.
        """

        comptime arg_count = args.__len__()

        comptime assert (
            Self.rank == arg_count or Self.num_strides == arg_count
        ), (
            "Indexed with "
            + String(arg_count)
            + " dims, but Self.rank, Self.num_strides = "
            + String(Self.rank)
            + ", "
            + String(self.num_strides)
        )

        var index_list = Self.idx_list_t[arg_count](fill=0)

        comptime for arg_idx in range(arg_count):
            index_list[arg_idx] = index(args[arg_idx])

        var strides = self.runtime_layout.stride.value
        var offset = Self._get_offset(strides, index_list)

        Element[index_type=Self.linear_idx_type](
            val, self.runtime_element_layout
        ).store(self.ptr.unsafe_mut_cast[True]() + offset)

    @always_inline("nodebug")
    def load[
        width: Int,
        load_alignment: Int = Self.alignment,
        non_temporal: Bool = False,
    ](self, m: Int, n: Int) -> SIMD[Self.dtype, width]:
        """Load a SIMD vector from the tensor at the specified 2D coordinates.

        Performs a vectorized load operation from the tensor's memory,
        retrieving `width` consecutive elements starting at position (m, n).
        This method enables efficient SIMD operations on tensor data.

        Parameters:
            width: The number of elements to load into the SIMD vector. Should match
                  the target hardware's vector width for optimal performance.
            load_alignment: The alignment to use. Defaults to Self.alignment.
            non_temporal: If True, issue a non-temporal (streaming) load hint,
                indicating the data has no temporal locality and should not
                pollute caches.

        Args:
            m: The row index (first dimension).
            n: The column index (second dimension).

        Returns:
            A SIMD vector containing 'width' consecutive elements from the tensor.

        Performance:

        - Uses unaligned memory access which may be slower on some
            architectures.
        - For aligned access, use `aligned_load` instead when data alignment is
            guaranteed.
        - The load operation is optimized based on the tensor's memory layout.

        Notes:

        - Bounds checking is performed via debug_assert for the base coordinate
            and the full SIMD width range. Enable assertions with `-D ASSERT=all`
            to catch out-of-bounds accesses during development.
        - The elements are loaded according to the tensor's stride configuration.
        """

        # Bounds checking for 2D load
        # Note: We skip bounds checking for nested layouts because computing the logical
        # dimension size requires calling IntTuple.__getitem__ which cannot be evaluated
        # at compile-time when assertions are enabled (it would trigger llvm.memcpy).
        #
        # We use a simple static error message to minimize register pressure on GPU kernels.
        comptime if ASSERT_MODE == "all" and depth(Self.layout.shape) <= 1:
            # Use self.dim which correctly handles both compile-time and
            # runtime layouts (including UNKNOWN_VALUE dimensions)
            var dim0 = self.dim[0]()
            var dim1 = self.dim[1]()
            assert 0 <= m < dim0, "LayoutTensor load out of bounds"
            assert (
                0 <= n and n + width <= dim1
            ), "LayoutTensor load out of bounds"

        return self.ptr.load[
            width=width, alignment=load_alignment, non_temporal=non_temporal
        ](self._offset(m, n))

    @always_inline("nodebug")
    def load[
        width: Int,
        load_alignment: Int = Self.alignment,
        non_temporal: Bool = False,
    ](self, coords: IndexList[...]) -> SIMD[Self.dtype, width]:
        """Load a SIMD vector from the tensor at the specified coordinates.

        Performs a vectorized load operation from the tensor's memory,
        retrieving `width` consecutive elements starting at the position specified
        by `coords`. This method enables efficient SIMD operations on tensor data
        and works with tensors of any rank.

        Parameters:
            width: The number of elements to load into the SIMD vector. Should match
                    the target hardware's vector width for optimal performance.
            load_alignment: The alignment to use. Defaults to Self.alignment.
            non_temporal: If True, issue a non-temporal (streaming) load hint,
                indicating the data has no temporal locality and should not
                pollute caches.

        Args:
            coords: The coordinates to index. Must have the same size as the tensor's rank.

        Returns:
            A SIMD vector containing 'width' consecutive elements from the tensor.

        Performance:

        - Uses unaligned memory access which may be slower on some
            architectures.
        - For aligned access, use `aligned_load` instead when data alignment is
            guaranteed.
        - The load operation is optimized based on the tensor's memory layout.

        Notes:

        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        - The elements are loaded according to the tensor's stride configuration.
        """
        comptime assert self.rank == coords.size
        assert self.runtime_layout.stride.value[self.rank - 1] == 1

        return self.ptr.load[
            width=width, alignment=load_alignment, non_temporal=non_temporal
        ](self._offset(coords))

    @always_inline
    def prefetch(self, m: Int, n: Int):
        """Prefetch tensor data at the specified 2D coordinates into cache.

        Issues a software prefetch hint to the processor to load the data at
        position (m, n) into the cache hierarchy. This can improve performance
        by reducing memory latency for subsequent accesses to the same location.

        Args:
            m: The row index (first dimension).
            n: The column index (second dimension).

        Performance:

        - Prefetching is a performance hint and does not guarantee data will be
            cached.
        - Most effective when issued sufficiently ahead of the actual data
            access.
        - Uses high locality prefetch to the data cache, optimized for data that
            will be accessed multiple times.
        - Can reduce memory access latency by 50-90% when used correctly.

        Notes:

        - Excessive prefetching can pollute the cache and degrade performance.
        - Most beneficial for predictable access patterns that would otherwise
            cause cache misses.
        - No operation is performed on the prefetched data.
        """
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            self.ptr + self._offset(m, n)
        )

    @always_inline
    def prefetch(self, coords: IndexList):
        """Prefetch tensor data at the specified coordinates into cache.

        Issues a software prefetch hint to the processor to load the data at
        coords into the cache hierarchy. This can improve performance
        by reducing memory latency for subsequent accesses to the same location.

        Args:
            coords: The indices.

        Performance:

        - Prefetching is a performance hint and does not guarantee data will be
            cached.
        - Most effective when issued sufficiently ahead of the actual data
            access.
        - Uses high locality prefetch to the data cache, optimized for data that
            will be accessed multiple times.
        - Can reduce memory access latency by 50-90% when used correctly.

        Notes:

        - Excessive prefetching can pollute the cache and degrade performance.
        - Most beneficial for predictable access patterns that would otherwise
            cause cache misses.
        - No operation is performed on the prefetched data.
        """
        prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](
            self.ptr + self._offset(coords)
        )

    @always_inline("nodebug")
    def aligned_load[
        width: Int
    ](self, m: Int, n: Int) -> SIMD[Self.dtype, width]:
        """Load a SIMD vector with alignment guarantees from the tensor.

        Performs an aligned vectorized load operation from the tensor's memory,
        retrieving `width` consecutive elements starting at position (m, n). The
        alignment is automatically calculated based on the SIMD width and dtype.

        Parameters:
            width: The number of elements to load into the SIMD vector. Should
                match the target hardware's vector width for optimal performance.

        Args:
            m: The row index (first dimension).
            n: The column index (second dimension).

        Returns:
            A SIMD vector containing 'width' consecutive elements from the tensor.

        Performance:

        - Uses aligned memory access which is faster than unaligned access on
            most architectures.
        - The alignment is automatically calculated based on the SIMD width and
            dtype.
        - Can be up to 2x faster than unaligned loads on architectures that
            require alignment.

        Notes:

        - The caller must ensure that the memory at (m, n) is properly aligned.
            Misaligned access with this method may cause hardware exceptions on
            some architectures.
        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        """

        comptime _alignment = align_of[SIMD[Self.dtype, width]]()
        return self.ptr.load[width=width, alignment=_alignment](
            self._offset(m, n)
        )

    @always_inline("nodebug")
    def aligned_load[
        width: Int
    ](self, coords: IndexList[...]) -> SIMD[Self.dtype, width]:
        """Load a SIMD vector with alignment guarantees from the tensor.

        Performs an aligned vectorized load operation from the tensor's memory,
        retrieving `width` consecutive elements starting at the position specified
        by `coords`. The alignment is automatically calculated based on the SIMD width
        and dtype. This method enables efficient SIMD operations on tensor data and
        works with tensors of any rank.

        Parameters:
            width: The number of elements to load into the SIMD vector. Should
                match the target hardware's vector width for optimal performance.

        Args:
            coords: The coordinates to index. Must have the same size as the tensor's rank.

        Returns:
            A SIMD vector containing 'width' consecutive elements from the tensor.

        Performance (copied from `aligned_load[width](m,n)`):

        - Uses aligned memory access which is faster than unaligned access on
            most architectures.
        - The alignment is automatically calculated based on the SIMD width and
            dtype.
        - Can be up to 2x faster than unaligned loads on architectures that
            require alignment.

        Notes:

        - The caller must ensure that the memory at the specified coordinates is
            properly aligned. Misaligned access with this method may cause hardware
            exceptions on some architectures.
        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        - The elements are loaded according to the tensor's stride configuration.
        - The last dimension must have unit stride (stride == 1) for this operation
            to be valid.
        """
        comptime assert self.rank == coords.size
        comptime _alignment = align_of[SIMD[Self.dtype, width]]()
        return self.ptr.load[width=width, alignment=_alignment](
            self._offset(coords)
        )

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def store[
        width: SIMDLength, store_alignment: Int = Self.alignment
    ](
        self: LayoutTensor[mut=True, Self.dtype, ...],
        m: Int,
        n: Int,
        val: SIMD[Self.dtype, width],
    ):
        """Store a SIMD vector to the tensor at the specified 2D coordinates.

        Performs a vectorized store operation to the tensor's memory, writing
        'width' consecutive elements starting at position (m, n). This method
        enables efficient SIMD operations on tensor data.

        Parameters:
            width: The number of elements in the SIMD vector to store. Should
                match the target hardware's vector width for optimal performance.
            store_alignment: The alignment to use. Defaults to Self.alignment.

        Args:
            m: The row index (first dimension) where the store operation begins.
            n: The column index (second dimension) where the store operation
                begins.
            val: The SIMD vector containing the values to store in the tensor.

        Performance:

        - Uses unaligned memory access which may be slower on some
            architectures.
        - For aligned access, use aligned_store instead when data alignment is
            guaranteed.
        - The store operation is optimized based on the tensor's memory layout.

        Notes:

        - Bounds checking is performed via debug_assert for the base coordinate
            and the full SIMD width range. Enable assertions with `-D ASSERT=all`
            to catch out-of-bounds accesses during development.
        - The elements are stored according to the tensor's stride configuration.
        - This operation modifies the tensor's data in-place.
        """

        # Bounds checking for 2D store
        # Note: We skip bounds checking for nested layouts because computing the logical
        # dimension size requires calling IntTuple.__getitem__ which cannot be evaluated
        # at compile-time when assertions are enabled (it would trigger llvm.memcpy).
        #
        # We use a simple static error message to minimize register pressure on GPU kernels.
        comptime if ASSERT_MODE == "all" and depth(Self.layout.shape) <= 1:
            # Use self.dim which correctly handles both compile-time and
            # runtime layouts (including UNKNOWN_VALUE dimensions)
            var dim0 = self.dim[0]()
            var dim1 = self.dim[1]()
            debug_assert(
                0 <= m < dim0,
                "LayoutTensor store out of bounds: m=",
                m,
                " (valid range: [0, ",
                dim0,
                "))",
            )
            debug_assert(
                0 <= n and n + width <= dim1,
                "LayoutTensor store out of bounds: n=",
                n,
                ", width=",
                Int(width),
                " (valid range for n+width: [0, ",
                dim1,
                "])",
            )

        return self.ptr.store[alignment=store_alignment](
            self._offset(m, n), val
        )

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def store[
        width: SIMDLength, store_alignment: Int = Self.alignment
    ](
        self: LayoutTensor[mut=True, Self.dtype, ...],
        coords: IndexList[...],
        val: SIMD[Self.dtype, width],
    ):
        """Store a SIMD vector to the tensor at the specified ND coordinates.

        Performs a vectorized store operation to the tensor's memory, writing
        'width' consecutive elements starting at position (m, n). This method
        enables efficient SIMD operations on tensor data.

        Parameters:
            width: The number of elements in the SIMD vector to store. Should
                match the target hardware's vector width for optimal performance.
            store_alignment: The alignment to use. Defaults to Self.alignment.

        Args:
            coords: The coordinates to index.
            val: The SIMD vector containing the values to store in the tensor.

        Performance:

        - Uses unaligned memory access which may be slower on some
            architectures.
        - For aligned access, use aligned_store instead when data alignment is
            guaranteed.
        - The store operation is optimized based on the tensor's memory layout.

        Notes:

        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        - The elements are stored according to the tensor's stride configuration.
        - This operation modifies the tensor's data in-place.
        """
        comptime assert self.rank == coords.size
        assert self.runtime_layout.stride.value[self.rank - 1] == 1

        return self.ptr.store[alignment=store_alignment](
            self._offset(coords), val
        )

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def aligned_store[
        width: SIMDLength
    ](self: Self._AsMut, m: Int, n: Int, val: SIMD[Self.dtype, width]):
        """Store a SIMD vector with alignment guarantees to the tensor.

        Performs an aligned vectorized store operation to the tensor's memory,
        writing `width` consecutive elements starting at position (m, n). The
        alignment is automatically calculated based on the SIMD width and dtype.

        Parameters:
            width: The number of elements in the SIMD vector to store. Should
                match the target hardware's vector width for optimal performance.

        Args:
            m: The row index (first dimension) where the store operation begins.
            n: The column index (second dimension) where the store operation
                begins.
            val: The SIMD vector containing the values to store in the tensor.

        Performance:

        - Uses aligned memory access which is faster than unaligned access on
            most architectures.
        - The alignment is automatically calculated based on the SIMD width and
            dtype.
        - Can be up to 2x faster than unaligned stores on architectures that
            require alignment.
        - Particularly important for streaming stores that bypass the cache.

        Notes:

        - The caller must ensure that the memory at (m, n) is properly aligned.
            Misaligned access with this method may cause hardware exceptions on
            some architectures.
        - No bounds checking is performed. Accessing out-of-bounds indices will
            result in undefined behavior.
        - This operation modifies the tensor's data in-place.
        """

        comptime _alignment = align_of[SIMD[Self.dtype, width]]()
        return self.ptr.store[alignment=_alignment](self._offset(m, n), val)

    @always_inline("nodebug")
    def size(self) -> Int:
        """
        Get the total number of elements that the tensor can contain.

        Returns:
          The total number of elements that can be stores in the tensor.
        """

        comptime if Self.layout.all_dims_known():
            comptime size = Self.layout.size()
            return size
        else:
            return self.runtime_layout.size()

    @staticmethod
    @always_inline("nodebug")
    def stack_allocation[
        *, stack_alignment: Int = Self.alignment
    ]() -> Self.StackTensorType:
        """Allocates stack memory for a `LayoutTensor` with a fully static
        layout.

        Creates a new `LayoutTensor` instance with memory allocated on the stack
        rather than the heap. This provides deterministic memory management and
        potentially better performance for tensors with known sizes at compile
        time.

        Constraints:
            - The layout must be fully static (all dimensions known at compile
                time).
            - The alignment must be a multiple of the tensor's minimum required
                alignment.

        Parameters:
             stack_alignment: Memory alignment value for the allocation in bytes. Must
                be a multiple of the tensor's minimum required alignment.
                Default is the tensor's natural alignment based on its data type
                and layout.

        Returns:
            A new `LayoutTensor` instance with memory allocated on the stack.

        Performance:

        - Stack allocation is typically faster than heap allocation.
        - Proper alignment can significantly improve memory access performance,
            especially for vectorized operations.
        - No dynamic memory management overhead (no malloc/free calls).

        Notes:

        - Only works with tensors that have fully static layouts known at
            compile time.
        - Stack memory is limited, so this should only be used for reasonably
            sized tensors.
        - The allocated memory is automatically freed when the function returns.
        """

        comptime assert (
            Self.layout.all_dims_known()
        ), "Requires fully static layout"
        comptime assert stack_alignment % Self.alignment == 0, (
            "Stack allocation alignment "
            + String(stack_alignment)
            + " must be multiple of tensor alignment "
            + String(Self.alignment)
        )

        return Self.StackTensorType(
            unsafe_stack_allocation[
                Self.layout.cosize() * Self.element_layout.size(),
                Self.dtype,
                alignment=stack_alignment,
                address_space=Self.address_space,
            ]().as_unsafe_any_origin()
        )

    @staticmethod
    @always_inline("nodebug")
    def null() -> Self.StackTensorType:
        """
        Returns a null `LayoutTensor` object.

        Returns:
            A null `LayoutTensor` object.
        """
        return Self.StackTensorType(None)

    comptime StackTensorType = LayoutTensor[
        Self.dtype,
        Self.layout,
        MutAnyOrigin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
        alignment=Self.alignment,
    ]
    """LayoutTensor type for stack-allocated tensors."""

    @always_inline
    def to_device_buffer(self, ctx: DeviceContext) -> DeviceBuffer[Self.dtype]:
        """Convert the tensor to a `DeviceBuffer`.

        Args:
            ctx: The device context to use.

        Returns:
            A `DeviceBuffer` containing the tensor's data.
        """
        comptime assert (
            Self.address_space == Self.address_space.GENERIC
        ), "DeviceBuffer is only used on GENERIC address space"
        return DeviceBuffer[Self.dtype](
            ctx,
            self.ptr,
            self.size(),
            owning=False,
        )

    @always_inline("nodebug")
    def _stack_copy(
        self,
    ) -> Self.StackTensorType:
        var copy: Self.StackTensorType
        comptime if Self.layout.all_dims_known():
            copy = self.stack_allocation()
        else:
            copy = Self.StackTensorType(
                self.ptr.unsafe_mut_cast[True]().as_unsafe_any_origin(),
                self.runtime_layout,
            )

        def self_value(
            lhs: Self.element_type, rhs: Self.element_type
        ) capturing -> Self.element_type:
            return rhs

        return copy._elementwise_binary_with_broadcast[self_value](self)

    @staticmethod
    @always_inline("nodebug")
    def _to_static[
        t: IntTuple, element_type: DType
    ]() -> IndexList[len(t), element_type=element_type]:
        var st = IndexList[len(t), element_type=element_type]()

        comptime for i in range(len(t)):
            # Use product() to handle both scalar and nested tuples
            st[i] = product(t[i])
        return st

    @staticmethod
    @always_inline("nodebug")
    def _get_rank_stride_offset(rank_idx: Int) -> Int:
        var offset = 0
        for i in range(rank_idx):
            offset += len(flatten(Self.layout.shape[i]))
        return offset

    @staticmethod
    @always_inline("nodebug")
    def _get_rank_offset[
        num_strides: Int, rank: Int, //, rank_idx: Int
    ](stride: IndexList[num_strides, ...], vals: IndexList[rank, ...]) -> Int:
        comptime sub_layout = Self.layout[rank_idx]
        comptime stride_idx = Self._get_rank_stride_offset(rank_idx)

        comptime if len(sub_layout) == 1:
            return stride[stride_idx] * vals[rank_idx]
        return 0

    @staticmethod
    @always_inline("nodebug")
    def _expand_indices(
        ridx: Self.idx_list_t[Self.rank],
    ) -> Self.idx_list_t[Self.num_strides]:
        var eidx = IndexList[
            Self.num_strides, element_type=Self.linear_idx_type
        ]()
        var eidx_offset = 0

        var r: Int
        comptime for rank_idx in range(Self.rank):
            comptime sub_layout = flatten(Self.layout.shape[rank_idx])
            comptime sub_layout_size = len(sub_layout)
            comptime assert sub_layout_size > 0

            comptime if sub_layout_size == 1:
                # not nested
                eidx[eidx_offset] = ridx[rank_idx]
                eidx_offset += 1
            else:
                # map from linear to column-major cartesian indices
                var idx = ridx[rank_idx]

                comptime for i in range(sub_layout_size - 1):
                    comptime sz: Int = sub_layout[i].value()
                    comptime assert sz != UNKNOWN_VALUE, (
                        "unknown shapes not supported in non-trailing"
                        " positions of nested dimensions"
                    )
                    idx, r = divmod(idx, sz)
                    eidx[eidx_offset] = r
                    eidx_offset += 1
                eidx[eidx_offset] = idx
                eidx_offset += 1

        return eidx

    @staticmethod
    @always_inline("nodebug")
    def _get_offset[
        rank: Int,
    ](
        stride: Self.idx_list_t[Self.num_strides],
        vals: Self.idx_list_t[rank],
    ) -> Int:
        comptime assert rank == Self.rank or rank == Self.num_strides, (
            "idx rank = "
            + String(rank)
            + "\nTensor rank = "
            + String(Self.rank)
            + "\nnum_strides = "
            + String(Self.num_strides)
        )

        var offset: Scalar[Self.linear_idx_type] = 0

        var idxs: Self.idx_list_t[Self.num_strides]

        comptime if Self.num_strides == rank:
            idxs = rebind[Self.idx_list_t[Self.num_strides]](vals)
        else:
            idxs = Self._expand_indices(
                rebind[Self.idx_list_t[Self.rank]](vals)
            )

        comptime for i in range(Self.num_strides):
            offset += Scalar[Self.linear_idx_type](idxs[i] * stride[i])
        return Int(offset)

    @always_inline
    @staticmethod
    def is_static_shape[idx: Int]() -> Bool where idx != UNKNOWN_VALUE:
        """Returns the whether the specified dimension is statically known.

        Parameters:
            idx: The dimension index to query (0-based).
                    For example, in a 3D tensor with shape [10, UNKNOWN_VALUE, 30]:
                    - `shape[0]()` returns True (first dimension).
                    - `shape[1]()` returns False (second dimension).
                    - `shape[2]()` returns True (third dimension).

        Returns:
            The True if the dimension is statically known, False otherwise.

        Performance:

        - This is a compile-time operation with no runtime cost when used
            with static dimensions.

        Notes:

        - This is a static method that operates on the tensor's type information,
            not on a specific tensor instance.
        """

        comptime shape = Self._to_static[
            Self.layout.shape, Self.layout_int_type
        ]()
        return shape[idx] != UNKNOWN_VALUE

    @always_inline
    @staticmethod
    def shape[idx: Int]() -> Int where idx != UNKNOWN_VALUE:
        """Returns the size of the tensor along the specified dimension.

        Provides static access to the tensor's shape information. This method
        returns the size of a specific dimension without requiring an instance
        of the tensor, as the shape is part of the tensor's static type
        information.

        Parameters:
            idx: The dimension index to query (0-based).
                 For example, in a 3D tensor with shape [10, 20, 30]:
                 - `shape[0]()` returns 10 (first dimension).
                 - `shape[1]()` returns 20 (second dimension).
                 - `shape[2]()` returns 30 (third dimension).

        Returns:
            The size of the tensor along the specified dimension as an integer.

        Performance:

        - This is a compile-time operation with no runtime cost when used
            with static dimensions.

        Notes:

        - This is a static method that operates on the tensor's type information,
            not on a specific tensor instance.
        """

        comptime shape = Self._to_static[
            Self.layout.shape, Self.layout_int_type
        ]()
        return shape[idx]

    @always_inline("nodebug")
    def get_shape(self) -> IndexList[Self.rank]:
        """Get the flattened shape of a LayoutTensor.

        Returns:
           The flattened shape of a LayoutTensor.
        """
        return rebind[IndexList[Self.rank]](
            self.runtime_layout.shape.value.canonicalize()
        )

    @always_inline("nodebug")
    def get_stride(self) -> IndexList[Self.rank]:
        """Get the flattened stride of a LayoutTensor.

        Returns:
           The flattened shape of a LayoutTensor.
        """
        return rebind[IndexList[Self.rank]](
            self.runtime_layout.stride.value.canonicalize()
        )

    @always_inline
    @staticmethod
    def stride[idx: Int]() -> Int where idx != UNKNOWN_VALUE:
        """Returns the memory stride of the tensor along the specified
        dimension.

        Provides static access to the tensor's stride information. The stride
        represents the number of elements to skip in memory to move one position
        along a particular dimension. This method returns the stride without
        requiring an instance of the tensor, as the stride is part of the
        tensor's static type information.

        Parameters:
            idx: The dimension index to query (0-based).
                 For example, in a 2D tensor with shape [10, 20] and row-major
                 layout:
                 - `stride[0]()` might return 20 (moving one row requires
                   skipping 20 elements).
                 - `stride[1]()` might return 1 (moving one column requires
                   skipping 1 element).

        Returns:
            The memory stride of the tensor along the specified dimension as an
            integer.

        Performance:

        - This is a compile-time operation with no runtime cost when used
            with static dimensions.
        - Understanding stride patterns is crucial for optimizing memory access
            patterns in performance-critical code.

        Notes:

        - Strides depend on the memory layout (row-major, column-major, or
            custom).
        - For non-contiguous tensors (e.g., tensor slices), strides may not
            follow a simple pattern.
        """

        comptime stride = Self._to_static[
            Self.layout.stride, Self.linear_idx_type
        ]()
        return stride[idx]

    @always_inline
    def dim(self, idx: Int) -> Int:
        """Returns the runtime dimension size of the tensor along the specified
        axis.

        Unlike the static `dim` method, this instance method takes a runtime
        dimension index.

        Args:
            idx: The dimension index to query (0-based).
                 For example, in a 3D tensor with shape `[10, 20, 30]`:
                 - `dim(0)` returns 10 (first dimension).
                 - `dim(1)` returns 20 (second dimension).
                 - `dim(2)` returns 30 (third dimension).

        Returns:
            The dimension of the tensor along the specified axis as an integer.
        """

        comptime assert depth(Self.layout.shape) in (0, 1), String(
            (
                "This method only works with tensors that have depth-1"
                " layouts (no nested shapes). Received: "
            ),
            Self.layout,
        )

        return self.runtime_layout.shape.value[idx]

    @always_inline
    def stride(self, idx: Int) -> Int:
        """Returns the runtime stride of the tensor along the specified
        axis.

        Unlike the static `stride` method, this instance method takes a runtime
        dimension index.

        Args:
            idx: The dimension index to query (0-based).
                 For example, in a row-major 3D tensor with shape `[10, 20, 30]`:
                 - `stride(0)` returns 600 (first dimension).
                 - `stride(1)` returns 30 (second dimension).
                 - `stride(2)` returns 1 (third dimension).

        Returns:
            The dimension of the tensor along the specified axis as an integer.
        """

        comptime assert 0 <= depth(Self.layout.stride) <= 1, String(
            (
                "This method only works with tensors that have depth-1"
                " layouts (no nested shapes). Received: "
            ),
            Self.layout,
        )

        return self.runtime_layout.stride.value[idx]

    @always_inline
    def dim[idx: Int](self) -> Int:
        """Returns the dimension size of the tensor along the specified
        axis.

        Unlike the static `shape` method, this instance method provides access
        to the tensor's actual dimension sizes. If the dimension is unknown,
        the runtime layout is used to get the dimension size.

        Parameters:
            idx: The dimension index to query (0-based).
                 For example, in a 3D tensor with shape `[10, 20, 30]`:
                 - `dim[0]()` returns 10 (first dimension).
                 - `dim[1]()` returns 20 (second dimension).
                 - `dim[2]()` returns 30 (third dimension).

        Constraints:
            - Only works with tensors that have depth-1 layouts (no nested
                shapes).

        Returns:
            The size of the tensor along the specified dimension as an integer.

        Performance:

        - For static dimensions known at compile time, prefer the static
            `shape` method when possible for better performance.

        Notes:

        - This method works with both static and dynamic dimensions.
        - For tensors with masked or partial views, this returns the actual
            size of the view, not the original tensor.
        """
        comptime assert depth(Self.layout.shape) in (0, 1), String(
            (
                "This method only works with tensors that have depth-1"
                " layouts (no nested shapes). Received: "
            ),
            Self.layout,
        )

        comptime shape = Self._to_static[
            Self.layout.shape, Self.layout_int_type
        ]()

        comptime if not Self.layout.shape[idx].all_known() or Self.masked:
            return self.runtime_layout.shape.value[idx]
        else:
            return shape[idx]

    comptime CoalesceType[element_layout: Layout] = LayoutTensor[
        Self.dtype,
        coalesce(Self.layout),
        Self.origin,
        address_space=Self.address_space,
        element_layout=element_layout,
    ]
    """Type alias for coalesced result tensors.

    Parameters:
        element_layout: The element layout for the coalesced tensor.
    """

    @always_inline
    def coalesce(self) -> Self.CoalesceType[Self.element_layout]:
        """Creates a tensor with a coalesced memory layout from this tensor.

        Coalescing a tensor's layout means reorganizing its memory
        representation to be as contiguous as possible, which can improve memory
        access patterns and performance. This operation does not move or copy
        data; it only changes how the same memory is interpreted.

        Returns:
            A tensor with the same data but with a coalesced memory layout.
            The returned tensor has type `LayoutTensor` with the same dtype but
            with a coalesced layout.

        Performance:

        - Coalesced layouts typically provide better cache utilization and
            memory access patterns.
        - This operation is zero-cost at runtime as it only changes the
            layout information, not the actual data.
        - Particularly beneficial before operations that perform sequential
            memory access or vectorized operations.

        Notes:

        - The coalesced tensor shares the same memory as the original tensor,
            so modifications to one will affect the other.
        - The shape of the tensor remains the same, only the stride information
            is optimized.
        - For already optimally coalesced tensors, this operation has no effect.
        """
        return Self.CoalesceType[Self.element_layout](self.ptr)

    @staticmethod
    def _compute_tile_layout[*tile_sizes: Int]() -> Layout:
        return Self._divide_tiles[*tile_sizes]()

    @staticmethod
    def _divide_tiles[*tile_sizes: Int]() -> Layout:
        comptime tiler = MakeTileLayoutList[*tile_sizes]()
        return zipped_divide(materialize[Self.layout](), materialize[tiler]())

    @staticmethod
    def _fast_varying_dim_tiler(shape: Int) -> Layout:
        var flat_stride = flatten(Self.layout.stride)
        var min_stride = Int.MAX
        var min_idx = -1
        for i in range(len(flat_stride)):
            var s = flat_stride[i]
            if s == UNKNOWN_VALUE:
                continue
            if s < min_stride:
                min_stride = Int(flat_stride[i])
                min_idx = i
        if min_stride != 1:
            abort(
                "Linear vectorization is limited to tensors with a contiguous"
                " dimension"
            )
        if min_idx == -1:
            abort("No known stride found to vectorize!")
        var flat_tiler_shape = IntTuple()
        for i in range(len(flat_stride)):
            if i == min_idx:
                flat_tiler_shape.append(shape)
            else:
                flat_tiler_shape.append(Int(1))
        var tiler_shape = to_nest(Self.layout.stride, flat_tiler_shape)
        var unit_stride = fill_like(Self.layout.shape, 1)
        return Layout(tiler_shape, unit_stride)

    @staticmethod
    def _tuple_divide_tiler(
        shape: IntTuple, linear_vectorize: Bool = False
    ) -> Layout:
        if is_int(shape):
            if linear_vectorize:
                # If the shape is a single int, and we are vectorizing wrt the
                # linear indexing. We should then vectorize the fastest varying
                # dimension.
                return Self._fast_varying_dim_tiler(Int(shape))
            else:
                # If the shape is a single int, we need to use the LayoutList
                # dispatch
                return Layout(shape, 1)
        else:
            # Otherwise, the shape should be compatible, so we can use the
            # nested layout dispatch
            var tiler_stride = fill_like(shape, 1)
            return Layout(shape, tiler_stride)

    @staticmethod
    def _tuple_divide_tiles(
        shape: IntTuple, linear_vectorize: Bool = False
    ) -> Layout:
        var tiler = Self._tuple_divide_tiler(shape, linear_vectorize)
        if is_int(shape) and not linear_vectorize:
            # legacy behavior
            return zipped_divide(materialize[Self.layout](), LayoutList(tiler))
        return zipped_divide(materialize[Self.layout](), tiler)

    @staticmethod
    @always_inline
    def _prop_unknown_shape[idx: Int, src: Layout, target: Layout]() -> Layout:
        """Propagate all unknown dim from target to a new layout."""
        var new_shape = src.shape
        # new_shape[idx] = propagate_unknown(src.shape[idx], target.shape)
        new_shape = new_shape.replace_entry(
            idx, propagate_unknown(src.shape[idx], target.shape)
        )
        return Layout(new_shape, src.stride)

    @staticmethod
    def _compute_tile_layout[*, tile_size: Int, axis: Int]() -> Layout:
        var tiler = LayoutList()
        var i = 0
        for dim in Self.layout.shape:
            if i == axis:
                tiler.append(Layout(tile_size))
            else:
                tiler.append(Layout(dim))
            i += 1
        return zipped_divide(materialize[Self.layout](), tiler)

    comptime TileType[*tile_sizes: Int] = LayoutTensor[
        Self.dtype,
        Self._compute_tile_layout[*tile_sizes]()[0],
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked or _tile_is_masked[Self.layout, *tile_sizes](),
        alignment=Self.alignment,
    ]
    """The tile type returned by the `tile()` method given
    the specified set of tile sizes.

    Parameters:
        tile_sizes: The dimensions of each tile along each axis of the
            tensor.
    """

    @always_inline
    def tile[
        *tile_sizes: Int
    ](self, *tile_coords: Int) -> self.TileType[*tile_sizes]:
        """Extract a tile (sub-tensor) from this tensor with specified
        dimensions and position.

        Tiling is a fundamental operation for high-performance tensor
        computations that divides a tensor into smaller blocks for better cache
        locality and parallelism. This method extracts a specific tile at the
        given coordinates without copying data.

        Parameters:
            tile_sizes: The dimensions of each tile along each axis of the
                tensor. For example, in a 2D tensor, `tile[32, 32]` creates
                32x32 tiles.

        Args:
            tile_coords: The coordinates of the specific tile to extract. For
                example, `tile[32, 32](1, 2)` extracts the tile at position
                (1, 2) in the grid of 32x32 tiles.

        Returns:
            A view into the original tensor representing the specified tile.

        Example:

        For a 4x4 tensor with values:

        ```
        [1 2 3 4]
        [2 3 4 5]
        [5 4 3 2]
        [1 1 1 1]
        ```

        `tile[2, 2](1, 0)` will extract the tile:

        ```
        [5 4]
        [1 1]
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Optimized for both static and dynamic layouts with different code paths.
        - Properly handles edge cases where tiles may be partially outside the tensor.
        - Maintains stride information for efficient memory access within the tile.

        Notes:

        - The resulting tile is a view into the original tensor, so modifications
            to the tile will affect the original tensor.
        - For tiles at the edges of the tensor, the actual dimensions may be smaller
            than the requested tile_sizes if masking is enabled.
        - The implementation automatically selects between static and dynamic tiling
            based on the tensor's layout properties.
        """

        comptime num_tiles = tile_sizes.size

        # need to calculate this again because _tiled_layout[1] is required for the offset calculation
        comptime _tiled_layout = Self._compute_tile_layout[*tile_sizes]()

        comptime assert (
            _tiled_layout[1].rank() == num_tiles
        ), "Number of tiles should match the rank"

        comptime tile_type = self.TileType[*tile_sizes]

        var offset = 0
        var runtime_shape = tile_type.RuntimeLayoutType.ShapeType()
        var runtime_stride = tile_type.RuntimeLayoutType.StrideType()

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        comptime if tile_type.layout.all_dims_known():
            comptime for i in range(num_tiles):
                comptime stride = product(_tiled_layout[1].stride[i])
                offset += tile_coords[i] * stride

            var runtime_layout = tile_type.RuntimeLayoutType(
                runtime_shape, runtime_stride
            )

            # Adjust runtime layout, so the shape is clipped to the unmasked sizes.
            comptime if tile_type.masked:
                comptime for i in range(tile_type.layout.rank()):
                    var cur_dim = self.dim[i]() - (
                        tile_coords[i] * tile_sizes[i]
                    )
                    var shape_i = max(min(tile_sizes[i], cur_dim), 0)
                    runtime_layout.shape.value[i] = shape_i

            return tile_type(self.ptr + offset, runtime_layout)

        else:
            # Dynamic layout, use strides

            comptime for i in range(num_tiles):
                var stride = self.runtime_layout.stride.value[i] * tile_sizes[i]
                runtime_stride.value[i] = self.runtime_layout.stride.value[i]
                offset += tile_coords[i] * stride

            var runtime_layout = tile_type.RuntimeLayoutType(
                runtime_shape, runtime_stride
            )

            # Adjusts the runtime layout so that the shape is clipped to the unmasked sizes.
            comptime for i in range(tile_type.layout.rank()):
                var cur_dim = self.dim[i]() - (tile_coords[i] * tile_sizes[i])
                var shape_i = max(min(tile_sizes[i], cur_dim), 0)
                runtime_layout.shape.value[i] = shape_i

            return tile_type(self.ptr + offset, runtime_layout)

    comptime SIMDTileType[tile_size: Int] = Self.TileType[
        tile_size, simd_width_of[Self.dtype]()
    ]
    """Type alias for SIMD-sized tile tensors.

    Parameters:
        tile_size: The size of the tile along the tiled axis.
    """

    @always_inline
    def simd_tile[
        tile_size: Int
    ](self, tile_idx: Int) -> Self.SIMDTileType[tile_size]:
        """Return a SIMD[dtype] sized tile of size `tile_size` at `tile_idx`.

        Parameters:
            tile_size: The size of the tile along the tiled axis used for
                vectorization.

        Args:
            tile_idx: The index of the tile to extract along the tiled axis.

        Returns:
            A SIMD[dtype] tile of size `tile_size` at `tile_idx`
        """
        return self.tile[tile_size, simd_width_of[Self.dtype]()](tile_idx)

    comptime CornerCoordsType = IndexList[
        len(flatten(Self.layout.shape)),
        element_type=Self.layout_int_type,
    ]
    """Index list type for corner coordinates."""

    @always_inline
    def tile_with_offset[
        *tile_sizes: Int,
    ](
        self,
        *tile_coords: Int,
    ) -> Tuple[
        Self.TileType[*tile_sizes],
        Self.CornerCoordsType,
        Scalar[Self.linear_idx_type],
    ]:
        """Similar to `tile`, but also returns the corner coordinates of the
        tile as well as the offset.

        Parameters:
            tile_sizes: The dimensions of each tile along each axis of the
                tensor.

        Args:
            tile_coords: The coordinates of the specific tile to extract.

        Returns:
            A tuple containing:
                - The extracted tile as a `LayoutTensor`.
                - The corner coordinates of the tile.
                - The offset of the tile.
        """
        comptime num_tiles = tile_sizes.size

        # need to calculate this again because _tiled_layout[1] is required for the offset calculation
        comptime _tiled_layout = Self._compute_tile_layout[*tile_sizes]()

        comptime assert (
            _tiled_layout[1].rank() == num_tiles
        ), "Number of tiles should match the rank"

        comptime tile_type = self.TileType[*tile_sizes]

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        var corner_coords = IndexList[
            len(flatten(self.layout.shape)), element_type=Self.layout_int_type
        ]()
        var offset: Scalar[Self.linear_idx_type] = 0
        var runtime_shape = tile_type.RuntimeLayoutType.ShapeType()
        var runtime_stride = tile_type.RuntimeLayoutType.StrideType()

        comptime if tile_type.layout.all_dims_known():
            comptime for i in range(num_tiles):
                comptime stride = Int(_tiled_layout[1].stride[i])
                offset += Scalar[Self.linear_idx_type](tile_coords[i] * stride)
                corner_coords[i] = tile_coords[i] * tile_sizes[i]

            var runtime_layout = tile_type.RuntimeLayoutType(
                runtime_shape, runtime_stride
            )

            # Adjust runtime layout, so the shape is clipped to the unmasked sizes.
            comptime if tile_type.masked:
                comptime for i in range(tile_type.layout.rank()):
                    var cur_dim = self.dim[i]() - (
                        tile_coords[i] * tile_sizes[i]
                    )
                    var shape_i = max(min(tile_sizes[i], cur_dim), 0)
                    runtime_layout.shape.value[i] = shape_i

            return (
                tile_type(self.ptr + offset, runtime_layout),
                corner_coords,
                offset,
            )

        else:
            # Dynamic layout, use strides
            comptime for i in range(num_tiles):
                var corner_coord = tile_coords[i] * tile_sizes[i]
                corner_coords[i] = corner_coord
                runtime_stride.value[i] = self.runtime_layout.stride.value[i]
                offset += Scalar[Self.linear_idx_type](
                    self.runtime_layout.stride.value[i] * corner_coord
                )

            var runtime_layout = tile_type.RuntimeLayoutType(
                runtime_shape, runtime_stride
            )

            # Adjusts the runtime layout so that the shape is clipped to the unmasked sizes.
            comptime for i in range(tile_type.layout.rank()):
                var cur_dim = self.dim[i]() - (tile_coords[i] * tile_sizes[i])
                var shape_i = max(min(tile_sizes[i], cur_dim), 0)
                runtime_layout.shape.value[i] = shape_i

            return (
                tile_type(self.ptr + offset, runtime_layout),
                corner_coords,
                offset,
            )

    comptime TiledIteratorType[
        *tile_sizes: Int,
        axis: Int = 0,
    ] = LayoutTensorIter[
        Self.dtype,
        Self._compute_tile_layout[*tile_sizes]()[0],
        Self.origin,
        address_space=Self.address_space,
        circular=False,
        axis=axis,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked or _tile_is_masked[Self.layout, *tile_sizes](),
    ]
    """Type alias for tiled iterator types.

    Parameters:
        tile_sizes: The dimensions of each tile along each axis.
        axis: The axis along which to iterate.
    """

    @always_inline
    def tiled_iterator[
        *tile_sizes: Int,
        axis: Int = 0,
    ](self, *tile_coords: Int) -> Self.TiledIteratorType[
        *tile_sizes, axis=axis
    ]:
        """Create an iterator that traverses tiles along a specified axis.

        This method creates an iterator that allows efficient traversal of tiles
        within a tensor. The iterator starts at the specified tile coordinates
        and can move along the specified axis, providing access to consecutive
        tiles.

        Parameters:
            tile_sizes: The dimensions of each tile along each axis of the
                tensor. For example, in a 2D tensor, `tiled_iterator[32, 32]`
                creates an iterator over 32x32 tiles.
            axis: The axis along which the iterator will traverse. Default is 0
                (first dimension). For example, with axis=0, the iterator will
                move vertically through tiles.

        Args:
            tile_coords: The starting coordinates of the tile where iteration
                begins.

        Returns:
            A `LayoutTensorIter` that can be used to traverse tiles along the
                specified axis.

        Performance:

        - Provides efficient sequential access to tiles with good cache
            locality.
        - Optimized for both static and dynamic layouts with different code
            paths.
        - Maintains stride information for efficient memory access within each
            tile.
        - Properly handles edge cases where tiles may be partially outside the
            tensor.

        Notes:

        - The iterator provides views into the original tensor, so modifications
            through the iterator will affect the original tensor.
        - For tiles at the edges of the tensor, the actual dimensions may be smaller
            than the requested tile_sizes if masking is enabled.
        - The iterator is not circular by default, meaning it will not wrap around
            when reaching the end of the tensor along the iteration axis.
        - The implementation automatically selects between static and dynamic tiling
            based on the tensor's layout properties.

        Example:

        ```mojo
        var iter = tensor.tiled_iterator[16, 16, axis=0](0, 0)
        for i in range(num_tiles_along_axis):
            var tile = iter.get()
            # Process tile
            iter.next()
        ```
        """

        comptime tiles_rank = tile_sizes.size
        comptime __tiled_layout = Self._compute_tile_layout[*tile_sizes]()
        comptime assert (
            __tiled_layout[1].rank() == tiles_rank
        ), "Number of tiles should match the rank"

        comptime tiled_iterator_type = Self.TiledIteratorType[
            *tile_sizes, axis=axis
        ]

        var ptr_offset = 0

        comptime if Self.layout.all_dims_known():
            var runtime_shape = (
                tiled_iterator_type.RuntimeLayoutType.ShapeType()
            )
            var runtime_stride = (
                tiled_iterator_type.RuntimeLayoutType.StrideType()
            )

            comptime for i in range(tiles_rank):
                comptime stride = Int(__tiled_layout[1].stride[i])
                ptr_offset += tile_coords[i] * stride

            # fmt: off

            # A nested LayoutTensor may have shape=(16, 64) and stride=(1, 16)
            # In order to calculate the bound we only need to use the last
            # element in the IntTuple.
            comptime is_axis_val = Self.layout.shape[axis].is_value()
            comptime axis_shape = Self.layout.shape[axis]
            comptime axis_stride = Self.layout.stride[axis]
            comptime bound = axis_shape.value() * axis_stride.value() \
                if is_axis_val \
                else axis_shape[len(axis_shape) - 1].value() * axis_stride[len(axis_stride) - 1].value()
            comptime assert axis != UNKNOWN_VALUE
            comptime dim_bound = Self.shape[axis]() \
                if is_axis_val \
                else product(Self.layout.shape[axis])
            comptime stride = __tiled_layout[1].stride[axis].value()
            # fmt: on

            comptime if tiled_iterator_type.masked:
                comptime for i in range(tiled_iterator_type.layout.rank()):
                    var cur_dim = self.dim[i]() - (
                        tile_coords[i] * tile_sizes[i]
                    )
                    var shape_i = max(min(tile_sizes[i], cur_dim), 0)
                    runtime_shape.value[i] = shape_i

                return tiled_iterator_type(
                    self.ptr + ptr_offset,
                    tiled_iterator_type.linear_uint_type(bound),
                    tiled_iterator_type.RuntimeLayoutType(
                        runtime_shape, runtime_stride
                    ),
                    stride=tiled_iterator_type.linear_uint_type(stride),
                    offset=0,
                    dimension_bound=tiled_iterator_type.layout_uint_type(
                        dim_bound
                    ),
                    idx=tiled_iterator_type.linear_uint_type(tile_coords[axis]),
                )
            else:
                return tiled_iterator_type(
                    self.ptr + ptr_offset,
                    tiled_iterator_type.linear_uint_type(bound),
                    stride=tiled_iterator_type.linear_uint_type(stride),
                    offset=0,
                )

        else:
            var runtime_shape = (
                tiled_iterator_type.RuntimeLayoutType.ShapeType()
            )
            var runtime_stride = (
                tiled_iterator_type.RuntimeLayoutType.StrideType()
            )

            comptime for i in range(tiles_rank):
                var stride = self.runtime_layout.stride.value[i] * tile_sizes[i]
                runtime_stride.value[i] = self.runtime_layout.stride.value[i]
                ptr_offset += tile_coords[i] * stride

            var axis_dim = self.runtime_layout.shape.value[axis]
            var axis_stride = self.runtime_layout.stride.value[axis]
            var iter_bound = axis_dim * axis_stride
            var iter_stride = tile_sizes[axis] * axis_stride

            comptime for i in range(tiled_iterator_type.layout.rank()):
                var cur_dim = self.dim[i]() - (tile_coords[i] * tile_sizes[i])
                var shape_i = max(min(tile_sizes[i], cur_dim), 0)
                runtime_shape.value[i] = shape_i

            return tiled_iterator_type(
                self.ptr + ptr_offset,
                tiled_iterator_type.linear_uint_type(iter_bound),
                stride=tiled_iterator_type.linear_uint_type(iter_stride),
                offset=0,
                runtime_layout=tiled_iterator_type.RuntimeLayoutType(
                    runtime_shape, runtime_stride
                ),
                dimension_bound=tiled_iterator_type.layout_uint_type(
                    self.dim[axis]()
                ),
                idx=tiled_iterator_type.linear_uint_type(tile_coords[axis]),
            )

    comptime SplitElementType[
        count: Int,
        axis: Int = 0,
    ] = LayoutTensor[
        Self.dtype,
        Self._compute_tile_layout[
            tile_size=Self.layout.shape[axis].value() // count, axis=axis
        ]()[0],
        # Splitting inherently introduces mutable aliases of the same origin -
        # each chunk won't overlap, but the origin can't indicate that.
        AnyOrigin[mut=Self.mut],
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        alignment=Self.alignment,
    ]
    """Type alias for split element tensors.

    Parameters:
        count: Number of portions to split into.
        axis: The axis along which to split.
    """

    comptime StaticSplitType[
        count: Int,
        axis: Int = 0,
    ] = StaticTuple[
        Self.SplitElementType[count, axis],
        count,
    ]
    """Type alias for static split result tuples.

    Parameters:
        count: Number of portions to split into.
        axis: The axis along which to split.
    """

    @always_inline
    def split[
        count: Int,
        axis: Int = 0,
    ](self) -> Self.StaticSplitType[count, axis]:
        """Split the `LayoutTensor` along a axis and return a `StaticTuple` of
        `LayoutTensor`.

        Parameters:
            count: Number of portion to split.
            axis: The axis where the split is applied to.

        Returns:
            A `StaticTuple` containing `count` `LayoutTensors`, each
            representing an equal-sized partition of the original tensor along
            the specified axis. Each partition has the same data type and memory
            characteristics as the original tensor, but with a reduced size
            along the split axis.
        """

        comptime assert Self.layout.shape[
            axis
        ].is_value(), "Only support partition modes that are plain values."

        comptime assert (
            Self.layout.shape[axis].value() % count == 0
        ), "The input dimension must be divisible over the input count."

        comptime stride = Self.layout.stride[axis].value()
        var tiles = Self.StaticSplitType[count, axis]()

        # Safety: this is to turn off the mutable aliasing origin check, so we
        # can have multiple LayoutTensors using the same pointer/origin. We've
        # ensured that we're not overlapping any of the pointers.
        var ptr = self.ptr.unsafe_origin_cast[AnyOrigin[mut=Self.mut]]()
        comptime for i in range(count):
            # Need tile_size alias to ensure that the ptr passed to LayoutTensor is
            # known at compile time. Otherwise we get compile time failure.
            # The compiler can't allocate LayoutTensor on stack if ptr is not known at compile time.
            # See MOCO-1081 for more details.
            comptime tile_size = Self.layout.shape[axis].value() // count
            tiles[i] = LayoutTensor[
                Self.dtype,
                Self._compute_tile_layout[
                    tile_size=Self.layout.shape[axis].value() // count,
                    axis=axis,
                ]()[0],
                address_space=Self.address_space,
                element_layout=Self.element_layout,
                alignment=Self.alignment,
            ](ptr + i * tile_size * stride)

        return tiles

    comptime DynamicSplitType[
        axis: Int = 0,
    ] = LayoutTensor[
        Self.dtype,
        Self.layout.make_shape_unknown[axis](),
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for dynamic split result tensors.

    Parameters:
        axis: The axis along which to split.
    """

    @always_inline
    def split[
        axis: Int = 0,
        split_alignment: Int = 1,
    ](self, count: Int, idx: Int) -> Self.DynamicSplitType[axis]:
        """Retrieve a specific partition of the tensor after splitting along a
        specified axis.

        This method divides the tensor into 'count' partitions along the
        specified axis and returns the partition at index 'idx'. The
        partitioning is done with alignment considerations to optimize memory
        access patterns.

        Unlike the overloaded split method that returns all partitions, this
        method returns only a single partition, making it more memory-efficient
        for cases where only one partition is needed at a time.

        Constraints:
            - The dimension being split must have a statically known size.
            - Cannot split dimensions with unknown or dynamic sizes.

        Parameters:
            axis: The axis along which to split the tensor. Defaults to 0 (first
                dimension).
            split_alignment: Memory alignment value for the partition size. Defaults
                to 1.

        Args:
            count: The number of partitions to divide the tensor into.
            idx: The index of the partition to return (0-based).

        Returns:
            A `LayoutTensor` representing the requested partition.

        Notes:

        - The shape along the split axis becomes unknown at compile time.
        - Only works with dimensions that have statically known sizes.
        - The last partition may be smaller than others if the dimension size
            is not evenly divisible by `count`.
        - Partition sizes are aligned up to the specified alignment value,
            which can improve performance for vectorized operations.

        Performance:

        - Uses aligned partitioning to improve memory access patterns.
        - Avoids creating all partitions in memory, reducing memory usage.
        - Maintains the original tensor's stride information for efficient
            element access within the partition.
        """
        comptime assert Self.layout.shape[
            axis
        ].is_value(), "Can't split non-scalar dimension."

        # We can split dynamic dimension but that should be audited carefully with
        # other parts when we really want to support arbitrary K, N in matmul.
        # Restrict to static case for now.
        comptime assert (
            Self.layout.shape[axis].value() != UNKNOWN_VALUE
            and Self.layout.stride[axis].value() != UNKNOWN_VALUE
        ), "Shouldn't split dynamic dimension."

        comptime axis_dim = Self.layout.shape[axis].value()
        comptime axis_stride = Self.layout.stride[axis].value()
        comptime flatten_rank = len(flatten(Self.layout.shape))
        comptime axis_in_flatten_tuple = runtime_shape.offset_until[axis]()

        var runtime_shape = Self.DynamicSplitType[
            axis
        ].RuntimeLayoutType.ShapeType()
        var axis_partition_dim = align_up(axis_dim // count, split_alignment)

        comptime for i in range(flatten_rank):
            var shape_i = self.runtime_layout.shape.value[i]

            comptime if i == axis_in_flatten_tuple:
                runtime_shape.value[i] = min(
                    axis_partition_dim, shape_i - idx * axis_partition_dim
                )
            else:
                runtime_shape.value[i] = shape_i

        return Self.DynamicSplitType[axis](
            # Only the last partition can have size other than axis_partition_dim.
            self.ptr + idx * axis_partition_dim * axis_stride,
            Self.DynamicSplitType[axis].RuntimeLayoutType(
                runtime_shape,
                rebind[
                    Self.DynamicSplitType[axis].RuntimeLayoutType.StrideType
                ](self.runtime_layout.stride),
            ),
        )

    @always_inline
    def _clamp_distribute_shape[
        thread_layout: Layout,
    ](self, thread_id: Int) -> IndexList[
        Self.rank, element_type=Self.layout_int_type
    ]:
        comptime assert (
            len(flatten(thread_layout.shape)) <= 2
            and len(flatten(thread_layout.stride)) <= 2
        ), "Only supporting rank-2 or less thread layout for dynamic tile."

        # clamp IndexList using thread_id and thread_layout
        var tile_shape = IndexList[
            Self.rank, element_type=Self.layout_int_type
        ]()
        comptime thread_shape = thread_layout.shape
        comptime thread_stride = thread_layout.stride

        # this would only work for rank-2 thread layout, need to extend this
        # to support thread layout such as Layout((2, 2), 2)
        comptime for i in range(Self.rank):
            comptime thread_stride_i = Int(thread_stride[i])
            comptime thread_shape_i = Int(thread_shape[i])
            var tile_idx = umod(
                ufloordiv(thread_id, thread_stride_i),
                thread_shape_i,
            )
            var tile_shape_i = ceildiv(self.dim[i](), thread_shape_i)
            var bound_i = (tile_shape_i - 1) * thread_shape_i + tile_idx
            tile_shape[i] = min(self.dim[i]() - bound_i, tile_shape_i)

        return tile_shape

    comptime DistributeType[
        threads_layout: Layout,
        axis: Optional[Int] = None,
    ] = LayoutTensor[
        Self.dtype,
        _compute_distribute_layout[
            Self.layout,
            threads_layout,
            axis,
        ]()[1],
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        # TODO: This is a workaround as we don't need masking support for AMD GPU
        # if we use buffer stores and loads. Probably need a better solution
        # in the long term, if someone ends up using global loads and stores
        # it may lead to out of bounds access.
        masked=(
            Self.masked
            or _distribute_is_masked[Self.layout, threads_layout, axis]()
        ) if is_nvidia_gpu() else False,
    ]
    """Type alias for distributed tensor types.

    Parameters:
        threads_layout: The layout describing thread distribution.
        axis: Optional axis to distribute along.
    """

    @always_inline
    def distribute[
        threads_layout: Layout,
        axis: Optional[Int] = None,
        swizzle: Optional[Swizzle] = None,
        submode_axis: Optional[Int] = None,
    ](self, thread_id: Int) -> Self.DistributeType[threads_layout, axis]:
        """Distribute tensor workload across multiple threads in a structured
        pattern.

        This method partitions a tensor across multiple threads for parallel
        processing, assigning each thread a specific portion of the tensor. The
        distribution pattern is determined by the threads_layout parameter,
        which defines the logical arrangement of threads.

        Constraints:
            - For dynamic layouts, the shape must be known at runtime and the
                threads_layout must be fully static.

        Parameters:
            threads_layout: Defines the logical arrangement of threads (e.g.,
                2x2 grid of 4 threads). This layout determines how the tensor is
                partitioned.
            axis: Optional. If specified, restricts distribution to only this
                axis. For example, with axis=0 in a 2D thread layout, threads
                that differ only in their second coordinate will receive the
                same data.
            swizzle: Optional. A function that remaps the distribution pattern
                to improve memory access patterns or cache locality.
            submode_axis: Optional. Specifies an axis for specialized
                distribution modes.

        Args:
            thread_id: The ID of the current thread (0-based).

        Returns:
            A view into the original tensor representing the portion assigned to
            this thread.

        Example:

        For a 4x4 row-major tensor distributed across 4 threads in a 2x2 row-major grid:

        - Thread 0 will receive a LayoutTensor with a view into
            (0,0), (0,2), (2,0), (2,2) of the original tensor.
        - Thread 1 will receive a LayoutTensor with a view into
            (0,1), (0,3), (2,1), (2,3) of the original tensor.
        - Thread 2 will receive a LayoutTensor with a view into
            (1,0), (1,2), (3,0), (3,2) of the original tensor.
        - Thread 3 will receive a LayoutTensor with a view into
            (1,1), (1,3), (3,1), (3,3) of the original tensor.

        If axis=0 is specified with the same setup:

        - Thread (0, 0) and Thread (0, 1) would get the same data (top half)
        - Thread (1, 0) and Thread (1, 1) would get the same data (bottom half)

        Performance:

        - Creates a view without copying data, making it very efficient for
            parallel processing.
        - The swizzle parameter can significantly improve cache locality and
            memory access patterns.
        - Optimized for both static and dynamic layouts with different code
            paths.

        Notes:

        - The resulting tensor is a view into the original tensor, so
            modifications will affect the original tensor.
        - For optimal performance, the `threads_layout` should match the
            hardware's thread organization (e.g., warp/wavefront size and shape).
        - When using swizzling, carefully consider the memory access patterns to
            avoid cache thrashing or bank conflicts.
        - This function is particularly useful for GPU programming where threads
            are organized in structured grids.
        """

        comptime distribute_type = Self.DistributeType[threads_layout, axis]
        comptime runtime_layout_type = distribute_type.RuntimeLayoutType
        comptime runtime_shape_type = runtime_layout_type.ShapeType
        comptime runtime_stride_type = runtime_layout_type.StrideType

        comptime distributed_layout = _compute_distribute_layout[
            Self.layout,
            threads_layout,
            axis,
        ]()

        var runtime_shape: runtime_shape_type

        comptime if distribute_type.masked:
            runtime_shape = runtime_shape_type(
                self._clamp_distribute_shape[threads_layout](thread_id)
            )
        else:
            runtime_shape = runtime_shape_type()

        var runtime_stride = runtime_stride_type()

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        comptime if Self.layout.all_dims_known():
            comptime fragments_layout_stride = flatten(
                distributed_layout[0].stride
            )

            # Only extract coordinates in the given axis.
            # Example: axis = 0 for 2x2 threads, we only need thread 0 and 1's
            # coordinates since thread 2 and 3 are getting the same tile.
            comptime thread_projected_stride = flatten(
                threads_layout.stride[
                    axis.value()
                ] if axis else threads_layout.stride
            )
            comptime thread_projected_shape = flatten(
                threads_layout.shape[
                    axis.value()
                ] if axis else threads_layout.shape
            )

            var offset: Scalar[Self.linear_idx_type] = 0

            comptime for i in range(len(fragments_layout_stride)):
                comptime fragments_stride_i = Int(fragments_layout_stride[i])
                comptime shape_i = Int(thread_projected_shape[i])
                comptime stride_i = Int(thread_projected_stride[i])
                var thread_coord_i = umod(
                    ufloordiv(thread_id, stride_i), shape_i
                )
                offset += Scalar[Self.linear_idx_type](
                    thread_coord_i * fragments_stride_i
                )

            # Swizzling applies to the index of elements rather than scalars because
            # the former is the unit in distribution.
            var swizzled_offset = offset

            comptime if swizzle:
                comptime swizzle_fn = swizzle.value()
                swizzled_offset = swizzle_fn(
                    offset // Scalar[Self.linear_idx_type](self.element_size)
                ) * Scalar[Self.linear_idx_type](self.element_size)

            comptime if distribute_type.masked:
                return distribute_type(
                    self.ptr + Int(swizzled_offset),
                    runtime_layout_type(runtime_shape, runtime_stride),
                )
            else:
                return distribute_type(
                    self.ptr + Int(swizzled_offset),
                )

        else:
            comptime assert (
                Self.layout.known_shape() and threads_layout.all_dims_known()
            ), (
                "Distribute expecting layout with static shapes and"
                " fully static threads_layout"
            )

            # Only extract coordinates in the given axis.
            # Example: axis = 0 for 2x2 threads, we only need thread 0 and 1's
            # coordinates since thread 2 and 3 are getting the same tile.
            comptime thread_projected_stride = flatten(
                threads_layout.stride[
                    axis.value()
                ] if axis else threads_layout.stride
            )
            comptime thread_projected_shape = flatten(
                threads_layout.shape[
                    axis.value()
                ] if axis else threads_layout.shape
            )

            var offset: Scalar[Self.linear_idx_type] = 0

            comptime for i in range(runtime_shape.scalar_length):
                comptime thread_shape_i = threads_layout[i].size()
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * thread_shape_i
                )

            comptime for i in range(len(flatten(Self.layout.stride))):
                var fragments_stride_i = self.runtime_layout.stride.value[i]
                comptime shape_i = Int(thread_projected_shape[i])
                comptime stride_i = Int(thread_projected_stride[i])
                var thread_coord_i = umod(
                    ufloordiv(thread_id, stride_i), shape_i
                )
                offset += Scalar[Self.linear_idx_type](
                    thread_coord_i * fragments_stride_i
                )

            # Swizzling applies to the index of elements rather than scalars because
            # the former is the unit in distribution.
            var swizzled_offset = offset

            comptime if swizzle:
                comptime swizzle_fn = swizzle.value()
                swizzled_offset = swizzle_fn(
                    offset // Scalar[Self.linear_idx_type](self.element_size)
                ) * Scalar[Self.linear_idx_type](self.element_size)

            comptime if self.element_layout.all_dims_known():
                return distribute_type(
                    self.ptr + Int(swizzled_offset),
                    runtime_layout_type(runtime_shape, runtime_stride),
                )
            else:
                return distribute_type(
                    self.ptr + Int(swizzled_offset),
                    runtime_layout_type(runtime_shape, runtime_stride),
                    self.runtime_element_layout,
                )

    @always_inline
    def distribute_with_offset[
        threads_layout: Layout,
        axis: Optional[Int] = None,
        swizzle: Optional[Swizzle] = None,
        submode_axis: Optional[Int] = None,
    ](
        self,
        thread_id: Int,
    ) -> Tuple[
        Self.DistributeType[threads_layout, axis],
        IndexList[threads_layout.rank(), element_type=Self.layout_int_type],
        Scalar[Self.linear_idx_type],
    ]:
        """Similar to `distribute`, but also returns the corner coordinates of
        the tile as well as the offset.

        Parameters:
            threads_layout: The layout of the threads.
            axis: The axis to distribute along.
            swizzle: An optional swizzle function.
            submode_axis: An optional submode axis.

        Args:
            thread_id: The ID of the current thread (0-based).

        Returns:
            A tuple containing:
                - The distributed tensor.
                - The corner coordinates of the tile.
                - The offset of the tile.
        """
        comptime ret_tensor_type = Self.DistributeType[threads_layout, axis]
        comptime distributed_layout = _compute_distribute_layout[
            Self.layout,
            threads_layout,
            axis,
        ]()

        var runtime_shape: ret_tensor_type.RuntimeLayoutType.ShapeType
        comptime if ret_tensor_type.masked:
            runtime_shape = ret_tensor_type.RuntimeLayoutType.ShapeType(
                self._clamp_distribute_shape[threads_layout](thread_id)
            )
        else:
            runtime_shape = ret_tensor_type.RuntimeLayoutType.ShapeType()

        var runtime_stride = ret_tensor_type.RuntimeLayoutType.StrideType()
        var offset_coords = IndexList[
            threads_layout.rank(), element_type=Self.layout_int_type
        ]()
        var offset: Scalar[Self.linear_idx_type] = 0

        # Static layout tiling
        # TODO: Consider merge the two cases in away that won't slowdown the fully static layout.
        comptime if Self.layout.all_dims_known():
            comptime fragments_layout_stride = flatten(
                distributed_layout[0].stride
            )

            # Only extract coordinates in the given axis.
            # Example: axis = 0 for 2x2 threads, we only need thread 0 and 1's
            # coordinates since thread 2 and 3 are getting the same tile.
            comptime thread_projected_stride = flatten(
                threads_layout.stride[
                    axis.value()
                ] if axis else threads_layout.stride
            )
            comptime thread_projected_shape = flatten(
                threads_layout.shape[
                    axis.value()
                ] if axis else threads_layout.shape
            )

            comptime for i in range(len(fragments_layout_stride)):
                comptime fragments_stride_i = Int(fragments_layout_stride[i])
                comptime shape_i = Int(thread_projected_shape[i])
                comptime stride_i = Int(thread_projected_stride[i])
                var thread_coord_i = umod(
                    ufloordiv(thread_id, stride_i), shape_i
                )
                offset_coords[i] = thread_coord_i
                offset += Scalar[Self.linear_idx_type](
                    thread_coord_i * fragments_stride_i
                )

            # Swizzling applies to the index of elements rather than scalars because
            # the former is the unit in distribution.
            var swizzled_offset = offset

            comptime if swizzle:
                comptime swizzle_fn = swizzle.value()
                swizzled_offset = swizzle_fn(
                    offset // Scalar[Self.linear_idx_type](self.element_size)
                ) * Scalar[Self.linear_idx_type](self.element_size)

            comptime if ret_tensor_type.masked:
                return (
                    ret_tensor_type(
                        self.ptr + Int(swizzled_offset),
                        ret_tensor_type.RuntimeLayoutType(
                            runtime_shape, runtime_stride
                        ),
                    ),
                    offset_coords,
                    swizzled_offset,
                )
            else:
                return (
                    ret_tensor_type(
                        self.ptr + Int(swizzled_offset),
                    ),
                    offset_coords,
                    swizzled_offset,
                )

        else:
            comptime assert (
                Self.layout.known_shape() and threads_layout.all_dims_known()
            ), (
                "Distribute expecting layout with static shapes and"
                " fully static threads_layout"
            )

            # Only extract coordinates in the given axis.
            # Example: axis = 0 for 2x2 threads, we only need thread 0 and 1's
            # coordinates since thread 2 and 3 are getting the same tile.
            comptime thread_projected_stride = flatten(
                threads_layout.stride[
                    axis.value()
                ] if axis else threads_layout.stride
            )
            comptime thread_projected_shape = flatten(
                threads_layout.shape[
                    axis.value()
                ] if axis else threads_layout.shape
            )

            comptime for i in range(runtime_shape.scalar_length):
                comptime thread_shape_i = threads_layout[i].size()
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * thread_shape_i
                )

            comptime for i in range(len(flatten(Self.layout.stride))):
                var fragments_stride_i = self.runtime_layout.stride.value[i]
                comptime shape_i = Int(thread_projected_shape[i])
                comptime stride_i = Int(thread_projected_stride[i])
                var thread_coord_i = umod(
                    ufloordiv(thread_id, stride_i), shape_i
                )
                offset_coords[i] = thread_coord_i
                offset += Scalar[Self.linear_idx_type](
                    thread_coord_i * fragments_stride_i
                )

            # Swizzling applies to the index of elements rather than scalars because
            # the former is the unit in distribution.
            var swizzled_offset = offset

            comptime if swizzle:
                comptime swizzle_fn = swizzle.value()
                swizzled_offset = swizzle_fn(
                    offset // Scalar[Self.linear_idx_type](self.element_size)
                ) * Scalar[Self.linear_idx_type](self.element_size)

            comptime if self.element_layout.all_dims_known():
                return (
                    ret_tensor_type(
                        self.ptr + Int(swizzled_offset),
                        ret_tensor_type.RuntimeLayoutType(
                            runtime_shape, runtime_stride
                        ),
                    ),
                    offset_coords,
                    swizzled_offset,
                )
            else:
                return (
                    ret_tensor_type(
                        self.ptr + Int(swizzled_offset),
                        ret_tensor_type.RuntimeLayoutType(
                            runtime_shape, runtime_stride
                        ),
                        self.runtime_element_layout,
                    ),
                    offset_coords,
                    swizzled_offset,
                )

    comptime ShapeVectorizedType[
        origin: ImmOrigin,
        vector_shape: IntTuple,
        linear_vectorize: Bool,
    ] = LayoutTensor[
        Self.dtype,
        coalesce(
            Self._tuple_divide_tiles(vector_shape, linear_vectorize)[1],
            keep_rank=True,
        ),
        origin,
        address_space=Self.address_space,
        element_layout=Self._tuple_divide_tiles(vector_shape, linear_vectorize)[
            0
        ],
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
    ]
    """Type alias for shape-vectorized tensor types.

    Parameters:
        origin: The origin of the result tensor.
        vector_shape: The shape of each vector unit.
        linear_vectorize: Whether to vectorize in a linear manner.
    """

    @always_inline
    def _vectorize_2[
        vector_len: Int,
        linear_vectorize: Bool = True,
    ](self) -> Self.ShapeVectorizedType[
        origin_of(),
        IntTuple(vector_len),
        linear_vectorize=linear_vectorize,
    ]:
        """Wrap the integer `vector_len` in an `IntTuple` and call the
        `_vectorize_2` function.

        Parameters:
            vector_len: The length of vectorization.
            linear_vectorize: Whether to vectorize in a linear manner. Defaults to True.

        Returns:
            A view of the tensor with a vectorized layout based on the specified
            vector length.
        """
        return self._vectorize_2[
            origin_of(),
            IntTuple(vector_len),
            linear_vectorize=linear_vectorize,
        ]()

    @always_inline
    def _vectorize_2[
        _origin: ImmOrigin,  # FIXME: MOCO-1912
        vector_shape: IntTuple,
        check_rank: Bool = True,
        linear_vectorize: Bool = vector_shape.is_value(),
    ](self) -> Self.ShapeVectorizedType[
        _origin, vector_shape, linear_vectorize
    ]:
        """Experimental implementation of the generalized vectorize operation
        using IntTuple.

        This function creates a vectorized view of the tensor using an IntTuple
        to specify the vector dimensions rather than variadic parameters.

        Parameters:
            _origin: The origin of the IntTuple.
            vector_shape: The dimensions of each vector unit as an IntTuple.
            check_rank: Whether to verify that vector_shape is congruent with
                the tensor's shape. Defaults to True.
            linear_vectorize: Whether to vectorize in a linear manner. Defaults to True.

        Returns:
            A view of the tensor with a vectorized layout based on the specified
            vector shape.
        """
        comptime assert (vector_shape.is_value() and linear_vectorize) or (
            not linear_vectorize
        ), (
            "Only contiguous vectorization or vectorization of a"
            " congruent shape is supported!"
        )

        comptime vectorized_type = Self.ShapeVectorizedType[
            _origin, vector_shape, linear_vectorize
        ]
        var runtime_shape = vectorized_type.RuntimeLayoutType.ShapeType()
        var runtime_stride = vectorized_type.RuntimeLayoutType.StrideType()

        comptime if check_rank:
            comptime assert is_int(vector_shape) or congruent(
                vector_shape, Self.layout.shape
            ), "vector_shape has to be congruent to layout.shape = " + String(
                Self.layout.shape
            )

        comptime tiler = Self._tuple_divide_tiler(
            vector_shape, linear_vectorize
        )
        comptime flat_vector_shape = flatten(tiler.shape)

        comptime if vectorized_type.masked or not Self.layout.all_dims_known():
            comptime for i in range(len(flat_vector_shape)):
                comptime vector_shape_i = Int(flat_vector_shape[i])
                runtime_shape.value[i] = ceildiv(
                    self.runtime_layout.shape.value[i], vector_shape_i
                )
                runtime_stride.value[i] = (
                    self.runtime_layout.stride.value[i] * vector_shape_i
                )

        var ptr = self.ptr.as_imm().unsafe_origin_cast[_origin]()

        comptime if Self.layout.all_dims_known():
            comptime if vectorized_type.masked:
                return vectorized_type(
                    ptr,
                    vectorized_type.RuntimeLayoutType(
                        runtime_shape, runtime_stride
                    ),
                )
            else:
                return vectorized_type(ptr)
        else:
            comptime assert coalesce(
                vectorized_type.element_layout
            ).known_shape(), "Result element layout should have known shape"

            var runtime_element_layout_shape = (
                vectorized_type.RuntimeElementLayoutType.ShapeType()
            )
            var runtime_element_layout_stride = (
                vectorized_type.RuntimeElementLayoutType.StrideType(
                    self.runtime_layout.stride.value
                )
            )

            return Self.ShapeVectorizedType[
                _origin, vector_shape, linear_vectorize
            ](
                ptr,
                vectorized_type.RuntimeLayoutType(
                    runtime_shape, runtime_stride
                ),
                vectorized_type.RuntimeElementLayoutType(
                    runtime_element_layout_shape,
                    runtime_element_layout_stride,
                ),
            )

    comptime VectorizedType[*vector_shape: Int] = LayoutTensor[
        Self.dtype,
        coalesce(Self._compute_tile_layout[*vector_shape]()[1], keep_rank=True),
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self._divide_tiles[*vector_shape]()[0],
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
    ]
    """Type alias for vectorized tensor types.

    Parameters:
        vector_shape: The shape of each vector unit along each axis.
    """

    @always_inline
    def vectorize[
        *vector_shape: Int
    ](self) -> Self.VectorizedType[*vector_shape]:
        """Reshape a tensor into a vectorized form for efficient SIMD
        operations.

        This method transforms the tensor's logical layout to enable efficient
        vectorized processing, treating blocks of elements as vector units. The
        transformation is particularly useful for SIMD (Single Instruction
        Multiple Data) operations and hardware acceleration.

        Constraints:
            - Each tensor dimension must be divisible by the corresponding
                vector dimension.
            - Vector dimensions must be smaller than or equal to the
                corresponding tensor dimensions.
            - For dimensions with unknown size, the vector dimension must be 1.

        Parameters:
            vector_shape: The dimensions of each vector unit along each axis of
                the tensor. or example, in a 2D tensor, `vectorize[4, 4]` treats
                4x4 blocks as vector units.

        Returns:
            A view of the tensor with a vectorized layout, where each element in
            the resulting tensor represents a vector of elements from the
            original tensor.

        Example:

        For a 16x16 tensor, `vectorize[4, 4]` will produce a 4x4 tensor
        where each element represents a 4x4 block from the original tensor.

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Enables hardware-accelerated vector operations on blocks of data.
        - Improves cache locality by grouping related elements together.
        - Particularly beneficial for operations that can leverage SIMD
            instructions.

        Notes:

        - The tensor dimensions must be divisible by the corresponding vector
            dimensions.
        - For dimensions with unknown size, the corresponding vector dimension
            must be 1.
        - The resulting tensor has the same data but a different logical
            organization.
        - Modifications to the vectorized tensor affect the original tensor.
        - This transformation is particularly useful for GPU and vector
            processor optimizations.
        """

        comptime shape = IntTuple.__init__[*vector_shape]()
        comptime _origin = origin_of()  # FIXME: MOCO-1912
        var ret = self._vectorize_2[
            _origin,
            shape,
            check_rank=False,
            linear_vectorize=False,
        ]()
        # FIXME: this is ugly, is there a simpler way to do this?
        return rebind[Self.VectorizedType[*vector_shape]](ret)

    comptime SIMDVectorizedType = Self.VectorizedType[
        1, simd_width_of[Self.dtype]()
    ]
    """Result type for SIMD-width vectorization."""

    @always_inline
    def vectorize(self) -> Self.SIMDVectorizedType:
        """Return a SIMD[dtype] vectorized view of this tensor.

        Returns:
            A `Self.VectorizedType[1, simd_width_of[Self.dtype]()]` view whose
            width equals the SIMD width for the tensor's dtype.
        """
        return self.vectorize[1, simd_width_of[Self.dtype]()]()

    @staticmethod
    def _compute_slice_layout(d0_slice: Slice, d1_slice: Slice) -> Layout:
        comptime assert (
            Self.layout.shape.__len__() == 2
        ), "Only rank-2 tensors slices are supported for now!"
        return Layout(
            [
                _get_slice_size(materialize[Self.layout](), d0_slice, 0),
                _get_slice_size(materialize[Self.layout](), d1_slice, 1),
            ],
            Self.layout.stride,
        )

    @staticmethod
    def _compute_slice_layout(
        slice_0: Slice, slice_1: Slice, slice_0_axis: Int, slice_1_axis: Int
    ) -> Layout:
        comptime assert Self.layout.rank() >= 2, "Rank should be >= 2"

        var sliced_layout = sublayout(
            materialize[Self.layout](), slice_0_axis, slice_1_axis
        )
        return Layout(
            [
                _get_slice_size(sliced_layout, slice_0, 0),
                _get_slice_size(sliced_layout, slice_1, 1),
            ],
            sliced_layout.stride,
        )

    @staticmethod
    def _compute_slice_layout(slice_0: Slice, slice_0_axis: Int) -> Layout:
        comptime assert Self.layout.shape.__len__() > 1, "Rank should be >= 1"
        var sliced_layout = sublayout(materialize[Self.layout](), slice_0_axis)
        return Layout(
            [_get_slice_size(sliced_layout, slice_0, 0)],
            sliced_layout.stride[0],
        )

    comptime SliceType[
        d0_slice: Slice,
        d1_slice: Slice,
    ] = LayoutTensor[
        Self.dtype,
        Self._compute_slice_layout(
            d0_slice,
            d1_slice,
        ),
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for 2D slice result tensors.

    Parameters:
        d0_slice: Slice specification for the first dimension.
        d1_slice: Slice specification for the second dimension.
    """

    @always_inline
    def slice[
        d0_slice: Slice, d1_slice: Slice
    ](self) -> Self.SliceType[d0_slice, d1_slice]:
        """Extract a slice from a rank-2 tensor using slice objects.

        This method creates a view into a subset of the tensor defined by the
        slice specifications for each dimension. The slice is a continuous
        region of the tensor with no gaps (step size must be 1).

        Constraints:
            - Only works with rank-2 tensors.

        Parameters:
            d0_slice: Slice specification for the first dimension (rows).
                Defines the start and end indices for the slice along this
                dimension.
            d1_slice: Slice specification for the second dimension (columns).
                Defines the start and end indices for the slice along this
                dimension.

        Returns:
            A view into the original tensor representing the specified slice.

        Example:

        For a 4x4 tensor, `t` with values:

        ```
        [1 2 3 4]
        [5 6 7 8]
        [9 10 11 12]
        [13 14 15 16]
        ```

        ```mojo
        t.slice[Slice(1, 3), Slice(0, 2)]
        ```

        will extract:

        ```
        [5 6]
        [9 10]
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Maintains the original tensor's stride information for efficient
            memory access.
        - Zero-cost abstraction at runtime when used with compile-time constant
            slices.

        Notes:

        - The slice is a view into the original tensor, so modifications to the
            slice will affect the original tensor.
        - Only supports rank-2 tensors. For higher-rank tensors, use the
            overloaded version with slice indices.
        - The step size must be 1 (no gaps allowed in the slice).
        - Slice bounds are not checked at runtime; accessing out-of-bounds
            indices will result in undefined behavior.
        """
        comptime assert (
            d0_slice.step.or_else(1) == 1 and d1_slice.step.or_else(1) == 1
        ), "Slice should have no gaps"

        comptime return_type = Self.SliceType[d0_slice, d1_slice]
        comptime stride_m = Int(return_type.layout.stride[0])
        comptime stride_n = Int(return_type.layout.stride[1])

        comptime d0_slice_start = d0_slice.start.or_else(0)
        comptime d1_slice_start = d1_slice.start.or_else(0)

        var offset = d0_slice_start * stride_m + d1_slice_start * stride_n

        return Self.SliceType[d0_slice, d1_slice](self.ptr + offset)

    comptime SliceType2D[
        d0_slice: Slice,
        d1_slice: Slice,
        slice_indices: IndexList[2],
        __offset_dims: Int = Self.rank - 2,
    ] = LayoutTensor[
        Self.dtype,
        Self._compute_slice_layout(
            d0_slice, d1_slice, slice_indices[0], slice_indices[1]
        ),
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for 2D slice result tensors from higher-rank tensors.

    Parameters:
        d0_slice: Slice specification for the first selected dimension.
        d1_slice: Slice specification for the second selected dimension.
        slice_indices: Indices of the two dimensions to slice.
        __offset_dims: Number of fixed dimensions.
    """

    @always_inline
    def slice[
        d0_slice: Slice,
        d1_slice: Slice,
        slice_indices: IndexList[2],
        __offset_dims: Int = Self.rank - 2,
    ](
        self,
        offsets: IndexList[__offset_dims],
    ) -> Self.SliceType2D[
        d0_slice, d1_slice, slice_indices, __offset_dims
    ]:
        """Extract a 2D slice from a higher-rank tensor at specific indices.

        This method creates a view into a 2D subset of a higher-rank tensor:

        Selecting two dimensions to slice using the slice_indices parameter.
        Applying slice specifications to those dimensions.
        Using fixed offsets for all other dimensions.

        Constraints:
            - Slice step size must be 1 (no gaps).
            - Slice indices must be ordered (ascending).
            - Tensor rank must be at least 2.

        Parameters:
            d0_slice: Slice specification for the first selected dimension.
            d1_slice: Slice specification for the second selected dimension.
            slice_indices: Indices of the two dimensions to slice (must be
                ordered).
            __offset_dims: Internal parameter representing number of fixed
                dimensions.

        Args:
            offsets: Fixed index values for all dimensions not being sliced.

        Returns:
            A 2D view into the original tensor representing the specified slice.

        Example:

        Given a 3x4x5 tensor, `t`, the following example extracts a 2x2 slice
        from dimensions 0 and 2, with dimension 1 fixed at index 1.

        ```mojo
        t.slice = t.slice[Slice(1, 3), Slice(0, 2), IndexList[2](0, 2)](1)
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Maintains the original tensor's stride information for efficient
            memory access.
        - Zero-cost abstraction at runtime when used with compile-time constant
            slices.

        Notes:

        - The slice is a view into the original tensor, so modifications to the
            slice will affect the original tensor.
        - The slice indices must be ordered (e.g., [0, 2] is valid, [2, 0] is
            not).
        - The step size must be 1 (no gaps allowed in the slice).
        - Slice bounds are not checked at runtime; accessing out-of-bounds
            indices will result in undefined behavior.
        """
        comptime assert (
            d0_slice.step.or_else(1) == 1 and d1_slice.step.or_else(1) == 1
        ), "Slice should have no gaps"
        comptime assert (
            slice_indices[0] < slice_indices[1]
        ), "Slice indices should be ordered"
        comptime slice_type = Self.SliceType2D[
            d0_slice, d1_slice, slice_indices, __offset_dims
        ]

        comptime stride_0 = Int(slice_type.layout.stride[0])
        comptime stride_1 = Int(slice_type.layout.stride[1])

        comptime d0_slice_start = d0_slice.start.or_else(0)
        comptime d1_slice_start = d1_slice.start.or_else(0)

        var slice_offset = d0_slice_start * stride_0 + d1_slice_start * stride_1

        var idx = 0

        comptime for i in range(Self.rank):
            comptime stride_i = Int(Self.layout.stride[i])

            comptime offset_index = _not_in_tuple[i, 2, slice_indices]()

            comptime if offset_index:
                slice_offset += offsets[idx] * stride_i
                idx += 1

        return Self.SliceType2D[
            d0_slice, d1_slice, slice_indices, __offset_dims
        ](self.ptr + slice_offset)

    comptime SliceType1D[
        d0_slice: Slice,
        slice_indices: IndexList[1],
        __offset_dims: Int = Self.rank - 1,
    ] = LayoutTensor[
        Self.dtype,
        Self._compute_slice_layout(d0_slice, slice_indices[0]),
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for 1D slice result tensors from higher-rank tensors.

    Parameters:
        d0_slice: Slice specification for the selected dimension.
        slice_indices: Index of the dimension to slice.
        __offset_dims: Number of fixed dimensions.
    """

    # FIXME: Can't overload slice, hitting compiler issue.
    # https://linear.app/modularml/issue/MOCO-174
    @always_inline
    def slice_1d[
        d0_slice: Slice,
        slice_indices: IndexList[1],
        __offset_dims: Int = Self.rank - 1,
    ](
        self,
        offsets: IndexList[__offset_dims],
    ) -> Self.SliceType1D[
        d0_slice, slice_indices, __offset_dims
    ]:
        """Extract a 1D slice from a higher-rank tensor at a specific index.

        This method creates a view into a 1D subset of a higher-rank tensor by:
        1. Selecting one dimension to slice using the slice_indices parameter
        2. Applying a slice specification to that dimension
        3. Using fixed offsets for all other dimensions

        Constraints:
            - Slice step size must be 1 (no gaps).
            - Tensor rank must be at least 1.

        Parameters:
            d0_slice: Slice specification for the selected dimension.
            slice_indices: Index of the dimension to slice.
            __offset_dims: Internal parameter representing number of fixed
                dimensions.

        Args:
            offsets: Fixed index values for all dimensions not being sliced.

        Returns:
            A 1D view into the original tensor representing the specified slice.

        Example:

        For a 3x4x5 tensor, `t`, the following example extracts a 1D slice from
        dimension 0, with dimensions 1 and 2 fixed at indices 1 and 2:

        ```mojo
        t.slice_1d[Slice(1, 3), IndexList[1](0)](1, 2)
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - Maintains the original tensor's stride information for efficient
            memory access.
        - Zero-cost abstraction at runtime when used with compile-time constant
            slices.

        Notes:

        - The slice is a view into the original tensor, so modifications
            to the slice will affect the original tensor.
        - The step size must be 1 (no gaps allowed in the slice).
        - Slice bounds are not checked at runtime; accessing out-of-bounds
            indices will result in undefined behavior.
        - This function exists as a workaround for compiler limitations with
            overloading.
        """
        comptime assert (
            d0_slice.step.or_else(1) == 1
        ), "Slice should have no gaps"

        comptime slice_type = Self.SliceType1D[
            d0_slice, slice_indices, __offset_dims
        ]

        comptime stride_0 = Int(slice_type.layout.stride[0])

        comptime d0_slice_start = d0_slice.start.or_else(0)

        var slice_offset = d0_slice_start * stride_0

        var idx = 0

        comptime for i in range(Self.rank):
            comptime stride_i = Int(Self.layout.stride[i])

            comptime offset_index = _not_in_tuple[i, 1, slice_indices]()

            comptime if offset_index:
                slice_offset += offsets[idx] * stride_i
                idx += 1

        return Self.SliceType1D[d0_slice, slice_indices, __offset_dims](
            self.ptr + slice_offset
        )

    comptime TransposeType = LayoutTensor[
        Self.dtype,
        Self.layout.transpose(),
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Result type for transpose operations."""

    @always_inline
    def transpose(self) -> Self.TransposeType:
        """Create a transposed view of a tensor.

        This method creates a view of the tensor with its dimensions swapped, effectively
        converting rows to columns and columns to rows. The transposition is performed
        without copying data, by adjusting the tensor's layout information.

        Returns:
            A view of the tensor with dimensions transposed (rows become columns and vice versa).

        Example:

        For a 2x3 tensor with values:

        ```
        [1 2 3]
        [4 5 6]
        ```

        `transpose()` will produce a 3x2 tensor:

        ```
        [1 4]
        [2 5]
        [3 6]
        ```

        Performance:

        - Creates a view without copying data, making it very efficient.
        - The operation is zero-cost at runtime as it only changes the layout
            information.
        - Memory access patterns may be less efficient in the transposed view
            due to non-contiguous memory access, especially for row-major
            storage.

        Notes:

        - The transposed tensor shares the same memory as the original tensor,
            so modifications to one will affect the other.
        - For optimal performance when repeatedly accessing the transposed data,
            consider creating a physical copy with the transposed layout.
        - Transpose only works with statically known shapes.
        """
        comptime assert (
            Self.layout.all_dims_known()
        ), "Transpose only works with statically known shapes."
        return Self.TransposeType(self.ptr)

    comptime ReshapeType[
        dst_layout: Layout,
    ] = LayoutTensor[
        Self.dtype,
        dst_layout,
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
        alignment=Self.alignment,
    ]
    """Type alias for reshaped tensor types.

    Parameters:
        dst_layout: The target layout for the reshaped tensor.
    """

    @always_inline
    def reshape[
        dst_layout: Layout,
    ](self) -> Self.ReshapeType[dst_layout]:
        """Create a view of the tensor with a different shape.

        This method creates a view of the tensor with a new shape, without changing
        the underlying data. The total number of elements must remain the same.

        Constraints:
            - Cannot reshape masked tensors.
            - The total number of elements must be the same in both layouts.

        Parameters:
            dst_layout: The target layout for the reshaped tensor. Must have the same
                       total number of elements as the original tensor.

        Returns:
            A view of the tensor with the new shape specified by dst_layout.

        Example:

        Given a 2x6 row-major tensor, `reshape[Layout.col_major(3, 4)]()`
        produces a 3x4 tensor with the same elements in column-major order.

        Performance:

        - Creates a view without copying data, making it very efficient.
        - The operation is zero-cost at runtime as it only changes the layout
            information.
        - Memory access patterns may change, potentially affecting performance
            depending on the original and target layouts.

        Notes:

        - The reshaped tensor shares the same memory as the original tensor,
            so modifications to one will affect the other.
        - The total number of elements must remain the same after reshaping.
        - The reshape operation assumes a row-major (C-style) memory layout.
        - For tensors with complex strides or non-contiguous memory, reshaping
            may not produce the expected results.
        - Masked tensors cannot be reshaped.
        """
        comptime assert (
            not Self.masked
        ), "Masked tensor does not support reshape."
        return Self.ReshapeType[dst_layout](self.ptr)

    @always_inline
    def reshape[
        dst_layout: Layout,
    ](self, runtime_layout: RuntimeLayout[dst_layout]) -> Self.ReshapeType[
        dst_layout
    ]:
        """Create a view of the tensor with a different shape.

        This method creates a view of the tensor with a new shape, without changing
        the underlying data. The total number of elements must remain the same.

        Constraints:
            - Cannot reshape masked tensors.
            - The total number of elements must be the same in both layouts.

        Parameters:
            dst_layout: The target layout for the reshaped tensor. Must have the same
                       total number of elements as the original tensor.

        Args:
            runtime_layout: The target RuntimeLayout for the reshaped tensor.

        Returns:
            A view of the tensor with the new shape specified by dst_layout.

        Example:

        Given a 2x6 row-major tensor, `reshape[Layout.col_major(3, 4)]()`
        produces a 3x4 tensor with the same elements in column-major order.

        Performance:

        - Creates a view without copying data, making it very efficient.
        - The operation is zero-cost at runtime as it only changes the layout
            information.
        - Memory access patterns may change, potentially affecting performance
            depending on the original and target layouts.

        Notes:

        - The reshaped tensor shares the same memory as the original tensor,
            so modifications to one will affect the other.
        - The total number of elements must remain the same after reshaping.
        - The reshape operation assumes a row-major (C-style) memory layout.
        - For tensors with complex strides or non-contiguous memory, reshaping
            may not produce the expected results.
        - Masked tensors cannot be reshaped.
        """
        comptime assert (
            not Self.masked
        ), "Masked tensor does not support reshape."
        return Self.ReshapeType[dst_layout](self.ptr, runtime_layout)

    comptime FlattenedType = LayoutTensor[
        Self.dtype,
        Layout(UNKNOWN_VALUE),
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
        alignment=Self.alignment,
    ]
    """Type alias for flattened tensor types.
    """

    @always_inline("nodebug")
    def flatten(self) -> Self.FlattenedType:
        """Convert a LayoutTensor to a flattened dynamic layout.

        Returns:
            A LayoutTensor to a flattened dynamic layout.
        """
        return Self.FlattenedType(
            self.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                IndexList[1](self.size())
            ),
        )

    comptime CompositionType[
        rhs_layout: Layout,
        dst_layout: Layout = composition(Self.layout, rhs_layout),
    ] = LayoutTensor[
        Self.dtype,
        dst_layout,
        Self.origin,
        address_space=Self.address_space,
        element_layout=Self.element_layout,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for composed layout tensor types.

    Parameters:
        rhs_layout: The layout to compose with.
        dst_layout: The resulting composed layout.
    """

    @always_inline
    def composition[
        rhs_layout: Layout,
        dst_layout: Layout = composition(Self.layout, rhs_layout),
    ](self, out result: self.CompositionType[rhs_layout, dst_layout]):
        """Create a view of the tensor with a composed layout.

        This method creates a view of the tensor with a new layout that is the
        composition of the original layout with another layout. Layout
        composition allows for complex transformations of the tensor's logical
        structure without copying data.

        Constraints:
            - The layouts must be compatible for composition.
            - The total number of elements must remain the same after
                composition.

        Parameters:
            rhs_layout: The layout to compose with the tensor's current layout.
            dst_layout: The resulting layout after composition. Defaults to the
                       composition of the tensor's layout with rhs_layout.

        Returns:
            A view of the tensor with the composed layout.

        Example:

        For a 4x4 tensor with a standard row-major layout, composing with a
        layout that represents a 2x2 tiling would result in a tensor that
        logically views the data as 2x2 blocks.

        Performance:

        - Creates a view without copying data, making it very efficient.
        - The operation is zero-cost at runtime as it only changes the layout information.
        - Can be used to optimize memory access patterns for specific algorithms.

        Notes:

        - The composed tensor shares the same memory as the original tensor,
            so modifications to one will affect the other.
        - Layout composition is a powerful tool for expressing complex data
            transformations like tiling, transposition, and reshaping in a
            unified framework.
        - Understanding the mathematical properties of layout composition is
            important for correctly using this function.
        """
        return self.CompositionType[rhs_layout, dst_layout](self.ptr)

    @__allow_legacy_custom_self_type
    @always_inline
    def distance(
        self: Self.Immut,
        addr: ImmPointer[
            Scalar[Self.dtype], address_space=Self.address_space, ...
        ],
    ) -> Scalar[Self.linear_idx_type]:
        """Calculate the element-wise distance between this tensor's pointer
        and another pointer.

        This method computes the number of elements (not bytes) between the
        tensor's pointer and the provided address. This is useful for
        determining offsets within a larger memory allocation or for pointer
        arithmetic operations.

        Args:
            addr: The target pointer to calculate the distance to.

        Returns:
            The number of elements between this tensor's pointer and the
            provided address. The result is of type `_uint_dtype`.

        Example:

        If `tensor.ptr` points to an element at index 100 in a buffer, and
        `addr` points to element at index 50, then `distance(addr)` returns 50.

        Performance:

        - This is a lightweight operation that only involves pointer arithmetic.
        - The operation is optimized based on the address space, using smaller
            integer types for shared memory to improve efficiency.

        Notes:

        - The distance is calculated in elements, not bytes.
        - The result can be positive or negative depending on the relative positions
            of the pointers.
        - This function is particularly useful for GPU programming where understanding
            memory offsets is critical for performance.
        - Care should be taken when using this with pointers from different allocations,
            as the result would be meaningless.
        """
        return Scalar[Self.linear_idx_type](
            Int(self.ptr) - Int(addr)
        ) // Scalar[Self.linear_idx_type](size_of[Self.dtype]())

    @__allow_legacy_custom_self_type
    @always_inline
    def distance[
        _layout: Layout,
        _uint_dtype: DType = _get_unsigned_type(_layout, Self.address_space),
    ](
        self: Self.Immut,
        src: LayoutTensor[
            mut=False,
            Self.dtype,
            _layout,
            address_space=Self.address_space,
            ...,
        ],
    ) -> Scalar[_uint_dtype]:
        """Calculate the element-wise distance between this tensor and another
        tensor.

        This method computes the number of elements (not bytes) between this
        tensor's pointer and another tensor's pointer. This is useful for
        determining the relative positions of tensors within a larger memory
        allocation.

        Parameters:
            _layout: The layout of the source tensor.
            _uint_dtype: The unsigned integer type to use for the result.
                Automatically determined based on the layout and address space.

        Args:
            src: The source tensor to calculate the distance to.

        Returns:
            The number of elements between this tensor's pointer and the source
            tensor's pointer. The result is of type _uint_dtype.

        Example:

        If tensor1 points to element at index 100 in a buffer, and tensor2 points
        to element at index 50, then `tensor1.distance(tensor2)` would return 50.

        Performance:

        - This is a lightweight operation that only involves pointer arithmetic.
        - The operation is optimized based on the address space and layout,
            using appropriate integer types for efficiency.

        Notes:

        - The distance is calculated in elements, not bytes.
        - The result can be positive or negative depending on the relative
            positions of the tensors.
        - This function is particularly useful for GPU programming where
            understanding memory offsets is critical for performance.
        - Both tensors must be in the same address space for the result to be
            meaningful.
        - This overload is more type-safe than the pointer-based version as it
            ensures the tensors have compatible data types and address spaces.
        """

        return Scalar[_uint_dtype](
            (Int(self.ptr) - Int(src.ptr)) // size_of[Self.dtype]()
        )

    # Returns the linear index of an elem_i 0 ... size(layout).
    #
    @always_inline
    def _get_element_idx[elem_i: Int](self) -> Scalar[Self.linear_idx_type]:
        comptime element_size = self.element_size

        comptime if Self.layout.all_dims_known():
            comptime idx = make_layout(Self.element_layout, Self.layout)(
                elem_i * element_size
            )
            return Scalar[Self.linear_idx_type](idx)
        else:
            # FIXME: this used to be simpler
            var rt = RuntimeTuple[IntTuple(UNKNOWN_VALUE)](
                elem_i * element_size
            )
            var idx = make_runtime_layout[linear_idx_type=Self.linear_idx_type](
                self.runtime_element_layout, self.runtime_layout
            )(rt)
            return idx

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def copy_from(self: Self._AsMut, other: LayoutTensor):
        """Copy data from another tensor to this tensor.

        This method performs an element-by-element copy from the source tensor
        to this tensor, respecting the layouts of both tensors. The copy
        operation handles different memory layouts correctly, ensuring that
        elements are copied to their proper positions regardless of how the data
        is arranged in memory.

        Constraints:
        - Both tensors must have statically known shapes.
        - The total number of elements must be the same in both tensors.
        - The element sizes must match between the tensors.

        Args:
            other: The source tensor to copy data from. Must have the same total
                number of elements as this tensor.

        Example:

        ```mojo
        from layout import LayoutTensor, Layout

        var src_engine = Array[Float32, 2 * 3](uninitialized=True)
        var dst_storage = Array[Float32, 3 * 2](uninitialized=True)
        var src = LayoutTensor[
            DType.float32,
            Layout([2, 3]),
        ](src_engine).fill(1.0)

        var dst = LayoutTensor[
            DType.float32,
            Layout([3, 2]),
        ](dst_storage)

        dst.copy_from(src)  # Copies all elements from src to dst
        ```

        Performance:

        - Performs element-by-element copying, which may be less efficient than
            vectorized or bulk memory operations.
        - The copy respects the memory layout of both tensors, which may involve
            non-contiguous memory access patterns.
        - For optimal performance with large tensors, consider using specialized
            copy functions that can leverage hardware acceleration.

        Notes:

        - Both tensors must have statically known shapes.
        - The total number of elements must be the same in both tensors.
        - The element sizes must match between the tensors.
        - This function handles different memory layouts correctly, making it suitable
            for copying between tensors with different shapes or strides.
        - The copy is performed element by element, not as a bulk memory copy.
        """
        comptime other_layout = other.layout

        comptime dst_element_size = self.element_size
        comptime src_element_size = other.element_size

        comptime dst_size = Self.layout.size()
        comptime src_size = other_layout.size()

        comptime assert (
            Self.layout.known_shape() and other_layout.known_shape()
        ), "copy_from must move data of statically known shape"

        comptime assert dst_size == src_size, (
            "copy_from should move data of the same size, getting dst size "
            + String(dst_size)
            + " and src size "
            + String(src_size)
        )

        comptime assert (
            dst_element_size == src_element_size
        ), "copy_from should move"

        comptime for i in range(dst_size):
            var src_idx = other._get_element_idx[i]()
            var dst_idx = self._get_element_idx[i]()

            var src_element = MemoryElement[index_type=other.linear_idx_type](
                other.ptr + src_idx, other.runtime_element_layout
            )

            var dst_element = MemoryElement[index_type=Self.linear_idx_type](
                self.ptr + dst_idx, self.runtime_element_layout
            )

            dst_element.transfer(src_element)

    @always_inline("nodebug")
    def copy_from_async[
        is_masked: Bool = False,
        swizzle: Optional[Swizzle] = None,
        fill: Fill = Fill.NONE,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        src: LayoutTensor,
        src_idx_bound: Scalar[src.linear_idx_type] = 0,
        base_offset: Scalar[Self.linear_idx_type] = 0,
    ):
        """Asynchronously copy data from another tensor to this tensor using GPU
        hardware.

        This method performs an asynchronous copy from the source tensor to this
        tensor using GPU hardware acceleration. It's specifically designed for
        copying data from global memory to shared memory in GPU kernels,
        leveraging hardware-specific asynchronous copy mechanisms for improved
        performance.

        For optimal performance, you need to arrange the copy correctly. Use the
        [`distribute()`](/api/mojo/layout/layout_tensor/LayoutTensor/#distribute)
        method to create thread-local fragments of the source and
        destination tensors, assigning each thread one or more elements to copy.

        Optionally, use the
        [`vectorize()`](/api/mojo/layout/layout_tensor/LayoutTensor/#vectorize)
        method to get vectorized views of both tensors before calling
        `distribute()`. This allows each thread to copy multiple elements of the
        tensor. For example:

        ```mojo
        var fragment = tensor.vectorize[1, simd_width]().distribute[
            thread_layout
        ](thread_id)
        ```

        The copy operation is asynchronous, so you must call
        [`async_copy_wait_all()`](/api/mojo/max/gpu/memory/memory/async_copy_wait_all/)
        or
        [`async_copy_wait_group()`](/api/mojo/max/gpu/memory/memory/async_copy_wait_group/)
        to ensure the copy has completed before using the data.

        Constraints:
            - Destination must be in shared memory.
            - Source and destination data types must match.
            - Element size must be 4, 8, or 16 bytes.
            - Destination tensor must have a static layout.

        Parameters:
            is_masked: Whether to perform a masked copy, where elements outside
                the `src_idx_bound` are not copied or filled with zeros.
            swizzle: Optional swizzling function to rearrange the destination
                indices, which can improve memory access patterns.
            fill: Fill policy for elements that are not copied (only used with
                masked copies).
            eviction_policy: Cache eviction policy for the source data.

        Args:
            src: The source tensor to copy data from.
            src_idx_bound: For masked copies, the upper bound index for valid
                source elements.
            base_offset: Base offset for swizzling calculations.

        Example:

        ```mojo
        from layout import LayoutTensor, Layout
        from max.gpu import thread_idx, block_idx, block_dim
        from max.gpu.memory import async_copy_wait_all

        comptime dtype = DType.float32
        comptime in_size = 128
        comptime block_size = 16
        num_blocks = in_size // block_size
        comptime input_layout = Layout.row_major(in_size, in_size)

        def kernel(tensor: LayoutTensor[dtype, input_layout, MutAnyOrigin]):
            # extract a tile from the input tensor.
            var global_tile = tensor.tile[block_size, block_size](block_idx.x, block_idx.y)

            # allocate a shared memory tile
            comptime tile_layout = Layout.row_major(block_size, block_size)
            var shared_tile = LayoutTensor[
                dtype,
                tile_layout,
                MutAnyOrigin,
                address_space = AddressSpace.SHARED,
            ].stack_allocation()

            # Create per-thread tile fragments for copying
            var tid = thread_idx.y + thread_idx.x * block_dim.x
            comptime thread_layout = Layout.row_major(block_size, block_size)
            var global_fragment = global_tile.distribute[thread_layout](tid)
            var shared_fragment = shared_tile.distribute[thread_layout](tid)

            # async copy to shared memory
            shared_fragment.copy_from_async(global_fragment)
            async_copy_wait_all()
            # ... do something with the shared tile
        ```

        Performance:

        - Supports vectorized copies for 4, 8, or 16-byte elements for better
            throughput.
        - Can bypass L1 cache with appropriate eviction policies for specific
            access patterns.
        - Swizzling can improve memory access patterns and reduce bank
            conflicts.

        Notes:

        - For vectorized copies, both tensors must have contiguous element
            layouts.
        - Asynchronous copies allow computation to overlap with memory
            transfers.
        - A synchronization barrier is required before using the copied data.
        """
        comptime assert (
            self.address_space == .SHARED
        ), "Async is only supported for destinations in shared memory"

        comptime assert (
            src.dtype == Self.dtype
        ), "src dtype must be the same as dst dtype."

        comptime dst_size = Self.layout.size()
        comptime src_size = src.layout.size()

        comptime dst_element_size = self.element_size
        comptime src_element_size = src.element_size
        comptime assert (
            dst_element_size == src_element_size
        ), "copy_from_async should move data of the same element size"

        # Eligibility for 4, 8, 16 bytes async load.
        comptime element_size_bytes = size_of[Self.dtype]() * src_element_size
        comptime assert (
            element_size_bytes == 4
            or element_size_bytes == 8
            or element_size_bytes == 16
        ), "copy_from_async only allows 4, 8, 16 bytes element"

        # Share memory must always have static layout.
        comptime dst_dims_known = (
            self.layout.all_dims_known()
            and self.element_layout.all_dims_known()
        )
        comptime assert dst_dims_known, "dst tensor must have static layout"

        comptime src_dims_known = (
            src.layout.all_dims_known() and src.element_layout.all_dims_known()
        )

        var dst_ptr = self.ptr.address_space_cast[.SHARED]().unsafe_mut_cast[
            True
        ]()
        var src_ptr = src.ptr.address_space_cast[.GLOBAL]()

        # Coalesce element layouts to simplify vectorization condition.
        comptime coalesce_src_element_layout = coalesce(src.element_layout)
        comptime coalesce_dst_element_layout = coalesce(self.element_layout)

        comptime if (
            src.element_layout.all_dims_known()
            and coalesce_src_element_layout.rank() == 1
            and coalesce_src_element_layout.stride[0] == 1
            and coalesce_dst_element_layout.rank() == 1
            and coalesce_dst_element_layout.stride[0] == 1
        ):
            comptime num_vecs = Self.layout.size()

            comptime for i in range(num_vecs):
                var src_idx: Scalar[src.linear_idx_type]
                comptime src_static_idx: Scalar[src.linear_idx_type] = Scalar[
                    src.linear_idx_type
                ](src.layout(i))

                comptime if src_dims_known:
                    src_idx = src_static_idx
                else:
                    src_idx = src.runtime_layout(i)
                comptime dst_idx = Self.layout(i)
                var swizzled_idx: Scalar[self.linear_idx_type]

                comptime if swizzle:
                    comptime swizzle_fn = swizzle.value()
                    comptime dst_idx_base = dst_idx % swizzle_fn.size()
                    comptime dst_idx_diff = dst_idx - dst_idx_base
                    swizzled_idx = (
                        swizzle_fn(
                            base_offset
                            + Scalar[Self.linear_idx_type](dst_idx_base)
                        )
                        + Scalar[Self.linear_idx_type](dst_idx_diff)
                        - base_offset
                    ).cast[Self.linear_idx_type]()
                else:
                    swizzled_idx = Scalar[Self.linear_idx_type](dst_idx)

                comptime if is_masked:
                    var src_copy_size = (
                        Int32(element_size_bytes) if src_idx
                        < src_idx_bound else 0
                    )
                    async_copy[
                        element_size_bytes, fill=Scalar[Self.dtype](0.0)
                    ](
                        src_ptr.bitcast[Scalar[Self.dtype]]() + src_idx,
                        dst_ptr + Int(swizzled_idx),
                        src_copy_size,
                    )
                else:
                    async_copy[
                        element_size_bytes,
                        eviction_policy=eviction_policy,
                    ](
                        src_ptr.bitcast[Scalar[Self.dtype]]() + src_idx,
                        dst_ptr + swizzled_idx,
                    )

        # Async copy should only be used for 16B vector for bypassing L1.
        # Scalar path is only for kernel tests.
        else:
            comptime assert not swizzle, "Should not swizzle scalar copy."

            comptime for i in range(dst_size * dst_element_size):
                var src_idx: Scalar[src.linear_idx_type]
                comptime src_static_idx = make_layout(
                    src.element_layout, src.layout
                )(i)
                comptime dst_idx = make_layout(
                    self.element_layout, self.layout
                )(i)

                comptime if src_dims_known:
                    src_idx = Scalar[src.linear_idx_type](src_static_idx)
                else:
                    # FIXME: this used to be simpler
                    var rt = RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i)
                    src_idx = make_runtime_layout[
                        linear_idx_type=src.linear_idx_type
                    ](src.runtime_element_layout, src.runtime_layout)(rt)

                async_copy[4, eviction_policy=eviction_policy](
                    src_ptr.bitcast[Scalar[Self.dtype]]() + src_idx,
                    dst_ptr + dst_idx,
                )

    @__allow_legacy_custom_self_type
    @always_inline
    def fill[
        *,
        use_runtime_layout: Bool = (
            not Self.layout.all_dims_known() or Self.layout.size() > BATCH_SIZE
        ),
    ](
        self: LayoutTensor[mut=True, Self.dtype, ...], val: Scalar[Self.dtype]
    ) -> type_of(self):
        """Fill the entire tensor with a single value.

        This method sets all elements of the tensor to the specified value. It
        works with both statically and dynamically shaped tensors.

        For statically known layouts, the fill operation is unrolled at compile
        time. For dynamic layouts, a runtime loop is used. No vectorization is
        applied, so performance may be suboptimal for large tensors. Consider
        using hardware-specific fill operations for better performance with
        large tensors.

        This method can be used with tensors of any rank and shape. The
        fill operation respects the tensor's layout, filling all
        elements regardless of how they are arranged in memory. For
        tensors with `element_layout`, all elements within each logical element
        are filled with the same value.

        Parameters:
            use_runtime_layout: Whether to use the runtime layout for filling.
                This parameter is defaulted to `True` if the layout is not
                statically known. If loop bounds are too large, it's better to
                use the runtime layout to avoid long compilation time.

        Args:
            val: The value to fill the tensor with. Must be of the same data
                type as the tensor.

        Returns:
            The tensor itself (self), allowing for method chaining.

        Example:

        ```mojo
        from layout import Layout, LayoutTensor

        def main() raises:
            var storage = Array[Float32, 3 * 4](uninitialized=True)
            var tensor = LayoutTensor[
                DType.float32,
                Layout([3, 4]),
            ](storage).fill(0.0)
            print(tensor)
        ```

        If not using method chaining, you can either reassign the result to the
        tensor variable, or assign the result to the discard pattern (`_`) to
        avoid warnings about an unused value:

        ```mojo
        tensor = tensor.fill(0.0)
        # or
        _ = tensor.fill(0.0)
        ```
        """

        comptime if not use_runtime_layout:
            comptime num_elements = Self.layout.size()

            # TODO: MSTDL-1352 we can use memory element to fill the tensor.
            comptime for i in range(num_elements):
                comptime idx = Self.layout(i)

                comptime for j in range(Self.element_size):
                    comptime element_offset = Self.element_layout(j)
                    self.ptr[idx + element_offset] = val
        else:
            var num_elements = self.runtime_layout.size()

            for i in range(num_elements):
                var idx = self.runtime_layout(i)

                comptime if Self.element_layout.all_dims_known():
                    comptime for j in range(Self.element_size):
                        comptime element_offset = Self.element_layout(j)
                        self.ptr[
                            idx + Scalar[self.linear_idx_type](element_offset)
                        ] = val
                else:
                    for j in range(self.runtime_element_layout.size()):
                        var element_offset = self.runtime_element_layout(j)
                        self.ptr[idx + element_offset] = val
        return self

    def write_to(self, mut writer: Some[Writer]):
        """Format and write the tensor's contents to a writer.

        This method formats the tensor's contents and writes them to the
        provided writer. For 2D tensors, it formats the output in a 2D grid. For
        tensors of other ranks, it prints all values in column-major coordinate
        order.

        Args:
            writer: The writer instance to write the formatted output to.

        Example:

        ```mojo
        from layout import Layout, LayoutTensor

        def main() raises:
            var storage = Array[Float32, 2 * 3](uninitialized=True)
            var tensor = LayoutTensor[
                DType.float32,
                Layout([2, 3]),
            ](storage).fill(1.0)
            print(tensor)  # Internally calls `write_to` with a StringWriter
        ```

        Output for a 2x3 tensor:

        ```
        [[1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]]
        ```

        Notes:

        - For 2D tensors, the output is formatted as a 2D grid with rows and
            columns.
        - For tensors of other ranks, values are printed in column-major
            coordinate order.
        - Empty tensors (size 0) produce no output.
        - This method is used by the `__str__` method to convert the tensor to a
            string.
        - The formatting is designed for human readability rather than parsing.
        - For large tensors, the output may be truncated to avoid excessive
            output.
        """

        if self.runtime_layout.size() == 0:
            return

        @always_inline
        def is_2d_print(layout: Layout) -> Bool:
            return (
                len(layout) == 2
                and layout.shape[0].is_value()
                and layout.shape[1].is_value()
            )

        # The 2D print works only for layout shape (M, N).
        # Check both original and coalesced layouts so that (M, 1) and
        # ((M), (N)) can all be printed in 2D. Shapes like ((2, 2), 2) will be
        # printed elementwise.
        comptime if is_2d_print(Self.layout):
            _pretty_print_2d_tensor(self, writer)
            return
        elif is_2d_print(coalesce(Self.layout)):
            _pretty_print_2d_tensor(self.coalesce(), writer)
            return

        comptime layout_size = Self.layout.size()
        for i in range(self.runtime_layout.size()):
            var vec_offset = self.runtime_layout(i)
            var vec = SIMD[Self.dtype, Self.element_size]()

            comptime for idx in range(Self.element_size):
                comptime element_offset = self.element_layout(idx)
                vec[idx] = self.ptr.load(
                    vec_offset + Scalar[Self.linear_idx_type](element_offset)
                )

            writer.write(vec)
            if i != layout_size - 1:
                writer.write(" ")


@always_inline
def _pretty_print_2d_tensor[W: Writer](tensor: LayoutTensor, mut writer: W):
    comptime assert tensor.layout.rank() == 2

    var m_dim = tensor.runtime_layout.shape[0].value[0]
    var n_dim = tensor.runtime_layout.shape[1].value[0]
    for m in range(m_dim):
        for n in range(n_dim):
            writer.write(tensor[m, n], " ")
        if m < m_dim - 1:
            writer.write("\n")


def stack_allocation_like[
    layout: Layout,
    dtype: DType,
    *,
    address_space: AddressSpace,
    target_address_space: AddressSpace = .GENERIC,
](
    in_tensor: LayoutTensor[dtype, layout, address_space=address_space, ...],
) -> LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space=target_address_space,
    masked=in_tensor.masked,
]:
    """Create a stack-allocated tensor with the same layout as an existing
    tensor.

    This function creates a new tensor on the stack with the same layout, data
    type, and masking properties as the input tensor, but potentially with a
    different address space. This is useful for creating temporary tensors that
    match the structure of existing tensors.

    Parameters:
        layout: The layout of the tensor to allocate.
        dtype: The data type of the tensor elements.
        address_space: The address space of the input tensor.
        target_address_space: The address space for the new tensor. Defaults to
            GENERIC.

    Args:
        in_tensor: The input tensor to match the layout of.

    Returns:
        A new tensor allocated on the stack with the same layout as the input
        tensor.

    Example:

    ```mojo
    from layout import LayoutTensor, Layout
    from layout.layout_tensor import stack_allocation_like

    var global_tensor = LayoutTensor[
        DType.float32,
        Layout([10, 10]),
        MutAnyOrigin,
        address_space=.GLOBAL
    ].stack_allocation()

    var shared_tensor = stack_allocation_like[
        target_address_space=.SHARED
    ](global_tensor)
    ```

    Performance:

    - Creates a tensor on the stack, which is typically faster to allocate and
        access than heap-allocated memory.
    - Stack allocations have automatic lifetime management, reducing memory
        management overhead.
    - Stack size is limited, so be cautious with large tensor allocations.

    Notes:

    - The new tensor will have the same layout, data type, and masking properties
        as the input tensor.
    - The address space can be changed, which is useful for moving data between
        different memory regions (e.g., from global to shared memory).
    - Stack allocations are automatically freed when they go out of scope.
    - The function uses the stack_allocation method of the result tensor type.
    """
    return LayoutTensor[
        dtype,
        layout,
        MutAnyOrigin,
        address_space=target_address_space,
        masked=in_tensor.masked,
    ].stack_allocation()


struct ThreadScope(TrivialRegisterPassable, Writable):
    """Represents the scope of thread operations in GPU programming.

    This struct defines the scope at which thread operations are performed,
    particularly for operations like tensor distribution and synchronization.
    It provides two main scopes: `BLOCK` and `WARP`, which correspond to
    different levels of thread grouping in GPU programming models.

    Example:

    ```mojo
    from layout.layout_tensor import copy_dram_to_sram, ThreadScope

    # Distribute tensor at block level (all threads in block participate)
    copy_dram_to_sram[layout, thread_scope=ThreadScope.BLOCK](dst, src)

    # Distribute tensor at warp level (only threads in same warp participate)
    copy_dram_to_sram[layout, thread_scope=ThreadScope.WARP](dst, src)
    ```

    Performance:

    - WARP scope operations typically have lower synchronization overhead
        than BLOCK scope operations.
    - BLOCK scope operations allow coordination across all threads in a block,
        which is necessary for certain algorithms.
    - The choice of scope can significantly impact performance and correctness
        of parallel algorithms.

    Notes:

    - The appropriate scope depends on the specific algorithm and hardware.
    - WARP scope operations may be more efficient for operations that only
        require coordination within a warp.
    - BLOCK scope operations are necessary when threads from different warps
        need to coordinate.
    - The actual size of a warp or block is hardware-dependent.
    """

    var _value: Int32
    """The internal integer value representing the thread scope."""

    comptime BLOCK = Self(0)
    """Represents operations at the thread block level, where all threads in a
    block participate."""

    comptime WARP = Self(1)
    """Represents operations at the warp level, where only threads within the
    same warp participate."""

    def __init__(out self, value: Int):
        """Initialize a `ThreadScope` with the given integer value.

        Args:
            value: An integer representing the thread scope (0 for `BLOCK`,
                1 for `WARP`).
        """
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        """Compare two `ThreadScope` objects for equality.

        Args:
            other: Another `ThreadScope` object to compare with.

        Returns:
            True if the thread scopes are equal, False otherwise.
        """
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        """Compare two `ThreadScope` objects for inequality.

        Args:
            other: Another `ThreadScope` object to compare with.

        Returns:
            True if the thread scopes are not equal, False otherwise.
        """
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        """Write the `ThreadScope` as a human-readable string representation.

        Args:
            writer: The writer to write the string representation to.

        Aborts:
            If the thread scope has an invalid value.
        """
        if self == Self.BLOCK:
            return writer.write("BLOCK")
        if self == Self.WARP:
            return writer.write("WARP")
        abort("invalid ThreadScope entry")

    def __int__(self) -> Int:
        """Convert the `ThreadScope` to an integer value.

        Returns:
            The integer value of the thread scope (0 for BLOCK, 1 for WARP).
        """
        return Int(self._value)


@always_inline("nodebug")
def _get_worker_idx[
    thread_scope: ThreadScope, block_dim_count: Int = 1
]() -> Int:
    """
    Returns the worker index for the current thread scope.

    This function determines the index of the current worker (thread) based on the
    specified thread scope. If the scope is `BLOCK`, it returns the thread's index
    within the block (`thread_idx.x`). If the scope is `WARP`, it returns the lane
    ID within the warp (`lane_id()`).

    Parameters:
        thread_scope: The scope at which the worker index is determined.
        block_dim_count: The number of dimensions in the thread block.

    Returns:
        Int: The worker index within the specified scope.

    """

    comptime assert block_dim_count >= 1 and block_dim_count <= 3, (
        "block_dim_count = "
        + String(block_dim_count)
        + ". Thread blocks contain between 1 (x) and 3 (x,y,z) dimensions"
    )

    comptime if thread_scope == ThreadScope.BLOCK:
        comptime if block_dim_count == 1:
            return thread_idx.x
        elif block_dim_count == 2:
            return thread_idx.y * block_dim.x + thread_idx.x
        else:
            return (
                thread_idx.z * block_dim.y * block_dim.x
                + thread_idx.y * block_dim.x
                + thread_idx.x
            )
    else:
        return lane_id()


@always_inline("nodebug")
def _copy_dram_to_sram_validate_args(
    dst: LayoutTensor[mut=True, ...], src: LayoutTensor
):
    """Validate arguments for DRAM to SRAM copy operations.

    This internal function validates that the source and destination tensors
    have compatible properties for a DRAM to SRAM copy operation. It checks
    data types and address spaces to ensure the copy operation can be performed
    correctly.

    Constraints:
        - Source and destination tensors must have the same data type.
        - Source tensor must be in GENERIC or GLOBAL address space.
        - Destination tensor must be in SHARED address space.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory
            (DRAM).

    Notes:

    - This is an internal helper function used by copy_dram_to_sram.
    - The function enforces that the source and destination tensors have
        the same data type.
    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - These constraints ensure that the copy operation follows the expected
        memory hierarchy flow from slower global memory to faster shared memory.
    """
    comptime assert (
        dst.dtype == src.dtype
    ), "src dtype and dst dtype must be the same."

    comptime assert src.address_space in (
        AddressSpace.GENERIC,
        AddressSpace.GLOBAL,
    ), "src address space must be GENERIC or GLOBAL."

    comptime assert (
        dst.address_space == .SHARED
    ), "dst address space must be SHARED."


@always_inline("nodebug")
def copy_dram_to_sram[
    src_thread_layout: Layout,
    dst_thread_layout: Layout = src_thread_layout,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor[mut=True, ...], src: LayoutTensor[mut=False, ...]):
    """Synchronously copy data from DRAM (global memory) to SRAM (shared memory)
    in a GPU context.

    This function performs a synchronous copy operation from global memory
    (DRAM) to shared memory (SRAM) in a GPU context, distributing the workload
    across multiple threads for parallel execution. It uses thread affinity
    mapping to ensure efficient work distribution and supports vectorized memory
    operations for optimal performance.

    Constraints:
        - Source and destination tensors must have the same data type.
        - Source tensor must be in GENERIC or GLOBAL address space.
        - Destination tensor must be in SHARED address space.
        - For non-masked tensors, the fragment sizes must match.

    Parameters:
        src_thread_layout: Layout defining how threads are organized for the
            source tensor. This determines how the workload is distributed among
            threads.
        dst_thread_layout: Layout defining how threads are organized for the
            destination tensor. Defaults to the same as `src_thread_layout` if
            not specified.
        swizzle: Optional swizzling function to rearrange the destination
            indices, which can improve memory access patterns and reduce bank
            conflicts.
        num_threads: Total number of threads in the thread block. Threads
            beyond `src_thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Scope at which thread operations are performed (`BLOCK` or
            `WARP`). Defaults to `ThreadScope.BLOCK`, where all threads in a
            block participate.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory
            (DRAM).

    Performance:

    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Supports vectorized loads and stores for better memory throughput.
    - Can use swizzling to optimize memory access patterns and reduce bank
        conflicts.
    - Thread affinity mapping ensures efficient work distribution.
    - For masked tensors, performs bounds checking to handle edge cases
        correctly.

    Notes:

    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - Both tensors must have the same data type.
    - This function is synchronous, meaning all threads must complete their
        copy operations before proceeding.
    - For optimal performance, the thread layouts should match the memory
        access patterns of the tensors.
    - This function is particularly useful in GPU kernels for loading data
        from global memory to shared memory for faster access.
    """
    _copy_dram_to_sram_validate_args(dst, src)

    comptime num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var src_fragments = src.distribute[src_thread_layout](worker_idx)
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        worker_idx
    )

    comptime simd_width = simd_width_of[dst.dtype]()
    comptime src_align = align_of[SIMD[src.dtype, simd_width]]()
    comptime dst_align = align_of[SIMD[dst.dtype, simd_width]]()

    comptime coalesce_src_element_layout = coalesce(src.element_layout)
    comptime coalesce_dst_element_layout = coalesce(dst.element_layout)

    comptime is_scalar = not (
        src.element_layout.all_dims_known()
        and coalesce_src_element_layout.rank() == 1
        and coalesce_src_element_layout.stride[0] == 1
        and coalesce_dst_element_layout.rank() == 1
        and coalesce_dst_element_layout.stride[0] == 1
    )

    var stride: Int
    comptime if not src_fragments.masked or is_scalar:
        comptime assert (
            dst_fragments.layout.size() == src_fragments.layout.size()
        ), (
            "Fragment size mismatch: dst fragments size ("
            + String(dst_fragments.layout.size())
            + ") does not match src fragments size ("
            + String(src_fragments.layout.size())
            + ")"
        )

        dst_fragments.copy_from(src_fragments)
    else:
        comptime num_stores_per_thread = dst_fragments.layout.size()
        comptime static_stride = src.layout.stride[0].value()

        comptime if src.layout.all_dims_known():
            stride = static_stride
        else:
            stride = src.runtime_layout.stride.value[0]
        var src_frag_offset = src_fragments.distance(src.ptr)

        # NOTE: This can be a negative number, so we cannot use unsigned type
        # in layout tensor.
        var src_idx_bound = (
            Scalar[src.linear_idx_type](src.dim[0]() * stride) - src_frag_offset
        ).cast[src_fragments.linear_idx_type]()

        comptime for i in range(num_stores_per_thread):
            comptime src_static_idx = src_fragments.layout(i)

            comptime dst_idx = dst_fragments.layout(i)

            var src_idx: Scalar[src_fragments.linear_idx_type]

            comptime if src.layout.all_dims_known():
                src_idx = Scalar[src.linear_idx_type](src_static_idx)
            else:
                src_idx = src_fragments.runtime_layout(i)

            if src_idx < src_idx_bound:
                var src_vec = (src.ptr).load[
                    width=simd_width, alignment=src_align
                ](Scalar[src.linear_idx_type](Int(src_frag_offset)) + src_idx)
                dst_fragments.ptr.store[alignment=dst_align](
                    dst_idx, src_vec.cast[dst.dtype]()
                )


@always_inline("nodebug")
def copy_dram_to_sram[
    src_thread_layout: Layout,
    dst_thread_layout: Layout = src_thread_layout,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor[mut=True, ...], src_iter: LayoutTensorIter, bound: Int):
    """Efficiently copy data from global memory (DRAM) to shared memory (SRAM)
    on AMD GPUs.

    This function implements an optimized memory transfer operation specifically
    for AMD GPU architectures. It utilizes the hardware's `buffer_load`
    intrinsic to efficiently transfer data while handling bounds checking. The
    function distributes the copy operation across multiple threads for maximum
    throughput.

    Parameters:
        src_thread_layout: The layout used to distribute the source tensor
            across threads. This determines how the workload is divided among
            participating threads.
        dst_thread_layout: The layout used to distribute the destination tensor
            across threads. Defaults to the same layout as `src_thread_layout`.
        swizzle: Optional swizzling pattern to apply when distributing the
            destination tensor. This can improve memory access patterns and
            reduce bank conflicts. Defaults to None (no swizzling).
        num_threads: Total number of threads in the thread block. Threads
            beyond `src_thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor in shared memory (SRAM).
        src_iter: The source tensor iterator in global memory (DRAM) to be
            copied.
        bound: The bound of the source tensor iterator.
    """
    comptime assert is_amd_gpu(), "This function is only supported on AMD GPUs."

    var src_tensor = src_iter[].vectorize[
        dst.element_layout.shape[0].value(), dst.element_layout.shape[1].value()
    ]()
    _copy_dram_to_sram_validate_args(dst, src_tensor)

    comptime num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var src_fragments = src_tensor.distribute[src_thread_layout](worker_idx)
    var dst_fragments = dst.distribute[dst_thread_layout, swizzle=swizzle](
        worker_idx
    )

    comptime simd_width = src_tensor.element_layout.size()
    comptime dst_align = align_of[SIMD[dst.dtype, simd_width]]()

    comptime num_stores_per_thread = dst_fragments.layout.size()
    var buffer = make_amd_buffer_resource(src_iter, bound)
    var src_frag_offset = src_fragments.distance(src_tensor.ptr) + Scalar[
        src_iter.linear_idx_type
    ](Int(src_iter.offset))

    comptime for i in range(num_stores_per_thread):
        var src_frag_idx: Scalar[src_fragments.linear_idx_type]

        comptime if src_tensor.layout.all_dims_known():
            comptime frag_layout = src_fragments.layout(i)
            src_frag_idx = Scalar[src_iter.linear_idx_type](frag_layout)
        else:
            src_frag_idx = src_fragments.runtime_layout(i)

        comptime dst_frag_idx = dst_fragments.layout(i)
        dst_fragments.ptr.store[alignment=dst_align](
            dst_frag_idx,
            buffer.load[src_tensor.dtype, simd_width](
                Int32(src_frag_offset),
                scalar_offset=Int32(src_frag_idx),
            ).cast[dst.dtype](),
        )


@always_inline("nodebug")
def cp_async_k_major[
    dtype: DType,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    dst: LayoutTensor[
        mut=True,
        dtype,
        _,
        address_space=gpu_memory.AddressSpace.SHARED,
        ...,
    ],
    src: LayoutTensor[
        dtype, _, address_space=gpu_memory.AddressSpace.GENERIC, ...
    ],
):
    """Asynchronously copy data from DRAM to SRAM using TMA (Tensor Memory
    Accelerator) with K-major layout.

    This function performs an asynchronous copy operation from global memory
    (DRAM) to shared memory (SRAM) using NVIDIA's Tensor Memory Accelerator
    (TMA) hardware. It optimizes for K-major memory access patterns, which is
    particularly beneficial for certain tensor operations like matrix
    multiplications where the inner dimension (K) is accessed contiguously.

    The function automatically determines the optimal tile size and thread
    distribution based on the tensor shapes and hardware capabilities,
    leveraging TMA's efficient memory transfer mechanisms.

    Constraints:
        - Requires NVIDIA GPUs with TMA support (compute capability 9.0+).
        - Source tensor must be in GENERIC or GLOBAL address space.
        - Destination tensor must be in SHARED address space.
        - Both tensors must have the same data type.
        - Source and destination tensors must be 2D.

    Parameters:
        dtype: The data type of the tensor elements.
        eviction_policy: The cache eviction policy to use. Default is `CacheEviction.EVICT_NORMAL`.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory
            (DRAM).

    Performance:

    - Uses TMA hardware acceleration for optimal memory transfer performance.
    - Optimizes for K-major access patterns, which can significantly improve
        performance for certain tensor operations like matrix multiplications.
    - Performs asynchronous transfers, allowing computation to overlap with
        memory operations.
    - Automatically determines optimal tile sizes based on tensor dimensions.
    - Uses hardware-accelerated swizzling to reduce shared memory bank
        conflicts.

    Notes:

    - This function requires NVIDIA GPUs with TMA support (compute capability
        9.0+).
    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - Both tensors must have the same data type.
    - This function is asynchronous, so you must call
        [`async_copy_wait_all()`](/api/mojo/max/gpu/memory/memory/async_copy_wait_all/)
        or
        [`async_copy_wait_group()`](/api/mojo/max/gpu/memory/memory/async_copy_wait_group/)
        to ensure the copy has completed before using the data.
    - K-major layout is particularly beneficial for matrix multiplication
        operations where the inner dimension (K) is accessed contiguously.
    """
    comptime dst_layout = dst.layout

    comptime src_layout = src.layout
    comptime src_shape0 = src_layout.shape[0].value()
    comptime src_shape1 = src_layout.shape[1].value()

    comptime tile_desc_shape = _tma_desc_tile_shape[
        dtype,
        2,
        Index(src_shape0, src_shape1),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
    ]()
    comptime desc_shape0 = tile_desc_shape[0]
    comptime desc_shape1 = tile_desc_shape[1]
    comptime desc_size = desc_shape0 * desc_shape1
    comptime desc_layout = Layout.row_major(desc_shape0, desc_shape1)

    comptime assert (
        desc_shape0 == src_shape0
    ), "k-major desc layout shouldn't alter 1st dim"

    comptime num_tiles = src_shape1 // desc_shape1
    comptime simd_size = simd_width_of[dtype]()
    # single warp group
    comptime thread_layout = Layout.row_major(
        128 * simd_size // desc_shape1, desc_shape1 // simd_size
    )

    comptime for tile_id in range(num_tiles):
        var src_tile = src.tile[desc_shape0, desc_shape1](0, tile_id)
        var dst_tile = LayoutTensor[
            dtype, desc_layout, address_space=gpu_memory.AddressSpace.SHARED
        ](dst.ptr + tile_id * desc_size)

        copy_dram_to_sram_async[
            thread_layout, swizzle=True, eviction_policy=eviction_policy
        ](
            dst_tile.vectorize[1, simd_size](),
            src_tile.vectorize[1, simd_size](),
        )


@always_inline("nodebug")
def copy_dram_to_sram[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor[mut=True, ...], src_iter: LayoutTensorIter, bound: Int):
    """Synchronously copy data from DRAM to SRAM using a unified thread layout
    for AMD GPUs.

    This is a convenience wrapper around the more general `copy_dram_to_sram()`
    function that uses the same layout for both source and destination tensors.
    It's specifically designed for AMD GPUs where the buffer_load intrinsic
    requires the original base tensor.

    Parameters:
        thread_layout: Layout defining how threads are organized for both source
            and destination. This determines how the workload is distributed
            among threads.
        swizzle: Optional swizzling function to rearrange the destination
            indices, which can improve memory access patterns and reduce bank
            conflicts.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Scope at which thread operations are performed (`BLOCK` or
            `WARP`). Defaults to `BLOCK`, where all threads in a block
            participate.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src_iter: The source tensor iterator, which must be in global or generic
            memory (DRAM).
        bound: The bound of the source tensor iterator.

    Performance:

    - Simplifies API usage when the same thread layout is appropriate for both
        source and destination tensors.
    - Optimized for AMD GPUs using buffer_load intrinsics for efficient memory
        transfers.
    - Distributes the copy workload across multiple threads for parallel
        execution.

    Notes:

    - This function is only supported on AMD GPUs.
    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - Both tensors must have the same data type.
    """
    copy_dram_to_sram[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
        thread_scope=thread_scope,
    ](dst, src_iter, bound)


@always_inline("nodebug")
def copy_dram_to_sram[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor[mut=True, ...], src: LayoutTensor):
    """Synchronously copy data from DRAM to SRAM using a unified thread layout.

    This is a convenience wrapper around the more general `copy_dram_to_sram()`
    function that uses the same layout for both source and destination tensors.
    It simplifies the API for the common case where the same thread distribution
    pattern works well for both tensors.

    Parameters:
        thread_layout: Layout defining how threads are organized for both source
            and destination. This determines how the workload is distributed
            among threads.
        swizzle: Optional swizzling function to rearrange the destination
            indices, which can improve memory access patterns and reduce bank
            conflicts.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Scope at which thread operations are performed
                (`BLOCK` or `WARP`). Defaults to `ThreadScope.BLOCK`, where all
                threads in a block participate.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory
            (DRAM).

    Performance:

    - Simplifies API usage when the same thread layout is appropriate for both
        source and destination tensors.
    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Supports vectorized loads and stores for better memory throughput.
    - Can use swizzling to optimize memory access patterns and reduce bank
        conflicts.

    Notes:

    - The source tensor must be in `GENERIC` or `GLOBAL` address space (DRAM).
    - The destination tensor must be in `SHARED` address space (SRAM).
    - Both tensors must have the same data type.
    - This function is synchronous, meaning all threads must complete their
        copy operations before proceeding.
    """
    copy_dram_to_sram[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
        thread_scope=thread_scope,
    ](dst, src)


@always_inline("nodebug")
def copy_dram_to_sram_async[
    src_thread_layout: Layout,
    dst_thread_layout: Layout,
    swizzle: Bool = False,
    fill: Fill = Fill.NONE,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = src_thread_layout.size(),
    block_dim_count: Int = 1,
](dst: LayoutTensor[mut=True, ...], src: LayoutTensor):
    """Asynchronously copy data from DRAM (global memory) to SRAM (shared
    memory) in a GPU context.

    This function performs an asynchronous copy operation from global memory
    (DRAM) to shared memory (SRAM) in a GPU context, using NVIDIA's cp.async
    hardware mechanism. It distributes the workload across multiple threads and
    allows computation to overlap with memory transfers for improved
    performance.

    Constraints:
        - Requires NVIDIA GPUs with cp.async support (compute capability 8.0+).
        - Source tensor must be in `GENERIC` or `GLOBAL` address space.
        - Destination tensor must be in `SHARED` address space.
        - Both tensors must have the same data type.
        - Element size must be 4, 8, or 16 bytes.

    Parameters:
        src_thread_layout: Layout defining how threads are organized for the
            source tensor. This determines how the workload is distributed among
            threads.
        dst_thread_layout: Layout defining how threads are organized for the
            destination tensor.
        swizzle: Whether to apply swizzling to the destination indices to
            reduce bank conflicts. Defaults to False.
        fill: Fill policy for handling out-of-bounds accesses. Options
            include:
            - `Fill.NONE`: No special handling (default).
            - `Fill.ZERO`: Fill out-of-bounds elements with zeros.
        eviction_policy: Cache eviction policy for the source data. Options
            include:
            - `CacheEviction.EVICT_NORMAL`: Normal eviction (default).
            - `CacheEviction.EVICT_FIRST`: Evict data after first use.
            - `CacheEviction.EVICT_LAST`: Keep data in cache until last use.
        num_threads: Total number of threads in the thread block. Threads
            beyond `src_thread_layout.size()` will be disabled and not
            participate in the copy operation.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in global or generic memory (DRAM).

    Performance:

    - Performs asynchronous transfers, allowing computation to overlap with
        memory operations.
    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Can use swizzling to optimize memory access patterns and reduce bank
        conflicts.
    - Supports different cache eviction policies to optimize memory hierarchy
        usage.
    - For masked tensors, performs bounds checking to handle edge cases
        correctly.

    Notes:

    - This function requires NVIDIA GPUs with `cp.async` support (compute
        capability 8.0+).
    - The source tensor must be in GENERIC or GLOBAL address space (DRAM).
    - The destination tensor must be in SHARED address space (SRAM).
    - Both tensors must have the same data type.
    - This function is asynchronous, so you must call
        [`async_copy_wait_all()`](/api/mojo/max/gpu/memory/memory/async_copy_wait_all/)
        or
        [`async_copy_wait_group()`](/api/mojo/max/gpu/memory/memory/async_copy_wait_group/)
        to ensure the copy has completed before using the data.
    - The maximum size of each element that can be copied is 16 bytes.
    """
    comptime assert src.address_space in (
        AddressSpace.GENERIC,
        AddressSpace.GLOBAL,
    ), "src address space must be GENERIC or GLOBAL."

    comptime assert (
        dst.address_space == .SHARED
    ), "dst address space must be SHARED."

    comptime assert src_thread_layout.size() == dst_thread_layout.size(), (
        "src thread layout size "
        + String(src_thread_layout.size())
        + " does not match dst thread layout size "
        + String(dst_thread_layout.size())
    )

    comptime num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[ThreadScope.BLOCK, block_dim_count]()

    # We know at compile time that only partial threads copy based on the size
    # of input tensors. Return if current thread doesn't have work.
    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    comptime row_size = dst.stride[0]()
    # See make_ldmatrix_swizzle in Swizzle.mojo for `conflict_ways`.
    # TODO: use the above when MOCO-1048 is fixed.
    comptime bytes_32_banks = 128
    comptime conflict_ways = min(
        8 * row_size * size_of[dst.dtype]() // bytes_32_banks, 8
    )
    comptime assert (
        swizzle and (conflict_ways in (4, 8))
    ) or not swizzle, "Only support swizzle for 4 or 8 ways conflict."

    comptime assert (
        swizzle and row_size in (16, 32, 64, 128, 256, 512)
    ) or not swizzle, (
        "Only support 2^4-2^9 elements per row in shared memory tile for"
        " async copy with swizzling."
    )

    comptime swizzle_option = None if not swizzle else (
        Optional[Swizzle](
            make_ldmatrix_swizzle[
                dst.dtype, row_size, log2_floor(dst_fragments.element_size)
            ]()
        )
    )

    var src_fragments = src.distribute[src_thread_layout](worker_idx)
    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var dst_frag_offset = dst_fragments.distance(dst.ptr) if swizzle else 0

    comptime if not src_fragments.masked:
        dst_fragments.copy_from_async[
            swizzle=swizzle_option, eviction_policy=eviction_policy
        ](
            src_fragments,
            base_offset=Scalar[dst.linear_idx_type](Int(dst_frag_offset)),
        )
    else:
        var src_frag_offset = src_fragments.distance(src.ptr)

        # Stride between two rows
        comptime static_row_stride = Scalar[src_fragments.linear_idx_type](
            src.layout.stride[0].value()
        )
        var row_stride = static_row_stride

        comptime if src.layout.stride[0].value() == UNKNOWN_VALUE:
            row_stride = Scalar[src_fragments.linear_idx_type](
                src.runtime_layout.stride.value[0]
            )

        var src_idx_bound = (
            Scalar[src_fragments.linear_idx_type](src.dim[0]()) * row_stride
            - src_frag_offset
        )

        dst_fragments.copy_from_async[
            is_masked=True,
            swizzle=swizzle_option,
            eviction_policy=eviction_policy,
        ](
            src_fragments,
            src_idx_bound=src_idx_bound,
            base_offset=dst_frag_offset,
        )


@always_inline("nodebug")
def copy_dram_to_sram_async[
    thread_layout: Layout,
    swizzle: Bool = False,
    masked: Bool = False,
    fill: Fill = Fill.NONE,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = thread_layout.size(),
    block_dim_count: Int = 1,
](dst: LayoutTensor[mut=True, ...], src: LayoutTensor):
    """
    Asynchronous copy from DRAM to SRAM with thread affinity mapping.

    This function performs an asynchronous memory transfer from DRAM (global
    memory) to SRAM (shared memory) using the specified thread layout for
    distribution.

    Parameters:
        thread_layout: The layout used to distribute work across threads.
        swizzle: Whether to apply memory access swizzling for better performance.
        masked: Whether the copy operation should use masking.
        fill: Fill policy for uninitialized memory regions.
        eviction_policy: Cache eviction policy to use during the transfer.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` will be disabled and not
            participate in the copy operation.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: Destination tensor in SRAM.
        src: Source tensor in DRAM.

    Notes:

    This is a convenience wrapper around the more general
    `copy_dram_to_sram_async()` function, using the same thread layout for
    both source and destination.
    """
    copy_dram_to_sram_async[
        src_thread_layout=thread_layout,
        dst_thread_layout=thread_layout,
        swizzle=swizzle,
        eviction_policy=eviction_policy,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](dst, src)


comptime binary_op_type = def[dtype: DType, width: SIMDLength](
    lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]
) thin -> SIMD[dtype, width]
"""
Type alias for binary operations on SIMD vectors.

This type represents a function that takes two SIMD vectors of the same type and
width and returns a SIMD vector of the same type and width.

Args:
    dtype: The data type of the SIMD vector elements.
    width: The width of the SIMD vector.
    lhs: Left-hand side SIMD vector operand.
    rhs: Right-hand side SIMD vector operand.

Returns:
    A SIMD vector containing the result of the binary operation.
"""


@always_inline("nodebug")
def copy_sram_to_dram[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    block_dim_count: Int = 1,
    binary_op: Optional[binary_op_type] = None,
](dst: LayoutTensor[mut=True, ...], src: LayoutTensor):
    """Synchronously copy data from SRAM (shared memory) to DRAM (global
    memory).

    This function performs a synchronous memory transfer from SRAM (shared
    memory) to DRAM (global memory) using the specified thread layout for
    workload distribution. It supports optional swizzling for optimized memory
    access patterns and binary operations for combining data during the
    transfer.

    Constraints:
        - Source tensor must be in SHARED address space with a static layout.
        - Destination tensor must be in GENERIC or GLOBAL address space.
        - For type conversion, only FP32 to half-precision is supported.
        - For vectorized copy with type conversion, both tensors must have
          element layouts matching the SIMD width of the destination type.

    Parameters:
        thread_layout: Layout defining how threads are organized for both source
            and destination. This determines how the workload is distributed
            among threads.
        swizzle: Optional swizzling function to rearrange the source indices,
            which can improve memory access patterns and reduce bank conflicts.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` will be disabled and not
            participate in the copy operation.
        block_dim_count: The number of dimensions in the thread block.
        binary_op: Optional binary operation to apply during the copy, combining
            source data with existing destination data.

    Args:
        dst: The destination tensor, which must be in global or generic memory
            (DRAM).
        src: The source tensor, which must be in shared memory (SRAM).

    Performance:

    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Supports vectorized loads and stores for better memory throughput.
    - Can use swizzling to optimize memory access patterns.
    - Supports binary operations to combine data during transfer (e.g., for
        reduction operations).

    Notes:

    - The source tensor must be in `SHARED` address space (SRAM).
    - The destination tensor must be in `GENERIC` or `GLOBAL` address space
        (DRAM).
    - Supports FP32 to half-precision downcast during copy if needed.
    - Handles masked tensors with proper bounds checking.
    - This function is synchronous, meaning all threads must complete their
        copy operations before proceeding.
    """
    comptime assert dst.address_space in (
        AddressSpace.GENERIC,
        AddressSpace.GLOBAL,
    ), "dst address space must be GENERIC or GLOBAL."

    comptime assert (
        src.address_space == .SHARED
    ), "src address space must be SHARED."

    comptime assert (
        src.layout.all_dims_known()
    ), "Shared memory must have static layout"

    comptime num_busy_threads = thread_layout.size()
    var worker_idx = _get_worker_idx[ThreadScope.BLOCK, block_dim_count]()

    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var src_fragments = src.distribute[thread_layout](worker_idx)
    var dst_fragments = dst.distribute[thread_layout](worker_idx)

    var stride: Int
    # TODO: copy_from only allows static layout
    comptime if src.dtype == dst.dtype and not swizzle and not dst.masked:
        dst_fragments.copy_from(src_fragments)
    else:
        comptime assert src.dtype == dst.dtype or (
            src.dtype == .float32 and dst.dtype.is_half_float()
        ), "Only support FP32 -> half precision downcast during copy."

        comptime simd_size = simd_width_of[dst.dtype]()
        # TODO: generalize the copy to non-scalar case if possible.
        comptime assert (
            src.element_layout.size() == simd_size
            and dst.element_layout.size() == simd_size
        ), "Only FP32 -> half precision downcast for vectorized copy."

        comptime src_align = align_of[
            SIMD[src.dtype, simd_width_of[src.dtype]()]
        ]()
        comptime dst_align = align_of[SIMD[dst.dtype, simd_size]]()

        var src_frag_offset = src_fragments.distance(src.ptr)

        comptime num_stores_per_thread = dst_fragments.layout.size()

        comptime if not dst_fragments.masked:
            comptime for i in range(num_stores_per_thread):
                comptime src_idx = src_fragments.layout(i)
                comptime dst_idx = dst_fragments.layout(i)
                var swizzled_idx = src_frag_offset + Scalar[
                    src.linear_idx_type
                ](src_idx)

                comptime if swizzle:
                    comptime swizzle_fn = swizzle.value()
                    comptime src_idx_base = src_idx % swizzle_fn.size()
                    comptime src_idx_diff = src_idx - src_idx_base
                    # `src_frag_offset + src_idx_base` should be a value already seen
                    # in the unrolled loop. Hopefully compiler can eliminate the duplicated
                    # xor computation.
                    swizzled_idx = swizzle_fn(
                        src_frag_offset
                        + Scalar[src.linear_idx_type](src_idx_base)
                    ) + Scalar[src.linear_idx_type](src_idx_diff)

                var src_vec = src.ptr.load[
                    width=simd_size, alignment=src_align
                ](swizzled_idx).cast[dst.dtype]()

                comptime if binary_op:
                    comptime binop = binary_op.value()
                    var dst_vec = dst_fragments.ptr.load[
                        width=simd_size, alignment=dst_align
                    ](dst_idx)
                    src_vec = binop(src_vec, dst_vec)

                dst_fragments.ptr.store[alignment=dst_align](dst_idx, src_vec)
        else:
            comptime static_stride = dst.layout.stride[0].value()

            comptime if dst.layout.all_dims_known():
                stride = static_stride
            else:
                stride = dst.runtime_layout.stride.value[0]
            var dst_frag_offset = dst_fragments.distance(dst.ptr)
            var dst_idx_bound = (
                Scalar[dst.linear_idx_type](dst.dim[0]() * stride)
                - dst_frag_offset
            ).cast[dst_fragments.linear_idx_type]()

            comptime for i in range(num_stores_per_thread):
                comptime src_idx = src_fragments.layout(i)

                comptime dst_uint_dtype = _get_unsigned_type(
                    dst_fragments.layout, dst_fragments.address_space
                )
                comptime dst_static_idx = dst_fragments.layout(i)

                var dst_idx: Scalar[dst_fragments.linear_idx_type]

                comptime if dst.layout.all_dims_known():
                    dst_idx = Scalar[dst.linear_idx_type](dst_static_idx)
                else:
                    dst_idx = dst_fragments.runtime_layout(i)

                var swizzled_idx = src_frag_offset + Scalar[
                    src.linear_idx_type
                ](src_idx)

                comptime if swizzle:
                    comptime swizzle_fn = swizzle.value()
                    comptime src_idx_base = src_idx % swizzle_fn.size()
                    comptime src_idx_diff = src_idx - src_idx_base
                    # `src_frag_offset + src_idx_base` should be a value already seen
                    # in the unrolled loop. Hopefully compiler can eliminate the duplicated
                    # xor computation.
                    swizzled_idx = swizzle_fn(
                        src_frag_offset
                        + Scalar[src.linear_idx_type](src_idx_base)
                    ) + Scalar[src.linear_idx_type](src_idx_diff)

                if dst_idx < dst_idx_bound:
                    var src_vec = (
                        (src.ptr)
                        .load[width=simd_size, alignment=src_align](
                            swizzled_idx
                        )
                        .cast[dst.dtype]()
                    )

                    comptime if binary_op:
                        comptime binop = binary_op.value()
                        var dst_vec = dst_fragments.ptr.load[
                            width=simd_size, alignment=dst_align
                        ](dst_idx)
                        src_vec = binop(src_vec, dst_vec)

                    dst_fragments.ptr.store[alignment=dst_align](
                        dst_idx, src_vec
                    )


@always_inline("nodebug")
def copy_sram_to_local[
    src_warp_layout: Layout,
    axis: Optional[Int] = None,
](dst: LayoutTensor[mut=True, ...], src: LayoutTensor):
    """Synchronously copy data from SRAM (shared memory) to local memory.

    This function performs a synchronous memory transfer from SRAM (shared
    memory) to local memory (registers) using the specified thread layout for
    workload distribution.

    Constraints:
        - The source tensor must be in SHARED address space (SRAM).
        - The destination tensor must be in LOCAL address space (registers).
        - Both tensors must have the same data type.

    Parameters:
        src_warp_layout: Layout defining how threads are organized for the
            source tensor. This determines how the workload is distributed among
            threads.
        axis: Optional parameter specifying which axis to distribute along.
            When provided, distribution happens along the specified axis.
            When None (default), distribution uses the standard layout pattern.

    Args:
        dst: The destination tensor, which must be in local memory (registers).
        src: The source tensor, which must be in shared memory (SRAM).

    Performance:

    - Distributes the copy workload across multiple threads for parallel
        execution.
    - Optimized for transferring data from shared memory to registers.
    - Supports optional axis-specific distribution for specialized access
        patterns.
    """
    comptime assert (
        dst.dtype == src.dtype
    ), "dst dtype must be the same as src dtype."

    comptime assert (
        src.address_space == .SHARED
    ), "src address space must be SHARED."

    comptime assert (
        dst.address_space == .LOCAL
    ), "dst address space must be LOCAL."

    comptime if axis:
        var src_fragments = src.distribute[src_warp_layout, axis=axis.value()](
            thread_idx.x
        )
        dst.copy_from(src_fragments)
    else:
        var src_fragments = src.distribute[src_warp_layout](thread_idx.x)
        dst.copy_from(src_fragments)


@always_inline("nodebug")
def _copy_local_to_dram_validate_args(dst: LayoutTensor, src: LayoutTensor):
    comptime assert (
        src.address_space == .LOCAL
    ), "src address space must be LOCAL."

    comptime assert dst.address_space in (
        AddressSpace.GENERIC,
        AddressSpace.GLOBAL,
    ), "dst address space must be GENERIC or GLOBAL."


@always_inline("nodebug")
def copy_local_to_dram[
    dst_thread_layout: Layout,
    num_threads: Int = dst_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor[mut=True, ...], src: LayoutTensor):
    """Efficiently copy data from registers (LOCAL) to global memory (DRAM).

    This function implements a high-performance memory transfer operation from
    register memory to global memory. It distributes the copy operation across
    multiple threads for maximum throughput while handling bounds checking for
    safety.

    Constraints:
        - The source tensor must be in LOCAL address space (registers).
        - The destination tensor must be in GENERIC or GLOBAL address space (DRAM).
        - Both tensors must have compatible data types.

    Parameters:
        dst_thread_layout: The layout used to distribute the destination tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads in the thread block. Threads
            beyond `dst_thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor in global memory (DRAM).
        src: The source tensor in register memory (LOCAL) to be copied.
    """
    _copy_local_to_dram_validate_args(dst, src)

    comptime num_busy_threads = dst_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var stride: Int
    comptime if not dst_fragments.masked:
        dst_fragments.copy_from(src)
    else:
        var dst_frag_offset = dst_fragments.distance(dst.ptr)
        comptime static_stride = dst.layout.stride[0].value()

        comptime if dst.layout.all_dims_known():
            stride = static_stride
        else:
            stride = dst.runtime_layout.stride.value[0]
        var dst_idx_bound = (
            Scalar[dst.linear_idx_type](dst.dim[0]() * stride) - dst_frag_offset
        ).cast[dst_fragments.linear_idx_type]()

        comptime num_stores_per_thread = dst_fragments.layout.size()

        comptime for i in range(num_stores_per_thread):
            comptime src_idx = src.layout(i)
            comptime dst_uint_dtype = _get_unsigned_type(
                dst_fragments.layout, dst_fragments.address_space
            )
            comptime dst_static_idx = dst_fragments.layout(i)

            var dst_idx: Scalar[dst_fragments.linear_idx_type]

            comptime if dst_fragments.layout.all_dims_known():
                dst_idx = Scalar[dst.linear_idx_type](dst_static_idx)
            else:
                dst_idx = dst_fragments.runtime_layout(i)

            if dst_idx < dst_idx_bound:
                var src_element = Element[index_type=src.linear_idx_type].load(
                    src.ptr + src_idx,
                    src.runtime_element_layout,
                )
                comptime dst_element_type = Element[
                    dst.dtype, dst.element_layout, dst.linear_idx_type
                ]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data.cast[dst.dtype]()
                    )
                ).store(dst_fragments.ptr + dst_idx)


@always_inline("nodebug")
def _copy_local_to_dram_static_row_major[
    dst_type: DType
](
    src: LayoutTensor,
    dst_fragments: LayoutTensor,
    dst_frag_offset: Int32,
    buffer: AMDBufferResource,
):
    comptime M = dst_fragments.shape[0]()
    comptime N = dst_fragments.shape[1]()

    comptime for i in range(M):
        comptime for j in range(N):
            comptime idx = Layout.col_major(M, N)([i, j])
            comptime src_frag_idx = src.layout(idx)
            comptime dst_frag_idx = Int32(dst_fragments.layout(idx))

            var src_element = Element[index_type=src.linear_idx_type].load(
                src.ptr + src_frag_idx,
                src.runtime_element_layout,
            )

            buffer.store(
                dst_frag_offset + dst_frag_idx,
                src_element.element_data.cast[dst_type](),
            )


@always_inline("nodebug")
def _copy_local_to_dram[
    dst_thread_layout: Layout,
    num_threads: Int = dst_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](
    dst: LayoutTensor[mut=True, ...],
    src: LayoutTensor,
    buffer: AMDBufferResource,
):
    comptime assert is_amd_gpu(), "This function is only supported on AMD GPUs."

    _copy_local_to_dram_validate_args(dst, src)

    comptime num_busy_threads = dst_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var base_ptr = buffer.get_base_ptr()
    var offset = (Int(dst.ptr) - base_ptr) // size_of[dst.dtype]()
    var dst_frag_offset = dst_fragments.distance(dst.ptr) + Scalar[
        dst.linear_idx_type
    ](offset)

    comptime dst_element_stride = dst_fragments.element_layout.stride[1].value()

    comptime if dst_element_stride == 1 and dst_fragments.layout.all_dims_known():
        _copy_local_to_dram_static_row_major[dst.dtype](
            src,
            dst_fragments,
            Int32(dst_frag_offset),
            buffer,
        )
    else:
        comptime num_stores_per_thread = dst_fragments.layout.size()

        comptime for i in range(num_stores_per_thread):
            comptime src_idx = src.layout(i)
            comptime dst_static_idx = dst_fragments.layout(i)
            var dst_idx = dst_frag_offset

            comptime if dst_fragments.layout.all_dims_known():
                dst_idx += Scalar[dst.linear_idx_type](dst_static_idx)
            else:
                dst_idx += dst_fragments.runtime_layout(i)

            var src_element = Element[index_type=src.linear_idx_type].load(
                src.ptr + src_idx,
                src.runtime_element_layout,
            )

            comptime if dst_element_stride == 1:
                buffer.store(
                    Int32(dst_idx),
                    src_element.element_data.cast[dst.dtype](),
                )
            else:
                comptime for i in range(dst_fragments.element_layout.size()):
                    comptime element_offset = dst_fragments.element_layout(i)
                    var src = src_element.element_data[i].cast[dst.dtype]()
                    buffer.store(
                        Int32(
                            dst_idx
                            + Scalar[dst.linear_idx_type](element_offset)
                        ),
                        src,
                    )


@always_inline("nodebug")
def copy_local_to_dram[
    dst_thread_layout: Layout,
    num_threads: Int = dst_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor[mut=True, ...], src: LayoutTensor, dst_base: LayoutTensor,):
    """Efficiently copy data from registers (LOCAL) to global memory (DRAM) on
    AMD GPUs.

    This function implements an optimized memory transfer operation specifically
    for AMD GPU architectures. It utilizes the hardware's buffer_store intrinsic
    to efficiently transfer data from registers to global memory while handling
    bounds checking. The function distributes the copy operation across multiple
    threads for maximum throughput.

    Constraints:
        - Only supported on AMD GPUs.
        - Destination tensor must be in GLOBAL address space.
        - Source tensor must be in LOCAL address space.
        - Data types must match between source and destination tensors.

    Parameters:
        dst_thread_layout: The layout used to distribute the destination tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads in the thread block. Threads
            beyond `dst_thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor in global memory (DRAM).
        src: The source tensor in register memory (LOCAL address space) to be
            copied.
        dst_base: The original global memory tensor from which dst is derived.
            This is used to construct the buffer descriptor required by AMD's
            `buffer_store` intrinsic.

    Notes:

    - This function is particularly useful for writing computed results from
        registers back to global memory with minimal latency.
    - The offset calculation is optimized for performance rather than
        flexibility.
    """
    comptime assert is_amd_gpu(), "This function is only supported on AMD GPUs."
    var buffer = make_amd_buffer_resource(dst_base)

    _copy_local_to_dram[
        dst_thread_layout, num_threads, thread_scope, block_dim_count
    ](dst, src, buffer)


@always_inline("nodebug")
def _copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](
    dst: LayoutTensor[mut=True, ...],
    src: LayoutTensor[mut=False, ...],
    buffer: AMDBufferResource,
    offset: Optional[Int] = None,
):
    comptime assert is_amd_gpu(), "This function is only supported on AMD GPUs."
    comptime simd_width = src.element_layout.size()
    _copy_local_to_dram_validate_args(src, dst)

    comptime num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var src_fragments = src.distribute[src_thread_layout](worker_idx)

    comptime M = src_fragments.shape[0]()
    comptime N = src_fragments.shape[1]()

    comptime assert (
        src_fragments.layout.rank() == 2
    ), "src_fragments must be rank 2."

    comptime assert (
        src_fragments.layout.all_dims_known()
    ), "src_fragments must have known layout."

    @always_inline
    @__parameter
    def offset_helper(offset_val: Int):
        var src_frag_offset = Int32(
            src_fragments.distance(src.ptr)
            + Scalar[src.linear_idx_type](offset_val)
        )

        # These loads need to be row-major for L1 cache performance
        comptime for i in range(M):
            comptime for j in range(N):
                comptime dst_frag_idx = Layout.col_major(M, N)([i, j])
                comptime src_frag_idx = Int32(src_fragments.layout([i, j]))
                dst[dst_frag_idx, 0] = rebind[dst.element_type](
                    buffer.load[
                        src.dtype, simd_width, cache_policy=cache_policy
                    ](
                        src_frag_offset,
                        scalar_offset=src_frag_idx,
                    )
                )

    if offset:
        offset_helper(offset.value())
    else:
        var base_ptr = buffer.get_base_ptr()
        offset_helper(ufloordiv(Int(src.ptr) - base_ptr, size_of[src.dtype]()))


@always_inline("nodebug")
def copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](
    dst: LayoutTensor[mut=True, ...],
    src: LayoutTensor[mut=False, ...],
    src_base: LayoutTensor[mut=False, ...],
    offset: Optional[Int] = None,
):
    """Efficiently copy data from global memory (DRAM) to registers for AMD GPUs.

    This function implements an optimized memory transfer operation specifically
    for AMD GPU architectures. It utilizes the hardware's buffer_load intrinsic
    to efficiently transfer data from global memory to registers while handling
    bounds checking. The function distributes the copy operation across multiple
    threads for maximum throughput.

    Constraints:
        - Only supported on AMD GPUs.
        - The destination element layout size must match the SIMD width.
        - Source fragments must be rank 2 with known dimensions.

    Parameters:
        src_thread_layout: The layout used to distribute the source tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads in the thread block. Threads
            beyond `src_thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.
        cache_policy: The cache policy to use for the copy operation.
            Defaults to `CacheOperation.ALWAYS`.

    Args:
        dst: The destination tensor in register memory (LOCAL address space).
        src: The source tensor in global memory (DRAM) to be copied.
        src_base: The original global memory tensor from which src is derived.
            This is used to construct the buffer struct required by AMD's
            `buffer_load` intrinsic.
        offset: The offset in the global memory.

    Notes:

    - The offset calculation method significantly impacts performance.
        Current implementation optimizes for throughput over flexibility.
    - This function is particularly useful for prefetching data into registers
        before performing computations, reducing memory access latency.
    """
    comptime assert is_amd_gpu(), "This function is only supported on AMD GPUs."
    var buffer = make_amd_buffer_resource(src_base)

    _copy_dram_to_local[
        src_thread_layout,
        num_threads,
        thread_scope,
        block_dim_count,
        cache_policy,
    ](dst, src, buffer, offset)


@always_inline("nodebug")
def _copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](
    dst: LayoutTensor[mut=True, ...],
    src_iter: LayoutTensorIter[mut=False, ...],
    buffer: AMDBufferResource,
):
    comptime assert is_amd_gpu(), "This function is only supported on AMD GPUs."
    var src_tensor = src_iter[].vectorize[
        dst.element_layout.shape[0].value(), dst.element_layout.shape[1].value()
    ]()

    _copy_dram_to_local[
        src_thread_layout,
        num_threads,
        thread_scope,
        block_dim_count,
        cache_policy,
    ](dst, src_tensor, buffer, Int(src_iter.offset))


@always_inline("nodebug")
def copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](
    dst: LayoutTensor[mut=True, ...],
    src_iter: LayoutTensorIter[mut=False, ...],
    bounds: UInt32,
):
    """Efficiently copy data from global memory (DRAM) to registers for AMD GPUs.

    This function implements an optimized memory transfer operation specifically
    for AMD GPU architectures. It utilizes the hardware's buffer_load intrinsic
    to efficiently transfer data from global memory to registers while handling
    bounds checking. The function distributes the copy operation across multiple
    threads for maximum throughput.

    Parameters:
        src_thread_layout: The layout used to distribute the source tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads in the thread block. Threads
            beyond `src_thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.
        cache_policy: The cache policy to use for the copy operation.
            Defaults to `CacheOperation.ALWAYS`.

    Args:
        dst: The destination tensor in register memory (LOCAL address space).
        src_iter: The source tensor iterator.
        bounds: Bounds of the buffer, based on the ptr of the src_iter.

    Constraints:
        - Only supported on AMD GPUs.
        - The destination element layout size must match the SIMD width.
        - Source fragments must be rank 2 with known dimensions.

    Notes:

    - The offset calculation method significantly impacts performance.
        Current implementation optimizes for throughput over flexibility.
    - This function is particularly useful for prefetching data into registers
        before performing computations, reducing memory access latency.
    """
    var buffer = make_amd_buffer_resource(src_iter, Int(bounds))

    _copy_dram_to_local[
        src_thread_layout,
        num_threads,
        thread_scope,
        block_dim_count,
        cache_policy,
    ](dst, src_iter, buffer)


@always_inline("nodebug")
def copy_dram_to_local[
    src_thread_layout: Layout,
    num_threads: Int = src_thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
](dst: LayoutTensor[mut=True, ...], src: LayoutTensor[mut=False, ...]):
    """Efficiently copy data from global memory (DRAM) to registers.

    This function implements an optimized memory transfer operation from
    global memory to register memory. It distributes the copy operation across
    multiple threads for maximum throughput while handling bounds checking for
    safety.

    Constraints:
        - The source tensor must be in GLOBAL address space (DRAM).
        - The destination tensor must be in LOCAL address space (registers).
        - Both tensors must have compatible data types.

    Parameters:
        src_thread_layout: The layout used to distribute the source tensor
            across threads. This determines how the workload is divided among
            participating threads.
        num_threads: Total number of threads in the thread block. Threads
            beyond `src_thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.

    Args:
        dst: The destination tensor in register memory (LOCAL address space).
        src:  The source tensor in global memory (DRAM).
    """

    comptime num_busy_threads = src_thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    var src_fragments = src.distribute[src_thread_layout](worker_idx)

    var stride: Int
    comptime if not src_fragments.masked:
        dst.copy_from(src_fragments)
    else:
        var src_frag_offset = src_fragments.distance(src.ptr)
        comptime static_stride = src.layout.stride[0].value()

        comptime if src.layout.all_dims_known():
            stride = static_stride
        else:
            stride = src.runtime_layout.stride.value[0]
        var src_idx_bound = (
            Scalar[src.linear_idx_type](src.dim[0]() * stride) - src_frag_offset
        ).cast[src_fragments.linear_idx_type]()

        comptime num_stores_per_thread = src_fragments.layout.size()

        comptime for i in range(num_stores_per_thread):
            comptime dst_idx = dst.layout(i)
            comptime src_uint_dtype = _get_unsigned_type(
                src_fragments.layout, src_fragments.address_space
            )
            comptime src_static_idx = src_fragments.layout(i)

            var src_idx: Scalar[src_fragments.linear_idx_type]

            comptime if src_fragments.layout.all_dims_known():
                src_idx = Scalar[src.linear_idx_type](src_static_idx)
            else:
                src_idx = src_fragments.runtime_layout(i)

            if src_idx < src_idx_bound:
                var src_element = Element[index_type=src.linear_idx_type].load(
                    src_fragments.ptr + src_idx,
                    src_fragments.runtime_element_layout,
                )
                comptime dst_element_type = Element[
                    dst.dtype, dst.element_layout, dst.linear_idx_type
                ]
                dst_element_type(
                    rebind[dst_element_type.element_data_type](
                        src_element.element_data.cast[dst.dtype]()
                    )
                ).store(dst.ptr + dst_idx)


@always_inline("nodebug")
def copy_local_to_shared[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
    *,
    row_major: Bool = False,
](
    dst: LayoutTensor[mut=True, address_space=.SHARED, ...],
    src: LayoutTensor[address_space=.LOCAL, ...],
):
    """Synchronously copy data from local memory (registers) to SRAM (shared
    memory).

    This function performs a synchronous copy operation from register memory to
    shared memory in a GPU context, distributing the workload across multiple
    threads for parallel execution. It's particularly useful for transferring
    processed data from registers to shared memory for inter-thread
    communication.

    Constraints:

        - Destination tensor must be in SHARED address space.
        - Source tensor must be in LOCAL address space.
        - For optimal performance, the thread layout should match the memory
          access patterns of the tensors.

    Parameters:
        thread_layout: Layout defining how threads are organized for the
            operation. This determines how the workload is distributed among
            threads.
        swizzle: Optional swizzling function to rearrange the destination
            indices, which can improve memory access patterns and reduce bank
            conflicts.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` will be disabled and not
            participate in the copy operation.
        thread_scope: Defines whether operations are performed at `BLOCK` or
            `WARP` level. `BLOCK` scope involves all threads in a thread block,
            while `WARP` scope restricts operations to threads within the same
            warp. Defaults to `ThreadScope.BLOCK`.
        block_dim_count: The number of dimensions in the thread block.
        row_major: Whether to use row-major ordering for the copy operation.
            This is particularly relevant when prefetching from DRAM to SRAM
            via registers on AMD GPUs. Defaults to False.

    Args:
        dst: The destination tensor, which must be in shared memory (SRAM).
        src: The source tensor, which must be in local memory (registers).

    Performance:

    - Distributes the copy workload across multiple threads for parallel execution.
    - Can use swizzling to optimize memory access patterns and reduce bank conflicts.
    - Optimized for transferring data from registers to shared memory.
    - On AMD GPUs, the `row_major` parameter can be used to match the memory
        access pattern used during prefetching from DRAM to registers.

    Notes:

    - The destination tensor must be in `SHARED` address space (SRAM).
    - The source tensor must be in `LOCAL` address space (registers).
    - This function is particularly useful in GPU kernels for sharing processed
        data between threads in the same block.
    - The `row_major` parameter is specifically designed for AMD GPUs when using
        a prefetching pattern from DRAM to SRAM via registers.
    """
    comptime assert (
        dst.address_space == .SHARED
    ), "dst address space must be SHARED."

    comptime assert (
        src.address_space == .LOCAL
    ), "src address space must be LOCAL."

    comptime num_busy_threads = thread_layout.size()
    var worker_idx = _get_worker_idx[thread_scope, block_dim_count]()

    comptime if num_threads > num_busy_threads:
        if worker_idx >= num_busy_threads:
            return

    comptime assert src.dtype == dst.dtype or (
        src.dtype == .float32
        and (dst.dtype.is_half_float() or dst.dtype.is_float8())
    ), "Only support FP32 -> half-precision or FP8 downcast during copy."
    comptime assert (
        src.element_size == dst.element_size
    ), "src and dst element size mismatch."

    comptime if not row_major:
        var dst_frag = dst.distribute[thread_layout](worker_idx)

        comptime if swizzle:
            comptime swizzle_fn = swizzle.value()
            comptime num_vecs = src.layout.size()
            comptime align_src = align_of[SIMD[src.dtype, src.element_size]]()
            comptime align_dst = align_of[SIMD[dst.dtype, dst.element_size]]()
            var dst_frag_offset = dst_frag.distance(dst.ptr)

            comptime for i in range(num_vecs):
                comptime src_idx = src.layout(i)
                comptime dst_idx = dst_frag.layout(i)
                comptime dst_idx_base = dst_idx % swizzle_fn.size()
                comptime dst_idx_diff = dst_idx - dst_idx_base
                var swizzled_idx = swizzle_fn(
                    dst_frag_offset + Scalar[dst.linear_idx_type](dst_idx_base)
                ) + Scalar[dst.linear_idx_type](dst_idx_diff)
                var src_vec = src.ptr.load[
                    width=src.element_size, alignment=align_src
                ](src_idx).cast[dst.dtype]()
                dst.ptr.store[alignment=align_dst](
                    swizzled_idx, src_vec.cast[dst.dtype]()
                )

        else:
            dst_frag.copy_from(src)
    else:
        comptime assert (
            is_amd_gpu()
        ), "This function is only supported on AMD GPUs."
        var dst_frag = dst.distribute[thread_layout, swizzle=swizzle](
            worker_idx
        )
        comptime M = product(dst_frag.layout.shape[0])
        comptime N = product(dst_frag.layout.shape[1])

        comptime assert dst_frag.layout.rank() == 2, "dst_frag must be rank 2."

        comptime for i in range(M):
            comptime for j in range(N):
                # The order here needs to match the order of the loads in copy_dram_to_local
                comptime idx = Layout.col_major(M, N)([i, j])
                var src_idx = src._get_element_idx[idx]()
                var dst_idx = dst_frag._get_element_idx[idx]()

                var src_element = MemoryElement(
                    src.ptr + src_idx, src.runtime_element_layout
                )

                var dst_element = MemoryElement(
                    dst_frag.ptr + dst_idx,
                    dst_frag.runtime_element_layout,
                )
                dst_element.transfer(src_element)


@always_inline
def copy_local_to_local(dst: LayoutTensor[mut=True, ...], src: LayoutTensor):
    """Synchronously copy data between local memory (register) tensors with type
    conversion.

    This function performs a synchronous copy operation between register tensors
    in a GPU context, with support for converting from float32 to half-precision
    formats (bfloat16/float16). It's particularly optimized for specific tensor
    layouts commonly used in matrix multiplication operations.

    Constraints:
        - Destination tensor must be in `LOCAL` address space.
        - Source tensor must be in `LOCAL` address space.
        - Destination tensor must have a half-precision floating-point data type.
        - Source tensor must have float32 data type.
        - Both tensors must have the same total size.

    Args:
        dst: The destination tensor, which must be in local memory (registers)
            and have a half-precision floating-point data type (bfloat16 or
            float16).
        src: The source tensor, which must be in local memory (registers) and
            have float32 data type.

    Example:

    ```mojo
    from layout import LayoutTensor, Layout
    from layout.layout_tensor import copy_local_to_local

    def kernel():
        ...
        var src_reg = LayoutTensor[.float32,
            Layout.row_major(16, 8),
            MutAnyOrigin,
            address_space = .LOCAL,
        ].stack_allocation().fill(1)

        var dst_reg = LayoutTensor[.bfloat16,
            Layout.row_major(16, 8),
            MutAnyOrigin,
            address_space = .LOCAL,
        ].stack_allocation()

        # Process data in float32 registers
        # ...

        # Convert and copy to bfloat16 registers
        copy_local_to_local(dst_reg, src_reg)
    ```

    Performance:

    - Optimized for specific 2D tensor layouts with contiguous inner dimensions.
    - Special fast path for 2D tensors with specific layouts used in matrix
        multiplication.
    - For MMA (Matrix Multiply-Accumulate) operations, efficiently handles the
        conversion between output fragments and input fragments with different
        layouts.
    - Falls back to element-wise copy for general cases.

    Notes:

    - Both source and destination tensors must be in `LOCAL` address space
        (registers).
    - This function currently only supports copying from float32 to half-precision formats.
    - For 2D tensors with stride[1] == 1, a specialized fast path is used that's optimized
        for matrix multiplication patterns.
    - This function is particularly useful in GPU kernels for converting between different
        precision formats while keeping data in registers.
    """
    comptime assert (
        dst.address_space == .LOCAL
    ), "dst address space must be LOCAL."

    comptime assert (
        src.address_space == .LOCAL
    ), "src address space must be LOCAL."

    comptime assert (
        dst.dtype.is_half_float() and src.dtype == .float32
    ), "Only support copy float32 to bfloat16 for now"

    comptime assert (
        dst.layout.size() == src.layout.size()
    ), "dst and src should have the same size."

    # Fast for 2D fragments
    comptime if (
        dst.rank == 2
        and src.rank == 2
        and dst.stride[1]() == 1
        and src.stride[1]() == 1
    ):
        # This path is to map 16x8x16 mma output (16x8) to 16x8x16 mma input (16x16).
        # Output fragment has layout [2 * num_m_mmas, 4]
        # Input  fragment has layout [num_m_mmas, 8]
        comptime num_mmas = src.layout.shape[0].value()
        comptime src_frag_size = src.layout.shape[1].value()
        comptime a_frag_layout = composition(
            src.layout,
            make_layout(Layout.row_major(num_mmas // 2, 2), src.layout[1]),
        )
        # [num_m_mmas, 8] vectorized and transposed to [2, num_m_mmas] x 4
        var dst_vectorized = dst.vectorize[1, src_frag_size]().transpose()
        # [2*num_m_mmas, 4] reshaped and vectorized row_major(num_m_mmas, 2) x 4
        var src_vectorized = src.reshape[a_frag_layout]().vectorize[
            1, src_frag_size
        ]()

        comptime for i in range(dst_vectorized.layout.size()):
            comptime dst_idx = dst_vectorized.layout(i)
            comptime src_idx = src_vectorized.layout(i)

            dst_vectorized.ptr.store(
                dst_idx,
                src_vectorized.ptr.load[width=src_frag_size](src_idx).cast[
                    dst.dtype
                ](),
            )

    # Default elementwise copy
    else:
        comptime for i in range(dst.layout.size()):
            comptime dst_idx = dst.layout(i)
            comptime src_idx = src.layout(i)
            dst.ptr.store(dst_idx, src.ptr[src_idx].cast[dst.dtype]())


# ===-----------------------------------------------------------------------===#
# LayoutTensorIter                                                             #
# ===-----------------------------------------------------------------------===#


struct LayoutTensorIter[
    mut: Bool,
    //,
    dtype: DType,
    layout: Layout,
    origin: Origin[mut=mut],
    /,
    *,
    address_space: AddressSpace = .GENERIC,
    alignment: Int = align_of[dtype](),
    circular: Bool = False,
    axis: Optional[Int] = None,
    layout_int_type: DType = _get_index_type(address_space),
    linear_idx_type: DType = _get_index_type(address_space),
    masked: Bool = False,
](Defaultable, TrivialRegisterPassable):
    """Iterator for traversing a memory buffer with a specific layout.

    `LayoutTensorIter` provides a way to iterate through memory according to a
    specific layout pattern, constructing layout tensors at each position. This
    enables efficient traversal of multi-dimensional data structures with custom
    memory layouts.

    Parameters:
        mut: Whether the iterator allows mutation of the underlying data.
        dtype: The data type of the tensor elements.
        layout: The memory layout pattern to follow during iteration.
        origin: Origin tracking for memory safety.
        address_space: The memory address space (`GLOBAL`, `SHARED`, etc.).
        alignment: Memory alignment requirement for the data.
        circular: Whether iteration wraps around at boundaries.
        axis: Optional axis for dimension-specific operations.
        layout_int_type: Integer type used for layout indices.
        linear_idx_type: Integer type used for indexing into memory.
        masked: Whether to apply bounds masking during iteration.

    Notes:

    The returned layout tensor is NOT vectorized. Users should explicitly vectorize
    if needed for performance-critical operations.
    """

    comptime layout_uint_type = Scalar[
        _unsigned_integral_type_of[Self.layout_int_type]()
    ]
    """The unsigned integer type used for layout, based on layout and address space."""

    comptime linear_uint_type = Scalar[
        _unsigned_integral_type_of[Self.linear_idx_type]()
    ]
    """The unsigned integer type used for indexing into memory."""

    var ptr: Pointer[
        Scalar[Self.dtype], address_space=Self.address_space, origin=Self.origin
    ]
    """Pointer to the memory region being iterated, with appropriate type and memory attributes."""

    var offset: Self.linear_uint_type
    """Current offset from the base pointer, representing the iterator's position in memory."""

    var stride: Self.linear_uint_type
    """Step size between consecutive elements or blocks in memory during iteration."""

    var bound: Self.linear_uint_type
    """Upper bound of the memory region, limiting the iteration range."""

    comptime RuntimeLayoutType = RuntimeLayout[
        Self.layout,
        element_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
    ]
    """Type alias for the runtime layout."""

    var runtime_layout: Self.RuntimeLayoutType
    """Runtime representation of the layout pattern used for mapping logical indices to memory locations."""

    var dimension_bound: Self.layout_uint_type
    """Boundary value for the current dimension when iterating along a specific axis."""

    var idx: Self.linear_uint_type
    """Current logical index position within the iteration sequence."""

    @always_inline
    def __init__(out self):
        """Initialize an empty iterator.

        Creates a default iterator with zero values, typically used as a
        placeholder or default value.
        """

        comptime if Self.axis:
            comptime assert (
                not Self.circular
            ), "Circular use case is not supported if an axis is defined."

        # TODO: Temporary stop-gap to avoid refactoring all `LayoutTensor`s
        # to expect a non-null pointer. Do NOT copy this pattern; new code
        # should use a properly-initialized `Pointer` instead.
        # Or to explicitly model nullability, use `Optional[Pointer]`.
        var this_is_a_hack = 0
        self.ptr = Pointer[
            Scalar[Self.dtype],
            address_space=Self.address_space,
            origin=Self.origin,
        ](unsafe_from_address=this_is_a_hack)
        self.offset = 0
        self.stride = 0
        self.bound = 0
        self.runtime_layout = {}
        self.dimension_bound = 0
        self.idx = 0

    @always_inline
    def __init__(
        out self,
        ptr: Pointer[
            Scalar[Self.dtype],
            address_space=Self.address_space,
            origin=Self.origin,
        ],
        bound: Self.linear_uint_type,
        stride: Self.linear_uint_type = Self.linear_uint_type(
            Self.layout.size()
        ),
        offset: Self.linear_uint_type = 0,
    ):
        """Initialize an iterator with a pointer and basic parameters.

        Creates an iterator for a memory region with the specified bounds and
        stride.

        Args:
            ptr: Pointer to the beginning of the memory region.
            bound: Upper bound of the memory region.
            stride: Step size between consecutive elements (defaults to layout
                size).
            offset: Initial offset from the base pointer.

        Constraints:
            The layout must have all dimensions known at compile time.
        """
        comptime assert (
            Self.layout.all_dims_known()
        ), "Cannot construct LayoutTensorIter with unknown layout."

        comptime assert (
            Self.layout_int_type.is_signed()
            and Self.linear_idx_type.is_signed()
        ), "Layout integer type and linear index type must be signed."

        self.ptr = ptr
        self.bound = bound
        self.stride = stride
        self.runtime_layout = {}
        self.offset = offset
        self.dimension_bound = 0
        self.idx = 0

    @always_inline
    def __init__(
        out self,
        ptr: Pointer[
            Scalar[Self.dtype],
            address_space=Self.address_space,
            origin=Self.origin,
        ],
        bound: Int,
    ):
        """Initialize an iterator with a pointer and `Int` bound.

        Creates an iterator for a memory region with the specified bounds and
        stride.

        Args:
            ptr: Pointer to the beginning of the memory region.
            bound: Upper bound of the memory region.
        """
        return Self(ptr, Self.linear_uint_type(bound))

    @always_inline
    def __init__(
        out self,
        ptr: Pointer[
            Scalar[Self.dtype],
            address_space=Self.address_space,
            origin=Self.origin,
        ],
        bound: Self.linear_uint_type,
        runtime_layout: RuntimeLayout[Self.layout, ...],
        stride: Self.linear_uint_type = Self.linear_uint_type(
            Self.layout.size() if Self.layout.all_dims_known() else UNKNOWN_VALUE
        ),
        offset: Self.linear_uint_type = 0,
        dimension_bound: Self.layout_uint_type = 0,
        idx: Self.linear_uint_type = 0,
    ):
        """Initialize an iterator with a runtime layout.

        Creates an iterator with a runtime-determined layout, allowing for more
        flexible memory traversal patterns.

        Args:
            ptr: Pointer to the beginning of the memory region.
            bound: Upper bound of the memory region.
            runtime_layout: Layout determined at runtime.
            stride: Step size between consecutive elements.
            offset: Initial offset from the base pointer.
            dimension_bound: Bound for the specified dimension when using masked
                iteration.
            idx: Initial index position.

        Constraints:
            The runtime layout must have the same bitwidth as specified for the
            iterator. Circular iteration is not supported when an axis is
            defined.
        """

        comptime assert (
            runtime_layout.linear_idx_type == Self.linear_idx_type
        ), "Mismatch of index type for RuntimeLayout and LayoutTensorIter."

        comptime assert (
            runtime_layout.element_type == Self.layout_int_type
        ), "Mismatch of dimension type for RuntimeLayout and LayoutTensorIter."

        comptime assert (
            Self.layout_int_type.is_signed()
            and Self.linear_idx_type.is_signed()
        ), "Layout integer type and linear index type must be signed."

        comptime if Self.axis:
            comptime assert (
                not Self.circular
            ), "Circular use case is not supported if an axis is defined."

        self.ptr = ptr
        self.offset = offset
        self.stride = (
            Self.linear_uint_type(runtime_layout.size()) if stride
            == UNKNOWN_VALUE else stride
        )
        self.bound = bound
        self.runtime_layout = rebind[Self.RuntimeLayoutType](runtime_layout)
        self.dimension_bound = dimension_bound
        self.idx = idx

    comptime _OriginCastType[
        to_mut: Bool, //, to_origin: Origin[mut=to_mut]
    ] = LayoutTensorIter[
        Self.dtype,
        Self.layout,
        to_origin,
        address_space=Self.address_space,
        alignment=Self.alignment,
        circular=Self.circular,
        axis=Self.axis,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
    ]

    @always_inline("builtin")
    @implicit
    def __init__(
        other: LayoutTensorIter,
        out self: type_of(other)._OriginCastType[ImmOrigin(other.origin)],
    ):
        """Implicitly cast a mutable LayoutTensorIter to immutable.

        Args:
            other: The mutable LayoutTensorIter to cast from.
        """
        self.ptr = other.ptr
        self.bound = other.bound
        self.stride = other.stride
        self.runtime_layout = other.runtime_layout
        self.offset = other.offset
        self.dimension_bound = other.dimension_bound
        self.idx = other.idx

    @always_inline("builtin")
    @implicit
    @doc_hidden
    def __init__[
        __disambig: Int = 0,
    ](
        other: LayoutTensorIter[mut=True, ...],
        out self: type_of(other)._OriginCastType[MutAnyOrigin],
    ):
        self.ptr = other.ptr.as_unsafe_any_origin()
        self.bound = other.bound
        self.stride = other.stride
        self.runtime_layout = other.runtime_layout
        self.offset = other.offset
        self.dimension_bound = other.dimension_bound
        self.idx = other.idx

    @always_inline("builtin")
    @implicit
    @doc_hidden
    def __init__[
        __disambig: Int = 0,
    ](
        other: LayoutTensorIter[...],
        out self: type_of(other)._OriginCastType[ImmutAnyOrigin],
    ):
        self.ptr = other.ptr.as_unsafe_any_origin()
        self.bound = other.bound
        self.stride = other.stride
        self.runtime_layout = other.runtime_layout
        self.offset = other.offset
        self.dimension_bound = other.dimension_bound
        self.idx = other.idx

    comptime LayoutTensorType = LayoutTensor[
        Self.dtype,
        Self.layout,
        Self.origin,
        address_space=Self.address_space,
        masked=Self.masked,
        alignment=Self.alignment,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
    ]
    """The LayoutTensor type returned by this iterator."""

    @always_inline
    def get(self) -> Self.LayoutTensorType:
        """Get the layout tensor at the current iterator position.

        Returns a layout tensor representing the data at the current position
        of the iterator.

        Returns:
            A tensor view at the current iterator position with the
            same type, layout, and memory characteristics as specified by the
            output parameter.
        """
        # TODO: Use deref `[]` to be consistent with mojo feature.

        return Self.LayoutTensorType(
            self.ptr + Int(self.offset),
            self.runtime_layout,
        )

    @always_inline
    def __getitem__(
        self,
    ) -> Self.LayoutTensorType:
        """Get the layout tensor at the current iterator position.

        Operator overload that returns a layout tensor representing the data
        at the current position of the iterator.

        Returns:
            A layout tensor at the current iterator position.
        """
        return self.get()

    @always_inline
    def _clip_shape(self) -> Self.RuntimeLayoutType:
        """Clip the shape based on dimension bounds.

        Internal method that adjusts the shape of the layout tensor based on
        dimension bounds when using masked iteration.

        Returns:
            A new runtime layout with adjusted shape.
        """
        var new_shape = self.runtime_layout.shape
        var cur_dim = new_shape.value[Self.axis.value()]
        new_shape.value[Self.axis.value()] = max(
            0, min(Int(self.dimension_bound) - Int(self.idx) * cur_dim, cur_dim)
        )
        return Self.RuntimeLayoutType(new_shape, self.runtime_layout.stride)

    @always_inline
    def __iadd__[T: Intable](mut self, rhs: T):
        """Increment the iterator by an integer value.

        Advances the iterator by the specified number of positions.

        Parameters:
            T: A type that can be converted to an integer.

        Args:
            rhs: The number of positions to advance.

        Notes:

        This function is unsafe. It omits bound checking for performance
        reasons. Caller must ensure the index doesn't go out-of-bounds.
        """
        self += Self.linear_uint_type(Int(rhs))

    @always_inline
    def __iadd__(mut self, rhs: Self.linear_uint_type):
        """Increment the iterator by a uint value.

        Advances the iterator by the specified number of positions.

        Args:
            rhs: The number of positions to advance.

        Notes:

        This function is unsafe. It omits bound checking for performance
        reasons. Caller must ensure the index doesn't go out-of-bounds.
        """
        self.offset += rhs * self.stride

        comptime if Self.axis:
            self.idx += rhs

        comptime if Self.masked and Self.axis:
            self.runtime_layout = self._clip_shape()

        comptime if Self.circular:
            self.offset = self.offset % self.bound

    @always_inline
    def _incr(mut self):
        """Increment the iterator by 1.

        Advances the iterator by a single position. This is equivalent to
        `iter += 1` but without the division operation, making it more
        efficient.
        """
        self.offset += self.stride

        comptime if Self.circular:
            self.offset = (
                self.offset - self.bound if self.offset
                >= self.bound else self.offset
            )

    @always_inline
    def next[T: Intable](self, rhs: T) -> Self:
        """Return an iterator pointing to a position ahead by rhs steps.

        Creates a new iterator that points rhs positions ahead of the current
        one.

        Parameters:
            T: An integer-convertible type for the step size.

        Args:
            rhs: The number of positions to advance.

        Returns:
            A new iterator pointing to the advanced position.
        """
        var next_idx = Self.linear_uint_type(0)
        var next_offset = (
            self.offset + Self.linear_uint_type(Int(rhs)) * self.stride
        )

        comptime if Self.axis:
            next_idx = self.idx + Self.linear_uint_type(Int(rhs))

        var runtime_layout: Self.RuntimeLayoutType
        comptime if Self.masked:
            runtime_layout = self._clip_shape()
        else:
            runtime_layout = self.runtime_layout

        comptime if Self.circular:
            next_offset = next_offset % self.bound

        return Self(
            self.ptr,
            self.bound,
            stride=self.stride,
            offset=Self.linear_uint_type(Int(next_offset)),
            runtime_layout=runtime_layout,
            dimension_bound=self.dimension_bound,
            idx=next_idx,
        )

    @always_inline
    def next(self, rhs: Self.linear_uint_type = 1) -> Self:
        """Return an iterator pointing to a position ahead by rhs steps.

        Creates a new iterator that points rhs positions ahead of the current
        one.

        Args:
            rhs: The number of positions to advance (defaults to 1).

        Returns:
            A new iterator pointing to the advanced position.
        """
        return self.next(Int(rhs))

    @always_inline
    def next_unsafe(self, rhs: Self.linear_uint_type = 1) -> Self:
        """Return an iterator pointing to a position ahead by rhs steps (unsafe
        version).

        Creates a new iterator that points rhs positions ahead of the current
        one. This is an unsafe version that omits certain checks for
        performance.

        Args:
            rhs: The number of positions to advance (defaults to 1).

        Returns:
            A new iterator pointing to the advanced position.

        Constraints:
            Cannot be used with masked iterators.
            User must ensure rhs < bound / stride.
        """
        comptime assert (
            not Self.masked
        ), "Cannot use unsafe increment for masked iterator."

        var next_offset = self.offset + rhs * self.stride

        comptime if Self.circular:
            next_offset = (
                next_offset - self.bound if next_offset
                >= self.bound else next_offset
            )

        return Self(
            self.ptr,
            self.bound,
            stride=self.stride,
            offset=next_offset,
        )

    comptime ReshapeType[dst_layout: Layout] = LayoutTensorIter[
        Self.dtype,
        dst_layout,
        Self.origin,
        address_space=Self.address_space,
        alignment=Self.alignment,
        circular=Self.circular,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
    ]
    """Type alias for reshaped iterator types.

    Parameters:
        dst_layout: The target layout for the reshaped iterator.
    """

    @always_inline
    def reshape[dst_layout: Layout](self) -> Self.ReshapeType[dst_layout]:
        """Reshape the iterator to a new layout.

        This method creates a new iterator with a different layout while
        preserving the underlying data. The new layout must have the same total
        size as the original.

        Parameters:
            dst_layout: The target layout to reshape to.

        Returns:
            A new iterator with the specified layout.

        Constraints:
            - The destination layout must have the same total size as the original.
            - Both layouts must be contiguous.
            - Both layouts must have compile-time known dimensions.
        """
        comptime assert (
            dst_layout.size() == Self.layout.size()
        ), "Destination layout doesn't match the original."

        comptime assert (
            dst_layout.size() == dst_layout.cosize()
            and Self.layout.size() == Self.layout.cosize()
        ), "Iterator reshape only supports contiguous layout."

        comptime assert (
            Self.layout.all_dims_known() and dst_layout.all_dims_known()
        ), "Iterator reshape only supports compile time layout."

        return Self.ReshapeType[dst_layout](
            self.ptr,
            Self.linear_uint_type(Int(self.bound)),
            Self.ReshapeType[dst_layout].RuntimeLayoutType(),
            Self.linear_uint_type(Int(self.stride)),
            Self.linear_uint_type(Int(self.offset)),
            dimension_bound=Self.layout_uint_type(Int(self.dimension_bound)),
            idx=Self.linear_uint_type(Int(self.idx)),
        )

    comptime BitcastType[
        new_type: DType,
        *,
        address_space: AddressSpace = Self.address_space,
        alignment: Int = Self.alignment,
    ] = LayoutTensorIter[
        new_type,
        Self.layout,
        Self.origin,
        address_space=address_space,
        alignment=alignment,
        circular=Self.circular,
        layout_int_type=Self.layout_int_type,
        linear_idx_type=Self.linear_idx_type,
        masked=Self.masked,
    ]
    """Type alias for bitcast iterator types.

    Parameters:
        new_type: The target data type.
        address_space: The target address space.
        alignment: The target memory alignment.
    """

    @always_inline
    def bitcast[
        new_type: DType,
        *,
        target_address_space: AddressSpace = Self.address_space,
        target_alignment: Int = Self.alignment,
    ](self) -> Self.BitcastType[
        new_type, address_space=Self.address_space, alignment=Self.alignment
    ]:
        """Reinterpret the iterator's underlying pointer as a different data
        type.

        This method performs a bitcast operation, allowing you to view the same
        memory location as a different data type without copying or converting
        the data.

        Parameters:
            new_type: The target data type to cast to.
            target_address_space: The memory address space for the new
                iterator (defaults to current).
            target_alignment: Memory alignment requirement for the new
                iterator (defaults to current).

        Returns:
            A new LayoutTensorIter with the same layout but different data type.
        """
        return Self.BitcastType[
            new_type,
            address_space=Self.address_space,
            alignment=Self.alignment,
        ](
            self.ptr.bitcast[Scalar[new_type]]().address_space_cast[
                Self.address_space
            ](),
            Self.linear_uint_type(Int(self.bound)),
            self.runtime_layout,
            Self.linear_uint_type(Int(self.stride)),
            Self.linear_uint_type(Int(self.offset)),
            dimension_bound=Self.layout_uint_type(Int(self.dimension_bound)),
            idx=Self.linear_uint_type(Int(self.idx)),
        )
