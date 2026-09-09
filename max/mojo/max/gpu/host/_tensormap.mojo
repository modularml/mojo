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

from std.sys import size_of
from std.utils import IndexList, StaticTuple

from max.gpu.host import DeviceBuffer


@fieldwise_init("implicit")
struct DataType(TrivialRegisterPassable):
    """
    Enum representing acceptable data types for the TensorMap descriptor.
    """

    var _value: Int32

    comptime UINT8 = Self(0)
    comptime UINT16 = Self(1)
    comptime UINT32 = Self(2)
    comptime INT32 = Self(3)
    comptime UINT64 = Self(4)
    comptime INT64 = Self(5)
    comptime FLOAT16 = Self(6)
    comptime FLOAT32 = Self(7)
    comptime FLOAT64 = Self(8)
    comptime BFLOAT16 = Self(9)
    comptime FLOAT32_FTZ = Self(10)
    comptime TFLOAT32 = Self(11)
    comptime TFLOAT32_FTZ = Self(12)

    @staticmethod
    def from_dtype[dtype: DType]() -> Self:
        """
        Convert a DType to a DataType enum value.

        Parameters:
            dtype: The data type to convert.

        Returns:
            The DataType enum value corresponding to the input data type.
        """
        comptime assert dtype in (
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.uint8,
            DType.float8_e4m3fn,
            DType.float8_e8m0fnu,
        ), "Unsupported dtype"

        comptime if dtype == .float32:
            return Self.FLOAT32
        elif dtype == .float16:
            return Self.FLOAT16
        elif dtype in (DType.float8_e4m3fn, DType.float8_e8m0fnu, DType.uint8):
            return Self.UINT8
        else:
            return Self.BFLOAT16


@fieldwise_init("implicit")
struct InterleaveMode(TrivialRegisterPassable):
    """Enum representing interleave modes for tensor memory access.

    Interleaving controls how data is distributed across memory channels
    to optimize memory bandwidth utilization.
    """

    var _value: Int32

    comptime NONE = Self(0)
    comptime _16B = Self(1)
    comptime _32B = Self(2)


@fieldwise_init("implicit")
struct SwizzleMode(
    Equatable,
    ImplicitlyCopyable,
    Intable,
    TrivialRegisterPassable,
    Writable,
):
    """Enum representing memory swizzling patterns for tensor access optimization.

    Swizzling rearranges memory layout to improve cache performance and reduce
    memory bank conflicts. Different swizzle modes correspond to different
    memory access patterns optimized for specific tensor operations.
    """

    var _value: Int32

    comptime NONE = Self(0)
    comptime _32B = Self(1)
    comptime _64B = Self(2)
    comptime _128B = Self(3)

    @always_inline("nodebug")
    def __int__(self) -> Int:
        """Convert SwizzleMode to integer representation.

        Returns:
            The integer value of the swizzle mode.
        """
        return Int(self._value)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        """Check equality between two SwizzleMode instances.

        Args:
            other: The SwizzleMode to compare against.

        Returns:
            True if both instances have the same swizzle mode value.
        """
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        """Check inequality between two SwizzleMode instances.

        Args:
            other: The SwizzleMode to compare against.

        Returns:
            True if the instances have different swizzle mode values.
        """
        return self._value != other._value

    @always_inline
    def bytes(self) -> Int:
        """Get the swizzle size in bytes.

        Returns:
            The swizzle pattern size in bytes. For example:
            - NONE (0): 16 bytes.
            - _32B (1): 32 bytes.
            - _64B (2): 64 bytes.
            - _128B (3): 128 bytes.
        """
        return Int((2**self._value) * 16)

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        """Write a human-readable representation of the SwizzleMode to a writer.

        Args:
            writer: The writer to output the string representation to.
        """
        if self._value == 1:
            writer.write("32B swizzle")
        elif self._value == 2:
            writer.write("64B swizzle")
        elif self._value == 3:
            writer.write("128B swizzle")
        elif self._value == 0:
            writer.write("no swizzle")
        else:
            writer.write("invalid swizzle")


@fieldwise_init("implicit")
struct L2Promotion(TrivialRegisterPassable):
    """Enum representing L2 cache promotion policies for tensor data.

    L2 promotion controls how tensor data is cached in the L2 cache
    to improve memory access performance.
    """

    var _value: Int32

    comptime NONE = Self(0)
    comptime _64B = Self(1)
    comptime _128B = Self(2)
    comptime _256B = Self(3)


@fieldwise_init("implicit")
struct OOBFill(TrivialRegisterPassable):
    """Enum representing out-of-bounds fill behavior for tensor access.

    Controls what values are returned when accessing tensor elements
    outside the defined tensor boundaries.
    """

    var _value: Int32

    comptime NONE = Self(0)
    comptime NAN_REQUEST_ZERO_FMA = Self(1)


# The TMA descriptor is a 128-byte opaque object filled by the driver API.
# It should be 64-byte aligned both on the host and the device (if passed to constant memory).
@align(64)
struct TensorMap(ImplicitlyCopyable):
    """A tensor memory access descriptor for optimized GPU tensor operations.

    TensorMap encapsulates a 128-byte opaque descriptor that is filled by the
    CUDA driver API. This descriptor contains all the information needed for
    efficient tensor memory access patterns, including tiling, swizzling, and
    caching policies.

    The descriptor must be 64-byte aligned both on the host and device memory
    to ensure proper hardware access patterns.
    """

    var data: StaticTuple[UInt8, 128]
    """The underlying 128-byte opaque descriptor data filled by the CUDA driver API."""

    @always_inline
    def __init__(out self):
        """Initialize an empty TensorMap descriptor.

        Creates a zero-initialized 128-byte tensor map descriptor.
        The descriptor will need to be populated using create_tensormap()
        before it can be used for tensor operations.
        """
        self.data = StaticTuple[UInt8, 128]()


@always_inline
def create_tensormap[
    dtype: DType,
    rank: Int,
    //,
](
    global_buf: DeviceBuffer[dtype],
    global_shape: IndexList[rank],
    global_strides: IndexList[rank],
    shared_mem_shape: IndexList[rank],
    swizzle_mode: SwizzleMode = SwizzleMode.NONE,
) raises -> TensorMap:
    """Create a tensor map descriptor object representing tiled memory region.

    Creates a TensorMap descriptor that enables efficient tensor memory access
    patterns on GPU devices. The descriptor encapsulates information about tensor
    layout, memory addressing, and access patterns to optimize data movement
    between global memory and shared memory.

    The function handles the conversion from row-major tensor layout (where the
    last dimension is contiguous) to the column-major layout expected by the
    underlying CUDA TensorMap API.

    Parameters:
        dtype: The data type of tensor elements (float32, bfloat16, or float8_e4m3fn).
        rank: The number of tensor dimensions.

    Args:
        global_buf: Device buffer containing the tensor data in global memory.
        global_shape: Shape of the tensor in global memory for each dimension.
        global_strides: Stride values for each dimension in the global tensor.
        shared_mem_shape: Shape of the tensor tile to be loaded into shared memory.
        swizzle_mode: Memory swizzling pattern to optimize cache performance. Defaults to SwizzleMode.NONE.

    Returns:
        A TensorMap descriptor configured for the specified tensor layout and
        access patterns.

    Raises:
        Error if the tensor configuration is invalid or if the underlying
        CUDA driver call fails.
    """
    # TensorMap is @align(64) so stack allocation is automatically 64-byte aligned.
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
    var tensormap = TensorMap()
    var tensormap_ptr = Pointer(to=tensormap).unsafe_bitcast[NoneType]()

    var global_dim_arg = Array[Int64, rank](uninitialized=True)
    var global_strides_arg = Array[Int64, rank](uninitialized=True)
    var box_dim_arg = Array[Int32, rank](uninitialized=True)
    var element_stride_arg = Array[Int32, rank](fill=1)

    # The input are row-major i.e. last dim is contiguous. The tensormap arguments
    # goes from the least rapidly varying dim to the highest. Here we inverse the
    # inputs for the tensormap constructor arguments.

    comptime for i in range(rank):
        global_dim_arg[i] = Int64(global_shape[rank - i - 1])
        global_strides_arg[i] = Int64(
            global_strides[rank - i - 1] * size_of[dtype]()
        )
        box_dim_arg[i] = Int32(shared_mem_shape[rank - i - 1])

    debug_assert(
        global_strides_arg[0] == Int64(size_of[dtype]()),
        "TMA GMEM should be row-major, global stride",
        " at dim 0 should be size_of[dtype](): ",
        size_of[dtype](),
        " but is: ",
        global_strides_arg[0],
    )

    global_buf._tensor_map_encode_tiled(
        tensormap_ptr,
        DataType.from_dtype[dtype]()._value,
        Int32(rank),
        global_dim_arg.unsafe_ptr(),
        # global_strides_arg[0] is implicitly size_of[dtype]()
        global_strides_arg.unsafe_ptr().unsafe_offset(1),
        box_dim_arg.unsafe_ptr(),
        element_stride_arg.unsafe_ptr(),
        InterleaveMode.NONE._value,
        swizzle_mode._value,
        L2Promotion.NONE._value,
        OOBFill.NONE._value,
    )

    return tensormap


@always_inline
def create_tensormap_im2col[
    dtype: DType,
    rank: Int,
    spatial_rank: Int,
](
    global_buf: DeviceBuffer[dtype],
    global_shape: IndexList[rank],
    global_strides: IndexList[rank],
    lower_corner: IndexList[spatial_rank],
    upper_corner: IndexList[spatial_rank],
    channels_per_pixel: Int,
    pixels_per_column: Int,
    swizzle_mode: SwizzleMode = SwizzleMode.NONE,
) raises -> TensorMap:
    """Create a TMA descriptor for im2col transformation.

    Creates a TensorMap descriptor that performs im2col coordinate transformation
    in hardware during TMA loads. This is used for implicit GEMM convolution
    where the activation tensor is accessed with im2col addressing.

    The tensor is interpreted in CWHDN order (channels, width, height, depth, batch)
    where the last spatial dimensions are optional based on rank.

    For 2D convolution (rank=4): tensor shape is [N, H, W, C] (NHWC)
    - Dimensions are reordered to CWHDN: [C, W, H, N]
    - Spatial dimensions are [W, H] (spatial_rank=2)

    Parameters:
        dtype: The data type of tensor elements.
        rank: Total number of tensor dimensions (4 for 2D conv, 5 for 3D conv).
        spatial_rank: Number of spatial dimensions (2 for 2D, 3 for 3D).

    Args:
        global_buf: Device buffer containing the tensor data.
        global_shape: Shape of the tensor in global memory [N, (D,) H, W, C].
        global_strides: Stride values for each dimension.
        lower_corner: Lower corner offsets for spatial dimensions (negative for padding).
        upper_corner: Upper corner offsets for spatial dimensions.
        channels_per_pixel: Number of channels per im2col column element.
        pixels_per_column: Number of output pixels per im2col column.
        swizzle_mode: Memory swizzling pattern. Defaults to SwizzleMode.NONE.

    Returns:
        A TensorMap descriptor configured for im2col access patterns.

    Raises:
        Error if the tensor configuration is invalid or driver call fails.
    """
    comptime assert (
        rank >= 3
    ), "Im2col requires at least 3D tensor (NHC or NHWC)"
    comptime assert rank <= 5, "Im2col supports at most 5D tensor (NDHWC)"
    comptime assert (
        spatial_rank == rank - 2
    ), "spatial_rank must equal rank - 2 (excluding batch and channel)"

    var tensormap = TensorMap()
    var tensormap_ptr = Pointer(to=tensormap).unsafe_bitcast[NoneType]()

    # Convert from row-major NHWC to column-major CWHDN for TMA API
    # Row-major NHWC: N is dim 0, H is dim 1, W is dim 2, C is dim 3
    # TMA expects CWHDN order (least rapidly varying to most)
    var global_dim_arg = Array[Int64, rank](uninitialized=True)
    var global_strides_arg = Array[Int64, rank](uninitialized=True)
    var lower_corner_arg = Array[Int32, spatial_rank](uninitialized=True)
    var upper_corner_arg = Array[Int32, spatial_rank](uninitialized=True)
    var element_stride_arg = Array[Int32, rank](fill=1)

    # Reverse dimension order for TMA API (CWHDN from NHWC)
    comptime for i in range(rank):
        global_dim_arg[i] = Int64(global_shape[rank - i - 1])
        global_strides_arg[i] = Int64(
            global_strides[rank - i - 1] * size_of[dtype]()
        )

    # Reverse spatial corners (W, H, D from D, H, W or H, W)
    comptime for i in range(spatial_rank):
        lower_corner_arg[i] = Int32(lower_corner[spatial_rank - i - 1])
        upper_corner_arg[i] = Int32(upper_corner[spatial_rank - i - 1])

    global_buf._tensor_map_encode_im2col(
        tensormap_ptr,
        DataType.from_dtype[dtype]()._value,
        Int32(rank),
        global_dim_arg.unsafe_ptr(),
        # global_strides_arg[0] is implicitly size_of[dtype]()
        global_strides_arg.unsafe_ptr().unsafe_offset(1),
        lower_corner_arg.unsafe_ptr(),
        upper_corner_arg.unsafe_ptr(),
        Int32(channels_per_pixel),
        Int32(pixels_per_column),
        element_stride_arg.unsafe_ptr(),
        InterleaveMode.NONE._value,
        swizzle_mode._value,
        L2Promotion.NONE._value,
        OOBFill.NONE._value,  # OOB returns 0 (matches CUTLASS OOBFill::ZERO)
    )

    return tensormap
