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
"""NVIDIA Tensor Memory Accelerator (TMA) module.

Provides types and functions for working with NVIDIA's Tensor Memory Accelerator,
which enables efficient asynchronous data movement between global and shared memory
on GPUs with Hopper architecture and newer.

The TMA hardware provides hardware-accelerated multi-dimensional memory copies with
features like swizzling for bank conflict avoidance, L2 cache promotion hints, and
support for various data types and memory layouts.
"""

from .. import DeviceBuffer
from std.sys import size_of

from std._gpu._utils import to_llvm_ptr

from std.utils import IndexList, StaticTuple
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


@fieldwise_init("implicit")
struct TensorMapDataType(TrivialRegisterPassable):
    """Data type enumeration for TMA tensor map descriptors.

    Specifies the element data type for TMA operations. The TMA hardware supports
    various numeric types including integers, floating-point, and specialized formats.
    """

    var _value: Int32

    comptime UINT8 = Self(0)
    """Unsigned 8-bit integer."""
    comptime UINT16 = Self(1)
    """Unsigned 16-bit integer."""
    comptime UINT32 = Self(2)
    """Unsigned 32-bit integer."""
    comptime INT32 = Self(3)
    """Signed 32-bit integer."""
    comptime UINT64 = Self(4)
    """Unsigned 64-bit integer."""
    comptime INT64 = Self(5)
    """Signed 64-bit integer."""
    comptime FLOAT16 = Self(6)
    """IEEE 754 16-bit floating-point."""
    comptime FLOAT32 = Self(7)
    """IEEE 754 32-bit floating-point."""
    comptime FLOAT64 = Self(8)
    """IEEE 754 64-bit floating-point."""
    comptime BFLOAT16 = Self(9)
    """Brain floating-point 16-bit format."""
    comptime FLOAT32_FTZ = Self(10)
    """32-bit float with flush-to-zero for denormals."""
    comptime TFLOAT32 = Self(11)
    """TensorFloat-32 format."""
    comptime TFLOAT32_FTZ = Self(12)
    """TensorFloat-32 with flush-to-zero for denormals."""
    comptime PACKED_FP4_ALIGN8B = Self(13)
    """Nibble-packed 4-bit source, landed packed in shared memory.

    Copies 16 U4 values into 8 shared-memory bytes with no gaps.
    """
    comptime PACKED_FP4_ALIGN16B = Self(14)
    """Nibble-packed 4-bit source, landed padded in shared memory.

    Copies 16 U4 values into 16 shared-memory bytes: 8 bytes of packed data
    then an 8-byte gap. The values stay nibble-packed -- it is the padding that
    makes a K extent span one byte per element. This is the layout the SM100
    `kind::mxf8f6f4` UMMA reads an E2M1 operand from, so it is what lets FP4
    weights stay 4-bit in global memory while feeding a tensor-core kind whose
    operands are byte addressed.
    """

    @staticmethod
    def from_dtype[dtype: DType]() -> Self:
        """Converts a Mojo `DType` to the corresponding TMA data type.

        Parameters:
            dtype: The Mojo data type to convert. Must be one of `DType.float32`,
                `DType.float16`, `DType.bfloat16`, `DType.uint8`, `DType.uint16`,
                `DType.uint32`, `DType.int32`, `DType.int64`, `DType.uint64`,
                `DType.float8_e4m3fn`, or `DType.float8_e8m0fnu`.

        Constraints:
            The dtype must be one of the supported types listed above.

        Returns:
            The corresponding `TensorMapDataType` value.
        """
        comptime assert dtype in (
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.uint8,
            DType.uint16,
            DType.uint32,
            DType.int32,
            DType.int64,
            DType.uint64,
            DType.float8_e4m3fn,
            DType.float8_e8m0fnu,
        ), "Unsupported dtype"

        comptime if dtype == .float32:
            return Self.FLOAT32
        elif dtype == .float16:
            return Self.FLOAT16
        elif dtype == .uint16:
            return Self.UINT16
        elif dtype in (DType.float8_e4m3fn, DType.float8_e8m0fnu, DType.uint8):
            return Self.UINT8
        elif dtype == .uint32:
            return Self.UINT32
        elif dtype == .int32:
            return Self.INT32
        elif dtype == .int64:
            return Self.INT64
        elif dtype == .uint64:
            return Self.UINT64
        else:
            return Self.BFLOAT16


@fieldwise_init("implicit")
struct TensorMapInterleave(TrivialRegisterPassable):
    """Interleave mode for TMA tensor map descriptors.

    Specifies how data elements are interleaved in memory for TMA operations.
    Interleaving can improve memory access patterns for certain workloads.
    """

    var _value: Int32

    comptime INTERLEAVE_NONE = Self(0)
    """No interleaving."""
    comptime INTERLEAVE_16B = Self(1)
    """16-byte interleaving."""
    comptime INTERLEAVE_32B = Self(2)
    """32-byte interleaving."""


@fieldwise_init("implicit")
struct TensorMapSwizzle(
    Equatable,
    Hashable,
    ImplicitlyCopyable,
    Intable,
    TrivialRegisterPassable,
    Writable,
):
    """Swizzle mode for TMA tensor map descriptors.

    Swizzling permutes memory addresses to reduce shared memory bank conflicts
    on NVIDIA GPUs. Different swizzle modes apply XOR-based address transformations
    with different granularities (32B, 64B, or 128B).
    """

    var _value: Int32

    comptime SWIZZLE_NONE = Self(0)
    """No swizzling applied."""
    comptime SWIZZLE_32B = Self(1)
    """32-byte swizzle pattern."""
    comptime SWIZZLE_64B = Self(2)
    """64-byte swizzle pattern."""
    comptime SWIZZLE_128B = Self(3)
    """128-byte swizzle pattern."""

    @always_inline("nodebug")
    def __int__(self) -> Int:
        """Converts the swizzle mode to an integer value.

        Returns:
            The integer representation of the swizzle mode.
        """
        return Int(self._value)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        """Checks if two swizzle modes are equal.

        Args:
            other: The swizzle mode to compare with.

        Returns:
            True if the swizzle modes are equal, False otherwise.
        """
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        """Checks if two swizzle modes are not equal.

        Args:
            other: The swizzle mode to compare with.

        Returns:
            True if the swizzle modes are not equal, False otherwise.
        """
        return self._value != other._value

    @always_inline
    def bytes(self) -> Int:
        """Gets the swizzle size in bytes.

        Returns:
            The swizzle size in bytes (0, 32, 64, or 128).
        """
        return Int((2**self._value) * 16)

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the swizzle mode to a writer.

        Args:
            writer: The writer to write to.
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
struct TensorMapL2Promotion(TrivialRegisterPassable):
    """L2 cache promotion hint for TMA tensor map descriptors.

    Specifies how much data to promote into the L2 cache during TMA operations.
    Promoting data to L2 can improve performance when the same data will be
    accessed multiple times.
    """

    var _value: Int32

    comptime NONE = Self(0)
    """No L2 promotion."""
    comptime L2_64B = Self(1)
    """Promote 64 bytes to L2 cache."""
    comptime L2_128B = Self(2)
    """Promote 128 bytes to L2 cache."""
    comptime L2_256B = Self(3)
    """Promote 256 bytes to L2 cache."""


@fieldwise_init("implicit")
struct TensorMapFloatOOBFill(TrivialRegisterPassable):
    """Out-of-bounds fill mode for floating-point TMA operations.

    Specifies how out-of-bounds memory accesses are handled for floating-point
    data types during TMA operations.
    """

    var _value: Int32

    comptime NONE = Self(0)
    """No special out-of-bounds handling."""
    comptime NAN_REQUEST_ZERO_FMA = Self(1)
    """Fill out-of-bounds values with NaN, request zero for FMA operations."""


# The TMA descriptor is a 128-byte opaque object filled by the driver API.
# It should be 64-byte aligned both on the host and the device (if passed to constant memory).
@align(64)
struct TMADescriptor(DevicePassable, ImplicitlyCopyable):
    """TMA tensor map descriptor.

    An opaque 128-byte descriptor that encodes all parameters for a TMA operation,
    including tensor dimensions, strides, data type, swizzle mode, and other
    configuration. This descriptor is created on the host using `create_tma_descriptor()`
    and can be passed to device code for use with TMA hardware instructions.

    The descriptor must be 64-byte aligned both on the host and device.
    """

    var data: StaticTuple[UInt8, 128]
    """The opaque 128-byte descriptor data."""

    comptime device_type: AnyType = TMADescriptor
    """The device-side type for this TMA descriptor."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Gets the type name for this descriptor.

        Returns:
            The string "TMADescriptor".
        """
        return "TMADescriptor"

    @always_inline
    def __init__(out self):
        """Initializes an empty TMA descriptor.

        The descriptor data is uninitialized and must be filled using
        `create_tma_descriptor()` before use.
        """
        self.data = StaticTuple[UInt8, 128]()


def prefetch_tma_descriptor(desc_ptr: OpaquePointer[mut=False, _]):
    """Prefetches a TMA descriptor into the constant cache.

    Issues a hardware prefetch instruction to bring the TMA descriptor into
    the constant cache, which can improve performance when the descriptor
    will be used soon.

    Args:
        desc_ptr: Pointer to the TMA descriptor to prefetch.
    """
    __mlir_op.`nvvm.prefetch`[tensormap=__mlir_attr.unit](
        to_llvm_ptr(desc_ptr),
    )


@always_inline
def create_tma_descriptor[
    dtype: DType,
    rank: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
    unpack_fp4: Bool = False,
](
    global_buf: DeviceBuffer[dtype],
    global_shape: IndexList[rank],
    global_strides: IndexList[rank],
    shared_mem_shape: IndexList[rank],
) raises -> TMADescriptor:
    """Creates a TMA descriptor for tiled memory operations.

    Encodes tensor layout information into a 128-byte TMA descriptor that can
    be used with TMA hardware instructions to efficiently copy data between
    global and shared memory on NVIDIA GPUs.

    The descriptor specifies a mapping from a tile in shared memory to a region
    in global memory, including dimensions, strides, data type, and optional
    swizzling for bank conflict avoidance.

    Parameters:
        dtype: The element data type of the tensor.
        rank: The number of dimensions (1-5).
        swizzle_mode: The swizzle pattern to apply in shared memory.
        l2_promotion: L2 cache promotion hint for TMA loads. Defaults to NONE.
        unpack_fp4: When True, `global_buf` holds nibble-packed E2M1 (two
            values per `uint8`) and the copy pads it on the way into shared
            memory: each 16-value group lands in 16 bytes as 8 packed bytes
            followed by an 8-byte gap (see `PACKED_FP4_ALIGN16B`). The values
            stay nibble-packed; it is the padding that makes a K extent span
            one byte per element. `global_shape` and `shared_mem_shape` are
            then counted in FP4 ELEMENTS on the innermost dimension, while
            `global_strides` stays in `uint8` units.

    Args:
        global_buf: Device buffer containing the global memory tensor.
        global_shape: Dimensions of the tensor in global memory.
        global_strides: Strides (in elements) for each dimension in global memory.
            The tensor must be row-major (stride at innermost dimension equals 1).
        shared_mem_shape: Dimensions of the tile to be copied to shared memory.

    Returns:
        A TMA descriptor configured for the specified tensor layout.

    Raises:
        An error if the descriptor creation fails.
    """
    # TMADescriptor is @align(64) so stack allocation is automatically 64-byte aligned.
    var tma_descriptor = TMADescriptor()
    var tensor_map_ptr = Pointer(to=tma_descriptor).unsafe_bitcast[NoneType]()

    # NOTE: These are initialized in the comptime loop below.
    var global_dim_arg = Array[Int64, rank](uninitialized=True)
    var global_strides_arg = Array[Int64, rank](uninitialized=True)
    var box_dim_arg = Array[Int32, rank](uninitialized=True)
    var element_stride_arg = Array[Int32, rank](fill=1)

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
    # cuTensorMapEncodeTiled requires globalAddress aligned to 32 bytes for
    # PACKED_FP4_ALIGN16B and 16 bytes otherwise.
    comptime required_addr_align = 32 if unpack_fp4 else 16
    debug_assert(
        Int(global_buf.unsafe_ptr()) % required_addr_align == 0,
        "TMA globalAddress must be ",
        required_addr_align,
        "-byte aligned, but is: ",
        Int(global_buf.unsafe_ptr()),
    )
    comptime if unpack_fp4:
        comptime assert (
            dtype == .uint8
        ), "packed FP4 must be presented as uint8, two E2M1 values per byte"
        # `cuTensorMapEncodeTiled` requires globalDim[0] % 128 == 0 for the
        # ALIGN16B packed types.
        debug_assert(
            global_dim_arg[0] % 128 == 0,
            (
                "packed FP4 TMA requires the innermost extent to be a multiple"
                " of 128 elements, but is: "
            ),
            global_dim_arg[0],
        )

    comptime data_type = (
        TensorMapDataType.PACKED_FP4_ALIGN16B if unpack_fp4 else TensorMapDataType.from_dtype[
            dtype
        ]()
    )

    global_buf._tensor_map_encode_tiled(
        tensor_map_ptr,
        data_type._value,
        Int32(rank),
        global_dim_arg.unsafe_ptr(),
        # global_strides_arg[0] is implicitly size_of[dtype]()
        global_strides_arg.unsafe_ptr().unsafe_offset(1),
        box_dim_arg.unsafe_ptr(),
        element_stride_arg.unsafe_ptr(),
        TensorMapInterleave.INTERLEAVE_NONE._value,
        swizzle_mode._value,
        l2_promotion._value,
        TensorMapFloatOOBFill.NONE._value,
    )
    return tma_descriptor
