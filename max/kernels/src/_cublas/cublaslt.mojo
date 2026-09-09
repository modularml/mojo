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

from std.os import abort
from std.pathlib import Path
from std.ffi import _find_dylib
from std.ffi import _get_dylib_function as _ffi_get_dylib_function
from std.ffi import _Global, OwnedDLHandle

from max.gpu.host._nvidia_cuda import _CUstream_st

from std.utils import StaticTuple

from .cublas import ComputeType
from .dtype import DataType, Property
from .result import Result


comptime cublasLtHandle_t = OptionalPointer[NoneType, _]

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUBLASLT_LIBRARY_PATHS: List[Path] = [
    "libcublasLt.so.13",
    "/usr/local/cuda-13.1/lib64/libcublasLt.so.13",
    "/usr/local/cuda-13.0/lib64/libcublasLt.so.13",
    "/usr/local/cuda/lib64/libcublasLt.so.13",
    "libcublasLt.so.12",
    "/usr/local/cuda-12.8/lib64/libcublasLt.so.12",
    "/usr/local/cuda/lib64/libcublasLt.so.12",
]


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the cuBLASLT libraries. Please make sure that "
            "the CUDA toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[CUDA_CUBLASLT_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUBLASLT_LIBRARY = _Global[
    "CUDA_CUBLASLT_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUBLASLT_LIBRARY_PATHS]()
    )


@always_inline
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUBLASLT_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


def cublasLtMatmulAlgoConfigSetAttribute(
    algo: UnsafePointer[MatmulAlgorithm, MutAnyOrigin],
    attr: AlgorithmConfig,
    buf: OpaquePointer[ImmutAnyOrigin],
    size_in_bytes: Int,
) raises -> Result:
    """Set algo configuration attribute.

    algo         The algo descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoConfigSetAttribute",
        def(
            UnsafePointer[MatmulAlgorithm, MutAnyOrigin],
            AlgorithmConfig,
            OpaquePointer[ImmutAnyOrigin],
            Int,
        ) thin -> Result,
    ]()(algo, attr, buf, size_in_bytes)


def cublasLtCreate(
    light_handle: UnsafePointer[OpaquePointer[AnyOrigin[mut=True]], _],
) raises -> Result:
    return _get_dylib_function[
        "cublasLtCreate",
        def(type_of(light_handle)) thin -> Result,
    ]()(light_handle)


def cublasLtMatrixTransformDescCreate(
    transform_desc: UnsafePointer[
        UnsafePointer[Transform, MutAnyOrigin], MutAnyOrigin
    ],
    scale_type: DataType,
) raises -> Result:
    """Create new matrix transform operation descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if descriptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescCreate",
        def(
            type_of(transform_desc),
            DataType,
        ) thin -> Result,
    ]()(transform_desc, scale_type)


@fieldwise_init
struct Order(TrivialRegisterPassable, Writable):
    """Enum for data ordering ."""

    var _value: Int32
    comptime COL = Self(0)
    """Column-major.

    Leading dimension is the stride (in elements) to the beginning of next column in memory.
    """
    comptime ROW = Self(1)
    """Row major.

    Leading dimension is the stride (in elements) to the beginning of next row in memory.
    """
    comptime COL32 = Self(2)
    """Column-major ordered tiles of 32 columns.

    Leading dimension is the stride (in elements) to the beginning of next group of 32-columns. E.g. if matrix has 33
    columns and 2 rows, ld must be at least (32) * 2 = 64.
    """
    comptime COL4_4R2_8C = Self(3)
    """Column-major ordered tiles of composite tiles with total 32 columns and 8 rows, tile composed of interleaved
    inner tiles of 4 columns within 4 even or odd rows in an alternating pattern.

    Leading dimension is the stride (in elements) to the beginning of the first 32 column x 8 row tile for the next
    32-wide group of columns. E.g. if matrix has 33 columns and 1 row, ld must be at least (32 * 8) * 1 = 256.
    """
    comptime COL32_2R_4R4 = Self(4)
    """Column-major ordered tiles of composite tiles with total 32 columns and 32 rows.
    Element offset within the tile is calculated as (((row%8)/2*4+row/8)*2+row%2)*32+col.

    Leading dimension is the stride (in elements) to the beginning of the first 32 column x 32 row tile for the next
    32-wide group of columns. E.g. if matrix has 33 columns and 1 row, ld must be at least (32*32)*1 = 1024.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.COL:
            writer.write_string("COL")
            return
        if self == Self.ROW:
            writer.write_string("ROW")
            return
        if self == Self.COL32:
            writer.write_string("COL32")
            return
        if self == Self.COL4_4R2_8C:
            writer.write_string("COL4_4R2_8C")
            return
        if self == Self.COL32_2R_4R4:
            writer.write_string("COL32_2R_4R4")
            return
        abort("invalid Order entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtMatrixLayoutSetAttribute(
    mat_layout: cublasLtMatrixLayout_t,
    attr: LayoutAttribute,
    buf: OpaquePointer[ImmutAnyOrigin],
    size_in_bytes: Int,
) raises -> Result:
    """Set matrix layout descriptor attribute.

    matLayout    The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutSetAttribute",
        def(
            cublasLtMatrixLayout_t,
            LayoutAttribute,
            OpaquePointer[ImmutAnyOrigin],
            Int,
        ) thin -> Result,
    ]()(mat_layout, attr, buf, size_in_bytes)


@fieldwise_init
struct ClusterShape(TrivialRegisterPassable, Writable):
    """Thread Block Cluster size.

    Typically dimensioned similar to Tile, with the third coordinate unused at this time.
    ."""

    var _value: Int32
    comptime SHAPE_AUTO = Self(0)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x1x1 = Self(2)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_2x1x1 = Self(3)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_4x1x1 = Self(4)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x2x1 = Self(5)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_2x2x1 = Self(6)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_4x2x1 = Self(7)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x4x1 = Self(8)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_2x4x1 = Self(9)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_4x4x1 = Self(10)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_8x1x1 = Self(11)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x8x1 = Self(12)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_8x2x1 = Self(13)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_2x8x1 = Self(14)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_16x1x1 = Self(15)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x16x1 = Self(16)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_3x1x1 = Self(17)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_5x1x1 = Self(18)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_6x1x1 = Self(19)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_7x1x1 = Self(20)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_9x1x1 = Self(21)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_10x1x1 = Self(22)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_11x1x1 = Self(23)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_12x1x1 = Self(24)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_13x1x1 = Self(25)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_14x1x1 = Self(26)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_15x1x1 = Self(27)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_3x2x1 = Self(28)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_5x2x1 = Self(29)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_6x2x1 = Self(30)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_7x2x1 = Self(31)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x3x1 = Self(32)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_2x3x1 = Self(33)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_3x3x1 = Self(34)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_4x3x1 = Self(35)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_5x3x1 = Self(36)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_3x4x1 = Self(37)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x5x1 = Self(38)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_2x5x1 = Self(39)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_3x5x1 = Self(40)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x6x1 = Self(41)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_2x6x1 = Self(42)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x7x1 = Self(43)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_2x7x1 = Self(44)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x9x1 = Self(45)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x10x1 = Self(46)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x11x1 = Self(47)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x12x1 = Self(48)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x13x1 = Self(49)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x14x1 = Self(50)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_1x15x1 = Self(51)
    """Let library pick cluster shape automatically.
    """
    comptime SHAPE_END = Self(52)
    """Let library pick cluster shape automatically.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.SHAPE_AUTO:
            writer.write_string("SHAPE_AUTO")
        elif self == Self.SHAPE_1x1x1:
            writer.write_string("SHAPE_1x1x1")
        elif self == Self.SHAPE_2x1x1:
            writer.write_string("SHAPE_2x1x1")
        elif self == Self.SHAPE_4x1x1:
            writer.write_string("SHAPE_4x1x1")
        elif self == Self.SHAPE_1x2x1:
            writer.write_string("SHAPE_1x2x1")
        elif self == Self.SHAPE_2x2x1:
            writer.write_string("SHAPE_2x2x1")
        elif self == Self.SHAPE_4x2x1:
            writer.write_string("SHAPE_4x2x1")
        elif self == Self.SHAPE_1x4x1:
            writer.write_string("SHAPE_1x4x1")
        elif self == Self.SHAPE_2x4x1:
            writer.write_string("SHAPE_2x4x1")
        elif self == Self.SHAPE_4x4x1:
            writer.write_string("SHAPE_4x4x1")
        elif self == Self.SHAPE_8x1x1:
            writer.write_string("SHAPE_8x1x1")
        elif self == Self.SHAPE_1x8x1:
            writer.write_string("SHAPE_1x8x1")
        elif self == Self.SHAPE_8x2x1:
            writer.write_string("SHAPE_8x2x1")
        elif self == Self.SHAPE_2x8x1:
            writer.write_string("SHAPE_2x8x1")
        elif self == Self.SHAPE_16x1x1:
            writer.write_string("SHAPE_16x1x1")
        elif self == Self.SHAPE_1x16x1:
            writer.write_string("SHAPE_1x16x1")
        elif self == Self.SHAPE_3x1x1:
            writer.write_string("SHAPE_3x1x1")
        elif self == Self.SHAPE_5x1x1:
            writer.write_string("SHAPE_5x1x1")
        elif self == Self.SHAPE_6x1x1:
            writer.write_string("SHAPE_6x1x1")
        elif self == Self.SHAPE_7x1x1:
            writer.write_string("SHAPE_7x1x1")
        elif self == Self.SHAPE_9x1x1:
            writer.write_string("SHAPE_9x1x1")
        elif self == Self.SHAPE_10x1x1:
            writer.write_string("SHAPE_10x1x1")
        elif self == Self.SHAPE_11x1x1:
            writer.write_string("SHAPE_11x1x1")
        elif self == Self.SHAPE_12x1x1:
            writer.write_string("SHAPE_12x1x1")
        elif self == Self.SHAPE_13x1x1:
            writer.write_string("SHAPE_13x1x1")
        elif self == Self.SHAPE_14x1x1:
            writer.write_string("SHAPE_14x1x1")
        elif self == Self.SHAPE_15x1x1:
            writer.write_string("SHAPE_15x1x1")
        elif self == Self.SHAPE_3x2x1:
            writer.write_string("SHAPE_3x2x1")
        elif self == Self.SHAPE_5x2x1:
            writer.write_string("SHAPE_5x2x1")
        elif self == Self.SHAPE_6x2x1:
            writer.write_string("SHAPE_6x2x1")
        elif self == Self.SHAPE_7x2x1:
            writer.write_string("SHAPE_7x2x1")
        elif self == Self.SHAPE_1x3x1:
            writer.write_string("SHAPE_1x3x1")
        elif self == Self.SHAPE_2x3x1:
            writer.write_string("SHAPE_2x3x1")
        elif self == Self.SHAPE_3x3x1:
            writer.write_string("SHAPE_3x3x1")
        elif self == Self.SHAPE_4x3x1:
            writer.write_string("SHAPE_4x3x1")
        elif self == Self.SHAPE_5x3x1:
            writer.write_string("SHAPE_5x3x1")
        elif self == Self.SHAPE_3x4x1:
            writer.write_string("SHAPE_3x4x1")
        elif self == Self.SHAPE_1x5x1:
            writer.write_string("SHAPE_1x5x1")
        elif self == Self.SHAPE_2x5x1:
            writer.write_string("SHAPE_2x5x1")
        elif self == Self.SHAPE_3x5x1:
            writer.write_string("SHAPE_3x5x1")
        elif self == Self.SHAPE_1x6x1:
            writer.write_string("SHAPE_1x6x1")
        elif self == Self.SHAPE_2x6x1:
            writer.write_string("SHAPE_2x6x1")
        elif self == Self.SHAPE_1x7x1:
            writer.write_string("SHAPE_1x7x1")
        elif self == Self.SHAPE_2x7x1:
            writer.write_string("SHAPE_2x7x1")
        elif self == Self.SHAPE_1x9x1:
            writer.write_string("SHAPE_1x9x1")
        elif self == Self.SHAPE_1x10x1:
            writer.write_string("SHAPE_1x10x1")
        elif self == Self.SHAPE_1x11x1:
            writer.write_string("SHAPE_1x11x1")
        elif self == Self.SHAPE_1x12x1:
            writer.write_string("SHAPE_1x12x1")
        elif self == Self.SHAPE_1x13x1:
            writer.write_string("SHAPE_1x13x1")
        elif self == Self.SHAPE_1x14x1:
            writer.write_string("SHAPE_1x14x1")
        elif self == Self.SHAPE_1x15x1:
            writer.write_string("SHAPE_1x15x1")
        elif self == Self.SHAPE_END:
            writer.write_string("SHAPE_END")
        else:
            abort("invalid ClusterShape entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtHeuristicsCacheSetCapacity(capacity: Int) raises -> Result:
    return _get_dylib_function[
        "cublasLtHeuristicsCacheSetCapacity", def(Int) thin -> Result
    ]()(capacity)


struct MatmulAlgorithmCapability(TrivialRegisterPassable, Writable):
    """Capabilities Attributes that can be retrieved from an initialized Algo structure
    ."""

    var _value: Int32
    comptime SPLITK_SUPPORT = Self(0)
    """Support for split K, see SPLITK_NUM.

    int32_t, 0 means no support, supported otherwise.
    """
    comptime REDUCTION_SCHEME_MASK = Self(1)
    """Reduction scheme mask, see ReductionScheme.

    Shows supported reduction schemes, if reduction scheme is not masked out it is supported.
    e.g. int isReductionSchemeComputeTypeSupported ? (reductionSchemeMask & COMPUTE_TYPE) ==
    COMPUTE_TYPE ? 1 : 0;

    uint32_t.
    """
    comptime CTA_SWIZZLING_SUPPORT = Self(2)
    """Support for cta swizzling, see CTA_SWIZZLING.

    uint32_t, 0 means no support, 1 means supported value of 1, other values are reserved.
    """
    comptime STRIDED_BATCH_SUPPORT = Self(3)
    """Support strided batch.

    int32_t, 0 means no support, supported otherwise.
    """
    comptime OUT_OF_PLACE_RESULT_SUPPORT = Self(4)
    """Support results out of place (D != C in D = alpha.A.B + beta.C).

    int32_t, 0 means no support, supported otherwise.
    """
    comptime UPLO_SUPPORT = Self(5)
    """Syrk/herk support (on top of regular gemm).

    int32_t, 0 means no support, supported otherwise.
    """
    comptime TILE_IDS = Self(6)
    """Tile ids possible to use, see Tile.

    If no tile ids are supported use TILE_UNDEFINED.
    Use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count.

    array of uint32_t.
    """
    comptime CUSTOM_OPTION_MAX = Self(7)
    """Custom option range is from 0 to CUSTOM_OPTION_MAX (inclusive), see CUSTOM_OPTION.

    int32_t.
    """
    comptime CUSTOM_MEMORY_ORDER = Self(10)
    """Whether algorithm supports custom (not COL or ROW memory order), see Order.

    int32_t 0 means only COL and ROW memory order is allowed, non-zero means that algo might have different
    requirements.
    """
    comptime POINTER_MODE_MASK = Self(11)
    """Bitmask enumerating pointer modes algorithm supports.

    uint32_t, see PointerModeMask.
    """
    comptime EPILOGUE_MASK = Self(12)
    """Bitmask enumerating kinds of postprocessing algorithm supports in the epilogue.

    uint32_t, see Epilogue.
    """
    comptime STAGES_IDS = Self(13)
    """Stages ids possible to use, see Stages.

    If no stages ids are supported use STAGES_UNDEFINED.
    Use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count.

    array of uint32_t.
    """
    comptime LD_NEGATIVE = Self(14)
    """Support for negative ld for all of the matrices.

    int32_t 0 means no support, supported otherwise.
    """
    comptime NUMERICAL_IMPL_FLAGS = Self(15)
    """Details about algorithm's implementation that affect it's numerical behavior.

    uint64_t, see cublasLtNumericalImplFlags_t.
    """
    comptime MIN_ALIGNMENT_A_BYTES = Self(16)
    """Minimum alignment required for A matrix in bytes.

    Required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order.
    uint32_t.
    """
    comptime MIN_ALIGNMENT_B_BYTES = Self(17)
    """Minimum alignment required for B matrix in bytes.

    Required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order.
    uint32_t.
    """
    comptime MIN_ALIGNMENT_C_BYTES = Self(18)
    """Minimum alignment required for C matrix in bytes.

    Required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order.
    uint32_t.
    """
    comptime MIN_ALIGNMENT_D_BYTES = Self(19)
    """Minimum alignment required for D matrix in bytes.

    Required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order.
    uint32_t.
    """
    comptime ATOMIC_SYNC = Self(20)
    """EXPERIMENTAL: support for synchronization via atomic counters.

    int32_t.
    """

    comptime POINTER_ARRAY_BATCH_SUPPORT = Self(21)
    """Support pointer array batch.

    int32_t, 0 means no support, supported otherwise.
    """
    comptime FLOATING_POINT_EMULATION_SUPPORT = Self(22)
    """Describes if the algorithm supports floating point emulation.

    int32_t.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.SPLITK_SUPPORT:
            writer.write_string("SPLITK_SUPPORT")
        elif self == Self.REDUCTION_SCHEME_MASK:
            writer.write_string("REDUCTION_SCHEME_MASK")
        elif self == Self.CTA_SWIZZLING_SUPPORT:
            writer.write_string("CTA_SWIZZLING_SUPPORT")
        elif self == Self.STRIDED_BATCH_SUPPORT:
            writer.write_string("STRIDED_BATCH_SUPPORT")
        elif self == Self.OUT_OF_PLACE_RESULT_SUPPORT:
            writer.write_string("OUT_OF_PLACE_RESULT_SUPPORT")
        elif self == Self.UPLO_SUPPORT:
            writer.write_string("UPLO_SUPPORT")
        elif self == Self.TILE_IDS:
            writer.write_string("TILE_IDS")
        elif self == Self.CUSTOM_OPTION_MAX:
            writer.write_string("CUSTOM_OPTION_MAX")
        elif self == Self.CUSTOM_MEMORY_ORDER:
            writer.write_string("CUSTOM_MEMORY_ORDER")
        elif self == Self.POINTER_MODE_MASK:
            writer.write_string("POINTER_MODE_MASK")
        elif self == Self.EPILOGUE_MASK:
            writer.write_string("EPILOGUE_MASK")
        elif self == Self.STAGES_IDS:
            writer.write_string("STAGES_IDS")
        elif self == Self.LD_NEGATIVE:
            writer.write_string("LD_NEGATIVE")
        elif self == Self.NUMERICAL_IMPL_FLAGS:
            writer.write_string("NUMERICAL_IMPL_FLAGS")
        elif self == Self.MIN_ALIGNMENT_A_BYTES:
            writer.write_string("MIN_ALIGNMENT_A_BYTES")
        elif self == Self.MIN_ALIGNMENT_B_BYTES:
            writer.write_string("MIN_ALIGNMENT_B_BYTES")
        elif self == Self.MIN_ALIGNMENT_C_BYTES:
            writer.write_string("MIN_ALIGNMENT_C_BYTES")
        elif self == Self.MIN_ALIGNMENT_D_BYTES:
            writer.write_string("MIN_ALIGNMENT_D_BYTES")
        elif self == Self.ATOMIC_SYNC:
            writer.write_string("ATOMIC_SYNC")
        elif self == Self.POINTER_ARRAY_BATCH_SUPPORT:
            writer.write_string("POINTER_ARRAY_BATCH_SUPPORT")
        elif self == Self.FLOATING_POINT_EMULATION_SUPPORT:
            writer.write_string("FLOATING_POINT_EMULATION_SUPPORT")
        else:
            abort("invalid MatmulAlgorithmCapability entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtGetStatusString(
    status: Result,
) raises -> UnsafePointer[Int8, ImmutAnyOrigin]:
    return _get_dylib_function[
        "cublasLtGetStatusString",
        def(Result) thin raises -> UnsafePointer[Int8, ImmutAnyOrigin],
    ]()(status)


@fieldwise_init
struct PointerMode(TrivialRegisterPassable, Writable):
    """UnsafePointer mode to use for alpha/beta ."""

    var _value: Int32
    comptime HOST = PointerMode(0)
    """Matches CUBLAS_POINTER_MODE_HOST, pointer targets a single value host memory.
    """
    comptime DEVICE = PointerMode(1)
    """Matches CUBLAS_POINTER_MODE_DEVICE, pointer targets a single value device memory.
    """
    comptime DEVICE_VECTOR = PointerMode(2)
    """Pointer targets an array in device memory.
    """
    comptime ALPHA_DEVICE_VECTOR_BETA_ZERO = PointerMode(3)
    """Alpha pointer targets an array in device memory, beta is zero.

    Note: CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE is not supported, must be 0.
    """
    comptime ALPHA_DEVICE_VECTOR_BETA_HOST = PointerMode(4)
    """Alpha pointer targets an array in device memory, beta is a single value in host memory.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.HOST:
            writer.write_string("HOST")
        elif self == Self.DEVICE:
            writer.write_string("DEVICE")
        elif self == Self.DEVICE_VECTOR:
            writer.write_string("DEVICE_VECTOR")
        elif self == Self.ALPHA_DEVICE_VECTOR_BETA_ZERO:
            writer.write_string("ALPHA_DEVICE_VECTOR_BETA_ZERO")
        elif self == Self.ALPHA_DEVICE_VECTOR_BETA_HOST:
            writer.write_string("ALPHA_DEVICE_VECTOR_BETA_HOST")
        else:
            abort("invalid PointerMode entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtMatmulDescGetAttribute(
    matmul_desc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: OpaquePointer[MutAnyOrigin],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int, MutAnyOrigin],
) raises -> Result:
    """Get matmul operation descriptor attribute.

    matmulDesc   The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescGetAttribute",
        def(
            cublasLtMatmulDesc_t,
            cublasLtMatmulDescAttributes_t,
            OpaquePointer[MutAnyOrigin],
            Int,
            UnsafePointer[Int, MutAnyOrigin],
        ) thin -> Result,
    ]()(matmul_desc, attr, buf, size_in_bytes, size_written)


# Opaque descriptor for matrix memory layout
# .
comptime cublasLtMatrixLayout_t = OptionalPointer[MatrixLayout, MutAnyOrigin]

# Opaque descriptor for cublasLtMatrixTransform() operation details
# .
comptime cublasLtMatrixTransformDesc_t = UnsafePointer[Transform, MutAnyOrigin]


def cublasLtMatmulAlgoCheck(
    light_handle: cublasLtHandle_t,
    operation_desc: cublasLtMatmulDesc_t,
    _adesc: cublasLtMatrixLayout_t,
    _bdesc: cublasLtMatrixLayout_t,
    _cdesc: cublasLtMatrixLayout_t,
    _ddesc: cublasLtMatrixLayout_t,
    algo: UnsafePointer[MatmulAlgorithm, ImmutAnyOrigin],
    result: UnsafePointer[cublasLtMatmulHeuristicResult_t, MutAnyOrigin],
) raises -> Result:
    """Check configured algo descriptor for correctness and support on current device.

    Result includes required workspace size and calculated wave count.

    CUBLAS_STATUS_SUCCESS doesn't fully guarantee algo will run (will fail if e.g. buffers are not correctly aligned);
    but if cublasLtMatmulAlgoCheck fails, the algo will not run.

    algo    algo configuration to check
    result  result structure to report algo runtime characteristics; algo field is never updated

    \retval     CUBLAS_STATUS_INVALID_VALUE  if matrix layout descriptors or operation descriptor don't match algo
                                            descriptor
    \retval     CUBLAS_STATUS_NOT_SUPPORTED  if algo configuration or data type combination is not currently supported on
                                            given device
    \retval     CUBLAS_STATUS_ARCH_MISMATCH  if algo configuration cannot be run using the selected device
    \retval     CUBLAS_STATUS_SUCCESS        if check was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoCheck",
        def(
            type_of(light_handle),
            cublasLtMatmulDesc_t,
            cublasLtMatrixLayout_t,
            cublasLtMatrixLayout_t,
            cublasLtMatrixLayout_t,
            cublasLtMatrixLayout_t,
            UnsafePointer[MatmulAlgorithm, ImmutAnyOrigin],
            UnsafePointer[cublasLtMatmulHeuristicResult_t, MutAnyOrigin],
        ) thin -> Result,
    ]()(
        light_handle,
        operation_desc,
        _adesc,
        _bdesc,
        _cdesc,
        _ddesc,
        algo,
        result,
    )


@fieldwise_init
struct Search(TrivialRegisterPassable, Writable):
    """Matmul heuristic search mode
    ."""

    var _value: Int32
    comptime BEST_FIT = Self(0)
    """Ask heuristics for best algo for given usecase.
    """
    comptime LIMITED_BY_ALGO_ID = Self(1)
    """Only try to find best config for preconfigured algo id.
    """
    comptime RESERVED_02 = Self(2)
    """Reserved for future use.
    """
    comptime RESERVED_03 = Self(3)
    """Reserved for future use.
    """
    comptime RESERVED_04 = Self(4)
    """Reserved for future use.
    """
    comptime RESERVED_05 = Self(5)
    """Reserved for future use.
    """
    comptime RESERVED_06 = Self(6)
    """Reserved for future use.
    """
    comptime RESERVED_07 = Self(7)
    """Reserved for future use.
    """
    comptime RESERVED_08 = Self(8)
    """Reserved for future use.
    """
    comptime RESERVED_09 = Self(9)
    """Reserved for future use.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.BEST_FIT:
            writer.write_string("BEST_FIT")
        elif self == Self.LIMITED_BY_ALGO_ID:
            writer.write_string("LIMITED_BY_ALGO_ID")
        elif self == Self.RESERVED_02:
            writer.write_string("RESERVED_02")
        elif self == Self.RESERVED_03:
            writer.write_string("RESERVED_03")
        elif self == Self.RESERVED_04:
            writer.write_string("RESERVED_04")
        elif self == Self.RESERVED_05:
            writer.write_string("RESERVED_05")
        elif self == Self.RESERVED_06:
            writer.write_string("RESERVED_06")
        elif self == Self.RESERVED_07:
            writer.write_string("RESERVED_07")
        elif self == Self.RESERVED_08:
            writer.write_string("RESERVED_08")
        elif self == Self.RESERVED_09:
            writer.write_string("RESERVED_09")
        else:
            abort("invalid Search entry")

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct ReductionScheme(TrivialRegisterPassable, Writable):
    """Reduction scheme for portions of the dot-product calculated in parallel (a. k. a. "split - K").
    ."""

    var _value: Int32
    comptime NONE = ReductionScheme(0)
    """No reduction scheme, dot-product shall be performed in one sequence.
    """
    comptime INPLACE = ReductionScheme(1)
    """Reduction is performed "in place" - using the output buffer (and output data type) and counters (in workspace) to
    guarantee the sequentiality.
    """
    comptime COMPUTE_TYPE = ReductionScheme(2)
    """Intermediate results are stored in compute type in the workspace and reduced in a separate step.
    """
    comptime OUTPUT_TYPE = ReductionScheme(4)
    """Intermediate results are stored in output type in the workspace and reduced in a separate step.
    """
    comptime MASK = ReductionScheme(0x7)
    """Intermediate results are stored in output type in the workspace and reduced in a separate step.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.NONE:
            writer.write_string("NONE")
        elif self == Self.INPLACE:
            writer.write_string("INPLACE")
        elif self == Self.COMPUTE_TYPE:
            writer.write_string("COMPUTE_TYPE")
        elif self == Self.OUTPUT_TYPE:
            writer.write_string("OUTPUT_TYPE")
        elif self == Self.MASK:
            writer.write_string("MASK")
        else:
            abort("invalid ReductionScheme entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtLoggerSetCallback(
    callback: def(
        Int16, UnsafePointer[Int8, ImmutAnyOrigin], OpaquePointer[MutAnyOrigin]
    ) thin raises -> None
) raises -> Result:
    """Experimental: Logger callback setter.

    callback                     a user defined callback function to be called by the logger

    \retval     CUBLAS_STATUS_SUCCESS        if callback was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetCallback",
        def(
            def(
                Int16,
                UnsafePointer[Int8, ImmutAnyOrigin],
                OpaquePointer[MutAnyOrigin],
            ) thin raises -> None
        ) thin -> Result,
    ]()(callback)


def cublasLtGetProperty(
    type: Property, value: UnsafePointer[Int16, MutAnyOrigin]
) raises -> Result:
    return _get_dylib_function[
        "cublasLtGetProperty",
        def(Property, UnsafePointer[Int16, MutAnyOrigin]) thin -> Result,
    ]()(type, value)


def cublasLtGetVersion() raises -> Int:
    return _get_dylib_function["cublasLtGetVersion", def() thin -> Int]()()


def cublasLtMatrixLayoutGetAttribute(
    mat_layout: cublasLtMatrixLayout_t,
    attr: LayoutAttribute,
    buf: OpaquePointer[MutAnyOrigin],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int, MutAnyOrigin],
) raises -> Result:
    """Get matrix layout descriptor attribute.

    matLayout    The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutGetAttribute",
        def(
            cublasLtMatrixLayout_t,
            LayoutAttribute,
            OpaquePointer[MutAnyOrigin],
            Int,
            UnsafePointer[Int, MutAnyOrigin],
        ) thin -> Result,
    ]()(mat_layout, attr, buf, size_in_bytes, size_written)


struct PreferenceOpaque(TrivialRegisterPassable):
    """Semi-opaque descriptor for cublasLtMatmulSelf() operation details
    ."""

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]


@fieldwise_init
struct cublasLtMatmulDescAttributes_t(TrivialRegisterPassable, Writable):
    """Matmul descriptor attributes to define details of the operation. ."""

    var _value: Int32
    comptime CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = Self(0)
    """Compute type, see cudaDataType. Defines data type used for multiply and accumulate operations and the
    accumulator during matrix multiplication.

    int32_t.
    """
    comptime CUBLASLT_MATMUL_DESC_SCALE_TYPE = Self(1)
    """Scale type, see cudaDataType. Defines data type of alpha and beta. Accumulator and value from matrix C are
    typically converted to scale type before final scaling. Value is then converted from scale type to type of matrix
    D before being stored in memory.

    int32_t, default: same as CUBLASLT_MATMUL_DESC_COMPUTE_TYPE.
    """
    comptime CUBLASLT_MATMUL_DESC_POINTER_MODE = Self(2)
    """UnsafePointer mode of alpha and beta, see PointerMode. When DEVICE_VECTOR is in use,
    alpha/beta vector lengths must match number of output matrix rows.

    int32_t, default: HOST.
    """
    comptime CUBLASLT_MATMUL_DESC_TRANSA = Self(3)
    """Transform of matrix A, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    comptime CUBLASLT_MATMUL_DESC_TRANSB = Self(4)
    """Transform of matrix B, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    comptime CUBLASLT_MATMUL_DESC_TRANSC = Self(5)
    """Transform of matrix C, see cublasOperation_t.

    Currently only CUBLAS_OP_N is supported.

    int32_t, default: CUBLAS_OP_N.
    """
    comptime CUBLASLT_MATMUL_DESC_FILL_MODE = Self(6)
    """Matrix fill mode, see cublasFillMode_t.

    int32_t, default: CUBLAS_FILL_MODE_FULL.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE = Self(7)
    """Epilogue function, see Epilogue.

    uint32_t, default: DEFAULT.
    """
    comptime CUBLASLT_MATMUL_DESC_BIAS_POINTER = Self(8)
    """Bias or bias gradient vector pointer in the device memory.

    Bias case. See BIAS.
    For bias data type see CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE.

    Bias vector length must match matrix D rows count.

    Bias gradient case. See DRELU_BGRAD and DGELU_BGRAD.
    Bias gradient vector elements are the same type as the output elements
    (Ctype) with the exception of IMMA kernels (see above).

    Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
    depend on its value to determine expected pointer alignment.

    Bias case: const void *, default: NULL
    Bias gradient case: void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = Self(10)
    """Batch stride for bias or bias gradient vector.

    Used together with CUBLASLT_MATMUL_DESC_BIAS_POINTER when matrix D's BATCH_COUNT > 1.

    int64_t, default: 0.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = Self(11)
    """UnsafePointer for epilogue auxiliary buffer.

    - Output vector for ReLu bit-mask in forward pass when RELU_AUX
     or RELU_AUX_BIAS epilogue is used.
    - Input vector for ReLu bit-mask in backward pass when
     DRELU_BGRAD epilogue is used.

    - Output of GELU input matrix in forward pass when
     GELU_AUX_BIAS epilogue is used.
    - Input of GELU input matrix for backward pass when
     DGELU_BGRAD epilogue is used.

    For aux data type see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE.

    Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
    depend on its value to determine expected pointer alignment.

    Requires setting CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD attribute.

    Forward pass: void *, default: NULL
    Backward pass: const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = Self(12)
    """Leading dimension for epilogue auxiliary buffer.

    - ReLu bit-mask matrix leading dimension in elements (i.e. bits)
     when RELU_AUX, RELU_AUX_BIAS or DRELU_BGRAD epilogue is
    used. Must be divisible by 128 and be no less than the number of rows in the output matrix.

    - GELU input matrix leading dimension in elements
     when GELU_AUX_BIAS or DGELU_BGRAD epilogue used.
     Must be divisible by 8 and be no less than the number of rows in the output matrix.

    int64_t, default: 0.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = Self(13)
    """Batch stride for epilogue auxiliary buffer.

    - ReLu bit-mask matrix batch stride in elements (i.e. bits)
     when RELU_AUX, RELU_AUX_BIAS or DRELU_BGRAD epilogue is
    used. Must be divisible by 128.

    - GELU input matrix batch stride in elements
     when GELU_AUX_BIAS or DGELU_BGRAD epilogue used.
     Must be divisible by 8.

    int64_t, default: 0.
    """
    comptime CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = Self(14)
    """Batch stride for alpha vector.

    Used together with ALPHA_DEVICE_VECTOR_BETA_HOST when matrix D's
    BATCH_COUNT > 1. If ALPHA_DEVICE_VECTOR_BETA_ZERO is set then
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE must be set to 0 as this mode doesn't support batched alpha vector.

    int64_t, default: 0.
    """
    comptime CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = Self(15)
    """Number of SMs to target for parallel execution. Optimizes heuristics for execution on a different number of SMs
    when user expects a concurrent stream to be using some of the device resources.

    int32_t, default: 0 - use the number reported by the device.
    """
    comptime CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = Self(17)
    """Device pointer to the scale factor value that converts data in matrix A to the compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = Self(18)
    """Device pointer to the scale factor value to convert data in matrix B to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = Self(19)
    """Device pointer to the scale factor value to convert data in matrix C to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = Self(20)
    """Device pointer to the scale factor value to convert data in matrix D to compute data type range.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    const void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_AMAX_D_POINTER = Self(21)
    """Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
    output matrix.

    The computed value has the same type as the compute type.

    If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
    data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = Self(22)
    """Type of the data to be stored to the memory pointed to by CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    If unset, the data type defaults to the type of elements of the output matrix with some exceptions, see details
    below.

    ReLu uses a bit-mask.

    GELU input matrix elements type is the same as the type of elements of
    the output matrix with some exceptions, see details below.

    For fp8 kernels with output type CUDA_R_8F_E4M3 the aux data type can be CUDA_R_8F_E4M3 or CUDA_R_16F with some
    restrictions.  See https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulDescAttributes_t for more details.

    If set for an unsupported matrix data, scale, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    int32_t based on cudaDataType, default: -1.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = Self(23)
    """Device pointer to the scaling factor value to convert results from compute type data range to storage
    data range in the auxiliary matrix that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    The scaling factor value must have the same type as the compute type.

    If not specified, or set to NULL, the scaling factor is assumed to be 1. If set for an unsupported matrix data,
    scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER = Self(24)
    """Device pointer to the memory location that on completion will be set to the maximum of absolute values in the
    buffer that is set via CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.

    The computed value has the same type as the compute type.

    If not specified or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix
    data, scale, and compute type combination, calling cublasLtMatmul() will return CUBLAS_INVALID_VALUE.

    void *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_FAST_ACCUM = Self(25)
    """Flag for managing fp8 fast accumulation mode.
    When enabled, problem execution might be faster but at the cost of lower accuracy because intermediate results
    will not periodically be promoted to a higher precision.

    int8_t, default: 0 - fast accumulation mode is disabled.
    """
    comptime CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = Self(26)
    """Type of bias or bias gradient vector in the device memory.

    Bias case: see BIAS.

    Bias vector elements are the same type as the elements of output matrix (Dtype) with the following exceptions:
    - IMMA kernels with computeType=CUDA_R_32I and Ctype=CUDA_R_8I where the bias vector elements
     are the same type as alpha, beta (CUBLASLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F)
    - fp8 kernels with an output type of CUDA_R_32F, CUDA_R_8F_E4M3 or CUDA_R_8F_E5M2, See
     https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul for details.

    int32_t based on cudaDataType, default: -1.
    """
    comptime CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS = Self(27)
    """EXPERIMENTAL: Number of atomic synchronization chunks in the row dimension of the output matrix D.

    int32_t, default 0 (atomic synchronization disabled).
    """
    comptime CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS = Self(28)
    """EXPERIMENTAL: Number of atomic synchronization chunks in the column dimension of the output matrix D.

    int32_t, default 0 (atomic synchronization disabled).
    """
    comptime CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER = Self(29)
    """EXPERIMENTAL: UnsafePointer to a device array of input atomic counters consumed by a matmul.

    int32_t *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER = Self(30)
    """EXPERIMENTAL: UnsafePointer to a device array of output atomic counters produced by a matmul.

    int32_t *, default: NULL.
    """
    comptime CUBLASLT_MATMUL_DESC_A_SCALE_MODE = Self(31)
    """Scaling mode that defines how the matrix scaling factor for matrix A is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_B_SCALE_MODE = Self(32)
    """Scaling mode that defines how the matrix scaling factor for matrix B is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_C_SCALE_MODE = Self(33)
    """Scaling mode that defines how the matrix scaling factor for matrix C is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_D_SCALE_MODE = Self(34)
    """Scaling mode that defines how the matrix scaling factor for matrix D is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE = Self(35)
    """Scaling mode that defines how the matrix scaling factor for the auxiliary matrix is interpreted.

    int32_t, default: 0.
    """

    comptime CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER = Self(36)
    """Device pointer to the scale factors that are used to convert data in matrix D to the compute data type range.

    The scaling factor value type is defined by the scaling mode (see CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE)

    If set for an unsupported matrix data, scale, scale mode, and compute type combination, calling cublasLtMatmul()
    will return CUBLAS_INVALID_VALUE.

    void *, default: NULL
    """

    comptime CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE = Self(37)
    """Scaling mode that defines how the output matrix scaling factor for matrix D is interpreted.

    int32_t, default: 0.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.CUBLASLT_MATMUL_DESC_COMPUTE_TYPE:
            writer.write_string("CUBLASLT_MATMUL_DESC_COMPUTE_TYPE")
        elif self == Self.CUBLASLT_MATMUL_DESC_SCALE_TYPE:
            writer.write_string("CUBLASLT_MATMUL_DESC_SCALE_TYPE")
        elif self == Self.CUBLASLT_MATMUL_DESC_POINTER_MODE:
            writer.write_string("CUBLASLT_MATMUL_DESC_POINTER_MODE")
        elif self == Self.CUBLASLT_MATMUL_DESC_TRANSA:
            writer.write_string("CUBLASLT_MATMUL_DESC_TRANSA")
        elif self == Self.CUBLASLT_MATMUL_DESC_TRANSB:
            writer.write_string("CUBLASLT_MATMUL_DESC_TRANSB")
        elif self == Self.CUBLASLT_MATMUL_DESC_TRANSC:
            writer.write_string("CUBLASLT_MATMUL_DESC_TRANSC")
        elif self == Self.CUBLASLT_MATMUL_DESC_FILL_MODE:
            writer.write_string("CUBLASLT_MATMUL_DESC_FILL_MODE")
        elif self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE:
            writer.write_string("CUBLASLT_MATMUL_DESC_EPILOGUE")
        elif self == Self.CUBLASLT_MATMUL_DESC_BIAS_POINTER:
            writer.write_string("CUBLASLT_MATMUL_DESC_BIAS_POINTER")
        elif self == Self.CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE:
            writer.write_string("CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE")
        elif self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
            writer.write_string("CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER")
        elif self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD:
            writer.write_string("CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD")
        elif self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE:
            writer.write_string(
                "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE"
            )
        elif self == Self.CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE:
            writer.write_string(
                "CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE"
            )
        elif self == Self.CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET:
            writer.write_string("CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET")
        elif self == Self.CUBLASLT_MATMUL_DESC_A_SCALE_POINTER:
            writer.write_string("CUBLASLT_MATMUL_DESC_A_SCALE_POINTER")
        elif self == Self.CUBLASLT_MATMUL_DESC_B_SCALE_POINTER:
            writer.write_string("CUBLASLT_MATMUL_DESC_B_SCALE_POINTER")
        elif self == Self.CUBLASLT_MATMUL_DESC_C_SCALE_POINTER:
            writer.write_string("CUBLASLT_MATMUL_DESC_C_SCALE_POINTER")
        elif self == Self.CUBLASLT_MATMUL_DESC_D_SCALE_POINTER:
            writer.write_string("CUBLASLT_MATMUL_DESC_D_SCALE_POINTER")
        elif self == Self.CUBLASLT_MATMUL_DESC_AMAX_D_POINTER:
            writer.write_string("CUBLASLT_MATMUL_DESC_AMAX_D_POINTER")
        elif self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE:
            writer.write_string("CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE")
        elif self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER:
            writer.write_string(
                "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER"
            )
        elif self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER:
            writer.write_string(
                "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER"
            )
        elif self == Self.CUBLASLT_MATMUL_DESC_FAST_ACCUM:
            writer.write_string("CUBLASLT_MATMUL_DESC_FAST_ACCUM")
        elif self == Self.CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
            writer.write_string("CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE")
        elif self == Self.CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS:
            writer.write_string(
                "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS"
            )
        elif self == Self.CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS:
            writer.write_string(
                "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS"
            )
        elif self == Self.CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER:
            writer.write_string(
                "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER"
            )
        elif self == Self.CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER:
            writer.write_string(
                "CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER"
            )
        elif self == Self.CUBLASLT_MATMUL_DESC_A_SCALE_MODE:
            writer.write_string("CUBLASLT_MATMUL_DESC_A_SCALE_MODE")
        elif self == Self.CUBLASLT_MATMUL_DESC_B_SCALE_MODE:
            writer.write_string("CUBLASLT_MATMUL_DESC_B_SCALE_MODE")
        elif self == Self.CUBLASLT_MATMUL_DESC_C_SCALE_MODE:
            writer.write_string("CUBLASLT_MATMUL_DESC_C_SCALE_MODE")
        elif self == Self.CUBLASLT_MATMUL_DESC_D_SCALE_MODE:
            writer.write_string("CUBLASLT_MATMUL_DESC_D_SCALE_MODE")
        elif self == Self.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE:
            writer.write_string("CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE")
        elif self == Self.CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER:
            writer.write_string("CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER")
        elif self == Self.CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE:
            writer.write_string("CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE")
        else:
            abort("invalid cublasLtMatmulDescAttributes_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtMatrixTransformDescInit_internal(
    transform_desc: cublasLtMatrixTransformDesc_t,
    size: Int,
    scale_type: DataType,
) raises -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescInit_internal",
        def(cublasLtMatrixTransformDesc_t, Int, DataType) thin -> Result,
    ]()(transform_desc, size, scale_type)


def cublasLtMatrixLayoutDestroy(
    mat_layout: cublasLtMatrixLayout_t,
) raises -> Result:
    """Destroy matrix layout descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutDestroy",
        def(cublasLtMatrixLayout_t) thin -> Result,
    ]()(mat_layout)


# Opaque descriptor for cublasLtMatmul() operation details
# .
comptime cublasLtMatmulDesc_t = OptionalPointer[Descriptor, MutAnyOrigin]

# Opaque descriptor for cublasLtMatmulAlgoGetHeuristic() configuration
# .
comptime cublasLtMatmulPreference_t = OptionalPointer[
    PreferenceOpaque, MutAnyOrigin
]


def cublasLtMatmul(
    light_handle: cublasLtHandle_t,
    compute_desc: cublasLtMatmulDesc_t,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _adesc: cublasLtMatrixLayout_t,
    _b: OptionalPointer[NoneType, ImmutAnyOrigin],
    _bdesc: cublasLtMatrixLayout_t,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OptionalPointer[NoneType, ImmutAnyOrigin],
    _cdesc: cublasLtMatrixLayout_t,
    _d: OptionalPointer[NoneType, MutAnyOrigin],
    _ddesc: cublasLtMatrixLayout_t,
    algo: UnsafePointer[MatmulAlgorithm, ImmutAnyOrigin],
    workspace: OpaquePointer[MutAnyOrigin],
    workspace_size_in_bytes: Int,
    stream: _CUstream_st,
) raises -> Result:
    """Execute matrix multiplication (D = alpha * op(A) * op(B) + beta * C).

    \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been initialized
    \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in conflict or in an impossible configuration; e.g.
                                               when workspaceSizeInBytes is less than workspace required by configured
                                               algo
    \retval     CUBLAS_STATUS_NOT_SUPPORTED     if current implementation on selected device doesn't support configured
                                               operation
    \retval     CUBLAS_STATUS_ARCH_MISMATCH     if configured operation cannot be run using selected device
    \retval     CUBLAS_STATUS_EXECUTION_FAILED  if cuda reported execution error from the device
    \retval     CUBLAS_STATUS_SUCCESS           if the operation completed successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmul",
        def(
            type_of(light_handle),
            type_of(compute_desc),
            type_of(alpha),
            type_of(_a),
            type_of(_adesc),
            type_of(_b),
            type_of(_bdesc),
            type_of(beta),
            type_of(_c),
            type_of(_cdesc),
            type_of(_d),
            type_of(_ddesc),
            type_of(algo),
            type_of(workspace),
            type_of(workspace_size_in_bytes),
            type_of(stream),
        ) thin -> Result,
    ]()(
        light_handle,
        compute_desc,
        alpha,
        _a,
        _adesc,
        _b,
        _bdesc,
        beta,
        _c,
        _cdesc,
        _d,
        _ddesc,
        algo,
        workspace,
        workspace_size_in_bytes,
        stream,
    )


def cublasLtMatrixTransformDescDestroy(
    transform_desc: cublasLtMatrixTransformDesc_t,
) raises -> Result:
    """Destroy matrix transform operation descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescDestroy",
        def(cublasLtMatrixTransformDesc_t) thin -> Result,
    ]()(transform_desc)


def cublasLtMatmulAlgoCapGetAttribute(
    algo: UnsafePointer[MatmulAlgorithm, ImmutAnyOrigin],
    attr: MatmulAlgorithmCapability,
    buf: OpaquePointer[MutAnyOrigin],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int, MutAnyOrigin],
) raises -> Result:
    """Get algo capability attribute.

    E.g. to get list of supported Tile IDs:
        Tile tiles[TILE_END];
        size_t num_tiles, size_written;
        if (cublasLtMatmulAlgoCapGetAttribute(algo, TILE_IDS, tiles, size_of(tiles), size_written) ==
    CUBLAS_STATUS_SUCCESS) { num_tiles = size_written / size_of(tiles[0]);
        }

    algo         The algo descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoCapGetAttribute",
        def(
            UnsafePointer[MatmulAlgorithm, ImmutAnyOrigin],
            MatmulAlgorithmCapability,
            OpaquePointer[MutAnyOrigin],
            Int,
            UnsafePointer[Int, MutAnyOrigin],
        ) thin -> Result,
    ]()(algo, attr, buf, size_in_bytes, size_written)


def cublasLtMatmulDescSetAttribute(
    matmul_desc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    buf: OpaquePointer[ImmutAnyOrigin],
    size_in_bytes: Int,
) raises -> Result:
    """Set matmul operation descriptor attribute.

    matmulDesc   The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescSetAttribute",
        def(
            cublasLtMatmulDesc_t,
            cublasLtMatmulDescAttributes_t,
            OpaquePointer[ImmutAnyOrigin],
            Int,
        ) thin -> Result,
    ]()(matmul_desc, attr, buf, size_in_bytes)


def cublasLtMatmulPreferenceSetAttribute(
    pref: cublasLtMatmulPreference_t,
    attr: Preference,
    buf: OpaquePointer[ImmutAnyOrigin],
    size_in_bytes: Int,
) raises -> Result:
    """Set matmul heuristic search preference descriptor attribute.

    pref         The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceSetAttribute",
        def(
            cublasLtMatmulPreference_t,
            Preference,
            OpaquePointer[ImmutAnyOrigin],
            Int,
        ) thin -> Result,
    ]()(pref, attr, buf, size_in_bytes)


# Experimental: Logger callback type.
# .
comptime cublasLtLoggerCallback_t = def(
    Int32,
    UnsafePointer[Int8, ImmutAnyOrigin],
    UnsafePointer[Int8, ImmutAnyOrigin],
) thin -> None


def cublasLtMatrixLayoutInit_internal(
    mat_layout: cublasLtMatrixLayout_t,
    size: Int,
    type: DataType,
    rows: UInt64,
    cols: UInt64,
    ld: Int64,
) raises -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutInit_internal",
        def(
            cublasLtMatrixLayout_t,
            Int,
            DataType,
            UInt64,
            UInt64,
            Int64,
        ) thin -> Result,
    ]()(mat_layout, size, type, rows, cols, ld)


@fieldwise_init
struct Preference(TrivialRegisterPassable, Writable):
    """Algo search preference to fine tune the heuristic function. ."""

    var _value: Int32
    comptime SEARCH_MODE = Self(0)
    """Search mode, see Search.

    uint32_t, default: BEST_FIT.
    """
    comptime MAX_WORKSPACE_BYTES = Self(1)
    """Maximum allowed workspace size in bytes.

    uint64_t, default: 0 - no workspace allowed.
    """
    comptime REDUCTION_SCHEME_MASK = Self(3)
    """Reduction scheme mask, see ReductionScheme. Filters heuristic result to only include algo configs that
    use one of the required modes.

    E.g. mask value of 0x03 will allow only INPLACE and COMPUTE_TYPE reduction schemes.

    uint32_t, default: MASK (allows all reduction schemes).
    """
    comptime MIN_ALIGNMENT_A_BYTES = Self(5)
    """Minimum buffer alignment for matrix A (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix A that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    comptime MIN_ALIGNMENT_B_BYTES = Self(6)
    """Minimum buffer alignment for matrix B (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix B that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    comptime MIN_ALIGNMENT_C_BYTES = Self(7)
    """Minimum buffer alignment for matrix C (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix C that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    comptime MIN_ALIGNMENT_D_BYTES = Self(8)
    """Minimum buffer alignment for matrix D (in bytes).

    Selecting a smaller value will exclude algorithms that can not work with matrix D that is not as strictly aligned
    as they need.

    uint32_t, default: 256.
    """
    comptime MAX_WAVES_COUNT = Self(9)
    """Maximum wave count.

    See cublasLtMatmulHeuristicResult_t::wavesCount.

    Selecting a non-zero value will exclude algorithms that report device utilization higher than specified.

    float, default: 0.0f.
    """
    comptime IMPL_MASK = Self(12)
    """Numerical implementation details mask, see cublasLtNumericalImplFlags_t. Filters heuristic result to only include
    algorithms that use the allowed implementations.

    uint64_t, default: uint64_t(-1) (allow everything).
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.SEARCH_MODE:
            writer.write_string("SEARCH_MODE")
        elif self == Self.MAX_WORKSPACE_BYTES:
            writer.write_string("MAX_WORKSPACE_BYTES")
        elif self == Self.REDUCTION_SCHEME_MASK:
            writer.write_string("REDUCTION_SCHEME_MASK")
        elif self == Self.MIN_ALIGNMENT_A_BYTES:
            writer.write_string("MIN_ALIGNMENT_A_BYTES")
        elif self == Self.MIN_ALIGNMENT_B_BYTES:
            writer.write_string("MIN_ALIGNMENT_B_BYTES")
        elif self == Self.MIN_ALIGNMENT_C_BYTES:
            writer.write_string("MIN_ALIGNMENT_C_BYTES")
        elif self == Self.MIN_ALIGNMENT_D_BYTES:
            writer.write_string("MIN_ALIGNMENT_D_BYTES")
        elif self == Self.MAX_WAVES_COUNT:
            writer.write_string("MAX_WAVES_COUNT")
        elif self == Self.IMPL_MASK:
            writer.write_string("IMPL_MASK")
        else:
            abort("invalid Preference entry")

    def __int__(self) -> Int:
        return Int(self._value)


struct MatmulAlgorithm(Defaultable, TrivialRegisterPassable):
    """Semi-opaque algorithm descriptor (to avoid complicated alloc/free schemes).

    This structure can be trivially serialized and later restored for use with the same version of cuBLAS library to save
    on selecting the right configuration again.
    """

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]

    def __init__(out self):
        self.data = StaticTuple[UInt64, 8](0)


comptime cublasLtNumericalImplFlags_t = UInt64


@fieldwise_init
struct AlgorithmConfig(TrivialRegisterPassable, Writable):
    """Algo Configuration Attributes that can be set according to the Algo capabilities
    ."""

    var _value: Int32
    comptime ID = Self(0)
    """Algorithm index, see `cublasLtMatmulAlgoGetIds()`.

    Readonly, set by cublasLtMatmulAlgoInit().
    int32_t.
    """
    comptime TILE_ID = Self(1)
    """Tile id, see Tile.

    uint32_t, default: TILE_UNDEFINED.
    """
    comptime SPLITK_NUM = Self(2)
    """Number of K splits.

    If the number of K splits is greater than one, SPLITK_NUM parts
    of matrix multiplication will be computed in parallel. The results will be accumulated
    according to REDUCTION_SCHEME.

    int32_t, default: 1.
    """
    comptime REDUCTION_SCHEME = Self(3)
    """Reduction scheme, see ReductionScheme.

    uint32_t, default: NONE.
    """
    comptime CTA_SWIZZLING = Self(4)
    """CTA swizzling, change mapping from CUDA grid coordinates to parts of the matrices.

    Possible values: 0, 1, other values reserved.

    uint32_t, default: 0.
    """
    comptime CUSTOM_OPTION = Self(5)
    """Custom option.

    Each algorithm can support some custom options that don't fit description of the other config
    attributes, see CUSTOM_OPTION_MAX to get accepted range for any specific case.

    uint32_t, default: 0.
    """
    comptime STAGES_ID = Self(6)
    """Stages id, see Stages.

    uint32_t, default: STAGES_UNDEFINED.
    """
    comptime INNER_SHAPE_ID = Self(7)
    """Inner shape id, see InnerShape.

    uint16_t, default: 0 (UNDEFINED).
    """
    comptime CLUSTER_SHAPE_ID = Self(8)
    """Thread Block Cluster shape id, see ClusterShape. Defines cluster size to use.

    uint16_t, default: 0 (SHAPE_AUTO).
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.ID:
            writer.write_string("ID")
        elif self == Self.TILE_ID:
            writer.write_string("TILE_ID")
        elif self == Self.SPLITK_NUM:
            writer.write_string("SPLITK_NUM")
        elif self == Self.REDUCTION_SCHEME:
            writer.write_string("REDUCTION_SCHEME")
        elif self == Self.CTA_SWIZZLING:
            writer.write_string("CTA_SWIZZLING")
        elif self == Self.CUSTOM_OPTION:
            writer.write_string("CUSTOM_OPTION")
        elif self == Self.STAGES_ID:
            writer.write_string("STAGES_ID")
        elif self == Self.INNER_SHAPE_ID:
            writer.write_string("INNER_SHAPE_ID")
        elif self == Self.CLUSTER_SHAPE_ID:
            writer.write_string("CLUSTER_SHAPE_ID")
        else:
            abort("invalid AlgorithmConfig entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtMatmulPreferenceDestroy(
    pref: cublasLtMatmulPreference_t,
) raises -> Result:
    """Destroy matmul heuristic search preference descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceDestroy",
        def(cublasLtMatmulPreference_t) thin -> Result,
    ]()(pref)


def cublasLtMatmulAlgoGetHeuristic(
    light_handle: cublasLtHandle_t,
    operation_desc: cublasLtMatmulDesc_t,
    _adesc: cublasLtMatrixLayout_t,
    _bdesc: cublasLtMatrixLayout_t,
    _cdesc: cublasLtMatrixLayout_t,
    _ddesc: cublasLtMatrixLayout_t,
    preference: cublasLtMatmulPreference_t,
    requested_algo_count: Int,
    heuristic_results_array: UnsafePointer[
        cublasLtMatmulHeuristicResult_t, MutAnyOrigin
    ],
    return_algo_count: UnsafePointer[Int, MutAnyOrigin],
) raises -> Result:
    """Query cublasLt heuristic for algorithm appropriate for given use case.

        lightHandle            UnsafePointer to the allocated cuBLASLt handle for the cuBLASLt
                                          context. See cublasLtHandle_t.
        operationDesc          Handle to the matrix multiplication descriptor.
        Adesc                  Handle to the layout descriptors for matrix A.
        Bdesc                  Handle to the layout descriptors for matrix B.
        Cdesc                  Handle to the layout descriptors for matrix C.
        Ddesc                  Handle to the layout descriptors for matrix D.
        preference             UnsafePointer to the structure holding the heuristic search
                                          preferences descriptor. See cublasLtMatrixLayout_t.
        requestedAlgoCount     Size of heuristicResultsArray (in elements) and requested
                                          maximum number of algorithms to return.
        returnAlgoCount        The number of heuristicResultsArray elements written.

    \retval  CUBLAS_STATUS_INVALID_VALUE   if requestedAlgoCount is less or equal to zero
    \retval  CUBLAS_STATUS_NOT_SUPPORTED   if no heuristic function available for current configuration
    \retval  CUBLAS_STATUS_SUCCESS         if query was successful, inspect
                                          heuristicResultsArray[0 to (returnAlgoCount - 1)].state
                                          for detail status of results
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoGetHeuristic",
        def(
            type_of(light_handle),
            cublasLtMatmulDesc_t,
            cublasLtMatrixLayout_t,
            cublasLtMatrixLayout_t,
            cublasLtMatrixLayout_t,
            cublasLtMatrixLayout_t,
            cublasLtMatmulPreference_t,
            Int16,
            UnsafePointer[cublasLtMatmulHeuristicResult_t, MutAnyOrigin],
            UnsafePointer[Int, MutAnyOrigin],
        ) thin -> Result,
    ]()(
        light_handle,
        operation_desc,
        _adesc,
        _bdesc,
        _cdesc,
        _ddesc,
        preference,
        Int16(requested_algo_count),
        heuristic_results_array,
        return_algo_count,
    )


@fieldwise_init
struct InnerShape(TrivialRegisterPassable, Writable):
    """Inner size of the kernel.

    Represents various aspects of internal kernel design, that don't impact CUDA grid size but may have other more subtle
    effects.
    """

    var _value: Int32
    comptime UNDEFINED = Self(0)
    comptime MMA884 = Self(1)
    comptime MMA1684 = Self(2)
    comptime MMA1688 = Self(3)
    comptime MMA16816 = Self(4)
    comptime END = Self(5)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.UNDEFINED:
            writer.write_string("UNDEFINED")
        elif self == Self.MMA884:
            writer.write_string("MMA884")
        elif self == Self.MMA1684:
            writer.write_string("MMA1684")
        elif self == Self.MMA1688:
            writer.write_string("MMA1688")
        elif self == Self.MMA16816:
            writer.write_string("MMA16816")
        elif self == Self.END:
            writer.write_string("END")
        else:
            abort("invalid InnerShape entry")

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cublasLtMatmulMatrixScale_t(TrivialRegisterPassable, Writable):
    """Scaling mode for per-matrix scaling."""

    var _value: Int32
    comptime MATRIX_SCALE_SCALAR_32F = Self(0)
    """Scaling factors are single precision scalars applied to the whole tensor.
    """
    comptime MATRIX_SCALE_VEC16_UE4M3 = Self(1)
    """Scaling factors are tensors with a dedicated 8-bit CUDA_R_8F_UE4M3 value per 16-element block.

    The scaling factor is stored for each 16-element block in the innermost dimension of the
    corresponding data tensor.
    """
    comptime MATRIX_SCALE_VEC32_UE8M0 = Self(2)
    """Same as VEC16_UE4M3, but with CUDA_R_8F_UE8M0 type and block size of 32 elements.
    """
    comptime MATRIX_SCALE_OUTER_VEC_32F = Self(3)
    """Scaling factors are single-precision vectors.

    This mode is only applicable to matrices A and B, in which case the vectors are expected to
    have M and N elements respectively, and each (i, j)-th element of product of A and B is
    multiplied by i-th element of A scale and j-th element of B scale.
    """
    comptime MATRIX_SCALE_VEC128_32F = Self(4)
    """Scaling factors are tensors with a dedicated FP32 scaling factor per 128-element block.

    The scaling factor is stored for each 128-element block in the innermost dimension of the
    corresponding data tensor.
    """
    comptime MATRIX_SCALE_BLK128x128_32F = Self(5)
    """Scaling factors are tensors with a dedicated FP32 scaling factor per 128x128-element block.
    """
    comptime MATRIX_SCALE_END = Self(6)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.MATRIX_SCALE_SCALAR_32F:
            writer.write_string("MATRIX_SCALE_SCALAR_32F")
        elif self == Self.MATRIX_SCALE_VEC16_UE4M3:
            writer.write_string("MATRIX_SCALE_VEC16_UE4M3")
        elif self == Self.MATRIX_SCALE_VEC32_UE8M0:
            writer.write_string("MATRIX_SCALE_VEC32_UE8M0")
        elif self == Self.MATRIX_SCALE_OUTER_VEC_32F:
            writer.write_string("MATRIX_SCALE_OUTER_VEC_32F")
        elif self == Self.MATRIX_SCALE_VEC128_32F:
            writer.write_string("MATRIX_SCALE_VEC128_32F")
        elif self == Self.MATRIX_SCALE_BLK128x128_32F:
            writer.write_string("MATRIX_SCALE_BLK128x128_32F")
        elif self == Self.MATRIX_SCALE_END:
            writer.write_string("MATRIX_SCALE_END")
        else:
            abort("invalid MatmulMatrixScale entry")

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cublasLtBatchMode_t(TrivialRegisterPassable, Writable):
    """Batch mode."""

    var _value: Int32
    comptime STRIDED = Self(0)
    """
    The matrices of each instance of the batch are located at fixed offsets in number of elements from their locations
    in the previous instance.
    """
    comptime POINTER_ARRAY = Self(1)
    """
    The address of the matrix of each instance of the batch are read from arrays of pointers.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.STRIDED:
            writer.write_string("BATCH_MODE_STRIDED")
        elif self == Self.POINTER_ARRAY:
            writer.write_string("BATCH_MODE_POINTER_ARRAY")
        else:
            abort("invalid cublasLtBatchMode_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct LayoutAttribute(TrivialRegisterPassable, Writable):
    """Attributes of memory layout ."""

    var _value: Int32
    comptime TYPE = Self(0)
    """Data type, see cudaDataType.

    uint32_t.
    """
    comptime ORDER = Self(1)
    """Memory order of the data, see Order.

    int32_t, default: COL.
    """
    comptime ROWS = Self(2)
    """Number of rows.

    Usually only values that can be expressed as int32_t are supported.

    uint64_t.
    """
    comptime COLS = Self(3)
    """Number of columns.

    Usually only values that can be expressed as int32_t are supported.

    uint64_t.
    """
    comptime LD = Self(4)
    """Matrix leading dimension.

    For COL this is stride (in elements) of matrix column, for more details and documentation for
    other memory orders see documentation for Order values.

    Currently only non-negative values are supported, must be large enough so that matrix memory locations are not
    overlapping (e.g. greater or equal to ROWS in case of COL).

    int64_t;.
    """
    comptime BATCH_COUNT = Self(5)
    """Number of matmul operations to perform in the batch.

    See also STRIDED_BATCH_SUPPORT

    int32_t, default: 1.
    """
    comptime STRIDED_BATCH_OFFSET = Self(6)
    """Stride (in elements) to the next matrix for strided batch operation.

    When matrix type is planar-complex (PLANE_OFFSET != 0), batch stride
    is interpreted by cublasLtMatmul() in number of real valued sub-elements. E.g. for data of type CUDA_C_16F,
    offset of 1024B is encoded as a stride of value 512 (since each element of the real and imaginary matrices
    is a 2B (16bit) floating point type).

    NOTE: A bug in cublasLtMatrixTransform() causes it to interpret the batch stride for a planar-complex matrix
    as if it was specified in number of complex elements. Therefore an offset of 1024B must be encoded as stride
    value 256 when calling cublasLtMatrixTransform() (each complex element is 4B with real and imaginary values 2B
    each). This behavior is expected to be corrected in the next major cuBLAS version.

    int64_t, default: 0.
    """
    comptime PLANE_OFFSET = Self(7)
    """Stride (in bytes) to the imaginary plane for planar complex layout.

    int64_t, default: 0 - 0 means that layout is regular (real and imaginary parts of complex numbers are interleaved
    in memory in each element).
    """

    comptime BATCH_MODE = Self(8)
    """Batch mode.
    uint32_t, default: 0 - 0 means that batch mode is CUBLASLT_BATCH_MODE_STRIDED.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.TYPE:
            writer.write_string("TYPE")
        elif self == Self.ORDER:
            writer.write_string("ORDER")
        elif self == Self.ROWS:
            writer.write_string("ROWS")
        elif self == Self.COLS:
            writer.write_string("COLS")
        elif self == Self.LD:
            writer.write_string("LD")
        elif self == Self.BATCH_COUNT:
            writer.write_string("BATCH_COUNT")
        elif self == Self.STRIDED_BATCH_OFFSET:
            writer.write_string("STRIDED_BATCH_OFFSET")
        elif self == Self.PLANE_OFFSET:
            writer.write_string("PLANE_OFFSET")
        elif self == Self.BATCH_MODE:
            writer.write_string("BATCH_MODE")
        else:
            abort("invalid LayoutAttribute entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtDestroy(light_handle: cublasLtHandle_t) raises -> Result:
    return _get_dylib_function[
        "cublasLtDestroy", def(type_of(light_handle)) thin -> Result
    ]()(light_handle)


def cublasLtGetCudartVersion() raises -> Int:
    return _get_dylib_function[
        "cublasLtGetCudartVersion", def() thin -> Int
    ]()()


def cublasLtMatmulAlgoConfigGetAttribute(
    algo: UnsafePointer[MatmulAlgorithm, ImmutAnyOrigin],
    attr: AlgorithmConfig,
    buf: OpaquePointer[MutAnyOrigin],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int, MutAnyOrigin],
) raises -> Result:
    """Get algo configuration attribute.

    algo         The algo descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoConfigGetAttribute",
        def(
            UnsafePointer[MatmulAlgorithm, ImmutAnyOrigin],
            AlgorithmConfig,
            OpaquePointer[MutAnyOrigin],
            Int,
            UnsafePointer[Int, MutAnyOrigin],
        ) thin -> Result,
    ]()(algo, attr, buf, size_in_bytes, size_written)


def cublasLtLoggerForceDisable() raises -> Result:
    """Experimental: Disable logging for the entire session.

    \retval     CUBLAS_STATUS_SUCCESS        if disabled logging

    Raises:
        If the dynamic library cannot be found.
    ."""
    return _get_dylib_function[
        "cublasLtLoggerForceDisable", def() thin -> Result
    ]()()


def cublasLtHeuristicsCacheGetCapacity(
    capacity: UnsafePointer[Int, MutAnyOrigin],
) raises -> Result:
    return _get_dylib_function[
        "cublasLtHeuristicsCacheGetCapacity",
        def(UnsafePointer[Int, MutAnyOrigin]) thin -> Result,
    ]()(capacity)


def cublasLtDisableCpuInstructionsSetMask(mask: Int16) raises -> Int16:
    """Restricts usage of CPU instructions (ISA) specified by the flags in the mask.

    Flags can be combined with bitwise OR(|) operator. Supported flags:
    - 0x1 -- x86-64 AVX512 ISA

    Default mask: 0 (any applicable ISA is allowed).

    The function returns the previous value of the mask.
    The function takes precedence over the environment variable CUBLASLT_DISABLE_CPU_INSTRUCTIONS_MASK.

    Raises:
        If the dynamic library cannot be found.
    ."""
    return _get_dylib_function[
        "cublasLtDisableCpuInstructionsSetMask", def(Int16) thin raises -> Int16
    ]()(mask)


def cublasLtLoggerSetLevel(level: Int16) raises -> Result:
    """Experimental: Log level setter.

    level                        log level, should be one of the following:
                                            0. Off
                                            1. Errors
                                            2. Performance Trace
                                            3. Performance Hints
                                            4. Heuristics Trace
                                            5. API Trace

    \retval     CUBLAS_STATUS_INVALID_VALUE  if log level is not one of the above levels

    \retval     CUBLAS_STATUS_SUCCESS        if log level was set successfully

    Raises:
        If the dynamic library cannot be found.
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetLevel", def(Int16) thin -> Result
    ]()(level)


@fieldwise_init
struct Stages(TrivialRegisterPassable, Writable):
    """Size and number of stages in which elements are read into shared memory.

    General order of stages IDs is sorted by stage size first and by number of stages second.
    ."""

    var _value: Int32
    comptime STAGES_UNDEFINED = Self(0)
    comptime STAGES_16x1 = Self(1)
    comptime STAGES_16x2 = Self(2)
    comptime STAGES_16x3 = Self(3)
    comptime STAGES_16x4 = Self(4)
    comptime STAGES_16x5 = Self(5)
    comptime STAGES_16x6 = Self(6)
    comptime STAGES_32x1 = Self(7)
    comptime STAGES_32x2 = Self(8)
    comptime STAGES_32x3 = Self(9)
    comptime STAGES_32x4 = Self(10)
    comptime STAGES_32x5 = Self(11)
    comptime STAGES_32x6 = Self(12)
    comptime STAGES_64x1 = Self(13)
    comptime STAGES_64x2 = Self(14)
    comptime STAGES_64x3 = Self(15)
    comptime STAGES_64x4 = Self(16)
    comptime STAGES_64x5 = Self(17)
    comptime STAGES_64x6 = Self(18)
    comptime STAGES_128x1 = Self(19)
    comptime STAGES_128x2 = Self(20)
    comptime STAGES_128x3 = Self(21)
    comptime STAGES_128x4 = Self(22)
    comptime STAGES_128x5 = Self(23)
    comptime STAGES_128x6 = Self(24)
    comptime STAGES_32x10 = Self(25)
    comptime STAGES_8x4 = Self(26)
    comptime STAGES_16x10 = Self(27)
    comptime STAGES_8x5 = Self(28)
    comptime STAGES_8x3 = Self(31)
    comptime STAGES_8xAUTO = Self(32)
    comptime STAGES_16xAUTO = Self(33)
    comptime STAGES_32xAUTO = Self(34)
    comptime STAGES_64xAUTO = Self(35)
    comptime STAGES_128xAUTO = Self(36)
    comptime STAGES_256xAUTO = Self(37)
    comptime STAGES_END = Self(38)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.STAGES_UNDEFINED:
            writer.write_string("STAGES_UNDEFINED")
        elif self == Self.STAGES_16x1:
            writer.write_string("STAGES_16x1")
        elif self == Self.STAGES_16x2:
            writer.write_string("STAGES_16x2")
        elif self == Self.STAGES_16x3:
            writer.write_string("STAGES_16x3")
        elif self == Self.STAGES_16x4:
            writer.write_string("STAGES_16x4")
        elif self == Self.STAGES_16x5:
            writer.write_string("STAGES_16x5")
        elif self == Self.STAGES_16x6:
            writer.write_string("STAGES_16x6")
        elif self == Self.STAGES_32x1:
            writer.write_string("STAGES_32x1")
        elif self == Self.STAGES_32x2:
            writer.write_string("STAGES_32x2")
        elif self == Self.STAGES_32x3:
            writer.write_string("STAGES_32x3")
        elif self == Self.STAGES_32x4:
            writer.write_string("STAGES_32x4")
        elif self == Self.STAGES_32x5:
            writer.write_string("STAGES_32x5")
        elif self == Self.STAGES_32x6:
            writer.write_string("STAGES_32x6")
        elif self == Self.STAGES_64x1:
            writer.write_string("STAGES_64x1")
        elif self == Self.STAGES_64x2:
            writer.write_string("STAGES_64x2")
        elif self == Self.STAGES_64x3:
            writer.write_string("STAGES_64x3")
        elif self == Self.STAGES_64x4:
            writer.write_string("STAGES_64x4")
        elif self == Self.STAGES_64x5:
            writer.write_string("STAGES_64x5")
        elif self == Self.STAGES_64x6:
            writer.write_string("STAGES_64x6")
        elif self == Self.STAGES_128x1:
            writer.write_string("STAGES_128x1")
        elif self == Self.STAGES_128x2:
            writer.write_string("STAGES_128x2")
        elif self == Self.STAGES_128x3:
            writer.write_string("STAGES_128x3")
        elif self == Self.STAGES_128x4:
            writer.write_string("STAGES_128x4")
        elif self == Self.STAGES_128x5:
            writer.write_string("STAGES_128x5")
        elif self == Self.STAGES_128x6:
            writer.write_string("STAGES_128x6")
        elif self == Self.STAGES_32x10:
            writer.write_string("STAGES_32x10")
        elif self == Self.STAGES_8x4:
            writer.write_string("STAGES_8x4")
        elif self == Self.STAGES_16x10:
            writer.write_string("STAGES_16x10")
        elif self == Self.STAGES_8x5:
            writer.write_string("STAGES_8x5")
        elif self == Self.STAGES_8x3:
            writer.write_string("STAGES_8x3")
        elif self == Self.STAGES_8xAUTO:
            writer.write_string("STAGES_8xAUTO")
        elif self == Self.STAGES_16xAUTO:
            writer.write_string("STAGES_16xAUTO")
        elif self == Self.STAGES_32xAUTO:
            writer.write_string("STAGES_32xAUTO")
        elif self == Self.STAGES_64xAUTO:
            writer.write_string("STAGES_64xAUTO")
        elif self == Self.STAGES_128xAUTO:
            writer.write_string("STAGES_128xAUTO")
        elif self == Self.STAGES_256xAUTO:
            writer.write_string("STAGES_256xAUTO")
        elif self == Self.STAGES_END:
            writer.write_string("STAGES_END")
        else:
            abort("invalid Stages entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtMatmulDescDestroy(
    matmul_desc: cublasLtMatmulDesc_t,
) raises -> Result:
    """Destroy matmul operation descriptor.

    \retval     CUBLAS_STATUS_SUCCESS  if operation was successful
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescDestroy",
        def(cublasLtMatmulDesc_t) thin -> Result,
    ]()(matmul_desc)


def cublasLtMatrixTransformDescSetAttribute(
    transform_desc: cublasLtMatrixTransformDesc_t,
    attr: TransformDescriptor,
    buf: OpaquePointer[ImmutAnyOrigin],
    size_in_bytes: Int,
) raises -> Result:
    """Set matrix transform operation descriptor attribute.

    transformDesc  The descriptor
    attr           The attribute
    buf            memory address containing the new value
    sizeInBytes    size of buf buffer for verification (in bytes)

    \retval     CUBLAS_STATUS_INVALID_VALUE  if buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute was set successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescSetAttribute",
        def(
            cublasLtMatrixTransformDesc_t,
            TransformDescriptor,
            OpaquePointer[ImmutAnyOrigin],
            Int,
        ) thin -> Result,
    ]()(transform_desc, attr, buf, size_in_bytes)


def cublasLtMatmulPreferenceGetAttribute(
    pref: cublasLtMatmulPreference_t,
    attr: Preference,
    buf: OpaquePointer[MutAnyOrigin],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int, MutAnyOrigin],
) raises -> Result:
    """Get matmul heuristic search preference descriptor attribute.

    pref         The descriptor
    attr         The attribute
    buf          memory address containing the new value
    sizeInBytes  size of buf buffer for verification (in bytes)
    sizeWritten  only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number of
                            bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceGetAttribute",
        def(
            cublasLtMatmulPreference_t,
            Preference,
            OpaquePointer[MutAnyOrigin],
            Int,
            UnsafePointer[Int, MutAnyOrigin],
        ) thin -> Result,
    ]()(pref, attr, buf, size_in_bytes, size_written)


def cublasLtMatmulAlgoInit(
    light_handle: cublasLtHandle_t,
    compute_type: ComputeType,
    scale_type: DataType,
    _atype: DataType,
    _btype: DataType,
    _ctype: DataType,
    _dtype: DataType,
    algo_id: Int16,
    algo: UnsafePointer[MatmulAlgorithm, MutAnyOrigin],
) raises -> Result:
    """Initialize algo structure.

    \retval     CUBLAS_STATUS_INVALID_VALUE  if algo is NULL or algoId is outside of recognized range
    \retval     CUBLAS_STATUS_NOT_SUPPORTED  if algoId is not supported for given combination of data types
    \retval     CUBLAS_STATUS_SUCCESS        if the structure was successfully initialized
    ."""
    return _get_dylib_function[
        "cublasLtMatmulAlgoInit",
        def(
            type_of(light_handle),
            ComputeType,
            DataType,
            DataType,
            DataType,
            DataType,
            DataType,
            Int16,
            UnsafePointer[MatmulAlgorithm, MutAnyOrigin],
        ) thin -> Result,
    ]()(
        light_handle,
        compute_type,
        scale_type,
        _atype,
        _btype,
        _ctype,
        _dtype,
        algo_id,
        algo,
    )


@fieldwise_init
struct Epilogue(TrivialRegisterPassable, Writable):
    """Postprocessing options for the epilogue
    ."""

    var _value: Int32
    comptime DEFAULT = Self(1)
    """No special postprocessing, just scale and quantize results if necessary.
    """
    comptime RELU = Self(2)
    """ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).
    """
    comptime RELU_AUX = Self(Self.RELU._value | 128)
    """ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime BIAS = Self(4)
    """Bias, apply (broadcasted) Bias from bias vector. Bias vector length must match matrix D rows, it must be packed
    (stride between vector elements is 1). Bias vector is broadcasted to all columns and added before applying final
    postprocessing.
    """
    comptime RELU_BIAS = Self(Self.RELU._value | Self.BIAS._value)
    """ReLu and Bias, apply Bias and then ReLu transform.
    """
    comptime RELU_AUX_BIAS = Self(Self.RELU_AUX._value | Self.BIAS._value)
    """ReLu and Bias, apply Bias and then ReLu transform.

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime DRELU = Self(8 | 128)
    """DReLu, apply derivative of ReLu transform.

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime DRELU_BGRAD = Self(Self.DRELU._value | 16)
    """DReLu with bias gradient.

    This epilogue mode produces an extra output, a ReLu bit-mask matrix,
    see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime GELU = Self(32)
    """GELU, apply GELU point-wise transform to the results (x:=GELU(x)).
    """
    comptime GELU_AUX = Self(Self.GELU._value | 128)
    """GELU, apply GELU point-wise transform to the results (x:=GELU(x)).

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime GELU_BIAS = Self(Self.GELU._value | Self.BIAS._value)
    """GELU and Bias, apply Bias and then GELU transform.
    """
    comptime GELU_AUX_BIAS = Self(Self.GELU_AUX._value | Self.BIAS._value)
    """GELU and Bias, apply Bias and then GELU transform.

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime DGELU = Self(64 | 128)
    """DGELU, apply derivative of GELU transform.

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime DGELU_BGRAD = Self(Self.DGELU._value | 16)
    """DGELU with bias gradient.

    This epilogue mode outputs GELU input as a separate matrix (useful for training).
    See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
    """
    comptime BGRADA = Self(256)
    """Bias gradient based on the input matrix A.

    The bias size corresponds to the number of rows of the matrix D.
    The reduction happens over the GEMM's "k" dimension.

    Stores Bias gradient in the auxiliary output
    (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
    """
    comptime BGRADB = Self(512)
    """Bias gradient based on the input matrix B.

    The bias size corresponds to the number of columns of the matrix D.
    The reduction happens over the GEMM's "k" dimension.

    Stores Bias gradient in the auxiliary output
    (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.DEFAULT:
            writer.write_string("DEFAULT")
        elif self == Self.RELU:
            writer.write_string("RELU")
        elif self == Self.RELU_AUX:
            writer.write_string("RELU_AUX")
        elif self == Self.BIAS:
            writer.write_string("BIAS")
        elif self == Self.RELU_BIAS:
            writer.write_string("RELU_BIAS")
        elif self == Self.RELU_AUX_BIAS:
            writer.write_string("RELU_AUX_BIAS")
        elif self == Self.DRELU:
            writer.write_string("DRELU")
        elif self == Self.DRELU_BGRAD:
            writer.write_string("DRELU_BGRAD")
        elif self == Self.GELU:
            writer.write_string("GELU")
        elif self == Self.GELU_AUX:
            writer.write_string("GELU_AUX")
        elif self == Self.GELU_BIAS:
            writer.write_string("GELU_BIAS")
        elif self == Self.GELU_AUX_BIAS:
            writer.write_string("GELU_AUX_BIAS")
        elif self == Self.DGELU:
            writer.write_string("DGELU")
        elif self == Self.DGELU_BGRAD:
            writer.write_string("DGELU_BGRAD")
        elif self == Self.BGRADA:
            writer.write_string("BGRADA")
        elif self == Self.BGRADB:
            writer.write_string("BGRADB")
        else:
            abort("invalid Epilogue entry")

    def __int__(self) -> Int:
        return Int(self._value)


struct Descriptor(TrivialRegisterPassable):
    """Semi-opaque descriptor for cublasLtMatmul() operation details
    ."""

    var data: StaticTuple[UInt64, 32]


def cublasLtMatrixLayoutCreate(
    mat_layout: UnsafePointer[cublasLtMatrixLayout_t, MutAnyOrigin],
    type: DataType,
    rows: UInt64,
    cols: UInt64,
    ld: Int64,
) raises -> Result:
    """Create new matrix layout descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if descriptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixLayoutCreate",
        def(
            UnsafePointer[cublasLtMatrixLayout_t, MutAnyOrigin],
            DataType,
            UInt64,
            UInt64,
            Int64,
        ) thin -> Result,
    ]()(mat_layout, type, rows, cols, ld)


@fieldwise_init
struct PointerModeMask(TrivialRegisterPassable, Writable):
    """Mask to define pointer mode capability."""

    var _value: Int32
    comptime HOST = Self(1)
    """See HOST."""
    comptime DEVICE = Self(2)
    """See DEVICE."""
    comptime DEVICE_VECTOR = Self(4)
    """See DEVICE_VECTOR."""
    comptime ALPHA_DEVICE_VECTOR_BETA_ZERO = Self(8)
    """See ALPHA_DEVICE_VECTOR_BETA_ZERO."""
    comptime ALPHA_DEVICE_VECTOR_BETA_HOST = Self(16)
    """See ALPHA_DEVICE_VECTOR_BETA_HOST."""

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        if self == Self.HOST:
            writer.write_string("HOST")
        elif self == Self.DEVICE:
            writer.write_string("DEVICE")
        elif self == Self.DEVICE_VECTOR:
            writer.write_string("DEVICE_VECTOR")
        elif self == Self.ALPHA_DEVICE_VECTOR_BETA_ZERO:
            writer.write_string("ALPHA_DEVICE_VECTOR_BETA_ZERO")
        elif self == Self.ALPHA_DEVICE_VECTOR_BETA_HOST:
            writer.write_string("ALPHA_DEVICE_VECTOR_BETA_HOST")
        else:
            abort("invalid PointerModeMask entry")

    def __int__(self) -> Int:
        return Int(self._value)


struct MatrixLayout(TrivialRegisterPassable):
    """Semi-opaque descriptor for matrix memory layout
    ."""

    var data: StaticTuple[UInt64, 8]


def cublasLtMatmulDescCreate(
    matmul_desc: UnsafePointer[cublasLtMatmulDesc_t, MutAnyOrigin],
    compute_type: ComputeType,
    scale_type: DataType,
) raises -> Result:
    """Create new matmul operation descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if descriptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescCreate",
        def(
            UnsafePointer[cublasLtMatmulDesc_t, MutAnyOrigin],
            ComputeType,
            DataType,
        ) thin -> Result,
    ]()(matmul_desc, compute_type, scale_type)


@fieldwise_init
struct Tile(TrivialRegisterPassable, Writable):
    """Tile size (in C/D matrix Rows x Cols).

    General order of tile IDs is sorted by size first and by first dimension second.
    ."""

    var _value: Int32
    comptime TILE_UNDEFINED = Self(0)
    comptime TILE_8x8 = Self(1)
    comptime TILE_8x16 = Self(2)
    comptime TILE_16x8 = Self(3)
    comptime TILE_8x32 = Self(4)
    comptime TILE_16x16 = Self(5)
    comptime TILE_32x8 = Self(6)
    comptime TILE_8x64 = Self(7)
    comptime TILE_16x32 = Self(8)
    comptime TILE_32x16 = Self(9)
    comptime TILE_64x8 = Self(10)
    comptime TILE_32x32 = Self(11)
    comptime TILE_32x64 = Self(12)
    comptime TILE_64x32 = Self(13)
    comptime TILE_32x128 = Self(14)
    comptime TILE_64x64 = Self(15)
    comptime TILE_128x32 = Self(16)
    comptime TILE_64x128 = Self(17)
    comptime TILE_128x64 = Self(18)
    comptime TILE_64x256 = Self(19)
    comptime TILE_128x128 = Self(20)
    comptime TILE_256x64 = Self(21)
    comptime TILE_64x512 = Self(22)
    comptime TILE_128x256 = Self(23)
    comptime TILE_256x128 = Self(24)
    comptime TILE_512x64 = Self(25)
    comptime TILE_64x96 = Self(26)
    comptime TILE_96x64 = Self(27)
    comptime TILE_96x128 = Self(28)
    comptime TILE_128x160 = Self(29)
    comptime TILE_160x128 = Self(30)
    comptime TILE_192x128 = Self(31)
    comptime TILE_128x192 = Self(32)
    comptime TILE_128x96 = Self(33)
    comptime TILE_32x256 = Self(34)
    comptime TILE_256x32 = Self(35)
    comptime TILE_8x128 = Self(36)
    comptime TILE_8x192 = Self(37)
    comptime TILE_8x256 = Self(38)
    comptime TILE_8x320 = Self(39)
    comptime TILE_8x384 = Self(40)
    comptime TILE_8x448 = Self(41)
    comptime TILE_8x512 = Self(42)
    comptime TILE_8x576 = Self(43)
    comptime TILE_8x640 = Self(44)
    comptime TILE_8x704 = Self(45)
    comptime TILE_8x768 = Self(46)
    comptime TILE_16x64 = Self(47)
    comptime TILE_16x128 = Self(48)
    comptime TILE_16x192 = Self(49)
    comptime TILE_16x256 = Self(50)
    comptime TILE_16x320 = Self(51)
    comptime TILE_16x384 = Self(52)
    comptime TILE_16x448 = Self(53)
    comptime TILE_16x512 = Self(54)
    comptime TILE_16x576 = Self(55)
    comptime TILE_16x640 = Self(56)
    comptime TILE_16x704 = Self(57)
    comptime TILE_16x768 = Self(58)
    comptime TILE_24x64 = Self(59)
    comptime TILE_24x128 = Self(60)
    comptime TILE_24x192 = Self(61)
    comptime TILE_24x256 = Self(62)
    comptime TILE_24x320 = Self(63)
    comptime TILE_24x384 = Self(64)
    comptime TILE_24x448 = Self(65)
    comptime TILE_24x512 = Self(66)
    comptime TILE_24x576 = Self(67)
    comptime TILE_24x640 = Self(68)
    comptime TILE_24x704 = Self(69)
    comptime TILE_24x768 = Self(70)
    comptime TILE_32x192 = Self(71)
    comptime TILE_32x320 = Self(72)
    comptime TILE_32x384 = Self(73)
    comptime TILE_32x448 = Self(74)
    comptime TILE_32x512 = Self(75)
    comptime TILE_32x576 = Self(76)
    comptime TILE_32x640 = Self(77)
    comptime TILE_32x704 = Self(78)
    comptime TILE_32x768 = Self(79)
    comptime TILE_40x64 = Self(80)
    comptime TILE_40x128 = Self(81)
    comptime TILE_40x192 = Self(82)
    comptime TILE_40x256 = Self(83)
    comptime TILE_40x320 = Self(84)
    comptime TILE_40x384 = Self(85)
    comptime TILE_40x448 = Self(86)
    comptime TILE_40x512 = Self(87)
    comptime TILE_40x576 = Self(88)
    comptime TILE_40x640 = Self(89)
    comptime TILE_40x704 = Self(90)
    comptime TILE_40x768 = Self(91)
    comptime TILE_48x64 = Self(92)
    comptime TILE_48x128 = Self(93)
    comptime TILE_48x192 = Self(94)
    comptime TILE_48x256 = Self(95)
    comptime TILE_48x320 = Self(96)
    comptime TILE_48x384 = Self(97)
    comptime TILE_48x448 = Self(98)
    comptime TILE_48x512 = Self(99)
    comptime TILE_48x576 = Self(100)
    comptime TILE_48x640 = Self(101)
    comptime TILE_48x704 = Self(102)
    comptime TILE_48x768 = Self(103)
    comptime TILE_56x64 = Self(104)
    comptime TILE_56x128 = Self(105)
    comptime TILE_56x192 = Self(106)
    comptime TILE_56x256 = Self(107)
    comptime TILE_56x320 = Self(108)
    comptime TILE_56x384 = Self(109)
    comptime TILE_56x448 = Self(110)
    comptime TILE_56x512 = Self(111)
    comptime TILE_56x576 = Self(112)
    comptime TILE_56x640 = Self(113)
    comptime TILE_56x704 = Self(114)
    comptime TILE_56x768 = Self(115)
    comptime TILE_64x192 = Self(116)
    comptime TILE_64x320 = Self(117)
    comptime TILE_64x384 = Self(118)
    comptime TILE_64x448 = Self(119)
    comptime TILE_64x576 = Self(120)
    comptime TILE_64x640 = Self(121)
    comptime TILE_64x704 = Self(122)
    comptime TILE_64x768 = Self(123)
    comptime TILE_72x64 = Self(124)
    comptime TILE_72x128 = Self(125)
    comptime TILE_72x192 = Self(126)
    comptime TILE_72x256 = Self(127)
    comptime TILE_72x320 = Self(128)
    comptime TILE_72x384 = Self(129)
    comptime TILE_72x448 = Self(130)
    comptime TILE_72x512 = Self(131)
    comptime TILE_72x576 = Self(132)
    comptime TILE_72x640 = Self(133)
    comptime TILE_80x64 = Self(134)
    comptime TILE_80x128 = Self(135)
    comptime TILE_80x192 = Self(136)
    comptime TILE_80x256 = Self(137)
    comptime TILE_80x320 = Self(138)
    comptime TILE_80x384 = Self(139)
    comptime TILE_80x448 = Self(140)
    comptime TILE_80x512 = Self(141)
    comptime TILE_80x576 = Self(142)
    comptime TILE_88x64 = Self(143)
    comptime TILE_88x128 = Self(144)
    comptime TILE_88x192 = Self(145)
    comptime TILE_88x256 = Self(146)
    comptime TILE_88x320 = Self(147)
    comptime TILE_88x384 = Self(148)
    comptime TILE_88x448 = Self(149)
    comptime TILE_88x512 = Self(150)
    comptime TILE_96x192 = Self(151)
    comptime TILE_96x256 = Self(152)
    comptime TILE_96x320 = Self(153)
    comptime TILE_96x384 = Self(154)
    comptime TILE_96x448 = Self(155)
    comptime TILE_96x512 = Self(156)
    comptime TILE_104x64 = Self(157)
    comptime TILE_104x128 = Self(158)
    comptime TILE_104x192 = Self(159)
    comptime TILE_104x256 = Self(160)
    comptime TILE_104x320 = Self(161)
    comptime TILE_104x384 = Self(162)
    comptime TILE_104x448 = Self(163)
    comptime TILE_112x64 = Self(164)
    comptime TILE_112x128 = Self(165)
    comptime TILE_112x192 = Self(166)
    comptime TILE_112x256 = Self(167)
    comptime TILE_112x320 = Self(168)
    comptime TILE_112x384 = Self(169)
    comptime TILE_120x64 = Self(170)
    comptime TILE_120x128 = Self(171)
    comptime TILE_120x192 = Self(172)
    comptime TILE_120x256 = Self(173)
    comptime TILE_120x320 = Self(174)
    comptime TILE_120x384 = Self(175)
    comptime TILE_128x320 = Self(176)
    comptime TILE_128x384 = Self(177)
    comptime TILE_136x64 = Self(178)
    comptime TILE_136x128 = Self(179)
    comptime TILE_136x192 = Self(180)
    comptime TILE_136x256 = Self(181)
    comptime TILE_136x320 = Self(182)
    comptime TILE_144x64 = Self(183)
    comptime TILE_144x128 = Self(184)
    comptime TILE_144x192 = Self(185)
    comptime TILE_144x256 = Self(186)
    comptime TILE_144x320 = Self(187)
    comptime TILE_152x64 = Self(188)
    comptime TILE_152x128 = Self(189)
    comptime TILE_152x192 = Self(190)
    comptime TILE_152x256 = Self(191)
    comptime TILE_152x320 = Self(192)
    comptime TILE_160x64 = Self(193)
    comptime TILE_160x192 = Self(194)
    comptime TILE_160x256 = Self(195)
    comptime TILE_168x64 = Self(196)
    comptime TILE_168x128 = Self(197)
    comptime TILE_168x192 = Self(198)
    comptime TILE_168x256 = Self(199)
    comptime TILE_176x64 = Self(200)
    comptime TILE_176x128 = Self(201)
    comptime TILE_176x192 = Self(202)
    comptime TILE_176x256 = Self(203)
    comptime TILE_184x64 = Self(204)
    comptime TILE_184x128 = Self(205)
    comptime TILE_184x192 = Self(206)
    comptime TILE_184x256 = Self(207)
    comptime TILE_192x64 = Self(208)
    comptime TILE_192x192 = Self(209)
    comptime TILE_192x256 = Self(210)
    comptime TILE_200x64 = Self(211)
    comptime TILE_200x128 = Self(212)
    comptime TILE_200x192 = Self(213)
    comptime TILE_208x64 = Self(214)
    comptime TILE_208x128 = Self(215)
    comptime TILE_208x192 = Self(216)
    comptime TILE_216x64 = Self(217)
    comptime TILE_216x128 = Self(218)
    comptime TILE_216x192 = Self(219)
    comptime TILE_224x64 = Self(220)
    comptime TILE_224x128 = Self(221)
    comptime TILE_224x192 = Self(222)
    comptime TILE_232x64 = Self(223)
    comptime TILE_232x128 = Self(224)
    comptime TILE_232x192 = Self(225)
    comptime TILE_240x64 = Self(226)
    comptime TILE_240x128 = Self(227)
    comptime TILE_240x192 = Self(228)
    comptime TILE_248x64 = Self(229)
    comptime TILE_248x128 = Self(230)
    comptime TILE_248x192 = Self(231)
    comptime TILE_256x192 = Self(232)
    comptime TILE_264x64 = Self(233)
    comptime TILE_264x128 = Self(234)
    comptime TILE_272x64 = Self(235)
    comptime TILE_272x128 = Self(236)
    comptime TILE_280x64 = Self(237)
    comptime TILE_280x128 = Self(238)
    comptime TILE_288x64 = Self(239)
    comptime TILE_288x128 = Self(240)
    comptime TILE_296x64 = Self(241)
    comptime TILE_296x128 = Self(242)
    comptime TILE_304x64 = Self(243)
    comptime TILE_304x128 = Self(244)
    comptime TILE_312x64 = Self(245)
    comptime TILE_312x128 = Self(246)
    comptime TILE_320x64 = Self(247)
    comptime TILE_320x128 = Self(248)
    comptime TILE_328x64 = Self(249)
    comptime TILE_328x128 = Self(250)
    comptime TILE_336x64 = Self(251)
    comptime TILE_336x128 = Self(252)
    comptime TILE_344x64 = Self(253)
    comptime TILE_344x128 = Self(254)
    comptime TILE_352x64 = Self(255)
    comptime TILE_352x128 = Self(256)
    comptime TILE_360x64 = Self(257)
    comptime TILE_360x128 = Self(258)
    comptime TILE_368x64 = Self(259)
    comptime TILE_368x128 = Self(260)
    comptime TILE_376x64 = Self(261)
    comptime TILE_376x128 = Self(262)
    comptime TILE_384x64 = Self(263)
    comptime TILE_384x128 = Self(264)
    comptime TILE_392x64 = Self(265)
    comptime TILE_400x64 = Self(266)
    comptime TILE_408x64 = Self(267)
    comptime TILE_416x64 = Self(268)
    comptime TILE_424x64 = Self(269)
    comptime TILE_432x64 = Self(270)
    comptime TILE_440x64 = Self(271)
    comptime TILE_448x64 = Self(272)
    comptime TILE_456x64 = Self(273)
    comptime TILE_464x64 = Self(274)
    comptime TILE_472x64 = Self(275)
    comptime TILE_480x64 = Self(276)
    comptime TILE_488x64 = Self(277)
    comptime TILE_496x64 = Self(278)
    comptime TILE_504x64 = Self(279)
    comptime TILE_520x64 = Self(280)
    comptime TILE_528x64 = Self(281)
    comptime TILE_536x64 = Self(282)
    comptime TILE_544x64 = Self(283)
    comptime TILE_552x64 = Self(284)
    comptime TILE_560x64 = Self(285)
    comptime TILE_568x64 = Self(286)
    comptime TILE_576x64 = Self(287)
    comptime TILE_584x64 = Self(288)
    comptime TILE_592x64 = Self(289)
    comptime TILE_600x64 = Self(290)
    comptime TILE_608x64 = Self(291)
    comptime TILE_616x64 = Self(292)
    comptime TILE_624x64 = Self(293)
    comptime TILE_632x64 = Self(294)
    comptime TILE_640x64 = Self(295)
    comptime TILE_648x64 = Self(296)
    comptime TILE_656x64 = Self(297)
    comptime TILE_664x64 = Self(298)
    comptime TILE_672x64 = Self(299)
    comptime TILE_680x64 = Self(300)
    comptime TILE_688x64 = Self(301)
    comptime TILE_696x64 = Self(302)
    comptime TILE_704x64 = Self(303)
    comptime TILE_712x64 = Self(304)
    comptime TILE_720x64 = Self(305)
    comptime TILE_728x64 = Self(306)
    comptime TILE_736x64 = Self(307)
    comptime TILE_744x64 = Self(308)
    comptime TILE_752x64 = Self(309)
    comptime TILE_760x64 = Self(310)
    comptime TILE_768x64 = Self(311)
    comptime TILE_64x16 = Self(312)
    comptime TILE_64x24 = Self(313)
    comptime TILE_64x40 = Self(314)
    comptime TILE_64x48 = Self(315)
    comptime TILE_64x56 = Self(316)
    comptime TILE_64x72 = Self(317)
    comptime TILE_64x80 = Self(318)
    comptime TILE_64x88 = Self(319)
    comptime TILE_64x104 = Self(320)
    comptime TILE_64x112 = Self(321)
    comptime TILE_64x120 = Self(322)
    comptime TILE_64x136 = Self(323)
    comptime TILE_64x144 = Self(324)
    comptime TILE_64x152 = Self(325)
    comptime TILE_64x160 = Self(326)
    comptime TILE_64x168 = Self(327)
    comptime TILE_64x176 = Self(328)
    comptime TILE_64x184 = Self(329)
    comptime TILE_64x200 = Self(330)
    comptime TILE_64x208 = Self(331)
    comptime TILE_64x216 = Self(332)
    comptime TILE_64x224 = Self(333)
    comptime TILE_64x232 = Self(334)
    comptime TILE_64x240 = Self(335)
    comptime TILE_64x248 = Self(336)
    comptime TILE_64x264 = Self(337)
    comptime TILE_64x272 = Self(338)
    comptime TILE_64x280 = Self(339)
    comptime TILE_64x288 = Self(340)
    comptime TILE_64x296 = Self(341)
    comptime TILE_64x304 = Self(342)
    comptime TILE_64x312 = Self(343)
    comptime TILE_64x328 = Self(344)
    comptime TILE_64x336 = Self(345)
    comptime TILE_64x344 = Self(346)
    comptime TILE_64x352 = Self(347)
    comptime TILE_64x360 = Self(348)
    comptime TILE_64x368 = Self(349)
    comptime TILE_64x376 = Self(350)
    comptime TILE_64x392 = Self(351)
    comptime TILE_64x400 = Self(352)
    comptime TILE_64x408 = Self(353)
    comptime TILE_64x416 = Self(354)
    comptime TILE_64x424 = Self(355)
    comptime TILE_64x432 = Self(356)
    comptime TILE_64x440 = Self(357)
    comptime TILE_64x456 = Self(358)
    comptime TILE_64x464 = Self(359)
    comptime TILE_64x472 = Self(360)
    comptime TILE_64x480 = Self(361)
    comptime TILE_64x488 = Self(362)
    comptime TILE_64x496 = Self(363)
    comptime TILE_64x504 = Self(364)
    comptime TILE_64x520 = Self(365)
    comptime TILE_64x528 = Self(366)
    comptime TILE_64x536 = Self(367)
    comptime TILE_64x544 = Self(368)
    comptime TILE_64x552 = Self(369)
    comptime TILE_64x560 = Self(370)
    comptime TILE_64x568 = Self(371)
    comptime TILE_64x584 = Self(372)
    comptime TILE_64x592 = Self(373)
    comptime TILE_64x600 = Self(374)
    comptime TILE_64x608 = Self(375)
    comptime TILE_64x616 = Self(376)
    comptime TILE_64x624 = Self(377)
    comptime TILE_64x632 = Self(378)
    comptime TILE_64x648 = Self(379)
    comptime TILE_64x656 = Self(380)
    comptime TILE_64x664 = Self(381)
    comptime TILE_64x672 = Self(382)
    comptime TILE_64x680 = Self(383)
    comptime TILE_64x688 = Self(384)
    comptime TILE_64x696 = Self(385)
    comptime TILE_64x712 = Self(386)
    comptime TILE_64x720 = Self(387)
    comptime TILE_64x728 = Self(388)
    comptime TILE_64x736 = Self(389)
    comptime TILE_64x744 = Self(390)
    comptime TILE_64x752 = Self(391)
    comptime TILE_64x760 = Self(392)
    comptime TILE_128x8 = Self(393)
    comptime TILE_128x16 = Self(394)
    comptime TILE_128x24 = Self(395)
    comptime TILE_128x40 = Self(396)
    comptime TILE_128x48 = Self(397)
    comptime TILE_128x56 = Self(398)
    comptime TILE_128x72 = Self(399)
    comptime TILE_128x80 = Self(400)
    comptime TILE_128x88 = Self(401)
    comptime TILE_128x104 = Self(402)
    comptime TILE_128x112 = Self(403)
    comptime TILE_128x120 = Self(404)
    comptime TILE_128x136 = Self(405)
    comptime TILE_128x144 = Self(406)
    comptime TILE_128x152 = Self(407)
    comptime TILE_128x168 = Self(408)
    comptime TILE_128x176 = Self(409)
    comptime TILE_128x184 = Self(410)
    comptime TILE_128x200 = Self(411)
    comptime TILE_128x208 = Self(412)
    comptime TILE_128x216 = Self(413)
    comptime TILE_128x224 = Self(414)
    comptime TILE_128x232 = Self(415)
    comptime TILE_128x240 = Self(416)
    comptime TILE_128x248 = Self(417)
    comptime TILE_128x264 = Self(418)
    comptime TILE_128x272 = Self(419)
    comptime TILE_128x280 = Self(420)
    comptime TILE_128x288 = Self(421)
    comptime TILE_128x296 = Self(422)
    comptime TILE_128x304 = Self(423)
    comptime TILE_128x312 = Self(424)
    comptime TILE_128x328 = Self(425)
    comptime TILE_128x336 = Self(426)
    comptime TILE_128x344 = Self(427)
    comptime TILE_128x352 = Self(428)
    comptime TILE_128x360 = Self(429)
    comptime TILE_128x368 = Self(430)
    comptime TILE_128x376 = Self(431)
    comptime TILE_128x392 = Self(432)
    comptime TILE_128x400 = Self(433)
    comptime TILE_128x408 = Self(434)
    comptime TILE_128x416 = Self(435)
    comptime TILE_128x424 = Self(436)
    comptime TILE_128x432 = Self(437)
    comptime TILE_128x440 = Self(438)
    comptime TILE_128x448 = Self(439)
    comptime TILE_128x456 = Self(440)
    comptime TILE_128x464 = Self(441)
    comptime TILE_128x472 = Self(442)
    comptime TILE_128x480 = Self(443)
    comptime TILE_128x488 = Self(444)
    comptime TILE_128x496 = Self(445)
    comptime TILE_128x504 = Self(446)
    comptime TILE_128x512 = Self(447)
    comptime TILE_192x8 = Self(448)
    comptime TILE_192x16 = Self(449)
    comptime TILE_192x24 = Self(450)
    comptime TILE_192x32 = Self(451)
    comptime TILE_192x40 = Self(452)
    comptime TILE_192x48 = Self(453)
    comptime TILE_192x56 = Self(454)
    comptime TILE_192x72 = Self(455)
    comptime TILE_192x80 = Self(456)
    comptime TILE_192x88 = Self(457)
    comptime TILE_192x96 = Self(458)
    comptime TILE_192x104 = Self(459)
    comptime TILE_192x112 = Self(460)
    comptime TILE_192x120 = Self(461)
    comptime TILE_192x136 = Self(462)
    comptime TILE_192x144 = Self(463)
    comptime TILE_192x152 = Self(464)
    comptime TILE_192x160 = Self(465)
    comptime TILE_192x168 = Self(466)
    comptime TILE_192x176 = Self(467)
    comptime TILE_192x184 = Self(468)
    comptime TILE_192x200 = Self(469)
    comptime TILE_192x208 = Self(470)
    comptime TILE_192x216 = Self(471)
    comptime TILE_192x224 = Self(472)
    comptime TILE_192x232 = Self(473)
    comptime TILE_192x240 = Self(474)
    comptime TILE_192x248 = Self(475)
    comptime TILE_192x264 = Self(476)
    comptime TILE_192x272 = Self(477)
    comptime TILE_192x280 = Self(478)
    comptime TILE_192x288 = Self(479)
    comptime TILE_192x296 = Self(480)
    comptime TILE_192x304 = Self(481)
    comptime TILE_192x312 = Self(482)
    comptime TILE_192x320 = Self(483)
    comptime TILE_192x328 = Self(484)
    comptime TILE_192x336 = Self(485)
    comptime TILE_256x8 = Self(486)
    comptime TILE_256x16 = Self(487)
    comptime TILE_256x24 = Self(488)
    comptime TILE_256x40 = Self(489)
    comptime TILE_256x48 = Self(490)
    comptime TILE_256x56 = Self(491)
    comptime TILE_256x72 = Self(492)
    comptime TILE_256x80 = Self(493)
    comptime TILE_256x88 = Self(494)
    comptime TILE_256x96 = Self(495)
    comptime TILE_256x104 = Self(496)
    comptime TILE_256x112 = Self(497)
    comptime TILE_256x120 = Self(498)
    comptime TILE_256x136 = Self(499)
    comptime TILE_256x144 = Self(500)
    comptime TILE_256x152 = Self(501)
    comptime TILE_256x160 = Self(502)
    comptime TILE_256x168 = Self(503)
    comptime TILE_256x176 = Self(504)
    comptime TILE_256x184 = Self(505)
    comptime TILE_256x200 = Self(506)
    comptime TILE_256x208 = Self(507)
    comptime TILE_256x216 = Self(508)
    comptime TILE_256x224 = Self(509)
    comptime TILE_256x232 = Self(510)
    comptime TILE_256x240 = Self(511)
    comptime TILE_256x248 = Self(512)
    comptime TILE_256x256 = Self(513)
    comptime TILE_320x8 = Self(514)
    comptime TILE_320x16 = Self(515)
    comptime TILE_320x24 = Self(516)
    comptime TILE_320x32 = Self(517)
    comptime TILE_320x40 = Self(518)
    comptime TILE_320x48 = Self(519)
    comptime TILE_320x56 = Self(520)
    comptime TILE_320x72 = Self(521)
    comptime TILE_320x80 = Self(522)
    comptime TILE_320x88 = Self(523)
    comptime TILE_320x96 = Self(524)
    comptime TILE_320x104 = Self(525)
    comptime TILE_320x112 = Self(526)
    comptime TILE_320x120 = Self(527)
    comptime TILE_320x136 = Self(528)
    comptime TILE_320x144 = Self(529)
    comptime TILE_320x152 = Self(530)
    comptime TILE_320x160 = Self(531)
    comptime TILE_320x168 = Self(532)
    comptime TILE_320x176 = Self(533)
    comptime TILE_320x184 = Self(534)
    comptime TILE_320x192 = Self(535)
    comptime TILE_320x200 = Self(536)
    comptime TILE_384x8 = Self(537)
    comptime TILE_384x16 = Self(538)
    comptime TILE_384x24 = Self(539)
    comptime TILE_384x32 = Self(540)
    comptime TILE_384x40 = Self(541)
    comptime TILE_384x48 = Self(542)
    comptime TILE_384x56 = Self(543)
    comptime TILE_384x72 = Self(544)
    comptime TILE_384x80 = Self(545)
    comptime TILE_384x88 = Self(546)
    comptime TILE_384x96 = Self(547)
    comptime TILE_384x104 = Self(548)
    comptime TILE_384x112 = Self(549)
    comptime TILE_384x120 = Self(550)
    comptime TILE_384x136 = Self(551)
    comptime TILE_384x144 = Self(552)
    comptime TILE_384x152 = Self(553)
    comptime TILE_384x160 = Self(554)
    comptime TILE_384x168 = Self(555)
    comptime TILE_448x8 = Self(556)
    comptime TILE_448x16 = Self(557)
    comptime TILE_448x24 = Self(558)
    comptime TILE_448x32 = Self(559)
    comptime TILE_448x40 = Self(560)
    comptime TILE_448x48 = Self(561)
    comptime TILE_448x56 = Self(562)
    comptime TILE_448x72 = Self(563)
    comptime TILE_448x80 = Self(564)
    comptime TILE_448x88 = Self(565)
    comptime TILE_448x96 = Self(566)
    comptime TILE_448x104 = Self(567)
    comptime TILE_448x112 = Self(568)
    comptime TILE_448x120 = Self(569)
    comptime TILE_448x128 = Self(570)
    comptime TILE_448x136 = Self(571)
    comptime TILE_448x144 = Self(572)
    comptime TILE_512x8 = Self(573)
    comptime TILE_512x16 = Self(574)
    comptime TILE_512x24 = Self(575)
    comptime TILE_512x32 = Self(576)
    comptime TILE_512x40 = Self(577)
    comptime TILE_512x48 = Self(578)
    comptime TILE_512x56 = Self(579)
    comptime TILE_512x72 = Self(580)
    comptime TILE_512x80 = Self(581)
    comptime TILE_512x88 = Self(582)
    comptime TILE_512x96 = Self(583)
    comptime TILE_512x104 = Self(584)
    comptime TILE_512x112 = Self(585)
    comptime TILE_512x120 = Self(586)
    comptime TILE_512x128 = Self(587)
    comptime TILE_576x8 = Self(588)
    comptime TILE_576x16 = Self(589)
    comptime TILE_576x24 = Self(590)
    comptime TILE_576x32 = Self(591)
    comptime TILE_576x40 = Self(592)
    comptime TILE_576x48 = Self(593)
    comptime TILE_576x56 = Self(594)
    comptime TILE_576x72 = Self(595)
    comptime TILE_576x80 = Self(596)
    comptime TILE_576x88 = Self(597)
    comptime TILE_576x96 = Self(598)
    comptime TILE_576x104 = Self(599)
    comptime TILE_576x112 = Self(600)
    comptime TILE_640x8 = Self(601)
    comptime TILE_640x16 = Self(602)
    comptime TILE_640x24 = Self(603)
    comptime TILE_640x32 = Self(604)
    comptime TILE_640x40 = Self(605)
    comptime TILE_640x48 = Self(606)
    comptime TILE_640x56 = Self(607)
    comptime TILE_640x72 = Self(608)
    comptime TILE_640x80 = Self(609)
    comptime TILE_640x88 = Self(610)
    comptime TILE_640x96 = Self(611)
    comptime TILE_704x8 = Self(612)
    comptime TILE_704x16 = Self(613)
    comptime TILE_704x24 = Self(614)
    comptime TILE_704x32 = Self(615)
    comptime TILE_704x40 = Self(616)
    comptime TILE_704x48 = Self(617)
    comptime TILE_704x56 = Self(618)
    comptime TILE_704x72 = Self(619)
    comptime TILE_704x80 = Self(620)
    comptime TILE_704x88 = Self(621)
    comptime TILE_768x8 = Self(622)
    comptime TILE_768x16 = Self(623)
    comptime TILE_768x24 = Self(624)
    comptime TILE_768x32 = Self(625)
    comptime TILE_768x40 = Self(626)
    comptime TILE_768x48 = Self(627)
    comptime TILE_768x56 = Self(628)
    comptime TILE_768x72 = Self(629)
    comptime TILE_768x80 = Self(630)
    comptime TILE_256x512 = Self(631)
    comptime TILE_256x1024 = Self(632)
    comptime TILE_512x512 = Self(633)
    comptime TILE_512x1024 = Self(634)

    comptime TILE_END = Self(635)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.TILE_UNDEFINED:
            writer.write_string("TILE_UNDEFINED")
        elif self == Self.TILE_8x8:
            writer.write_string("TILE_8x8")
        elif self == Self.TILE_8x16:
            writer.write_string("TILE_8x16")
        elif self == Self.TILE_16x8:
            writer.write_string("TILE_16x8")
        elif self == Self.TILE_8x32:
            writer.write_string("TILE_8x32")
        elif self == Self.TILE_16x16:
            writer.write_string("TILE_16x16")
        elif self == Self.TILE_32x8:
            writer.write_string("TILE_32x8")
        elif self == Self.TILE_8x64:
            writer.write_string("TILE_8x64")
        elif self == Self.TILE_16x32:
            writer.write_string("TILE_16x32")
        elif self == Self.TILE_32x16:
            writer.write_string("TILE_32x16")
        elif self == Self.TILE_64x8:
            writer.write_string("TILE_64x8")
        elif self == Self.TILE_32x32:
            writer.write_string("TILE_32x32")
        elif self == Self.TILE_32x64:
            writer.write_string("TILE_32x64")
        elif self == Self.TILE_64x32:
            writer.write_string("TILE_64x32")
        elif self == Self.TILE_32x128:
            writer.write_string("TILE_32x128")
        elif self == Self.TILE_64x64:
            writer.write_string("TILE_64x64")
        elif self == Self.TILE_128x32:
            writer.write_string("TILE_128x32")
        elif self == Self.TILE_64x128:
            writer.write_string("TILE_64x128")
        elif self == Self.TILE_128x64:
            writer.write_string("TILE_128x64")
        elif self == Self.TILE_64x256:
            writer.write_string("TILE_64x256")
        elif self == Self.TILE_128x128:
            writer.write_string("TILE_128x128")
        elif self == Self.TILE_256x64:
            writer.write_string("TILE_256x64")
        elif self == Self.TILE_64x512:
            writer.write_string("TILE_64x512")
        elif self == Self.TILE_128x256:
            writer.write_string("TILE_128x256")
        elif self == Self.TILE_256x128:
            writer.write_string("TILE_256x128")
        elif self == Self.TILE_512x64:
            writer.write_string("TILE_512x64")
        elif self == Self.TILE_64x96:
            writer.write_string("TILE_64x96")
        elif self == Self.TILE_96x64:
            writer.write_string("TILE_96x64")
        elif self == Self.TILE_96x128:
            writer.write_string("TILE_96x128")
        elif self == Self.TILE_128x160:
            writer.write_string("TILE_128x160")
        elif self == Self.TILE_160x128:
            writer.write_string("TILE_160x128")
        elif self == Self.TILE_192x128:
            writer.write_string("TILE_192x128")
        elif self == Self.TILE_128x192:
            writer.write_string("TILE_128x192")
        elif self == Self.TILE_128x96:
            writer.write_string("TILE_128x96")
        elif self == Self.TILE_32x256:
            writer.write_string("TILE_32x256")
        elif self == Self.TILE_256x32:
            writer.write_string("TILE_256x32")
        elif self == Self.TILE_END:
            writer.write_string("TILE_END")
        else:
            abort("invalid Tile entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasLtGetStatusName(
    status: Result,
) raises -> UnsafePointer[Int8, ImmutAnyOrigin]:
    return _get_dylib_function[
        "cublasLtGetStatusName",
        def(Result) thin raises -> UnsafePointer[Int8, ImmutAnyOrigin],
    ]()(status)


def cublasLtMatmulPreferenceCreate(
    pref: UnsafePointer[cublasLtMatmulPreference_t, MutAnyOrigin],
) raises -> Result:
    """Create new matmul heuristic search preference descriptor.

    \retval     CUBLAS_STATUS_ALLOC_FAILED  if memory could not be allocated
    \retval     CUBLAS_STATUS_SUCCESS       if descriptor was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceCreate",
        def(
            UnsafePointer[cublasLtMatmulPreference_t, MutAnyOrigin]
        ) thin -> Result,
    ]()(pref)


struct cublasLtMatmulHeuristicResult_t(Defaultable, TrivialRegisterPassable):
    """Results structure used by cublasLtMatmulGetAlgo.

    Holds returned configured algo descriptor and its runtime properties.
    ."""

    # Matmul algorithm descriptor.
    #
    # Must be initialized with cublasLtMatmulAlgoInit() if preferences' CUBLASLT_MATMUL_PERF_SEARCH_MODE is set to
    # LIMITED_BY_ALGO_ID
    # .
    var algo: MatmulAlgorithm
    # Actual size of workspace memory required.
    # .
    var workspaceSize: Int
    # Result status, other fields are only valid if after call to cublasLtMatmulAlgoGetHeuristic() this member is set to
    # CUBLAS_STATUS_SUCCESS.
    # .
    var state: Result
    # Waves count - a device utilization metric.
    #
    # wavesCount value of 1.0f suggests that when kernel is launched it will fully occupy the GPU.
    # .
    var wavesCount: Float32
    var reserved: StaticTuple[Int32, 4]

    def __init__(out self):
        self.algo = MatmulAlgorithm()
        self.workspaceSize = 0
        self.state = Result.NOT_INITIALIZED
        self.wavesCount = 0.0
        self.reserved = StaticTuple[Int32, 4](0)


def cublasLtLoggerSetFile(file: OpaquePointer[MutAnyOrigin]) raises -> Result:
    """Experimental: Log file setter.

    file                         an open file with write permissions

    \retval     CUBLAS_STATUS_SUCCESS        if log file was set successfully

    Raises:
        If the dynamic library cannot be found.
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetFile", def(OpaquePointer[MutAnyOrigin]) thin -> Result
    ]()(file)


def cublasLtLoggerOpenFile(
    log_file: UnsafePointer[Int8, ImmutAnyOrigin]
) raises -> Result:
    """Experimental: Open log file.

    logFile                      log file path. if the log file does not exist, it will be created

    \retval     CUBLAS_STATUS_SUCCESS        if log file was created successfully
    ."""
    return _get_dylib_function[
        "cublasLtLoggerOpenFile",
        def(UnsafePointer[Int8, ImmutAnyOrigin]) thin -> Result,
    ]()(log_file)


def cublasLtMatrixTransform(
    light_handle: cublasLtHandle_t,
    transform_desc: cublasLtMatrixTransformDesc_t,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _adesc: cublasLtMatrixLayout_t,
    beta: OpaquePointer[ImmutAnyOrigin],
    _b: OpaquePointer[ImmutAnyOrigin],
    _bdesc: cublasLtMatrixLayout_t,
    _c: OpaquePointer[MutAnyOrigin],
    _cdesc: cublasLtMatrixLayout_t,
    stream: _CUstream_st,
) raises -> Result:
    """Matrix layout conversion helper (C = alpha * op(A) + beta * op(B)).

    Can be used to change memory order of data or to scale and shift the values.

    \retval     CUBLAS_STATUS_NOT_INITIALIZED   if cuBLASLt handle has not been initialized
    \retval     CUBLAS_STATUS_INVALID_VALUE     if parameters are in conflict or in an impossible configuration; e.g.
                                               when A is not NULL, but Adesc is NULL
    \retval     CUBLAS_STATUS_NOT_SUPPORTED     if current implementation on selected device doesn't support configured
                                               operation
    \retval     CUBLAS_STATUS_ARCH_MISMATCH     if configured operation cannot be run using selected device
    \retval     CUBLAS_STATUS_EXECUTION_FAILED  if cuda reported execution error from the device
    \retval     CUBLAS_STATUS_SUCCESS           if the operation completed successfully
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransform",
        def(
            type_of(light_handle),
            cublasLtMatrixTransformDesc_t,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            cublasLtMatrixLayout_t,
            OpaquePointer[ImmutAnyOrigin],
            OpaquePointer[ImmutAnyOrigin],
            cublasLtMatrixLayout_t,
            OpaquePointer[MutAnyOrigin],
            cublasLtMatrixLayout_t,
            _CUstream_st,
        ) thin -> Result,
    ]()(
        light_handle,
        transform_desc,
        alpha,
        _a,
        _adesc,
        beta,
        _b,
        _bdesc,
        _c,
        _cdesc,
        stream,
    )


def cublasLtLoggerSetMask(mask: Int16) raises -> Result:
    """Experimental: Log mask setter.

    mask                         log mask, should be a combination of the following masks:
                                            0.  Off
                                            1.  Errors
                                            2.  Performance Trace
                                            4.  Performance Hints
                                            8.  Heuristics Trace
                                            16. API Trace

    \retval     CUBLAS_STATUS_SUCCESS        if log mask was set successfully

    Raises:
        If the dynamic library cannot be found.
    ."""
    return _get_dylib_function[
        "cublasLtLoggerSetMask", def(Int16) thin -> Result
    ]()(mask)


# Opaque structure holding CUBLASLT context (typedef defined at top of file)


def cublasLtMatrixTransformDescGetAttribute(
    transform_desc: cublasLtMatrixTransformDesc_t,
    attr: TransformDescriptor,
    buf: OpaquePointer[MutAnyOrigin],
    size_in_bytes: Int,
    size_written: UnsafePointer[Int, MutAnyOrigin],
) raises -> Result:
    """Get matrix transform operation descriptor attribute.

    transformDesc  The descriptor
    attr           The attribute
    buf            memory address containing the new value
    sizeInBytes    size of buf buffer for verification (in bytes)
    sizeWritten    only valid when return value is CUBLAS_STATUS_SUCCESS. If sizeInBytes is non-zero: number
    of bytes actually written, if sizeInBytes is 0: number of bytes needed to write full contents

    \retval     CUBLAS_STATUS_INVALID_VALUE  if sizeInBytes is 0 and sizeWritten is NULL, or if  sizeInBytes is non-zero
                                            and buf is NULL or sizeInBytes doesn't match size of internal storage for
                                            selected attribute
    \retval     CUBLAS_STATUS_SUCCESS        if attribute's value was successfully written to user memory
    ."""
    return _get_dylib_function[
        "cublasLtMatrixTransformDescGetAttribute",
        def(
            cublasLtMatrixTransformDesc_t,
            TransformDescriptor,
            OpaquePointer[MutAnyOrigin],
            Int,
            UnsafePointer[Int, MutAnyOrigin],
        ) thin -> Result,
    ]()(transform_desc, attr, buf, size_in_bytes, size_written)


def cublasLtMatmulDescInit_internal(
    matmul_desc: cublasLtMatmulDesc_t,
    size: Int,
    compute_type: ComputeType,
    scale_type: DataType,
) raises -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatmulDescInit_internal",
        def(
            cublasLtMatmulDesc_t,
            Int,
            ComputeType,
            DataType,
        ) thin -> Result,
    ]()(matmul_desc, size, compute_type, scale_type)


def cublasLtMatmulPreferenceInit_internal(
    pref: cublasLtMatmulPreference_t, size: Int
) raises -> Result:
    """Internal. Do not use directly.
    ."""
    return _get_dylib_function[
        "cublasLtMatmulPreferenceInit_internal",
        def(cublasLtMatmulPreference_t, Int) thin -> Result,
    ]()(pref, size)


struct Transform(TrivialRegisterPassable):
    """Semi-opaque descriptor for cublasLtMatrixTransform() operation details
    ."""

    var data: StaticTuple[UInt64, 8]  # uint64_t data[8]


@fieldwise_init
struct TransformDescriptor(TrivialRegisterPassable, Writable):
    """Matrix transform descriptor attributes to define details of the operation.
    ."""

    var _value: Int32
    comptime SCALE_TYPE = TransformDescriptor(0)
    """Scale type, see cudaDataType. Inputs are converted to scale type for scaling and summation and results are then
    converted to output type to store in memory.

    int32_t.
    """
    comptime POINTER_MODE = TransformDescriptor(1)
    """UnsafePointer mode of alpha and beta, see PointerMode.

    int32_t, default: HOST.
    """
    comptime TRANSA = TransformDescriptor(2)
    """Transform of matrix A, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """
    comptime TRANSB = TransformDescriptor(3)
    """Transform of matrix B, see cublasOperation_t.

    int32_t, default: CUBLAS_OP_N.
    """

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.SCALE_TYPE:
            writer.write_string("SCALE_TYPE")
        elif self == Self.POINTER_MODE:
            writer.write_string("POINTER_MODE")
        elif self == Self.TRANSA:
            writer.write_string("TRANSA")
        elif self == Self.TRANSB:
            writer.write_string("TRANSB")
        else:
            abort("invalid TransformDescriptor entry")

    def __int__(self) -> Int:
        return Int(self._value)


# def cublasLtMatmulAlgoGetIds(
#     light_handle: cublasLtHandle_t,
#     compute_type: ComputeType,
#     scale_type: DataType,
#     _atype: DataType,
#     _btype: DataType,
#     _ctype: DataType,
#     _dtype: DataType,
#     requested_algo_count: Int16,
#     algo_ids_array: UNKNOWN,
#     return_algo_count: UnsafePointer[Int16, MutAnyOrigin],
# )raises -> Result:
#     """Routine to get all algo IDs that can potentially run

#     int              requestedAlgoCount requested number of algos (must be less or equal to size of algoIdsA
#     (in elements)) algoIdsA         array to write algoIds to returnAlgoCount  number of algoIds
#     actually written

#     \retval     CUBLAS_STATUS_INVALID_VALUE  if requestedAlgoCount is less or equal to zero
#     \retval     CUBLAS_STATUS_SUCCESS        if query was successful, inspect returnAlgoCount to get actual number of IDs
#                                             available
#     ."""
#     return _get_dylib_function[
#         "cublasLtMatmulAlgoGetIds",
#         def (
#             cublasLtHandle_t,
#             ComputeType,
#             DataType,
#             DataType,
#             DataType,
#             DataType,
#             DataType,
#             Int16,
#             UNKNOWN,
#             UnsafePointer[Int16],
#         ) -> Result,
#     ]()(
#         light_handle,
#         compute_type,
#         scale_type,
#         _atype,
#         _btype,
#         _ctype,
#         _dtype,
#         requested_algo_count,
#         algo_ids_array,
#         return_algo_count,
#     )
