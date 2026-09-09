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
from std.utils import StaticTuple

from max.gpu.host._amdgpu_hip import hipStream_t

comptime hipblasLtHandle_t = Optional[OpaquePointer[MutAnyOrigin]]
comptime hipblasLtMatmulDesc_t = Optional[OpaquePointer[MutAnyOrigin]]
comptime hipblasLtMatrixLayout_t = Optional[OpaquePointer[MutAnyOrigin]]
comptime hipblasLtMatmulPreference_t = Optional[OpaquePointer[MutAnyOrigin]]


@fieldwise_init
struct Status(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime SUCCESS = Self(0)
    comptime NOT_INITIALIZED = Self(1)
    comptime ALLOC_FAILED = Self(2)
    comptime INVALID_VALUE = Self(3)
    comptime MAPPING_ERROR = Self(4)
    comptime EXECUTION_FAILED = Self(5)
    comptime INTERNAL_ERROR = Self(6)
    comptime NOT_SUPPORTED = Self(7)
    comptime ARCH_MISMATCH = Self(8)
    comptime HANDLE_IS_NULLPTR = Self(9)
    comptime INVALID_ENUM = Self(10)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.SUCCESS:
            return writer.write_string("SUCCESS")
        if self == Self.NOT_INITIALIZED:
            return writer.write_string("NOT_INITIALIZED")
        if self == Self.ALLOC_FAILED:
            return writer.write_string("ALLOC_FAILED")
        if self == Self.INVALID_VALUE:
            return writer.write_string("INVALID_VALUE")
        if self == Self.MAPPING_ERROR:
            return writer.write_string("MAPPING_ERROR")
        if self == Self.EXECUTION_FAILED:
            return writer.write_string("EXECUTION_FAILED")
        if self == Self.INTERNAL_ERROR:
            return writer.write_string("INTERNAL_ERROR")
        if self == Self.NOT_SUPPORTED:
            return writer.write_string("NOT_SUPPORTED")
        if self == Self.ARCH_MISMATCH:
            return writer.write_string("ARCH_MISMATCH")
        if self == Self.HANDLE_IS_NULLPTR:
            return writer.write_string("HANDLE_IS_NULLPTR")
        if self == Self.INVALID_ENUM:
            return writer.write_string("INVALID_ENUM")

        abort("unreachable: invalid Status entry")

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct hipDataType_t(TrivialRegisterPassable):
    var _value: Int32
    comptime R_32F = Self(0)
    comptime R_64F = Self(1)
    comptime R_16F = Self(2)
    comptime R_8I = Self(3)
    comptime R_16BF = Self(14)
    comptime R_8F_E4M3 = Self(28)
    comptime R_8F_E5M2 = Self(29)
    comptime R_4F_E2M1 = Self(33)
    comptime R_8F_E4M3_FNUZ = Self(1000)
    comptime R_8F_E5M2_FNUZ = Self(1001)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasComputeType_t(TrivialRegisterPassable):
    var _value: Int32
    comptime COMPUTE_16F = Self(0)
    comptime COMPUTE_16F_PEDANTIC = Self(1)
    comptime COMPUTE_32F = Self(2)
    comptime COMPUTE_32F_PEDANTIC = Self(3)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasOperation_t(TrivialRegisterPassable):
    var _value: Int32
    comptime OP_N = Self(111)
    comptime OP_T = Self(112)
    comptime OP_C = Self(113)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasLtOrder_t(TrivialRegisterPassable):
    var _value: Int32
    comptime COL = Self(0)
    comptime ROW = Self(1)
    comptime COL16_4R16 = Self(100)
    comptime COL16_4R8 = Self(101)
    comptime COL16_4R4 = Self(102)
    comptime COL16_4R2 = Self(103)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasLtMatmulMatrixScale_t(TrivialRegisterPassable):
    var _value: Int32
    comptime SCALAR_32F = Self(0)
    comptime VEC32_UE8M0 = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasLtMatmulDescAttributes_t(TrivialRegisterPassable):
    var _value: Int32
    comptime TRANSA = Self(0)
    comptime TRANSB = Self(1)
    comptime A_SCALE_POINTER = Self(5)
    comptime B_SCALE_POINTER = Self(6)
    comptime A_SCALE_MODE = Self(31)
    comptime B_SCALE_MODE = Self(32)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


@fieldwise_init
struct hipblasLtMatmulLayoutAttribute_t(TrivialRegisterPassable):
    var _value: Int32
    comptime BATCH_COUNT = Self(0)
    comptime STRIDED_BATCH_OFFSET = Self(1)
    comptime TYPE = Self(2)
    comptime ORDER = Self(3)
    comptime ROWS = Self(4)
    comptime COLS = Self(5)
    comptime LD = Self(6)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)


struct hipblasLtMatmulAlgo_t(Defaultable, TrivialRegisterPassable):
    var data: StaticTuple[UInt8, 16]
    var maxWorkspaceBytes: Int

    def __init__(out self):
        self.data = StaticTuple[UInt8, 16](0)
        self.maxWorkspaceBytes = 0


struct hipblasLtMatmulHeuristicResult_t(Defaultable, TrivialRegisterPassable):
    var algo: hipblasLtMatmulAlgo_t
    var workspaceSize: Int
    var state: Status
    var wavesCount: Float32
    var reserved: StaticTuple[Int32, 4]

    def __init__(out self):
        self.algo = hipblasLtMatmulAlgo_t()
        self.workspaceSize = 0
        self.state = Status.SUCCESS
        self.wavesCount = 1.0
        self.reserved = StaticTuple[Int32, 4](0)


# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime HIPBLASLT_LIBRARY_PATHS: List[Path] = [
    "libhipblaslt.so.0",
    "libhipblaslt.so.1",
    "/opt/rocm/lib/libhipblaslt.so.0",
    "/opt/rocm/lib/libhipblaslt.so.1",
]

comptime HIPBLASLT_LIBRARY = _Global["HIPBLASLT_LIBRARY", _init_dylib]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib["HIP BLAS LT"](materialize[HIPBLASLT_LIBRARY_PATHS]())


@always_inline
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        HIPBLASLT_LIBRARY(), func_name, result_type
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


def hipblasLtCreate(
    light_handle: UnsafePointer[hipblasLtHandle_t, _],
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtCreate",
        def(type_of(light_handle)) thin -> Status,
    ]()(light_handle)


def hipblasLtDestroy(light_handle: hipblasLtHandle_t) raises -> Status:
    return _get_dylib_function[
        "hipblasLtDestroy", def(hipblasLtHandle_t) thin -> Status
    ]()(light_handle)


def hipblasLtMatmulDescCreate(
    matmul_desc: UnsafePointer[hipblasLtMatmulDesc_t, _],
    compute_type: hipblasComputeType_t,
    scale_type: hipDataType_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulDescCreate",
        def(
            type_of(matmul_desc),
            hipblasComputeType_t,
            hipDataType_t,
        ) thin -> Status,
    ]()(matmul_desc, compute_type, scale_type)


def hipblasLtMatmulDescSetAttribute(
    matmul_desc: hipblasLtMatmulDesc_t,
    attr: hipblasLtMatmulDescAttributes_t,
    buf: OpaquePointer[_],
    size_in_bytes: Int,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulDescSetAttribute",
        def(
            hipblasLtMatmulDesc_t,
            hipblasLtMatmulDescAttributes_t,
            type_of(buf),
            Int,
        ) thin -> Status,
    ]()(matmul_desc, attr, buf, size_in_bytes)


def hipblasLtMatmulDescDestroy(
    matmul_desc: hipblasLtMatmulDesc_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulDescDestroy", def(hipblasLtMatmulDesc_t) thin -> Status
    ]()(matmul_desc)


def hipblasLtMatrixLayoutCreate(
    mat_layout: UnsafePointer[hipblasLtMatrixLayout_t, _],
    type: hipDataType_t,
    rows: UInt64,
    cols: UInt64,
    ld: Int64,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatrixLayoutCreate",
        def(
            type_of(mat_layout),
            hipDataType_t,
            UInt64,
            UInt64,
            Int64,
        ) thin -> Status,
    ]()(mat_layout, type, rows, cols, ld)


def hipblasLtMatrixLayoutSetAttribute(
    mat_layout: hipblasLtMatrixLayout_t,
    attr: hipblasLtMatmulLayoutAttribute_t,
    buf: OpaquePointer[_],
    size_in_bytes: Int,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatrixLayoutSetAttribute",
        def(
            hipblasLtMatrixLayout_t,
            hipblasLtMatmulLayoutAttribute_t,
            type_of(buf),
            Int,
        ) thin -> Status,
    ]()(mat_layout, attr, buf, size_in_bytes)


def hipblasLtMatrixLayoutDestroy(
    mat_layout: hipblasLtMatrixLayout_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatrixLayoutDestroy",
        def(hipblasLtMatrixLayout_t) thin -> Status,
    ]()(mat_layout)


def hipblasLtMatmulPreferenceCreate(
    pref: UnsafePointer[hipblasLtMatmulPreference_t, _],
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulPreferenceCreate",
        def(type_of(pref)) thin -> Status,
    ]()(pref)


def hipblasLtMatmulAlgoGetHeuristic(
    light_handle: hipblasLtHandle_t,
    operation_desc: hipblasLtMatmulDesc_t,
    _adesc: hipblasLtMatrixLayout_t,
    _bdesc: hipblasLtMatrixLayout_t,
    _cdesc: hipblasLtMatrixLayout_t,
    _ddesc: hipblasLtMatrixLayout_t,
    preference: hipblasLtMatmulPreference_t,
    requested_algo_count: Int,
    heuristic_results_array: UnsafePointer[hipblasLtMatmulHeuristicResult_t, _],
    return_algo_count: UnsafePointer[Int, _],
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulAlgoGetHeuristic",
        def(
            hipblasLtHandle_t,
            hipblasLtMatmulDesc_t,
            hipblasLtMatrixLayout_t,
            hipblasLtMatrixLayout_t,
            hipblasLtMatrixLayout_t,
            hipblasLtMatrixLayout_t,
            hipblasLtMatmulPreference_t,
            Int,
            type_of(heuristic_results_array),
            type_of(return_algo_count),
        ) thin -> Status,
    ]()(
        light_handle,
        operation_desc,
        _adesc,
        _bdesc,
        _cdesc,
        _ddesc,
        preference,
        requested_algo_count,
        heuristic_results_array,
        return_algo_count,
    )


def hipblasLtMatmulPreferenceDestroy(
    pref: hipblasLtMatmulPreference_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmulPreferenceDestroy",
        def(hipblasLtMatmulPreference_t) thin -> Status,
    ]()(pref)


def hipblasLtMatmul(
    light_handle: hipblasLtHandle_t,
    compute_desc: hipblasLtMatmulDesc_t,
    alpha: OpaquePointer[_],
    _a: OpaquePointer[_],
    _adesc: hipblasLtMatrixLayout_t,
    _b: OpaquePointer[_],
    _bdesc: hipblasLtMatrixLayout_t,
    beta: OpaquePointer[_],
    _c: OptionalPointer[NoneType, _],
    _cdesc: hipblasLtMatrixLayout_t,
    _d: OptionalPointer[NoneType, _],
    _ddesc: hipblasLtMatrixLayout_t,
    algo: UnsafePointer[hipblasLtMatmulAlgo_t, _],
    workspace: OpaquePointer[_],
    workspace_size_in_bytes: Int,
    stream: hipStream_t,
) raises -> Status:
    return _get_dylib_function[
        "hipblasLtMatmul",
        def(
            hipblasLtHandle_t,
            hipblasLtMatmulDesc_t,
            type_of(alpha),
            type_of(_a),
            hipblasLtMatrixLayout_t,
            type_of(_b),
            hipblasLtMatrixLayout_t,
            type_of(beta),
            type_of(_c),
            hipblasLtMatrixLayout_t,
            type_of(_d),
            hipblasLtMatrixLayout_t,
            type_of(algo),
            type_of(workspace),
            Int,
            hipStream_t,
        ) thin -> Status,
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


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def _check_hipblas_error(status: Status) raises:
    if status != Status.SUCCESS:
        raise Error(t"HIPBLASLT ERROR:{status}")


@always_inline
def _convert_to_hip_datatype[dtype: DType]() -> hipDataType_t:
    comptime if dtype == .float32:
        return hipDataType_t.R_32F
    elif dtype == .float16:
        return hipDataType_t.R_16F
    elif dtype == .float8_e4m3fn:
        return hipDataType_t.R_8F_E4M3
    elif dtype == .float8_e5m2:
        return hipDataType_t.R_8F_E5M2
    elif dtype == .float8_e4m3fnuz:
        return hipDataType_t.R_8F_E4M3_FNUZ
    elif dtype == .float8_e5m2fnuz:
        return hipDataType_t.R_8F_E5M2_FNUZ
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # Replace this with float4-e2m1fn when GENAI-337 is fixed.
    elif dtype == .uint8:
        return hipDataType_t.R_4F_E2M1
    else:
        comptime assert dtype == .bfloat16, (
            "Only support FP32, FP16, BF16, E4M3(FNUZ), E5M2(FNUZ), and E2M1x2"
            " (UInt8). Please extend it if more dtypes are needed."
        )
        return hipDataType_t.R_16BF
