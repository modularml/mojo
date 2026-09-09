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

from max.gpu.host._nvidia_cuda import CUstream

from .dtype import DataType, Property
from .result import Result

comptime cublasContext = NoneType
comptime cublasHandle_t = OptionalPointer[cublasContext, _]

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUBLAS_LIBRARY_PATHS: List[Path] = [
    "libcublas.so.13",
    "/usr/local/cuda-13.1/lib64/libcublas.so.13",
    "/usr/local/cuda-13.0/lib64/libcublas.so.13",
    "/usr/local/cuda/lib64/libcublas.so.13",
    "libcublas.so.12",
    "/usr/local/cuda-12.8/lib64/libcublas.so.12",
    "/usr/local/cuda/lib64/libcublas.so.12",
]


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the cuBLAS libraries. Please make sure that "
            "the CUDA toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[CUDA_CUBLAS_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUBLAS_LIBRARY = _Global[
    "CUDA_CUBLAS_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUBLAS_LIBRARY_PATHS]()
    )


@always_inline
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUBLAS_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def check_cublas_error(stat: Result) raises:
    if stat != Result.SUCCESS:
        raise Error(t"failed to operate on CUBLAS due to error: {stat}")


@always_inline
def check_cublas_error(stat: Result, msg: StringSlice) raises:
    if stat != Result.SUCCESS:
        raise Error(t"{msg}. Got a CUBLAS error: {stat}")


@always_inline
def _convert_to_cublas_datatype[mojo_type: DType]() -> DataType:
    comptime if mojo_type == .float32:
        return DataType.R_32F
    elif mojo_type == .float16:
        return DataType.R_16F
    elif mojo_type == .float8_e4m3fn:
        return DataType.R_8F_E4M3
    elif mojo_type == .float8_e5m2:
        return DataType.R_8F_E5M2
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # Replace this with float4-e2m1fn when GENAI-337 is fixed.
    elif mojo_type == .uint8:
        return DataType.R_4F_E2M1
    else:
        comptime assert mojo_type == .bfloat16, (
            "Only support FP32, FP16, BF16, E4M3, E5M2, and E2M1x2 (UInt8)."
            " Please extend it if more types are needed."
        )
        return DataType.R_16BF


@always_inline
def _convert_to_cublas_transpose(transpose: Bool) -> cublasOperation_t:
    return (
        cublasOperation_t.CUBLAS_OP_T if transpose else cublasOperation_t.CUBLAS_OP_N
    )


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


def cublasScopy(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    y: UnsafePointer[Float32, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasScopy_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy)


def cublasDgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemv_v2",
        def(
            type_of(handle),
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


def cublasStpsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[Float32, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpsv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            type_of(_ap),
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


def cublasDgbmv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    kl: Int16,
    ku: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgbmv_v2",
        def(
            type_of(handle),
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


def cublasDgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    stride_a: Int64,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int64,
    stride_b: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int64,
    stride_c: Int64,
    batch_count: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemmStridedBatched_64",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            Int64,
            type_of(_b),
            Int64,
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
            Int64,
            Int64,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        lda,
        stride_a,
        _b,
        ldb,
        stride_b,
        beta,
        _c,
        ldc,
        stride_c,
        batch_count,
    )


def cublasDsyrkx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyrkx_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasUint8gemmBias(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    transc: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[mut=False, Int8, _],
    _a_bias: Int16,
    lda: Int16,
    _b: UnsafePointer[mut=False, Int8, _],
    _b_bias: Int16,
    ldb: Int16,
    _c: UnsafePointer[Int8, _],
    _c_bias: Int16,
    ldc: Int16,
    _c_mult: Int16,
    _c_shift: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasUint8gemmBias",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            Int16,
            type_of(_b),
            Int16,
            Int16,
            type_of(_c),
            Int16,
            Int16,
            Int16,
            Int16,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        transc,
        m,
        n,
        k,
        _a,
        _a_bias,
        lda,
        _b,
        _b_bias,
        ldb,
        _c,
        _c_bias,
        ldc,
        _c_mult,
        _c_shift,
    )


def cublasGetProperty(
    type: Property, value: UnsafePointer[Int16, _]
) raises -> Result:
    return _get_dylib_function[
        "cublasGetProperty",
        def(Property, type_of(value)) thin -> Result,
    ]()(type, value)


def cublasSsyr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    _a: UnsafePointer[Float32, _],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(_a),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


def cublasIdamax(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    result: UnsafePointer[Int16, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIdamax_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasGetMatrix(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int16,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMatrix",
        def(
            Int16,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
        ) thin -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


def cublasSgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    stride_a: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    stridex: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int16,
    stridey: Int64,
    batch_count: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemvStridedBatched",
        def(
            type_of(handle),
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            Int64,
            type_of(x),
            Int16,
            Int64,
            type_of(beta),
            type_of(y),
            Int16,
            Int64,
            Int16,
        ) thin -> Result,
    ]()(
        handle,
        trans,
        m,
        n,
        alpha,
        _a,
        lda,
        stride_a,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batch_count,
    )


def cublasStrsm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    _b: UnsafePointer[Float32, _],
    ldb: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrsm_v2",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
        ) thin -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


def cublasRotmEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    param: OpaquePointer[ImmutAnyOrigin],
    param_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotmEx",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(y),
            DataType,
            Int16,
            type_of(param),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        param,
        param_type,
        executiontype,
    )


def cublasSgemm(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemm_v2_64",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasSgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int64,
    _c: UnsafePointer[Float32, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgeam_64",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(beta),
            type_of(_b),
            Int64,
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


def cublasStrttp(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    _ap: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasStrttp",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(_a),
            Int16,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, _a, lda, _ap)


def cublasRotmgEx(
    handle: cublasHandle_t,
    d1: OpaquePointer[MutAnyOrigin],
    d1_type: DataType,
    d2: OpaquePointer[MutAnyOrigin],
    d2_type: DataType,
    x1: OpaquePointer[MutAnyOrigin],
    x1_type: DataType,
    y1: OpaquePointer[ImmutAnyOrigin],
    y1_type: DataType,
    param: OpaquePointer[MutAnyOrigin],
    param_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotmgEx",
        def(
            type_of(handle),
            type_of(d1),
            DataType,
            type_of(d2),
            DataType,
            type_of(x1),
            DataType,
            type_of(y1),
            DataType,
            type_of(param),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        d1,
        d1_type,
        d2,
        d2_type,
        x1,
        x1_type,
        y1,
        y1_type,
        param,
        param_type,
        executiontype,
    )


def cublasStrmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    x: UnsafePointer[Float32, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrmv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


@fieldwise_init
struct cublasPointerMode_t(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUBLAS_POINTER_MODE_HOST = cublasPointerMode_t(0)
    comptime CUBLAS_POINTER_MODE_DEVICE = cublasPointerMode_t(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.CUBLAS_POINTER_MODE_HOST:
            return writer.write_string("CUBLAS_POINTER_MODE_HOST")
        if self == Self.CUBLAS_POINTER_MODE_DEVICE:
            return writer.write_string("CUBLAS_POINTER_MODE_DEVICE")
        abort("invalid cublasPointerMode_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasDnrm2(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    result: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDnrm2_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasIaminEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    result: UnsafePointer[Int16, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIaminEx",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, result)


def cublasDger(
    handle: cublasHandle_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    y: UnsafePointer[mut=False, Float64, _],
    incy: Int64,
    _a: UnsafePointer[Float64, _],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDger_v2_64",
        def(
            type_of(handle),
            Int64,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(_a),
            Int64,
        ) thin -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


def cublasDgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    stride_a: Int64,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int16,
    stride_b: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int16,
    stride_c: Int64,
    batch_count: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemmStridedBatched",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            Int64,
            type_of(_b),
            Int16,
            Int64,
            type_of(beta),
            type_of(_c),
            Int16,
            Int64,
            Int16,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        lda,
        stride_a,
        _b,
        ldb,
        stride_b,
        beta,
        _c,
        ldc,
        stride_c,
        batch_count,
    )


@fieldwise_init
struct cublasMath_t(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUBLAS_DEFAULT_MATH = cublasMath_t(0)
    comptime CUBLAS_TENSOR_OP_MATH = cublasMath_t(1)
    comptime CUBLAS_PEDANTIC_MATH = cublasMath_t(2)
    comptime CUBLAS_TF32_TENSOR_OP_MATH = cublasMath_t(3)
    comptime CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = cublasMath_t(4)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.CUBLAS_DEFAULT_MATH:
            return writer.write_string("CUBLAS_DEFAULT_MATH")
        if self == Self.CUBLAS_TENSOR_OP_MATH:
            return writer.write_string("CUBLAS_TENSOR_OP_MATH")
        if self == Self.CUBLAS_PEDANTIC_MATH:
            return writer.write_string("CUBLAS_PEDANTIC_MATH")
        if self == Self.CUBLAS_TF32_TENSOR_OP_MATH:
            return writer.write_string("CUBLAS_TF32_TENSOR_OP_MATH")
        if self == Self.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION:
            return writer.write_string(
                "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION"
            )
        abort("invalid cublasMath_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasSdot(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    y: UnsafePointer[mut=False, Float32, _],
    incy: Int64,
    result: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSdot_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, result)


def cublasGetMatrixAsync(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int16,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int16,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMatrixAsync",
        def(
            Int16,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            CUstream,
        ) thin -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


def cublasGetVector(
    n: Int64,
    elem_size: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVector_64",
        def(
            Int64,
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(n, elem_size, x, incx, y, incy)


def cublasStrsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    x: UnsafePointer[Float32, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrsv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


def cublasSgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemv_v2_64",
        def(
            type_of(handle),
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


def cublasXerbla(sr_name: UnsafePointer[Int8, _], info: Int16) raises:
    return _get_dylib_function[
        "cublasXerbla", def(type_of(sr_name), Int16) thin -> None
    ]()(sr_name, info)


def cublasGetMatrixAsync(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int64,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int64,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMatrixAsync_64",
        def(
            Int64,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            CUstream,
        ) thin -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


def cublasStbsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    x: UnsafePointer[Float32, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStbsv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


def cublasGetSmCountTarget(
    handle: cublasHandle_t,
    sm_count_target: UnsafePointer[Int16, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetSmCountTarget",
        def(type_of(handle), type_of(sm_count_target)) thin -> Result,
    ]()(handle, sm_count_target)


def cublasSetMathMode(
    handle: cublasHandle_t, mode: cublasMath_t
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMathMode",
        def(type_of(handle), cublasMath_t) thin -> Result,
    ]()(handle, mode)


def cublasDsbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsbmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


def cublasSdot(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    y: UnsafePointer[mut=False, Float32, _],
    incy: Int16,
    result: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSdot_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, result)


def cublasSsbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsbmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


def cublasIsamax(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    result: UnsafePointer[Int64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIsamax_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasSdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: Int64,
    n: Int64,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    _c: UnsafePointer[Float32, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSdgmm_64",
        def(
            type_of(handle),
            cublasSideMode_t,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


def cublasSwapEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSwapEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(y),
            DataType,
            Int64,
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


def cublasDotcEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[ImmutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasDotcEx",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(y),
            DataType,
            Int16,
            type_of(result),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    )


def cublasRotEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    c: OpaquePointer[ImmutAnyOrigin],
    s: OpaquePointer[ImmutAnyOrigin],
    cs_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotEx",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(y),
            DataType,
            Int16,
            type_of(c),
            type_of(s),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        c,
        s,
        cs_type,
        executiontype,
    )


def cublasSsymv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsymv_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


def cublasSsyr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    y: UnsafePointer[mut=False, Float32, _],
    incy: Int16,
    _a: UnsafePointer[Float32, _],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr2_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(_a),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


def cublasGetStream(
    handle: cublasHandle_t,
    stream_id: UnsafePointer[CUstream, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetStream_v2",
        def(type_of(handle), type_of(stream_id)) thin -> Result,
    ]()(handle, stream_id)


def cublasIsamin(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    result: UnsafePointer[Int16, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIsamin_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasStbsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    x: UnsafePointer[Float32, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStbsv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


def cublasSetMatrixAsync(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int16,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int16,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMatrixAsync",
        def(
            Int16,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            CUstream,
        ) thin -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


def cublasSaxpy(
    handle: cublasHandle_t,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    y: UnsafePointer[Float32, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSaxpy_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


def cublasDgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int16,
    _c: UnsafePointer[Float64, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgeam",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(beta),
            type_of(_b),
            Int16,
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


def cublasCopyEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasCopyEx",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(y),
            DataType,
            Int16,
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


def cublasGetCudartVersion() raises -> Int:
    return _get_dylib_function["cublasGetCudartVersion", def() thin -> Int]()()


def cublasIdamax(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    result: UnsafePointer[Int64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIdamax_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasSsyr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    y: UnsafePointer[mut=False, Float32, _],
    incy: Int64,
    _a: UnsafePointer[Float32, _],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr2_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(_a),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


def cublasDaxpy(
    handle: cublasHandle_t,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    y: UnsafePointer[Float64, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDaxpy_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


def cublasDsyr2k(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr2k_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasSetLoggerCallback(
    user_callback: def(UnsafePointer[Int8, ImmutAnyOrigin]) thin -> None,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetLoggerCallback",
        def(type_of(user_callback)) thin -> Result,
    ]()(user_callback)


def cublasSgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int16,
    _c: UnsafePointer[Float32, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgeam",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(beta),
            type_of(_b),
            Int16,
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


def cublasDtpttr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    _ap: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[Float64, _],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpttr",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(_ap),
            type_of(_a),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, _ap, _a, lda)


def cublasIamaxEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    result: UnsafePointer[Int16, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIamaxEx",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, result)


def cublasSspmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _ap: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSspmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(_ap),
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


def cublasSsymv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsymv_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


def cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    stride_a: Int64,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int64,
    stride_b: Int64,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
    stride_c: Int64,
    batch_count: Int64,
    compute_type: ComputeType,
    algo: Algorithm,
) raises -> Result:
    return _get_dylib_function[
        "cublasGemmStridedBatchedEx_64",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int64,
            Int64,
            type_of(_b),
            DataType,
            Int64,
            Int64,
            type_of(beta),
            type_of(_c),
            DataType,
            Int64,
            Int64,
            Int64,
            ComputeType,
            Algorithm,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        stride_a,
        _b,
        _btype,
        ldb,
        stride_b,
        beta,
        _c,
        _ctype,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
    )


def cublasNrm2Ex(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasNrm2Ex_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(result),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, execution_type)


def cublasGetPointerMode(
    handle: cublasHandle_t,
    mode: UnsafePointer[cublasPointerMode_t, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetPointerMode_v2",
        def(
            type_of(handle),
            type_of(mode),
        ) thin -> Result,
    ]()(handle, mode)


def cublasSrotm(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, _],
    incx: Int64,
    y: UnsafePointer[Float32, _],
    incy: Int64,
    param: UnsafePointer[mut=False, Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrotm_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(param),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, param)


@fieldwise_init
struct Algorithm(TrivialRegisterPassable, Writable):
    var _value: Int32

    # According to https://docs.nvidia.com/cuda/cublas/#cublasgemmalgo-t, the
    # only useful algorithm options are default and algo0 - algo23.
    # We never specify 0-23 in practice.

    comptime DEFAULT = Self(-1)
    comptime ALGO0 = Self(0)
    comptime ALGO1 = Self(1)
    comptime ALGO2 = Self(2)
    comptime ALGO3 = Self(3)
    comptime ALGO4 = Self(4)
    comptime ALGO5 = Self(5)
    comptime ALGO6 = Self(6)
    comptime ALGO7 = Self(7)
    comptime ALGO8 = Self(8)
    comptime ALGO9 = Self(9)
    comptime ALGO10 = Self(10)
    comptime ALGO11 = Self(11)
    comptime ALGO12 = Self(12)
    comptime ALGO13 = Self(13)
    comptime ALGO14 = Self(14)
    comptime ALGO15 = Self(15)
    comptime ALGO16 = Self(16)
    comptime ALGO17 = Self(17)
    comptime ALGO18 = Self(18)
    comptime ALGO19 = Self(19)
    comptime ALGO20 = Self(20)
    comptime ALGO21 = Self(21)
    comptime ALGO22 = Self(22)
    comptime ALGO23 = Self(23)
    comptime DEFAULT_TENSOR_OP = Self(99)
    comptime ALGO0_TENSOR_OP = Self(100)
    comptime ALGO1_TENSOR_OP = Self(101)
    comptime ALGO2_TENSOR_OP = Self(102)
    comptime ALGO3_TENSOR_OP = Self(103)
    comptime ALGO4_TENSOR_OP = Self(104)
    comptime ALGO5_TENSOR_OP = Self(105)
    comptime ALGO6_TENSOR_OP = Self(106)
    comptime ALGO7_TENSOR_OP = Self(107)
    comptime ALGO8_TENSOR_OP = Self(108)
    comptime ALGO9_TENSOR_OP = Self(109)
    comptime ALGO10_TENSOR_OP = Self(110)
    comptime ALGO11_TENSOR_OP = Self(111)
    comptime ALGO12_TENSOR_OP = Self(112)
    comptime ALGO13_TENSOR_OP = Self(113)
    comptime ALGO14_TENSOR_OP = Self(114)
    comptime ALGO15_TENSOR_OP = Self(115)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.DEFAULT:
            return writer.write_string("DEFAULT")
        if self == Self.ALGO0:
            return writer.write_string("ALGO0")
        if self == Self.ALGO1:
            return writer.write_string("ALGO1")
        if self == Self.ALGO2:
            return writer.write_string("ALGO2")
        if self == Self.ALGO3:
            return writer.write_string("ALGO3")
        if self == Self.ALGO4:
            return writer.write_string("ALGO4")
        if self == Self.ALGO5:
            return writer.write_string("ALGO5")
        if self == Self.ALGO6:
            return writer.write_string("ALGO6")
        if self == Self.ALGO7:
            return writer.write_string("ALGO7")
        if self == Self.ALGO8:
            return writer.write_string("ALGO8")
        if self == Self.ALGO9:
            return writer.write_string("ALGO9")
        if self == Self.ALGO10:
            return writer.write_string("ALGO10")
        if self == Self.ALGO11:
            return writer.write_string("ALGO11")
        if self == Self.ALGO12:
            return writer.write_string("ALGO12")
        if self == Self.ALGO13:
            return writer.write_string("ALGO13")
        if self == Self.ALGO14:
            return writer.write_string("ALGO14")
        if self == Self.ALGO15:
            return writer.write_string("ALGO15")
        if self == Self.ALGO16:
            return writer.write_string("ALGO16")
        if self == Self.ALGO17:
            return writer.write_string("ALGO17")
        if self == Self.ALGO18:
            return writer.write_string("ALGO18")
        if self == Self.ALGO19:
            return writer.write_string("ALGO19")
        if self == Self.ALGO20:
            return writer.write_string("ALGO20")
        if self == Self.ALGO21:
            return writer.write_string("ALGO21")
        if self == Self.ALGO22:
            return writer.write_string("ALGO22")
        if self == Self.ALGO23:
            return writer.write_string("ALGO23")
        if self == Self.DEFAULT_TENSOR_OP:
            return writer.write_string("DEFAULT_TENSOR_OP")
        if self == Self.ALGO0_TENSOR_OP:
            return writer.write_string("ALGO0_TENSOR_OP")
        if self == Self.ALGO1_TENSOR_OP:
            return writer.write_string("ALGO1_TENSOR_OP")
        if self == Self.ALGO2_TENSOR_OP:
            return writer.write_string("ALGO2_TENSOR_OP")
        if self == Self.ALGO3_TENSOR_OP:
            return writer.write_string("ALGO3_TENSOR_OP")
        if self == Self.ALGO4_TENSOR_OP:
            return writer.write_string("ALGO4_TENSOR_OP")
        if self == Self.ALGO5_TENSOR_OP:
            return writer.write_string("ALGO5_TENSOR_OP")
        if self == Self.ALGO6_TENSOR_OP:
            return writer.write_string("ALGO6_TENSOR_OP")
        if self == Self.ALGO7_TENSOR_OP:
            return writer.write_string("ALGO7_TENSOR_OP")
        if self == Self.ALGO8_TENSOR_OP:
            return writer.write_string("ALGO8_TENSOR_OP")
        if self == Self.ALGO9_TENSOR_OP:
            return writer.write_string("ALGO9_TENSOR_OP")
        if self == Self.ALGO10_TENSOR_OP:
            return writer.write_string("ALGO10_TENSOR_OP")
        if self == Self.ALGO11_TENSOR_OP:
            return writer.write_string("ALGO11_TENSOR_OP")
        if self == Self.ALGO12_TENSOR_OP:
            return writer.write_string("ALGO12_TENSOR_OP")
        if self == Self.ALGO13_TENSOR_OP:
            return writer.write_string("ALGO13_TENSOR_OP")
        if self == Self.ALGO14_TENSOR_OP:
            return writer.write_string("ALGO14_TENSOR_OP")
        if self == Self.ALGO15_TENSOR_OP:
            return writer.write_string("ALGO15_TENSOR_OP")
        abort("invalid Algorithm entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasSsyrk(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyrk_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


def cublasDsyr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    _a: UnsafePointer[Float64, _],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(_a),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


def cublasStrmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    x: UnsafePointer[Float32, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


def cublasDcopy(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    y: UnsafePointer[Float64, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDcopy_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy)


def cublasDtrmm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int64,
    _c: UnsafePointer[Float64, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrmm_v2_64",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


def cublasDdot(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    y: UnsafePointer[mut=False, Float64, _],
    incy: Int16,
    result: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDdot_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, result)


def cublasSscal(
    handle: cublasHandle_t,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[Float32, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSscal_v2",
        def(
            type_of(handle),
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, n, alpha, x, incx)


def cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    stride_a: Int64,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int64,
    stride_b: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int64,
    stride_c: Int64,
    batch_count: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemmStridedBatched_64",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            Int64,
            type_of(_b),
            Int64,
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
            Int64,
            Int64,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        lda,
        stride_a,
        _b,
        ldb,
        stride_b,
        beta,
        _c,
        ldc,
        stride_c,
        batch_count,
    )


def cublasDdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: Int64,
    n: Int64,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    _c: UnsafePointer[Float64, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDdgmm_64",
        def(
            type_of(handle),
            cublasSideMode_t,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


def cublasStpttr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    _ap: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[Float32, _],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpttr",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(_ap),
            type_of(_a),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, _ap, _a, lda)


def cublasDsyr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    _a: UnsafePointer[Float64, _],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(_a),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


def cublasSetVector(
    n: Int16,
    elem_size: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    incx: Int16,
    device_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetVector",
        def(
            Int16,
            Int16,
            type_of(x),
            Int16,
            type_of(device_ptr),
            Int16,
        ) thin -> Result,
    ]()(n, elem_size, x, incx, device_ptr, incy)


def cublasSetMatrixAsync(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int64,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int64,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMatrixAsync_64",
        def(
            Int64,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            CUstream,
        ) thin -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb, stream)


# def cublasGetLoggerCallback(user_callback: UNKNOWN) raises -> Result:
#     return _get_dylib_function[
#         "cublasGetLoggerCallback", def (UNKNOWN) -> Result
#     ]()(user_callback)


def cublasSasum(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    result: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSasum_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasRotgEx(
    handle: cublasHandle_t,
    a: OpaquePointer[MutAnyOrigin],
    b: OpaquePointer[MutAnyOrigin],
    ab_type: DataType,
    c: OpaquePointer[MutAnyOrigin],
    s: OpaquePointer[MutAnyOrigin],
    cs_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotgEx",
        def(
            type_of(handle),
            type_of(a),
            type_of(b),
            DataType,
            type_of(c),
            type_of(s),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(handle, a, b, ab_type, c, s, cs_type, executiontype)


@fieldwise_init
struct cublasDiagType_t(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUBLAS_DIAG_NON_UNIT = cublasDiagType_t(0)
    comptime CUBLAS_DIAG_UNIT = cublasDiagType_t(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.CUBLAS_DIAG_NON_UNIT:
            return writer.write_string("CUBLAS_DIAG_NON_UNIT")
        if self == Self.CUBLAS_DIAG_UNIT:
            return writer.write_string("CUBLAS_DIAG_UNIT")
        abort("invalid cublasDiagType_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct ComputeType(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime COMPUTE_16F = Self(64)
    comptime COMPUTE_16F_PEDANTIC = Self(65)
    comptime COMPUTE_32F = Self(68)
    comptime COMPUTE_32F_PEDANTIC = Self(69)
    comptime COMPUTE_32F_FAST_16F = Self(74)
    comptime COMPUTE_32F_FAST_16BF = Self(75)
    comptime COMPUTE_32F_FAST_TF32 = Self(77)
    comptime COMPUTE_64F = Self(70)
    comptime COMPUTE_64F_PEDANTIC = Self(71)
    comptime COMPUTE_32I = Self(72)
    comptime COMPUTE_32I_PEDANTIC = Self(73)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.COMPUTE_16F:
            return writer.write_string("COMPUTE_16F")
        if self == Self.COMPUTE_16F_PEDANTIC:
            return writer.write_string("COMPUTE_16F_PEDANTIC")
        if self == Self.COMPUTE_32F:
            return writer.write_string("COMPUTE_32F")
        if self == Self.COMPUTE_32F_PEDANTIC:
            return writer.write_string("COMPUTE_32F_PEDANTIC")
        if self == Self.COMPUTE_32F_FAST_16F:
            return writer.write_string("COMPUTE_32F_FAST_16F")
        if self == Self.COMPUTE_32F_FAST_16BF:
            return writer.write_string("COMPUTE_32F_FAST_16BF")
        if self == Self.COMPUTE_32F_FAST_TF32:
            return writer.write_string("COMPUTE_32F_FAST_TF32")
        if self == Self.COMPUTE_64F:
            return writer.write_string("COMPUTE_64F")
        if self == Self.COMPUTE_64F_PEDANTIC:
            return writer.write_string("COMPUTE_64F_PEDANTIC")
        if self == Self.COMPUTE_32I:
            return writer.write_string("COMPUTE_32I")
        if self == Self.COMPUTE_32I_PEDANTIC:
            return writer.write_string("COMPUTE_32I_PEDANTIC")
        abort("invalid ComputeType entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasDsymm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsymm_v2_64",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasSspr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    _ap: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSspr_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


def cublasIdamin(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    result: UnsafePointer[Int64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIdamin_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasGetVectorAsync(
    n: Int16,
    elem_size: Int16,
    device_ptr: OpaquePointer[ImmutAnyOrigin],
    incx: Int16,
    host_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int16,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVectorAsync",
        def(
            Int16,
            Int16,
            type_of(device_ptr),
            Int16,
            type_of(host_ptr),
            Int16,
            CUstream,
        ) thin -> Result,
    ]()(n, elem_size, device_ptr, incx, host_ptr, incy, stream)


def cublasGetMatrix(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int64,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMatrix_64",
        def(
            Int64,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
        ) thin -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


def cublasDaxpy(
    handle: cublasHandle_t,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    y: UnsafePointer[Float64, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDaxpy_v2",
        def(
            type_of(handle),
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


def cublasDsyr2k(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr2k_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasSger(
    handle: cublasHandle_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    y: UnsafePointer[mut=False, Float32, _],
    incy: Int64,
    _a: UnsafePointer[Float32, _],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSger_v2_64",
        def(
            type_of(handle),
            Int64,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(_a),
            Int64,
        ) thin -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


def cublasSdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: Int16,
    n: Int16,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    _c: UnsafePointer[Float32, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSdgmm",
        def(
            type_of(handle),
            cublasSideMode_t,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


def cublasDtbsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    x: UnsafePointer[Float64, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtbsv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


def cublasDtrsm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    _b: UnsafePointer[Float64, _],
    ldb: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrsm_v2",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
        ) thin -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


def cublasStbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    x: UnsafePointer[Float32, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStbmv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


def cublasDspmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _ap: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDspmv_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(_ap),
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


def cublasSswap(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, _],
    incx: Int64,
    y: UnsafePointer[Float32, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSswap_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy)


def cublasDspmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _ap: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDspmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(_ap),
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


def cublasSrotmg(
    handle: cublasHandle_t,
    d1: UnsafePointer[Float32, _],
    d2: UnsafePointer[Float32, _],
    x1: UnsafePointer[Float32, _],
    y1: UnsafePointer[Float32, _],
    param: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrotmg_v2",
        def(
            type_of(handle),
            type_of(d1),
            type_of(d2),
            type_of(x1),
            type_of(y1),
            type_of(param),
        ) thin -> Result,
    ]()(handle, d1, d2, x1, y1, param)


def cublasDtpmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[Float64, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpmv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            type_of(_ap),
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


def cublasDasum(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    result: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDasum_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasRotEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    c: OpaquePointer[ImmutAnyOrigin],
    s: OpaquePointer[ImmutAnyOrigin],
    cs_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(y),
            DataType,
            Int64,
            type_of(c),
            type_of(s),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        c,
        s,
        cs_type,
        executiontype,
    )


def cublasDrotm(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, _],
    incx: Int16,
    y: UnsafePointer[Float64, _],
    incy: Int16,
    param: UnsafePointer[mut=False, Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrotm_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(param),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, param)


def cublasAxpyEx(
    handle: cublasHandle_t,
    n: Int16,
    alpha: OpaquePointer[ImmutAnyOrigin],
    alpha_type: DataType,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasAxpyEx",
        def(
            type_of(handle),
            Int16,
            type_of(alpha),
            DataType,
            type_of(x),
            DataType,
            Int16,
            type_of(y),
            DataType,
            Int16,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        alpha,
        alpha_type,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        executiontype,
    )


def cublasSgemm(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemm_v2",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasSsymm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsymm_v2_64",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasCopyEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasCopyEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(y),
            DataType,
            Int64,
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


def cublasSwapEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSwapEx",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(y),
            DataType,
            Int16,
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, y, y_type, incy)


def cublasSrot(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float32, _],
    incx: Int64,
    y: UnsafePointer[Float32, _],
    incy: Int64,
    c: UnsafePointer[mut=False, Float32, _],
    s: UnsafePointer[mut=False, Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrot_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(c),
            type_of(s),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


def cublasGetVector(
    n: Int16,
    elem_size: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    incx: Int16,
    y: OpaquePointer[MutAnyOrigin],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVector",
        def(
            Int16,
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(n, elem_size, x, incx, y, incy)


def cublasDtrsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    x: UnsafePointer[Float64, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrsv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


def cublasSsymm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsymm_v2",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasDtrmm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int16,
    _c: UnsafePointer[Float64, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrmm_v2",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


def cublasCherk3mEx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasCherk3mEx_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int64,
            type_of(beta),
            type_of(_c),
            DataType,
            Int64,
        ) thin -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


comptime cublasLogCallback = def(
    UnsafePointer[Int8, ImmutAnyOrigin]
) thin -> None


def cublasDtrmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    x: UnsafePointer[Float64, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrmv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


def cublasDdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: Int16,
    n: Int16,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    _c: UnsafePointer[Float64, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDdgmm",
        def(
            type_of(handle),
            cublasSideMode_t,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, mode, m, n, _a, lda, x, incx, _c, ldc)


def cublasDtbsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    x: UnsafePointer[Float64, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtbsv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


def cublasSsyr2k(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr2k_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasDgemm(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemm_v2",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasGetMathMode(
    handle: cublasHandle_t,
    mode: UnsafePointer[cublasMath_t, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetMathMode",
        def(
            type_of(handle),
            type_of(mode),
        ) thin -> Result,
    ]()(handle, mode)


def cublasDrot(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, _],
    incx: Int64,
    y: UnsafePointer[Float64, _],
    incy: Int64,
    c: UnsafePointer[mut=False, Float64, _],
    s: UnsafePointer[mut=False, Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrot_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(c),
            type_of(s),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


def cublasSspr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    _ap: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSspr_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


def cublasGemmEx64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int64,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OptionalPointer[NoneType, MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
    compute_type: ComputeType,
    algo: Algorithm,
) raises -> Result:
    return _get_dylib_function[
        "cublasGemmEx_64",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int64,
            type_of(_b),
            DataType,
            Int64,
            type_of(beta),
            type_of(_c),
            DataType,
            Int64,
            ComputeType,
            Algorithm,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        _b,
        _btype,
        ldb,
        beta,
        _c,
        _ctype,
        ldc,
        compute_type,
        algo,
    )


def cublasDotEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    y: OpaquePointer[ImmutAnyOrigin],
    y_type: DataType,
    incy: Int16,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasDotEx",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(y),
            DataType,
            Int16,
            type_of(result),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    )


def cublasSswap(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, _],
    incx: Int16,
    y: UnsafePointer[Float32, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSswap_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy)


def cublasDrotm(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, _],
    incx: Int64,
    y: UnsafePointer[Float64, _],
    incy: Int64,
    param: UnsafePointer[mut=False, Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrotm_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(param),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, param)


def cublasSgemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemmEx_64",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int64,
            type_of(_b),
            DataType,
            Int64,
            type_of(beta),
            type_of(_c),
            DataType,
            Int64,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        _b,
        _btype,
        ldb,
        beta,
        _c,
        _ctype,
        ldc,
    )


def cublasDgemm(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemm_v2_64",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, transa, transb, m, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasSsyrk(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyrk_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


def cublasDnrm2(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    result: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDnrm2_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasDasum(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    result: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDasum_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasDsyrkx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyrkx",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasRotmEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    param: OpaquePointer[ImmutAnyOrigin],
    param_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasRotmEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(y),
            DataType,
            Int64,
            type_of(param),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        param,
        param_type,
        executiontype,
    )


def cublasDtpsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[Float64, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpsv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            type_of(_ap),
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


def cublasSspr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    y: UnsafePointer[mut=False, Float32, _],
    incy: Int16,
    _ap: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSspr2_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


def cublasSetMatrix(
    rows: Int64,
    cols: Int64,
    elem_size: Int64,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int64,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMatrix_64",
        def(
            Int64,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
        ) thin -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


def cublasDrotg(
    handle: cublasHandle_t,
    a: UnsafePointer[Float64, _],
    b: UnsafePointer[Float64, _],
    c: UnsafePointer[Float64, _],
    s: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrotg_v2",
        def(
            type_of(handle),
            type_of(a),
            type_of(b),
            type_of(c),
            type_of(s),
        ) thin -> Result,
    ]()(handle, a, b, c, s)


def cublasGetAtomicsMode(
    handle: cublasHandle_t,
    mode: UnsafePointer[cublasAtomicsMode_t, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetAtomicsMode",
        def(
            type_of(handle),
            type_of(mode),
        ) thin -> Result,
    ]()(handle, mode)


def cublasStbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    x: UnsafePointer[Float32, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStbmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


def cublasAxpyEx(
    handle: cublasHandle_t,
    n: Int64,
    alpha: OpaquePointer[ImmutAnyOrigin],
    alpha_type: DataType,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[MutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasAxpyEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(alpha),
            DataType,
            type_of(x),
            DataType,
            Int64,
            type_of(y),
            DataType,
            Int64,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        alpha,
        alpha_type,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        executiontype,
    )


def cublasIaminEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    result: UnsafePointer[Int64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIaminEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, result)


def cublasDspr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    y: UnsafePointer[mut=False, Float64, _],
    incy: Int16,
    _ap: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDspr2_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


def cublasDotEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[ImmutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasDotEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(y),
            DataType,
            Int64,
            type_of(result),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    )


def cublasScopy(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    y: UnsafePointer[Float32, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasScopy_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy)


def cublasDsyrk(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyrk_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


def cublasDestroy(handle: cublasHandle_t) raises -> Result:
    return _get_dylib_function[
        "cublasDestroy_v2",
        def(type_of(handle)) thin -> Result,
    ]()(handle)


def cublasSetVectorAsync(
    n: Int16,
    elem_size: Int16,
    host_ptr: OpaquePointer[ImmutAnyOrigin],
    incx: Int16,
    device_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int16,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetVectorAsync",
        def(
            Int16,
            Int16,
            type_of(host_ptr),
            Int16,
            type_of(device_ptr),
            Int16,
            CUstream,
        ) thin -> Result,
    ]()(n, elem_size, host_ptr, incx, device_ptr, incy, stream)


def cublasIamaxEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    result: UnsafePointer[Int64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIamaxEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, result)


def cublasSsyrkx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyrkx_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasDswap(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[Float64, _],
    incx: Int64,
    y: UnsafePointer[Float64, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDswap_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy)


def cublasAsumEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasAsumEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(result),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, executiontype)


@fieldwise_init
struct FillMode(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime LOWER = Self(0)
    comptime UPPER = Self(1)
    comptime FULL = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        if self == Self.LOWER:
            return writer.write_string("LOWER")
        if self == Self.UPPER:
            return writer.write_string("UPPER")
        if self == Self.FULL:
            return writer.write_string("FULL")
        abort("invalid FillMode entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasSspr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    y: UnsafePointer[mut=False, Float32, _],
    incy: Int64,
    _ap: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSspr2_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


def cublasSgbmv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    kl: Int64,
    ku: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgbmv_v2_64",
        def(
            type_of(handle),
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


def cublasAsumEx(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    executiontype: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasAsumEx",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(result),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, executiontype)


def cublasGetVersion(
    handle: cublasHandle_t,
    version: UnsafePointer[Int16, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVersion_v2",
        def(type_of(handle), type_of(version)) thin -> Result,
    ]()(handle, version)


def cublasScalEx(
    handle: cublasHandle_t,
    n: Int64,
    alpha: OpaquePointer[ImmutAnyOrigin],
    alpha_type: DataType,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasScalEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(alpha),
            DataType,
            type_of(x),
            DataType,
            Int64,
            DataType,
        ) thin -> Result,
    ]()(handle, n, alpha, alpha_type, x, x_type, incx, execution_type)


def cublasSetPointerMode(
    handle: cublasHandle_t,
    mode: cublasPointerMode_t,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetPointerMode_v2",
        def(type_of(handle), cublasPointerMode_t) thin -> Result,
    ]()(handle, mode)


def cublasDgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemv_v2_64",
        def(
            type_of(handle),
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


def cublasGetStatusString(
    status: Result,
) raises -> UnsafePointer[Int8, ImmutAnyOrigin]:
    return _get_dylib_function[
        "cublasGetStatusString",
        def(Result) thin -> UnsafePointer[Int8, ImmutAnyOrigin],
    ]()(status)


def cublasSnrm2(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    result: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSnrm2_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasDgbmv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    kl: Int64,
    ku: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgbmv_v2_64",
        def(
            type_of(handle),
            cublasOperation_t,
            Int64,
            Int64,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


def cublasDsyr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    y: UnsafePointer[mut=False, Float64, _],
    incy: Int16,
    _a: UnsafePointer[Float64, _],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr2_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(_a),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


def cublasDtpsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[Float64, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpsv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            type_of(_ap),
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


def cublasSetVector(
    n: Int64,
    elem_size: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    incx: Int64,
    device_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetVector_64",
        def(
            Int64,
            Int64,
            type_of(x),
            Int64,
            type_of(device_ptr),
            Int64,
        ) thin -> Result,
    ]()(n, elem_size, x, incx, device_ptr, incy)


def cublasDgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    stride_a: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    stridex: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int64,
    stridey: Int64,
    batch_count: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemvStridedBatched_64",
        def(
            type_of(handle),
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            Int64,
            type_of(x),
            Int64,
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
            Int64,
            Int64,
        ) thin -> Result,
    ]()(
        handle,
        trans,
        m,
        n,
        alpha,
        _a,
        lda,
        stride_a,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batch_count,
    )


def cublasSsyrkx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyrkx",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasGetStatusName(
    status: Result,
) raises -> UnsafePointer[Int8, ImmutAnyOrigin]:
    return _get_dylib_function[
        "cublasGetStatusName",
        def(Result) thin -> UnsafePointer[Int8, ImmutAnyOrigin],
    ]()(status)


def cublasDtbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    k: Int64,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    x: UnsafePointer[Float64, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtbmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


def cublasSrotg(
    handle: cublasHandle_t,
    a: UnsafePointer[Float32, _],
    b: UnsafePointer[Float32, _],
    c: UnsafePointer[Float32, _],
    s: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrotg_v2",
        def(
            type_of(handle),
            type_of(a),
            type_of(b),
            type_of(c),
            type_of(s),
        ) thin -> Result,
    ]()(handle, a, b, c, s)


def cublasCherkEx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasCherkEx",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int16,
            type_of(beta),
            type_of(_c),
            DataType,
            Int16,
        ) thin -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


def cublasDrotmg(
    handle: cublasHandle_t,
    d1: UnsafePointer[Float64, _],
    d2: UnsafePointer[Float64, _],
    x1: UnsafePointer[Float64, _],
    y1: UnsafePointer[Float64, _],
    param: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrotmg_v2",
        def(
            type_of(handle),
            type_of(d1),
            type_of(d2),
            type_of(x1),
            type_of(y1),
            type_of(param),
        ) thin -> Result,
    ]()(handle, d1, d2, x1, y1, param)


def cublasDger(
    handle: cublasHandle_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    y: UnsafePointer[mut=False, Float64, _],
    incy: Int16,
    _a: UnsafePointer[Float64, _],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDger_v2",
        def(
            type_of(handle),
            Int16,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(_a),
            Int16,
        ) thin -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


def cublasSscal(
    handle: cublasHandle_t,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[Float32, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSscal_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, n, alpha, x, incx)


def cublasSetWorkspace(
    handle: cublasHandle_t,
    workspace: OpaquePointer[MutAnyOrigin],
    workspace_size_in_bytes: Int,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetWorkspace_v2",
        def(type_of(handle), type_of(workspace), Int) thin -> Result,
    ]()(handle, workspace, workspace_size_in_bytes)


def cublasStpsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[Float32, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpsv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            type_of(_ap),
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


def cublasDspr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    _ap: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDspr_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


def cublasGemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int32,
    n: Int32,
    k: Int32,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int32,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int32,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OptionalPointer[NoneType, MutAnyOrigin],
    _ctype: DataType,
    ldc: Int32,
    compute_type: ComputeType,
    algo: Algorithm,
) raises -> Result:
    return _get_dylib_function[
        "cublasGemmEx",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int32,
            Int32,
            Int32,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int32,
            type_of(_b),
            DataType,
            Int32,
            type_of(beta),
            type_of(_c),
            DataType,
            Int32,
            ComputeType,
            Algorithm,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        _b,
        _btype,
        ldb,
        beta,
        _c,
        _ctype,
        ldc,
        compute_type,
        algo,
    )


def cublasSsbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsbmv_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


def cublasDgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    stride_a: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    stridex: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int16,
    stridey: Int64,
    batch_count: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgemvStridedBatched",
        def(
            type_of(handle),
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            Int64,
            type_of(x),
            Int16,
            Int64,
            type_of(beta),
            type_of(y),
            Int16,
            Int64,
            Int16,
        ) thin -> Result,
    ]()(
        handle,
        trans,
        m,
        n,
        alpha,
        _a,
        lda,
        stride_a,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batch_count,
    )


def cublasDsymv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsymv_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


def cublasLoggerConfigure(
    log_is_on: Int16,
    log_to_std_out: Int16,
    log_to_std_err: Int16,
    log_file_name: OptionalPointer[Int8, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasLoggerConfigure",
        def(Int16, Int16, Int16, type_of(log_file_name)) thin -> Result,
    ]()(log_is_on, log_to_std_out, log_to_std_err, log_file_name)


def cublasStpmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[Float32, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            type_of(_ap),
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


def cublasSgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    stride_a: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    stridex: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int64,
    stridey: Int64,
    batch_count: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemvStridedBatched_64",
        def(
            type_of(handle),
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            Int64,
            type_of(x),
            Int64,
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
            Int64,
            Int64,
        ) thin -> Result,
    ]()(
        handle,
        trans,
        m,
        n,
        alpha,
        _a,
        lda,
        stride_a,
        x,
        incx,
        stridex,
        beta,
        y,
        incy,
        stridey,
        batch_count,
    )


def cublasIsamin(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    result: UnsafePointer[Int64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIsamin_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasDrot(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, _],
    incx: Int16,
    y: UnsafePointer[Float64, _],
    incy: Int16,
    c: UnsafePointer[mut=False, Float64, _],
    s: UnsafePointer[mut=False, Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDrot_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(c),
            type_of(s),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


def cublasDgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int64,
    _c: UnsafePointer[Float64, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDgeam_64",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(beta),
            type_of(_b),
            Int64,
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, transa, transb, m, n, alpha, _a, lda, beta, _b, ldb, _c, ldc)


def cublasGetVectorAsync(
    n: Int64,
    elem_size: Int64,
    device_ptr: OpaquePointer[ImmutAnyOrigin],
    incx: Int64,
    host_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int64,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasGetVectorAsync_64",
        def(
            Int64,
            Int64,
            type_of(device_ptr),
            Int64,
            type_of(host_ptr),
            Int64,
            CUstream,
        ) thin -> Result,
    ]()(n, elem_size, device_ptr, incx, host_ptr, incy, stream)


def cublasStrsm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    _b: UnsafePointer[Float32, _],
    ldb: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrsm_v2_64",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
        ) thin -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


def cublasSgemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int16,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemmEx",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int16,
            type_of(_b),
            DataType,
            Int16,
            type_of(beta),
            type_of(_c),
            DataType,
            Int16,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        _b,
        _btype,
        ldb,
        beta,
        _c,
        _ctype,
        ldc,
    )


def cublasStpmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    _ap: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[Float32, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStpmv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            type_of(_ap),
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


def cublasDtrmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    x: UnsafePointer[Float64, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


def cublasDtrsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    x: UnsafePointer[Float64, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrsv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


def cublasDsyr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    y: UnsafePointer[mut=False, Float64, _],
    incy: Int64,
    _a: UnsafePointer[Float64, _],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyr2_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(_a),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _a, lda)


def cublasSrot(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, _],
    incx: Int16,
    y: UnsafePointer[Float32, _],
    incy: Int16,
    c: UnsafePointer[mut=False, Float32, _],
    s: UnsafePointer[mut=False, Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrot_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(c),
            type_of(s),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, c, s)


def cublasDscal(
    handle: cublasHandle_t,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[Float64, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDscal_v2",
        def(
            type_of(handle),
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, n, alpha, x, incx)


def cublasCreate(
    handle: UnsafePointer[cublasHandle_t[MutAnyOrigin], _],
) raises -> Result:
    return _get_dylib_function[
        "cublasCreate_v2",
        def(type_of(handle)) thin -> Result,
    ]()(handle)


def cublasSetSmCountTarget(
    handle: cublasHandle_t, sm_count_target: Int16
) raises -> Result:
    return _get_dylib_function[
        "cublasSetSmCountTarget",
        def(type_of(handle), Int16) thin -> Result,
    ]()(handle, sm_count_target)


def cublasDswap(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float64, _],
    incx: Int16,
    y: UnsafePointer[Float64, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDswap_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy)


def cublasStrsv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    x: UnsafePointer[Float32, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrsv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _a, lda, x, incx)


def cublasDspr2(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    y: UnsafePointer[mut=False, Float64, _],
    incy: Int64,
    _ap: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDspr2_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, y, incy, _ap)


def cublasSsyr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    _a: UnsafePointer[Float32, _],
    lda: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
            type_of(_a),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _a, lda)


def cublasNrm2Ex(
    handle: cublasHandle_t,
    n: Int16,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasNrm2Ex",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            DataType,
            Int16,
            type_of(result),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(handle, n, x, x_type, incx, result, result_type, execution_type)


def cublasDtbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int16,
    k: Int16,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    x: UnsafePointer[Float64, _],
    incx: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtbmv_v2",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, k, _a, lda, x, incx)


@fieldwise_init
struct cublasAtomicsMode_t(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUBLAS_ATOMICS_NOT_ALLOWED = cublasAtomicsMode_t(0)
    comptime CUBLAS_ATOMICS_ALLOWED = cublasAtomicsMode_t(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.CUBLAS_ATOMICS_NOT_ALLOWED:
            return writer.write_string("CUBLAS_ATOMICS_NOT_ALLOWED")
        if self == Self.CUBLAS_ATOMICS_ALLOWED:
            return writer.write_string("CUBLAS_ATOMICS_ALLOWED")
        abort("invalid cublasAtomicsMode_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasSsyr2k(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasSsyr2k_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasCherk3mEx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasCherk3mEx",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int16,
            type_of(beta),
            type_of(_c),
            DataType,
            Int16,
        ) thin -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


def cublasScalEx(
    handle: cublasHandle_t,
    n: Int16,
    alpha: OpaquePointer[ImmutAnyOrigin],
    alpha_type: DataType,
    x: OpaquePointer[MutAnyOrigin],
    x_type: DataType,
    incx: Int16,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasScalEx",
        def(
            type_of(handle),
            Int16,
            type_of(alpha),
            DataType,
            type_of(x),
            DataType,
            Int16,
            DataType,
        ) thin -> Result,
    ]()(handle, n, alpha, alpha_type, x, x_type, incx, execution_type)


def cublasDotcEx(
    handle: cublasHandle_t,
    n: Int64,
    x: OpaquePointer[ImmutAnyOrigin],
    x_type: DataType,
    incx: Int64,
    y: OpaquePointer[ImmutAnyOrigin],
    y_type: DataType,
    incy: Int64,
    result: OpaquePointer[MutAnyOrigin],
    result_type: DataType,
    execution_type: DataType,
) raises -> Result:
    return _get_dylib_function[
        "cublasDotcEx_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            DataType,
            Int64,
            type_of(y),
            DataType,
            Int64,
            type_of(result),
            DataType,
            DataType,
        ) thin -> Result,
    ]()(
        handle,
        n,
        x,
        x_type,
        incx,
        y,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    )


def cublasDsymm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float64, _],
    ldb: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsymm_v2",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(beta),
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, side, uplo, m, n, alpha, _a, lda, _b, ldb, beta, _c, ldc)


def cublasIsamax(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    result: UnsafePointer[Int16, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIsamax_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasSaxpy(
    handle: cublasHandle_t,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    y: UnsafePointer[Float32, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSaxpy_v2",
        def(
            type_of(handle),
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, n, alpha, x, incx, y, incy)


def cublasSnrm2(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    result: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSnrm2_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasCherkEx(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasCherkEx_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int64,
            type_of(beta),
            type_of(_c),
            DataType,
            Int64,
        ) thin -> Result,
    ]()(
        handle, uplo, trans, n, k, alpha, _a, _atype, lda, beta, _c, _ctype, ldc
    )


@fieldwise_init
struct cublasSideMode_t(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUBLAS_SIDE_LEFT = cublasSideMode_t(0)
    comptime CUBLAS_SIDE_RIGHT = cublasSideMode_t(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.CUBLAS_SIDE_LEFT:
            return writer.write_string("CUBLAS_SIDE_LEFT")
        if self == Self.CUBLAS_SIDE_RIGHT:
            return writer.write_string("CUBLAS_SIDE_RIGHT")
        abort("invalid cublasSideMode_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasSetMatrix(
    rows: Int16,
    cols: Int16,
    elem_size: Int16,
    _a: OpaquePointer[ImmutAnyOrigin],
    lda: Int16,
    _b: OpaquePointer[MutAnyOrigin],
    ldb: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetMatrix",
        def(
            Int16,
            Int16,
            Int16,
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
        ) thin -> Result,
    ]()(rows, cols, elem_size, _a, lda, _b, ldb)


def cublasDtrsm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    _b: UnsafePointer[Float64, _],
    ldb: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrsm_v2_64",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
        ) thin -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb)


def cublasDcopy(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    y: UnsafePointer[Float64, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDcopy_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy)


def cublasSetVectorAsync(
    n: Int64,
    elem_size: Int64,
    host_ptr: OpaquePointer[ImmutAnyOrigin],
    incx: Int64,
    device_ptr: OpaquePointer[MutAnyOrigin],
    incy: Int64,
    stream: CUstream,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetVectorAsync_64",
        def(
            Int64,
            Int64,
            type_of(host_ptr),
            Int64,
            type_of(device_ptr),
            Int64,
            CUstream,
        ) thin -> Result,
    ]()(n, elem_size, host_ptr, incx, device_ptr, incy, stream)


def cublasDspr(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    _ap: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDspr_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, x, incx, _ap)


def cublasSgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemv_v2",
        def(
            type_of(handle),
            cublasOperation_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, trans, m, n, alpha, _a, lda, x, incx, beta, y, incy)


def cublasDtrttp(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    _ap: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDtrttp",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(_a),
            Int16,
            type_of(_ap),
        ) thin -> Result,
    ]()(handle, uplo, n, _a, lda, _ap)


def cublasDdot(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    y: UnsafePointer[mut=False, Float64, _],
    incy: Int64,
    result: UnsafePointer[Float64, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasDdot_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(y),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, result)


def cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: OpaquePointer[ImmutAnyOrigin],
    _a: OpaquePointer[ImmutAnyOrigin],
    _atype: DataType,
    lda: Int16,
    stride_a: Int64,
    _b: OpaquePointer[ImmutAnyOrigin],
    _btype: DataType,
    ldb: Int16,
    stride_b: Int64,
    beta: OpaquePointer[ImmutAnyOrigin],
    _c: OpaquePointer[MutAnyOrigin],
    _ctype: DataType,
    ldc: Int16,
    stride_c: Int64,
    batch_count: Int16,
    compute_type: ComputeType,
    algo: Algorithm,
) raises -> Result:
    return _get_dylib_function[
        "cublasGemmStridedBatchedEx",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            DataType,
            Int16,
            Int64,
            type_of(_b),
            DataType,
            Int16,
            Int64,
            type_of(beta),
            type_of(_c),
            DataType,
            Int16,
            Int64,
            Int16,
            ComputeType,
            Algorithm,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        _atype,
        lda,
        stride_a,
        _b,
        _btype,
        ldb,
        stride_b,
        beta,
        _c,
        _ctype,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
    )


def cublasStrmm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int64,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int64,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int64,
    _c: UnsafePointer[Float32, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrmm_v2_64",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(_b),
            Int64,
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


def cublasDsyrk(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    n: Int64,
    k: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    _c: UnsafePointer[Float64, _],
    ldc: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsyrk_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            Int64,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(beta),
            type_of(_c),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, n, k, alpha, _a, lda, beta, _c, ldc)


def cublasDscal(
    handle: cublasHandle_t,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[Float64, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDscal_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(alpha),
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, n, alpha, x, incx)


def cublasDtpmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: Int64,
    _ap: UnsafePointer[mut=False, Float64, _],
    x: UnsafePointer[Float64, _],
    incx: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDtpmv_v2_64",
        def(
            type_of(handle),
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int64,
            type_of(_ap),
            type_of(x),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, trans, diag, n, _ap, x, incx)


def cublasSgbmv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: Int16,
    n: Int16,
    kl: Int16,
    ku: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgbmv_v2",
        def(
            type_of(handle),
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, trans, m, n, kl, ku, alpha, _a, lda, x, incx, beta, y, incy)


def cublasSrotm(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[Float32, _],
    incx: Int16,
    y: UnsafePointer[Float32, _],
    incy: Int16,
    param: UnsafePointer[mut=False, Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSrotm_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(param),
        ) thin -> Result,
    ]()(handle, n, x, incx, y, incy, param)


def cublasSetAtomicsMode(
    handle: cublasHandle_t,
    mode: cublasAtomicsMode_t,
) raises -> Result:
    return _get_dylib_function[
        "cublasSetAtomicsMode",
        def(type_of(handle), cublasAtomicsMode_t) thin -> Result,
    ]()(handle, mode)


def cublasDsbmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsbmv_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, k, alpha, _a, lda, x, incx, beta, y, incy)


def cublasSger(
    handle: cublasHandle_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    y: UnsafePointer[mut=False, Float32, _],
    incy: Int16,
    _a: UnsafePointer[Float32, _],
    lda: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSger_v2",
        def(
            type_of(handle),
            Int16,
            Int16,
            type_of(alpha),
            type_of(x),
            Int16,
            type_of(y),
            Int16,
            type_of(_a),
            Int16,
        ) thin -> Result,
    ]()(handle, m, n, alpha, x, incx, y, incy, _a, lda)


def cublasDsymv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int64,
    alpha: UnsafePointer[mut=False, Float64, _],
    _a: UnsafePointer[mut=False, Float64, _],
    lda: Int64,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int64,
    beta: UnsafePointer[mut=False, Float64, _],
    y: UnsafePointer[Float64, _],
    incy: Int64,
) raises -> Result:
    return _get_dylib_function[
        "cublasDsymv_v2_64",
        def(
            type_of(handle),
            FillMode,
            Int64,
            type_of(alpha),
            type_of(_a),
            Int64,
            type_of(x),
            Int64,
            type_of(beta),
            type_of(y),
            Int64,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, _a, lda, x, incx, beta, y, incy)


def cublasSetStream(
    handle: cublasHandle_t, stream_id: CUstream
) raises -> Result:
    return _get_dylib_function[
        "cublasSetStream_v2",
        def(type_of(handle), CUstream) thin -> Result,
    ]()(handle, stream_id)


def cublasStrmm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: FillMode,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: Int16,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int16,
    _c: UnsafePointer[Float32, _],
    ldc: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasStrmm_v2",
        def(
            type_of(handle),
            cublasSideMode_t,
            FillMode,
            cublasOperation_t,
            cublasDiagType_t,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            type_of(_b),
            Int16,
            type_of(_c),
            Int16,
        ) thin -> Result,
    ]()(handle, side, uplo, trans, diag, m, n, alpha, _a, lda, _b, ldb, _c, ldc)


@fieldwise_init
struct cublasOperation_t(TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime CUBLAS_OP_N = cublasOperation_t(0)
    comptime CUBLAS_OP_T = cublasOperation_t(1)
    comptime CUBLAS_OP_C = cublasOperation_t(2)
    comptime CUBLAS_OP_HERMITAN = cublasOperation_t(2)
    comptime CUBLAS_OP_CONJG = cublasOperation_t(3)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.CUBLAS_OP_N:
            return writer.write_string("CUBLAS_OP_N")
        if self == Self.CUBLAS_OP_T:
            return writer.write_string("CUBLAS_OP_T")
        if self == Self.CUBLAS_OP_C:
            return writer.write_string("CUBLAS_OP_C")
        if self == Self.CUBLAS_OP_HERMITAN:
            return writer.write_string("CUBLAS_OP_HERMITAN")
        if self == Self.CUBLAS_OP_CONJG:
            return writer.write_string("CUBLAS_OP_CONJG")
        abort("invalid cublasOperation_t entry")

    def __int__(self) -> Int:
        return Int(self._value)


def cublasIdamin(
    handle: cublasHandle_t,
    n: Int16,
    x: UnsafePointer[mut=False, Float64, _],
    incx: Int16,
    result: UnsafePointer[Int16, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasIdamin_v2",
        def(
            type_of(handle),
            Int16,
            type_of(x),
            Int16,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)


def cublasSspmv(
    handle: cublasHandle_t,
    uplo: FillMode,
    n: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _ap: UnsafePointer[mut=False, Float32, _],
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int16,
    beta: UnsafePointer[mut=False, Float32, _],
    y: UnsafePointer[Float32, _],
    incy: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSspmv_v2",
        def(
            type_of(handle),
            FillMode,
            Int16,
            type_of(alpha),
            type_of(_ap),
            type_of(x),
            Int16,
            type_of(beta),
            type_of(y),
            Int16,
        ) thin -> Result,
    ]()(handle, uplo, n, alpha, _ap, x, incx, beta, y, incy)


def cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: Int16,
    n: Int16,
    k: Int16,
    alpha: UnsafePointer[mut=False, Float32, _],
    _a: UnsafePointer[mut=False, Float32, _],
    lda: Int16,
    stride_a: Int64,
    _b: UnsafePointer[mut=False, Float32, _],
    ldb: Int16,
    stride_b: Int64,
    beta: UnsafePointer[mut=False, Float32, _],
    _c: UnsafePointer[Float32, _],
    ldc: Int16,
    stride_c: Int64,
    batch_count: Int16,
) raises -> Result:
    return _get_dylib_function[
        "cublasSgemmStridedBatched",
        def(
            type_of(handle),
            cublasOperation_t,
            cublasOperation_t,
            Int16,
            Int16,
            Int16,
            type_of(alpha),
            type_of(_a),
            Int16,
            Int64,
            type_of(_b),
            Int16,
            Int64,
            type_of(beta),
            type_of(_c),
            Int16,
            Int64,
            Int16,
        ) thin -> Result,
    ]()(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        _a,
        lda,
        stride_a,
        _b,
        ldb,
        stride_b,
        beta,
        _c,
        ldc,
        stride_c,
        batch_count,
    )


def cublasSasum(
    handle: cublasHandle_t,
    n: Int64,
    x: UnsafePointer[mut=False, Float32, _],
    incx: Int64,
    result: UnsafePointer[Float32, _],
) raises -> Result:
    return _get_dylib_function[
        "cublasSasum_v2_64",
        def(
            type_of(handle),
            Int64,
            type_of(x),
            Int64,
            type_of(result),
        ) thin -> Result,
    ]()(handle, n, x, incx, result)
