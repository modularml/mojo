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

from .infer import (
    cudnnContext,
    cudnnConvolutionBwdDataAlgo_t,
    cudnnConvolutionFwdAlgo_t,
    cudnnDataType_t,
    cudnnDeterminism_t,
    cudnnMathType_t,
    cudnnStatus_t,
    cudnnTensorFormat_t,
    cudnnTensorTransformStruct,
    AnyOpaquePointer,
    DoubleNestedPointer,
)

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUDNN_CNN_INFER_LIBRARY_PATHS: List[Path] = [
    "libcudnn.so",
    "libcudnn.so.9",
    "libcudnn.so.8",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so.9",
    "/usr/lib/x86_64-linux-gnu/libcudnn.so.8",
]


def _on_error_msg() -> Error:
    return Error(
        (
            "Cannot find the CUDNN libraries. Please make sure that "
            "the CUDA toolkit is installed and that the library path is "
            "correctly set in one of the following paths ["
        ),
        ", ".join(materialize[CUDA_CUDNN_CNN_INFER_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUDNN_CNN_INFER_LIBRARY = _Global[
    "CUDA_CUDNN_CNN_INFER_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUDNN_CNN_INFER_LIBRARY_PATHS]()
    )


@always_inline
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUDNN_CNN_INFER_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#

comptime cudnnTensorStruct = AnyOpaquePointer
comptime cudnnConvolutionStruct = AnyOpaquePointer
comptime cudnnFilterStruct = AnyOpaquePointer
comptime cudnnActivationStruct = AnyOpaquePointer
comptime cudnnFusedOpsPlanStruct = AnyOpaquePointer
comptime cudnnFusedOpsVariantParamStruct = NoneType
comptime cudnnFusedOpsConstParamStruct = NoneType


def cudnnGetConvolutionMathType(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    math_type: UnsafePointer[cudnnMathType_t, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionMathType",
        def(
            type_of(conv_desc),
            type_of(math_type),
        ) thin -> cudnnStatus_t,
    ]()(conv_desc, math_type)


def cudnnIm2Col(
    handle: UnsafePointer[cudnnContext, _],
    x_desc: UnsafePointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    col_buffer: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnIm2Col",
        def(
            type_of(handle),
            type_of(x_desc),
            type_of(x),
            type_of(w_desc),
            type_of(conv_desc),
            type_of(col_buffer),
        ) thin -> cudnnStatus_t,
    ]()(handle, x_desc, x, w_desc, conv_desc, col_buffer)


def cudnnConvolutionBiasActivationForward(
    handle: UnsafePointer[cudnnContext, _],
    alpha1: OpaquePointer,
    x_desc: UnsafePointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    w: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    algo: cudnnConvolutionFwdAlgo_t,
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
    alpha2: OpaquePointer,
    z_desc: UnsafePointer[cudnnTensorStruct, _],
    z: OpaquePointer,
    bias_desc: UnsafePointer[cudnnTensorStruct, _],
    bias: OpaquePointer,
    activation_desc: UnsafePointer[cudnnActivationStruct, _],
    y_desc: UnsafePointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnConvolutionBiasActivationForward",
        def(
            type_of(handle),
            type_of(alpha1),
            type_of(x_desc),
            type_of(x),
            type_of(w_desc),
            type_of(w),
            type_of(conv_desc),
            type_of(algo),
            type_of(work_space),
            type_of(work_space_size_in_bytes),
            type_of(alpha2),
            type_of(z_desc),
            type_of(z),
            type_of(bias_desc),
            type_of(bias),
            type_of(activation_desc),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        alpha1,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        alpha2,
        z_desc,
        z,
        bias_desc,
        bias,
        activation_desc,
        y_desc,
        y,
    )


struct cudnnConvolutionFwdAlgoPerfStruct(TrivialRegisterPassable):
    var algo: cudnnConvolutionFwdAlgo_t
    var status: cudnnStatus_t
    var time: Float32
    var memory: Int
    var determinism: cudnnDeterminism_t
    var mathType: cudnnMathType_t
    var reserved: StaticTuple[Int32, 3]


def cudnnSetConvolution2dDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    pad_h: Int16,
    pad_w: Int16,
    u: Int16,
    v: Int16,
    dilation_h: Int16,
    dilation_w: Int16,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolution2dDescriptor",
        def(
            type_of(conv_desc),
            type_of(pad_h),
            type_of(pad_w),
            type_of(u),
            type_of(v),
            type_of(dilation_h),
            type_of(dilation_w),
            type_of(mode),
            type_of(compute_type),
        ) thin -> cudnnStatus_t,
    ]()(
        conv_desc,
        pad_h,
        pad_w,
        u,
        v,
        dilation_h,
        dilation_w,
        mode,
        compute_type,
    )


def cudnnCreateConvolutionDescriptor(
    conv_desc: DoubleNestedPointer[cudnnConvolutionStruct],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateConvolutionDescriptor",
        def(type_of(conv_desc)) thin -> cudnnStatus_t,
    ]()(conv_desc)


def cudnnSetConvolutionGroupCount(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _], group_count: Int16
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionGroupCount",
        def(type_of(conv_desc), type_of(group_count)) thin -> cudnnStatus_t,
    ]()(conv_desc, group_count)


comptime cudnnFusedOpsVariantParamPack_t = UnsafePointer[
    cudnnFusedOpsVariantParamStruct, _
]

comptime cudnnConvolutionFwdAlgoPerf_t = cudnnConvolutionFwdAlgoPerfStruct


struct cudnnConvolutionBwdDataAlgoPerfStruct(TrivialRegisterPassable):
    var algo: cudnnConvolutionBwdDataAlgo_t
    var status: cudnnStatus_t
    var time: Float32
    var memory: Int
    var determinism: cudnnDeterminism_t
    var mathType: cudnnMathType_t
    var reserved: StaticTuple[Int32, 3]


def cudnnGetConvolutionForwardWorkspaceSize(
    handle: UnsafePointer[cudnnContext, _],
    x_desc: UnsafePointer[cudnnTensorStruct, _],
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    y_desc: UnsafePointer[cudnnTensorStruct, _],
    algo: cudnnConvolutionFwdAlgo_t,
    size_in_bytes: UnsafePointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionForwardWorkspaceSize",
        def(
            type_of(handle),
            type_of(x_desc),
            type_of(w_desc),
            type_of(conv_desc),
            type_of(y_desc),
            type_of(algo),
            type_of(size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(handle, x_desc, w_desc, conv_desc, y_desc, algo, size_in_bytes)


def cudnnGetConvolution2dDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    pad_h: UnsafePointer[Int16, _],
    pad_w: UnsafePointer[Int16, _],
    u: UnsafePointer[Int16, _],
    v: UnsafePointer[Int16, _],
    dilation_h: UnsafePointer[Int16, _],
    dilation_w: UnsafePointer[Int16, _],
    mode: UnsafePointer[cudnnConvolutionMode_t, _],
    compute_type: UnsafePointer[cudnnDataType_t, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolution2dDescriptor",
        def(
            type_of(conv_desc),
            type_of(pad_h),
            type_of(pad_w),
            type_of(u),
            type_of(v),
            type_of(dilation_h),
            type_of(dilation_w),
            type_of(mode),
            type_of(compute_type),
        ) thin -> cudnnStatus_t,
    ]()(
        conv_desc,
        pad_h,
        pad_w,
        u,
        v,
        dilation_h,
        dilation_w,
        mode,
        compute_type,
    )


@fieldwise_init
struct cudnnFusedOpsConstParamLabel_t(
    Equatable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_PARAM_XDESC = Self(0)
    comptime CUDNN_PARAM_XDATA_PLACEHOLDER = Self(1)
    comptime CUDNN_PARAM_BN_MODE = Self(2)
    comptime CUDNN_PARAM_BN_EQSCALEBIAS_DESC = Self(3)
    comptime CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = Self(4)
    comptime CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = Self(5)
    comptime CUDNN_PARAM_ACTIVATION_DESC = Self(6)
    comptime CUDNN_PARAM_CONV_DESC = Self(7)
    comptime CUDNN_PARAM_WDESC = Self(8)
    comptime CUDNN_PARAM_WDATA_PLACEHOLDER = Self(9)
    comptime CUDNN_PARAM_DWDESC = Self(10)
    comptime CUDNN_PARAM_DWDATA_PLACEHOLDER = Self(11)
    comptime CUDNN_PARAM_YDESC = Self(12)
    comptime CUDNN_PARAM_YDATA_PLACEHOLDER = Self(13)
    comptime CUDNN_PARAM_DYDESC = Self(14)
    comptime CUDNN_PARAM_DYDATA_PLACEHOLDER = Self(15)
    comptime CUDNN_PARAM_YSTATS_DESC = Self(16)
    comptime CUDNN_PARAM_YSUM_PLACEHOLDER = Self(17)
    comptime CUDNN_PARAM_YSQSUM_PLACEHOLDER = Self(18)
    comptime CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = Self(19)
    comptime CUDNN_PARAM_BN_SCALE_PLACEHOLDER = Self(20)
    comptime CUDNN_PARAM_BN_BIAS_PLACEHOLDER = Self(21)
    comptime CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = Self(22)
    comptime CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = Self(23)
    comptime CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = Self(24)
    comptime CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = Self(25)
    comptime CUDNN_PARAM_ZDESC = Self(26)
    comptime CUDNN_PARAM_ZDATA_PLACEHOLDER = Self(27)
    comptime CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = Self(28)
    comptime CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = Self(29)
    comptime CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = Self(30)
    comptime CUDNN_PARAM_ACTIVATION_BITMASK_DESC = Self(31)
    comptime CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = Self(32)
    comptime CUDNN_PARAM_DXDESC = Self(33)
    comptime CUDNN_PARAM_DXDATA_PLACEHOLDER = Self(34)
    comptime CUDNN_PARAM_DZDESC = Self(35)
    comptime CUDNN_PARAM_DZDATA_PLACEHOLDER = Self(36)
    comptime CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = Self(37)
    comptime CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = Self(38)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_PARAM_XDESC:
            return writer.write_string("CUDNN_PARAM_XDESC")
        if self is Self.CUDNN_PARAM_XDATA_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_XDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_MODE:
            return writer.write_string("CUDNN_PARAM_BN_MODE")
        if self is Self.CUDNN_PARAM_BN_EQSCALEBIAS_DESC:
            return writer.write_string("CUDNN_PARAM_BN_EQSCALEBIAS_DESC")
        if self is Self.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_ACTIVATION_DESC:
            return writer.write_string("CUDNN_PARAM_ACTIVATION_DESC")
        if self is Self.CUDNN_PARAM_CONV_DESC:
            return writer.write_string("CUDNN_PARAM_CONV_DESC")
        if self is Self.CUDNN_PARAM_WDESC:
            return writer.write_string("CUDNN_PARAM_WDESC")
        if self is Self.CUDNN_PARAM_WDATA_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_WDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_DWDESC:
            return writer.write_string("CUDNN_PARAM_DWDESC")
        if self is Self.CUDNN_PARAM_DWDATA_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_DWDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_YDESC:
            return writer.write_string("CUDNN_PARAM_YDESC")
        if self is Self.CUDNN_PARAM_YDATA_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_YDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_DYDESC:
            return writer.write_string("CUDNN_PARAM_DYDESC")
        if self is Self.CUDNN_PARAM_DYDATA_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_DYDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_YSTATS_DESC:
            return writer.write_string("CUDNN_PARAM_YSTATS_DESC")
        if self is Self.CUDNN_PARAM_YSUM_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_YSUM_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_YSQSUM_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_YSQSUM_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC:
            return writer.write_string("CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC")
        if self is Self.CUDNN_PARAM_BN_SCALE_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_SCALE_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_BIAS_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_BIAS_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER:
            return writer.write_string(
                "CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER"
            )
        if self is Self.CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER:
            return writer.write_string(
                "CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER"
            )
        if self is Self.CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_ZDESC:
            return writer.write_string("CUDNN_PARAM_ZDESC")
        if self is Self.CUDNN_PARAM_ZDATA_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_ZDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC:
            return writer.write_string("CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC")
        if self is Self.CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_ACTIVATION_BITMASK_DESC:
            return writer.write_string("CUDNN_PARAM_ACTIVATION_BITMASK_DESC")
        if self is Self.CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER:
            return writer.write_string(
                "CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER"
            )
        if self is Self.CUDNN_PARAM_DXDESC:
            return writer.write_string("CUDNN_PARAM_DXDESC")
        if self is Self.CUDNN_PARAM_DXDATA_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_DXDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_DZDESC:
            return writer.write_string("CUDNN_PARAM_DZDESC")
        if self is Self.CUDNN_PARAM_DZDATA_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_DZDATA_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_DSCALE_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_DSCALE_PLACEHOLDER")
        if self is Self.CUDNN_PARAM_BN_DBIAS_PLACEHOLDER:
            return writer.write_string("CUDNN_PARAM_BN_DBIAS_PLACEHOLDER")
        abort("invalid cudnnFusedOpsConstParamLabel_t entry")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnFusedOpsConstParamLabel_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSetConvolutionReorderType(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    reorder_type: cudnnReorderType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionReorderType",
        def(type_of(conv_desc), type_of(reorder_type)) thin -> cudnnStatus_t,
    ]()(conv_desc, reorder_type)


@fieldwise_init
struct cudnnReorderType_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_DEFAULT_REORDER = Self(0)
    comptime CUDNN_NO_REORDER = Self(1)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_DEFAULT_REORDER:
            return writer.write_string("CUDNN_DEFAULT_REORDER")
        if self is Self.CUDNN_NO_REORDER:
            return writer.write_string("CUDNN_NO_REORDER")
        abort("invalid cudnnReorderType_t entry")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnReorderType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


comptime cudnnConvolutionBwdDataAlgoPerf_t = cudnnConvolutionBwdDataAlgoPerfStruct


def cudnnGetConvolution2dForwardOutputDim(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    input_tensor_desc: UnsafePointer[cudnnTensorStruct, _],
    filter_desc: UnsafePointer[cudnnFilterStruct, _],
    n: UnsafePointer[Int16, _],
    c: UnsafePointer[Int16, _],
    h: UnsafePointer[Int16, _],
    w: UnsafePointer[Int16, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolution2dForwardOutputDim",
        def(
            type_of(conv_desc),
            type_of(input_tensor_desc),
            type_of(filter_desc),
            type_of(n),
            type_of(c),
            type_of(h),
            type_of(w),
        ) thin -> cudnnStatus_t,
    ]()(conv_desc, input_tensor_desc, filter_desc, n, c, h, w)


def cudnnFindConvolutionForwardAlgorithm(
    handle: UnsafePointer[cudnnContext, _],
    x_desc: UnsafePointer[cudnnTensorStruct, _],
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    y_desc: UnsafePointer[cudnnTensorStruct, _],
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16, _],
    perf_results: UnsafePointer[cudnnConvolutionFwdAlgoPerfStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnFindConvolutionForwardAlgorithm",
        def(
            type_of(handle),
            type_of(x_desc),
            type_of(w_desc),
            type_of(conv_desc),
            type_of(y_desc),
            type_of(requested_algo_count),
            type_of(returned_algo_count),
            type_of(perf_results),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )


comptime cudnnFusedOpsConstParamPack_t = UnsafePointer[
    cudnnFusedOpsConstParamStruct, _
]


def cudnnGetConvolutionForwardAlgorithm_v7(
    handle: UnsafePointer[cudnnContext, _],
    src_desc: UnsafePointer[cudnnTensorStruct, _],
    filter_desc: UnsafePointer[cudnnFilterStruct, _],
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    dest_desc: UnsafePointer[cudnnTensorStruct, _],
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16, _],
    perf_results: UnsafePointer[cudnnConvolutionFwdAlgoPerfStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionForwardAlgorithm_v7",
        def(
            type_of(handle),
            type_of(src_desc),
            type_of(filter_desc),
            type_of(conv_desc),
            type_of(dest_desc),
            type_of(requested_algo_count),
            type_of(returned_algo_count),
            type_of(perf_results),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        src_desc,
        filter_desc,
        conv_desc,
        dest_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )


@fieldwise_init
struct cudnnFusedOps_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = Self(0)
    comptime CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = Self(1)
    comptime CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = Self(2)
    comptime CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = Self(3)
    comptime CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = Self(4)
    comptime CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = Self(5)
    comptime CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = Self(6)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS:
            return writer.write(
                "CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS"
            )
        if self is Self.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD:
            return writer.write_string(
                "CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD"
            )
        if self is Self.CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING:
            return writer.write_string(
                "CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING"
            )
        if self is Self.CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE:
            return writer.write_string(
                "CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE"
            )
        if self is Self.CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION:
            return writer.write_string(
                "CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION"
            )
        if self is Self.CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK:
            return writer.write(
                "CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK"
            )
        if self is Self.CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM:
            return writer.write_string(
                "CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM"
            )
        abort("invalid cudnnFusedOps_t entry")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnFusedOps_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
    handle: UnsafePointer[cudnnContext, _], count: UnsafePointer[Int16, _]
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionBackwardDataAlgorithmMaxCount",
        def(type_of(handle), type_of(count)) thin -> cudnnStatus_t,
    ]()(handle, count)


def cudnnDestroyConvolutionDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyConvolutionDescriptor",
        def(type_of(conv_desc)) thin -> cudnnStatus_t,
    ]()(conv_desc)


@fieldwise_init
struct cudnnFusedOpsPointerPlaceHolder_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_PTR_NULL = Self(0)
    comptime CUDNN_PTR_ELEM_ALIGNED = Self(1)
    comptime CUDNN_PTR_16B_ALIGNED = Self(2)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_PTR_NULL:
            return writer.write_string("CUDNN_PTR_NULL")
        if self is Self.CUDNN_PTR_ELEM_ALIGNED:
            return writer.write_string("CUDNN_PTR_ELEM_ALIGNED")
        if self is Self.CUDNN_PTR_16B_ALIGNED:
            return writer.write_string("CUDNN_PTR_16B_ALIGNED")
        abort("invalid cudnnFusedOpsPointerPlaceHolder_t entry")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnFusedOpsPointerPlaceHolder_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


comptime cudnnFusedOpsPlan_t = UnsafePointer[cudnnFusedOpsPlanStruct, _]

comptime cudnnConvolutionDescriptor_t = UnsafePointer[cudnnConvolutionStruct, _]


def cudnnConvolutionForward(
    handle: UnsafePointer[cudnnContext, _],
    alpha: OpaquePointer,
    x_desc: UnsafePointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    w: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    algo: cudnnConvolutionFwdAlgo_t,
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
    beta: OpaquePointer,
    y_desc: UnsafePointer[cudnnTensorStruct, _],
    y: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnConvolutionForward",
        def(
            type_of(handle),
            type_of(alpha),
            type_of(x_desc),
            type_of(x),
            type_of(w_desc),
            type_of(w),
            type_of(conv_desc),
            type_of(algo),
            type_of(work_space),
            type_of(work_space_size_in_bytes),
            type_of(beta),
            type_of(y_desc),
            type_of(y),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        alpha,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        y_desc,
        y,
    )


def cudnnGetConvolutionReorderType(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    reorder_type: UnsafePointer[cudnnReorderType_t, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionReorderType",
        def(
            type_of(conv_desc),
            type_of(reorder_type),
        ) thin -> cudnnStatus_t,
    ]()(conv_desc, reorder_type)


@fieldwise_init
struct cudnnFusedOpsVariantParamLabel_t(
    Equatable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_PTR_XDATA = Self(0)
    comptime CUDNN_PTR_BN_EQSCALE = Self(1)
    comptime CUDNN_PTR_BN_EQBIAS = Self(2)
    comptime CUDNN_PTR_WDATA = Self(3)
    comptime CUDNN_PTR_DWDATA = Self(4)
    comptime CUDNN_PTR_YDATA = Self(5)
    comptime CUDNN_PTR_DYDATA = Self(6)
    comptime CUDNN_PTR_YSUM = Self(7)
    comptime CUDNN_PTR_YSQSUM = Self(8)
    comptime CUDNN_PTR_WORKSPACE = Self(9)
    comptime CUDNN_PTR_BN_SCALE = Self(10)
    comptime CUDNN_PTR_BN_BIAS = Self(11)
    comptime CUDNN_PTR_BN_SAVED_MEAN = Self(12)
    comptime CUDNN_PTR_BN_SAVED_INVSTD = Self(13)
    comptime CUDNN_PTR_BN_RUNNING_MEAN = Self(14)
    comptime CUDNN_PTR_BN_RUNNING_VAR = Self(15)
    comptime CUDNN_PTR_ZDATA = Self(16)
    comptime CUDNN_PTR_BN_Z_EQSCALE = Self(17)
    comptime CUDNN_PTR_BN_Z_EQBIAS = Self(18)
    comptime CUDNN_PTR_ACTIVATION_BITMASK = Self(19)
    comptime CUDNN_PTR_DXDATA = Self(20)
    comptime CUDNN_PTR_DZDATA = Self(21)
    comptime CUDNN_PTR_BN_DSCALE = Self(22)
    comptime CUDNN_PTR_BN_DBIAS = Self(23)
    comptime CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = Self(24)
    comptime CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = Self(25)
    comptime CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = Self(26)
    comptime CUDNN_SCALAR_DOUBLE_BN_EPSILON = Self(27)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_PTR_XDATA:
            return writer.write_string("CUDNN_PTR_XDATA")
        if self is Self.CUDNN_PTR_BN_EQSCALE:
            return writer.write_string("CUDNN_PTR_BN_EQSCALE")
        if self is Self.CUDNN_PTR_BN_EQBIAS:
            return writer.write_string("CUDNN_PTR_BN_EQBIAS")
        if self is Self.CUDNN_PTR_WDATA:
            return writer.write_string("CUDNN_PTR_WDATA")
        if self is Self.CUDNN_PTR_DWDATA:
            return writer.write_string("CUDNN_PTR_DWDATA")
        if self is Self.CUDNN_PTR_YDATA:
            return writer.write_string("CUDNN_PTR_YDATA")
        if self is Self.CUDNN_PTR_DYDATA:
            return writer.write_string("CUDNN_PTR_DYDATA")
        if self is Self.CUDNN_PTR_YSUM:
            return writer.write_string("CUDNN_PTR_YSUM")
        if self is Self.CUDNN_PTR_YSQSUM:
            return writer.write_string("CUDNN_PTR_YSQSUM")
        if self is Self.CUDNN_PTR_WORKSPACE:
            return writer.write_string("CUDNN_PTR_WORKSPACE")
        if self is Self.CUDNN_PTR_BN_SCALE:
            return writer.write_string("CUDNN_PTR_BN_SCALE")
        if self is Self.CUDNN_PTR_BN_BIAS:
            return writer.write_string("CUDNN_PTR_BN_BIAS")
        if self is Self.CUDNN_PTR_BN_SAVED_MEAN:
            return writer.write_string("CUDNN_PTR_BN_SAVED_MEAN")
        if self is Self.CUDNN_PTR_BN_SAVED_INVSTD:
            return writer.write_string("CUDNN_PTR_BN_SAVED_INVSTD")
        if self is Self.CUDNN_PTR_BN_RUNNING_MEAN:
            return writer.write_string("CUDNN_PTR_BN_RUNNING_MEAN")
        if self is Self.CUDNN_PTR_BN_RUNNING_VAR:
            return writer.write_string("CUDNN_PTR_BN_RUNNING_VAR")
        if self is Self.CUDNN_PTR_ZDATA:
            return writer.write_string("CUDNN_PTR_ZDATA")
        if self is Self.CUDNN_PTR_BN_Z_EQSCALE:
            return writer.write_string("CUDNN_PTR_BN_Z_EQSCALE")
        if self is Self.CUDNN_PTR_BN_Z_EQBIAS:
            return writer.write_string("CUDNN_PTR_BN_Z_EQBIAS")
        if self is Self.CUDNN_PTR_ACTIVATION_BITMASK:
            return writer.write_string("CUDNN_PTR_ACTIVATION_BITMASK")
        if self is Self.CUDNN_PTR_DXDATA:
            return writer.write_string("CUDNN_PTR_DXDATA")
        if self is Self.CUDNN_PTR_DZDATA:
            return writer.write_string("CUDNN_PTR_DZDATA")
        if self is Self.CUDNN_PTR_BN_DSCALE:
            return writer.write_string("CUDNN_PTR_BN_DSCALE")
        if self is Self.CUDNN_PTR_BN_DBIAS:
            return writer.write_string("CUDNN_PTR_BN_DBIAS")
        if self is Self.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES:
            return writer.write_string(
                "CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES"
            )
        if self is Self.CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT:
            return writer.write_string(
                "CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT"
            )
        if self is Self.CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR:
            return writer.write_string("CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR")
        if self is Self.CUDNN_SCALAR_DOUBLE_BN_EPSILON:
            return writer.write_string("CUDNN_SCALAR_DOUBLE_BN_EPSILON")
        abort("invalid cudnnFusedOpsVariantParamLabel_t entry")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnFusedOpsVariantParamLabel_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnGetConvolutionBackwardDataAlgorithm_v7(
    handle: UnsafePointer[cudnnContext, _],
    filter_desc: UnsafePointer[cudnnFilterStruct, _],
    diff_desc: UnsafePointer[cudnnTensorStruct, _],
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    grad_desc: UnsafePointer[cudnnTensorStruct, _],
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16, _],
    perf_results: UnsafePointer[cudnnConvolutionBwdDataAlgoPerfStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionBackwardDataAlgorithm_v7",
        def(
            type_of(handle),
            type_of(filter_desc),
            type_of(diff_desc),
            type_of(conv_desc),
            type_of(grad_desc),
            type_of(requested_algo_count),
            type_of(returned_algo_count),
            type_of(perf_results),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        filter_desc,
        diff_desc,
        conv_desc,
        grad_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )


def cudnnGetConvolutionGroupCount(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    group_count: UnsafePointer[Int16, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionGroupCount",
        def(type_of(conv_desc), type_of(group_count)) thin -> cudnnStatus_t,
    ]()(conv_desc, group_count)


def cudnnGetConvolutionNdDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    array_length_requested: Int16,
    array_length: UnsafePointer[Int16, _],
    pad_a: OpaquePointer,
    stride_a: OpaquePointer,
    dilation_a: OpaquePointer,
    mode: UnsafePointer[cudnnConvolutionMode_t, _],
    compute_type: UnsafePointer[cudnnDataType_t, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionNdDescriptor",
        def(
            type_of(conv_desc),
            type_of(array_length_requested),
            type_of(array_length),
            type_of(pad_a),
            type_of(stride_a),
            type_of(dilation_a),
            type_of(mode),
            type_of(compute_type),
        ) thin -> cudnnStatus_t,
    ]()(
        conv_desc,
        array_length_requested,
        array_length,
        pad_a,
        stride_a,
        dilation_a,
        mode,
        compute_type,
    )


def cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle: UnsafePointer[cudnnContext, _],
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    dy_desc: UnsafePointer[cudnnTensorStruct, _],
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    dx_desc: UnsafePointer[cudnnTensorStruct, _],
    algo: cudnnConvolutionBwdDataAlgo_t,
    size_in_bytes: UnsafePointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionBackwardDataWorkspaceSize",
        def(
            type_of(handle),
            type_of(w_desc),
            type_of(dy_desc),
            type_of(conv_desc),
            type_of(dx_desc),
            type_of(algo),
            type_of(size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(handle, w_desc, dy_desc, conv_desc, dx_desc, algo, size_in_bytes)


def cudnnReorderFilterAndBias(
    handle: UnsafePointer[cudnnContext, _],
    filter_desc: UnsafePointer[cudnnFilterStruct, _],
    reorder_type: cudnnReorderType_t,
    filter_data: OpaquePointer,
    reordered_filter_data: OpaquePointer,
    reorder_bias: Int16,
    bias_data: OpaquePointer,
    reordered_bias_data: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnReorderFilterAndBias",
        def(
            type_of(handle),
            type_of(filter_desc),
            type_of(reorder_type),
            type_of(filter_data),
            type_of(reordered_filter_data),
            type_of(reorder_bias),
            type_of(bias_data),
            type_of(reordered_bias_data),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        filter_desc,
        reorder_type,
        filter_data,
        reordered_filter_data,
        reorder_bias,
        bias_data,
        reordered_bias_data,
    )


def cudnnFindConvolutionBackwardDataAlgorithm(
    handle: UnsafePointer[cudnnContext, _],
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    dy_desc: UnsafePointer[cudnnTensorStruct, _],
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    dx_desc: UnsafePointer[cudnnTensorStruct, _],
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16, _],
    perf_results: UnsafePointer[cudnnConvolutionBwdDataAlgoPerfStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnFindConvolutionBackwardDataAlgorithm",
        def(
            type_of(handle),
            type_of(w_desc),
            type_of(dy_desc),
            type_of(conv_desc),
            type_of(dx_desc),
            type_of(requested_algo_count),
            type_of(returned_algo_count),
            type_of(perf_results),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        w_desc,
        dy_desc,
        conv_desc,
        dx_desc,
        requested_algo_count,
        returned_algo_count,
        perf_results,
    )


def cudnnConvolutionBackwardData(
    handle: UnsafePointer[cudnnContext, _],
    alpha: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    w: OpaquePointer,
    dy_desc: UnsafePointer[cudnnTensorStruct, _],
    dy: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    algo: cudnnConvolutionBwdDataAlgo_t,
    work_space: OptionalPointer[NoneType, MutAnyOrigin],
    work_space_size_in_bytes: Int,
    beta: OpaquePointer,
    dx_desc: UnsafePointer[cudnnTensorStruct, _],
    dx: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnConvolutionBackwardData",
        def(
            type_of(handle),
            type_of(alpha),
            type_of(w_desc),
            type_of(w),
            type_of(dy_desc),
            type_of(dy),
            type_of(conv_desc),
            type_of(algo),
            type_of(work_space),
            type_of(work_space_size_in_bytes),
            type_of(beta),
            type_of(dx_desc),
            type_of(dx),
        ) thin abi("C") -> cudnnStatus_t,
    ]()(
        handle,
        alpha,
        w_desc,
        w,
        dy_desc,
        dy,
        conv_desc,
        algo,
        work_space,
        work_space_size_in_bytes,
        beta,
        dx_desc,
        dx,
    )


def cudnnSetConvolutionNdDescriptor(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    array_length: Int16,
    pad_a: OpaquePointer,
    filter_stride_a: OpaquePointer,
    dilation_a: OpaquePointer,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionNdDescriptor",
        def(
            type_of(conv_desc),
            type_of(array_length),
            type_of(pad_a),
            type_of(filter_stride_a),
            type_of(dilation_a),
            type_of(mode),
            type_of(compute_type),
        ) thin -> cudnnStatus_t,
    ]()(
        conv_desc,
        array_length,
        pad_a,
        filter_stride_a,
        dilation_a,
        mode,
        compute_type,
    )


def cudnnCnnInferVersionCheck() raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCnnInferVersionCheck", def() thin -> cudnnStatus_t
    ]()()


def cudnnSetConvolutionMathType(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    math_type: cudnnMathType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetConvolutionMathType",
        def(type_of(conv_desc), type_of(math_type)) thin -> cudnnStatus_t,
    ]()(conv_desc, math_type)


def cudnnFindConvolutionBackwardDataAlgorithmEx(
    handle: UnsafePointer[cudnnContext, _],
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    w: OpaquePointer,
    dy_desc: UnsafePointer[cudnnTensorStruct, _],
    dy: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    dx_desc: UnsafePointer[cudnnTensorStruct, _],
    dx: OpaquePointer,
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16, _],
    perf_results: UnsafePointer[cudnnConvolutionBwdDataAlgoPerfStruct, _],
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnFindConvolutionBackwardDataAlgorithmEx",
        def(
            type_of(handle),
            type_of(w_desc),
            type_of(w),
            type_of(dy_desc),
            type_of(dy),
            type_of(conv_desc),
            type_of(dx_desc),
            type_of(dx),
            type_of(requested_algo_count),
            type_of(returned_algo_count),
            type_of(perf_results),
            type_of(work_space),
            type_of(work_space_size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        w_desc,
        w,
        dy_desc,
        dy,
        conv_desc,
        dx_desc,
        dx,
        requested_algo_count,
        returned_algo_count,
        perf_results,
        work_space,
        work_space_size_in_bytes,
    )


def cudnnFindConvolutionForwardAlgorithmEx(
    handle: UnsafePointer[cudnnContext, _],
    x_desc: UnsafePointer[cudnnTensorStruct, _],
    x: OpaquePointer,
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    w: OpaquePointer,
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    y_desc: UnsafePointer[cudnnTensorStruct, _],
    y: OpaquePointer,
    requested_algo_count: Int16,
    returned_algo_count: UnsafePointer[Int16, _],
    perf_results: UnsafePointer[cudnnConvolutionFwdAlgoPerfStruct, _],
    work_space: OpaquePointer,
    work_space_size_in_bytes: Int,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnFindConvolutionForwardAlgorithmEx",
        def(
            type_of(handle),
            type_of(x_desc),
            type_of(x),
            type_of(w_desc),
            type_of(w),
            type_of(conv_desc),
            type_of(y_desc),
            type_of(y),
            type_of(requested_algo_count),
            type_of(returned_algo_count),
            type_of(perf_results),
            type_of(work_space),
            type_of(work_space_size_in_bytes),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        y_desc,
        y,
        requested_algo_count,
        returned_algo_count,
        perf_results,
        work_space,
        work_space_size_in_bytes,
    )


def cudnnGetConvolutionNdForwardOutputDim(
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    input_tensor_desc: UnsafePointer[cudnnTensorStruct, _],
    filter_desc: UnsafePointer[cudnnFilterStruct, _],
    nb_dims: Int16,
    tensor_output_dim_a: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionNdForwardOutputDim",
        def(
            type_of(conv_desc),
            type_of(input_tensor_desc),
            type_of(filter_desc),
            type_of(nb_dims),
            type_of(tensor_output_dim_a),
        ) thin -> cudnnStatus_t,
    ]()(conv_desc, input_tensor_desc, filter_desc, nb_dims, tensor_output_dim_a)


def cudnnGetConvolutionForwardAlgorithmMaxCount(
    handle: UnsafePointer[cudnnContext, _], count: UnsafePointer[Int16, _]
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetConvolutionForwardAlgorithmMaxCount",
        def(type_of(handle), type_of(count)) thin -> cudnnStatus_t,
    ]()(handle, count)


@fieldwise_init
struct cudnnConvolutionMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_CONVOLUTION = Self(0)
    comptime CUDNN_CROSS_CORRELATION = Self(1)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_CONVOLUTION:
            return writer.write_string("CUDNN_CONVOLUTION")
        if self is Self.CUDNN_CROSS_CORRELATION:
            return writer.write_string("CUDNN_CROSS_CORRELATION")
        abort("invalid cudnnConvolutionMode_t entry")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnConvolutionMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnGetFoldedConvBackwardDataDescriptors(
    handle: UnsafePointer[cudnnContext, _],
    filter_desc: UnsafePointer[cudnnFilterStruct, _],
    diff_desc: UnsafePointer[cudnnTensorStruct, _],
    conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    grad_desc: UnsafePointer[cudnnTensorStruct, _],
    transform_format: cudnnTensorFormat_t,
    folded_filter_desc: UnsafePointer[cudnnFilterStruct, _],
    padded_diff_desc: UnsafePointer[cudnnTensorStruct, _],
    folded_conv_desc: UnsafePointer[cudnnConvolutionStruct, _],
    folded_grad_desc: UnsafePointer[cudnnTensorStruct, _],
    filter_fold_trans_desc: UnsafePointer[cudnnTensorTransformStruct, _],
    diff_pad_trans_desc: UnsafePointer[cudnnTensorTransformStruct, _],
    grad_fold_trans_desc: UnsafePointer[cudnnTensorTransformStruct, _],
    grad_unfold_trans_desc: UnsafePointer[cudnnTensorTransformStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetFoldedConvBackwardDataDescriptors",
        def(
            type_of(handle),
            type_of(filter_desc),
            type_of(diff_desc),
            type_of(conv_desc),
            type_of(grad_desc),
            type_of(transform_format),
            type_of(folded_filter_desc),
            type_of(padded_diff_desc),
            type_of(folded_conv_desc),
            type_of(folded_grad_desc),
            type_of(filter_fold_trans_desc),
            type_of(diff_pad_trans_desc),
            type_of(grad_fold_trans_desc),
            type_of(grad_unfold_trans_desc),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        filter_desc,
        diff_desc,
        conv_desc,
        grad_desc,
        transform_format,
        folded_filter_desc,
        padded_diff_desc,
        folded_conv_desc,
        folded_grad_desc,
        filter_fold_trans_desc,
        diff_pad_trans_desc,
        grad_fold_trans_desc,
        grad_unfold_trans_desc,
    )
