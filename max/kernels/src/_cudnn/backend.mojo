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

from .infer import cudnnContext, cudnnStatus_t

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUDNN_LIBRARY_PATHS: List[Path] = [
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
        ", ".join(materialize[CUDA_CUDNN_LIBRARY_PATHS]()),
        (
            "]. You may need to make sure that you are using the non-slim"
            " version of the MAX container."
        ),
    )


comptime CUDA_CUDNN_LIBRARY = _Global[
    "CUDA_CUDNN_LIBRARY", _init_dylib, on_error_msg=_on_error_msg
]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib[abort_on_failure=False](
        materialize[CUDA_CUDNN_LIBRARY_PATHS]()
    )


@inline(.always)
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUDNN_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#


def cudnnBackendInitialize(
    descriptor: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendInitialize",
        def(type_of(descriptor)) thin -> cudnnStatus_t,
    ]()(descriptor)


@fieldwise_init
struct cudnnBackendKnobType_t(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int8
    comptime CUDNN_KNOB_TYPE_SPLIT_K = Self(0)
    comptime CUDNN_KNOB_TYPE_SWIZZLE = Self(1)
    comptime CUDNN_KNOB_TYPE_TILE_SIZE = Self(2)
    comptime CUDNN_KNOB_TYPE_USE_TEX = Self(3)
    comptime CUDNN_KNOB_TYPE_EDGE = Self(4)
    comptime CUDNN_KNOB_TYPE_KBLOCK = Self(5)
    comptime CUDNN_KNOB_TYPE_LDGA = Self(6)
    comptime CUDNN_KNOB_TYPE_LDGB = Self(7)
    comptime CUDNN_KNOB_TYPE_CHUNK_K = Self(8)
    comptime CUDNN_KNOB_TYPE_SPLIT_H = Self(9)
    comptime CUDNN_KNOB_TYPE_WINO_TILE = Self(10)
    comptime CUDNN_KNOB_TYPE_MULTIPLY = Self(11)
    comptime CUDNN_KNOB_TYPE_SPLIT_K_BUF = Self(12)
    comptime CUDNN_KNOB_TYPE_TILEK = Self(13)
    comptime CUDNN_KNOB_TYPE_STAGES = Self(14)
    comptime CUDNN_KNOB_TYPE_REDUCTION_MODE = Self(15)
    comptime CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE = Self(16)
    comptime CUDNN_KNOB_TYPE_SPLIT_K_SLC = Self(17)
    comptime CUDNN_KNOB_TYPE_IDX_MODE = Self(18)
    comptime CUDNN_KNOB_TYPE_SLICED = Self(19)
    comptime CUDNN_KNOB_TYPE_SPLIT_RS = Self(20)
    comptime CUDNN_KNOB_TYPE_SINGLEBUFFER = Self(21)
    comptime CUDNN_KNOB_TYPE_LDGC = Self(22)
    comptime CUDNN_KNOB_TYPE_SPECFILT = Self(23)
    comptime CUDNN_KNOB_TYPE_KERNEL_CFG = Self(24)
    comptime CUDNN_KNOB_TYPE_WORKSPACE = Self(25)
    comptime CUDNN_KNOB_TYPE_TILE_CGA = Self(26)
    comptime CUDNN_KNOB_TYPE_TILE_CGA_M = Self(27)
    comptime CUDNN_KNOB_TYPE_TILE_CGA_N = Self(28)
    comptime CUDNN_KNOB_TYPE_BLOCK_SIZE = Self(29)
    comptime CUDNN_KNOB_TYPE_OCCUPANCY = Self(30)
    comptime CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD = Self(31)
    comptime CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK = Self(32)
    comptime CUDNN_KNOB_TYPE_SPLIT_COLS = Self(33)
    comptime CUDNN_KNOB_TYPE_TILE_ROWS = Self(34)
    comptime CUDNN_KNOB_TYPE_TILE_COLS = Self(35)
    comptime CUDNN_KNOB_TYPE_LOAD_SIZE = Self(36)
    comptime CUDNN_KNOB_TYPE_COUNTS = Self(37)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_KNOB_TYPE_SPLIT_K:
            return writer.write_string("CUDNN_KNOB_TYPE_SPLIT_K")
        if self is Self.CUDNN_KNOB_TYPE_SWIZZLE:
            return writer.write_string("CUDNN_KNOB_TYPE_SWIZZLE")
        if self is Self.CUDNN_KNOB_TYPE_TILE_SIZE:
            return writer.write_string("CUDNN_KNOB_TYPE_TILE_SIZE")
        if self is Self.CUDNN_KNOB_TYPE_USE_TEX:
            return writer.write_string("CUDNN_KNOB_TYPE_USE_TEX")
        if self is Self.CUDNN_KNOB_TYPE_EDGE:
            return writer.write_string("CUDNN_KNOB_TYPE_EDGE")
        if self is Self.CUDNN_KNOB_TYPE_KBLOCK:
            return writer.write_string("CUDNN_KNOB_TYPE_KBLOCK")
        if self is Self.CUDNN_KNOB_TYPE_LDGA:
            return writer.write_string("CUDNN_KNOB_TYPE_LDGA")
        if self is Self.CUDNN_KNOB_TYPE_LDGB:
            return writer.write_string("CUDNN_KNOB_TYPE_LDGB")
        if self is Self.CUDNN_KNOB_TYPE_CHUNK_K:
            return writer.write_string("CUDNN_KNOB_TYPE_CHUNK_K")
        if self is Self.CUDNN_KNOB_TYPE_SPLIT_H:
            return writer.write_string("CUDNN_KNOB_TYPE_SPLIT_H")
        if self is Self.CUDNN_KNOB_TYPE_WINO_TILE:
            return writer.write_string("CUDNN_KNOB_TYPE_WINO_TILE")
        if self is Self.CUDNN_KNOB_TYPE_MULTIPLY:
            return writer.write_string("CUDNN_KNOB_TYPE_MULTIPLY")
        if self is Self.CUDNN_KNOB_TYPE_SPLIT_K_BUF:
            return writer.write_string("CUDNN_KNOB_TYPE_SPLIT_K_BUF")
        if self is Self.CUDNN_KNOB_TYPE_TILEK:
            return writer.write_string("CUDNN_KNOB_TYPE_TILEK")
        if self is Self.CUDNN_KNOB_TYPE_STAGES:
            return writer.write_string("CUDNN_KNOB_TYPE_STAGES")
        if self is Self.CUDNN_KNOB_TYPE_REDUCTION_MODE:
            return writer.write_string("CUDNN_KNOB_TYPE_REDUCTION_MODE")
        if self is Self.CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE:
            return writer.write_string("CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE")
        if self is Self.CUDNN_KNOB_TYPE_SPLIT_K_SLC:
            return writer.write_string("CUDNN_KNOB_TYPE_SPLIT_K_SLC")
        if self is Self.CUDNN_KNOB_TYPE_IDX_MODE:
            return writer.write_string("CUDNN_KNOB_TYPE_IDX_MODE")
        if self is Self.CUDNN_KNOB_TYPE_SLICED:
            return writer.write_string("CUDNN_KNOB_TYPE_SLICED")
        if self is Self.CUDNN_KNOB_TYPE_SPLIT_RS:
            return writer.write_string("CUDNN_KNOB_TYPE_SPLIT_RS")
        if self is Self.CUDNN_KNOB_TYPE_SINGLEBUFFER:
            return writer.write_string("CUDNN_KNOB_TYPE_SINGLEBUFFER")
        if self is Self.CUDNN_KNOB_TYPE_LDGC:
            return writer.write_string("CUDNN_KNOB_TYPE_LDGC")
        if self is Self.CUDNN_KNOB_TYPE_SPECFILT:
            return writer.write_string("CUDNN_KNOB_TYPE_SPECFILT")
        if self is Self.CUDNN_KNOB_TYPE_KERNEL_CFG:
            return writer.write_string("CUDNN_KNOB_TYPE_KERNEL_CFG")
        if self is Self.CUDNN_KNOB_TYPE_WORKSPACE:
            return writer.write_string("CUDNN_KNOB_TYPE_WORKSPACE")
        if self is Self.CUDNN_KNOB_TYPE_TILE_CGA:
            return writer.write_string("CUDNN_KNOB_TYPE_TILE_CGA")
        if self is Self.CUDNN_KNOB_TYPE_TILE_CGA_M:
            return writer.write_string("CUDNN_KNOB_TYPE_TILE_CGA_M")
        if self is Self.CUDNN_KNOB_TYPE_TILE_CGA_N:
            return writer.write_string("CUDNN_KNOB_TYPE_TILE_CGA_N")
        if self is Self.CUDNN_KNOB_TYPE_BLOCK_SIZE:
            return writer.write_string("CUDNN_KNOB_TYPE_BLOCK_SIZE")
        if self is Self.CUDNN_KNOB_TYPE_OCCUPANCY:
            return writer.write_string("CUDNN_KNOB_TYPE_OCCUPANCY")
        if self is Self.CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD:
            return writer.write_string("CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD")
        if self is Self.CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK:
            return writer.write_string("CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK")
        if self is Self.CUDNN_KNOB_TYPE_SPLIT_COLS:
            return writer.write_string("CUDNN_KNOB_TYPE_SPLIT_COLS")
        if self is Self.CUDNN_KNOB_TYPE_TILE_ROWS:
            return writer.write_string("CUDNN_KNOB_TYPE_TILE_ROWS")
        if self is Self.CUDNN_KNOB_TYPE_TILE_COLS:
            return writer.write_string("CUDNN_KNOB_TYPE_TILE_COLS")
        if self is Self.CUDNN_KNOB_TYPE_LOAD_SIZE:
            return writer.write_string("CUDNN_KNOB_TYPE_LOAD_SIZE")
        if self is Self.CUDNN_KNOB_TYPE_COUNTS:
            return writer.write_string("CUDNN_KNOB_TYPE_COUNTS")
        abort("invalid cudnnBackendKnobType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendKnobType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnPointwiseMode_t(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int8
    comptime CUDNN_POINTWISE_ADD = Self(0)
    comptime CUDNN_POINTWISE_ADD_SQUARE = Self(1)
    comptime CUDNN_POINTWISE_DIV = Self(2)
    comptime CUDNN_POINTWISE_MAX = Self(3)
    comptime CUDNN_POINTWISE_MIN = Self(4)
    comptime CUDNN_POINTWISE_MOD = Self(5)
    comptime CUDNN_POINTWISE_MUL = Self(6)
    comptime CUDNN_POINTWISE_POW = Self(7)
    comptime CUDNN_POINTWISE_SUB = Self(8)
    comptime CUDNN_POINTWISE_ABS = Self(9)
    comptime CUDNN_POINTWISE_CEIL = Self(10)
    comptime CUDNN_POINTWISE_COS = Self(11)
    comptime CUDNN_POINTWISE_EXP = Self(12)
    comptime CUDNN_POINTWISE_FLOOR = Self(13)
    comptime CUDNN_POINTWISE_LOG = Self(14)
    comptime CUDNN_POINTWISE_NEG = Self(15)
    comptime CUDNN_POINTWISE_RSQRT = Self(16)
    comptime CUDNN_POINTWISE_SIN = Self(17)
    comptime CUDNN_POINTWISE_SQRT = Self(18)
    comptime CUDNN_POINTWISE_TAN = Self(19)
    comptime CUDNN_POINTWISE_ERF = Self(20)
    comptime CUDNN_POINTWISE_IDENTITY = Self(21)
    comptime CUDNN_POINTWISE_RECIPROCAL = Self(22)
    comptime CUDNN_POINTWISE_RELU_FWD = Self(23)
    comptime CUDNN_POINTWISE_TANH_FWD = Self(24)
    comptime CUDNN_POINTWISE_SIGMOID_FWD = Self(25)
    comptime CUDNN_POINTWISE_ELU_FWD = Self(26)
    comptime CUDNN_POINTWISE_GELU_FWD = Self(27)
    comptime CUDNN_POINTWISE_SOFTPLUS_FWD = Self(28)
    comptime CUDNN_POINTWISE_SWISH_FWD = Self(29)
    comptime CUDNN_POINTWISE_GELU_APPROX_TANH_FWD = Self(30)
    comptime CUDNN_POINTWISE_RELU_BWD = Self(31)
    comptime CUDNN_POINTWISE_TANH_BWD = Self(32)
    comptime CUDNN_POINTWISE_SIGMOID_BWD = Self(33)
    comptime CUDNN_POINTWISE_ELU_BWD = Self(34)
    comptime CUDNN_POINTWISE_GELU_BWD = Self(35)
    comptime CUDNN_POINTWISE_SOFTPLUS_BWD = Self(36)
    comptime CUDNN_POINTWISE_SWISH_BWD = Self(37)
    comptime CUDNN_POINTWISE_GELU_APPROX_TANH_BWD = Self(38)
    comptime CUDNN_POINTWISE_CMP_EQ = Self(39)
    comptime CUDNN_POINTWISE_CMP_NEQ = Self(40)
    comptime CUDNN_POINTWISE_CMP_GT = Self(41)
    comptime CUDNN_POINTWISE_CMP_GE = Self(42)
    comptime CUDNN_POINTWISE_CMP_LT = Self(43)
    comptime CUDNN_POINTWISE_CMP_LE = Self(44)
    comptime CUDNN_POINTWISE_LOGICAL_AND = Self(45)
    comptime CUDNN_POINTWISE_LOGICAL_OR = Self(46)
    comptime CUDNN_POINTWISE_LOGICAL_NOT = Self(47)
    comptime CUDNN_POINTWISE_GEN_INDEX = Self(48)
    comptime CUDNN_POINTWISE_BINARY_SELECT = Self(49)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_POINTWISE_ADD:
            return writer.write_string("CUDNN_POINTWISE_ADD")
        if self is Self.CUDNN_POINTWISE_ADD_SQUARE:
            return writer.write_string("CUDNN_POINTWISE_ADD_SQUARE")
        if self is Self.CUDNN_POINTWISE_DIV:
            return writer.write_string("CUDNN_POINTWISE_DIV")
        if self is Self.CUDNN_POINTWISE_MAX:
            return writer.write_string("CUDNN_POINTWISE_MAX")
        if self is Self.CUDNN_POINTWISE_MIN:
            return writer.write_string("CUDNN_POINTWISE_MIN")
        if self is Self.CUDNN_POINTWISE_MOD:
            return writer.write_string("CUDNN_POINTWISE_MOD")
        if self is Self.CUDNN_POINTWISE_MUL:
            return writer.write_string("CUDNN_POINTWISE_MUL")
        if self is Self.CUDNN_POINTWISE_POW:
            return writer.write_string("CUDNN_POINTWISE_POW")
        if self is Self.CUDNN_POINTWISE_SUB:
            return writer.write_string("CUDNN_POINTWISE_SUB")
        if self is Self.CUDNN_POINTWISE_ABS:
            return writer.write_string("CUDNN_POINTWISE_ABS")
        if self is Self.CUDNN_POINTWISE_CEIL:
            return writer.write_string("CUDNN_POINTWISE_CEIL")
        if self is Self.CUDNN_POINTWISE_COS:
            return writer.write_string("CUDNN_POINTWISE_COS")
        if self is Self.CUDNN_POINTWISE_EXP:
            return writer.write_string("CUDNN_POINTWISE_EXP")
        if self is Self.CUDNN_POINTWISE_FLOOR:
            return writer.write_string("CUDNN_POINTWISE_FLOOR")
        if self is Self.CUDNN_POINTWISE_LOG:
            return writer.write_string("CUDNN_POINTWISE_LOG")
        if self is Self.CUDNN_POINTWISE_NEG:
            return writer.write_string("CUDNN_POINTWISE_NEG")
        if self is Self.CUDNN_POINTWISE_RSQRT:
            return writer.write_string("CUDNN_POINTWISE_RSQRT")
        if self is Self.CUDNN_POINTWISE_SIN:
            return writer.write_string("CUDNN_POINTWISE_SIN")
        if self is Self.CUDNN_POINTWISE_SQRT:
            return writer.write_string("CUDNN_POINTWISE_SQRT")
        if self is Self.CUDNN_POINTWISE_TAN:
            return writer.write_string("CUDNN_POINTWISE_TAN")
        if self is Self.CUDNN_POINTWISE_ERF:
            return writer.write_string("CUDNN_POINTWISE_ERF")
        if self is Self.CUDNN_POINTWISE_IDENTITY:
            return writer.write_string("CUDNN_POINTWISE_IDENTITY")
        if self is Self.CUDNN_POINTWISE_RECIPROCAL:
            return writer.write_string("CUDNN_POINTWISE_RECIPROCAL")
        if self is Self.CUDNN_POINTWISE_RELU_FWD:
            return writer.write_string("CUDNN_POINTWISE_RELU_FWD")
        if self is Self.CUDNN_POINTWISE_TANH_FWD:
            return writer.write_string("CUDNN_POINTWISE_TANH_FWD")
        if self is Self.CUDNN_POINTWISE_SIGMOID_FWD:
            return writer.write_string("CUDNN_POINTWISE_SIGMOID_FWD")
        if self is Self.CUDNN_POINTWISE_ELU_FWD:
            return writer.write_string("CUDNN_POINTWISE_ELU_FWD")
        if self is Self.CUDNN_POINTWISE_GELU_FWD:
            return writer.write_string("CUDNN_POINTWISE_GELU_FWD")
        if self is Self.CUDNN_POINTWISE_SOFTPLUS_FWD:
            return writer.write_string("CUDNN_POINTWISE_SOFTPLUS_FWD")
        if self is Self.CUDNN_POINTWISE_SWISH_FWD:
            return writer.write_string("CUDNN_POINTWISE_SWISH_FWD")
        if self is Self.CUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
            return writer.write_string("CUDNN_POINTWISE_GELU_APPROX_TANH_FWD")
        if self is Self.CUDNN_POINTWISE_RELU_BWD:
            return writer.write_string("CUDNN_POINTWISE_RELU_BWD")
        if self is Self.CUDNN_POINTWISE_TANH_BWD:
            return writer.write_string("CUDNN_POINTWISE_TANH_BWD")
        if self is Self.CUDNN_POINTWISE_SIGMOID_BWD:
            return writer.write_string("CUDNN_POINTWISE_SIGMOID_BWD")
        if self is Self.CUDNN_POINTWISE_ELU_BWD:
            return writer.write_string("CUDNN_POINTWISE_ELU_BWD")
        if self is Self.CUDNN_POINTWISE_GELU_BWD:
            return writer.write_string("CUDNN_POINTWISE_GELU_BWD")
        if self is Self.CUDNN_POINTWISE_SOFTPLUS_BWD:
            return writer.write_string("CUDNN_POINTWISE_SOFTPLUS_BWD")
        if self is Self.CUDNN_POINTWISE_SWISH_BWD:
            return writer.write_string("CUDNN_POINTWISE_SWISH_BWD")
        if self is Self.CUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
            return writer.write_string("CUDNN_POINTWISE_GELU_APPROX_TANH_BWD")
        if self is Self.CUDNN_POINTWISE_CMP_EQ:
            return writer.write_string("CUDNN_POINTWISE_CMP_EQ")
        if self is Self.CUDNN_POINTWISE_CMP_NEQ:
            return writer.write_string("CUDNN_POINTWISE_CMP_NEQ")
        if self is Self.CUDNN_POINTWISE_CMP_GT:
            return writer.write_string("CUDNN_POINTWISE_CMP_GT")
        if self is Self.CUDNN_POINTWISE_CMP_GE:
            return writer.write_string("CUDNN_POINTWISE_CMP_GE")
        if self is Self.CUDNN_POINTWISE_CMP_LT:
            return writer.write_string("CUDNN_POINTWISE_CMP_LT")
        if self is Self.CUDNN_POINTWISE_CMP_LE:
            return writer.write_string("CUDNN_POINTWISE_CMP_LE")
        if self is Self.CUDNN_POINTWISE_LOGICAL_AND:
            return writer.write_string("CUDNN_POINTWISE_LOGICAL_AND")
        if self is Self.CUDNN_POINTWISE_LOGICAL_OR:
            return writer.write_string("CUDNN_POINTWISE_LOGICAL_OR")
        if self is Self.CUDNN_POINTWISE_LOGICAL_NOT:
            return writer.write_string("CUDNN_POINTWISE_LOGICAL_NOT")
        if self is Self.CUDNN_POINTWISE_GEN_INDEX:
            return writer.write_string("CUDNN_POINTWISE_GEN_INDEX")
        if self is Self.CUDNN_POINTWISE_BINARY_SELECT:
            return writer.write_string("CUDNN_POINTWISE_BINARY_SELECT")
        abort("invalid cudnnPointwiseMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnPointwiseMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnBackendDescriptorType_t(
    Equatable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_BACKEND_POINTWISE_DESCRIPTOR = Self(0)
    comptime CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR = Self(1)
    comptime CUDNN_BACKEND_ENGINE_DESCRIPTOR = Self(2)
    comptime CUDNN_BACKEND_ENGINECFG_DESCRIPTOR = Self(3)
    comptime CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR = Self(4)
    comptime CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR = Self(5)
    comptime CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR = Self(6)
    comptime CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR = Self(7)
    comptime CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR = Self(8)
    comptime CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR = Self(9)
    comptime CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = Self(10)
    comptime CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = Self(
        11
    )
    comptime CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = Self(
        12
    )
    comptime CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = Self(13)
    comptime CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = Self(14)
    comptime CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR = Self(15)
    comptime CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = Self(16)
    comptime CUDNN_BACKEND_TENSOR_DESCRIPTOR = Self(17)
    comptime CUDNN_BACKEND_MATMUL_DESCRIPTOR = Self(18)
    comptime CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = Self(19)
    comptime CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR = Self(
        20
    )
    comptime CUDNN_BACKEND_REDUCTION_DESCRIPTOR = Self(21)
    comptime CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = Self(22)
    comptime CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR = Self(23)
    comptime CUDNN_BACKEND_RESAMPLE_DESCRIPTOR = Self(24)
    comptime CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = Self(25)
    comptime CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = Self(26)
    comptime CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR = Self(27)
    comptime CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR = Self(28)
    comptime CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = Self(29)
    comptime CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = Self(30)
    comptime CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR = Self(31)
    comptime CUDNN_BACKEND_RNG_DESCRIPTOR = Self(32)
    comptime CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR = Self(33)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_BACKEND_POINTWISE_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_POINTWISE_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_ENGINE_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_ENGINE_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_ENGINECFG_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_ENGINECFG_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
            return writer.write(
                "CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR"
            )
        if (
            self
            is Self.CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR
        ):
            return writer.write(
                "CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR"
            )
        if (
            self
            is Self.CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
        ):
            return writer.write(
                "CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_TENSOR_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_TENSOR_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_MATMUL_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_MATMUL_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR"
            )
        if (
            self
            is Self.CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR
        ):
            return writer.write(
                "CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_REDUCTION_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_REDUCTION_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
            return writer.write(
                "CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_RESAMPLE_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_RESAMPLE_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR:
            return writer.write(
                "CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR:
            return writer.write(
                "CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR:
            return writer.write(
                "CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR:
            return writer.write(
                "CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR"
            )
        if self is Self.CUDNN_BACKEND_RNG_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_RNG_DESCRIPTOR")
        if self is Self.CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR:
            return writer.write_string("CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR")
        abort("invalid cudnnBackendDescriptorType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendDescriptorType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnBackendSetAttribute(
    descriptor: OpaquePointer,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    element_count: Int64,
    array_of_elements: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendSetAttribute",
        def(
            type_of(descriptor),
            type_of(attribute_name),
            type_of(attribute_type),
            type_of(element_count),
            type_of(array_of_elements),
        ) thin -> cudnnStatus_t,
    ]()(
        descriptor,
        attribute_name,
        attribute_type,
        element_count,
        array_of_elements,
    )


@fieldwise_init
struct cudnnBackendBehaviorNote_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION = Self(0)
    comptime CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER = Self(1)
    comptime CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER = Self(2)
    comptime CUDNN_BEHAVIOR_NOTE_TYPE_COUNT = Self(3)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
            return writer.write_string(
                "CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION"
            )
        if self is Self.CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
            return writer.write(
                "CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER"
            )
        if self is Self.CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
            return writer.write(
                "CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER"
            )
        if self is Self.CUDNN_BEHAVIOR_NOTE_TYPE_COUNT:
            return writer.write_string("CUDNN_BEHAVIOR_NOTE_TYPE_COUNT")
        abort("invalid cudnnBackendBehaviorNote_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendBehaviorNote_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnBackendLayoutType_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_LAYOUT_TYPE_PREFERRED_NCHW = Self(0)
    comptime CUDNN_LAYOUT_TYPE_PREFERRED_NHWC = Self(1)
    comptime CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK = Self(2)
    comptime CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK = Self(3)
    comptime CUDNN_LAYOUT_TYPE_COUNT = Self(4)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_LAYOUT_TYPE_PREFERRED_NCHW:
            return writer.write_string("CUDNN_LAYOUT_TYPE_PREFERRED_NCHW")
        if self is Self.CUDNN_LAYOUT_TYPE_PREFERRED_NHWC:
            return writer.write_string("CUDNN_LAYOUT_TYPE_PREFERRED_NHWC")
        if self is Self.CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK:
            return writer.write_string("CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK")
        if self is Self.CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK:
            return writer.write_string("CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK")
        if self is Self.CUDNN_LAYOUT_TYPE_COUNT:
            return writer.write_string("CUDNN_LAYOUT_TYPE_COUNT")
        abort("invalid cudnnBackendLayoutType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendLayoutType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnBackendNormFwdPhase_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_NORM_FWD_INFERENCE = Self(0)
    comptime CUDNN_NORM_FWD_TRAINING = Self(1)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_NORM_FWD_INFERENCE:
            return writer.write_string("CUDNN_NORM_FWD_INFERENCE")
        if self is Self.CUDNN_NORM_FWD_TRAINING:
            return writer.write_string("CUDNN_NORM_FWD_TRAINING")
        abort("invalid cudnnBackendNormFwdPhase_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendNormFwdPhase_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnBackendHeurMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_HEUR_MODE_INSTANT = Self(0)
    comptime CUDNN_HEUR_MODE_B = Self(1)
    comptime CUDNN_HEUR_MODE_FALLBACK = Self(2)
    comptime CUDNN_HEUR_MODE_A = Self(3)
    comptime CUDNN_HEUR_MODES_COUNT = Self(4)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_HEUR_MODE_INSTANT:
            return writer.write_string("CUDNN_HEUR_MODE_INSTANT")
        if self is Self.CUDNN_HEUR_MODE_B:
            return writer.write_string("CUDNN_HEUR_MODE_B")
        if self is Self.CUDNN_HEUR_MODE_FALLBACK:
            return writer.write_string("CUDNN_HEUR_MODE_FALLBACK")
        if self is Self.CUDNN_HEUR_MODE_A:
            return writer.write_string("CUDNN_HEUR_MODE_A")
        if self is Self.CUDNN_HEUR_MODES_COUNT:
            return writer.write_string("CUDNN_HEUR_MODES_COUNT")
        abort("invalid cudnnBackendHeurMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendHeurMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


struct cudnnFractionStruct(TrivialRegisterPassable):
    var numerator: Int64
    var denominator: Int64


@fieldwise_init
struct cudnnBackendNumericalNote_t(
    Equatable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_NUMERICAL_NOTE_TENSOR_CORE = Self(0)
    comptime CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS = Self(1)
    comptime CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION = Self(2)
    comptime CUDNN_NUMERICAL_NOTE_FFT = Self(3)
    comptime CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC = Self(4)
    comptime CUDNN_NUMERICAL_NOTE_WINOGRAD = Self(5)
    comptime CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4 = Self(6)
    comptime CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6 = Self(7)
    comptime CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13 = Self(8)
    comptime CUDNN_NUMERICAL_NOTE_TYPE_COUNT = Self(9)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
            return writer.write_string("CUDNN_NUMERICAL_NOTE_TENSOR_CORE")
        if self is Self.CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
            return writer.write_string(
                "CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS"
            )
        if self is Self.CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
            return writer.write(
                "CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION"
            )
        if self is Self.CUDNN_NUMERICAL_NOTE_FFT:
            return writer.write_string("CUDNN_NUMERICAL_NOTE_FFT")
        if self is Self.CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
            return writer.write_string("CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC")
        if self is Self.CUDNN_NUMERICAL_NOTE_WINOGRAD:
            return writer.write_string("CUDNN_NUMERICAL_NOTE_WINOGRAD")
        if self is Self.CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
            return writer.write_string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4")
        if self is Self.CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
            return writer.write_string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6")
        if self is Self.CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13:
            return writer.write_string(
                "CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13"
            )
        if self is Self.CUDNN_NUMERICAL_NOTE_TYPE_COUNT:
            return writer.write_string("CUDNN_NUMERICAL_NOTE_TYPE_COUNT")
        abort("invalid cudnnBackendNumericalNote_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendNumericalNote_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnBackendCreateDescriptor(
    descriptor_type: cudnnBackendDescriptorType_t,
    descriptor: UnsafePointer[OpaquePointer[AnyOrigin[mut=True]], _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendCreateDescriptor",
        def(
            type_of(descriptor_type), type_of(descriptor)
        ) thin -> cudnnStatus_t,
    ]()(descriptor_type, descriptor)


@fieldwise_init
struct cudnnBackendAttributeType_t(
    Equatable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_TYPE_HANDLE = Self(0)
    comptime CUDNN_TYPE_DATA_TYPE = Self(1)
    comptime CUDNN_TYPE_BOOLEAN = Self(2)
    comptime CUDNN_TYPE_INT64 = Self(3)
    comptime CUDNN_TYPE_FLOAT = Self(4)
    comptime CUDNN_TYPE_DOUBLE = Self(5)
    comptime CUDNN_TYPE_VOID_PTR = Self(6)
    comptime CUDNN_TYPE_CONVOLUTION_MODE = Self(7)
    comptime CUDNN_TYPE_HEUR_MODE = Self(8)
    comptime CUDNN_TYPE_KNOB_TYPE = Self(9)
    comptime CUDNN_TYPE_NAN_PROPOGATION = Self(10)
    comptime CUDNN_TYPE_NUMERICAL_NOTE = Self(11)
    comptime CUDNN_TYPE_LAYOUT_TYPE = Self(12)
    comptime CUDNN_TYPE_ATTRIB_NAME = Self(13)
    comptime CUDNN_TYPE_POINTWISE_MODE = Self(14)
    comptime CUDNN_TYPE_BACKEND_DESCRIPTOR = Self(15)
    comptime CUDNN_TYPE_GENSTATS_MODE = Self(16)
    comptime CUDNN_TYPE_BN_FINALIZE_STATS_MODE = Self(17)
    comptime CUDNN_TYPE_REDUCTION_OPERATOR_TYPE = Self(18)
    comptime CUDNN_TYPE_BEHAVIOR_NOTE = Self(19)
    comptime CUDNN_TYPE_TENSOR_REORDERING_MODE = Self(20)
    comptime CUDNN_TYPE_RESAMPLE_MODE = Self(21)
    comptime CUDNN_TYPE_PADDING_MODE = Self(22)
    comptime CUDNN_TYPE_INT32 = Self(23)
    comptime CUDNN_TYPE_CHAR = Self(24)
    comptime CUDNN_TYPE_SIGNAL_MODE = Self(25)
    comptime CUDNN_TYPE_FRACTION = Self(26)
    comptime CUDNN_TYPE_NORM_MODE = Self(27)
    comptime CUDNN_TYPE_NORM_FWD_PHASE = Self(28)
    comptime CUDNN_TYPE_RNG_DISTRIBUTION = Self(29)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_TYPE_HANDLE:
            return writer.write_string("CUDNN_TYPE_HANDLE")
        if self is Self.CUDNN_TYPE_DATA_TYPE:
            return writer.write_string("CUDNN_TYPE_DATA_TYPE")
        if self is Self.CUDNN_TYPE_BOOLEAN:
            return writer.write_string("CUDNN_TYPE_BOOLEAN")
        if self is Self.CUDNN_TYPE_INT64:
            return writer.write_string("CUDNN_TYPE_INT64")
        if self is Self.CUDNN_TYPE_FLOAT:
            return writer.write_string("CUDNN_TYPE_FLOAT")
        if self is Self.CUDNN_TYPE_DOUBLE:
            return writer.write_string("CUDNN_TYPE_DOUBLE")
        if self is Self.CUDNN_TYPE_VOID_PTR:
            return writer.write_string("CUDNN_TYPE_VOID_PTR")
        if self is Self.CUDNN_TYPE_CONVOLUTION_MODE:
            return writer.write_string("CUDNN_TYPE_CONVOLUTION_MODE")
        if self is Self.CUDNN_TYPE_HEUR_MODE:
            return writer.write_string("CUDNN_TYPE_HEUR_MODE")
        if self is Self.CUDNN_TYPE_KNOB_TYPE:
            return writer.write_string("CUDNN_TYPE_KNOB_TYPE")
        if self is Self.CUDNN_TYPE_NAN_PROPOGATION:
            return writer.write_string("CUDNN_TYPE_NAN_PROPOGATION")
        if self is Self.CUDNN_TYPE_NUMERICAL_NOTE:
            return writer.write_string("CUDNN_TYPE_NUMERICAL_NOTE")
        if self is Self.CUDNN_TYPE_LAYOUT_TYPE:
            return writer.write_string("CUDNN_TYPE_LAYOUT_TYPE")
        if self is Self.CUDNN_TYPE_ATTRIB_NAME:
            return writer.write_string("CUDNN_TYPE_ATTRIB_NAME")
        if self is Self.CUDNN_TYPE_POINTWISE_MODE:
            return writer.write_string("CUDNN_TYPE_POINTWISE_MODE")
        if self is Self.CUDNN_TYPE_BACKEND_DESCRIPTOR:
            return writer.write_string("CUDNN_TYPE_BACKEND_DESCRIPTOR")
        if self is Self.CUDNN_TYPE_GENSTATS_MODE:
            return writer.write_string("CUDNN_TYPE_GENSTATS_MODE")
        if self is Self.CUDNN_TYPE_BN_FINALIZE_STATS_MODE:
            return writer.write_string("CUDNN_TYPE_BN_FINALIZE_STATS_MODE")
        if self is Self.CUDNN_TYPE_REDUCTION_OPERATOR_TYPE:
            return writer.write_string("CUDNN_TYPE_REDUCTION_OPERATOR_TYPE")
        if self is Self.CUDNN_TYPE_BEHAVIOR_NOTE:
            return writer.write_string("CUDNN_TYPE_BEHAVIOR_NOTE")
        if self is Self.CUDNN_TYPE_TENSOR_REORDERING_MODE:
            return writer.write_string("CUDNN_TYPE_TENSOR_REORDERING_MODE")
        if self is Self.CUDNN_TYPE_RESAMPLE_MODE:
            return writer.write_string("CUDNN_TYPE_RESAMPLE_MODE")
        if self is Self.CUDNN_TYPE_PADDING_MODE:
            return writer.write_string("CUDNN_TYPE_PADDING_MODE")
        if self is Self.CUDNN_TYPE_INT32:
            return writer.write_string("CUDNN_TYPE_INT32")
        if self is Self.CUDNN_TYPE_CHAR:
            return writer.write_string("CUDNN_TYPE_CHAR")
        if self is Self.CUDNN_TYPE_SIGNAL_MODE:
            return writer.write_string("CUDNN_TYPE_SIGNAL_MODE")
        if self is Self.CUDNN_TYPE_FRACTION:
            return writer.write_string("CUDNN_TYPE_FRACTION")
        if self is Self.CUDNN_TYPE_NORM_MODE:
            return writer.write_string("CUDNN_TYPE_NORM_MODE")
        if self is Self.CUDNN_TYPE_NORM_FWD_PHASE:
            return writer.write_string("CUDNN_TYPE_NORM_FWD_PHASE")
        if self is Self.CUDNN_TYPE_RNG_DISTRIBUTION:
            return writer.write_string("CUDNN_TYPE_RNG_DISTRIBUTION")
        abort("invalid cudnnBackendAttributeType_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendAttributeType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnRngDistribution_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_RNG_DISTRIBUTION_BERNOULLI = Self(0)
    comptime CUDNN_RNG_DISTRIBUTION_UNIFORM = Self(1)
    comptime CUDNN_RNG_DISTRIBUTION_NORMAL = Self(2)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_RNG_DISTRIBUTION_BERNOULLI:
            return writer.write_string("CUDNN_RNG_DISTRIBUTION_BERNOULLI")
        if self is Self.CUDNN_RNG_DISTRIBUTION_UNIFORM:
            return writer.write_string("CUDNN_RNG_DISTRIBUTION_UNIFORM")
        if self is Self.CUDNN_RNG_DISTRIBUTION_NORMAL:
            return writer.write_string("CUDNN_RNG_DISTRIBUTION_NORMAL")
        abort("invalid cudnnRngDistribution_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnRngDistribution_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnBackendFinalize(descriptor: OpaquePointer) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendFinalize", def(type_of(descriptor)) thin -> cudnnStatus_t
    ]()(descriptor)


@fieldwise_init
struct cudnnBackendTensorReordering_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_TENSOR_REORDERING_NONE = Self(0)
    comptime CUDNN_TENSOR_REORDERING_INT8x32 = Self(1)
    comptime CUDNN_TENSOR_REORDERING_F16x16 = Self(2)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_TENSOR_REORDERING_NONE:
            return writer.write_string("CUDNN_TENSOR_REORDERING_NONE")
        if self is Self.CUDNN_TENSOR_REORDERING_INT8x32:
            return writer.write_string("CUDNN_TENSOR_REORDERING_INT8x32")
        if self is Self.CUDNN_TENSOR_REORDERING_F16x16:
            return writer.write_string("CUDNN_TENSOR_REORDERING_F16x16")
        abort("invalid cudnnBackendTensorReordering_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendTensorReordering_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnBackendAttributeName_t(
    Equatable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_ATTR_POINTWISE_MODE = Self(0)
    comptime CUDNN_ATTR_POINTWISE_MATH_PREC = Self(1)
    comptime CUDNN_ATTR_POINTWISE_NAN_PROPAGATION = Self(2)
    comptime CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP = Self(3)
    comptime CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP = Self(4)
    comptime CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = Self(5)
    comptime CUDNN_ATTR_POINTWISE_ELU_ALPHA = Self(6)
    comptime CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA = Self(7)
    comptime CUDNN_ATTR_POINTWISE_SWISH_BETA = Self(8)
    comptime CUDNN_ATTR_POINTWISE_AXIS = Self(9)
    comptime CUDNN_ATTR_CONVOLUTION_COMP_TYPE = Self(10)
    comptime CUDNN_ATTR_CONVOLUTION_CONV_MODE = Self(11)
    comptime CUDNN_ATTR_CONVOLUTION_DILATIONS = Self(12)
    comptime CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = Self(13)
    comptime CUDNN_ATTR_CONVOLUTION_POST_PADDINGS = Self(14)
    comptime CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS = Self(15)
    comptime CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS = Self(16)
    comptime CUDNN_ATTR_ENGINEHEUR_MODE = Self(17)
    comptime CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = Self(18)
    comptime CUDNN_ATTR_ENGINEHEUR_RESULTS = Self(19)
    comptime CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET = Self(20)
    comptime CUDNN_ATTR_ENGINECFG_ENGINE = Self(21)
    comptime CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO = Self(22)
    comptime CUDNN_ATTR_ENGINECFG_KNOB_CHOICES = Self(23)
    comptime CUDNN_ATTR_EXECUTION_PLAN_HANDLE = Self(24)
    comptime CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = Self(25)
    comptime CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = Self(26)
    comptime CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = Self(27)
    comptime CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = Self(28)
    comptime CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION = Self(29)
    comptime CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID = Self(30)
    comptime CUDNN_ATTR_INTERMEDIATE_INFO_SIZE = Self(31)
    comptime CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS = Self(32)
    comptime CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = Self(33)
    comptime CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE = Self(34)
    comptime CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE = Self(35)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = Self(36)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA = Self(37)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = Self(38)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W = Self(39)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X = Self(40)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y = Self(41)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = Self(42)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA = Self(43)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC = Self(44)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = Self(45)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = Self(46)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = Self(47)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA = Self(48)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA = Self(49)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = Self(50)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = Self(51)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X = Self(52)
    comptime CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = Self(53)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = Self(54)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_XDESC = Self(55)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_BDESC = Self(56)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_YDESC = Self(57)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 = Self(58)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 = Self(59)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_DXDESC = Self(60)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_DYDESC = Self(61)
    comptime CUDNN_ATTR_OPERATION_POINTWISE_TDESC = Self(62)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_MODE = Self(63)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC = Self(64)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_XDESC = Self(65)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC = Self(66)
    comptime CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC = Self(67)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE = Self(68)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC = Self(69)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC = Self(70)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC = Self(71)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC = Self(72)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC = Self(73)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC = Self(74)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC = Self(75)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = Self(
        76
    )
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC = Self(
        77
    )
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC = Self(78)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC = Self(79)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC = Self(80)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC = Self(81)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC = Self(82)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC = Self(83)
    comptime CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC = Self(84)
    comptime CUDNN_ATTR_OPERATIONGRAPH_HANDLE = Self(85)
    comptime CUDNN_ATTR_OPERATIONGRAPH_OPS = Self(86)
    comptime CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = Self(87)
    comptime CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT = Self(88)
    comptime CUDNN_ATTR_TENSOR_DATA_TYPE = Self(89)
    comptime CUDNN_ATTR_TENSOR_DIMENSIONS = Self(90)
    comptime CUDNN_ATTR_TENSOR_STRIDES = Self(91)
    comptime CUDNN_ATTR_TENSOR_VECTOR_COUNT = Self(92)
    comptime CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION = Self(93)
    comptime CUDNN_ATTR_TENSOR_UNIQUE_ID = Self(94)
    comptime CUDNN_ATTR_TENSOR_IS_VIRTUAL = Self(95)
    comptime CUDNN_ATTR_TENSOR_IS_BY_VALUE = Self(96)
    comptime CUDNN_ATTR_TENSOR_REORDERING_MODE = Self(97)
    comptime CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC = Self(98)
    comptime CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = Self(99)
    comptime CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = Self(100)
    comptime CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES = Self(101)
    comptime CUDNN_ATTR_VARIANT_PACK_WORKSPACE = Self(102)
    comptime CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID = Self(103)
    comptime CUDNN_ATTR_LAYOUT_INFO_TYPES = Self(104)
    comptime CUDNN_ATTR_KNOB_INFO_TYPE = Self(105)
    comptime CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE = Self(106)
    comptime CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE = Self(107)
    comptime CUDNN_ATTR_KNOB_INFO_STRIDE = Self(108)
    comptime CUDNN_ATTR_ENGINE_OPERATION_GRAPH = Self(109)
    comptime CUDNN_ATTR_ENGINE_GLOBAL_INDEX = Self(110)
    comptime CUDNN_ATTR_ENGINE_KNOB_INFO = Self(111)
    comptime CUDNN_ATTR_ENGINE_NUMERICAL_NOTE = Self(112)
    comptime CUDNN_ATTR_ENGINE_LAYOUT_INFO = Self(113)
    comptime CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE = Self(114)
    comptime CUDNN_ATTR_ENGINE_SM_COUNT_TARGET = Self(115)
    comptime CUDNN_ATTR_MATMUL_COMP_TYPE = Self(116)
    comptime CUDNN_ATTR_MATMUL_PADDING_VALUE = Self(117)
    comptime CUDNN_ATTR_OPERATION_MATMUL_ADESC = Self(118)
    comptime CUDNN_ATTR_OPERATION_MATMUL_BDESC = Self(119)
    comptime CUDNN_ATTR_OPERATION_MATMUL_CDESC = Self(120)
    comptime CUDNN_ATTR_OPERATION_MATMUL_DESC = Self(121)
    comptime CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT = Self(
        122
    )
    comptime CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC = Self(123)
    comptime CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC = Self(124)
    comptime CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC = Self(125)
    comptime CUDNN_ATTR_REDUCTION_OPERATOR = Self(126)
    comptime CUDNN_ATTR_REDUCTION_COMP_TYPE = Self(127)
    comptime CUDNN_ATTR_OPERATION_REDUCTION_XDESC = Self(128)
    comptime CUDNN_ATTR_OPERATION_REDUCTION_YDESC = Self(129)
    comptime CUDNN_ATTR_OPERATION_REDUCTION_DESC = Self(130)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC = Self(131)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC = Self(132)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC = Self(133)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC = Self(134)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC = Self(135)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC = Self(136)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC = Self(137)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC = Self(138)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = Self(139)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC = Self(140)
    comptime CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS = Self(141)
    comptime CUDNN_ATTR_RESAMPLE_MODE = Self(142)
    comptime CUDNN_ATTR_RESAMPLE_COMP_TYPE = Self(143)
    comptime CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS = Self(144)
    comptime CUDNN_ATTR_RESAMPLE_POST_PADDINGS = Self(145)
    comptime CUDNN_ATTR_RESAMPLE_PRE_PADDINGS = Self(146)
    comptime CUDNN_ATTR_RESAMPLE_STRIDES = Self(147)
    comptime CUDNN_ATTR_RESAMPLE_WINDOW_DIMS = Self(148)
    comptime CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION = Self(149)
    comptime CUDNN_ATTR_RESAMPLE_PADDING_MODE = Self(150)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC = Self(151)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC = Self(152)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = Self(153)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA = Self(154)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA = Self(155)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC = Self(156)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC = Self(157)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC = Self(158)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = Self(159)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA = Self(160)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA = Self(161)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC = Self(162)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC = Self(163)
    comptime CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC = Self(164)
    comptime CUDNN_ATTR_OPERATION_CONCAT_AXIS = Self(165)
    comptime CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS = Self(166)
    comptime CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX = Self(167)
    comptime CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC = Self(168)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_MODE = Self(169)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC = Self(170)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_VALUE = Self(171)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_XDESC = Self(172)
    comptime CUDNN_ATTR_OPERATION_SIGNAL_YDESC = Self(173)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_MODE = Self(174)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_PHASE = Self(175)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_XDESC = Self(176)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC = Self(177)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC = Self(178)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC = Self(179)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC = Self(180)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC = Self(181)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC = Self(182)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC = Self(183)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC = Self(184)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = Self(185)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC = Self(186)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_YDESC = Self(187)
    comptime CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS = Self(188)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_MODE = Self(189)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_XDESC = Self(190)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC = Self(191)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = Self(192)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC = Self(193)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC = Self(194)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC = Self(195)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC = Self(196)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC = Self(197)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC = Self(198)
    comptime CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS = Self(199)
    comptime CUDNN_ATTR_OPERATION_RESHAPE_XDESC = Self(200)
    comptime CUDNN_ATTR_OPERATION_RESHAPE_YDESC = Self(201)
    comptime CUDNN_ATTR_RNG_DISTRIBUTION = Self(202)
    comptime CUDNN_ATTR_RNG_NORMAL_DIST_MEAN = Self(203)
    comptime CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = Self(204)
    comptime CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM = Self(205)
    comptime CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM = Self(206)
    comptime CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY = Self(207)
    comptime CUDNN_ATTR_OPERATION_RNG_YDESC = Self(208)
    comptime CUDNN_ATTR_OPERATION_RNG_SEED = Self(209)
    comptime CUDNN_ATTR_OPERATION_RNG_DESC = Self(210)
    comptime CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC = Self(211)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_ATTR_POINTWISE_MODE:
            return writer.write_string("CUDNN_ATTR_POINTWISE_MODE")
        if self is Self.CUDNN_ATTR_POINTWISE_MATH_PREC:
            return writer.write_string("CUDNN_ATTR_POINTWISE_MATH_PREC")
        if self is Self.CUDNN_ATTR_POINTWISE_NAN_PROPAGATION:
            return writer.write_string("CUDNN_ATTR_POINTWISE_NAN_PROPAGATION")
        if self is Self.CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP:
            return writer.write_string("CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP")
        if self is Self.CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP:
            return writer.write_string("CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP")
        if self is Self.CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
            return writer.write_string(
                "CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE"
            )
        if self is Self.CUDNN_ATTR_POINTWISE_ELU_ALPHA:
            return writer.write_string("CUDNN_ATTR_POINTWISE_ELU_ALPHA")
        if self is Self.CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA:
            return writer.write_string("CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA")
        if self is Self.CUDNN_ATTR_POINTWISE_SWISH_BETA:
            return writer.write_string("CUDNN_ATTR_POINTWISE_SWISH_BETA")
        if self is Self.CUDNN_ATTR_POINTWISE_AXIS:
            return writer.write_string("CUDNN_ATTR_POINTWISE_AXIS")
        if self is Self.CUDNN_ATTR_CONVOLUTION_COMP_TYPE:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_COMP_TYPE")
        if self is Self.CUDNN_ATTR_CONVOLUTION_CONV_MODE:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_CONV_MODE")
        if self is Self.CUDNN_ATTR_CONVOLUTION_DILATIONS:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_DILATIONS")
        if self is Self.CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES")
        if self is Self.CUDNN_ATTR_CONVOLUTION_POST_PADDINGS:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_POST_PADDINGS")
        if self is Self.CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS")
        if self is Self.CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS:
            return writer.write_string("CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS")
        if self is Self.CUDNN_ATTR_ENGINEHEUR_MODE:
            return writer.write_string("CUDNN_ATTR_ENGINEHEUR_MODE")
        if self is Self.CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
            return writer.write_string("CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH")
        if self is Self.CUDNN_ATTR_ENGINEHEUR_RESULTS:
            return writer.write_string("CUDNN_ATTR_ENGINEHEUR_RESULTS")
        if self is Self.CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET:
            return writer.write_string("CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET")
        if self is Self.CUDNN_ATTR_ENGINECFG_ENGINE:
            return writer.write_string("CUDNN_ATTR_ENGINECFG_ENGINE")
        if self is Self.CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
            return writer.write_string("CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO")
        if self is Self.CUDNN_ATTR_ENGINECFG_KNOB_CHOICES:
            return writer.write_string("CUDNN_ATTR_ENGINECFG_KNOB_CHOICES")
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_HANDLE:
            return writer.write_string("CUDNN_ATTR_EXECUTION_PLAN_HANDLE")
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
            return writer.write_string(
                "CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
            return writer.write_string(
                "CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS:
            return writer.write(
                "CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS:
            return writer.write(
                "CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS"
            )
        if self is Self.CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION:
            return writer.write_string(
                "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION"
            )
        if self is Self.CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID:
            return writer.write_string("CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID")
        if self is Self.CUDNN_ATTR_INTERMEDIATE_INFO_SIZE:
            return writer.write_string("CUDNN_ATTR_INTERMEDIATE_INFO_SIZE")
        if self is Self.CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS:
            return writer.write(
                "CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS"
            )
        if self is Self.CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES:
            return writer.write(
                "CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES"
            )
        if self is Self.CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE:
            return writer.write_string("CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE")
        if self is Self.CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE:
            return writer.write_string("CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE")
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY:
            return writer.write(
                "CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY"
            )
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR"
            )
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_BDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_BDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_DXDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_DXDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_DYDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_DYDESC")
        if self is Self.CUDNN_ATTR_OPERATION_POINTWISE_TDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_POINTWISE_TDESC")
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_MODE:
            return writer.write_string("CUDNN_ATTR_OPERATION_GENSTATS_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_GENSTATS_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC")
        if self is Self.CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC
        ):
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC
        ):
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC"
            )
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC
        ):
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATIONGRAPH_HANDLE:
            return writer.write_string("CUDNN_ATTR_OPERATIONGRAPH_HANDLE")
        if self is Self.CUDNN_ATTR_OPERATIONGRAPH_OPS:
            return writer.write_string("CUDNN_ATTR_OPERATIONGRAPH_OPS")
        if self is Self.CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT:
            return writer.write_string(
                "CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT"
            )
        if self is Self.CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT:
            return writer.write_string("CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT")
        if self is Self.CUDNN_ATTR_TENSOR_DATA_TYPE:
            return writer.write_string("CUDNN_ATTR_TENSOR_DATA_TYPE")
        if self is Self.CUDNN_ATTR_TENSOR_DIMENSIONS:
            return writer.write_string("CUDNN_ATTR_TENSOR_DIMENSIONS")
        if self is Self.CUDNN_ATTR_TENSOR_STRIDES:
            return writer.write_string("CUDNN_ATTR_TENSOR_STRIDES")
        if self is Self.CUDNN_ATTR_TENSOR_VECTOR_COUNT:
            return writer.write_string("CUDNN_ATTR_TENSOR_VECTOR_COUNT")
        if self is Self.CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION:
            return writer.write_string("CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION")
        if self is Self.CUDNN_ATTR_TENSOR_UNIQUE_ID:
            return writer.write_string("CUDNN_ATTR_TENSOR_UNIQUE_ID")
        if self is Self.CUDNN_ATTR_TENSOR_IS_VIRTUAL:
            return writer.write_string("CUDNN_ATTR_TENSOR_IS_VIRTUAL")
        if self is Self.CUDNN_ATTR_TENSOR_IS_BY_VALUE:
            return writer.write_string("CUDNN_ATTR_TENSOR_IS_BY_VALUE")
        if self is Self.CUDNN_ATTR_TENSOR_REORDERING_MODE:
            return writer.write_string("CUDNN_ATTR_TENSOR_REORDERING_MODE")
        if self is Self.CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC:
            return writer.write_string("CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC")
        if self is Self.CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
            return writer.write_string("CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS")
        if self is Self.CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
            return writer.write_string("CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS")
        if self is Self.CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES:
            return writer.write_string("CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES")
        if self is Self.CUDNN_ATTR_VARIANT_PACK_WORKSPACE:
            return writer.write_string("CUDNN_ATTR_VARIANT_PACK_WORKSPACE")
        if self is Self.CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID:
            return writer.write_string("CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID")
        if self is Self.CUDNN_ATTR_LAYOUT_INFO_TYPES:
            return writer.write_string("CUDNN_ATTR_LAYOUT_INFO_TYPES")
        if self is Self.CUDNN_ATTR_KNOB_INFO_TYPE:
            return writer.write_string("CUDNN_ATTR_KNOB_INFO_TYPE")
        if self is Self.CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE:
            return writer.write_string("CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE")
        if self is Self.CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE:
            return writer.write_string("CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE")
        if self is Self.CUDNN_ATTR_KNOB_INFO_STRIDE:
            return writer.write_string("CUDNN_ATTR_KNOB_INFO_STRIDE")
        if self is Self.CUDNN_ATTR_ENGINE_OPERATION_GRAPH:
            return writer.write_string("CUDNN_ATTR_ENGINE_OPERATION_GRAPH")
        if self is Self.CUDNN_ATTR_ENGINE_GLOBAL_INDEX:
            return writer.write_string("CUDNN_ATTR_ENGINE_GLOBAL_INDEX")
        if self is Self.CUDNN_ATTR_ENGINE_KNOB_INFO:
            return writer.write_string("CUDNN_ATTR_ENGINE_KNOB_INFO")
        if self is Self.CUDNN_ATTR_ENGINE_NUMERICAL_NOTE:
            return writer.write_string("CUDNN_ATTR_ENGINE_NUMERICAL_NOTE")
        if self is Self.CUDNN_ATTR_ENGINE_LAYOUT_INFO:
            return writer.write_string("CUDNN_ATTR_ENGINE_LAYOUT_INFO")
        if self is Self.CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
            return writer.write_string("CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE")
        if self is Self.CUDNN_ATTR_ENGINE_SM_COUNT_TARGET:
            return writer.write_string("CUDNN_ATTR_ENGINE_SM_COUNT_TARGET")
        if self is Self.CUDNN_ATTR_MATMUL_COMP_TYPE:
            return writer.write_string("CUDNN_ATTR_MATMUL_COMP_TYPE")
        if self is Self.CUDNN_ATTR_MATMUL_PADDING_VALUE:
            return writer.write_string("CUDNN_ATTR_MATMUL_PADDING_VALUE")
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_ADESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_MATMUL_ADESC")
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_BDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_MATMUL_BDESC")
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_CDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_MATMUL_CDESC")
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_MATMUL_DESC")
        if (
            self
            is Self.CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT
        ):
            return writer.write(
                "CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC"
            )
        if self is Self.CUDNN_ATTR_REDUCTION_OPERATOR:
            return writer.write_string("CUDNN_ATTR_REDUCTION_OPERATOR")
        if self is Self.CUDNN_ATTR_REDUCTION_COMP_TYPE:
            return writer.write_string("CUDNN_ATTR_REDUCTION_COMP_TYPE")
        if self is Self.CUDNN_ATTR_OPERATION_REDUCTION_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_REDUCTION_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_REDUCTION_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_REDUCTION_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_REDUCTION_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_REDUCTION_DESC")
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS"
            )
        if self is Self.CUDNN_ATTR_RESAMPLE_MODE:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_MODE")
        if self is Self.CUDNN_ATTR_RESAMPLE_COMP_TYPE:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_COMP_TYPE")
        if self is Self.CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS")
        if self is Self.CUDNN_ATTR_RESAMPLE_POST_PADDINGS:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_POST_PADDINGS")
        if self is Self.CUDNN_ATTR_RESAMPLE_PRE_PADDINGS:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_PRE_PADDINGS")
        if self is Self.CUDNN_ATTR_RESAMPLE_STRIDES:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_STRIDES")
        if self is Self.CUDNN_ATTR_RESAMPLE_WINDOW_DIMS:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_WINDOW_DIMS")
        if self is Self.CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION")
        if self is Self.CUDNN_ATTR_RESAMPLE_PADDING_MODE:
            return writer.write_string("CUDNN_ATTR_RESAMPLE_PADDING_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC")
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONCAT_AXIS:
            return writer.write_string("CUDNN_ATTR_OPERATION_CONCAT_AXIS")
        if self is Self.CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX"
            )
        if self is Self.CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_MODE:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_VALUE:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_VALUE")
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_SIGNAL_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_SIGNAL_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_MODE:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_FWD_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_PHASE:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_FWD_PHASE")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_FWD_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_FWD_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_MODE:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_BWD_MODE")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_BWD_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC:
            return writer.write(
                "CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC"
            )
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC")
        if self is Self.CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS:
            return writer.write_string(
                "CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RESHAPE_XDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESHAPE_XDESC")
        if self is Self.CUDNN_ATTR_OPERATION_RESHAPE_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RESHAPE_YDESC")
        if self is Self.CUDNN_ATTR_RNG_DISTRIBUTION:
            return writer.write_string("CUDNN_ATTR_RNG_DISTRIBUTION")
        if self is Self.CUDNN_ATTR_RNG_NORMAL_DIST_MEAN:
            return writer.write_string("CUDNN_ATTR_RNG_NORMAL_DIST_MEAN")
        if self is Self.CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION:
            return writer.write_string(
                "CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION"
            )
        if self is Self.CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM:
            return writer.write_string("CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM")
        if self is Self.CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM:
            return writer.write_string("CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM")
        if self is Self.CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY:
            return writer.write_string(
                "CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY"
            )
        if self is Self.CUDNN_ATTR_OPERATION_RNG_YDESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RNG_YDESC")
        if self is Self.CUDNN_ATTR_OPERATION_RNG_SEED:
            return writer.write_string("CUDNN_ATTR_OPERATION_RNG_SEED")
        if self is Self.CUDNN_ATTR_OPERATION_RNG_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RNG_DESC")
        if self is Self.CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC:
            return writer.write_string("CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC")
        abort("invalid cudnnBackendAttributeName_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendAttributeName_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnBackendNormMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_LAYER_NORM = Self(0)
    comptime CUDNN_INSTANCE_NORM = Self(1)
    comptime CUDNN_BATCH_NORM = Self(2)
    comptime CUDNN_GROUP_NORM = Self(3)
    comptime CUDNN_RMS_NORM = Self(4)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_LAYER_NORM:
            return writer.write_string("CUDNN_LAYER_NORM")
        if self is Self.CUDNN_INSTANCE_NORM:
            return writer.write_string("CUDNN_INSTANCE_NORM")
        if self is Self.CUDNN_BATCH_NORM:
            return writer.write_string("CUDNN_BATCH_NORM")
        if self is Self.CUDNN_GROUP_NORM:
            return writer.write_string("CUDNN_GROUP_NORM")
        if self is Self.CUDNN_RMS_NORM:
            return writer.write_string("CUDNN_RMS_NORM")
        abort("invalid cudnnBackendNormMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBackendNormMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnSignalMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_SIGNAL_SET = Self(0)
    comptime CUDNN_SIGNAL_WAIT = Self(1)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_SIGNAL_SET:
            return writer.write_string("CUDNN_SIGNAL_SET")
        if self is Self.CUDNN_SIGNAL_WAIT:
            return writer.write_string("CUDNN_SIGNAL_WAIT")
        abort("invalid cudnnSignalMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnSignalMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


comptime cudnnBackendDescriptor_t = OpaquePointer[AnyOrigin[mut=True]]


@fieldwise_init
struct cudnnBnFinalizeStatsMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_BN_FINALIZE_STATISTICS_TRAINING = Self(0)
    comptime CUDNN_BN_FINALIZE_STATISTICS_INFERENCE = Self(1)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_BN_FINALIZE_STATISTICS_TRAINING:
            return writer.write_string("CUDNN_BN_FINALIZE_STATISTICS_TRAINING")
        if self is Self.CUDNN_BN_FINALIZE_STATISTICS_INFERENCE:
            return writer.write_string("CUDNN_BN_FINALIZE_STATISTICS_INFERENCE")
        abort("invalid cudnnBnFinalizeStatsMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnBnFinalizeStatsMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct cudnnGenStatsMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_GENSTATS_SUM_SQSUM = Self(0)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_GENSTATS_SUM_SQSUM:
            return writer.write_string("CUDNN_GENSTATS_SUM_SQSUM")
        abort("invalid cudnnGenStatsMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnGenStatsMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnBackendDestroyDescriptor(
    descriptor: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendDestroyDescriptor",
        def(type_of(descriptor)) thin -> cudnnStatus_t,
    ]()(descriptor)


def cudnnBackendExecute(
    handle: UnsafePointer[cudnnContext, _],
    execution_plan: OpaquePointer,
    variant_pack: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendExecute",
        def(
            type_of(handle),
            type_of(execution_plan),
            type_of(variant_pack),
        ) thin -> cudnnStatus_t,
    ]()(handle, execution_plan, variant_pack)


@fieldwise_init
struct cudnnResampleMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_RESAMPLE_NEAREST = Self(0)
    comptime CUDNN_RESAMPLE_BILINEAR = Self(1)
    comptime CUDNN_RESAMPLE_AVGPOOL = Self(2)
    comptime CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING = Self(3)
    comptime CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING = Self(4)
    comptime CUDNN_RESAMPLE_MAXPOOL = Self(5)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_RESAMPLE_NEAREST:
            return writer.write_string("CUDNN_RESAMPLE_NEAREST")
        if self is Self.CUDNN_RESAMPLE_BILINEAR:
            return writer.write_string("CUDNN_RESAMPLE_BILINEAR")
        if self is Self.CUDNN_RESAMPLE_AVGPOOL:
            return writer.write_string("CUDNN_RESAMPLE_AVGPOOL")
        if self is Self.CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING:
            return writer.write_string("CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING")
        if self is Self.CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING:
            return writer.write_string("CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING")
        if self is Self.CUDNN_RESAMPLE_MAXPOOL:
            return writer.write_string("CUDNN_RESAMPLE_MAXPOOL")
        abort("invalid cudnnResampleMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnResampleMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


comptime cudnnFraction_t = cudnnFractionStruct


def cudnnBackendGetAttribute(
    descriptor: OpaquePointer,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    requested_element_count: Int64,
    element_count: UnsafePointer[Int64, _],
    array_of_elements: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnBackendGetAttribute",
        def(
            type_of(descriptor),
            type_of(attribute_name),
            type_of(attribute_type),
            type_of(requested_element_count),
            type_of(element_count),
            type_of(array_of_elements),
        ) thin -> cudnnStatus_t,
    ]()(
        descriptor,
        attribute_name,
        attribute_type,
        requested_element_count,
        element_count,
        array_of_elements,
    )


@fieldwise_init
struct cudnnPaddingMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_ZERO_PAD = Self(0)
    comptime CUDNN_NEG_INF_PAD = Self(1)
    comptime CUDNN_EDGE_VAL_PAD = Self(2)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_ZERO_PAD:
            return writer.write_string("CUDNN_ZERO_PAD")
        if self is Self.CUDNN_NEG_INF_PAD:
            return writer.write_string("CUDNN_NEG_INF_PAD")
        if self is Self.CUDNN_EDGE_VAL_PAD:
            return writer.write_string("CUDNN_EDGE_VAL_PAD")
        abort("invalid cudnnPaddingMode_t entry")

    @inline(.never)
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnPaddingMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)
