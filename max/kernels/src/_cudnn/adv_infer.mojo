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

from .infer import (
    cudnnContext,
    cudnnDataType_t,
    cudnnFilterStruct,
    cudnnMathType_t,
    cudnnNanPropagation_t,
    cudnnRNNAlgo_t,
    cudnnStatus_t,
    AnyOpaquePointer,
    DoubleNestedPointer,
)

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

comptime CUDA_CUDNN_ADV_INFER_LIBRARY_PATHS: List[Path] = [
    "libcudnn_adv_infer.so",
    "libcudnn_adv_infer.so.9",
    "libcudnn_adv_infer.so.8",
    "/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.9",
    "/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8",
]

comptime CUDA_CUDNN_ADV_INFER_LIBRARY = _Global[
    "CUDA_CUDNN_ADV_INFER_LIBRARY", _init_dylib
]


def _init_dylib() -> OwnedDLHandle:
    return _find_dylib["CUDA cuDNN Adv Infer"](
        materialize[CUDA_CUDNN_ADV_INFER_LIBRARY_PATHS]()
    )


@always_inline
def _get_dylib_function[
    func_name: StaticString, result_type: TrivialRegisterPassable
]() raises -> result_type:
    return _ffi_get_dylib_function[
        CUDA_CUDNN_ADV_INFER_LIBRARY(),
        func_name,
        result_type,
    ]()


# ===-----------------------------------------------------------------------===#
# Bindings
# ===-----------------------------------------------------------------------===#

comptime cudnnRNNStruct = AnyOpaquePointer
comptime cudnnDropoutStruct = AnyOpaquePointer
comptime cudnnAlgorithmStruct = AnyOpaquePointer
comptime cudnnRNNDataStruct = AnyOpaquePointer
comptime cudnnAttnStruct = AnyOpaquePointer
comptime cudnnTensorStruct = AnyOpaquePointer
comptime cudnnSeqDataStruct = AnyOpaquePointer
comptime cudnnPersistentRNNPlan = NoneType


@fieldwise_init
struct cudnnRNNInputMode_t(TrivialRegisterPassable):
    var _value: Int32

    comptime LINEAR_INPUT = Self(0)
    """Adjustable weight matrix in first layer input GEMM."""
    comptime SKIP_INPUT = Self(1)
    """Fixed identity matrix in the first layer input GEMM."""


@fieldwise_init
struct cudnnDirectionMode_t(TrivialRegisterPassable):
    var _value: Int32

    comptime UNIDIRECTIONAL = Self(0)
    """Single direction network."""
    comptime BIDIRECTIONAL = Self(1)
    """Output concatenation at each layer."""


@fieldwise_init
struct cudnnRNNClipMode_t(TrivialRegisterPassable):
    var _value: Int32

    comptime NONE = Self(0)
    """Disables LSTM cell clipping."""
    comptime MINMAX = Self(1)
    """Enables LSTM cell clipping."""


@fieldwise_init
struct cudnnRNNMode_t(TrivialRegisterPassable):
    var _value: Int32
    comptime RNN_RELU = Self(0)
    """Basic RNN cell type with ReLu activation."""
    comptime RNN_TANH = Self(1)
    """Basic RNN cell type with tanh activation."""
    comptime LTSM = Self(2)
    """LSTM with optional recurrent projection and clipping."""
    comptime GRU = Self(3)
    """Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1)."""


@fieldwise_init
struct cudnnMultiHeadAttnWeightKind_t(TrivialRegisterPassable):
    var _value: Int32

    comptime ATTN_Q_WEIGHTS = Self(0)
    "Input projection weights for 'queries'."

    comptime ATTN_K_WEIGHTS = Self(1)
    "Input projection weights for 'keys'."

    comptime ATTN_V_WEIGHTS = Self(2)
    "Input projection weights for 'values'."

    comptime ATTN_O_WEIGHTS = Self(3)
    "Output projection weights."

    comptime ATTN_Q_BIASES = Self(4)
    "Input projection bias for 'queries'."

    comptime ATTN_K_BIASES = Self(5)
    "Input projection bias for 'keys'."

    comptime ATTN_V_BIASES = Self(6)
    "Input projection bias for 'values'."

    comptime ATTN_O_BIASES = Self(6)
    "Output projection bias."


@fieldwise_init
struct cudnnRNNBiasMode_t(TrivialRegisterPassable):
    var _value: Int32

    comptime NO_BIAS = Self(0)
    """Rnn cell formulas do not use biases."""
    comptime SINGLE_INP_BIAS = Self(1)
    """Rnn cell formulas use one input bias in input GEMM."""
    comptime DOUBLE_BIAS = Self(2)
    """Default, rnn cell formulas use two bias vectors."""
    comptime SINGLE_REC_BIAS = Self(3)
    """Rrnn cell formulas use one recurrent bias in recurrent GEMMs."""


@fieldwise_init
struct cudnnRNNDataLayout_t(TrivialRegisterPassable):
    var _value: Int32
    comptime SEQ_MAJOR_UNPACKED = Self(0)
    """Padded, outer stride from one time-step to the next."""
    comptime SEQ_MAJOR_PACKED = Self(1)
    """Sequence length sorted and packed as in basic RNN api."""
    comptime BATCH_MAJOR_UNPACKED = Self(2)
    """Padded, outer stride from one batch to the next."""


def cudnnGetRNNDescriptor_v6(
    handle: UnsafePointer[cudnnContext, _],
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    hidden_size: UnsafePointer[Int16, _],
    num_layers: UnsafePointer[Int16, _],
    dropout_desc: DoubleNestedPointer[cudnnDropoutStruct],
    input_mode: UnsafePointer[cudnnRNNInputMode_t, _],
    direction: UnsafePointer[cudnnDirectionMode_t, _],
    cell_mode: UnsafePointer[cudnnRNNMode_t, _],
    algo: UnsafePointer[cudnnRNNAlgo_t, _],
    math_prec: UnsafePointer[cudnnDataType_t, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNDescriptor_v6",
        def(
            type_of(handle),
            type_of(rnn_desc),
            type_of(hidden_size),
            type_of(num_layers),
            type_of(dropout_desc),
            type_of(input_mode),
            type_of(direction),
            type_of(cell_mode),
            type_of(algo),
            type_of(math_prec),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        rnn_desc,
        hidden_size,
        num_layers,
        dropout_desc,
        input_mode,
        direction,
        cell_mode,
        algo,
        math_prec,
    )


@fieldwise_init
struct cudnnForwardMode_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_FWD_MODE_INFERENCE = Self(0)
    comptime CUDNN_FWD_MODE_TRAINING = Self(1)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_FWD_MODE_INFERENCE:
            return writer.write_string("CUDNN_FWD_MODE_INFERENCE")
        if self is Self.CUDNN_FWD_MODE_TRAINING:
            return writer.write_string("CUDNN_FWD_MODE_TRAINING")
        abort("invalid cudnnForwardMode_t entry")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnForwardMode_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnDestroyAttnDescriptor(
    attn_desc: UnsafePointer[cudnnAttnStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyAttnDescriptor",
        def(type_of(attn_desc)) thin -> cudnnStatus_t,
    ]()(attn_desc)


def cudnnGetRNNTempSpaceSizes(
    handle: UnsafePointer[cudnnContext, _],
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    fwd_mode: cudnnForwardMode_t,
    x_desc: UnsafePointer[cudnnRNNDataStruct, _],
    work_space_size: UnsafePointer[Int, _],
    reserve_space_size: UnsafePointer[Int, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNTempSpaceSizes",
        def(
            type_of(handle),
            type_of(rnn_desc),
            type_of(fwd_mode),
            type_of(x_desc),
            type_of(work_space_size),
            type_of(reserve_space_size),
        ) thin -> cudnnStatus_t,
    ]()(handle, rnn_desc, fwd_mode, x_desc, work_space_size, reserve_space_size)


def cudnnSetRNNDescriptor_v6(
    handle: UnsafePointer[cudnnContext, _],
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    hidden_size: Int16,
    num_layers: Int16,
    dropout_desc: UnsafePointer[cudnnDropoutStruct, _],
    input_mode: cudnnRNNInputMode_t,
    direction: cudnnDirectionMode_t,
    cell_mode: cudnnRNNMode_t,
    algo: cudnnRNNAlgo_t,
    math_prec: cudnnDataType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNDescriptor_v6",
        def(
            type_of(handle),
            type_of(rnn_desc),
            type_of(hidden_size),
            type_of(num_layers),
            type_of(dropout_desc),
            type_of(input_mode),
            type_of(direction),
            type_of(cell_mode),
            type_of(algo),
            type_of(math_prec),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        rnn_desc,
        hidden_size,
        num_layers,
        dropout_desc,
        input_mode,
        direction,
        cell_mode,
        algo,
        math_prec,
    )


def cudnnCreatePersistentRNNPlan(
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    minibatch: Int16,
    data_type: cudnnDataType_t,
    plan: DoubleNestedPointer[cudnnPersistentRNNPlan],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreatePersistentRNNPlan",
        def(
            type_of(rnn_desc),
            type_of(minibatch),
            type_of(data_type),
            type_of(plan),
        ) thin -> cudnnStatus_t,
    ]()(rnn_desc, minibatch, data_type, plan)


def cudnnGetSeqDataDescriptor(
    seq_data_desc: UnsafePointer[cudnnSeqDataStruct, _],
    data_type: UnsafePointer[cudnnDataType_t, _],
    nb_dims: UnsafePointer[Int16, _],
    nb_dims_requested: Int16,
    dim_a: OpaquePointer,
    axes: OpaquePointer,
    seq_length_array_size: UnsafePointer[Int, _],
    seq_length_size_requested: Int,
    seq_length_array: OpaquePointer,
    padding_fill: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetSeqDataDescriptor",
        def(
            type_of(seq_data_desc),
            type_of(data_type),
            type_of(nb_dims),
            type_of(nb_dims_requested),
            type_of(dim_a),
            type_of(axes),
            type_of(seq_length_array_size),
            type_of(seq_length_size_requested),
            type_of(seq_length_array),
            type_of(padding_fill),
        ) thin -> cudnnStatus_t,
    ]()(
        seq_data_desc,
        data_type,
        nb_dims,
        nb_dims_requested,
        dim_a,
        axes,
        seq_length_array_size,
        seq_length_size_requested,
        seq_length_array,
        padding_fill,
    )


def cudnnRNNGetClip_v8(
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    clip_mode: UnsafePointer[cudnnRNNClipMode_t, _],
    clip_nan_opt: UnsafePointer[cudnnNanPropagation_t, _],
    lclip: UnsafePointer[Float64, _],
    rclip: UnsafePointer[Float64, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnRNNGetClip_v8",
        def(
            type_of(rnn_desc),
            type_of(clip_mode),
            type_of(clip_nan_opt),
            type_of(lclip),
            type_of(rclip),
        ) thin -> cudnnStatus_t,
    ]()(rnn_desc, clip_mode, clip_nan_opt, lclip, rclip)


def cudnnSetRNNAlgorithmDescriptor(
    handle: UnsafePointer[cudnnContext, _],
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    algo_desc: UnsafePointer[cudnnAlgorithmStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNAlgorithmDescriptor",
        def(
            type_of(handle),
            type_of(rnn_desc),
            type_of(algo_desc),
        ) thin -> cudnnStatus_t,
    ]()(handle, rnn_desc, algo_desc)


def cudnnGetRNNParamsSize(
    handle: UnsafePointer[cudnnContext, _],
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    x_desc: UnsafePointer[cudnnTensorStruct, _],
    size_in_bytes: UnsafePointer[Int, _],
    data_type: cudnnDataType_t,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNParamsSize",
        def(
            type_of(handle),
            type_of(rnn_desc),
            type_of(x_desc),
            type_of(size_in_bytes),
            type_of(data_type),
        ) thin -> cudnnStatus_t,
    ]()(handle, rnn_desc, x_desc, size_in_bytes, data_type)


def cudnnSetRNNMatrixMathType(
    rnn_desc: UnsafePointer[cudnnRNNStruct, _], m_type: cudnnMathType_t
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNMatrixMathType",
        def(type_of(rnn_desc), type_of(m_type)) thin -> cudnnStatus_t,
    ]()(rnn_desc, m_type)


def cudnnGetAttnDescriptor(
    attn_desc: UnsafePointer[cudnnAttnStruct, _],
    attn_mode: UnsafePointer[Int16, _],
    n_heads: UnsafePointer[Int16, _],
    sm_scaler: UnsafePointer[Float64, _],
    data_type: UnsafePointer[cudnnDataType_t, _],
    compute_prec: UnsafePointer[cudnnDataType_t, _],
    math_type: UnsafePointer[cudnnMathType_t, _],
    attn_dropout_desc: DoubleNestedPointer[cudnnDropoutStruct],
    post_dropout_desc: DoubleNestedPointer[cudnnDropoutStruct],
    q_size: UnsafePointer[Int16, _],
    k_size: UnsafePointer[Int16, _],
    v_size: UnsafePointer[Int16, _],
    q_proj_size: UnsafePointer[Int16, _],
    k_proj_size: UnsafePointer[Int16, _],
    v_proj_size: UnsafePointer[Int16, _],
    o_proj_size: UnsafePointer[Int16, _],
    qo_max_seq_length: UnsafePointer[Int16, _],
    kv_max_seq_length: UnsafePointer[Int16, _],
    max_batch_size: UnsafePointer[Int16, _],
    max_beam_size: UnsafePointer[Int16, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetAttnDescriptor",
        def(
            type_of(attn_desc),
            type_of(attn_mode),
            type_of(n_heads),
            type_of(sm_scaler),
            type_of(data_type),
            type_of(compute_prec),
            type_of(math_type),
            type_of(attn_dropout_desc),
            type_of(post_dropout_desc),
            type_of(q_size),
            type_of(k_size),
            type_of(v_size),
            type_of(q_proj_size),
            type_of(k_proj_size),
            type_of(v_proj_size),
            type_of(o_proj_size),
            type_of(qo_max_seq_length),
            type_of(kv_max_seq_length),
            type_of(max_batch_size),
            type_of(max_beam_size),
        ) thin -> cudnnStatus_t,
    ]()(
        attn_desc,
        attn_mode,
        n_heads,
        sm_scaler,
        data_type,
        compute_prec,
        math_type,
        attn_dropout_desc,
        post_dropout_desc,
        q_size,
        k_size,
        v_size,
        q_proj_size,
        k_proj_size,
        v_proj_size,
        o_proj_size,
        qo_max_seq_length,
        kv_max_seq_length,
        max_batch_size,
        max_beam_size,
    )


def cudnnRNNSetClip(
    handle: UnsafePointer[cudnnContext, _],
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    clip_mode: cudnnRNNClipMode_t,
    clip_nan_opt: cudnnNanPropagation_t,
    lclip: Float64,
    rclip: Float64,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnRNNSetClip",
        def(
            type_of(handle),
            type_of(rnn_desc),
            type_of(clip_mode),
            type_of(clip_nan_opt),
            type_of(lclip),
            type_of(rclip),
        ) thin -> cudnnStatus_t,
    ]()(handle, rnn_desc, clip_mode, clip_nan_opt, lclip, rclip)


def cudnnGetMultiHeadAttnWeights(
    handle: UnsafePointer[cudnnContext, _],
    attn_desc: UnsafePointer[cudnnAttnStruct, _],
    w_kind: cudnnMultiHeadAttnWeightKind_t,
    weight_size_in_bytes: Int,
    weights: OpaquePointer,
    w_desc: UnsafePointer[cudnnTensorStruct, _],
    w_addr: UnsafePointer[AnyOpaquePointer, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetMultiHeadAttnWeights",
        def(
            type_of(handle),
            type_of(attn_desc),
            type_of(w_kind),
            type_of(weight_size_in_bytes),
            type_of(weights),
            type_of(w_desc),
            type_of(w_addr),
        ) thin -> cudnnStatus_t,
    ]()(
        handle, attn_desc, w_kind, weight_size_in_bytes, weights, w_desc, w_addr
    )


def cudnnSetSeqDataDescriptor(
    seq_data_desc: UnsafePointer[cudnnSeqDataStruct, _],
    data_type: cudnnDataType_t,
    nb_dims: Int16,
    dim_a: OpaquePointer,
    axes: OpaquePointer,
    seq_length_array_size: Int,
    seq_length_array: OpaquePointer,
    padding_fill: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetSeqDataDescriptor",
        def(
            type_of(seq_data_desc),
            type_of(data_type),
            type_of(nb_dims),
            type_of(dim_a),
            type_of(axes),
            type_of(seq_length_array_size),
            type_of(seq_length_array),
            type_of(padding_fill),
        ) thin -> cudnnStatus_t,
    ]()(
        seq_data_desc,
        data_type,
        nb_dims,
        dim_a,
        axes,
        seq_length_array_size,
        seq_length_array,
        padding_fill,
    )


def cudnnCreateSeqDataDescriptor(
    seq_data_desc: UnsafePointer[
        UnsafePointer[cudnnSeqDataStruct, UntrackedOrigin[mut=True]], _
    ],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnCreateSeqDataDescriptor",
        def(type_of(seq_data_desc)) thin -> cudnnStatus_t,
    ]()(seq_data_desc)


def cudnnGetRNNPaddingMode(
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    padding_mode: UnsafePointer[Int16, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNPaddingMode",
        def(type_of(rnn_desc), type_of(padding_mode)) thin -> cudnnStatus_t,
    ]()(rnn_desc, padding_mode)


comptime cudnnAttnDescriptor_t = UnsafePointer[
    cudnnAttnStruct, UntrackedOrigin[mut=True]
]

comptime cudnnAttnQueryMap_t = Int16


def cudnnGetRNNLinLayerBiasParams(
    handle: UnsafePointer[cudnnContext, _],
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    pseudo_layer: Int16,
    x_desc: UnsafePointer[cudnnTensorStruct, _],
    w_desc: UnsafePointer[cudnnFilterStruct, _],
    w: OpaquePointer,
    lin_layer_id: Int16,
    lin_layer_bias_desc: UnsafePointer[cudnnFilterStruct, _],
    lin_layer_bias: UnsafePointer[AnyOpaquePointer, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNLinLayerBiasParams",
        def(
            type_of(handle),
            type_of(rnn_desc),
            type_of(pseudo_layer),
            type_of(x_desc),
            type_of(w_desc),
            type_of(w),
            type_of(lin_layer_id),
            type_of(lin_layer_bias_desc),
            type_of(lin_layer_bias),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        rnn_desc,
        pseudo_layer,
        x_desc,
        w_desc,
        w,
        lin_layer_id,
        lin_layer_bias_desc,
        lin_layer_bias,
    )


def cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    handle: UnsafePointer[cudnnContext, _],
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    count: UnsafePointer[Int16, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNForwardInferenceAlgorithmMaxCount",
        def(
            type_of(handle),
            type_of(rnn_desc),
            type_of(count),
        ) thin -> cudnnStatus_t,
    ]()(handle, rnn_desc, count)


def cudnnGetRNNWeightParams(
    handle: UnsafePointer[cudnnContext, _],
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    pseudo_layer: Int32,
    weight_space_size: Int,
    weight_space: OpaquePointer,
    lin_layer_id: Int32,
    m_desc: UnsafePointer[cudnnTensorStruct, _],
    m_addr: UnsafePointer[AnyOpaquePointer, _],
    b_desc: UnsafePointer[cudnnTensorStruct, _],
    b_addr: UnsafePointer[AnyOpaquePointer, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNWeightParams",
        def(
            type_of(handle),
            type_of(rnn_desc),
            type_of(pseudo_layer),
            type_of(weight_space_size),
            type_of(weight_space),
            type_of(lin_layer_id),
            type_of(m_desc),
            type_of(m_addr),
            type_of(b_desc),
            type_of(b_addr),
        ) thin -> cudnnStatus_t,
    ]()(
        handle,
        rnn_desc,
        pseudo_layer,
        weight_space_size,
        weight_space,
        lin_layer_id,
        m_desc,
        m_addr,
        b_desc,
        b_addr,
    )


def cudnnGetRNNDescriptor_v8(
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
    algo: UnsafePointer[cudnnRNNAlgo_t, _],
    cell_mode: UnsafePointer[cudnnRNNMode_t, _],
    bias_mode: UnsafePointer[cudnnRNNBiasMode_t, _],
    dir_mode: UnsafePointer[cudnnDirectionMode_t, _],
    input_mode: UnsafePointer[cudnnRNNInputMode_t, _],
    data_type: UnsafePointer[cudnnDataType_t, _],
    math_prec: UnsafePointer[cudnnDataType_t, _],
    math_type: UnsafePointer[cudnnMathType_t, _],
    input_size: UnsafePointer[Int32, _],
    hidden_size: UnsafePointer[Int32, _],
    proj_size: UnsafePointer[Int32, _],
    num_layers: UnsafePointer[Int32, _],
    dropout_desc: DoubleNestedPointer[cudnnDropoutStruct],
    aux_flags: UnsafePointer[UInt32, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnGetRNNDescriptor_v8",
        def(
            type_of(rnn_desc),
            type_of(algo),
            type_of(cell_mode),
            type_of(bias_mode),
            type_of(dir_mode),
            type_of(input_mode),
            type_of(data_type),
            type_of(math_prec),
            type_of(math_type),
            type_of(input_size),
            type_of(hidden_size),
            type_of(proj_size),
            type_of(num_layers),
            type_of(dropout_desc),
            type_of(aux_flags),
        ) thin -> cudnnStatus_t,
    ]()(
        rnn_desc,
        algo,
        cell_mode,
        bias_mode,
        dir_mode,
        input_mode,
        data_type,
        math_prec,
        math_type,
        input_size,
        hidden_size,
        proj_size,
        num_layers,
        dropout_desc,
        aux_flags,
    )


@fieldwise_init
struct cudnnSeqDataAxis_t(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUDNN_SEQDATA_TIME_DIM = Self(0)
    comptime CUDNN_SEQDATA_BATCH_DIM = Self(1)
    comptime CUDNN_SEQDATA_BEAM_DIM = Self(2)
    comptime CUDNN_SEQDATA_VECT_DIM = Self(3)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __is__(self, other: Self) -> Bool:
        return self == other

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUDNN_SEQDATA_TIME_DIM:
            return writer.write_string("CUDNN_SEQDATA_TIME_DIM")
        if self is Self.CUDNN_SEQDATA_BATCH_DIM:
            return writer.write_string("CUDNN_SEQDATA_BATCH_DIM")
        if self is Self.CUDNN_SEQDATA_BEAM_DIM:
            return writer.write_string("CUDNN_SEQDATA_BEAM_DIM")
        if self is Self.CUDNN_SEQDATA_VECT_DIM:
            return writer.write_string("CUDNN_SEQDATA_VECT_DIM")
        abort("invalid cudnnSeqDataAxis_t entry")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        t"cudnnSeqDataAxis_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


def cudnnSetRNNPaddingMode(
    rnn_desc: UnsafePointer[cudnnRNNStruct, _], padding_mode: Int16
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNPaddingMode",
        def(type_of(rnn_desc), type_of(padding_mode)) thin -> cudnnStatus_t,
    ]()(rnn_desc, padding_mode)


def cudnnDestroyRNNDescriptor(
    rnn_desc: UnsafePointer[cudnnRNNStruct, _],
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnDestroyRNNDescriptor",
        def(type_of(rnn_desc)) thin -> cudnnStatus_t,
    ]()(rnn_desc)


def cudnnSetRNNDataDescriptor(
    rnn_data_desc: UnsafePointer[cudnnRNNDataStruct, _],
    data_type: cudnnDataType_t,
    layout: cudnnRNNDataLayout_t,
    max_seq_length: Int16,
    batch_size: Int16,
    vector_size: Int16,
    seq_length_array: OpaquePointer,
    padding_fill: OpaquePointer,
) raises -> cudnnStatus_t:
    return _get_dylib_function[
        "cudnnSetRNNDataDescriptor",
        def(
            type_of(rnn_data_desc),
            type_of(data_type),
            type_of(layout),
            type_of(max_seq_length),
            type_of(batch_size),
            type_of(vector_size),
            type_of(seq_length_array),
            type_of(padding_fill),
        ) thin -> cudnnStatus_t,
    ]()(
        rnn_data_desc,
        data_type,
        layout,
        max_seq_length,
        batch_size,
        vector_size,
        seq_length_array,
        padding_fill,
    )
