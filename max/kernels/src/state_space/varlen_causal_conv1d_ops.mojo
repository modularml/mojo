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
"""Varlen Causal Conv1D operation registrations for Mamba SSM.

This module registers operations for variable-length causal 1D convolution:
- causal_conv1d_varlen_fwd: Forward pass for varlen sequences
- causal_conv1d_varlen_update: Update for decode/autoregressive inference
- causal_conv1d_varlen_states: Extract states from varlen sequences
"""

from std.math import ceildiv

import extensibility
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu

from extensibility import InputTensor, OutputTensor
from std.utils.index import IndexList

from state_space.varlen_causal_conv1d import (
    causal_conv1d_varlen_update_cpu,
    causal_conv1d_varlen_update_gpu,
    causal_conv1d_varlen_states_cpu,
    causal_conv1d_varlen_states_gpu,
)


# ============================================================================
# Varlen Causal Conv1D Forward Registration
# ============================================================================
#
# NOTE: `causal_conv1d_varlen_fwd` is registered in the built-in kernel
# library (`graph_compiler/builtin_kernels/kernels.mojo`), NOT here, mirroring
# the `gated_delta_conv1d_fwd` precedent. Registering it here too would
# double-register the op (this `_ops` module is part of `//max:state_space`,
# a dep of `builtin_kernels`). The kernel math lives in
# `state_space.varlen_causal_conv1d`. The `_update` / `_states` variants below
# are still registered here (they have no built-in registration).


# ============================================================================
# Varlen Causal Conv1D Update Registration
# ============================================================================


@extensibility.register("causal_conv1d_varlen_update")
struct CausalConv1DVarlenUpdate[activation: StaticString]:
    """Varlen causal conv1d update for autoregressive decoding.

    Performs incremental convolution update for token-by-token generation.
    Updates the conv_state in-place with new input values.

    Parameters:
        activation: Activation function - "none" or "silu".

    Tensor Shapes:
        - output: (batch, dim, seqlen) - Output tensor
        - x: (batch, dim, seqlen) - Input tensor
        - weight: (dim, width) - Convolution weights
        - bias: (dim,) - Per-channel bias
        - conv_state: (batch, dim, state_len) - Conv state (in/out)
        - cache_seqlens: (batch,) - Current sequence lengths (optional)
        - conv_state_indices: (batch,) - Indices into conv_state (optional)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        conv_state: OutputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        cache_seqlens: InputTensor[dtype=.int32, rank=1, ...],
        conv_state_indices: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var batch = x.dim_size(0)
        var dim = x.dim_size(1)
        var seqlen = x.dim_size(2)
        var width = weight.dim_size(1)
        var state_len = conv_state.dim_size(2)

        var output_tt = output.to_tile_tensor[.int32]()
        var x_tt = x.to_tile_tensor[.int32]()
        var weight_tt = weight.to_tile_tensor[.int32]()
        var bias_tt = bias.to_tile_tensor[.int32]()
        var conv_state_tt = conv_state.to_tile_tensor[.int32]()
        var cache_seqlens_tt = cache_seqlens.to_tile_tensor[.int32]()
        var conv_state_indices_tt = conv_state_indices.to_tile_tensor[
            DType.int32
        ]()

        var x_strides = x.strides()
        var weight_strides = weight.strides()
        var output_strides = output.strides()
        var conv_state_strides = conv_state.strides()

        var x_batch_stride = UInt32(x_strides[0])
        var x_dim_stride = UInt32(x_strides[1])
        var x_seqlen_stride = UInt32(x_strides[2])
        var weight_dim_stride = UInt32(weight_strides[0])
        var weight_width_stride = UInt32(weight_strides[1])
        var conv_state_batch_stride = UInt32(conv_state_strides[0])
        var conv_state_dim_stride = UInt32(conv_state_strides[1])
        var conv_state_seqlen_stride = UInt32(conv_state_strides[2])
        var out_batch_stride = UInt32(output_strides[0])
        var out_dim_stride = UInt32(output_strides[1])
        var out_seqlen_stride = UInt32(output_strides[2])

        var has_conv_state_indices = conv_state_indices.dim_size(0) > 0
        var has_cache_seqlens = cache_seqlens.dim_size(0) > 0
        var has_bias = bias.dim_size(0) > 0

        var silu_activation = Self.activation == "silu"
        comptime PAD_SLOT_ID: Int32 = -1

        comptime if is_cpu[target]():
            causal_conv1d_varlen_update_cpu[
                x_tt.dtype,
                weight_tt.dtype,
                bias_tt.dtype,
                output_tt.dtype,
                conv_state_tt.dtype,
                cache_seqlens_tt.dtype,
                conv_state_indices_tt.dtype,
            ](
                batch,
                dim,
                seqlen,
                width,
                state_len,
                x_tt,
                weight_tt,
                bias_tt,
                conv_state_tt,
                cache_seqlens_tt,
                conv_state_indices_tt,
                output_tt,
                x_batch_stride,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                conv_state_batch_stride,
                conv_state_dim_stride,
                conv_state_seqlen_stride,
                out_batch_stride,
                out_dim_stride,
                out_seqlen_stride,
                silu_activation,
                PAD_SLOT_ID,
                has_conv_state_indices,
                has_cache_seqlens,
                has_bias,
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            comptime BLOCK_DIM = 128
            var silu_activation_int8 = Int8(silu_activation)

            if width == 1:
                comptime kWidth = 1
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_update_gpu[
                        x_tt.dtype,
                        weight_tt.dtype,
                        bias_tt.dtype,
                        output_tt.dtype,
                        conv_state_tt.dtype,
                        cache_seqlens_tt.dtype,
                        conv_state_indices_tt.dtype,
                        kWidth,
                        BLOCK_DIM,
                        x_tt.LayoutType,
                        weight_tt.LayoutType,
                        bias_tt.LayoutType,
                        conv_state_tt.LayoutType,
                        cache_seqlens_tt.LayoutType,
                        conv_state_indices_tt.LayoutType,
                        output_tt.LayoutType,
                        x_tt.Engine,
                        weight_tt.Engine,
                        bias_tt.Engine,
                        conv_state_tt.Engine,
                        cache_seqlens_tt.Engine,
                        conv_state_indices_tt.Engine,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(state_len),
                    x_tt,
                    weight_tt,
                    bias_tt,
                    conv_state_tt,
                    cache_seqlens_tt,
                    conv_state_indices_tt,
                    output_tt,
                    x_batch_stride,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    conv_state_batch_stride,
                    conv_state_dim_stride,
                    conv_state_seqlen_stride,
                    out_batch_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_conv_state_indices),
                    Int8(has_cache_seqlens),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM,),
                )
            elif width == 2:
                comptime kWidth = 2
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_update_gpu[
                        x_tt.dtype,
                        weight_tt.dtype,
                        bias_tt.dtype,
                        output_tt.dtype,
                        conv_state_tt.dtype,
                        cache_seqlens_tt.dtype,
                        conv_state_indices_tt.dtype,
                        kWidth,
                        BLOCK_DIM,
                        x_tt.LayoutType,
                        weight_tt.LayoutType,
                        bias_tt.LayoutType,
                        conv_state_tt.LayoutType,
                        cache_seqlens_tt.LayoutType,
                        conv_state_indices_tt.LayoutType,
                        output_tt.LayoutType,
                        x_tt.Engine,
                        weight_tt.Engine,
                        bias_tt.Engine,
                        conv_state_tt.Engine,
                        cache_seqlens_tt.Engine,
                        conv_state_indices_tt.Engine,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(state_len),
                    x_tt,
                    weight_tt,
                    bias_tt,
                    conv_state_tt,
                    cache_seqlens_tt,
                    conv_state_indices_tt,
                    output_tt,
                    x_batch_stride,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    conv_state_batch_stride,
                    conv_state_dim_stride,
                    conv_state_seqlen_stride,
                    out_batch_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_conv_state_indices),
                    Int8(has_cache_seqlens),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM,),
                )
            elif width == 3:
                comptime kWidth = 3
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_update_gpu[
                        x_tt.dtype,
                        weight_tt.dtype,
                        bias_tt.dtype,
                        output_tt.dtype,
                        conv_state_tt.dtype,
                        cache_seqlens_tt.dtype,
                        conv_state_indices_tt.dtype,
                        kWidth,
                        BLOCK_DIM,
                        x_tt.LayoutType,
                        weight_tt.LayoutType,
                        bias_tt.LayoutType,
                        conv_state_tt.LayoutType,
                        cache_seqlens_tt.LayoutType,
                        conv_state_indices_tt.LayoutType,
                        output_tt.LayoutType,
                        x_tt.Engine,
                        weight_tt.Engine,
                        bias_tt.Engine,
                        conv_state_tt.Engine,
                        cache_seqlens_tt.Engine,
                        conv_state_indices_tt.Engine,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(state_len),
                    x_tt,
                    weight_tt,
                    bias_tt,
                    conv_state_tt,
                    cache_seqlens_tt,
                    conv_state_indices_tt,
                    output_tt,
                    x_batch_stride,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    conv_state_batch_stride,
                    conv_state_dim_stride,
                    conv_state_seqlen_stride,
                    out_batch_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_conv_state_indices),
                    Int8(has_cache_seqlens),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM,),
                )
            elif width == 4:
                comptime kWidth = 4
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_update_gpu[
                        x_tt.dtype,
                        weight_tt.dtype,
                        bias_tt.dtype,
                        output_tt.dtype,
                        conv_state_tt.dtype,
                        cache_seqlens_tt.dtype,
                        conv_state_indices_tt.dtype,
                        kWidth,
                        BLOCK_DIM,
                        x_tt.LayoutType,
                        weight_tt.LayoutType,
                        bias_tt.LayoutType,
                        conv_state_tt.LayoutType,
                        cache_seqlens_tt.LayoutType,
                        conv_state_indices_tt.LayoutType,
                        output_tt.LayoutType,
                        x_tt.Engine,
                        weight_tt.Engine,
                        bias_tt.Engine,
                        conv_state_tt.Engine,
                        cache_seqlens_tt.Engine,
                        conv_state_indices_tt.Engine,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    Int32(batch),
                    Int32(dim),
                    Int32(seqlen),
                    Int32(state_len),
                    x_tt,
                    weight_tt,
                    bias_tt,
                    conv_state_tt,
                    cache_seqlens_tt,
                    conv_state_indices_tt,
                    output_tt,
                    x_batch_stride,
                    x_dim_stride,
                    x_seqlen_stride,
                    weight_dim_stride,
                    weight_width_stride,
                    conv_state_batch_stride,
                    conv_state_dim_stride,
                    conv_state_seqlen_stride,
                    out_batch_stride,
                    out_dim_stride,
                    out_seqlen_stride,
                    silu_activation_int8,
                    PAD_SLOT_ID,
                    Int8(has_conv_state_indices),
                    Int8(has_cache_seqlens),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM,),
                )
            else:
                raise Error(
                    "Unsupported kernel width: only widths 1, 2, 3, 4 are"
                    " supported"
                )
        else:
            raise Error("Unsupported target device")


@extensibility.register_shape_function("causal_conv1d_varlen_update")
def causal_conv1d_varlen_update_shape[
    dtype: DType,
](
    x: InputTensor[dtype=dtype, rank=3, ...],
    weight: InputTensor[dtype=dtype, rank=2, ...],
    bias: InputTensor[dtype=dtype, rank=1, ...],
    cache_seqlens: InputTensor[dtype=.int32, rank=1, ...],
    conv_state_indices: InputTensor[dtype=.int32, rank=1, ...],
) -> IndexList[3]:
    """Returns the output shape for the `causal_conv1d_varlen_update` op.

    The output shape equals the input shape `(batch, dim, seqlen)`.

    Parameters:
        dtype: Element type of the `x`, `weight`, and `bias` input tensors.

    Args:
        x: Input tensor with shape `(batch, dim, seqlen)`.
        weight: Convolution weights with shape `(dim, width)`.
        bias: Per-channel bias with shape `(dim,)`.
        cache_seqlens: Current sequence lengths per batch entry with shape
            `(batch,)`.
        conv_state_indices: Indices into the conv state buffer with shape
            `(batch,)`.

    Returns:
        The output tensor shape, equal to `x.shape()`.
    """
    return x.shape()


# ============================================================================
# Varlen Causal Conv1D States Registration
# ============================================================================


@extensibility.register("causal_conv1d_varlen_states")
struct CausalConv1DVarlenStates:
    """Extract conv states from variable-length sequences.

    Extracts the last state_len elements from each sequence to initialize
    conv_state for subsequent autoregressive generation.

    Tensor Shapes:
        - states: (batch, dim, state_len) - Output states tensor
        - x: (total_tokens, dim) - Input tensor (concatenated sequences)
        - cu_seqlens: (batch + 1,) - Cumulative sequence lengths
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        states: OutputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        cu_seqlens: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var total_tokens = x.dim_size(0)
        var dim = x.dim_size(1)
        var batch = cu_seqlens.dim_size(0) - 1
        var state_len = states.dim_size(2)

        var states_tt = states.to_tile_tensor[.int32]()
        var x_tt = x.to_tile_tensor[.int32]()
        var cu_seqlens_tt = cu_seqlens.to_tile_tensor[.int32]()

        var x_strides = x.strides()
        var states_strides = states.strides()

        var x_seqlen_stride = UInt32(x_strides[0])
        var x_dim_stride = UInt32(x_strides[1])
        var states_batch_stride = UInt32(states_strides[0])
        var states_dim_stride = UInt32(states_strides[1])
        var states_seqlen_stride = UInt32(states_strides[2])

        comptime if is_cpu[target]():
            causal_conv1d_varlen_states_cpu[
                x_tt.dtype,
                cu_seqlens_tt.dtype,
                states_tt.dtype,
            ](
                total_tokens,
                dim,
                batch,
                state_len,
                x_tt,
                cu_seqlens_tt,
                states_tt,
                x_seqlen_stride,
                x_dim_stride,
                states_batch_stride,
                states_dim_stride,
                states_seqlen_stride,
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            comptime BLOCK_DIM = 128
            var compiled_func = gpu_ctx.compile_function[
                causal_conv1d_varlen_states_gpu[
                    x_tt.dtype,
                    cu_seqlens_tt.dtype,
                    states_tt.dtype,
                    BLOCK_DIM,
                    BLOCK_DIM,
                    x_tt.LayoutType,
                    cu_seqlens_tt.LayoutType,
                    states_tt.LayoutType,
                    x_tt.Engine,
                    cu_seqlens_tt.Engine,
                    states_tt.Engine,
                ]
            ]()
            gpu_ctx.enqueue_function(
                compiled_func,
                Int32(total_tokens),
                Int32(dim),
                Int32(batch),
                Int32(state_len),
                x_tt,
                cu_seqlens_tt,
                states_tt,
                x_seqlen_stride,
                x_dim_stride,
                states_batch_stride,
                states_dim_stride,
                states_seqlen_stride,
                grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                block_dim=(BLOCK_DIM,),
            )
        else:
            raise Error("Unsupported target device")


@extensibility.register_shape_function("causal_conv1d_varlen_states")
def causal_conv1d_varlen_states_shape[
    dtype: DType,
](
    x: InputTensor[dtype=dtype, rank=2, ...],
    cu_seqlens: InputTensor[dtype=.int32, rank=1, ...],
) -> IndexList[3]:
    """Returns the output shape for the `causal_conv1d_varlen_states` op.

    The output is a state buffer with one entry per sequence: shape
    `(batch, dim, state_len)`. The `state_len` dimension is determined by
    the output allocation at runtime; this function returns 0 for that
    dimension as a placeholder.

    Parameters:
        dtype: Element type of the packed input tensor `x`.

    Args:
        x: Packed input tensor with shape `(total_tokens, dim)`.
        cu_seqlens: Cumulative sequence lengths with shape `(batch + 1,)`.

    Returns:
        The output state shape `(batch, dim, 0)` where `batch` is inferred
        from `cu_seqlens` and `state_len` is filled at runtime.
    """
    var batch = cu_seqlens.dim_size(0) - 1
    var dim = x.dim_size(1)
    # state_len is derived from the output tensor shape at runtime
    # Return a placeholder shape; actual shape determined by output allocation
    return IndexList[3](batch, dim, 0)
