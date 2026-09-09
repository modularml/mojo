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
"""Python wrappers for the Gated DeltaNet two-pass kernels.

Provides two graph-level wrappers that call the Mojo ops registered in the
``state_space`` package:

  ``gated_delta_conv1d_fwd()``
    Pass 1: causal depthwise conv1d over a ragged batch of sequences.
    Reads/writes a single mutable conv-state pool of shape
    ``[max_slots, conv_dim, K-1]`` at slot ``slot_idx[batch_item]`` —
    no gather/scatter, no working buffers.

  ``gated_delta_recurrence_fwd()``
    Pass 2: gated delta rule recurrence over the conv1d outputs. Reads/
    writes a single mutable recurrent-state pool of shape
    ``[max_slots, nv, KD, VD]`` at slot ``slot_idx[batch_item]``.

Both ops mutate their pool inputs in place (``ops.inplace_custom``); the
graph output is just the per-token tensor (conv output / recurrence
output respectively). This matches vLLM's ``selective_state_update``
design — kernel does pointer arithmetic ``state_ptr += slot * stride``
into a long-lived pool, no per-step pool allocation.


Usage
-----
::

    from max.nn.state_space import (
        gated_delta_conv1d_fwd,
        gated_delta_recurrence_fwd,
    )

    # Pass 1: conv_pool is a BufferValue mutated in place at slot_idx[b].
    conv_output = gated_delta_conv1d_fwd(
        qkv_input_ragged=qkv_f32,         # [total_N, conv_dim]
        conv_weight=conv_weight_flat,     # [conv_dim, K]
        conv_state=conv_pool,             # [max_slots, conv_dim, K-1] (mut)
        slot_idx=slot_idx_uint32,         # [B]
        input_row_offsets=offsets_uint32, # [B+1]
    )

    # Pass 2: recurrent_pool is mutated in place too; there is no
    # state-out graph output.
    recurrence_output = gated_delta_recurrence_fwd(
        qkv_conv_output=conv_output,      # [total_N, conv_dim]
        decay_per_token=decay,            # [total_N, nv]
        beta_per_token=beta,              # [total_N, nv]
        recurrent_state=recurrent_pool,   # [max_slots, nv, kd, vd] (mut)
        slot_idx=slot_idx_uint32,         # [B]
        input_row_offsets=offsets_uint32, # [B+1]
    )
"""

from __future__ import annotations

from typing import cast

from max.dtype import DType
from max.graph import BufferValue, TensorType, TensorValue, ops


def gated_delta_conv1d_fwd(
    qkv_input_ragged: TensorValue,
    conv_weight: TensorValue,
    conv_state: BufferValue,
    slot_idx: TensorValue,
    input_row_offsets: TensorValue,
) -> TensorValue:
    """Pass 1: causal conv1d that mutates a slot-indexed pool in place.

    ``conv_state`` is a mutable pool of shape ``[max_slots, conv_dim, K-1]``
    and the kernel reads/writes slot ``slot_idx[batch_item]`` directly.
    There is no ``conv_state_out`` graph output: the pool is mutated in
    place.

    Args:
        qkv_input_ragged: ``[total_seq_len, conv_dim]`` projected QKV input.
        conv_weight: ``[conv_dim, kernel_size]`` depthwise conv weights.
        conv_state: ``[max_slots, conv_dim, kernel_size-1]`` mutable pool.
        slot_idx: ``[batch_size]`` uint32 slot indices into the pool.
        input_row_offsets: ``[batch_size + 1]`` uint32 ragged offsets.

    Returns:
        ``conv_output_ragged: [total_seq_len, conv_dim]``.
    """
    device = qkv_input_ragged.device
    total_seq_len = qkv_input_ragged.shape[0]
    conv_dim = qkv_input_ragged.shape[1]

    conv_output_ragged_type = TensorType(
        DType.float32, [total_seq_len, conv_dim], device
    )

    # input_row_offsets must be uint32 for the Mojo op
    offsets_uint32 = (
        input_row_offsets
        if input_row_offsets.type.dtype == DType.uint32
        else input_row_offsets.cast(DType.uint32)
    )
    slot_idx_uint32 = (
        slot_idx
        if slot_idx.type.dtype == DType.uint32
        else slot_idx.cast(DType.uint32)
    )

    results = ops.inplace_custom(
        "gated_delta_conv1d_fwd",
        device,
        [
            qkv_input_ragged,
            conv_weight,
            conv_state,
            slot_idx_uint32,
            offsets_uint32,
        ],
        [conv_output_ragged_type],
    )
    return cast(TensorValue, results[0])


def gated_delta_recurrence_fwd(
    qkv_conv_output: TensorValue,
    decay_per_token: TensorValue,
    beta_per_token: TensorValue,
    recurrent_state: BufferValue,
    slot_idx: TensorValue,
    input_row_offsets: TensorValue,
) -> TensorValue:
    """Pass 2: gated delta recurrence mutating a slot-indexed pool in place.

    ``recurrent_state`` is a mutable pool of shape ``[max_slots, nv, KD, VD]``
    and the kernel reads/writes slot ``slot_idx[batch_item]`` directly.
    There is no ``recurrent_state_out`` graph output: the pool is mutated
    in place.

    Args:
        qkv_conv_output: ``[total_seq_len, conv_dim]`` from
            :func:`gated_delta_conv1d_fwd`.
        decay_per_token: ``[total_seq_len, num_value_heads]`` decays.
        beta_per_token: ``[total_seq_len, num_value_heads]`` beta gates.
        recurrent_state: ``[max_slots, nv, KD, VD]`` mutable pool.
        slot_idx: ``[batch_size]`` uint32 slot indices into the pool.
        input_row_offsets: ``[batch_size + 1]`` uint32 ragged offsets.

    Returns:
        ``recurrence_output: [total_seq_len, value_dim]``.
    """
    device = qkv_conv_output.device
    total_seq_len = qkv_conv_output.shape[0]
    num_value_heads = decay_per_token.shape[1]
    # recurrent_state.shape is [max_slots, nv, KD, VD]; index 3 is value_head_dim.
    value_head_dim = recurrent_state.shape[3]
    value_dim = num_value_heads * value_head_dim

    recurrence_output_type = TensorType(
        DType.float32, [total_seq_len, value_dim], device
    )

    # input_row_offsets must be uint32 for the Mojo op
    offsets_uint32 = (
        input_row_offsets
        if input_row_offsets.type.dtype == DType.uint32
        else input_row_offsets.cast(DType.uint32)
    )
    slot_idx_uint32 = (
        slot_idx
        if slot_idx.type.dtype == DType.uint32
        else slot_idx.cast(DType.uint32)
    )

    results = ops.inplace_custom(
        "gated_delta_recurrence_fwd",
        device,
        [
            qkv_conv_output,
            decay_per_token,
            beta_per_token,
            recurrent_state,
            slot_idx_uint32,
            offsets_uint32,
        ],
        [recurrence_output_type],
    )
    return cast(TensorValue, results[0])
