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
"""Causal depthwise conv1d for the Gated DeltaNet two-pass prefill.

This is Pass 1 of the two-pass gated delta rule prefill path.  It computes
the causal 1-D convolution over a ragged (variable-length) batch of sequences
and updates the per-sequence sliding-window conv state.

Unlike the existing causal_conv1d_varlen_fwd (which uses [dim, total_seqlen]
layout for Mamba compatibility), this kernel uses [total_seqlen, conv_dim]
layout to match the gated_deltanet.py convention where all per-token tensors
are seqlen-first.

Tensor shapes
-------------
Inputs:
  qkv_input_ragged   : [total_seq_len, conv_dim]              float32
      Flat projected QKV input, all sequences concatenated.
  conv_weight        : [conv_dim, kernel_size]                float32
      Depthwise conv weights (one weight per channel per time offset).
  conv_state         : [max_slots, conv_dim, kernel_size-1]
      Mutable sliding-window conv state pool.  The kernel reads/writes
      slot `slot_idx[batch_item]` in place; all other slots are
      untouched.  Slots within a single pool entry are ordered
      oldest-to-newest: window slot 0 is the token at position -(K-1)
      relative to the current sequence start.  Pool dtype is independent
      of the working dtype.
  slot_idx           : [batch_size]                           uint32
      Pool slot index for each batch item.
  input_row_offsets  : [batch_size + 1]                       uint32
      Exclusive prefix sums of sequence lengths.  Sequence b spans
      token indices [input_row_offsets[b], input_row_offsets[b+1]).

Outputs:
  conv_output_ragged : [total_seq_len, conv_dim]              float32
      Causal conv output in the same ragged layout as the input.
  (conv_state is mutated in place; there is no separate state-out
   tensor.  Window slot j ends up holding the raw input at position
   seq_len - (kernel_size-1) + j within the sequence, carrying forward
   from the old window when seq_len is shorter.)

Thread mapping (GPU)
--------------------
  Grid  : (batch_size, ceildiv(conv_dim, CONV1D_BLOCK_DIM))
  Block : (CONV1D_BLOCK_DIM,)
  One thread per (batch_item, conv_channel).  Each thread processes its
  channel's full sequence sequentially, reading from conv_state at slot
  `slot_idx[batch_item]` for the look-back that extends before the
  current sequence.
"""

from max.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)
from layout import TensorLayout, TileTensor
from std.utils.index import IndexList


# ===----------------------------------------------------------------------=== #
# GPU Kernel
# ===----------------------------------------------------------------------=== #


def gated_delta_conv1d_fwd_gpu[
    work_dtype: DType,  # for qkv_input_ragged / conv_weight / conv_output_ragged
    state_dtype: DType,  # for the conv_state pool (typically bf16)
    KERNEL_SIZE: Int,
    CONV1D_BLOCK_DIM: Int,
    qkv_input_ragged_LT: TensorLayout,
    conv_weight_LT: TensorLayout,
    conv_state_LT: TensorLayout,
    slot_idx_LT: TensorLayout,
    input_row_offsets_LT: TensorLayout,
    conv_output_ragged_LT: TensorLayout,
](
    batch_size: Int32,
    total_seq_len: Int32,
    conv_dim: Int32,
    qkv_input_ragged: TileTensor[
        work_dtype, qkv_input_ragged_LT, MutUntrackedOrigin
    ],
    conv_weight: TileTensor[work_dtype, conv_weight_LT, MutUntrackedOrigin],
    conv_state: TileTensor[state_dtype, conv_state_LT, MutUntrackedOrigin],
    slot_idx: TileTensor[.uint32, slot_idx_LT, MutUntrackedOrigin],
    input_row_offsets: TileTensor[
        .uint32, input_row_offsets_LT, MutUntrackedOrigin
    ],
    conv_output_ragged: TileTensor[
        work_dtype, conv_output_ragged_LT, MutUntrackedOrigin
    ],
    # Strides for [total_seq_len, conv_dim] tensors
    qkv_input_seqlen_stride: UInt32,  # stride along total_seq_len axis
    qkv_input_channel_stride: UInt32,  # stride along conv_dim axis (usually 1)
    conv_weight_channel_stride: UInt32,  # stride along conv_dim axis
    conv_weight_offset_stride: UInt32,  # stride along kernel_size axis
    # Strides for the [max_slots, conv_dim, kernel_size-1] conv state pool.
    conv_state_pool_stride: UInt32,
    conv_state_channel_stride: UInt32,
    conv_state_window_stride: UInt32,
    # Output strides (match input strides for conv_output_ragged)
    conv_output_seqlen_stride: UInt32,
    conv_output_channel_stride: UInt32,
):
    """GPU kernel: slot-indexed causal depthwise conv1d over a ragged batch.

    The conv state lives in a single mutable pool of shape
    ``[max_slots, conv_dim, K-1]``; the kernel reads/writes slot
    ``slot_idx[batch_item_idx]`` for batch item ``batch_item_idx``. Reads of
    the old window during look-back precede all writes of the new window, so
    in-place mutation is safe. One thread handles one (batch_item,
    conv_channel) pair for the entire sequence; the channel's K weights live
    in registers across the token loop.
    """
    var _batch_size = Int(batch_size)
    var _total_seq_len = Int(total_seq_len)
    var _conv_dim = Int(conv_dim)
    var batch_item_idx = Int(block_idx.x)
    var channel_block_idx = Int(block_idx.y)
    var thread_within_block = Int(thread_idx.x)

    var conv_channel_idx = (
        channel_block_idx * CONV1D_BLOCK_DIM + thread_within_block
    )

    if batch_item_idx >= _batch_size or conv_channel_idx >= _conv_dim:
        return

    # Read the pool slot for this batch item exactly once. The caller
    # (`GatedDeltaNetStateCache.claim`) guarantees `slot < max_slots`.
    var slot = Int(slot_idx._storage[batch_item_idx])

    # ── Sequence boundaries from ragged offsets ─────────────────────────────
    var sequence_start_flat_idx = Int(
        input_row_offsets._storage[batch_item_idx]
    )
    var sequence_end_flat_idx = Int(
        input_row_offsets._storage[batch_item_idx + 1]
    )
    var sequence_length = sequence_end_flat_idx - sequence_start_flat_idx

    # ── Load conv weights for this channel into registers ───────────────────
    # Avoids re-reading the same KERNEL_SIZE weights on every token step.
    var weight_register = SIMD[work_dtype, KERNEL_SIZE](0)
    comptime for kernel_offset_k in range(KERNEL_SIZE):
        var weight_flat_offset = (
            UInt32(conv_channel_idx) * conv_weight_channel_stride
            + UInt32(kernel_offset_k) * conv_weight_offset_stride
        )
        weight_register[kernel_offset_k] = conv_weight._storage[
            weight_flat_offset
        ]

    comptime KERNEL_SIZE_MINUS_ONE = KERNEL_SIZE - 1

    # ── Process each token in this sequence ─────────────────────────────────
    for token_position_in_sequence in range(sequence_length):
        var flat_token_idx = (
            sequence_start_flat_idx + token_position_in_sequence
        )
        var conv_sum = Float32(0.0)

        # Convolve: sum over K offsets, looking back K-1 tokens.
        # For kernel offset k, the input token is at relative position:
        #   lookback_position = token_position_in_sequence - (KERNEL_SIZE - 1 - k)
        # A negative lookback_position falls before the current sequence and
        # is read from conv_state.  Non-negative positions come from the
        # ragged input buffer.
        comptime for kernel_offset_k in range(KERNEL_SIZE):
            var lookback_position = token_position_in_sequence - (
                KERNEL_SIZE_MINUS_ONE - kernel_offset_k
            )

            # Cast on read so the work-dtype qkv input and the state-dtype
            # pool produce the same Float32 conv_sum accumulator.
            var input_value: Float32 = 0

            if lookback_position >= 0:
                # Within the current sequence: read from ragged input buffer
                var ragged_flat_offset = (
                    UInt32(sequence_start_flat_idx + lookback_position)
                    * qkv_input_seqlen_stride
                    + UInt32(conv_channel_idx) * qkv_input_channel_stride
                )
                input_value = Float32(
                    qkv_input_ragged._storage[ragged_flat_offset]
                )
            else:
                # Before the current sequence: read from conv_state pool slot.
                # Map the negative lookback to a position within the K-1 window:
                #   window_idx = KERNEL_SIZE_MINUS_ONE + lookback_position
                # When lookback_position = -(KERNEL_SIZE-1) this is window_idx 0
                # (oldest). When lookback_position = -1 this is window_idx
                # KERNEL_SIZE-2 (newest).
                var window_idx = KERNEL_SIZE_MINUS_ONE + lookback_position
                if window_idx >= 0:
                    var state_flat_offset = (
                        UInt32(slot) * conv_state_pool_stride
                        + UInt32(conv_channel_idx) * conv_state_channel_stride
                        + UInt32(window_idx) * conv_state_window_stride
                    )
                    input_value = Float32(
                        conv_state._storage[state_flat_offset]
                    )

            conv_sum += input_value * Float32(weight_register[kernel_offset_k])

        var output_flat_offset = (
            UInt32(flat_token_idx) * conv_output_seqlen_stride
            + UInt32(conv_channel_idx) * conv_output_channel_stride
        )
        conv_output_ragged._storage[output_flat_offset] = Scalar[work_dtype](
            conv_sum
        )

    # ── Update conv_state: the last KERNEL_SIZE-1 raw input tokens ──────────
    # slot j should hold the raw input at position seq_len - (KERNEL_SIZE-1) + j.
    # If that position is negative (sequence shorter than KERNEL_SIZE-1), the
    # slot carries forward from the old conv_state at position
    # KERNEL_SIZE_MINUS_ONE + (seq_len - KERNEL_SIZE_MINUS_ONE + j) = seq_len + j.
    # This loop runs after the per-token loop, so all reads of the old window
    # have completed before any write to the same buffer takes place.
    comptime for state_slot_j in range(KERNEL_SIZE_MINUS_ONE):
        var source_position_in_sequence = (
            sequence_length - KERNEL_SIZE_MINUS_ONE + state_slot_j
        )

        # The pool stores state_dtype; sources may be work_dtype (qkv_input)
        # or state_dtype (carry-forward). Cast both into state_dtype on read,
        # then write at state_dtype.
        var state_value: Scalar[state_dtype] = 0
        if source_position_in_sequence >= 0:
            # Source token is within the current sequence
            var source_flat_offset = (
                UInt32(sequence_start_flat_idx + source_position_in_sequence)
                * qkv_input_seqlen_stride
                + UInt32(conv_channel_idx) * qkv_input_channel_stride
            )
            state_value = Scalar[state_dtype](
                qkv_input_ragged._storage[source_flat_offset]
            )
        else:
            # Source token is before the current sequence; carry from old state.
            # The old window position is
            #   KERNEL_SIZE_MINUS_ONE + source_position_in_sequence
            # (still negative-offset from the old window end).
            var old_window_idx = (
                KERNEL_SIZE_MINUS_ONE + source_position_in_sequence
            )
            if old_window_idx >= 0:
                var old_state_flat_offset = (
                    UInt32(slot) * conv_state_pool_stride
                    + UInt32(conv_channel_idx) * conv_state_channel_stride
                    + UInt32(old_window_idx) * conv_state_window_stride
                )
                state_value = conv_state._storage[old_state_flat_offset]

        var new_state_flat_offset = (
            UInt32(slot) * conv_state_pool_stride
            + UInt32(conv_channel_idx) * conv_state_channel_stride
            + UInt32(state_slot_j) * conv_state_window_stride
        )
        conv_state._storage[new_state_flat_offset] = state_value
