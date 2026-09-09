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
"""Gated DeltaNet recurrence kernel for Qwen3.5: Pass 2 of two-pass prefill.

Implements the gated delta rule recurrence over a ragged (variable-length)
batch of sequences. This is Pass 2 of the prefill path; it consumes the
conv1d output produced by Pass 1 (gated_delta_conv1d_fwd).

The five steps of the gated delta rule at each token t for value-dim element
vd_i and value head h are:

  1. Apply per-head scalar decay to the entire state column:
       state_col[k]  ←  decay[t,h] * state_col[k]    for k in [0, KD)

  2. Compute kv_memory by taking the dot product of the decayed state column
     with the L2-normalised key vector (summing over the key_dim axis):
       kv_memory_vd_i  =  Σ_k  state_col[k] * key_normalised[t,h,k]

  3. Compute the delta correction using beta and the value residual:
       delta_correction_vd_i  =  beta[t,h] * (value[t,h,vd_i] - kv_memory_vd_i)

  4. Outer-product update of the state column with the key and delta:
       state_col[k]  ←  state_col[k]  +  key_normalised[t,h,k] * delta_correction_vd_i

  5. Read out the output by dotting the updated state with the scaled,
     L2-normalised query vector:
       output[t, h*VD + vd_i]  =  Σ_k  state_col[k] * query_scaled[t,h,k]

Thread mapping (GPU)
--------------------
One CTA owns one (batch_item, value_head); the block has VALUE_HEAD_DIM
threads. Thread `tid == vd_element` owns the KD-element state column

    state_col[k] = recurrent_state[slot_idx[batch_item], value_head, k, tid]

in registers and iterates over its sequence sequentially. KEY_HEAD_DIM is a
compile-time constant, so the inner k-loop is fully unrolled and the state
column lives in registers (no spill to local memory) across the whole
sequence.

  Grid  : (batch_size * num_value_heads,) 1-D
  Block : (VALUE_HEAD_DIM,) 1-D

The per-token raw Q and K vectors for this value head's key head are loaded
once per block into shared memory (one element per thread, coalesced), so the
KD reduction reads them from shared memory instead of every vd-thread
re-reading the same KD elements from global memory. L2 normalisation and the
1/sqrt(KD) query scale are folded in as scalars factored out of the KD
reductions, so no normalised Q/K array is materialised.

GQA (grouped query attention) is handled by computing the key head index as:
  key_head_idx = value_head_idx // heads_expansion_ratio

where heads_expansion_ratio = num_value_heads / num_key_heads is a runtime
integer, so no compile-time specialisation per model is required.

Tensor shapes
-------------
Inputs:
  qkv_conv_output    : [total_seq_len, conv_dim]              float32
      Conv1d output from Pass 1. Channel layout:
        Q: channels [0, key_dim)
        K: channels [key_dim, 2*key_dim)
        V: channels [2*key_dim, 2*key_dim + value_dim)
      where key_dim  = num_key_heads  * key_head_dim
            value_dim = num_value_heads * value_head_dim
            conv_dim  = key_dim * 2 + value_dim
  decay_per_token    : [total_seq_len, num_value_heads]        float32
      Per-token, per-head scalar decay factor (exp(-softplus) pre-applied).
  beta_per_token     : [total_seq_len, num_value_heads]        float32
      Per-token, per-head beta gate (sigmoid pre-applied).
  recurrent_state    : [max_slots, num_value_heads, key_head_dim, value_head_dim]
      Mutable recurrent-state pool. The kernel reads/writes slot
      `slot_idx[batch_item]` in place; all other slots are untouched.
      Pool dtype is independent of the working dtype, so the caller can
      keep per-token tensors at float32 while storing the pool at the
      model's native dtype (bfloat16).
  slot_idx           : [batch_size]                            uint32
      Pool slot index for each batch item.
  input_row_offsets  : [batch_size + 1]                        uint32
      Ragged offsets: sequence b spans flat indices
      [input_row_offsets[b], input_row_offsets[b+1]).

Outputs:
  recurrence_output  : [total_seq_len, value_dim]              float32
      Flat output for all tokens. Indexed as
      output[flat_t, value_head_idx * value_head_dim + vd_element_idx].
  (recurrent_state is mutated in place; there is no separate state-out
   tensor.)
"""

import std.math
from max.gpu import (
    block_idx,
    thread_idx,
)
from max.gpu.sync import barrier
from std.math import rsqrt
from std.memory import unsafe_stack_allocation
from layout import TensorLayout, TileTensor


# ===----------------------------------------------------------------------=== #
# GPU Kernel
# ===----------------------------------------------------------------------=== #


def gated_delta_recurrence_fwd_gpu[
    work_dtype: DType,  # for qkv/decay/beta/recurrence_output (fp32)
    state_dtype: DType,  # for the recurrent_state pool (bf16)
    KEY_HEAD_DIM: Int,  # key_head_dim, compile-time (e.g. 128 for Qwen3.5)
    VALUE_HEAD_DIM: Int,  # value_head_dim, compile-time (e.g. 128 for Qwen3.5)
    recurrence_output_LT: TensorLayout,
    qkv_conv_output_LT: TensorLayout,
    decay_per_token_LT: TensorLayout,
    beta_per_token_LT: TensorLayout,
    recurrent_state_LT: TensorLayout,
    slot_idx_LT: TensorLayout,
    input_row_offsets_LT: TensorLayout,
](
    batch_size: Int32,
    num_value_heads: Int32,  # nv
    num_key_heads: Int32,  # nk; heads_expansion_ratio = nv / nk
    key_dim: Int32,  # num_key_heads * key_head_dim
    recurrence_output: TileTensor[
        work_dtype, recurrence_output_LT, MutUntrackedOrigin
    ],
    recurrent_state: TileTensor[
        state_dtype, recurrent_state_LT, MutUntrackedOrigin
    ],
    slot_idx: TileTensor[.uint32, slot_idx_LT, MutUntrackedOrigin],
    qkv_conv_output: TileTensor[
        work_dtype, qkv_conv_output_LT, MutUntrackedOrigin
    ],
    decay_per_token: TileTensor[
        work_dtype, decay_per_token_LT, MutUntrackedOrigin
    ],
    beta_per_token: TileTensor[
        work_dtype, beta_per_token_LT, MutUntrackedOrigin
    ],
    input_row_offsets: TileTensor[
        .uint32, input_row_offsets_LT, MutUntrackedOrigin
    ],
    # Strides for [total_seq_len, conv_dim] tensors
    qkv_conv_output_seqlen_stride: UInt32,
    qkv_conv_output_channel_stride: UInt32,
    # Strides for [total_seq_len, num_value_heads] tensors (decay, beta)
    per_token_seqlen_stride: UInt32,
    per_token_head_stride: UInt32,
    # Strides for [max_slots, nv, KD, VD] recurrent state pool.
    recurrent_state_slot_stride: UInt32,
    recurrent_state_value_head_stride: UInt32,
    recurrent_state_key_dim_stride: UInt32,
    recurrent_state_value_dim_stride: UInt32,
    # Strides for [total_seq_len, value_dim] recurrence output
    recurrence_output_seqlen_stride: UInt32,
    recurrence_output_valuedim_stride: UInt32,
):
    """GPU kernel: slot-indexed gated delta rule recurrence, one CTA per head.

    One CTA owns one (batch_item, value_head); thread `tid == vd_element` owns
    the KD-element state column ``recurrent_state[slot, value_head, :, tid]`` in
    registers for the whole sequence. The per-token raw Q/K for this value
    head's key head are staged once per block in shared memory (one element per
    thread, coalesced) so the KD reductions read them from shared memory rather
    than every vd-thread re-reading the same KD elements from global memory;
    L2 normalisation and the 1/sqrt(KD) query scale are folded in as scalars
    factored out of the reductions.

    Parameters:
        work_dtype: `DType` for the per-token input and output tensors
            (`qkv_conv_output`, `decay_per_token`, `beta_per_token`,
            `recurrence_output`), `float32`.
        state_dtype: `DType` for the `recurrent_state` pool (`bfloat16`).
        KEY_HEAD_DIM: Compile-time key head dimension (e.g. 128 for
            Qwen3.5).
        VALUE_HEAD_DIM: Compile-time value head dimension; must equal
            `KEY_HEAD_DIM`.
        recurrence_output_LT: `TensorLayout` for `recurrence_output`.
        qkv_conv_output_LT: `TensorLayout` for `qkv_conv_output`.
        decay_per_token_LT: `TensorLayout` for `decay_per_token`.
        beta_per_token_LT: `TensorLayout` for `beta_per_token`.
        recurrent_state_LT: `TensorLayout` for `recurrent_state`.
        slot_idx_LT: `TensorLayout` for `slot_idx`.
        input_row_offsets_LT: `TensorLayout` for
            `input_row_offsets`.

    Args:
        batch_size: Number of sequences in the ragged batch.
        num_value_heads: Number of value heads (`nv`).
        num_key_heads: Number of key heads (`nk`); the GQA expansion
            ratio is `num_value_heads / num_key_heads`.
        key_dim: Total key dimension, equal to
            `num_key_heads * KEY_HEAD_DIM`.
        recurrence_output: Flat output of shape
            `[total_seq_len, value_dim]` holding the recurrence result
            for every token.
        recurrent_state: Mutable state pool of shape
            `[max_slots, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM]`;
            the kernel reads and writes slot `slot_idx[batch_item]` in
            place.
        slot_idx: Pool slot index for each batch item, shape
            `[batch_size]`, `uint32`.
        qkv_conv_output: Conv1d output from Pass 1, shape
            `[total_seq_len, conv_dim]`, with Q in channels
            `[0, key_dim)`, K in `[key_dim, 2*key_dim)`, V in
            `[2*key_dim, 2*key_dim + value_dim)`.
        decay_per_token: Per-token per-head scalar decay factor, shape
            `[total_seq_len, num_value_heads]`.
        beta_per_token: Per-token per-head beta gate, shape
            `[total_seq_len, num_value_heads]`.
        input_row_offsets: Ragged offsets of shape `[batch_size + 1]`;
            sequence `b` spans flat indices
            `[input_row_offsets[b], input_row_offsets[b+1])`.
        qkv_conv_output_seqlen_stride: Stride between consecutive
            sequence positions in `qkv_conv_output`.
        qkv_conv_output_channel_stride: Stride between consecutive
            channels in `qkv_conv_output`.
        per_token_seqlen_stride: Stride between consecutive sequence
            positions in `decay_per_token` and `beta_per_token`.
        per_token_head_stride: Stride between consecutive heads in
            `decay_per_token` and `beta_per_token`.
        recurrent_state_slot_stride: Stride between consecutive slots
            in `recurrent_state`.
        recurrent_state_value_head_stride: Stride between consecutive
            value heads in `recurrent_state`.
        recurrent_state_key_dim_stride: Stride between consecutive
            key-dim elements in `recurrent_state`.
        recurrent_state_value_dim_stride: Stride between consecutive
            value-dim elements in `recurrent_state`.
        recurrence_output_seqlen_stride: Stride between consecutive
            sequence positions in `recurrence_output`.
        recurrence_output_valuedim_stride: Stride between consecutive
            value-dim elements in `recurrence_output`.
    """
    var _batch_size = Int(batch_size)
    var _num_value_heads = Int(num_value_heads)
    var _num_key_heads = Int(num_key_heads)
    var _key_dim = Int(key_dim)
    comptime assert (
        KEY_HEAD_DIM == VALUE_HEAD_DIM
    ), "gated_delta_recurrence_fwd_gpu requires KEY_HEAD_DIM == VALUE_HEAD_DIM"

    var tid = Int(thread_idx.x)
    var block = Int(block_idx.x)

    # ── block -> (batch_item, value_head) ───────────────────────────────────
    var batch_item_idx = block // _num_value_heads
    var value_head_idx = block % _num_value_heads
    if batch_item_idx >= _batch_size:
        return

    # GQA: map value head to key head.
    var heads_expansion_ratio = _num_value_heads // _num_key_heads
    var key_head_idx = value_head_idx // heads_expansion_ratio

    # Read the pool slot for this batch item exactly once. The caller
    # (`GatedDeltaNetStateCache.claim`) guarantees `slot < max_slots`.
    var slot = Int(slot_idx._storage[batch_item_idx])

    # Shared memory: raw Q and K for the current token (one element per kd).
    var q_raw_s = unsafe_stack_allocation[
        KEY_HEAD_DIM,
        Float32,
        address_space=.SHARED,
    ]()
    var k_raw_s = unsafe_stack_allocation[
        KEY_HEAD_DIM,
        Float32,
        address_space=.SHARED,
    ]()

    # ── Load this thread's KD-element state column from pool[slot, ...] ──────
    var state_col = SIMD[.float32, KEY_HEAD_DIM](0.0)
    comptime for kd in range(KEY_HEAD_DIM):
        var off = (
            UInt32(slot) * recurrent_state_slot_stride
            + UInt32(value_head_idx) * recurrent_state_value_head_stride
            + UInt32(kd) * recurrent_state_key_dim_stride
            + UInt32(tid) * recurrent_state_value_dim_stride
        )
        state_col[kd] = Float32(recurrent_state._storage[off])

    var sequence_start_flat_idx = Int(
        input_row_offsets._storage[batch_item_idx]
    )
    var sequence_end_flat_idx = Int(
        input_row_offsets._storage[batch_item_idx + 1]
    )
    var sequence_length = sequence_end_flat_idx - sequence_start_flat_idx

    # Precompute constant channel offsets for Q, K, V in the conv_dim layout.
    var query_channel_base = UInt32(key_head_idx * KEY_HEAD_DIM)
    var key_channel_base = UInt32(_key_dim + key_head_idx * KEY_HEAD_DIM)
    var value_channel = UInt32(
        2 * _key_dim + value_head_idx * VALUE_HEAD_DIM + tid
    )
    var query_scale = Float32(1.0) / std.math.sqrt(Float32(KEY_HEAD_DIM))

    for token_position_in_sequence in range(sequence_length):
        var flat_token_idx = (
            sequence_start_flat_idx + token_position_in_sequence
        )
        var token_qkv_row_offset = (
            UInt32(flat_token_idx) * qkv_conv_output_seqlen_stride
        )

        # ── Cooperative load of raw Q/K into SMEM (coalesced) ─────────────
        # One thread per kd; the block has VALUE_HEAD_DIM == KEY_HEAD_DIM
        # threads, so every kd element is covered.
        var q_off = (
            token_qkv_row_offset
            + (query_channel_base + UInt32(tid))
            * qkv_conv_output_channel_stride
        )
        var k_off = (
            token_qkv_row_offset
            + (key_channel_base + UInt32(tid)) * qkv_conv_output_channel_stride
        )
        q_raw_s[tid] = Float32(qkv_conv_output._storage[q_off])
        k_raw_s[tid] = Float32(qkv_conv_output._storage[k_off])
        barrier()

        # ── L2 norms from SMEM (shared across all vd-threads of this head) ─
        var q_squared_sum = Float32(0.0)
        var key_squared_sum = Float32(0.0)
        comptime for kd in range(KEY_HEAD_DIM):
            var qv = Float32(q_raw_s[kd])
            var kv = Float32(k_raw_s[kd])
            q_squared_sum = q_squared_sum + qv * qv
            key_squared_sum = key_squared_sum + kv * kv
        # Fold the query scale into the query inverse-norm scalar.
        var query_factor = rsqrt(q_squared_sum + Float32(1e-6)) * query_scale
        var key_inv_norm = rsqrt(key_squared_sum + Float32(1e-6))

        # ── V element (only this thread's vd column) ──────────────────────
        var value_element = Float32(
            qkv_conv_output._storage[
                token_qkv_row_offset
                + value_channel * qkv_conv_output_channel_stride
            ]
        )

        # ── Per-token decay and beta for this value head ──────────────────
        var head_token_offset = (
            UInt32(flat_token_idx) * per_token_seqlen_stride
            + UInt32(value_head_idx) * per_token_head_stride
        )
        var decay_value = Float32(decay_per_token._storage[head_token_offset])
        var beta_value = Float32(beta_per_token._storage[head_token_offset])

        # ── Step 1+2: decay state, accumulate kv_memory ───────────────────
        # kv_memory = key_inv_norm * Σ_k (decay·state_col[k]) · k_raw[k].
        var kv_raw = Float32(0.0)
        comptime for kd in range(KEY_HEAD_DIM):
            state_col[kd] = state_col[kd] * decay_value
            kv_raw = kv_raw + state_col[kd] * Float32(k_raw_s[kd])
        var kv_memory_value = kv_raw * key_inv_norm

        # ── Step 3: delta correction ──────────────────────────────────────
        var delta_correction_vd_i = beta_value * (
            value_element - kv_memory_value
        )
        # k_normalised[k]·delta = k_raw[k] · (key_inv_norm·delta).
        var key_update_factor = delta_correction_vd_i * key_inv_norm

        # ── Step 4+5: outer-product update, query readout ─────────────────
        # output = query_factor · Σ_k state_col[k] · q_raw[k].
        var out_raw = Float32(0.0)
        comptime for kd in range(KEY_HEAD_DIM):
            state_col[kd] = (
                state_col[kd] + Float32(k_raw_s[kd]) * key_update_factor
            )
            out_raw = out_raw + state_col[kd] * Float32(q_raw_s[kd])
        var output_value = out_raw * query_factor

        var recurrence_output_flat_offset = (
            UInt32(flat_token_idx) * recurrence_output_seqlen_stride
            + UInt32(value_head_idx * VALUE_HEAD_DIM + tid)
            * recurrence_output_valuedim_stride
        )
        recurrence_output._storage[recurrence_output_flat_offset] = Scalar[
            work_dtype
        ](output_value)

        # WAR: all reads of q_raw_s/k_raw_s must finish before the next
        # token's cooperative load overwrites them.
        barrier()

    # ── Write final state column back into pool[slot, ...] ──────────────────
    comptime for kd in range(KEY_HEAD_DIM):
        var off = (
            UInt32(slot) * recurrent_state_slot_stride
            + UInt32(value_head_idx) * recurrent_state_value_head_stride
            + UInt32(kd) * recurrent_state_key_dim_stride
            + UInt32(tid) * recurrent_state_value_dim_stride
        )
        recurrent_state._storage[off] = Scalar[state_dtype](state_col[kd])
