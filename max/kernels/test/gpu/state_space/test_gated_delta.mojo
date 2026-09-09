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

import std.math
from std.math import sqrt

from max.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.random import rand
from state_space.gated_delta import gated_delta_recurrence_fwd_gpu
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index, IndexList


def run_slot_indexed_gpu[
    work_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
](
    batch_size: Int,
    total_seq_len: Int,
    num_value_heads: Int,
    num_key_heads: Int,
    max_slots: Int,
    seq_lengths: IndexList,
    slot_assignments: IndexList,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Run the slot-indexed recurrence kernel and check it against a CPU reference.

    Verifies (a) recurrence_output matches the scalar five-step recurrence and
    (b) only the pool slots named in ``slot_assignments`` are mutated; the
    remaining slots must equal their initial random fill.
    """
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var key_dim = num_key_heads * KEY_HEAD_DIM
    var value_dim = num_value_heads * VALUE_HEAD_DIM
    var conv_dim = key_dim * 2 + value_dim

    # ── Host tensors ────────────────────────────────────────────────────────
    # qkv_conv_output: [total_seq_len, conv_dim]
    var qkv_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var qkv_h = LayoutTensor[work_dtype, layout_2d, _](
        qkv_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, conv_dim)),
    )
    rand[work_dtype](qkv_h.ptr, qkv_h.size())

    # decay_per_token: [total_seq_len, num_value_heads], values in (0, 1)
    var decay_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    var decay_h = LayoutTensor[work_dtype, layout_2d, _](
        decay_heap,
        RuntimeLayout[layout_2d].row_major(
            Index(total_seq_len, num_value_heads)
        ),
    )
    rand[work_dtype](decay_h.ptr, decay_h.size())
    # Decay in (0, 1): use |x| / (|x| + 1) to keep values in (0, 1)
    for i in range(total_seq_len * num_value_heads):
        var v = abs(Float32(decay_h.ptr[i]))
        decay_h.ptr.store(i, Scalar[work_dtype](v / (v + Float32(1.0))))

    # beta_per_token: [total_seq_len, num_value_heads], values in (0, 1)
    var beta_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    var beta_h = LayoutTensor[work_dtype, layout_2d, _](
        beta_heap,
        RuntimeLayout[layout_2d].row_major(
            Index(total_seq_len, num_value_heads)
        ),
    )
    rand[work_dtype](beta_h.ptr, beta_h.size())
    # Beta in (0, 1): same trick
    for i in range(total_seq_len * num_value_heads):
        var v = abs(Float32(beta_h.ptr[i]))
        beta_h.ptr.store(i, Scalar[work_dtype](v / (v + Float32(1.0))))

    # input_row_offsets: [batch_size + 1]
    var offsets_heap = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var offsets_h = LayoutTensor[.uint32, layout_1d, _](
        offsets_heap,
        RuntimeLayout[layout_1d].row_major(Index(batch_size + 1)),
    )
    var cumsum = 0
    offsets_h.ptr.store(0, UInt32(0))
    for b in range(batch_size):
        cumsum += seq_lengths[b]
        offsets_h.ptr.store(b + 1, UInt32(cumsum))

    # Pool [max_slots, nv, KD, VD] zeroed so initial state for any slot is 0.
    var pool_size = max_slots * num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    var pool_initial_heap = ctx.enqueue_create_host_buffer[state_dtype](
        pool_size
    )
    var pool_initial_h = LayoutTensor[state_dtype, layout_4d, _](
        pool_initial_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(max_slots, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM)
        ),
    )
    for i in range(pool_size):
        pool_initial_h.ptr.store(i, Scalar[state_dtype](0))

    var slot_idx_heap = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var slot_idx_h = LayoutTensor[.uint32, layout_1d, _](
        slot_idx_heap,
        RuntimeLayout[layout_1d].row_major(Index(batch_size)),
    )
    for b in range(batch_size):
        slot_idx_h.ptr.store(b, UInt32(slot_assignments[b]))

    var recur_out_gpu_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * value_dim
    )
    var recur_out_gpu_h = LayoutTensor[work_dtype, layout_2d, _](
        recur_out_gpu_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, value_dim)),
    )
    var pool_after_gpu_heap = ctx.enqueue_create_host_buffer[state_dtype](
        pool_size
    )
    var pool_after_gpu_h = LayoutTensor[state_dtype, layout_4d, _](
        pool_after_gpu_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(max_slots, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM)
        ),
    )

    # ── Device buffers ──────────────────────────────────────────────────────
    var qkv_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var decay_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    var beta_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    var offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var pool_device = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var slot_idx_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var recur_out_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * value_dim
    )

    with ctx.push_context():
        ctx.enqueue_copy(qkv_device, qkv_h.ptr)
        ctx.enqueue_copy(decay_device, decay_h.ptr)
        ctx.enqueue_copy(beta_device, beta_h.ptr)
        ctx.enqueue_copy(offsets_device, offsets_h.ptr)
        ctx.enqueue_copy(pool_device, pool_initial_h.ptr)
        ctx.enqueue_copy(slot_idx_device, slot_idx_h.ptr)

    var qkv_tt = TileTensor(qkv_device, row_major(total_seq_len, conv_dim))
    var decay_tt = TileTensor(
        decay_device, row_major(total_seq_len, num_value_heads)
    )
    var beta_tt = TileTensor(
        beta_device, row_major(total_seq_len, num_value_heads)
    )
    var offsets_tt = TileTensor(offsets_device, row_major(batch_size + 1))
    var pool_tt = TileTensor(
        pool_device,
        row_major(
            max_slots,
            num_value_heads,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
        ),
    )
    var slot_idx_tt = TileTensor(slot_idx_device, row_major(batch_size))
    var recur_out_tt = TileTensor(
        recur_out_device, row_major(total_seq_len, value_dim)
    )

    var qkv_seqlen_stride: UInt32 = UInt32(conv_dim)
    var qkv_channel_stride: UInt32 = 1
    var per_token_seqlen_stride: UInt32 = UInt32(num_value_heads)
    var per_token_head_stride: UInt32 = 1
    var pool_slot_stride: UInt32 = UInt32(
        num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    )
    var pool_value_head_stride: UInt32 = UInt32(KEY_HEAD_DIM * VALUE_HEAD_DIM)
    var pool_key_dim_stride: UInt32 = UInt32(VALUE_HEAD_DIM)
    var pool_value_dim_stride: UInt32 = 1
    var output_seqlen_stride: UInt32 = UInt32(value_dim)
    var output_valuedim_stride: UInt32 = 1

    # One CTA per (batch, value_head); the block has VALUE_HEAD_DIM threads.
    var num_blocks = batch_size * num_value_heads

    var compiled_func = ctx.compile_function[
        gated_delta_recurrence_fwd_gpu[
            work_dtype,
            state_dtype,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            recur_out_tt.LayoutType,
            qkv_tt.LayoutType,
            decay_tt.LayoutType,
            beta_tt.LayoutType,
            pool_tt.LayoutType,
            slot_idx_tt.LayoutType,
            offsets_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch_size),
            Int32(num_value_heads),
            Int32(num_key_heads),
            Int32(key_dim),
            recur_out_tt,
            pool_tt,
            slot_idx_tt,
            qkv_tt,
            decay_tt,
            beta_tt,
            offsets_tt,
            qkv_seqlen_stride,
            qkv_channel_stride,
            per_token_seqlen_stride,
            per_token_head_stride,
            pool_slot_stride,
            pool_value_head_stride,
            pool_key_dim_stride,
            pool_value_dim_stride,
            output_seqlen_stride,
            output_valuedim_stride,
            grid_dim=(num_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(recur_out_gpu_h.ptr, recur_out_device)
        ctx.enqueue_copy(pool_after_gpu_h.ptr, pool_device)
    ctx.synchronize()

    # ── CPU reference: scalar implementation of the five-step gated delta rule ─
    # Mirrors the GPU kernel logic exactly, iterating over every
    # (batch, value_head, vd_element) thread and every token.
    var pool_ref_heap = ctx.enqueue_create_host_buffer[state_dtype](pool_size)
    var pool_ref_h = LayoutTensor[state_dtype, layout_4d, _](
        pool_ref_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(max_slots, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM)
        ),
    )
    for i in range(pool_size):
        pool_ref_h.ptr.store(i, pool_initial_h.ptr[i])

    var recur_out_ref_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * value_dim
    )
    var recur_out_ref_h = LayoutTensor[work_dtype, layout_2d, _](
        recur_out_ref_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, value_dim)),
    )

    var heads_expansion_ratio = num_value_heads // num_key_heads
    var query_scale = Float32(1.0) / sqrt(Float32(KEY_HEAD_DIM))

    for b in range(batch_size):
        var slot = slot_assignments[b]
        var seq_start = Int(offsets_h.ptr.load(b))
        var seq_end = Int(offsets_h.ptr.load(b + 1))
        var seq_len = seq_end - seq_start

        for vh in range(num_value_heads):
            var kh = vh // heads_expansion_ratio
            for vd in range(VALUE_HEAD_DIM):
                # Load initial state column for this (batch, value_head, vd_element)
                var state_col = SIMD[.float32, KEY_HEAD_DIM](0.0)
                comptime for kd in range(KEY_HEAD_DIM):
                    state_col[kd] = Float32(
                        pool_ref_h.ptr[
                            UInt32(slot) * pool_slot_stride
                            + UInt32(vh) * pool_value_head_stride
                            + UInt32(kd) * pool_key_dim_stride
                            + UInt32(vd) * pool_value_dim_stride
                        ]
                    )

                var q_base = UInt32(kh * KEY_HEAD_DIM)
                var k_base = UInt32(key_dim + kh * KEY_HEAD_DIM)
                var v_off_const = UInt32(2 * key_dim + vh * VALUE_HEAD_DIM + vd)

                for t in range(seq_len):
                    var token = seq_start + t
                    var token_row = UInt32(token) * qkv_seqlen_stride

                    # Load Q, K raw vectors and accumulate squared norms
                    var q_raw = SIMD[.float32, KEY_HEAD_DIM](0.0)
                    var k_raw = SIMD[.float32, KEY_HEAD_DIM](0.0)
                    var q_sq = Float32(0.0)
                    var k_sq = Float32(0.0)
                    comptime for kd in range(KEY_HEAD_DIM):
                        var q_val = Float32(
                            qkv_h.ptr[
                                token_row
                                + (q_base + UInt32(kd)) * qkv_channel_stride
                            ]
                        )
                        var k_val = Float32(
                            qkv_h.ptr[
                                token_row
                                + (k_base + UInt32(kd)) * qkv_channel_stride
                            ]
                        )
                        q_raw[kd] = q_val
                        k_raw[kd] = k_val
                        q_sq = q_sq + q_val * q_val
                        k_sq = k_sq + k_val * k_val

                    # L2 normalise Q (also scaled) and K
                    var q_inv = Float32(1.0) / sqrt(q_sq + Float32(1e-6))
                    var k_inv = Float32(1.0) / sqrt(k_sq + Float32(1e-6))
                    var q_ns = SIMD[.float32, KEY_HEAD_DIM](0.0)
                    var k_n = SIMD[.float32, KEY_HEAD_DIM](0.0)
                    comptime for kd in range(KEY_HEAD_DIM):
                        q_ns[kd] = q_raw[kd] * q_inv * query_scale
                        k_n[kd] = k_raw[kd] * k_inv

                    # Load V element
                    var v_elem = Float32(
                        qkv_h.ptr[token_row + v_off_const * qkv_channel_stride]
                    )
                    # Load decay and beta
                    var head_off = (
                        UInt32(token) * per_token_seqlen_stride
                        + UInt32(vh) * per_token_head_stride
                    )
                    var dec = Float32(decay_h.ptr[head_off])
                    var bet = Float32(beta_h.ptr[head_off])

                    # Step 1+2: decay state, accumulate kv_memory
                    var kv_mem = Float32(0.0)
                    comptime for kd in range(KEY_HEAD_DIM):
                        state_col[kd] = state_col[kd] * dec
                        kv_mem = kv_mem + state_col[kd] * k_n[kd]

                    # Step 3: delta correction
                    var delta = bet * (v_elem - kv_mem)

                    # Step 4+5: update state, read out output
                    var out_val = Float32(0.0)
                    comptime for kd in range(KEY_HEAD_DIM):
                        state_col[kd] = state_col[kd] + k_n[kd] * delta
                        out_val = out_val + state_col[kd] * q_ns[kd]

                    recur_out_ref_h.ptr.store(
                        UInt32(token) * output_seqlen_stride
                        + UInt32(vh * VALUE_HEAD_DIM + vd)
                        * output_valuedim_stride,
                        Scalar[work_dtype](out_val),
                    )

                # Write final state column
                comptime for kd in range(KEY_HEAD_DIM):
                    pool_ref_h.ptr.store(
                        UInt32(slot) * pool_slot_stride
                        + UInt32(vh) * pool_value_head_stride
                        + UInt32(kd) * pool_key_dim_stride
                        + UInt32(vd) * pool_value_dim_stride,
                        Scalar[state_dtype](state_col[kd]),
                    )

    # ── Compare GPU vs CPU ─────────────────────────────────────────────────────
    for i in range(total_seq_len * value_dim):
        assert_almost_equal(
            recur_out_gpu_h.ptr[i], recur_out_ref_h.ptr[i], rtol=rtol
        )
    for i in range(pool_size):
        assert_almost_equal(
            pool_after_gpu_h.ptr[i], pool_ref_h.ptr[i], rtol=rtol
        )


def test_slot_indexed_single_sequence_targets_chosen_slot() raises:
    """Single sequence, KD=VD=128: writes only to slot 1 of a 3-slot pool."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.float32, 128, 128](
        batch_size=1,
        total_seq_len=4,
        num_value_heads=1,
        num_key_heads=1,
        max_slots=3,
        seq_lengths=Index(4),
        slot_assignments=Index(1),
        ctx=ctx,
    )


def test_slot_indexed_gqa_two_sequences() raises:
    """GQA (nv=2 nk=1), two sequences hitting non-adjacent slots, bf16 pool."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 128, 128](
        batch_size=2,
        total_seq_len=5,
        num_value_heads=2,
        num_key_heads=1,
        max_slots=4,
        seq_lengths=Index(3, 2),
        slot_assignments=Index(3, 0),
        ctx=ctx,
        rtol=0.05,
    )


def test_decode_gqa_batch() raises:
    """Decode shape (seq_len 1 per request), GQA ratio 2, multi-request batch
    hitting distinct slots — mirrors Qwen3.5 decode."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 128, 128](
        batch_size=4,
        total_seq_len=4,
        num_value_heads=8,
        num_key_heads=4,
        max_slots=8,
        seq_lengths=Index(1, 1, 1, 1),
        slot_assignments=Index(5, 1, 7, 2),
        ctx=ctx,
        rtol=0.05,
    )


def test_prefill_gqa_multi_seq() raises:
    """Prefill shape (longer seqs), GQA ratio 2, two requests in distinct slots.
    """
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 128, 128](
        batch_size=2,
        total_seq_len=24,
        num_value_heads=8,
        num_key_heads=4,
        max_slots=4,
        seq_lengths=Index(16, 8),
        slot_assignments=Index(2, 0),
        ctx=ctx,
        rtol=0.05,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
