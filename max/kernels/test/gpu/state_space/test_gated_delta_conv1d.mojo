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

from std.math import ceildiv

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
from state_space.gated_delta_conv1d import gated_delta_conv1d_fwd_gpu
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index, IndexList


def run_slot_indexed_gpu[
    work_dtype: DType,
    state_dtype: DType,
    KERNEL_SIZE: Int,
](
    batch_size: Int,
    total_seq_len: Int,
    conv_dim: Int,
    max_slots: Int,
    seq_lengths: IndexList,
    slot_assignments: IndexList,  # [batch_size] slot indices into the pool
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Run the slot-indexed conv1d kernel and check it against a CPU reference.

    Differences from the old gather/scatter path that this exercises:
      - The conv-state pool has shape [max_slots, conv_dim, K-1] and the
        kernel reads/writes slot ``slot_assignments[b]`` for batch item b,
        so slots not referenced by ``slot_assignments`` must remain
        untouched.
      - In-place mutation: there is no conv_state_out tensor.
    """
    comptime CONV1D_BLOCK_DIM = 128
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_3d = Layout.row_major[3]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var state_len = KERNEL_SIZE - 1

    # ── Host tensors ────────────────────────────────────────────────────────
    # qkv_input_ragged: [total_seq_len, conv_dim]
    var qkv_input_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var qkv_input_h = LayoutTensor[work_dtype, layout_2d, _](
        qkv_input_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, conv_dim)),
    )

    # conv_weight: [conv_dim, KERNEL_SIZE]
    var conv_weight_heap = ctx.enqueue_create_host_buffer[work_dtype](
        conv_dim * KERNEL_SIZE
    )
    var conv_weight_h = LayoutTensor[work_dtype, layout_2d, _](
        conv_weight_heap,
        RuntimeLayout[layout_2d].row_major(Index(conv_dim, KERNEL_SIZE)),
    )

    # Pool: [max_slots, conv_dim, K-1]. Filled with a recognisable pattern so
    # that any unintended write to the wrong slot is caught by the equality
    # check at the end.
    var pool_size = max_slots * conv_dim * state_len
    var conv_state_initial_h_heap = ctx.enqueue_create_host_buffer[state_dtype](
        pool_size
    )
    var conv_state_initial_h = LayoutTensor[state_dtype, layout_3d, _](
        conv_state_initial_h_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(max_slots, conv_dim, state_len)
        ),
    )
    rand[state_dtype](conv_state_initial_h.ptr, pool_size)

    # Slot assignments device buffer.
    var slot_idx_heap = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var slot_idx_h = LayoutTensor[.uint32, layout_1d, _](
        slot_idx_heap,
        RuntimeLayout[layout_1d].row_major(Index(batch_size)),
    )
    for b in range(batch_size):
        slot_idx_h.ptr.store(b, UInt32(slot_assignments[b]))

    # input_row_offsets: [batch_size + 1]
    var input_row_offsets_heap = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var input_row_offsets_h = LayoutTensor[.uint32, layout_1d, _](
        input_row_offsets_heap,
        RuntimeLayout[layout_1d].row_major(Index(batch_size + 1)),
    )
    var cumsum = 0
    input_row_offsets_h.ptr.store(0, UInt32(0))
    for b in range(batch_size):
        cumsum += seq_lengths[b]
        input_row_offsets_h.ptr.store(b + 1, UInt32(cumsum))

    var conv_output_gpu_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var conv_output_gpu_h = LayoutTensor[work_dtype, layout_2d, _](
        conv_output_gpu_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, conv_dim)),
    )

    var pool_after_gpu_heap = ctx.enqueue_create_host_buffer[state_dtype](
        pool_size
    )
    var pool_after_gpu_h = LayoutTensor[state_dtype, layout_3d, _](
        pool_after_gpu_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(max_slots, conv_dim, state_len)
        ),
    )

    rand[work_dtype](qkv_input_h.ptr, qkv_input_h.size())
    rand[work_dtype](conv_weight_h.ptr, conv_weight_h.size())

    # ── Device buffers ──────────────────────────────────────────────────────
    var qkv_input_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var conv_weight_device = ctx.enqueue_create_buffer[work_dtype](
        conv_dim * KERNEL_SIZE
    )
    var conv_state_device = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var slot_idx_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var conv_output_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )

    with ctx.push_context():
        ctx.enqueue_copy(qkv_input_device, qkv_input_h.ptr)
        ctx.enqueue_copy(conv_weight_device, conv_weight_h.ptr)
        ctx.enqueue_copy(conv_state_device, conv_state_initial_h.ptr)
        ctx.enqueue_copy(slot_idx_device, slot_idx_h.ptr)
        ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_h.ptr)

    var qkv_input_tt = TileTensor(
        qkv_input_device, row_major(total_seq_len, conv_dim)
    )
    var conv_weight_tt = TileTensor(
        conv_weight_device, row_major(conv_dim, KERNEL_SIZE)
    )
    var conv_state_tt = TileTensor(
        conv_state_device,
        row_major(max_slots, conv_dim, state_len),
    )
    var slot_idx_tt = TileTensor(slot_idx_device, row_major(batch_size))
    var input_row_offsets_tt = TileTensor(
        input_row_offsets_device, row_major(batch_size + 1)
    )
    var conv_output_tt = TileTensor(
        conv_output_device, row_major(total_seq_len, conv_dim)
    )

    var qkv_input_seqlen_stride: UInt32 = UInt32(conv_dim)
    var qkv_input_channel_stride: UInt32 = 1
    var conv_weight_channel_stride: UInt32 = UInt32(KERNEL_SIZE)
    var conv_weight_offset_stride: UInt32 = 1
    var conv_state_pool_stride: UInt32 = UInt32(conv_dim * state_len)
    var conv_state_channel_stride: UInt32 = UInt32(state_len)
    var conv_state_window_stride: UInt32 = 1
    var conv_output_seqlen_stride: UInt32 = UInt32(conv_dim)
    var conv_output_channel_stride: UInt32 = 1

    var compiled_func = ctx.compile_function[
        gated_delta_conv1d_fwd_gpu[
            work_dtype,
            state_dtype,
            KERNEL_SIZE,
            CONV1D_BLOCK_DIM,
            qkv_input_tt.LayoutType,
            conv_weight_tt.LayoutType,
            conv_state_tt.LayoutType,
            slot_idx_tt.LayoutType,
            input_row_offsets_tt.LayoutType,
            conv_output_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch_size),
            Int32(total_seq_len),
            Int32(conv_dim),
            qkv_input_tt,
            conv_weight_tt,
            conv_state_tt,
            slot_idx_tt,
            input_row_offsets_tt,
            conv_output_tt,
            qkv_input_seqlen_stride,
            qkv_input_channel_stride,
            conv_weight_channel_stride,
            conv_weight_offset_stride,
            conv_state_pool_stride,
            conv_state_channel_stride,
            conv_state_window_stride,
            conv_output_seqlen_stride,
            conv_output_channel_stride,
            grid_dim=(batch_size, ceildiv(conv_dim, CONV1D_BLOCK_DIM)),
            block_dim=(CONV1D_BLOCK_DIM,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(conv_output_gpu_h.ptr, conv_output_device)
        ctx.enqueue_copy(pool_after_gpu_h.ptr, conv_state_device)
    ctx.synchronize()

    # ── CPU reference: scalar gather/scatter to the same pool. ───────────────
    # Only the slots referenced by slot_assignments should change.
    var pool_ref_heap = ctx.enqueue_create_host_buffer[state_dtype](pool_size)
    var pool_ref_h = LayoutTensor[state_dtype, layout_3d, _](
        pool_ref_heap,
        RuntimeLayout[layout_3d].row_major(
            Index(max_slots, conv_dim, state_len)
        ),
    )
    for i in range(pool_size):
        pool_ref_h.ptr.store(i, conv_state_initial_h.ptr[i])

    var conv_output_ref_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var conv_output_ref_h = LayoutTensor[work_dtype, layout_2d, _](
        conv_output_ref_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, conv_dim)),
    )

    comptime KERNEL_SIZE_MINUS_ONE = KERNEL_SIZE - 1

    for b in range(batch_size):
        var slot = slot_assignments[b]
        var seq_start = Int(input_row_offsets_h.ptr.load(b))
        var seq_end = Int(input_row_offsets_h.ptr.load(b + 1))
        var seq_len = seq_end - seq_start

        for c in range(conv_dim):
            for t in range(seq_len):
                var conv_sum = Float32(0.0)
                comptime for k in range(KERNEL_SIZE):
                    var lookback = t - (KERNEL_SIZE_MINUS_ONE - k)
                    var input_value = Float32(0.0)
                    if lookback >= 0:
                        input_value = Float32(
                            qkv_input_h.ptr[
                                UInt32(seq_start + lookback)
                                * qkv_input_seqlen_stride
                                + UInt32(c) * qkv_input_channel_stride
                            ]
                        )
                    else:
                        var slot_pos = KERNEL_SIZE_MINUS_ONE + lookback
                        if slot_pos >= 0:
                            input_value = Float32(
                                pool_ref_h.ptr[
                                    UInt32(slot) * conv_state_pool_stride
                                    + UInt32(c) * conv_state_channel_stride
                                    + UInt32(slot_pos)
                                    * conv_state_window_stride
                                ]
                            )
                    var w = Float32(
                        conv_weight_h.ptr[
                            UInt32(c) * conv_weight_channel_stride
                            + UInt32(k) * conv_weight_offset_stride
                        ]
                    )
                    conv_sum = conv_sum + input_value * w

                conv_output_ref_h.ptr.store(
                    UInt32(seq_start + t) * conv_output_seqlen_stride
                    + UInt32(c) * conv_output_channel_stride,
                    Scalar[work_dtype](conv_sum),
                )

            # Update the slot's window with the last K-1 raw inputs (or
            # carry-forward if seq_len < K-1). Reads of the old window
            # complete before any write because the write loop runs after the
            # token loop.
            var old_window = Array[Scalar[state_dtype], KERNEL_SIZE_MINUS_ONE](
                fill=0
            )
            comptime for j in range(KERNEL_SIZE_MINUS_ONE):
                old_window[j] = pool_ref_h.ptr[
                    UInt32(slot) * conv_state_pool_stride
                    + UInt32(c) * conv_state_channel_stride
                    + UInt32(j) * conv_state_window_stride
                ]

            comptime for j in range(KERNEL_SIZE_MINUS_ONE):
                var src = seq_len - KERNEL_SIZE_MINUS_ONE + j
                var v: Scalar[state_dtype] = 0
                if src >= 0:
                    v = Scalar[state_dtype](
                        qkv_input_h.ptr[
                            UInt32(seq_start + src) * qkv_input_seqlen_stride
                            + UInt32(c) * qkv_input_channel_stride
                        ]
                    )
                else:
                    var old_slot = KERNEL_SIZE_MINUS_ONE + src
                    if old_slot >= 0:
                        v = old_window[old_slot]
                pool_ref_h.ptr.store(
                    UInt32(slot) * conv_state_pool_stride
                    + UInt32(c) * conv_state_channel_stride
                    + UInt32(j) * conv_state_window_stride,
                    v,
                )

    # ── Compare ──────────────────────────────────────────────────────────────
    for i in range(total_seq_len * conv_dim):
        assert_almost_equal(
            conv_output_gpu_h.ptr[i],
            conv_output_ref_h.ptr[i],
            rtol=rtol,
        )

    for i in range(pool_size):
        assert_almost_equal(
            pool_after_gpu_h.ptr[i],
            pool_ref_h.ptr[i],
            rtol=rtol,
        )


def test_slot_indexed_single_sequence_targets_chosen_slot() raises:
    """One-sequence smoke test: writes only to the slot named in slot_idx."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.float32, 4](
        batch_size=1,
        total_seq_len=5,
        conv_dim=8,
        max_slots=3,
        seq_lengths=Index(5),
        slot_assignments=Index(2),
        ctx=ctx,
    )


def test_slot_indexed_two_sequences_disjoint_slots() raises:
    """Two sequences hitting non-adjacent slots; bf16 pool, fp32 work."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 4](
        batch_size=2,
        total_seq_len=7,
        conv_dim=8,
        max_slots=4,
        seq_lengths=Index(4, 3),
        slot_assignments=Index(3, 0),
        ctx=ctx,
        rtol=0.05,
    )


def test_slot_indexed_short_sequence_carries_state_forward() raises:
    """When seq_len < K-1: window must carry forward from existing pool entry.
    """
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.float32, 4](
        batch_size=1,
        total_seq_len=2,
        conv_dim=8,
        max_slots=2,
        seq_lengths=Index(2),
        slot_assignments=Index(1),
        ctx=ctx,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
