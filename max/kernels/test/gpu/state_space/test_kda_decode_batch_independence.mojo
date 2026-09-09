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
"""Batch independence of the KDA decode kernel (`kda_decode_gpu`).

A served request's tokens and recurrent state must not depend on which other
requests share its batch. This test is the kernel-level form of the serving
symptom it was written for: three identical prompts issued concurrently came
back with three different continuations, each correct for a few tokens and then
continuing as a batch-mate's.

The oracle is the kernel itself run one sequence at a time. Batched and
one-at-a-time launches read the same device buffers and differ only in
`cu_seqlens` / `state_indices` / grid size, so the per-thread arithmetic is
identical and the comparison is bit-exact, not tolerance-based -- any
difference is cross-sequence interference, never rounding.

Serving-shaped details the earlier tests did not cover, each of which is a way
the batch could leak:
  - the pool is a slot POOL: `max_slots` (8) > `batch_size` (4), so a
    pool-vs-batch dimension confusion is observable.
  - slots are permuted and non-contiguous (`[6, 2, 7, 0]`), so batch item
    `b` addressing slot `b` would fail.
  - mixed sequence lengths in one batch (continuous batching runs a prefill
    alongside decodes), and the all-length-1 pure decode step.
  - the batched launch is repeated: a race between CTAs shows up as
    run-to-run variation even when a single run happens to look right.

Geometry is Kimi-K3's (HV = H = 8, K = V = 32), bf16 q/k/v with an fp32 and a
bf16 state pool.
"""

import std.math
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_true
from max.gpu.host import DeviceContext

from layout import TileTensor, row_major
from kda.recurrent import kda_decode_gpu


comptime NUM_HEADS = 8
comptime HEAD_DIM = 32
comptime MAX_SLOTS = 8


struct _Batch:
    """Host-side inputs shared by the batched and one-at-a-time launches."""

    var seq_lengths: List[Int]
    var slots: List[Int]
    var total_T: Int

    def __init__(out self, var seq_lengths: List[Int], var slots: List[Int]):
        self.seq_lengths = seq_lengths^
        self.slots = slots^
        self.total_T = 0
        for i in range(len(self.seq_lengths)):
            self.total_T += self.seq_lengths[i]

    def batch_size(self) -> Int:
        return len(self.seq_lengths)

    def start_of(self, b: Int) -> Int:
        var s = 0
        for i in range(b):
            s += self.seq_lengths[i]
        return s


def _run[
    state_dtype: DType
](
    batch: _Batch,
    one_at_a_time: Bool,
    out_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    pool_h: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    ctx: DeviceContext,
) raises:
    """Runs the batch, writing the output and the whole post-run pool to host.

    With `one_at_a_time` the same work is issued as one launch per sequence
    (`batch_size == 1`, `cu_seqlens == [start_b, end_b]`), which is the
    oracle: a sequence that cannot see a batch-mate cannot be corrupted by
    one. Every other buffer is byte-identical between the two modes.
    """
    comptime H = NUM_HEADS
    comptime HV = NUM_HEADS
    comptime K = HEAD_DIM
    comptime V = HEAD_DIM

    var B = batch.batch_size()
    var total_T = batch.total_T
    var pool_size = MAX_SLOTS * HV * K * V

    var q_h = alloc[Scalar[DType.bfloat16]](total_T * H * K)
    var k_h = alloc[Scalar[DType.bfloat16]](total_T * H * K)
    var v_h = alloc[Scalar[DType.bfloat16]](total_T * HV * V)
    var rg_h = alloc[Scalar[DType.float32]](total_T * HV * K)
    var bl_h = alloc[Scalar[DType.float32]](total_T * HV)
    var al_h = alloc[Scalar[DType.float32]](HV)
    var dt_h = alloc[Scalar[DType.float32]](HV * K)
    var state_init_h = alloc[Scalar[state_dtype]](pool_size)

    for i in range(total_T * H * K):
        q_h[i] = Scalar[DType.bfloat16](
            std.math.sin(Float32(i + 1) * Float32(0.313))
        )
        k_h[i] = Scalar[DType.bfloat16](
            std.math.cos(Float32(i + 1) * Float32(0.217))
        )
    for i in range(total_T * HV * V):
        v_h[i] = Scalar[DType.bfloat16](
            std.math.sin(Float32(i + 7) * Float32(0.491)) * Float32(0.5)
        )
    for i in range(total_T * HV * K):
        rg_h[i] = std.math.cos(Float32(i + 3) * Float32(0.137)) * Float32(0.75)
    for i in range(total_T * HV):
        bl_h[i] = std.math.sin(Float32(i + 11) * Float32(0.229))
    for i in range(HV):
        al_h[i] = std.math.cos(Float32(i + 2) * Float32(0.401)) * Float32(0.5)
    for i in range(HV * K):
        dt_h[i] = std.math.sin(Float32(i + 5) * Float32(0.173)) * Float32(0.25)
    # Every slot starts non-zero, including the ones this batch does not use:
    # a kernel that walked off its slot would otherwise land on zeros and hide.
    for i in range(pool_size):
        state_init_h[i] = Scalar[state_dtype](
            std.math.sin(Float32(i + 13) * Float32(0.0917)) * Float32(0.3)
        )

    var q_dev = ctx.enqueue_create_buffer[DType.bfloat16](total_T * H * K)
    var k_dev = ctx.enqueue_create_buffer[DType.bfloat16](total_T * H * K)
    var v_dev = ctx.enqueue_create_buffer[DType.bfloat16](total_T * HV * V)
    var rg_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * K)
    var bl_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV)
    var al_dev = ctx.enqueue_create_buffer[DType.float32](HV)
    var dt_dev = ctx.enqueue_create_buffer[DType.float32](HV * K)
    var cu_dev = ctx.enqueue_create_buffer[DType.int32](B + 1)
    var si_dev = ctx.enqueue_create_buffer[DType.int32](B)
    var pool_dev = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](total_T * HV * V)

    var cu_h = alloc[Scalar[DType.int32]](B + 1)
    var si_h = alloc[Scalar[DType.int32]](B)

    with ctx.push_context():
        ctx.enqueue_copy(q_dev, q_h)
        ctx.enqueue_copy(k_dev, k_h)
        ctx.enqueue_copy(v_dev, v_h)
        ctx.enqueue_copy(rg_dev, rg_h)
        ctx.enqueue_copy(bl_dev, bl_h)
        ctx.enqueue_copy(al_dev, al_h)
        ctx.enqueue_copy(dt_dev, dt_h)
        ctx.enqueue_copy(pool_dev, state_init_h)
    out_dev.enqueue_fill(0.0)

    var q_tt = TileTensor(q_dev, row_major(total_T, H * K))
    var k_tt = TileTensor(k_dev, row_major(total_T, H * K))
    var v_tt = TileTensor(v_dev, row_major(total_T, HV * V))
    var rg_tt = TileTensor(rg_dev, row_major(total_T, HV * K))
    var bl_tt = TileTensor(bl_dev, row_major(total_T, HV))
    var al_tt = TileTensor(al_dev, row_major(HV))
    var dt_tt = TileTensor(dt_dev, row_major(HV, K))
    var out_tt = TileTensor(out_dev, row_major(total_T, HV * V))
    var pool_tt = TileTensor(pool_dev, row_major(MAX_SLOTS, HV, K, V))

    # One launch per (cu_seqlens, state_indices) pair: the whole batch, or one
    # sequence at a time. Distinct slots mean the one-at-a-time launches write
    # disjoint pool rows, so running them back to back on one pool is
    # equivalent to running each against a private pool.
    var num_launches = B if one_at_a_time else 1
    for launch in range(num_launches):
        var launch_B: Int
        if one_at_a_time:
            launch_B = 1
            cu_h[0] = Int32(batch.start_of(launch))
            cu_h[1] = Int32(batch.start_of(launch) + batch.seq_lengths[launch])
            si_h[0] = Int32(batch.slots[launch])
        else:
            launch_B = B
            var cumulative = 0
            cu_h[0] = Int32(0)
            for b in range(B):
                cumulative += batch.seq_lengths[b]
                cu_h[b + 1] = Int32(cumulative)
                si_h[b] = Int32(batch.slots[b])

        with ctx.push_context():
            ctx.enqueue_copy(cu_dev, cu_h)
            ctx.enqueue_copy(si_dev, si_h)

        var cu_tt = TileTensor(cu_dev, row_major(launch_B + 1))
        var si_tt = TileTensor(si_dev, row_major(launch_B))

        ctx.enqueue_function[
            kda_decode_gpu[
                DType.bfloat16,
                DType.float32,
                state_dtype,
                DType.float32,
                HEAD_DIM,
                HEAD_DIM,
                out_tt.LayoutType,
                q_tt.LayoutType,
                k_tt.LayoutType,
                v_tt.LayoutType,
                rg_tt.LayoutType,
                bl_tt.LayoutType,
                al_tt.LayoutType,
                dt_tt.LayoutType,
                cu_tt.LayoutType,
                pool_tt.LayoutType,
                si_tt.LayoutType,
                "original",
                "logits",
                "K_FIRST",
            ]
        ](
            Int32(launch_B),
            Int32(HV),
            Int32(H),
            out_tt,
            q_tt,
            k_tt,
            v_tt,
            rg_tt,
            bl_tt,
            al_tt,
            dt_tt,
            cu_tt,
            pool_tt,
            si_tt,
            UInt32(H * K),
            UInt32(K),
            UInt32(1),
            UInt32(H * K),
            UInt32(K),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            UInt32(HV * K),
            UInt32(K),
            UInt32(1),
            UInt32(HV),
            UInt32(1),
            UInt32(K),
            UInt32(1),
            UInt32(HV * K * V),
            UInt32(K * V),
            UInt32(V),
            UInt32(1),
            UInt32(HV * V),
            UInt32(V),
            UInt32(1),
            UInt32(1),
            grid_dim=(launch_B * HV,),
            block_dim=(HEAD_DIM,),
        )

    var pool_typed = alloc[Scalar[state_dtype]](pool_size)
    with ctx.push_context():
        ctx.enqueue_copy(out_h, out_dev)
        ctx.enqueue_copy(pool_typed, pool_dev)
    ctx.synchronize()
    for i in range(pool_size):
        pool_h[i] = Float32(pool_typed[i])

    pool_typed.free()
    q_h.free()
    k_h.free()
    v_h.free()
    rg_h.free()
    bl_h.free()
    al_h.free()
    dt_h.free()
    state_init_h.free()
    cu_h.free()
    si_h.free()


def _check[
    state_dtype: DType
](var seq_lengths: List[Int], var slots: List[Int], repeats: Int) raises:
    """Asserts the batch is bit-identical to the same sequences run alone."""
    comptime HV = NUM_HEADS
    comptime V = HEAD_DIM

    var batch = _Batch(seq_lengths^, slots^)
    var out_n = batch.total_T * HV * V
    var pool_n = MAX_SLOTS * HV * HEAD_DIM * HEAD_DIM

    var ctx = DeviceContext()

    var out_alone = alloc[Scalar[DType.float32]](out_n)
    var pool_alone = alloc[Scalar[DType.float32]](pool_n)
    _run[state_dtype](batch, True, out_alone, pool_alone, ctx)

    var out_batched = alloc[Scalar[DType.float32]](out_n)
    var pool_batched = alloc[Scalar[DType.float32]](pool_n)
    for repeat in range(repeats):
        _run[state_dtype](batch, False, out_batched, pool_batched, ctx)
        for i in range(out_n):
            assert_equal(
                out_batched[i],
                out_alone[i],
                String(
                    "output element ",
                    i,
                    " of batch repeat ",
                    repeat,
                    " differs from the same sequence run alone",
                ),
            )
        for i in range(pool_n):
            assert_equal(
                pool_batched[i],
                pool_alone[i],
                String(
                    "state-pool element ",
                    i,
                    " (slot ",
                    i // (HV * HEAD_DIM * HEAD_DIM),
                    ") of batch repeat ",
                    repeat,
                    " differs from the same sequence run alone",
                ),
            )

    out_alone.free()
    pool_alone.free()
    out_batched.free()
    pool_batched.free()


def test_decode_step_batch_of_four_fp32_state() raises:
    """A pure decode step: four requests, one token each, permuted slots."""
    comptime if has_accelerator():
        _check[DType.float32]([1, 1, 1, 1], [6, 2, 7, 0], 4)


def test_decode_step_batch_of_four_bf16_state() raises:
    """Same, with the bf16 state pool a served model may be configured with."""
    comptime if has_accelerator():
        _check[DType.bfloat16]([1, 1, 1, 1], [6, 2, 7, 0], 4)


def test_mixed_prefill_and_decode_batch_fp32_state() raises:
    """Continuous batching: a prefill and a chunk alongside decode steps."""
    comptime if has_accelerator():
        _check[DType.float32]([1, 3, 1, 5], [6, 2, 7, 0], 4)


def test_single_sequence_matches_itself_fp32_state() raises:
    """Control: batch of one must be a no-op difference (guards the harness)."""
    comptime if has_accelerator():
        _check[DType.float32]([4], [3], 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
