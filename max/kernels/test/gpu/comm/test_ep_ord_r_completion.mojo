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
"""Two-GPU protocol test for rank-level EP completion (`ep_ord_r`).

Rank-level completion lets a fused consumer acquire ONCE per source rank
instead of checking each expert's word. That is only sound if the elected
producer's single system-scope release carries every other producer's state --
which the `ACQUIRE_RELEASE` election chain establishes. This test runs the
protocol across REAL peer-mapped memory on two GPUs, both directions:

  - producers on the SOURCE GPU write payload markers and per-expert count
    words directly into the DESTINATION GPU's buffers (true P2P stores);
  - each producer publishes its count word before joining the election;
  - the elected last producer release-stores the dedicated rank flag with the
    frozen sentinel discipline (flag value == n_experts_per_device);
  - the consumer on the DESTINATION GPU spins with system-scope ACQUIRE on the
    rank flag only, then validates every count word and payload marker with no
    further acquire, and restores the `MAX_FINITE` sentinel -- the
    destination-restore rule of the frozen protocol;
  - repeated back-to-back generations reuse the counter without host resets
    between generations (the elected producer restores it).

Per-expert words are progress data only; the consumer never waits on them.
"""

from comm.sync import enable_p2p
from max.gpu.host import DeviceBuffer, DeviceContext
from shmem.ep_comm import ep_signal_completion
from std.atomic import Atomic, Ordering
from max.gpu import block_idx, thread_idx
from std.testing import assert_equal, assert_true

comptime N_EXPERTS = 16  # producers per direction = destination local experts
comptime PAYLOAD_PER_EXPERT = 32
comptime RANK_FLAG_OFFSET = N_EXPERTS  # dedicated tail word past the grid
comptime SENTINEL = UInt64.MAX_FINITE
comptime FLAG_VALUE = UInt64(N_EXPERTS)  # frozen: n_experts_per_device
comptime N_GENERATIONS = 8


def producer_kernel(
    peer_recv_count: UnsafePointer[UInt64, MutUntrackedOrigin],
    peer_payload: UnsafePointer[UInt64, MutUntrackedOrigin],
    rank_completion_counter: UnsafePointer[Int32, MutUntrackedOrigin],
    generation: UInt64,
):
    """One single-thread block per expert on the SOURCE GPU (peer writes).

    Payload precedes the election in this thread's program order, so it sits
    in the producer's release set exactly as the protocol requires -- the
    cross-warp ordering production adds on top is the monitor ACQUIRE chain,
    which is out of scope for this primitive test.
    """
    var expert = Int(block_idx.x)

    # Unique per-(generation, expert, slot) payload markers, stored directly
    # into the destination GPU's buffer, BEFORE the election.
    for i in range(PAYLOAD_PER_EXPERT):
        peer_payload[expert * PAYLOAD_PER_EXPERT + i] = (
            generation * 100_000 + UInt64(expert) * 100 + UInt64(i)
        )

    if thread_idx.x == 0:
        var ptrs = Array[UnsafePointer[UInt64, MutUntrackedOrigin], 1](
            fill=peer_recv_count
        )
        # The 1-entry peer array already points AT the peer, so index 0 is
        # the true remote buffer.
        ep_signal_completion[
            False,
            n_experts_per_device=N_EXPERTS,
            has_rank_flag=True,
            ep_ord_r=True,
        ](
            Int32(0),
            Int32(0),
            ptrs,
            Int32(expert),
            generation * 100_000 + UInt64(expert),
            rank_completion_counter,
            Int32(RANK_FLAG_OFFSET),
        )


def consumer_kernel(
    recv_count: UnsafePointer[UInt64, MutUntrackedOrigin],
    payload: UnsafePointer[UInt64, MutUntrackedOrigin],
    result: UnsafePointer[Int32, MutUntrackedOrigin],
    generation: UInt64,
):
    """Destination GPU: one system-scope ACQUIRE on the rank flag only."""
    if thread_idx.x != 0:
        return

    # Spin exactly as the fused activation gate does: sentinel until the
    # elected producer's release lands from the peer GPU.
    var flag = Atomic[UInt64].load[ordering=Ordering.ACQUIRE](
        recv_count + RANK_FLAG_OFFSET
    )
    while flag == SENTINEL:
        flag = Atomic[UInt64].load[ordering=Ordering.ACQUIRE](
            recv_count + RANK_FLAG_OFFSET
        )

    var bad: Int32 = 0
    # Frozen flag value: n_experts_per_device, never the per-expert signal.
    if flag != FLAG_VALUE:
        bad += 1
    # After the single acquire, every per-expert count word and every peer
    # payload byte must be visible -- with NO further acquire and NO wait on
    # any per-expert word (progress data, never eligibility).
    for e in range(N_EXPERTS):
        if recv_count[e] != generation * 100_000 + UInt64(e):
            bad += 1
        for i in range(PAYLOAD_PER_EXPERT):
            var want = generation * 100_000 + UInt64(e) * 100 + UInt64(i)
            if payload[e * PAYLOAD_PER_EXPERT + i] != want:
                bad += 1
    result[0] = bad

    # Destination-restore rule (frozen protocol): the CONSUMER restores the
    # sentinel after consuming the dispatch, so a stale flag can never satisfy
    # the next generation's wait.
    Atomic[UInt64].store[ordering=Ordering.RELEASE](
        recv_count + RANK_FLAG_OFFSET, SENTINEL
    )


def _run_direction(
    src: DeviceContext, dst: DeviceContext, label: StaticString
) raises:
    """Producers on `src` write into `dst`'s memory; consumer runs on `dst`."""
    # Buffers live on the DESTINATION; the source kernel receives raw peer
    # pointers to them (real P2P addresses, not local aliases).
    var recv_count = dst.enqueue_create_buffer[DType.uint64](N_EXPERTS + 1)
    var payload = dst.enqueue_create_buffer[DType.uint64](
        N_EXPERTS * PAYLOAD_PER_EXPERT
    )
    var result = dst.enqueue_create_buffer[DType.int32](1)
    # The election counter is SOURCE-local, exactly as in production.
    var counter = src.enqueue_create_buffer[DType.int32](1)

    dst.enqueue_memset(recv_count, SENTINEL)
    src.enqueue_memset(counter, Int32(0))
    dst.synchronize()
    src.synchronize()

    for gen in range(1, N_GENERATIONS + 1):
        # Per-expert words are re-armed by the host here for test brevity; the
        # rank flag is NOT -- its sentinel restore is the consumer's job
        # (asserted below, and by the next generation's wait working at all).
        with recv_count.map_to_host() as h:
            for e in range(N_EXPERTS):
                h[e] = SENTINEL
        dst.enqueue_memset(payload, UInt64(0))
        dst.enqueue_memset(result, Int32(-1))
        dst.synchronize()

        src.enqueue_function[producer_kernel](
            recv_count.unsafe_ptr(),
            payload.unsafe_ptr(),
            counter.unsafe_ptr(),
            UInt64(gen),
            grid_dim=N_EXPERTS,
            block_dim=1,
        )
        dst.enqueue_function[consumer_kernel](
            recv_count.unsafe_ptr(),
            payload.unsafe_ptr(),
            result.unsafe_ptr(),
            UInt64(gen),
            grid_dim=1,
            block_dim=1,
        )
        src.synchronize()
        dst.synchronize()

        with result.map_to_host() as h:
            assert_equal(
                Int(h[0]),
                0,
                String(label)
                + " generation "
                + String(gen)
                + ": consumer acquired the rank flag but observed stale or"
                " missing count/payload state",
            )
        # The consumer must have restored the sentinel (destination-restore).
        with recv_count.map_to_host() as h:
            assert_true(
                h[RANK_FLAG_OFFSET] == SENTINEL,
                String(label)
                + " generation "
                + String(gen)
                + ": rank-flag sentinel was not restored by the consumer",
            )
        # The elected producer must have reset the source-local counter.
        with counter.map_to_host() as h:
            assert_equal(
                Int(h[0]),
                0,
                String(label)
                + " generation "
                + String(gen)
                + ": election counter was not reset for reuse",
            )


def test_ord_r_two_gpu_bidirectional() raises:
    """Real peer-mapped ORD-R, both directions, repeated generations."""
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)
    _run_direction(ctx0, ctx1, "GPU0->GPU1")
    _run_direction(ctx1, ctx0, "GPU1->GPU0")


def test_rank_flag_is_distinct_from_expert_grid() raises:
    """Per-expert words are progress only; the flag is a word of its own."""
    assert_true(
        RANK_FLAG_OFFSET >= N_EXPERTS,
        "the rank flag must live outside the per-expert count grid",
    )


def main() raises:
    # Direct calls, not TestSuite discovery: discovery host-instantiates every
    # module function, and the GPU kernels here are device-only.
    test_rank_flag_is_distinct_from_expert_grid()
    test_ord_r_two_gpu_bidirectional()
    print("test_ep_ord_r_completion: all checks passed")
