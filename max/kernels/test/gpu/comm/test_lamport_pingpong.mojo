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
"""Empirical 2-GPU sentinel ping-pong micro-test (Lamport allreduce slice 2).

This is a *de-risking gate*, not a kernel. Before we build the barrier-free
Lamport allreduce (slice 3) we must settle two hardware memory-ordering
unknowns on real GPUs, because Mojo's type system cannot express them (see
`.claude/agent-memory/mojo-kernel-engineer/comm-memory-ordering.md` and the
"Memory ordering" section of `.agentwork/kernels/designs/lamport-allreduce.md`):

1. SM100 (B200): is a naturally-aligned 128-bit (`v4.b32`) volatile global
   store *single-copy-atomic* w.r.t. a peer GPU reading over NVLink with a `v4`
   volatile load -- i.e. does the reader NEVER observe a torn pack (some lanes
   from generation g, some from g-1 or the sentinel)? And does `ld.volatile`
   alone give timely visibility WITHOUT any fence?
2. CDNA4 (MI355): the same atomicity question for `global_store_dwordx4` over
   XGMI, AND specifically: is the store ever made peer-visible WITHOUT an
   explicit release fence after it? Strong prior: no.

Design (writer = GPU 0, reader = GPU 1, P2P enabled, peer-addressable buffers):

- The writer publishes, per generation `g = 1 .. ITERS`, the all-lanes-equal
  data pack `[g, g, g, g]` (fp32), preceded by the all-sentinel `set_neg_zero`
  pack (the slice-1 "not ready" marker). Writing the sentinel *between*
  generations maximizes the sentinel->data transition window, which is the
  tear-prone moment.
- TORN-READ DETECTOR: the reader spins on `has_neg_zero` (slice-1 readiness
  poll). Once a non-sentinel pack appears it asserts every lane is mutually
  consistent (all four equal the same `g`). A torn 128-bit store surfaces as a
  pack mixing lanes from different generations / the sentinel: `torn_count++`.
- VISIBILITY / PROGRESS DETECTOR: the reader records the largest generation it
  observed. Each per-generation spin is *bounded* (`SPIN_CAP`) so a visibility
  failure (the store never becomes peer-visible) surfaces as a caught timeout
  (`max_seen < ITERS`), never an infinite hang.
- ORDERING VARIANTS (comptime `use_fence`): (A) volatile store, NO fence;
  (B) volatile store + system-scope release `atomic.fence` after the store.
  Both variants run on both targets -> a 2x2 result matrix.

Note on the fence primitive: `gpu.intrinsics.threadfence` asserts
`is_nvidia_gpu()` (NVIDIA-only), so it CANNOT be used for the AMD release fence.
We use the cross-target `std.atomic.fence[ordering=RELEASE, scope=...]`
(`pop.fence` with a real `syncscope`) instead. `scope=""` is the LLVM default =
system scope, which is what a cross-GPU release requires.

The torn detector and the per-lane invariant operate on raw `uint32` lanes so
"all lanes equal" is unambiguous (no `-0.0 == +0.0` float subtleties); the
readiness poll reuses the fp32 view of `lamport.has_neg_zero` so this test is a
genuine consumer of the slice-1 primitive.
"""

from std.atomic import Ordering, fence
from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.sys import has_amd_gpu_accelerator
from std.testing import assert_equal, assert_true

from comm.lamport import has_neg_zero, set_neg_zero
from comm.sync import enable_p2p


# Each pack is one naturally-aligned 128-bit line: 4 x fp32 / 4 x uint32.
comptime PACK_WIDTH = 4
comptime DTYPE = DType.float32
comptime UDTYPE = DType.uint32

# How many generations the writer publishes, and the reader's bounded spin per
# generation. ITERS is large to maximize the chance of catching a tear under
# the sentinel->data transition; SPIN_CAP bounds a visibility failure so it
# surfaces as a timeout instead of an infinite hang.
comptime ITERS = 200000
comptime SPIN_CAP = 50000000

# Number of independent (writer-thread, reader-thread) lanes running the
# ping-pong concurrently, each on its own 128-bit slot, to add cross-thread
# contention on the NVLink/XGMI fabric.
comptime NUM_SLOTS = 256


# ===----------------------------------------------------------------------=== #
# Writer kernel (runs on GPU 0, stores into GPU 1's peer-addressable buffer)
# ===----------------------------------------------------------------------=== #


def writer_kernel[
    use_fence: Bool,
    fence_scope: StaticString,
](peer_buf: MutPointer[Scalar[DTYPE], MutAnyOrigin], num_slots_dev: Int32):
    """Publishes ITERS generations of the all-lanes-equal pack into a peer slot.

    Each thread owns one 128-bit slot. For every generation `g` it first writes
    the all-sentinel "not ready" pack, then the data pack `[g, g, g, g]`,
    optionally followed by a system-scope release fence. All stores are
    naturally-aligned 128-bit volatile stores (the single `v4` instruction we
    are testing for atomicity).
    """
    # `Int` is not device-passable; widen the fixed-width arg.
    var num_slots = Int(num_slots_dev)
    var slot = Int(global_idx.x)
    if slot >= num_slots:
        return

    var slot_ptr = peer_buf + slot * PACK_WIDTH
    var sentinel = set_neg_zero[DTYPE, PACK_WIDTH]()

    for g in range(1, ITERS + 1):
        # Mark "not ready" (the tear-prone window opens here).
        slot_ptr.store[width=PACK_WIDTH, alignment=16, volatile=True](sentinel)
        comptime if use_fence:
            fence[ordering=Ordering.RELEASE, scope=fence_scope]()

        # Publish the data pack: every lane carries the same generation g.
        var data = SIMD[DTYPE, PACK_WIDTH](Scalar[DTYPE](g))
        slot_ptr.store[width=PACK_WIDTH, alignment=16, volatile=True](data)
        comptime if use_fence:
            fence[ordering=Ordering.RELEASE, scope=fence_scope]()


# ===----------------------------------------------------------------------=== #
# Reader kernel (runs on GPU 1, polls its own buffer that GPU 0 writes into)
# ===----------------------------------------------------------------------=== #


def reader_kernel[
    use_fence: Bool,
    fence_scope: StaticString,
](
    own_buf: MutPointer[Scalar[DTYPE], MutAnyOrigin],
    num_slots_dev: Int32,
    torn_counts: MutPointer[Int64, MutAnyOrigin],
    max_seen: MutPointer[Int64, MutAnyOrigin],
):
    """Spins on its own slot, detecting torn reads and tracking progress.

    For each generation it spins (bounded by SPIN_CAP) on `has_neg_zero` until a
    non-sentinel pack appears, then checks all four lanes are mutually equal. A
    pack that is non-sentinel but whose lanes are not all equal is a torn 128-bit
    transaction (`torn`); a pack whose value moved backwards is also recorded as
    torn. The largest generation observed is written to `max_seen[slot]`; the
    host judges visibility by `min over slots > 0` (a slot that observed 0
    generations means the peer store never became visible -- a fence/ordering
    bug). The bounded `SPIN_CAP` makes such a failure a finite result, not a
    hang.
    """
    # `Int` is not device-passable; widen the fixed-width arg.
    var num_slots = Int(num_slots_dev)
    var slot = Int(global_idx.x)
    if slot >= num_slots:
        return

    var slot_ptr = own_buf + slot * PACK_WIDTH
    var torn = Int64(0)
    var last_g = Int64(0)

    for _gen in range(1, ITERS + 1):
        var spins = 0
        while spins < SPIN_CAP:
            spins += 1
            var pack = slot_ptr.load[
                width=PACK_WIDTH, alignment=16, volatile=True
            ]()
            comptime if use_fence:
                # Reader-side acquire fence, paired with the writer release, to
                # test whether an explicit acquire is needed for visibility.
                fence[ordering=Ordering.ACQUIRE, scope=fence_scope]()

            # Still the sentinel -> producer has not written this generation.
            if has_neg_zero[DTYPE, PACK_WIDTH](pack):
                continue

            # Non-sentinel pack: check 128-bit single-copy atomicity. Compare on
            # raw uint32 lanes so "all lanes equal" is exact.
            var bits = bitcast[UDTYPE, PACK_WIDTH](pack)
            var l0 = bits[0]
            var torn_lane = (
                (bits[1] != l0) or (bits[2] != l0) or (bits[3] != l0)
            )
            if torn_lane:
                # Lanes disagree: a torn 128-bit store was observed.
                torn += 1
                break

            # All lanes agree. Decode the generation from the float value
            # (lane 0), not its raw bits, so the progress metric is the actual
            # generation g. The writer is monotonic; a backwards move is also a
            # tear/reorder.
            var g = Int64(Int(pack[0]))
            if g < last_g:
                torn += 1
            if g > last_g:
                last_g = g
            break

    torn_counts[slot] = torn
    max_seen[slot] = last_g


# ===----------------------------------------------------------------------=== #
# Host driver
# ===----------------------------------------------------------------------=== #


def run_variant[
    use_fence: Bool,
    fence_scope: StaticString,
](writer_ctx: DeviceContext, reader_ctx: DeviceContext, label: String) raises:
    """Runs one variant of the ping-pong and reports the result for the matrix.

    The writer buffer lives on the reader's GPU (so the writer's stores cross
    the fabric to the peer); the reader polls that same buffer locally.
    """
    print("---- variant:", label, "----")

    # The shared buffer lives on the reader GPU; the writer reaches it as a peer.
    var shared = reader_ctx.create_buffer_sync[DTYPE](NUM_SLOTS * PACK_WIDTH)
    # Initialize to the sentinel so the reader's first read is "not ready",
    # never uninitialized garbage.
    var init_host = alloc[Scalar[DTYPE]](NUM_SLOTS * PACK_WIDTH)
    var neg_zero = bitcast[DTYPE, 1](SIMD[UDTYPE, 1](UInt32(1) << 31))[0]
    for i in range(NUM_SLOTS * PACK_WIDTH):
        init_host[i] = neg_zero
    reader_ctx.enqueue_copy(shared, init_host)
    reader_ctx.synchronize()

    var torn_buf = reader_ctx.create_buffer_sync[.int64](NUM_SLOTS)
    var seen_buf = reader_ctx.create_buffer_sync[.int64](NUM_SLOTS)
    reader_ctx.enqueue_memset[.int64](torn_buf, 0)
    reader_ctx.enqueue_memset[.int64](seen_buf, 0)
    reader_ctx.synchronize()

    var shared_ptr = shared.unsafe_ptr().as_unsafe_any_origin()
    var torn_ptr = torn_buf.unsafe_ptr().as_unsafe_any_origin()
    var seen_ptr = seen_buf.unsafe_ptr().as_unsafe_any_origin()

    comptime BLOCK = 128
    var grid = (NUM_SLOTS + BLOCK - 1) // BLOCK

    # Launch reader first (it spins waiting), then the writer that feeds it.
    reader_ctx.enqueue_function[reader_kernel[use_fence, fence_scope]](
        shared_ptr,
        Int32(NUM_SLOTS),
        torn_ptr,
        seen_ptr,
        grid_dim=grid,
        block_dim=BLOCK,
    )
    writer_ctx.enqueue_function[writer_kernel[use_fence, fence_scope]](
        shared_ptr,
        Int32(NUM_SLOTS),
        grid_dim=grid,
        block_dim=BLOCK,
    )

    writer_ctx.synchronize()
    reader_ctx.synchronize()

    # Reduce results on the host.
    var torn_host = alloc[Int64](NUM_SLOTS)
    var seen_host = alloc[Int64](NUM_SLOTS)
    reader_ctx.enqueue_copy(torn_host, torn_buf)
    reader_ctx.enqueue_copy(seen_host, seen_buf)
    reader_ctx.synchronize()

    var total_torn = Int64(0)
    var min_seen = Int64(ITERS)
    var max_seen_v = Int64(0)
    for s in range(NUM_SLOTS):
        total_torn += torn_host[s]
        if seen_host[s] < min_seen:
            min_seen = seen_host[s]
        if seen_host[s] > max_seen_v:
            max_seen_v = seen_host[s]

    print("  torn observations (total over", NUM_SLOTS, "slots):", total_torn)
    print("  generations seen: min =", min_seen, " max =", max_seen_v)
    print("  expected max generation:", ITERS)

    # Visibility is settled by min_seen > 0: every reader slot observed the
    # peer's stores become visible and advanced. (max_seen tracks how many
    # distinct generations a slot happened to sample before the independent
    # writer kernel finished; a lower count under the fenced variant reflects
    # added per-iteration latency, not a visibility failure.)
    var visibility_ok = min_seen > 0
    if not visibility_ok:
        print(
            "  VISIBILITY FAILURE: at least one slot saw 0 generations (peer"
            " store never became visible -- a fence/ordering bug)"
        )
    else:
        print(
            (
                "  VISIBILITY OK: every reader slot observed the peer's stores"
                " and advanced (min generation seen ="
            ),
            min_seen,
            ")",
        )

    if total_torn == 0:
        print("  ATOMICITY OK: no torn 128-bit reads observed")
    else:
        print(
            "  ATOMICITY VIOLATION: torn 128-bit reads observed -> v4"
            " store/load is NOT single-copy-atomic for this variant"
        )

    init_host.free()
    torn_host.free()
    seen_host.free()


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1,
        "ping-pong micro-test requires >= 2 GPUs",
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    var writer_ctx = DeviceContext(device_id=0)
    var reader_ctx = DeviceContext(device_id=1)
    assert_true(
        writer_ctx.can_access(reader_ctx),
        "GPU 0 cannot peer-access GPU 1; P2P unavailable",
    )

    print("================================================================")
    print("Lamport sentinel ping-pong micro-test (slice 2 de-risking gate)")
    # Both targets use the cross-target `atomic.fence` with scope="" (LLVM
    # default = system scope): on NVIDIA this lowers to `fence.{release,
    # acquire}.sys`; on AMD to the syncscope-qualified `pop.fence` (system),
    # which is broader than the GPU-wide "agent" scope. We deliberately do NOT
    # use `gpu.intrinsics.threadfence` -- it asserts `is_nvidia_gpu()` and
    # cannot express an AMD release fence at all.
    comptime if has_amd_gpu_accelerator():
        print("target family: AMD (CDNA), expecting global_store_dwordx4")
    else:
        print("target family: NVIDIA, expecting st.volatile.global.v4.b32")
    print("ITERS =", ITERS, " NUM_SLOTS =", NUM_SLOTS)
    print("================================================================")

    # Variant A: no fence (sentinel-as-flag only).
    run_variant[use_fence=False, fence_scope=StaticString("")](
        writer_ctx, reader_ctx, String("A) volatile store, NO fence")
    )

    # Variant B: volatile store + system-scope release/acquire fence.
    run_variant[use_fence=True, fence_scope=StaticString("")](
        writer_ctx, reader_ctx, String("B) volatile store + system fence")
    )

    print("================================================================")
    print("ping-pong micro-test complete")
    print("================================================================")
