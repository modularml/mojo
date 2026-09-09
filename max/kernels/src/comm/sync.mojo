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

"""Signal-based GPU synchronization primitives for multi-GPU collective kernels.

Provides the `Signal` struct for counter-based cross-GPU thread-block
synchronization, peer-to-peer memory access helpers, and the barrier utilities
consumed by allreduce, scatter, and other collective operations.
"""

from std.collections import Array
from std.utils import StaticTuple
from std.math.uutils import umod
from std.sys import size_of, is_amd_gpu

from std.atomic import Atomic, Ordering
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu import (
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.sync import named_barrier
from .lamport import LAMPORT_SENTINEL_U32, Lamport, LamportGeneration


# No-op (currently) group operation functions (enables vendor_ccl drop in replacement)
def group_start():
    """Marks the start of a collective operation group.

    Currently a no-op that exists as a drop-in placeholder for vendor CCL
    (`ncclGroupStart` / `rcclGroupStart`). Wrapping a series of collective
    calls between `group_start()` and `group_end()` allows a future
    implementation to batch them as a single fused CCL operation without
    changing call sites.
    """
    return


def group_end():
    """Marks the end of a collective operation group.

    Currently a no-op that exists as a drop-in placeholder for vendor CCL
    (`ncclGroupEnd` / `rcclGroupEnd`). Wrapping a series of collective
    calls between `group_start()` and `group_end()` allows a future
    implementation to batch them as a single fused CCL operation without
    changing call sites.
    """
    return


def enable_p2p() -> Bool:
    """Enable peer-to-peer memory access between all GPU pairs if supported.

    Attempts to enable P2P access between all device pairs. Returns False
    if P2P is not supported by the hardware rather than raising.

    Returns:
        True if P2P access is enabled between all GPU pairs, False otherwise.
    """
    try:
        DeviceContext.enable_all_peer_access()
        return DeviceContext.all_peer_access_enabled()
    except:
        return False


def is_p2p_enabled() raises -> Bool:
    """Checks whether P2P access is available between GPUs.

    This is a read-only status check. Callers must ensure `enable_p2p()`
    has been called during initialization (e.g., during model setup or
    via `Signals.buffers()`) before relying on this function.

    Returns:
        True if P2P access is available between all GPU pairs, False otherwise.

    Raises:
        If the P2P access check fails.
    """
    return DeviceContext.all_peer_access_enabled()


# NOTE: the above result was true on A100, but on H100 we need more SMs to
# sature the NVLink in the bandwidth-bound regime.
# TODO(bduke): Dispatch based on device after completing parameter sweep.

comptime MAX_NUM_BLOCKS_UPPER_BOUND = 1024
"""Maximum number of thread blocks to use for reduction kernels.

This value has been empirically optimized through grid search across different GPU architectures.
While this value is optimal for A100 GPUs, H100 GPUs may benefit from more blocks to fully
saturate NVLink bandwidth.
"""


@always_inline
def circular_add[n: Int](x: Int, y: Int) -> Int:
    """Addition modulo n, assuming 0 <= x < n and 0 <= y < n.

    Equivalent to (x + y) % n. When n is a power of 2, uses unsigned
    modulo which compiles to a single `and` instruction. Otherwise uses
    a conditional subtract to avoid expensive integer division on GPU.

    Parameters:
        n: The modulus. Must be positive. Both addends are assumed to
            already lie in `[0, n)`.

    Args:
        x: The first addend, in `[0, n)`.
        y: The second addend, in `[0, n)`.
    """

    comptime if n.is_power_of_two():
        return umod(x + y, n)
    else:
        var z = x + y
        return z - n if z >= n else z


comptime MAX_GPUS = 8
"""Maximum number of GPUs supported in the allreduce implementation.

This constant sets the upper bound for the number of GPUS supported in this algorithm.
"""


comptime NUM_BARRIER_DOMAINS = MAX_GPUS
"""Number of disjoint barrier counter banks held by a `Signal` buffer.

`_multi_gpu_barrier` keys its `self_counter`/`peer_counter` slots by the
in-block thread index, not the global device rank. When subgroup collectives
(e.g. a `group_size`-of-4 TP collective) and full-world collectives share one
`Signal` buffer, they advance the same counter slots at different rates and the
generation counters desync -- a subsequent full-world barrier then spins
forever waiting for a generation a peer will never publish.

Giving each collective domain its own counter bank makes the barrier histories
disjoint so domains can never poison each other. `MAX_GPUS` banks is the loosest
sufficient bound: there can be at most one distinct device-group per GPU.

Callers map by group WIDTH (`0 if group_size == num_devices else group_size`),
so DIFFERENT ops of the same width share a bank. Sharing is sound only while
every rank in the domain issues the same barrier sequence: every collective must
pair start with end (a start-only one lets a fast rank overwrite a shard its
peers are still P2P-reading), and any conditional skip must be group-uniform.
Sibling groups are safe regardless -- their `rank_sigs` are disjoint. The cost is
negligible next to the embedded Lamport region.
"""


@fieldwise_init
struct Signal:
    """A synchronization primitive for coordinating GPU thread blocks across multiple devices.

    This struct provides counter-based synchronization between thread blocks on different GPUs.
    It maintains two sets of counters:
    1. self_counter: Used by blocks on the current GPU to signal their progress
    2. peer_counter: Used to track progress of blocks on other GPUs

    Note:
        The counters use unsigned integers that may overflow, but this is safe since
        unsigned integer overflow has well-defined behavior.
    """

    # Counter may overflow, but it's fine since unsigned int overflow is
    # well-defined behavior.
    comptime flag_t = DType.uint32

    var self_counter: StaticTuple[
        StaticTuple[
            StaticTuple[Scalar[Self.flag_t], MAX_GPUS],
            MAX_NUM_BLOCKS_UPPER_BOUND,
        ],
        NUM_BARRIER_DOMAINS,
    ]
    """
    A 3D array of counters with shape
    (NUM_BARRIER_DOMAINS, MAX_NUM_BLOCKS_UPPER_BOUND, MAX_GPUS).
    Each counter tracks the progress of a specific thread block on the current
    GPU for a domain. Thread blocks increment their corresponding counter to
    signal completion of a phase, allowing other GPUs to detect when
    synchronization points are reached. The counters use atomic operations to
    ensure proper synchronization across devices.
    """

    var peer_counter: StaticTuple[
        StaticTuple[
            StaticTuple[
                StaticTuple[Scalar[Self.flag_t], MAX_GPUS],
                MAX_NUM_BLOCKS_UPPER_BOUND,
            ],
            2,
        ],
        NUM_BARRIER_DOMAINS,
    ]
    """
    A 4D array of counters with shape
    (NUM_BARRIER_DOMAINS, 2, MAX_NUM_BLOCKS_UPPER_BOUND, MAX_GPUS).
    Each domain contains two sets of counters to handle two synchronization
    points safely. The dual counter design prevents race conditions where a peer
    block arrives at the second sync point before the current block passes the
    first sync point.
    """

    var lamport_state: StaticTuple[Scalar[Self.flag_t], 4]
    """Device-resident state for the in-kernel Lamport generation advance:
    `[flag, prev_num_elements, arrival, reserved]`.

    - `flag`: monotonically-increasing generation counter (read at kernel entry,
      advanced once per call in the grid-barrier epilogue).
    - `prev_num_elements`: previous call's element count = this call's
      `clear_size`.
    - `arrival`: per-call block-arrival counter for the exactly-once advance.
    - reserved: pads to 16 bytes so `lamport_region` stays 16-byte aligned (the
      128-bit atomic stores require it).

    Zero-fill is the correct start state (flag 0, nothing to clear). This is the
    sole source of the generation counter / clear extent -- the kernel reads and
    advances it in-kernel, so the public op needs no per-call argument.
    """

    comptime _REGION_BYTES = (
        LamportGeneration.NUM_GENERATIONS
        * MAX_GPUS
        * Lamport.MAX_SMALL_MESSAGE_BYTES
    )
    """Bytes of the embedded Lamport region: 3 generations x MAX_GPUS slots x the
    per-slot max. Sized for MAX_GPUS so `Signal` is not parameterized by the
    runtime gpu count (a smaller run leaves the tail slots unused)."""

    var lamport_region: StaticTuple[UInt8, Self._REGION_BYTES]
    """The barrier-free Lamport comm region: 3 rotating generations x MAX_GPUS
    rank slots x the per-slot max message.

    Disjoint from self_counter & peer_counter to enable a mix of lamport &
    barrier-based collectives on the same signal buffer.
    """

    @always_inline
    def lamport_state_ptr(
        mut self,
    ) -> MutPointer[Scalar[Self.flag_t], MutAnyOrigin]:
        """Typed pointer to this `Signal`'s `lamport_state` block.

        Index it with the `Lamport.STATE_*` constants. The field is located by
        its own address (`Pointer(to=...)`), so there is no hand-computed
        byte offset to keep in sync with the field order.
        """
        return (
            Pointer(to=self.lamport_state)
            .bitcast[Scalar[Self.flag_t]]()
            .as_unsafe_any_origin()
        )

    @always_inline
    def lamport_region_ptr[
        dtype: DType
    ](mut self) -> MutPointer[Scalar[dtype], MutAnyOrigin]:
        """Typed pointer to the start of this `Signal`'s embedded Lamport region.

        Parameters:
            dtype: The element type to reinterpret the raw region bytes as.
                The returned pointer is typed `Pointer[Scalar[dtype], ...]`
                so callers can read and write Lamport message packs in this
                dtype without an extra cast.
        """
        return (
            Pointer(to=self.lamport_region)
            .bitcast[Scalar[dtype]]()
            .as_unsafe_any_origin()
        )


def _lamport_init(
    signal_buffer: DeviceBuffer[.uint8], ctx: DeviceContext
) raises:
    """Sets a signal buffer's embedded Lamport region to the sentinel.

    The region-only half of `init_signal_buffer`: every pack in the region must
    read as "not ready" (hold a `-0.0` lane) so no rank mistakes an unwritten
    peer slot for real data. The universal sentinel is a single `uint32` pattern
    (fp32 `-0.0`) that `has_neg_zero` detects under every supported transport
    dtype, so the fill is a dtype-agnostic `uint32` memset needing no dedicated
    kernel.

    The region is the trailing `Signal.lamport_region` field, at byte offset
    `sizeof(Signal) - Signal._REGION_BYTES`. `create_sub_buffer` takes its
    offset and size in the view dtype's elements (`uint32`), hence the `// 4`.

    Args:
        signal_buffer: This rank's signal buffer (at least `sizeof(Signal)`
            bytes).
        ctx: The device context for this rank's GPU.
    """
    comptime offset = (size_of[Signal]() - Signal._REGION_BYTES) // 4
    var region = signal_buffer.create_sub_buffer[.uint32](
        offset, Signal._REGION_BYTES // 4
    )
    ctx.enqueue_memset(region, LAMPORT_SENTINEL_U32)


def init_signal_buffer(
    signal_buffer: DeviceBuffer[.uint8], ctx: DeviceContext
) raises:
    """Initializes a freshly allocated signal buffer for any comm collective.

    Mojo-side equivalent of `Signals.buffers()` in allreduce.py: zero-fills the
    whole buffer (the correct start state for the barrier counters and
    `lamport_state`), then overwrites the embedded Lamport region with the
    `-0.0` sentinel so the barrier-free Lamport allreduce reads every pack as
    "not ready". Both memsets are enqueued on `ctx`'s stream, so the sentinel
    fill is correctly ordered after the zero-fill.

    Args:
        signal_buffer: This rank's signal buffer (at least `sizeof(Signal)`
            bytes).
        ctx: The device context for this rank's GPU.
    """
    ctx.enqueue_memset[.uint8](signal_buffer, 0)
    _lamport_init(signal_buffer, ctx)


@always_inline
def _multi_gpu_barrier[
    ngpus: Int,
    is_start: Bool,
    need_fence: Bool = False,
    *,
    named_barrier_threads: Int = 0,
    named_barrier_id: Int = 1,
    domain_id: Int = 0,
](
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    self_sg: MutPointer[Signal, MutAnyOrigin],
    my_rank: Int,
):
    """Implements a barrier synchronization across multiple GPUs to ensure all
    GPU blocks reach a certain point before proceeding.

    Parameters:
        ngpus: Number of GPUs participating in barrier.
        is_start: Whether this is the start barrier.
        need_fence: Whether memory fence is needed.
            If True, uses release/acquire semantics.
            If False, uses volatile memory operations for faster communication.
        named_barrier_threads: When > 0, use a named barrier scoped to this
            many threads instead of a CTA-wide `barrier()` (PTX `bar.sync 0`).
        named_barrier_id: Hardware named-barrier id (0..15). Only used when
            `named_barrier_threads > 0`. Defaults to 1 to avoid collision
            with the implicit id 0 used by `barrier()`.
        domain_id: Selects which disjoint counter bank in the `Signal` buffer
            this barrier uses (0..`NUM_BARRIER_DOMAINS`-1). Full-world
            collectives use 0; grouped (subgroup) collectives must pass a
            distinct nonzero domain so their barrier counters never alias the
            full-world bank on a shared `Signal` buffer.

    Args:
        rank_sigs: Signal pointers for all GPUs.
        self_sg: Signal pointer for current GPU.
        my_rank: Current GPU rank.

    Uses atomic counters and memory fences to ensure all GPUs reach barrier before proceeding.
    Implementation ported from VLLM's _multi_gpu_barrier in
    https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh#L169-L198

    Note: while named_barrier_threads makes it possible for only some threads/warps to
    participate in this barrier, the inter-GPU sync still expects local thread idx 0-ngpus
    to participate! In other words, it is safe to call this with only warp-0, but not safe
    to call this with only warp 1, 2 etc.
    """
    comptime assert (
        ngpus <= MAX_GPUS
    ), "too many GPUs for barrier implementation"
    comptime assert (
        0 <= domain_id < NUM_BARRIER_DOMAINS
    ), "domain_id out of range for barrier counter banks"

    comptime if not is_start:
        comptime if named_barrier_threads > 0:
            named_barrier[Int32(named_barrier_threads)](Int32(named_barrier_id))
        else:
            barrier()

    comptime assert not (
        need_fence and is_start
    ), "Start barrier should not need fence"
    comptime flag_t = Signal.flag_t
    # Linearise block_idx
    var bid = (
        block_idx.x
        + block_idx.y * grid_dim.x
        + (block_idx.z * (grid_dim.x * grid_dim.y))
    )

    if thread_idx.x < ngpus:
        # NOTE: (MOCO-1431) the use of pointer arithmetic here is a temporary workaround
        # to avoid functional issues that arise with increased register pressure when
        # dealing with static tuples
        var my_gpu = thread_idx.x
        # Each thread increments its own counter
        # Technically we only need one counter, but we use
        # multiple per block to eliminate the need to share the counter via smem.
        #
        # self_counter is [NUM_BARRIER_DOMAINS][MAX_NUM_BLOCKS][MAX_GPUS]; index
        # the scope bank so subgroup and full-world barriers never alias.
        comptime self_domain_offset = domain_id * (
            MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS
        )
        var internal_counter_ptr = (
            self_sg.bitcast[Scalar[flag_t]]()
            + self_domain_offset
            + bid * MAX_GPUS
            + my_gpu
        )
        var val = internal_counter_ptr[] + 1
        internal_counter_ptr[] = val

        # peer_counter follows the whole self_counter array
        comptime peer_counter_offset = size_of[
            StaticTuple[
                StaticTuple[
                    StaticTuple[Scalar[flag_t], MAX_GPUS],
                    MAX_NUM_BLOCKS_UPPER_BOUND,
                ],
                NUM_BARRIER_DOMAINS,
            ]
        ]() // size_of[flag_t]()
        # and is laid out [NUM_BARRIER_DOMAINS][2][MAX_NUM_BLOCKS][MAX_GPUS].
        comptime peer_domain_offset = domain_id * (
            2 * MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS
        )

        # this line should compute &rank_sigs[my_gpu]->peer_counter[domain_id][val % 2][bid][my_rank]
        var peer_counter_ptr = (
            rank_sigs[my_gpu].bitcast[Scalar[flag_t]]()
            + peer_counter_offset
            + peer_domain_offset
            + (val % 2) * (MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS)
            + bid * MAX_GPUS
            + my_rank
        )
        # this line should compute
        # &self_sg->peer_counter[domain_id][val % 2][bid][my_gpu]
        var self_counter_ptr = (
            self_sg.bitcast[Scalar[flag_t]]()
            + peer_counter_offset
            + peer_domain_offset
            + (val % 2) * (MAX_NUM_BLOCKS_UPPER_BOUND * MAX_GPUS)
            + bid * MAX_GPUS
            + my_gpu
        )

        # Write the expected counter value to peer and wait for correct value from
        # peer.
        # TODO(KERN-3443): Investigate why AMD GPUs require this to be a full fence
        # instead of the lighter weight volatile loop used for nvidia GPUs.
        comptime if need_fence or is_amd_gpu():
            # broadcast the value to all peers that I reached the barrier
            Atomic[Scalar[flag_t]].store[ordering=Ordering.RELEASE](
                peer_counter_ptr, val
            )
            while (
                Atomic[Scalar[flag_t]].load[ordering=Ordering.ACQUIRE](
                    self_counter_ptr
                )
                != val
            ):
                pass
        else:
            peer_counter_ptr.store[volatile=True](val)
            while self_counter_ptr.load[volatile=True]() != val:
                pass

    comptime if is_start or need_fence:
        comptime if named_barrier_threads > 0:
            named_barrier[Int32(named_barrier_threads)](Int32(named_barrier_id))
        else:
            barrier()
