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

"""Zero-overhead per-CTA trace buffer for GPU kernel instrumentation.

A `TraceBuf` is a kernel-arg–shaped handle to a per-CTA timestamp slot
buffer. Implementations:

- `NullTrace` is zero-sized; passing it as a kernel argument adds no
  bytes to the kernel ABI. Its `store` is `pass`, so the body of the
  surrounding `comptime if enable_trace:` strips entirely at compile
  time.
- `GmemTrace` wraps a single `Pointer[UInt64]` to a buffer sized
  for `num_blocks * events_per_block` slots and records timestamps via
  PTX `globaltimer` (lowered from `global_perf_counter_ns`).

Usage pattern (see `nn/gemv_partial_norm.mojo` and the SM100 grouped
SwiGLU+NVFP4 kernel):

    fn my_kernel[..., enable_trace: Bool = False, TraceBufT: TraceBuf](
        ..., trace_buf: TraceBufT
    ):
        comptime if enable_trace:
            if thread_idx.x == 0:
                trace_buf.store(
                    Int(block_idx.x) * EVENTS_PER_BLOCK + role,
                    UInt64(global_perf_counter_ns()),
                )

When `enable_trace=False` (default), every `comptime if` block strips
to nothing and the resulting PTX is byte-identical to a build with no
trace plumbing at all.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.memory import UnsafePointer
from std.sys import llvm_intrinsic
from std.sys.info import is_nvidia_gpu
from std.time import global_perf_counter_ns


trait TraceBuf(DevicePassable, TrivialRegisterPassable):
    """Trace-buffer interface. Implementations: `NullTrace`, `GmemTrace`."""

    def store(self, offset: Int, val: UInt64):
        """Stores a timestamp at a slot in the trace buffer.

        Args:
            offset: Slot index. Callers typically encode roles as
                `block_idx * events_per_block + role`.
            val: Timestamp value (ns from `global_perf_counter_ns`).
        """
        ...

    def base_ptr(self) -> UnsafePointer[UInt64, MutUntrackedOrigin]:
        """Base pointer of the underlying slot buffer.

        For handing the buffer to code that cannot accept a `TraceBuf` --
        `P3PeerSendConfig` is a plain struct reaching the epilogue and the
        send warp class. `NullTrace` returns a DANGLING pointer, so every
        consumer must gate on its own runtime enable; a dangling pointer is
        not a valid "uninitialized" sentinel.
        """
        ...

    def load(self, offset: Int) -> UInt64:
        """Reads a `u64` slot (the Section B ring `WRITE_COUNT` cursor).

        Args:
            offset: Slot index.

        Returns:
            The stored `u64`, or 0 for the no-op `NullTrace`.
        """
        ...


struct NullTrace(TraceBuf):
    """Zero-sized no-op trace buffer.

    `store` is `pass`; the struct has no fields so it contributes 0
    kernel-arg bytes when passed as an argument. Combined with a
    `comptime if enable_trace:` guard at every call site, the no-trace
    path emits zero PTX for instrumentation.
    """

    comptime device_type: AnyType = Self
    """Device-side type alias. `NullTrace` is trivially device-passable."""

    @always_inline
    def base_ptr(self) -> UnsafePointer[UInt64, MutUntrackedOrigin]:
        """Returns a dangling pointer; there is no buffer."""
        return UnsafePointer[UInt64, MutUntrackedOrigin].unsafe_dangling()

    @always_inline
    def __init__(out self):
        """Constructs a zero-sized no-op trace buffer."""
        pass

    @always_inline
    def store(self, offset: Int, val: UInt64):
        """No-op store. The body compiles away entirely.

        Args:
            offset: Unused.
            val: Unused.
        """
        pass

    @always_inline
    def load(self, offset: Int) -> UInt64:
        """No-op load. Always returns 0.

        Args:
            offset: Unused.

        Returns:
            Always 0.
        """
        return 0

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        pass

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"NullTrace"`.
        """
        return "NullTrace"


struct GmemTrace(TraceBuf):
    """HBM-backed trace buffer.

    `store(offset, ts)` writes `ts` to `ptr[offset]`. 8 bytes of kernel
    arg.
    """

    comptime device_type: AnyType = Self
    """Device-side type alias. `GmemTrace` is trivially device-passable."""

    var ptr: UnsafePointer[UInt64, MutUntrackedOrigin]
    """Device pointer to a `u64` buffer sized for the caller's
    `num_blocks * events_per_block` slot count, zero-initialized on
    first use."""

    @always_inline
    def base_ptr(self) -> UnsafePointer[UInt64, MutUntrackedOrigin]:
        """Returns the device buffer's base pointer."""
        return self.ptr

    @always_inline
    def __init__(out self, ptr: UnsafePointer[UInt64, MutUntrackedOrigin]):
        """Wraps a device pointer as a trace buffer.

        Args:
            ptr: Device-side `Pointer[UInt64]` with room for
                `num_blocks * events_per_block` slots, zero-initialized
                on first use.
        """
        self.ptr = ptr

    @always_inline
    def store(self, offset: Int, val: UInt64):
        """Writes a timestamp into the device-side trace buffer.

        Args:
            offset: Slot index.
            val: Timestamp value (ns).
        """
        self.ptr.store(offset, val)

    @always_inline
    def load(self, offset: Int) -> UInt64:
        """Reads a `u64` slot from the device-side trace buffer.

        Args:
            offset: Slot index.

        Returns:
            The stored `u64`.
        """
        return self.ptr.load(offset)

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"GmemTrace"`.
        """
        return "GmemTrace"


# ===----------------------------------------------------------------------=== #
# Section B: per-owner-warp event rings (E40+ schema)
# ===----------------------------------------------------------------------=== #
#
# A second section appended to the SAME `GmemTrace` buffer after the existing
# positional per-tile tile-slot section (Section A). Each traced warp role owns
# ONE pre-carved ring (region `ring_base + ws * (2 + 2 * capacity)` for warp-slot
# `ws = cta * rings_per_cta + ring_id`). The owning warp elects ONE lane to
# stamp; no shared cursors, no atomics. Every call site is guarded by
# `comptime if enable_trace` so a no-trace build strips the whole section
# (byte-identical to no plumbing).
#
# Wire contract, frozen because a host decoder walks the flat u64 dump with
# nothing else to go on. Per ring: two head slots, then `capacity` two-slot
# records, record `i` at `2 + 2 * (i % capacity)`.
#   slot 0  WRITE_COUNT -- records ever appended, monotone, NOT wrapped;
#           written LAST, so a count never covers an unwritten record. The
#           live record count is `min(WRITE_COUNT, capacity)`.
#   slot 1  INFO, stamped once with the first record: bit 63 = ring written,
#           bit 40 = Section B schema tag, bits 31:0 = physical SM id.
#   word 0  `event_id:u8 << 56 | payload:u24 << 32 | seq:u32`, where `seq`
#           is `i` -- how a decoder orders a wrapped ring.
#   word 1  `global_perf_counter_ns()` at the stamp.


@always_inline
def _trace_smid() -> UInt32:
    """Physical SM id via the raw PTX `%smid` register.

    Unlike `std.gpu.sm_id`, this does NOT `warp.broadcast` — safe to call from a
    single elected lane (a broadcast from one active lane would hang).

    Returns:
        The physical SM id, or 0 on non-NVIDIA targets.
    """
    comptime if is_nvidia_gpu():
        return UInt32(
            Int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.smid",
                    Int32,
                    has_side_effect=False,
                ]()
            )
        )
    else:
        return 0


@always_inline
def pack_payload2(hi: Int, lo: Int) -> Int:
    """Packs a two-field ring payload as `(hi:u12 << 12) | lo:u12`.

    Mirrors the existing `pack_phase_pool` little-fields-low style. Used for the
    `(expert, m-block/pool)` and `(expert, target-rank)` event payloads.

    Args:
        hi: High 12-bit field (max 4095).
        lo: Low 12-bit field (max 4095).

    Returns:
        The packed u24 payload.
    """
    return ((hi & 0xFFF) << 12) | (lo & 0xFFF)


@always_inline
def ring_emit[
    TraceBufT: TraceBuf,
    //,
    rings_per_cta: Int,
    ring_capacity: Int,
](
    trace_buf: TraceBufT,
    ring_base: Int,
    cta: Int,
    ring_id: Int,
    event_id: Int,
    payload: Int,
):
    """Appends one Section B record to the `(cta, ring_id)` ring.

    Must be called by a SINGLE owner lane per `(cta, ring_id)` (monotone stamps).
    Stateless drop-oldest: reads `WRITE_COUNT`, writes the record's two words,
    stamps `INFO` on the first record, then writes `WRITE_COUNT` LAST (release
    order — a host snapshot never sees a count covering an unwritten record).

    Parameters:
        TraceBufT: The trace-buffer type (inferred; only `GmemTrace` is ever
            instantiated since every call site is `enable_trace`-guarded).
        rings_per_cta: Rings carved per physical CTA.
        ring_capacity: Records per ring (drop-oldest beyond this).

    Args:
        trace_buf: The device trace buffer.
        ring_base: First Section B slot (`num_ctas * slots_per_cta`).
        cta: Physical CTA index (`block_idx.x`).
        ring_id: Which ring this warp role owns.
        event_id: Event id (>= 40).
        payload: The u24 payload (see `pack_payload2`).
    """
    var ws = cta * rings_per_cta + ring_id
    var region = ring_base + ws * (2 + 2 * ring_capacity)
    var wc = Int(trace_buf.load(region))
    var slot = region + 2 + 2 * (wc % ring_capacity)
    trace_buf.store(
        slot,
        (UInt64(event_id) << 56)
        | ((UInt64(payload) & 0xFFFFFF) << 32)
        | (UInt64(wc) & 0xFFFFFFFF),
    )
    trace_buf.store(slot + 1, UInt64(global_perf_counter_ns()))
    if wc == 0:
        trace_buf.store(
            region + 1,
            (UInt64(1) << 63) | (UInt64(1) << 40) | UInt64(_trace_smid()),
        )
    trace_buf.store(region, UInt64(wc + 1))
