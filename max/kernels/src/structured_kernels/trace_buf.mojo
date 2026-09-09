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

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"GmemTrace"`.
        """
        return "GmemTrace"
