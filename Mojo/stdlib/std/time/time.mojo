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
"""Implements basic utils for working with time.

You can import these APIs from the `time` package. For example:

```mojo
from std.time import perf_counter_ns
```
"""

from std.math import floor
from std.os import abort
from std.ffi import external_call
from std.sys import (
    CompilationTarget,
    is_amd_gpu,
    is_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
)
from std.sys._assembly import inlined_assembly

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#

# Enums used in time.h 's glibc
comptime _CLOCK_REALTIME = 0
comptime _CLOCK_MONOTONIC = 1 if CompilationTarget.is_linux() else 6
comptime _CLOCK_PROCESS_CPUTIME_ID = 2 if CompilationTarget.is_linux() else 12
comptime _CLOCK_THREAD_CPUTIME_ID = 3 if CompilationTarget.is_linux() else 16
comptime _CLOCK_MONOTONIC_RAW = 4

# Constants
comptime _NSEC_PER_USEC = 1000
comptime _NSEC_PER_MSEC = 1_000_000
comptime _USEC_PER_MSEC = 1000
comptime _MSEC_PER_SEC = 1000
comptime _NSEC_PER_SEC = _NSEC_PER_USEC * _USEC_PER_MSEC * _MSEC_PER_SEC


@fieldwise_init
struct _CTimeSpec(Defaultable, TrivialRegisterPassable, Writable):
    var tv_sec: Int  # Seconds
    var tv_nsec: Int  # Nanoseconds

    def __init__(out self):
        self.tv_sec = 0
        self.tv_nsec = 0

    def as_nanoseconds(self) -> Int:
        return self.tv_sec * _NSEC_PER_SEC + self.tv_nsec

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.as_nanoseconds(), "ns")


@always_inline
def _clock_gettime(clockid: Int) -> _CTimeSpec:
    """Low-level call to the clock_gettime libc function"""
    var ts = _CTimeSpec()

    # Call libc's clock_gettime.
    _ = external_call["clock_gettime", Int32](Int32(clockid), Pointer(to=ts))

    return ts


@always_inline
def _gettime_as_nsec_unix(clockid: Int) -> Int:
    comptime if CompilationTarget.is_linux():
        var ts = _clock_gettime(clockid)
        return ts.as_nanoseconds()
    else:
        return Int(
            external_call["clock_gettime_nsec_np", Int64](Int32(clockid))
        )


@always_inline
def _amd_gpu_realtime() -> UInt64:
    """Returns the AMD GPU real-time counter (constant-speed clock).

    This reads the s_memrealtime register which provides a constant-speed
    64-bit clock counter, independent of GPU core clock frequency scaling.
    This is suitable for wall-clock timing measurements.

    The counter frequency is 100 MHz (10ns per tick) on MI355 (gfx950) and
    likely other modern AMD GPUs, though this may vary by architecture.
    Use _AMD_GPU_REALTIME_FREQ_HZ for conversions.
    """
    return llvm_intrinsic["llvm.amdgcn.s.memrealtime", UInt64]()


# AMD GPU real-time counter frequency in Hz.
# MI355 (gfx950) and likely other modern AMD GPUs use 100 MHz.
# This can be queried at runtime via hipDeviceAttributeWallClockRate if needed.
comptime _AMD_GPU_REALTIME_FREQ_HZ: UInt64 = 100_000_000


@always_inline
def _realtime_nanoseconds() -> Int:
    """Returns the current realtime time in nanoseconds"""
    return _gettime_as_nsec_unix(_CLOCK_REALTIME)


@always_inline
def _monotonic_nanoseconds() -> Int:
    """Returns the current monotonic time in nanoseconds"""

    comptime if is_gpu():
        return Int(global_perf_counter_ns())
    else:
        return _gettime_as_nsec_unix(_CLOCK_MONOTONIC)


@always_inline
def _monotonic_raw_nanoseconds() -> Int:
    """Returns the current monotonic time in nanoseconds"""
    return _gettime_as_nsec_unix(_CLOCK_MONOTONIC_RAW)


@always_inline
def _process_cputime_nanoseconds() -> Int:
    """Returns the high-resolution per-process timer from the CPU"""

    return _gettime_as_nsec_unix(_CLOCK_PROCESS_CPUTIME_ID)


@always_inline
def _thread_cputime_nanoseconds() -> Int:
    """Returns the thread-specific CPU-time clock"""

    return _gettime_as_nsec_unix(_CLOCK_THREAD_CPUTIME_ID)


# ===-----------------------------------------------------------------------===#
# perf_counter
# ===-----------------------------------------------------------------------===#


@always_inline
def perf_counter() -> Float64:
    """Return the value (in fractional seconds) of a performance counter, i.e.
    a clock with the highest available resolution to measure a short duration.
    It does include time elapsed during sleep and is system-wide. The reference
    point of the returned value is undefined, so that only the difference
    between the results of two calls is valid.

    Returns:
        The current time in ns.
    """
    return Float64(_monotonic_nanoseconds()) / _NSEC_PER_SEC


# ===-----------------------------------------------------------------------===#
# perf_counter_ns
# ===-----------------------------------------------------------------------===#


@always_inline
def perf_counter_ns() -> Int:
    """Return the value (in nanoseconds) of a performance counter, i.e.
    a clock with the highest available resolution to measure a short duration.
    It does include time elapsed during sleep and is system-wide. The reference
    point of the returned value is undefined, so that only the difference
    between the results of two calls is valid.

    Returns:
        The current time in ns.
    """
    return _monotonic_nanoseconds()


# ===-----------------------------------------------------------------------===#
# global perf_counter_ns
# ===-----------------------------------------------------------------------===#


@always_inline
def global_perf_counter_ns() -> UInt64:
    """Returns the current value in the global nanosecond resolution timer. This value
    is common across all SM's.

    On NVIDIA GPUs, this uses the globaltimer register which provides nanosecond
    resolution. On AMD GPUs, this uses the s_memrealtime counter (constant-speed
    clock) converted to nanoseconds. On other platforms, this falls back to
    perf_counter_ns().

    Returns:
        The current time in ns.
    """

    comptime if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.read.ptx.sreg.globaltimer",
            UInt64,
            has_side_effect=True,
        ]()
    elif is_amd_gpu():
        # Convert s_memrealtime ticks to nanoseconds.
        # At 100 MHz, each tick is 10ns (1e9 / 100e6 = 10).
        var ticks = _amd_gpu_realtime()
        return (ticks * 1_000_000_000) // _AMD_GPU_REALTIME_FREQ_HZ

    return UInt64(perf_counter_ns())


# ===-----------------------------------------------------------------------===#
# monotonic
# ===-----------------------------------------------------------------------===#


@always_inline
def monotonic() -> Int:
    """
    Returns the current monotonic time time in nanoseconds. This function
    queries the current platform's monotonic clock, making it useful for
    measuring time differences, but the significance of the returned value
    varies depending on the underlying implementation.

    Returns:
        The current time in ns.
    """
    return perf_counter_ns()


# ===-----------------------------------------------------------------------===#
# time_function
# ===-----------------------------------------------------------------------===#


@always_inline
def time_function[FuncType: def() raises -> None](func: FuncType) raises -> Int:
    """Measures the time spent in the function.

    Parameters:
        FuncType: The function type to time.

    Args:
        func: The closure carrying the captured state of the timed function.

    Returns:
        The time elapsed in the function in ns.

    Raises:
        If the operation fails.
    """
    var tic = perf_counter_ns()
    func()
    var toc = perf_counter_ns()
    return toc - tic


@always_inline
def time_function[FuncType: def() -> None](func: FuncType) -> Int:
    """Measures the time spent in the function.

    Parameters:
        FuncType: The function type to time.

    Args:
        func: The closure carrying the captured state of the timed function.

    Returns:
        The time elapsed in the function in ns.
    """
    var tic = perf_counter_ns()
    func()
    var toc = perf_counter_ns()
    return toc - tic


# ===-----------------------------------------------------------------------===#
# sleep
# ===-----------------------------------------------------------------------===#


def sleep(sec: Float64):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for. Values <= 0 return immediately.
    """
    # Guard against non-positive sleep durations.
    if sec <= 0.0:
        return

    comptime if is_gpu():
        comptime if is_nvidia_gpu():
            # NVIDIA's nanosleep has a max duration of 1ms (1,000,000 ns).
            # Loop to handle longer sleep durations.
            comptime MAX_SLEEP_NS = 1_000_000  # 1ms in nanoseconds
            var total_ns = UInt64(sec * 1.0e9)
            var start = global_perf_counter_ns()
            var elapsed = global_perf_counter_ns() - start
            while elapsed < total_ns:
                var remaining = total_ns - elapsed
                var sleep_ns = Int32(min(remaining, UInt64(MAX_SLEEP_NS)))
                llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](sleep_ns)
                elapsed = global_perf_counter_ns() - start
            return
        elif is_amd_gpu():
            # AMD GPU sleep using s_memrealtime for timing feedback.
            # This approach is based on ROCm's ockl rtcwait implementation.
            #
            # We use the constant-speed s_memrealtime counter to track actual
            # elapsed time and loop with progressive s_sleep calls until the
            # target duration is reached. The s_sleep instruction accepts
            # values 0-127 and sleeps for approximately that many cycles.
            #
            # The tiered approach (127 -> 15 -> 1) balances power efficiency
            # (longer sleeps when far from target) with timing accuracy
            # (shorter sleeps as we approach the target).
            var total_ticks = UInt64(sec * Float64(_AMD_GPU_REALTIME_FREQ_HZ))
            var start = _amd_gpu_realtime()
            var end = start + total_ticks
            var now = start

            # Use tiered sleep intervals for efficiency.
            # Thresholds are approximate tick counts where each sleep level
            # is appropriate (based on 10ns per tick at 100 MHz).
            while now < end:
                var remaining = end - now
                if remaining > 3000:  # > ~30us remaining
                    llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(127))
                elif remaining > 400:  # > ~4us remaining
                    llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(15))
                else:
                    llvm_intrinsic["llvm.amdgcn.s.sleep", NoneType](Int32(1))
                now = _amd_gpu_realtime()
            return
        else:
            # Other GPUs are not supported.
            CompilationTarget.unsupported_target_error[
                operation="time.sleep()",
                note="time.sleep() is only supported on NVIDIA and AMD GPUs",
            ]()

    comptime NANOSECONDS_IN_SECOND = 1_000_000_000
    var total_secs = floor(sec)
    var tv_spec = _CTimeSpec(
        Int(total_secs),
        Int((sec - total_secs) * NANOSECONDS_IN_SECOND),
    )
    var req = Pointer(to=tv_spec)
    var rem = OptionalPointer[_CTimeSpec, MutUntrackedOrigin]()
    _ = external_call["nanosleep", Int32](req, rem)


def sleep(sec: Int):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for.
    """

    comptime if is_gpu():
        return sleep(Float64(sec))

    external_call["sleep", NoneType](Int32(sec))
