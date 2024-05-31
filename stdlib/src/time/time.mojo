# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
from time import now
```
"""

from sys import external_call, os_is_linux, os_is_windows

from memory import UnsafePointer

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#

# Enums used in time.h 's glibc
alias _CLOCK_REALTIME = 0
alias _CLOCK_MONOTONIC = 1 if os_is_linux() else 6
alias _CLOCK_PROCESS_CPUTIME_ID = 2 if os_is_linux() else 12
alias _CLOCK_THREAD_CPUTIME_ID = 3 if os_is_linux() else 16
alias _CLOCK_MONOTONIC_RAW = 4

# Constants
alias _NSEC_PER_USEC = 1000
alias _NSEC_PER_MSEC = 1000000
alias _USEC_PER_MSEC = 1000
alias _MSEC_PER_SEC = 1000
alias _NSEC_PER_SEC = _NSEC_PER_USEC * _USEC_PER_MSEC * _MSEC_PER_SEC

# LARGE_INTEGER in Windows represent a signed 64 bit integer. Internally it
# is implemented as a union of one 64 bit integer or two 32 bit integers
# for 64/32 bit compilers.
# https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-large_integer-r1
alias _WINDOWS_LARGE_INTEGER = Int64


@value
@register_passable("trivial")
struct _CTimeSpec(Stringable):
    var tv_sec: Int  # Seconds
    var tv_subsec: Int  # subsecond (nanoseconds on linux and usec on mac)

    fn __init__(inout self):
        self.tv_sec = 0
        self.tv_subsec = 0

    fn as_nanoseconds(self) -> Int:
        @parameter
        if os_is_linux():
            return self.tv_sec * _NSEC_PER_SEC + self.tv_subsec
        else:
            return self.tv_sec * _NSEC_PER_SEC + self.tv_subsec * _NSEC_PER_USEC

    fn __str__(self) -> String:
        return str(self.as_nanoseconds()) + "ns"


@value
@register_passable("trivial")
struct _FILETIME:
    var dwLowDateTime: UInt32
    var dwHighDateTime: UInt32

    fn __init__(inout self):
        self.dwLowDateTime = 0
        self.dwHighDateTime = 0

    fn as_nanoseconds(self) -> Int:
        # AFTER subtracting windows offset the return value fits in a signed int64
        # BEFORE subtracting windows offset the return value does not fit in a signed int64
        # Taken from https://github.com/microsoft/STL/blob/c8d1efb6d504f6392acf8f6d01fd703f7c8826c0/stl/src/xtime.cpp#L50
        alias windowsToUnixEpochOffsetNs: Int = 0x19DB1DED53E8000
        var interval_count: UInt64 = (
            self.dwHighDateTime.cast[DType.uint64]() << 32
        ) + self.dwLowDateTime.cast[DType.uint64]() - windowsToUnixEpochOffsetNs
        return int(interval_count * 100)


@always_inline
fn _clock_gettime(clockid: Int) -> _CTimeSpec:
    """Low-level call to the clock_gettime libc function"""
    var ts = _CTimeSpec()

    # Call libc's clock_gettime.
    _ = external_call["clock_gettime", Int32](
        Int32(clockid), UnsafePointer.address_of(ts)
    )

    return ts


@always_inline
fn _gettime_as_nsec_unix(clockid: Int) -> Int:
    if os_is_linux():
        var ts = _clock_gettime(clockid)
        return ts.as_nanoseconds()
    else:
        return int(
            external_call["clock_gettime_nsec_np", Int64](Int32(clockid))
        )


@always_inline
fn _realtime_nanoseconds() -> Int:
    """Returns the current realtime time in nanoseconds"""
    return _gettime_as_nsec_unix(_CLOCK_REALTIME)


@always_inline
fn _monotonic_nanoseconds() -> Int:
    """Returns the current monotonic time in nanoseconds"""

    @parameter
    if os_is_windows():
        var ft = _FILETIME()
        external_call["GetSystemTimePreciseAsFileTime", NoneType](
            UnsafePointer.address_of(ft)
        )

        return ft.as_nanoseconds()
    else:
        return _gettime_as_nsec_unix(_CLOCK_MONOTONIC)


@always_inline
fn _monotonic_raw_nanoseconds() -> Int:
    """Returns the current monotonic time in nanoseconds"""

    return _gettime_as_nsec_unix(_CLOCK_MONOTONIC_RAW)


@always_inline
fn _process_cputime_nanoseconds() -> Int:
    """Returns the high-resolution per-process timer from the CPU"""

    return _gettime_as_nsec_unix(_CLOCK_PROCESS_CPUTIME_ID)


@always_inline
fn _thread_cputime_nanoseconds() -> Int:
    """Returns the thread-specific CPU-time clock"""

    return _gettime_as_nsec_unix(_CLOCK_THREAD_CPUTIME_ID)


# ===----------------------------------------------------------------------===#
# now
# ===----------------------------------------------------------------------===#


@always_inline
fn now() -> Int:
    """Returns the current monotonic time time in nanoseconds. This function
    queries the current platform's monotonic clock, making it useful for
    measuring time differences, but the significance of the returned value
    varies depending on the underlying implementation.

    Returns:
        The current time in ns.
    """
    return _monotonic_nanoseconds()


# ===----------------------------------------------------------------------===#
# time_function
# ===----------------------------------------------------------------------===#


@always_inline
@parameter
fn _time_function_windows[func: fn () capturing -> None]() -> Int:
    """Calculates elapsed time in Windows"""

    var ticks_per_sec: _WINDOWS_LARGE_INTEGER = 0
    var ticks_per_sec_ptr = UnsafePointer[_WINDOWS_LARGE_INTEGER].address_of(
        ticks_per_sec
    )
    external_call["QueryPerformanceFrequency", NoneType](ticks_per_sec_ptr)

    var starting_tick_count: _WINDOWS_LARGE_INTEGER = 0
    var start_ptr = UnsafePointer[_WINDOWS_LARGE_INTEGER].address_of(
        starting_tick_count
    )
    var ending_tick_count: _WINDOWS_LARGE_INTEGER = 0
    var end_ptr = UnsafePointer[_WINDOWS_LARGE_INTEGER].address_of(
        ending_tick_count
    )

    external_call["QueryPerformanceCounter", NoneType](start_ptr)
    func()
    external_call["QueryPerformanceCounter", NoneType](end_ptr)

    var elapsed_ticks = ending_tick_count - starting_tick_count

    # Note: Windows performance counter resolution is in µs.
    var elapsed_time_in_ns = (elapsed_ticks * 1_000_000_000) // ticks_per_sec
    return int(elapsed_time_in_ns)


@always_inline
@parameter
fn time_function[func: fn () capturing -> None]() -> Int:
    """Measures the time spent in the function.

    Parameters:
        func: The function to time.

    Returns:
        The time elapsed in the function in ns.
    """

    @parameter
    if os_is_windows():
        return _time_function_windows[func]()

    var tic = now()
    func()
    var toc = now()
    return toc - tic


# ===----------------------------------------------------------------------===#
# sleep
# ===----------------------------------------------------------------------===#


fn sleep(sec: Float64):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for.
    """
    alias NANOSECONDS_IN_SECOND = 1_000_000_000
    var total_secs = sec.__floor__()
    var tv_spec = _CTimeSpec(
        int(total_secs.cast[DType.index]()),
        int((sec - total_secs) * NANOSECONDS_IN_SECOND),
    )
    var req = UnsafePointer[_CTimeSpec].address_of(tv_spec)
    var rem = UnsafePointer[_CTimeSpec]()
    _ = external_call["nanosleep", Int32](req, rem)


fn sleep(sec: Int):
    """Suspends the current thread for the seconds specified.

    Args:
        sec: The number of seconds to sleep for.
    """

    @parameter
    if os_is_windows():
        # In Windows the argument is in milliseconds.
        external_call["Sleep", NoneType](sec * 1000)
    else:
        external_call["sleep", NoneType](sec)
