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

"""Interface for accessing GPU statistics of NVIDIA GPUs."""

from __future__ import annotations

import ctypes
from types import TracebackType
from typing import Annotated, Any, NamedTuple, Protocol, runtime_checkable

from . import _bindtools
from .types import (
    ClockStats,
    GPUStats,
    MemoryStats,
    ThrottleReason,
    UtilizationStats,
)

_nvmlReturn_t = Annotated[int, ctypes.c_int]
_nvmlDevice_t = ctypes.c_void_p
_nvmlClockType_t = Annotated[int, ctypes.c_int]

_NVML_SUCCESS = 0
_NVML_ERROR_NO_PERMISSION = 4
_NVML_ERROR_GPU_NOT_FOUND = 28
_NVML_ERROR_NOT_SUPPORTED = 3

_NVML_CLOCK_GRAPHICS = 0
_NVML_CLOCK_SM = 1
_NVML_CLOCK_MEM = 2

# Sentinel value NVML writes into usedGpuMemory when the figure is unavailable.
_NVML_VALUE_NOT_AVAILABLE: int = 0xFFFFFFFFFFFFFFFF

# Buffer capacity for nvmlDeviceGetComputeRunningProcesses.
_MAX_PROCS = 64

# Bit-to-name mapping for the bitfield returned by
# ``nvmlDeviceGetCurrentClocksThrottleReasons``. Names match
# ``nvmlClocksThrottleReasonXxx`` (deprecated upstream; equivalent to the
# newer ``nvmlClocksEventReasonXxx``); values are mapped to the
# vendor-neutral :data:`ThrottleReason` vocabulary.
_NVML_THROTTLE_REASON_BITS: tuple[tuple[int, ThrottleReason], ...] = (
    (0x0000000000000001, "gpu_idle"),
    (0x0000000000000002, "applications_clocks_setting"),
    (0x0000000000000004, "sw_power_cap"),
    (0x0000000000000008, "hw_slowdown"),
    (0x0000000000000010, "sync_boost"),
    (0x0000000000000020, "sw_thermal_slowdown"),
    (0x0000000000000040, "hw_thermal_slowdown"),
    (0x0000000000000080, "hw_power_brake_slowdown"),
    (0x0000000000000100, "display_clock_setting"),
)


def _decode_throttle_bits(bits: int) -> list[ThrottleReason]:
    # Bits not present in the table above are silently ignored. Extend
    # ``_NVML_THROTTLE_REASON_BITS`` (and the ``ThrottleReason`` literal in
    # ``_types.py``) when NVML adds new reason flags.
    return [name for bit, name in _NVML_THROTTLE_REASON_BITS if bits & bit]


class _ClockPair(NamedTuple):
    current_mhz: int
    max_mhz: int


class _nvmlMemory_v2_t(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]

    version: int
    total: int
    reserved: int
    free: int
    used: int


class _nvmlUtilization_t(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]

    gpu: int
    memory: int


class _nvmlProcessInfo_t(ctypes.Structure):
    # Two-field layout compatible with all NVML versions.  Newer drivers (11+)
    # extend this struct with gpuInstanceId/computeInstanceId for MIG, but
    # defining only pid+usedGpuMemory avoids struct-size mismatch when the
    # runtime library predates the MIG extension.
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
    ]

    pid: int
    usedGpuMemory: int


@runtime_checkable
class _NVMLLibrary(Protocol):
    def nvmlInit_v2(self) -> _nvmlReturn_t: ...
    def nvmlShutdown(self) -> _nvmlReturn_t: ...
    def nvmlErrorString(
        self, result: _nvmlReturn_t
    ) -> Annotated[bytes, ctypes.c_char_p]: ...
    def nvmlDeviceGetCount_v2(
        self, device_count: ctypes._Pointer[ctypes.c_uint]
    ) -> _nvmlReturn_t: ...
    def nvmlDeviceGetHandleByIndex_v2(
        self, index: ctypes.c_uint, device: ctypes._Pointer[_nvmlDevice_t]
    ) -> _nvmlReturn_t: ...
    def nvmlDeviceGetMemoryInfo_v2(
        self, device: _nvmlDevice_t, memory: ctypes._Pointer[_nvmlMemory_v2_t]
    ) -> _nvmlReturn_t: ...
    def nvmlDeviceGetUtilizationRates(
        self,
        device: _nvmlDevice_t,
        utilization: ctypes._Pointer[_nvmlUtilization_t],
    ) -> _nvmlReturn_t: ...


# Optional clock-related entrypoints. nvmlDeviceGetCurrentClocksThrottleReasons
# in particular has been deprecated upstream in favor of
# nvmlDeviceGetCurrentClocksEventReasons and may be absent on stripped or very
# old libnvidia-ml builds. Bind these separately so a missing symbol disables
# clock reporting instead of failing context entry.
@runtime_checkable
class _NVMLClockExtensions(Protocol):
    def nvmlDeviceGetClockInfo(
        self,
        device: _nvmlDevice_t,
        clock_type: _nvmlClockType_t,
        clock: ctypes._Pointer[ctypes.c_uint],
    ) -> _nvmlReturn_t: ...
    def nvmlDeviceGetMaxClockInfo(
        self,
        device: _nvmlDevice_t,
        clock_type: _nvmlClockType_t,
        clock: ctypes._Pointer[ctypes.c_uint],
    ) -> _nvmlReturn_t: ...
    def nvmlDeviceGetCurrentClocksThrottleReasons(
        self,
        device: _nvmlDevice_t,
        clocks_throttle_reasons: ctypes._Pointer[ctypes.c_ulonglong],
    ) -> _nvmlReturn_t: ...


@runtime_checkable
class _NVMLProcessExtensions(Protocol):
    def nvmlDeviceGetComputeRunningProcesses(
        self,
        device: _nvmlDevice_t,
        info_count: ctypes._Pointer[ctypes.c_uint],
        infos: ctypes.Array[_nvmlProcessInfo_t],
    ) -> _nvmlReturn_t: ...


class NVMLError(Exception):
    def __init__(self, code: _nvmlReturn_t, message: str, /) -> None:
        super().__init__(message)
        self.code = code


class NoPermissionError(NVMLError):
    pass


class GPUNotFoundError(NVMLError):
    pass


class NotSupportedError(NVMLError):
    pass


_SPECIFIC_ERROR_TYPES: dict[int, type[NVMLError]] = {
    _NVML_ERROR_NO_PERMISSION: NoPermissionError,
    _NVML_ERROR_GPU_NOT_FOUND: GPUNotFoundError,
    _NVML_ERROR_NOT_SUPPORTED: NotSupportedError,
}


def _check_nvml_return(library: _NVMLLibrary, result: _nvmlReturn_t) -> None:
    if result != _NVML_SUCCESS:
        cls = _SPECIFIC_ERROR_TYPES.get(result, NVMLError)
        error_bytes = library.nvmlErrorString(result)
        if error_bytes is None:
            error_string = "(Unknown)"
        else:
            error_string = error_bytes.decode()
        raise cls(result, error_string)


class NVMLContext:
    """Context for accessing NVML and accessing GPU information."""

    def __init__(self) -> None:
        self._library: _NVMLLibrary | None = None
        self._clock_library: _NVMLClockExtensions | None = None
        self._process_library: _NVMLProcessExtensions | None = None

    def __enter__(self) -> NVMLContext:
        if self._library is not None:
            raise AssertionError("Context already active")
        cdll = ctypes.CDLL("libnvidia-ml.so.1")
        lib = _bindtools.bind_protocol(cdll, _NVMLLibrary)
        _check_nvml_return(lib, lib.nvmlInit_v2())
        self._library = lib
        # Bind clock-related entrypoints separately. dlsym-style binding
        # (via ctypes.CDLL.__getattr__ inside bind_protocol) raises
        # AttributeError when a symbol is missing, so a stripped/old
        # libnvidia-ml that lacks the deprecated throttle-reasons symbol
        # would otherwise fail the entire context. Treat that case as
        # "clock stats unavailable" instead.
        try:
            self._clock_library = _bindtools.bind_protocol(
                cdll, _NVMLClockExtensions
            )
        except AttributeError:
            self._clock_library = None
        try:
            self._process_library = _bindtools.bind_protocol(
                cdll, _NVMLProcessExtensions
            )
        except AttributeError:
            self._process_library = None
        return self

    def __exit__(
        self,
        exc_type: type[Any] | None,
        exc_value: Any,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._library is not None:
            _check_nvml_return(self._library, self._library.nvmlShutdown())
        self._library = None
        self._clock_library = None
        self._process_library = None

    def _get_count(self) -> int:
        if self._library is None:
            return 0
        count = ctypes.c_uint()
        _check_nvml_return(
            self._library,
            self._library.nvmlDeviceGetCount_v2(_bindtools.byref(count)),
        )
        return count.value

    def _get_device(self, index: int) -> _nvmlDevice_t:
        if self._library is None:
            raise IndexError("All indices are invalid when uninitialized")
        device = _nvmlDevice_t()
        _check_nvml_return(
            self._library,
            self._library.nvmlDeviceGetHandleByIndex_v2(
                ctypes.c_uint(index), _bindtools.byref(device)
            ),
        )
        return device

    def _get_clock_pair(
        self, device: _nvmlDevice_t, clock_type: int
    ) -> _ClockPair | None:
        """Return (current_mhz, max_mhz) for a clock domain, or None if unsupported."""
        assert self._library is not None
        assert self._clock_library is not None
        current = ctypes.c_uint()
        maximum = ctypes.c_uint()
        try:
            _check_nvml_return(
                self._library,
                self._clock_library.nvmlDeviceGetClockInfo(
                    device, clock_type, _bindtools.byref(current)
                ),
            )
            _check_nvml_return(
                self._library,
                self._clock_library.nvmlDeviceGetMaxClockInfo(
                    device, clock_type, _bindtools.byref(maximum)
                ),
            )
        except NotSupportedError:
            return None
        return _ClockPair(current_mhz=current.value, max_mhz=maximum.value)

    def _get_throttle_reasons(
        self, device: _nvmlDevice_t
    ) -> list[ThrottleReason] | None:
        assert self._library is not None
        assert self._clock_library is not None
        reasons = ctypes.c_ulonglong()
        try:
            _check_nvml_return(
                self._library,
                self._clock_library.nvmlDeviceGetCurrentClocksThrottleReasons(
                    device, _bindtools.byref(reasons)
                ),
            )
        except NotSupportedError:
            return None
        return _decode_throttle_bits(reasons.value)

    def _get_clock_stats(self, device: _nvmlDevice_t) -> ClockStats | None:
        # Skip entirely if clock-related symbols were not present at bind
        # time (very old / stripped libnvidia-ml).
        if self._clock_library is None:
            return None
        # Prefer SM clock; fall back to graphics clock if SM is unsupported on
        # this device. Memory clock is reported when available but its absence
        # does not suppress core-clock reporting.
        core_clocks = self._get_clock_pair(device, _NVML_CLOCK_SM)
        if core_clocks is None:
            core_clocks = self._get_clock_pair(device, _NVML_CLOCK_GRAPHICS)
        if core_clocks is None:
            return None
        mem_clocks = self._get_clock_pair(device, _NVML_CLOCK_MEM)
        return ClockStats(
            core_mhz=core_clocks.current_mhz,
            core_max_mhz=core_clocks.max_mhz,
            mem_mhz=mem_clocks.current_mhz if mem_clocks is not None else None,
            mem_max_mhz=mem_clocks.max_mhz if mem_clocks is not None else None,
            throttle_reasons=self._get_throttle_reasons(device),
        )

    def _get_device_stats(self, device: _nvmlDevice_t) -> GPUStats:
        assert self._library is not None
        mem = _nvmlMemory_v2_t()
        # Documentation says version must be 2, but the required value is
        # actually more complicated -- see
        # https://github.com/NVIDIA/nvidia-settings/issues/78#issuecomment-1012837988
        mem.version = ctypes.sizeof(_nvmlMemory_v2_t) | (2 << 24)
        _check_nvml_return(
            self._library,
            self._library.nvmlDeviceGetMemoryInfo_v2(
                device, _bindtools.byref(mem)
            ),
        )
        util = _nvmlUtilization_t()
        _check_nvml_return(
            self._library,
            self._library.nvmlDeviceGetUtilizationRates(
                device, _bindtools.byref(util)
            ),
        )
        return GPUStats(
            memory=MemoryStats(
                total_bytes=mem.total,
                free_bytes=mem.free,
                used_bytes=mem.used,
                reserved_bytes=mem.reserved,
            ),
            utilization=UtilizationStats(
                gpu_usage_percent=util.gpu,
                memory_activity_percent=util.memory,
            ),
            clocks=self._get_clock_stats(device),
        )

    def get_process_memory_bytes(self, pid: int) -> int | None:
        """Return total GPU memory in bytes used by pid across all devices.

        Queries each device's compute-process list and sums the allocations
        attributed to ``pid``.  Entries whose ``usedGpuMemory`` equals the
        NVML sentinel ``0xFFFFFFFFFFFFFFFF`` (unavailable) are skipped.

        Args:
            pid: OS process ID to look up.

        Returns:
            Total bytes across all devices where ``pid`` has an allocation,
            or ``None`` if the process is not found on any device or if the
            per-process query is unsupported by this driver version.
        """
        if self._library is None or self._process_library is None:
            return None
        total = 0
        found = False
        for i in range(self._get_count()):
            try:
                device = self._get_device(i)
            except (NoPermissionError, GPUNotFoundError):
                continue
            info_count = ctypes.c_uint(_MAX_PROCS)
            infos = (_nvmlProcessInfo_t * _MAX_PROCS)()
            ret = self._process_library.nvmlDeviceGetComputeRunningProcesses(
                device, _bindtools.byref(info_count), infos
            )
            if ret != _NVML_SUCCESS:
                continue
            for j in range(info_count.value):
                if (
                    infos[j].pid == pid
                    and infos[j].usedGpuMemory != _NVML_VALUE_NOT_AVAILABLE
                ):
                    total += infos[j].usedGpuMemory
                    found = True
        return total if found else None

    def get_stats(self) -> dict[int, GPUStats]:
        """Get GPU statistics for all GPUs."""
        stats: dict[int, GPUStats] = {}
        count = self._get_count()
        for i in range(count):
            try:
                device = self._get_device(i)
            except NoPermissionError:
                continue
            stats[i] = self._get_device_stats(device)
        return stats
