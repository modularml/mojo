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

"""Interface for accessing GPU statistics of AMD GPUs."""

from __future__ import annotations

import ctypes
from types import TracebackType
from typing import Annotated, Any, Protocol, runtime_checkable

from . import _bindtools
from .types import ClockStats, GPUStats, MemoryStats, UtilizationStats

_rsmi_status_t = Annotated[int, ctypes.c_int]
_rsmi_memory_type_t = Annotated[int, ctypes.c_int]
_rsmi_clk_type_t = Annotated[int, ctypes.c_int]
_rsmi_device_index_t = Annotated[int, ctypes.c_uint32]

_RSMI_STATUS_SUCCESS = 0
_RSMI_STATUS_NOT_SUPPORTED = 2

_RSMI_MEM_TYPE_VRAM = 0

_RSMI_CLK_TYPE_SYS = 0
_RSMI_CLK_TYPE_MEM = 4

# rsmi_frequencies_t layout for ROCm 6.0+ (verified against the rocm-6.0.0
# through rocm-7.0.0 headers). ROCm 6.0 is the documented floor for this
# binding: ROCm <= 5.7 omits the leading ``has_deep_sleep`` field and uses
# RSMI_MAX_NUM_FREQUENCIES=32, which would silently misalign the struct and
# produce garbage frequencies. The validation in ``_get_clock_freqs`` (range
# checks on ``num_supported`` and ``current``) catches the worst cases so the
# stats path returns None rather than a bogus number, but environments still
# on rocm-smi <6.0 should treat clock reporting as unsupported.
_RSMI_MAX_NUM_FREQUENCIES = 33


class _rsmi_process_info_t(ctypes.Structure):
    _fields_ = [
        ("process_id", ctypes.c_uint32),
        ("pasid", ctypes.c_uint32),
        ("vram_usage", ctypes.c_uint64),  # bytes
        ("sdma_usage", ctypes.c_uint64),  # microseconds
        ("cu_occupancy", ctypes.c_uint32),
    ]

    process_id: int
    pasid: int
    vram_usage: int
    sdma_usage: int
    cu_occupancy: int


class _rsmi_frequencies_t(ctypes.Structure):
    # When `has_deep_sleep` is true, frequency[0] is the deep-sleep clock
    # (very low) and `current` may be 0 to indicate the GPU is currently in
    # deep sleep. _get_clock_freqs filters that case out.
    _fields_ = [
        ("has_deep_sleep", ctypes.c_bool),
        ("num_supported", ctypes.c_uint32),
        ("current", ctypes.c_uint32),
        ("frequency", ctypes.c_uint64 * _RSMI_MAX_NUM_FREQUENCIES),
    ]

    has_deep_sleep: bool
    num_supported: int
    current: int
    frequency: ctypes.Array[ctypes.c_uint64]


@runtime_checkable
class _RSMILibrary(Protocol):
    def rsmi_init(
        self, init_flags: Annotated[int, ctypes.c_uint64]
    ) -> _rsmi_status_t: ...
    def rsmi_shut_down(self) -> None: ...
    def rsmi_num_monitor_devices(
        self, num_devices: ctypes._Pointer[ctypes.c_uint32]
    ) -> _rsmi_status_t: ...
    def rsmi_dev_memory_total_get(
        self,
        device_index: _rsmi_device_index_t,
        mem_type: _rsmi_memory_type_t,
        total: ctypes._Pointer[ctypes.c_uint64],
    ) -> _rsmi_status_t: ...
    def rsmi_dev_memory_usage_get(
        self,
        device_index: _rsmi_device_index_t,
        mem_type: _rsmi_memory_type_t,
        total: ctypes._Pointer[ctypes.c_uint64],
    ) -> _rsmi_status_t: ...
    def rsmi_dev_memory_busy_percent_get(
        self,
        device_index: _rsmi_device_index_t,
        busy_percent: ctypes._Pointer[ctypes.c_uint32],
    ) -> _rsmi_status_t: ...
    def rsmi_dev_busy_percent_get(
        self,
        device_index: _rsmi_device_index_t,
        busy_percent: ctypes._Pointer[ctypes.c_uint32],
    ) -> _rsmi_status_t: ...
    def rsmi_dev_gpu_clk_freq_get(
        self,
        device_index: _rsmi_device_index_t,
        clk_type: _rsmi_clk_type_t,
        f: ctypes._Pointer[_rsmi_frequencies_t],
    ) -> _rsmi_status_t: ...
    def rsmi_status_string(
        self, status: _rsmi_status_t, string: ctypes._Pointer[ctypes.c_char_p]
    ) -> _rsmi_status_t: ...


@runtime_checkable
class _RSMIProcessExtensions(Protocol):
    def rsmi_compute_process_info_by_pid_get(
        self,
        pid: Annotated[int, ctypes.c_uint32],
        proc: ctypes._Pointer[_rsmi_process_info_t],
    ) -> _rsmi_status_t: ...


class RSMIError(Exception):
    def __init__(self, code: _rsmi_status_t, message: str, /) -> None:
        super().__init__(message)
        self.code = code


def _check_rsmi_status(library: _RSMILibrary, status: _rsmi_status_t) -> None:
    if status != _RSMI_STATUS_SUCCESS:
        error_ptr = ctypes.c_char_p()
        if (
            library.rsmi_status_string(status, _bindtools.byref(error_ptr))
            != _RSMI_STATUS_SUCCESS
            or (error_bytes := error_ptr.value) is None
        ):
            error_string = "(Unknown)"
        else:
            error_string = error_bytes.decode()
        raise RSMIError(status, error_string)


class RSMIContext:
    """Context for accessing ROCm-SMI and accessing GPU information."""

    def __init__(self) -> None:
        self._library: _RSMILibrary | None = None
        self._process_library: _RSMIProcessExtensions | None = None

    def __enter__(self) -> RSMIContext:
        if self._library is not None:
            raise AssertionError("Context already active")
        cdll = ctypes.CDLL("librocm_smi64.so")
        lib = _bindtools.bind_protocol(cdll, _RSMILibrary)
        _check_rsmi_status(lib, lib.rsmi_init(0))
        self._library = lib
        try:
            self._process_library = _bindtools.bind_protocol(
                cdll, _RSMIProcessExtensions
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
            self._library.rsmi_shut_down()
        self._library = None
        self._process_library = None

    def _get_count(self) -> int:
        if self._library is None:
            return 0
        count = ctypes.c_uint32()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_num_monitor_devices(_bindtools.byref(count)),
        )
        return count.value

    def _get_memory_total(self, index: int) -> int:
        assert self._library is not None
        result = ctypes.c_uint64()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_dev_memory_total_get(
                index, _RSMI_MEM_TYPE_VRAM, _bindtools.byref(result)
            ),
        )
        return result.value

    def _get_memory_usage(self, index: int) -> int:
        assert self._library is not None
        result = ctypes.c_uint64()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_dev_memory_usage_get(
                index, _RSMI_MEM_TYPE_VRAM, _bindtools.byref(result)
            ),
        )
        return result.value

    def _get_memory_busy_percent(self, index: int) -> int:
        assert self._library is not None
        result = ctypes.c_uint32()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_dev_memory_busy_percent_get(
                index, _bindtools.byref(result)
            ),
        )
        return result.value

    def _get_device_busy_percent(self, index: int) -> int:
        assert self._library is not None
        result = ctypes.c_uint32()
        _check_rsmi_status(
            self._library,
            self._library.rsmi_dev_busy_percent_get(
                index, _bindtools.byref(result)
            ),
        )
        return result.value

    def _get_clock_freqs(
        self, index: int, clk_type: int
    ) -> tuple[int, int] | None:
        """Return (current_mhz, max_mhz) for a clock domain, or None on error.

        ROCm-SMI returns frequencies in Hz. The "max" clock is taken as the
        highest supported DPM level (``frequency[num_supported - 1]``).
        Deep-sleep samples (``has_deep_sleep`` set with ``current == 0``)
        are reported as ``None``: ``frequency[0]`` would otherwise be the
        very-low deep-sleep clock and would falsely look like severe
        throttling.
        """
        assert self._library is not None
        freqs = _rsmi_frequencies_t()
        try:
            _check_rsmi_status(
                self._library,
                self._library.rsmi_dev_gpu_clk_freq_get(
                    index, clk_type, _bindtools.byref(freqs)
                ),
            )
        except RSMIError as e:
            # Mirror the NVML side: only swallow "not supported" so genuinely
            # unexpected RSMI failures (permission errors, init failures, etc.)
            # still propagate to the caller.
            if e.code == _RSMI_STATUS_NOT_SUPPORTED:
                return None
            raise
        if (
            freqs.num_supported == 0
            or freqs.num_supported > _RSMI_MAX_NUM_FREQUENCIES
        ):
            # May be dealing with a different struct layout from ROCm <= 5.7
            # (see comment on `_RSMI_MAX_NUM_FREQUENCIES`); bail.
            return None
        if freqs.current >= freqs.num_supported:
            # Either a layout mismatch slipped past the size check above, or
            # ROCm-SMI returned an inconsistent snapshot.
            return None
        if freqs.has_deep_sleep and freqs.current == 0:
            return None
        # `freqs.current` is in [0, num_supported) and num_supported is in
        # (0, _RSMI_MAX_NUM_FREQUENCIES] (the size of `freqs.frequency`), so
        # both indices below are in bounds.
        current_hz = freqs.frequency[freqs.current]
        max_hz = freqs.frequency[freqs.num_supported - 1]
        return (current_hz // 1_000_000, max_hz // 1_000_000)

    def _get_clock_stats(self, index: int) -> ClockStats | None:
        # ROCm-SMI has no analogue of NVML's SM->GRAPHICS clock-type fallback:
        # `RSMI_CLK_TYPE_SYS` is the only universally-supported core clock,
        # so a NOT_SUPPORTED here means we report no clocks at all.
        sys_clocks = self._get_clock_freqs(index, _RSMI_CLK_TYPE_SYS)
        if sys_clocks is None:
            return None
        mem_clocks = self._get_clock_freqs(index, _RSMI_CLK_TYPE_MEM)
        return ClockStats(
            core_mhz=sys_clocks[0],
            core_max_mhz=sys_clocks[1],
            mem_mhz=mem_clocks[0] if mem_clocks is not None else None,
            mem_max_mhz=mem_clocks[1] if mem_clocks is not None else None,
            # ROCm-SMI does not expose a single-call throttle-reasons API
            # comparable to NVML's; leave unset for now.
            throttle_reasons=None,
        )

    def _get_device_stats(self, index: int) -> GPUStats:
        mem_total = self._get_memory_total(index)
        mem_used = self._get_memory_usage(index)
        mem_busy_pct = self._get_memory_busy_percent(index)
        dev_busy_pct = self._get_device_busy_percent(index)
        return GPUStats(
            memory=MemoryStats(
                total_bytes=mem_total,
                free_bytes=mem_total - mem_used,
                used_bytes=mem_used,
                reserved_bytes=None,
            ),
            utilization=UtilizationStats(
                gpu_usage_percent=dev_busy_pct,
                memory_activity_percent=mem_busy_pct,
            ),
            clocks=self._get_clock_stats(index),
        )

    def get_process_memory_bytes(self, pid: int) -> int | None:
        """Return VRAM in bytes used by pid, or None if not found or unsupported.

        Calls ``rsmi_compute_process_info_by_pid_get`` which returns the total
        VRAM allocated by the process across all GPU devices it is using.

        Args:
            pid: OS process ID to look up.

        Returns:
            VRAM usage in bytes, or ``None`` if the process is not running on
            any GPU or if the per-process API is unavailable on this ROCm version.
        """
        if self._library is None or self._process_library is None:
            return None
        proc = _rsmi_process_info_t()
        try:
            status = self._process_library.rsmi_compute_process_info_by_pid_get(
                pid, _bindtools.byref(proc)
            )
        except Exception:
            return None
        if status != _RSMI_STATUS_SUCCESS:
            return None
        return proc.vram_usage

    def get_stats(self) -> dict[int, GPUStats]:
        """Get GPU statistics for all GPUs."""
        stats: dict[int, GPUStats] = {}
        count = self._get_count()
        for i in range(count):
            stats[i] = self._get_device_stats(i)
        return stats
