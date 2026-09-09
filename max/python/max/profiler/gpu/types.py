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

"""Public types returned by the GPU diagnostics API."""

from __future__ import annotations

from typing import Literal

import msgspec

ThrottleReason = Literal[
    "gpu_idle",
    "applications_clocks_setting",
    "sw_power_cap",
    "hw_slowdown",
    "sync_boost",
    "sw_thermal_slowdown",
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
    "display_clock_setting",
]
"""Vendor-neutral reason that a GPU's clock is below its boost ceiling.

The vocabulary mirrors the categories NVML exposes today; new vendors
(e.g. ROCm-SMI, once it exposes an equivalent API) should map their
reasons onto these names where possible and extend the list otherwise.
"""

# Reasons that indicate the GPU was performance-capped by hardware
# (thermal, power, sync-boost) rather than by user/application settings
# or simply being idle. Use ``set(stats.throttle_reasons) &
# HARDWARE_THROTTLE_REASONS`` to decide whether a benchmark run was
# throttled. A bare non-empty list is misleading because ``"gpu_idle"``
# is reported whenever the GPU was idle at sample time.
HARDWARE_THROTTLE_REASONS: frozenset[ThrottleReason] = frozenset(
    {
        "sw_power_cap",
        "hw_slowdown",
        "sync_boost",
        "sw_thermal_slowdown",
        "hw_thermal_slowdown",
        "hw_power_brake_slowdown",
    }
)


class GPUStats(msgspec.Struct):
    """GPU state snapshot: memory, utilization, and clock metrics.

    This class provides a complete view of a GPU's current state, including
    detailed memory usage statistics, utilization percentages, and current
    clock rates. It serves as the primary data structure returned by GPU
    diagnostic queries.
    """

    memory: MemoryStats
    """Detailed memory usage statistics for the GPU."""
    utilization: UtilizationStats
    """Current GPU compute and memory utilization percentages."""
    clocks: ClockStats | None = None
    """Current and maximum clock rates plus throttle reasons, if available."""


class MemoryStats(msgspec.Struct):
    """Detailed GPU memory usage statistics including total, free, used, and reserved memory.

    This class provides comprehensive memory information for a GPU, allowing
    developers to monitor memory consumption and identify potential memory
    bottlenecks during model inference or training.
    """

    total_bytes: int
    """Total GPU memory capacity in bytes."""
    free_bytes: int
    """Currently available GPU memory in bytes."""
    used_bytes: int
    """Currently allocated GPU memory in bytes."""
    reserved_bytes: int | None
    """Memory reserved by the driver, if available from the GPU vendor."""


class UtilizationStats(msgspec.Struct):
    """GPU compute and memory activity utilization percentages.

    This class captures the current utilization levels of a GPU's compute
    units and memory subsystem, providing insights into how effectively
    the GPU resources are being utilized during workload execution.
    """

    gpu_usage_percent: int
    """Current GPU compute utilization as a percentage (0-100)."""

    memory_activity_percent: int | None
    """Memory controller activity percentage, if available from the GPU vendor."""


class ClockStats(msgspec.Struct):
    """Current and maximum GPU clock rates plus throttle reasons.

    Captures the core (SM/graphics on NVIDIA, system clock on AMD) and memory
    clock rates measured at sample time along with the device's boost ceiling
    for each domain. Consumers can compare current to max clocks (or inspect
    ``throttle_reasons``) to detect runs whose performance is limited by
    hardware throttling rather than the workload itself.

    ``throttle_reasons`` is a list of vendor-neutral
    :data:`ThrottleReason` values. Note that ``"gpu_idle"`` is reported
    whenever the GPU is idle, so a non-empty list does *not* on its own
    indicate throttling -- intersect with :data:`HARDWARE_THROTTLE_REASONS`
    to detect hardware-induced throttling specifically. On AMD,
    ``throttle_reasons`` is ``None`` because ROCm-SMI does not expose an
    equivalent single-call API.
    """

    core_mhz: int
    """Current core clock rate in MHz (SM clock on NVIDIA, system clock on AMD)."""

    core_max_mhz: int
    """Maximum (boost) core clock rate in MHz."""

    mem_mhz: int | None
    """Current memory clock rate in MHz, or ``None`` if the vendor library did not report it."""

    mem_max_mhz: int | None
    """Maximum (boost) memory clock rate in MHz, or ``None`` if the vendor library did not report it."""

    throttle_reasons: list[ThrottleReason] | None
    """Vendor-neutral reasons the clock is below boost.

    ``"gpu_idle"`` is reported whenever the GPU is idle, so consumers
    should intersect with :data:`HARDWARE_THROTTLE_REASONS` when checking
    for hardware throttling. ``None`` when the vendor library does not
    expose a throttle-reason API.
    """
