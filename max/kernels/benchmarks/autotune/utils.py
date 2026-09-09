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

import functools
import os
import pickle
import shutil
import subprocess
from collections.abc import Iterable
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import rich

LINE: str = "\n" + 70 * "-"


def _percentage(x: int, y: int) -> int:
    if x > 0 and y > 0:
        return int((x / y) * 100.0)
    return 0


def pretty_exception_handler(exception_type, exception, traceback) -> None:  # noqa: ANN001
    rich.print(f"[bold red]{exception_type.__name__}[/bold red]: {exception}")


def store_pickle(path: Path | str, data: Any) -> None:
    """Serialize data to a pickle file."""
    with Path(path).open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path | str) -> Any:
    """Deserialize data from a pickle file."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def flatten(value: int | object | Iterable) -> list[Any]:  # type: ignore[type-arg]
    """Flatten nested iterables into a single list."""
    if not isinstance(value, Iterable) or isinstance(value, str):
        return [value]
    return [
        item
        for sublist in (flatten(item) for item in value)
        for item in sublist
    ]


def _get_core_count() -> int:
    try:
        # The 'os.sched_getaffinity' method is only available on some Unix platforms
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined, unused-ignore]
    except AttributeError:
        # To cover other platforms, including mac
        return cpu_count()


def _get_visible_device_prefix(target_accelerator: str = "") -> str:
    """Returns the environment variable prefix for visible devices based on accelerator type."""
    if "nvidia" in target_accelerator or "cuda" in target_accelerator:
        return "CUDA_VISIBLE_DEVICES"
    elif "amd" in target_accelerator:
        return "ROCR_VISIBLE_DEVICES"
    else:
        return ""


def _get_gpu_count(target_accelerator: str = "") -> int | None:
    """Detect available GPUs, capped by any visibility env var."""
    hw_count = _get_hw_gpu_count(target_accelerator)
    if hw_count is None:
        return None
    prefix = _get_visible_device_prefix(target_accelerator)
    if prefix:
        vis = os.environ.get(prefix, "").strip()
        if vis:
            return min(hw_count, len({v.strip() for v in vis.split(",")}))
    return hw_count


def _get_hw_gpu_count(target_accelerator: str) -> int | None:
    """Query physical GPU count from vendor tools."""
    if "nvidia" in target_accelerator or "cuda" in target_accelerator:
        smi = get_nvidia_smi()
        if not smi:
            return None
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=name", "--format=csv,noheader"],
                timeout=10,
            )
            return len(out.decode().strip().splitlines())
        except (subprocess.SubprocessError, OSError):
            return None
    if "amd" in target_accelerator:
        smi = shutil.which("rocm-smi")
        if not smi:
            return None
        try:
            out = subprocess.check_output(
                [smi, "--showproductname", "--csv"],
                timeout=10,
                stderr=subprocess.DEVNULL,
            )
            lines = out.decode().strip().splitlines()
            # CSV: "device,Card series,..." header, then "card0,..." per GPU
            return sum(1 for line in lines if line.startswith("card"))
        except (subprocess.SubprocessError, OSError):
            return None
    return None


@functools.cache
def get_nvidia_smi():  # noqa: ANN201
    return shutil.which("nvidia-smi")


def check_gpu_clock() -> None:
    """Warn when the SM clock is free to move during a benchmark.

    Persistence mode used to stand in for this. It cannot: it is the one step
    of ``setup-gpu-clock.sh`` that succeeds on a modern driver, it survives
    reboots, and it says nothing about clock state — so it reported "locked"
    on boards whose clocks were floating by 900 MHz. Read the board instead.
    """
    nvidia_smi = get_nvidia_smi()
    if not nvidia_smi:
        return
    try:
        output = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu",
                "clocks_event_reasons.sw_power_cap,persistence_mode",
                "--format",
                "csv,noheader",
            ],
        ).decode("utf-8")
    except subprocess.CalledProcessError:
        return

    # Exact match: the field reads "Not Active" when idle, so a substring
    # test for "Active" matches every healthy board.
    capped = [
        ln for ln in output.splitlines() if ln.split(",")[0].strip() == "Active"
    ]
    if capped:
        # At the power cap the clock is set by the power budget, so no lock
        # holds at or above the sustainable frequency: `-lgc` is accepted and
        # then ignored. Pinning below it is the only thing that works.
        raise Exception(
            "the GPU is at its software power cap, so the SM clock is floating"
            " and benchmark results will not be reproducible. Pin below the"
            " power-limited clock (`sudo nvidia-smi -lgc <mhz>,<mhz>`, released"
            " with `-rgc` afterwards). `utils/setup-gpu-clock.sh` does NOT pin"
            " clocks on such a board — its `nvidia-smi -ac` step is refused by"
            " recent drivers and still exits 0. See"
            " docs/internal/GpuClockPinning.md."
        )
    if "Disabled" in output:
        raise Exception(
            "persistence mode is disabled, so the driver may reset clock state"
            " between runs; enable it with"
            " `sudo nvidia-smi --persistence-mode=1`. Note that persistence"
            " alone does not pin the SM clock — see"
            " docs/internal/GpuClockPinning.md."
        )


target_accelerator_values = {
    "NVIDIA": [
        "nvidia:sm_52",
        "nvidia:sm_60",
        "nvidia:sm_61",
        "nvidia:sm_75",
        "nvidia:sm_80",
        "nvidia:sm_86",
        "nvidia:sm_87",
        "nvidia:sm_89",
        "nvidia:sm_90",
        "nvidia:sm_90a",
        "nvidia:sm_100",
        "nvidia:sm_100a",
        "nvidia:sm_120",
        "nvidia:sm_120a",
    ],
    "AMD": [
        "amdgpu:mi300x",
        "amdgpu:mi355x",
        "amdgpu:gfx942",
        "amdgpu:gfx950",
        "amdgpu:gfx1030",
        "amdgpu:gfx1033",
        "amdgpu:gfx1100",
        "amdgpu:gfx1101",
        "amdgpu:gfx1102",
        "amdgpu:gfx1103",
        "amdgpu:gfx1150",
        "amdgpu:gfx1151",
        "amdgpu:gfx1152",
        "amdgpu:gfx1200",
        "amdgpu:gfx1201",
    ],
    "Apple": ["metal:1", "metal:2", "metal:3", "metal:4"],
}


@functools.cache
def get_target_accelerator_helpstr() -> str:
    helpstr = ""
    for arch, target_list in target_accelerator_values.items():
        helpstr += f"\n\n# {arch}\n\n"
        helpstr += ",\n".join([f"'{x}'" for x in target_list])
    return helpstr


def check_valid_target_accelerator(target_accelerator: str) -> bool:
    return target_accelerator in flatten(target_accelerator_values.values())


def format_time(ms: float) -> str:
    """Format time in human-readable units.

    Args:
        ms: Time in milliseconds.

    Returns:
        Human-readable time string with appropriate units (ns/µs/ms/s).
    """
    if ms < 0.001:  # < 1µs
        return f"{ms * 1e6:.0f} ns"
    elif ms < 1:  # < 1ms
        return f"{ms * 1e3:.1f} µs"
    elif ms < 1000:  # < 1s
        return f"{ms:.1f} ms"
    else:
        return f"{ms / 1000:.2f} s"
