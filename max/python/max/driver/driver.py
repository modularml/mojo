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
"""MAX Driver APIs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from max._core.driver import (  # noqa: F401
    CPU,
    Accelerator,
    Device,
    DeviceQueue,
    accelerator_count,
)


def accelerator_api() -> str:
    """Returns the API used to program the accelerator."""
    if accelerator_count() > 0:
        return Accelerator().api
    return CPU().api


def accelerator_architecture_name() -> str:
    """Returns the architecture name of the accelerator device."""
    if accelerator_count() > 0:
        return Accelerator().architecture_name
    return CPU().architecture_name


@dataclass(frozen=True)
class DeviceSpec:
    """A device specification containing an ID and device type.

    Args:
        id: The device identifier.
        device_type: The device type, either ``cpu`` or ``gpu``.
            Defaults to ``cpu``.
    """

    id: int
    """The device identifier."""

    device_type: Literal["cpu", "gpu"] = "cpu"
    """The device type, either ``cpu`` or ``gpu``."""

    def __post_init__(self) -> None:
        if self.device_type == "gpu" and self.id < 0:
            raise ValueError(
                f"id provided {self.id} for accelerator must always be greater than 0"
            )

    @staticmethod
    def cpu(id: int = -1):  # noqa: ANN205
        """Creates a CPU device specification."""
        return DeviceSpec(id, "cpu")

    @staticmethod
    def accelerator(id: int = 0):  # noqa: ANN205
        """Creates an accelerator (GPU) device specification."""
        return DeviceSpec(id, "gpu")


def load_device(device_spec: DeviceSpec) -> Device:
    """Loads a :class:`Device` from a :class:`DeviceSpec`."""
    if device_spec.device_type == "cpu":
        return CPU(device_spec.id)

    num_devices_available = accelerator_count()
    if device_spec.id >= num_devices_available:
        if num_devices_available == 0:
            reason = "no devices were found."
        else:
            reason = f"only found {num_devices_available} devices."
        raise ValueError(f"Device {device_spec.id} was requested but {reason}")

    return Accelerator(device_spec.id)


def load_devices(device_specs: Sequence[DeviceSpec]) -> list[Device]:
    """Initializes and returns a list of :class:`Device` objects from a sequence of :class:`DeviceSpec` objects."""
    devices: list[Device] = []
    for device_spec in device_specs:
        devices.append(load_device(device_spec))
    return devices


def scan_available_devices() -> list[DeviceSpec]:
    """Returns all available accelerators, or a CPU device if none are available."""
    accel_count = accelerator_count()
    if accel_count == 0:
        return [DeviceSpec.cpu()]
    else:
        return [DeviceSpec.accelerator(i) for i in range(accel_count)]


def devices_exist(devices: list[DeviceSpec]) -> bool:
    """Returns ``True`` if all specified devices exist."""
    available_devices = scan_available_devices()
    for device in devices:
        if device.device_type != "cpu" and device not in available_devices:
            return False

    return True


def calculate_virtual_device_count(*device_spec_lists: list[DeviceSpec]) -> int:
    """Calculates the minimum virtual device count needed for the given device specs.

    Args:
        *device_spec_lists: One or more lists of :class:`DeviceSpec` objects (for example, main
            devices and draft devices).

    Returns:
        The minimum number of virtual devices needed (max GPU ID + 1), or 1 if no GPUs.
    """
    max_gpu_id = -1
    for device_specs in device_spec_lists:
        for device_spec in device_specs:
            if device_spec.device_type == "gpu":
                max_gpu_id = max(max_gpu_id, device_spec.id)

    return max(1, max_gpu_id + 1)


def calculate_virtual_device_count_from_cli(
    *device_inputs: str | list[int],
) -> int:
    """Calculates the minimum virtual device count from raw CLI inputs (before parsing).

    This helper works with the raw device input strings or lists before they're
    parsed into :class:`DeviceSpec` objects. Used when virtual device mode needs to be
    enabled before device validation occurs.

    Args:
        *device_inputs: One or more raw device inputs, either strings like ``gpu:0,1,2``
            or lists of integers like ``[0, 1, 2]``.

    Returns:
        ``max(1, max_gpu_id + 1)``: the slot count implied by the highest GPU index in the
        raw inputs, with a floor of ``1``. In particular, ``gpu:all`` when
        ``accelerator_count()`` is ``0`` leaves ``max_gpu_id == -1`` and still returns
        ``1`` (same as no GPU IDs in the CLI). That floor is for virtual-device setup;
        actual ``gpu:all`` with zero GPUs still fails later when device specs are built.
    """
    max_gpu_id = -1
    for device_input in device_inputs:
        if isinstance(device_input, list):
            # Handle list of GPU IDs like [0, 1, 2]
            if len(device_input) > 0:
                max_gpu_id = max(max_gpu_id, max(device_input))
        elif device_input == "gpu":
            # Handle "gpu" which means GPU 0.
            max_gpu_id = max(max_gpu_id, 0)
        elif (
            isinstance(device_input, str) and device_input.lower() == "gpu:all"
        ):
            # Every visible GPU: need slots 0 .. N-1. If N==0, max_gpu_id stays -1 and the
            # final max(1, ...) enforces the same minimum as other no-GPU CLI paths.
            max_gpu_id = max(max_gpu_id, accelerator_count() - 1)
        elif isinstance(device_input, str) and device_input.lower().startswith(
            "gpu:"
        ):
            suffix = device_input.split(":", 1)[1]
            for part in suffix.split(","):
                part_stripped = part.strip()
                if not part_stripped or part_stripped.lower() == "all":
                    continue
                max_gpu_id = max(max_gpu_id, int(part_stripped))

    return max(1, max_gpu_id + 1)
