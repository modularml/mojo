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


"""Custom Click Options used in pipelines"""

from __future__ import annotations

from typing import Any

import click
from max.driver import DeviceSpec
from max.pipelines.lib.device_specs import (
    DeviceHandle,
    device_specs_from_normalized_device_handle,
    get_requested_gpu_ids,
    normalize_device_specs_input,
    validate_gpu_ids,
)


# TODO: This whole interface will eventually be deprecated once we move away from
# click.
class DevicesOptionType(click.ParamType):
    name = "devices"

    @staticmethod
    def _get_requested_gpu_ids(devices: DeviceHandle) -> list[int]:
        return get_requested_gpu_ids(devices=devices)

    @staticmethod
    def _validate_gpu_ids(
        gpu_ids: list[int], available_gpu_ids: list[int]
    ) -> None:
        validate_gpu_ids(gpu_ids, available_gpu_ids)

    @staticmethod
    def device_specs(devices: DeviceHandle) -> list[DeviceSpec]:
        """Converts parsed devices input into validated :obj:`DeviceSpec` objects.

        Args:
            devices: The value provided by the --devices option.
                Valid arguments:
                - "cpu"   → use the CPU,
                - "gpu"   → default to GPU 0, or,
                - "gpu:all" → use every visible GPU, or,
                - a list of ints (GPU IDs).

        Raises:
            ValueError: If a requested GPU ID is invalid.

        Returns:
            A list of DeviceSpec objects.
        """
        return device_specs_from_normalized_device_handle(devices=devices)

    def convert(
        self,
        value: Any,
        param: click.Parameter | None = None,
        ctx: click.Context | None = None,
    ) -> DeviceHandle:
        try:
            return normalize_device_specs_input(value)
        except ValueError as e:
            self.fail(str(e), param, ctx)
