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

import pytest
from max.driver import CPU, Accelerator


def test_cpu_device() -> None:
    # We should be able to create a CPU device.
    cpu = CPU()
    assert "cpu" in str(cpu)
    assert cpu.is_host


@pytest.mark.skip(reason="MSDK-834")
def test_accelerator_device_creation_error() -> None:
    # Creating a Accelerator device on a machine without a GPU should raise an error.
    with pytest.raises(ValueError, match="failed to create device:"):
        _ = Accelerator()


def test_cpu_is_host_unified() -> None:
    # The CPU device is the host, so there is no second pool to unify with.
    assert CPU().is_host_unified


def test_equality() -> None:
    # We should be able to validate that two devices are the same.
    cpu_one = CPU()
    cpu_two = CPU()
    assert cpu_one == cpu_two


def test_stats() -> None:
    # We should be able to query utilization stats for the device.
    cpu = CPU()
    assert "free_memory" in cpu.stats
    assert "total_memory" in cpu.stats
    assert cpu.stats["free_memory"] <= cpu.stats["total_memory"]


def test_max_single_alloc_size() -> None:
    # Default implementation returns the device's total memory; should be a
    # positive integer.
    cpu = CPU()
    assert isinstance(cpu.max_single_alloc_size, int)
    assert cpu.max_single_alloc_size > 0


def test_api() -> None:
    # We should be to check the API used for programming the device.
    # This is more relevant for accelerators, for the host device, expect "cpu".
    cpu = CPU()
    assert "cpu" == cpu.api


def test_cpu_id() -> None:
    # The CPU id should always be 0.
    cpu = CPU()
    assert 0 == cpu.id


def test_cpu_architecture_name() -> None:
    cpu = CPU()
    with pytest.raises(
        Exception, match="failed to get device architecture name"
    ):
        _ = cpu.architecture_name
