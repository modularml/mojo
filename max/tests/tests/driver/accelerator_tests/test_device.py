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

from max.driver import (
    CPU,
    Accelerator,
    accelerator_architecture_name,
)
from max.graph import DeviceKind, DeviceRef


def test_accelerator_device() -> None:
    # We should be able to create a Accelerator device.
    dev = Accelerator()
    assert "gpu" in str(dev)
    assert not dev.is_host


def test_is_host_unified_is_reachable_and_host_is_unified() -> None:
    # Whether an accelerator is unified depends on the model, so the value is
    # pinned in GPUDeviceContextTest; here we only need the property to reach
    # the driver without raising.
    assert isinstance(Accelerator().is_host_unified, bool)
    assert CPU().is_host_unified


def test_accelerator_is_compatible() -> None:
    accelerator = Accelerator()
    assert accelerator.is_compatible


def test_accelerator_device_label_id() -> None:
    # Test the label property and attempt to map to graph.DeviceRef.
    dev_id = 0
    default_device = Accelerator()
    device = Accelerator(id=dev_id)
    assert "gpu" in device.label
    assert dev_id == device.id
    assert dev_id == default_device.id
    dev_from_runtime = DeviceRef(device.label, device.id)
    dev1_from_runtime = DeviceRef(DeviceKind(device.label), device.id)
    assert dev_from_runtime == DeviceRef.GPU(dev_id)
    assert dev1_from_runtime == DeviceRef.GPU(dev_id)


def scoped_device() -> None:
    _ = Accelerator(0)  # NOTE: device ID is intentionally explicit.


def test_stress_accelerator_device() -> None:
    # We should be able to call Accelerator() many times, and get cached outputs.
    devices = [Accelerator() for _ in range(64)]
    assert len({id(dev) for dev in devices}) == 1

    # TODO(MSDK-1220): move this before the above assert when the context no
    # longer leaks. Until then, this should still test that the default device
    # ID and explicit 0 ID share a device cache entry.
    for _ in range(64):
        scoped_device()


def test_equality() -> None:
    # We should be able to test the equality of devices.
    cpu = CPU()
    accel = Accelerator()

    assert cpu != accel


def test_stats() -> None:
    # We should be able to query utilization stats for the device.
    accel = Accelerator()
    stats = accel.stats
    assert "free_memory" in stats
    assert "total_memory" in stats
    assert "graph_mem_reserved" in stats
    assert "graph_mem_used" in stats


def test_accelerator_can_access_self() -> None:
    """Accelerator should not be able to access itself."""
    accel = Accelerator()
    assert not accel.can_access(accel), "Device should not access itself."


def test_accelerator_can_access_cpu() -> None:
    """Accelerator should typically not have direct peer access to CPU."""
    gpu = Accelerator()
    cpu = CPU()
    assert not gpu.can_access(cpu), "GPU should not directly access CPU memory."


def test_cpu_can_access_accelerator() -> None:
    """CPUs normally cannot directly access accelerator memory."""
    gpu = Accelerator()
    cpu = CPU()
    assert not cpu.can_access(gpu), (
        "CPUs shouldn't be able to access accelerator memory."
    )


def test_cpu_can_access_cpu() -> None:
    """CPU should not report peer access to itself."""
    cpu = CPU()
    another_cpu = CPU()
    assert not cpu.can_access(another_cpu), "CPU should not access another CPU."


def test_accelerator_architecture_name() -> None:
    """Accelerator should return architecture name (e.g., gfx942, sm_80)."""
    accelerator = Accelerator()

    archname = accelerator.architecture_name
    assert isinstance(archname, str)
    assert len(archname) > 0, (
        "Accelerator should have a non-empty architecture name"
    )


def test_accelerator_architecture_name_function() -> None:
    arch = accelerator_architecture_name()
    assert isinstance(arch, str)
    assert len(arch) > 0, (
        "Accelerator should have a non-empty architecture name"
    )
