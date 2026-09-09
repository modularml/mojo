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

import numpy as np
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    DevicePinnedBuffer,
    batch_inplace_copy,
)
from max.dtype import DType


def test_accelerator_peer_access() -> None:
    """Test peer access between multiple accelerators."""
    gpu0 = Accelerator(id=0)
    gpu1 = Accelerator(id=1)

    can_access_0_to_1 = gpu0.can_access(gpu1)
    can_access_1_to_0 = gpu1.can_access(gpu0)

    # Typically, peer access is symmetric, but hardware-dependent.
    assert can_access_0_to_1 == can_access_1_to_0, (
        "Peer access should be symmetric."
    )


def test_to_multiple_devices() -> None:
    cpu = CPU()
    acc0 = Accelerator(id=0)
    acc1 = Accelerator(id=1)

    tensor = Buffer(dtype=DType.int32, shape=(3, 3), device=cpu)
    # GEX-2624: CPU to CPU copies not supported via Buffer.to()
    tensors = tensor.to([acc0, acc1])
    assert len(tensors) == 2
    assert tensors[0].device == acc0
    assert tensors[1].device == acc1


def test_from_device() -> None:
    cpu = CPU()
    acc0 = Accelerator(id=0)
    acc1 = Accelerator(id=1)

    tensor = Buffer(dtype=DType.int32, shape=(3, 3), device=acc0)
    tensors = tensor.to([cpu, acc0, acc1])
    assert len(tensors) == 3
    assert tensors[0].device == cpu
    assert tensors[1].device == acc0
    assert tensors[2].device == acc1


def _test_pinned_device_copy(
    pinned_gpu_id: int,
    target_gpu_id: int,
    pattern_before: int,
    pattern_after: int,
    tensor_size: int = 1024,
) -> bool:
    """
    Helper to test if copying from pinned host memory to device memory works.

    Creates pinned host memory on pinned_gpu_id, fills it with pattern_before,
    copies to target_gpu_id, then modifies the pinned memory to pattern_after.
    Returns True if the GPU tensor still contains pattern_before (correct copy).
    """
    gpus = [Accelerator(id=i) for i in range(2)]
    cpu_device = CPU()

    # Create pinned memory on specified GPU
    pinned_cpu_tensor = DevicePinnedBuffer(
        dtype=DType.int8,
        shape=(tensor_size,),
        device=gpus[pinned_gpu_id],
    )

    # Fill with pattern_before
    pattern_before_data = np.full(tensor_size, pattern_before, dtype=np.int8)
    pattern_before_tensor = Buffer.from_numpy(pattern_before_data)
    pinned_cpu_tensor.inplace_copy_from(pattern_before_tensor)

    # Copy to target GPU
    gpu_tensor = pinned_cpu_tensor.to(gpus[target_gpu_id])
    assert not gpu_tensor.pinned

    # Synchronize to ensure the copy from pinned memory has completed
    # before we modify the source buffer
    gpus[target_gpu_id].synchronize()

    # Modify pinned memory to pattern_after
    pattern_after_data = np.full(tensor_size, pattern_after, dtype=np.int8)
    pattern_after_tensor = Buffer.from_numpy(pattern_after_data)
    pinned_cpu_tensor.inplace_copy_from(pattern_after_tensor)

    # Copy GPU tensor back to CPU and check contents
    gpu_on_cpu = gpu_tensor.copy(device=cpu_device)
    gpu_values = gpu_on_cpu.to_numpy()

    # Should contain pattern_before (the value before modification)
    return bool(np.all(gpu_values == pattern_before))


def test_pinned_to_same_device_copy() -> None:
    """Test copying from pinned host memory to same device memory.

    Regression test for GEX-2864: Copying from GPU 0 pinned host memory to
    GPU 0 device memory should allocate device memory and perform a copy,
    not return a reference to the pinned memory.
    """
    result = _test_pinned_device_copy(
        pinned_gpu_id=0,
        target_gpu_id=0,
        pattern_before=42,
        pattern_after=99,
    )
    assert result, (
        "Copy from pinned host memory to same GPU device memory failed. "
        "Expected GPU tensor to contain original data (42), not modified data (99). "
    )


def test_pinned_to_different_device_copy() -> None:
    """Test copying from pinned host memory to different device memory.

    This should work correctly - validates the test infrastructure.
    """
    result = _test_pinned_device_copy(
        pinned_gpu_id=0,
        target_gpu_id=1,
        pattern_before=77,
        pattern_after=55,
    )
    assert result, (
        "Copy from pinned host memory to different GPU device memory failed."
        "Expected GPU tensor to contain original data (77), not modified data (55). "
    )


def test_non_pinned_cross_device_copy() -> None:
    """Test copying non-pinned memory from one GPU to another."""
    gpu0 = Accelerator(id=0)
    gpu1 = Accelerator(id=1)
    cpu = CPU()
    tensor_size = 1024

    # Create tensor on GPU 0
    data = np.full(tensor_size, 42, dtype=np.int8)
    gpu0_tensor = Buffer.from_numpy(data).to(gpu0)

    # Copy to GPU 1
    gpu1_tensor = gpu0_tensor.to(gpu1)
    assert gpu1_tensor.device == gpu1

    # Verify the data was copied correctly
    gpu1.synchronize()
    result = gpu1_tensor.copy(device=cpu).to_numpy()
    assert np.all(result == 42), (
        "Cross-device copy of non-pinned memory failed. "
        "Expected tensor to contain 42."
    )


def test_batch_inplace_copy_mixed_destination_devices_parity() -> None:
    """Destinations spanning two GPUs are grouped by the driver, not rejected.

    A submission only orders the writes on its own stream, so the driver splits
    the batch per destination device. Interleave the destinations so a naive
    single-stream submit would be observable as a wrong-stream write.
    """
    gpu0 = Accelerator(id=0)
    gpu1 = Accelerator(id=1)

    n = 4
    srcs = [
        Buffer.from_numpy(np.full((8,), float(i + 1), dtype=np.float32))
        for i in range(n)
    ]
    devices = [gpu0, gpu1, gpu0, gpu1]
    batch_dsts = [Buffer(DType.float32, (8,), device=dev) for dev in devices]
    loop_dsts = [Buffer(DType.float32, (8,), device=dev) for dev in devices]

    batch_inplace_copy(batch_dsts, srcs)
    for dst, src in zip(loop_dsts, srcs, strict=True):
        dst.inplace_copy_from(src)

    gpu0.synchronize()
    gpu1.synchronize()
    for i, (batch_dst, loop_dst) in enumerate(
        zip(batch_dsts, loop_dsts, strict=True)
    ):
        np.testing.assert_array_equal(
            batch_dst.to(CPU()).to_numpy(),
            loop_dst.to(CPU()).to_numpy(),
            err_msg=f"mixed-destination batch vs per-copy mismatch at {i}",
        )


def test_batch_inplace_copy_mixed_source_devices_parity() -> None:
    """One destination device fed by local, peer and host sources at once.

    The driver brackets each foreign source stream around the single submit
    rather than de-batching, so the mix must match a per-copy loop.
    """
    gpu0 = Accelerator(id=0)
    gpu1 = Accelerator(id=1)
    assert gpu0.can_access(gpu1), "peer access required for peer DtoD batch"

    values = [np.full((8,), float(i + 1), dtype=np.float32) for i in range(3)]
    srcs = [
        Buffer.from_numpy(values[0]).to(gpu0),
        Buffer.from_numpy(values[1]).to(gpu1),
        Buffer.from_numpy(values[2]),
    ]
    batch_dsts = [Buffer(DType.float32, (8,), device=gpu0) for _ in srcs]
    loop_dsts = [Buffer(DType.float32, (8,), device=gpu0) for _ in srcs]

    gpu0.synchronize()
    gpu1.synchronize()
    batch_inplace_copy(batch_dsts, srcs)
    for dst, src in zip(loop_dsts, srcs, strict=True):
        dst.inplace_copy_from(src)

    gpu0.synchronize()
    for i, (batch_dst, loop_dst) in enumerate(
        zip(batch_dsts, loop_dsts, strict=True)
    ):
        np.testing.assert_array_equal(
            batch_dst.to(CPU()).to_numpy(),
            loop_dst.to(CPU()).to_numpy(),
            err_msg=f"mixed-source batch vs per-copy mismatch at index {i}",
        )


def test_batch_inplace_copy_peer_device_parity() -> None:
    """Batched peer DtoD (GPU1→GPU0) matches a per-copy inplace_copy_from loop.

    All destinations share one device (GPU0); sources live on GPU1. Exercises
    the memBatchCopy peer path vs sequential enqueueCopyDeviceToDevice.
    """
    gpu0 = Accelerator(id=0)
    gpu1 = Accelerator(id=1)
    assert gpu0.can_access(gpu1), "peer access required for peer DtoD batch"

    n = 5
    srcs = [
        Buffer.from_numpy(np.full((8,), float(i + 1), dtype=np.float32)).to(
            gpu1
        )
        for i in range(n)
    ]
    batch_dsts = [Buffer(DType.float32, (8,), device=gpu0) for _ in range(n)]
    loop_dsts = [Buffer(DType.float32, (8,), device=gpu0) for _ in range(n)]

    gpu1.synchronize()
    batch_inplace_copy(batch_dsts, srcs)
    for dst, src in zip(loop_dsts, srcs, strict=True):
        dst.inplace_copy_from(src)

    gpu0.synchronize()
    for i, (batch_dst, loop_dst) in enumerate(
        zip(batch_dsts, loop_dsts, strict=True)
    ):
        np.testing.assert_array_equal(
            batch_dst.to(CPU()).to_numpy(),
            loop_dst.to(CPU()).to_numpy(),
            err_msg=f"peer DtoD batch vs per-copy mismatch at index {i}",
        )
