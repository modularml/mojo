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
"""Test DevicePinnedBuffer class."""

import threading

import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    CompletionFlag,
    DevicePinnedBuffer,
    Usage,
    accelerator_count,
)
from max.dtype import DType


def test_device_pinned_buffer_type_cpu() -> None:
    """Test that DevicePinnedBuffer raises error on CPU."""
    cpu = CPU()

    # DevicePinnedBuffer should raise ValueError on CPU devices
    with pytest.raises(
        ValueError, match="DevicePinnedBuffer requires a non-host device"
    ):
        DevicePinnedBuffer(dtype=DType.float32, shape=[10], device=cpu)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer GPU tests require GPU",
)
def test_device_pinned_buffer_type_gpu() -> None:
    """Test that DevicePinnedBuffer creates pinned memory on GPU."""
    gpu = Accelerator()
    buffer = DevicePinnedBuffer(dtype=DType.float32, shape=[10], device=gpu)

    # Should be an instance of DevicePinnedBuffer
    assert isinstance(buffer, DevicePinnedBuffer)
    assert isinstance(buffer, Buffer)

    # On GPU, should be pinned
    assert buffer.pinned
    # Pinned buffers are associated with GPU device context
    assert not buffer.device.is_host


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer GPU tests require GPU",
)
def test_device_pinned_buffer_zeros() -> None:
    """Test DevicePinnedBuffer.zeros() static method."""
    gpu = Accelerator()
    buffer = DevicePinnedBuffer.zeros(
        shape=[5, 3], dtype=DType.int32, device=gpu
    )

    # Should be an instance of DevicePinnedBuffer
    assert isinstance(buffer, DevicePinnedBuffer)
    assert isinstance(buffer, Buffer)

    # Should be pinned
    assert buffer.pinned

    # Should be initialized to zeros
    np_array = buffer.to_numpy()
    assert np_array.shape == (5, 3)
    assert np.all(np_array == 0)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer GPU tests require GPU",
)
def test_device_pinned_buffer_data_transfer() -> None:
    """Test that DevicePinnedBuffer can transfer data to/from GPU."""
    gpu = Accelerator()

    # Create a pinned buffer and fill it with data
    host_buffer = DevicePinnedBuffer(
        dtype=DType.float32, shape=[100], device=gpu
    )
    host_np = host_buffer.to_numpy()
    host_np[:] = np.arange(100, dtype=np.float32)

    # Transfer to GPU
    gpu_buffer = host_buffer.to(gpu)

    # Transfer back to a new pinned buffer
    result_buffer = DevicePinnedBuffer(
        dtype=DType.float32, shape=[100], device=gpu
    )
    result_buffer.inplace_copy_from(gpu_buffer)

    # DevicePinnedBuffer skips automatic synchronization in to_numpy(),
    # so we must explicitly synchronize to ensure the async copy completes.
    gpu.synchronize()

    # Verify data is correct
    result_np = result_buffer.to_numpy()
    expected = np.arange(100, dtype=np.float32)
    assert np.allclose(result_np, expected)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer GPU tests require GPU",
)
def test_device_pinned_buffer_slice_preserves_type() -> None:
    """Slicing a DevicePinnedBuffer should stay a DevicePinnedBuffer (DRIV-7).

    A slice that decays to a plain Buffer loses the no-synchronization
    behavior, so reads like to_numpy() would trigger an unexpected
    cuStreamSynchronize.
    """
    gpu = Accelerator()
    buffer = DevicePinnedBuffer(dtype=DType.float32, shape=[10], device=gpu)

    sliced = buffer[:5]
    assert isinstance(sliced, DevicePinnedBuffer)
    assert sliced.pinned
    assert sliced.shape == (5,)

    # A multi-dimensional slice should keep the type too.
    buffer2d = DevicePinnedBuffer(dtype=DType.float32, shape=[4, 3], device=gpu)
    sliced2d = buffer2d[1:3, :]
    assert isinstance(sliced2d, DevicePinnedBuffer)
    assert sliced2d.pinned


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer GPU tests require GPU",
)
def test_device_pinned_buffer_view_preserves_type() -> None:
    """view() on a DevicePinnedBuffer should stay a DevicePinnedBuffer."""
    gpu = Accelerator()
    buffer = DevicePinnedBuffer(dtype=DType.float32, shape=[10], device=gpu)

    viewed = buffer.view(DType.uint8)
    assert isinstance(viewed, DevicePinnedBuffer)
    assert viewed.pinned


def test_device_pinned_buffer_cpu_zeros() -> None:
    """Test DevicePinnedBuffer.zeros() raises error on CPU."""
    cpu = CPU()

    # DevicePinnedBuffer.zeros() should raise ValueError on CPU devices
    with pytest.raises(
        ValueError, match="DevicePinnedBuffer requires a non-host device"
    ):
        DevicePinnedBuffer.zeros(shape=[4, 2], dtype=DType.float64, device=cpu)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer with events requires GPU",
)
def test_device_pinned_buffer_with_events() -> None:
    """Test DevicePinnedBuffer with DeviceEvent for explicit synchronization."""
    gpu = Accelerator()
    stream = gpu.default_queue

    # Create a pinned buffer for efficient host-device transfers
    host_buffer = DevicePinnedBuffer(
        dtype=DType.float32, shape=[1000], device=gpu
    )
    host_np = host_buffer.to_numpy()
    host_np[:] = np.arange(1000, dtype=np.float32)

    # Transfer to GPU and record an event after the transfer
    gpu_buffer = host_buffer.to(gpu)
    event = stream.record_event()

    # Event should eventually be ready
    event.synchronize()
    assert event.is_ready()

    # Transfer back using pinned buffer
    result_buffer = DevicePinnedBuffer(
        dtype=DType.float32, shape=[1000], device=gpu
    )
    result_buffer.inplace_copy_from(gpu_buffer)

    # Record another event after the copy
    copy_event = stream.record_event()
    copy_event.synchronize()

    # Verify data is correct
    result_np = result_buffer.to_numpy()
    expected = np.arange(1000, dtype=np.float32)
    assert np.allclose(result_np, expected)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer sync-contract tests require GPU",
)
def test_device_pinned_buffer_to_numpy_does_not_synchronize() -> None:
    """``to_numpy`` on a pinned buffer returns without synchronizing (DRIV-311).

    Contract 4a: ``DevicePinnedBuffer`` overrides ``__dlpack__`` to skip the
    device synchronize a plain ``Buffer`` does, so the read proceeds while the
    default stream is gated. A full ``device.synchronize()`` is the positive
    control: it must stay blocked while the gate is held, proving the gate is
    engaged and the pinned result meaningful.
    """
    gpu = Accelerator()
    if gpu.api not in ("cuda", "hip"):
        pytest.skip("stream host-value gating requires CUDA/HIP")

    sentinel = np.arange(1, 5, dtype=np.float32)
    pinned = DevicePinnedBuffer(dtype=DType.float32, shape=[4], device=gpu)
    pinned.to_numpy()[:] = sentinel

    flag = CompletionFlag(gpu)
    gate_stream = gpu.default_queue
    pinned_done = threading.Event()
    control_done = threading.Event()
    pinned_box: dict[str, object] = {}

    def read_pinned() -> None:
        try:
            pinned_box["value"] = pinned.to_numpy()
        finally:
            pinned_done.set()

    def sync_device() -> None:
        try:
            gpu.synchronize()
        finally:
            control_done.set()

    pinned_thread = threading.Thread(target=read_pinned, daemon=True)
    control_thread = threading.Thread(target=sync_device, daemon=True)

    gate_stream.wait_for_host_value(flag, 1)
    try:
        pinned_thread.start()
        control_thread.start()
        assert pinned_done.wait(timeout=10.0), (
            "pinned to_numpy blocked on the gated stream; it synchronized"
        )
        assert not control_done.wait(timeout=2.0), (
            "device.synchronize() returned while gated; the gate is not"
            " engaged so the pinned result is not meaningful"
        )
    finally:
        flag.signal(1)
        pinned_thread.join(timeout=60.0)
        control_thread.join(timeout=60.0)

    assert not pinned_thread.is_alive() and not control_thread.is_alive()
    np.testing.assert_array_equal(pinned_box["value"], sentinel)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer sync-contract tests require GPU",
)
def test_device_pinned_buffer_to_numpy_is_zero_copy_view() -> None:
    """``to_numpy`` on a pinned buffer aliases its memory (DRIV-311).

    Pins contract 4b: because the export is zero-copy, the numpy array does not
    own its data and a later host write to the pinned buffer is visible through
    the array returned earlier.
    """
    gpu = Accelerator()
    pinned = DevicePinnedBuffer(dtype=DType.int32, shape=[8], device=gpu)

    view = pinned.to_numpy()
    assert not view.flags.owndata

    written = np.arange(10, 18, dtype=np.int32)
    pinned.to_numpy()[:] = written
    np.testing.assert_array_equal(view, written)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="staging Buffer GPU tests require GPU",
)
def test_buffer_staging_matches_device_pinned() -> None:
    """Buffer(usage=STAGING) allocates what DevicePinnedBuffer allocates."""
    gpu = Accelerator()
    buf = Buffer(
        dtype=DType.float32, shape=[10], device=gpu, usage=Usage.STAGING
    )
    pinned = DevicePinnedBuffer(dtype=DType.float32, shape=[10], device=gpu)

    assert buf.pinned == pinned.pinned == True  # noqa: E712
    assert buf.usage == pinned.usage == Usage.STAGING
    assert not buf.device.is_host


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="staging Buffer GPU tests require GPU",
)
def test_staging_usage_survives_slice_and_view() -> None:
    """Slices/views share storage, so they keep the staging contract."""
    gpu = Accelerator()
    buf = Buffer(
        dtype=DType.float32, shape=[4, 4], device=gpu, usage=Usage.STAGING
    )
    # Buffer.get requires as many indices as the rank.
    assert buf[0, :].usage == Usage.STAGING
    assert buf[0, :].pinned
    assert buf.view(DType.int32, [4, 4]).usage == Usage.STAGING


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="staging Buffer GPU tests require GPU",
)
def test_staging_zeros() -> None:
    gpu = Accelerator()
    buf = Buffer.zeros(
        shape=[5, 3], dtype=DType.int32, device=gpu, usage=Usage.STAGING
    )
    assert buf.usage == Usage.STAGING
    # zeros() enqueues an async memset and the staging export does not
    # synchronize, so the caller must order the read itself.
    gpu.synchronize()
    np.testing.assert_array_equal(
        buf.to_numpy(), np.zeros((5, 3), dtype=np.int32)
    )


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="staging Buffer GPU tests require GPU",
)
@pytest.mark.parametrize("make_buffer", ["staging", "device_pinned"])
def test_staging_dlpack_does_not_synchronize(make_buffer: str) -> None:
    """The staging read returns while gated work is still pending.

    Inverse of DRIV-311 contract 1: a host read of staging memory must NOT
    absorb pending device work. The stream is gated on a host-signalled
    CompletionFlag; a non-synchronizing read finishes while the gate is
    held. DevicePinnedBuffer is parametrized alongside to pin parity.
    """
    gpu = Accelerator()
    if gpu.api not in ("cuda", "hip"):
        pytest.skip("stream host-value gating requires CUDA/HIP")

    if make_buffer == "staging":
        host_buf = Buffer(
            dtype=DType.int32, shape=[4], device=gpu, usage=Usage.STAGING
        )
    else:
        host_buf = DevicePinnedBuffer(dtype=DType.int32, shape=[4], device=gpu)
    src = Buffer.from_numpy(np.arange(1, 5, dtype=np.int32)).to(gpu)

    flag = CompletionFlag(gpu)
    done = threading.Event()

    def worker() -> None:
        host_buf.to_numpy()
        done.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    gpu.default_queue.wait_for_host_value(flag, 1)
    try:
        host_buf.inplace_copy_from(src)
        worker_thread.start()
        completed_while_gated = done.wait(timeout=2.0)
    finally:
        flag.signal(1)
        worker_thread.join(timeout=60.0)

    assert not worker_thread.is_alive()
    assert completed_while_gated, (
        "to_numpy()/__dlpack__ on a staging buffer blocked on gated device"
        " work; the staging export must not synchronize"
    )
