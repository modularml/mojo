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
import pytest
import torch
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    DevicePinnedBuffer,
    DeviceQueue,
    Usage,
    accelerator_api,
    batch_inplace_copy,
)
from max.dtype import DType


def test_from_numpy_accelerator() -> None:
    # A user should be able to create an accelerator tensor from a numpy array.
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    tensor = Buffer.from_numpy(arr).to(Accelerator())
    assert tensor.shape == (2, 3)
    assert tensor.dtype == DType.int32


def test_is_host_accelerator() -> None:
    # Accelerator tensors should be marked as not being on-host.
    assert not Buffer(DType.int32, (1, 1), device=Accelerator()).is_host


def test_host_device_copy() -> None:
    # We should be able to freely copy tensors between host and device.
    host_tensor = Buffer.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    device_tensor = host_tensor.copy(Accelerator())
    tensor = device_tensor.copy(CPU())

    assert tensor.shape == host_tensor.shape
    assert tensor.dtype == host_tensor.dtype
    assert tensor[0].item() == 1
    assert tensor[1].item() == 2
    assert tensor[2].item() == 3


def test_device_device_copy() -> None:
    # We should be able to freely copy tensors between device and device.
    acc = Accelerator()

    device_tensor1 = Buffer.from_numpy(np.array([1, 2, 3], dtype=np.int32)).to(
        acc
    )
    device_tensor2 = device_tensor1.copy(acc)
    tensor = device_tensor2.copy(CPU())

    assert tensor.shape == device_tensor1.shape
    assert tensor.dtype == DType.int32
    assert tensor[0].item() == 1
    assert tensor[1].item() == 2
    assert tensor[2].item() == 3


def test_torch_tensor_conversion() -> None:
    # Our tensors should be convertible to and from Torch tensors. We have to a
    # bunch of juggling between host and device because we don't have a
    # Accelerator-compatible version of torch available yet.
    torch_tensor = torch.reshape(torch.arange(1, 11, dtype=torch.int32), (2, 5))
    host_tensor = Buffer.from_dlpack(torch_tensor)
    acc_tensor = host_tensor.to(Accelerator())
    assert acc_tensor.shape == (2, 5)
    assert acc_tensor.dtype == DType.int32
    host_tensor = acc_tensor.to(CPU())
    torch_tensor_copy = torch.from_dlpack(host_tensor)
    assert torch.all(torch.eq(torch_tensor, torch_tensor_copy))


def test_to_device() -> None:
    cpu = CPU()
    acc = Accelerator()

    host_tensor = Buffer(dtype=DType.int32, shape=(3, 3), device=cpu)
    acc_tensor = host_tensor.to(acc)

    assert cpu == host_tensor.device
    assert acc == acc_tensor.device

    assert acc != host_tensor.device
    assert cpu != acc_tensor.device


def test_zeros() -> None:
    # We should be able to initialize an all-zero tensor.
    tensor = Buffer.zeros((3, 3), DType.int32, device=Accelerator())
    host_tensor = tensor.to(CPU())
    assert np.array_equal(
        host_tensor.to_numpy(), np.zeros((3, 3), dtype=np.int32)
    )


DLPACK_DTYPES = {
    DType.int8: torch.int8,
    DType.int16: torch.int16,
    DType.int32: torch.int32,
    DType.int64: torch.int64,
    DType.uint8: torch.uint8,
    DType.uint16: torch.uint16,
    DType.uint32: torch.uint32,
    DType.uint64: torch.uint64,
    DType.float16: torch.float16,
    DType.float32: torch.float32,
    DType.float64: torch.float64,
}


def test_dlpack_accelerator() -> None:
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for dtype, torch_dtype in DLPACK_DTYPES.items():
        tensor = Buffer(dtype, (1, 4))
        for j in range(4):
            tensor[0, j] = j
        acc_tensor = tensor.to(Accelerator())

        torch_tensor = torch.from_dlpack(acc_tensor)
        assert torch_tensor.dtype == torch_dtype
        assert acc_tensor.shape == torch_tensor.shape

        torch_tensor[0, 0] = 7
        assert acc_tensor[0, 0].to(CPU()).item() == 7


def test_from_dlpack() -> None:
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for dtype, torch_dtype in DLPACK_DTYPES.items():
        torch_tensor = torch.tensor([0, 1, 2, 3], dtype=torch_dtype).cuda()
        acc_tensor = Buffer.from_dlpack(torch_tensor)
        assert acc_tensor.dtype == dtype
        assert acc_tensor.shape == torch_tensor.shape

        torch_tensor[0] = 7
        assert acc_tensor[0].to(CPU()).item() == 7


def test_dlpack_device() -> None:
    tensor = Buffer(DType.int32, (3, 3), device=Accelerator())
    device_tuple = tensor.__dlpack_device__()
    assert len(device_tuple) == 2
    assert isinstance(device_tuple[0], int)
    if accelerator_api() == "hip":
        # 10 is the value of DLDeviceType::kDLROCM
        assert device_tuple[0] == 10
    else:
        # 2 is the value of DLDeviceType::kDLCUDA
        assert device_tuple[0] == 2
    assert isinstance(device_tuple[1], int)
    assert device_tuple[1] == 0  # should be the default device


def test_dlpack_device_pinned() -> None:
    gpu_device = Accelerator()

    pinned_tensor = DevicePinnedBuffer(DType.int32, (3, 3), device=gpu_device)
    device_tuple = pinned_tensor.__dlpack_device__()
    assert len(device_tuple) == 2
    assert isinstance(device_tuple[0], int)
    if gpu_device.api == "cuda":
        assert device_tuple[0] == 3  # DLDeviceType::kDLCUDAHost
    elif gpu_device.api == "hip":
        assert device_tuple[0] == 11  # DLDeviceType::kDLROCMHost
    else:
        raise ValueError(f"Unsupported device API: {gpu_device.api}")
    assert device_tuple[1] == 0  # should be the default device


def test_scalar() -> None:
    # We should be able to create scalar values on accelerators.
    acc = Accelerator()
    scalar = Buffer.scalar(5, DType.int32, device=acc)
    assert scalar.device == acc

    host_scalar = scalar.to(CPU())
    assert host_scalar.item() == 5


def test_pinned_zeros() -> None:
    tensor = DevicePinnedBuffer.zeros((2, 1), DType.int32, device=Accelerator())
    assert tensor.pinned
    assert tensor[0, 0].item() == 0
    assert tensor[1, 0].item() == 0

    if accelerator_api() == "hip":
        pytest.skip(
            "FIXME SERVOPT-947: __setitem__ / __getitem__ is buggy on HIP."
        )

    tensor[1, 0] = 42
    assert tensor[1, 0].item() == 42


def test_accelerator_to_numpy() -> None:
    acc = Accelerator()
    tensor = Buffer.zeros((3, 3), DType.int32, device=acc)

    assert np.array_equal(tensor.to_numpy(), np.zeros((3, 3), dtype=np.int32))


def test_d2h_inplace_copy_from_tensor_view() -> None:
    enumerated = np.zeros((5, 2, 3), dtype=np.int32)
    for i, j, k in np.ndindex(enumerated.shape):
        enumerated[i, j, k] = 100 * i + 10 * j + k

    all_nines = np.full((5, 2, 3), 999, dtype=np.int32)

    gpu_tensor = Buffer.from_numpy(enumerated).to(Accelerator())
    host_tensor = Buffer.from_numpy(all_nines)

    # Copy 3rd row of gpu_tensor into 1st row of host_tensor.
    host_tensor[1, :, :].inplace_copy_from(gpu_tensor[3, :, :])

    expected = np.array(
        [
            [[999, 999, 999], [999, 999, 999]],
            [[300, 301, 302], [310, 311, 312]],
            [[999, 999, 999], [999, 999, 999]],
            [[999, 999, 999], [999, 999, 999]],
            [[999, 999, 999], [999, 999, 999]],
        ]
    )
    assert np.array_equal(host_tensor.to_numpy(), expected)


@pytest.mark.parametrize("is_pinned", [True, False])
def test_to_numpy_inplace(is_pinned: bool) -> None:
    tensor: Buffer
    if is_pinned:
        tensor = DevicePinnedBuffer(
            shape=(4,), dtype=DType.int32, device=Accelerator()
        )
    else:
        tensor = Buffer(shape=(4,), dtype=DType.int32, device=CPU())

    # This should alias the existing memory.
    # It must NOT copy the memory to another buffer.
    tensor_np = tensor.to_numpy()

    # Assert that numpy does not own the underlying buffer. This means that it
    # is up to MAX to eventually deallocate the memory.
    assert not tensor_np.flags.owndata

    # Overwrite the numpy array.
    for i in range(4):
        tensor_np[i] = 2**i

    # Check if the memory is aliased.
    actual = tensor.to_numpy()
    expected = np.array([1, 2, 4, 8])
    assert np.array_equal(actual, expected)


def test_pinned_concatenate() -> None:
    arrs = [
        np.full((5,), 11, dtype=np.int32),
        np.full((5,), 22, dtype=np.int32),
        np.full((5,), 33, dtype=np.int32),
    ]

    pinned = DevicePinnedBuffer(
        shape=(15,), dtype=DType.int32, device=Accelerator()
    )

    pinned_np = pinned.to_numpy()
    np.concatenate(arrs, out=pinned_np)

    expected = np.array(
        [11, 11, 11, 11, 11, 22, 22, 22, 22, 22, 33, 33, 33, 33, 33]
    )

    assert np.array_equal(pinned.to_numpy(), expected)


@pytest.mark.parametrize("is_pinned", [True, False])
def test_zero_copy_on_to_stream_on_same_device(is_pinned: bool) -> None:
    gpu = Accelerator()
    data = np.arange(24).reshape(2, 3, 4).astype(np.int32)
    tensor = Buffer(
        shape=data.shape,
        dtype=DType.int32,
        device=gpu,
        usage=Usage.STAGING if is_pinned else Usage.DEFAULT,
    )
    tensor.inplace_copy_from(Buffer.from_numpy(data))
    stream1 = tensor.stream
    tensor1 = tensor.to(stream1)
    assert tensor1 is tensor
    assert tensor1.stream is stream1
    stream2 = DeviceQueue(gpu)
    assert stream2 != stream1
    tensor2 = tensor.to(stream2)
    assert tensor2 is not tensor
    assert tensor2.rank == tensor.rank
    assert tensor2.shape == tensor.shape
    assert tensor2.dtype == tensor.dtype
    assert tensor2.element_size == tensor.element_size
    assert tensor2.is_contiguous == tensor.is_contiguous
    assert tensor2.is_host == tensor.is_host
    assert tensor2.device is tensor.device
    assert tensor2.stream is stream2
    assert tensor2.stream is not stream1
    assert (tensor2.to_numpy() == tensor.to_numpy()).all()


def test_batch_inplace_copy_same_device_parity() -> None:
    """Batched H2D into one GPU matches a per-copy loop bit-exactly.

    Positive-path smoke for memBatchCopy (cuMemcpyBatchAsync on CUDA 12.8+,
    sequential fallback otherwise).
    """
    gpu = Accelerator()
    n = 5
    srcs = [
        Buffer.from_numpy(np.full((8,), float(i + 1), dtype=np.float32))
        for i in range(n)
    ]
    batch_dsts = [Buffer(DType.float32, (8,), device=gpu) for _ in range(n)]
    loop_dsts = [Buffer(DType.float32, (8,), device=gpu) for _ in range(n)]

    batch_inplace_copy(batch_dsts, srcs)
    for dst, src in zip(loop_dsts, srcs, strict=True):
        dst.inplace_copy_from(src)

    gpu.synchronize()
    for i, (batch_dst, loop_dst) in enumerate(
        zip(batch_dsts, loop_dsts, strict=True)
    ):
        np.testing.assert_array_equal(
            batch_dst.to(CPU()).to_numpy(),
            loop_dst.to(CPU()).to_numpy(),
            err_msg=f"batch vs per-copy mismatch at index {i}",
        )


def test_batch_inplace_copy_device_to_device_same_gpu_parity() -> None:
    """Batched same-GPU DtoD matches a per-copy inplace_copy_from loop.

    Exercises memBatchCopy on device-resident sources (cuMemcpyBatchAsync when
    available, sequential copyTo otherwise).
    """
    gpu = Accelerator()
    n = 5
    srcs = [
        Buffer.from_numpy(np.full((8,), float(i + 1), dtype=np.float32)).to(gpu)
        for i in range(n)
    ]
    batch_dsts = [Buffer(DType.float32, (8,), device=gpu) for _ in range(n)]
    loop_dsts = [Buffer(DType.float32, (8,), device=gpu) for _ in range(n)]

    batch_inplace_copy(batch_dsts, srcs)
    for dst, src in zip(loop_dsts, srcs, strict=True):
        dst.inplace_copy_from(src)

    gpu.synchronize()
    for i, (batch_dst, loop_dst) in enumerate(
        zip(batch_dsts, loop_dsts, strict=True)
    ):
        np.testing.assert_array_equal(
            batch_dst.to(CPU()).to_numpy(),
            loop_dst.to(CPU()).to_numpy(),
            err_msg=f"DtoD batch vs per-copy mismatch at index {i}",
        )


def test_batch_inplace_copy_mixed_host_device_sources_parity() -> None:
    """Mixed CPU-host + device sources in one batch succeed with value parity.

    Both kinds share one destination device, so they ride a single submission:
    UVA resolves each source pointer's direction inside cuMemcpyBatchAsync.
    """
    gpu = Accelerator()
    host_src = Buffer.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    device_src = Buffer.from_numpy(np.array([3.0, 4.0], dtype=np.float32)).to(
        gpu
    )
    batch_dsts = [
        Buffer(DType.float32, (2,), device=gpu),
        Buffer(DType.float32, (2,), device=gpu),
    ]
    loop_dsts = [
        Buffer(DType.float32, (2,), device=gpu),
        Buffer(DType.float32, (2,), device=gpu),
    ]
    srcs = [host_src, device_src]

    batch_inplace_copy(batch_dsts, srcs)
    for dst, src in zip(loop_dsts, srcs, strict=True):
        dst.inplace_copy_from(src)

    gpu.synchronize()
    for i, (batch_dst, loop_dst) in enumerate(
        zip(batch_dsts, loop_dsts, strict=True)
    ):
        np.testing.assert_array_equal(
            batch_dst.to(CPU()).to_numpy(),
            loop_dst.to(CPU()).to_numpy(),
            err_msg=f"mixed-source batch vs per-copy mismatch at index {i}",
        )


def test_batch_inplace_copy_skips_identity_on_accelerator() -> None:
    """Identity pairs on GPU are no-ops; adjacent pairs still copy."""
    gpu = Accelerator()
    stable = Buffer.from_numpy(np.array([42.0], dtype=np.float32)).to(gpu)
    other_src = Buffer.from_numpy(np.array([99.0], dtype=np.float32))
    other_dst = Buffer(DType.float32, (1,), device=gpu)
    other_dst.inplace_copy_from(
        Buffer.from_numpy(np.array([0.0], dtype=np.float32))
    )
    gpu.synchronize()

    # stable is both src and dst (identity no-op); other_* is a real H2D copy.
    batch_inplace_copy([stable, other_dst], [stable, other_src])
    gpu.synchronize()

    np.testing.assert_array_equal(stable.to(CPU()).to_numpy(), [42.0])
    np.testing.assert_array_equal(other_dst.to(CPU()).to_numpy(), [99.0])


def test_batch_inplace_copy_pinned_poison_pill_parity() -> None:
    """Mostly DtoD plus DevicePinnedBuffer pairs: value parity vs per-copy loop.

    Matches the production device-0 preface poison-pill shape: several pure
    device→device pairs plus pinned src→device and pinned→pinned. Pinned
    buffers are storage-host but ``is_host`` is false (device-based), so they
    used to de-batch the whole group. Asserts value parity only; no in-tree
    memcpy-API counter for submission shape.
    """
    gpu = Accelerator()
    n_dtod = 4

    def _fill_pinned(value: float) -> DevicePinnedBuffer:
        pinned = DevicePinnedBuffer(dtype=DType.float32, shape=(8,), device=gpu)
        pinned.inplace_copy_from(
            Buffer.from_numpy(np.full((8,), value, dtype=np.float32))
        )
        return pinned

    device_srcs = [
        Buffer.from_numpy(np.full((8,), float(i + 1), dtype=np.float32)).to(gpu)
        for i in range(n_dtod)
    ]
    pinned_to_device_src = _fill_pinned(100.0)
    pinned_to_pinned_src = _fill_pinned(200.0)

    # Predicate mismatch that caused the regression: pinned but not is_host.
    assert pinned_to_device_src.pinned and not pinned_to_device_src.is_host
    assert pinned_to_pinned_src.pinned and not pinned_to_pinned_src.is_host

    srcs: list[Buffer] = [
        *device_srcs,
        pinned_to_device_src,
        pinned_to_pinned_src,
    ]
    batch_dsts: list[Buffer] = [
        *[Buffer(DType.float32, (8,), device=gpu) for _ in range(n_dtod)],
        Buffer(DType.float32, (8,), device=gpu),
        DevicePinnedBuffer(dtype=DType.float32, shape=(8,), device=gpu),
    ]
    loop_dsts: list[Buffer] = [
        *[Buffer(DType.float32, (8,), device=gpu) for _ in range(n_dtod)],
        Buffer(DType.float32, (8,), device=gpu),
        DevicePinnedBuffer(dtype=DType.float32, shape=(8,), device=gpu),
    ]
    assert batch_dsts[-1].pinned and loop_dsts[-1].pinned

    batch_inplace_copy(batch_dsts, srcs)
    for dst, src in zip(loop_dsts, srcs, strict=True):
        dst.inplace_copy_from(src)

    gpu.synchronize()
    for i, (batch_dst, loop_dst) in enumerate(
        zip(batch_dsts, loop_dsts, strict=True)
    ):
        np.testing.assert_array_equal(
            batch_dst.to_numpy(),
            loop_dst.to_numpy(),
            err_msg=f"pinned-poison-pill batch vs per-copy mismatch at index {i}",
        )
