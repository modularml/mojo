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
"""Test max.driver Buffers."""

import math
import sys
import tempfile
from collections.abc import Generator
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from max.driver import CPU, Accelerator, Buffer, Usage, accelerator_count
from max.dtype import DType


def test_tensor() -> None:
    # Validate that metadata shows up correctly
    tensor = Buffer(DType.float32, (3, 4, 5))
    assert DType.float32 == tensor.dtype
    assert "DType.float32" == str(tensor.dtype)
    assert (3, 4, 5) == tensor.shape
    assert 3 == tensor.rank

    # Validate that shape can be specified as a list and we copy the dims.
    shape = [2, 3]
    tensor2 = Buffer(DType.float32, shape)
    shape[0] = 1
    assert (2, 3) == tensor2.shape


def test_repr() -> None:
    # repr shows metadata only, not the element values.
    tensor = Buffer(DType.float32, (3, 4))
    text = repr(tensor)
    assert text.startswith("max.driver.Buffer(")
    assert "DType.float32" in text
    assert "(3, 4)" in text


def test_str_scalar() -> None:
    tensor = Buffer.scalar(5, DType.int32)
    text = str(tensor)
    assert text.startswith("Buffer(")
    assert "5" in text
    assert "dtype=DType.int32" in text
    assert "shape=()" in text
    assert "device=" in text


def test_str_shows_data() -> None:
    # str should include the actual data values.
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    tensor = Buffer.from_numpy(arr)
    text = str(tensor)
    assert text.startswith("Buffer(")
    for value in range(1, 7):
        assert str(value) in text
    assert "dtype=DType.int32" in text
    assert "shape=(2, 3)" in text
    # The formatted data should match numpy's own rendering.
    assert np.array2string(arr, prefix="Buffer(") in text


def test_str_1d_float() -> None:
    tensor = Buffer.from_numpy(np.array([1.5, 2.5, 3.5], dtype=np.float32))
    text = str(tensor)
    assert "1.5" in text
    assert "2.5" in text
    assert "3.5" in text
    assert "dtype=DType.float32" in text


def test_str_summarizes_large_buffer() -> None:
    # Large buffers should be summarized with an ellipsis rather than dumping
    # every element.
    tensor = Buffer.from_numpy(np.arange(100_000, dtype=np.int32))
    text = str(tensor)
    assert "..." in text
    assert "shape=(100000,)" in text


def test_str_bfloat16() -> None:
    # bfloat16 is not natively representable in numpy, so values must be read
    # element-wise and shown as decoded numbers (not raw bytes).
    torch_value = torch.tensor([1.0, 2.0, 3.0]).type(torch.bfloat16)
    tensor = Buffer.from_dlpack(torch_value)
    assert tensor.dtype == DType.bfloat16
    text = str(tensor)
    assert "1" in text
    assert "2" in text
    assert "3" in text
    assert "dtype=DType.bfloat16" in text
    assert "shape=(3,)" in text


@pytest.mark.parametrize("dtype", list(DType))
def test_allocate(dtype: DType) -> None:
    Buffer(shape=[2], dtype=dtype, device=CPU())


def test_get_and_set() -> None:
    tensor = Buffer(DType.int32, (3, 4, 5))
    tensor[0, 1, 3] = 68
    # Get should return zero-d tensor
    elt = tensor[0, 1, 3]
    assert 0 == elt.rank
    assert () == elt.shape
    assert DType.int32 == elt.dtype
    assert 68 == elt.item()

    # Setting negative indices
    tensor[-1, -1, -1] = 72
    assert 72 == tensor[2, 3, 4].item()

    # Validate that passing the indices as a sequence object works.
    assert 72 == tensor[(2, 3, 4)].item()

    # Ensure we're not writing to the same memory location with each index
    assert 68 == tensor[0, 1, 3].item()

    # Cannot do out-of-bounds indexing
    with pytest.raises(IndexError):
        tensor[5, 2, 2] = 23

    # Cannot do out-of-bounds with negative indices
    with pytest.raises(IndexError):
        tensor[-4, -3, -3] = 42

    # Indexes need to be equal to the tensor rank
    with pytest.raises(ValueError):
        tensor[2, 2] = 2
    with pytest.raises(ValueError):
        tensor[2, 2, 2, 2] = 2

    # Cannot call item (without arguments) on a non-zero-rank tensor
    with pytest.raises(ValueError):
        tensor.item()


def test_slice() -> None:
    # Tensor slices should have the desired shape and should preserve
    # reference semantics.
    tensor = Buffer(DType.int32, (3, 3, 3))
    subtensor = tensor[:2, :2, :2]
    assert subtensor.shape == (2, 2, 2)
    subtensor[0, 0, 0] = 25
    assert tensor[0, 0, 0].item() == 25

    # We can take arbitrary slices of slices and preserve reference
    # semantics to the original tensor and any derived slices.
    subsubtensor = subtensor[:1, :1, :1]
    assert subsubtensor[0, 0, 0].item() == 25
    subsubtensor[0, 0, 0] = 37
    assert tensor[0, 0, 0].item() == 37
    assert subtensor[0, 0, 0].item() == 37

    # Users should be able to specify step sizes and get tensors of
    # an expected size and beginning at the expected offset.
    strided_subtensor = tensor[1::2, 1::2, 1::2]
    assert strided_subtensor.shape == (1, 1, 1)
    strided_subtensor[0, 0, 0] = 256
    assert tensor[1, 1, 1].item() == 256

    # Invalid slice semantics should throw an exception
    with pytest.raises(ValueError):
        tensor[::0, ::0, ::0]


def test_drop_dimensions() -> None:
    tensor = Buffer(DType.int32, (5, 5, 5))
    # When indexing into a tensor with a mixture of slices and integral
    # indices, the slice should drop any dimensions that correspond to
    # integral indices.
    droptensor = tensor[:, 2, :]
    assert droptensor.rank == 2
    assert droptensor.shape == (5, 5)

    for i in range(4):
        droptensor[i, i] = i
    droptensor[-1, -1] = 4
    for i in range(5):
        assert tensor[i, 2, i].item() == i


def test_negative_step() -> None:
    tensor = Buffer(DType.int32, (3, 3))
    tensor[0, 0] = 1
    tensor[0, 1] = 2
    tensor[0, 2] = 3
    tensor[1, 0] = 4
    tensor[1, 1] = 5
    tensor[1, 2] = 6
    tensor[2, 0] = 7
    tensor[2, 1] = 8
    tensor[2, 2] = 9
    # Tensors should support slices with negative steps.
    revtensor = tensor[::-1, ::-1]
    assert revtensor[0, 0].item() == 9
    assert revtensor[0, 1].item() == 8
    assert revtensor[0, 2].item() == 7
    assert revtensor[1, 0].item() == 6
    assert revtensor[1, 1].item() == 5
    assert revtensor[1, 2].item() == 4
    assert revtensor[2, 0].item() == 3
    assert revtensor[2, 1].item() == 2
    assert revtensor[2, 2].item() == 1


def test_out_of_bounds_slices() -> None:
    tensor = Buffer(DType.int32, (3, 3, 3))

    # Out of bounds indexes are allowed in slices.
    assert tensor[4:, :2, 8:10:-1].shape == (0, 2, 0)

    # Out of bounds indexes are not allowed in integral indexing.
    with pytest.raises(IndexError):
        tensor[4:, :2, 4]


def test_one_dimensional_tensor() -> None:
    tensor = Buffer(DType.int32, (10,))
    for i in range(10):
        tensor[i] = i

    for i in range(i):  # noqa: B020
        assert tensor[i].item() == i


def test_contiguous_tensor() -> None:
    # Initialized tensors should be contiguous, and tensor slices should not be.
    tensor = Buffer(DType.int32, (3, 3))
    assert tensor.is_contiguous
    val = 1
    for x, y in product(range(3), range(3)):
        tensor[x, y] = val
        val += 1

    subtensor = tensor[:2, :2]
    assert not subtensor.is_contiguous

    # There's a special case where reversed slices (which are "technically"
    # contiguous) should not be considered as such.
    assert not tensor[::-1, ::-1].is_contiguous

    subsubtensor = tensor[:2, :2]
    assert subsubtensor.shape == (2, 2)
    cont_tensor = subsubtensor.contiguous()
    assert cont_tensor.shape == (2, 2)
    assert cont_tensor.is_contiguous
    assert cont_tensor[0, 0].item() == 1
    assert cont_tensor[0, 1].item() == 2
    assert cont_tensor[1, 0].item() == 4
    assert cont_tensor[1, 1].item() == 5


def test_modify_contiguous_tensor() -> None:
    # Modifications made to the original tensor should not be reflected
    # on the contiguous copy, and vice-versa.
    tensor = Buffer(DType.int32, (3, 3))
    for x, y in product(range(3), range(3)):
        tensor[x, y] = 1

    cont_tensor = tensor.contiguous()

    cont_tensor[1, 1] = 22
    assert tensor[1, 1].item() == 1

    tensor[2, 2] = 25
    assert cont_tensor[2, 2].item() == 1


def test_contiguous_slice() -> None:
    # A contiguous slice of a tensor should be considered contiguous. An
    # example of this is taking a single row from a 2-d array.
    singlerow = Buffer.from_numpy(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    )[0, :]
    assert singlerow.shape == (3,)
    assert singlerow.is_contiguous

    # This should also work in cases where we take a couple of adjacent rows
    # from a multi-dimensional array.
    multirow = Buffer.from_numpy(np.ones((5, 5), dtype=np.int32))[2:4, :]
    assert multirow.shape == (2, 5)
    assert multirow.is_contiguous

    # We also need this work in cases where we're just taking subarrays of 1-d
    # arrays.
    subarray = Buffer.from_numpy(np.ones((10,), dtype=np.int32))[2:5]
    assert subarray.shape == (3,)
    assert subarray.is_contiguous


def test_from_numpy() -> None:
    # A user should be able to create a tensor from a numpy array.
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    tensor = Buffer.from_numpy(arr)
    assert tensor.shape == (2, 3)
    assert tensor.dtype == DType.int32
    assert tensor[0, 0].item() == 1
    assert tensor[0, 1].item() == 2
    assert tensor[0, 2].item() == 3
    assert tensor[1, 0].item() == 4
    assert tensor[1, 1].item() == 5
    assert tensor[1, 2].item() == 6


def test_from_numpy_scalar() -> None:
    # Also test that scalar numpy arrays remain scalar.
    arr = np.array(1.0, dtype=np.float32)
    tensor = Buffer.from_numpy(arr)

    assert tensor.dtype == DType.float32
    assert tensor.shape == arr.shape


def test_is_host() -> None:
    # CPU tensors should be marked as being on-host.
    assert Buffer(DType.int32, (1, 1), device=CPU()).is_host


def test_host_to_self() -> None:
    cpu = CPU()
    t = Buffer(DType.int32, (1, 1), device=cpu)
    t2 = t.to(cpu)
    assert t2 is t
    t3s = t.to([cpu])
    assert len(t3s) == 1
    assert t3s[0] is t


def test_host_host_copy() -> None:
    # We should be able to freely copy tensors between host and host.
    host_tensor = Buffer.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    cpu = CPU()
    tensor = host_tensor.copy(cpu)
    cpu.synchronize()

    assert tensor.shape == host_tensor.shape
    assert tensor.dtype == host_tensor.dtype
    host_tensor[0] = 0  # Ensure we got a deep copy.
    assert tensor[0].item() == 1
    assert tensor[1].item() == 2
    assert tensor[2].item() == 3

    tensor2 = host_tensor.copy(cpu)
    cpu.synchronize()
    assert tensor2[0].item() == 0
    assert tensor2[1].item() == 2
    assert tensor2[2].item() == 3


def test_pinning() -> None:
    # A host device can't page-lock, so staging there is a plain allocation.
    assert not Buffer(DType.int32, (1, 1), device=CPU()).pinned

    staging_on_host = Buffer(
        DType.int32, (1, 1), device=CPU(), usage=Usage.STAGING
    )
    assert not staging_on_host.pinned
    assert staging_on_host.usage == Usage.STAGING

    if accelerator_count():
        tensor = Buffer(DType.int32, (1, 1), device=CPU())
        gpu_tensor = tensor.to(Accelerator())
        assert not gpu_tensor.pinned
        # I don't actually know what the behavior _should_ be here, but
        # testing what we actually do for clarity.
        assert not gpu_tensor.to(CPU()).pinned


DLPACK_DTYPES = [
    DType.bool,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float16,
    DType.float32,
    DType.float64,
]


def test_from_dlpack() -> None:
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for dtype in DLPACK_DTYPES:
        np_dtype = dtype.to_numpy()
        array = np.array([0, 1, 2, 3], np_dtype)
        tensor = Buffer.from_dlpack(array)
        assert tensor.dtype == dtype
        assert tensor.shape == array.shape

        if dtype is dtype.bool:
            assert not array[0]
            tensor[0] = True
            assert array[0]
        else:
            tensor[0] = np_dtype.type(7)
            assert array[0] == np_dtype.type(7)


def test_from_dlpack_short_circuit() -> None:
    tensor = Buffer(DType.int8, (4,))
    for i in range(4):
        tensor[i] = i

    # Test short circuiting.
    same_tensor = Buffer.from_dlpack(tensor)
    assert tensor is same_tensor
    copy_tensor = Buffer.from_dlpack(tensor, copy=True)
    assert tensor is not copy_tensor
    assert tensor.dtype == copy_tensor.dtype
    assert tensor.shape == copy_tensor.shape


def test_from_dlpack_double_transfer() -> None:
    tensor1 = Buffer(DType.int8, (4,))
    for i in range(4):
        tensor1[i] = i

    arr = np.from_dlpack(tensor1)  # Transfer ownership
    tensor2 = Buffer.from_dlpack(arr)  # Transfer ownership back
    tensor2[0] = np.int8(7)
    assert arr[0] == np.int8(7)

    tensor3 = Buffer.from_dlpack(arr)  # No transfer, but still able to view
    # TODO: Why does this not work if we omit .item() ?
    assert tensor3[0].item() == np.int8(7)


def test_to_numpy_writable() -> None:
    # Ensure that to_numpy returns a writable array.
    # This is similar to test_from_dlpack_double_transfer above
    # but explicitly tests that the numpy array is writable.
    tensor = Buffer(shape=(20, 10), dtype=DType.int32)
    np_array = tensor.to_numpy()
    np_array[0, 0] = 100


def test_dlpack_device() -> None:
    tensor = Buffer(DType.int32, (3, 3))
    device_tuple = tensor.__dlpack_device__()
    assert len(device_tuple) == 2
    assert isinstance(device_tuple[0], int)
    assert device_tuple[0] == 1  # 1 is the value of DLDeviceType::kDLCPU
    assert isinstance(device_tuple[1], int)
    assert device_tuple[1] == 0  # should be the default device


def test_dlpack() -> None:
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for dtype in DLPACK_DTYPES:
        # Numpy's dlpack implementation cannot handle its own bool types.
        # Also don't try to put ints into a boolean vector.
        if dtype is dtype.bool:
            continue

        tensor = Buffer(dtype, (1, 4))
        for j in range(4):
            tensor[0, j] = j

        np_dtype = dtype.to_numpy()
        array = np.from_dlpack(tensor)
        assert array.dtype == np_dtype
        assert tensor.shape == array.shape

        # Numpy creates a read-only array, so we modify ours.
        tensor[0, 0] = np_dtype.type(7)
        assert array[0, 0] == np_dtype.type(7)


def test_torch_tensor_conversion() -> None:
    # Our tensors should be convertible to and from Torch tensors.
    torch_tensor = torch.reshape(torch.arange(1, 11, dtype=torch.int32), (2, 5))
    driver_tensor = Buffer.from_dlpack(torch_tensor)
    assert driver_tensor.shape == (2, 5)
    assert driver_tensor.dtype == DType.int32
    for x, y in product(range(2), range(5)):
        assert torch_tensor[x, y].item() == driver_tensor[x, y].item()

    converted_tensor = torch.from_dlpack(driver_tensor)
    assert torch.all(torch.eq(torch_tensor, converted_tensor))

    # We should also be able to get this running for boolean tensors.
    bool_tensor = torch.tensor([False, True, False, True])
    converted_bool = Buffer.from_dlpack(bool_tensor)
    assert converted_bool.shape == (4,)
    assert converted_bool.dtype == DType.bool
    for x in range(4):
        assert bool_tensor[x].item() == converted_bool[x].item()

    reconverted_bool = torch.from_dlpack(converted_bool)
    assert torch.all(torch.eq(bool_tensor, reconverted_bool))


# Whichever of these runs first pays torch's one-time lazy init inside its
# first example -- ~1.5s against hypothesis' 200ms per-example deadline on a
# contended CI worker. Neither asserts anything about speed.
@settings(deadline=None)
@given(st.floats())
def test_setitem_bfloat16(value: float) -> None:
    tensor = Buffer(DType.bfloat16, (1,))
    tensor[0] = value
    expected = torch.tensor([value]).type(torch.bfloat16)
    # Torch rounds values up, whereas we currently truncate.
    # In particular this is an issue near infinity, as there's certain values
    # that torch will represent as inf, while we will instead represent them
    # as bfloat16_max.
    result = torch.from_dlpack(tensor)
    bf16info = torch.finfo(torch.bfloat16)
    if value > bf16info.max and math.isfinite(result.item()):
        assert result.item() == bf16info.max
    elif value < bf16info.min and math.isfinite(result.item()):
        assert result.item() == bf16info.min
    else:
        torch.testing.assert_close(expected, result, equal_nan=True)


@settings(deadline=None)
@given(st.floats())
def test_getitem_bfloat16(value: float) -> None:
    torch_value = torch.tensor([value]).type(torch.bfloat16)
    tensor = Buffer.from_dlpack(torch_value)
    assert tensor.dtype == DType.bfloat16
    result = tensor[0].item()
    torch.testing.assert_close(torch_value.item(), result, equal_nan=True)


def test_device() -> None:
    # We should be able to set and query the device that a tensor is resident on.
    cpu = CPU()
    tensor = Buffer(dtype=DType.int32, shape=(3, 3), device=cpu)
    assert cpu == tensor.device


def test_to_numpy() -> None:
    # We should be able to convert a tensor to a numpy array.
    base_arr = np.arange(1, 6, dtype=np.int32)
    tensor = Buffer.from_numpy(base_arr)
    new_arr = tensor.to_numpy()
    assert np.array_equal(base_arr, new_arr)


def test_zeros() -> None:
    # We should be able to initialize an all-zero tensor.
    tensor = Buffer.zeros((3, 3), DType.int32)
    assert np.array_equal(tensor.to_numpy(), np.zeros((3, 3), dtype=np.int32))


def test_scalar() -> None:
    # We should be able to create scalar values.
    scalar = Buffer.scalar(5, DType.int32)
    assert scalar.item() == 5

    # We allow some ability to mutate scalars.
    scalar[0] = 8
    assert scalar.item() == 8


# NOTE: This is kept at function scope intentionally to avoid issues if tests
# mutate the stored data.
@pytest.fixture(scope="function")
def memmap_example_file() -> Generator[Path, None, None]:
    with tempfile.NamedTemporaryFile(mode="w+b") as f:
        f.write(b"\x00\x01\x02\x03\x04\x05\x06\x07")
        f.flush()
        yield Path(f.name)


@pytest.mark.xfail(sys.platform == "darwin", reason="GEX-2968")
def test_memmap(memmap_example_file: Path) -> None:
    tensor = Buffer.mmap(memmap_example_file, dtype=DType.int8, shape=(2, 4))
    assert tensor.shape == (2, 4)
    assert tensor.dtype == DType.int8
    for i, j in product(range(2), range(4)):
        assert tensor[i, j].item() == i * 4 + j

    # Test that offsets work.
    offset_tensor = Buffer.mmap(
        memmap_example_file, dtype=DType.int8, shape=(2, 3), offset=2, mode="r"
    )
    assert offset_tensor.shape == (2, 3)
    assert offset_tensor.dtype == DType.int8
    for i, j in product(range(2), range(3)):
        assert offset_tensor[i, j].item() == i * 3 + j + 2

    # Test that read-only arrays don't modify underlying data
    offset_tensor[0, 0] = 0
    assert offset_tensor[0, 0].item() == 0
    assert tensor[0, 1].item() == 1

    # Test that a different type works and we can modify the array.
    tensor_16 = Buffer.mmap(
        memmap_example_file, dtype=DType.int16, shape=(2,), offset=2, mode="r+"
    )
    tensor_16[0] = 0  # Intentional to avoid endianness issues.

    assert tensor[0, 1].item() == 1
    assert tensor[0, 2].item() == 0
    assert tensor[0, 3].item() == 0
    assert tensor[1, 0].item() == 4

    # offset_tensor is a copy because it's not writeable, so check that changes aren't reflected
    assert offset_tensor[0, 1].item() == 3
    assert offset_tensor[0, 2].item() == 4


def test_dlpack_memmap(memmap_example_file: Path) -> None:
    tensor = Buffer.mmap(memmap_example_file, dtype=DType.int8, shape=(2, 4))
    array = np.from_dlpack(tensor)
    assert array.dtype == np.int8
    assert tensor.shape == array.shape

    # Numpy creates a read-only array, so we modify ours.
    tensor[0, 0] = np.int8(8)
    assert array[0, 0] == np.int8(8)


def test_dlpack_memmap_view(memmap_example_file: Path) -> None:
    tensor = Buffer.mmap(memmap_example_file, dtype=DType.int8, shape=(2, 4))
    tensor_view = tensor.view(DType.uint8)
    assert isinstance(tensor_view, Buffer)

    array = np.from_dlpack(tensor_view)
    assert array.dtype == np.uint8
    assert tensor.shape == array.shape

    # Numpy creates a read-only array, so we modify ours.
    tensor[0, 0] = np.uint8(8)
    assert array[0, 0] == np.uint8(8)


def test_from_dlpack_memmap(memmap_example_file: Path) -> None:
    # We test that we can call from_dlpack on a read-only numpy memmap array.
    # TODO(MSDK-976): remove this test when we upgraded numpy to 2.1.
    array = np.memmap(memmap_example_file, dtype=np.int8, mode="r")
    assert not array.flags.writeable

    tensor = Buffer.from_dlpack(array)
    assert isinstance(tensor, Buffer)
    assert array.dtype == np.int8
    assert tensor.shape == array.shape

    # Test that we don't overwrite the read-only memmap data
    tensor[0] = 0
    assert array[1].item() != 0


def test_num_elements() -> None:
    tensor1 = Buffer(DType.int8, (2, 4, 3))
    assert tensor1.num_elements == 24

    tensor2 = Buffer(DType.int8, (1, 4))
    assert tensor2.num_elements == 4

    tensor3 = Buffer(DType.int8, ())
    assert tensor3.num_elements == 1

    tensor4 = Buffer(DType.int8, (1, 1, 1, 1, 1))
    assert tensor4.num_elements == 1


def test_element_size() -> None:
    for dtype in DLPACK_DTYPES:
        tensor = Buffer(dtype, ())
        assert tensor.element_size == np.dtype(dtype.to_numpy()).itemsize


def test_view() -> None:
    tensor = Buffer(DType.int8, (2, 6))
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            tensor[i, j] = i * 10 + j
    assert tensor[0, 0].item() == 0
    assert tensor[1, 0].item() == 10
    assert tensor[1, 3].item() == 13
    assert tensor[1, 5].item() == 15
    # Check that the new shape is properly backed by the original
    tensor_view = tensor.view(DType.int8, (6, 2))
    assert tensor_view[0, 0].item() == 0
    assert tensor_view[1, 0].item() == 2
    assert tensor_view[3, 1].item() == 11
    assert tensor_view[5, 1].item() == 15

    tensor8 = Buffer(DType.int8, (2, 4))
    for i, j in product(range(2), range(4)):
        tensor8[i, j] = 1

    # Check that we correctly deduce the shape if not given
    tensor16 = tensor8.view(DType.int16)
    assert tensor16.dtype is DType.int16
    assert tensor16.shape == (2, 2)
    assert tensor16[0, 0].item() == 2**8 + 1
    assert tensor16[0, 1].item() == 2**8 + 1

    # Check that it works with explicit shape.
    tensor32 = tensor8.view(DType.int32, (2,))
    assert tensor32.dtype is DType.int32
    assert tensor32.shape == (2,)
    assert tensor32[0].item() == 2**24 + 2**16 + 2**8 + 1

    # Check that this is not a copy.
    tensor16[0, 0] = 0
    assert tensor8[0, 0].item() == 0
    assert tensor8[0, 1].item() == 0

    # Check that shape deduction fails if the last axis is the wrong size.
    with pytest.raises(ValueError):
        _ = tensor8.view(DType.int64)


def test_from_dlpack_noncontiguous() -> None:
    array = np.arange(4).reshape(2, 2).transpose(1, 0)
    assert not array.flags.c_contiguous

    with pytest.raises(
        ValueError,
        match=r"from_dlpack only accepts contiguous arrays. First call np.ascontiguousarray",
    ):
        Buffer.from_dlpack(array)


def test_from_dlpack_torch_noncontiguous() -> None:
    """Test that Buffer.from_dlpack correctly handles non-contiguous torch tensors."""
    # Create a non-contiguous torch tensor via transpose.
    torch_tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    assert not torch_tensor.is_contiguous()

    with pytest.raises(
        ValueError, match="from_dlpack only accepts contiguous tensors"
    ):
        Buffer.from_dlpack(torch_tensor)


def test_item_success() -> None:
    """Test successful item() calls for valid single-element tensors."""
    # Zero-rank case
    scalar = Buffer.scalar(8, DType.int32)
    assert scalar.item() == 8

    # Single-element tensors of various ranks
    for shape in [(), (1,), (1, 1), (1, 1, 1)]:
        tensor = Buffer(DType.float32, shape)
        tensor[tuple(0 for _ in shape)] = 3.14
        assert math.isclose(tensor.item(), 3.14, rel_tol=1e-6)


def test_item_multiple_elements() -> None:
    """Test item() fails when tensor contains multiple elements"""
    tensor = Buffer(DType.int32, (2,))
    with pytest.raises(
        ValueError,
        match="calling `item` on a tensor with 2 items but expected only 1",
    ):
        tensor.item()


def test_aligned() -> None:
    tensor = Buffer(DType.int32, (5,))
    assert tensor._aligned()
    assert tensor._aligned(DType.int32.align)

    tensor_uint8 = tensor.view(DType.uint8)
    assert tensor_uint8[1]._aligned()
    assert not tensor_uint8[1]._aligned(DType.int32.align)


def test_unaligned_tensor_copy() -> None:
    """Tests tensor copying and viewing with unaligned memory."""
    cpu = CPU()
    expected = np.array([1005, 2510, 1325], np.int32)

    # Construct a uint8 tensor so that when converted to int32, it becomes the
    # expected tensor - with a twist that there's an extra byte to force an
    # unaligned view.
    unaligned_array = np.array(
        [15] + expected.view(np.uint8).tolist(),
        np.uint8,
    )
    tensor8 = Buffer.from_numpy(unaligned_array)

    # Copy operation now works correctly (fixed by GEX-2576).
    tensor8_copy = tensor8[1:].copy()
    cpu.synchronize()
    # Should correctly preserve element values after copy
    assert tensor8_copy[0].item() == tensor8[1].item()

    tensor32 = tensor8[1:].view(DType.int32)
    assert not tensor8[1:]._aligned(DType.int32.align)

    # The int32 view of unaligned data correctly reports as unaligned
    assert not tensor32._aligned()

    assert tensor32[0].item() == 1005

    tensor32_copy = tensor32.copy()
    cpu.synchronize()
    assert tensor32_copy._aligned()
    # This now passes because the source view has correct data
    np.testing.assert_array_equal(tensor32_copy.to_numpy(), expected)


def test_inplace_copy_from_raises() -> None:
    tensor_3_3 = Buffer(DType.int32, (3, 3))

    tensor_3_10_3 = Buffer(DType.int32, (3, 10, 3))
    tensor_3_3_noncontig = tensor_3_10_3[:, 0, :]
    assert tensor_3_3_noncontig.shape == (3, 3)
    assert not tensor_3_3_noncontig.is_contiguous

    with pytest.raises(ValueError) as e:
        tensor_3_3.inplace_copy_from(tensor_3_3_noncontig)
        assert "Cannot copy from non-contiguous tensor" in str(e.value)

    tensor_3_2 = Buffer(DType.int32, (3, 2))
    with pytest.raises(ValueError) as e:
        tensor_3_3.inplace_copy_from(tensor_3_2)
        assert "Cannot copy tensors of different sizes" in str(e.value)

    tensor_i32 = Buffer(DType.int32, (2, 3))
    tensor_i16 = Buffer(DType.int16, (2, 3))
    with pytest.raises(ValueError) as e:
        tensor_i32.inplace_copy_from(tensor_i16)
        assert "Cannot copy tensors of different dtypes" in str(e.value)


def test_inplace_copy_from_self_copy_noncontiguous_noop() -> None:
    tensor_3_10_3 = Buffer(DType.int32, (3, 10, 3))
    tensor_3_3_noncontig = tensor_3_10_3[:, 0, :]
    assert not tensor_3_3_noncontig.is_contiguous

    tensor_3_3_noncontig.inplace_copy_from(tensor_3_3_noncontig)


def test_inplace_copy_from_tensor_view() -> None:
    enumerated = np.zeros((5, 2, 3), dtype=np.int32)
    for i, j, k in np.ndindex(enumerated.shape):
        enumerated[i, j, k] = 100 * i + 10 * j + k

    all_nines = np.full((5, 2, 3), 999, dtype=np.int32)

    src = Buffer.from_numpy(enumerated)
    dst = Buffer.from_numpy(all_nines)

    # Copy 3rd row of dst tensor into 1st row of src tensor.
    dst[1, :, :].inplace_copy_from(src[3, :, :])

    expected = np.array(
        [
            [[999, 999, 999], [999, 999, 999]],
            [[300, 301, 302], [310, 311, 312]],
            [[999, 999, 999], [999, 999, 999]],
            [[999, 999, 999], [999, 999, 999]],
            [[999, 999, 999], [999, 999, 999]],
        ]
    )
    assert np.array_equal(dst.to_numpy(), expected)


def test_inplace_copy_from_cross_dtype_view() -> None:
    """Regression test: inplace_copy_from on slices of a view'd buffer.

    When a buffer is view'd from one dtype to another (e.g. int16 -> uint8),
    the underlying storage retains the original dtype. Slicing the view'd
    buffer and calling inplace_copy_from must correctly compute byte offsets
    using the view's dtype, not the storage's dtype.
    """
    num_rows = 4
    cols_i16 = 3
    cols_u8 = cols_i16 * 2

    # Create source data as int16, fill with a known pattern
    src_data = np.zeros((num_rows, cols_i16), dtype=np.int16)
    for i in range(num_rows):
        for j in range(cols_i16):
            src_data[i, j] = i * 10 + j
    src_buf = Buffer.from_numpy(src_data)

    # View as uint8 (same total bytes, different element count)
    src_u8 = src_buf.view(DType.uint8, [num_rows, cols_u8])

    # Create a destination buffer of the same uint8 shape, filled with 0xFF
    dst_data = np.full((num_rows, cols_u8), 0xFF, dtype=np.uint8)
    dst_buf = Buffer.from_numpy(dst_data)

    # Copy slices from the LAST row of source to the FIRST row of destination.
    # This exercises a large startOffset that would fail if byte offset
    # computation used the wrong dtype.
    dst_buf[0, :].inplace_copy_from(src_u8[num_rows - 1, :])

    result = dst_buf.to_numpy()
    expected_row = src_u8.to_numpy()[num_rows - 1]
    np.testing.assert_array_equal(result[0], expected_row)

    # Other rows should be unchanged
    for i in range(1, num_rows):
        np.testing.assert_array_equal(
            result[i], np.full(cols_u8, 0xFF, dtype=np.uint8)
        )


def test_GEX_2088() -> None:
    t = Buffer.zeros([2], DType.uint32)
    assert t[1].to_numpy().item() == 0


@pytest.mark.parametrize(
    "device_factory",
    [
        CPU,
        pytest.param(
            Accelerator,
            marks=pytest.mark.skipif(
                not accelerator_count(), reason="GPU not available"
            ),
        ),
    ],
)
def test_tensor_slicing_to_numpy_basic(device_factory: type) -> None:
    """Test basic tensor slicing with .to_numpy() on CPU and GPU - reproduces GEX-2576."""
    # Create original test case from reproduction
    tensor = Buffer.from_numpy(np.arange(10).reshape(2, 5)).to(device_factory())
    original_data = tensor.to_numpy()

    # Expected: [[0 1 2 3 4], [5 6 7 8 9]]
    expected_original = np.array(
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32
    )
    np.testing.assert_array_equal(original_data, expected_original)

    # Test first row slice - should return [[0 1 2 3 4]]
    first_row_slice = tensor[0:1, :]
    first_row_result = first_row_slice.to_numpy()
    expected_first_row = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)
    np.testing.assert_array_equal(
        first_row_result,
        expected_first_row,
        f"First row slice failed on {device_factory.__name__}",
    )

    # Test second row slice - should return [[5 6 7 8 9]]
    # This is the failing case from the bug report (used to fail on GPU, worked on CPU)
    second_row_slice = tensor[1:2, :]
    second_row_result = second_row_slice.to_numpy()
    expected_second_row = np.array([[5, 6, 7, 8, 9]], dtype=np.int32)
    np.testing.assert_array_equal(
        second_row_result,
        expected_second_row,
        f"Second row slice failed on {device_factory.__name__}",
    )


@pytest.mark.parametrize(
    "device_factory",
    [
        CPU,
        pytest.param(
            Accelerator,
            marks=pytest.mark.skipif(
                not accelerator_count(), reason="GPU not available"
            ),
        ),
    ],
)
def test_tensor_slicing_to_numpy_comprehensive(device_factory: type) -> None:
    """Comprehensive tensor slicing tests for various slice patterns on CPU and GPU."""
    # Create a 3x4 tensor for more comprehensive testing
    data = np.arange(12).reshape(3, 4).astype(np.int32)
    tensor = Buffer.from_numpy(data).to(device_factory())

    # Test various single row slices
    for i in range(3):
        row_slice = tensor[i : i + 1, :]
        result = row_slice.to_numpy()
        expected = data[i : i + 1, :]
        np.testing.assert_array_equal(
            result,
            expected,
            f"Row slice [%d:%d, :] failed on {device_factory.__name__}"
            % (i, i + 1),
        )

    # Test multi-row slices
    for start in range(2):
        for end in range(start + 1, 3):
            row_slice = tensor[start:end, :]
            result = row_slice.to_numpy()
            expected = data[start:end, :]
            np.testing.assert_array_equal(
                result,
                expected,
                f"Row slice [%d:%d, :] failed on {device_factory.__name__}"
                % (start, end),
            )


@pytest.mark.parametrize(
    "device_factory",
    [
        CPU,
        pytest.param(
            Accelerator,
            marks=pytest.mark.skipif(
                not accelerator_count(), reason="GPU not available"
            ),
        ),
    ],
)
def test_tensor_slicing_to_numpy_edge_cases(device_factory: type) -> None:
    """Test edge cases for tensor slicing on CPU and GPU."""
    data = np.arange(20).reshape(4, 5).astype(np.int32)
    tensor = Buffer.from_numpy(data).to(device_factory())

    # Test single element slices
    for i in range(4):
        for j in range(5):
            element_slice = tensor[i : i + 1, j : j + 1]
            result = element_slice.to_numpy()
            expected = data[i : i + 1, j : j + 1]
            np.testing.assert_array_equal(
                result,
                expected,
                f"Single element slice [%d:%d, %d:%d] failed on {device_factory.__name__}"
                % (i, i + 1, j, j + 1),
            )

    # Test last row and column
    last_row = tensor[-1:, :]
    result = last_row.to_numpy()
    expected = data[-1:, :]
    np.testing.assert_array_equal(
        result, expected, f"Last row slice failed on {device_factory.__name__}"
    )

    # Test negative indexing
    second_last_row = tensor[-2:-1, :]
    result = second_last_row.to_numpy()
    expected = data[-2:-1, :]
    np.testing.assert_array_equal(
        result,
        expected,
        f"Second last row slice failed on {device_factory.__name__}",
    )


@pytest.mark.parametrize(
    "device_factory",
    [
        CPU,
        pytest.param(
            Accelerator,
            marks=pytest.mark.skipif(
                not accelerator_count(), reason="GPU not available"
            ),
        ),
    ],
)
def test_tensor_slicing_to_numpy_different_dtypes(device_factory: type) -> None:
    """Test tensor slicing with different data types on CPU and GPU."""
    shapes_and_dtypes = [
        ((3, 4), DType.float32),
        ((2, 6), DType.float64),
        ((4, 3), DType.int64),
        ((3, 3), DType.int16),
    ]

    for shape, dtype in shapes_and_dtypes:
        np_dtype = dtype.to_numpy()
        data = np.arange(shape[0] * shape[1]).reshape(shape).astype(np_dtype)
        tensor = Buffer.from_numpy(data).to(device_factory())

        # Test middle row slice
        mid_row = shape[0] // 2
        row_slice = tensor[mid_row : mid_row + 1, :]
        result = row_slice.to_numpy()
        expected = data[mid_row : mid_row + 1, :]
        np.testing.assert_array_equal(
            result,
            expected,
            f"Row slice failed for dtype {dtype} and shape {shape} on {device_factory.__name__}",
        )


@pytest.mark.parametrize(
    "device_factory",
    [
        CPU,
        pytest.param(
            Accelerator,
            marks=pytest.mark.skipif(
                not accelerator_count(), reason="GPU not available"
            ),
        ),
    ],
)
def test_tensor_slicing_to_numpy_3d(device_factory: type) -> None:
    """Test tensor slicing with 3D tensors on CPU and GPU."""
    data = np.arange(24).reshape(2, 3, 4).astype(np.int32)
    tensor = Buffer.from_numpy(data).to(device_factory())

    # Test slicing along first dimension
    first_3d_slice = tensor[0:1, :, :]
    result = first_3d_slice.to_numpy()
    expected = data[0:1, :, :]
    np.testing.assert_array_equal(
        result,
        expected,
        f"3D first dimension slice failed on {device_factory.__name__}",
    )

    second_3d_slice = tensor[1:2, :, :]
    result = second_3d_slice.to_numpy()
    expected = data[1:2, :, :]
    np.testing.assert_array_equal(
        result,
        expected,
        f"3D second dimension slice failed on {device_factory.__name__}",
    )


# The cases below replace the C++ unit tests over `DeviceMemory::createView`
# (CreateViewTest.cpp) that went away with the GenericML Driver layer. That
# logic survives as `Python::createStorageView`.
#
# Only the unaligned view below actually enters `createStorageView`: `view()`
# takes an early return that just adjusts `startOffset` whenever the byte
# offset divides evenly into the target dtype, and element reads compute their
# own offset in `Buffer::data()`. The rest cover the slice arithmetic around
# it, which nothing else pins this directly.


def test_view_byte_offset_basic() -> None:
    # A slice starting partway into a buffer must read from the corresponding
    # byte offset, not from the base pointer.
    tensor = Buffer(DType.float32, (16,))
    for i in range(16):
        tensor[i] = np.float32(i)

    tail = tensor[8:]
    assert tail.shape == (8,)
    for i in range(8):
        assert tail[i].item() == float(i + 8)


def test_view_byte_offset_cross_dtype() -> None:
    # The core bug scenario: storage allocated as one dtype, viewed as another,
    # then offset. The byte offset must come from the view's dtype rather than
    # the storage's, or the reinterpreted slice reads the wrong bytes.
    tensor = Buffer(DType.bfloat16, (8,))  # 8 * 2 == 16 bytes
    as_bytes = tensor.view(DType.uint8, (16,))
    for i in range(16):
        as_bytes[i] = np.uint8(i)

    tail = as_bytes[12:]
    assert tail.shape == (4,)
    for i in range(4):
        assert tail[i].item() == i + 12


def test_view_zero_offset_shares_storage() -> None:
    # A view with no offset and an equivalent spec shares the original buffer
    # rather than creating a sub-buffer, so writes are visible both ways.
    tensor = Buffer(DType.float32, (4,))
    for i in range(4):
        tensor[i] = np.float32(i)

    same = tensor.view(DType.float32, (4,))
    same[0] = np.float32(99)
    assert tensor[0].item() == 99.0

    # Reinterpreting the whole buffer at a wider dtype also aliases it.
    as_int = tensor.view(DType.int32, (4,))
    assert as_int.shape == (4,)


def test_view_out_of_bounds_rejected() -> None:
    # A view wider than its backing storage must be refused rather than
    # silently reading past the end.
    tensor = Buffer(DType.float32, (4,))  # 16 bytes
    with pytest.raises((ValueError, RuntimeError)):
        tensor.view(DType.float32, (8,))  # 32 bytes


def test_unaligned_cross_dtype_view_offsets_storage() -> None:
    # The core bug scenario from CreateViewTest, and the one path that reaches
    # `Python::createStorageView`: a slice whose byte offset is not a multiple
    # of the target dtype's size cannot be folded into a startOffset, so the
    # view has to materialize a sub-buffer starting at that byte offset. If the
    # offset were dropped, these reads would come from the base of the buffer.
    tensor = Buffer(DType.uint8, (16,))
    for i in range(16):
        tensor[i] = np.uint8(i)

    window = tensor[3:7]  # 4 bytes, byte offset 3
    assert window.shape == (4,)

    as_i16 = window.view(DType.int16, (2,))  # 4 bytes; 3 % 2 != 0 -> unaligned
    assert as_i16.shape == (2,)

    # Little-endian: bytes (3, 4) -> 0x0403, bytes (5, 6) -> 0x0605.
    assert as_i16[0].item() == 0x0403
    assert as_i16[1].item() == 0x0605
