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
"""Tests `max.driver.Buffer` basic behaviors."""

import asyncio

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.tensor import (
    Tensor,
    TensorType,
    _default_device,
    _default_dtype,
    default_device,
    default_dtype,
    defaults_like,
    driver_tensor_type,
)
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    ops,
)
from max.graph import (
    TensorType as GraphTensorType,
)


def test_tensor_basic() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    b = a + 1
    assert isinstance(b, Tensor)
    assert b.type == a.type == expected_type
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_tensor_basic_lazy() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    with F.lazy():
        b = a + 1
    assert isinstance(b, Tensor)
    assert not b.real
    assert b.type == a.type == expected_type

    asyncio.run(b.realize)
    assert b.real
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_tensor_with_intermediate() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    b = a + a + 1
    assert isinstance(b, Tensor)
    assert b.type == a.type == expected_type
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_tensor_with_intermediate_lazy() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    expected_type = TensorType(
        DType.float32, [5, 5], DeviceRef.from_device(DEVICE)
    )
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.type == expected_type
    assert a.driver_tensor.to(CPU())[0, 0].item() == 0.0
    with F.lazy():
        b = a + a + 1
    assert isinstance(b, Tensor)
    assert not b.real
    assert b.type == a.type == expected_type

    asyncio.run(b.realize)
    assert b.real
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_compilation_failure() -> None:
    a_data = Buffer.zeros([5, 5], DType.float8_e4m3fn, CPU())
    b_data = Buffer.zeros([5, 5], DType.float32, CPU())
    a = Tensor(storage=a_data)

    with F.lazy():
        # Adding fp8 on cpu is unsupported
        fails_compilation = a + 1

    with pytest.raises(Exception):
        asyncio.run(fails_compilation.realize)

    # Test that new tensor ops can still execute
    b = Tensor(storage=b_data)
    c = b + 1
    assert c.real


def test_tensor_dlpack() -> None:
    data = Buffer.zeros([5, 5], DType.float32, CPU())
    t = Tensor(storage=data)
    assert t.type == driver_tensor_type(data)
    assert t.real
    npt = np.from_dlpack(t)
    assert npt.dtype == t.dtype.to_numpy()
    assert list(npt.shape) == t.shape


def test_tensor_lazy_dlpack() -> None:
    with F.lazy():
        t = random.normal([5, 5], dtype=DType.float32, device=CPU())
    npt = np.from_dlpack(t)
    assert npt.dtype == t.dtype.to_numpy()
    assert list(npt.shape) == t.shape


def test_tensor_from_dlpack() -> None:
    npt = np.random.normal([5, 5])
    t = Tensor.from_dlpack(npt)
    assert t.real
    assert npt.dtype == t.dtype.to_numpy()
    assert list(npt.shape) == t.shape


def test_tensor_constructor_preserves_dlpack_dtype() -> None:
    """Tensor(array) inherits the array's dtype without silent casting."""
    arr_f32 = np.ones([3, 4], dtype=np.float32)
    t_f32 = Tensor(arr_f32, device=CPU())
    assert t_f32.dtype == DType.float32
    assert t_f32.real

    arr_i16 = np.array([1, 2, 3], dtype=np.int16)
    t_i16 = Tensor(arr_i16, device=CPU())
    assert t_i16.dtype == DType.int16
    assert list(t_i16.shape) == [3]

    arr_f64 = np.zeros([2, 2], dtype=np.float64)
    t_f64 = Tensor(arr_f64, device=CPU())
    assert t_f64.dtype == DType.float64


def test_tensor_constructor_dlpack_conflicting_dtype_raises() -> None:
    """Tensor(array, dtype=...) raises when dtype conflicts with the array's dtype."""
    arr = np.ones([4], dtype=np.float32)
    with pytest.raises(ValueError, match="DType must match"):
        Tensor(arr, dtype=DType.float64, device=CPU())


def test_functional_in_graph() -> None:
    with Graph("test_functional") as graph:
        graph.output(F.constant(1, dtype=DType.float32, device=DeviceRef.CPU()))


def test_tensor_default_dtype() -> None:
    t = Tensor(1, device=CPU())
    assert t.dtype == _default_dtype(CPU())
    assert t.device == CPU()

    assert DType.float64 != _default_dtype(CPU())
    with default_dtype(DType.float64):
        t = Tensor(1, device=CPU())
    assert t.dtype == DType.float64


def test_tensor_default_device() -> None:
    t = Tensor(1)
    assert t.device == _default_device()
    assert t.dtype == _default_dtype(_default_device())


def test_defaults_like() -> None:
    t = Tensor(1, dtype=DType.float64)
    with defaults_like(t):
        t2 = Tensor(1)
        assert t.type == t2.type
    with defaults_like(t):
        t3 = Tensor(1)
        assert t.type == t3.type


@pytest.mark.skipif(
    not accelerator_count(), reason="requires at least 2 devices"
)
def test_tensor_default_device_context() -> None:
    assert _default_device() != CPU()
    with default_device(CPU()):
        t = Tensor(1)

    assert t.device == CPU()
    assert t.dtype == _default_dtype(CPU())


def test_tensor_constant_deprecated() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = Tensor.constant(1, device=CPU())
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "Tensor.constant" in str(w[0].message)
    assert t.dtype == _default_dtype(CPU())
    assert t.device == CPU()


def test_realized_tensor_as_buffer() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    b = Tensor.ones_like(a)
    F.buffer_store(a, b)
    assert a.real


def test_realized_tensor_as_buffer_lazy() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.real
    with F.lazy():
        b = Tensor.ones_like(a)
        # Woof. `a` is a `Tensor`, not a `LazyTensor`. What does this do?
        F.buffer_store(a, b)
    assert not a.real
    asyncio.run(a.realize)
    assert a.real


def test_unrealized_value_as_buffer() -> None:
    with F.lazy():
        a = Tensor.zeros([5, 5])
        b = Tensor.ones_like(a)
        assert not a.real
        F.buffer_store(a, b)
        assert not a.real
    asyncio.run(a.realize)
    assert a.real


def test_buffervalue_on_realized_tensor() -> None:
    DEVICE = Accelerator() if accelerator_count() else CPU()
    a_data = Buffer.zeros([5, 5], DType.float32, DEVICE)
    a = Tensor(storage=a_data)
    assert a.real
    with F.lazy():
        _ = BufferValue(a)
        # Don't know whether the value was thrown away or used
        # in a mutating op!
        assert not a.real
    asyncio.run(a.realize)
    assert a.real


def test_mutation_op_order() -> None:
    a = Tensor.zeros([1])
    b = Tensor.ones_like(a)
    c = a + b
    F.buffer_store(a, b)
    d = a + b
    assert a.item() == 1.0
    assert b.item() == 1.0
    assert c.item() == 1.0
    assert d.item() == 2.0


@pytest.mark.skip(reason="flaky - GEX-3682")
def test_mutation_op_order_lazy() -> None:
    with F.lazy():
        a = Tensor.zeros([1])
        b = Tensor.ones_like(a)
        c = a + b
        F.buffer_store(a, b)
        d = a + b
    asyncio.run(c.realize)
    asyncio.run(d.realize)
    assert a.item() == 1.0
    assert b.item() == 1.0
    assert c.item() == 1.0
    assert d.item() == 2.0


# ---------------------------------------------------------------------------
# Tensor.to() fast-path tests
# ---------------------------------------------------------------------------


def test_tensor_to_same_device_returns_self() -> None:
    """Realized tensor.to(same_device) returns the same object (no-op)."""
    buf = Buffer.zeros([3, 4], DType.float32, CPU())
    t = Tensor(storage=buf)
    assert t.real
    result = t.to(CPU())
    assert result is t


@pytest.mark.skipif(not accelerator_count(), reason="requires GPU")
def test_tensor_to_same_device_gpu_returns_self() -> None:
    """Same-device fast path works on GPU."""
    gpu = Accelerator()
    buf = Buffer.zeros([2, 3], DType.float32, gpu)
    t = Tensor(storage=buf)
    result = t.to(gpu)
    assert result is t


@pytest.mark.skipif(not accelerator_count(), reason="requires GPU")
def test_tensor_to_different_device() -> None:
    """Realized tensor.to(other_device) returns a new tensor via Buffer.to()."""
    cpu_buf = Buffer(DType.float32, [2, 3], CPU())
    for idx in cpu_buf._iterate_indices():
        cpu_buf[idx] = 1.0
    t_cpu = Tensor(storage=cpu_buf)

    t_gpu = t_cpu.to(Accelerator())
    assert t_gpu is not t_cpu
    assert t_gpu.device == Accelerator()
    assert t_gpu.real
    assert list(t_gpu.shape) == [2, 3]
    assert t_gpu.dtype == DType.float32

    roundtrip = t_gpu.to(CPU())
    assert roundtrip.device == CPU()
    np.testing.assert_array_equal(
        np.from_dlpack(roundtrip.driver_tensor),
        np.ones([2, 3], dtype=np.float32),
    )


@pytest.mark.skipif(not accelerator_count(), reason="requires GPU")
def test_tensor_to_roundtrip_data_integrity() -> None:
    """CPU -> GPU -> CPU preserves data exactly."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = Tensor(storage=Buffer.from_numpy(src))
    assert t.device == CPU()

    t_gpu = t.to(Accelerator())
    t_back = t_gpu.to(CPU())

    np.testing.assert_array_equal(np.from_dlpack(t_back.driver_tensor), src)


def test_tensor_to_unrealized_uses_graph_path() -> None:
    """Unrealized tensor.to() still goes through graph-based F.transfer_to."""
    DEVICE = Accelerator() if accelerator_count() else CPU()
    with F.lazy():
        a = Tensor.zeros([2, 2], device=DEVICE)
        b = a.to(DEVICE)
        assert not b.real

    asyncio.run(b.realize)
    assert b.real
    assert b.device == DEVICE


@pytest.mark.skipif(not accelerator_count(), reason="requires GPU")
def test_tensor_to_idempotent_module() -> None:
    """Module.to(device) twice doesn't re-allocate parameters already there."""
    from max.experimental.nn import Linear

    model = Linear(4, 3)
    gpu = Accelerator()
    model.to(gpu)

    param_ids_first = {name: id(t) for name, t in model.parameters}

    model.to(gpu)

    param_ids_second = {name: id(t) for name, t in model.parameters}
    assert param_ids_first == param_ids_second


# ---------------------------------------------------------------------------
# Tensor.cast() idempotency tests
# ---------------------------------------------------------------------------


def test_tensor_cast_same_dtype_returns_self() -> None:
    """Realized tensor.cast(same_dtype) returns the same object (no-op)."""
    buf = Buffer.zeros([3, 4], DType.float32, CPU())
    t = Tensor(storage=buf)
    assert t.real
    result = t.cast(DType.float32)
    assert result is t


def test_ops_cast_different_dtype_emits_op() -> None:
    """Graph-level ops.cast(x, different_dtype) emits a new CastOp."""
    input_type = GraphTensorType(DType.float32, shape=[2, 3], device=CPU())
    with Graph("cast_diff_test", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        y = ops.cast(x, DType.int32)
        assert y._mlir_value is not x._mlir_value
        assert y.dtype == DType.int32


def test_ops_cast_same_dtype_no_op() -> None:
    """Graph-level ops.cast(x, x.dtype) returns the same underlying value."""
    input_type = GraphTensorType(DType.float32, shape=[2, 3], device=CPU())
    with Graph("cast_noop_test", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        y = ops.cast(x, DType.float32)
        assert y._mlir_value is x._mlir_value
