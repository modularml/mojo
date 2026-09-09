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
"""mutable ops tests."""

import os
from pathlib import Path

import numpy as np
import pytest
from conftest import buffer_types, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max import mlir
from max._core.dialects import mo
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Value,
    _ChainType,
    _ChainValue,
    ops,
)

shared_dtypes = st.shared(st.from_type(DType))
shared_shapes = st.shared(shapes().filter(lambda shape: 0 not in shape))
tensor_type = tensor_types(shapes=shared_shapes, dtypes=shared_dtypes)
buffer_type = buffer_types(shapes=shared_shapes, dtypes=shared_dtypes)


def _custom_ops_path() -> Path:
    path = os.getenv("CUSTOM_OPS_PATH")
    if path is None:
        pytest.skip("CUSTOM_OPS_PATH is not set; custom ops unavailable")
    return Path(path)


@given(buffer_type=...)
def test_mlir_type_checking(buffer_type: BufferType) -> None:
    with Graph(
        "buffer",
        input_types=[
            buffer_type,
        ],
    ) as graph:
        buffer = graph.inputs[0]
        type = buffer.type
        assert isinstance(buffer, BufferValue)
        assert type == buffer_type
        assert not isinstance(buffer, mlir.Value)
        assert isinstance(buffer._mlir_value.type, mo.BufferType)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_value_constructor(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    with Graph(
        "buffer_store",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        buffer = Value.from_mlir(graph.inputs[1]._mlir_value)
        assert isinstance(buffer, BufferValue)
        assert isinstance(buffer.type, BufferType)
        tensor = Value.from_mlir(graph.inputs[0]._mlir_value)
        assert isinstance(tensor, TensorValue)
        assert isinstance(tensor.type, TensorType)
        with pytest.raises(Exception):
            TensorValue.from_mlir(graph.inputs[1]._mlir_value)
        with pytest.raises(Exception):
            BufferValue.from_mlir(graph.inputs[0]._mlir_value)

        buffer = BufferValue(graph.inputs[1]._mlir_value)
        assert isinstance(buffer, BufferValue)
        assert isinstance(buffer.type, BufferType)
        tensor = TensorValue(graph.inputs[0]._mlir_value)
        assert isinstance(tensor, TensorValue)
        assert isinstance(tensor.type, TensorType)

        with pytest.raises(AssertionError):
            BufferValue(graph.inputs[0]._mlir_value)
        with pytest.raises(AssertionError):
            TensorValue(graph.inputs[1]._mlir_value)

        buffer = BufferValue(graph.inputs[1])
        assert isinstance(buffer, BufferValue)
        assert isinstance(buffer.type, BufferType)
        tensor = TensorValue(graph.inputs[0].tensor)
        assert isinstance(tensor, TensorValue)
        assert isinstance(tensor.type, TensorType)

        with pytest.raises(TypeError):
            BufferValue(graph.inputs[0])
        with pytest.raises(TypeError):
            TensorValue(graph.inputs[1].tensor)

        with pytest.raises(TypeError):
            TensorValue(0)

        with pytest.raises(TypeError):
            BufferValue(0)  # type: ignore


# buffer and tensor inputs share dtype and shape
@given(buffer_type=...)
def test_load(buffer_type: BufferType) -> None:
    with Graph(
        "buffer_load",
        input_types=[
            buffer_type,
        ],
    ) as graph:
        buffer = graph.inputs[0].buffer
        chain_0 = graph.device_chains[buffer.device]
        assert isinstance(chain_0, _ChainValue)
        assert isinstance(chain_0.type, _ChainType)

        y = ops.buffer_load(buffer)
        chain_1 = graph.device_chains[buffer.device]

        assert isinstance(chain_1, _ChainValue)
        assert isinstance(chain_1.type, _ChainType)

        assert y.shape == buffer.shape
        assert y.dtype == buffer.dtype
        assert isinstance(y, TensorValue)
        # Check the chain is updated.

        assert chain_0 != chain_1

        graph.output()
        graph._mlir_op.verify()
        assert "rmo.mo.mutable.load" in str(graph)
        assert "mo.chain.create" in str(graph)


# buffer and tensor inputs share dtype and shape
@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_store(tensor_type: TensorType, buffer_type: BufferType) -> None:
    with Graph(
        "buffer_store",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        tensor = graph.inputs[0].tensor
        buffer = graph.inputs[1].buffer
        chain_0 = graph.device_chains[buffer.device]
        ops.buffer_store(buffer, tensor)
        chain_1 = graph.device_chains[buffer.device]

        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype

        # Check the chain is updated.
        assert chain_0 != chain_1

        graph.output()
        graph._mlir_op.verify()
        assert "rmo.mo.mutable.store" in str(graph)
        assert "mo.chain.create" in str(graph)


@given(buffer_type=...)
def test_load_store(buffer_type: BufferType) -> None:
    with Graph(
        "buffer_load_store",
        input_types=[
            buffer_type,
        ],
    ) as graph:
        buffer = graph.inputs[0].buffer
        chain_0 = graph.device_chains[buffer.device]
        tensor = ops.buffer_load(buffer)
        chain_1 = graph.device_chains[buffer.device]

        assert tensor.shape == buffer.shape
        assert tensor.dtype == buffer.dtype
        assert isinstance(tensor, TensorValue)
        assert chain_0 != chain_1

        ops.buffer_store(buffer, tensor)
        chain_2 = graph.device_chains[buffer.device]

        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype
        assert chain_0 != chain_2
        assert chain_1 != chain_2

        graph.output()
        graph._mlir_op.verify()
        assert "mo.chain.create" in str(graph)
        assert "rmo.mo.mutable.load" in str(graph)
        assert "rmo.mo.mutable.store" in str(graph)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_load_store_ellipsis_slice(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    assume(tensor_type.rank > 1 and buffer_type.rank > 1)

    with Graph(
        "buffer_load",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        tensor = graph.inputs[0].tensor
        buffer = graph.inputs[1].buffer
        chain_0 = graph.device_chains[buffer.device]
        buffer[...] = tensor + buffer[...]
        chain_1 = graph.device_chains[buffer.device]

        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype
        # Check the chain is updated.
        assert chain_0 != chain_1

        graph.output()
        graph._mlir_op.verify()
        assert "rmo.mo.mutable.load" in str(graph)
        assert "rmo.mo.mutable.store" in str(graph)
        assert "rmo.mo.mutable.store.slice" not in str(graph)
        assert "mo.chain.create" in str(graph)


@given(tensor_type=tensor_type)
def test_store_slice_mismatched_devices(tensor_type: TensorType) -> None:
    buffer_device = DeviceRef.GPU(1)
    assume(tensor_type.device != buffer_device)
    buffer_type = BufferType(
        tensor_type.dtype, tensor_type.shape, buffer_device
    )
    with Graph(
        "buffer_store_slice", input_types=[tensor_type, buffer_type]
    ) as graph:
        tensor, buffer = graph.inputs
        with pytest.raises(ValueError, match="same device"):
            ops.buffer_store_slice(buffer.buffer, tensor.tensor, [...])


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_load_store_slice(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    assume(tensor_type.rank > 1 and buffer_type.rank > 1)

    with Graph(
        "buffer_load",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        tensor = graph.inputs[0].tensor
        buffer = graph.inputs[1].buffer
        chain_0 = graph.device_chains[buffer.device]
        buffer[0] = tensor[0] + buffer[0]
        chain_1 = graph.device_chains[buffer.device]

        assert buffer.shape == tensor.shape
        assert buffer.dtype == tensor.dtype
        # Check the chain is updated.
        assert chain_0 != chain_1

        graph.output()
        graph._mlir_op.verify()
        assert "rmo.mo.mutable.load" in str(graph)
        assert "rmo.mo.mutable.store" in str(graph)
        assert "rmo.mo.mutable.store.slice" in str(graph)
        assert "mo.chain.create" in str(graph)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_no_implicit_load(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    assume(tensor_type.rank > 1 and buffer_type.rank > 1)

    with Graph(
        "buffer_load",
        input_types=[
            tensor_type,
            buffer_type,
        ],
    ) as graph:
        tensor = graph.inputs[0]
        buffer = graph.inputs[1]

        with pytest.raises(TypeError):  # binary ops
            tensor + buffer  # type: ignore

        with pytest.raises(TypeError):  # unary ops
            abs(buffer)  # type: ignore

        assert "rmo.mo.mutable.load" not in str(graph)
        assert "rmo.mo.slice" not in str(graph)


@given(tensor_type=tensor_type, buffer_type=buffer_type)
def test_prints_with_buffer_ops(
    tensor_type: TensorType, buffer_type: BufferType
) -> None:
    with Graph(
        "debug_prints_and_mutable_ops",
        input_types=[buffer_type, tensor_type],
    ) as graph:
        buffer: BufferValue = graph.inputs[0].buffer
        tensor: TensorValue = graph.inputs[1].tensor

        chain_0 = graph.device_chains[DeviceRef.CPU()]

        tensor.print()
        chain_1 = graph.device_chains[DeviceRef.CPU()]

        # Buffer load on a CPU buffer shares the host chain (HOST ==
        # ``DeviceRef.CPU()``), so it also advances chain_1 forward.
        x = buffer[...]
        chain_2 = graph.device_chains[DeviceRef.CPU()]

        x.print()
        chain_3 = graph.device_chains[DeviceRef.CPU()]

        ops.buffer_store(buffer, tensor)
        chain_4 = graph.device_chains[DeviceRef.CPU()]

        graph.output()

        assert chain_0 != chain_1
        assert chain_1 != chain_2
        assert chain_2 != chain_3
        assert chain_3 != chain_4


def test_buffer_ops_sequence_after_inplace_custom() -> None:
    """Buffer ops observe effects of preceding inplace_custom on same device."""
    custom_ops_path = _custom_ops_path()
    buffer_type = BufferType(
        DType.float32,
        shape=[4],
        device=DeviceRef.CPU(),
    )

    with Graph(
        "buffer_device_chain_merge",
        input_types=[buffer_type],
        custom_extensions=[custom_ops_path],
    ) as graph:
        buffer = graph.inputs[0].buffer

        # @extensibility.register("mutable_test_op") increments the first element.
        ops.inplace_custom(
            "mutable_test_op",
            device=buffer.device,
            values=[buffer],
        )

        tensor = ops.buffer_load(buffer)
        ops.buffer_store(buffer, tensor)

        graph.output(buffer)

    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)

    input_buffer = Buffer.from_numpy(np.zeros((4,), dtype=np.float32)).to(
        model.input_devices[0]
    )
    (result,) = model.execute(input_buffer)
    assert isinstance(result, Buffer)
    np.testing.assert_allclose(
        result.to_numpy(),
        np.array([1, 0, 0, 0], dtype=np.float32),
    )
