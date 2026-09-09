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
"""Test the max.engine Python bindings with Max Graph."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def create_test_graph() -> Graph:
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    with Graph("add", input_types=(input_type, input_type)) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
    return graph


def test_max_graph(session: InferenceSession) -> None:
    graph = create_test_graph()
    compiled = session.load(graph)
    a_np = np.ones((1, 1)).astype(np.float32)
    b_np = np.ones((1, 1)).astype(np.float32)
    a = Buffer.from_numpy(a_np).to(compiled.input_devices[0])
    b = Buffer.from_numpy(b_np).to(compiled.input_devices[1])
    output = compiled.execute(a, b)
    assert isinstance(output[0], Buffer)
    assert np.allclose((a_np + b_np), output[0].to_numpy())


def test_max_graph_export(session: InferenceSession) -> None:
    """Creates a graph via max-graph API, exports the mef to a tempfile, and
    check to ensure that the file contents are non-empty."""

    with tempfile.NamedTemporaryFile() as mef_file:
        graph = create_test_graph()
        compiled = session.load(graph)
        compiled._export_mef(mef_file.name)
        assert os.path.getsize(mef_file.name) > 0


def test_max_graph_export_import_mef(session: InferenceSession) -> None:
    """Creates a graph via max-graph API, exports the mef to a tempfile, and
    loads the mef. Both the original model from the max-graph and the model
    from the mef are executed to ensure that they produce the same output."""

    with tempfile.NamedTemporaryFile() as mef_file:
        graph = create_test_graph()
        compiled = session.load(graph)
        compiled._export_mef(mef_file.name)
        a_np = np.ones((1, 1)).astype(np.float32)
        b_np = np.ones((1, 1)).astype(np.float32)
        a = Buffer.from_numpy(a_np).to(compiled.input_devices[0])
        b = Buffer.from_numpy(b_np).to(compiled.input_devices[1])
        output = compiled.execute(a, b)[0]
        assert isinstance(output, Buffer)
        output_numpy = output.to_numpy()
        assert np.allclose((a_np + b_np), output_numpy)
        compiled2 = session.load(mef_file.name)
        # Executing a mef-loaded model with a device tensor seems to not work.
        output2 = compiled2.execute(a, b)[0]
        assert isinstance(output2, Buffer)
        output2_numpy = output2.to_numpy()
        assert np.allclose((a_np + b_np), output2_numpy)
        assert np.allclose(output_numpy, output2_numpy)


def test_compiled_model_export_mef(session: InferenceSession) -> None:
    """Compiles a graph without initializing it on a device, exports the mef
    from the CompiledModel artifact, and checks that the file contents are
    non-empty."""

    with tempfile.NamedTemporaryFile() as mef_file:
        graph = create_test_graph()
        compiled = session.compile(graph)
        compiled.export_mef(Path(mef_file.name))
        assert os.path.getsize(mef_file.name) > 0


def test_compiled_model_export_mef_roundtrip(
    session: InferenceSession,
) -> None:
    """Exports a mef from an uninitialized CompiledModel, then loads and
    executes it to verify the exported artifact is valid."""

    with tempfile.NamedTemporaryFile() as mef_file:
        graph = create_test_graph()
        compiled = session.compile(graph)
        compiled.export_mef(mef_file.name)
        model = session.load(mef_file.name)
        a_np = np.ones((1, 1)).astype(np.float32)
        b_np = np.ones((1, 1)).astype(np.float32)
        a = Buffer.from_numpy(a_np).to(model.input_devices[0])
        b = Buffer.from_numpy(b_np).to(model.input_devices[1])
        output = model.execute(a, b)[0]
        assert isinstance(output, Buffer)
        assert np.allclose((a_np + b_np), output.to_numpy())


def test_max_graph_device(session: InferenceSession) -> None:
    graph = create_test_graph()
    device = CPU()
    session = InferenceSession(devices=[device])
    compiled = session.load(graph)
    assert str(device) == str(compiled.devices[0])


def test_identity(session: InferenceSession) -> None:
    # Create identity graph.
    graph = Graph(
        "identity",
        lambda x: x,
        input_types=[TensorType(DType.int32, (1,), device=DeviceRef.CPU())],
    )

    # Compile and execute identity.
    model = session.load(graph)
    input = Buffer(shape=(1,), dtype=DType.int32)
    output = model.execute(input.to(model.input_devices[0]))
    # Test that using output's storage is still valid after destroying input.
    del input
    assert isinstance(output[0], Buffer)
    _ = output[0].to(CPU())[0]


def test_max_graph_export_import_mlir(session: InferenceSession) -> None:
    """Creates a graph via max-graph API, exports the mlir to a tempfile, and
    loads the mlir. Both the original model from the max-graph and the model
    from the mlir are executed to ensure that they produce the same output."""

    with tempfile.NamedTemporaryFile(mode="w+") as mlir_file:
        graph = create_test_graph()
        compiled = session.load(graph)
        mlir_file.write(graph._module.asm())
        a_np = np.ones((1, 1)).astype(np.float32)
        b_np = np.ones((1, 1)).astype(np.float32)
        a = Buffer.from_numpy(a_np).to(compiled.input_devices[0])
        b = Buffer.from_numpy(b_np).to(compiled.input_devices[1])
        output = compiled.execute(a, b)[0]
        assert isinstance(output, Buffer)
        output_np = output.to_numpy()
        assert output_np == a_np + b_np

        mlir_file.flush()

        # Now load the model from mlir.
        graph2 = Graph(name="add", path=Path(mlir_file.name))
        compiled2 = session.load(graph2)
        output2 = compiled2.execute(a, b)[0]
        assert isinstance(output2, Buffer)
        output2_np = output2.to_numpy()
        assert output2_np == a_np + b_np
        assert output_np == output2_np


def test_no_output_error_message(session: InferenceSession) -> None:
    with Graph("test", input_types=()) as graph:
        pass

    # TODO(MAXPLAT-331): Improve error message and test it here
    with pytest.raises(Exception):
        session.load(graph)
