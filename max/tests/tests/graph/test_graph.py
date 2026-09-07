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
"""Test the max.graph Python bindings."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import mock

import pytest
from conftest import buffer_types, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max import mlir
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops
from max.graph.graph import _location
from max.mlir.dialects import rmo

empty_graphs = st.builds(
    Graph, st.text(), input_types=st.lists(st.from_type(TensorType))
)

shared_dtypes = st.shared(st.from_type(DType))
shared_shapes = st.shared(shapes().filter(lambda shape: 0 not in shape))
tensor_type = tensor_types(shapes=shared_shapes, dtypes=shared_dtypes)
buffer_type = buffer_types(shapes=shared_shapes, dtypes=shared_dtypes)


@given(graph=empty_graphs)
def test_simple_graphs(graph: Graph) -> None:
    assume(len(graph.inputs) > 0)
    with graph:
        graph.output(graph.inputs[0])


def test_mlir_module_create() -> None:
    """Tests whether we can import mlir and create a Module.

    This is a basic smoke test for max.graph Python packaging.
    """
    with mlir.Context(), mlir.Location.unknown():
        _ = mlir.Module.create()


def test_elementwise_add_graph() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.CPU(),
            )
        ],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 1)


def test_elementwise_add_graph_with_device_prop() -> None:
    """Builds a simple graph with explicit device on inputs and checks for output device propagation in the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.GPU(0),
            ),
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.GPU(0),
            ),
        ],
    ) as graph:
        graph.output(graph.inputs[0].tensor + graph.inputs[1].tensor)
        # Ensure input tensor has cuda
        for input in graph.inputs:
            assert "gpu" in str(input)
        # Ensure output tensor has cuda propagated
        assert " -> !mo.tensor<[batch, channels], f32, gpu:0>" in str(
            graph._mlir_op
        )


def test_transpose_graph_with_device_prop() -> None:
    """Builds a simple graph with an transpose and checks the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.GPU(0),
            )
        ],
    ) as graph:
        graph.output(ops.transpose(graph.inputs[0].tensor, -1, -2))
        for input in graph.inputs:
            assert "gpu" in str(input)
        assert " -> !mo.tensor<[channels, batch], f32, gpu:0>" in str(
            graph._mlir_op
        )


def test_location_no_stack() -> None:
    with Graph("location"):
        with mock.patch("traceback.extract_stack") as mock_stack:
            mock_stack.return_value = []

            loc = _location()
            assert loc == mlir.Location.unknown()


def test_add_op() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    with Graph("add", input_types=(input_type, input_type)) as graph:
        lhs, rhs = graph.inputs
        elemwise_sum = ops.add(lhs, rhs)
        graph.output(elemwise_sum)

        # Check that the arg/result name attributes were added.
        assert "argument_names = " in str(graph._mlir_op)
        assert "result_names = " in str(graph._mlir_op)


def test_add_op_closure() -> None:
    """Uses a closure to build a simple graph with an elementwise addition
    and checks the IR.
    """

    def elementwise_add(lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        return ops.add(lhs, rhs)

    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    add_graph = Graph("add", elementwise_add, (input_type, input_type))

    assert "rmo.add" in str(add_graph._mlir_op)
    assert "mo.output" in str(add_graph._mlir_op)


def test_profile_scope_nests() -> None:
    """Nested profile_scope() wraps the innermost scope outermost."""
    input_type = TensorType(
        dtype=DType.float32, shape=[4], device=DeviceRef.CPU()
    )
    with Graph("profile_scope", input_types=[input_type]) as graph:
        with graph.profile_scope("outer"):
            with graph.profile_scope("inner"):
                y = ops.add(graph.inputs[0], graph.inputs[0])
        graph.output(y)

    asm = graph._mlir_op.get_asm(enable_debug_info=True, use_local_scope=True)
    # Innermost scope is the outermost ProfileScopeLocationAttr layer.
    assert 'loc(#mogg.profile_scope<"inner",' in asm
    assert 'profile_scope<"outer",' in asm


def test_profile_scope_color() -> None:
    """A colored scope stores its color in the ProfileScopeLocationAttr."""
    input_type = TensorType(
        dtype=DType.float32, shape=[4], device=DeviceRef.CPU()
    )
    with Graph("profile_scope_color", input_types=[input_type]) as graph:
        with graph.profile_scope("outer", color="orange"):
            with graph.profile_scope("inner"):
                y = ops.add(graph.inputs[0], graph.inputs[0])
        graph.output(y)

    asm = graph._mlir_op.get_asm(enable_debug_info=True)
    # The colored scope carries the color field; the uncolored sibling does not.
    assert 'loc(#mogg.profile_scope<"inner",' in asm
    assert 'color = "orange"' in asm
    assert 'profile_scope<"outer", loc(unknown), color = "orange">' in asm


def test_profile_scope_skips_constants_and_weights() -> None:
    """Constants and weights created inside a profile_scope carry no label."""
    weight = Weight(
        name="w",
        dtype=DType.float32,
        shape=[2],
        device=DeviceRef.CPU(),
    )
    with Graph("psc", input_types=[]) as graph:
        with graph.profile_scope("scope"):
            c = ops.constant(1.0, DType.float32, device=DeviceRef.CPU())
            w = graph.add_weight(weight)
            graph.output(c + w)

    asm = graph._mlir_op.get_asm(enable_debug_info=True, use_local_scope=True)
    # The graph name is intentionally not "profile_scope_*" so the broad
    # assertion is meaningful. Constants, weights, and the output bookkeeping
    # op are compile-time/structural and must not carry a profile_scope label,
    # even when created inside an active scope.
    for line in asm.splitlines():
        if (
            "mo.constant.external" in line
            or "mo.constant {value" in line
            or "mo.output" in line
        ):
            assert "profile_scope" not in line


def test_side_stream_stages_mo_sequence() -> None:
    """``ops.side_stream`` stages an ``mo.sequence`` region tagged with the
    requested stream id, with the body mapped through its block arguments.
    """
    input_type = TensorType(
        dtype=DType.float32, shape=[4], device=DeviceRef.GPU(0)
    )
    with Graph("side_stream", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        results = ops.side_stream(
            [x], lambda t: ops.relu(t), result_types=[x.type], stream_id=1
        )
        graph.output(results[0])

    ir = str(graph._mlir_op)
    assert "mo.sequence[1]" in ir
    assert "mo.yield" in ir


def test_invalid_operand() -> None:
    """Test that passing an invalid operand raises an error."""
    with Graph(
        "invalid_operand",
        input_types=[
            TensorType(DType.int64, [2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        input_tensor = graph.inputs[0]
        with pytest.raises(ValueError):
            Graph.current._add_op(rmo.add, [2, 5], input_tensor)
        graph.output(input_tensor)


def test_load_from_file() -> None:
    """Tests printing to and loading from a file."""
    graph = Graph(
        "identity",
        forward=lambda x: x,
        input_types=[TensorType(DType.int64, [1], device=DeviceRef.CPU())],
    )
    with NamedTemporaryFile("w") as mlir_text_file:
        # Flush so that the subsequent read works.
        print(graph, file=mlir_text_file, flush=True)
        loaded_graph = Graph("loaded", path=Path(mlir_text_file.name))

    assert isinstance(loaded_graph._mlir_op, mlir.Operation | mlir.OpView) and (
        str(loaded_graph) == str(graph)
    )


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        # f32 needing 7 significant digits, so MLIR spells it as a hex
        # literal. GLM-5.2's rope range stop; used to abort on print.
        (DType.float32, 2097152.0),
        (DType.float32, 1.0),
        # Any f64 at all: used to abort on parse against an assertions build
        # and silently reload as a denormal otherwise.
        (DType.float64, 10000.0),
        (DType.float64, 1.0),
    ],
)
def test_float_constants_survive_text_round_trip(
    dtype: DType, value: float
) -> None:
    """Tests that a float constant is unchanged by a dump-and-reload."""
    graph = Graph(
        "floats",
        forward=lambda: ops.constant(value, dtype, device=DeviceRef.CPU()),
        input_types=[],
    )
    text = graph.module._to_mlir_str()
    # Without a constant to carry the value, the comparison below would pass
    # for the wrong reason.
    assert "mo.constant" in text

    with NamedTemporaryFile("w") as mlir_text_file:
        print(text, file=mlir_text_file, flush=True)
        reloaded = Graph("floats", path=Path(mlir_text_file.name))

    assert reloaded.module._to_mlir_str() == text


def test_source_locations_round_trip_floats() -> None:
    """Tests that the source-location dump keeps float constants intact."""
    graph = Graph(
        "floats",
        forward=lambda: ops.constant(
            2097152.0, DType.float32, device=DeviceRef.CPU()
        ),
        input_types=[],
    )

    annotated = graph.module._to_mlir_str(source_locations=True)

    assert "mo.constant" in annotated
    # 0x4A000000 is 2097152.0; MLIR prints f32 needing 7 digits as hex.
    assert "0x4A000000" in annotated


def test_device_graph() -> None:
    with Graph(
        "device_graph",
        input_types=[],
        is_device_graph=True,
    ) as graph:
        graph.output()

    assert "isDeviceGraph" in str(graph)
