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
"""Tests for MO graph interpreter module."""

from typing import Any

import numpy as np
import pytest
from max import _interpreter
from max._interpreter_ops import _MO_OP_HANDLERS, register_op_handler
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental.executor import (
    CompilingExecutor,
    InterpreterExecutor,
    default_executor,
)
from max.experimental.realization_context import EagerRealizationContext
from max.experimental.tensor import Tensor, realization_context
from max.graph import Graph, TensorType, ops


class TestValidateInputs:
    """Tests for input validation."""

    def test_validate_inputs_wrong_count(self) -> None:
        """Test that validate_inputs catches input count mismatch."""

        class MockGraph:
            @property
            def inputs(self) -> list[int]:
                return [1, 2, 3]  # 3 inputs expected

        graph: Any = MockGraph()
        inputs: Any = [object()]
        with pytest.raises(ValueError, match="Expected 3 inputs, got 1"):
            _interpreter._validate_inputs(graph, inputs)


class TestCanExecute:
    """Tests for the can_execute() pre-flight check."""

    def test_can_execute_supported_graph(self) -> None:
        """Test can_execute returns True for a graph with only supported ops."""
        with Graph("supported", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            c = ops.add(a, b)
            graph.output(c)

        assert _interpreter.can_execute(graph) is True

    def test_can_execute_returns_false_for_compilation_required_op(
        self, monkeypatch: Any
    ) -> None:
        """Test can_execute returns False when graph contains a
        compilation-required op (e.g. mo.CustomOp)."""
        # Temporarily add ConstantOp to _COMPILATION_REQUIRED_OP_NAMES to
        # simulate a compilation-required op without needing a real custom
        # kernel.  ConstantOp is used because it's always present in simple
        # graphs and its name is stable.
        monkeypatch.setattr(
            _interpreter, "_COMPILATION_REQUIRED_OP_NAMES", ("ConstantOp",)
        )
        with Graph("custom", input_types=[]) as graph:
            c = ops.constant([1.0, 1.0], dtype=DType.float32, device=CPU())
            graph.output(c)

        assert _interpreter.can_execute(graph) is False

    def test_can_execute_max_ops_within_limit(self) -> None:
        """Test can_execute respects max_ops when graph is within limit."""
        with Graph("small", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            c = ops.add(a, b)
            graph.output(c)

        # 3 dispatchable ops (2 constants + 1 add), limit at 5 -> ok
        assert _interpreter.can_execute(graph, max_ops=5) is True

    def test_can_execute_max_ops_exceeds_limit(self) -> None:
        """Test can_execute returns False when graph exceeds max_ops."""
        with Graph("big", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            c = ops.add(a, b)
            d = ops.mul(a, b)
            e = ops.sub(c, d)
            graph.output(e)

        # 5 dispatchable ops (2 constants + add + mul + sub), limit at 2 -> too many
        assert _interpreter.can_execute(graph, max_ops=2) is False

    def test_can_execute_no_limit(self) -> None:
        """Test can_execute with max_ops=None imposes no limit."""
        with Graph("unlimited", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            c = ops.add(a, b)
            d = ops.mul(c, b)
            e = ops.sub(d, a)
            graph.output(e)

        assert _interpreter.can_execute(graph, max_ops=None) is True

    def test_can_execute_returns_false_for_unhandled_op(self) -> None:
        """Test can_execute returns False when an op has no registered handler."""
        with Graph("unsupported", input_types=[]) as graph:
            a = ops.constant(
                [[1.0, 2.0], [3.0, 4.0]], dtype=DType.float32, device=CPU()
            )
            # pad has no interpreter handler
            result = ops.pad(a, [0, 1, 0, 1])
            graph.output(result)

        assert _interpreter.can_execute(graph) is False


class TestOpHandlerRegistry:
    """Tests for op handler registration."""

    def test_register_handler(self) -> None:
        """Test handler registration decorator."""

        class MockOpType:
            pass

        op_type: Any = MockOpType

        @register_op_handler(op_type)
        def test_handler(op: Any, inputs: Any) -> list[Any]:
            return []

        assert op_type in _MO_OP_HANDLERS
        assert _MO_OP_HANDLERS[op_type] is test_handler

        # Cleanup
        del _MO_OP_HANDLERS[op_type]

    def test_register_handler_overwrites(self) -> None:
        """Test that registering same op type overwrites previous handler."""

        class MockOpType:
            pass

        op_type: Any = MockOpType

        @register_op_handler(op_type)
        def handler1(op: Any, inputs: Any) -> list[Any]:
            return [1]

        @register_op_handler(op_type)
        def handler2(op: Any, inputs: Any) -> list[Any]:
            return [2]

        assert _MO_OP_HANDLERS[op_type] is handler2

        # Cleanup
        del _MO_OP_HANDLERS[op_type]


class TestGraphExecution:
    """Integration tests that build MO graphs and run them through the interpreter."""

    def test_constant_only_graph(self) -> None:
        """Test executing a graph with only a constant output."""
        # Create a graph that just outputs a constant
        with Graph("constant_graph", input_types=[]) as graph:
            c = ops.constant([1.0, 2.0, 3.0], dtype=DType.float32, device=CPU())
            graph.output(c)

        # Execute through interpreter
        outputs = _interpreter.execute(graph, [])

        # Verify output
        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_graph_with_input(self) -> None:
        """Test executing a graph with an input tensor."""
        input_type = TensorType(DType.float32, [2, 3], CPU())
        with Graph("input_graph", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            c = ops.constant(
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                dtype=DType.float32,
                device=CPU(),
            )
            y = ops.add(x, c)
            graph.output(y)

        # Create input buffer
        input_np = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        input_buffer = Buffer.from_numpy(input_np)

        outputs = _interpreter.execute(graph, [input_buffer])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)

        expected = np.array(
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_chained_operations(self) -> None:
        """Test a graph with multiple chained operations."""
        with Graph("chained_ops", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0, 3.0], dtype=DType.float32, device=CPU())
            b = ops.constant([2.0, 2.0, 2.0], dtype=DType.float32, device=CPU())
            # (a + b) * 3 - 1
            c = ops.add(a, b)  # [3, 4, 5]
            three = ops.constant(
                [3.0, 3.0, 3.0], dtype=DType.float32, device=CPU()
            )
            d = ops.mul(c, three)  # [9, 12, 15]
            one = ops.constant(
                [1.0, 1.0, 1.0], dtype=DType.float32, device=CPU()
            )
            e = ops.sub(d, one)  # [8, 11, 14]
            graph.output(e)

        outputs = _interpreter.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)

        expected = np.array([8.0, 11.0, 14.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_multiple_outputs(self) -> None:
        """Test a graph with multiple outputs."""
        with Graph("multi_output", input_types=[]) as graph:
            a = ops.constant([1.0, 2.0], dtype=DType.float32, device=CPU())
            b = ops.constant([3.0, 4.0], dtype=DType.float32, device=CPU())
            sum_ab = ops.add(a, b)
            prod_ab = ops.mul(a, b)
            graph.output(sum_ab, prod_ab)

        outputs = _interpreter.execute(graph, [])

        assert len(outputs) == 2
        sum_result = outputs[0]
        assert isinstance(sum_result, Buffer)
        prod_result = outputs[1]
        assert isinstance(prod_result, Buffer)

        np.testing.assert_array_almost_equal(
            sum_result.to_numpy(), np.array([4.0, 6.0], dtype=np.float32)
        )
        np.testing.assert_array_almost_equal(
            prod_result.to_numpy(), np.array([3.0, 8.0], dtype=np.float32)
        )


class TestRealizationContextIntegration:
    """The context binds an executor from its constructor arguments."""

    def test_default_uses_default_executor(self) -> None:
        """With no arguments the context uses the ambient default executor."""
        assert EagerRealizationContext()._executor is default_executor()

    def test_explicit_executor_is_used(self) -> None:
        """An injected executor takes precedence over the default."""
        executor = InterpreterExecutor()
        assert EagerRealizationContext(executor=executor)._executor is executor

    def test_use_interpreter_true_selects_interpreter(self) -> None:
        """The deprecated ``use_interpreter=True`` shim forces the interpreter."""
        ctx = EagerRealizationContext(use_interpreter=True)
        assert isinstance(ctx._executor, InterpreterExecutor)

    def test_use_interpreter_false_selects_compile(self) -> None:
        """The deprecated ``use_interpreter=False`` shim forces compilation."""
        ctx = EagerRealizationContext(use_interpreter=False)
        assert isinstance(ctx._executor, CompilingExecutor)


class TestRuntimeErrors:
    """Interpreter runtime errors always propagate, however it was selected."""

    def test_auto_mode_recovers_from_execute_error(
        self, monkeypatch: Any
    ) -> None:
        """The auto-selected executor serves a graph the interpreter failed
        on with its compiled model. TODO(MXF-595)."""
        call_log: list[str] = []

        def _failing_execute(graph: Any, inputs: Any) -> Any:
            call_log.append("interpreter_called")
            raise RuntimeError("simulated unsupported dtype")

        monkeypatch.setattr(_interpreter, "execute", _failing_execute)

        with EagerRealizationContext() as ctx, realization_context(ctx):
            result = Tensor.zeros([3]) + 1.0

        assert result.real
        assert call_log == ["interpreter_called"]

    def test_explicit_interpreter_does_not_swallow_error(
        self, monkeypatch: Any
    ) -> None:
        """When the interpreter is explicitly requested, runtime errors
        must propagate instead of silently falling back."""

        def _failing_execute(graph: Any, inputs: Any) -> Any:
            raise RuntimeError("simulated unsupported dtype")

        monkeypatch.setattr(_interpreter, "execute", _failing_execute)

        with pytest.raises(RuntimeError, match="simulated unsupported dtype"):
            with (
                EagerRealizationContext(use_interpreter=True) as ctx,
                realization_context(ctx),
            ):
                _ = Tensor.zeros([3]) + 1.0


class TestDebugPrintOps:
    """Tests for DebugPrintOp and DebugTensorPrintOp interpreter handlers."""

    def test_can_execute_with_debug_print(self) -> None:
        """Test can_execute returns True for a graph using ops.print (string)."""
        with Graph("debug_print", input_types=[]) as graph:
            ops.print("hello world", label="test")
            c = ops.constant([1.0], dtype=DType.float32, device=CPU())
            graph.output(c)

        assert _interpreter.can_execute(graph) is True

    def test_can_execute_with_debug_tensor_print(self) -> None:
        """Test can_execute returns True for a graph using ops.print on a tensor."""
        with Graph("debug_tensor", input_types=[]) as graph:
            t = ops.constant([1.0, 2.0, 3.0], dtype=DType.float32, device=CPU())
            ops.print(t, label="my_tensor")
            graph.output(t)

        assert _interpreter.can_execute(graph) is True

    def test_debug_print_executes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that DebugPrintOp handler prints the string to stdout."""
        with Graph("print_exec", input_types=[]) as graph:
            ops.print("hello from graph", label="dbg")
            c = ops.constant([1.0], dtype=DType.float32, device=CPU())
            graph.output(c)

        outputs = _interpreter.execute(graph, [])
        assert len(outputs) == 1

        captured = capsys.readouterr()
        assert "[dbg] hello from graph" in captured.out

    def test_debug_tensor_print_executes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that DebugTensorPrintOp handler prints tensor data to stdout."""
        with Graph("tensor_print_exec", input_types=[]) as graph:
            t = ops.constant(
                [10.0, 20.0, 30.0], dtype=DType.float32, device=CPU()
            )
            ops.print(t, label="vec")
            graph.output(t)

        outputs = _interpreter.execute(graph, [])
        assert len(outputs) == 1

        captured = capsys.readouterr()
        assert "[vec]" in captured.out
        assert "10." in captured.out
        assert "20." in captured.out
        assert "30." in captured.out

    def test_debug_print_no_label(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test DebugPrintOp with empty label prints just the value."""
        with Graph("print_no_label", input_types=[]) as graph:
            ops.print("bare message", label="")
            c = ops.constant([1.0], dtype=DType.float32, device=CPU())
            graph.output(c)

        _interpreter.execute(graph, [])

        captured = capsys.readouterr()
        assert "bare message" in captured.out


class TestInterpreterMaxOps:
    """Tests for the MAX_INTERPRETER_MAX_OPS threshold."""

    def test_default_max_ops(self, monkeypatch: Any) -> None:
        """Test that the default max_ops is 1024."""
        monkeypatch.delenv("MAX_INTERPRETER_MAX_OPS", raising=False)
        from max.experimental.executor import _interpreter_max_ops

        assert _interpreter_max_ops() == 1024

    def test_env_var_overrides_max_ops(self, monkeypatch: Any) -> None:
        """Test that MAX_INTERPRETER_MAX_OPS env var overrides the default."""
        monkeypatch.setenv("MAX_INTERPRETER_MAX_OPS", "5")
        from max.experimental.executor import _interpreter_max_ops

        assert _interpreter_max_ops() == 5

    def test_env_var_max_ops_2(self, monkeypatch: Any) -> None:
        """Test setting the threshold to 2."""
        monkeypatch.setenv("MAX_INTERPRETER_MAX_OPS", "2")
        from max.experimental.executor import _interpreter_max_ops

        assert _interpreter_max_ops() == 2

    def test_invalid_env_var_uses_default(self, monkeypatch: Any) -> None:
        """Test that non-numeric env var falls back to default."""
        monkeypatch.setenv("MAX_INTERPRETER_MAX_OPS", "abc")
        from max.experimental.executor import _interpreter_max_ops

        assert _interpreter_max_ops() == 1024


class TestConstantScalarOp:
    """Tests for ConstantScalarOp interpreter handler.

    Uses mock ops to directly invoke _handle_constant_scalar because there
    is no public graph API that emits mo.constant.scalar -- it is an
    internal MO dialect op.
    """

    def test_integer_scalar(self) -> None:
        """ConstantScalarOp with an integer value produces a rank-0 buffer."""
        from unittest.mock import MagicMock

        from max._core.dialects import builtin, mo
        from max._interpreter_ops.handlers import _handle_constant_scalar

        mock_op = MagicMock(spec=mo.ConstantScalarOp)

        mock_result = MagicMock()
        mock_scalar_type = MagicMock(spec=mo.ScalarType)
        mock_scalar_type.dtype = DType.int64
        mock_result.type = mock_scalar_type
        mock_op.results = [mock_result]

        mock_op.value = builtin.IntegerAttr(
            builtin.IntegerType(64, builtin.SignednessSemantics.signed), 42
        )

        outputs = _handle_constant_scalar(mock_op, [])
        assert len(outputs) == 1
        buf = outputs[0]
        assert isinstance(buf, Buffer)
        assert buf.to_numpy().item() == 42

    def test_float_scalar(self) -> None:
        """ConstantScalarOp with a float value."""
        from unittest.mock import MagicMock

        from max._core.dialects import builtin, mo
        from max._interpreter_ops.handlers import _handle_constant_scalar

        mock_op = MagicMock(spec=mo.ConstantScalarOp)

        mock_result = MagicMock()
        mock_scalar_type = MagicMock(spec=mo.ScalarType)
        mock_scalar_type.dtype = DType.float32
        mock_result.type = mock_scalar_type
        mock_op.results = [mock_result]

        mock_op.value = MagicMock(spec=builtin.FloatAttr)
        mock_op.value.value = 3.14

        outputs = _handle_constant_scalar(mock_op, [])
        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)
        np.testing.assert_allclose(result.to_numpy().item(), 3.14)

    def test_bool_scalar(self) -> None:
        """ConstantScalarOp with a boolean value."""
        from unittest.mock import MagicMock

        from max._core.dialects import builtin, mo
        from max._interpreter_ops.handlers import _handle_constant_scalar

        mock_op = MagicMock(spec=mo.ConstantScalarOp)

        mock_result = MagicMock()
        mock_scalar_type = MagicMock(spec=mo.ScalarType)
        mock_scalar_type.dtype = DType.bool
        mock_result.type = mock_scalar_type
        mock_op.results = [mock_result]

        mock_op.value = builtin.BoolAttr(True)

        outputs = _handle_constant_scalar(mock_op, [])
        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)
        assert result.to_numpy().item() != 0


class TestConstantExternalOp:
    """The interpreter must statically refuse graphs with external constants.

    mo.constant.external requires a weights registry that only the compiled
    execution path provides, so can_execute() routes such graphs to the
    graph compiler.
    """

    def test_can_execute_returns_false(self) -> None:
        with Graph("external_const_graph", input_types=[]) as graph:
            w = ops.constant_external(
                "my_weight",
                TensorType(DType.float32, [2, 2], CPU()),
            )
            graph.output(w)

        assert _interpreter.can_execute(graph) is False
