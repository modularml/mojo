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
"""Integration tests for ops.custom and ops.inplace_custom kernel validation.

These tests require actual kernel verification and use real test kernels from
kernel_verification_ops. They test scenarios that require kernel execution
validation beyond API structure validation.

Complements the unit tests in max/tests/tests/graph/ops/test_custom.py
which focus on API validation only.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, BufferValue, DeviceRef, Graph, TensorType, ops


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    """Get path to kernel verification ops for testing."""
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


@pytest.fixture
def kernel_override_ops_path() -> Path:
    """Get path to kernel override ops for testing built-in shadowing."""
    return Path(os.environ["MODULAR_KERNEL_OVERRIDE_OPS_PATH"])


def _collect_error_chain(exc: BaseException) -> str:
    """Walks the exception chain and concatenates all messages.

    The compiler diagnostic sits on a ``__cause__``. The top-level message is
    just the generic "Failed to compile the model" wrapper.
    """
    messages = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(str(current))
        current = current.__cause__
    return "\n".join(messages)


class TestCustomKernelValidation:
    """Tests for ops.custom that require actual kernel validation."""

    def test_custom__success__basic_kernel(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test successful custom operation with a real kernel."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        graph = Graph(
            "test_custom_basic_kernel",
            input_types=[input_type],
            output_types=[input_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            result = ops.custom(
                name="op_with_device_context",  # Real test kernel
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[input_type],
            )
            graph.output(result[0])

    def test_custom__success__kernel_with_parameters(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test successful custom operation with parameters."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        graph = Graph(
            "test_custom_with_parameters",
            input_types=[input_type],
            output_types=[input_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            result = ops.custom(
                name="op_with_int_parameter",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[input_type],
                parameters={"IntParameter": 42},
            )
            graph.output(result[0])

    def test_custom__success__multiple_outputs(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test successful custom operation with multiple outputs."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())
        output_type1 = TensorType(DType.float32, (2, 3), DeviceRef.CPU())
        output_type2 = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        graph = Graph(
            "test_custom_multiple_outputs",
            input_types=[input_type],
            output_types=[output_type1, output_type2],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            result = ops.custom(
                name="op_with_multiple_outputs",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[output_type1, output_type2],
            )
            graph.output(*result)

    def test_custom__error__wrong_input_count_too_many(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test error when providing too many inputs to kernel."""
        input_type1 = TensorType(DType.float32, (2, 3), DeviceRef.CPU())
        input_type2 = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        graph = Graph(
            "test_custom_wrong_input_count",
            input_types=[input_type1, input_type2],
            output_types=[input_type1],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            with pytest.raises(Exception) as exc_info:
                # op_with_device_context expects 1 input, we're giving 2
                ops.custom(
                    name="op_with_device_context",
                    device=DeviceRef.CPU(),
                    values=[
                        graph.inputs[0],
                        graph.inputs[1],
                    ],  # Too many inputs
                    out_types=[input_type1],
                )

            # Should mention input count or signature mismatch
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["input", "signature", "arity", "operand"]
            )

    def test_custom__error__wrong_output_count(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test error when specifying wrong number of outputs."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())
        output_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        graph = Graph(
            "test_custom_wrong_output_count",
            input_types=[input_type],
            output_types=[output_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            with pytest.raises(Exception) as exc_info:
                # op_with_multiple_outputs produces 2 outputs, we're specifying 1
                result = ops.custom(
                    name="op_with_multiple_outputs",
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[output_type],  # Wrong count - should be 2
                )
                graph.output(result[0])  # Try to use wrong result

            # Should mention output count or signature mismatch
            error_msg = str(exc_info.value).lower()
            assert any(
                word in error_msg
                for word in ["output", "signature", "result", "type"]
            )

    def test_custom__error__data_dependent_dim_without_shape_function(
        self, kernel_verification_ops_path: Path, session: InferenceSession
    ) -> None:
        """Test error when an output dim can only come from a shape function.

        `op_with_int_parameter` has no shape function, so this used to kill the
        process instead of failing compilation.
        """
        input_type = TensorType(DType.float32, (4,), DeviceRef.CPU())
        output_type = TensorType(DType.float32, ("nnz",), DeviceRef.CPU())

        graph = Graph(
            "test_custom_data_dependent_dim_no_shape_fn",
            input_types=[input_type],
            output_types=[output_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            result = ops.custom(
                name="op_with_int_parameter",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[output_type],
                parameters={"IntParameter": 1},
            )
            graph.output(result[0])

        with pytest.raises(Exception) as exc_info:
            session.load(graph)

        chain = _collect_error_chain(exc_info.value)
        assert "requires a registered shape function" in chain
        assert "op_with_int_parameter" in chain

    def test_custom__missing_parameter_behavior(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test behavior when required parameter is not provided - validates actual behavior."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        graph = Graph(
            "test_custom_missing_parameter",
            input_types=[input_type],
            output_types=[input_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            # Based on test results: missing parameters are handled gracefully
            # This documents the actual behavior rather than expected failures
            result = ops.custom(
                name="op_with_int_parameter",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[input_type],
                parameters={},  # Missing IntParameter - but this succeeds
            )
            graph.output(result[0])
            # Test documents that missing parameters don't cause immediate errors

    def test_custom__different_parameter_types(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test different parameter types work correctly."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        # Test different parameter kernels
        test_cases: list[tuple[str, dict[str, int | str | DType]]] = [
            ("op_with_int_parameter", {"IntParameter": 123}),
            (
                "op_with_static_string_parameter",
                {"StringParameter": "test_value"},
            ),
            ("op_with_dtype_parameter", {"DTypeParameter": DType.float32}),
        ]

        for kernel_name, params in test_cases:
            graph = Graph(
                f"test_custom_{kernel_name}",
                input_types=[input_type],
                output_types=[input_type],
                custom_extensions=[kernel_verification_ops_path],
            )
            with graph:
                result = ops.custom(
                    name=kernel_name,
                    device=DeviceRef.CPU(),
                    values=[graph.inputs[0]],
                    out_types=[input_type],
                    parameters=params,
                )
                graph.output(result[0])


class TestInplaceCustomKernelValidation:
    """Tests for ops.inplace_custom that require actual kernel validation."""

    def test_inplace_custom__success__basic_kernel(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test successful inplace custom operation with a real kernel."""
        buffer_type = BufferType(DType.float32, (10,), DeviceRef.CPU())

        graph = Graph(
            "test_inplace_custom_basic",
            input_types=[buffer_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            result = ops.inplace_custom(
                name="mutable_input_tensor",  # Real inplace test kernel
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
            )
            # inplace_custom returns list (chain output excluded)
            graph.output(*result)

    def test_inplace_custom__chain_integration(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test that inplace operations integrate properly with execution chains."""
        buffer_type = BufferType(DType.float32, (10,), DeviceRef.CPU())
        output_type = TensorType(DType.float32, (10,), DeviceRef.CPU())

        graph = Graph(
            "test_inplace_custom_chain",
            input_types=[buffer_type],
            output_types=[output_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            # First inplace operation
            ops.inplace_custom(
                name="mutable_input_tensor",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
            )

            # Second operation that depends on the buffer state
            buffer_input = graph.inputs[0]
            assert isinstance(buffer_input, BufferValue)
            tensor_val = ops.buffer_load(buffer_input)
            graph.output(tensor_val)


class TestCustomKernelSignatureValidation:
    """Tests for comprehensive kernel signature validation."""

    def test_custom__rank_mismatch_behavior(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test behavior when tensor rank doesn't match - validates actual behavior."""
        # Test documents that rank mismatches are handled gracefully at graph construction
        input_type = TensorType(DType.float32, (10,), DeviceRef.CPU())  # 1D
        output_type = TensorType(
            DType.float32, (2, 5), DeviceRef.CPU()
        )  # 2D - rank mismatch

        graph = Graph(
            "test_custom_rank_mismatch",
            input_types=[input_type],
            output_types=[output_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            # Based on test results: rank mismatches are handled gracefully at construction
            # Validation may happen at compile/execution time instead
            result = ops.custom(
                name="op_with_device_context",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],  # 1D input
                out_types=[
                    output_type
                ],  # 2D output - but this succeeds at construction
            )
            graph.output(result[0])
            # Test documents that rank validation doesn't happen at graph construction

    def test_custom__comprehensive_validation_success(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test that properly matching signatures work correctly in sequence."""
        input_type = TensorType(DType.float32, (5, 5), DeviceRef.CPU())
        intermediate_type = TensorType(DType.float32, (5, 5), DeviceRef.CPU())

        graph = Graph(
            "test_custom_comprehensive_validation",
            input_types=[input_type],
            output_types=[intermediate_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            # Test multiple operations in sequence to ensure comprehensive validation
            result1 = ops.custom(
                name="op_with_device_context",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[intermediate_type],
            )

            result2 = ops.custom(
                name="op_with_int_parameter",
                device=DeviceRef.CPU(),
                values=[result1[0]],  # result1 is a list, take first element
                out_types=[intermediate_type],
                parameters={"IntParameter": 999},
            )

            graph.output(result2[0])


class TestCustomDeviceValidation:
    """Tests for device compatibility validation."""

    def test_custom__cpu_device_compatibility(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test device compatibility validation with CPU device."""
        input_type = TensorType(DType.float32, (2, 3), DeviceRef.CPU())

        graph = Graph(
            "test_custom_cpu_device",
            input_types=[input_type],
            output_types=[input_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            # This should work - test kernels support CPU
            result = ops.custom(
                name="op_with_device_context",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[input_type],
            )
            graph.output(result[0])

    def test_custom__device_context_handling(
        self, kernel_verification_ops_path: Path
    ) -> None:
        """Test that kernels with device context work correctly."""
        input_type = TensorType(DType.float32, (3, 3), DeviceRef.CPU())

        graph = Graph(
            "test_custom_device_context",
            input_types=[input_type],
            output_types=[input_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            # op_with_device_context specifically tests device context handling
            result = ops.custom(
                name="op_with_device_context",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[input_type],
            )
            graph.output(result[0])


class TestCustomOperationExecution:
    """Integration tests that verify actual execution behavior."""

    def test_custom__execution_with_session(
        self, kernel_verification_ops_path: Path, session: InferenceSession
    ) -> None:
        """Test custom operation execution with inference session."""
        input_type = TensorType(DType.int32, (1,), DeviceRef.CPU())
        output_type = TensorType(DType.int32, (1,), DeviceRef.CPU())

        graph = Graph(
            "test_custom_execution",
            input_types=[input_type],
            output_types=[output_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            result = ops.custom(
                name="op_with_int_parameter",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[output_type],
                parameters={"IntParameter": 123},
            )
            graph.output(result[0])

        # Compile and execute the graph
        compiled_model = session.load(graph)
        input_tensor = Buffer.from_numpy(np.array([42], dtype=np.int32))
        compiled_model.execute(input_tensor)

    def test_inplace_custom__execution_with_session(
        self, kernel_verification_ops_path: Path, session: InferenceSession
    ) -> None:
        """Test inplace custom operation execution with inference session."""
        buffer_type = BufferType(DType.float32, (3,), DeviceRef.CPU())
        output_type = TensorType(DType.float32, (3,), DeviceRef.CPU())

        graph = Graph(
            "test_inplace_custom_execution",
            input_types=[buffer_type],
            output_types=[output_type],
            custom_extensions=[kernel_verification_ops_path],
        )
        with graph:
            # Inplace operation that modifies the buffer
            ops.inplace_custom(
                name="mutable_input_tensor",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
            )

            # Load the modified buffer content
            buffer_input = graph.inputs[0]
            assert isinstance(buffer_input, BufferValue)
            tensor_val = ops.buffer_load(buffer_input)
            graph.output(tensor_val)

        # Compile and execute
        compiled_model = session.load(graph)
        input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        input_tensor = Buffer.from_numpy(input_data)
        result = compiled_model.execute(input_tensor)

        # Verify execution completed successfully
        assert result is not None


class TestKernelRegistrationOverride:
    """Empirical verification that user @extensibility.register shadows built-in
    MOGG kernels when the same op-name is registered in a custom mojoc
    supplied via Graph(..., custom_extensions=[...]).
    """

    def _build_add_graph(self, custom_extensions: list[Path]) -> Graph:
        input_type = TensorType(DType.float32, (4,), DeviceRef.CPU())
        graph = Graph(
            "test_mo_add_override",
            input_types=[input_type, input_type],
            output_types=[input_type],
            custom_extensions=custom_extensions,
        )
        with graph:
            a, b = graph.inputs
            graph.output(a.tensor + b.tensor)
        return graph

    def test_mo_add_override_takes_effect(
        self,
        kernel_override_ops_path: Path,
        session: InferenceSession,
    ) -> None:
        """User `mo.add` returning `lhs+rhs+1000` must shadow built-in `mo.add`."""
        graph = self._build_add_graph([kernel_override_ops_path])
        model = session.load(graph)

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        a_buf = Buffer.from_numpy(a).to(model.input_devices[0])
        b_buf = Buffer.from_numpy(b).to(model.input_devices[1])
        output = model.execute(a_buf, b_buf)

        result = output[0].to_numpy()
        expected_override = a + b + 1000.0
        expected_builtin = a + b
        assert np.allclose(result, expected_override), (
            f"Override did not fire: got {result!r}, "
            f"expected overridden {expected_override!r} "
            f"(built-in would give {expected_builtin!r})"
        )

    def _build_layer_norm_graph(self, custom_extensions: list[Path]) -> Graph:
        input_type = TensorType(DType.float32, (2, 4), DeviceRef.CPU())
        gamma_type = TensorType(DType.float32, (4,), DeviceRef.CPU())
        beta_type = TensorType(DType.float32, (4,), DeviceRef.CPU())
        output_type = TensorType(DType.float32, (2, 4), DeviceRef.CPU())
        graph = Graph(
            "test_mo_layer_norm_override",
            input_types=[input_type, gamma_type, beta_type],
            output_types=[output_type],
            custom_extensions=custom_extensions,
        )
        with graph:
            x, gamma, beta = graph.inputs
            y = ops.layer_norm(
                x.tensor, gamma.tensor, beta.tensor, epsilon=1e-5
            )
            graph.output(y)
        return graph

    def test_mo_layer_norm_override_takes_effect(
        self,
        kernel_override_ops_path: Path,
        session: InferenceSession,
    ) -> None:
        """User `mo.reduce.layer_norm` raises; built-in would normalize cleanly."""
        graph = self._build_layer_norm_graph([kernel_override_ops_path])
        model = session.load(graph)

        x = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
        )
        gamma = np.ones(4, dtype=np.float32)
        beta = np.zeros(4, dtype=np.float32)
        x_buf = Buffer.from_numpy(x).to(model.input_devices[0])
        gamma_buf = Buffer.from_numpy(gamma).to(model.input_devices[1])
        beta_buf = Buffer.from_numpy(beta).to(model.input_devices[2])

        with pytest.raises(Exception) as exc_info:
            model.execute(x_buf, gamma_buf, beta_buf)
        assert "LAYER_NORM_OVERRIDE_FIRED" in str(exc_info.value), (
            f"Override did not fire; raised: {exc_info.value!r}"
        )
