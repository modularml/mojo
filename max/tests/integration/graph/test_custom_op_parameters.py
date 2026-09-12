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
"""Test the behavior of custom op parameters."""

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


def test_custom_op_with_int_parameter(
    kernel_verification_ops_path: Path,
    session: InferenceSession,
    capfd: pytest.CaptureFixture,  # type: ignore[type-arg]
) -> None:
    expected_int = 42

    unimportant_for_test_tensor_type = TensorType(
        DType.int32, [1], device=DeviceRef.CPU()
    )
    unimportant_for_test_tensor = Buffer.from_numpy(np.full([1], 1, np.int32))
    graph = Graph(
        "test_op_with_int_parameter",
        input_types=[unimportant_for_test_tensor_type],
        output_types=[unimportant_for_test_tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        graph.output(
            ops.custom(
                "op_with_int_parameter",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[unimportant_for_test_tensor_type],
                parameters={"IntParameter": expected_int},
            )[0]
        )

    compiled_model = session.load(graph)
    compiled_model.execute(unimportant_for_test_tensor)

    execution_output = capfd.readouterr()
    assert str(expected_int) in execution_output.out


def test_custom_op_with_dtype_parameter(
    kernel_verification_ops_path: Path,
    session: InferenceSession,
    capfd: pytest.CaptureFixture,  # type: ignore[type-arg]
) -> None:
    expected_dtype = DType.int32
    expected_dtype_str = "int32"

    unimportant_for_test_tensor_type = TensorType(
        DType.int32, [1], device=DeviceRef.CPU()
    )
    unimportant_for_test_tensor = Buffer.from_numpy(np.full([1], 1, np.int32))
    graph = Graph(
        "test_op_with_dtype_parameter",
        input_types=[unimportant_for_test_tensor_type],
        output_types=[unimportant_for_test_tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        graph.output(
            ops.custom(
                "op_with_dtype_parameter",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[unimportant_for_test_tensor_type],
                parameters={"DTypeParameter": expected_dtype},
            )[0]
        )

    compiled_model = session.load(graph)
    compiled_model.execute(unimportant_for_test_tensor)

    execution_output = capfd.readouterr()
    assert expected_dtype_str in execution_output.out


def test_custom_op_with_static_string_parameter(
    kernel_verification_ops_path: Path,
    session: InferenceSession,
    capfd: pytest.CaptureFixture,  # type: ignore[type-arg]
) -> None:
    expected_string = "Socrates is a man"

    unimportant_for_test_tensor_type = TensorType(
        DType.int32, [1], device=DeviceRef.CPU()
    )
    unimportant_for_test_tensor = Buffer.from_numpy(np.full([1], 1, np.int32))
    graph = Graph(
        "test_op_with_static_string_parameter",
        input_types=[unimportant_for_test_tensor_type],
        output_types=[unimportant_for_test_tensor_type],
        custom_extensions=[kernel_verification_ops_path],
    )
    with graph:
        graph.output(
            ops.custom(
                "op_with_static_string_parameter",
                device=DeviceRef.CPU(),
                values=[graph.inputs[0]],
                out_types=[unimportant_for_test_tensor_type],
                parameters={"StringParameter": expected_string},
            )[0]
        )

    compiled_model = session.load(graph)
    compiled_model.execute(unimportant_for_test_tensor)

    execution_output = capfd.readouterr()
    assert expected_string in execution_output.out
