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
"""Tests for matmul operations."""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from test_common.modular_graph_test import ACCURACY_ATOL, ACCURACY_RTOL


def test_matmul_dense(session: InferenceSession) -> None:
    in_features = 4
    out_features = 7

    input_type = TensorType(
        DType.float32, ["batch", in_features], device=DeviceRef.CPU()
    )

    with Graph("matmul_dense", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        # Create constant weights and bias (using random values for testing)
        weights = ops.constant(
            np.random.randn(in_features, out_features).astype(np.float32),
            DType.float32,
            device=DeviceRef.CPU(),
        )
        bias = ops.constant(
            np.zeros(out_features, dtype=np.float32),
            DType.float32,
            device=DeviceRef.CPU(),
        )
        # Matmul + bias
        matmul_result = ops.matmul(x, weights)
        result = matmul_result + bias
        graph.output(result)

    model = session.load(graph)

    # Extract the actual weights and bias from the graph for comparison
    # We need to get the constant values that were used
    weights_np = np.random.randn(in_features, out_features).astype(np.float32)
    bias_np = np.zeros(out_features, dtype=np.float32)

    # Rebuild graph with known weights for testing
    with Graph("matmul_dense_test", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        weights = ops.constant(
            weights_np, DType.float32, device=DeviceRef.CPU()
        )
        bias = ops.constant(bias_np, DType.float32, device=DeviceRef.CPU())
        matmul_result = ops.matmul(x, weights)
        result = matmul_result + bias
        graph.output(result)

    model = session.load(graph)

    batch_sizes = st.integers(min_value=1, max_value=32)

    @settings(max_examples=20, deadline=None)
    @given(batch_size=batch_sizes)
    def check_matmul_dense(batch_size: int) -> None:
        # Generate random input data
        input_data = torch.randn((batch_size, in_features), dtype=torch.float32)

        # Run through MAX
        max_input = Buffer.from_dlpack(input_data).to(model.input_devices[0])
        max_result = model(max_input)[0]
        assert isinstance(max_result, Buffer)
        max_result_np = max_result.to_numpy()

        # Compute expected result with torch/numpy
        expected = input_data.numpy() @ weights_np + bias_np

        np.testing.assert_allclose(
            max_result_np,
            expected,
            rtol=ACCURACY_RTOL,
            atol=ACCURACY_ATOL,
        )

    check_matmul_dense()


def test_matmul_transpose(session: InferenceSession) -> None:
    in_features = 4
    out_features = 7

    input_type = TensorType(
        DType.float32, ["batch", in_features], device=DeviceRef.CPU()
    )

    # Create known weights for testing (shape [out_features, in_features] before transpose)
    weights_np = np.random.randn(out_features, in_features).astype(np.float32)

    with Graph("matmul_transpose", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        # Create constant weights in [7, 4] shape, then transpose to [4, 7]
        weights = ops.constant(
            weights_np, DType.float32, device=DeviceRef.CPU()
        )
        weights_transposed = ops.transpose(weights, 0, 1)
        result = ops.matmul(x, weights_transposed)
        graph.output(result)

    model = session.load(graph)

    batch_sizes = st.integers(min_value=1, max_value=32)

    @settings(max_examples=20, deadline=None)
    @given(batch_size=batch_sizes)
    def check_matmul_transpose(batch_size: int) -> None:
        # Generate random input data
        input_data = torch.randn((batch_size, in_features), dtype=torch.float32)

        # Run through MAX
        max_input = Buffer.from_dlpack(input_data).to(model.input_devices[0])
        max_result = model(max_input)[0]
        assert isinstance(max_result, Buffer)
        max_result_np = max_result.to_numpy()

        # Compute expected result: input @ weights.T (weights is [7, 4], so weights.T is [4, 7])
        expected = input_data.numpy() @ weights_np.T

        np.testing.assert_allclose(
            max_result_np,
            expected,
            rtol=ACCURACY_RTOL,
            atol=ACCURACY_ATOL,
        )

    check_matmul_transpose()
