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
"""Unit tests for Identity layer."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Identity
from test_common.modular_graph_test import (
    are_all_tensor_values,
    modular_graph_test,
)


def test_identity_passthrough() -> None:
    """Verify Identity layer passes through input unchanged."""
    identity = Identity()

    input_types = [
        TensorType(DType.float32, shape=(2, 3), device=DeviceRef.CPU())
    ]
    graph = Graph(
        "test_identity_passthrough", input_types=input_types, forward=identity
    )

    assert graph.output_types == input_types


def test_identity_execution(session: InferenceSession) -> None:
    """Verify Identity layer returns the exact same values when executed."""
    dtype = DType.float32

    with Graph(
        "identity_execution",
        input_types=[
            TensorType(dtype, ["batch", "dim"], device=DeviceRef.CPU())
        ],
    ) as graph:
        assert are_all_tensor_values(graph.inputs)
        (x,) = graph.inputs
        identity = Identity()
        graph.output(identity(x))

        @modular_graph_test(session, graph)
        def test_correctness(
            execute: Callable[[Sequence[Buffer]], Buffer],
            inputs: Sequence[Buffer],
            torch_inputs: Sequence[torch.Tensor],
        ) -> None:
            result = execute(inputs).to_numpy()
            expected = torch_inputs[0].numpy()

            # Identity should return exactly the same values
            np.testing.assert_array_equal(result, expected)
