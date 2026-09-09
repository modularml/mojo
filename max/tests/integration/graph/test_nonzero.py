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
"""Tests for ops.nonzero."""

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType, ops


def test_nonzero_1d(session: InferenceSession) -> None:
    """nonzero on a 1D CPU tensor returns correct row-major indices."""
    with Graph(
        "nonzero_1d",
        input_types=[TensorType(DType.int32, [5], device=DeviceRef.CPU())],
    ) as graph:
        result = ops.nonzero(graph.inputs[0].tensor, out_dim=2)
        graph.output(result)

    model = session.load(graph)
    # Input [0, 1, 0, 2, 0] has nonzero elements at indices 1 and 3.
    input_data = np.array([0, 1, 0, 2, 0], dtype=np.int32)
    output = model.execute(Buffer.from_dlpack(input_data))[0]
    assert isinstance(output, Buffer)
    np.testing.assert_array_equal(output.to_numpy(), np.array([[1], [3]]))


def test_nonzero_2d(session: InferenceSession) -> None:
    """nonzero on a 2D CPU tensor returns correct row-major indices."""
    with Graph(
        "nonzero_2d",
        input_types=[TensorType(DType.int32, [2, 2], device=DeviceRef.CPU())],
    ) as graph:
        result = ops.nonzero(graph.inputs[0].tensor, out_dim=2)
        graph.output(result)

    model = session.load(graph)
    # Input [[0, 1], [2, 0]] has nonzero elements at (0,1) and (1,0).
    input_data = np.array([[0, 1], [2, 0]], dtype=np.int32)
    output = model.execute(Buffer.from_dlpack(input_data))[0]
    assert isinstance(output, Buffer)
    np.testing.assert_array_equal(output.to_numpy(), np.array([[0, 1], [1, 0]]))


@pytest.mark.skipif(
    accelerator_count() == 0, reason="requires a GPU to test device check"
)
def test_nonzero_raises_on_gpu() -> None:
    """ops.nonzero raises ValueError at graph construction time on GPU input."""
    with pytest.raises(ValueError, match=r"ops\.nonzero"):
        with Graph(
            "nonzero_gpu",
            input_types=[TensorType(DType.int32, [5], device=DeviceRef.GPU())],
            strict_device_placement=DevicePlacementPolicy.Error,
        ):
            ops.nonzero(Graph.current.inputs[0].tensor, out_dim=2)
