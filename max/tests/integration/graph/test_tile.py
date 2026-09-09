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
"""Tests for ops.tile."""

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType, ops


def test_tile_1d(session: InferenceSession) -> None:
    """tile on a 1D CPU tensor repeats elements correctly."""
    with Graph(
        "tile_1d",
        input_types=[TensorType(DType.float32, [3], device=DeviceRef.CPU())],
    ) as graph:
        out = ops.tile(graph.inputs[0].tensor, repeats=[2])
        graph.output(out)

    model = session.load(graph)
    input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    output = model.execute(Buffer.from_dlpack(input_data))[0]
    assert isinstance(output, Buffer)
    np.testing.assert_array_equal(
        output.to_numpy(), np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    )


def test_tile_2d(session: InferenceSession) -> None:
    """tile on a 2D CPU tensor tiles along both dimensions correctly."""
    with Graph(
        "tile_2d",
        input_types=[TensorType(DType.float32, [2, 2], device=DeviceRef.CPU())],
    ) as graph:
        out = ops.tile(graph.inputs[0].tensor, repeats=[2, 3])
        graph.output(out)

    model = session.load(graph)
    input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    output = model.execute(Buffer.from_dlpack(input_data))[0]
    assert isinstance(output, Buffer)
    expected = np.tile(input_data, (2, 3))
    np.testing.assert_array_equal(output.to_numpy(), expected)


@pytest.mark.skipif(
    accelerator_count() == 0, reason="requires a GPU to test device check"
)
def test_tile_raises_on_gpu() -> None:
    """ops.tile raises ValueError at graph construction time on GPU input."""
    with pytest.raises(ValueError, match=r"ops\.tile"):
        with Graph(
            "tile_gpu",
            input_types=[
                TensorType(DType.float32, [3], device=DeviceRef.GPU())
            ],
            strict_device_placement=DevicePlacementPolicy.Error,
        ):
            ops.tile(Graph.current.inputs[0].tensor, repeats=[2])
