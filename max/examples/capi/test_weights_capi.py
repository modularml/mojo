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

"""Build a graph with external weights and run the C weights example."""

import os

import numpy as np
from max import engine
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def build_graph() -> None:
    """Build a graph that multiplies an input by an external weight."""
    input_type = TensorType(
        dtype=DType.float32, shape=(4,), device=DeviceRef.CPU()
    )
    weight_type = TensorType(
        dtype=DType.float32, shape=(4,), device=DeviceRef.CPU()
    )

    with Graph("weighted_multiply", input_types=(input_type,)) as graph:
        inp = graph.inputs[0].tensor
        weight = ops.constant_external("weight", weight_type)
        graph.output(inp * weight)

    # Dummy weights for compilation — real values are provided at runtime
    # via M_newWeightsRegistry in the C program.
    dummy_weights = {"weight": np.zeros(4, dtype=np.float32)}

    session = engine.InferenceSession(devices=[CPU()])
    compiled = session.compile(graph)
    model = session.init(compiled, weights_registry=dummy_weights)
    model._export_mef("weights_graph.mef")


def test_weights_capi() -> None:
    build_graph()

    path = os.environ["WEIGHTS_EXECUTOR"]
    os.execv(path, [path])


if __name__ == "__main__":
    test_weights_capi()
