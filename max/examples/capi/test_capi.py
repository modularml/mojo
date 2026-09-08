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

"""Simple MAX Graph example that adds two vectors."""

import os

from max import engine
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType


def build_graph() -> None:
    # Build the graph for the accelerator
    device = Accelerator()

    # Input tensors are expected on the accelerator. `vector_width` is a
    # symbolic dimension allowing for dynamic shapes on the vector inputs.
    input_type = TensorType(
        dtype=DType.float32,
        shape=("vector_width",),
        device=DeviceRef.from_device(device),
    )

    # We'll just do simple one-operation vector addition for our graph
    with Graph("vector_add", input_types=(input_type, input_type)) as graph:
        vector1, vector2 = graph.inputs[0].tensor, graph.inputs[1].tensor

        output = vector1 + vector2  # Same as ops.add()

        graph.output(output)

    # Compile the graph for the accelerator
    session = engine.InferenceSession(devices=[device])
    compiled = session.compile(graph)
    model = session.init(compiled)

    # Save the graph to a MEF file
    model._export_mef("graph.mef")


def test_capi() -> None:
    build_graph()

    path = os.environ["GRAPH_EXECUTOR"]
    os.execv(path, [path])


if __name__ == "__main__":
    test_capi()
