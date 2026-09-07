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

"""Build a vector-add graph, save it as MEF, and run the graph capture example."""

import os

from max import engine
from max.driver import Accelerator
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType


def test_graph_capture() -> None:
    device = Accelerator()

    input_type = TensorType(
        dtype=DType.float32,
        shape=(8,),
        device=DeviceRef.from_device(device),
    )

    with Graph("vector_add", input_types=(input_type, input_type)) as graph:
        vector1, vector2 = graph.inputs[0].tensor, graph.inputs[1].tensor
        graph.output(vector1 + vector2)

    session = engine.InferenceSession(devices=[device])
    compiled = session.compile(graph)
    model = session.init(compiled)
    model._export_mef("graph.mef")

    path = os.environ["GRAPH_EXECUTOR"]
    os.execv(path, [path])


if __name__ == "__main__":
    test_graph_capture()
