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
# DOC: max/develop/graph.mdx

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops

# Step 1: Define the graph structure
cpu = CPU()
input_type = TensorType(DType.float32, shape=[5], device=cpu)

with Graph("relu_graph", input_types=[input_type]) as graph:
    x = graph.inputs[0]
    y = ops.relu(x)
    graph.output(y)

# Step 2: Compile the graph
session = InferenceSession(devices=[cpu])
compiled = session.compile(graph)
model = session.init(compiled)

# Step 3: Execute with data
input_data = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
result = model.execute(input_data)
print(np.from_dlpack(result[0]))
