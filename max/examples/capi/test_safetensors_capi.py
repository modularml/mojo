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

"""Build a graph with external weights and run the C safetensors example."""

import json
import os
import struct

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
    # from the safetensors file by the C program.
    dummy_weights = {"weight": np.zeros(4, dtype=np.float32)}

    session = engine.InferenceSession(devices=[CPU()])
    compiled = session.compile(graph)
    model = session.init(compiled, weights_registry=dummy_weights)
    model._export_mef("weights_graph.mef")


def write_safetensors() -> None:
    """Write a minimal Safetensors file containing the "weight" tensor."""
    weight = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    blob = weight.tobytes()
    header = {
        "weight": {
            "dtype": "F32",
            "shape": list(weight.shape),
            "data_offsets": [0, len(blob)],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    # Pad the header (8-byte length prefix + JSON) up to an 8-byte boundary.
    padding = (-(8 + len(header_bytes))) % 8
    header_bytes += b" " * padding
    with open("weights.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(blob)


def test_safetensors_capi() -> None:
    build_graph()
    write_safetensors()

    path = os.environ["SAFETENSORS_EXECUTOR"]
    os.execv(path, [path])


if __name__ == "__main__":
    test_safetensors_capi()
