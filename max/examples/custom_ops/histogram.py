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

from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

if __name__ == "__main__":
    mojo_kernels = Path(__file__).parent / "kernels"

    n = 2**20

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Configure our simple one-operation graph.
    graph = Graph(
        "histogram",
        # The custom Mojo operation is referenced by its string name, and we
        # need to provide inputs as a list as well as expected output types.
        forward=lambda x: (
            ops.custom(
                name="histogram",
                device=DeviceRef.from_device(device),
                values=[x],
                out_types=[
                    TensorType(
                        dtype=DType.int64,
                        shape=[256],
                        device=DeviceRef.from_device(device),
                    )
                ],
            )[0].tensor
        ),
        input_types=[
            TensorType(
                DType.uint8, shape=[n], device=DeviceRef.from_device(device)
            ),
        ],
        custom_extensions=[mojo_kernels],
    )

    # Set up an inference session for running the graph.
    session = InferenceSession(devices=[device])

    # Compile the graph.
    compiled = session.compile(graph)
    model = session.init(compiled)

    # Fill an input with random values.
    x_values = np.random.randint(0, 256, size=n, dtype=np.uint8)

    # Create a driver tensor from this, and move it to the accelerator.
    x = Buffer.from_numpy(x_values).to(device)

    # Perform the calculation on the target device.
    model_result = model.execute(x)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(model_result, Buffer)

    print("Graph result:")
    result = model_result.to_numpy()
    print(result)
    print()

    print("Expected result:")
    expected = np.histogram(x_values, bins=256, range=(0, 256))[0]
    print(expected)

    assert all(result == expected), "Result does not match expected"
