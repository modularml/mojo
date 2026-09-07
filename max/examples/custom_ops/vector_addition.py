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

    vector_width = 10
    dtype = DType.float32

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    # Configure our simple one-operation graph.
    with Graph(
        "vector_addition",
        input_types=[
            TensorType(
                dtype,
                shape=[vector_width],
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=[vector_width],
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        lhs, rhs = graph.inputs
        output = ops.custom(
            name="vector_addition",
            device=DeviceRef.from_device(device),
            values=[lhs, rhs],
            out_types=[
                TensorType(
                    dtype=lhs.tensor.dtype,
                    shape=lhs.tensor.shape,
                    device=DeviceRef.from_device(device),
                )
            ],
        )[0].tensor
        graph.output(output)

    # Set up an inference session for running the graph.
    session = InferenceSession(
        devices=[device],
    )

    # Compile the graph.
    compiled = session.compile(graph)
    model = session.init(compiled)

    # Fill input matrices with random values.
    lhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)
    rhs_values = np.random.uniform(size=(vector_width)).astype(np.float32)

    # Create driver tensors from this, and move them to the accelerator.
    lhs_tensor = Buffer.from_numpy(lhs_values).to(device)
    rhs_tensor = Buffer.from_numpy(rhs_values).to(device)

    # Perform the calculation on the target device.
    result = model.execute(lhs_tensor, rhs_tensor)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Buffer)
    result = result.to(CPU())

    print("Left-hand-side values:")
    print(lhs_values)
    print()

    print("Right-hand-side values:")
    print(rhs_values)
    print()

    print("Graph result:")
    print(result.to_numpy())
    print()

    print("Expected result:")
    print(lhs_values + rhs_values)
