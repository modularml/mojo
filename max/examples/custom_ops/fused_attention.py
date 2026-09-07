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
from max.engine.api import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def main() -> None:
    mojo_kernels = Path(__file__).parent / "kernels"

    dtype = DType.float32

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()
    if accelerator_count() == 0:
        N = 8
        D = 8
        BD = 4
        BN = 4
    else:
        N = 32
        D = 32
        BD = 8
        BN = 16

    with Graph(
        "fused_attention",
        input_types=[
            TensorType(
                dtype, shape=[N, D], device=DeviceRef.from_device(device)
            ),
            TensorType(
                dtype, shape=[N, D], device=DeviceRef.from_device(device)
            ),
            TensorType(
                dtype, shape=[N, D], device=DeviceRef.from_device(device)
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        q, k, v, *_ = graph.inputs
        results = ops.custom(
            name="modular_ops::fused_attention_custom",
            device=DeviceRef.from_device(device),
            parameters={"BD": BD, "BN": BN},
            values=[q, k, v],
            out_types=[
                TensorType(
                    dtype, shape=[N, D], device=DeviceRef.from_device(device)
                )
            ],
        )
        graph.output(*results)

    # Set up an inference session for running the graph.
    session = InferenceSession(devices=[device])

    # Compile the graph.
    compiled = session.compile(graph)
    model = session.init(compiled)

    np.random.seed(123)
    Q = Buffer.from_numpy(np.random.randn(N, D).astype("f")).to(device)
    K = Buffer.from_numpy(np.random.randn(N, D).astype("f")).to(device)
    V = Buffer.from_numpy(np.random.randn(N, D).astype("f")).to(device)

    output = model.execute(Q, K, V)
    print(output)


if __name__ == "__main__":
    main()
