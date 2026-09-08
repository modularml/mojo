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

# DOC: max/develop/modules.mdx

import numpy as np
import numpy.typing as npt
from max import nn
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops

# --- Define and compose modules ---


class FeedForward(nn.Module):
    """Two linear projections with SiLU activation."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, DType.float32, DeviceRef.CPU())
        self.fc2 = nn.Linear(hidden_dim, dim, DType.float32, DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.fc2(ops.silu(self.fc1(x)))


# --- Chain multiple modules ---


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            [FeedForward(dim, hidden_dim) for _ in range(n_layers)]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.layers(x)


# --- Write a module with explicit weights ---


class ScaledLinear(nn.Module):
    """Built-in Linear with a learnable output scale."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, DType.float32, DeviceRef.CPU())
        self.scale = Weight("scale", DType.float32, [out_dim], DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.linear(x) * self.scale


if __name__ == "__main__":
    dim = 8
    hidden_dim = 16
    n_layers = 2

    # --- Instantiate and load weights ---

    model = FeedForwardBlock(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers)

    # Build a random state dict matching the module's weight names
    rng = np.random.default_rng(42)

    def rand(*shape: int) -> npt.NDArray[np.float32]:
        return rng.standard_normal(shape).astype(np.float32)

    state_dict = {}
    for i in range(n_layers):
        state_dict[f"layers.{i}.fc1.weight"] = rand(hidden_dim, dim)
        state_dict[f"layers.{i}.fc2.weight"] = rand(dim, hidden_dim)

    model.load_state_dict(state_dict)

    # Inspect fully-qualified weight names
    print(model.state_dict().keys())

    # --- Turn a module into a graph ---

    graph = Graph(
        "my_model",
        model,
        input_types=[
            TensorType(DType.float32, shape=[1, dim], device=DeviceRef.CPU())
        ],
    )

    # --- Run the model locally ---

    session = InferenceSession(devices=[CPU()])
    compiled = session.compile(graph)
    runnable_model = session.init(compiled, weights_registry=model.state_dict())

    input_data = rng.standard_normal((1, dim)).astype(np.float32)
    result = runnable_model(input_data)

    print("result:", result[0].to_numpy())
