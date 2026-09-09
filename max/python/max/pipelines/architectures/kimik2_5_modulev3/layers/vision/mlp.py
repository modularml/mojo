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

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor


# TODO: (MODELS-1084) generalize this (non-gated) MLP layer and move it somewhere central
class MLP2(Module[[Tensor], Tensor]):
    """Simple multi-layer perceptron composed of two :obj:`Linear` layers."""

    def __init__(
        self,
        dim: tuple[int, int, int],
        has_bias: bool = False,
    ) -> None:
        """Initializes the MLP2 layer.

        Args:
            dim: (in_dim, hidden_dim, out_dim) of the MLP2.
            has_bias: Whether to include bias terms in the linear layers.
        """
        super().__init__()
        self.up_proj = Linear(in_dim=dim[0], out_dim=dim[1], bias=has_bias)
        self.down_proj = Linear(in_dim=dim[1], out_dim=dim[2], bias=has_bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up_proj(x)
        # The vision encoder uses the tanh approximation of GELU.
        x = F.gelu(x, approximate="tanh")
        return self.down_proj(x)
