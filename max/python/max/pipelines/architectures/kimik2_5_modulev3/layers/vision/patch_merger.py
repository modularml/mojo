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
from max.experimental.nn.norm.layer_norm import LayerNorm
from max.experimental.tensor import Tensor


class PatchMergerMLP(Module[[Tensor], Tensor]):
    """Two-layer MLP with LayerNorm that merges spatially adjacent patches.

    Expects input of shape ``(total_patches, N_k, mm_hidden_size)`` with
    ``N_k = merge_kernel_size[0] * merge_kernel_size[1]``. Applies layer
    normalization, reshapes to ``(total_patches, N_k * mm_hidden_size)``,
    then projects through a two-layer MLP with GELU activation. Math matches
    the HuggingFace reference (nvidia/Kimi-K2.5-NVFP4 modeling_kimi_k25.py
    PatchMergerMLP).
    """

    def __init__(
        self,
        mm_hidden_size: int,
        hidden_size: int,
        merge_kernel_size: tuple[int, int],
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        self.merge_kernel_size = merge_kernel_size
        self.eps = eps

        self.input_dim = mm_hidden_size * (
            merge_kernel_size[0] * merge_kernel_size[1]
        )

        self.pre_norm = LayerNorm(dim=mm_hidden_size, eps=eps)
        self.linear1 = Linear(
            in_dim=self.input_dim, out_dim=self.input_dim, bias=True
        )
        self.linear2 = Linear(
            in_dim=self.input_dim, out_dim=hidden_size, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (total_patches, N_k, mm_hidden_size)
        x = self.pre_norm(x)
        # Use self.input_dim explicitly instead of -1: the symbolic shape engine
        # cannot infer the collapsed dim when x.shape[0] is a dynamic dimension.
        x = x.reshape((x.shape[0], self.input_dim))
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x  # (total_patches, hidden_size)
