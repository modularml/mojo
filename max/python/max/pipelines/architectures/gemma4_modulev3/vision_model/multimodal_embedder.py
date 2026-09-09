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

"""Gemma4 multimodal embedder for the ModuleV3 API."""

from __future__ import annotations

from max.dtype import DType
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor

from ..layers.rms_norm import Gemma4RMSNorm


class Gemma4MultimodalEmbedder(Module[[Tensor], Tensor]):
    """Weightless RMSNorm + projection into LM hidden space.

    Port of the graph arch's Gemma4MultimodalEmbedder.
    """

    def __init__(
        self,
        multimodal_hidden_size: int,
        text_hidden_size: int,
        dtype: DType,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.embedding_projection = Linear(
            multimodal_hidden_size, text_hidden_size, bias=False
        )
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            multimodal_hidden_size, eps=eps, with_weight=False
        )

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed.cast(self.dtype))
