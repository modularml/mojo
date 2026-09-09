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

"""Gemma4 vision patch embedder for the ModuleV3 API."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)


class Gemma4VisionPatchEmbedder(Module[..., Tensor]):
    """Pixel [0,1]->[-1,1] normalize + linear projection + 2-D position
    embedding lookup. Port of the graph arch's Gemma4VisionPatchEmbedder.
    """

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        vision_cfg = config.vision_config
        assert vision_cfg is not None
        self.dtype = config.unquantized_dtype
        self.hidden_size = vision_cfg.hidden_size
        self.position_embedding_size = vision_cfg.position_embedding_size

        self.input_proj = Linear(
            3 * vision_cfg.patch_size**2, self.hidden_size, bias=False
        )
        self.position_embedding_table = Tensor.zeros(
            [2, self.position_embedding_size, self.hidden_size]
        )

    def forward(
        self, patches_flat: Tensor, pixel_position_ids: Tensor
    ) -> Tensor:
        patches = patches_flat.cast(self.dtype) * 2.0 - 1.0
        hidden = self.input_proj(patches)

        table_flat = self.position_embedding_table.reshape(
            (2 * self.position_embedding_size, self.hidden_size)
        )
        x_ids = pixel_position_ids[:, 0].cast(DType.int64)
        y_ids = (
            pixel_position_ids[:, 1].cast(DType.int64)
            + self.position_embedding_size
        )
        position_emb = F.gather(table_flat, x_ids, axis=0) + F.gather(
            table_flat, y_ids, axis=0
        )
        return hidden + position_emb
