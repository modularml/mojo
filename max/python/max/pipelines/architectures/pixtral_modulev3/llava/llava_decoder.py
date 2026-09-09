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

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import ops
from max.nn.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.transformer import ReturnLogits

from ...llama3_modulev3.layers.transformer_block import LlamaTransformerBlock


class Transformer(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[LlamaTransformerBlock],
        norm: RMSNorm,
        output: Linear,
        embedding: Embedding,
        kv_params: KVCacheParams,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        embedding_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = ModuleList(layers)
        self.norm = norm
        self.lm_head = output
        self.embed_tokens = embedding
        self.kv_params = kv_params
        self.embedding_multiplier = embedding_multiplier
        self.return_logits = return_logits

    def forward(
        self,
        embeds: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        """Transformer model consisting of TransformerBlock layers.

        Args:
            embeds: embeddings of the sequence of text tokens and possibly images.
                shape = [batch_size, n_patches, hidden_dim]
            kv_collection: Paged KV cache values.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged batching.
        """
        h = embeds

        for idx, layer in enumerate(self.layers):
            h = layer(
                F.constant(idx, DType.uint32, device=h.device),
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        # Logits postprocessing (inline, replacing LogitsPostprocessMixin)
        last_token_indices = input_row_offsets[1:] - 1
        last_h = F.gather(h, last_token_indices, axis=0)
        last_logits = F.cast(self.lm_head(self.norm(last_h)), DType.float32)

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets_expanded = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(offsets_expanded, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = F.cast(self.lm_head(self.norm(last_tokens)), DType.float32)
            logit_offsets = F.cast(
                ops.range(
                    0,
                    last_indices.shape[0] + return_n_logits[0],
                    return_n_logits[0],
                    out_dim="logit_offsets",
                    device=h.device,
                    dtype=DType.int64,
                ),
                DType.int64,
            )
            ret_val += (logits, logit_offsets)
        elif self.return_logits == ReturnLogits.ALL:
            logits = F.cast(self.lm_head(self.norm(h)), DType.float32)
            ret_val += (
                logits,
                F.cast(input_row_offsets, input_row_offsets.dtype),
            )

        return ret_val
