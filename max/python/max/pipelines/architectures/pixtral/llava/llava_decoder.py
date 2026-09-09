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
from max.graph import DeviceRef, TensorValue, ops
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Layer, LayerList, Module
from max.nn.linear import Linear
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import ReturnLogits, TransformerBlock
from max.nn.transformer.transformer import LogitsPostprocessMixin


class Transformer(LogitsPostprocessMixin, Module):
    """Transformer model consisting for TransformerBlock layers.

    The differences between this transformer and the transformer in nn:

    - It takes as input the token embeddings rather than the token ids.
    - It skips the embedding generation (first step in nn.Transformer).

    TODO(AIPIPE-273): Once we have mo.if, we can update nn.Transformer
    to only generate embeddings if token ids are passed. That would
    eliminate the need for this class.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[TransformerBlock],
        norm: Layer,
        output: Linear,
        embedding: Embedding,
        kv_params: KVCacheParams,
        rope: RotaryEmbedding,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        embedding_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        self.lm_head = output
        self.embed_tokens = embedding
        self.kv_params = kv_params
        self.embedding_multiplier = embedding_multiplier
        self.rope = rope
        self.return_logits = return_logits

    def __call__(
        self,
        embeds: TensorValue,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Transformer model consisting of TransformerBlock layers.

        Args:
            embeds: embeddings of the sequence of text tokens and possibly images.
                shape = [batch_size, n_patches, hidden_dim]
            kv_cache_inputs: A tuple of 4 tensor values. In the case of paged attention,
                (blocks, cache_lengths, lookup_table, is_cache_empty). In the case of
                continuous attention, (blocks, cache_lengths, lookup_table,
                max_prompt_length, max_cache_length).
        """
        h = embeds

        freqs_cis = self.rope.freqs_cis
        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )

        return self._postprocess_logits(h, input_row_offsets, return_n_logits)
