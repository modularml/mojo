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
from max.experimental.nn.common_layers.attention import AttentionWithRope
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.kv_cache import KVCacheInputs, KVCacheParamInterface, KVCacheParams
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from ..llama3_modulev3.layers.transformer_block import LlamaTransformerBlock
from .llava.llava_decoder import Transformer
from .llava.llava_projector import LlavaMultiModalConnector
from .model_config import PixtralConfig
from .vision_encoder.attention import Attention as VisionEncoderAttention
from .vision_encoder.rotary_embedding_2d import RotaryEmbedding2D
from .vision_encoder.transformer import MLP as VisionEncoderMLP
from .vision_encoder.transformer import Transformer as VisionEncoderTransformer
from .vision_encoder.transformer import TransformerBlock as VisionEncoderBlock


class PixtralVision(Module[..., tuple[Tensor, ...]]):
    """Vision encoder + multi-modal projector for a ragged batch of images.

    Accepts pre-extracted patches from all images concatenated into a single
    ragged tensor, along with a block-diagonal attention mask and position IDs.
    This allows compiling the vision graph once and running it once per batch,
    regardless of the number of images.
    """

    patch_conv: Tensor
    _vision_hidden_size: int
    _patch_dim: int
    layer_norm: RMSNorm
    patch_positional_embedding: RotaryEmbedding2D
    transformer: VisionEncoderTransformer
    multi_modal_projector: LlavaMultiModalConnector

    def __init__(self, config: PixtralConfig) -> None:
        self._vision_hidden_size = config.vision_hidden_size
        self._patch_dim = (
            config.num_channels * config.patch_size * config.patch_size
        )
        # Weight must match checkpoint shape [out_ch, in_ch, kh, kw] since
        # compile() validates loaded weight shapes against the parameter.
        # Reshaped to [out_ch, patch_dim] in the graph at forward time.
        self.patch_conv = Tensor.zeros(
            [
                config.vision_hidden_size,
                config.num_channels,
                config.patch_size,
                config.patch_size,
            ]
        )
        self.layer_norm = RMSNorm(
            dim=config.vision_hidden_size,
            eps=1e-5,
        )
        self.patch_positional_embedding = RotaryEmbedding2D(
            dim=config.vision_hidden_size,
            n_heads=config.vision_num_attention_heads,
            theta=config.vision_rope_theta,
            max_patches_per_side=config.image_size // config.patch_size,
        )
        self.transformer = VisionEncoderTransformer(
            n_heads=config.vision_num_attention_heads,
            layers=[
                VisionEncoderBlock(
                    attention=VisionEncoderAttention(
                        n_heads=config.vision_num_attention_heads,
                        dim=config.vision_hidden_size,
                        head_dim=config.vision_head_dim,
                    ),
                    feed_forward=VisionEncoderMLP(
                        hidden_size=config.vision_hidden_size,
                        intermediate_size=config.vision_intermediate_size,
                    ),
                    attention_norm=RMSNorm(
                        dim=config.vision_hidden_size,
                        eps=1e-5,
                    ),
                    ffn_norm=RMSNorm(
                        dim=config.vision_hidden_size,
                        eps=1e-5,
                    ),
                )
                for i in range(config.vision_num_hidden_layers)
            ],
            dtype=config.dtype,
        )
        self.multi_modal_projector = LlavaMultiModalConnector(
            hidden_size=config.hidden_size,
            vision_hidden_size=config.vision_hidden_size,
        )

    def forward(
        self,
        pixel_patches: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, ...]:
        """Process a ragged batch of pre-extracted image patches.

        Args:
            pixel_patches: Flattened patches from all images,
                shape ``[total_patches, patch_dim]``.
            attention_mask: Block-diagonal mask ensuring patches from different
                images do not attend to each other,
                shape ``[1, 1, total_patches, total_patches]``.
            position_ids: 2D spatial position index for each patch,
                shape ``[total_patches]``.
        """
        # Reshape Conv2d weight [out_ch, in_ch, kh, kw] -> [out_ch, patch_dim]
        # and project: patches @ weight^T -> [total_patches, hidden]
        weight_2d = ops.reshape(
            TensorValue(self.patch_conv),
            [self._vision_hidden_size, self._patch_dim],
        )
        patches_cast = ops.cast(TensorValue(pixel_patches), weight_2d.dtype)
        patch_embeds = Tensor.from_graph_value(
            patches_cast @ ops.transpose(weight_2d, 0, 1)
        )

        # Add batch dim: [1, total_patches, hidden]
        patch_embeds = F.unsqueeze(patch_embeds, 0)

        # Pre-attention layer norm
        patch_embeds = self.layer_norm(patch_embeds)

        # 2D rotary positional embeddings
        position_embedding = self.patch_positional_embedding(
            patch_embeds, TensorValue(position_ids)
        )

        # Transformer encoder
        encoder_output = self.transformer(
            patch_embeds, attention_mask, position_embedding
        )

        # Multi-modal projector
        projected = self.multi_modal_projector(encoder_output)

        # Squeeze batch dim: [1, total_patches, hidden] -> [total_patches, hidden]
        return (F.reshape(projected, [-1, projected.shape[-1]]),)


class PixtralLanguage(Module[..., tuple[Tensor, ...]]):
    """Language model with multimodal embedding merge for batched inference."""

    language_model: Transformer
    image_token_index: int
    kv_params: KVCacheParamInterface | None

    def __init__(self, config: PixtralConfig) -> None:
        self.language_model = self._build_language_model(config)
        self.image_token_index = config.image_token_index
        self.kv_params = config.kv_params

    def forward(
        self,
        input_ids: Tensor,
        input_row_offsets: Tensor,
        return_n_logits: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        assert self.kv_params is not None
        kv_inputs = iter(x._graph_value for x in variadic_args)
        symbolic_inputs = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(symbolic_inputs, KVCacheInputs)
        kv_collections = symbolic_inputs.inputs

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        # Merge image embeddings at pre-computed token positions
        merged_value = merge_multimodal_embeddings(
            inputs_embeds=TensorValue(inputs_embeds),
            multimodal_embeddings=TensorValue(image_embeddings),
            image_token_indices=TensorValue(image_token_indices),
        )
        merged = Tensor.from_graph_value(merged_value)

        return self.language_model(
            merged, kv_collections[0], return_n_logits, input_row_offsets
        )

    @staticmethod
    def _build_language_model(config: PixtralConfig) -> Transformer:
        rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            interleaved=False,
        )

        kv_params = config.kv_params
        assert isinstance(kv_params, KVCacheParams)

        layers = [
            LlamaTransformerBlock(
                attention=AttentionWithRope(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=kv_params,
                    layer_idx=i,
                    scale=config.attention_multiplier,
                    stacked_qkv=False,
                    has_bias=False,
                ),
                mlp=MLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.feed_forward_length,
                ),
                input_layernorm=RMSNorm(
                    dim=config.hidden_size,
                    eps=config.rms_norm_eps,
                ),
                post_attention_layernorm=RMSNorm(
                    dim=config.hidden_size,
                    eps=config.rms_norm_eps,
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

        embedding_layer = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )
        output = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
        )

        return Transformer(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=RMSNorm(
                dim=config.hidden_size,
                eps=config.rms_norm_eps,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            return_logits=config.return_logits,
        )
