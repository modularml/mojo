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

from max.graph import TensorValue, ops
from max.nn.attention import AttentionWithRope
from max.nn.conv import Conv2d
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import MLP, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import TransformerBlock
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from .llava.llava_decoder import Transformer
from .llava.llava_projector import LlavaMultiModalConnector
from .model_config import PixtralConfig
from .vision_encoder.attention import Attention as VisionEncoderAttention
from .vision_encoder.rotary_embedding_2d import RotaryEmbedding2D
from .vision_encoder.transformer import MLP as VisionEncoderMLP
from .vision_encoder.transformer import Transformer as VisionEncoderTransformer
from .vision_encoder.transformer import TransformerBlock as VisionEncoderBlock


class PixtralVision(Module):
    """Vision encoder + multi-modal projector for a ragged batch of images.

    Accepts pre-extracted patches from all images concatenated into a single
    ragged tensor, along with a block-diagonal attention mask and position IDs.
    """

    patch_conv: Conv2d
    layer_norm: RMSNorm
    patch_positional_embedding: RotaryEmbedding2D
    transformer: VisionEncoderTransformer
    multi_modal_projector: LlavaMultiModalConnector

    _vision_hidden_size: int
    _patch_dim: int

    def __init__(self, config: PixtralConfig) -> None:
        super().__init__()
        self._vision_hidden_size = config.vision_hidden_size
        self._patch_dim = (
            config.num_channels * config.patch_size * config.patch_size
        )
        # Conv2d stores weight in checkpoint shape [out_ch, in_ch, kh, kw].
        # We reshape to [out_ch, patch_dim] and matmul at forward time.
        self.patch_conv = Conv2d(
            permute=True,
            kernel_size=(config.patch_size, config.patch_size),
            in_channels=config.num_channels,
            out_channels=config.vision_hidden_size,
            stride=(config.patch_size, config.patch_size),
            has_bias=False,
            device=config.devices[0],
            dtype=config.dtype,
        )
        self.layer_norm = RMSNorm(
            config.vision_hidden_size,
            config.dtype,
            1e-5,
            multiply_before_cast=False,
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
                        dtype=config.dtype,
                        device=config.devices[0],
                    ),
                    feed_forward=VisionEncoderMLP(
                        hidden_size=config.vision_hidden_size,
                        intermediate_size=config.vision_intermediate_size,
                        dtype=config.dtype,
                        device=config.devices[0],
                    ),
                    attention_norm=RMSNorm(
                        config.vision_hidden_size,
                        config.dtype,
                        1e-5,
                        multiply_before_cast=False,
                    ),
                    ffn_norm=RMSNorm(
                        config.vision_hidden_size,
                        config.dtype,
                        1e-5,
                        multiply_before_cast=False,
                    ),
                )
                for _ in range(config.vision_num_hidden_layers)
            ],
            dtype=config.dtype,
        )
        self.multi_modal_projector = LlavaMultiModalConnector(
            hidden_size=config.hidden_size,
            vision_hidden_size=config.vision_hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
        )

    def __call__(
        self,
        pixel_patches: TensorValue,
        attention_mask: TensorValue,
        position_ids: TensorValue,
    ) -> TensorValue:
        """Process a ragged batch of pre-extracted image patches.

        Args:
            pixel_patches: Flattened patches, shape [total_patches, patch_dim].
            attention_mask: Block-diagonal mask, shape [1, 1, total_patches, total_patches].
            position_ids: 2D spatial position index, shape [total_patches].

        Returns:
            Image embeddings, shape [total_patches, hidden_size].
        """
        # Reshape Conv2d weight [out_ch, in_ch, kh, kw] -> [out_ch, patch_dim]
        # and project: patches @ weight^T -> [total_patches, hidden]
        weight = TensorValue(self.patch_conv.filter)
        weight_2d = ops.reshape(
            weight, [self._vision_hidden_size, self._patch_dim]
        )
        patches_cast = ops.cast(pixel_patches, weight_2d.dtype)
        patch_embeds = patches_cast @ ops.transpose(weight_2d, 0, 1)

        # Add batch dim: [1, total_patches, hidden]
        patch_embeds = ops.unsqueeze(patch_embeds, 0)

        # Pre-attention layer norm
        patch_embeds = self.layer_norm(patch_embeds)

        # 2D rotary positional embeddings
        position_embedding = self.patch_positional_embedding(
            patch_embeds, position_ids
        )

        # Transformer encoder
        encoder_output = self.transformer(
            patch_embeds=patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=position_embedding,
        )

        # Multi-modal projector
        projected = self.multi_modal_projector(encoder_output)

        # Squeeze batch dim: [1, total_patches, hidden] -> [total_patches, hidden]
        return ops.reshape(projected, [-1, projected.shape[-1]])


class PixtralLanguage(Module):
    """Language model with multimodal embedding merge."""

    language_model: Transformer

    def __init__(self, config: PixtralConfig) -> None:
        super().__init__()
        self.language_model = _build_language_model(config)

    def __call__(
        self,
        tokens: TensorValue,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        image_embeddings: TensorValue,
        image_token_indices: TensorValue,
    ) -> tuple[TensorValue, ...]:
        h = self.language_model.embed_tokens(tokens)

        h = merge_multimodal_embeddings(
            inputs_embeds=h,
            multimodal_embeddings=image_embeddings,
            image_token_indices=image_token_indices,
        )

        return self.language_model(
            embeds=h,
            kv_collection=kv_collection,
            return_n_logits=return_n_logits,
            input_row_offsets=input_row_offsets,
        )


def _build_language_model(config: PixtralConfig) -> Transformer:
    rope = RotaryEmbedding(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        theta=config.rope_theta,
        max_seq_len=config.max_seq_len,
        interleaved=False,
    )

    kv_params = config.kv_params
    assert isinstance(kv_params, KVCacheParams)

    layers = [
        TransformerBlock(
            attention=AttentionWithRope(
                rope=rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=kv_params,
                dtype=config.dtype,
                devices=config.devices,
                scale=config.attention_multiplier,
                stacked_qkv=False,
                has_bias=False,
            ),
            mlp=MLP(
                dtype=config.dtype,
                quantization_encoding=None,
                hidden_dim=config.hidden_size,
                feed_forward_length=config.feed_forward_length,
                devices=config.devices,
            ),
            attention_norm=RMSNorm(
                config.hidden_size,
                config.dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,
            ),
            mlp_norm=RMSNorm(
                config.hidden_size,
                config.dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,
            ),
        )
        for _ in range(config.num_hidden_layers)
    ]

    embedding_layer = Embedding(
        config.vocab_size,
        config.hidden_size,
        config.dtype,
        config.devices[0],
    )
    output = Linear(
        config.hidden_size,
        config.vocab_size,
        config.dtype,
        config.devices[0],
    )

    return Transformer(
        dim=config.hidden_size,
        n_heads=config.num_attention_heads,
        layers=layers,
        norm=RMSNorm(
            config.hidden_size,
            config.dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        ),
        output=output,
        embedding=embedding_layer,
        kv_params=kv_params,
        rope=rope,
        return_logits=config.return_logits,
    )
