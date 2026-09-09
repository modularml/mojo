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
"""FLUX.2 transformer (top-level) in the experimental.nn Module style."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.pipelines.architectures.flux2.model_config import Flux2Config

from .layers import (
    AdaLayerNormContinuous,
    Flux2Modulation,
    Flux2PosEmbed,
    Flux2TimestepGuidanceEmbeddings,
)
from .transformer_blocks import (
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)


class Flux2Transformer2DModel(Module[..., Tensor]):
    """FLUX.2 transformer (8 dual-stream + 48 single-stream blocks).

    Mirrors :class:`max.pipelines.architectures.flux2.flux2.Flux2Transformer2DModel`
    single-device path: the legacy ``Allreduce`` / ``Signals`` plumbing
    and per-device stream replication are out of scope for the
    single-GPU + BF16 first port.
    """

    def __init__(self, config: Flux2Config) -> None:
        patch_size = config.patch_size
        in_channels = config.in_channels
        out_channels = config.out_channels
        num_layers = config.num_layers
        num_single_layers = config.num_single_layers
        attention_head_dim = config.attention_head_dim
        num_attention_heads = config.num_attention_heads
        joint_attention_dim = config.joint_attention_dim
        timestep_guidance_channels = config.timestep_guidance_channels
        mlp_ratio = config.mlp_ratio
        axes_dims_rope = config.axes_dims_rope
        rope_theta = config.rope_theta
        eps = config.eps

        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim

        self.pos_embed = Flux2PosEmbed(
            theta=rope_theta, axes_dim=axes_dims_rope
        )
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=getattr(config, "guidance_embeds", True),
        )
        self.double_stream_modulation_img = Flux2Modulation(
            self.inner_dim, mod_param_sets=2, bias=False
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            self.inner_dim, mod_param_sets=2, bias=False
        )
        self.single_stream_modulation = Flux2Modulation(
            self.inner_dim, mod_param_sets=1, bias=False
        )
        self.x_embedder = Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = Linear(
            joint_attention_dim, self.inner_dim, bias=False
        )
        self.transformer_blocks = ModuleList[Flux2TransformerBlock](
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = ModuleList[
            Flux2SingleTransformerBlock
        ](
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            elementwise_affine=False,
            eps=eps,
            bias=False,
        )
        self.proj_out = Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Tensor,
        txt_ids: Tensor,
        guidance: Tensor,
    ) -> Tensor:
        """Run the FLUX.2 transformer.

        Args:
            hidden_states: Image latents ``[B, S_img, in_channels]``.
            encoder_hidden_states: Text embeddings
                ``[B, S_txt, joint_attention_dim]``.
            timestep: Per-batch sigma values ``[B]`` (float32).
            img_ids: Image position IDs ``[B, S_img, 4]`` or
                ``[S_img, 4]``.
            txt_ids: Text position IDs ``[B, S_txt, 4]`` or
                ``[S_txt, 4]``.
            guidance: Per-batch guidance scales ``[B]`` (float32).

        Returns:
            Tensor ``[B, S_img, patch_size^2 * out_channels]`` in the
            transformer dtype.
        """
        if len(img_ids.shape) == 3:
            img_ids = img_ids[0]
        if len(txt_ids.shape) == 3:
            txt_ids = txt_ids[0]

        num_txt_tokens = encoder_hidden_states.shape[1]
        # Multiply by 1000 *before* the cast; bf16 rounding on the raw
        # sigma can shift the scaled result by up to 4 ulps for small
        # sigmas, which is why the legacy executor also keeps the
        # multiply in float32 (cf. ``flux2/components/denoise_compute.py:92``).
        timestep = (timestep * 1000.0).cast(hidden_states.dtype)
        guidance = (guidance * 1000.0).cast(hidden_states.dtype)
        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img_tuple = self.double_stream_modulation_img(temb)
        double_stream_mod_txt_tuple = self.double_stream_modulation_txt(temb)
        single_stream_mod_tuple = self.single_stream_modulation(temb)
        double_stream_mod_img = (
            double_stream_mod_img_tuple[0],
            double_stream_mod_img_tuple[1],
        )
        double_stream_mod_txt = (
            double_stream_mod_txt_tuple[0],
            double_stream_mod_txt_tuple[1],
        )
        single_stream_mod = single_stream_mod_tuple[0]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        ids = F.concat([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = F.concat([encoder_hidden_states, hidden_states], axis=1)

        for single_block in self.single_transformer_blocks:
            hidden_states = single_block(
                hidden_states,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = hidden_states[:, num_txt_tokens:, :]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)
