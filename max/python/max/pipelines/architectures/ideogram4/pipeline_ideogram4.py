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
"""Ideogram 4 flow-matching text-to-image diffusion pipeline.

Wires the Qwen3-VL text encoder (13-layer concat), the conditional and
unconditional ``Ideogram4Transformer2DModel`` DiTs (asymmetric dual-branch
CFG), and the FLUX.2-architecture VAE decoder. The denoise loop follows the
torch reference exactly: per-step ``z += v * delta`` with
``v = gw * pos_v + (1 - gw) * neg_v``.

Bring-up assumes ``batch_size == 1`` (a single prompt, no left padding).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.pipelines.context import PixelContext
from max.pipelines.diffusion.interface import (
    DiffusionPipeline,
    DiffusionPipelineOutput,
)
from max.profiler import traced

from ..qwen3_modulev3.text_encoder.model import Qwen3TextEncoderModel
from .denoise_step import Ideogram4CombineStep, Ideogram4PackStep
from .ideogram4 import Ideogram4Transformer2DModel
from .model import Ideogram4TransformerModel
from .model_config import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
)
from .text_encoder import Ideogram4TextEncoderModel
from .vae import Ideogram4VAEModel


@dataclass(kw_only=True)
class Ideogram4ModelInputs:
    """Device tensors + host metadata for one denoise run."""

    # Conditional (full packed sequence) branch.
    x_image: Tensor
    """Initial image-token noise ``(1, num_image, 128)``."""
    text_z_padding: Tensor
    """Zero latents over the text span ``(1, num_text, 128)``."""
    llm_features: Tensor
    """Full-sequence text features ``(1, total_seq, 53248)``."""
    position_ids: Tensor
    """Full-sequence 3D positions ``(1, total_seq, 3)`` int64."""
    indicator: Tensor
    """Full-sequence token-type indicator ``(1, total_seq)`` int64."""

    # Unconditional (image-only) branch.
    neg_llm_features: Tensor
    neg_position_ids: Tensor
    neg_indicator: Tensor

    # Host schedule + geometry.
    timesteps: npt.NDArray[np.float32]
    deltas: npt.NDArray[np.float32]
    guidance_scale: float
    num_text: int
    num_image: int
    grid_h: int
    grid_w: int

    # Fields read by the base ``DiffusionPipeline.execute`` wrapper.
    num_images_per_prompt: int
    height: int
    width: int
    num_inference_steps: int


class Ideogram4Pipeline(DiffusionPipeline):
    """Diffusion pipeline for Ideogram 4 text-to-image generation."""

    default_num_inference_steps = 20

    vae: Ideogram4VAEModel
    text_encoder: Qwen3TextEncoderModel
    transformer: Ideogram4TransformerModel
    unconditional_transformer: Ideogram4TransformerModel

    components = {
        "vae": Ideogram4VAEModel,
        "text_encoder": Ideogram4TextEncoderModel,
        "transformer": Ideogram4TransformerModel,
        "unconditional_transformer": Ideogram4TransformerModel,
    }

    @traced(message="Ideogram4Pipeline.init_remaining_components")
    def init_remaining_components(self) -> None:
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self._build_denoise_step()

    @traced(message="Ideogram4Pipeline.build_denoise_step")
    def _build_denoise_step(self) -> None:
        """Compile the per-step graphs: pack, cond DiT, uncond DiT, combine.

        Each transformer is compiled directly with its native single-``seq``
        contract (~1.3 s each); the concat (pack) and the velocity slice +
        CFG + Euler step (combine) live in their own attention-free graphs so
        the transformer graphs never see a symbolic sum/difference sequence
        dim. The loop in :meth:`execute` only orchestrates these tensor-in /
        tensor-out calls, so no numeric op runs eagerly between steps.
        """
        cond_config = self.transformer.config
        uncond_config = self.unconditional_transformer.config
        device = self.devices[0]
        in_channels = cond_config.in_channels

        cond_weights = dict(self.transformer.state_dict)
        uncond_weights = dict(self.unconditional_transformer.state_dict)

        with F.lazy():
            cond_transformer = Ideogram4Transformer2DModel(cond_config)
            cond_transformer.to(device)
            self._cond_transformer = cond_transformer.compile(
                *cond_transformer.input_types(), weights=cond_weights
            )

            uncond_transformer = Ideogram4Transformer2DModel(uncond_config)
            uncond_transformer.to(device)
            self._uncond_transformer = uncond_transformer.compile(
                *uncond_transformer.input_types(), weights=uncond_weights
            )

            pack_step = Ideogram4PackStep(
                in_channels, cond_config.dtype, cond_config.device
            )
            pack_step.to(device)
            self._pack_step = pack_step.compile(*pack_step.input_types())

            combine_step = Ideogram4CombineStep(
                in_channels, cond_config.dtype, cond_config.device
            )
            combine_step.to(device)
            self._combine_step = combine_step.compile(
                *combine_step.input_types()
            )

    # -----------------------------------------------------------------
    # Input preparation
    # -----------------------------------------------------------------
    @traced(message="Ideogram4Pipeline.prepare_inputs")
    def prepare_inputs(
        self,
        context: PixelContext,
    ) -> Ideogram4ModelInputs:
        device = self.transformer.devices[0]
        dtype = self.transformer.config.dtype

        tokens_np = np.asarray(context.tokens.array, dtype=np.int64)
        if tokens_np.ndim == 2:
            if tokens_np.shape[0] != 1:
                raise ValueError("Ideogram 4 bring-up supports batch_size=1.")
            tokens_np = tokens_np[0]
        num_text = int(tokens_np.shape[0])

        latents_np = np.asarray(context.latents, dtype=np.float32)
        # Latents arrive packed as (1, num_image, 128).
        if latents_np.ndim != 3 or latents_np.shape[0] != 1:
            raise ValueError(
                "Ideogram 4 expects packed latents of shape (1, num_image, 128)."
            )
        num_image = int(latents_np.shape[1])
        latent_dim = int(latents_np.shape[2])

        # Packed-token grid: the 128-channel DiT latent is a 2x2 patch over the
        # 8x-downsampled VAE latent, so the token grid per side is
        # ``(dim // vae_scale_factor) // 2`` (e.g. 512 -> 64 -> 32).
        grid_h = (int(context.height) // self.vae_scale_factor) // 2
        grid_w = (int(context.width) // self.vae_scale_factor) // 2
        if grid_h * grid_w != num_image:
            raise ValueError(
                f"grid {grid_h}x{grid_w} != num_image_tokens {num_image}."
            )

        total_seq = num_text + num_image

        # ----- packed position ids / indicator (layout [text][image]) -----
        position_ids = np.zeros((1, total_seq, 3), dtype=np.int64)
        text_pos = np.arange(num_text, dtype=np.int64)
        position_ids[0, :num_text, 0] = text_pos
        position_ids[0, :num_text, 1] = text_pos
        position_ids[0, :num_text, 2] = text_pos

        h_idx = np.repeat(np.arange(grid_h), grid_w)
        w_idx = np.tile(np.arange(grid_w), grid_h)
        position_ids[0, num_text:, 0] = IMAGE_POSITION_OFFSET
        position_ids[0, num_text:, 1] = h_idx + IMAGE_POSITION_OFFSET
        position_ids[0, num_text:, 2] = w_idx + IMAGE_POSITION_OFFSET

        indicator = np.zeros((1, total_seq), dtype=np.int64)
        indicator[0, :num_text] = LLM_TOKEN_INDICATOR
        indicator[0, num_text:] = OUTPUT_IMAGE_INDICATOR

        # ----- text features: encode text tokens, place at text positions ----
        tokens_tensor = Tensor(
            storage=Buffer.from_dlpack(np.ascontiguousarray(tokens_np)).to(
                self.text_encoder.devices[0]
            )
        )
        text_feats = self.text_encoder(tokens_tensor)
        if isinstance(text_feats, (list, tuple)):
            text_feats = text_feats[0]
        if text_feats.rank == 2:
            text_feats = F.unsqueeze(text_feats, 0)
        text_feats = text_feats.cast(dtype)

        # The dense Qwen3 encoder concatenates the selected layers tap-major
        # (``[block0_dims][block1_dims]...``). The Ideogram reference (and thus
        # the DiT input projection) expects interleaved (hidden-major) order
        # ``[dim0_taps][dim1_taps]...``. Re-order: [1, S, L*D] -> [1, S, D*L].
        seq = int(text_feats.shape[1])
        num_layers = len(QWEN3_VL_ACTIVATION_LAYERS)
        hidden = int(text_feats.shape[-1]) // num_layers
        text_feats = F.reshape(text_feats, (1, seq, num_layers, hidden))
        text_feats = F.permute(text_feats, (0, 1, 3, 2))
        text_feats = F.reshape(text_feats, (1, seq, hidden * num_layers))
        feat_dim = int(text_feats.shape[-1])

        image_feat_zeros = F.constant(
            0.0, dtype=dtype, device=device
        ).broadcast_to([1, num_image, feat_dim])
        llm_features = F.concat([text_feats, image_feat_zeros], axis=1)

        # ----- device tensors -----
        # Latents flow through the loop in the bf16 compute dtype; the combine
        # graph casts to float32 internally for the CFG + Euler step and back.
        x_image = Tensor(
            storage=Buffer.from_dlpack(np.ascontiguousarray(latents_np)).to(
                device
            )
        ).cast(dtype)
        text_z_padding = F.constant(
            0.0, dtype=dtype, device=device
        ).broadcast_to([1, num_text, latent_dim])

        position_ids_t = Tensor(
            storage=Buffer.from_dlpack(position_ids).to(device)
        )
        indicator_t = Tensor(storage=Buffer.from_dlpack(indicator).to(device))
        neg_position_ids_t = Tensor(
            storage=Buffer.from_dlpack(
                np.ascontiguousarray(position_ids[:, num_text:])
            ).to(device)
        )
        neg_indicator_t = Tensor(
            storage=Buffer.from_dlpack(
                np.ascontiguousarray(indicator[:, num_text:])
            ).to(device)
        )
        neg_llm_features = F.constant(
            0.0, dtype=dtype, device=device
        ).broadcast_to([1, num_image, feat_dim])

        return Ideogram4ModelInputs(
            x_image=x_image,
            text_z_padding=text_z_padding,
            llm_features=llm_features,
            position_ids=position_ids_t,
            indicator=indicator_t,
            neg_llm_features=neg_llm_features,
            neg_position_ids=neg_position_ids_t,
            neg_indicator=neg_indicator_t,
            timesteps=np.asarray(context.timesteps, dtype=np.float32),
            deltas=np.asarray(context.sigmas, dtype=np.float32),
            guidance_scale=float(context.guidance_scale),
            num_text=num_text,
            num_image=num_image,
            grid_h=grid_h,
            grid_w=grid_w,
            num_images_per_prompt=int(context.num_images_per_prompt),
            height=int(context.height),
            width=int(context.width),
            num_inference_steps=int(context.num_inference_steps),
        )

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------
    @staticmethod
    def _unwrap(result: object) -> Tensor:
        if isinstance(result, (list, tuple)):
            result = result[0]
        assert isinstance(result, Tensor)
        return result

    @traced(message="Ideogram4Pipeline.execute")
    def execute(
        self, model_inputs: Ideogram4ModelInputs, **kwargs: object
    ) -> DiffusionPipelineOutput:
        mi = model_inputs
        device = self.devices[0]

        # Each denoise step is a sequence of compiled-graph executions (pack,
        # cond transformer, uncond transformer, CFG+Euler combine), so ``z``
        # stays a realized float32 tensor between steps with no eager glue.
        z = mi.x_image  # (1, num_image, 128) float32
        guidance = Tensor(
            storage=Buffer.from_dlpack(
                np.asarray([mi.guidance_scale], dtype=np.float32)
            ).to(device)
        )
        # 1-D CPU carrier whose *length* encodes num_text for the combine
        # graph's velocity slice (contents unused).
        num_text_carrier = Tensor(
            storage=Buffer.from_dlpack(np.empty(mi.num_text, dtype=np.float32))
        )
        num_steps = int(mi.timesteps.shape[0])

        for step in range(num_steps):
            t = Tensor(
                storage=Buffer.from_dlpack(
                    np.asarray([float(mi.timesteps[step])], dtype=np.float32)
                ).to(device)
            )
            dt = Tensor(
                storage=Buffer.from_dlpack(
                    np.asarray([float(mi.deltas[step])], dtype=np.float32)
                ).to(device)
            )
            pos_z = self._pack_step(mi.text_z_padding, z)
            pos_out = self._unwrap(
                self._cond_transformer(
                    pos_z, mi.llm_features, t, mi.position_ids, mi.indicator
                )
            )
            neg_v = self._unwrap(
                self._uncond_transformer(
                    z,
                    mi.neg_llm_features,
                    t,
                    mi.neg_position_ids,
                    mi.neg_indicator,
                )
            )
            z = self._combine_step(
                z, pos_out, neg_v, num_text_carrier, dt, guidance
            )

        z = z.cast(self.vae.config.dtype)
        image_tensor = self.vae.decode_packed(z, mi.grid_h, mi.grid_w)
        images = np.asarray(image_tensor.to_numpy(), dtype=np.uint8)
        return DiffusionPipelineOutput(images=images)
