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

"""Wan Image-to-Video (I2V) pipeline.

Extends WanPipeline with image conditioning: the input image is encoded
via the VAE, combined with a temporal mask, and concatenated with noise
latents at each denoising step to produce 36-channel transformer input.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from max.driver import Buffer, Device
from max.profiler import Tracer, traced

from ..autoencoders.autoencoder_kl_wan import (
    _buffer_to_numpy_f32,
    _numpy_f32_to_buffer,
)
from .pipeline_wan import WanModelInputs, WanPipeline, WanPipelineOutput

logger = logging.getLogger(__name__)


class WanI2VPipeline(WanPipeline):
    """Wan I2V pipeline — extends WanPipeline with image conditioning."""

    _i2v_concat_model: Any = None

    def _prepare_i2v_condition(
        self,
        model_inputs: WanModelInputs,
        latent_shape: tuple[int, ...],
        device: Device,
    ) -> Buffer:
        """Prepare I2V condition tensor [B, 20, T_l, H_l, W_l].

        Encodes the input image via VAE, builds a temporal mask, and
        concatenates them.
        """
        image = model_inputs.input_image
        if image is None:
            raise ValueError("I2V pipeline requires input_image")
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        logger.info("Preparing I2V condition")

        # Normalize to [-1, 1] float32, shape [1, 3, H, W]
        image_f32 = image.astype(np.float32) / 127.5 - 1.0
        if image_f32.ndim == 3:
            image_f32 = image_f32.transpose(2, 0, 1)[np.newaxis]  # [1,3,H,W]

        batch_size = int(latent_shape[0])
        num_frames = int(model_inputs.num_frames)
        # Use target height/width from model_inputs (pixel space)
        h = int(model_inputs.height)
        w = int(model_inputs.width)

        # Resize image to target resolution if needed
        if image_f32.shape[2] != h or image_f32.shape[3] != w:
            import PIL.Image

            pil_img = PIL.Image.fromarray(
                ((image_f32[0].transpose(1, 2, 0) + 1.0) * 127.5)
                .clip(0, 255)
                .astype(np.uint8)
            )
            pil_img = pil_img.resize((w, h), PIL.Image.Resampling.LANCZOS)
            image_f32 = (
                np.array(pil_img).astype(np.float32) / 127.5 - 1.0
            ).transpose(2, 0, 1)[np.newaxis]

        enc_latent = self.vae.encode_zero_padded_video_condition(
            image_f32,
            batch_size=batch_size,
            num_frames=num_frames,
        )
        latent_cond_np = _buffer_to_numpy_f32(enc_latent)

        logger.debug(
            "VAE encode output: shape=%s min=%.4f max=%.4f mean=%.4f",
            latent_cond_np.shape,
            latent_cond_np.min(),
            latent_cond_np.max(),
            latent_cond_np.mean(),
        )

        expected_t = int(latent_shape[2])
        if latent_cond_np.shape[2] != expected_t:
            raise ValueError(
                "VAE encode temporal shape mismatch for I2V condition: "
                f"got {latent_cond_np.shape[2]}, expected {expected_t} "
                f"for num_frames={num_frames}."
            )

        expected_h = int(latent_shape[3])
        expected_w = int(latent_shape[4])
        if (
            latent_cond_np.shape[3] != expected_h
            or latent_cond_np.shape[4] != expected_w
        ):
            raise ValueError(
                "VAE encode spatial shape mismatch for I2V condition: "
                f"got {latent_cond_np.shape[3:5]}, expected "
                f"({expected_h}, {expected_w})."
            )

        z_dim = self.vae.config.z_dim
        mean = np.array(self.vae.config.latents_mean, dtype=np.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        inv_std = 1.0 / np.array(
            self.vae.config.latents_std, dtype=np.float32
        ).reshape(1, z_dim, 1, 1, 1)
        latent_cond_np = (latent_cond_np - mean) * inv_std

        # Build mask [B, vae_scale_factor_temporal, T_l, H_l, W_l]
        t_latent = latent_cond_np.shape[2]
        h_latent = latent_cond_np.shape[3]
        w_latent = latent_cond_np.shape[4]

        mask = np.zeros(
            (batch_size, 1, num_frames, h_latent, w_latent),
            dtype=np.float32,
        )
        mask[:, :, 0, :, :] = 1.0  # First frame is conditioned

        vae_t = self.vae_scale_factor_temporal
        first_mask = np.repeat(mask[:, :, 0:1, :, :], vae_t, axis=2)
        mask_expanded = np.concatenate(
            [first_mask, mask[:, :, 1:, :, :]], axis=2
        )
        # Reshape: [B, 1, T_pixel, H_l, W_l] -> [B, vae_t, T_l, H_l, W_l]
        mask_expanded = mask_expanded.reshape(
            batch_size, -1, vae_t, h_latent, w_latent
        )
        mask_expanded = mask_expanded.transpose(0, 2, 1, 3, 4)

        # Concat: [mask, latent_condition] -> [B, vae_t+z_dim, T_l, H_l, W_l]
        condition = np.concatenate(
            [mask_expanded, latent_cond_np], axis=1
        ).astype(np.float32)

        return _numpy_f32_to_buffer(condition, self.vae.config.dtype, device)

    def _compile_i2v_concat(
        self, latent_model_input: Buffer, condition: Buffer
    ) -> Any:
        """Compile a GPU graph that concatenates latents + condition along axis=1.

        Uses symbolic spatial dims so the same graph works for different
        resolutions (e.g. 1280x720 and 720x1280).
        """
        from max.graph import Graph, TensorType, ops

        device = self.transformer.devices[0]
        dtype = latent_model_input.dtype
        # batch, channels are concrete; T, H, W are symbolic
        lat_shape = [
            int(latent_model_input.shape[0]),
            int(latent_model_input.shape[1]),
            "T", "H", "W",
        ]
        cond_shape = [
            int(condition.shape[0]),
            int(condition.shape[1]),
            "T", "H", "W",
        ]

        with Graph(
            "wan_i2v_concat",
            input_types=[
                TensorType(dtype, lat_shape, device=device),
                TensorType(dtype, cond_shape, device=device),
            ],
        ) as g:
            lat = g.inputs[0].tensor
            cond = g.inputs[1].tensor
            g.output(ops.concat([lat, cond], axis=1))
        return self.session.load(g)

    def _concat_i2v_condition(
        self, latent_model_input: Buffer, condition: Buffer
    ) -> Buffer:
        """Concat latents [B,C_l,T,H,W] with condition [B,C_c,T,H,W] on GPU."""
        if self._i2v_concat_model is None:
            self._i2v_concat_model = self._compile_i2v_concat(
                latent_model_input, condition
            )
        return self._i2v_concat_model.execute(latent_model_input, condition)[0]

    @traced(message="WanI2VPipeline.execute")
    def execute(
        self,
        model_inputs: WanModelInputs,
        **kwargs: object,
    ) -> WanPipelineOutput:
        del kwargs
        device = self.transformer.devices[0]
        if not self._moe_dual_loaded:
            self._activate_transformer_weights(use_secondary=False)

        with Tracer("prepare_prompt_embeddings"):
            prompt_embeds, negative_prompt_embeds, do_cfg = (
                self.prepare_prompt_embeddings(model_inputs)
            )

        with Tracer("preprocess_latents"):
            logger.info("Preparing Wan latents")
            latents = Buffer.from_numpy(
                np.ascontiguousarray(model_inputs.latents, dtype=np.float32)
            ).to(device)

        # Prepare I2V condition from input image (includes VAE encode)
        with Tracer("prepare_i2v_condition"):
            i2v_condition = self._prepare_i2v_condition(
                model_inputs, tuple(int(d) for d in latents.shape), device
            )

        # Pre-compile I2V concat graph (latent dtype, not f32)
        if self._i2v_concat_model is None:
            latent_model_input = self._cast_f32_to_model_dtype.execute(latents)[
                0
            ]
            self._i2v_concat_model = self._compile_i2v_concat(
                latent_model_input, i2v_condition
            )

        with Tracer("prepare_scheduler"):
            if model_inputs.step_coefficients is None:
                raise ValueError(
                    "WanPipeline requires precomputed step_coefficients."
                )
            timesteps = np.ascontiguousarray(
                model_inputs.timesteps, dtype=np.float32
            )
            boundary_timestep = model_inputs.boundary_timestep
            if boundary_timestep is None and self.boundary_ratio is not None:
                boundary_timestep = (
                    self.boundary_ratio * self.num_train_timesteps
                )
            rope_cos, rope_sin = self.transformer.compute_rope(
                num_frames=int(latents.shape[2]),
                height=int(latents.shape[3]),
                width=int(latents.shape[4]),
            )
            batched_timesteps = self._get_batched_timesteps(
                scheduler_timesteps=timesteps,
                batch_size=int(latents.shape[0]),
                device=device,
            )
            coeff_buffers = [
                Buffer.from_numpy(
                    np.ascontiguousarray(row, dtype=np.float32)
                ).to(device)
                for row in model_inputs.step_coefficients
            ]
            guidance_scale_high: Buffer | None = None
            guidance_scale_low: Buffer | None = None
            if do_cfg:
                guidance_scale_high = self._get_guidance_scale(
                    float(model_inputs.guidance_scale),
                    dtype=prompt_embeds.dtype,
                    device=device,
                )
                guidance_scale_low = self._get_guidance_scale(
                    float(
                        model_inputs.guidance_scale_2
                        if model_inputs.guidance_scale_2 is not None
                        else model_inputs.guidance_scale
                    ),
                    dtype=prompt_embeds.dtype,
                    device=device,
                )
            has_moe = (
                self.transformer_2 is not None and boundary_timestep is not None
            )
            boundary_step_idx = len(timesteps)
            if boundary_timestep is not None:
                for idx, t in enumerate(timesteps):
                    if float(t) < boundary_timestep:
                        boundary_step_idx = idx
                        break
            p_t, p_h, p_w = self.transformer.config.patch_size
            spatial_shape = self._get_spatial_shape(
                int(latents.shape[2]) // p_t,
                int(latents.shape[3]) // p_h,
                int(latents.shape[4]) // p_w,
                device,
            )

        with Tracer("denoising_loop"):
            latents = self._run_i2v_denoising(
                latents,
                i2v_condition,
                prompt_embeds,
                negative_prompt_embeds,
                do_cfg,
                rope_cos,
                rope_sin,
                batched_timesteps,
                coeff_buffers,
                boundary_step_idx,
                spatial_shape,
                has_moe,
                guidance_scale_high,
                guidance_scale_low,
            )
        with Tracer("decode_outputs"):
            images = self.decode_latents(
                latents,
                int(model_inputs.num_frames),
                model_inputs.height,
                model_inputs.width,
            )
        return WanPipelineOutput(images=images)

    def _run_i2v_denoising(
        self,
        latents: Buffer,
        i2v_condition: Buffer,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
        do_cfg: bool,
        rope_cos: Buffer,
        rope_sin: Buffer,
        batched_timesteps: list[Buffer],
        coeff_buffers: list[Buffer],
        boundary_step_idx: int,
        spatial_shape: Buffer,
        has_moe: bool,
        guidance_scale_high: Buffer | None,
        guidance_scale_low: Buffer | None,
    ) -> Buffer:
        """Denoising loop with I2V condition concatenation."""
        from .pipeline_wan import WanUniPCState

        step_state: WanUniPCState = (None, None, None)
        latents, step_state = self._run_i2v_denoising_phase(
            latents=latents,
            i2v_condition=i2v_condition,
            transformer_model=self.transformer,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            batched_timesteps=batched_timesteps,
            coeff_buffers=coeff_buffers,
            do_cfg=do_cfg,
            guidance_scale=guidance_scale_high,
            step_range=range(boundary_step_idx),
            desc="Denoising (high-noise)" if has_moe else "Denoising",
            spatial_shape=spatial_shape,
            step_state=step_state,
        )

        if has_moe and boundary_step_idx < len(batched_timesteps):
            if self._moe_dual_loaded:
                low_noise_model = self.transformer_2
            else:
                self._activate_transformer_weights(use_secondary=True)
                low_noise_model = self.transformer
            latents, _ = self._run_i2v_denoising_phase(
                latents=latents,
                i2v_condition=i2v_condition,
                transformer_model=low_noise_model,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                batched_timesteps=batched_timesteps,
                coeff_buffers=coeff_buffers,
                do_cfg=do_cfg,
                guidance_scale=guidance_scale_low,
                step_range=range(boundary_step_idx, len(batched_timesteps)),
                desc="Denoising (low-noise)",
                spatial_shape=spatial_shape,
                step_state=step_state,
            )

        return latents

    def _run_i2v_denoising_phase(
        self,
        latents: Buffer,
        i2v_condition: Buffer,
        transformer_model: Any,
        prompt_embeds: Buffer,
        negative_prompt_embeds: Buffer | None,
        rope_cos: Buffer,
        rope_sin: Buffer,
        batched_timesteps: list[Buffer],
        coeff_buffers: list[Buffer],
        do_cfg: bool,
        guidance_scale: Buffer | None,
        step_range: range,
        desc: str,
        spatial_shape: Buffer,
        step_state: tuple[Buffer | None, Buffer | None, Buffer | None],
    ) -> tuple[Buffer, tuple[Buffer | None, Buffer | None, Buffer | None]]:
        """Denoising phase with I2V condition concat at each step."""
        import sys

        from tqdm.auto import tqdm

        progress = tqdm(  # type: ignore[call-arg]
            step_range,
            desc=desc,
            leave=True,
            disable=not sys.stderr.isatty(),
        )
        for i in progress:  # type: ignore[attr-defined]
            with Tracer(f"{desc}:step_{i}"):
                dit_timestep = batched_timesteps[i]
                latent_model_input = self._cast_f32_to_model_dtype.execute(
                    latents
                )[0]
                # I2V: concat condition with latents → 36 channels
                latent_model_input = self._concat_i2v_condition(
                    latent_model_input, i2v_condition
                )
                with Tracer("transformer"):
                    noise_pred_buf = self._run_transformer_forward(
                        transformer_model=transformer_model,
                        latent_model_input=latent_model_input,
                        dit_timestep=dit_timestep,
                        prompt_embeds=prompt_embeds,
                        batched_prompt_embeds=None,
                        negative_prompt_embeds=negative_prompt_embeds,
                        rope_cos=rope_cos,
                        rope_sin=rope_sin,
                        spatial_shape=spatial_shape,
                        do_cfg=do_cfg,
                        guidance_scale=guidance_scale,
                    )
                with Tracer("scheduler_step"):
                    latents, step_state = self.scheduler_step(
                        latents,
                        noise_pred_buf,
                        coeff_buffers[i],
                        step_state,
                    )
        return latents, step_state
