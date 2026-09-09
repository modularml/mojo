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
"""Wan-specific pixel generation tokenizer."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import numpy.typing as npt
import PIL.Image
from max.pipelines.lib.pixel_tokenizer import PixelGenerationTokenizer
from max.pipelines.request import OpenResponsesRequest

from .context import WanContext

logger = logging.getLogger("max.pipelines")


class WanTokenizer(PixelGenerationTokenizer):
    """Wan-specific tokenizer that produces WanContext with video/MoE fields."""

    def _select_wan_flow_shift(self, height: int, width: int) -> float:
        # Use explicit flow_shift from scheduler if set (user override).
        cfg_shift = getattr(self._scheduler, "flow_shift", None)
        if cfg_shift is not None and float(cfg_shift) != 1.0:
            return float(cfg_shift)
        # Default: interpolate based on pixel count.
        # 480p (480*832 = 399_360) -> 3.0, 720p (720*1280 = 921_600) -> 5.0
        pixels = height * width
        lo_px, hi_px = 399_360, 921_600
        lo_shift, hi_shift = 3.0, 5.0
        t = max(0.0, min(1.0, (pixels - lo_px) / (hi_px - lo_px)))
        return lo_shift + t * (hi_shift - lo_shift)

    async def new_context(
        self,
        request: OpenResponsesRequest,
        input_image: PIL.Image.Image | None = None,
    ) -> WanContext:
        base = await super().new_context(request, input_image=input_image)

        video_options = request.body.provider_options.video
        pixel_options = video_options or request.body.provider_options.image

        # Wan's execution path always expects 5D latents. Keep image requests
        # image-like at the API layer, but represent them internally as a
        # single-frame video so the transformer and VAE follow the same path.
        num_frames: int = (
            video_options.num_frames
            if video_options and video_options.num_frames is not None
            else 1
        )
        guidance_scale_2: float | None = (
            video_options.guidance_scale_2 if video_options else None
        )

        # Resolve video dimensions, falling back to what the base computed.
        height = (video_options and video_options.height) or base.height
        width = (video_options and video_options.width) or base.width

        # Resolve inference steps from video options.
        video_steps = (
            video_options.steps
            if video_options and video_options.steps is not None
            else None
        )
        num_inference_steps = (
            video_steps if video_steps is not None else base.num_inference_steps
        )

        # Recompute scheduler with video dimensions and Wan flow shift.
        latent_height = 2 * (int(height) // (self._vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self._vae_scale_factor * 2))
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        # Resolve flow_shift per request and pass it through instead of
        # mutating the shared scheduler, so an override can't leak into
        # later requests.
        extra_kwargs: dict[str, float] = {}
        if getattr(self._scheduler, "use_flow_sigmas", False):
            request_flow_shift = (
                pixel_options.flow_shift if pixel_options else None
            )
            extra_kwargs["flow_shift"] = (
                float(request_flow_shift)
                if request_flow_shift is not None
                else self._select_wan_flow_shift(height, width)
            )

        timesteps, sigmas = self._scheduler.retrieve_timesteps_and_sigmas(
            image_seq_len, num_inference_steps, **extra_kwargs
        )

        num_warmup_steps: int = max(
            len(timesteps) - num_inference_steps * self._scheduler.order, 0
        )

        # Compute Wan MoE boundary timestep from model metadata.
        boundary_timestep: float | None = None
        boundary_ratio = self._manifest_metadata.get("boundary_ratio")
        if boundary_ratio is not None:
            boundary_timestep = float(boundary_ratio) * float(
                getattr(self._scheduler, "num_train_timesteps", 1000)
            )

        step_coefficients: npt.NDArray[np.float32] | None = None
        if hasattr(self._scheduler, "build_step_coefficients"):
            step_coefficients = self._scheduler.build_step_coefficients()

        vae_scale_factor_temporal = 4
        latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        shape_5d = (
            base.num_images_per_prompt,
            self._num_channels_latents,
            latent_frames,
            latent_height,
            latent_width,
        )
        latents = self._randn_tensor(shape_5d, request.body.seed)

        # Build WanContext from base fields + Wan-specific overrides.
        base_fields = {
            f.name: getattr(base, f.name) for f in dataclasses.fields(base)
        }
        base_fields.update(
            height=height,
            width=width,
            timesteps=timesteps,
            sigmas=sigmas,
            latents=latents,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
        )
        return WanContext(
            **base_fields,
            num_frames=num_frames,
            guidance_scale_2=guidance_scale_2,
            step_coefficients=step_coefficients,
            boundary_timestep=boundary_timestep,
        )
