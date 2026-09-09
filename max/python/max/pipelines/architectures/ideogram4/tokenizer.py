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
"""Ideogram 4 pixel-generation tokenizer.

Tokenizes the prompt with the Qwen2 chat template and builds a
:class:`PixelContext` carrying the Ideogram logit-normal flow-matching
schedule. For bring-up the raw prompt is tokenized verbatim (no hosted
magic-prompt expansion, no safety screening).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import PIL.Image
from max.pipelines.context import PixelContext, TokenBuffer
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.pixel_tokenizer import (
    PixelGenerationTokenizer,
    run_with_default_executor,
)
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.provider_options import ImageProviderOptions

logger = logging.getLogger("max.pipelines")

# Logit-normal schedule clamp (ideogram4.scheduler.LogitNormalSchedule).
_LOGSNR_MIN = -15.0
_LOGSNR_MAX = 18.0


def _ndtri(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (Acklam's algorithm, ~1e-9 abs error).

    Pure-numpy replacement for ``scipy.special.ndtri`` so the tokenizer does
    not pull in scipy. Maps ``p in (0, 1)`` to the quantile; endpoints map to
    +/- inf, which the caller clamps via the schedule's t_min/t_max.
    """
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p = p.astype(np.float64)
    out = np.empty_like(p)
    plow, phigh = 0.02425, 1.0 - 0.02425

    lo = p < plow
    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(p[lo]))
        out[lo] = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    hi = p > phigh
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log(1.0 - p[hi]))
        out[hi] = -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    mid = ~(lo | hi)
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (
            (
                ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]
            )
            * q
            / (
                ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r
                + 1.0
            )
        )
    with np.errstate(divide="ignore"):
        out[p <= 0.0] = -np.inf
        out[p >= 1.0] = np.inf
    return out


def _logit_normal_schedule(
    t: np.ndarray, mean: float, std: float
) -> np.ndarray:
    """Vectorized ``LogitNormalSchedule`` (see ideogram4.scheduler)."""
    z = _ndtri(t.astype(np.float64))
    y = mean + std * z
    t_ = 1.0 - (1.0 / (1.0 + np.exp(-y)))  # 1 - sigmoid(y)
    t_min = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MAX))
    t_max = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MIN))
    return np.clip(t_, t_min, t_max).astype(np.float32)


def _resolution_mean(height: int, width: int, mu: float) -> float:
    """``get_schedule_for_resolution`` mean (known_resolution = 512x512).

    ``mu`` is the schedule mean at the known 512x512 resolution; the mean
    shifts by ``0.5 * log(num_pixels / known_pixels)`` for other resolutions.
    """
    num_pixels = height * width
    known_pixels = 512 * 512
    return mu + 0.5 * math.log(num_pixels / known_pixels)


class Ideogram4Tokenizer(PixelGenerationTokenizer):
    """Tokenizer + context builder for ``Ideogram4Pipeline``."""

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        subfolder: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        default_num_inference_steps: int = 20,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__(
            model_path,
            pipeline_config,
            subfolder,
            revision=revision,
            max_length=max_length or 2048,
            trust_remote_code=trust_remote_code,
            default_num_inference_steps=default_num_inference_steps,
        )
        transformer_config = pipeline_config.models[
            "transformer"
        ].huggingface_config.to_dict()
        self._num_channels_latents = int(
            transformer_config.get("in_channels", 128)
        )

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        latent_height: int,
        latent_width: int,
        seed: int | None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Sample noise in the *packed* ``(B, grid_h * grid_w, 128)`` layout.

        Ideogram 4 (like its torch reference) operates on packed image tokens:
        the ``in_channels = 128`` DiT latent is a 2x2 patch over the
        ``latent_channels = 32`` VAE latent. The token grid is therefore half
        the VAE-latent spatial size (``latent_height // 2``). The base
        implementation samples the unpacked ``(B, C, H, W)`` VAE layout, which
        the Ideogram DiT/VAE do not consume, so override it here.
        """
        grid_h = latent_height // 2
        grid_w = latent_width // 2
        num_image = grid_h * grid_w
        latents = self._randn_tensor(
            (batch_size, num_image, num_channels_latents), seed
        )
        latent_image_ids = self._prepare_latent_image_ids(
            grid_h, grid_w, batch_size
        )
        return latents, latent_image_ids

    async def encode(
        self,
        prompt: str,
        add_special_tokens: bool = True,
        *,
        use_secondary: bool = False,
        images: list[PIL.Image.Image] | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        delegate = self.delegate

        def _encode_fn(prompt_str: str) -> Any:
            assert delegate is not None
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_str}],
                }
            ]
            text = delegate.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            return delegate(text, add_special_tokens=False)

        out = await run_with_default_executor(_encode_fn, prompt)
        input_ids = np.asarray(out["input_ids"], dtype=np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if self.max_length is not None and input_ids.shape[1] > self.max_length:
            raise PromptTooLongError(
                input_ids.shape[1],
                self.max_length,
                limit_description="text encoder's maximum sequence length",
            )
        mask = np.ones_like(input_ids, dtype=np.bool_)
        return input_ids[0], mask[0]

    async def new_context(
        self,
        request: OpenResponsesRequest,
        input_image: PIL.Image.Image | None = None,
    ) -> PixelContext:
        prompt = self._retrieve_prompt(request)
        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")

        image_options = request.body.provider_options.image
        pixel_options = image_options or ImageProviderOptions()
        guidance_scale = (
            pixel_options.guidance_scale
            if pixel_options.guidance_scale is not None
            else 7.0
        )

        token_ids, attn_mask = await self.encode(prompt)
        token_buffer = TokenBuffer(array=token_ids.astype(np.int64, copy=False))

        vae_scale_factor = self._vae_scale_factor
        default_sample_size = self._default_sample_size
        height = pixel_options.height or default_sample_size * vae_scale_factor
        width = pixel_options.width or default_sample_size * vae_scale_factor

        # 2x2-patch over the 8x-downsampled latent (same packing as the donor).
        latent_height = 2 * (int(height) // (vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (vae_scale_factor * 2))

        num_inference_steps = (
            pixel_options.steps
            if pixel_options.steps is not None
            else self._default_num_inference_steps
        )

        # Ideogram logit-normal schedule with resolution-dependent shift.
        # Oracle parity defaults: mu=0.5, std=1.75 (see dump_ref.py).
        mean = _resolution_mean(int(height), int(width), mu=0.5)
        intervals = np.linspace(
            0.0, 1.0, num_inference_steps + 1, dtype=np.float32
        )
        sched_vals = _logit_normal_schedule(intervals, mean=mean, std=1.75)
        # Reference loop runs i = N-1..0 with t_val = sched(interval[i+1]),
        # s_val = sched(interval[i]), delta = s_val - t_val, z += v * delta.
        # In loop order j = N-1-i (j = 0..N-1):
        #   t_val[j] = sched_vals[N - j] = rev[j],
        #   s_val[j] = sched_vals[N-1-j] = rev[j + 1],  rev = sched_vals[::-1].
        rev = sched_vals[::-1]
        timesteps = rev[:-1].astype(np.float32)  # (N,) t per step
        # Store per-step deltas (s - t) in the sigmas channel.
        sigmas = (rev[1:] - rev[:-1]).astype(np.float32)

        latents, latent_image_ids = self._prepare_latents(
            pixel_options.num_images,
            self._num_channels_latents,
            latent_height,
            latent_width,
            request.body.seed,
        )

        return PixelContext(
            request_id=request.request_id,
            tokens=token_buffer,
            mask=attn_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            latents=latents,
            latent_image_ids=latent_image_ids,
            text_ids=np.array([], dtype=np.int64),
            negative_text_ids=np.array([], dtype=np.int64),
            height=int(height),
            width=int(width),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=pixel_options.num_images,
            model_name=request.body.model,
            output_format=pixel_options.output_format,
        )
