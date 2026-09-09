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
"""Z-Image-specific pixel generation tokenizer."""

from __future__ import annotations

import logging
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


class ZImageTokenizer(PixelGenerationTokenizer):
    """Z-Image tokenizer for ZImagePipeline."""

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        subfolder: str,
        *,
        subfolder_2: str | None = None,
        revision: str | None = None,
        max_length: int | None = None,
        secondary_max_length: int | None = None,
        trust_remote_code: bool = False,
        default_num_inference_steps: int = 50,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__(
            model_path,
            pipeline_config,
            subfolder,
            subfolder_2=subfolder_2,
            revision=revision,
            max_length=max_length,
            secondary_max_length=secondary_max_length,
            trust_remote_code=trust_remote_code,
            default_num_inference_steps=default_num_inference_steps,
        )
        # Use the transformer's in_channels directly rather than the base's
        # out_channels // 4 fallback, preserving Z-Image's prior behavior.
        transformer_config = pipeline_config.models[
            "transformer"
        ].huggingface_config.to_dict()
        self._num_channels_latents = int(transformer_config["in_channels"])

    async def encode(
        self,
        prompt: str,
        add_special_tokens: bool = True,
        *,
        use_secondary: bool = False,
        images: list[PIL.Image.Image] | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """Transforms the provided prompt into a token array."""
        delegate = self.delegate_2 if use_secondary else self.delegate
        max_sequence_length = (
            self.secondary_max_length if use_secondary else self.max_length
        )

        tokenizer_output: Any

        def _encode_fn(prompt_str: str) -> Any:
            assert delegate is not None
            messages = [{"role": "user", "content": prompt_str}]
            if not hasattr(delegate, "apply_chat_template"):
                raise ValueError(
                    "Z-Image requires tokenizer.apply_chat_template, "
                    "but the loaded tokenizer does not provide it."
                )
            return delegate.apply_chat_template(
                messages,
                enable_thinking=True,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
                return_length=False,
                return_overflowing_tokens=False,
            )

        tokenizer_output = await run_with_default_executor(_encode_fn, prompt)

        input_ids: Any
        attention_mask: Any | None
        if hasattr(tokenizer_output, "__getitem__") and (
            hasattr(tokenizer_output, "keys")
            and "input_ids" in tokenizer_output
        ):
            input_ids = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output.get("attention_mask", None)
        elif hasattr(tokenizer_output, "input_ids"):
            input_ids = tokenizer_output.input_ids
            attention_mask = getattr(tokenizer_output, "attention_mask", None)
        else:
            raise ValueError(
                "Tokenizer output does not contain `input_ids`; cannot build PixelContext."
            )

        input_ids_array = np.asarray(input_ids, dtype=np.int64)
        if input_ids_array.ndim == 1:
            input_ids_array = input_ids_array[None, :]

        if attention_mask is None:
            attention_mask_array = np.ones_like(input_ids_array, dtype=np.bool_)
        else:
            attention_mask_array = np.asarray(attention_mask, dtype=np.bool_)
            if attention_mask_array.ndim == 1:
                attention_mask_array = attention_mask_array[None, :]

        if input_ids_array.shape != attention_mask_array.shape:
            raise ValueError(
                "Tokenizer produced mismatched `input_ids` and `attention_mask` shapes: "
                f"{input_ids_array.shape} vs {attention_mask_array.shape}."
            )

        if (
            max_sequence_length is not None
            and input_ids_array.shape[1] > max_sequence_length
        ):
            raise PromptTooLongError(
                input_ids_array.shape[1],
                max_sequence_length,
                limit_description="text encoder's maximum sequence length",
            )

        encoded_prompt = input_ids_array[0].astype(np.int64, copy=False)
        attention_mask_flat = attention_mask_array[0].astype(
            np.bool_, copy=False
        )

        return encoded_prompt, attention_mask_flat

    async def new_context(
        self,
        request: OpenResponsesRequest,
        input_image: PIL.Image.Image | None = None,
    ) -> PixelContext:
        """Creates a new :class:`PixelContext` for Z-Image."""
        prompt = self._retrieve_prompt(request)
        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")

        input_image = self._retrieve_image(request) or input_image

        image_options = request.body.provider_options.image
        video_options = request.body.provider_options.video
        pixel_options = video_options or image_options or ImageProviderOptions()
        image_specific = image_options or ImageProviderOptions()

        guidance_scale = pixel_options.guidance_scale

        if guidance_scale < 1.0 or pixel_options.true_cfg_scale < 1.0:
            logger.warning(
                f"Guidance scales < 1.0 detected (guidance_scale={guidance_scale}, "
                f"true_cfg_scale={pixel_options.true_cfg_scale}). This is mathematically possible"
                " but may produce lower quality or unexpected results."
            )

        negative_prompt_resolved = pixel_options.negative_prompt

        if (
            pixel_options.true_cfg_scale > 1.0
            and negative_prompt_resolved is None
        ):
            logger.warning(
                f"true_cfg_scale={pixel_options.true_cfg_scale} is set, but no negative_prompt "
                "is provided. True classifier-free guidance requires a negative prompt; "
                "falling back to standard generation."
            )

        # Z-Image enables CFG-style negative-prompt tokenization whenever
        # guidance_scale > 0.0, on top of the standard true-CFG path.
        do_zimage_cfg = guidance_scale > 0.0
        do_true_cfg = (
            pixel_options.true_cfg_scale > 1.0
            and negative_prompt_resolved is not None
        )

        images_for_tokenization: list[PIL.Image.Image] | None = None
        if input_image is not None:
            input_img: PIL.Image.Image
            if isinstance(input_image, np.ndarray):
                input_img = PIL.Image.fromarray(input_image.astype(np.uint8))
            else:
                input_img = input_image
            images_for_tokenization = [input_img]

        (
            token_ids,
            attn_mask,
            token_ids_2,
            _attn_mask_2,
            negative_token_ids,
            negative_attn_mask,
            negative_token_ids_2,
        ) = await self._generate_tokens_ids(
            prompt,
            image_specific.secondary_prompt,
            negative_prompt_resolved,
            image_specific.secondary_negative_prompt,
            do_true_cfg or do_zimage_cfg,
            images=images_for_tokenization,
        )

        token_buffer = TokenBuffer(
            array=token_ids.astype(np.int64, copy=False),
        )
        token_buffer_2 = None
        if token_ids_2 is not None:
            token_buffer_2 = TokenBuffer(
                array=token_ids_2.astype(np.int64, copy=False),
            )
        negative_token_buffer = None
        if negative_token_ids is not None:
            negative_token_buffer = TokenBuffer(
                array=negative_token_ids.astype(np.int64, copy=False),
            )
        negative_token_buffer_2 = None
        if negative_token_ids_2 is not None:
            negative_token_buffer_2 = TokenBuffer(
                array=negative_token_ids_2.astype(np.int64, copy=False),
            )

        default_sample_size = self._default_sample_size
        vae_scale_factor = self._vae_scale_factor

        # Z-Image preprocesses input images by direct resize to the requested
        # output dims rather than aspect-preserving center-crop.
        preprocessed_image_array = None
        if input_image is not None:
            preprocessed_image = self._preprocess_input_image(
                input_image,
                target_height=pixel_options.height,
                target_width=pixel_options.width,
                preserve_aspect_ratio=False,
            )
            height = pixel_options.height or preprocessed_image.height
            width = pixel_options.width or preprocessed_image.width
            preprocessed_image_array = np.array(
                preprocessed_image, dtype=np.uint8
            ).copy()
        else:
            height = (
                pixel_options.height or default_sample_size * vae_scale_factor
            )
            width = (
                pixel_options.width or default_sample_size * vae_scale_factor
            )

        latent_height = 2 * (int(height) // (self._vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self._vae_scale_factor * 2))
        image_seq_len = (latent_height // 2) * (latent_width // 2)

        num_inference_steps = (
            pixel_options.steps
            if pixel_options.steps is not None
            else self._default_num_inference_steps
        )
        timesteps, sigmas = self._scheduler.retrieve_timesteps_and_sigmas(
            image_seq_len, num_inference_steps, sigma_min=0.0
        )
        if self._scheduler_shift != 1.0:
            # Match diffusers FlowMatchEulerDiscreteScheduler static shift behavior.
            # Z-Image scheduler config uses shift=6.0.
            shifted_timesteps = (
                self._scheduler_shift
                * timesteps
                / (1.0 + (self._scheduler_shift - 1.0) * timesteps)
            ).astype(np.float32)
            timesteps = shifted_timesteps
            sigmas = np.append(shifted_timesteps, np.float32(0.0))

        # Z-Image img2img follows diffusers strength behavior by starting
        # denoising from a later timestep.
        if input_image is not None:
            init_timestep = min(
                num_inference_steps * image_specific.strength,
                float(num_inference_steps),
            )
            t_start = int(max(num_inference_steps - init_timestep, 0.0))
            timesteps = timesteps[t_start:]
            sigmas = sigmas[t_start:]
            num_inference_steps = int(timesteps.shape[0])

        num_warmup_steps: int = max(
            len(timesteps) - num_inference_steps * self._scheduler.order, 0
        )

        latents, latent_image_ids = self._prepare_latents(
            image_specific.num_images,
            self._num_channels_latents,
            latent_height,
            latent_width,
            request.body.seed,
        )

        text_ids = np.array([], dtype=np.int64)
        negative_text_ids = np.array([], dtype=np.int64)

        return PixelContext(
            request_id=request.request_id,
            tokens=token_buffer,
            mask=attn_mask,
            tokens_2=token_buffer_2,
            negative_tokens=negative_token_buffer,
            negative_mask=negative_attn_mask,
            negative_tokens_2=negative_token_buffer_2,
            explicit_negative_prompt=pixel_options.negative_prompt is not None,
            timesteps=timesteps,
            sigmas=sigmas,
            latents=latents,
            latent_image_ids=latent_image_ids,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=image_specific.num_images,
            true_cfg_scale=pixel_options.true_cfg_scale,
            strength=image_specific.strength,
            cfg_normalization=pixel_options.cfg_normalization,
            cfg_truncation=pixel_options.cfg_truncation,
            num_warmup_steps=num_warmup_steps,
            model_name=request.body.model,
            input_image=preprocessed_image_array,
            output_format=image_specific.output_format,
        )
