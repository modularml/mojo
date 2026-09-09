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
"""FLUX.2 / FLUX.2-Klein-specific pixel generation tokenizer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import PIL.Image
from max.pipelines.context import PixelContext, TokenBuffer
from max.pipelines.context.exceptions import InputError, PromptTooLongError
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.pixel_tokenizer import (
    PipelineClassName,
    PixelGenerationTokenizer,
    run_with_default_executor,
)
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.provider_options import ImageProviderOptions

from .system_messages import SYSTEM_MESSAGE, format_input, format_input_klein

logger = logging.getLogger("max.pipelines")

# Distilled FLUX.2-Klein is trained for exactly this many denoising steps;
# other values produce incoherent images.
_DISTILLED_KLEIN_NUM_STEPS: int = 4


def tokenize_klein_text(
    delegate: Any,
    prompt: str,
    max_length: int | None,
    add_special_tokens: bool = True,
) -> Any:
    """Apply the FLUX.2-Klein tokenization recipe to a single prompt.

    Wraps the prompt as a user-only message via ``format_input_klein``,
    applies the Qwen3 chat template with ``add_generation_prompt=True``
    and ``enable_thinking=False`` when supported, then encodes. When
    ``max_length`` is set the result is padded/truncated to that length;
    otherwise the result has the prompt's natural length.

    Raises ``PromptTooLongError`` if the natural-length tokenization exceeds
    ``max_length`` (rather than silently truncating).

    Shared by ``Flux2Tokenizer.encode`` (production) and the
    ``run_max_text_encoder`` verification harness so both paths tokenize
    identically.
    """
    messages_batch = format_input_klein(prompts=[prompt], images=None)
    kwargs = dict(
        add_generation_prompt=True,
        tokenize=False,
    )
    try:
        prompt_text = delegate.apply_chat_template(
            messages_batch[0],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        prompt_text = delegate.apply_chat_template(
            messages_batch[0],
            **kwargs,
        )
    raw_ids = delegate.encode(
        prompt_text, add_special_tokens=add_special_tokens
    )
    if max_length and len(raw_ids) > max_length:
        raise PromptTooLongError(
            len(raw_ids),
            max_length,
            limit_description="text encoder's maximum sequence length",
        )

    if max_length is None:
        return delegate(
            prompt_text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=True,
        )
    return delegate(
        prompt_text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=add_special_tokens,
        return_attention_mask=True,
    )


class Flux2Tokenizer(PixelGenerationTokenizer):
    """FLUX.2-family tokenizer for Flux2Pipeline and Flux2KleinPipeline."""

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
            scheduler_config_overrides={"use_empirical_mu": True},
        )
        self._max_pixel_size = 1024 * 1024

        self._is_distilled_klein = (
            self._pipeline_class_name == PipelineClassName.FLUX2_KLEIN
            and bool(self._manifest_metadata.get("is_distilled", False))
        )
        # Distilled FLUX.2-Klein is trained for exactly
        # ``_DISTILLED_KLEIN_NUM_STEPS`` steps; override the executor-level
        # default so requests that omit ``steps`` pick up the right value.
        if self._is_distilled_klein:
            self._default_num_inference_steps = _DISTILLED_KLEIN_NUM_STEPS

    @staticmethod
    def _build_text_ids(batch_size: int, seq_len: int) -> npt.NDArray[np.int64]:
        """Creates 4D ``(T=0, H=0, W=0, L=arange(seq_len))`` text position IDs.

        Returns:
            Array of shape ``(batch_size, seq_len, 4)`` int64.
        """
        coords = np.stack(
            [
                np.zeros(seq_len, dtype=np.int64),
                np.zeros(seq_len, dtype=np.int64),
                np.zeros(seq_len, dtype=np.int64),
                np.arange(seq_len, dtype=np.int64),
            ],
            axis=-1,
        )
        return np.ascontiguousarray(
            np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
        )

    def _prepare_latent_image_ids(
        self, height: int, width: int, batch_size: int = 1
    ) -> npt.NDArray[np.float32]:
        t_coords, h_coords, w_coords, l_coords = np.meshgrid(
            np.array([0]),
            np.arange(height),
            np.arange(width),
            np.array([0]),
            indexing="ij",
        )
        latent_image_ids = np.stack(
            [t_coords, h_coords, w_coords, l_coords], axis=-1
        )
        latent_image_ids = latent_image_ids.reshape(-1, 4)

        latent_image_ids = np.tile(
            latent_image_ids[np.newaxis, :, :], (batch_size, 1, 1)
        )
        return latent_image_ids

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
            if self._pipeline_class_name == PipelineClassName.FLUX2:
                messages_batch = format_input(
                    prompts=[prompt_str],
                    system_message=SYSTEM_MESSAGE,
                    images=None,
                )

                # apply_chat_template with truncation=True silently
                # drops tokens; error early instead.
                precheck = delegate.apply_chat_template(
                    messages_batch[0],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    truncation=False,
                )
                precheck_ids = precheck["input_ids"]
                precheck_len = (
                    len(precheck_ids[0])
                    if precheck_ids and isinstance(precheck_ids[0], list)
                    else len(precheck_ids)
                )
                if max_sequence_length and precheck_len > max_sequence_length:
                    raise PromptTooLongError(
                        precheck_len,
                        max_sequence_length,
                        limit_description="text encoder's maximum sequence length",
                    )

                return delegate.apply_chat_template(
                    messages_batch[0],
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    padding="max_length",
                    truncation=True,
                    max_length=max_sequence_length,
                    return_length=False,
                    return_overflowing_tokens=False,
                )
            elif self._pipeline_class_name == PipelineClassName.FLUX2_KLEIN:
                return tokenize_klein_text(
                    delegate,
                    prompt_str,
                    max_length=max_sequence_length,
                    add_special_tokens=add_special_tokens,
                )
            else:
                raise ValueError(
                    f"Flux2Tokenizer received unexpected pipeline class "
                    f"{self._pipeline_class_name!r}; expected FLUX2 or "
                    "FLUX2_KLEIN."
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

        # The FLUX.2 text encoder path does not consume an explicit
        # attention mask, so strip padded tokens here and keep a dense
        # mask. FLUX.2-Klein consumes attention_mask directly.
        if self._pipeline_class_name == PipelineClassName.FLUX2:
            token_row = input_ids_array[0]
            mask_row = attention_mask_array[0]
            real_token_ids = token_row[mask_row]
            if real_token_ids.size == 0:
                raise ValueError(
                    f"{self._pipeline_class_name.value} tokenization produced "
                    "an empty effective prompt after attention masking."
                )
            input_ids_array = np.expand_dims(
                real_token_ids.astype(np.int64, copy=False), axis=0
            )
            attention_mask_array = np.ones_like(input_ids_array, dtype=np.bool_)

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
        """Creates a new :class:`PixelContext` for FLUX.2 / FLUX.2-Klein."""
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

        if self._pipeline_class_name == PipelineClassName.FLUX2_KLEIN:
            if (
                self._is_distilled_klein
                and pixel_options.steps is not None
                and pixel_options.steps != _DISTILLED_KLEIN_NUM_STEPS
            ):
                raise InputError(
                    f"FLUX.2-Klein distilled requires steps="
                    f"{_DISTILLED_KLEIN_NUM_STEPS} (got {pixel_options.steps})."
                )
            # for non-distilled models, CFG is enabled
            # whenever guidance_scale > 1.0; negative prompt defaults to "".
            do_true_cfg = guidance_scale > 1.0 and not self._is_distilled_klein
        else:
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
            do_true_cfg,
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

        preprocessed_image_array = None
        if input_image is not None:
            preprocessed_image = self._preprocess_input_image(
                input_image,
                target_height=None,
                target_width=None,
                preserve_aspect_ratio=True,
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
            image_seq_len, num_inference_steps, sigma_min=None
        )

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

        # Emit ``text_ids`` at the canonical padded length so the
        # executor can left-pad the encoder output to match before the
        # denoiser sees it. The Flux2 denoiser was trained with the
        # full padded ``text_seq_len`` (typically 512) present in joint
        # attention. Fall back to the token-buffer length if no
        # ``max_length`` is configured.
        text_seq_len = (
            int(self.max_length)
            if self.max_length is not None
            else int(token_buffer.array.shape[0])
        )
        text_ids = self._build_text_ids(
            image_specific.num_images,
            text_seq_len,
        )
        negative_text_ids: npt.NDArray[np.int64] = np.array([], dtype=np.int64)
        if negative_token_buffer is not None:
            negative_text_seq_len = (
                int(self.max_length)
                if self.max_length is not None
                else int(negative_token_buffer.array.shape[0])
            )
            negative_text_ids = self._build_text_ids(
                image_specific.num_images,
                negative_text_seq_len,
            )

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
            seed=request.body.seed,
        )
