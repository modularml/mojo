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
# mypy: disable-error-code="import-not-found"
"""Pixel generation tokenizer implementation."""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
from collections.abc import Callable
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import PIL.Image
from max.pipelines.context import PixelContext, TokenBuffer
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.diffusion.schedulers import SchedulerFactory
from max.pipelines.lib.request_text import retrieve_input_text
from max.pipelines.modeling.types import PipelineTokenizer
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import InputImageContent
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    PixelProviderOptionsBase,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")


async def run_with_default_executor(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Runs a callable in the default thread pool executor.

    Args:
        fn: Callable to run.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        The result of ``fn(*args, **kwargs)``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args, **kwargs)


def _load_diffusion_tokenizer(
    model_path: str,
    *,
    subfolder: str,
    revision: str | None,
    trust_remote_code: bool,
    model_max_length: int | None,
) -> Any:
    """Load a diffusion-pipeline tokenizer subfolder.

    Loads `tokenizer.json` directly via `PreTrainedTokenizerFast`,
    falling back to `AutoTokenizer` only if that fails. We avoid
    `AutoTokenizer` as the primary path because it dispatches on the
    `tokenizer_class` field in `tokenizer_config.json`, and in
    `transformers` v5 some of those classes (notably `LlamaTokenizerFast`,
    now aliased to the slow `LlamaTokenizer`) overwrite the loaded
    pre-tokenizer with a SentencePiece `Metaspace` regardless of what
    was in `tokenizer.json`. That silently breaks ByteLevel-BPE
    tokenizers like FLUX.2's Mistral-Small-3.
    """
    try:
        return PreTrainedTokenizerFast.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=model_max_length,
            subfolder=subfolder,
        )
    except Exception:
        logger.warning(
            "PreTrainedTokenizerFast.from_pretrained failed for %s/%s; "
            "falling back to AutoTokenizer.",
            model_path,
            subfolder,
        )
        return AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=model_max_length,
            subfolder=subfolder,
        )


class LockedTokenizer:
    """Serialize access to tokenizer interfaces that may race across threads."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._lock = threading.Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped tokenizer under the shared lock."""
        with self._lock:
            return self._delegate(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the wrapped tokenizer chat template under the shared lock."""
        with self._lock:
            return self._delegate.apply_chat_template(*args, **kwargs)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        """Encode with the wrapped tokenizer under the shared lock."""
        with self._lock:
            return self._delegate.encode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class PipelineClassName(str, Enum):
    """Known pipeline class names for image generation models."""

    FLUX = "FluxPipeline"
    FLUX2 = "Flux2Pipeline"
    FLUX2_KLEIN = "Flux2KleinPipeline"
    ZIMAGE = "ZImagePipeline"
    IDEOGRAM4 = "Ideogram4Pipeline"
    WAN = "WanPipeline"
    WAN_I2V = "WanImageToVideoPipeline"
    QWEN_IMAGE_EDIT = "QwenImageEditPipeline"
    QWEN_IMAGE_EDIT_PLUS = "QwenImageEditPlusPipeline"

    @classmethod
    def from_class_name(cls, class_name: str) -> PipelineClassName:
        """Resolve a PipelineClassName from a ``_class_name`` string."""
        try:
            return cls(class_name)
        except ValueError as e:
            allowed = ", ".join([m.value for m in cls])
            raise ValueError(
                f"Unsupported _class_name={class_name!r}. Allowed: {allowed}"
            ) from e


class PixelGenerationTokenizer(
    PipelineTokenizer[
        PixelContext,
        tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]],
        OpenResponsesRequest,
    ]
):
    """Encapsulates creation of PixelContext and specific token encode/decode logic.

    Args:
        model_path: Path to the model/tokenizer.
        pipeline_config: Pipeline configuration with ModelManifest metadata.
        subfolder: Subfolder within the model path for the primary tokenizer.
        subfolder_2: Optional subfolder for a second tokenizer (e.g. text encoder).
        revision: Git revision/branch to use.
        max_length: Maximum sequence length for the primary tokenizer.
        secondary_max_length: Maximum sequence length for the secondary tokenizer, if used.
        trust_remote_code: Whether to trust remote code from the model.
    """

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
        scheduler_config_overrides: dict[str, Any] | None = None,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        self._default_num_inference_steps = default_num_inference_steps

        if max_length is None:
            raise ValueError(
                "diffusion models frequently have an unbounded max length. Please provide a max length"
            )

        self.max_length = max_length

        if secondary_max_length is None and subfolder_2 is not None:
            raise ValueError(
                "diffusion models frequently have an unbounded max length. Please provide a max length"
            )

        self.secondary_max_length = secondary_max_length
        self.delegate: LockedTokenizer
        self.delegate_2: LockedTokenizer | None

        try:
            self.delegate = LockedTokenizer(
                _load_diffusion_tokenizer(
                    model_path,
                    subfolder=subfolder,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    model_max_length=self.max_length,
                )
            )

            if subfolder_2 is not None:
                self.delegate_2 = LockedTokenizer(
                    _load_diffusion_tokenizer(
                        model_path,
                        subfolder=subfolder_2,
                        revision=revision,
                        trust_remote_code=trust_remote_code,
                        model_max_length=self.secondary_max_length,
                    )
                )
            else:
                self.delegate_2 = None
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer from {model_path}. "
                "This can happen if:\n"
                "- The model is not fully supported by the transformers python package\n"
                "- Required configuration files are missing\n"
                "- The model path is incorrect\n"
                "- '--trust-remote-code' is needed but not set\n"
            ) from e

        # Extract config from ModelManifest metadata and per-component configs.
        if not pipeline_config:
            raise ValueError(
                "pipeline_config is required for PixelGenerationTokenizer."
            )
        models = pipeline_config.models
        metadata = models.metadata

        class_name = metadata.get("_class_name")
        if not class_name:
            raise ValueError(
                "ModelManifest metadata is missing required key '_class_name'."
            )
        self._pipeline_class_name = PipelineClassName.from_class_name(
            class_name
        )
        self._manifest_metadata = metadata

        # Extract static config values from per-component huggingface_config.
        vae_config = (
            models["vae"].huggingface_config.to_dict()
            if "vae" in models
            else {}
        )
        transformer_config = (
            models["transformer"].huggingface_config.to_dict()
            if "transformer" in models
            else {}
        )

        # Compute static VAE scale factor
        block_out_channels = vae_config.get("block_out_channels", None)
        self._vae_scale_factor = (
            2 ** (len(block_out_channels) - 1) if block_out_channels else 8
        )

        # Store static model dimensions
        self._default_sample_size = 128
        out_channels = transformer_config.get("out_channels")
        self._num_channels_latents = (
            out_channels
            if out_channels is not None
            else transformer_config["in_channels"] // 4
        )

        # Create scheduler from its component config.
        scheduler_config = models["scheduler"].huggingface_config
        scheduler_class_name: str | None = getattr(
            scheduler_config, "_class_name", None
        )
        if scheduler_class_name is None:
            raise ValueError(
                "Scheduler config missing '_class_name' attribute."
            )
        scheduler_cfg = scheduler_config.to_dict()
        scheduler_cfg["use_empirical_mu"] = False
        if scheduler_config_overrides:
            scheduler_cfg.update(scheduler_config_overrides)
        self._scheduler = SchedulerFactory.create(
            class_name=scheduler_class_name,
            config_dict=scheduler_cfg,
        )
        self._scheduler_shift = float(scheduler_cfg.get("shift", 1.0))

        self._max_pixel_size: int | None = None

    def _prepare_latent_image_ids(
        self, height: int, width: int, batch_size: int = 1
    ) -> npt.NDArray[np.float32]:
        latent_image_ids = np.zeros((height, width, 3))
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + np.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + np.arange(width)[None, :]
        )
        return latent_image_ids.reshape(-1, latent_image_ids.shape[-1]).astype(
            np.float32
        )

    def _randn_tensor(
        self,
        shape: tuple[int, ...],
        seed: int | None,
    ) -> npt.NDArray[np.float32]:
        rng = np.random.RandomState(seed)
        return rng.standard_normal(shape).astype(np.float32)

    @staticmethod
    def _resize_with_center_crop(
        image: PIL.Image.Image, target_width: int, target_height: int
    ) -> PIL.Image.Image:
        # When the source is already at or above the target on both dims,
        # do a pure PIL center crop to match the reference behavior.
        # See `Flux2ImageProcessor._resize_and_crop` in `diffusers`.
        w, h = image.size
        if w >= target_width and h >= target_height:
            left = (w - target_width) // 2
            top = (h - target_height) // 2
            return image.crop(
                (left, top, left + target_width, top + target_height)
            )
        ratio = target_width / target_height
        src_ratio = image.width / image.height

        src_w = (
            target_width
            if ratio > src_ratio
            else image.width * target_height // image.height
        )
        src_h = (
            target_height
            if ratio <= src_ratio
            else image.height * target_width // image.width
        )

        resized = image.resize(
            (src_w, src_h), resample=PIL.Image.Resampling.LANCZOS
        )
        canvas = PIL.Image.new("RGB", (target_width, target_height))
        canvas.paste(
            resized,
            box=(
                target_width // 2 - src_w // 2,
                target_height // 2 - src_h // 2,
            ),
        )
        return canvas

    def _preprocess_input_image(
        self,
        image: PIL.Image.Image | npt.NDArray[np.uint8],
        *,
        target_height: int | None = None,
        target_width: int | None = None,
        preserve_aspect_ratio: bool = True,
    ) -> PIL.Image.Image:
        """Preprocess input image for image-to-image generation.

        Matches diffusers FLUX2 behavior:
        - cap image area when needed
        - floor dimensions to multiples of vae_scale_factor * 2
        - apply aspect-ratio preserving center-crop resize to the floored size

        Args:
            image: PIL Image or numpy array (uint8) to preprocess.
            target_height: Optional requested output height before latent prep.
            target_width: Optional requested output width before latent prep.
            preserve_aspect_ratio: Whether to keep the source aspect ratio when
                resizing. When false, resize directly to the target dimensions.

        Returns:
            Preprocessed PIL Image with adjusted dimensions.
        """
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image.astype(np.uint8))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_width, image_height = image.size
        multiple_of = self._vae_scale_factor * 2

        if self._max_pixel_size is not None:
            if image_width * image_height > self._max_pixel_size:
                scale = (
                    self._max_pixel_size / (image_width * image_height)
                ) ** 0.5
                new_width = int(image_width * scale)
                new_height = int(image_height * scale)
                image = image.resize(
                    (new_width, new_height), PIL.Image.Resampling.LANCZOS
                )
            image_width, image_height = image.size

        image_width = max(
            (image_width // multiple_of) * multiple_of, multiple_of
        )
        image_height = max(
            (image_height // multiple_of) * multiple_of, multiple_of
        )

        if target_width is not None:
            image_width = max(
                (int(target_width) // multiple_of) * multiple_of, multiple_of
            )
        if target_height is not None:
            image_height = max(
                (int(target_height) // multiple_of) * multiple_of, multiple_of
            )

        if image.size != (image_width, image_height):
            if preserve_aspect_ratio:
                image = self._resize_with_center_crop(
                    image, image_width, image_height
                )
            else:
                image = image.resize(
                    (image_width, image_height),
                    resample=PIL.Image.Resampling.LANCZOS,
                )

        return image

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        latent_height: int,
        latent_width: int,
        seed: int | None,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        shape = (batch_size, num_channels_latents, latent_height, latent_width)

        latents = self._randn_tensor(shape, seed)
        latent_image_ids = self._prepare_latent_image_ids(
            latent_height // 2, latent_width // 2, batch_size
        )

        return latents, latent_image_ids

    async def _generate_tokens_ids(
        self,
        prompt: str,
        prompt_2: str | None = None,
        negative_prompt: str | None = None,
        negative_prompt_2: str | None = None,
        do_true_cfg: bool = False,
        images: list[PIL.Image.Image] | None = None,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.bool_],
        npt.NDArray[np.int64] | None,
        npt.NDArray[np.bool_] | None,
        npt.NDArray[np.int64] | None,
        npt.NDArray[np.bool_] | None,
        npt.NDArray[np.int64] | None,
    ]:
        """Tokenize prompt(s) with encoder model(s).

        Args:
            prompt: Primary prompt to tokenize.
            prompt_2: Secondary prompt (optional).
            negative_prompt: Negative prompt (optional).
            negative_prompt_2: Secondary negative prompt (optional).
            do_true_cfg: Whether to use true classifier-free guidance.
            images: Optional list of images for image-to-image generation (Flux2 only).

        Returns:
            Tuple of (
                token_ids,
                attn_mask,
                token_ids_2,
                attn_mask_2,
                negative_token_ids,
                negative_attn_mask,
                negative_token_ids_2,
            ).
            token_ids_2 and negative_token_ids_2 are None if no secondary tokenizer is configured.
        """
        token_ids, attn_mask = await self.encode(prompt, images=images)

        token_ids_2: npt.NDArray[np.int64] | None = None
        attn_mask_2: npt.NDArray[np.bool_] | None = None
        if self.delegate_2 is not None:
            token_ids_2, attn_mask_2 = await self.encode(
                prompt_2 or prompt,
                use_secondary=True,
            )

        negative_token_ids: npt.NDArray[np.int64] | None = None
        negative_attn_mask: npt.NDArray[np.bool_] | None = None
        negative_token_ids_2: npt.NDArray[np.int64] | None = None
        if do_true_cfg:
            negative_token_ids, negative_attn_mask = await self.encode(
                negative_prompt or ""
            )
            if self.delegate_2 is not None:
                negative_token_ids_2, _negative_attn_mask_2 = await self.encode(
                    negative_prompt_2 or negative_prompt or "",
                    use_secondary=True,
                )

        return (
            token_ids,
            attn_mask,
            token_ids_2,
            attn_mask_2,
            negative_token_ids,
            negative_attn_mask,
            negative_token_ids_2,
        )

    @property
    def eos_token_ids(self) -> set[int]:
        """Returns the end-of-sequence token IDs."""
        eos = self.delegate.eos_token_id
        return {eos} if eos is not None else set()

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns whether this tokenizer expects content wrapping."""
        return False

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
            # The tokenizer's truncation=True silently drops tokens beyond
            # max_sequence_length; error early instead.
            raw_ids = delegate.encode(
                prompt_str,
                add_special_tokens=add_special_tokens,
            )
            if max_sequence_length and len(raw_ids) > max_sequence_length:
                raise PromptTooLongError(
                    len(raw_ids),
                    max_sequence_length,
                    limit_description="text encoder's maximum sequence length",
                )

            return delegate(
                prompt_str,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=add_special_tokens,
            )

        tokenizer_output = await run_with_default_executor(_encode_fn, prompt)

        # Extract input_ids and attention_mask from both dict-like and object-like
        # tokenizer outputs (e.g. BatchEncoding).
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

    async def decode(
        self,
        encoded: tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]],
        **kwargs,
    ) -> str:
        """Decodes token arrays to text (not implemented for this tokenizer)."""
        raise NotImplementedError(
            "Decoding is not implemented for this tokenizer."
        )

    async def postprocess(
        self,
        output: Any,
    ) -> Any:
        """Post-processes pipeline output.

        Accepts either a raw numpy array or a GenerationOutput.
        For raw numpy arrays, denormalizes from [-1, 1] to [0, 1].
        For GenerationOutput, returns as-is (denormalization is handled
        in the pipeline variant before encoding to OutputImageContent).
        """
        if isinstance(output, GenerationOutput):
            return output

        # Raw numpy path
        pixel_data = (output * 0.5 + 0.5).clip(min=0.0, max=1.0)
        return pixel_data

    @staticmethod
    def _retrieve_prompt(request: OpenResponsesRequest) -> str:
        """Retrieve the text prompt from an OpenResponsesRequest.

        Args:
            request: The OpenResponsesRequest to extract the prompt from.

        Returns:
            The extracted text prompt.

        Raises:
            ValueError: If no valid prompt can be extracted from the request.
        """
        return retrieve_input_text(request)

    @staticmethod
    def _retrieve_image(
        request: OpenResponsesRequest,
    ) -> PIL.Image.Image | None:
        """Retrieve the input image from an OpenResponsesRequest.

        Extracts InputImageContent from the first message's content list and converts
        the data URI to a PIL Image.

        Args:
            request: The OpenResponsesRequest to extract the image from.

        Returns:
            PIL Image if found, None otherwise.
        """
        # Only check list inputs
        if not isinstance(request.body.input, list):
            return None

        if not request.body.input:
            return None

        first_message = request.body.input[0]

        # Only check list content
        if not isinstance(first_message.content, list):
            return None

        # Find first InputImageContent item
        for item in first_message.content:
            if isinstance(item, InputImageContent):
                # Parse data URI and convert to PIL Image
                image_url = item.image_url
                if image_url.startswith("data:"):
                    # Extract base64 data from data URI
                    # Format: data:image/png;base64,<base64_data>
                    _, base64_data = image_url.split(",", 1)
                    image_bytes = base64.b64decode(base64_data)
                    return PIL.Image.open(BytesIO(image_bytes))

        return None

    async def new_context(
        self,
        request: OpenResponsesRequest,
        input_image: PIL.Image.Image | None = None,
    ) -> PixelContext:
        """Creates a new :class:`PixelContext` object, leveraging necessary information from :class:`OpenResponsesRequest`."""
        # Extract prompt from request using the helper method
        prompt = self._retrieve_prompt(request)
        if not prompt:
            raise ValueError("Prompt must be a non-empty string.")

        # Extract input image from request content (takes precedence over input_image parameter)
        input_image = self._retrieve_image(request) or input_image

        # Resolve generation options. Shared (modality-agnostic) fields are
        # read from whichever modality block the request carries; image-only
        # fields (e.g., secondary_prompt, num_images) come from the image
        # block when present and fall back to defaults otherwise. This keeps
        # pure-video requests from needing a stub image block.
        image_options = request.body.provider_options.image
        video_options = request.body.provider_options.video
        pixel_options: PixelProviderOptionsBase = (
            video_options or image_options or ImageProviderOptions()
        )
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

        if self._pipeline_class_name in (
            PipelineClassName.WAN,
            PipelineClassName.WAN_I2V,
        ):
            # Wan uses classical CFG: diffusers enables it whenever
            # guidance_scale > 1.0, with an empty-string negative prompt as
            # the standard "unconditional" conditioning.
            do_true_cfg = guidance_scale > 1.0
            if do_true_cfg and negative_prompt_resolved is None:
                negative_prompt_resolved = ""
        else:
            do_true_cfg = (
                pixel_options.true_cfg_scale > 1.0
                and negative_prompt_resolved is not None
            )

        # 1. Tokenize prompts
        # Convert input_image to list format for _generate_tokens_ids
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

        # 2. Preprocess input image if provided
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

        # 3. Resolve image dimensions using cached static values
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

        text_ids = np.array([], dtype=np.int64)
        negative_text_ids = np.array([], dtype=np.int64)

        # 5. Build the context
        context = PixelContext(
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
            input_image=preprocessed_image_array,  # Pass numpy array instead of PIL.Image
            output_format=image_specific.output_format,
        )

        return context
