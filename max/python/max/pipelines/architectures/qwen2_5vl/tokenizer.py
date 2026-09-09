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

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    get_rope_index,
    get_seqlens,
    get_window_index,
    mrope_pos_ids_3d,
)
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import (
    MAX_PIXELS,
    fetch_image,
    process_vision_info,
)
from max.pipelines.context import (
    EOSTracker,
    ImageMetadata,
    TokenBuffer,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import (
    TextAndVisionTokenizer,
    VisionPreprocessCache,
    float32_to_bfloat16_as_uint16,
    max_tokens_to_generate,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import encode_dkv_cache_hint
from max.pipelines.modeling.types import (
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.support.image import find_contiguous_ranges, hash_image
from PIL import Image
from transformers import AutoTokenizer, Qwen2_5_VLConfig

from .context import Qwen2_5VLTextAndVisionContext, VisionEncodingData

logger = logging.getLogger("max.pipelines")


# Pre-computed normalization constants for ImageNet
# These are computed as: scale = 1 / (255 * std), offset = -mean / std
# This allows simplified normalization: normalized = pixel * scale + offset
_IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_NORM_SCALE = (1.0 / (255.0 * _IMAGENET_STD)).astype(np.float32)
_NORM_OFFSET = (-_IMAGENET_MEAN / _IMAGENET_STD).astype(np.float32)


def qwen2_5vl_image_preprocessing(
    image: Image.Image,
    *,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> tuple[npt.NDArray[np.uint16], tuple[int, int, int]]:
    """Preprocess image for Qwen2.5VL vision model.

    This function assumes the image has already been processed by fetch_image
    and is correctly sized. It only handles normalization and patch extraction.

    Args:
        image: PIL Image to preprocess (already resized by fetch_image)
        patch_size: Patch size for vision transformer (default 14)
        temporal_patch_size: Temporal patch size (default 2)
        merge_size: Spatial merge size (default 2)

    Returns:
        Tuple of (pixel_values, image_grid_thw) where:
        - pixel_values: Flattened patch values as numpy array
        - image_grid_thw: Grid dimensions (temporal, height, width)
    """
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Get actual dimensions
    width, height = image.size

    # Calculate grid dimensions based on actual image dimensions
    grid_h = height // patch_size
    grid_w = width // patch_size

    # Check if spatial merging is possible early
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError(
            f"Spatial merging is not possible because grid_h {grid_h} % merge_size {merge_size} != 0 or grid_w {grid_w} % merge_size {merge_size} != 0"
        )

    # Convert to numpy array (float32) with simplified normalization
    # This combines: (pixel / 255.0 - mean) / std = pixel * scale + offset
    # Using in-place operations to reduce memory allocations
    img_array = np.array(image, dtype=np.float32)
    np.multiply(img_array, _NORM_SCALE, out=img_array)
    np.add(img_array, _NORM_OFFSET, out=img_array)

    # For single images, temporal dimension is always 1 and we need to repeat
    # for temporal_patch_size.
    channel = 3
    grid_t = 1

    # Transpose to channel-first: (H, W, 3) -> (3, H, W)
    img_chw = img_array.transpose(2, 0, 1)

    # Add temporal dimension (single frame for images, will tile to temporal_patch_size at the end)
    patches = img_chw[np.newaxis]  # (1, 3, H, W)

    # Reshape with spatial merging
    # Input shape: (1, channel, height, width) - single temporal frame
    patches = patches.reshape(
        grid_t,  # Temporal groups (1 for images)
        1,  # Single frame, will tile at the end
        channel,  # RGB channels (3)
        grid_h // merge_size,  # Spatial groups in height
        merge_size,  # Patches per spatial group (2)
        patch_size,  # Patch height (14)
        grid_w // merge_size,  # Spatial groups in width
        merge_size,  # Patches per spatial group (2)
        patch_size,  # Patch width (14)
    )

    # Transpose following transformers library logic
    # This reorders dimensions to get the correct patch ordering
    # Output shape: (grid_t, gh//m, gw//m, m, m, channel, 1, ps, ps)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

    # Tile for temporal dimension: images have 1 frame but model expects
    # temporal_patch_size frames, so we replicate the single frame.
    num_patches = grid_t * grid_h * grid_w
    # Reshape to expose temporal dimension: (num_patches, channel, 1, patch_size^2)
    patches_4d = patches.reshape(
        num_patches, channel, 1, patch_size * patch_size
    )
    # Tile to (num_patches, channel, temporal_patch_size, patch_size^2)
    patches_tiled = np.tile(patches_4d, (1, 1, temporal_patch_size, 1))
    # Flatten to final shape: (num_patches, channel * temporal_patch_size * patch_size^2)
    flatten_patches = patches_tiled.reshape(
        num_patches,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    flatten_patches_uint16 = float32_to_bfloat16_as_uint16(flatten_patches)

    # Create grid dimensions (temporal, height, width)
    image_grid_thw = (grid_t, grid_h, grid_w)

    return flatten_patches_uint16, image_grid_thw


# One image's patchified pixels and its (t, h, w) grid, as cached.
_PreprocessedImage = tuple[npt.NDArray[np.uint16], npt.NDArray[np.int32]]


class Qwen2_5VLImageProcessor:
    """Custom image processor for Qwen2.5VL that handles image processing without PyTorch dependencies.

    This processor mimics the interface of AutoImageProcessor but uses pure NumPy/PIL
    for image preprocessing.
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
    ):
        """Initialize the custom image processor.

        Args:
            patch_size: Patch size for vision transformer
            temporal_patch_size: Temporal patch size
            merge_size: Spatial merge size (used for calculating image tokens)
        """
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size

    def __call__(
        self,
        images: list[Image.Image] | Image.Image,
        return_tensors: str = "np",
        **kwargs,
    ) -> tuple[dict[str, npt.NDArray[Any]], list[npt.NDArray[np.uint16]]]:
        """Process images for Qwen2.5VL.

        Args:
            images: Single image or list of images to process
            return_tensors: Ignored (always returns numpy arrays)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary containing processed image data with keys:
            - pixel_values: Normalized pixel values as numpy array of shape (num_patches, patch_features)
            - image_grid_thw: Grid dimensions as numpy array of shape (num_images, 3) where each row is (temporal, height, width)
            List of pixel values for each image
        """
        # Handle single image vs list of images
        if isinstance(images, Image.Image):
            images = [images]

        # Process each image
        pixel_values_list: list[npt.NDArray[np.uint16]] = []
        image_grid_thw_list: list[tuple[int, int, int]] = []

        for image in images:
            pixel_values, image_grid_thw_tuple = qwen2_5vl_image_preprocessing(
                image,
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.merge_size,
            )
            pixel_values_list.append(pixel_values)
            image_grid_thw_list.append(image_grid_thw_tuple)

        # Stack results
        pixel_values = np.vstack(pixel_values_list)
        image_grid_thw_array: npt.NDArray[np.int32] = np.array(
            image_grid_thw_list, dtype=np.int32
        )

        return {
            "concatenated_pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw_array,
        }, pixel_values_list

    def preprocess(
        self,
        images: list[Image.Image] | Image.Image,
        return_tensors: str = "np",
        **kwargs,
    ) -> tuple[dict[str, npt.NDArray[Any]], list[npt.NDArray[np.uint16]]]:
        """Alias for __call__ to match transformers interface."""
        return self.__call__(images, return_tensors=return_tensors, **kwargs)


class Qwen2_5VLTokenizer(TextAndVisionTokenizer):
    """Qwen2.5VL-specific tokenizer that handles vision and text processing.

    This tokenizer uses separate AutoTokenizer and custom image processor
    to handle multimodal inputs for the Qwen2.5VL model.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs,
    ):
        """Initialize the tokenizer with custom image processor instead of AutoProcessor."""
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        # Use the pre-loaded HuggingFace config from pipeline_config
        config = pipeline_config.model.huggingface_config
        config = cast(Qwen2_5_VLConfig, config)

        # Extract vision config parameters
        vision_config = config.vision_config
        self.patch_size = vision_config.patch_size
        self.window_size = vision_config.window_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size

        # Create custom image processor instead of AutoImageProcessor
        self.img_processor = Qwen2_5VLImageProcessor(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            merge_size=self.spatial_merge_size,
        )

        self._preprocess_cache: VisionPreprocessCache[_PreprocessedImage] = (
            VisionPreprocessCache.for_images(pipeline_config.runtime)
        )

        # Initialize EOS token IDs
        eos_token_id = self.delegate.eos_token_id
        self._eos_token_ids = (
            {eos_token_id} if eos_token_id is not None else set()
        )

        if eos_token_id := getattr(config, "eos_token_id", None):
            if isinstance(eos_token_id, int):
                self._eos_token_ids.add(eos_token_id)
            elif isinstance(eos_token_id, list):
                self._eos_token_ids.update(eos_token_id)

        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )

        self.vision_start_token_id = config.vision_start_token_id

        # Extract the vision config from the HuggingFace config.
        vision_cfg = config.vision_config
        self.tokens_per_second = vision_cfg.tokens_per_second

        self.executor: ThreadPoolExecutor | None = None

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Apply chat template using tokenizer directly (not processor).

        Args:
            messages: List of messages for the chat template.
            tools: Optional tools available for the model to invoke.
            **chat_template_options: Template options to forward to the Jinja
                template. Merged with ``add_generation_prompt=True`` default.

        Returns:
            The templated chat message as a string.
        """
        chat_template_options = {
            "add_generation_prompt": True,
            **chat_template_options,
        }
        messages_dicts = [msg.model_dump(exclude_none=True) for msg in messages]
        templated_message = self.delegate.apply_chat_template(
            messages_dicts,
            tokenize=False,
            tools=tools,
            **chat_template_options,
        )
        assert isinstance(templated_message, str)
        return templated_message

    def _build_eos_tracker(self, request: TextGenerationRequest) -> EOSTracker:
        """Builds an EOSTracker synchronously for use in new_context_blocking.

        All tokenizer access must happen in the same executor to avoid
        concurrent borrow errors from the HuggingFace tokenizer's Rust internals.
        """
        params = request.sampling_params
        eos_token_ids = set(self._eos_token_ids)
        eos_sequences: list[list[int]] = []
        if params.ignore_eos:
            eos_token_ids = set()
        else:
            if params.stop_token_ids:
                eos_token_ids.update(params.stop_token_ids)
            if params.stop:
                for stop_string in params.stop:
                    tokenized = self.delegate.encode(
                        stop_string, add_special_tokens=False
                    )
                    if tokenized:
                        eos_sequences.append(list(tokenized))
        return EOSTracker(
            eos_token_ids=eos_token_ids,
            eos_sequences=eos_sequences,
            eos_stop_strings=params.stop or [],
        )

    async def new_context(
        self, request: TextGenerationRequest
    ) -> Qwen2_5VLTextAndVisionContext:
        """Create a new Qwen2_5VLTextAndVisionContext for Qwen2.5VL processing.

        This method processes both text and vision inputs using the Qwen2.5VL
        processor and extracts the necessary components for model execution.
        """
        if self.executor is None:
            # lazy init the executor because the tokenizer gets pickled
            # when launching the model worker, and the executor is not pickle-safe.
            # max_workers=1 to serialize all tokenizer access — the HF
            # tokenizer's Rust internals are not thread-safe and concurrent
            # calls to delegate.encode trigger a borrow error.
            self.executor = ThreadPoolExecutor(max_workers=1)

        return await asyncio.get_running_loop().run_in_executor(
            self.executor,
            self.new_context_blocking,
            request,
        )

    def _retrieve_prompt(self, request: TextGenerationRequest) -> str:
        # If a text prompt is provided, immediately pass through.
        if request.prompt is not None:
            if not isinstance(request.prompt, str):
                raise ValueError(
                    "Qwen2.5VL tokenizer only supports string prompts, not token sequences"
                )
            return request.prompt

        if request.messages:
            return self.apply_chat_template(
                request.messages,
                request.tools,
                **(request.chat_template_options or {}),
            )

        raise ValueError(f"{request} does not provide messages or prompt.")

    def _preprocess_image(
        self, image_hash: int | None, image: Image.Image
    ) -> _PreprocessedImage:
        """Preprocesses one image, reusing a cached result when available.

        The cache key is the digest of the raw encoded bytes plus the
        resolution size class -- the same key ``new_context`` hands to the
        vision encoder cache, computed once and shared by both. Hitting here
        skips the smart-resize, rescale and patchify that the encoder cache
        cannot skip, since it is consulted only after that work has run.

        ``Qwen2_5VLImageProcessor`` calls
        :func:`qwen2_5vl_image_preprocessing` once per image with no
        cross-image state and only stacks at the end, so preprocessing one
        image at a time and stacking afterwards is bit-identical to the
        batched call it replaces.

        Args:
            image_hash: The image's content digest, or ``None`` when no media
                caching needs one, or when the image arrived without raw bytes
                to key on.
            image: The decoded image to preprocess.

        Returns:
            The image's patchified pixels and its ``(t, h, w)`` grid.
        """

        def preprocess() -> _PreprocessedImage:
            processed_dict, pixel_values_list = self.img_processor(
                images=[image],
                return_tensors="np",
            )
            return pixel_values_list[0], processed_dict["image_grid_thw"][0]

        return self._preprocess_cache.get_or_preprocess(image_hash, preprocess)

    def _process_images(
        self, request: TextGenerationRequest
    ) -> tuple[
        npt.NDArray[Any],
        npt.NDArray[Any],
        list[npt.NDArray[np.uint16]],
        list[int | None],
    ]:
        image_inputs = None
        image_hashes: list[int | None] = []
        if request.images:
            # fetch_image accepts both a PIL.Image and raw bytes.
            image_inputs = [
                fetch_image({"image": image})
                for image in request.images_for_processing()
            ]
            # Key each image on its raw encoded bytes (+ the resolution size
            # class), not on hash_image(pixel_values): the raw-byte key is
            # byte-identical across torch/BLAS/device so a separate encoder can
            # reproduce it for cache-aware routing, whereas the post-resize
            # float hash cannot. smart_resize is deterministic from the encoded
            # bytes + fixed config, so the process-wide max-pixels resolution
            # bound is the size tier. request.images is 1:1 with image_inputs.
            image_hashes = (
                [
                    hash_image(raw_bytes, MAX_PIXELS)
                    for raw_bytes in request.images
                ]
                if self.enable_prefix_caching or self._preprocess_cache.enabled
                else [None] * len(request.images)
            )
        elif request.messages:
            image_inputs, _, _ = process_vision_info(
                [msg.model_dump() for msg in request.messages]
            )
            # Images embedded in the messages arrive already decoded, with no
            # raw encoded bytes to key on, so they are preprocessed uncached
            # rather than keyed on something the encoder could not reproduce.
            image_hashes = [None] * len(image_inputs or [])

        if image_inputs is None:
            raise ValueError("No images provided in the request")

        per_image = [
            self._preprocess_image(image_hash, image)
            for image_hash, image in zip(
                image_hashes, image_inputs, strict=True
            )
        ]
        pixel_values_list = [pixels for pixels, _ in per_image]
        # Reassembles exactly what the batched call returned, so every caller
        # downstream sees the same arrays it always did.
        return (
            np.vstack(pixel_values_list),
            np.array([grid for _, grid in per_image], dtype=np.int32),
            pixel_values_list,
            image_hashes,
        )

    def _tokenize_inputs(
        self,
        request: TextGenerationRequest,
        image_grid_thw: npt.NDArray[np.int32] | None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        tokenized_inputs = self.delegate(
            self._retrieve_prompt(request), padding=True, return_tensors="np"
        )

        input_ids = tokenized_inputs["input_ids"].squeeze(0)
        attention_mask = tokenized_inputs["attention_mask"].squeeze(0)

        # Expand input_ids/attention_mask for image token ids
        if image_grid_thw is None:
            if self.max_length and input_ids.shape[0] > self.max_length:
                raise PromptTooLongError(input_ids.shape[0], self.max_length)

            return input_ids, attention_mask

        merge_len = self.img_processor.merge_size**2
        image_token_indices = np.where(input_ids == self.image_token_id)[0]

        # Only expand as many image tokens as we have grids for
        num_images = len(image_grid_thw)
        if len(image_token_indices) < num_images:
            raise ValueError(
                f"Found {len(image_token_indices)} image tokens but have {num_images} images"
            )

        # Process only the image tokens that correspond to actual images
        # (there might be extra placeholder tokens we should ignore)
        for i in range(num_images):
            idx = image_token_indices[-(i + 1)]  # Process in reverse order
            t, h, w = image_grid_thw[-(i + 1)]
            num_img_tokens = (t * h * w) // merge_len
            # Insert num_img_tokens - 1 additional tokens to expand the single token to num_img_tokens total
            input_ids = np.insert(
                input_ids, idx, [self.image_token_id] * (num_img_tokens - 1)
            )
            # Also expand attention_mask to match the new input_ids length
            attention_mask = np.insert(
                attention_mask, idx, [1] * (num_img_tokens - 1)
            )

        if self.max_length and input_ids.shape[0] > self.max_length:
            raise PromptTooLongError(input_ids.shape[0], self.max_length)

        return input_ids, attention_mask

    def _max_length_of_request(
        self, input_ids_length: int, request: TextGenerationRequest
    ) -> int:
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        tokens_to_generate = max_tokens_to_generate(
            input_ids_length, self.max_length, max_new_tokens
        )
        if tokens_to_generate is None:
            # If max_tokens_to_generate returns None, use max_length as fallback
            return self.max_length
        return input_ids_length + tokens_to_generate

    def _calculate_rope_params(
        self,
        input_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
        image_grid_thw: npt.NDArray[np.int32] | None,
    ) -> tuple[npt.NDArray[np.int64], int]:
        """Calculate rope delta and decoder position ids."""
        # Ensure attention_mask is 2D for get_rope_index
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.reshape(1, -1)

        # Convert attention_mask to float as required by get_rope_index
        attention_mask_float = attention_mask.astype(np.float32)

        decoder_position_ids, rope_delta_array = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            tokens_per_second=self.tokens_per_second,
            input_ids=input_ids.reshape(1, -1),
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask_float,
        )
        decoder_position_ids = decoder_position_ids.squeeze(1)
        rope_delta = int(rope_delta_array.item())

        return decoder_position_ids, rope_delta

    def _create_context(
        self,
        request: TextGenerationRequest,
        input_ids: npt.NDArray[np.int64],
        attention_mask: npt.NDArray[np.int64],
        image_grid_thw: npt.NDArray[np.int32] | None,
        images: list[ImageMetadata],
        vision_data: VisionEncodingData | None,
        eos_tracker: EOSTracker,
    ) -> Qwen2_5VLTextAndVisionContext:
        """Create a Qwen2_5VLTextAndVisionContext with common fields."""
        # Calculate image token indices from input_ids
        image_token_indices = (
            (input_ids == self.image_token_id).nonzero()[0].astype(np.int32)
        )

        # Calculate rope parameters
        decoder_position_ids, rope_delta = self._calculate_rope_params(
            input_ids, attention_mask, image_grid_thw
        )

        # Handle JSON schema if provided
        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format
            and request.response_format.json_schema is not None
            else None
        )

        # Calculate max_length using common logic
        max_length = self._max_length_of_request(input_ids.shape[0], request)

        return Qwen2_5VLTextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=eos_tracker,
            tokens=TokenBuffer(input_ids),
            max_length=max_length,
            json_schema=json_schema,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            images=images,
            vision_token_ids=[self.image_token_id],
            spatial_merge_size=self.spatial_merge_size,
            rope_delta=rope_delta,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            tokens_per_second=self.tokens_per_second,
            image_token_indices=image_token_indices,
            decoder_position_ids=decoder_position_ids,
            vision_data=vision_data,
            vocab_size=self.tokenizer_vocab_size,
            cache_salt=request.cache_salt,
        )

    def new_context_blocking(
        self,
        request: TextGenerationRequest,
    ) -> Qwen2_5VLTextAndVisionContext:
        eos_tracker = self._build_eos_tracker(request)
        # Exit early, if no images are provided.
        if not request.images:
            input_ids, attention_mask = self._tokenize_inputs(request, None)

            return self._create_context(
                request=request,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=None,
                images=[],
                vision_data=None,
                eos_tracker=eos_tracker,
            )

        # Get Image Inputs (already processed by img_processor inside _process_images)
        (
            concatenated_pixel_values,
            image_grid_thw,
            pixel_values_list,
            image_hashes,
        ) = self._process_images(request)

        # Tokenize (and expand the input ids)
        input_ids, attention_mask = self._tokenize_inputs(
            request, image_grid_thw
        )

        # Precompute vision_position_ids for this context
        vision_position_ids = mrope_pos_ids_3d(
            grid_thw=image_grid_thw,
            spatial_merge_size=self.spatial_merge_size,
        )

        # Precompute window index and cu_window_seqlens
        window_index, cu_window_seqlens = get_window_index(
            grid_thw=image_grid_thw,
            window_size=self.window_size,
            spatial_merge_size=self.spatial_merge_size,
            patch_size=self.patch_size,
            spatial_merge_unit=self.spatial_merge_size**2,
        )
        # Note: cu_window_seqlens is only used locally, not passed to model

        # Precompute seqlens values
        (
            cu_seqlens,
            cu_window_seqlens_unique,
            max_seqlen,
            window_max_seqlen,
        ) = get_seqlens(
            grid_thw=image_grid_thw,
            cu_win_seqlens=cu_window_seqlens,
        )
        max_seqlen_arr = np.array(max_seqlen, dtype=np.uint32)
        window_max_seqlen_arr = np.array(window_max_seqlen, dtype=np.uint32)

        # Precompute max_grid_size (max of height and width dimensions)
        max_grid_size = np.array(
            int(np.max(image_grid_thw[:, 1:])), dtype=np.int32
        )

        # Create VisionEncodingData with all vision-specific fields
        vision_data = VisionEncodingData(
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            vision_position_ids=vision_position_ids,
            window_index=window_index,
            max_grid_size=max_grid_size,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens_unique=cu_window_seqlens_unique,
            max_seqlen=max_seqlen_arr,
            window_max_seqlen=window_max_seqlen_arr,
            concatenated_pixel_values=concatenated_pixel_values,
        )

        # Build images list from pixel values
        if pixel_values_list:
            start_and_end_idxs = find_contiguous_ranges(
                input_ids, [self.image_token_id]
            )
            # Reuses the digests _process_images already computed for the
            # preprocessed-tensor cache (see there for why the key is the raw
            # encoded bytes rather than the post-resize pixels); computing them
            # again here would hash every image's bytes twice.
            images = [
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixel_values,
                    image_hash=image_hash
                    if self.enable_prefix_caching
                    else None,
                )
                for (start_idx, end_idx), pixel_values, image_hash in zip(
                    start_and_end_idxs,
                    pixel_values_list,
                    image_hashes,
                    strict=True,
                )
            ]
        else:
            images = []

        # Create and return context using shared method
        return self._create_context(
            request=request,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            images=images,
            vision_data=vision_data,
            eos_tracker=eos_tracker,
        )
