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

import base64
import copy
import io
import json
import logging
from collections.abc import Sequence
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt
import requests
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    mrope_pos_ids_3d,
)
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import smart_resize
from max.pipelines.architectures.qwen3vl_moe.nn.data_processing import (
    QWEN3VL_MAX_PIXELS,
    get_bilinear_interpolation_weights_and_indices,
    get_rope_index,
    get_seqlens,
)
from max.pipelines.context import (
    ImageMetadata,
    TokenBuffer,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import (
    TextAndVisionTokenizer,
    VisionPreprocessCache,
    max_tokens_to_generate,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import encode_dkv_cache_hint
from max.pipelines.modeling.types import (
    ImageContentPart,
    MessageContent,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.support.image import find_contiguous_ranges, hash_image
from PIL import Image
from transformers import AutoTokenizer

from .context import Qwen3VLTextAndVisionContext, VisionEncodingData

logger = logging.getLogger("max.pipelines")


def _load_image(image_input: dict[str, Any]) -> Image.Image:
    """Load an image from diverse input formats without resizing.

    Matches `fetch_image` input handling but skips smart_resize/resize entirely.
    Always converts to RGB.

    image_input: dict[str, Any] = {
        "image": Image.Image | bytes | str | dict[str, Any] = {
            "file_id": str,
            "url": str,
        },
        "image_url": str,
    }
    """
    image = image_input.get("image") or image_input.get("image_url")
    if isinstance(image, dict):
        image = image.get("file_id") or image.get("url")

    image_obj: Image.Image | None = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, bytes):
        image_obj = Image.open(io.BytesIO(image))
    elif isinstance(image, str) and (image.startswith(("http://", "https://"))):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif isinstance(image, str) and image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif isinstance(image, str) and image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif isinstance(image, str):
        # fallback path for plain filesystem paths
        image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(f"Unsupported image input: {image_input}")

    # Ensure RGB
    if image_obj.mode != "RGB":
        image_obj = image_obj.convert("RGB")
    return image_obj


def qwen3vl_image_preprocessing(
    image: Image.Image,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 65536,
    max_pixels: int = QWEN3VL_MAX_PIXELS,
) -> tuple[npt.NDArray[np.float32], tuple[int, int, int]]:
    """Preprocess image for Qwen3VL vision model.

    This function matches the behavior of the transformers Qwen2VLImageProcessorFast.
    It uses smart_resize to calculate target dimensions and resizes the PIL image
    before processing.

    Args:
        image: PIL Image in RGB mode to preprocess (not resized)
        patch_size: Patch size for vision transformer (default 16)
        temporal_patch_size: Temporal patch size (default 2)
        merge_size: Spatial merge size (default 2)
        min_pixels: Minimum pixels for smart_resize (default 65536)
        max_pixels: Maximum pixels for smart_resize

    Returns:
        Tuple of (pixel_values, image_grid_thw) where:
        - pixel_values: Flattened patch values as numpy array
        - image_grid_thw: Grid dimensions (temporal, height, width)
    """

    width, height = image.size

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    if resized_height != height or resized_width != width:
        image = image.resize(
            (resized_width, resized_height), resample=Image.Resampling.BICUBIC
        )
        height, width = resized_height, resized_width

    height_patches = height // patch_size
    width_patches = width // patch_size

    img_array = np.array(image, dtype=np.float32)

    # Qwen3VL uses mean=0.5, std=0.5 for normalization to [-1, 1] range
    # Also, Rescale to [0, 1] using the same rescale_factor as transformers (1/255)
    # img_array = img_array / 255.0
    # This is equivalent to: (x - 0.5) / 0.5 = 2*x - 1
    img_array = (img_array - (0.5 * 255.0)) / (0.5 * 255.0)

    patches = np.array([img_array])  # Shape: (n_images, height, width, 3)
    patches = patches.transpose(
        0, 3, 1, 2
    )  # Shape: (n_images, 3, height, width)

    channel = patches.shape[1]
    grid_h, grid_w = height_patches, width_patches

    # Handle temporal dimension padding if not divisible by temporal_patch_size
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = np.repeat(
            patches[-1][np.newaxis],
            temporal_patch_size - (patches.shape[0] % temporal_patch_size),
            axis=0,
        )
        patches = np.concatenate([patches, repeats], axis=0)

    # For images, grid_t should be 1 (single temporal group)
    grid_t = 1

    # Now reshape with spatial merging
    # Grid dimensions are divisible by merge_size (after padding if needed)
    patches = patches.reshape(
        grid_t,  # Temporal groups (1 for images)
        temporal_patch_size,  # Patches per temporal group (2)
        channel,  # RGB channels (3)
        grid_h // merge_size,  # Spatial groups in height
        merge_size,  # Patches per spatial group (2)
        patch_size,  # Patch height (16 for Qwen3VL)
        grid_w // merge_size,  # Spatial groups in width
        merge_size,  # Patches per spatial group (2)
        patch_size,  # Patch width (16 for Qwen3VL)
    )

    # Reorder dimensions to get the correct patch ordering
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    image_grid_thw = (grid_t, grid_h, grid_w)

    return flatten_patches, image_grid_thw


# One image's patchified pixels and its (t, h, w) grid, as cached.
_PreprocessedImage = tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]


class Qwen3VLImageProcessor:
    """Custom image processor for Qwen3VL that handles image processing without PyTorch dependencies.

    This processor mimics the interface of AutoImageProcessor but uses pure NumPy/PIL
    for image preprocessing.
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 65536,
        max_pixels: int = QWEN3VL_MAX_PIXELS,
    ):
        """Initialize the custom image processor.

        Args:
            patch_size: Patch size for vision transformer
            temporal_patch_size: Temporal patch size
            merge_size: Spatial merge size (used for calculating image tokens)
            min_pixels: Minimum pixels for smart_resize (default 65536)
            max_pixels: Maximum pixels for smart_resize
        """
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __call__(
        self,
        images: list[Image.Image] | Image.Image,
        return_tensors: str = "np",
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
        list[npt.NDArray[np.float32]],
    ]:
        """Process images for Qwen3VL.

        Args:
            images: Single image or list of images to process
            return_tensors: Ignored (always returns numpy arrays)

        Returns:
            Tuple of:
            - pixel_values: Normalized pixel values as numpy array of shape (num_patches, patch_features)
            - image_grid_thw: Grid dimensions as numpy array of shape (num_images, 3) where each row is (temporal, height, width)
            - pixel_values_list: List of pixel values for each image
        """
        # Handle single image vs list of images
        if isinstance(images, Image.Image):
            images = [images]

        # Process each image
        pixel_values_list: list[npt.NDArray[np.float32]] = []
        image_grid_thw_list: list[tuple[int, int, int]] = []

        for image in images:
            pixel_values, image_grid_thw_tuple = qwen3vl_image_preprocessing(
                image,
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            pixel_values_list.append(pixel_values)
            image_grid_thw_list.append(image_grid_thw_tuple)

        # TODO: Replace this with a parallel operation.
        # In that case, replace concatenated_pixel_values in context by pixel_values_list.
        pixel_values = np.vstack(pixel_values_list)
        image_grid_thw_array: npt.NDArray[np.int32] = np.array(
            image_grid_thw_list, dtype=np.int32
        )

        return pixel_values, image_grid_thw_array, pixel_values_list

    def preprocess(
        self,
        images: list[Image.Image] | Image.Image,
        return_tensors: str = "np",
        **kwargs,
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.int32],
        list[npt.NDArray[np.float32]],
    ]:
        """Alias for __call__ to match transformers interface."""
        return self.__call__(images, return_tensors=return_tensors, **kwargs)


class Qwen3VLTokenizer(TextAndVisionTokenizer):
    """Qwen3VL-specific processor that handles vision and text processing.

    This processor uses separate AutoTokenizer and custom image processor
    to handle multimodal inputs for the Qwen3VL model. It mimics the interface
    of the transformers Qwen3VLProcessor.

    - image_processor is a custom image processor that handles image preprocessing without PyTorch dependencies.
    - tokenizer uses transformers' tokenizer directly instead of AutoProcessor to avoid dependency on PyTorch.
    - no video support yet.
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
        assert config is not None

        # Extract vision config parameters
        vision_config = config.vision_config

        self.spatial_merge_size = getattr(
            vision_config, "spatial_merge_size", 2
        )
        self.num_position_embeddings = getattr(
            vision_config, "num_position_embeddings", None
        )

        # Store patch_size for potential future use
        self.patch_size = getattr(vision_config, "patch_size", 16)
        self.temporal_patch_size = getattr(
            vision_config, "temporal_patch_size", 2
        )

        # Create custom image processor instead of AutoImageProcessor
        self.img_processor = Qwen3VLImageProcessor(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            merge_size=self.spatial_merge_size,
            min_pixels=65536,  # Qwen3VL default shortest_edge
            max_pixels=QWEN3VL_MAX_PIXELS,
        )

        self._preprocess_cache: VisionPreprocessCache[_PreprocessedImage] = (
            VisionPreprocessCache.for_images(pipeline_config.runtime)
        )

        # Initialize EOS token IDs
        eos_token_id = self.delegate.eos_token_id
        self._eos_token_ids = (
            {eos_token_id} if eos_token_id is not None else set()
        )

        huggingface_config = pipeline_config.model.huggingface_config
        if eos_token_id := getattr(huggingface_config, "eos_token_id", None):
            if isinstance(eos_token_id, int):
                self._eos_token_ids.add(eos_token_id)
            elif isinstance(eos_token_id, list):
                self._eos_token_ids.update(eos_token_id)

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.enable_vision_caching = (
            pipeline_config.runtime.vision_cache_utilization != 0
        )

        if image_token_id := getattr(
            huggingface_config, "image_token_id", None
        ):
            self.image_token_id = image_token_id
        else:
            raise ValueError("image_token_id not found in model_config config")

        if video_token_id := getattr(
            huggingface_config, "video_token_id", None
        ):
            self.video_token_id = video_token_id

        # Qwen3VL specific: vision_start_token and vision_end_token
        if vision_start_token_id := getattr(
            huggingface_config, "vision_start_token_id", None
        ):
            self.vision_start_token_id = vision_start_token_id
        else:
            raise ValueError(
                "vision_start_token_id not found in model_config config"
            )

        if vision_end_token_id := getattr(
            huggingface_config, "vision_end_token_id", None
        ):
            self.vision_end_token_id = vision_end_token_id
        else:
            raise ValueError(
                "vision_end_token_id not found in model_config config"
            )

        vision_cfg = getattr(huggingface_config, "vision_config", None)
        if vision_cfg is not None:
            # If num_position_embeddings wasn't found in config, try from huggingface_config
            if self.num_position_embeddings is None:
                self.num_position_embeddings = getattr(
                    vision_cfg, "num_position_embeddings", None
                )
        else:
            raise ValueError(
                "vision_config must be provided in HuggingFace Config"
            )

        if self.num_position_embeddings is None:
            raise ValueError(
                "num_position_embeddings not found in vision_config. "
                "This is required for bilinear interpolation position embeddings."
            )

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
        templated_message = self.delegate.apply_chat_template(
            [msg.model_dump(exclude_none=True) for msg in messages],
            tokenize=False,
            tools=tools,
            **chat_template_options,
        )
        assert isinstance(templated_message, str)
        return templated_message

    def _preprocess_image(
        self, image_hash: int | None, image: Image.Image
    ) -> _PreprocessedImage:
        """Preprocesses one image, reusing a cached result when available.

        The cache key is the digest of the raw encoded bytes plus the
        resolution size class -- the same key ``new_context`` hands to the
        vision encoder cache, computed once and shared by both. Hitting here
        skips the smart-resize, rescale and patchify that the encoder cache
        cannot skip, since it is consulted only after that work has run.

        ``Qwen3VLImageProcessor`` calls :func:`qwen3vl_image_preprocessing`
        once per image with no cross-image state and only stacks at the end, so
        preprocessing one image at a time and stacking afterwards is
        bit-identical to the batched call it replaces.

        Args:
            image_hash: The image's content digest, or ``None`` when no media
                caching needs one.
            image: The decoded image to preprocess.

        Returns:
            The image's patchified pixels and its ``(t, h, w)`` grid.
        """

        def preprocess() -> _PreprocessedImage:
            _, image_grid_thw, pixel_values_list = self.img_processor(
                images=[image], return_tensors="pt"
            )
            return pixel_values_list[0], image_grid_thw[0]

        return self._preprocess_cache.get_or_preprocess(image_hash, preprocess)

    async def new_context(
        self, request: TextGenerationRequest
    ) -> Qwen3VLTextAndVisionContext:
        """Create a new Qwen3VLTextAndVisionContext for Qwen3VL processing.

        This method processes both text and vision inputs using the Qwen3VL
        processor and extracts the necessary components for model execution.
        """
        # Check for video inputs and raise error; we do not support video inputs in MAX
        if request.messages:
            messages_data = [msg.model_dump() for msg in request.messages]
            for msg in messages_data:
                contents = msg.get("content", [])
                if not isinstance(contents, list):
                    continue
                for item in contents:
                    if isinstance(item, dict) and item.get("type") == "video":
                        raise ValueError(
                            "Qwen3VL processor in MAX framework does not support video inputs. "
                            "Please remove video inputs from the request."
                        )

        # Step 1: Extract prompt from request and apply chat template if needed
        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
            if request.images:
                if not isinstance(request.prompt, str):
                    raise ValueError(
                        "prompt must be a string when images are provided"
                    )
                content: list[MessageContent] = [
                    TextContentPart(text=request.prompt)
                ]
                content.extend(ImageContentPart() for _ in request.images)
                messages = [
                    TextGenerationRequestMessage(role="user", content=content)
                ]
                new_request = TextGenerationRequest(
                    request_id=request.request_id,
                    model_name=request.model_name,
                    messages=messages,
                )
                assert new_request.messages
                prompt = self.apply_chat_template(
                    new_request.messages,
                    tools=request.tools,
                    **(request.chat_template_options or {}),
                )
        elif request.messages:
            prompt = self.apply_chat_template(
                request.messages,
                tools=request.tools,
                **(request.chat_template_options or {}),
            )
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        # Step 2: Load and process images
        image_inputs = None
        if request.images:
            # _load_image accepts both a PIL.Image and raw bytes.
            image_inputs = [
                _load_image({"image": image})
                for image in request.images_for_processing()
            ]

        # Check for BOS token BEFORE image expansion
        # This matches transformers' processing_utils.py line 1693-1694
        if (
            self.delegate.bos_token is not None
            and isinstance(prompt, str)
            and prompt.startswith(self.delegate.bos_token)
        ):
            add_special_tokens = False

        # Step 3: Process images with custom image processor (if any) and expand <|image_pad|> placeholders in text
        pixel_values_list: list[npt.NDArray[np.float32]] = []
        image_grid_thw: npt.NDArray[np.int32] | None = None
        pixel_values: npt.NDArray[np.float32] | None = None
        image_hashes: list[int | None] = []
        if image_inputs:
            # Key each image on its raw encoded bytes (+ the resolution size
            # class), not on hash_image(pixel_values): the raw-byte key is
            # byte-identical across torch/BLAS/device so a separate encoder can
            # reproduce it for cache-aware routing, whereas the post-resize
            # float hash cannot. smart_resize is deterministic from the encoded
            # bytes + fixed config, so the process-wide max-pixels resolution
            # bound is the size tier. request.images is 1:1 with image_inputs.
            #
            # One digest per image, shared by the preprocessed-tensor cache
            # here and the vision encoder cache downstream, so the bytes are
            # hashed once rather than twice.
            image_hashes = (
                [
                    hash_image(raw_bytes, self.img_processor.max_pixels)
                    for raw_bytes in request.images
                ]
                if self.enable_prefix_caching
                or self.enable_vision_caching
                or self._preprocess_cache.enabled
                else [None] * len(image_inputs)
            )
            per_image = [
                self._preprocess_image(image_hash, image)
                for image_hash, image in zip(
                    image_hashes, image_inputs, strict=True
                )
            ]
            pixel_values_list = [pixels for pixels, _ in per_image]
            # Reassembles exactly what the batched call returned, so every
            # caller downstream sees the same arrays it always did.
            pixel_values = np.vstack(pixel_values_list)
            image_grid_thw = np.array(
                [grid for _, grid in per_image], dtype=np.int32
            )

            # Expand <|image_pad|> placeholders using image_grid_thw and merge_size**2
            # Match transformers logic: process text as a list, expand each image token one by one
            merge_len = self.img_processor.merge_size**2

            # Match transformers: process as list, replace each <|image_pad|> sequentially
            assert isinstance(prompt, str)
            text_list = [prompt]

            index = 0
            for i in range(len(text_list)):
                while "<|image_pad|>" in text_list[i]:
                    if index >= len(image_grid_thw):
                        raise ValueError(
                            f"More <|image_pad|> tokens than images. "
                            f"Found {text_list[i].count('<|image_pad|>')} tokens but only {len(image_grid_thw)} images"
                        )
                    grid_values = image_grid_thw[index]
                    num_img_tokens = int(np.prod(grid_values)) // merge_len
                    # Replace first occurrence of <|image_pad|> with placeholder tokens
                    text_list[i] = text_list[i].replace(
                        "<|image_pad|>",
                        "<|placeholder|>" * num_img_tokens,
                        1,
                    )
                    index += 1
                # Convert all placeholders back to <|image_pad|> tokens
                text_list[i] = text_list[i].replace(
                    "<|placeholder|>", "<|image_pad|>"
                )

            text = text_list
        else:
            assert isinstance(prompt, str)
            text = [prompt]

        # Step 4: Tokenize the expanded text
        # See processing_qwen3_vl.py line 52-57 for defaults
        tokenizer_kwargs = {
            "padding": False,
            "return_token_type_ids": False,
            "add_special_tokens": add_special_tokens,
        }

        tokenizer_outputs = self.delegate(text, **tokenizer_kwargs)

        # Extract input_ids from tokenizer outputs
        if isinstance(tokenizer_outputs["input_ids"][0], int):
            encoded_prompt = np.array(tokenizer_outputs["input_ids"])
        else:
            encoded_prompt = np.array(tokenizer_outputs["input_ids"][0])

        # Extract attention_mask for use in get_rope_index
        # This should be extracted regardless of whether images are present
        # since the tokenizer always provides attention_mask
        attention_mask: npt.NDArray[np.floating[Any]] | None = None
        if "attention_mask" in tokenizer_outputs:
            attention_mask_raw = tokenizer_outputs["attention_mask"]
            # Handle various formats from tokenizer
            if hasattr(attention_mask_raw, "numpy"):
                attention_mask = attention_mask_raw.numpy()
            elif isinstance(attention_mask_raw, list):
                attention_mask = np.array(attention_mask_raw)
            elif isinstance(attention_mask_raw, np.ndarray):
                attention_mask = attention_mask_raw
            else:
                attention_mask = np.array(attention_mask_raw)

        # Calculate max generation tokens
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens
        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        # Handle JSON schema if provided
        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format
            and request.response_format.json_schema is not None
            else None
        )

        if self.max_length and encoded_prompt.shape[0] > self.max_length:
            raise PromptTooLongError(encoded_prompt.shape[0], self.max_length)

        # Step 5: Process vision model inputs for Qwen3VL using image processing results
        vision_data: VisionEncodingData | None = None
        images: list[ImageMetadata] = []
        if image_inputs:
            assert image_grid_thw is not None
            assert pixel_values is not None
            # pixel_values is already set from img_processor above if images were present
            image_token_indices = (
                (encoded_prompt == self.image_token_id)
                .nonzero()[0]
                .astype(np.int32)
            )
            # Precompute vision_position_ids for this context
            vision_position_ids = mrope_pos_ids_3d(
                grid_thw=image_grid_thw,
                spatial_merge_size=self.spatial_merge_size,
            )

            # Precompute bilinear interpolation weights and indices
            if self.num_position_embeddings is None:
                raise ValueError(
                    "num_position_embeddings is required for bilinear interpolation"
                )
            num_grid_per_side = int(self.num_position_embeddings**0.5)
            bilinear_indices, bilinear_weights = (
                get_bilinear_interpolation_weights_and_indices(
                    grid_thw=image_grid_thw,
                    num_grid_per_side=num_grid_per_side,
                )
            )

            # Precompute seqlens values (Qwen3VL uses simpler get_seqlens without window attention)
            cu_seqlens_arr, max_seqlen = get_seqlens(
                grid_thw=image_grid_thw,
            )
            # max_seqlen is already uint32, convert to array for VisionEncodingData
            max_seqlen_arr = np.array(max_seqlen, dtype=np.uint32)

            # Precompute max_grid_size (max of height and width dimensions)
            max_grid_size = np.array(
                int(np.max(image_grid_thw[:, 1:])), dtype=np.int32
            )

            # Create VisionEncodingData with all vision-specific fields
            vision_data = VisionEncodingData(
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                vision_position_ids=vision_position_ids,
                max_grid_size=max_grid_size,
                weights=bilinear_weights,
                indices=bilinear_indices,
                cu_seqlens=cu_seqlens_arr,
                max_seqlen=max_seqlen_arr,
                concatenated_pixel_values=pixel_values,
            )
        else:
            # TODO:consistently handle image_token_indices when we don't get images. Here or model.py?
            image_token_indices = np.array([], dtype=np.int32)

        # process images for prefix caching
        if pixel_values_list:
            start_and_end_idxs = find_contiguous_ranges(
                encoded_prompt, [self.image_token_id]
            )
            # Reuses the digests computed above for the preprocessed-tensor
            # cache (see there for why the key is the raw encoded bytes rather
            # than the post-resize pixels); computing them again here would
            # hash every image's bytes twice.
            images = [
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixel_values,
                    image_hash=image_hash
                    if self.enable_prefix_caching or self.enable_vision_caching
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

        # Calculate Rope Delta and position ids
        decoder_position_ids, rope_delta_array = get_rope_index(
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            input_ids=encoded_prompt.reshape(1, -1),
            image_grid_thw=vision_data.image_grid_thw
            if vision_data is not None
            else None,
            # Video processing not supported in MAX
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )
        decoder_position_ids = decoder_position_ids.squeeze(1)
        rope_delta = int(rope_delta_array.item())

        # Create and return context
        # TODO: Why are we passing images and vision_data to the context?
        # images are redundant since we have the pixel values in the vision_data.
        context = Qwen3VLTextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            tokens=TokenBuffer(
                array=encoded_prompt.astype(np.int64, copy=False),
            ),
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            images=images,
            vision_token_ids=[self.image_token_id],
            # Qwen3VL-specific fields
            spatial_merge_size=self.spatial_merge_size,
            rope_delta=rope_delta,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            image_token_indices=image_token_indices,
            decoder_position_ids=decoder_position_ids,
            vision_data=vision_data,
            vocab_size=self.tokenizer_vocab_size,
            cache_salt=request.cache_salt,
        )

        return context
