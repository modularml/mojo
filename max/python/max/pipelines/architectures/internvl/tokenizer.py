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

"""InternVL-specific tokenizer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
from max.pipelines.lib import (
    TextAndVisionTokenizer,
    float32_to_bfloat16_as_uint16,
)
from PIL import Image
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig

_T = TypeVar("_T", bound=np.generic)

# Number of dimensions for the pixel values tensor for InternVL.
IMAGE_NDIMS = 6

# ImageNet normalization constants.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_image_context_token_id(huggingface_config: Any) -> int:
    """Get the correct token ID for "<IMG_CONTEXT>" based on the InternVL
    version. This is used to identify where to insert image embeddings in the
    text.

    Args:
        huggingface_config: HuggingFace model configuration

    Returns:
        The correct token ID for "<IMG_CONTEXT>"
    """
    llm_config = getattr(huggingface_config, "llm_config", huggingface_config)
    model_type = getattr(llm_config, "model_type", None)
    architectures = getattr(llm_config, "architectures", None) or []

    if model_type == "qwen3" or "Qwen3ForCausalLM" in architectures:
        return 151671  # InternVL3.5+ (Qwen3)
    else:
        return 151667  # InternVL3 and earlier (Qwen2)


class InternVLImageConfig:
    """InternVL image-specific configuration values for processing and memory estimation."""

    num_image_token: int
    """Number of tokens per image patch."""

    max_dynamic_patch: int
    """Maximum number of dynamic patches."""

    image_size: int
    """Size of input images."""

    patch_size: int
    """Size of each patch."""

    def __init__(
        self, config: AutoConfig, vision_overrides: dict[str, Any]
    ) -> None:
        """Initialize from HuggingFace model configuration.

        Args:
            config: HuggingFace model configuration
            vision_overrides: Vision config overrides from pipeline config
        """
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            raise ValueError(
                "`vision_config` not found in `config` (HuggingFace config)"
            )

        # Get configuration values with defaults.
        self.image_size = getattr(vision_config, "image_size", 448)
        self.patch_size = getattr(vision_config, "patch_size", 14)

        # Check for override first, then fall back to config attribute.
        self.max_dynamic_patch = vision_overrides.get(
            "max_dynamic_patch", getattr(config, "max_dynamic_patch", 12)
        )

        downsample_ratio = getattr(config, "downsample_ratio", 0.5)

        # Calculate number of image tokens per patch.
        self.num_image_token = int(
            (self.image_size // self.patch_size) ** 2 * (downsample_ratio**2)
        )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def calculate_num_patches_for_image(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> int:
    """Calculate the number of patches for an image without actual preprocessing."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio.
    target_ratios = list(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        }
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target.
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the number of blocks
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
        blocks += 1

    return blocks


def imagenet_normalize(
    img: Image.Image, input_size: int
) -> npt.NDArray[np.float32]:
    """Normalize image using ImageNet normalization.

    This converts PIL image to normalized numpy array with proper preprocessing.
    """
    # Convert to RGB if needed.
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize with BICUBIC interpolation.
    img = img.resize((input_size, input_size), Image.Resampling.BICUBIC)

    # Convert to numpy array and normalize to [0, 1].
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Apply ImageNet normalization.
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

    return img_array


def crop_into_patches(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    """Dynamically preprocess image with adaptive tiling."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio.
    target_ratios = list(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        }
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target..
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height.
    # Note: Each individual patch will be square (image_size x image_size),
    # but the overall image is resized to maintain aspect ratio by using
    # different numbers of patches horizontally and vertically.
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split the image into patches.
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for y_block in range(target_aspect_ratio[1]):
        for x_block in range(target_aspect_ratio[0]):
            box = (
                x_block * image_size,
                y_block * image_size,
                (x_block + 1) * image_size,
                (y_block + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def extract_patches_from_image(
    normalized_image: npt.NDArray[_T],
    *,
    patch_size: int = 14,
) -> npt.NDArray[_T]:
    """Extract patches from a normalized image array.

    This replicates the exact patch extraction operations from
    InternVisionEmbeddings.__call__ to move them to preprocessing.

    Args:
        normalized_image: Normalized image array of shape (H, W, C).
        patch_size: Size of each patch.

    Returns:
        Array of shape (height_patches, width_patches, C, patch_size, patch_size).
    """
    height, width, channels = normalized_image.shape

    # Calculate number of patches
    h_patches = height // patch_size
    w_patches = width // patch_size

    # 1. Reshape to extract patches
    # From (H, W, C) to (H/P, P, W/P, P, C)
    reshaped = normalized_image.reshape(
        h_patches, patch_size, w_patches, patch_size, channels
    )

    # 2. Single transpose to rearrange patches and convert HWC to CHW
    # From (H/P, P, W/P, P, C) directly to (H/P, W/P, C, P, P)
    # This combines the patch grouping and HWC->CHW conversion
    return np.transpose(reshaped, (0, 2, 4, 1, 3))


def preprocess_image_to_tensor(
    pil_image: Image.Image,
    *,
    input_size: int = 448,
    max_num: int = 12,
    patch_size: int = 14,
) -> npt.NDArray[np.uint16]:
    """Preprocess image to tensor with dynamic patching - must match InternVLProcessor.

    This function replicates the exact patch extraction operations from
    InternVisionEmbeddings.__call__ to move them to preprocessing for memory efficiency.

    Returns:
        Buffer of shape (batch_size, height_patches, width_patches, C, patch_size, patch_size)
        where each image's patches preserve spatial dimensions.
    """
    images = crop_into_patches(
        pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )

    # Process each image separately to maintain batch structure
    processed_images = []
    for image in images:
        # Normalize the image - shape: (H, W, C)
        normalized = imagenet_normalize(image, input_size)

        # Extract patches using the shared function
        patches = extract_patches_from_image(normalized, patch_size=patch_size)

        processed_images.append(patches)

    # Stack all images together - shape: (batch_size, num_patches, features)
    # batch_size = number of images (including thumbnail if applicable)
    stacked = np.stack(processed_images).astype(np.float32)

    # Convert to bfloat16 representation as uint16.
    return float32_to_bfloat16_as_uint16(stacked)


class InternVLProcessor:
    """Custom processor for InternVL that handles text tokenization and image token insertion.

    This mimics the interface expected by TextAndVisionTokenizer while handling
    InternVL's tokenizer-only approach (no AutoProcessor available).

    NOTE: This processor does NOT do actual image preprocessing (no torch/torchvision).
    It only handles text formatting and token insertion. Image preprocessing happens
    later in the model layer.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        config: AutoConfig,
        vision_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config

        # Extract InternVL image configuration.
        image_config = InternVLImageConfig(config, vision_overrides or {})
        self.image_size = image_config.image_size
        self.max_dynamic_patch = image_config.max_dynamic_patch
        self.num_image_token = image_config.num_image_token

        # Image token configuration
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str | dict[str, str]]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str | list[int] | list[str] | list[list[int]] | BatchEncoding:
        """Converts a list of TextGenerationRequestMessage with `role` and `content` to a formatted string.

        This method handles multimodal messages by extracting text content before
        forwarding to the HF tokenizer.
        """

        # Convert multimodal messages to text-only for the tokenizer
        text_messages = []
        for message in messages:
            text_message = {"role": message["role"]}
            content = message["content"]

            if isinstance(content, str):
                text_message["content"] = content
            elif isinstance(content, list):
                # Process multimodal content in order, placing <image> placeholders
                # where image content appears to match VLMEvalKit behavior.
                content_parts = []
                for item in content:
                    if item["type"] == "text":
                        content_parts.append(item["text"])
                    elif item["type"] == "image":
                        # Insert image placeholder where image content appears.
                        content_parts.append("<|image|>")

                # Join content parts preserving order.
                text_message["content"] = " ".join(content_parts)
            else:
                text_message["content"] = ""

            text_messages.append(text_message)

        # Forward to the HF tokenizer with text-only messages
        return self.tokenizer.apply_chat_template(
            text_messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def __call__(
        self,
        text: str,
        images: list[Image.Image] | None = None,
        add_special_tokens: bool = True,
        return_tensors: str = "np",
    ) -> dict[str, Any] | BatchEncoding:
        """Process text and images for InternVL.

        This method needs to match the interface expected by TextAndVisionTokenizer.new_context().
        """
        if images is None or len(images) == 0:
            # Text-only case - just tokenize normally
            return self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
            )

        # Process images and format prompt (without actual image preprocessing)
        processed_text = text
        raw_pixel_values = []

        for image in images:
            # Calculate number of patches based on image dimensions
            num_patches = calculate_num_patches_for_image(
                image,
                min_num=1,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=True,
            )

            # Format text with image tokens
            num_image_tokens = self.num_image_token * num_patches
            image_tokens = (
                self.IMG_START_TOKEN
                + self.IMG_CONTEXT_TOKEN * num_image_tokens
                + self.IMG_END_TOKEN
            )

            # Replace <image> or <|image|> with InternVL's format.
            if "<image>" in processed_text:
                processed_text = processed_text.replace(
                    "<image>", image_tokens, 1
                )
            elif "<|image|>" in processed_text:
                # Handle the test prompt format with pipes.
                processed_text = processed_text.replace(
                    "<|image|>", image_tokens, 1
                )
            else:
                # If no <image> placeholder, find the last user prompt and
                # insert the image tokens at the beginning of it.
                user_prompt_start = "<|im_start|>user\n"
                last_user_prompt_idx = processed_text.rfind(user_prompt_start)

                if last_user_prompt_idx != -1:
                    # Insert after "<|im_start|>user\n"
                    insertion_point = last_user_prompt_idx + len(
                        user_prompt_start
                    )
                    processed_text = (
                        processed_text[:insertion_point]
                        + image_tokens
                        + "\n"
                        + processed_text[insertion_point:]
                    )
                else:
                    # Fallback for plain text prompts (no chat template)
                    # or if no user turn is present.
                    processed_text = image_tokens + "\n" + processed_text

            vision_config = getattr(self.config, "vision_config", None)
            patch_size = (
                getattr(vision_config, "patch_size", 14)
                if vision_config
                else 14
            )
            image_array = preprocess_image_to_tensor(
                image,
                input_size=self.image_size,
                max_num=self.max_dynamic_patch,
                patch_size=patch_size,
            )
            # Store the uint16 array (bfloat16 representation)
            raw_pixel_values.append(image_array)

        # Tokenize the processed text
        text_inputs = self.tokenizer(
            processed_text, add_special_tokens=add_special_tokens
        )

        # Add basic pixel values - TextAndVisionTokenizer expects this
        # These are raw image arrays that will be preprocessed later in the model layer
        if raw_pixel_values:
            text_inputs["pixel_values"] = [raw_pixel_values]

            # Compute image token indices for optimization
            input_ids = text_inputs.get("input_ids", [])
            # Handle various input formats (list, nested list, numpy array)
            # by converting to numpy and flattening.
            seq = np.asarray(input_ids).ravel()

            image_context_token_id = _get_image_context_token_id(self.config)
            image_token_indices = (
                (seq == image_context_token_id).nonzero()[0].astype(np.int32)
            )
            text_inputs["image_token_indices"] = image_token_indices

        return text_inputs


class InternVLTokenizer(TextAndVisionTokenizer):
    """InternVL-specific tokenizer that handles tokenizer-only models with custom image processing.

    This class only overrides __init__ to create a custom processor, while inheriting
    all other methods (including new_context) from TextAndVisionTokenizer.
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
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )

        # Set max_length after delegate is created (like parent class)
        self.max_length = max_length or self.delegate.model_max_length

        # Use the pre-loaded HuggingFace config from pipeline_config
        config: Any = pipeline_config.model.huggingface_config

        # Set the vision_token_id for use in super's new_context method
        self.vision_token_ids = [_get_image_context_token_id(config)]

        # Get vision config overrides from pipeline config.
        vision_overrides = pipeline_config.model.vision_config_overrides

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )

        # Create custom processor instead of AutoProcessor (which doesn't exist for InternVL)
        self.processor = InternVLProcessor(
            self.delegate, config, vision_overrides
        )

        # Initialize default EOS token IDs (required by parent class new_context method)
        eos_token_id = self.delegate.eos_token_id
        self._eos_token_ids = (
            {eos_token_id} if eos_token_id is not None else set()
        )

        self._reasoning_parser_name: str | None = (
            pipeline_config.runtime.reasoning_parser
        )
