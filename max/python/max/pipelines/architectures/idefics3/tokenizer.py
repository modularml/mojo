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

"""Idefics3-specific tokenizer implementation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TokenBuffer,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import TextAndVisionTokenizer
from max.pipelines.lib.tokenizer import encode_dkv_cache_hint, open_image
from max.pipelines.modeling.types import (
    ImageContentPart,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.support.image import find_contiguous_ranges, hash_image
from PIL.Image import Image as ImageType
from transformers import AutoProcessor, AutoTokenizer

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig


class Idefics3Tokenizer(TextAndVisionTokenizer):
    """Idefics3-specific tokenizer. This class has new_context modified to use the transformers tokenizer and processor.

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
        config = pipeline_config.model.huggingface_config

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )

        if vision_token_id := getattr(config, "image_token_id", None):
            self.vision_token_ids = [vision_token_id]
        else:
            raise ValueError("image_token_id not found in model config")

        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=revision
        )

        # Initialize default EOS token IDs (required by parent class new_context method)
        eos_token_id = self.delegate.eos_token_id
        self._eos_token_ids = (
            {eos_token_id} if eos_token_id is not None else set()
        )

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]] | int, **kwargs
    ) -> str:
        """Decode token array back into readable text, filtering out special tokens."""
        # Log-probability responses decode one token id (a plain int) at a
        # time; match the text tokenizer's handling.
        if isinstance(encoded, int):
            encoded = np.array(encoded)
        # Force skip_special_tokens=True to filter out tokens like <end_of_utterance>
        kwargs_with_special_filter = kwargs.copy()
        kwargs_with_special_filter["skip_special_tokens"] = True
        return self.delegate.decode(
            encoded.tolist(), **kwargs_with_special_filter
        )

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Apply the chat template to the messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Content can be a string or list of multimodal content parts.
            tools: Optional tools available for the model to invoke.
            **chat_template_options: Template options to forward to the Jinja
                template. Merged with ``add_generation_prompt=True`` default.

        Returns:
            The formatted prompt string with chat template applied.

        Raises:
            ValueError: If template application fails.
        """
        chat_template_options = {
            "add_generation_prompt": True,
            **chat_template_options,
        }

        text_messages: list[dict[str, Any]] = []
        for message in messages:
            text_message: dict[str, Any] = {"role": message.role}
            content = message.content

            if isinstance(content, str):
                text_message["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                image_parts: list[dict[str, Any]] = []
                text_parts: list[dict[str, Any]] = []
                for item in content:
                    if isinstance(item, ImageContentPart):
                        image_parts.append({"type": "image"})
                    elif isinstance(item, TextContentPart) and item.text:
                        text_parts.append({"type": "text", "text": item.text})
                # Idefics3 was fine-tuned with images before text:
                # User:<image>text<end_of_utterance>\nAssistant:
                text_context = image_parts + text_parts
                if not text_context:
                    raise ValueError(
                        f"Message from '{message.role}' has no usable content"
                    )
                text_message["content"] = text_context
            else:
                raise ValueError(f"Unexpected content type: {type(content)}")

            text_messages.append(text_message)

        return self.processor.apply_chat_template(
            text_messages,
            tokenize=False,
            tools=tools,
            **chat_template_options,
        )

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext object, leveraging necessary information from TextGenerationRequest."""

        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages:
            prompt = self.apply_chat_template(
                request.messages,
                request.tools,
                **(request.chat_template_options or {}),
            )
            add_special_tokens = False
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        # Convert image bytes to PIL Image objects (open_image reuses the
        # API server's decode-once result, or decodes raw bytes as a fallback).
        if request.images:
            images = []
            for image in request.images_for_processing():
                try:
                    img: ImageType = open_image(image)
                    # Ensure image is in RGB format to avoid channel format issues
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    # Validate image has reasonable dimensions
                    if img.size[0] == 0 or img.size[1] == 0:
                        raise ValueError(
                            f"Invalid image dimensions: {img.size}"
                        )
                    images.append(img)
                except Exception as e:
                    raise ValueError(f"Failed to process image: {e}") from e
        else:
            images = None

        processed_inputs = self.processor(
            text=prompt,
            images=images,
            add_special_tokens=add_special_tokens,
            return_tensors="np",
        )

        if "input_ids" not in processed_inputs:
            raise ValueError(
                "input_ids not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
            )

        if isinstance(processed_inputs["input_ids"][0], int):
            encoded_prompt = np.array(processed_inputs["input_ids"])
        else:
            encoded_prompt = np.array(processed_inputs["input_ids"][0])

        # TODO(zheng): We should probably just make max_new_tokens an optional
        # instead of -1.
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        extra_model_args = dict()

        if images is not None:
            if "pixel_values" not in processed_inputs:
                raise ValueError(
                    "pixel_values not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
                )
            pixel_values = processed_inputs["pixel_values"]

            if isinstance(pixel_values, np.ndarray):
                # Handle the extra batch dimension that return_tensors="np" adds
                if pixel_values.ndim == 5:
                    # Remove the extra batch dimension (1, N, C, H, W) -> (N, C, H, W)
                    pixel_values = pixel_values.squeeze(0)
                # Convert from (N, C, H, W) -> [(C, H, W), ...] with len(N)
                pixel_values = list(pixel_values)

            if not isinstance(pixel_values, list):
                raise ValueError(
                    f"pixel_values is not a list but it is {type(pixel_values)}"
                )
        else:
            pixel_values = []

        # Pass through image token indices if present
        if "image_token_indices" in processed_inputs:
            extra_model_args["image_token_indices"] = processed_inputs[
                "image_token_indices"
            ]

        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format
            and request.response_format.json_schema is not None
            else None
        )

        if self.max_length and encoded_prompt.shape[0] > self.max_length:
            raise PromptTooLongError(encoded_prompt.shape[0], self.max_length)

        start_and_end_idxs = find_contiguous_ranges(
            encoded_prompt, self.vision_token_ids
        )
        if len(start_and_end_idxs) != len(pixel_values):
            raise ValueError(
                f"Number of image token indices ({len(start_and_end_idxs)}) does not match number of pixel values ({len(pixel_values)})"
            )

        token_buffer = TokenBuffer(
            array=encoded_prompt.astype(np.int64, copy=False),
        )

        context = TextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            extra_model_args=extra_model_args,
            tokens=token_buffer,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            images=[
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixels,
                    image_hash=hash_image(pixels)
                    if self.enable_prefix_caching
                    else None,
                )
                for (start_idx, end_idx), pixels in zip(
                    start_and_end_idxs, pixel_values, strict=True
                )
            ],
            vision_token_ids=self.vision_token_ids,
            vocab_size=self.tokenizer_vocab_size,
            cache_salt=request.cache_salt,
        )
        return context


def max_tokens_to_generate(
    prompt_size: int,
    max_length: int | None,
    max_new_tokens: int | None = None,
) -> int | None:
    """Returns the max number of new tokens to generate."""
    if max_length is None:
        return max_new_tokens
    _difference_between_max_and_prompt = max(max_length - prompt_size, 0)
    if max_new_tokens is None:
        return _difference_between_max_and_prompt
    return min(max_new_tokens, _difference_between_max_and_prompt)
