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

"""Test that chat template properly handles multi-part content with images and text."""

import base64
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, NonCallableMock

import numpy as np
from max.pipelines.architectures.qwen2_5vl.tokenizer import Qwen2_5VLTokenizer
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from max.pipelines.modeling.types.pipeline_variants.text_generation import (
    ImageContentPart,
    TextContentPart,
    TextGenerationRequestMessage,
)
from PIL import Image
from transformers import AutoConfig


def _create_mock_pipeline_config(model_path: str) -> Any:
    """Create a mock PipelineConfig with real HuggingFace config."""
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    mock_kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    mock_kv_cache_config.enable_prefix_caching = False

    mock_model_config = MagicMock()
    mock_model_config.huggingface_config = hf_config
    mock_model_config.kv_cache = mock_kv_cache_config

    pipeline_config = MagicMock()
    pipeline_config.model = mock_model_config
    # A real runtime config, not a MagicMock: the tokenizer sizes its
    # preprocessed image cache from these budgets, and arithmetic on a
    # MagicMock raises.
    pipeline_config.runtime = PipelineRuntimeConfig()
    return pipeline_config


def test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Creates a test image for testing."""
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def test_chat_template_preserves_text_with_image() -> None:
    """Test that apply_chat_template preserves text content when image is present.

    This is a regression test for PAQ-1381 where text content was being dropped
    when processing multi-part content containing both images and text.
    """
    # Create tokenizer.
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    pipeline_config = _create_mock_pipeline_config(model_path)
    tokenizer = Qwen2_5VLTokenizer(
        model_path=model_path,
        pipeline_config=pipeline_config,
    )

    # Create message structure that mimics OpenAI API format.
    test_question = (
        "When a spring does work on an object, we cannot find the work by "
        "simply multiplying the spring force by the object's displacement. "
        "What is the compression distance?"
    )

    messages: list[TextGenerationRequestMessage] = [
        TextGenerationRequestMessage(
            role="system",
            content="You are a helpful assistant.",
        ),
        TextGenerationRequestMessage(
            role="user",
            content=[
                ImageContentPart(),
                TextContentPart(text=test_question),
            ],
        ),
    ]

    # Apply chat template.
    prompt = tokenizer.apply_chat_template(messages)

    # The text content must be present in the prompt.
    assert test_question in prompt, (
        f"Text content was dropped from chat template output!\n"
        f"Expected to find: {test_question}\n"
        f"But prompt was: {prompt}"
    )

    # Check to ensure structure is correct.
    assert "<|im_start|>system" in prompt, "Missing system message marker"
    assert "You are a helpful assistant." in prompt, (
        "Missing system message content"
    )
    assert "<|im_start|>user" in prompt, "Missing user message marker"
    assert "<|vision_start|>" in prompt, "Missing vision start marker"
    assert "<|image_pad|>" in prompt, "Missing image placeholder"
    assert "<|vision_end|>" in prompt, "Missing vision end marker"

    # The question should come after the vision tokens.
    vision_end_idx = prompt.index("<|vision_end|>")
    question_idx = prompt.index(test_question)
    assert question_idx > vision_end_idx, (
        "Text content should appear after vision tokens, but found in wrong order"
    )


def test_chat_template_multiple_text_parts() -> None:
    """Test that chat template handles multiple text parts in content."""
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    pipeline_config = _create_mock_pipeline_config(model_path)
    tokenizer = Qwen2_5VLTokenizer(
        model_path=model_path,
        pipeline_config=pipeline_config,
    )

    text_part_1 = "First part of the question."
    text_part_2 = "Second part of the question."

    messages: list[TextGenerationRequestMessage] = [
        TextGenerationRequestMessage(
            role="user",
            content=[
                TextContentPart(text=text_part_1),
                ImageContentPart(),
                TextContentPart(text=text_part_2),
            ],
        ),
    ]

    prompt = tokenizer.apply_chat_template(messages)

    # Both text parts must be present.
    assert text_part_1 in prompt, (
        f"First text part missing from prompt: {prompt}"
    )
    assert text_part_2 in prompt, (
        f"Second text part missing from prompt: {prompt}"
    )


def test_chat_template_text_only() -> None:
    """Test that chat template still works correctly for text-only messages."""
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    pipeline_config = _create_mock_pipeline_config(model_path)
    tokenizer = Qwen2_5VLTokenizer(
        model_path=model_path,
        pipeline_config=pipeline_config,
    )

    test_question = "What is 2 + 2?"

    messages: list[TextGenerationRequestMessage] = [
        TextGenerationRequestMessage(
            role="user",
            content=test_question,
        ),
    ]

    prompt = tokenizer.apply_chat_template(messages)

    # Text must be present for simple text-only case.
    assert test_question in prompt, (
        f"Text content missing from prompt: {prompt}"
    )
    assert "<|im_start|>user" in prompt, "Missing user message marker"
