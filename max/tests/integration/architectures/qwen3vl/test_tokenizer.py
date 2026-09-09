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

"""Compare SDK Qwen3VLTokenizer outputs to Transformers AutoProcessor outputs."""

from __future__ import annotations

import asyncio
import io
from typing import Any
from unittest.mock import MagicMock, NonCallableMock

import numpy as np
from max.pipelines.architectures.qwen3vl_moe.tokenizer import Qwen3VLTokenizer
from max.pipelines.lib import KVCacheConfig, PipelineRuntimeConfig
from max.pipelines.modeling.types import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from PIL import Image
from transformers import AutoConfig, AutoProcessor


def _create_mock_pipeline_config(model_path: str) -> MagicMock:
    """Create a mock PipelineConfig with real HuggingFace config."""
    # Load real HuggingFace config for proper vision config values
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


def _build_messages(image: Image.Image) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]


def test_qwen3vl_tokenizer() -> None:
    # Create synthetic test image (matching InternVL/Idefics3 pattern)
    img_buffer = io.BytesIO()
    test_pil_image = Image.new("RGB", (448, 448), color="red")
    test_pil_image.save(img_buffer, format="PNG")
    test_image_bytes = img_buffer.getvalue()

    # Transformers reference outputs
    hf_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True
    )
    hf_inputs = hf_processor.apply_chat_template(
        _build_messages(test_pil_image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Convert tensors to numpy for comparison
    hf_input_ids = hf_inputs["input_ids"].cpu().numpy()
    hf_attention_mask = hf_inputs["attention_mask"].cpu().numpy()
    hf_pixel_values = hf_inputs["pixel_values"].cpu().numpy()
    hf_image_grid_thw = hf_inputs["image_grid_thw"].cpu().numpy()

    # SDK tokenizer under test
    model_path = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    pipeline_config = _create_mock_pipeline_config(model_path)

    tokenizer = Qwen3VLTokenizer(
        model_path=model_path,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
    )

    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    ImageContentPart(),
                    TextContentPart(text="Describe this image."),
                ],
            )
        ],
        images=[test_image_bytes],
        request_id=RequestID("test-id"),
        model_name=model_path,
    )

    context = asyncio.run(tokenizer.new_context(request))

    # Extract SDK outputs
    # Use all_tokens (tokens[:end_idx]) instead of full tokens array to exclude resize padding
    # The context resizes tokens to CHUNK_SIZE boundary, so we should only compare the actual tokens
    sdk_input_ids = (
        context.tokens.all
    )  # This is tokens[:end_idx], excluding resize padding
    assert isinstance(sdk_input_ids, np.ndarray)

    assert context.vision_data is not None, (
        "Expected vision_data for image input"
    )
    sdk_pixel_values = context.vision_data.concatenated_pixel_values
    sdk_image_grid_thw = context.vision_data.image_grid_thw

    # Compare input_ids length and infer attention mask expectations
    # hf_input_ids shape is (B, seq_len). SDK input_ids shape is a ragged tensor of shape(total_seq_len,).

    assert sdk_input_ids.shape == (hf_input_ids.shape[1],)
    assert np.array_equal(sdk_input_ids, hf_input_ids.flatten())

    # HF mask should be all ones for single-sample, no padding
    assert hf_attention_mask.shape == (1, hf_input_ids.shape[1])
    assert int(np.sum(hf_attention_mask)) == hf_input_ids.shape[1]

    # Compare vision tensors
    assert sdk_pixel_values.shape == hf_pixel_values.shape
    assert np.array_equal(sdk_image_grid_thw, hf_image_grid_thw)

    # Numerical closeness on pixel values
    assert np.allclose(sdk_pixel_values, hf_pixel_values, rtol=1e-2, atol=1e-2)


def test_qwen3vl_tokenizer_no_images() -> None:
    """Test Qwen3VL tokenizer with text-only input (no images)."""
    # Build text-only messages
    text_only_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the capital of France?"}
            ],
        }
    ]

    # Transformers reference outputs
    hf_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct", trust_remote_code=True
    )
    hf_inputs = hf_processor.apply_chat_template(
        text_only_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Convert tensors to numpy for comparison
    hf_input_ids = hf_inputs["input_ids"].cpu().numpy()
    hf_attention_mask = hf_inputs["attention_mask"].cpu().numpy()

    # SDK tokenizer under test
    model_path = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    pipeline_config = _create_mock_pipeline_config(model_path)

    tokenizer = Qwen3VLTokenizer(
        model_path=model_path,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
    )

    # Build MAX request mirroring the HF messages
    request = TextGenerationRequest(
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[
                    TextContentPart(text="What is the capital of France?")
                ],
            )
        ],
        request_id=RequestID("test-id"),
        model_name=model_path,
    )

    context = asyncio.run(tokenizer.new_context(request))

    # Extract SDK outputs
    # Use all_tokens (tokens[:end_idx]) instead of full tokens array to exclude resize padding
    # The context resizes tokens to CHUNK_SIZE boundary, so we should only compare the actual tokens
    sdk_input_ids = (
        context.tokens.all
    )  # This is tokens[:end_idx], excluding resize padding
    assert isinstance(sdk_input_ids, np.ndarray)

    # For text-only input, vision_data should be None
    assert context.vision_data is None, (
        "Expected vision_data to be None for text-only input"
    )

    # Compare input_ids length and infer attention mask expectations
    # hf_input_ids shape is (B, seq_len). SDK input_ids shape is a ragged tensor of shape(total_seq_len,).

    assert sdk_input_ids.shape == (hf_input_ids.shape[1],)
    assert np.array_equal(sdk_input_ids, hf_input_ids.flatten())

    # HF mask should be all ones for single-sample, no padding
    assert hf_attention_mask.shape == (1, hf_input_ids.shape[1])
    assert int(np.sum(hf_attention_mask)) == hf_input_ids.shape[1]

    # For text-only input, image_token_indices should be empty
    assert context.image_token_indices.shape == (0,), (
        "Expected empty image_token_indices for text-only input"
    )
