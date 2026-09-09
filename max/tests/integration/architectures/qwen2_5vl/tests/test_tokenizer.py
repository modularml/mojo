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

"""Tests for Qwen2.5VL tokenizer."""

from unittest.mock import MagicMock, NonCallableMock

import pytest
from max.pipelines.architectures.qwen2_5vl.tokenizer import Qwen2_5VLTokenizer
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from pytest_mock import MockerFixture
from transformers import Qwen2_5_VLConfig


def _create_mock_huggingface_config() -> NonCallableMock:
    """Create a mock HuggingFace config with spec=Qwen2_5_VLConfig.

    Using spec ensures that ONLY attributes present on the real Qwen2_5_VLConfig
    are accessible. This prevents tests from passing when code incorrectly
    accesses attributes that don't exist on the real config type.
    """
    mock_hf_config = NonCallableMock(spec=Qwen2_5_VLConfig)

    # Set up required attributes that the tokenizer accesses from HuggingFace config
    mock_hf_config.eos_token_id = [151645, 151643]
    mock_hf_config.image_token_id = 151655
    mock_hf_config.video_token_id = 151656
    mock_hf_config.vision_start_token_id = 151652

    # Set up vision_config with required attributes
    mock_vision_config = NonCallableMock()
    mock_vision_config.patch_size = 14
    mock_vision_config.window_size = 448
    mock_vision_config.temporal_patch_size = 2
    mock_vision_config.spatial_merge_size = 2
    mock_vision_config.tokens_per_second = 50
    mock_hf_config.vision_config = mock_vision_config

    return mock_hf_config


class MockModelConfig(MAXModelConfig):
    def __init__(self) -> None:
        # Mirror MockPipelineConfig below: seed pydantic state from
        # model_construct instead of assigning fields (the class is frozen).
        base = MAXModelConfig.model_construct(
            kv_cache=KVCacheConfig(enable_prefix_caching=True)
        )
        self.__dict__.update(base.__dict__)
        for attr in (
            "__pydantic_fields_set__",
            "__pydantic_extra__",
            "__pydantic_private__",
        ):
            if hasattr(base, attr):
                object.__setattr__(self, attr, getattr(base, attr))
        self._huggingface_config = _create_mock_huggingface_config()


class MockPipelineConfig(PipelineConfig):
    def __init__(self) -> None:
        mock_model = MockModelConfig()
        manifest = ModelManifest({"main": mock_model})
        base = PipelineConfig.model_construct(models=manifest)
        self.__dict__.update(base.__dict__)
        for attr in (
            "__pydantic_fields_set__",
            "__pydantic_extra__",
            "__pydantic_private__",
        ):
            if hasattr(base, attr):
                object.__setattr__(self, attr, getattr(base, attr))


@pytest.mark.asyncio
async def test_qwen2_5vl_tokenizer_initialization() -> None:
    """Test tokenizer initialization."""

    pipeline_config = MockPipelineConfig()
    tokenizer = Qwen2_5VLTokenizer(
        "HuggingFaceM4/Idefics3-8B-Llama3", pipeline_config=pipeline_config
    )
    assert tokenizer.image_token_id == 151655
    assert tokenizer.video_token_id == 151656
    assert tokenizer.vision_start_token_id == 151652
    assert tokenizer.enable_prefix_caching is True
    assert tokenizer.tokens_per_second == 50


@pytest.mark.asyncio
async def test_qwen2_5vl_tokenizer_new_context_smoke(
    mocker: MockerFixture,
) -> None:
    """Smoke test to ensure new_context() doesn't raise"""

    mock_tok = MagicMock()
    mock_tok.model_max_length = 2048
    mock_tok.eos_token_id = 2
    mock_tok.apply_chat_template.return_value = "test prompt"
    mock_tok.return_value = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
    }
    mocker.patch(
        "max.pipelines.architectures.qwen2_5vl.tokenizer.AutoTokenizer.from_pretrained",
        return_value=mock_tok,
    )

    pipeline_config = MockPipelineConfig()
    tokenizer = Qwen2_5VLTokenizer("test-model", pipeline_config)

    request = TextGenerationRequest(
        messages=[TextGenerationRequestMessage(role="user", content="test")],
        request_id=RequestID("test-id"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)
    assert context is not None
