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

"""Regression tests for per-request enable_thinking on Qwen3.5.

These exercise the chat_template_options plumbing fixed in commit 96bdd6e102f.
Before that fix, ``Qwen3VLTokenizer.new_context`` dropped
``request.chat_template_options`` on the floor, so OpenAI-style
``chat_template_kwargs`` overrides like ``{"enable_thinking": false}`` were
silently ignored.

The Qwen3.5 chat template only emits a closed empty ``<think></think>`` block
when ``enable_thinking`` is explicitly false; otherwise it emits an open
``<think>`` for the model to fill in. We use that template difference as the
observable signal.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, NonCallableMock

import pytest
from max.pipelines.architectures.qwen3_5.tokenizer import Qwen3_5Tokenizer
from max.pipelines.lib import KVCacheConfig
from max.pipelines.modeling.types import (
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from transformers import AutoConfig

# Text-only tokenizer downloads — config + tokenizer files only, no weights.
_MODEL_PATHS = [
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-9B",
]


def _mock_pipeline_config(model_path: str) -> MagicMock:
    """Build a PipelineConfig stand-in around the real HF config.

    The real HF config is needed because Qwen3VLTokenizer pulls vision config
    fields (spatial_merge_size, num_position_embeddings, image / vision token
    IDs) directly off it. Everything outside ``model.huggingface_config`` and
    ``model.kv_cache.enable_prefix_caching`` is mocked.
    """
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    mock_kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    mock_kv_cache_config.enable_prefix_caching = False

    mock_model_config = MagicMock()
    mock_model_config.huggingface_config = hf_config
    mock_model_config.kv_cache = mock_kv_cache_config

    pipeline_config = MagicMock()
    pipeline_config.model = mock_model_config
    return pipeline_config


def _build_request(
    model_path: str, *, chat_template_options: dict[str, Any] | None
) -> TextGenerationRequest:
    return TextGenerationRequest(
        request_id=RequestID("test-enable-thinking"),
        model_name=model_path,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[TextContentPart(text="What is 2+2?")],
            )
        ],
        chat_template_options=chat_template_options,
    )


@pytest.fixture(scope="module", params=_MODEL_PATHS)
def tokenizer(request: pytest.FixtureRequest) -> Qwen3_5Tokenizer:
    model_path = request.param
    return Qwen3_5Tokenizer(
        model_path=model_path,
        pipeline_config=_mock_pipeline_config(model_path),
        trust_remote_code=True,
    )


def _decoded_prompt(
    tokenizer: Qwen3_5Tokenizer, request: TextGenerationRequest
) -> str:
    """Run new_context end-to-end and decode the resulting tokens.

    ``skip_special_tokens=False`` keeps the ``<think>`` / ``</think>`` markers
    visible in the decoded string.
    """
    context = asyncio.run(tokenizer.new_context(request))
    return tokenizer.delegate.decode(
        context.tokens.all.tolist(), skip_special_tokens=False
    )


def test_new_context_per_request_enable_thinking_false(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """``enable_thinking=False`` produces a closed empty ``<think></think>`` block."""
    prompt = _decoded_prompt(
        tokenizer,
        _build_request(
            tokenizer.model_path,
            chat_template_options={"enable_thinking": False},
        ),
    )
    # The Qwen3.5 chat template emits '<think>\n\n</think>\n\n' only when
    # enable_thinking is explicitly false. The closing </think> in the
    # assistant prefix is the deterministic signal.
    assert "</think>" in prompt, (
        "Per-request enable_thinking=False was not forwarded into the chat "
        f"template. Decoded prompt: {prompt!r}"
    )


def test_new_context_per_request_enable_thinking_true(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """``enable_thinking=True`` leaves the ``<think>`` block open for the model."""
    prompt = _decoded_prompt(
        tokenizer,
        _build_request(
            tokenizer.model_path,
            chat_template_options={"enable_thinking": True},
        ),
    )
    assert "<think>" in prompt
    assert "</think>" not in prompt, (
        "Per-request enable_thinking=True should leave the <think> block "
        f"open. Decoded prompt: {prompt!r}"
    )


def test_new_context_default_enables_thinking(
    tokenizer: Qwen3_5Tokenizer,
) -> None:
    """Qwen3_5Tokenizer defaults to ``enable_thinking=True`` when no options given."""
    prompt = _decoded_prompt(
        tokenizer,
        _build_request(tokenizer.model_path, chat_template_options=None),
    )
    assert "<think>" in prompt
    assert "</think>" not in prompt, (
        "Default request behavior must be enable_thinking=True. "
        f"Decoded prompt: {prompt!r}"
    )


def test_new_context_forwards_logprobs(tokenizer: Qwen3_5Tokenizer) -> None:
    """``request.logprobs``/``request.echo`` reach the context.

    Regression test: ``Qwen3VLTokenizer.new_context`` (and several sibling
    multimodal tokenizers) dropped both fields, so
    ``TextGenerationInputs.enable_log_probs`` stayed ``False`` and logprob
    responses came back empty.
    """
    request = TextGenerationRequest(
        request_id=RequestID("test-logprobs"),
        model_name=tokenizer.model_path,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=[TextContentPart(text="What is 2+2?")],
            )
        ],
        logprobs=2,
        echo=True,
    )
    context = asyncio.run(tokenizer.new_context(request))
    assert context.log_probabilities == 2
    assert context.log_probabilities_echo is True
