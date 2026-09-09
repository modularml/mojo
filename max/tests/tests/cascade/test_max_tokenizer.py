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
"""Tests for the cascade ``MAXTokenizer`` worker.

The ``encode``/``decode`` paths run against a real (small) HuggingFace
tokenizer -- ``HuggingFaceTB/SmolLM2-135M-Instruct`` -- deployed on the local
runtime, and are checked against the reference ``transformers`` tokenizer.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.experimental.cascade import (
    ChatMessage,
    GenAIImageChunk,
    GenAITextChunk,
    LocalRuntime,
)
from max.experimental.cascade.core.pipeline_method import _pipeline_method_scope
from max.experimental.cascade.workers.max_tokenizer import MAXTokenizer
from max.pipelines.lib import generate_local_model_path
from transformers import AutoTokenizer, PreTrainedTokenizerBase

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MESSAGES = [ChatMessage.text("user", "What is the capital of France?")]


def _model_path() -> str:
    """Resolve a cached local path for the test model, else the repo id."""
    try:
        return generate_local_model_path(REPO_ID)
    except FileNotFoundError:
        # Not pre-cached; fall back to the repo id so the HF hub downloads it
        # (requires network; the bazel target is tagged ``requires-network``).
        return REPO_ID


@pytest.fixture(scope="module")
def model_path() -> str:
    return _model_path()


@pytest.mark.asyncio
async def test_open_loads_real_tokenizer(model_path: str) -> None:
    tokenizer = MAXTokenizer(model_path)
    assert tokenizer._tokenizer is None
    async with tokenizer.open():
        assert isinstance(tokenizer._tokenizer, PreTrainedTokenizerBase)


def _reference_ids(reference: PreTrainedTokenizerBase) -> list[int]:
    """Chat-template ids the worker is expected to reproduce."""
    return list(
        reference.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
    )


@pytest.mark.asyncio
async def test_encode_chat_messages_matches_reference(model_path: str) -> None:
    reference = AutoTokenizer.from_pretrained(model_path)
    async with LocalRuntime() as rt, _pipeline_method_scope():
        tokenizer = await rt.deploy(MAXTokenizer(model_path))
        tokens = await (await tokenizer.encode(MESSAGES))
    assert tokens.dtype == np.int32
    assert tokens.tolist() == _reference_ids(reference)


@pytest.mark.asyncio
async def test_encode_joins_multipart_text(model_path: str) -> None:
    reference = AutoTokenizer.from_pretrained(model_path)
    split = [
        ChatMessage(
            role="user",
            content=[
                GenAITextChunk(text="What is the capital "),
                GenAITextChunk(text="of France?"),
            ],
        )
    ]
    async with LocalRuntime() as rt, _pipeline_method_scope():
        tokenizer = await rt.deploy(MAXTokenizer(model_path))
        tokens = await (await tokenizer.encode(split))
    # Text parts concatenate, so a split message tokenizes as one whole.
    assert tokens.tolist() == _reference_ids(reference)


@pytest.mark.asyncio
async def test_decode_roundtrip(model_path: str) -> None:
    reference = AutoTokenizer.from_pretrained(model_path)
    async with LocalRuntime() as rt, _pipeline_method_scope():
        tokenizer = await rt.deploy(MAXTokenizer(model_path))
        tokens = await (await tokenizer.encode(MESSAGES))
        decoded = await (await tokenizer.decode(tokens, True))
    # The worker wraps the HF tokenizer, so its decode must match the reference
    # decode of the same ids.
    assert decoded == reference.decode(
        _reference_ids(reference), skip_special_tokens=True
    )


@pytest.mark.asyncio
async def test_encode_rejects_non_text_content(model_path: str) -> None:
    images = [
        ChatMessage(
            role="user", content=[GenAIImageChunk(url="http://example/x.png")]
        )
    ]
    async with LocalRuntime() as rt, _pipeline_method_scope():
        tokenizer = await rt.deploy(MAXTokenizer(model_path))
        with pytest.raises(ValueError, match="text content only"):
            await (await tokenizer.encode(images))


@pytest.mark.asyncio
async def test_encode_before_deploy_raises() -> None:
    tokenizer = MAXTokenizer(REPO_ID)
    with pytest.raises(AssertionError, match="must be deployed"):
        await tokenizer.encode(MESSAGES)
