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
"""Tests for the echo text-generation pipeline.

The :class:`EchoTransformer` replay logic is checked in-process (no network).
The pipeline construction test verifies the tokenizer path is plumbed through
without resolving a model architecture or downloading weights.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.experimental.cascade import LocalRuntime, TextGenOptions
from max.experimental.cascade.core.pipeline_method import _pipeline_method_scope
from max.experimental.cascade.pipelines.echo_textgen import (
    EchoTextGenPipeline,
    EchoTransformer,
)


@pytest.mark.asyncio
async def test_echo_transformer_replays_prompt() -> None:
    prompt = np.array([10, 20, 30], dtype=np.int32)
    async with LocalRuntime() as rt, _pipeline_method_scope():
        transformer = await rt.deploy(EchoTransformer())
        req = TextGenOptions(num_tokens=3)
        chunks = [c async for c in await transformer.decode(req, prompt)]

    # One token per chunk, replaying the prompt in order.
    assert [np.asarray(c).reshape(-1).tolist() for c in chunks] == [
        [10],
        [20],
        [30],
    ]


@pytest.mark.asyncio
async def test_echo_transformer_cycles_and_matches_num_tokens() -> None:
    prompt = np.array([1, 2], dtype=np.int32)
    async with LocalRuntime() as rt, _pipeline_method_scope():
        transformer = await rt.deploy(EchoTransformer())
        req = TextGenOptions(num_tokens=5)
        tokens = [
            int(np.asarray(c).reshape(-1)[0])
            async for c in await transformer.decode(req, prompt)
        ]

    # Exactly num_tokens tokens, cycling the prompt when it runs out.
    assert tokens == [1, 2, 1, 2, 1]


@pytest.mark.asyncio
async def test_echo_transformer_empty_prompt() -> None:
    prompt = np.array([], dtype=np.int32)
    async with LocalRuntime() as rt, _pipeline_method_scope():
        transformer = await rt.deploy(EchoTransformer())
        req = TextGenOptions(num_tokens=4)
        chunks = [c async for c in await transformer.decode(req, prompt)]

    assert chunks == []


def test_pipeline_builds_tokenizer_from_model_path() -> None:
    # Construction only: the tokenizer worker is seeded from the model path and
    # no model worker is created (config is never resolved, no download).
    pipeline = EchoTextGenPipeline("some-org/some-llm")
    assert pipeline.tokenizer.model_path == "some-org/some-llm"
    assert isinstance(pipeline.transformer, EchoTransformer)
