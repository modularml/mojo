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
"""Functional tests for the dummy text-generation pipeline."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from max.experimental.cascade import (
    ChatMessage,
    GenAIRequest,
    GenAITextChunk,
    LocalRuntime,
    TextGenOptions,
)
from max.experimental.cascade.pipelines.dummy_textgen import (
    build_dummy_textgen_pipeline,
)


@pytest.fixture()
async def runtime() -> AsyncIterator[LocalRuntime]:
    async with LocalRuntime() as rt:
        yield rt


def _request(prompt: str, num_tokens: int) -> GenAIRequest:
    return GenAIRequest(
        messages=[ChatMessage.text("user", prompt)],
        text=TextGenOptions(num_tokens=num_tokens),
    )


@pytest.mark.asyncio
async def test_textgen_pipeline(runtime: LocalRuntime) -> None:
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    chunks = [
        chunk async for chunk in pipeline.generate(_request("hello, ", 5))
    ]

    assert len(chunks) == 5
    assert all(chunk == GenAITextChunk(text="A") for chunk in chunks)


@pytest.mark.asyncio
async def test_textgen_different_lengths(runtime: LocalRuntime) -> None:
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    for num_tokens in [1, 3, 10]:
        chunks = [
            chunk
            async for chunk in pipeline.generate(_request("test", num_tokens))
        ]
        assert len(chunks) == num_tokens
