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
"""Functional tests for the dummy image-generation pipeline."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator

import pytest
from max.experimental.cascade import (
    ChatMessage,
    GenAIImageChunk,
    GenAIRequest,
    ImageGenOptions,
    LocalRuntime,
)
from max.experimental.cascade.pipelines.dummy_imgen import (
    build_dummy_imgen_pipeline,
)
from PIL import Image


@pytest.fixture()
async def runtime() -> AsyncIterator[LocalRuntime]:
    async with LocalRuntime() as rt:
        yield rt


def _request(prompt: str, num_steps: int) -> GenAIRequest:
    return GenAIRequest(
        messages=[ChatMessage.text("user", prompt)],
        image=ImageGenOptions(
            width=64, height=64, num_steps=num_steps, output_format="JPEG"
        ),
    )


@pytest.mark.asyncio
async def test_imgen_pipeline(runtime: LocalRuntime) -> None:
    pipeline = await build_dummy_imgen_pipeline()
    await pipeline.deploy(runtime)

    chunks = [
        chunk
        async for chunk in pipeline.generate(_request("a beautiful sunset", 3))
    ]

    final = chunks[-1]
    assert isinstance(final, GenAIImageChunk)
    assert final.format == "JPEG"
    assert final.data is not None
    image = Image.open(io.BytesIO(final.data))
    assert image.size[0] > 0
    assert image.size[1] > 0


@pytest.mark.asyncio
async def test_imgen_pipeline_streaming(runtime: LocalRuntime) -> None:
    pipeline = await build_dummy_imgen_pipeline()
    await pipeline.deploy(runtime)

    num_steps = 3
    chunks = [
        chunk async for chunk in pipeline.generate(_request("a cat", num_steps))
    ]
    # The denoiser emits one frame per denoising step and the downstream
    # streaming workers forward the stream verbatim.
    assert len(chunks) == num_steps
    for chunk in chunks:
        assert isinstance(chunk, GenAIImageChunk)
        assert chunk.data is not None
        image = Image.open(io.BytesIO(chunk.data))
        assert image.size == (64, 64)
