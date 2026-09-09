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
"""Tests for the OpenResponses route adapter."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from max.experimental.cascade import LocalRuntime
from max.experimental.cascade.pipelines.dummy_imgen import (
    build_dummy_imgen_pipeline,
)
from max.experimental.cascade.serve.open_responses import build_router
from PIL import Image


@pytest.fixture()
async def runtime() -> AsyncIterator[LocalRuntime]:
    async with LocalRuntime() as rt:
        yield rt


@pytest.fixture()
async def client(runtime: LocalRuntime) -> AsyncIterator[AsyncClient]:
    pipeline = await build_dummy_imgen_pipeline()
    await pipeline.deploy(runtime)

    app = FastAPI()
    app.include_router(build_router(pipeline))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


def _decode_image(b64_data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_data)))


@pytest.mark.asyncio
async def test_non_streaming_response(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "dummy",
            "input": "a beautiful sunset",
            "provider_options": {
                "image": {
                    "height": 128,
                    "width": 128,
                    "steps": 3,
                    "output_format": "jpeg",
                }
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["model"] == "dummy"
    assert body["status"] == "completed"
    assert body["id"].startswith("resp_")

    assert len(body["output"]) == 1
    message = body["output"][0]
    assert message["role"] == "assistant"
    assert message["status"] == "completed"

    assert len(message["content"]) == 1
    image_content = message["content"][0]
    assert image_content["format"] == "jpeg"

    image = _decode_image(image_content["image_data"])
    assert image.size == (128, 128)


@pytest.mark.asyncio
async def test_streaming_response(client: AsyncClient) -> None:
    num_steps = 3
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "dummy",
            "input": "a beautiful sunset",
            "stream": True,
            "provider_options": {
                "image": {
                    "height": 128,
                    "width": 128,
                    "steps": num_steps,
                    "output_format": "jpeg",
                }
            },
        },
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events from the response body.
    events = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            events.append(line[len("data: ") :])

    # Last event should be the [DONE] sentinel.
    assert events[-1] == "[DONE]"

    # The dummy pipeline emits one frame per denoising step; each becomes a
    # typed ResponseResource SSE event.
    chunks = [json.loads(e) for e in events[:-1]]
    assert len(chunks) == num_steps

    # Intermediate chunks are in_progress; the final chunk is completed.
    for chunk in chunks[:-1]:
        assert chunk["status"] == "in_progress"
        assert chunk["output"][0]["status"] == "in_progress"
    assert chunks[-1]["status"] == "completed"
    assert chunks[-1]["output"][0]["status"] == "completed"

    for chunk in chunks:
        assert chunk["object"] == "response"
        assert chunk["model"] == "dummy"
        image_content = chunk["output"][0]["content"][0]
        assert image_content["format"] == "jpeg"
        image = _decode_image(image_content["image_data"])
        assert image.size == (128, 128)


@pytest.mark.asyncio
async def test_multipart_input(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/responses",
        json={
            "model": "dummy",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "a beautiful "},
                        {"type": "input_text", "text": "sunset"},
                    ],
                }
            ],
            "provider_options": {
                "image": {
                    "height": 144,
                    "width": 144,
                    "steps": 2,
                    "output_format": "png",
                }
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    image_content = body["output"][0]["content"][0]
    assert image_content["format"] == "png"
    image = _decode_image(image_content["image_data"])
    assert image.size == (144, 144)
