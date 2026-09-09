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

from __future__ import annotations

from unittest import mock

import httpx
import pydantic
import pytest
from fastapi import FastAPI
from max.serve.recordreplay import replay, schema


class PostData(pydantic.BaseModel):
    ingredients: list[str]


@pytest.mark.asyncio(loop_scope="module")
async def test_replay() -> None:
    example_post_data = PostData(
        ingredients=["bacon", "lettuce", "tomato", "bread", "mayonnaise"]
    )
    recording: schema.Recording = [
        schema.Transaction(
            request=schema.Request(
                method="GET",
                path="/get-example",
            ),
        ),
        schema.Transaction(
            request=schema.Request(
                method="POST",
                path="/post-example",
                headers=[(b"content-type", b"application/json")],
                body=example_post_data.model_dump_json().encode(),
            ),
        ),
    ]

    get_handler_mock = mock.Mock(return_value="OK")
    post_handler_mock = mock.Mock(return_value="OK")

    app = FastAPI()

    @app.get("/get-example")
    def get_handler() -> str:
        return get_handler_mock()

    @app.post("/post-example")
    def post_handler(post_data: PostData) -> str:
        return post_handler_mock(post_data)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        await replay.replay_recording(recording, client=client)

    get_handler_mock.assert_called_once()
    post_handler_mock.assert_called_once_with(example_post_data)
