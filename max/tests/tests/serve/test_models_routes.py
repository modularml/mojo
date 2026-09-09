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
"""Tests for the max_model_len the /v1/models routes report.

A multi-component checkpoint -- the diffusion pipelines, and now audio
generation -- has no main language model and so no single context length to
report. These routes have to answer anyway, since every API type is mounted
regardless of what is being served.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.serve.request import register_request
from max.serve.router import openai_routes

MODEL = "test-multi-component-model"


class _Tokenizer:
    """A prompt tokenizer, which may or may not bound its own length."""

    def __init__(self, max_length: int | None) -> None:
        self.max_length = max_length


class _Pipeline:
    """The attributes the models routes read off the served pipeline."""

    def __init__(self, tokenizer_max_length: int | None) -> None:
        self.model_name = MODEL
        self.lora_queue = None
        self.tokenizer = _Tokenizer(tokenizer_max_length)


def _app(
    *,
    planned_max_length: int | None,
    tokenizer_max_length: int | None = None,
) -> FastAPI:
    app = FastAPI(title="MAX Serve Test")
    register_request(app)
    app.include_router(openai_routes.router)
    app.state.pipeline = _Pipeline(tokenizer_max_length)
    app.state.memory_plan = MemoryPlan(
        planned_max_batch_size=1,
        footprint=0,
        planned_max_length=planned_max_length,
    )
    return app


def test_a_multi_component_model_reports_no_max_model_len() -> None:
    """Both routes have to answer without a main model to ask."""
    with TestClient(_app(planned_max_length=None)) as client:
        listed = client.get("/v1/models")
        assert listed.status_code == 200
        assert listed.json()["data"][0]["max_model_len"] is None

        named = client.get(f"/v1/models/{MODEL}")
        assert named.status_code == 200
        assert named.json()["max_model_len"] is None


def test_a_model_with_a_planned_length_reports_it() -> None:
    """The control: the null above is the no-main-model case, not the norm."""
    with TestClient(_app(planned_max_length=512)) as client:
        listed = client.get("/v1/models")
        assert listed.json()["data"][0]["max_model_len"] == 512

        named = client.get(f"/v1/models/{MODEL}")
        assert named.json()["max_model_len"] == 512


def test_a_tokenizer_shorter_than_the_plan_lowers_max_model_len() -> None:
    app = _app(planned_max_length=512, tokenizer_max_length=128)
    with TestClient(app) as client:
        assert (
            client.get("/v1/models").json()["data"][0]["max_model_len"] == 128
        )


def test_a_tokenizer_without_a_length_leaves_the_plan_alone() -> None:
    """A media tokenizer bounds the prompt, not the model's context."""
    app = _app(planned_max_length=512, tokenizer_max_length=None)
    with TestClient(app) as client:
        assert (
            client.get("/v1/models").json()["data"][0]["max_model_len"] == 512
        )


def test_an_unknown_model_is_still_a_404() -> None:
    with TestClient(_app(planned_max_length=None)) as client:
        assert client.get("/v1/models/not-the-served-model").status_code == 404
