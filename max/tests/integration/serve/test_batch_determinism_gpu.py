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
"""Serving-level run-to-run determinism gate.

Locks in the invariant that a greedy (``temperature=0``) request with a fixed
seed produces bit-identical output on every run when nothing else changes.
This is the gating half of the determinism/batch-invariance test plan
(``.agentwork/max-serve/designs/determinism-batch-invariance-tests.md``,
Slice 3, item 4); the characterization halves (needle-in-batch and
cached-vs-uncached prefix consistency) live in the ``s33_batch_determinism``
``llm_fuzz`` scenario.

Prefix caching is disabled here so every request recomputes from a cold cache:
this isolates pure run-to-run determinism from the cached-vs-uncached path,
which is a distinct (characterization, not gating) property. Requests are sent
sequentially rather than concurrently so batch composition never varies —
co-batch (batch-invariance) behavior is characterized in the scenario, not
gated.

Uses the ``/v1/completions`` (raw-prompt) route with the base SmolLM2-135M so
the gate does not depend on a chat template (the base model has none).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from async_asgi_testclient import TestClient
from fastapi import FastAPI
from max.driver import DeviceSpec
from max.pipelines import PipelineConfig
from max.pipelines.lib import KVCacheConfig, MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig

_MODEL = "HuggingFaceTB/SmolLM2-135M"
# Number of identical sequential requests to compare. A handful is enough to
# expose a run-to-run divergence while keeping the test fast/stable.
_N_REPEATS = 5
_SEED = 12345
_PROMPT = "The first five prime numbers are"
# A clearly different prompt used as a negative control: its greedy output must
# differ from the gate prompt's, proving the comparison actually detects a
# divergence (i.e. the gate is not vacuously passing).
_OTHER_PROMPT = "Once upon a time in a distant galaxy,"


def _greedy_request(prompt: str, logprobs: bool = False) -> dict[str, Any]:
    """A fixed greedy completion request (temperature=0, fixed seed)."""
    payload: dict[str, Any] = {
        "model": _MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "seed": _SEED,
        "max_tokens": 64,
    }
    if logprobs:
        # /v1/completions logprobs is an int: number of top logprobs per token.
        payload["logprobs"] = 5
    return payload


def _completion_fingerprint(body: str) -> tuple[str, str | None]:
    """Extract (text, finish_reason) from a completion response body.

    ``text`` is the decoded token stream, so equality across runs is the
    token-identity check the gate asserts.
    """
    data = json.loads(body)
    choice = data["choices"][0]
    return choice.get("text") or "", choice.get("finish_reason")


def _logprobs_payload(body: str) -> dict[str, Any] | None:
    """Return the ``logprobs`` object if the completion response exposes it."""
    data = json.loads(body)
    lp = data["choices"][0].get("logprobs")
    if isinstance(lp, dict) and isinstance(lp.get("token_logprobs"), list):
        return lp
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_config",
    [
        PipelineConfig(
            models=ModelManifest(
                {
                    "main": MAXModelConfig(
                        model_path=_MODEL,
                        device_specs=[DeviceSpec.accelerator()],
                        quantization_encoding="bfloat16",
                        # Disable prefix caching so each request is cold: this
                        # isolates run-to-run determinism from the (separately
                        # characterized) cached-vs-uncached prefix path.
                        kv_cache=KVCacheConfig(enable_prefix_caching=False),
                        max_length=512,
                    )
                }
            ),
            runtime=PipelineRuntimeConfig(max_batch_size=16),
        )
    ],
    indirect=True,
)
async def test_batch_determinism_run_to_run(app: FastAPI) -> None:
    """Same greedy request N times sequentially must be bit-identical.

    Gates run-to-run determinism: identical completion tokens across every
    repeat, plus identical per-token logprobs when the route exposes them.
    Includes a negative control proving the comparison detects a divergence.
    """
    # Generous timeout: the first request absorbs the cold MAX graph compile
    # (can be several minutes), not just the (sub-second) SmolLM generation.
    async with TestClient(app, timeout=600.0) as client:
        # --- Gating check: completion tokens (text) must be identical. ---
        contents: list[tuple[str, str | None]] = []
        for _ in range(_N_REPEATS):
            resp = await client.post(
                "/v1/completions", json=_greedy_request(_PROMPT)
            )
            assert resp.status_code == 200, (
                f"request failed: HTTP {resp.status_code}: {resp.text[:500]}"
            )
            contents.append(_completion_fingerprint(resp.text))

        baseline_text, baseline_finish = contents[0]
        assert baseline_text, "baseline produced empty text"
        for i, (text, finish) in enumerate(contents[1:], start=1):
            assert text == baseline_text, (
                f"run-to-run divergence at repeat {i}: text differs from "
                f"repeat 0.\n  repeat 0: {baseline_text!r}\n"
                f"  repeat {i}: {text!r}"
            )
            assert finish == baseline_finish, (
                f"run-to-run divergence at repeat {i}: finish_reason "
                f"{finish!r} != {baseline_finish!r}"
            )

        # --- Negative control: a different prompt must produce different text,
        # proving the equality gate above is sensitive (not vacuously true). ---
        other_resp = await client.post(
            "/v1/completions", json=_greedy_request(_OTHER_PROMPT)
        )
        assert other_resp.status_code == 200, (
            f"negative-control request failed: HTTP {other_resp.status_code}"
        )
        other_text, _ = _completion_fingerprint(other_resp.text)
        assert other_text != baseline_text, (
            "negative control failed: a different prompt produced identical "
            "text, so the determinism gate cannot detect a divergence"
        )

        # --- Best-effort logprobs check (only if the route exposes them). ---
        # MAX's overlap pipeline does not currently support logprobs and
        # returns 400; that is expected and must not fail (or skip) this gate.
        # The token-identity gate above has already passed, so when logprobs
        # are unavailable we return normally rather than skipping the test.
        lp_resp_a = await client.post(
            "/v1/completions", json=_greedy_request(_PROMPT, logprobs=True)
        )
        lp_a = (
            _logprobs_payload(lp_resp_a.text)
            if lp_resp_a.status_code == 200
            else None
        )
        if lp_a is None:
            print(
                "logprobs unavailable on this pipeline "
                f"(HTTP {lp_resp_a.status_code}); token-identity gate passed"
            )
            return

        lp_resp_b = await client.post(
            "/v1/completions", json=_greedy_request(_PROMPT, logprobs=True)
        )
        assert lp_resp_b.status_code == 200, (
            f"second logprobs request failed: HTTP {lp_resp_b.status_code}"
        )
        lp_b = _logprobs_payload(lp_resp_b.text)
        assert lp_b == lp_a, (
            "run-to-run divergence: per-token logprobs differ between two "
            "identical greedy requests"
        )
