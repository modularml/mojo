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
"""
Scenario: parallel same-function tool_call_id uniqueness

N parallel invocations of the same function must each carry a distinct
``tool_call_id`` so an OpenAI client can correlate subsequent ``tool``
role messages back to a single call.

Regression (PixAI repro): Kimi's structural-tag grammar expands
``function_id_template.{index}`` using the tool's position in the
tools list (a build-time constant), so every invocation of the same
function was emitted with identical ``:0`` suffix.  Non-streaming
collapsed N calls to 1 via id-only dedup; streaming surfaced N
DeltaToolCall entries that all shared one id.

Repro: https://gist.github.com/alanjinqs/b60667a5d30c6008259fd1ed75ca311f
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from helpers import make_tool

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

_WEATHER_TOOL = make_tool(
    "get_weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def _parallel_payload(model: str, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Get weather for Tokyo, Seattle, Oxford in parallel.",
            }
        ],
        "tools": [_WEATHER_TOOL],
        "max_tokens": 256,
        "stream": stream,
    }


def _collect_stream_ids(chunks: list[str]) -> dict[int, str]:
    """Map delta index → first non-empty id seen for that index."""
    ids: dict[int, str] = {}
    for line in chunks or []:
        if line == "[DONE]":
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        for tc in (choices[0].get("delta") or {}).get("tool_calls") or []:
            tc_id = tc.get("id")
            idx = tc.get("index")
            if tc_id and idx is not None and idx not in ids:
                ids[idx] = tc_id
    return ids


def _verdict_for_ids(ids: list[str], label: str) -> tuple[Verdict, str]:
    """Classify a list of tool_call ids for parallel-invocation uniqueness."""
    if len(ids) < 2:
        # Model chose not to parallelize — record as INTERESTING, not
        # FAIL, since the contract only applies when parallelization
        # actually happens.
        return (
            Verdict.INTERESTING,
            f"{label}: {len(ids)} calls, cannot test uniqueness",
        )
    if any(not i for i in ids):
        return Verdict.FAIL, f"{label}: missing/empty ids: {ids}"
    if len(set(ids)) != len(ids):
        return Verdict.FAIL, f"{label}: duplicate ids for parallel calls: {ids}"
    return Verdict.PASS, f"{label}: {len(ids)} parallel calls, all ids unique"


@register_scenario
class ParallelToolCallIdUniqueness(BaseScenario):
    name = "parallel_tool_call_ids"
    description = (
        "Parallel invocations of the same function must emit distinct "
        "tool_call ids (non-streaming + streaming)"
    )
    tags = ["tools", "function_calling", "openai_spec"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        # ----- Non-streaming -----
        resp = await client.post_json(_parallel_payload(model, stream=False))
        if resp.status != 200:
            verdict, detail = (
                Verdict.FAIL,
                f"HTTP {resp.status} error={resp.error or ''}",
            )
        else:
            try:
                data = json.loads(resp.body)
                tool_calls = (data.get("choices") or [{}])[0].get(
                    "message", {}
                ).get("tool_calls") or []
                ids = [tc.get("id") or "" for tc in tool_calls]
                verdict, detail = _verdict_for_ids(ids, "non-streaming")
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                verdict, detail = Verdict.FAIL, f"failed to parse response: {e}"
        results.append(
            self.make_result(
                self.name,
                "nonstreaming_parallel_unique_ids",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- Streaming -----
        resp = await client.post_streaming(
            _parallel_payload(model, stream=True)
        )
        if resp.status != 200:
            verdict, detail = (
                Verdict.FAIL,
                f"HTTP {resp.status} error={resp.error or ''}",
            )
        else:
            id_by_index = _collect_stream_ids(resp.chunks or [])
            verdict, detail = _verdict_for_ids(
                list(id_by_index.values()),
                f"streaming (indices={sorted(id_by_index.keys())})",
            )
        results.append(
            self.make_result(
                self.name,
                "streaming_parallel_unique_ids",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        return results
