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
Scenario: streaming — reasoning_content must not resume after content starts

Uses a multi-turn + thinking-enabled chat completion stream. Validates that
delta ordering never emits non-empty reasoning_content after non-empty content
has appeared.

Default: 30 identical streams. Override with LLM_FUZZ_STREAMING_REASONING_MONOTONICITY_RUNS.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from scenarios import (
    BaseScenario,
    ScenarioResult,
    Verdict,
    register_scenario,
)

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


def _thinking_stream_payload(
    model: str, max_tokens: int = 65536
) -> dict[str, Any]:
    return {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {
                "role": "user",
                "content": (
                    "I have 4 apples. I give 2 to my friend. "
                    "How many apples do we have now?"
                ),
            },
            {"role": "assistant", "content": "What do you want me to do?"},
            {
                "role": "user",
                "content": "Please think and answer my original question",
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 1,
        "presence_penalty": 0,
        "repetition_penalty": 1,
        "frequency_penalty": 0,
        "top_p": 1,
        "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
    }


def _nonempty_str(val: object) -> bool:
    return isinstance(val, str) and bool(val.strip())


def _analyze_stream_chunks(chunks: list[str]) -> tuple[bool, bool, bool, str]:
    """
    Returns (saw_reasoning, saw_content, violation, detail).

    violation: non-empty reasoning_content in a chunk after some earlier chunk
    had non-empty content (cross-chunk re-entry). Same-chunk boundary (both
    non-empty) is allowed.
    """
    saw_reasoning = False
    saw_content = False
    content_started = False
    violation = False
    detail = ""

    for i, raw in enumerate(chunks):
        if raw == "[DONE]":
            break
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        rc = delta.get("reasoning_content") or delta.get("reasoning")
        ct = delta.get("content")

        if _nonempty_str(rc):
            assert isinstance(rc, str)
            saw_reasoning = True
            if content_started:
                violation = True
                detail = (
                    f"reasoning_content after content at chunk_index={i} "
                    f"(snippet={rc[:120]!r}…)"
                )
                break

        if _nonempty_str(ct):
            saw_content = True
            content_started = True

    if not detail and violation:
        detail = "reasoning resumed after content"

    return saw_reasoning, saw_content, violation, detail


@register_scenario
class StreamingReasoningMonotonicity(BaseScenario):
    name = "streaming_reasoning_monotonicity"
    description = (
        "30x streaming + thinking: reasoning_content must not reappear after "
        "content (chunk order); LLM_FUZZ_STREAMING_REASONING_MONOTONICITY_RUNS to override"
    )
    tags = ["validation", "streaming", "reasoning", "thinking"]
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model
        max_tokens = config.model_config.max_num_tokens
        payload = _thinking_stream_payload(model, max_tokens=max_tokens)

        n_runs = int(
            os.environ.get(
                "LLM_FUZZ_STREAMING_REASONING_MONOTONICITY_RUNS", "30"
            )
        )
        if n_runs < 1:
            n_runs = 30

        # Long max_tokens — allow a generous read timeout for the full stream.
        read_timeout = 600.0

        pass_n = 0
        interesting_n = 0
        fail_n = 0
        first_fail: str = ""

        for run_idx in range(n_runs):
            run = run_idx + 1
            resp = await client.post_streaming(
                payload, read_timeout=read_timeout
            )

            if resp.error == "TIMEOUT":
                fail_n += 1
                if not first_fail:
                    first_fail = f"run {run}: TIMEOUT ({read_timeout}s)"
                continue

            if resp.status != 200:
                fail_n += 1
                if not first_fail:
                    first_fail = (
                        f"run {run}: HTTP {resp.status} "
                        f"{(resp.body or '')[:300]!r}"
                    )
                continue

            chunks = resp.chunks or []
            saw_r, saw_c, violation, vdetail = _analyze_stream_chunks(chunks)

            if violation:
                fail_n += 1
                if not first_fail:
                    first_fail = f"run {run}: {vdetail}"
            elif saw_r and saw_c:
                pass_n += 1
            else:
                interesting_n += 1

            if run % 5 == 0 or run == n_runs:
                print(
                    f"    Progress: {run}/{n_runs} | "
                    f"pass={pass_n} interesting={interesting_n} fail={fail_n}"
                )

        if fail_n > 0:
            verdict = Verdict.FAIL
            detail = (
                f"runs={n_runs} pass={pass_n} interesting={interesting_n} fail={fail_n}"
                f" | first_issue: {first_fail}"
            )
        elif interesting_n > 0:
            verdict = Verdict.INTERESTING
            detail = (
                f"runs={n_runs} pass={pass_n} interesting={interesting_n} "
                f"(no monotonicity violation, but some runs lacked full reasoning→content)"
            )
        else:
            verdict = Verdict.PASS
            detail = f"runs={n_runs} all pass: reasoning→content with no reasoning after content"

        results.append(
            self.make_result(
                self.name,
                f"streaming_reasoning_monotonicity_{n_runs}_runs",
                verdict,
                detail=detail,
            )
        )

        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_streaming_reasoning_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Server still healthy after scenario"
                if health.status == 200
                else (health.error or f"HTTP {health.status}"),
            )
        )

        return results
