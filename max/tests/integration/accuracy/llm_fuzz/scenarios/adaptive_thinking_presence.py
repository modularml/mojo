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
Scenario: adaptive thinking must produce a thinking block under concurrency.

Regression guard for CENG-781. With ``thinking: {"type": "adaptive"}`` the model
decides whether to emit a reasoning block; a healthy deployment reliably emits
one for a reasoning-eliciting prompt (MiniMax's own endpoint scores 100/100 on
the equivalent MiniMax-Provider-Verifier ``test_15_04_extreme_agent_thinking``).

This scenario fires many concurrent adaptive-thinking requests and FAILs if the
"no thinking block" rate exceeds a threshold. It runs two paths:

- ``cache_hit_identical``    -- identical requests (prefix cache warms and is
                               reused; the condition under which it reproduced).
- ``cache_miss_unique``      -- a unique counter is prepended to each request so
                               every request is a full prefix-cache MISS.

Comparing the two localizes whether a regression rides on prefix-cache reuse.

Env overrides:
  LLM_FUZZ_ADAPTIVE_THINKING_RUNS   requests per path      (default 100)
  LLM_FUZZ_ADAPTIVE_THINKING_CONC   concurrency            (default 50)
  LLM_FUZZ_ADAPTIVE_THINKING_MAXPCT fail threshold percent (default 1.0)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def _messages(prefix: str) -> list[dict[str, Any]]:
    # Multi-turn tool-calling conversation ending in a reasoning-eliciting ask,
    # mirroring MiniMax-Provider-Verifier test_15_04_extreme_agent_thinking.
    return [
        {
            "role": "system",
            "content": prefix
            + "You are a weather assistant. Always use tools.",
        },
        {"role": "user", "content": "Weather in Beijing?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Beijing"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "25C sunny"},
        {"role": "user", "content": "And Shanghai?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Shanghai"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "28C cloudy"},
        {
            "role": "user",
            "content": "Compare them. Think step by step before answering.",
        },
    ]


def _payload(model: str, prefix: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "stream": True,
        "messages": _messages(prefix),
        "tools": [_WEATHER_TOOL],
        "thinking": {"type": "adaptive"},
        "max_tokens": max_tokens,
    }


def _thinking_present(chunks: list[str]) -> tuple[bool, str | None]:
    """Whether the stream carried any thinking signal.

    Present iff ``<think>`` appears in concatenated content OR any
    ``reasoning_content``/``reasoning`` delta is non-empty (matching the
    verifier's ``assert_thinking_present``). Returns ``(present, finish_reason)``.
    """
    content = ""
    reasoning = ""
    finish: str | None = None
    for raw in chunks:
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
        if delta.get("content"):
            content += delta["content"]
        rc = delta.get("reasoning_content") or delta.get("reasoning")
        if rc:
            reasoning += rc
        if choices[0].get("finish_reason"):
            finish = choices[0]["finish_reason"]
    present = ("<think>" in content.lower()) or bool(reasoning.strip())
    return present, finish


@register_scenario
class AdaptiveThinkingPresence(BaseScenario):
    name = "adaptive_thinking_presence"
    description = (
        "CENG-781: adaptive thinking must yield a thinking block under "
        "concurrency; cache-HIT vs cache-MISS split; "
        "LLM_FUZZ_ADAPTIVE_THINKING_{RUNS,CONC,MAXPCT} to override"
    )
    tags = ["reasoning", "thinking", "concurrency", "kv_cache", "regression"]
    scenario_type = "fuzz"
    # Adaptive thinking (`thinking: {"type": "adaptive"}`) is a MiniMax-M3
    # feature; other models don't support it and would fail this spuriously.
    # Gate it out of the default suite unless `--model-profile minimax-m3` is
    # selected. Naming it explicitly via `--scenarios adaptive_thinking_presence`
    # still runs it regardless of profile.
    model_filter = "minimax-m3"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        model = config.model
        max_tokens = min(config.model_config.max_num_tokens, 8192)
        n = int(os.environ.get("LLM_FUZZ_ADAPTIVE_THINKING_RUNS", "100"))
        if n < 1:
            n = 100
        conc = int(os.environ.get("LLM_FUZZ_ADAPTIVE_THINKING_CONC", "50"))
        if conc < 1:
            conc = 50
        max_pct = float(
            os.environ.get("LLM_FUZZ_ADAPTIVE_THINKING_MAXPCT", "1.0")
        )
        # Reasoning generations are long, so scale the global --timeout up for
        # the streaming read rather than hard-coding a value -- this keeps the
        # --timeout knob effective and avoids long CI hangs on an unhealthy
        # endpoint. (Matches the reasoning/tool-calling scenarios' convention.)
        read_timeout = config.timeout * 4
        nonce = uuid.uuid4().hex[:8]

        async def _one(prefix: str) -> tuple[str, int, str | None]:
            resp = await client.post_streaming(
                _payload(model, prefix, max_tokens),
                read_timeout=read_timeout,
            )
            if resp.error or resp.status != 200:
                return ("err", resp.status or 0, resp.error)
            present, finish = _thinking_present(resp.chunks or [])
            return ("present" if present else "absent", resp.status, finish)

        async def _path(
            test_name: str, prefix_fn: Callable[[int], str]
        ) -> ScenarioResult:
            sem = asyncio.Semaphore(conc)

            async def _guarded(i: int) -> tuple[str, int, str | None]:
                async with sem:
                    return await _one(prefix_fn(i))

            res = await asyncio.gather(*[_guarded(i) for i in range(n)])
            absent = sum(1 for r in res if r[0] == "absent")
            err = sum(1 for r in res if r[0] == "err")
            ok = n - err
            pct = (100.0 * absent / ok) if ok else 0.0
            detail = (
                f"{test_name}: n={n} conc={conc} ok={ok} "
                f"thinking_absent={absent} ({pct:.1f}%) errors={err}"
            )
            if ok == 0:
                verdict = Verdict.ERROR
            elif pct > max_pct:
                verdict = Verdict.FAIL
            elif absent > 0:
                verdict = Verdict.INTERESTING
            else:
                verdict = Verdict.PASS
            print(f"    {detail}")
            return self.make_result(
                self.name, test_name, verdict, detail=detail
            )

        results: list[ScenarioResult] = []
        # cache-HIT: identical prompts; prefix cache warms after the first.
        results.append(await _path("cache_hit_identical", lambda i: ""))
        # cache-MISS: unique counter prepended -> full prefix-cache miss.
        results.append(
            await _path("cache_miss_unique", lambda i: f"[trace {nonce}-{i}] ")
        )
        return results
