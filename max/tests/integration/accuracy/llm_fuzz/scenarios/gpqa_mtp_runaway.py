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
Scenario: GPQA MTP token runaway on known diagnostic prompts.

Runs five GPQA-diamond prompts (indices 12, 71, 118, 145, 147) that reliably
separate healthy termination from MTP length-cap runaways in production evals.
Each prompt is sampled ``_SAMPLES`` times at ``temperature=1.0``. By default
messages are a single ``user`` turn (Modular Cloud / OpenAI-compatible gateways).
Set ``GPQA_RUNAWAY_LOCAL_SERVE=1`` to use the ``gpqa_eval`` shape
(root + system + user) against a local MAX Serve. Any ``finish_reason='length'``
is a runaway and fails the case.

Motivated by MTP regressions where Eagle/v5-revert terminate cleanly but v7 MTP
repeatedly hits the output cap on organic-chemistry chain-of-thought items.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import re
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# Match gpqa_eval defaults.
_CAP = 98304
_SAMPLES = 10
_REQUEST_TIMEOUT = 600.0

_ROOT_PREAMBLE = (
    "Your model version is MiniMax-M3, developed by MiniMax. "
    "Knowledge cutoff: January 2026. Founded in early 2022, MiniMax is "
    "a global AI foundation model company committed to advancing the "
    "frontiers of AI towards AGI."
)
_SYSTEM_PROMPT = "You are a helpful assistant."

_DATA_PATH = (
    pathlib.Path(__file__).parent.parent / "data" / "gpqa_runaway_prompts.json"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"answer\s*:\s*([A-D])\b", re.IGNORECASE)


def _load_cases() -> list[dict[str, Any]]:
    with _DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _build_messages(user_prompt: str) -> list[dict[str, str]]:
    if os.environ.get("GPQA_RUNAWAY_LOCAL_SERVE", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return [
            {"role": "root", "content": _ROOT_PREAMBLE},
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    return [{"role": "user", "content": user_prompt}]


def _format_request_error(resp: Any) -> tuple[str, int | None]:
    if resp.error:
        return (resp.error, resp.status or None)
    if resp.status != 200:
        body = (resp.body or "").strip().replace("\n", " ")[:300]
        return (f"HTTP {resp.status}: {body or '(empty body)'}", resp.status)
    return ("unknown error", resp.status)


def _has_final_answer(text: str) -> bool:
    cleaned = _THINK_RE.sub("", text or "").strip()
    return bool(_ANSWER_RE.search(cleaned))


def _extract_finish(resp_body: str) -> tuple[str | None, int | None, str]:
    try:
        data = json.loads(resp_body)
        choice = data["choices"][0]
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        return None, None, ""
    finish_reason = choice.get("finish_reason")
    completion_tokens = data.get("usage", {}).get("completion_tokens")
    message = choice.get("message") or {}
    content = message.get("content") or ""
    return finish_reason, completion_tokens, content


@register_scenario
class GpqaMtpRunaway(BaseScenario):
    name = "gpqa_mtp_runaway"
    description = (
        "Detects GPQA chain-of-thought runaway (finish_reason=length) on five "
        "diagnostic MTP prompts, 10 samples each"
    )
    tags = ["reasoning", "runaway", "gpqa", "correctness", "minimax"]
    model_filter = "minimax-m3"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model
        cases = _load_cases()

        for case in cases:
            messages = _build_messages(case["user_prompt"])
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": _CAP,
                "temperature": 1.0,
                "top_p": 0.95,
            }

            async def one(
                payload: dict[str, Any] = payload,
            ) -> tuple[str | None, int | None, bool, str, int | None]:
                resp = await client.post_json(payload, timeout=_REQUEST_TIMEOUT)
                if resp.error or resp.status != 200:
                    msg, status = _format_request_error(resp)
                    return ("error", None, False, msg, status)
                fr, ct, content = _extract_finish(resp.body)
                return (fr, ct, _has_final_answer(content), "", resp.status)

            outcomes = await asyncio.gather(*[one() for _ in range(_SAMPLES)])
            n_len = sum(1 for fr, _, _, _, _ in outcomes if fr == "length")
            n_err = sum(1 for fr, _, _, _, _ in outcomes if fr == "error")
            n_noans = sum(
                1 for fr, _, ans, _, _ in outcomes if fr != "length" and not ans
            )
            toks = [ct for _, ct, _, _, _ in outcomes if ct is not None]
            max_ct = max(toks) if toks else None
            err_msgs = [
                msg for fr, _, _, msg, _ in outcomes if fr == "error" and msg
            ]
            err_status = next(
                (st for fr, _, _, _, st in outcomes if fr == "error" and st),
                None,
            )
            sample_err = err_msgs[0] if err_msgs else ""

            if n_len > 0:
                verdict = Verdict.FAIL
                detail = (
                    f"RUNAWAY: {n_len}/{_SAMPLES} samples hit "
                    f"finish_reason='length' at cap={_CAP} "
                    f"(errors={n_err}/{_SAMPLES}, no_answer={n_noans}/{_SAMPLES}, "
                    f"max_completion_tokens={max_ct}, prompt_index="
                    f"{case['prompt_index']}). These GPQA items should "
                    f"terminate cleanly; hitting the cap indicates an MTP "
                    f"reasoning runaway."
                )
            elif n_err > 0:
                verdict = Verdict.FAIL
                detail = (
                    f"{n_err}/{_SAMPLES} requests errored before completion "
                    f"(max_completion_tokens={max_ct}, prompt_index="
                    f"{case['prompt_index']}, first_error={sample_err!r})"
                )
            elif n_noans > 0:
                verdict = Verdict.INTERESTING
                detail = (
                    f"terminated but {n_noans}/{_SAMPLES} samples emitted no "
                    f"'Answer: X' (max_completion_tokens={max_ct}, "
                    f"expected ~{case.get('expected_clean_tokens')})"
                )
            else:
                verdict = Verdict.PASS
                detail = (
                    f"all {_SAMPLES} samples terminated with an answer "
                    f"(max_completion_tokens={max_ct}, "
                    f"expected ~{case.get('expected_clean_tokens')})"
                )

            results.append(
                self.make_result(
                    self.name,
                    case["id"],
                    verdict,
                    status_code=err_status,
                    detail=detail,
                    request_body=json.dumps(payload)[:1200],
                )
            )

        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_runaway_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail=(
                    "Server healthy after GPQA runaway scenario"
                    if health.status == 200
                    else (health.error or f"HTTP {health.status}")
                ),
            )
        )
        return results
