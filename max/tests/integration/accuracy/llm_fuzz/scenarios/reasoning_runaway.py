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
Scenario: Reasoning (chain-of-thought) token runaway

Distinct from ``token_runaway`` (which targets STRUCTURED-OUTPUT /
TOOL-CALLING runaways with tiny caps). This scenario targets PLAIN
chain-of-thought runaway on hard long-reasoning prompts: no tools, no
response_format — just a multiple-choice STEM question answered in
thinking mode.

The embedded prompts are real mmlu_pro engineering questions that have a
short, well-defined final answer and should be solved in at most a few
thousand output tokens. The server is expected to finish each with
finish_reason="stop" (and an "answer is (X)") well within the cap. A
response that instead runs to the token cap (finish_reason="length"),
emitting no final answer, is a reasoning runaway: the model failed to
terminate its chain of thought.

Thinking mode is the trigger (with thinking disabled the same prompts
answer in a few hundred tokens), so each request sets
``chat_template_kwargs={"enable_thinking": true}`` to force thinking on
regardless of the server's default. This relies only on a standard
gemma-4 chat template exposing the ``enable_thinking`` switch; no special
``--chat-template`` is required. If the deployed template ignores the
flag (thinking stays off), the scenario will report PASS — it cannot
reproduce a thinking-only runaway against a non-thinking deployment.

Each case records ``expected_clean_tokens`` — the approximate output
length of a clean, terminating answer for that prompt — for context when
triaging. Sampling matches Gemma's recommended thinking config
(temperature=1.0, top_p=0.95, top_k=64). Each prompt is sent ``_SAMPLES``
times because the runaway is probabilistic; the expected result is zero
length-finishes across all samples.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# Gemma's recommended max_gen_toks for thinking mode. A correct answer to these
# questions finishes well under this (see expected_clean_tokens per case).
_CAP = 16384
# Each prompt is sent this many times (runaway is probabilistic at temp=1.0).
_SAMPLES = 4
# Per-request timeout: a runaway runs to _CAP tokens before returning.
_REQUEST_TIMEOUT = 600.0
# Force thinking on regardless of the server's default chat template. The
# runaway is a thinking-mode behavior; with thinking off these prompts answer
# in a few hundred tokens. Relies on a standard gemma-4 template exposing this.
_THINKING_KWARGS = {"enable_thinking": True}

_LETTERS = "ABCDEFGHIJ"

# Real mmlu_pro engineering questions with short, well-defined answers that a
# healthy server solves well within the cap. Observed runaway on
# google/gemma-4-31B-it: the model fails to terminate and runs to the token
# cap (finish_reason="length") without emitting a final answer.
_CASES: list[dict[str, Any]] = [
    {
        "id": "engineering_q21",
        "gold": "A",
        "expected_clean_tokens": 11871,
        "question": (
            "Calculate the starting current and starting torque for the"
            " repulsion-start, induction-run motor described below: Rating:2"
            " hp220 volts4 poles60 cycles Full-load torque = 96.5 oz-ft Maximum"
            " torque as induction motor = 228 oz-ft. r_1 = 0.765 \\OmegaX_0 ="
            " 57.0 \\Omega X_1 = 1.88 \\OmegaK_r = 0.935 X_2 = 1.88 \\Omegar_2"
            " = 1.58 \\Omega a = 4.68C (stator conductors) = 576 k_wl = 0.78r_c"
            " (short-circuited coils) = 0.00745\\Omega a_3 = ratio of"
            " transformation stator to short-circuited coils a_3 ="
            " 56.25r_b(brush and brush contact) = 0.0140 \\Omega Brush-shift"
            " angle = 16°"
        ),
        "options": [
            "486 oz-ft, 30.65 amperes",
            "520 oz-ft, 33 amperes",
            "420 oz-ft, 31.5 amperes",
            "475 oz-ft, 37 amperes",
            "490 oz-ft, 32.5 amperes",
            "510 oz-ft, 27 amperes",
            "450 oz-ft, 28 amperes",
            "500 oz-ft, 35 amperes",
            "530 oz-ft, 29 amperes",
            "560 oz-ft, 26 amperes",
        ],
    },
    {
        "id": "engineering_q41",
        "gold": "D",
        "expected_clean_tokens": 4902,
        "question": (
            "Find the radiation resistance of a single-turn circular loop with a"
            " circumference of (1/4) wavelength."
        ),
        "options": [
            "1.27 ohm",
            "1.07 ohm",
            "0.107 ohm",
            "0.77 ohm",
            "0.87 ohm",
            "0.57 ohm",
            "0.47 ohm",
            "0.67 ohm",
            "0.97 ohm",
            "0.37 ohm",
        ],
    },
    {
        "id": "engineering_q46",
        "gold": "I",
        "expected_clean_tokens": 13525,
        "question": (
            "Steam, at 212°F, is condensing at 198°F on the outer surface of a"
            " bank of pipes with 20 pipes horizontally placed in a row. The"
            " diameter of each 3 ft long pipe is 1 in. Calculate the rate of"
            " condensate."
        ),
        "options": [
            "300 lbm/hr",
            "350 lbm/hr",
            "275 lbm/hr",
            "220 lbm/hr",
            "280 lbm/hr",
            "325 lbm/hr",
            "200 lbm/hr",
            "400 lbm/hr",
            "250 lbm/hr",
            "180 lbm/hr",
        ],
    },
]


def _build_prompt(question: str, options: list[str]) -> str:
    s = question + "\n\n"
    for i, opt in enumerate(options):
        s += f"{_LETTERS[i]}. {opt}\n"
    s += (
        "\nThink step by step, then end with 'The answer is (X)' where X is "
        "the letter of the correct option."
    )
    return s


def _has_final_answer(text: str) -> bool:
    return bool(
        re.search(r"answer is\s*\(?[A-J]\)?", text or "", re.IGNORECASE)
    )


@register_scenario
class ReasoningRunaway(BaseScenario):
    name = "reasoning_runaway"
    description = (
        "Detects plain chain-of-thought token runaway (finish_reason=length) "
        "on hard long-reasoning STEM prompts that should terminate well "
        "within the cap"
    )
    tags = ["reasoning", "runaway", "thinking", "correctness"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        # -- baseline: a trivial prompt must terminate (server-health gate) --
        baseline = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly one word: hello.",
                }
            ],
            "max_tokens": _CAP,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "chat_template_kwargs": _THINKING_KWARGS,
        }
        resp = await client.post_json(baseline, timeout=_REQUEST_TIMEOUT)
        if resp.error or resp.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "baseline_trivial",
                    Verdict.FAIL,
                    status_code=resp.status,
                    detail=f"API error: {resp.error or resp.body[:400]}",
                    request_body=json.dumps(baseline),
                    response_body=resp.body[:800],
                )
            )
        else:
            data = json.loads(resp.body)
            fr = data["choices"][0].get("finish_reason")
            ct = data.get("usage", {}).get("completion_tokens")
            results.append(
                self.make_result(
                    self.name,
                    "baseline_trivial",
                    Verdict.FAIL if fr == "length" else Verdict.PASS,
                    status_code=resp.status,
                    detail=f"finish_reason={fr!r} completion_tokens={ct}",
                    request_body=json.dumps(baseline),
                    response_body=resp.body[:800],
                )
            )

        # -- the runaway cases --
        for case in _CASES:
            prompt = _build_prompt(case["question"], case["options"])
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": _CAP,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "chat_template_kwargs": _THINKING_KWARGS,
            }

            async def one(
                payload: dict[str, Any] = payload,
            ) -> tuple[str | None, int | None, bool]:
                r = await client.post_json(payload, timeout=_REQUEST_TIMEOUT)
                if r.error or r.status != 200:
                    return ("error", None, False)
                d = json.loads(r.body)
                ch = d["choices"][0]
                fr = ch.get("finish_reason")
                ct = d.get("usage", {}).get("completion_tokens")
                content = (ch.get("message") or {}).get("content") or ""
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                )
                return (fr, ct, _has_final_answer(content))

            outcomes = await asyncio.gather(*[one() for _ in range(_SAMPLES)])
            n_len = sum(1 for fr, _, _ in outcomes if fr == "length")
            n_noans = sum(1 for fr, _, ans in outcomes if not ans)
            toks = [ct for _, ct, _ in outcomes if ct is not None]
            max_ct = max(toks) if toks else None

            if n_len > 0:
                verdict = Verdict.FAIL
                detail = (
                    f"RUNAWAY: {n_len}/{_SAMPLES} samples hit"
                    " finish_reason='length' at"
                    f" cap={_CAP} (no_answer={n_noans}/{_SAMPLES},"
                    f" max_completion_tokens={max_ct}). This prompt has a short"
                    " answer and should terminate in"
                    f" ~{case['expected_clean_tokens']} tokens — the model is"
                    " failing to terminate its reasoning."
                )
            elif n_noans > 0:
                verdict = Verdict.INTERESTING
                detail = (
                    f"terminated but {n_noans}/{_SAMPLES} samples emitted no "
                    f"'answer is (X)' (max_completion_tokens={max_ct}, "
                    f"expected ~{case['expected_clean_tokens']})"
                )
            else:
                verdict = Verdict.PASS
                detail = (
                    f"all {_SAMPLES} samples terminated with an answer "
                    f"(max_completion_tokens={max_ct}, "
                    f"expected ~{case['expected_clean_tokens']})"
                )
            results.append(
                self.make_result(
                    self.name,
                    case["id"],
                    verdict,
                    detail=detail,
                    request_body=json.dumps(payload),
                )
            )

        # -- post health check --
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_runaway_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail=(
                    "Server healthy after runaway scenario"
                    if health.status == 200
                    else (health.error or f"HTTP {health.status}")
                ),
            )
        )
        return results
