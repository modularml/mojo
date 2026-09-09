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
"""Checks that constrained turns preserve reasoning and schema enforcement.

Covers fresh response-format, forced-tool, combined, and resumed turns. Each
judged turn must expose reasoning, and every constrained document must conform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._constrained_stream import (
    accumulate_stream,
    constrained_document_errors,
)

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
        "unit": {"type": "string", "enum": ["seconds", "minutes"]},
    },
    "required": ["answer", "unit"],
    "additionalProperties": False,
}

_TOOL_NAME = "record_duration"

_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": "Record a computed duration.",
        "parameters": _SCHEMA,
    },
}

_ASK = (
    "A job runs 3 stages taking 95, 140 and 65 seconds. Two stages run in "
    "parallel and the third runs after. Work out the total wall-clock time, "
    "then report it."
)

_N_PER_CASE = 5
_MAX_OUTPUT_TOKENS = 1024


@dataclass(frozen=True)
class ConstraintCase:
    name: str
    required_tool_name: str | None = None
    tools: bool = False
    force_tool: bool = False
    response_format: bool = False
    resumed: bool = False


def _messages(resumed: bool) -> list[dict[str, object]]:
    if not resumed:
        return [{"role": "user", "content": _ASK}]
    return [
        {"role": "user", "content": "How long did stage one take?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": _TOOL_NAME,
                        "arguments": '{"answer": 95, "unit": "seconds"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "recorded"},
        {"role": "user", "content": _ASK},
    ]


def _payload(model: str, case: ConstraintCase) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "messages": _messages(case.resumed),
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": True, "thinking": True},
    }
    if case.response_format:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "duration", "schema": _SCHEMA},
        }
    if case.tools or case.force_tool:
        body["tools"] = [_TOOL]
    if case.force_tool:
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": _TOOL_NAME},
        }
    return body


CASES = (
    ConstraintCase("fresh_turn_response_format", response_format=True),
    ConstraintCase(
        "fresh_turn_forced_tool",
        force_tool=True,
        required_tool_name=_TOOL_NAME,
    ),
    ConstraintCase(
        "fresh_turn_tools_and_schema", tools=True, response_format=True
    ),
    ConstraintCase(
        "resumed_turn_response_format", response_format=True, resumed=True
    ),
)


@register_scenario
class ReasoningUnderConstraint(BaseScenario):
    name = "reasoning_under_constraint"
    description = (
        "A thinking model under response_format or a forced tool_choice must "
        "still emit reasoning and a conformant document"
    )
    tags = [
        "reasoning",
        "thinking",
        "structured",
        "json",
        "schema",
        "regression",
        "correctness",
    ]
    model_filter = "gemma4"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        verdicts: list[Verdict] = []
        summary_parts: list[str] = []

        for case in CASES:
            reasoned = conformant = truncated = 0
            attempts = 0
            notes: list[str] = []
            for _ in range(_N_PER_CASE):
                attempts += 1
                payload = _payload(config.model, case)
                resp = await client.post_streaming(
                    payload, read_timeout=config.timeout * 2
                )
                if resp.error or resp.status != 200:
                    if len(notes) < 2:
                        notes.append(
                            f"status={resp.status} err={resp.error}"
                            f" body={(resp.body or '')[:120]!r}"
                        )
                    continue
                response = accumulate_stream(resp.chunks or [])
                if response.finish_reason == "length":
                    truncated += 1
                    continue
                if response.reasoning.strip():
                    reasoned += 1
                elif len(notes) < 2:
                    notes.append(
                        "EMPTY reasoning_content"
                        f" (content={response.content[:60]!r})"
                    )
                errs = constrained_document_errors(
                    response,
                    _SCHEMA,
                    content_is_constrained=case.response_format,
                    required_tool_name=case.required_tool_name,
                )
                if errs:
                    if len(notes) < 2:
                        notes.append(str(errs[:2]))
                else:
                    conformant += 1

            judged = attempts - truncated
            detail = (
                f"{case.name}: reasoned {reasoned}/{judged},"
                f" conformant {conformant}/{judged}, truncated={truncated}"
            )
            print(f"    {detail}")
            for note in notes:
                print(f"      {note}")
            summary_parts.append(
                f"{case.name}=[reason {reasoned}/{judged},"
                f" conf {conformant}/{judged}]"
            )
            if judged == 0:
                fv = Verdict.INTERESTING
            elif reasoned != judged or conformant != judged:
                fv = Verdict.FAIL
            else:
                fv = Verdict.PASS
            verdicts.append(fv)
            results.append(
                self.make_result(
                    self.name, f"case_{case.name}", fv, detail=detail
                )
            )

        summary = " ".join(summary_parts)
        print(f"    SUMMARY: {summary}")
        if Verdict.FAIL in verdicts:
            overall = Verdict.FAIL
        elif Verdict.INTERESTING in verdicts:
            overall = Verdict.INTERESTING
        else:
            overall = Verdict.PASS
        results.append(
            self.make_result(
                self.name, "reasoning_under_constraint", overall, detail=summary
            )
        )

        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_reasoning_under_constraint_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
            )
        )
        return results
