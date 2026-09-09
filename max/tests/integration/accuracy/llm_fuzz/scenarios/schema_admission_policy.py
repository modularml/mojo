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
"""Pins which JSON Schema requests are served and which are refused.

Serve cases must return conformant content. Refuse cases cover unsatisfiable,
unenforceable, or draft-ambiguous schemas. Reasoning is disabled so the verdict
reflects the compiler rather than model behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._constrained_stream import classify_served_json, grade_admission
from scenarios._schema_admission_cases import CASES, REFUSE, SERVE

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

_N_PER_CASE = 3
_MAX_OUTPUT_TOKENS = 512


_GRADE_TO_VERDICT = {
    "pass": Verdict.PASS,
    "fail": Verdict.FAIL,
    "interesting": Verdict.INTERESTING,
}


def _payload(
    model: str, name: str, schema: dict[str, object], prompt: str
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY a JSON object."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema},
        },
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    }


@register_scenario
class SchemaAdmissionPolicy(BaseScenario):
    name = "schema_admission_policy"
    description = (
        "Pins admission for multi-byte text, enum siblings, and schemas that "
        "are unenforceable or draft-ambiguous"
    )
    tags = [
        "structured",
        "json",
        "schema",
        "admission",
        "regression",
        "correctness",
    ]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        case_verdicts: list[Verdict] = []
        summary_parts: list[str] = []

        for case in CASES:
            refused = served = conformant = truncated = 0
            other: list[str] = []
            for _ in range(_N_PER_CASE):
                resp = await client.post_json(
                    _payload(config.model, case.name, case.schema, case.prompt),
                    timeout=config.timeout * 2,
                )
                if resp.status == 400:
                    refused += 1
                    if case.expect == SERVE and len(other) < 2:
                        other.append(f"400: {(resp.body or '')[:160]!r}")
                    continue
                if resp.status != 200 or resp.error:
                    if len(other) < 2:
                        other.append(f"status={resp.status} err={resp.error}")
                    continue
                served += 1
                grade, detail = classify_served_json(resp.body, case.schema)
                if grade == "truncated":
                    truncated += 1
                elif grade == "conformant":
                    conformant += 1
                elif len(other) < 2 and detail:
                    other.append(detail)

            judged = served - truncated
            fv = _GRADE_TO_VERDICT[
                grade_admission(
                    case.expect,
                    _N_PER_CASE,
                    refused,
                    served,
                    conformant,
                    truncated,
                )
            ]
            if case.expect == REFUSE:
                detail = (
                    f"{case.name}: refused {refused}/{_N_PER_CASE}"
                    f" (expected: {case.why})"
                )
                summary_parts.append(
                    f"{case.name}=[refused {refused}/{_N_PER_CASE}]"
                )
            elif served == _N_PER_CASE and truncated == _N_PER_CASE:
                detail = (
                    f"{case.name}: served {served}/{_N_PER_CASE},"
                    f" truncated {truncated} (expected: {case.why})"
                )
                summary_parts.append(
                    f"{case.name}=[served {served}/{_N_PER_CASE}, truncated]"
                )
            else:
                detail = (
                    f"{case.name}: served {served}/{_N_PER_CASE},"
                    f" conformant {conformant}/{judged}"
                    f" (expected: {case.why})"
                )
                summary_parts.append(
                    f"{case.name}=[served {served}/{_N_PER_CASE},"
                    f" conf {conformant}/{judged}]"
                )
            case_verdicts.append(fv)
            print(f"    {detail}")
            for ex in other:
                print(f"      {ex}")
            results.append(
                self.make_result(
                    self.name, f"case_{case.name}", fv, detail=detail
                )
            )

        summary = " ".join(summary_parts)
        print(f"    SUMMARY: {summary}")
        if Verdict.FAIL in case_verdicts:
            overall = Verdict.FAIL
        elif Verdict.INTERESTING in case_verdicts:
            overall = Verdict.INTERESTING
        else:
            overall = Verdict.PASS
        results.append(
            self.make_result(
                self.name, "schema_admission_policy", overall, detail=summary
            )
        )

        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_schema_admission_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
            )
        )
        return results
