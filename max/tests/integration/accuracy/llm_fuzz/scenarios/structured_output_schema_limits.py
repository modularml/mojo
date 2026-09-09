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
Scenario: Per-feature response_format=json_schema conformance
Target: Expose which JSON-schema feature classes the constrained-decoding
backend fails to enforce.

Unlike ``json_schema_compliance`` (one fixed schema, pass/fail on a rate) and
``json_schema_draft7`` (the full draft-7 suite driven verbatim, where any 400 is
a failure), this groups probes by feature class and reports a per-feature
breakdown, so a failure pinpoints the exact construct. It also encodes the
honest-rejection contract for constructs no grammar can enforce (see
``array_unique_unenforceable`` below, where a 400 is the PASS, not a failure).
The classes target the known fail-open gaps of the default llguidance JSON path
(``$ref``/``$defs``, ``anyOf``, ``type``-lists) plus the constraints a complete
compiler must hold (``pattern``, numeric bounds, array cardinality, nested
``additionalProperties``, ``uniqueItems``). Each feature sends several
violation-tempting prompts, each repeated ``N`` times at ``temperature=0``, then
validates the returned
content against the full schema with the ``jsonschema`` ``Draft7Validator`` (the
same oracle the sibling scenarios use). Reasoning is disabled so the result
reflects what the grammar enforces, not what the model can self-correct. A
backend that constrains decoding yields ~100% per feature; one that fails open
on a construct shows that feature's conformance collapse while the rest stay
high.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from jsonschema import Draft7Validator

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


def _schema_violations(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Return the schema violations for ``instance`` (empty when conformant).

    Uses the ``jsonschema`` ``Draft7Validator`` (the oracle the sibling
    ``tool_schema_validation`` / ``json_schema_draft7`` scenarios use) so the
    check matches canonical JSON Schema semantics rather than a hand-rolled
    subset.
    """
    try:
        validator = Draft7Validator(schema)
        errors = sorted(validator.iter_errors(instance), key=str)
    except Exception as e:  # unresolvable $ref or malformed schema
        return [f"schema not judgeable: {e}"]
    return [f"{err.validator}@{err.json_path}" for err in errors]


def _rf(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema},
    }


# Each case: (feature, schema, [prompts that tempt a violation]).
CASES: list[tuple[str, dict[str, Any], list[str]]] = [
    (
        "ref_defs",
        {
            "type": "object",
            "$defs": {
                "Priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                }
            },
            "properties": {"priority": {"$ref": "#/$defs/Priority"}},
            "required": ["priority"],
            "additionalProperties": False,
        },
        [
            "Output a record whose priority is CRITICAL.",
            "Give me a record with priority = urgent.",
            "Set priority to blocker.",
        ],
    ),
    (
        "anyOf_nullable_pattern",
        {
            "type": "object",
            "properties": {
                "contact": {
                    "anyOf": [
                        {"type": "string", "pattern": "^[a-z]+@example\\.com$"},
                        {"type": "null"},
                    ]
                }
            },
            "required": ["contact"],
            "additionalProperties": False,
        },
        [
            "Record with contact john.smith@gmail.com.",
            "Set contact to Bob Jones (no email).",
            "contact: ADMIN@EXAMPLE.COM",
        ],
    ),
    (
        "type_list_bounded",
        {
            "type": "object",
            "properties": {
                "retries": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 5,
                }
            },
            "required": ["retries"],
            "additionalProperties": False,
        },
        [
            "Set retries to 999.",
            "retries should be -3.",
            "Use 'infinite' retries.",
        ],
    ),
    (
        "nested_additionalProperties",
        {
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "properties": {"owner": {"type": "string"}},
                    "additionalProperties": False,
                }
            },
            "required": ["meta"],
            "additionalProperties": False,
        },
        [
            (
                "meta.owner = alice, and also add meta.department = sales and"
                " meta.region = EU."
            ),
            "Set meta with owner bob, priority high, and tags x,y.",
        ],
    ),
    (
        "array_cardinality_bounded",
        {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 3,
                }
            },
            "required": ["tags"],
            "additionalProperties": False,
        },
        [
            "Use exactly one tag: solo.",
            "tags: a, b, c, d, e, f, g (seven of them).",
            "tags: red, green.",
        ],
    ),
    (
        # `uniqueItems` on an unbounded-domain array is a cross-element global
        # constraint no regular/CFG grammar can express, so no constrained-
        # decoding backend (llguidance, xgrammar) can enforce it. The honest
        # contract is to reject the request up front with a 400 naming the
        # construct (the orchestrator schema gate), not best-effort a 200 that
        # violates the schema: a clean 400 is the PASS, a 200 is the FAIL.
        # (uniqueItems on a finite/enum domain is enforced by the bespoke codecs
        # and covered by the Rust matrix test, not here.)
        "array_unique_unenforceable",
        {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 3,
                    "uniqueItems": True,
                }
            },
            "required": ["tags"],
            "additionalProperties": False,
        },
        [
            "tags: a, a, b, c, d, e (with the duplicate a).",
            "Use exactly one tag: solo.",
            "tags: red, red, red.",
        ],
    ),
]

# Features carrying a construct no constrained-decoding backend can enforce (a
# cross-element global constraint). These are graded on the rejection contract
# (an honest 400 up front), not on output conformance.
_EXPECT_REJECT = {"array_unique_unenforceable"}


def _is_honest_rejection(status: int) -> bool:
    """True if ``status`` is an honest refusal of the unenforceable construct.

    The request is well-formed (a valid ``uniqueItems`` schema), so the only
    reason to reject it is the construct itself: a clean 400 is the backend
    honestly declining a constraint it cannot enforce, whatever the exact
    message says. Matching on status alone -- rather than a ``"uniqueitems"`` /
    ``"enforce"`` substring that is backend- and version-specific -- keeps a
    correctly-refused request from grading FAIL just because a backend words its
    400 differently.
    """
    return status == 400


_N_PER_PROMPT = 5

_MAX_OUTPUT_TOKENS = 2048


@register_scenario
class StructuredOutputSchemaLimits(BaseScenario):
    name = "structured_output_schema_limits"
    description = (
        "Per-feature response_format=json_schema conformance ($ref/anyOf/"
        "type-list/nested-addlProps/array-cardinality), exposes which schema "
        "constructs the constrained-decoding backend fails to enforce"
    )
    tags = [
        "structured",
        "json",
        "schema",
        "conformance",
        "limits",
        "correctness",
    ]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model

        overall_conf = 0
        overall_attempts = 0
        # feature -> (conform, valid, attempts)
        per_feature: dict[str, tuple[int, int, int]] = {}
        feature_verdicts: list[Verdict] = []

        for feature, schema, prompts in CASES:
            conf = 0
            valid = 0
            attempts = 0
            empty = 0
            nonjson = 0
            truncated = 0
            rejected = 0
            expect_reject = feature in _EXPECT_REJECT
            examples: list[str] = []
            for prompt in prompts:
                for _ in range(_N_PER_PROMPT):
                    attempts += 1
                    payload = {
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Return ONLY a JSON object for the request."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": _rf(feature, schema),
                        "max_tokens": _MAX_OUTPUT_TOKENS,
                        "temperature": 0.0,
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                            "thinking": False,
                        },
                    }
                    resp = await client.post_json(
                        payload, timeout=config.timeout * 2
                    )
                    if expect_reject:
                        # The unenforceable construct must be refused up front
                        # with a 400 naming it; any other status means it leaked
                        # to a best-effort response.
                        if _is_honest_rejection(resp.status):
                            rejected += 1
                        elif len(examples) < 3:
                            examples.append(
                                f"LEAK status={resp.status}: "
                                f"{(resp.body or '')[:80]!r}"
                            )
                        continue
                    if resp.status != 200 or resp.error:
                        continue
                    try:
                        choice = json.loads(resp.body)["choices"][0]
                        content = choice["message"]["content"]
                    except (
                        json.JSONDecodeError,
                        KeyError,
                        IndexError,
                        TypeError,
                    ):
                        continue
                    # A truncated response (finish="length") is liveness noise,
                    # not a grammar fault, so an empty or non-JSON body counts
                    # against the grammar only when the response completed.
                    truncated_resp = choice.get("finish_reason") == "length"
                    # Empty/whitespace content under response_format=json_schema
                    # is a backend failure (the grammar admitted an immediate
                    # stop instead of forcing the object) -- unless it truncated.
                    if not (content or "").strip():
                        if truncated_resp:
                            truncated += 1
                        else:
                            empty += 1
                            if len(examples) < 3:
                                examples.append(
                                    "EMPTY content (grammar admitted stop)"
                                )
                        continue
                    try:
                        parsed = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        if truncated_resp:
                            truncated += 1
                        else:
                            nonjson += 1
                            if len(examples) < 3:
                                examples.append(f"NON-JSON: {content[:60]!r}")
                        continue
                    valid += 1
                    viol = _schema_violations(parsed, schema)
                    if not viol:
                        conf += 1
                    elif len(examples) < 3:
                        examples.append(f"{content[:70]} -> {viol[:2]}")
            if expect_reject:
                reject_rate = (rejected / attempts * 100) if attempts else 0.0
                detail = (
                    f"{feature}: honest-400 rejected"
                    f" {rejected}/{attempts} ({reject_rate:.0f}%);"
                    " unenforceable construct refused up front, never a"
                    " best-effort 200"
                )
                print(f"    {detail}")
                for ex in examples:
                    print(f"      {ex}")
                # Its counts stay out of the conformance rate (a different
                # contract), but its verdict still gates the overall result.
                per_feature[feature] = (rejected, rejected, attempts)
                # Deterministic static gate: a correct backend rejects the
                # unenforceable construct on every attempt, so require 100% --
                # any leak to a best-effort 200 is a real fail-open.
                fv = Verdict.PASS if rejected == attempts else Verdict.FAIL
                feature_verdicts.append(fv)
                results.append(
                    self.make_result(
                        self.name, f"feature_{feature}", fv, detail=detail
                    )
                )
                continue
            per_feature[feature] = (conf, valid, attempts)
            overall_conf += conf
            overall_attempts += attempts
            valid_rate = (valid / attempts * 100) if attempts else 0.0
            conf_rate = (conf / valid * 100) if valid else 0.0
            detail = (
                f"{feature}: valid_output"
                f" {valid}/{attempts} ({valid_rate:.0f}%), conformant"
                f" {conf}/{valid} ({conf_rate:.0f}%), empty={empty},"
                f" non_json={nonjson}, truncated={truncated}"
            )
            print(f"    {detail}")
            for ex in examples:
                print(f"      {ex}")
            # Conformance is binary (constrained decoding is deterministic): any
            # fail-open fault on a completed response -- schema violation, empty
            # object, or complete non-JSON body -- is a FAIL, never averaged away
            # by a pass rate. Truncation is separate liveness noise, so a feature
            # that only ever truncated is INTERESTING, not FAIL.
            violations = valid - conf
            if violations + empty + nonjson > 0:
                fv = Verdict.FAIL
            elif valid > 0:
                fv = Verdict.PASS
            else:
                fv = Verdict.INTERESTING
            feature_verdicts.append(fv)
            results.append(
                self.make_result(
                    self.name, f"feature_{feature}", fv, detail=detail
                )
            )

        overall_rate = (
            (overall_conf / overall_attempts * 100) if overall_attempts else 0.0
        )
        summary_parts = []
        for k, (c, v, a) in per_feature.items():
            if k in _EXPECT_REJECT:
                summary_parts.append(f"{k}=[rejected {v}/{a} (honest 400)]")
            else:
                summary_parts.append(f"{k}=[valid {v}/{a}, conf {c}/{v}]")
        summary = (
            f"overall conformant {overall_conf}/{overall_attempts} "
            f"enforceable-feature attempts ({overall_rate:.0f}%); "
            + " ".join(summary_parts)
        )
        print(f"    SUMMARY: {summary}")
        # Worst per-feature verdict, so one broken construct is not averaged
        # away by the others conforming.
        if Verdict.FAIL in feature_verdicts:
            verdict = Verdict.FAIL
        elif Verdict.INTERESTING in feature_verdicts:
            verdict = Verdict.INTERESTING
        else:
            verdict = Verdict.PASS
        results.append(
            self.make_result(
                self.name,
                "structured_output_conformance",
                verdict,
                detail=summary,
            )
        )

        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_structured_limits_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
            )
        )
        return results
