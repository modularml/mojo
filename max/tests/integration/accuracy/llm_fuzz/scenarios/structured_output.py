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
Scenarios: Structured output attacks
Target: State corruption from json mode, invalid schemas, guided generation crashes.
Known to crash vLLM (issue #4070, #17248).
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class StructuredOutputAttacks(BaseScenario):
    name = "structured_output"
    description = "JSON mode state corruption, invalid schemas, response_format edge cases"
    tags = ["structured", "json", "schema", "state_corruption", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        def base(
            content: str = "Return a JSON object with key 'name' and value 'test'",
            **extra: Any,
        ) -> dict[str, Any]:
            p: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 100,
            }
            p.update(extra)
            return p

        # ----- 1. JSON mode baseline (happy path) -----
        baseline_payload = base(response_format={"type": "json_object"})
        resp = await client.post_json(baseline_payload)
        results.append(
            self.make_result(
                self.name,
                "json_mode_baseline",
                Verdict.PASS if resp.status == 200 else Verdict.FAIL,
                status_code=resp.status,
            )
        )

        # ----- 2. JSON mode then normal (state corruption test) -----
        # This is the vLLM #4070 pattern: json mode can corrupt global state
        json_payload = base(response_format={"type": "json_object"})
        normal_payload = base(content="Say hello in plain text")

        await client.post_json(json_payload)
        resp2 = await client.post_json(normal_payload)

        # Check if normal response got corrupted (e.g., only \n\t chars)
        if resp2.status == 200:
            body = resp2.body
            try:
                data = json.loads(body)
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content and all(c in "\n\t " for c in content):
                    verdict = Verdict.FAIL
                    detail = "STATE CORRUPTION: Normal response contains only whitespace after JSON mode"
                else:
                    verdict = Verdict.PASS
                    detail = "Normal response unaffected by prior JSON mode"
            except (json.JSONDecodeError, KeyError, IndexError):
                verdict = Verdict.PASS
                detail = "Response not parseable as expected JSON"
        else:
            verdict = Verdict.PASS if resp2.status < 500 else Verdict.FAIL
            detail = f"Status {resp2.status}"

        results.append(
            self.make_result(
                self.name,
                "json_then_normal_state_corruption",
                verdict,
                detail=detail,
            )
        )

        # ----- 3. Rapid alternation: JSON ↔ normal -----
        corruption_found = False
        for i in range(20):
            if i % 2 == 0:
                r = await client.post_json(
                    base(response_format={"type": "json_object"})
                )
            else:
                r = await client.post_json(base(content="Plain text please"))
                if r.status == 200:
                    try:
                        d = json.loads(r.body)
                        c = (
                            d.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        if c and all(ch in "\n\t " for ch in c):
                            corruption_found = True
                            break
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

        results.append(
            self.make_result(
                self.name,
                "rapid_json_normal_alternation",
                Verdict.FAIL if corruption_found else Verdict.PASS,
                detail="State corruption detected"
                if corruption_found
                else "No corruption after 20 alternations",
            )
        )

        # ----- 4. Concurrent JSON mode requests -----
        json_payloads = [
            base(response_format={"type": "json_object"}) for _ in range(50)
        ]
        responses = await client.concurrent_requests(json_payloads)
        server_errors = sum(1 for r in responses if r.status >= 500)
        results.append(
            self.make_result(
                self.name,
                "concurrent_json_mode_50",
                Verdict.FAIL
                if server_errors > 10
                else (
                    Verdict.INTERESTING if server_errors > 0 else Verdict.PASS
                ),
                detail=f"{server_errors}/50 server errors",
            )
        )

        # ----- 5. JSON mode + streaming -----
        streaming_json = base(response_format={"type": "json_object"})
        resp = await client.post_streaming(streaming_json)
        results.append(
            self.make_result(
                self.name,
                "json_mode_streaming",
                Verdict.PASS if resp.status in (200, 400) else Verdict.FAIL,
                status_code=resp.status,
                detail=f"Got {len(resp.chunks or [])} chunks",
            )
        )

        # ----- 6. JSON schema (json_schema type) - valid schema -----
        valid_schema_payload = base(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test_response",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            },
        )
        resp = await client.post_json(valid_schema_payload)
        results.append(
            self.make_result(
                self.name,
                "json_schema_valid",
                Verdict.PASS if resp.status in (200, 400) else Verdict.FAIL,
                status_code=resp.status,
            )
        )

        # ----- 7. Invalid / edge-case JSON schemas -----
        # Each entry: (format_dict, expected) where expected is:
        #   "reject"  → 400 = PASS, 200 = FAIL
        #   "accept"  → 200 = PASS, 400 = INTERESTING (stricter than required)
        #   "either"  → 200 or 400 = PASS
        # 500 and TIMEOUT are always FAIL.
        #
        # JSON Schema spec allows missing "type", empty schemas, type
        # unions, non-object roots, and deep nesting.  xgrammar compiles
        # all of these successfully.  Only structurally broken schemas
        # (non-dict, null, recursive $ref) must be rejected.
        invalid_schemas = {
            # Schema value is a string, not a dict — structurally invalid
            "schema_not_object": (
                {
                    "type": "json_schema",
                    "json_schema": {"name": "bad", "schema": "not an object"},
                },
                "reject",
            ),
            # Missing "type" — valid JSON Schema (means "any type")
            "schema_missing_type": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bad",
                        "schema": {"properties": {"x": {}}},
                    },
                },
                "accept",
            ),
            # Recursive $ref — valid JSON Schema; grammar backends
            # (xgrammar, llguidance) support it with depth limits.
            "schema_recursive": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "recursive",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "children": {
                                    "type": "array",
                                    "items": {"$ref": "#"},
                                },
                            },
                        },
                    },
                },
                "accept",
            ),
            # 50-level nesting — valid but may truncate at max_tokens
            "schema_deeply_nested": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "deep",
                        "schema": _nested_schema(50),
                    },
                },
                "accept",
            ),
            # Type array union — valid JSON Schema
            "schema_conflicting_types": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "conflict",
                        "schema": {"type": ["object", "array", "string"]},
                    },
                },
                "accept",
            ),
            # Empty schema {} — means "accept any JSON", valid
            "schema_empty": (
                {
                    "type": "json_schema",
                    "json_schema": {"name": "empty", "schema": {}},
                },
                "accept",
            ),
            # Null json_schema — structurally invalid
            "schema_null": (
                {
                    "type": "json_schema",
                    "json_schema": None,
                },
                "reject",
            ),
            # Large enum — valid, xgrammar handles it
            "schema_huge_enum": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "huge_enum",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "value": {
                                    "type": "string",
                                    "enum": [
                                        f"value_{i}" for i in range(10000)
                                    ],
                                },
                            },
                        },
                    },
                },
                "accept",
            ),
        }

        for name, (fmt, expected) in invalid_schemas.items():
            resp = await client.post_json(
                base(response_format=fmt), timeout=config.timeout * 0.5
            )
            if resp.error == "TIMEOUT":
                verdict = Verdict.FAIL
                detail = "Server hung on invalid schema"
            elif resp.status >= 500:
                verdict = Verdict.FAIL
                detail = f"Server crash {resp.status} on invalid schema"
            elif expected == "reject":
                if 400 <= resp.status < 500:
                    verdict, detail = Verdict.PASS, "Properly rejected"
                elif resp.status == 200:
                    verdict, detail = (
                        Verdict.FAIL,
                        "Accepted schema that should be rejected",
                    )
                else:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        f"Status {resp.status}",
                    )
            elif expected == "accept":
                if resp.status == 200:
                    verdict, detail = Verdict.PASS, "Accepted valid schema"
                elif 400 <= resp.status < 500:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        "Rejected valid schema (stricter than required)",
                    )
                else:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        f"Status {resp.status}",
                    )
            else:  # "either"
                if resp.status == 200:
                    verdict, detail = Verdict.PASS, "Accepted"
                elif 400 <= resp.status < 500:
                    verdict, detail = Verdict.PASS, "Rejected"
                else:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        f"Status {resp.status}",
                    )

            results.append(
                self.make_result(
                    self.name,
                    f"invalid_{name}",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    response_body=resp.body[:300],
                )
            )

        # ----- 7b. Backend-uncompilable schemas (backend-conditional) -----
        # These schemas only a fail-closed backend (xgrammar) rejects at
        # compile time; llguidance fails open and compiles them. So we probe
        # all three, detect the backend's disposition at runtime (did it 4xx
        # any of them?), and assert the 400 rejection ONLY when it fails
        # closed -- any model that migrates to xgrammar gets this check for
        # free, and llguidance models aren't false-failed. A crash/hang is a
        # FAIL on every backend: that is the invariant admission validation
        # protects.
        uncompilable_schemas = {
            # `false` -> router lowers to {"anyOf": [false]}, unsatisfiable
            "boolean_false": {"name": "bf", "schema": False},
            # $ref to a definition that does not exist
            "unresolvable_ref": {
                "name": "bad_ref",
                "schema": {
                    "type": "object",
                    "properties": {"x": {"$ref": "#/$defs/Missing"}},
                },
            },
            # unbalanced-paren regex terminal fails to compile
            "invalid_regex_pattern": {
                "name": "bad_regex",
                "schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string", "pattern": "("},
                    },
                },
            },
        }
        uncompilable_responses = {}
        for uname, js in uncompilable_schemas.items():
            uncompilable_responses[uname] = await client.post_json(
                base(
                    response_format={"type": "json_schema", "json_schema": js}
                ),
                timeout=config.timeout * 0.5,
            )
        # Fail-closed iff the backend rejected (4xx) at least one without
        # crashing. xgrammar rejects all three; llguidance accepts all three.
        backend_fails_closed = any(
            400 <= r.status < 500 for r in uncompilable_responses.values()
        )
        for uname, resp in uncompilable_responses.items():
            if resp.error == "TIMEOUT":
                verdict, detail = (
                    Verdict.FAIL,
                    "Server hung on uncompilable schema",
                )
            elif resp.status == 0:
                verdict, detail = (
                    Verdict.FAIL,
                    "Connection failed (worker crash?) on uncompilable schema",
                )
            elif resp.status >= 500:
                verdict, detail = (
                    Verdict.FAIL,
                    f"Server crash {resp.status} on uncompilable schema",
                )
            elif backend_fails_closed:
                # xgrammar-class backend: admission validation must 400 these
                # rather than let them crash the worker.
                if 400 <= resp.status < 500:
                    verdict, detail = (
                        Verdict.PASS,
                        "Fail-closed backend rejected",
                    )
                elif resp.status == 200:
                    verdict, detail = (
                        Verdict.FAIL,
                        "Fail-closed backend accepted an uncompilable schema",
                    )
                else:
                    verdict, detail = (
                        Verdict.INTERESTING,
                        f"Status {resp.status}",
                    )
            else:
                # Fail-open backend (e.g. llguidance): compiling these is fine.
                verdict, detail = (
                    Verdict.PASS,
                    "Fail-open backend compiled schema (no crash)",
                )
            results.append(
                self.make_result(
                    self.name,
                    f"uncompilable_{uname}",
                    verdict,
                    status_code=resp.status,
                    detail=detail,
                    response_body=resp.body[:300],
                )
            )

        # ----- 8. JSON mode without instructing JSON in prompt -----
        # Some servers require "json" in the prompt when using json mode
        resp = await client.post_json(
            base(
                content="Tell me about cats",
                response_format={"type": "json_object"},
            )
        )
        results.append(
            self.make_result(
                self.name,
                "json_mode_no_json_instruction",
                Verdict.PASS if resp.status in (200, 400) else Verdict.FAIL,
                status_code=resp.status,
                detail=f"Status {resp.status}",
            )
        )

        # ----- 9. Post-attack health check -----
        await asyncio.sleep(2)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_structured_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
            )
        )

        return results


def _nested_schema(depth: int) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }
    for i in range(depth):
        schema = {
            "type": "object",
            "properties": {f"level_{i}": schema},
            "required": [f"level_{i}"],
        }
    return schema
