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
Scenario: JSON schema compliance validation
Target: Verify that response_format json_schema actually constrains model output.
Tests whether guided decoding is working. The engine should return valid JSON matching
the requested schema, not markdown-wrapped or wrong-field responses.

The error is random so the payload is 100 runs to get a good sample. Add random prefix
to avoid prefix caching. It outputs the error content in error log.

Motivated by: Kimi-K2.5-NVFP4 returns high error rates with response_format json_schema
 with same requests.
"""

from __future__ import annotations

import json
import os
import random
import string
from datetime import datetime
from typing import TYPE_CHECKING

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

REQUIRED_KEYS = {"is_violating", "category", "explanation_if_violating"}


def _rand(k: int = 300) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=k))


@register_scenario
class JsonSchemaCompliance(BaseScenario):
    name = "json_schema_compliance"
    description = "100-run cache-busted test that json_schema response_format actually constrains output"
    tags = [
        "validation",
        "structured",
        "json",
        "schema",
        "compliance",
        "correctness",
    ]
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model
        n_runs = 100

        valid_json = 0
        correct_schema = 0
        not_json = 0
        api_errors = 0
        logged_one_success = False
        wrong_schema_logs = []
        non_json_logs = []

        # Set up error log file
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_file = os.path.join(
            log_dir, f"json_schema_compliance_{timestamp_str}.jsonl"
        )

        with open(log_file, "w") as logf:
            logf.write(
                json.dumps(
                    {
                        "event": "run_start",
                        "timestamp": datetime.now().isoformat(),
                        "runs": n_runs,
                        "model": model,
                    }
                )
                + "\n"
            )

            for i in range(n_runs):
                rand_block = _rand(300)
                rand_block2 = _rand(200)

                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"{rand_block}\n\nYour task: determine if the user input violates content guidelines. Respond in the requested JSON format only.",
                        },
                        {
                            "role": "user",
                            "content": f"{rand_block2}\n\nHow do I prepare for a job interview?",
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "content_compliance",
                            "description": "Determines if content is violating specific moderation rules",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "is_violating": {"type": "boolean"},
                                    "category": {
                                        "anyOf": [
                                            {
                                                "type": "string",
                                                "enum": [
                                                    "violence",
                                                    "sexual",
                                                    "self_harm",
                                                ],
                                            },
                                            {"type": "null"},
                                        ],
                                    },
                                    "explanation_if_violating": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "null"},
                                        ],
                                    },
                                },
                                "required": [
                                    "is_violating",
                                    "category",
                                    "explanation_if_violating",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                }

                resp = await client.post_json(
                    payload, timeout=config.timeout * 2
                )

                if resp.error or resp.status != 200:
                    api_errors += 1
                    logf.write(
                        json.dumps(
                            {
                                "event": "api_error",
                                "timestamp": datetime.now().isoformat(),
                                "run": i + 1,
                                "error": resp.error or f"HTTP {resp.status}",
                            }
                        )
                        + "\n"
                    )
                    continue

                try:
                    data = json.loads(resp.body)
                    content = data["choices"][0]["message"]["content"]
                except Exception as e:
                    api_errors += 1
                    logf.write(
                        json.dumps(
                            {
                                "event": "api_error",
                                "timestamp": datetime.now().isoformat(),
                                "run": i + 1,
                                "error": str(e),
                            }
                        )
                        + "\n"
                    )
                    continue

                try:
                    parsed = json.loads(content)
                    valid_json += 1
                    if (
                        "is_violating" in parsed
                        and "category" in parsed
                        and "explanation_if_violating" in parsed
                    ):
                        correct_schema += 1
                        if not logged_one_success:
                            print(
                                f"    Run {i + 1}: VALID JSON (correct schema)"
                            )
                            print(f"      {json.dumps(parsed, indent=2)}")
                            logged_one_success = True
                    else:
                        wrong_schema_logs.append((i + 1, parsed))
                        logf.write(
                            json.dumps(
                                {
                                    "event": "wrong_schema",
                                    "timestamp": datetime.now().isoformat(),
                                    "run": i + 1,
                                    "content": parsed,
                                }
                            )
                            + "\n"
                        )
                except (json.JSONDecodeError, TypeError):
                    not_json += 1
                    non_json_logs.append((i + 1, content))
                    logf.write(
                        json.dumps(
                            {
                                "event": "not_json",
                                "timestamp": datetime.now().isoformat(),
                                "run": i + 1,
                                "content": content,
                            }
                        )
                        + "\n"
                    )

                if (i + 1) % 20 == 0:
                    print(
                        f"    Progress: {i + 1}/{n_runs} | json={valid_json} not_json={not_json} errors={api_errors}"
                    )

            logf.write(
                json.dumps(
                    {
                        "event": "run_end",
                        "timestamp": datetime.now().isoformat(),
                        "valid_json": valid_json,
                        "correct_schema": correct_schema,
                        "wrong_schema": valid_json - correct_schema,
                        "not_json": not_json,
                        "api_errors": api_errors,
                    }
                )
                + "\n"
            )

        # Print summary
        wrong_schema = valid_json - correct_schema
        schema_rate = correct_schema / n_runs * 100 if n_runs > 0 else 0
        json_rate = valid_json / n_runs * 100 if n_runs > 0 else 0

        print()
        print(f"    {'=' * 50}")
        print(f"    Total:               {n_runs}")
        print(f"    Valid JSON:          {valid_json}")
        print(f"      Correct schema:    {correct_schema}")
        print(f"      Wrong schema:      {wrong_schema}")
        print(f"    Not JSON:            {not_json}")
        print(f"    API errors:          {api_errors}")
        print(f"    JSON rate:           {json_rate:.0f}%")
        print(f"    Correct schema rate: {schema_rate:.0f}%")
        print()

        if wrong_schema_logs:
            print(f"    {'=' * 50}")
            print(
                f"    ALL WRONG SCHEMA RESPONSES ({len(wrong_schema_logs)} total)"
            )
            print(f"    {'=' * 50}")
            for r, p in wrong_schema_logs:
                print(f"    --- Run {r} ---")
                print(f"    {json.dumps(p, indent=2)}")

        if non_json_logs:
            print(f"    {'=' * 50}")
            print(f"    ALL NON-JSON RESPONSES ({len(non_json_logs)} total)")
            print(f"    {'=' * 50}")
            for r, s in non_json_logs:
                print(f"    --- Run {r} ---")
                print(f"    {s}")

        print(f"    Errors logged to: {log_file}")

        # Build verdict
        detail = (
            f"json={valid_json} correct_schema={correct_schema} wrong_schema={wrong_schema} "
            f"not_json={not_json} api_errors={api_errors} | "
            f"JSON rate: {json_rate:.0f}%, Schema rate: {schema_rate:.0f}%"
        )

        if schema_rate >= 90:
            verdict = Verdict.PASS
        elif schema_rate >= 50:
            verdict = Verdict.INTERESTING
        else:
            verdict = Verdict.FAIL

        results.append(
            self.make_result(
                self.name,
                "schema_compliance_100_runs",
                verdict,
                detail=detail,
            )
        )

        # Health check
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_schema_compliance_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
            )
        )

        return results
