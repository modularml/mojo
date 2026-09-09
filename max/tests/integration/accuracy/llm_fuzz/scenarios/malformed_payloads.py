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
Scenarios: Malformed payloads
Target: Server crashes or hangs when receiving structurally broken requests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class MalformedPayloads(BaseScenario):
    name = "malformed_payloads"
    description = "Structurally broken JSON, missing fields, wrong types, truncated bodies"
    tags = ["payload", "validation", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        tests = {
            # --- Missing / empty fields ---
            "empty_body": b"",
            "empty_json_object": {},
            "empty_messages_array": {"model": model, "messages": []},
            "missing_messages": {"model": model},
            "missing_model": {"messages": [{"role": "user", "content": "hi"}]},
            "missing_content": {"model": model, "messages": [{"role": "user"}]},
            "missing_role": {"model": model, "messages": [{"content": "hi"}]},
            "null_model": {
                "model": None,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "null_messages": {"model": model, "messages": None},
            "null_content": {
                "model": model,
                "messages": [{"role": "user", "content": None}],
            },
            "null_role": {
                "model": model,
                "messages": [{"role": None, "content": "hi"}],
            },
            # --- Wrong types ---
            "messages_is_string": {"model": model, "messages": "hello"},
            "messages_is_number": {"model": model, "messages": 42},
            "messages_is_bool": {"model": model, "messages": True},
            "message_is_string": {"model": model, "messages": ["hello"]},
            "message_is_number": {"model": model, "messages": [123]},
            "message_is_nested_array": {
                "model": model,
                "messages": [[{"role": "user", "content": "hi"}]],
            },
            "content_is_number": {
                "model": model,
                "messages": [{"role": "user", "content": 99999}],
            },
            "content_is_array": {
                "model": model,
                "messages": [{"role": "user", "content": [1, 2, 3]}],
            },
            "content_is_bool": {
                "model": model,
                "messages": [{"role": "user", "content": False}],
            },
            "content_is_object": {
                "model": model,
                "messages": [{"role": "user", "content": {"text": "hi"}}],
            },
            "model_is_number": {
                "model": 12345,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "model_is_array": {
                "model": ["gpt-4"],
                "messages": [{"role": "user", "content": "hi"}],
            },
            "max_tokens_is_string": {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": "lots",
            },
            "temperature_is_string": {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": "hot",
            },
            # --- Invalid role values ---
            "role_empty_string": {
                "model": model,
                "messages": [{"role": "", "content": "hi"}],
            },
            "role_unknown": {
                "model": model,
                "messages": [{"role": "admin", "content": "hi"}],
            },
            "role_numeric": {
                "model": model,
                "messages": [{"role": "123", "content": "hi"}],
            },
            "role_sql_injection": {
                "model": model,
                "messages": [
                    {"role": "user'; DROP TABLE messages;--", "content": "hi"}
                ],
            },
            # --- Truncated / malformed JSON ---
            "truncated_json": '{"model": "'
            + model
            + '", "messages": [{"role": "us',
            "unclosed_brace": '{"model": "' + model + '"',
            "unclosed_bracket": '{"model": "' + model + '", "messages": [',
            "double_comma": '{"model": "' + model + '",, "messages": []}',
            "trailing_comma": '{"model": "' + model + '", "messages": [],}',
            "not_json_xml": f"<request><model>{model}</model><message>hi</message></request>",
            "not_json_plain": "just a plain text string",
            "not_json_urlencoded": f"model={model}&messages=hello",
            "only_whitespace_body": "   \n\t\n   ",
            "only_newlines": "\n\n\n\n\n",
            # --- Binary junk ---
            "null_bytes": b"\x00\x00\x00\x00\x00",
            "random_binary": bytes(range(256)),
            "json_with_null_bytes": (
                '{"model":"'
                + model
                + '","messages":[{"role":"user","content":"he\x00llo"}]}'
            ),
            # --- Extra / unexpected fields ---
            "extra_unknown_fields": {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "destroy_server": True,
                "admin_mode": True,
                "debug": True,
                "__proto__": {"isAdmin": True},
            },
            # --- Deeply nested structures ---
            "deeply_nested_content": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "hi", "extra": _nest_dict(100)}
                ],
            },
            # --- Duplicate keys (raw JSON) ---
            "duplicate_keys": '{"model":"'
            + model
            + '","model":"other","messages":[{"role":"user","content":"hi"}]}',
            # --- Enormous single field ---
            "huge_model_name": {
                "model": "x" * 100_000,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "huge_role": {
                "model": model,
                "messages": [{"role": "x" * 100_000, "content": "hi"}],
            },
        }

        for test_name, payload in tests.items():
            try:
                if isinstance(payload, bytes):
                    resp = await client.post_raw_bytes(
                        payload, timeout=config.timeout * 0.33
                    )
                elif isinstance(payload, str):
                    resp = await client.post_raw_bytes(
                        payload.encode("utf-8", errors="replace"),
                        timeout=config.timeout * 0.33,
                    )
                else:
                    resp = await client.post_json(
                        payload, timeout=config.timeout * 0.33
                    )

                # A well-behaved server should return 4xx for all of these
                if resp.error == "TIMEOUT":
                    verdict = Verdict.FAIL
                    detail = "Server hung / timed out on malformed input"
                elif resp.status == 0:
                    verdict = Verdict.FAIL
                    detail = f"Connection error: {resp.error}"
                elif 400 <= resp.status < 500:
                    verdict = Verdict.PASS
                    detail = f"Properly rejected with {resp.status}"
                elif resp.status == 200:
                    verdict = Verdict.INTERESTING
                    detail = "Server ACCEPTED a malformed request"
                elif resp.status >= 500:
                    verdict = Verdict.FAIL
                    detail = f"Server error {resp.status} — potential crash"
                else:
                    verdict = Verdict.INTERESTING
                    detail = f"Unexpected status {resp.status}"

                results.append(
                    self.make_result(
                        self.name,
                        test_name,
                        verdict,
                        status_code=resp.status,
                        elapsed_ms=resp.elapsed_ms,
                        detail=detail,
                        response_body=resp.body[:500],
                    )
                )
            except Exception as e:
                results.append(
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.ERROR,
                        error=str(e),
                    )
                )

        # --- Post-attack health check ---
        health = await client.health_check()
        if health.status != 200:
            results.append(
                self.make_result(
                    self.name,
                    "post_attack_health_check",
                    Verdict.FAIL,
                    status_code=health.status,
                    detail="Server became unhealthy after malformed payload barrage",
                    error=health.error or "",
                )
            )
        else:
            results.append(
                self.make_result(
                    self.name,
                    "post_attack_health_check",
                    Verdict.PASS,
                    status_code=200,
                    detail="Server still healthy",
                )
            )

        return results


def _nest_dict(depth: int) -> dict[str, Any]:
    d: dict[str, Any] = {"value": "leaf"}
    for _ in range(depth):
        d = {"nested": d}
    return d
