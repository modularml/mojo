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
Scenarios: Parameter abuse
Target: Crashes from out-of-range, NaN, infinity, negative, or conflicting parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class ParameterAbuse(BaseScenario):
    name = "parameter_abuse"
    description = "Extreme, invalid, and conflicting parameter values"
    tags = ["parameters", "validation", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        def base(overrides: dict[str, Any]) -> dict[str, Any]:
            p: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": "Say hello"}],
            }
            p.update(overrides)
            return p

        tests = {
            # --- max_tokens extremes ---
            "max_tokens_zero": base({"max_tokens": 0}),
            "max_tokens_negative": base({"max_tokens": -1}),
            "max_tokens_negative_large": base({"max_tokens": -999999}),
            "max_tokens_one": base({"max_tokens": 1}),
            "max_tokens_huge": base({"max_tokens": 999999999}),
            "max_tokens_int_max": base({"max_tokens": 2**31 - 1}),
            "max_tokens_int64_max": base({"max_tokens": 2**63 - 1}),
            "max_tokens_float": base({"max_tokens": 10.5}),
            "max_tokens_nan": '{"model":"'
            + model
            + '","messages":[{"role":"user","content":"hi"}],"max_tokens":NaN}',
            "max_tokens_infinity": '{"model":"'
            + model
            + '","messages":[{"role":"user","content":"hi"}],"max_tokens":Infinity}',
            "max_tokens_neg_infinity": '{"model":"'
            + model
            + '","messages":[{"role":"user","content":"hi"}],"max_tokens":-Infinity}',
            # --- temperature extremes ---
            "temperature_zero": base({"temperature": 0}),
            "temperature_negative": base({"temperature": -1}),
            "temperature_large_negative": base({"temperature": -100}),
            "temperature_slightly_above_2": base({"temperature": 2.0001}),
            "temperature_100": base({"temperature": 100}),
            "temperature_tiny": base({"temperature": 0.0000001}),
            "temperature_nan_str": '{"model":"'
            + model
            + '","messages":[{"role":"user","content":"hi"}],"temperature":NaN}',
            # --- top_p extremes ---
            "top_p_zero": base({"top_p": 0}),
            "top_p_negative": base({"top_p": -0.5}),
            "top_p_above_one": base({"top_p": 1.5}),
            "top_p_tiny": base({"top_p": 0.0000001}),
            "top_p_and_temperature": base({"top_p": 0.1, "temperature": 2}),
            # --- frequency_penalty / presence_penalty ---
            "freq_penalty_extreme_positive": base({"frequency_penalty": 100}),
            "freq_penalty_extreme_negative": base({"frequency_penalty": -100}),
            "pres_penalty_extreme_positive": base({"presence_penalty": 100}),
            "pres_penalty_extreme_negative": base({"presence_penalty": -100}),
            "both_penalties_extreme": base(
                {"frequency_penalty": 2, "presence_penalty": 2}
            ),
            # --- n (number of completions) ---
            "n_zero": base({"n": 0}),
            "n_negative": base({"n": -1}),
            "n_1000": base({"n": 1000, "max_tokens": 1}),
            "n_very_large": base({"n": 100000, "max_tokens": 1}),
            # --- logprobs ---
            "logprobs_true": base({"logprobs": True, "top_logprobs": 5}),
            "logprobs_top_zero": base({"logprobs": True, "top_logprobs": 0}),
            "logprobs_top_negative": base(
                {"logprobs": True, "top_logprobs": -1}
            ),
            "logprobs_top_huge": base({"logprobs": True, "top_logprobs": 100}),
            "logprobs_without_top": base({"logprobs": True}),
            # --- stop sequences ---
            "stop_empty_string": base({"stop": ""}),
            "stop_empty_array": base({"stop": []}),
            "stop_array_with_empty": base({"stop": [""]}),
            "stop_too_many": base({"stop": [f"stop{i}" for i in range(100)]}),
            "stop_very_long": base({"stop": "x" * 100000}),
            "stop_null": base({"stop": None}),
            "stop_with_newlines": base({"stop": ["\n", "\n\n", "\r\n"]}),
            # --- seed ---
            "seed_zero": base({"seed": 0}),
            "seed_negative": base({"seed": -1}),
            "seed_huge": base({"seed": 2**63}),
            "seed_float": base({"seed": 3.14}),
            # --- response_format ---
            "response_format_invalid_type": base(
                {"response_format": {"type": "invalid"}}
            ),
            "response_format_empty": base({"response_format": {}}),
            "response_format_null": base({"response_format": None}),
            "response_format_string": base({"response_format": "json"}),
            "response_format_json": base(
                {"response_format": {"type": "json_object"}, "max_tokens": 50}
            ),
            # --- max_tokens vs max_completion_tokens ---
            "max_completion_tokens_only": base({"max_completion_tokens": 10}),
            "max_tokens_and_max_completion_tokens_same": base(
                {"max_tokens": 10, "max_completion_tokens": 10}
            ),
            "max_tokens_and_max_completion_tokens_conflict": base(
                {"max_tokens": 10, "max_completion_tokens": 20}
            ),
            # --- conflicting parameters ---
            "stream_true_with_n_5": base(
                {"stream": True, "n": 5, "max_tokens": 5}
            ),
            "temp_0_with_top_p_0": base({"temperature": 0, "top_p": 0}),
            # --- user field ---
            "user_empty": base({"user": ""}),
            "user_very_long": base({"user": "x" * 100000}),
            "user_special_chars": base(
                {"user": "<script>alert('xss')</script>"}
            ),
            "user_null": base({"user": None}),
            # --- Non-existent model ---
            "model_nonexistent": {
                "model": "gpt-nonexistent-turbo-99",
                "messages": [{"role": "user", "content": "hi"}],
            },
            "model_empty_string": {
                "model": "",
                "messages": [{"role": "user", "content": "hi"}],
            },
            # --- Multiple conflicting settings ---
            "everything_extreme": base(
                {
                    "temperature": 2,
                    "top_p": 0.001,
                    "max_tokens": 1,
                    "n": 10,
                    "frequency_penalty": 2,
                    "presence_penalty": 2,
                    "stop": ["a", "e", "i", "o", "u"],
                    "logprobs": True,
                    "top_logprobs": 5,
                    "seed": 42,
                }
            ),
        }

        # Tests that require an exact status code to pass.
        expected_status: dict[str, int] = {
            "max_completion_tokens_only": 200,
            "max_tokens_and_max_completion_tokens_same": 200,
            "max_tokens_and_max_completion_tokens_conflict": 400,
        }

        for test_name, payload in tests.items():
            try:
                if isinstance(payload, str):
                    resp = await client.post_raw_bytes(
                        payload.encode(), timeout=config.timeout * 0.5
                    )
                else:
                    resp = await client.post_json(
                        payload, timeout=config.timeout * 0.5
                    )

                required = expected_status.get(test_name)
                if resp.error == "TIMEOUT":
                    verdict = Verdict.FAIL
                    detail = "Server hung on parameter edge case"
                elif resp.status == 0:
                    verdict = Verdict.FAIL
                    detail = f"Connection error: {resp.error}"
                elif required is not None:
                    if resp.status == required:
                        verdict = Verdict.PASS
                        detail = f"Got expected status {resp.status}"
                    else:
                        verdict = Verdict.FAIL
                        detail = f"Expected {required}, got {resp.status}"
                elif resp.status >= 500:
                    verdict = Verdict.FAIL
                    detail = (
                        f"Server error {resp.status} — parameter not validated"
                    )
                elif 400 <= resp.status < 500:
                    verdict = Verdict.PASS
                    detail = f"Properly rejected with {resp.status}"
                elif resp.status == 200:
                    verdict = Verdict.PASS
                    detail = "Handled without crash"
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

        return results
