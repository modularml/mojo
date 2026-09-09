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
Scenarios: Protocol-level abuse
Target: Server crashes from malformed HTTP, wrong content types, slow clients, huge headers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class ProtocolAbuse(BaseScenario):
    name = "protocol_abuse"
    description = "Wrong content types, huge headers, slow body, method abuse, encoding attacks"
    tags = ["protocol", "http", "headers", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        valid_body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5,
            }
        ).encode()

        # ----- 1. Wrong Content-Type headers -----
        content_types = {
            "text_plain": {"Content-Type": "text/plain"},
            "text_html": {"Content-Type": "text/html"},
            "form_urlencoded": {
                "Content-Type": "application/x-www-form-urlencoded"
            },
            "multipart": {"Content-Type": "multipart/form-data"},
            "xml": {"Content-Type": "application/xml"},
            "octet_stream": {"Content-Type": "application/octet-stream"},
            "missing_content_type": {"Content-Type": ""},
            "garbage_content_type": {
                "Content-Type": "definitely/not-a-real-type"
            },
            "content_type_with_charset": {
                "Content-Type": "application/json; charset=utf-16"
            },
        }

        for name, headers in content_types.items():
            resp = await client.post_raw_bytes(
                valid_body, headers=headers, timeout=config.timeout * 0.33
            )
            if resp.status >= 500:
                verdict = Verdict.FAIL
            elif resp.status in (200, 400, 415):
                verdict = Verdict.PASS
            elif resp.error == "TIMEOUT":
                verdict = Verdict.FAIL
            else:
                verdict = Verdict.INTERESTING

            results.append(
                self.make_result(
                    self.name,
                    f"content_type_{name}",
                    verdict,
                    status_code=resp.status,
                    detail=f"Status {resp.status}"
                    + (f" error: {resp.error}" if resp.error else ""),
                )
            )

        # ----- 2. Huge headers -----
        huge_header_tests = {
            "huge_auth_header": {"Authorization": "Bearer " + "x" * 100_000},
            "huge_custom_header": {"X-Custom": "y" * 100_000},
            "many_small_headers": {
                f"X-Header-{i}": f"value-{i}" for i in range(500)
            },
            "huge_accept": {"Accept": "application/json, " * 10_000},
            "huge_user_agent": {"User-Agent": "Bot/" + "x" * 50_000},
        }

        for name, headers in huge_header_tests.items():
            resp = await client.post_raw_bytes(
                valid_body, headers=headers, timeout=config.timeout * 0.33
            )
            if resp.status >= 500 or resp.error == "TIMEOUT":
                verdict = Verdict.FAIL
            elif resp.status in (200, 400, 413, 431):
                verdict = Verdict.PASS
            else:
                verdict = Verdict.INTERESTING

            results.append(
                self.make_result(
                    self.name,
                    f"header_{name}",
                    verdict,
                    status_code=resp.status,
                    detail=f"Status {resp.status}"
                    + (f" error: {resp.error}" if resp.error else ""),
                )
            )

        # ----- 3. Authentication edge cases -----
        auth_tests = {
            "no_auth": {},
            "empty_bearer": {"Authorization": "Bearer "},
            "wrong_scheme": {"Authorization": "Basic dXNlcjpwYXNz"},
            "just_bearer": {"Authorization": "Bearer"},
            "double_bearer": {"Authorization": "Bearer Bearer token123"},
            "null_byte_in_token": {"Authorization": "Bearer tok\x00en"},
        }

        for name, headers in auth_tests.items():
            # Override the default auth by passing explicit headers
            full_headers = {"Content-Type": "application/json"}
            full_headers.update(headers)
            resp = await client.post_raw_bytes(
                valid_body, headers=full_headers, timeout=config.timeout * 0.33
            )
            results.append(
                self.make_result(
                    self.name,
                    f"auth_{name}",
                    Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                    status_code=resp.status,
                )
            )

        # ----- 4. Slow body delivery (Slowloris for POST body) -----
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
        }
        resp = await client.post_slow_body(
            payload, chunk_delay=0.5, chunk_size=5
        )
        results.append(
            self.make_result(
                self.name,
                "slowloris_body",
                Verdict.PASS
                if resp.status in (200, 400, 408)
                else Verdict.FAIL,
                status_code=resp.status,
                elapsed_ms=resp.elapsed_ms,
                detail=f"Slow body delivery: status {resp.status}, {resp.elapsed_ms:.0f}ms",
            )
        )

        # ----- 5. Encoding attacks -----
        encoding_tests = {
            "utf16_body": valid_body.decode().encode("utf-16"),
            "utf32_body": valid_body.decode().encode("utf-32"),
            "latin1_body": valid_body.decode().encode("latin-1"),
            "gzip_body_no_header": _gzip_compress(valid_body),
            "double_encoded_json": json.dumps(
                json.dumps(
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )
            ).encode(),
        }

        for name, data in encoding_tests.items():
            resp = await client.post_raw_bytes(
                data, timeout=config.timeout * 0.33
            )
            results.append(
                self.make_result(
                    self.name,
                    f"encoding_{name}",
                    Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                    status_code=resp.status,
                )
            )

        # ----- 6. Gzip with proper header -----
        resp = await client.post_raw_bytes(
            _gzip_compress(valid_body),
            headers={"Content-Encoding": "gzip"},
            timeout=config.timeout * 0.33,
        )
        results.append(
            self.make_result(
                self.name,
                "gzip_with_header",
                Verdict.PASS
                if resp.status in (200, 400, 415)
                else Verdict.FAIL,
                status_code=resp.status,
            )
        )

        # ----- 7. Extremely large payload -----
        huge_payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "x" * 10_000_000}
            ],  # ~10MB
            "max_tokens": 1,
        }
        resp = await client.post_json(huge_payload)
        results.append(
            self.make_result(
                self.name,
                "10mb_payload",
                Verdict.PASS
                if resp.status in (200, 400, 413)
                else Verdict.FAIL,
                status_code=resp.status,
                detail=f"10MB content: status {resp.status}",
            )
        )

        # ----- 8. Mismatched Content-Length -----
        resp = await client.post_raw_http(
            valid_body,
            auto_content_length=False,
            content_length=len(valid_body) * 2,
            timeout=config.timeout * 0.33,
        )
        results.append(
            self.make_result(
                self.name,
                "mismatched_content_length",
                Verdict.FAIL
                if resp.status == 0 or resp.status >= 500
                else Verdict.PASS,
                status_code=resp.status,
                detail=f"Status {resp.status}"
                + (f" error: {resp.error}" if resp.error else ""),
            )
        )

        # ----- 9. Duplicate Content-Length headers -----
        resp = await client.post_raw_http(
            valid_body,
            auto_content_length=False,
            header_items=[
                ("Content-Length", str(len(valid_body))),
                ("Content-Length", str(len(valid_body) * 2)),
            ],
            timeout=config.timeout * 0.33,
        )
        results.append(
            self.make_result(
                self.name,
                "duplicate_content_length_headers",
                Verdict.FAIL
                if resp.status == 0 or resp.status >= 500
                else Verdict.PASS,
                status_code=resp.status,
                detail=f"Status {resp.status}"
                + (f" error: {resp.error}" if resp.error else ""),
            )
        )

        # ----- 10. Empty body with valid headers -----
        resp = await client.post_raw_bytes(b"", timeout=config.timeout * 0.33)
        results.append(
            self.make_result(
                self.name,
                "empty_body_valid_headers",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
            )
        )

        return results


def _gzip_compress(data: bytes) -> bytes:
    import gzip
    import io

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(data)
    return buf.getvalue()
