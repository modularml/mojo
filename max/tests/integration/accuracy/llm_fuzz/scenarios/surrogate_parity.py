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
Scenario: lone-surrogate streaming/non-streaming parity

A JSON request may legally carry an unpaired UTF-16 surrogate (e.g. an emoji
split by client-side truncation). Such a prompt must be handled consistently:
the same surrogate-containing prompt must yield the same status class on the
streaming and non-streaming chat paths (which differ only by ``stream`` and the
Accept header), the server must never return a 5xx, and -- the CENG-790
regression -- the streaming path must never answer HTTP 200 while smuggling a
5xx error object into an in-band SSE ``data:`` chunk.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RawResponse, RunConfig

# Lone low surrogate embedded in otherwise-ordinary text.
_LONE_SURROGATE = "hi \ude00 there"


def _content_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": _LONE_SURROGATE}],
        "max_tokens": 5,
    }


def _system_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": f"be {_LONE_SURROGATE} nice"},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 5,
    }


def _inband_error(response: RawResponse) -> str | None:
    """Returns the code/type of an error object leaked into an SSE stream, else None.

    A well-formed HTTP-200 stream carries content chunks and ``[DONE]``; an error
    object embedded in the stream while the status is 200 is the in-band leak
    from CENG-790. Returns the error's ``code`` (or ``type``) verbatim so a
    non-numeric code is reported rather than misread as a 5xx.
    """
    pieces = list(response.chunks or [])
    if not pieces and response.body:
        pieces = response.body.splitlines()
    for piece in pieces:
        piece = piece.strip()
        if piece.startswith("data:"):
            piece = piece[len("data:") :].strip()
        if not piece or piece == "[DONE]":
            continue
        try:
            obj = json.loads(piece)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
            error = obj["error"]
            return str(error.get("code") or error.get("type") or "error")
    return None


@register_scenario
class SurrogateStreamingParity(BaseScenario):
    name = "surrogate_streaming_parity"
    description = (
        "Lone UTF-16 surrogates yield consistent streaming/non-streaming "
        "results and never an in-band 5xx at HTTP 200"
    )
    tags = ["content", "tokenizer", "encoding", "streaming", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        payloads = {
            "content": _content_payload(config.model),
            "system_prompt": _system_payload(config.model),
        }

        for test_name, payload in payloads.items():
            try:
                sync_resp, _ = await self.timed_request(
                    client.post_json(dict(payload))
                )
                stream_resp, stream_elapsed = await self.timed_request(
                    client.post_streaming(dict(payload))
                )
            except Exception as e:
                results.append(
                    self.make_result(
                        self.name, test_name, Verdict.ERROR, error=str(e)
                    )
                )
                continue

            inband = _inband_error(stream_resp)
            if sync_resp.status == 0 or stream_resp.status == 0:
                # status 0 means the connection reset / timed out -- i.e. the
                # server crashed or hung on the input, the worst outcome and
                # exactly what this scenario guards against.
                verdict = Verdict.FAIL
                detail = (
                    f"connection failure (crash/hang/timeout): "
                    f"sync=[{sync_resp.status} {sync_resp.error}] "
                    f"stream=[{stream_resp.status} {stream_resp.error}]"
                )
            elif sync_resp.status >= 500 or stream_resp.status >= 500:
                verdict = Verdict.FAIL
                detail = (
                    f"server error: sync={sync_resp.status} "
                    f"stream={stream_resp.status}"
                )
            elif stream_resp.status == 200 and inband is not None:
                verdict = Verdict.FAIL
                detail = (
                    f"streaming leaked an in-band error ({inband}) while "
                    "returning HTTP 200"
                )
            elif sync_resp.status // 100 != stream_resp.status // 100:
                verdict = Verdict.FAIL
                detail = (
                    f"streaming/non-streaming status mismatch: "
                    f"sync={sync_resp.status} stream={stream_resp.status}"
                )
            else:
                verdict = Verdict.PASS
                detail = (
                    f"consistent: sync={sync_resp.status} "
                    f"stream={stream_resp.status}"
                )

            results.append(
                self.make_result(
                    self.name,
                    test_name,
                    verdict,
                    status_code=stream_resp.status,
                    elapsed_ms=stream_elapsed,
                    detail=detail,
                    response_body=stream_resp.body[:500],
                )
            )

        return results
