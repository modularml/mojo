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
Scenario: Connection exhaustion
Target: Socket-level attacks that leave connections dangling, exhaust
        connection pools, or trigger resource leaks in the HTTP server layer.

Unlike payload-level attacks, these target the transport layer.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class ConnectionExhaustion(BaseScenario):
    name = "connection_exhaustion"
    description = "Socket exhaustion, half-open connections, abrupt resets, slowloris, pipelining"
    tags = ["connection", "crash", "resource_leak"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        # ----- 1. Open many connections without closing -----
        # Send partial HTTP headers but never complete the request
        hold_tasks = [
            client.raw_socket_hold(
                timeout=config.timeout * 0.17, send_partial=True
            )
            for _ in range(50)
        ]
        held_results = await asyncio.gather(*hold_tasks, return_exceptions=True)
        connected = sum(
            1
            for r in held_results
            if not isinstance(r, BaseException) and r.error is None
        )
        results.append(
            self.make_result(
                self.name,
                "open_50_connections_no_close",
                Verdict.PASS,  # This is about the server surviving, not about connection success
                detail=f"{connected}/50 connections held open for 5s with partial headers",
            )
        )

        # Clean up held sockets
        client.close_held_sockets()
        await asyncio.sleep(2)

        # ----- 2. Verify server still works after held connections -----
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "health_after_held_connections",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Healthy"
                if health.status == 200
                else f"Unhealthy after 50 held connections: {health.error}",
            )
        )

        # ----- 3. Client reset immediately after sending request -----
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a detailed essay about the history of computing.",
                    }
                ],
                "max_tokens": 4096,
            }
        ).encode()
        resp = await client.post_raw_http(
            body,
            timeout=config.timeout * 0.17,
            reset_after_send=True,
        )
        results.append(
            self.make_result(
                self.name,
                "client_reset_after_request_send",
                Verdict.PASS,  # We just care that the server survives
                detail=f"Socket reset right after request send: error={resp.error}",
            )
        )

        # ----- 4. Client reset after sending streaming request -----
        stream_body = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a very long story about dragons.",
                    }
                ],
                "max_tokens": 8192,
                "stream": True,
            }
        ).encode()
        resp = await client.post_raw_http(
            stream_body,
            header_items=[("Accept", "text/event-stream")],
            timeout=config.timeout * 0.17,
            reset_after_send=True,
        )
        results.append(
            self.make_result(
                self.name,
                "client_reset_after_stream_request_send",
                Verdict.PASS,
                detail=f"Streaming request reset after send: error={resp.error}",
            )
        )

        # ----- 5. Rapid connect-disconnect 100 times -----
        async def rapid_connect() -> bool:
            loop = asyncio.get_event_loop()

            def _connect_disconnect() -> bool:
                try:
                    s = client._make_socket(2)
                    s.close()
                    return True
                except Exception:
                    return False

            return await loop.run_in_executor(None, _connect_disconnect)

        connect_tasks = [rapid_connect() for _ in range(100)]
        connect_results = await asyncio.gather(
            *connect_tasks, return_exceptions=True
        )
        connected = sum(1 for r in connect_results if r is True)
        results.append(
            self.make_result(
                self.name,
                "rapid_connect_disconnect_100",
                Verdict.PASS,
                detail=f"{connected}/100 rapid connect-disconnects succeeded",
            )
        )

        # ----- 6. Verify health after rapid connections -----
        await asyncio.sleep(2)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "health_after_rapid_connections",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )

        # ----- 7. Slowloris: 50 slow connections -----
        # Each sends body at 1 byte per 2 seconds using the slow body method
        slow_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
        }
        slow_tasks = [
            client.post_slow_body(slow_payload, chunk_delay=2.0, chunk_size=5)
            for _ in range(20)  # 20 is enough to stress without taking forever
        ]
        slow_results = await asyncio.gather(*slow_tasks, return_exceptions=True)
        slow_ok = sum(
            1
            for r in slow_results
            if not isinstance(r, BaseException) and r.status == 200
        )
        slow_errors = sum(
            1
            for r in slow_results
            if not isinstance(r, BaseException)
            and (r.status >= 500 or r.status == 0)
        )
        results.append(
            self.make_result(
                self.name,
                "slowloris_20_connections",
                Verdict.FAIL
                if slow_errors > 10
                else (Verdict.INTERESTING if slow_errors > 0 else Verdict.PASS),
                detail=f"{slow_ok}/20 ok, {slow_errors} errors — slow body delivery",
            )
        )

        # ----- 8. HTTP pipelining: multiple requests on one connection -----
        req1 = client.build_raw_request(
            "POST",
            client._chat_path,
            body=json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "one"}],
                    "max_tokens": 5,
                }
            ).encode(),
        )
        req2 = client.build_raw_request(
            "POST",
            client._chat_path,
            body=json.dumps(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "two"}],
                    "max_tokens": 5,
                }
            ).encode(),
        )
        pipe_result = await client.send_pipelined_requests(
            [req1, req2], timeout=config.timeout * 0.33
        )
        per_response_statuses = []
        for line in pipe_result.body.splitlines():
            if line.startswith(("HTTP/1.1 ", "HTTP/1.0 ")):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    per_response_statuses.append(int(parts[1]))

        response_markers = pipe_result.body.count(
            "HTTP/1.1 "
        ) + pipe_result.body.count("HTTP/1.0 ")
        if pipe_result.status == 0 or pipe_result.error == "TIMEOUT":
            verdict = Verdict.FAIL
            detail = f"Pipelined request timed out or failed: error={pipe_result.error}"
        elif (
            any(status >= 500 for status in per_response_statuses)
            or pipe_result.status >= 500
        ):
            verdict = Verdict.FAIL
            status_str = ", ".join(
                str(status) for status in per_response_statuses
            ) or str(pipe_result.status)
            detail = (
                f"Pipelined request triggered server error(s): {status_str}"
            )
        elif response_markers >= 2:
            verdict = Verdict.PASS
            status_str = (
                ", ".join(str(status) for status in per_response_statuses)
                or "unknown"
            )
            detail = f"Received {response_markers} HTTP responses on one connection ({status_str})"
        elif response_markers == 1:
            verdict = Verdict.INTERESTING
            detail = (
                "Server produced only one HTTP response to pipelined requests"
            )
        else:
            verdict = Verdict.INTERESTING
            detail = (
                f"No complete HTTP response parsed, error={pipe_result.error}"
            )

        results.append(
            self.make_result(
                self.name,
                "pipelined_requests",
                verdict,
                detail=detail,
            )
        )

        # ----- 9. Keep-alive hold: get response then hold connection idle -----
        async def keepalive_hold() -> int | str:
            loop = asyncio.get_event_loop()

            def _hold() -> int | str:
                try:
                    conn = client._make_conn(timeout=config.timeout)
                    body = (
                        b'{"model": "'
                        + model.encode()
                        + b'", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}'
                    )
                    headers = client._base_headers(
                        {
                            "Content-Length": str(len(body)),
                            "Connection": "keep-alive",
                        }
                    )
                    conn.request(
                        "POST", client._chat_path, body=body, headers=headers
                    )
                    resp = conn.getresponse()
                    _ = resp.read()
                    status = resp.status
                    # Hold the connection open for 15 seconds without sending anything
                    time.sleep(15)
                    conn.close()
                    return status
                except Exception as e:
                    return str(e)[:200]

            return await loop.run_in_executor(None, _hold)

        keepalive_result = await keepalive_hold()
        results.append(
            self.make_result(
                self.name,
                "keep_alive_hold_15s",
                Verdict.PASS
                if isinstance(keepalive_result, int) and keepalive_result == 200
                else Verdict.INTERESTING,
                detail=f"Keep-alive held 15s: {keepalive_result}",
            )
        )

        # ----- 10. Recovery: final health check -----
        await asyncio.sleep(3)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_connection_exhaustion_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )

        return results
