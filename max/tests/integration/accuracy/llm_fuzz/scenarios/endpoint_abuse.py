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
Scenario: Endpoint and protocol abuse
Target: Non-inference endpoints (/health, /metrics, /v1/models), wrong HTTP
        methods, invalid paths, path traversal, and query string injection.

Dynamo frontend exposes: /v1/chat/completions, /v1/models, /health, /metrics
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class EndpointAbuse(BaseScenario):
    name = "endpoint_abuse"
    description = "Non-inference endpoints, wrong HTTP methods, path traversal, query injection"
    tags = ["endpoint", "protocol", "dynamo"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        # ----- 1. Metrics endpoint scrape -----
        resp = await client.get_path("/metrics", timeout=config.timeout * 0.33)
        is_prometheus = (
            "# HELP" in resp.body or "# TYPE" in resp.body or resp.status == 404
        )
        results.append(
            self.make_result(
                self.name,
                "metrics_scrape",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail=f"GET /metrics: status {resp.status}, prometheus format: {is_prometheus}",
            )
        )

        # ----- 2. Metrics under load -----
        # 100 concurrent metrics scrapes while also sending inference requests
        async def metrics_flood() -> list[Any]:
            tasks = [
                client.get_path("/metrics", timeout=config.timeout * 0.17)
                for _ in range(100)
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        async def inference_during() -> list[Any]:
            payloads = [
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "max_tokens": 5,
                }
                for _ in range(5)
            ]
            return await client.concurrent_requests(payloads, max_concurrent=5)

        gather_results = await asyncio.gather(
            metrics_flood(),
            inference_during(),
        )
        metrics_results_list: list[Any] = gather_results[0]
        metrics_ok = sum(
            1
            for r in metrics_results_list
            if not isinstance(r, BaseException)
            and r.status > 0
            and r.status < 500
        )
        metrics_server_errors = sum(
            1
            for r in metrics_results_list
            if not isinstance(r, BaseException) and r.status >= 500
        )
        metrics_conn_errors = sum(
            1
            for r in metrics_results_list
            if not isinstance(r, BaseException) and r.status == 0
        )
        metrics_exceptions = sum(
            1 for r in metrics_results_list if isinstance(r, Exception)
        )
        metrics_total_errors = (
            metrics_server_errors + metrics_conn_errors + metrics_exceptions
        )
        results.append(
            self.make_result(
                self.name,
                "metrics_under_load",
                Verdict.FAIL if metrics_total_errors > 20 else Verdict.PASS,
                detail=f"{metrics_ok}/100 ok, {metrics_server_errors} server errors, {metrics_conn_errors} connection errors, {metrics_exceptions} exceptions — concurrent metrics during inference",
            )
        )

        # ----- 3. Health rapid poll -----
        health_tasks = [
            client.get_path("/health", timeout=config.timeout * 0.17)
            for _ in range(200)
        ]
        health_results_list = await asyncio.gather(
            *health_tasks, return_exceptions=True
        )
        health_ok = sum(
            1
            for r in health_results_list
            if not isinstance(r, BaseException) and r.status == 200
        )
        health_errors = sum(
            1
            for r in health_results_list
            if not isinstance(r, BaseException) and r.status >= 500
        )
        results.append(
            self.make_result(
                self.name,
                "health_rapid_poll_200",
                Verdict.FAIL if health_errors > 20 else Verdict.PASS,
                detail=f"{health_ok}/200 ok, {health_errors} server errors — rapid health polling",
            )
        )

        # ----- 4. Invalid API paths -----
        invalid_paths = [
            "/v1/completions",  # Legacy completions (not chat)
            "/v2/chat/completions",  # Wrong version
            "/v1/embeddings",  # Embeddings endpoint
            "/v1/images/generations",  # Image generation
            "/v1/audio/transcriptions",  # Audio
            "/api/v1/chat/completions",  # Wrong prefix
            "/chat/completions",  # Missing /v1
            "/v1/chat",  # Incomplete path
        ]
        for path in invalid_paths:
            resp = await client.method_to_path(
                "POST",
                path,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5,
                },
                timeout=config.timeout * 0.17,
            )
            results.append(
                self.make_result(
                    self.name,
                    f"invalid_path_{path.replace('/', '_').strip('_')}",
                    Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                    status_code=resp.status,
                    detail=f"POST {path}: status {resp.status}",
                )
            )

        # ----- 5. HTTP method abuse on chat completions -----
        for method in ["GET", "PUT", "DELETE", "PATCH"]:
            resp = await client.method_to_path(
                method, "/v1/chat/completions", timeout=config.timeout * 0.17
            )
            results.append(
                self.make_result(
                    self.name,
                    f"method_{method.lower()}_on_chat",
                    Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                    status_code=resp.status,
                    detail=f"{method} /v1/chat/completions: status {resp.status}",
                )
            )

        # ----- 6. POST to health endpoint -----
        resp = await client.post_to_path(
            "/health", {"test": True}, timeout=config.timeout * 0.17
        )
        results.append(
            self.make_result(
                self.name,
                "post_to_health",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail=f"POST /health with body: status {resp.status}",
            )
        )

        # ----- 7. POST to metrics endpoint -----
        resp = await client.post_to_path(
            "/metrics", {"test": True}, timeout=config.timeout * 0.17
        )
        results.append(
            self.make_result(
                self.name,
                "post_to_metrics",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail=f"POST /metrics with body: status {resp.status}",
            )
        )

        # ----- 8. Path traversal -----
        traversal_paths = [
            "/../../../etc/passwd",
            "/v1/../v1/models",
            "/v1/chat/completions/../../../etc/passwd",
            "/%2e%2e/%2e%2e/etc/passwd",
            "/v1/chat/completions%00",
            "/v1/chat/completions/../../admin",
        ]
        for i, path in enumerate(traversal_paths):
            resp = await client.method_to_path(
                "GET", path, timeout=config.timeout * 0.17
            )
            suspicious = "root:" in resp.body or "bin/bash" in resp.body
            results.append(
                self.make_result(
                    self.name,
                    f"path_traversal_{i}",
                    Verdict.FAIL
                    if suspicious or resp.status >= 500
                    else Verdict.PASS,
                    status_code=resp.status,
                    detail=f"GET {path}: status {resp.status}, suspicious content: {suspicious}",
                )
            )

        # ----- 9. Query string abuse -----
        query_paths = [
            "/v1/chat/completions?admin=true",
            "/v1/chat/completions?debug=1",
            "/v1/chat/completions?format=raw",
            "/v1/models?" + "a=b&" * 1000,  # Long query string
        ]
        for i, path in enumerate(query_paths):
            resp = await client.method_to_path(
                "POST",
                path,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 5,
                },
                timeout=config.timeout * 0.33,
            )
            results.append(
                self.make_result(
                    self.name,
                    f"query_string_{i}",
                    Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                    status_code=resp.status,
                    detail=f"POST {path[:80]}: status {resp.status}",
                )
            )

        # ----- 10. OPTIONS request (CORS) -----
        resp = await client.method_to_path(
            "OPTIONS", "/v1/chat/completions", timeout=config.timeout * 0.17
        )
        results.append(
            self.make_result(
                self.name,
                "options_cors",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail=f"OPTIONS /v1/chat/completions: status {resp.status}",
            )
        )

        # ----- 11. v1/models rapid poll under load -----
        models_tasks = [
            client.get_path("/v1/models", timeout=config.timeout * 0.17)
            for _ in range(100)
        ]
        models_results_list = await asyncio.gather(
            *models_tasks, return_exceptions=True
        )
        models_ok = sum(
            1
            for r in models_results_list
            if not isinstance(r, BaseException) and r.status == 200
        )
        models_errors = sum(
            1
            for r in models_results_list
            if not isinstance(r, BaseException) and r.status >= 500
        )
        results.append(
            self.make_result(
                self.name,
                "v1_models_rapid_poll_100",
                Verdict.FAIL if models_errors > 10 else Verdict.PASS,
                detail=f"{models_ok}/100 ok, {models_errors} server errors — rapid /v1/models polling",
            )
        )

        # ----- 12. Health check -----
        await asyncio.sleep(2)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_endpoint_abuse_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )

        return results
