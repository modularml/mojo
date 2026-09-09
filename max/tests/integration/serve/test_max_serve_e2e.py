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
"""End-to-end tests for MAX serve.

This test suite validates server functionality by launching max serve in a
subprocess and testing various API endpoints.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import signal
import time
from collections.abc import Generator
from multiprocessing.context import SpawnProcess

import httpx
import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)

PORT = 8000
METRICS_PORT = 8001
MODEL = "modularai/SmolLM-135M-Instruct-FP32"
BASE_URL = f"http://127.0.0.1:{PORT}"
HEALTH_URL = f"{BASE_URL}/health"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/v1/chat/completions"

# Drain window for `test_sigterm_drains_in_flight_request`. Generous, because
# the point of that test is that shutdown is bounded by the in-flight request
# finishing, not by this timeout firing. Shutdown still returns as soon as the
# request completes, so the long window costs the other tests nothing.
DRAIN_TIMEOUT_S = 120
# Long enough that SIGTERM lands mid-generation even on a fast machine, short
# enough to still fit the drain window on a slow one.
DRAIN_MAX_TOKENS = 100


def serve_main() -> None:
    """Entrypoint to run max serve in subprocess.

    This function configures and launches the serve API server with model worker.
    It blocks in uvloop.run(server.serve()) until the server is shut down.
    """
    from max._entrypoints.cli.serve.serve_api_and_model_worker import (
        serve_api_server_and_model_worker,
    )
    from max.driver import DeviceSpec
    from max.pipelines import PipelineArgs
    from max.serve.config import Settings

    settings = Settings(
        port=PORT,
        metrics_port=METRICS_PORT,
        graceful_shutdown_timeout_s=DRAIN_TIMEOUT_S,
    )
    # Configure pipeline with GGUF model for fast loading on CPU
    pipeline_config = PipelineArgs(
        model_path=MODEL,
        device_specs=[DeviceSpec.cpu()],
        quantization_encoding="float32",
    )
    # Launch server (blocks until shutdown)
    serve_api_server_and_model_worker(settings, pipeline_config)


async def wait_for_server_ready(
    proc: SpawnProcess, url: str, timeout: float
) -> None:
    """Poll server health endpoint until ready or timeout.

    Args:
        proc: Server process to monitor
        url: Health check URL (e.g., "http://127.0.0.1:8000/health")
        timeout: Maximum seconds to wait for server readiness

    Raises:
        RuntimeError: If server process exits before becoming ready
        TimeoutError: If server doesn't become ready within timeout
    """
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        while time.monotonic() - start < timeout:
            try:
                response = await client.get(url, timeout=1.0)
                if response.status_code == 200:
                    elapsed = time.monotonic() - start
                    logger.info(f"Server ready after {elapsed:.1f}s")
                    return
            except (httpx.RequestError, httpx.TimeoutException):
                pass

            await asyncio.sleep(1)

            if not proc.is_alive():
                raise RuntimeError(
                    f"Server process exited with code {proc.exitcode}"
                )

    raise TimeoutError(
        f"Server at {url} did not become ready within {timeout}s"
    )


@pytest.fixture(scope="module")
def _server_processes() -> Generator[list[SpawnProcess], None, None]:
    """Owns every server started for this module so all of them get reaped.

    Yields:
        The list that each started server is appended to.
    """
    processes: list[SpawnProcess] = []
    yield processes

    for server_process in processes:
        # SIGTERM first: it unwinds the serving stack, which is what reaps the
        # model worker subprocess. Going straight to SIGKILL would orphan it.
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=30)
        # A server still up 30s after SIGTERM is itself a bug (SERVSYS-1197),
        # but failing teardown over it would only mask the tests' own results.
        if server_process.is_alive():
            server_process.kill()
            server_process.join(timeout=10)


@pytest_asyncio.fixture
async def max_serve_server(
    _server_processes: list[SpawnProcess],
) -> SpawnProcess:
    """Pytest fixture that launches max serve and waits for it to be ready.

    The server is started once and shared across all tests in this file for
    efficiency. The fixture is nonetheless function-scoped so that a test which
    shuts the server down (see ``test_sigterm_drains_in_flight_request``)
    doesn't strand the tests after it: the next test to ask for a server gets a
    fresh one, and only that test pays for the restart.

    Returns:
        The running server process, serving on ``BASE_URL``.

    Note:
        - Uses non-daemon Process so the server can spawn child processes
        - Timeout is generous to allow for model download and compilation
    """
    if _server_processes and _server_processes[-1].is_alive():
        return _server_processes[-1]

    # Use spawn method to ensure clean process separation
    ctx = multiprocessing.get_context("spawn")
    server_process = ctx.Process(target=serve_main)
    server_process.start()
    _server_processes.append(server_process)

    # Huge timeout for model download + compile (and ASAN CI is super slow)
    await wait_for_server_ready(server_process, HEALTH_URL, timeout=900)
    return server_process


@pytest.mark.asyncio
async def test_chat_completions(max_serve_server: SpawnProcess) -> None:
    """Test basic chat completions endpoint.

    This test validates:
    1. Server accepts chat completion requests
    2. Response structure matches OpenAI API format
    3. Generated text is present in response
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            CHAT_COMPLETIONS_URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 10,
            },
        )

    # Verify response structure
    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}: {response.text}"
    )

    data = response.json()
    assert "choices" in data, f"Missing 'choices' in response: {data}"
    assert len(data["choices"]) > 0, "Expected at least one choice in response"
    assert "message" in data["choices"][0], (
        f"Missing 'message' in choice: {data['choices'][0]}"
    )
    assert "content" in data["choices"][0]["message"], (
        f"Missing 'content' in message: {data['choices'][0]['message']}"
    )

    content = data["choices"][0]["message"]["content"]
    assert len(content) > 0, "Expected non-empty generated content"
    logger.info(f"Text generation successful. Response: {content}")


@pytest.mark.asyncio
async def test_health_endpoint(max_serve_server: SpawnProcess) -> None:
    """Test health check endpoint returns 200 OK."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(HEALTH_URL)

    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}"
    )


# Keep this test last: it shuts the shared server down, so anything after it
# pays for a restart.
@pytest.mark.asyncio
async def test_sigterm_drains_in_flight_request(
    max_serve_server: SpawnProcess,
) -> None:
    """SIGTERM lets an in-flight request finish instead of cutting it off.

    Streams a fixed-length generation, sends a real SIGTERM to the server
    process once tokens are flowing, and requires the stream to reach its
    normal end. A server that dies on the signal truncates the stream instead.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Tell me a long story."}],
        # Suppressing EOS pins the length to max_tokens, so the generation
        # can't stop early and leave the test asserting nothing.
        "max_tokens": DRAIN_MAX_TOKENS,
        "ignore_eos": True,
        "stream": True,
    }

    signal_sent = False
    tokens_after_signal = 0
    finish_reason: str | None = None
    stream_error: Exception | None = None

    async with httpx.AsyncClient(timeout=DRAIN_TIMEOUT_S) as client:
        try:
            async with client.stream(
                "POST", CHAT_COMPLETIONS_URL, json=payload
            ) as response:
                assert response.status_code == 200, (
                    f"Expected status 200, got {response.status_code}"
                )

                # Read to the end of the stream rather than breaking on
                # [DONE], so the response iterator closes itself.
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        continue

                    choice = json.loads(data)["choices"][0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    if not choice.get("delta", {}).get("content"):
                        continue

                    if signal_sent:
                        tokens_after_signal += 1
                        continue

                    # A token has been generated, so the request is genuinely
                    # in flight and the drain has something to protect.
                    assert max_serve_server.pid is not None
                    os.kill(max_serve_server.pid, signal.SIGTERM)
                    signal_sent = True
                    logger.info(
                        "Sent SIGTERM to server pid %d", max_serve_server.pid
                    )
        except httpx.HTTPError as e:
            stream_error = e

    assert signal_sent, "Server produced no tokens, so SIGTERM was never sent"
    assert stream_error is None, (
        f"Stream aborted {tokens_after_signal} tokens after SIGTERM instead of "
        f"draining: {stream_error!r}"
    )
    assert finish_reason == "length", (
        f"Expected the {DRAIN_MAX_TOKENS}-token generation to run to "
        f"completion, got finish_reason={finish_reason!r} after "
        f"{tokens_after_signal} tokens post-SIGTERM"
    )
    # Ordered after the assertion above so a broken drain reports as a broken
    # drain. Reaching here with no post-signal tokens means the generation beat
    # the signal, so the drain was never exercised and the pass is hollow.
    assert tokens_after_signal > 0, (
        "Generation finished before SIGTERM arrived, so the drain was never "
        f"exercised; raise DRAIN_MAX_TOKENS above {DRAIN_MAX_TOKENS}"
    )

    # Draining is an upper bound, not a floor: once the request is done the
    # server should go away on its own.
    max_serve_server.join(timeout=30)
    assert not max_serve_server.is_alive(), (
        "Server still running after draining the in-flight request"
    )
