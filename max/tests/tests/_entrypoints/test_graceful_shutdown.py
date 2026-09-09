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
"""Tests for the SIGTERM graceful-shutdown drain timeout.

Runs the real server inside ``api_server.lifespan`` (as the entrypoint does)
and asserts ``graceful_shutdown_timeout_s`` bounds how long an in-flight
request is drained before it is cancelled.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import FastAPI
from max.pipelines.context import TextContext, TextGenerationOutput
from max.pipelines.lib import PipelineConfig
from max.pipelines.modeling.types import RequestID, TextGenerationInputs
from max.serve import api_server
from max.serve.config import Settings
from max.serve.pipelines.echo_gen import (
    EchoPipelineTokenizer,
    EchoTokenGenerator,
)
from uvicorn import Server


class MockHangingTokenGenerator(EchoTokenGenerator):
    """A pipeline whose ``execute`` never returns, so a request never finishes."""

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        time.sleep(3600)
        raise AssertionError("unreachable")


class MockCompletingTokenGenerator(EchoTokenGenerator):
    """A pipeline that stays in flight briefly, then echoes to completion."""

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        # Block only the first step so the request is reliably in flight at
        # shutdown, then echo to completion.
        if not getattr(self, "_drain_test_slept", False):
            self._drain_test_slept = True
            time.sleep(2.0)
        return super().execute(inputs)


async def _start_server(server: Server) -> tuple[asyncio.Task[None], int]:
    """Run ``server.serve()`` in the background; return its task and bound port."""
    serve_task = asyncio.ensure_future(server.serve())
    deadline = time.monotonic() + 10
    while not server.started:
        if serve_task.done():  # surface a startup failure instead of hanging
            serve_task.result()
        if time.monotonic() > deadline:
            raise TimeoutError("uvicorn server did not start in time")
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return serve_task, port


def _chat_completions_request(port: int) -> bytes:
    """A minimal chat-completions POST as raw HTTP bytes (avoids an HTTP dep)."""
    body = (
        b'{"model": "echo", "messages": '
        b'[{"role": "user", "content": "hi"}], "max_tokens": 128}'
    )
    return (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1:%d\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: %d\r\n"
        b"Connection: close\r\n"
        b"\r\n%s" % (port, len(body), body)
    )


@pytest.mark.parametrize("drain_timeout_s", [2, 5])
@pytest.mark.asyncio
async def test_graceful_shutdown_waits_for_drain_timeout(
    mock_pipeline_config: PipelineConfig,
    drain_timeout_s: int,
) -> None:
    """A request that never finishes is drained for the timeout, then cancelled."""
    settings = Settings(
        host="127.0.0.1",
        port=0,  # ephemeral
        graceful_shutdown_timeout_s=drain_timeout_s,
        use_heartbeat=False,
    )
    serving_settings = api_server.ServingTokenGeneratorSettings(
        model_factory=MockHangingTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=EchoPipelineTokenizer(),
    )
    app = api_server.fastapi_app(settings, serving_settings)
    config = api_server.fastapi_config(app, settings)
    assert config.timeout_graceful_shutdown == drain_timeout_s
    server = Server(config)

    async with api_server.lifespan(
        app, settings, serving_settings, app.state.zmq_endpoint_base
    ):
        serve_task, port = await _start_server(server)

        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(_chat_completions_request(port))
            await writer.drain()

            # Wait until the request is in flight (uvicorn tracks it in
            # server_state.tasks) before triggering shutdown.
            deadline = time.monotonic() + 15
            while not server.server_state.tasks:
                if time.monotonic() > deadline:
                    raise TimeoutError("request never reached the server")
                await asyncio.sleep(0.02)

            # should_exit is what uvicorn's SIGTERM handler sets.
            start = time.monotonic()
            server.should_exit = True

            await asyncio.wait_for(serve_task, timeout=drain_timeout_s + 30)
            elapsed = time.monotonic() - start
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    assert elapsed >= drain_timeout_s, (
        f"shut down after {elapsed:.2f}s, expected to drain at least "
        f"{drain_timeout_s}s"
    )
    # Slack past the timeout is just worker-subprocess teardown.
    assert elapsed < drain_timeout_s + 15, (
        f"shut down after {elapsed:.2f}s, expected close to {drain_timeout_s}s"
    )


@pytest.mark.asyncio
async def test_idle_server_shuts_down_promptly() -> None:
    """An idle server shuts down promptly -- the timeout is an upper bound."""
    settings = Settings(
        host="127.0.0.1",
        port=0,
        graceful_shutdown_timeout_s=30,
        use_heartbeat=False,
    )
    config = api_server.fastapi_config(FastAPI(), settings)
    server = Server(config)
    serve_task, _ = await _start_server(server)

    start = time.monotonic()
    server.should_exit = True
    await asyncio.wait_for(serve_task, timeout=10)
    elapsed = time.monotonic() - start

    assert elapsed < 5, f"idle shutdown took {elapsed:.2f}s, expected prompt"


@pytest.mark.asyncio
async def test_completed_request_does_not_block_shutdown(
    mock_pipeline_config: PipelineConfig,
) -> None:
    """A request that finishes during the drain lets shutdown return promptly,
    with a normal 200 response."""
    drain_timeout_s = 30
    settings = Settings(
        host="127.0.0.1",
        port=0,
        graceful_shutdown_timeout_s=drain_timeout_s,
        use_heartbeat=False,
    )
    serving_settings = api_server.ServingTokenGeneratorSettings(
        model_factory=MockCompletingTokenGenerator,
        pipeline_config=mock_pipeline_config,
        tokenizer=EchoPipelineTokenizer(),
    )
    app = api_server.fastapi_app(settings, serving_settings)
    config = api_server.fastapi_config(app, settings)
    assert config.timeout_graceful_shutdown == drain_timeout_s
    server = Server(config)

    async with api_server.lifespan(
        app, settings, serving_settings, app.state.zmq_endpoint_base
    ):
        serve_task, port = await _start_server(server)

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(_chat_completions_request(port))
            await writer.drain()

            deadline = time.monotonic() + 15
            while not server.server_state.tasks:
                if time.monotonic() > deadline:
                    raise TimeoutError("request never reached the server")
                await asyncio.sleep(0.02)

            start = time.monotonic()
            server.should_exit = True

            await asyncio.wait_for(serve_task, timeout=drain_timeout_s + 30)
            elapsed = time.monotonic() - start
            response = await asyncio.wait_for(reader.read(), timeout=5)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    status_line = response.split(b"\r\n", 1)[0]
    assert b"200" in status_line, f"expected 200, got: {status_line!r}"
    # Returned shortly after the request finished, not at the full timeout.
    assert elapsed < 10, (
        f"shut down after {elapsed:.2f}s, expected prompt return after the "
        "request completed"
    )
