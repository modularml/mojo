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
"""End-to-end tests parameterized over every Cascade :py:class:`Runtime`.

Runs the same suite (scalar call, streaming call, dummy text-gen pipeline,
varying generation lengths) against:

* :py:class:`LocalRuntime` -- in-process baseline.
* :py:class:`SubprocHttpRuntime` -- uvicorn HTTP server in a child process,
  :py:class:`HttpRuntimeProxy` client over a unix domain socket.

The dummy text-gen pipeline chains
``tokenizer.encode -> transformer.decode -> tokenizer.decode_streaming``,
which forces the runtime to forward intermediate result IDs between workers
(p2p) rather than round-tripping every value through the client.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import socket
import sys
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
from max.experimental.cascade import (
    ChatMessage,
    GenAIRequest,
    GenAITextChunk,
    LocalRuntime,
    Result,
    Runtime,
    TextGenOptions,
    Worker,
    worker_method,
)
from max.experimental.cascade.core.pipeline_method import (
    _pipeline_method_scope,
)
from max.experimental.cascade.grpc_runtime import SubprocGrpcRuntimeClient
from max.experimental.cascade.grpc_runtime import server as grpc_server
from max.experimental.cascade.grpc_runtime.client import GrpcRuntimeClient
from max.experimental.cascade.http_runtime import SubprocHttpRuntime
from max.experimental.cascade.http_runtime import server as http_server
from max.experimental.cascade.http_runtime.client import HttpRuntimeProxy
from max.experimental.cascade.http_runtime.subproc import (
    _wait_until_alive as _http_wait_until_alive,
)
from max.experimental.cascade.pipelines.dummy_textgen import (
    build_dummy_textgen_pipeline,
)
from max.serve.process_control import subprocess_manager
from pydantic import BaseModel

# The HTTP server subprocess imports uvicorn, fastapi and prometheus_client,
# which costs ~35s before it answers `/alive` on a contended macOS CI worker --
# past the 30s readiness budget in `_wait_until_alive`. gRPC answers in ~11s.
# TODO(SERVSYS-1301): re-enable once HTTP server startup fits the budget.
_SKIP_HTTP = sys.platform == "darwin"
_SKIP_HTTP_REASON = "Cascade HTTP startup exceeds its budget on macOS CI"

# The transports the shared-runtime suite runs against. ``LocalRuntime`` is
# in-process; the other two are backed by a server subprocess (see
# ``_shared_server``).
_RUNTIME_TRANSPORTS = (
    ["local", "grpc"] if _SKIP_HTTP else ["local", "grpc", "http"]
)


async def _grpc_ready(target: str) -> None:
    """Block until the gRPC server at ``target`` answers (``__aenter__`` polls)."""
    async with GrpcRuntimeClient(target):
        pass


async def _http_ready(address: str) -> None:
    """Block until the HTTP server at ``address`` reports alive."""
    proxy = HttpRuntimeProxy(address)
    async with proxy.session() as session:
        await _http_wait_until_alive(session, proxy._base_url)


@pytest_asyncio.fixture(
    params=_RUNTIME_TRANSPORTS,
    ids=lambda transport: transport,
    scope="module",
)
async def _shared_server(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Callable[[], Runtime]]:
    """Boot one server per subprocess-backed transport, shared across the module.

    Yields a zero-arg opener returning an *unentered* :py:class:`Runtime`
    pointed at the shared server (or a fresh in-process :py:class:`LocalRuntime`,
    which is cheap enough to build per test). Booting the server once per module
    -- rather than once per test case -- avoids ~40 subprocess spawn/teardown
    cycles across this file, each of which pays the subprocess SIGTERM-grace
    teardown (especially costly under ASAN). The subprocess lifecycle is owned
    by :py:func:`~max.serve.process_control.subprocess_manager` (the same code
    the real runtimes use); tests just dial the server with lightweight
    per-test clients.
    """
    transport: str = request.param
    if transport == "local":
        yield LocalRuntime
        return

    sock_path = f"/tmp/max-test-{uuid.uuid4().hex[:12]}.sock"
    address = f"unix://{sock_path}"
    if transport == "grpc":
        server_fn: Callable[[str], None] = grpc_server.serve
        opener: Callable[[], Runtime] = functools.partial(
            GrpcRuntimeClient, f"unix:{sock_path}"
        )
        ready = _grpc_ready(f"unix:{sock_path}")
    else:
        server_fn = http_server.serve
        opener = functools.partial(HttpRuntimeProxy, address)
        ready = _http_ready(address)

    # Manage the server subprocess (loop-agnostic, module-lived) and its client
    # (event-loop-bound, per-test) separately rather than reusing the bundled
    # ``Subproc*Runtime`` classes, whose single ``async with`` opens a client on
    # this fixture's loop that the per-function test loops then can't await.
    try:
        async with subprocess_manager(
            f"cascade {transport} test server"
        ) as proc:
            proc.start(server_fn, address)
            await asyncio.wait_for(ready, timeout=30.0)
            yield opener
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path)


@pytest_asyncio.fixture
async def runtime(
    _shared_server: Callable[[], Runtime],
) -> AsyncIterator[Runtime]:
    """Open a per-test client to the module's shared runtime.

    The client (a gRPC channel / aiohttp session, or a fresh in-process
    ``LocalRuntime``) is cheap; the server subprocess it talks to is shared
    across the whole module by :py:func:`_shared_server`.
    """
    async with _shared_server() as rt:
        yield rt


class Person(BaseModel):
    name: str
    age: int
    is_lefthanded: bool


class _Echo(Worker):
    """Minimal worker covering scalar and streaming worker methods."""

    @worker_method()
    async def add(self, a: int, b: int) -> int:
        return a + b

    @worker_method()
    async def count(self, n: int) -> AsyncIterator[int]:
        for i in range(n):
            yield i

    @worker_method()
    async def hello(self, person: Person) -> str:
        return "Hello " + person.name + ", how are you?"

    @worker_method()
    async def add_person(
        self, name: str, age: int, is_lefthanded: bool
    ) -> Person:
        return Person(name=name, age=age, is_lefthanded=is_lefthanded)

    @worker_method()
    async def greet(self, name: str, excited: bool = False) -> str:
        return f"Hi {name}" + ("!" if excited else ".")

    @worker_method()
    async def total(self, nums: list[int]) -> int:
        return sum(nums)

    @worker_method()
    async def hello_all(self, people: list[Person]) -> str:
        return "Hello " + ", ".join(p.name for p in people) + "!"


class _Failing(Worker):
    """Worker whose methods fail in various ways."""

    @worker_method()
    async def raise_value_error(self, message: str) -> str:
        raise ValueError(message)

    @worker_method()
    async def exit_hard(self) -> str:
        # Ensure exit happens after server is setup, instead of
        # racing between server shutdown and `sys.exit(1)`.
        await asyncio.sleep(1)
        sys.exit(1)

    @worker_method()
    async def slow_ok(self) -> str:
        await asyncio.sleep(10)
        return "done"


@pytest.mark.asyncio
async def test_scalar_call(runtime: Runtime) -> None:
    """A coroutine worker method round-trips its scalar return value.

    Two awaits per terminal call: the first gets the :py:class:`Result`
    handle, the second resolves it. Code inside a ``@pipeline_method``
    body skips the second await -- the consuming worker method's
    ``MaybeAsync`` argument resolution does it automatically.
    """
    async with _pipeline_method_scope():
        echo = await runtime.deploy(_Echo())

        add_handle = await echo.add(2, 3)
        assert await add_handle == 5

        add_handle = await echo.add(10, -4)
        assert await add_handle == 6


@pytest.mark.asyncio
async def test_basemodel_type_inference_result(runtime: Runtime) -> None:
    """Test that `pydantic.BaseModel`s get type inferred and decoded properly

    We should not have to explicitly specify which class an argument gets decoded into.
    """
    async with _pipeline_method_scope():
        echo = await runtime.deploy(_Echo())
        person_handle = await echo.add_person("Lebron", 41, False)
        greeting_handle = await echo.hello(person_handle)
        assert await greeting_handle == "Hello Lebron, how are you?"


@pytest.mark.asyncio
async def test_kwargs_type_inference(runtime: Runtime) -> None:
    """Kwargs decode by name while positional args bind to the leading params.

    Proxy methods are statically positional-only, so exercise the kwargs
    wire path through ``call_method`` directly.
    """
    worker_id = await runtime.deploy_worker(_Echo())
    # All-kwargs call.
    async with runtime.call_method(
        worker_id, "add", (), {"a": 2, "b": 3}
    ) as result_id:
        assert await runtime.get_result(result_id) == 5
    # Mixed positional + kwargs call.
    async with runtime.call_method(
        worker_id, "add", (2,), {"b": 4}
    ) as result_id:
        assert await runtime.get_result(result_id) == 6


@pytest.mark.asyncio
async def test_kwargs_basemodel_type_inference(runtime: Runtime) -> None:
    """A result forwarded as a kwarg decodes into its `pydantic.BaseModel`."""
    worker_id = await runtime.deploy_worker(_Echo())
    async with runtime.call_method(
        worker_id, "add_person", ("Lebron", 41, False), {}
    ) as person_id:
        person_handle: Result[Person] = Result(person_id, runtime)
        async with runtime.call_method(
            worker_id, "hello", (), {"person": person_handle}
        ) as result_id:
            assert (
                await runtime.get_result(result_id)
                == "Hello Lebron, how are you?"
            )


@pytest.mark.asyncio
async def test_basemodel_list_type_inference(runtime: Runtime) -> None:
    """Test a container of pydantic.BaseModels"""
    async with _pipeline_method_scope():
        echo = await runtime.deploy(_Echo())
        people = [
            Person(name="Lebron", age=41, is_lefthanded=False),
            Person(name="Max", age=3, is_lefthanded=True),
        ]
        greeting_handle = await echo.hello_all(people)
        assert await greeting_handle == "Hello Lebron, Max!"


@pytest.mark.asyncio
async def test_container_type_inference(runtime: Runtime) -> None:
    """A non-``BaseModel`` hint (``list[int]``) decodes through the same
    hinted path as models, not the generic-JSON fallback."""
    async with _pipeline_method_scope():
        echo = await runtime.deploy(_Echo())
        total_handle = await echo.total([1, 2, 3])
        assert await total_handle == 6


@pytest.mark.asyncio
async def test_default_arg_omitted(runtime: Runtime) -> None:
    """A call omitting a defaulted param still works."""
    async with _pipeline_method_scope():
        echo = await runtime.deploy(_Echo())
        greeting_handle = await echo.greet("Max")
        assert await greeting_handle == "Hi Max."


@pytest.mark.asyncio
async def test_streaming_call(runtime: Runtime) -> None:
    """An async-generator worker method streams items inline."""
    async with _pipeline_method_scope():
        echo = await runtime.deploy(_Echo())
        items = [item async for item in await echo.count(5)]
        assert items == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_textgen_pipeline(runtime: Runtime) -> None:
    """Dummy text-gen pipeline runs end-to-end, exercising p2p forwarding."""
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    req = GenAIRequest(
        messages=[ChatMessage.text("user", "hello, ")],
        text=TextGenOptions(num_tokens=5),
    )
    # ``generate`` is decorated with ``@pipeline_method`` so it owns
    # its own scope; the test doesn't need one here.
    tokens = [token async for token in pipeline.generate(req)]

    assert len(tokens) == 5
    assert all(token == GenAITextChunk(text="A") for token in tokens)


@pytest.mark.asyncio
async def test_textgen_different_lengths(runtime: Runtime) -> None:
    """Repeated generations with different lengths reuse the deployed workers."""
    pipeline = await build_dummy_textgen_pipeline()
    await pipeline.deploy(runtime)

    for num_tokens in [1, 3, 10]:
        request = GenAIRequest(
            messages=[ChatMessage.text("user", "test")],
            text=TextGenOptions(num_tokens=num_tokens),
        )
        tokens = [token async for token in pipeline.generate(request)]
        assert len(tokens) == num_tokens


# ---------------------------------------------------------------------------
# Cross-runtime data forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grpc_cross_runtime_result_forwarding() -> None:
    """Two GrpcRuntimeClient instances on separate unix sockets communicate peer-to-peer.

    Each server is a standalone subprocess; the clients are the bare
    :py:class:`GrpcRuntimeClient` (no :py:class:`SubprocGrpcRuntimeClient`
    wrapper). A ``Result`` produced on server A is forwarded as an arg to a
    worker on server B; server B dials server A's socket directly to fetch
    the value via :py:func:`dial` (an unentered ``GrpcRuntimeClient``).
    """
    sock_a = f"/tmp/max-test-{uuid.uuid4().hex[:8]}.sock"
    sock_b = f"/tmp/max-test-{uuid.uuid4().hex[:8]}.sock"
    bind_a, target_a = f"unix://{sock_a}", f"unix:{sock_a}"
    bind_b, target_b = f"unix://{sock_b}", f"unix:{sock_b}"

    try:
        async with (
            subprocess_manager("gRPC server A") as proc_a,
            subprocess_manager("gRPC server B") as proc_b,
        ):
            proc_a.start(grpc_server.serve, bind_a)
            proc_b.start(grpc_server.serve, bind_b)

            async with (
                GrpcRuntimeClient(target_a) as rt_a,
                GrpcRuntimeClient(target_b) as rt_b,
                _pipeline_method_scope(),
            ):
                worker_a = await rt_a.deploy(_Echo())
                worker_b = await rt_b.deploy(_Echo())

                result_on_a = await worker_a.add(10, 5)

                # Server B's result store is empty — it cannot fetch a result
                # that lives on server A.
                with pytest.raises(RuntimeError):
                    await rt_b.get_result(result_on_a.result_id)

                # Pass the Result from server A as an arg to a worker on server B.
                # Server B dials server A's unix socket to resolve it.
                result_on_b = await worker_b.add(result_on_a, 3)
                assert await result_on_b == 18
    finally:
        for path in (sock_a, sock_b):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)


@pytest.mark.asyncio
@pytest.mark.skipif(_SKIP_HTTP, reason=_SKIP_HTTP_REASON)
async def test_http_cross_runtime_result_forwarding() -> None:
    """Two HttpRuntimeProxy instances on separate TCP ports communicate peer-to-peer.

    Each server is a standalone subprocess bound to a free loopback port;
    the clients are the bare :py:class:`HttpRuntimeProxy` (no
    :py:class:`SubprocHttpRuntime` wrapper). A ``Result`` produced on server A
    is forwarded as an arg to a worker on server B; server B dials server A's
    port to fetch the value (``SubprocHttpRuntime.__reduce__`` serializes to
    just the address, so the ``Result`` is a plain URL + result_id pair).
    """

    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    addr_a = f"http://127.0.0.1:{_free_port()}"
    addr_b = f"http://127.0.0.1:{_free_port()}"

    async with (
        subprocess_manager("HTTP server A") as proc_a,
        subprocess_manager("HTTP server B") as proc_b,
    ):
        proc_a.start(http_server.serve, addr_a)
        proc_b.start(http_server.serve, addr_b)

        # HttpRuntimeProxy.__aenter__ opens the session but does not wait for
        # the server — poll /alive manually before deploying workers.
        async with (
            HttpRuntimeProxy(addr_a) as rt_a,
            HttpRuntimeProxy(addr_b) as rt_b,
            _pipeline_method_scope(),
        ):
            async with rt_a.session() as s:
                await _http_wait_until_alive(s, rt_a._base_url)
            async with rt_b.session() as s:
                await _http_wait_until_alive(s, rt_b._base_url)

            worker_a = await rt_a.deploy(_Echo())
            worker_b = await rt_b.deploy(_Echo())

            result_on_a = await worker_a.add(10, 5)

            # Server B's result store is empty — it cannot fetch a result
            # that lives on server A.
            with pytest.raises(KeyError):
                await rt_b.get_result(result_on_a.result_id)

            # Pass the Result from server A as an arg to a worker on server B.
            # Server B dials server A's TCP port to resolve it.
            result_on_b = await worker_b.add(result_on_a, 3)
            assert await result_on_b == 18


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_exception(runtime: Runtime) -> None:
    """An exception raised inside a worker method propagates to the caller."""
    async with _pipeline_method_scope():
        worker = await runtime.deploy(_Failing())
        handle = await worker.raise_value_error("kaboom")
        # The HTTP protocol can raise specific errors since it is mostly for
        # Python testing, but the GrpcRuntimeClient raises a generic
        # RuntimeError because it has to serialize errors from
        # workers implemented in other languages.
        with pytest.raises((ValueError, RuntimeError), match="kaboom"):
            await handle


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_cls",
    [
        pytest.param(
            SubprocHttpRuntime,
            marks=pytest.mark.skipif(_SKIP_HTTP, reason=_SKIP_HTTP_REASON),
        ),
        SubprocGrpcRuntimeClient,
    ],
    ids=lambda c: c.__name__,
)
async def test_worker_sys_exit(runtime_cls: type) -> None:
    """A subprocess worker that calls ``sys.exit(1)`` surfaces an error.

    The error surfaces either on the client's ``await handle`` or as
    ``SubprocessExit`` from ``subprocess_manager``, since both are notified
    when the child process dies.

    The client is notified since the runtime drops the (gRPC) connection, and
    the ``subprocess_manager`` is notified since it watches the child's
    process' termination.

    So wrap the entire pipeline with `pytest.raises`, and check for a successful
    deploy so a deployment failure stays a real error.
    """
    deployed = False
    with pytest.raises(Exception):
        async with runtime_cls() as rt, _pipeline_method_scope():
            worker = await rt.deploy(_Failing())
            deployed = True
            handle = await worker.exit_hard()
            await handle
    assert deployed, "worker deploy failed before exit_hard could run"


@pytest.mark.asyncio
async def test_pipeline_exit_cancels_remote_work(runtime: Runtime) -> None:
    """Exiting the pipeline scope cancels in-flight work and cleans up results.

    Schedules a long-running worker call, then raises to force the
    pipeline scope closed. After the scope exits, the result handle
    should be invalidated -- fetching it must fail, proving the runtime
    released the in-flight task and its result buffer.

    """
    handle = None

    with pytest.raises(RuntimeError, match="orchestrator error"):
        async with _pipeline_method_scope():
            worker = await runtime.deploy(_Failing())
            handle = await worker.slow_ok()
            # Simulate the pipeline orchestrator failing somehow
            raise RuntimeError("orchestrator error")

    assert handle is not None
    # Give the server time to detect the client disconnect
    await asyncio.sleep(1)
    # The remote result should be deleted now
    # The HTTP protocol can raise specific errors since it is mostly for
    # Python testing, but the GrpcRuntimeClient raises a generic
    # RuntimeError because it has to serialize errors from
    # workers implemented in other languages.
    with pytest.raises((KeyError, RuntimeError), match="Unknown result id"):
        await handle
