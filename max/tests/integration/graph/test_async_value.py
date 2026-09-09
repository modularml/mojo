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
"""Integration tests for ``await async_value``.

The asyncio integration (``__await__``) is monkeypatched onto
``max._core.mlrt.AsyncValue`` at ``max.engine`` import time (see
``max.engine.mlrt``). This file exercises that bridge from Python end to
end; primitive-level tests for the C++ bindings live under
``MLRT/unittests/Nanobind``.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from max._core.mlrt import AsyncValue
from max.driver import CPU
from max.engine import InferenceSession


# Constructing an ``InferenceSession`` initializes the global MLRT
# ``CPUDevice`` that the ``AsyncValue()`` constructor needs to allocate
# storage. Importing ``max.engine`` also installs the ``__await__``
# monkeypatch as a side effect.
@pytest.fixture(scope="module", autouse=True)
def _cpu_device() -> None:
    InferenceSession(devices=[CPU()])


async def test_await_ready_value_fast_path() -> None:
    """Awaiting an already-done AsyncValue returns the value without
    ever touching the event loop."""
    av: AsyncValue[object] = AsyncValue()
    av.set_result("ready")
    assert await av == "ready"


async def test_await_ready_error_fast_path() -> None:
    av: AsyncValue[object] = AsyncValue()
    av.set_exception(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        await av


async def test_await_pending_resolves_later() -> None:
    """Awaiting an unresolved AsyncValue suspends; when another task
    resolves it, the awaiter resumes."""

    async def producer(av: AsyncValue[object]) -> None:
        await asyncio.sleep(0)  # ensure consumer suspends first
        av.set_result(42)

    async def consumer(av: AsyncValue[object]) -> object:
        return await av

    av: AsyncValue[object] = AsyncValue()
    consumer_task = asyncio.create_task(consumer(av))
    await producer(av)
    assert await consumer_task == 42


async def test_await_pending_propagates_exception() -> None:
    async def producer(av: AsyncValue[object]) -> None:
        await asyncio.sleep(0)
        av.set_exception(KeyError("missing"))

    async def consumer(av: AsyncValue[object]) -> None:
        await av

    av: AsyncValue[object] = AsyncValue()
    consumer_task = asyncio.create_task(consumer(av))
    await producer(av)
    with pytest.raises(KeyError, match="missing"):
        await consumer_task


def test_done_callback_runs_on_resolution() -> None:
    """The callback runs with the resolved value, without an event loop."""
    av: AsyncValue[object] = AsyncValue()
    seen: list[object] = []
    fired = threading.Event()

    def _cb(resolved: AsyncValue[object]) -> None:
        seen.append(resolved.result())
        fired.set()

    av.add_done_callback(_cb)
    av.set_result(42)
    assert fired.wait(timeout=30)
    assert seen == [42]


def test_done_callback_on_already_resolved_value() -> None:
    av: AsyncValue[object] = AsyncValue()
    av.set_result("ready")
    fired = threading.Event()
    av.add_done_callback(lambda _: fired.set())
    assert fired.wait(timeout=30)


def test_done_callback_sees_exception_state() -> None:
    av: AsyncValue[object] = AsyncValue()
    errors: list[BaseException | None] = []
    fired = threading.Event()

    def _cb(resolved: AsyncValue[object]) -> None:
        errors.append(resolved.exception())
        fired.set()

    av.add_done_callback(_cb)
    av.set_exception(ValueError("boom"))
    assert fired.wait(timeout=30)
    (error,) = errors
    assert isinstance(error, ValueError)


def test_and_then_chains_result() -> None:
    """The returned AsyncValue resolves to the callback's return value."""
    av: AsyncValue[object] = AsyncValue()
    chained = av.and_then(lambda resolved: resolved.result() + 1)
    av.set_result(41)
    chained.wait()
    assert chained.result() == 42


def test_and_then_owns_callback_exception() -> None:
    """An exception raised by the callback resolves the returned value."""
    av: AsyncValue[object] = AsyncValue()

    def _boom(_: AsyncValue[object]) -> object:
        raise ValueError("chained boom")

    chained = av.and_then(_boom)
    av.set_result("ready")
    chained.wait()
    assert isinstance(chained.exception(), ValueError)
    with pytest.raises(ValueError, match="chained boom"):
        chained.result()
