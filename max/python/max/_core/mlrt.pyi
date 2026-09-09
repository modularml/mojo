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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from typing import Generic, TypeVar

T = TypeVar("T")

class AsyncValue(Generic[T]):
    """
    Reference-counted handle to an asynchronous value.

    Maps to `AsyncRT::AsyncValueRef<T>`, but duck-types like
    `asyncio.Future`: `done()` / `result()` / `exception()`,
    `set_result()` / `set_exception()`, `add_done_callback()`,
    and `await` are all supported.

    `AsyncValue` instances handed in from C++ are read-only from
    Python — `set_result` / `set_exception` are rejected. Use
    the no-arg constructor or `from_future` to get a Python-owned
    `AsyncValue[object]` that the caller resolves.

    Example — read a result a C++ API handed back:

        compiled = session.compile(graph)         # AsyncValue[CompiledModels]
        models = await compiled._compiled         # suspends until ready

    Example — fulfill from Python and await:

        av: AsyncValue[object] = AsyncValue()
        asyncio.create_task(produce(av))          # calls av.set_result(...)
        value = await av

    Example — bridge an `asyncio.Future`:

        fut: asyncio.Future[int] = loop.create_future()
        av = AsyncValue.from_future(fut)          # AsyncValue[object]
        ...
        fut.set_result(42)                        # av resolves shortly after

    Example — fire-and-forget callback:

        av.add_done_callback(lambda x: log(x.result()))

    Note:
        Constructing an unresolved `AsyncValue` requires an MLRT
        `CPUDevice` on the current thread (any `InferenceSession`
        initializes one).
    """

    def __init__(self) -> None:
        """Allocate an unresolved AsyncValue with a Python-object payload."""

    def done(self) -> bool:
        """True if the value (or an error) is available."""

    def is_error(self) -> bool:
        """True if the value is in the error state."""

    def result(self) -> T:
        """
        Return the held value. Raises `asyncio.InvalidStateError` if not yet done; re-raises the stored exception (or a synthesized `RuntimeError` for C++-side errors) if errored. Mirrors `asyncio.Future.result`.
        """

    def wait(self) -> None:
        """Block the calling thread until this AsyncValue is done."""

    def __await__(self) -> Generator[Any, None, T]:
        """Yield control to the event loop until this AsyncValue resolves."""

    def exception(self) -> BaseException | None:
        """
        Return the exception, or `None` if the AsyncValue completed successfully. Raises `asyncio.InvalidStateError` if the AsyncValue is not yet done. Mirrors `asyncio.Future.exception`.
        """

    def set_result(self, arg: object, /) -> None:
        """
        Mark this AsyncValue as done with the given value. Raises `asyncio.InvalidStateError` if already done.
        """

    def set_exception(self, exc: BaseException, /) -> None:
        """
        Mark this AsyncValue as errored with the given exception. Raises `asyncio.InvalidStateError` if already done.
        """

    def and_then(self, fn: object, /) -> AsyncValue:
        """
        Run fn(self) when this AsyncValue resolves (promptly if already resolved) and return a new AsyncValue that resolves to the callback's return value, or to the exception it raises. The callback is invoked with the GIL held on a runtime worker thread (not the registering thread), concurrently with other Python threads.
        """

    def add_done_callback(self, arg: object, /) -> None:
        """
        Run fn(self) when this AsyncValue resolves, like `concurrent.futures.Future.add_done_callback`. The callback is invoked with the GIL held on a runtime worker thread (not the registering thread), concurrently with other Python threads. Exceptions it raises are reported via `sys.unraisablehook`.
        """

    @staticmethod
    def from_future(fut: object) -> AsyncValue[object]:
        """
        Wrap an asyncio.Future in an AsyncValue that resolves when the Future does.
        """
