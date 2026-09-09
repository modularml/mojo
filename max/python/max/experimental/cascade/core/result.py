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
"""``Result`` and ``ResultIter`` handles returned by proxy worker methods.

A handle carries a ``result_id`` plus the :py:class:`Runtime` that owns
the binding. The wire boundary keeps these handles uniform across
transports by relying on the runtime being picklable -- the local
in-process runtime pickles by reference (within a process), and remote
runtime proxies (e.g. :py:class:`HttpRuntimeProxy`) implement
``__getstate__`` / ``__setstate__`` to carry only their address.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pydantic import TypeAdapter

if TYPE_CHECKING:
    from max.experimental.cascade.core.interfaces import Runtime

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Result(Generic[T]):
    """Awaitable handle to a single worker-method result."""

    result_id: str
    runtime: Runtime
    # Expected type of the resolved value; local only, never sent on the wire.
    type_hint: object | None = None

    def _decode(self, value: object) -> T:
        # We can either get a fully formed value (i.e. from LocalRuntime),
        # a primitive type, or a CascadeValue, which is a Python view of
        # a partially decoded JSON string. For a CascadeValue, we want to
        # resolve it into the final type via `type_hint`.

        # Here, we handle primitive types.
        if self.type_hint is None or not isinstance(value, (dict, list)):
            return cast(T, value)
        # TypeAdapter should handle lists, dicts, and custom types, whether
        # fully resolved or partially resolved as a CascadeValue.
        return cast(T, TypeAdapter(self.type_hint).validate_python(value))

    def __await__(self) -> Generator[Any, None, T]:
        return self._resolve().__await__()

    async def _resolve(self) -> T:
        return self._decode(await self.runtime.get_result(self.result_id))


@dataclass(frozen=True, slots=True)
class ResultIter(Generic[T]):
    """Async-iterable handle to a streamed worker-method result.

    Streaming counterpart of :py:class:`Result`; same picklability model.
    """

    result_id: str
    runtime: Runtime
    # Expected element type; local only, never sent on the wire.
    type_hint: object | None = None

    def _decode(self, item: object) -> T:
        # See :py:meth:`Result._decode`: only raw JSON composites get rebuilt.
        if self.type_hint is None or not isinstance(item, (dict, list)):
            return cast(T, item)
        return cast(T, TypeAdapter(self.type_hint).validate_python(item))

    def __aiter__(self) -> AsyncIterator[T]:
        return self._resolve()

    async def _resolve(self) -> AsyncIterator[T]:
        async for item in self.runtime.stream_result(self.result_id):
            yield self._decode(item)
