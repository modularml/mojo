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
"""Compatibility shim for ``asyncio.TaskGroup`` plus :py:class:`CancelGroup`.

``asyncio.TaskGroup`` was added in Python 3.11. On older interpreters we
fall back to the ``taskgroup`` PyPI backport. Importers should always
use::

    from max.support.taskgroups import TaskGroup

so the version check and dependency live in a single place.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup  # builtin in 3.11+
    from taskgroup import TaskGroup
else:
    from asyncio import TaskGroup


class CancelGroup(TaskGroup):
    """:py:class:`TaskGroup` for *supervisor* scopes.

    Differs from :py:class:`asyncio.TaskGroup` in two ways:

    1. **Cancel-on-clean-exit.** When the ``async with`` body exits
       normally, background tasks are cancelled rather than awaited.
       Use this when child tasks are long-running watchers/monitors
       that exist only to support the body.
    2. **Single-exception unwrap.** :py:class:`asyncio.TaskGroup`
       always raises :py:class:`BaseExceptionGroup`; for supervisor
       scopes the typical outcome is "one logical thing failed and we
       want to surface that exception directly," so this class unwraps
       single non-cancellation exceptions and surfaces them as
       themselves. Callers can write ``except SubprocessExit:`` rather
       than ``except* SubprocessExit:``.

       More precisely, after :py:class:`asyncio.TaskGroup` collects its
       exceptions, this class:

       * filters out :py:class:`asyncio.CancelledError` (which are
         expected: the body either was cancelled externally or our
         cancel-on-clean-exit fired);
       * if exactly one non-cancellation exception remains, raises it
         directly (preserving its ``__cause__``);
       * if only cancellations remain, raises a bare
         :py:class:`asyncio.CancelledError`;
       * otherwise (multiple concurrent failures) re-raises the
         original :py:class:`BaseExceptionGroup` so callers can use
         ``except*`` to handle each.
    """

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool | None:
        # Cancel all background tasks on clean scope-exit so the group
        # doesn't wait forever for watchers that have no terminating
        # condition of their own.
        # Funny type-ignores, because mypy is very confused by
        # our version-dependent TaskGroup import.
        for task in list(self._tasks):  # type: ignore[attr-defined, unused-ignore]
            task.cancel()
        try:
            return await super().__aexit__(exc_type, exc, tb)
        except BaseExceptionGroup as eg:
            non_cancel = [
                e
                for e in eg.exceptions
                if not isinstance(e, asyncio.CancelledError)
            ]
            if len(non_cancel) == 1:
                # Preserve the cause chain (e.g. remote subprocess
                # tracebacks chained via `raise ... from cause`).
                raise non_cancel[0] from non_cancel[0].__cause__
            if not non_cancel:
                # Only cancellations -- surface a bare CancelledError
                # so callers can write `except asyncio.CancelledError:`.
                raise asyncio.CancelledError() from None
            # Multiple concurrent failures: keep the group so callers
            # can disambiguate with `except*`.
            raise


__all__ = ["BaseExceptionGroup", "CancelGroup", "TaskGroup"]
