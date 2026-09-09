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

"""Region-scoped solver selection via ``@mode(...)`` decorator or ``with mode(...):`` block."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

from max.experimental.sharding.picker import GreedyReshard, Solver
from max.experimental.sharding.placements import ShardingError

__all__ = [
    "ShardingError",
    "current_solver",
    "isolated_solver",
    "mode",
]


#: The active per-op solver for the current region. ``None`` means use
#: :class:`GreedyReshard` with default costs.
_CURRENT_SOLVER: contextvars.ContextVar[Solver | None] = contextvars.ContextVar(
    "max_current_solver", default=None
)


def current_solver() -> Solver:
    """Returns the active solver, falling back to :class:`GreedyReshard`."""
    s = _CURRENT_SOLVER.get()
    return s if s is not None else GreedyReshard()


@contextmanager
def mode(solver: Solver) -> Iterator[Solver]:
    """Binds ``solver`` for the duration of a ``with`` block or function call.

    Usable both as a context manager ``with mode(S):`` and as a
    decorator ``@mode(S)`` -- :func:`contextlib.contextmanager` returns
    a :class:`ContextDecorator` that supports both. ``solver`` must be
    a :data:`Solver` instance (one of :class:`GreedyReshard`,
    :class:`NoReshard`, :class:`PartialsOnly`, or any callable matching
    the :data:`Solver` protocol).
    """
    token = _CURRENT_SOLVER.set(solver)
    try:
        yield solver
    finally:
        _CURRENT_SOLVER.reset(token)


@contextmanager
def isolated_solver() -> Iterator[None]:
    """Resets the current solver for the duration of the block."""
    token = _CURRENT_SOLVER.set(None)
    try:
        yield
    finally:
        _CURRENT_SOLVER.reset(token)
