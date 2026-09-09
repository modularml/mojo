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
"""Tests for :mod:`max.experimental.sharding.mode`."""

from __future__ import annotations

import pytest
from max.experimental.sharding import (
    GreedyReshard,
    NoReshard,
    ShardingError,
    Solver,
    isolated_solver,
    mode,
)
from max.experimental.sharding.mode import current_solver


class TestShardingError:
    def test_is_runtime_error_subclass(self) -> None:
        assert issubclass(ShardingError, RuntimeError)


class TestModeContextManager:
    def test_sets_and_restores_solver(self) -> None:
        marker = GreedyReshard(on_reshard="warn")
        assert current_solver() is not marker
        with mode(marker):
            assert current_solver() is marker
        assert current_solver() is not marker

    def test_nesting_restores_outer_on_exit(self) -> None:
        outer = GreedyReshard()
        inner = NoReshard()
        with mode(outer):
            with mode(inner):
                assert current_solver() is inner
            assert current_solver() is outer

    def test_exception_inside_block_still_restores(self) -> None:
        outer = GreedyReshard(on_reshard="warn")
        with mode(outer):
            with pytest.raises(RuntimeError):
                with mode(NoReshard()):
                    raise RuntimeError("boom")
            assert current_solver() is outer


class TestModeAsDecorator:
    def test_calls_function_inside_mode_block(self) -> None:
        marker = NoReshard()
        observed: list[Solver] = []

        @mode(marker)
        def f() -> None:
            observed.append(current_solver())

        assert current_solver() is not marker
        f()
        assert observed == [marker]
        assert current_solver() is not marker


class TestIsolatedSolver:
    def test_clears_active_solver_inside_block(self) -> None:
        with mode(NoReshard()):
            with isolated_solver():
                assert isinstance(current_solver(), GreedyReshard)
