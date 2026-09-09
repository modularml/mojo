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
"""Tests for Placement types: Replicated, Sharded, Partial, ReduceOp,
and cross-type comparisons."""

from __future__ import annotations

import dataclasses

import pytest
from max.experimental.sharding import (
    Partial,
    ReduceOp,
    Replicated,
    Sharded,
)

# ═════════════════════════════════════════════════════════════════════════
#  Placement types
# ═════════════════════════════════════════════════════════════════════════


class TestReplicated:
    def test_repr(self) -> None:
        assert repr(Replicated()) == "Replicated()"

    def test_equality(self) -> None:
        assert Replicated() == Replicated()

    def test_value_equality(self) -> None:
        assert Replicated() == Replicated()
        assert hash(Replicated()) == hash(Replicated())

    def test_not_equal_to_other_types(self) -> None:
        assert Replicated() != Sharded(axis=0)
        assert Replicated() != Partial()

    def test_hash_consistent(self) -> None:
        assert hash(Replicated()) == hash(Replicated())
        assert {Replicated(), Replicated()} == {Replicated()}


class TestSharded:
    def test_repr(self) -> None:
        assert repr(Sharded(axis=0)) == "Sharded(axis=0)"

    def test_equality(self) -> None:
        assert Sharded(axis=0) == Sharded(axis=0)
        assert Sharded(axis=0) != Sharded(axis=1)

    def test_frozen(self) -> None:
        s = Sharded(axis=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.axis = 1  # type: ignore[misc]

    def test_hash(self) -> None:
        assert {Sharded(0), Sharded(0), Sharded(1)} == {
            Sharded(0),
            Sharded(1),
        }

    def test_is_placement_subclass(self) -> None:
        from max.experimental.sharding import Placement

        assert isinstance(Sharded(0), Placement)


class TestPartial:
    def test_repr(self) -> None:
        assert repr(Partial()) == "Partial(reduce_op='sum')"

    def test_default_reduce_op(self) -> None:
        assert Partial().reduce_op == ReduceOp.SUM

    def test_equality(self) -> None:
        assert Partial() == Partial(reduce_op=ReduceOp.SUM)
        assert Partial(ReduceOp.SUM) != Partial(ReduceOp.AVG)

    def test_value_equality(self) -> None:
        assert Partial() == Partial()
        assert Partial(ReduceOp.AVG) == Partial(ReduceOp.AVG)
        assert hash(Partial()) == hash(Partial())

    def test_reduce_op_readonly(self) -> None:
        p = Partial()
        with pytest.raises(AttributeError):
            p.reduce_op = ReduceOp.SUM  # type: ignore[misc]

    def test_hash(self) -> None:
        assert hash(Partial()) == hash(Partial(ReduceOp.SUM))
        assert hash(Partial(ReduceOp.SUM)) != hash(Partial(ReduceOp.AVG))


class TestReduceOp:
    def test_values(self) -> None:
        assert ReduceOp.SUM.value == "sum"
        assert ReduceOp.AVG.value == "avg"
        assert ReduceOp.MIN.value == "min"
        assert ReduceOp.MAX.value == "max"

    def test_is_string_enum(self) -> None:
        assert isinstance(ReduceOp.SUM, str)


# ═════════════════════════════════════════════════════════════════════════
#  Cross-type placement comparisons
# ═════════════════════════════════════════════════════════════════════════


class TestPlacementCrossType:
    def test_replicated_not_equal_sharded(self) -> None:
        assert Replicated() != Sharded(0)

    def test_replicated_not_equal_partial(self) -> None:
        assert Replicated() != Partial()

    def test_sharded_not_equal_partial(self) -> None:
        assert Sharded(0) != Partial()

    def test_sharded_not_equal_replicated(self) -> None:
        assert Sharded(0) != Replicated()

    def test_partial_not_equal_non_placement(self) -> None:
        assert Partial() != "not a placement"
        assert Replicated() != 42

    def test_all_are_placement_instances(self) -> None:
        from max.experimental.sharding import Placement

        assert isinstance(Replicated(), Placement)
        assert isinstance(Sharded(0), Placement)
        assert isinstance(Partial(), Placement)

    def test_set_of_mixed_placements(self) -> None:
        s = {Replicated(), Sharded(0), Partial(), Replicated(), Sharded(0)}
        assert len(s) == 3
