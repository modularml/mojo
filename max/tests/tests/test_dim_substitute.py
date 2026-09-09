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
"""Tests for Dim.substitute."""

import pytest
from max.graph import AlgebraicDim, Dim
from max.graph.dim import StaticDim


def test_substitute_static_dim_is_identity() -> None:
    assert Dim(4).substitute({"a": 7}) == Dim(4)


def test_substitute_symbolic_hit_and_miss() -> None:
    assert Dim("a").substitute({"a": 3}) == Dim(3)
    assert Dim("a").substitute({"b": 3}) == Dim("a")


def test_substitute_algebraic_folds_via_kgen() -> None:
    d = (Dim("a") + 1) * 2
    out = d.substitute({"a": 3})
    assert isinstance(out, StaticDim)
    assert out == Dim(8)


def test_substitute_renames_symbolic() -> None:
    out = (Dim("a") // 2).substitute({"a": Dim("batch")})
    assert isinstance(out, AlgebraicDim)
    assert str(out) == "batch // 2"


def test_substitute_div_matches_kgen_semantics() -> None:
    # POC.div truncates; graph dims are non-negative so this equals floor.
    assert (Dim("a") // 2).substitute({"a": 7}) == Dim(3)


def test_substitute_partial_mapping_nested() -> None:
    assert (Dim("a") + Dim("b")).substitute({"a": 5}) == Dim("b") + 5


def test_substitute_is_simultaneous_not_sequential() -> None:
    # Replacements are not themselves re-substituted.
    assert (Dim("a") - Dim("b")).substitute({"a": "b", "b": "a"}) == Dim(
        "b"
    ) - Dim("a")


def test_substitute_accepts_algebraic_replacement() -> None:
    assert (Dim("a") + 1).substitute({"a": Dim("b") * 2}) == Dim("b") * 2 + 1


def test_substitute_zero_divisor_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        (Dim("a") // Dim("b")).substitute({"b": 0})
    with pytest.raises(ZeroDivisionError):
        (Dim(8) // Dim("b")).substitute({"b": 0})


def test_substitute_zero_replacement_outside_division_is_fine() -> None:
    assert (Dim("a") + Dim("b")).substitute({"b": 0}) == Dim("a")
    assert (Dim("a") * Dim("b")).substitute({"b": 0}) == Dim(0)
