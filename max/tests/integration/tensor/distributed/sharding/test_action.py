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
"""Tests for :mod:`max.experimental.sharding.action`."""

from __future__ import annotations

import pytest
from max.experimental.sharding import PerShard


class TestPerShard:
    def test_returns_rank_indexed_value(self) -> None:
        p = PerShard([10, 20, 30])
        assert [p[i] for i in range(len(p))] == [10, 20, 30]

    def test_len_matches_input(self) -> None:
        assert len(PerShard([1, 2, 3, 4])) == 4

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(IndexError):
            _ = PerShard([1, 2])[5]

    def test_equality_and_hash(self) -> None:
        assert PerShard([1, 2, 3]) == PerShard([1, 2, 3])
        assert hash(PerShard([1, 2, 3])) == hash(PerShard([1, 2, 3]))
        assert PerShard([1, 2, 3]) != PerShard([1, 2, 4])

    def test_isinstance_check(self) -> None:
        assert isinstance(PerShard([1, 2]), PerShard)
        assert not isinstance(5, PerShard)
        assert not isinstance((1, 2), PerShard)
