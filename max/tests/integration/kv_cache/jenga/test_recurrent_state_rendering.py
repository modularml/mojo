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
"""Tests for folding a forward's state blocks to the rows a model reads."""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.nn.kv_cache import (
    KVCacheGroupId,
    RecurrentKVLeafRegion,
    RecurrentStateRegion,
)

A, B = "a", "b"

REGIONS = (
    RecurrentStateRegion(
        leaf_id=A, num_layers=3, row_shape=(4,), dtype=DType.bfloat16
    ),
    RecurrentStateRegion(
        leaf_id=B, num_layers=2, row_shape=(2,), dtype=DType.bfloat16
    ),
)
"""Two leaves with *different* layer counts, so a swap cannot pass."""

LEAVES = {
    region.leaf_id: RecurrentKVLeafRegion(
        leaf_id=region.leaf_id,
        group_id=KVCacheGroupId.recurrent(),
        bytes_per_page=region.bytes_per_state,
        region=region,
    )
    for region in REGIONS
}


def plan(runs_in: dict[str, int]) -> dict[str, list[int]]:
    return {leaf_id: [runs_in[leaf_id]] for leaf_id in (A, B)}


def rows(
    *plans: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    """The rows those plans name, staged the way the cache manager stages them."""
    folded: dict[str, np.ndarray] = {}
    for leaf_id, leaf in LEAVES.items():
        into = {
            key: np.zeros(shape, dtype=dtype.to_numpy())
            for key, (shape, dtype) in leaf.staged_input_shapes(
                len(plans), num_blocks=0
            ).items()
        }
        leaf.write_staged_inputs([p[leaf_id] for p in plans], into)
        folded[leaf_id] = into[leaf_id]
    return folded


def test_a_page_folds_to_the_rows_its_layers_occupy() -> None:
    # Page p of a 3-layer leaf is rows 3p..3p+2.
    folded = rows(plan({A: 1, B: 4}))

    assert np.array_equal(folded[A], [[3, 4, 5]])
    assert np.array_equal(folded[B], [[8, 9]])


def test_a_forward_sees_the_rows_it_was_given() -> None:
    folded = rows(plan({A: 1, B: 4}), plan({A: 2, B: 5}))

    assert np.array_equal(folded[A], [[3, 4, 5], [6, 7, 8]])
    assert np.array_equal(folded[B], [[8, 9], [10, 11]])


def test_a_leaf_keeps_its_own_layer_count() -> None:
    folded = rows(plan({A: 0, B: 0}), plan({A: 1, B: 1}))

    assert [folded[leaf_id].shape for leaf_id in (A, B)] == [(2, 3), (2, 2)]


def test_a_state_leaf_reserves_two_blocks_whatever_the_block_count() -> None:
    """The block it runs in and the checkpoint it publishes, at any num_blocks."""
    assert [
        LEAVES[A].blocks_to_reserve(num_blocks) for num_blocks in (1, 7, 4096)
    ] == [
        2,
        2,
        2,
    ]
