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

"""Unit tests for KVTransferEngine's transfer-strategy resolver.

``resolve_transfer_strategy`` is a pure planner: it classifies any ``(dp,tp)``
peer into a per-group :class:`_TransferStrategy` keyed on the TP axis --
``DIRECT`` (matching TP), ``BROADCAST`` (TP change + replicated), or
``GATHER_SCATTER`` (TP change + sharded). Connect is always the full DP
cartesian; the strategy picks the shard pattern within each replica pair
(DIRECT diagonal, BROADCAST full cross-shard mesh); DP is the scheduler's
routing. ``GATHER_SCATTER`` is planned but refused by the transport until
MXSERV-290.
"""

from __future__ import annotations

import pytest
from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
    _build_group_descriptors,
    _TransferStrategy,
    connect_pairing,
    resolve_transfer_strategy,
    transfer_shard_pairing,
)

DIRECT = _TransferStrategy.DIRECT
BROADCAST = _TransferStrategy.BROADCAST
GATHER_SCATTER = _TransferStrategy.GATHER_SCATTER


@pytest.mark.parametrize(
    "local_tp,local_rep,remote_tp,remote_rep,expected",
    [
        # Uniform TP -> DIRECT for every group, replication ignored.
        (4, [True], 4, [True], [DIRECT]),
        (2, [True, False], 2, [True, False], [DIRECT, DIRECT]),
        # TP change + replicated -> BROADCAST. The tp==1 side reports every
        # group False; the OR recovers the truth from the tp>1 side (either
        # direction), and both-tp>1 resolves too.
        (4, [True], 1, [False], [BROADCAST]),
        (1, [False], 8, [True], [BROADCAST]),
        (4, [True], 2, [True], [BROADCAST]),
        # TP change + sharded (False on both) -> GATHER_SCATTER.
        (4, [False], 1, [False], [GATHER_SCATTER]),
        # Mixed groups resolve per group -- no engine-wide collapse.
        (4, [True, False], 1, [False, False], [BROADCAST, GATHER_SCATTER]),
    ],
)
def test_resolve_transfer_strategy(
    local_tp: int,
    local_rep: list[bool],
    remote_tp: int,
    remote_rep: list[bool],
    expected: list[_TransferStrategy],
) -> None:
    assert (
        resolve_transfer_strategy(
            local_tp=local_tp,
            local_replicate=local_rep,
            remote_tp=remote_tp,
            remote_replicate=remote_rep,
        )
        == expected
    )


@pytest.mark.parametrize(
    "strategies,local_dp,local_tp,remote_dp,remote_tp,expected",
    [
        # Each expected row is a connect quad
        # (local_replica, local_shard, remote_replica, remote_shard): the local
        # tensor agent (replica, shard) that loads that remote agent's metadata.
        # Rows always span the full local x remote DP cartesian; the strategy
        # only picks the shard pattern within each replica pair.
        # DIRECT: shard diagonal within each of the 4 replica pairs.
        (
            [DIRECT],
            2,
            2,
            2,
            2,
            [
                (0, 0, 0, 0),
                (0, 1, 0, 1),
                (0, 0, 1, 0),
                (0, 1, 1, 1),
                (1, 0, 0, 0),
                (1, 1, 0, 1),
                (1, 0, 1, 0),
                (1, 1, 1, 1),
            ],
        ),
        # BROADCAST, tp'=1 (dp1/tp2 -> dp2/tp1): each local shard -> the single
        # remote shard, across the full DP cartesian.
        (
            [BROADCAST],
            1,
            2,
            2,
            1,
            [
                (0, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 1, 1, 0),
            ],
        ),
    ],
)
def test_connect_pairing(
    strategies: list[_TransferStrategy],
    local_dp: int,
    local_tp: int,
    remote_dp: int,
    remote_tp: int,
    expected: list[tuple[int, int, int, int]],
) -> None:
    assert (
        connect_pairing(strategies, local_dp, local_tp, remote_dp, remote_tp)
        == expected
    )


def test_connect_pairing_broadcast_cross_shard_mesh() -> None:
    """BROADCAST with remote_tp>1 (dp2/tp4 -> dp4/tp2): each of the 8 replica
    pairs wires the full 4x2 cross-shard mesh -> 64 loads. This equals
    flattening both grids to ``[dp*tp][1]`` (8x8)."""
    pairs = connect_pairing([BROADCAST], 2, 4, 4, 2)
    assert len(pairs) == 64
    # First replica pair (local 0, remote 0): all 4x2 shard combinations.
    assert pairs[:8] == [(0, s, 0, d) for s in range(4) for d in range(2)]


@pytest.mark.parametrize(
    "strategies", [[BROADCAST, GATHER_SCATTER], [GATHER_SCATTER]]
)
def test_connect_pairing_refuses_gather_scatter(
    strategies: list[_TransferStrategy],
) -> None:
    """The transport refuses a reshard plan until MXSERV-290. The resolver
    still produced a valid plan; the boundary is here, per peer."""
    with pytest.raises(NotImplementedError, match="MXSERV-290"):
        connect_pairing(strategies, 2, 4, 8, 1)


@pytest.mark.parametrize(
    "flatten_source,source_tp,dest_tp,expected",
    [
        # Equal TP, no flatten: 1:1 shard pairing.
        (False, 2, 2, [(0, 0), (1, 1)]),
        # Flatten source collapses to shard 0 (dest TP=1).
        (True, 4, 1, [(0, 0)]),
        # Flatten + fan-out: shard 0 to every dest shard.
        (True, 4, 2, [(0, 0), (0, 1)]),
        # Single source shard fans out without flatten.
        (False, 1, 4, [(0, 0), (0, 1), (0, 2), (0, 3)]),
    ],
)
def test_transfer_shard_pairing(
    flatten_source: bool,
    source_tp: int,
    dest_tp: int,
    expected: list[tuple[int, int]],
) -> None:
    assert (
        transfer_shard_pairing(
            flatten_source=flatten_source,
            source_tp=source_tp,
            dest_tp=dest_tp,
        )
        == expected
    )


def test_build_group_descriptors_multi_group_is_per_group_concat() -> None:
    """The multi-group descriptor plan equals the per-group plans concatenated.

    Byte-identity of the group-major ordering: slice 2 may plan each group
    independently (different strategy per group) and must still emit exactly
    these (addr, size, device) descriptors for groups that do not reshard.
    """
    base_addrs = [0x1000, 0x9000]
    bytes_per_group = [800, 16]
    page_idxs = [1, 3, 4]
    device_id = 2

    combined = _build_group_descriptors(
        base_addrs, bytes_per_group, page_idxs, device_id
    )
    per_group = _build_group_descriptors(
        base_addrs[:1], bytes_per_group[:1], page_idxs, device_id
    ) + _build_group_descriptors(
        base_addrs[1:], bytes_per_group[1:], page_idxs, device_id
    )
    assert combined == per_group

    # Golden offsets/sizes: base[g] + idx * bpp[g], size bpp[g].
    assert combined == [
        (0x1000 + 1 * 800, 800, 2),
        (0x1000 + 3 * 800, 800, 2),
        (0x1000 + 4 * 800, 800, 2),
        (0x9000 + 1 * 16, 16, 2),
        (0x9000 + 3 * 16, 16, 2),
        (0x9000 + 4 * 16, 16, 2),
    ]
