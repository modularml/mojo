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
"""Expert-parallel load-balance rearrangement (CPU only).

NumPy port of the DeepSeek EPLB rearrangement algorithm. Used by
max serve --eplb-stats to compute a static log->phy expert
indices once at server startup from a routing-histogram snapshot.

Adapted from https://github.com/deepseek-ai/EPLB (MIT-licensed). The
reference torch implementation lives at
utils/benchmarking/speed-of-light/speed_of_light/eplb/rebalance_algo.py;
that copy is retained for psim/benchmarking and is parity-tested against
this NumPy port in
max/tests/tests/serve/test_eplb_rebalance.py.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("max.serve")


def balanced_packing(
    weight: NDArray[np.float64], num_packs: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Pack n weighted objects into num_packs packs evenly.

    Each pack contains exactly n / num_packs objects, with weights
    balanced as much as possible by greedy assignment.

    Args:
        weight: [X, n] weight per item.
        num_packs: number of packs.

    Returns:
        pack_index: [X, n] int64 — pack assignment per item.
        rank_in_pack: [X, n] int64 — position within the pack.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = np.broadcast_to(
            np.arange(num_groups, dtype=np.int64), weight.shape
        ).copy()
        rank_in_pack = np.zeros_like(weight, dtype=np.int64)
        return pack_index, rank_in_pack

    indices = np.argsort(-weight.astype(np.float64), axis=-1, kind="stable")
    pack_index = np.full_like(weight, fill_value=-1, dtype=np.int64)
    rank_in_pack = np.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (
                    p
                    for p in range(num_packs)
                    if pack_items[p] < groups_per_pack
                ),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += float(weight[i, group])
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: NDArray[np.float64], num_phy: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Replicate num_log logical experts to num_phy physical slots.

    Greedy: at each step we duplicate whichever logical expert currently
    has the highest per-replica load.

    Returns:
        phy2log: [X, num_phy] — logical expert id of each physical slot.
        rank: [X, num_phy] — replica rank within its logical expert.
        logcnt: [X, num_log] — number of replicas per logical expert.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    phy2log = np.broadcast_to(
        np.arange(num_phy, dtype=np.int64), (n, num_phy)
    ).copy()
    rank = np.zeros((n, num_phy), dtype=np.int64)
    logcnt = np.ones((n, num_log), dtype=np.int64)
    arangen = np.arange(n, dtype=np.int64)
    for i in range(num_log, num_phy):
        redundant_indices = np.argmax(weight / logcnt, axis=-1)
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def _inverse(perm: NDArray[np.int64]) -> NDArray[np.int64]:
    """Row-wise inverse permutation. perm shape: [X, n]."""
    inv = np.empty_like(perm)
    np.put_along_axis(
        inv,
        perm,
        np.broadcast_to(np.arange(perm.shape[1], dtype=np.int64), perm.shape),
        axis=1,
    )
    return inv


def rebalance_experts_hierarchical(
    weight: NDArray[np.float64],
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Hierarchical rearrangement: groups -> nodes -> GPUs.

    See the reference implementation in
    utils/benchmarking/speed-of-light/speed_of_light/eplb/rebalance_algo.py
    for the algorithm walkthrough.
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    # Step 1: pack groups to nodes.
    tokens_per_group = weight.reshape(num_layers, num_groups, group_size).sum(
        -1
    )
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes
    )
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack)
            * group_size
        )[..., None]
        + np.arange(group_size, dtype=np.int64)
    ).reshape(num_layers, -1)
    mlog2log = _inverse(log2mlog)

    # Step 2: construct redundant experts within nodes.
    tokens_per_mlog = np.take_along_axis(weight, mlog2log, axis=-1).reshape(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical experts to GPUs within each node.
    tokens_per_phy = np.take_along_axis(
        tokens_per_mlog / mlogcnt, phy2mlog, axis=-1
    )
    pack_index, rank_in_pack = balanced_packing(
        tokens_per_phy, num_gpus // num_nodes
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = _inverse(phy2pphy)

    pphy2mlog = np.take_along_axis(phy2mlog, pphy2phy, axis=-1)
    pphy2mlog = (
        pphy2mlog.reshape(num_layers, num_nodes, -1)
        + np.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            dtype=np.int64,
        ).reshape(1, -1, 1)
    ).reshape(num_layers, -1)
    pphy2log = np.take_along_axis(mlog2log, pphy2mlog, axis=-1)
    pphyrank = np.take_along_axis(phyrank, pphy2phy, axis=-1).reshape(
        num_layers, -1
    )
    logcnt = np.take_along_axis(
        mlogcnt.reshape(num_layers, -1), log2mlog, axis=-1
    )
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: NDArray[np.int64] | NDArray[np.float64],
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Entry point for expert-parallelism load balancing.

    Identical contract to the speed-of-light torch reference: same
    inputs, same outputs, byte-equivalent for any well-shaped input.

    Args:
        weight: [layers, num_logical_experts] int64 or float load
            statistics for each logical expert (typically the
            cumulative routing histogram from
            :class:`max.serve.pipelines.ep_stats.EpStatsSnapshot`).
        num_replicas: total number of physical slots after replication.
            Must be a multiple of ``num_gpus``. For v1 set
            ``num_replicas == num_logical_experts`` (no redundancy).
        num_groups: number of expert groups (DeepSeek-V3=8,
            Kimi-K2.5=1).
        num_nodes: number of server nodes; intra-node network is
            assumed faster.
        num_gpus: number of GPUs; must be a multiple of ``num_nodes``.

    Returns:
        physical_to_logical_map: [layers, num_replicas] — logical
            expert id at each physical slot.
        logical_to_physical_map: ``[layers, num_logical_experts, X]``
            — physical slots holding each logical expert.
        expert_count: ``[layers, num_logical_experts]`` — replica
            count per logical expert.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.astype(np.float64, copy=True)
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    maxlogcnt = int(logcnt.max())
    log2phy = np.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=np.int64,
    )
    np.put_along_axis(
        log2phy.reshape(num_layers, -1),
        phy2log * maxlogcnt + phyrank,
        np.broadcast_to(
            np.arange(num_replicas, dtype=np.int64), (num_layers, num_replicas)
        ),
        axis=-1,
    )
    return phy2log, log2phy, logcnt


__all__ = [
    # "EPLBPlan"
    "balanced_packing",
    # "permute_expert_keys",
    "rebalance_experts",
    "rebalance_experts_hierarchical",
    "replicate_experts",
]
