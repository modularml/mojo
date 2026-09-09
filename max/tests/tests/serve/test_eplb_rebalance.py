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
"""Tests for the NumPy EPLB rebalance algorithm."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from max.pipelines.lib.eplb_rebalance import (
    balanced_packing,
    rebalance_experts,
    replicate_experts,
)
from max.pipelines.lib.eplb_stats import (
    EplbPlacement,
    EplbStatsMetadata,
    EplbStatsSnapshot,
)


def _unique_weights(
    rng: np.random.Generator, shape: tuple[int, int]
) -> np.ndarray:
    """Per-row distinct integer weights so sort tie-breaking is irrelevant."""
    L, N = shape
    out = np.empty((L, N), dtype=np.int64)
    for l in range(L):
        out[l] = rng.permutation(N).astype(np.int64) + 1
    return out


def test_balanced_packing_groups_per_pack_one() -> None:
    weight = np.array([[1.0, 2.0, 3.0, 4.0]])
    pack_index, rank_in_pack = balanced_packing(weight, num_packs=4)
    assert pack_index.tolist() == [[0, 1, 2, 3]]
    assert rank_in_pack.tolist() == [[0, 0, 0, 0]]


def test_replicate_experts_no_redundancy_is_identity() -> None:
    weight = np.array([[10.0, 20.0, 30.0, 40.0]])
    phy2log, rank, logcnt = replicate_experts(weight, num_phy=4)
    np.testing.assert_array_equal(phy2log, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(rank, [[0, 0, 0, 0]])
    np.testing.assert_array_equal(logcnt, [[1, 1, 1, 1]])


def test_replicate_experts_picks_hottest_first() -> None:
    weight = np.array([[10.0, 20.0, 30.0, 40.0]])
    phy2log, _rank, logcnt = replicate_experts(weight, num_phy=6)
    # Hottest expert (3) should be replicated first.
    assert phy2log[0, 4] == 3
    assert logcnt[0, 3] == 2


def test_rebalance_experts_uniform_input_is_well_formed() -> None:
    layers, num_log, num_phy = 4, 8, 8
    weight = np.ones((layers, num_log), dtype=np.int64)
    phy2log, log2phy, logcnt = rebalance_experts(
        weight,
        num_replicas=num_phy,
        num_groups=1,
        num_nodes=1,
        num_gpus=4,
    )
    assert phy2log.shape == (layers, num_phy)
    assert log2phy.shape == (layers, num_log, 1)
    assert logcnt.shape == (layers, num_log)
    np.testing.assert_array_equal(logcnt, np.ones_like(logcnt))
    # phy2log[l] is a permutation of arange(num_log) for each layer.
    for l in range(layers):
        assert sorted(phy2log[l].tolist()) == list(range(num_log))


def test_rebalance_experts_skew_improves_balance() -> None:
    """A heavily skewed histogram, after rebalance, has better
    per-GPU max/min load ratio than naive arange placement."""
    layers, num_log, num_phy, num_gpus = 1, 16, 16, 4
    weight = np.ones((layers, num_log), dtype=np.int64)
    weight[0, 0] = 10_000
    weight[0, 1] = 5_000

    phy2log, _, _ = rebalance_experts(
        weight,
        num_replicas=num_phy,
        num_groups=1,
        num_nodes=1,
        num_gpus=num_gpus,
    )

    # Naive baseline: contiguous arange placement.
    naive = np.broadcast_to(
        np.arange(num_log, dtype=np.int64), (layers, num_phy)
    )
    phy_per_gpu = num_phy // num_gpus

    def gpu_load(plan: np.ndarray) -> np.ndarray:
        per_phy = np.take_along_axis(weight, plan, axis=-1)
        return per_phy.reshape(layers, num_gpus, phy_per_gpu).sum(-1)

    naive_loads = gpu_load(naive)
    rebalanced_loads = gpu_load(phy2log)
    naive_ratio = naive_loads.max() / naive_loads.min()
    rebalanced_ratio = rebalanced_loads.max() / rebalanced_loads.min()
    assert rebalanced_ratio < naive_ratio


def test_rebalance_experts_is_deterministic() -> None:
    rng = np.random.default_rng(seed=0xEB1B)
    weight = rng.integers(0, 1000, size=(4, 32)).astype(np.int64)
    a = rebalance_experts(
        weight.copy(),
        num_replicas=32,
        num_groups=4,
        num_nodes=1,
        num_gpus=4,
    )
    b = rebalance_experts(
        weight.copy(),
        num_replicas=32,
        num_groups=4,
        num_nodes=1,
        num_gpus=4,
    )
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


# -------------------------------------------------------------------- #
# Parity test against the torch reference, when torch is available.
# -------------------------------------------------------------------- #


def _try_load_torch_reference() -> types.ModuleType | None:
    """Import the speed-of-light torch reference if torch is present."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    repo_root = Path(__file__).resolve()
    while repo_root.name != "modular" and repo_root.parent != repo_root:
        repo_root = repo_root.parent
    ref_path = (
        repo_root
        / "utils/benchmarking/speed-of-light/speed_of_light/eplb/rebalance_algo.py"
    )
    if not ref_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "speed_of_light_rebalance_algo", ref_path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


REFERENCE = _try_load_torch_reference()


@pytest.mark.skipif(REFERENCE is None, reason="torch reference unavailable")
@pytest.mark.parametrize(
    "shape,num_groups,num_nodes,num_gpus,num_replicas",
    [
        ((61, 256), 8, 1, 8, 256),
        ((30, 384), 1, 1, 8, 384),
    ],
)
def test_parity_with_torch_reference(
    shape: tuple[int, int],
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    num_replicas: int,
) -> None:
    """Strict array-equality parity on tie-free inputs.
    Tied inputs are tested separately in
    test_load_distribution_parity_under_ties — torch.sort defaults
    to stable=False, so phy2log can differ in tied-cell placement
    while still being a valid balanced layout.
    """
    import torch

    assert REFERENCE is not None

    rng = np.random.default_rng(seed=42)
    weight_np = _unique_weights(rng, shape)
    weight_pt = torch.from_numpy(weight_np.copy())
    np_out = rebalance_experts(
        weight_np.copy(), num_replicas, num_groups, num_nodes, num_gpus
    )
    pt_out = REFERENCE.rebalance_experts(
        weight_pt, num_replicas, num_groups, num_nodes, num_gpus
    )
    for arr_np, t_pt, name in zip(
        np_out, pt_out, ("phy2log", "log2phy", "logcnt"), strict=True
    ):
        np.testing.assert_array_equal(
            arr_np,
            t_pt.numpy(),
            err_msg=f"divergence in {name} for shape={shape}",
        )


@pytest.mark.skipif(REFERENCE is None, reason="torch reference unavailable")
@pytest.mark.parametrize(
    "shape,num_groups,num_nodes,num_gpus,num_replicas",
    [
        ((61, 256), 8, 1, 8, 256),
        ((30, 384), 1, 1, 8, 384),
        ((4, 64), 4, 2, 8, 80),
    ],
)
def test_load_distribution_parity_under_ties(
    shape: tuple[int, int],
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    num_replicas: int,
) -> None:
    """On inputs with ties, phy2log placement may diverge between the
    NumPy port and the torch reference because torch.sort uses a
    non-stable sort by default. The per-GPU load distribution must
    still match exactly, since both impls run the same balanced-packing
    greedy on the same weights."""
    import torch

    assert REFERENCE is not None
    rng = np.random.default_rng(seed=42)
    weight_np = rng.integers(0, 10_000, size=shape).astype(np.int64)
    weight_pt = torch.from_numpy(weight_np.copy())
    np_phy2log, _, np_logcnt = rebalance_experts(
        weight_np.copy(), num_replicas, num_groups, num_nodes, num_gpus
    )
    pt_phy2log_t, _, pt_logcnt_t = REFERENCE.rebalance_experts(
        weight_pt, num_replicas, num_groups, num_nodes, num_gpus
    )
    pt_phy2log = pt_phy2log_t.numpy()
    pt_logcnt = pt_logcnt_t.numpy()
    phy_per_gpu = num_replicas // num_gpus

    def gpu_loads(phy2log: np.ndarray, logcnt: np.ndarray) -> np.ndarray:
        per_rep = np.take_along_axis(weight_np / logcnt, phy2log, axis=-1)
        return per_rep.reshape(shape[0], num_gpus, phy_per_gpu).sum(-1)

    np.testing.assert_allclose(
        gpu_loads(np_phy2log, np_logcnt),
        gpu_loads(pt_phy2log, pt_logcnt),
        rtol=1e-9,
        atol=1e-9,
        err_msg=(
            f"per-GPU load divergence for shape={shape}; the two "
            f"implementations should produce identical balance even "
            f"when phy2log placement diverges on tied weights."
        ),
    )


def test_eplb_placement_identity() -> None:
    p = EplbPlacement.identity(4, 16)
    assert p.phy2log.shape == (4, 16)
    assert p.log2phy.shape == (4, 16, 1)
    assert p.log2phy.dtype == np.int32
    assert p.max_replicas == 1
    np.testing.assert_array_equal(
        p.log2phy[..., 0],
        np.broadcast_to(np.arange(16, dtype=np.int32), (4, 16)),
    )


def test_eplb_placement_from_uniform_snapshot() -> None:
    md = EplbStatsMetadata(
        num_layers=2,
        num_moe_layers=2,
        moe_layer_indices=(0, 1),
        num_logical_experts=16,
        num_experts_per_token=2,
    )
    snap = EplbStatsSnapshot.from_array(md, np.ones((2, 16), dtype=np.int64))
    p = EplbPlacement.from_snapshot(snap, ep_size=4, n_nodes=1, n_groups=1)
    assert p.max_replicas == 1
    for l in range(2):
        np.testing.assert_array_equal(np.sort(p.phy2log[l]), np.arange(16))


def test_eplb_placement_from_skewed_snapshot_logs_improvement(
    caplog: pytest.LogCaptureFixture,
) -> None:
    md = EplbStatsMetadata(
        num_layers=1,
        num_moe_layers=1,
        moe_layer_indices=(0,),
        num_logical_experts=16,
        num_experts_per_token=2,
    )
    h = np.ones((1, 16), dtype=np.int64)
    h[0, 0] = 10_000
    snap = EplbStatsSnapshot.from_array(md, h)
    with caplog.at_level("INFO", logger="max.serve"):
        EplbPlacement.from_snapshot(snap, ep_size=4, n_nodes=1, n_groups=1)
    assert "EPLB: per-GPU load max/min" in caplog.text
