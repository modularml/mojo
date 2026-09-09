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

"""Tests for MultiKVCacheParams memory estimation functionality."""

from __future__ import annotations

import math

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVConnectorType,
    MHAKVCacheParams,
    MultiKVCacheParams,
    compute_max_seq_len_fitting_in_cache,
    compute_num_device_blocks,
    estimated_memory_size,
)
from max.nn.kv_cache.input_types import (
    KVCacheInputs,
    MultiKVCacheInputs,
)
from max.pipelines.kv_cache.config import KVConnectorConfig


def create_kv_cache_params(
    num_layers: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 128,
    dtype: DType = DType.bfloat16,
) -> KVCacheParams:
    """Helper to create KVCacheParams with common defaults."""
    return MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        devices=[DeviceRef.GPU()],
        page_size=page_size,
    )


def create_leaf_params(
    n_devices: int = 1,
    dp_degree: int = 1,
    n_kv_heads: int = 8,
    num_layers: int = 8,
    head_dim: int = 64,
    page_size: int = 128,
) -> KVCacheParams:
    """Create a KVCacheParams leaf for any (n_devices, dp_degree) combination."""
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        devices=[DeviceRef.GPU()] * n_devices,
        page_size=page_size,
        data_parallel_degree=dp_degree,
    )


# ---------------------------------------------------------------------------
# Deep-tree constants
#
# Leaf configs: (name, num_layers, n_kv_heads, head_dim)
# Tree: root -> (target -> (a, b), draft -> (c -> (d, e -> (f, g), h)))
# ---------------------------------------------------------------------------

_PAGE_SIZE = 128
_LEAF_SPECS = [
    ("a", 4, 4, 64),
    ("b", 4, 8, 64),
    ("d", 2, 4, 64),
    ("f", 2, 4, 64),
    ("g", 2, 4, 32),
    ("h", 2, 4, 64),
]

# 2 (K+V) x num_layers x page_size x n_kv_heads x head_dim x 2 (bf16 bytes)
_DEEP_TREE_BYTES_PER_BLOCK = sum(
    2 * layers * _PAGE_SIZE * heads * dim * 2
    for _, layers, heads, dim in _LEAF_SPECS
)


def _build_deep_tree(
    n_devices: int = 1, dp_degree: int = 1
) -> MultiKVCacheParams:
    """Build the deeply nested tree used across tests.

    root
    ├── target
    │   ├── a  (leaf)
    │   └── b  (leaf)
    └── draft
        └── c
            ├── d  (leaf)
            ├── e
            │   ├── f  (leaf)
            │   └── g  (leaf)
            └── h  (leaf)
    """

    def make(layers: int, heads: int, dim: int) -> KVCacheParams:
        return create_leaf_params(
            n_devices=n_devices,
            dp_degree=dp_degree,
            n_kv_heads=heads,
            num_layers=layers,
            head_dim=dim,
        )

    a = make(4, 4, 64)
    b = make(4, 8, 64)
    target = MultiKVCacheParams.from_params({"a": a, "b": b})

    d = make(2, 4, 64)
    f = make(2, 4, 64)
    g = make(2, 4, 32)
    e = MultiKVCacheParams.from_params({"f": f, "g": g})
    h = make(2, 4, 64)
    c = MultiKVCacheParams.from_params({"d": d, "e": e, "h": h})
    draft = MultiKVCacheParams.from_params({"c": c})

    return MultiKVCacheParams.from_params({"target": target, "draft": draft})


class TestMultiKVCacheParamsValidation:
    """Tests for MultiKVCacheParams validation logic."""

    def test_empty_params_raises_error(self) -> None:
        """MultiKVCacheParams should raise an error if params list is empty."""
        with pytest.raises(ValueError, match="requires at least one param"):
            MultiKVCacheParams.from_params({})

    def test_mismatched_page_size_raises_error(self) -> None:
        """MultiKVCacheParams should raise if page sizes don't match."""
        params1 = create_kv_cache_params(page_size=128)
        params2 = create_kv_cache_params(page_size=256)
        with pytest.raises(ValueError, match="same page size"):
            MultiKVCacheParams.from_params(
                {"cache0": params1, "cache1": params2}
            )

    def test_mismatched_data_parallel_degree_raises_error(self) -> None:
        """MultiKVCacheParams should raise if data parallel degrees don't match."""
        params1 = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=32,
            devices=[DeviceRef.GPU()],
            page_size=128,
            data_parallel_degree=1,
        )
        params2 = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=32,
            devices=[DeviceRef.GPU(), DeviceRef.GPU()],
            page_size=128,
            data_parallel_degree=2,
        )
        with pytest.raises(ValueError, match="same data parallel degree"):
            MultiKVCacheParams.from_params(
                {"cache0": params1, "cache1": params2}
            )


class TestMultiKVCacheParamsBytesPerBlock:
    """Tests for MultiKVCacheParams.bytes_per_block aggregation."""

    def test_bytes_per_block_sums_across_params(self) -> None:
        """bytes_per_block should be the sum across all param sets."""
        params1 = create_kv_cache_params(num_layers=16, n_kv_heads=8)
        params2 = create_kv_cache_params(num_layers=16, n_kv_heads=4)

        multi_params = MultiKVCacheParams.from_params(
            {"cache0": params1, "cache1": params2}
        )

        expected = params1.bytes_per_block + params2.bytes_per_block
        assert multi_params.bytes_per_block == expected

    def test_single_param_bytes_per_block_unchanged(self) -> None:
        """With a single param, bytes_per_block should match that param."""
        params = create_kv_cache_params()
        multi_params = MultiKVCacheParams.from_params({"cache0": params})

        assert multi_params.bytes_per_block == params.bytes_per_block


class TestMultiKVCacheParamsMemoryEstimation:
    """Tests for memory estimation with MultiKVCacheParams."""

    def test_compute_max_seq_len_accounts_for_all_caches(self) -> None:
        """Max sequence length should account for memory from all caches."""
        # Create two cache configs with different sizes
        params1 = create_kv_cache_params(num_layers=16, n_kv_heads=8)
        params2 = create_kv_cache_params(num_layers=16, n_kv_heads=8)

        # Available memory that can fit some blocks
        available_memory = 100 * 1024 * 1024  # 100 MB

        # Compute max seq len for individual params
        max_seq_len_1 = compute_max_seq_len_fitting_in_cache(
            params1, available_memory
        )
        max_seq_len_2 = compute_max_seq_len_fitting_in_cache(
            params2, available_memory
        )

        # Compute max seq len for multi params
        multi_params = MultiKVCacheParams.from_params(
            {"cache0": params1, "cache1": params2}
        )
        max_seq_len_multi = compute_max_seq_len_fitting_in_cache(
            multi_params, available_memory
        )

        # With two identical caches, multi should fit roughly half the seq len
        # (since bytes_per_block is doubled)
        assert max_seq_len_multi < max_seq_len_1
        assert max_seq_len_multi < max_seq_len_2

        # More precisely: since bytes_per_block doubles, seq len should halve
        # (approximately, due to integer division)
        assert max_seq_len_multi == pytest.approx(
            max_seq_len_1 / 2, rel=0.1
        ) or max_seq_len_multi == pytest.approx(max_seq_len_2 / 2, rel=0.1)

    def test_compute_num_device_blocks_with_multi_params(self) -> None:
        """compute_num_device_blocks should work correctly with MultiKVCacheParams."""
        params1 = create_kv_cache_params(num_layers=16)
        params2 = create_kv_cache_params(num_layers=16)
        multi_params = MultiKVCacheParams.from_params(
            {"cache0": params1, "cache1": params2}
        )

        available_memory = 100 * 1024 * 1024  # 100 MB

        # With multi params, we should get half the blocks (double bytes_per_block)
        blocks_single = compute_num_device_blocks(
            params1,
            available_cache_memory=available_memory,
            max_batch_size=None,
            max_seq_len=None,
        )
        blocks_multi = compute_num_device_blocks(
            multi_params,
            available_cache_memory=available_memory,
            max_batch_size=None,
            max_seq_len=None,
        )

        assert blocks_multi == blocks_single // 2

    def test_estimated_memory_size_with_multi_params(self) -> None:
        """estimated_memory_size should work correctly with MultiKVCacheParams."""
        params1 = create_kv_cache_params(num_layers=16)
        params2 = create_kv_cache_params(num_layers=16)
        multi_params = MultiKVCacheParams.from_params(
            {"cache0": params1, "cache1": params2}
        )

        available_memory = 100 * 1024 * 1024  # 100 MB
        max_batch_size = 4
        max_seq_len = 1024

        # Estimate memory for multi params
        mem_estimate = estimated_memory_size(
            multi_params,
            available_cache_memory=available_memory,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # Memory estimate should be positive and reasonable
        assert mem_estimate > 0
        assert mem_estimate <= available_memory

    def test_multi_params_with_different_head_dims(self) -> None:
        """Test MultiKVCacheParams with different head dimensions."""
        # Simulate a model with two different cache configs (e.g., MLA + standard)
        params1 = create_kv_cache_params(
            num_layers=32, n_kv_heads=8, head_dim=128
        )
        params2 = create_kv_cache_params(
            num_layers=32, n_kv_heads=8, head_dim=64
        )

        multi_params = MultiKVCacheParams.from_params(
            {"cache0": params1, "cache1": params2}
        )

        # bytes_per_block should reflect the sum
        assert (
            multi_params.bytes_per_block
            == params1.bytes_per_block + params2.bytes_per_block
        )

        # Memory estimation should still work
        available_memory = 100 * 1024 * 1024
        max_seq_len = compute_max_seq_len_fitting_in_cache(
            multi_params, available_memory
        )
        assert max_seq_len > 0


class TestMultiKVCacheParamsProperties:
    """Tests for MultiKVCacheParams property accessors."""

    def test_properties_from_first_param(self) -> None:
        """Properties like page_size should come from the first param."""
        params1 = create_kv_cache_params(num_layers=16, page_size=128)
        params2 = create_kv_cache_params(num_layers=32, page_size=128)

        multi_params = MultiKVCacheParams.from_params(
            {"cache0": params1, "cache1": params2}
        )

        assert multi_params.page_size == 128
        assert multi_params.data_parallel_degree == 1
        assert multi_params.n_devices == 1

    def test_frozen_dataclass(self) -> None:
        """MultiKVCacheParams should be frozen (immutable)."""
        params = create_kv_cache_params()
        multi_params = MultiKVCacheParams.from_params({"cache0": params})

        with pytest.raises(AttributeError):
            multi_params.children = {}  # type: ignore[misc]  # ty:ignore[invalid-assignment]


class TestDeepNestedKVCacheTree:
    """Tests for a deeply nested MultiKVCacheParams tree.

    Tree shape: root -> (target -> (a, b), draft -> (c -> (d, e -> (f, g), h)))
    This exercises the recursive structure beyond the two-level shallow trees
    used in the other test classes.
    """

    def test_tree_construction_succeeds(self) -> None:
        """Building the nested tree should not raise."""
        root = _build_deep_tree()
        assert isinstance(root, MultiKVCacheParams)
        assert set(root.children.keys()) == {"target", "draft"}

    def test_bytes_per_block_is_sum_of_all_leaves(self) -> None:
        """bytes_per_block must equal the sum of 2*layers*page*heads*dim*bf16_bytes
        over every leaf — verified against the independently computed constant."""
        assert _build_deep_tree().bytes_per_block == _DEEP_TREE_BYTES_PER_BLOCK

    def test_target_subtree_bytes_per_block(self) -> None:
        """The 'target' subtree's bytes_per_block sums only its own leaves."""
        a = create_kv_cache_params(num_layers=4, n_kv_heads=4, head_dim=64)
        b = create_kv_cache_params(num_layers=4, n_kv_heads=8, head_dim=64)
        target = MultiKVCacheParams.from_params({"a": a, "b": b})
        assert target.bytes_per_block == a.bytes_per_block + b.bytes_per_block

    def test_get_symbolic_inputs_returns_matching_tree_shape(self) -> None:
        """get_symbolic_inputs should mirror the param tree structure."""
        root = _build_deep_tree()
        symbolic = root.get_symbolic_inputs()

        assert isinstance(symbolic, MultiKVCacheInputs)
        assert set(symbolic.children.keys()) == {"target", "draft"}

        target_sym = symbolic.children["target"]
        assert isinstance(target_sym, MultiKVCacheInputs)
        assert set(target_sym.children.keys()) == {"a", "b"}
        assert isinstance(target_sym.children["a"], KVCacheInputs)
        assert isinstance(target_sym.children["b"], KVCacheInputs)

        draft_sym = symbolic.children["draft"]
        assert isinstance(draft_sym, MultiKVCacheInputs)
        assert set(draft_sym.children.keys()) == {"c"}

        c_sym = draft_sym.children["c"]
        assert isinstance(c_sym, MultiKVCacheInputs)
        assert set(c_sym.children.keys()) == {"d", "e", "h"}
        assert isinstance(c_sym.children["d"], KVCacheInputs)
        assert isinstance(c_sym.children["h"], KVCacheInputs)

        e_sym = c_sym.children["e"]
        assert isinstance(e_sym, MultiKVCacheInputs)
        assert set(e_sym.children.keys()) == {"f", "g"}
        assert isinstance(e_sym.children["f"], KVCacheInputs)
        assert isinstance(e_sym.children["g"], KVCacheInputs)

    def test_flatten_unflatten_roundtrip(self) -> None:
        """flatten then unflatten should reconstruct an equivalent tree."""
        root = _build_deep_tree()
        symbolic = root.get_symbolic_inputs()

        flat = symbolic.flatten()
        # Each leaf (single GPU) contributes 6 tensors; 6 leaves → 36 total.
        assert len(flat) == 6 * 6

        it = iter(flat)
        reconstructed = symbolic.unflatten(it)
        assert list(it) == [], "unflatten left unconsumed elements"

        assert isinstance(reconstructed, MultiKVCacheInputs)
        assert set(reconstructed.children.keys()) == {"target", "draft"}

        target_rec = reconstructed.children["target"]
        assert isinstance(target_rec, MultiKVCacheInputs)
        assert set(target_rec.children.keys()) == {"a", "b"}

        draft_rec = reconstructed.children["draft"]
        assert isinstance(draft_rec, MultiKVCacheInputs)
        c_rec = draft_rec.children["c"]
        assert isinstance(c_rec, MultiKVCacheInputs)
        assert set(c_rec.children.keys()) == {"d", "e", "h"}

        e_rec = c_rec.children["e"]
        assert isinstance(e_rec, MultiKVCacheInputs)
        assert set(e_rec.children.keys()) == {"f", "g"}

    def test_unflatten_basic_kv_tree_raises_on_nested_tree(self) -> None:
        """unflatten_basic_kv_tree must raise for trees deeper than height 1."""
        root = _build_deep_tree()
        symbolic = root.get_symbolic_inputs()
        flat = symbolic.flatten()
        it = iter(flat)
        with pytest.raises(
            ValueError, match="Unable to flatten nested KV tree"
        ):
            root.unflatten_basic_kv_tree(it)

    def test_page_size_and_dp_degree_propagated_through_nesting(self) -> None:
        """page_size and data_parallel_degree should be uniform at every level."""
        root = _build_deep_tree()
        assert root.page_size == _PAGE_SIZE
        assert root.data_parallel_degree == 1

        target = root.children["target"]
        assert isinstance(target, MultiKVCacheParams)
        assert target.page_size == _PAGE_SIZE

        draft = root.children["draft"]
        assert isinstance(draft, MultiKVCacheParams)
        c = draft.children["c"]
        assert isinstance(c, MultiKVCacheParams)
        e = c.children["e"]
        assert isinstance(e, MultiKVCacheParams)
        assert e.page_size == _PAGE_SIZE

    def test_memory_estimation_accounts_for_whole_tree(self) -> None:
        """estimated_memory_size should equal the exact formula-derived value.

        Formula (compute_num_device_blocks + estimated_memory_size):
          per_replica_memory = available_memory // dp_degree
          num_blocks = min(per_replica_memory // bytes_per_block,
                           ceil(max_seq_len / page_size) * max_batch_size)
          result = num_blocks * bytes_per_block * dp_degree
        """
        root = _build_deep_tree()
        available_memory = 200 * 1024 * 1024  # 200 MB
        max_batch_size = 4
        max_seq_len = 512

        # Verify bytes_per_block is what we expect before using it.
        assert root.bytes_per_block == _DEEP_TREE_BYTES_PER_BLOCK

        dp = root.data_parallel_degree  # 1
        per_replica = available_memory // dp
        num_blocks = min(
            per_replica // _DEEP_TREE_BYTES_PER_BLOCK,
            math.ceil(max_seq_len / _PAGE_SIZE) * max_batch_size,
        )
        expected = num_blocks * _DEEP_TREE_BYTES_PER_BLOCK * dp

        assert (
            estimated_memory_size(
                root,
                available_cache_memory=available_memory,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
            )
            == expected
        )


# ---------------------------------------------------------------------------
# Parallelism tests: parametrized over TP and DP configurations
#
# (n_devices, dp_degree) → tp_degree = n_devices // dp_degree
#   tp2: (2, 1) → TP=2, heads sharded, 2 entries per leaf KVCacheInputs
#   tp4: (4, 1) → TP=4, heads sharded, 4 entries per leaf KVCacheInputs
#   dp2: (2, 2) → DP=2, heads replicated, 2 entries per leaf KVCacheInputs
#   dp4: (4, 4) → DP=4, heads replicated, 4 entries per leaf KVCacheInputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_devices,dp_degree",
    [(2, 1), (4, 1), (2, 2), (4, 4)],
    ids=["tp2", "tp4", "dp2", "dp4"],
)
class TestDeepTreeParallelism:
    """Tests for the nested KV tree under various TP and DP configurations."""

    def test_leaf_parallelism_properties(
        self, n_devices: int, dp_degree: int
    ) -> None:
        """A leaf should report the correct tp/dp degrees and heads per device."""
        tp = n_devices // dp_degree
        leaf = create_leaf_params(
            n_devices=n_devices, dp_degree=dp_degree, n_kv_heads=8
        )
        assert leaf.tensor_parallel_degree == tp
        assert leaf.data_parallel_degree == dp_degree
        # TP shards heads; DP replicates them.
        assert leaf.n_kv_heads_per_device == 8 // tp

    def test_shallow_tree_parallelism_propagated(
        self, n_devices: int, dp_degree: int
    ) -> None:
        """A two-leaf tree should expose the same parallelism as its leaves."""
        tp = n_devices // dp_degree
        a = create_leaf_params(n_devices=n_devices, dp_degree=dp_degree)
        b = create_leaf_params(n_devices=n_devices, dp_degree=dp_degree)
        root = MultiKVCacheParams.from_params({"a": a, "b": b})
        assert root.tensor_parallel_degree == tp
        assert root.data_parallel_degree == dp_degree
        assert root.n_devices == n_devices

    def test_deep_tree_parallelism_propagated_to_deepest_node(
        self, n_devices: int, dp_degree: int
    ) -> None:
        """Parallelism metadata must reach the deepest subtree (e, depth 4)."""
        tp = n_devices // dp_degree
        root = _build_deep_tree(n_devices=n_devices, dp_degree=dp_degree)
        draft = root.children["draft"]
        assert isinstance(draft, MultiKVCacheParams)
        c = draft.children["c"]
        assert isinstance(c, MultiKVCacheParams)
        e = c.children["e"]
        assert isinstance(e, MultiKVCacheParams)
        assert e.tensor_parallel_degree == tp
        assert e.data_parallel_degree == dp_degree

    def test_symbolic_inputs_shard_count_per_leaf(
        self, n_devices: int, dp_degree: int
    ) -> None:
        """Each leaf's KVCacheInputs.inputs has one entry per device."""
        a = create_leaf_params(n_devices=n_devices, dp_degree=dp_degree)
        b = create_leaf_params(n_devices=n_devices, dp_degree=dp_degree)
        root = MultiKVCacheParams.from_params({"a": a, "b": b})
        symbolic = root.get_symbolic_inputs()
        assert isinstance(symbolic, MultiKVCacheInputs)
        for child in symbolic.children.values():
            assert isinstance(child, KVCacheInputs)
            assert len(child.inputs) == n_devices

    def test_flatten_element_count(
        self, n_devices: int, dp_degree: int
    ) -> None:
        """Flat list length == n_devices x n_leaves x 6 items per device."""
        a = create_leaf_params(n_devices=n_devices, dp_degree=dp_degree)
        b = create_leaf_params(n_devices=n_devices, dp_degree=dp_degree)
        flat = (
            MultiKVCacheParams.from_params({"a": a, "b": b})
            .get_symbolic_inputs()
            .flatten()
        )
        assert len(flat) == n_devices * 2 * 6

    def test_deep_tree_flatten_unflatten_roundtrip(
        self, n_devices: int, dp_degree: int
    ) -> None:
        """Full 6-leaf nested tree should round-trip flatten/unflatten."""
        root = _build_deep_tree(n_devices=n_devices, dp_degree=dp_degree)
        symbolic = root.get_symbolic_inputs()
        flat = symbolic.flatten()
        # 6 leaves x n_devices entries x 6 items per device
        assert len(flat) == 6 * n_devices * 6

        it = iter(flat)
        reconstructed = symbolic.unflatten(it)
        assert list(it) == [], "unflatten left unconsumed elements"
        assert isinstance(reconstructed, MultiKVCacheInputs)
        assert set(reconstructed.children.keys()) == {"target", "draft"}


class TestParallelismValidation:
    """Error cases for mismatched parallelism in MultiKVCacheParams trees."""

    def test_mismatched_n_devices_raises(self) -> None:
        """Mixing TP=2 and TP=4 leaves in the same parent should raise."""
        a = create_leaf_params(n_devices=2)
        b = create_leaf_params(n_devices=4)
        with pytest.raises(ValueError, match="same number of devices"):
            MultiKVCacheParams.from_params({"a": a, "b": b})

    def test_mismatched_dp_degree_raises(self) -> None:
        """Mixing DP=2 and DP=4 leaves in the same parent should raise."""
        a = create_leaf_params(n_devices=2, dp_degree=2)
        b = create_leaf_params(n_devices=4, dp_degree=4)
        with pytest.raises(ValueError, match="same data parallel degree"):
            MultiKVCacheParams.from_params({"a": a, "b": b})

    def test_dp_and_tp_mixed_leaves_raises(self) -> None:
        """A DP leaf and a TP leaf (both 2-device) should fail at the parent."""
        dp_leaf = create_leaf_params(n_devices=2, dp_degree=2)
        tp_leaf = create_leaf_params(n_devices=2, dp_degree=1)
        with pytest.raises(ValueError, match="same data parallel degree"):
            MultiKVCacheParams.from_params({"dp": dp_leaf, "tp": tp_leaf})


class TestPerLayerBuffers:
    """Symbolic-ABI tests for the gated ``per_layer_buffers`` flag.

    A per-layer pool appends one single-layer buffer per layer at the *tail* of
    the flattened inputs, keeping the leading fields byte-identical for every
    non-per-layer cache. These are CPU-only (symbolic types; no device
    allocation or execution).
    """

    def _mha(
        self,
        *,
        num_layers: int,
        per_layer_buffers: bool,
        n_devices: int = 1,
    ) -> KVCacheParams:
        return MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=num_layers,
            devices=[DeviceRef.GPU()] * n_devices,
            page_size=128,
            per_layer_buffers=per_layer_buffers,
        )

    def test_default_off_is_byte_identical(self) -> None:
        """Off by default: no per-layer field and 6 flat items per device."""
        params = self._mha(num_layers=4, per_layer_buffers=False)
        symbolic = params.get_symbolic_inputs()
        assert symbolic.inputs[0].kv_blocks_per_layer is None
        assert len(symbolic.flatten()) == 6

    def test_per_layer_appends_num_layers_at_tail(self) -> None:
        """On: ``num_layers`` single-layer buffers appended after the 6 fields."""
        num_layers = 4
        params = self._mha(num_layers=num_layers, per_layer_buffers=True)
        symbolic = params.get_symbolic_inputs()
        per_device = symbolic.inputs[0]
        assert per_device.kv_blocks_per_layer is not None
        assert len(per_device.kv_blocks_per_layer) == num_layers
        # kv_blocks matches a single-layer buffer (layer dim pinned to 1) so it
        # stays a valid single-buffer alias for kv_blocks_per_layer[0].
        assert (
            per_device.kv_blocks.shape
            == per_device.kv_blocks_per_layer[0].shape
        )
        assert int(per_device.kv_blocks.shape[2]) == 1
        assert len(symbolic.flatten()) == 6 + num_layers

    def test_per_layer_flatten_unflatten_roundtrip(self) -> None:
        """flatten -> unflatten fully consumes the iterator and rebuilds tail."""
        num_layers = 3
        params = self._mha(num_layers=num_layers, per_layer_buffers=True)
        symbolic = params.get_symbolic_inputs()
        it = iter(symbolic.flatten())
        reconstructed = symbolic.unflatten(it)
        assert list(it) == [], "unflatten left unconsumed elements"
        rec = reconstructed.inputs[0]
        assert rec.kv_blocks_per_layer is not None
        assert len(rec.kv_blocks_per_layer) == num_layers

    def test_multi_tree_only_flagged_child_extends(self) -> None:
        """In a tree, only the per-layer child grows; others stay 6 per device."""
        sliding = self._mha(num_layers=2, per_layer_buffers=True)
        full = self._mha(num_layers=5, per_layer_buffers=False)
        root = MultiKVCacheParams.from_params(
            {"sliding": sliding, "full": full}
        )
        # sliding: 6 + 2 (per-layer tail); full: 6; one device each.
        assert len(root.get_symbolic_inputs().flatten()) == (6 + 2) + 6

    def test_allocate_zero_layers_raises(self) -> None:
        """``num_layers == 0`` fails fast with a clear error, not IndexError.

        The guards raise before any device buffer is materialized, so this runs
        without a GPU.
        """
        params = self._mha(num_layers=0, per_layer_buffers=True)
        with pytest.raises(ValueError, match="num_layers >= 1"):
            params.allocate_buffers(total_num_pages=4)

    def test_allocate_with_offload_connector_raises(self) -> None:
        """An off-device connector is rejected: ``all_buffers`` sees only layer 0."""
        params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=4,
            devices=[DeviceRef.GPU()],
            page_size=128,
            per_layer_buffers=True,
            enable_prefix_caching=True,
            kv_connector_config=KVConnectorConfig(
                type=KVConnectorType.tiered,
                host_offload_max_gb=1.0,
            ),
        )
        with pytest.raises(
            NotImplementedError, match="off-device KV connector"
        ):
            params.allocate_buffers(total_num_pages=4)

    def test_allocate_with_data_parallelism_raises(self) -> None:
        """Data parallelism is rejected: cross-replica copy sees only layer 0."""
        params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=8,
            head_dim=128,
            num_layers=4,
            devices=[DeviceRef.GPU()] * 2,
            page_size=128,
            per_layer_buffers=True,
            data_parallel_degree=2,
        )
        with pytest.raises(NotImplementedError, match="data parallelism"):
            params.allocate_buffers(total_num_pages=4)


def _page_dim(kv: KVCacheInputs[TensorType, BufferType]) -> str:
    """First (page-pool) symbolic dim name of a leaf's kv_blocks buffer."""
    return str(kv.inputs[0].kv_blocks.shape[0])


def _lookup_dims(kv: KVCacheInputs[TensorType, BufferType]) -> tuple[str, str]:
    """The (batch_size, max_num_pages) symbolic dim names of the lookup table."""
    shape = kv.inputs[0].lookup_table.shape
    return str(shape[0]), str(shape[1])


class TestPagePoolSymbolicNamespace:
    """A windowed model gives each KV group an independently sized page pool
    (mach sizes a sliding group smaller than the global one), so per-group
    symbolic dims must not collapse onto one shared name: both the pool's
    ``total_num_pages`` and the block table's ``max_num_pages`` (max pages per
    sequence, which the smaller group caps lower) get the group namespace.
    ``batch_size`` stays shared — the batch is identical across groups.
    """

    def test_sibling_groups_get_distinct_page_dims(self) -> None:
        a = create_leaf_params()
        b = create_leaf_params()
        root = MultiKVCacheParams.from_params({"global": a, "local": b})
        symbolic = root.get_symbolic_inputs()
        assert isinstance(symbolic, MultiKVCacheInputs)

        g = symbolic.children["global"]
        loc = symbolic.children["local"]
        assert isinstance(g, KVCacheInputs) and isinstance(loc, KVCacheInputs)
        assert _page_dim(g) == "global_total_num_pages"
        assert _page_dim(loc) == "local_total_num_pages"
        assert _page_dim(g) != _page_dim(loc)

    def test_sibling_groups_get_distinct_max_num_pages(self) -> None:
        """Each group caps pages-per-sequence independently, so their
        ``max_num_pages`` block-table dims must be distinct; ``batch_size``
        stays shared across groups."""
        a = create_leaf_params()
        b = create_leaf_params()
        root = MultiKVCacheParams.from_params({"global": a, "local": b})
        symbolic = root.get_symbolic_inputs()
        assert isinstance(symbolic, MultiKVCacheInputs)
        g = symbolic.children["global"]
        loc = symbolic.children["local"]
        assert isinstance(g, KVCacheInputs) and isinstance(loc, KVCacheInputs)
        g_batch, g_pages = _lookup_dims(g)
        loc_batch, loc_pages = _lookup_dims(loc)
        # batch_size is shared; max_num_pages is namespaced per group.
        assert g_batch == loc_batch == "replica_0_batch_size"
        assert g_pages == "replica_0_global_max_num_pages"
        assert loc_pages == "replica_0_local_max_num_pages"
        assert g_pages != loc_pages

    def test_nested_namespace_composes(self) -> None:
        root = _build_deep_tree()
        symbolic = root.get_symbolic_inputs()
        assert isinstance(symbolic, MultiKVCacheInputs)
        draft = symbolic.children["draft"]
        assert isinstance(draft, MultiKVCacheInputs)
        c = draft.children["c"]
        assert isinstance(c, MultiKVCacheInputs)
        e = c.children["e"]
        assert isinstance(e, MultiKVCacheInputs)
        f = e.children["f"]
        assert isinstance(f, KVCacheInputs)
        assert _page_dim(f) == "draft_c_e_f_total_num_pages"

    def test_single_group_names_unchanged(self) -> None:
        """A plain leaf keeps byte-identical names (empty namespace)."""
        leaf = create_leaf_params()
        symbolic = leaf.get_symbolic_inputs()
        assert isinstance(symbolic, KVCacheInputs)
        assert _page_dim(symbolic) == "total_num_pages"
        assert _lookup_dims(symbolic) == (
            "replica_0_batch_size",
            "replica_0_max_num_pages",
        )

    def test_single_group_quantized_scales_page_dim_unchanged(self) -> None:
        """The kv_scales page dim tracks total_num_pages and stays bare."""
        leaf = MHAKVCacheParams(
            dtype=DType.float8_e4m3fn,
            n_kv_heads=8,
            head_dim=128,
            num_layers=4,
            devices=[DeviceRef.GPU()],
            page_size=128,
            kvcache_quant_config=KVCacheQuantizationConfig(
                scale_dtype=DType.float32
            ),
        )
        symbolic = leaf.get_symbolic_inputs()
        assert isinstance(symbolic, KVCacheInputs)
        scales = symbolic.inputs[0].kv_scales
        assert scales is not None
        assert str(scales.shape[0]) == "total_num_pages"

    def test_quantized_sibling_scales_page_dim_namespaced(self) -> None:
        def quant_leaf() -> KVCacheParams:
            return MHAKVCacheParams(
                dtype=DType.float8_e4m3fn,
                n_kv_heads=8,
                head_dim=128,
                num_layers=4,
                devices=[DeviceRef.GPU()],
                page_size=128,
                kvcache_quant_config=KVCacheQuantizationConfig(
                    scale_dtype=DType.float32
                ),
            )

        root = MultiKVCacheParams.from_params(
            {"global": quant_leaf(), "local": quant_leaf()}
        )
        symbolic = root.get_symbolic_inputs()
        assert isinstance(symbolic, MultiKVCacheInputs)
        g = symbolic.children["global"]
        loc = symbolic.children["local"]
        assert isinstance(g, KVCacheInputs) and isinstance(loc, KVCacheInputs)
        g_scales = g.inputs[0].kv_scales
        loc_scales = loc.inputs[0].kv_scales
        assert g_scales is not None and loc_scales is not None
        assert str(g_scales.shape[0]) == "global_total_num_pages"
        assert str(loc_scales.shape[0]) == "local_total_num_pages"
        assert str(g_scales.shape[0]) == _page_dim(g)
