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
"""Tests for JengaKVCacheManager's runtime input preparation.

``JengaBlockManager`` (see ``test_jenga_block_manager.py``) owns allocation;
this file covers what ``JengaKVCacheManager`` adds on top of it --
``runtime_inputs`` / ``_compute_kv_cache_assignments`` -- since that's where
this PR's two real bugs lived: computing the per-request block count from a
leaf-id dict's *keys* instead of its *values*, and an alloc-time
``num_draft_tokens_per_step`` that silently diverged from ``params``'.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pytest
from max.driver import Buffer, Device, Usage
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsInterface,
    MHAKVCacheParams,
    MultiKVCacheParams,
)
from max.nn.kv_cache.cache_params import (
    KVCacheParamInterface,
    SpeculativeMethod,
)
from max.nn.kv_cache.metrics import KVCacheMetrics
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.paged_kv_cache import (
    jenga_cache_manager as jenga_mod,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from max.pipelines.request.base import RequestID

SLIDING = "sliding"
FULL = "full"


def make_leaf(
    *,
    n_kv_heads: int,
    window_size: int | None = None,
    page_size: int = 1,
    speculative_method: SpeculativeMethod | None = None,
    num_draft_tokens: int = 0,
    data_parallel_degree: int = 1,
    n_devices: int | None = None,
) -> MHAKVCacheParams:
    device_count = data_parallel_degree if n_devices is None else n_devices
    return MHAKVCacheParams(
        dtype=DType.float32,
        num_layers=1,
        n_kv_heads=n_kv_heads,
        head_dim=1,
        page_size=page_size,
        devices=[DeviceRef.CPU()] * device_count,
        data_parallel_degree=data_parallel_degree,
        window_size=window_size,
        speculative_method=speculative_method,
        num_draft_tokens=num_draft_tokens,
    )


def create_manager(
    params: KVCacheParamInterface, num_huge_blocks: int, max_batch_size: int
) -> JengaKVCacheManager:
    tp_degree = params.tensor_parallel_degree
    huge_page_bytes = max(
        leaf.bytes_per_page // tp_degree for leaf in params.leaves().values()
    )
    return JengaKVCacheManager.create(
        params=params,
        available_bytes=num_huge_blocks * huge_page_bytes * len(params.devices),
        max_batch_size=max_batch_size,
    )


def make_multi_leaf_manager(
    num_huge_blocks: int, max_batch_size: int = 8
) -> JengaKVCacheManager:
    """Two leaves with different bytes-per-page, so they land on different
    per-huge-block ratios -- mirrors gemma4's sliding (ratio 1) vs full
    (ratio 10) shape, just scaled down. n_kv_heads=1 vs 3 with page_size=1,
    head_dim=1, float32 gives 8 vs 24 bytes/page -> ratios 3 and 1.
    """
    params = MultiKVCacheParams.from_params(
        {
            SLIDING: make_leaf(n_kv_heads=1, window_size=4),
            FULL: make_leaf(n_kv_heads=3),
        }
    )
    return create_manager(params, num_huge_blocks, max_batch_size)


def make_single_leaf_manager(
    num_huge_blocks: int,
    max_batch_size: int = 8,
    *,
    speculative_method: SpeculativeMethod | None = None,
    num_draft_tokens: int = 0,
    data_parallel_degree: int = 1,
    page_size: int = 1,
) -> JengaKVCacheManager:
    params = make_leaf(
        n_kv_heads=1,
        speculative_method=speculative_method,
        num_draft_tokens=num_draft_tokens,
        data_parallel_degree=data_parallel_degree,
        page_size=page_size,
    )
    return create_manager(params, num_huge_blocks, max_batch_size)


def make_ctx(num_tokens: int) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=4096,
        tokens=TokenBuffer(np.arange(num_tokens, dtype=np.int64)),
    )


def get_lut(
    kv_inputs: KVCacheInputsInterface[Buffer, Buffer],
    device_idx: int = 0,
) -> list[list[int]]:
    """Returns the assigned block ids in the runtime LUT, trimming the
    tail padding. Unlike the legacy manager (sentinel = total_num_pages),
    Jenga's null block is index 0 -- real block ids start at 1 -- so
    padding columns fill with 0 and get trimmed the same way here.

    ``device_idx`` indexes the ``(replica, TP shard)`` devices in
    replica-major order, so with one shard per replica it is the replica.
    """
    assert isinstance(kv_inputs, KVCacheInputs)
    raw = kv_inputs.inputs[device_idx].lookup_table.to_numpy().tolist()
    return [[b for b in row if b != 0] for row in raw]


def test_runtime_inputs_boundary_matches_real_allocated_blocks() -> None:
    """The insufficient-blocks check must use the request's real per-leaf
    block count, not something derived from the leaf-id dict's keys.

    Regression test for a bug where ``min(len(bs) for bs in
    get_req_blocks_per_leaf(ctx))`` iterated dict *keys* (leaf-id strings)
    instead of ``.values()``, so the boundary was a bogus constant
    (``len("full_attention.full_group")``) instead of the request's actual
    block count.
    """
    mgr = make_multi_leaf_manager(num_huge_blocks=10)
    ctx = make_ctx(num_tokens=3)
    mgr.claim(ctx)
    mgr.alloc(ctx)

    min_blocks = min(
        len(bs) for bs in mgr.get_req_blocks_per_leaf(ctx).values()
    )
    page_size = mgr.params.page_size

    # Exactly at the allocated capacity: must not raise.
    while ctx.tokens.active_length < min_blocks * page_size:
        ctx.update(0)
    mgr.runtime_inputs([[ctx]])

    # One token past capacity: must raise, and say so.
    ctx.update(0)
    with pytest.raises(ValueError, match="does not have sufficient blocks"):
        mgr.runtime_inputs([[ctx]])


def test_runtime_inputs_lut_and_cache_lengths() -> None:
    """Happy path: runtime_inputs reports the blocks alloc() actually gave."""
    mgr = make_single_leaf_manager(num_huge_blocks=10)
    ctx_a = make_ctx(num_tokens=3)
    ctx_b = make_ctx(num_tokens=2)
    mgr.claim(ctx_a)
    mgr.claim(ctx_b)
    mgr.alloc(ctx_a)
    mgr.alloc(ctx_b)

    kv_inputs = mgr.runtime_inputs([[ctx_a, ctx_b]])
    assert isinstance(kv_inputs, KVCacheInputs)

    (leaf_id,) = mgr._leaf_infos
    assert get_lut(kv_inputs) == [
        mgr.get_req_blocks_per_leaf(ctx_a)[leaf_id],
        mgr.get_req_blocks_per_leaf(ctx_b)[leaf_id],
    ]
    cache_lengths = kv_inputs.inputs[0].cache_lengths.to_numpy().tolist()
    assert cache_lengths == [
        ctx_a.tokens.processed_length,
        ctx_b.tokens.processed_length,
    ]


def test_runtime_inputs_without_alloc_raises() -> None:
    """A request that was only claimed, never allocated, cannot run."""
    mgr = make_single_leaf_manager(num_huge_blocks=10)
    ctx = make_ctx(num_tokens=3)
    mgr.claim(ctx)

    with pytest.raises(ValueError, match="does not have sufficient blocks"):
        mgr.runtime_inputs([[ctx]])


def test_create_logs_the_pool_geometry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``create()`` logs one summary line plus one line per leaf, so an
    operator can read the huge/little page split straight from server logs.

    Asserted as one golden block (rather than piecemeal substring checks) so
    a failure prints the actual vs. expected geometry side by side.
    """
    with caplog.at_level(logging.INFO, logger="max.pipelines"):
        make_multi_leaf_manager(num_huge_blocks=10)

    expected = (
        "Jenga KV manager: 10 huge pages x 0.02 KiB = 0.23 KiB (per device), page_size 1 tokens\n"
        "\tsliding.sliding_window_group(4): 30 pages of 0.01 KiB  (3 per huge page)\n"
        "\tfull.full_group                : 10 pages of 0.02 KiB  (1 per huge page)"
    )
    assert "\n".join(r.message for r in caplog.records) == expected


def test_num_draft_tokens_per_step_threaded_from_params() -> None:
    """``__init__`` must forward ``params.num_draft_tokens_per_step`` to the
    block manager instead of relying on its hardcoded default.

    Regression test: the constructor used to omit this kwarg entirely, so
    the block manager's default of 1 could silently diverge from whatever
    ``params.num_draft_tokens_per_step`` returns (e.g. dflash's block-draft
    configs, where it equals ``num_draft_tokens``), desyncing alloc-time
    sizing from the runtime_inputs boundary check that reads the live
    property.
    """
    mgr = make_single_leaf_manager(
        num_huge_blocks=10,
        speculative_method="dflash",
        num_draft_tokens=3,
    )
    assert mgr.params.num_draft_tokens_per_step == 3
    assert mgr._num_draft_tokens_per_step == 3


def test_each_replica_gets_its_own_runtime_inputs() -> None:
    """Every replica prepares its own LUT from its own pool, so a page id in
    one replica's row says nothing about the other's."""
    mgr = make_single_leaf_manager(num_huge_blocks=10, data_parallel_degree=2)
    ctx_a = make_ctx(num_tokens=3)
    ctx_b = make_ctx(num_tokens=2)
    mgr.claim(ctx_a, replica_idx=0)
    mgr.claim(ctx_b, replica_idx=1)
    mgr.alloc(ctx_a)
    mgr.alloc(ctx_b)

    kv_inputs = mgr.runtime_inputs([[ctx_a], [ctx_b]])

    (leaf_id,) = mgr._leaf_infos
    assert get_lut(kv_inputs, device_idx=0) == [
        mgr.get_req_blocks_per_leaf(ctx_a)[leaf_id]
    ]
    assert get_lut(kv_inputs, device_idx=1) == [
        mgr.get_req_blocks_per_leaf(ctx_b)[leaf_id]
    ]
    # Both replicas started from an untouched pool, so both handed out page 1.
    assert get_lut(kv_inputs, device_idx=0)[0][0] == 1
    assert get_lut(kv_inputs, device_idx=1)[0][0] == 1


def test_runtime_inputs_rejects_a_batch_per_replica_mismatch() -> None:
    mgr = make_single_leaf_manager(num_huge_blocks=10, data_parallel_degree=2)
    ctx = make_ctx(num_tokens=3)
    mgr.claim(ctx)
    mgr.alloc(ctx)

    with pytest.raises(ValueError, match="must match number of replicas"):
        mgr.runtime_inputs([[ctx]])


def test_block_count_is_reported_per_replica() -> None:
    mgr = make_single_leaf_manager(num_huge_blocks=10, data_parallel_degree=2)
    ctx = make_ctx(num_tokens=3)
    mgr.claim(ctx, replica_idx=1)
    mgr.alloc(ctx)

    assert mgr.block_count(0).free == 9
    assert mgr.block_count(1).free == 6
    assert mgr.block_count(0).total == mgr.block_count(1).total == 9


def test_a_padding_dummy_runs_on_the_null_page() -> None:
    """DP padding equalizes replica batch sizes under graph capture. The
    dummy has to reach the graph like any other request, pointed at a page
    whose contents nobody reads.

    Uses a production-sized page: a dummy's two tokens fit in its single null
    page, which is what lets it pass the same sufficiency check as everything
    else in the batch rather than needing an exemption from it.
    """
    mgr = make_single_leaf_manager(
        num_huge_blocks=10, data_parallel_degree=2, page_size=128
    )
    real = make_ctx(num_tokens=3)
    mgr.claim(real, replica_idx=0)
    mgr.alloc(real)

    # Built the way DPBatchPadder builds them, so the manager sees a real
    # padding context rather than an ordinary one-token request.
    dummy = TextContext.new_padding_context(max_length=4096, model_name="test")
    dummy.update(0)
    mgr.alloc_dummy(dummy, replica_idx=1)

    kv_inputs = mgr.runtime_inputs([[real], [dummy]])

    assert isinstance(kv_inputs, KVCacheInputs)
    # Trimmed of its zeros the dummy's row is empty: every column is the null
    # page, which is exactly page 0.
    assert get_lut(kv_inputs, device_idx=1) == [[]]
    assert mgr.block_count(1).free == mgr.block_count(1).total

    mgr.release(dummy)

    assert mgr.block_count(1).free == mgr.block_count(1).total


def test_metrics_are_reported_and_reset() -> None:
    mgr = make_single_leaf_manager(num_huge_blocks=10)
    ctx = make_ctx(num_tokens=3)
    mgr.claim(ctx)
    mgr.alloc(ctx)

    assert mgr.get_metrics_aggregated().input_tokens == 3

    mgr.reset_metrics()

    assert mgr.get_metrics_aggregated() == KVCacheMetrics()


def test_alloc_reserves_the_draft_positions_runtime_inputs_will_ask_for() -> (
    None
):
    """Allocation and the runtime_inputs boundary check both size the request
    from ``params.num_draft_tokens``. When only the latter did, a speculative
    request was refused for pages the former never drew.
    """
    mgr = make_single_leaf_manager(
        num_huge_blocks=40,
        speculative_method="eagle",
        num_draft_tokens=3,
    )
    ctx = make_ctx(num_tokens=3)
    mgr.claim(ctx)
    mgr.alloc(ctx)

    # Not raising is the assertion: the boundary check reads the live
    # ``num_draft_tokens`` and must find the pages alloc already drew.
    mgr.runtime_inputs([[ctx]])


# ===--------------------------------------------------------------------=== #
# max_seq_len startup validation
# ===--------------------------------------------------------------------=== #


def test_create_rejects_oversized_max_seq_len() -> None:
    params = make_leaf(n_kv_heads=1, page_size=128)
    huge_page_bytes = next(iter(params.leaves().values())).bytes_per_page
    with pytest.raises(RuntimeError, match="max sequence length"):
        JengaKVCacheManager.create(
            params=params,
            available_bytes=2 * huge_page_bytes,
            max_batch_size=1,
            max_seq_len=10_000,
        )


def test_create_accepts_max_seq_len_that_fits() -> None:
    params = make_leaf(n_kv_heads=1, page_size=128)
    huge_page_bytes = next(iter(params.leaves().values())).bytes_per_page
    manager = JengaKVCacheManager.create(
        params=params,
        available_bytes=40 * huge_page_bytes,
        max_batch_size=8,
        max_seq_len=128,
    )
    assert manager.effective_max_seq_length is not None


def test_kv_budget_is_split_across_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``available_bytes`` is the KV budget across all devices; each device
    slab must not be that full amount. Memory estimation sums free memory
    across GPUs, then Jenga used to size every ``Buffer.zeros`` from that
    total.
    """
    params = make_leaf(n_kv_heads=1, data_parallel_degree=2)
    huge_page_bytes = next(iter(params.leaves().values())).bytes_per_page
    allocated_shapes: list[tuple[int, ...]] = []
    real_zeros = jenga_mod.Buffer.zeros

    def spy_zeros(
        shape: Sequence[int],
        dtype: DType,
        device: Device | None = None,
        usage: Usage = Usage.DEFAULT,
    ) -> Buffer:
        allocated_shapes.append(tuple(int(dim) for dim in shape))
        return real_zeros(shape=shape, dtype=dtype, device=device, usage=usage)

    monkeypatch.setattr(jenga_mod.Buffer, "zeros", spy_zeros)

    total_huge_blocks = 8
    mgr = JengaKVCacheManager.create(
        params=params,
        available_bytes=total_huge_blocks * huge_page_bytes,
        max_batch_size=8,
    )
    per_device_huge_blocks = total_huge_blocks // 2
    assert mgr._num_huge_blocks == per_device_huge_blocks
    expected = (per_device_huge_blocks, huge_page_bytes)
    assert allocated_shapes == [expected, expected]


def test_tp_huge_pages_use_per_device_page_size() -> None:
    """TP inflates ``leaf.bytes_per_page`` to the replica total. Each device
    slab is one shard, so huge-page count must use the per-device stride.
    """
    params = make_leaf(n_kv_heads=2, data_parallel_degree=1, n_devices=2)
    assert params.tensor_parallel_degree == 2
    replica_page = next(iter(params.leaves().values())).bytes_per_page
    per_device_page = replica_page // 2
    mgr = JengaKVCacheManager.create(
        params=params,
        available_bytes=8 * per_device_page * 2,
        max_batch_size=8,
    )
    assert mgr._num_huge_blocks == 8


def test_mixed_dp_tp_splits_budget_and_uses_per_device_pages() -> None:
    params = make_leaf(n_kv_heads=2, data_parallel_degree=2, n_devices=4)
    assert params.tensor_parallel_degree == 2
    replica_page = next(iter(params.leaves().values())).bytes_per_page
    per_device_page = replica_page // 2
    mgr = JengaKVCacheManager.create(
        params=params,
        available_bytes=8 * per_device_page * 4,
        max_batch_size=8,
    )
    assert mgr._num_huge_blocks == 8
