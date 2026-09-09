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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheInputs,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheInputs,
    MultiKVCacheParams,
)
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context


def _create_kv_manager(
    data_parallel_degree: int,
    num_devices: int,
    batch_size: int | None = None,
    session: InferenceSession | None = None,
) -> PagedKVCacheManager:
    """Creates a PagedKVCacheManager with the given data parallel degree
    and number of devices.

    The maximum batch size is 2 * num_devices.
    """
    batch_size = 2 * num_devices if batch_size is None else batch_size

    devices = [Accelerator(id=i) for i in range(num_devices)]
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=32,
        num_layers=10,
        devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        data_parallel_degree=data_parallel_degree,
    )
    session = (
        session if session is not None else InferenceSession(devices=devices)
    )
    manager = PagedKVCacheManager(
        params=params,
        session=session,
        total_num_pages=8,
        max_batch_size=128,
    )
    assert isinstance(manager, PagedKVCacheManager)
    return manager


def test_claim() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)

    max_batch_size = 10
    batch = []
    for i in range(max_batch_size * data_parallel_degree):
        # TokenBuffer requires at least one token, so start from 1
        context = create_text_context(np.empty(max(i, 1)))
        replica_idx = i % data_parallel_degree
        kv_manager.claim(context, replica_idx=replica_idx)
        batch.append((replica_idx, context))

    new_context = create_text_context(np.empty(max(i, 1)))

    # Release a slot.
    replica_idx, context = batch[0]
    kv_manager.release(context)
    assert not kv_manager.contains(context)

    # Check that the new context can be claimed using the released slot.
    kv_manager.claim(new_context, replica_idx=replica_idx)
    assert kv_manager.contains(new_context)


def test_step() -> None:
    data_parallel_degree = 2
    num_devices = 2

    kv_manager = _create_kv_manager(data_parallel_degree, num_devices)

    # Create text contexts and externally claim each using their request_id
    prompt_lens = [3, 4, 7]
    batch = []
    batches_by_replica: list[list[TextContext]] = [
        [] for _ in range(data_parallel_degree)
    ]
    for i, prompt_len in enumerate(prompt_lens):
        context = create_text_context(np.empty(prompt_len))
        replica_idx = i % data_parallel_degree
        kv_manager.claim(context, replica_idx=replica_idx)
        batch.append(context)
        batches_by_replica[replica_idx].append(context)

    # Assert that each cache_length is initialized appropriately as 0
    for ctx in batch:
        assert ctx.tokens.processed_length == 0

    # Update these values a few times
    for j in range(3):
        for ctx in batch:
            kv_manager.alloc(ctx)
        kv_manager.runtime_inputs(batches_by_replica)
        for ctx in batch:
            ctx.update(42)
        for replica_batch in batches_by_replica:
            for ctx in replica_batch:
                kv_manager.step(ctx)

        for i, ctx in enumerate(batch):
            assert ctx.tokens.processed_length == prompt_lens[i] * (j + 1)

        for i, ctx in enumerate(batch):
            orig_processed_length = ctx.tokens.processed_length
            for _ in range(prompt_lens[i] - 1):
                ctx.update(42)
            ctx.tokens.rewind_processing(
                ctx.tokens.processed_length - orig_processed_length
            )


def test_runtime_inputs_requires_per_replica_batches() -> None:
    kv_manager = _create_kv_manager(data_parallel_degree=2, num_devices=2)

    with pytest.raises(ValueError):
        kv_manager.runtime_inputs([[]])


@dataclass
class PrevModelInputs:
    input_row_offsets: Buffer
    data_parallel_splits: Buffer
    signal_buffers: list[Buffer] = field(default_factory=list)


def _bytes_per_block(buf: Buffer) -> int:
    return buf.num_elements * buf.dtype.size_in_bytes // buf.shape[0]


def _write_block_pattern(buf: Buffer, block_id: int, seed: int) -> np.ndarray:
    """Write a deterministic uint8 pattern into one device block.

    Returns the pattern (ground truth) as a 1-D uint8 array.
    """
    nbytes = _bytes_per_block(buf)
    pattern = np.random.RandomState(seed).randint(
        0, 256, size=(nbytes,), dtype=np.uint8
    )
    host = Buffer.from_numpy(pattern.copy())
    buf.view(dtype=DType.uint8, shape=[buf.shape[0], nbytes])[
        block_id, :
    ].inplace_copy_from(host.to(buf.device))
    return pattern


def _read_block_bytes(buf: Buffer, block_id: int) -> np.ndarray:
    nbytes = _bytes_per_block(buf)
    return (
        buf.view(dtype=DType.uint8, shape=[buf.shape[0], nbytes])[block_id, :]
        .to_numpy()
        .reshape(-1)
        .copy()
    )


def test_cross_replica_gpu_prefix_cache_hit() -> None:
    """A request on replica 1 reuses prefix blocks cached on replica 0.

    Replica 0's device prefix cache is seeded with two committed blocks holding
    known data. An identical prompt is then admitted on replica 1; the manager
    must materialize those blocks onto replica 1 via a device-to-device copy
    (SERVOPT-1500), advancing the request's token window and producing
    byte-identical KV data on the destination replica.
    """
    if accelerator_count() < 2:
        pytest.skip("Need at least 2 GPUs")

    num_devices = 2
    data_parallel_degree = 2
    page_size = 16

    devices = [Accelerator(id=i) for i in range(num_devices)]
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=4,
        head_dim=32,
        num_layers=2,
        page_size=page_size,
        enable_prefix_caching=True,
        devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        data_parallel_degree=data_parallel_degree,
    )
    session = InferenceSession(devices=devices)
    manager = PagedKVCacheManager(
        params=params,
        session=session,
        total_num_pages=16,
        max_batch_size=128,
    )

    bm = manager._block_manager
    pool0 = bm.device_block_pools[0]
    pool1 = bm.device_block_pools[1]

    # Build a request whose prompt spans two full prefix blocks (the trailing
    # token is never hashed), then derive its per-block hashes.
    num_prompt_tokens = 2 * page_size + 1
    ctx = create_text_context(np.arange(num_prompt_tokens))
    bm.compute_hashes_for_request(ctx)
    hashes = cast("list[bytes]", list(bm.req_to_hashes[ctx.request_id]))
    assert len(hashes) == 2

    # Seed replica 0's device prefix cache with the two blocks, each holding a
    # distinct known pattern in replica 0's KV buffer.
    buf0 = manager.get_device_buffer(0).all_buffers[0]
    buf1 = manager.get_device_buffer(1).all_buffers[0]
    expected: list[np.ndarray] = []
    for i, block_hash in enumerate(hashes):
        block = bm.allocate_device_block(0)
        expected.append(_write_block_pattern(buf0, block.bid, seed=100 + i))
        pool0.commit_into_prefix_cache(block_hash, block)

    assert len(pool1.prefix_cache) == 0

    # Admit the identical prompt on replica 1: triggers the cross-replica copy.
    manager.claim(ctx, replica_idx=1)
    manager.alloc(ctx)

    # Both prefix blocks were served cross-replica.
    metrics = manager.get_metrics_aggregated()
    assert metrics.cross_replica_blocks_copied == 2
    assert metrics.cross_replica_bytes_copied == 2 * _bytes_per_block(buf0)
    # A cross-replica D2D copy is not a local device hit: must not also tick
    # device_blocks_served (regression guard for the G0/G0-DP double-count
    # bug -- CENG-845).
    assert metrics.device_blocks_served == 0

    # The request's window advanced past the two reused blocks.
    assert ctx.cached_prefix_length == 2 * page_size

    # Replica 1 now holds the two blocks, byte-identical to replica 0's copies.
    for i, block_hash in enumerate(hashes):
        assert block_hash in pool1.prefix_cache
        dst_block = pool1.prefix_cache[block_hash]
        got = _read_block_bytes(buf1, dst_block.bid)
        np.testing.assert_array_equal(got, expected[i])


# ``test_get_metrics_aggregated_h2d_d2h``, ``test_get_metrics_aggregated_disk_ops``
# and ``test_cross_replica_host_prefix_cache_hit`` were removed with the Python
# host/disk tier: that tier is now the Rust ``rust_tiered`` connector, whose pyo3
# extension may only be depended on from an internal-only package. Its tier
# metrics and host/disk residency coverage lives in
# ``internal/dkv/test_rust_tiered_connector_gpu.py``; the shared-connector DP
# wiring stays covered by ``test_cross_replica_gpu_prefix_cache_hit`` above.


def test_runtime_inputs_mha_primary_mla_secondary_matches_graph() -> None:
    """Runtime KV input count must match the graph for an MLA secondary cache.

    Regression for MiniMax-M3 sparse attention: a ``MultiKVCacheParams`` whose
    primary is a non-MLA GQA cache and whose secondary is an ``is_mla`` index-K
    cache.  The graph declares ``mla_num_partitions`` for each MLA cache (via
    ``get_symbolic_inputs``), but the runtime previously derived that scalar
    only from the primary (non-MLA) cache and applied it to every cache, so the
    secondary cache's ``mla_num_partitions`` was dropped — the fed input count
    fell short of the compiled graph by one per device (``ValueError: Number of
    inputs ... does not match expected number``).  The flattened runtime inputs
    must match the flattened symbolic graph inputs exactly.
    """
    num_devices = 2

    devices = [Accelerator(id=i) for i in range(num_devices)]
    device_refs = [DeviceRef.GPU(i) for i in range(num_devices)]

    # Primary: non-MLA GQA cache (mirrors M3 main attention).
    main_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        num_layers=4,
        devices=device_refs,
        page_size=128,
    )
    # Secondary: is_mla index-K cache (1 KV head, replicated K), mirrors M3's
    # indexer cache and DeepSeek-V3.2's order *reversed* (there the MLA cache is
    # primary, so this asymmetry is exercised only by M3).
    indexer_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=128,
        num_layers=4,
        devices=device_refs,
        page_size=128,
        num_q_heads=64,
    )
    params = MultiKVCacheParams.from_params(
        {"main": main_params, "indexer": indexer_params}
    )

    session = InferenceSession(devices=devices)
    manager = PagedKVCacheManager(
        params=params,
        session=session,
        total_num_pages=8,
        max_batch_size=128,
    )

    context = create_text_context(np.empty(4))
    manager.claim(context)
    manager.alloc(context)

    kv_cache_inputs = manager.runtime_inputs([[context]])
    assert isinstance(kv_cache_inputs, MultiKVCacheInputs)

    # The compiled graph declares its KV inputs from the same symbolic params.
    num_graph_inputs = len(params.flattened_kv_inputs())
    num_runtime_inputs = len(kv_cache_inputs.flatten())

    assert num_runtime_inputs == num_graph_inputs, (
        f"runtime fed {num_runtime_inputs} KV inputs but the graph expects "
        f"{num_graph_inputs}"
    )

    # The MLA secondary cache must contribute its per-device mla_num_partitions.
    secondary_inputs = kv_cache_inputs.children["indexer"]
    assert isinstance(secondary_inputs, KVCacheInputs)
    assert len(secondary_inputs.inputs) == num_devices
    for per_device in secondary_inputs.inputs:
        assert per_device.mla_num_partitions is not None
