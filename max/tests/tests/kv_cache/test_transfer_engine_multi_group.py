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

"""CPU unit tests for KVTransferEngine's per-group NIXL registration.

Covers two things, using CPU buffers only (no NIXL/GPU objects required):

- ``to_memory``: how a KV cache buffer authors one NIXL group per
  logical ``(child, kind)`` tensor (values, scales, or a nested child cache).
- ``KVTransferEngine.__init__`` structural-validation guards, which raise
  before any NIXL registration.
"""

from __future__ import annotations

import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.nn.kv_cache.cache_params import (
    KVCacheBuffer,
    KVCacheMemory,
    MultiKVCacheBuffer,
)
from max.pipelines.kv_cache import KVTransferEngine
from max.pipelines.kv_cache.paged_kv_cache.transfer_engine import (
    _build_group_descriptors,
    _resolve_remote_bytes_per_group,
    _validate_tensor_shape,
)


def _cpu_buf(
    num_pages: int,
    elts_per_page: int,
    dtype: DType = DType.bfloat16,
) -> Buffer:
    """Allocate a 2-D CPU buffer in the original (non-uint8) dtype."""
    return Buffer(shape=(num_pages, elts_per_page), dtype=dtype, device=CPU())


def _kv(
    elts_per_page: int,
    *,
    tp: int = 2,
    num_pages: int = 8,
    replicated: bool = False,
    dtype: DType = DType.bfloat16,
    scale_elts: int | None = None,
) -> KVCacheBuffer:
    """A single-cache ``KVCacheBuffer`` with ``tp`` TP shards.

    Pass ``scale_elts`` to attach a float32 scales tensor (quantized cache).
    """
    return KVCacheBuffer(
        values=[_cpu_buf(num_pages, elts_per_page, dtype) for _ in range(tp)],
        scales=(
            [_cpu_buf(num_pages, scale_elts, DType.float32) for _ in range(tp)]
            if scale_elts is not None
            else None
        ),
        replicates_kv_across_tp=replicated,
    )


def _hetero_multi(*, num_pages: int = 8, tp: int = 2) -> MultiKVCacheBuffer:
    """Multi-child cache whose children differ in bytes_per_page.

    A 61-"layer" target next to a 1-"layer" draft: the two children have
    different per-page byte sizes and must land in separate NIXL groups. This
    is the case that motivated per-child grouping.
    """
    return MultiKVCacheBuffer(
        children={
            "target": _kv(61 * 8, tp=tp, num_pages=num_pages),
            "draft": _kv(8, tp=tp, num_pages=num_pages),
        }
    )


# ---------------------------------------------------------------------------
# to_memory: one authored NIXL group per (child, kind)
# ---------------------------------------------------------------------------


def test_to_memory_sharded() -> None:
    """Non-replicated single cache: one group holding all TP shards."""
    groups = _kv(64).to_memory()
    assert len(groups) == 1
    assert not groups[0].replicated
    assert len(groups[0].buffers) == 2  # 2 TP shards
    # bfloat16: 64 elts/page * 2 bytes = 128 bytes/page.
    assert groups[0].bytes_per_page == 128
    assert groups[0].total_num_pages == 8


def test_to_memory_replicated() -> None:
    """Replicated cache: one group carrying every TP shard."""
    groups = _kv(64, tp=3, replicated=True).to_memory()
    assert len(groups) == 1
    assert groups[0].replicated
    assert len(groups[0].buffers) == 3  # one view per TP shard


def test_to_memory_quantized() -> None:
    """Quantized cache: values and scales become separate NIXL groups."""
    groups = _kv(64, dtype=DType.uint8, scale_elts=4).to_memory()
    assert len(groups) == 2  # values group + scales group
    values, scales = groups
    assert len(values.buffers) == 2 and len(scales.buffers) == 2
    # uint8 values: 64 bytes/page; float32 scales: 4 elts * 4 bytes = 16.
    assert values.bytes_per_page == 64
    assert scales.bytes_per_page == 16


def test_to_memory_multi_child_heterogeneous() -> None:
    """Multi-child cache with different shapes: one group per child."""
    groups = _hetero_multi().to_memory()
    assert len(groups) == 2  # one group per child
    assert len(groups[0].buffers) == 2 and len(groups[1].buffers) == 2
    # Different per-page byte sizes -> kept in separate groups.
    assert groups[0].bytes_per_page != groups[1].bytes_per_page


def test_to_memory_nested_quantized_child_count() -> None:
    """A nested tree yields one group per leaf (child, kind).

    Mirrors the DISTINF-383 nested-tree case authored-side: a quantized
    ``target`` (values + scales) beside a non-quantized ``draft`` (values only)
    produces three groups, in child-major then kind order.
    """
    tree = MultiKVCacheBuffer(
        children={
            "target": _kv(64, dtype=DType.uint8, scale_elts=4),
            "draft": _kv(8),
        }
    )
    groups = tree.to_memory()
    assert len(groups) == 3  # target values, target scales, draft values
    assert [g.bytes_per_page for g in groups] == [64, 16, 16]
    assert all(not g.replicated for g in groups)


# ---------------------------------------------------------------------------
# from_paged memory build (guards the re-flatten regression)
# ---------------------------------------------------------------------------


def test_per_replica_groups_keep_children_separate() -> None:
    """``[buf.to_memory() ...]`` keeps a MultiKVCacheBuffer's children
    as separate authored groups.

    Guards against regressing to a flattened ``all_buffers`` layout (the
    multi-cache heterogeneous-shape crash). CPU-only: exercises the build step
    without constructing an engine (which needs NIXL).
    """
    memory = [
        buf.to_memory()
        for buf in [_hetero_multi(num_pages=4), _hetero_multi(num_pages=4)]
    ]

    assert len(memory) == 2
    assert all(isinstance(g, KVCacheMemory) for g in memory[0])
    # Children stay in separate groups with different bytes_per_page.
    assert len(memory[0]) == 2
    assert memory[0][0].bytes_per_page != memory[0][1].bytes_per_page


# ---------------------------------------------------------------------------
# KVTransferEngine.__init__ structural guards (raise before NIXL)
# ---------------------------------------------------------------------------


def _group(
    elts_per_page: int = 64,
    *,
    num_pages: int = 4,
    tp: int = 2,
    replicated: bool = False,
) -> KVCacheMemory:
    """A transport NIXL group of ``tp`` uint8-view TP shards."""
    return KVCacheMemory(
        replicated=replicated,
        buffers=[
            _cpu_buf(num_pages, elts_per_page, DType.uint8) for _ in range(tp)
        ],
    )


# ---------------------------------------------------------------------------
# KVCacheMemory: all shards must agree on shape
# ---------------------------------------------------------------------------


def test_group_rejects_mismatched_shard_shape() -> None:
    """``bytes_per_page``/``total_num_pages`` are read off shard 0, so every
    shard must agree with it or those properties would silently report the
    wrong value for a mismatched shard."""
    good = _cpu_buf(4, 64, DType.uint8)
    different_bytes_per_page = _cpu_buf(4, 32, DType.uint8)
    with pytest.raises(ValueError, match="must share a shape"):
        KVCacheMemory(
            replicated=False, buffers=[good, different_bytes_per_page]
        )

    different_num_pages = _cpu_buf(8, 64, DType.uint8)
    with pytest.raises(ValueError, match="must share a shape"):
        KVCacheMemory(replicated=False, buffers=[good, different_num_pages])


# ---------------------------------------------------------------------------
# Per-group replication
# ---------------------------------------------------------------------------
#
# Replication is per-group, so mixed replication within one engine is
# representable (no blanket "same replication kind" assertion). The per-group
# routing plan lives in resolve_transfer_strategy and is covered in
# test_transfer_engine_resolver.py.


def test_inconsistent_replication_across_replicas_raises() -> None:
    """A group index must agree on ``replicated`` across DP replicas.

    Unlike "all groups agree" (dropped), a logical cache is consistently
    replicated across replicas, so replica 1's group 0 disagreeing with replica
    0's group 0 is rejected. This raises before any NIXL registration, so it is
    CPU-testable.
    """
    replica0 = [_group(replicated=True)]
    replica1 = [_group()]  # same group index, different replication kind
    with pytest.raises(ValueError, match="consistently across DP replicas"):
        KVTransferEngine("engine", [replica0, replica1])


def test_replicas_with_different_group_counts_raises() -> None:
    """Replicas that provide different NIXL group counts are rejected."""
    # Replica 0: two groups; replica 1: one group.
    replica0 = [_group(64), _group(16)]
    replica1 = [_group(64)]

    with pytest.raises(ValueError, match="consistent buffer structure"):
        KVTransferEngine("engine", [replica0, replica1])


# ---------------------------------------------------------------------------
# Per-group descriptor arithmetic (the SERVOPT-1456 stride-mismatch guard)
# ---------------------------------------------------------------------------


def test_build_group_descriptors_uses_own_stride_per_group() -> None:
    """Each group is addressed with its OWN base + bytes_per_page.

    Guards the descriptor math behind SERVOPT-1456: page ``i`` of group ``g``
    lands at ``base[g] + i*bpp[g]`` with size ``bpp[g]``. A group with a
    different ``bytes_per_page`` (e.g. a 1-layer draft next to a 61-layer
    target) can never inherit another group's stride.
    """
    base_addrs = [0x1000, 0x5000]  # target group, draft group
    bytes_per_group = [800, 16]  # deliberately different per-page sizes
    page_idxs = [0, 2, 5]
    device_id = 3

    descs = _build_group_descriptors(
        base_addrs, bytes_per_group, page_idxs, device_id
    )

    # Group-major, page-inner ordering; one descriptor per (group, page).
    assert len(descs) == len(bytes_per_group) * len(page_idxs)

    # Group 0 (target): base 0x1000, stride 800.
    for i, k in enumerate(page_idxs):
        addr, size, dev = descs[i]
        assert addr == 0x1000 + k * 800
        assert size == 800
        assert dev == device_id

    # Group 1 (draft): base 0x5000, stride 16 -- its own bpp, not the target's.
    off = len(page_idxs)
    for i, k in enumerate(page_idxs):
        addr, size, _ = descs[off + i]
        assert addr == 0x5000 + k * 16
        assert size == 16


# ---------------------------------------------------------------------------
# _resolve_remote_bytes_per_group: a read transfer's remote-side stride list
# ---------------------------------------------------------------------------


def test_resolve_remote_bytes_per_group_uses_remote_advertised_strides() -> (
    None
):
    """Remote advertises an exactly matching per-group breakdown -- use it."""
    assert _resolve_remote_bytes_per_group([100, 16], [100, 16]) == [100, 16]


@pytest.mark.parametrize(
    "remote_bpg",
    [
        [100],  # fewer groups than local
        [],  # no per-group breakdown at all
        [100, 16, 8],  # more groups than local
        [100, 8],  # same count, but a differing stride value
    ],
)
def test_resolve_remote_bytes_per_group_raises_on_mismatch(
    remote_bpg: list[int],
) -> None:
    """Anything but an exact match must raise -- fewer groups means there is
    no stride to infer for a group the remote never advertised; more groups
    or a differing stride means assuming a positional correspondence
    connect() never validated."""
    with pytest.raises(ValueError, match="never advertised"):
        _resolve_remote_bytes_per_group([100, 16], remote_bpg)


# ---------------------------------------------------------------------------
# _validate_tensor_shape: cross-shard shape-mismatch rejection
# ---------------------------------------------------------------------------


def test_validate_tensor_shape_rejects_mismatched_shards() -> None:
    """Shards of one group must share a shape (subsumes elt-count + dtype).

    ``to_memory()`` emits 2-D uint8 views (``[total_num_pages,
    bytes_per_page]``), so shape-equality is the single invariant a NIXL group
    needs: a differing page count or per-page stride across shards is rejected
    outright rather than silently producing a mismatched transfer descriptor.
    """
    good = _cpu_buf(17, 24, DType.uint8)
    # Same per-page stride, different page count.
    fewer_pages = _cpu_buf(9, 24, DType.uint8)
    with pytest.raises(ValueError, match="same shape"):
        _validate_tensor_shape([good, fewer_pages])

    # Same page count, different per-page stride.
    wider = _cpu_buf(17, 48, DType.uint8)
    with pytest.raises(ValueError, match="same shape"):
        _validate_tensor_shape([good, wider])
