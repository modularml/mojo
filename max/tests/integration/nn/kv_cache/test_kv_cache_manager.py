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


import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheInputs,
    MultiKVCacheParams,
)
from max.nn.kv_cache.utils import padded_lut_cols
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context


def _make_kv_manager(
    *,
    num_devices: int = 1,
    page_size: int = 128,
    total_num_pages: int = 8,
    n_kv_heads: int = 1,
    head_dim: int = 16,
    num_layers: int = 10,
    max_batch_size: int = 128,
    devices: list[DeviceRef] | None = None,
    **extra_kv_params: object,
) -> PagedKVCacheManager:
    devices = [DeviceRef.CPU() for _ in range(num_devices)]
    params: KVCacheParams
    if extra_kv_params.pop("is_mla", False):
        params = MLAKVCacheParams(
            dtype=DType.float32,
            head_dim=head_dim,
            num_layers=num_layers,
            page_size=page_size,
            devices=devices or [DeviceRef.CPU()],
            **extra_kv_params,  # type: ignore[arg-type]
        )
    else:
        params = MHAKVCacheParams(
            dtype=DType.float32,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            page_size=page_size,
            devices=devices or [DeviceRef.CPU()],
            **extra_kv_params,  # type: ignore[arg-type]
        )
    return PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=[CPU()]),
        total_num_pages=total_num_pages,
        max_batch_size=max_batch_size,
    )


@pytest.mark.asyncio
async def test_step() -> None:
    kv_manager = _make_kv_manager(n_kv_heads=8, head_dim=128)

    prompt_lens = [3, 4, 7]
    batch = []
    for i in range(3):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context)
        batch.append(context)

    # Assert that each cache_length is initialized appropriately as 0
    for ctx in batch:
        assert ctx.tokens.processed_length == 0

    # Update these values a few times
    for j in range(3):
        for ctx in batch:
            kv_manager.alloc(ctx)
        kv_manager.runtime_inputs([batch])
        for ctx in batch:
            ctx.update(42)
        for ctx in batch:
            kv_manager.step(ctx)

        for i, ctx in enumerate(batch):
            assert ctx.tokens.processed_length == prompt_lens[i] * (j + 1)

        for i, ctx in enumerate(batch):
            orig_start_idx = ctx.tokens.processed_length
            for _ in range(prompt_lens[i] - 1):
                ctx.update(42)

            ctx.tokens.rewind_processing(
                ctx.tokens.processed_length - orig_start_idx
            )


@pytest.mark.asyncio
async def test_claim_and_release() -> None:
    # Initialize llama like params
    # claim and release are both cache_type independent,
    # so we can test with the KVCacheType.CONTINUOUS default
    kv_manager = _make_kv_manager(n_kv_heads=8, head_dim=128)

    contexts = []
    prompt_lens = [2, 3, 4, 5, 6]
    for i in range(5):
        context = create_text_context(np.empty(prompt_lens[i]))
        kv_manager.claim(context)
        contexts.append(context)

    # Claim 5 ids
    assert len(contexts) == 5
    assert all(kv_manager.contains(ctx) for ctx in contexts)

    # Claim another 3 ids
    contexts_2 = []
    prompt_lens_2 = [7, 8, 9]
    for i in range(3):
        context = create_text_context(np.empty(prompt_lens_2[i]))
        kv_manager.claim(context)
        contexts_2.append(context)

    assert all(kv_manager.contains(ctx) for ctx in contexts_2)

    # Release a request that has not been claimed
    with pytest.raises(ValueError):
        kv_manager.release(create_text_context(np.empty(2)))

    # Release all ids
    for context in contexts + contexts_2:
        kv_manager.release(context)
        assert not kv_manager.contains(context)


@pytest.mark.asyncio
async def test_release_returns_pages_to_the_claiming_replica_once() -> None:
    """A repeat release is a caller bug, not something the manager absorbs.

    The claim is what names the pool the pages return to, so a release without
    one has no replica to resolve. Defaulting it to replica 0 would hand a
    different replica's pool the pages, so it raises instead.
    """
    kv_manager = _make_kv_manager(num_devices=2, data_parallel_degree=2)

    ctx = create_text_context(np.zeros(256, dtype=np.int64))
    kv_manager.claim(ctx, replica_idx=1)
    kv_manager.alloc(ctx)
    assert kv_manager.block_count(replica_idx=1).used > 0

    kv_manager.release(ctx)

    # The pages went back to the replica that lent them, and replica 0 -- the
    # index a claim-less release would fall back to -- was never touched.
    assert kv_manager.block_count(replica_idx=1).used == 0
    assert kv_manager.block_count(replica_idx=0).used == 0

    with pytest.raises(ValueError, match="not claimed"):
        kv_manager.release(ctx)


@pytest.mark.asyncio
async def test_fetch_paged() -> None:
    kv_manager = _make_kv_manager()

    # Claim 5 items
    contexts = []
    for _ in range(5):
        context = create_text_context(np.empty(1))
        kv_manager.claim(context)
        contexts.append(context)

    # Fetch 3 of the 5 contexts created above
    for ctx in contexts[:3]:
        kv_manager.alloc(ctx)
    _ = kv_manager.runtime_inputs_for_leaf([contexts[:3]]).inputs[0]


@pytest.mark.asyncio
async def test_claim_alloc_and_release() -> None:
    kv_manager = _make_kv_manager()
    contexts = [
        create_text_context(np.zeros(1, dtype=np.int64)) for _ in range(2)
    ]

    for context in contexts:
        kv_manager.claim(context)
        kv_manager.alloc(context)
        assert kv_manager.contains(context)

    for context in contexts:
        kv_manager.release(context)
        assert not kv_manager.contains(context)


@pytest.mark.asyncio
async def test_fetch_paged_lookup_table_tracks_required_page_capacity() -> None:
    kv_manager = _make_kv_manager()

    short_context = create_text_context(np.zeros(1, dtype=np.int64))
    kv_manager.claim(short_context)

    kv_manager.alloc(short_context)
    first_inputs = kv_manager.runtime_inputs_for_leaf([[short_context]]).inputs[
        0
    ]
    assert tuple(first_inputs.lookup_table.shape) == (1, padded_lut_cols(1))

    long_context = create_text_context(np.zeros(256, dtype=np.int64))
    kv_manager.claim(long_context)

    kv_manager.alloc(long_context)
    second_inputs = kv_manager.runtime_inputs_for_leaf([[long_context]]).inputs[
        0
    ]
    assert tuple(second_inputs.lookup_table.shape) == (1, padded_lut_cols(2))


@pytest.mark.asyncio
async def test_runtime_inputs_lookup_table_uses_explicit_max_cache_length() -> (
    None
):
    total_num_pages = 8
    kv_manager = _make_kv_manager(total_num_pages=total_num_pages)

    context = create_text_context(np.zeros(1, dtype=np.int64))
    kv_manager.claim(context)
    kv_manager.alloc(context)

    runtime_inputs = kv_manager.runtime_inputs_for_leaf([[context]]).inputs[0]
    assert tuple(runtime_inputs.lookup_table.shape) == (1, padded_lut_cols(1))

    explicit_inputs = kv_manager.runtime_inputs_for_leaf(
        [[context]],
        max_cache_length=1024,
    ).inputs[0]
    assert tuple(explicit_inputs.lookup_table.shape) == (
        1,
        padded_lut_cols(total_num_pages),
    )


@pytest.mark.asyncio
async def test_mla_runtime_inputs_handles_empty_replica_batch() -> None:
    """MLA runtime inputs should handle empty per-replica batches.

    This exercises data-parallel serving where one replica has work and another
    replica is idle in the same scheduler iteration.
    """
    kv_manager = _make_kv_manager(
        num_devices=2,
        data_parallel_degree=2,
        is_mla=True,
        num_q_heads=8,
    )

    context = create_text_context(np.zeros(1, dtype=np.int64))
    kv_manager.claim(context, replica_idx=0)
    kv_manager.alloc(context)

    runtime_inputs = kv_manager.runtime_inputs_for_leaf([[context], []])
    assert len(runtime_inputs.inputs) == 2
    assert runtime_inputs.inputs[0].attention_dispatch_metadata is not None
    assert runtime_inputs.inputs[1].attention_dispatch_metadata is not None


@pytest.mark.asyncio
async def test_mixed_dp_tp_runtime_inputs_copy_lut_within_replica() -> None:
    """DP2 TP4 runtime inputs should be replica-major with shared TP LUTs."""
    kv_manager = _make_kv_manager(
        num_devices=8,
        total_num_pages=16,
        page_size=128,
        data_parallel_degree=2,
        is_mla=True,
        num_q_heads=128,
    )
    assert kv_manager.params.tensor_parallel_degree == 4

    replica_batches = []
    for replica_idx, token_count in enumerate((1, 129)):
        context = create_text_context(np.zeros(token_count, dtype=np.int64))
        kv_manager.claim(context, replica_idx=replica_idx)
        kv_manager.alloc(context)
        replica_batches.append([context])

    runtime_inputs = kv_manager.runtime_inputs_for_leaf(replica_batches)
    assert len(runtime_inputs.inputs) == 8

    assert tuple(runtime_inputs.inputs[0].lookup_table.shape) == (
        1,
        padded_lut_cols(1),
    )
    assert tuple(runtime_inputs.inputs[4].lookup_table.shape) == (
        1,
        padded_lut_cols(2),
    )

    for replica_start in (0, 4):
        base = runtime_inputs.inputs[replica_start]
        base_lut = base.lookup_table.to_numpy()
        base_cache_lengths = base.cache_lengths.to_numpy()
        for tp_shard in range(replica_start + 1, replica_start + 4):
            shard = runtime_inputs.inputs[tp_shard]
            np.testing.assert_array_equal(
                shard.lookup_table.to_numpy(), base_lut
            )
            np.testing.assert_array_equal(
                shard.cache_lengths.to_numpy(), base_cache_lengths
            )


@pytest.mark.parametrize("data_parallel_degree", [1, 2, 4])
@pytest.mark.asyncio
async def test_multi_cache_runtime_inputs_match_symbolic_order(
    data_parallel_degree: int,
) -> None:
    """Runtime KV input order must match the graph's symbolic input order."""
    kv_manager = _make_multi_kv_manager(
        num_devices=data_parallel_degree,
        data_parallel_degree=data_parallel_degree,
    )

    batches = []
    for replica_idx in range(data_parallel_degree):
        ctx = create_text_context(np.zeros(1, dtype=np.int64))
        kv_manager.claim(ctx, replica_idx=replica_idx)
        kv_manager.alloc(ctx)
        batches.append([ctx])

    symbolic_types = kv_manager.params.flattened_kv_inputs()
    runtime_buffers = kv_manager.runtime_inputs(batches).flatten()

    assert len(runtime_buffers) == len(symbolic_types), (
        "runtime produced a different number of KV inputs than the graph "
        "declares"
    )
    for i, (typ, buf) in enumerate(
        zip(symbolic_types, runtime_buffers, strict=True)
    ):
        assert typ.dtype == buf.dtype, (
            f"position {i}: runtime dtype {buf.dtype} != symbolic "
            f"{typ.dtype} (KV input order mismatch)"
        )
        # The graph engine validates each statically-known dim positionally
        # (the leading block-count dim is symbolic and is skipped). A wrong
        # cache ordering shows up as a head_dim mismatch here.
        for axis, dim in enumerate(typ.shape):
            try:
                static_dim = int(dim)
            except Exception:
                continue
            assert int(buf.shape[axis]) == static_dim, (
                f"position {i} axis {axis}: runtime {buf.shape[axis]} != "
                f"symbolic {static_dim} (likely MLA/indexer cache order "
                f"mismatch)"
            )


@pytest.mark.asyncio
async def test_alloc_num_speculative_steps_allocates_extra_blocks() -> None:
    """alloc with num_speculative_steps reserves blocks for spec tokens."""
    page_size = 4
    kv_manager = _make_kv_manager(page_size=page_size, total_num_pages=16)
    kv_manager_spec = _make_kv_manager(
        page_size=page_size, total_num_pages=16, num_draft_tokens=4
    )

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    kv_manager.claim(ctx)
    kv_manager_spec.claim(ctx)

    # Without speculative steps: 3 tokens + 1 step - 1 = 3 → 1 block
    kv_manager.alloc(ctx)
    blocks_base = len(
        kv_manager._replica[0].block_manager.req_to_blocks[ctx.request_id]
    )

    kv_manager.release(ctx)

    # With speculative steps: 3 + 0 maybe_accepted + 2*4 spec_steps + 1 - 1 = 11 → 3 blocks
    ctx2 = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    kv_manager_spec.claim(ctx2)
    kv_manager_spec.alloc(ctx2)
    blocks_spec = len(
        kv_manager_spec._replica[0].block_manager.req_to_blocks[ctx2.request_id]
    )

    assert blocks_spec > blocks_base


@pytest.mark.asyncio
async def test_alloc_spec_decoding_empty_draft_tokens_allocates_same_as_dummy() -> (
    None
):
    """Block count must not depend on whether draft_tokens_to_verify is populated."""
    page_size = 4
    num_speculative_tokens = 4
    kv_manager = _make_kv_manager(
        page_size=page_size,
        total_num_pages=16,
        num_draft_tokens=num_speculative_tokens,
    )

    tokens = np.array([1, 2, 3], dtype=np.int64)

    # Case 1: draft_tokens_to_verify is empty.
    ctx_empty = create_text_context(tokens)
    assert ctx_empty.spec_decoding_state.draft_tokens_to_verify == []
    kv_manager.claim(ctx_empty)
    kv_manager.alloc(ctx_empty)
    blocks_empty = len(
        kv_manager._replica[0].block_manager.req_to_blocks[ctx_empty.request_id]
    )
    kv_manager.release(ctx_empty)

    # Case 2: draft_tokens_to_verify is populated with dummy _MAGIC_DRAFT_TOKEN_ID.
    ctx_dummy = create_text_context(tokens)
    _MAGIC_DRAFT_TOKEN_ID = 42
    ctx_dummy.spec_decoding_state.draft_tokens_to_verify = [
        _MAGIC_DRAFT_TOKEN_ID
    ] * num_speculative_tokens
    kv_manager.claim(ctx_dummy)
    kv_manager.alloc(ctx_dummy)
    blocks_dummy = len(
        kv_manager._replica[0].block_manager.req_to_blocks[ctx_dummy.request_id]
    )
    kv_manager.release(ctx_dummy)

    assert blocks_empty == blocks_dummy, (
        f"Empty draft_tokens_to_verify allocated {blocks_empty} blocks but "
        f"dummy-populated draft_tokens_to_verify allocated {blocks_dummy} blocks. "
        "The scheduler must reserve the same capacity regardless of whether "
        "draft_tokens_to_verify has been populated yet."
    )


@pytest.mark.asyncio
async def test_alloc_with_draft_tokens_to_verify_reserves_more_blocks() -> None:
    """alloc with speculative tokens reserves more blocks than without."""
    page_size = 4
    kv_manager = _make_kv_manager(
        page_size=page_size, total_num_pages=16, num_draft_tokens=4
    )

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    ctx.spec_decoding_state.draft_tokens_to_verify = [10, 20, 30]
    kv_manager.claim(ctx)
    # seq_len = 3 tokens + 0 maybe_accepted + 2*4 spec_steps + 1 - 1 = 11 → 3 blocks
    kv_manager.alloc(ctx)
    blocks = len(
        kv_manager._replica[0].block_manager.req_to_blocks[ctx.request_id]
    )
    assert blocks == 3


@pytest.mark.asyncio
async def test_runtime_inputs_with_num_speculative_steps() -> None:
    """runtime_inputs passes through num_speculative_steps without error."""
    page_size = 4
    kv_manager = _make_kv_manager(
        page_size=page_size, total_num_pages=16, num_draft_tokens=4
    )

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    kv_manager.claim(ctx)
    kv_manager.alloc(ctx)

    inputs = kv_manager.runtime_inputs_for_leaf([[ctx]])
    assert len(inputs.inputs) == 1


@pytest.mark.asyncio
async def test_runtime_inputs_cache_length_bounded_by_accepted_not_speculative_length() -> (
    None
):
    """``cache_lengths`` reflects accepted tokens, never the larger
    speculative verify-width -- this is the read boundary that keeps a
    rejected draft token's cache row from ever being read by a later step.
    """
    page_size = 4
    kv_manager = _make_kv_manager(
        page_size=page_size, total_num_pages=64, num_draft_tokens=8
    )

    ctx = create_text_context(np.arange(7, dtype=np.int64))
    kv_manager.claim(ctx)
    kv_manager.alloc(ctx)
    kv_manager.runtime_inputs_for_leaf([[ctx]])
    ctx.update(42)
    kv_manager.step(ctx)
    assert ctx.tokens.processed_length == 7

    ctx.spec_decoding_state.maybe_accepted_draft_tokens = [1, 2, 3]
    # A verify width (6) larger than the accepted count (3) must be
    # irrelevant to cache_lengths -- it only affects block allocation.
    ctx.spec_decoding_state.draft_tokens_to_verify = [9, 9, 9, 9, 9, 9]
    kv_manager.alloc(ctx)

    inputs = kv_manager.runtime_inputs_for_leaf([[ctx]]).inputs[0]
    cache_length = int(inputs.cache_lengths.to_numpy()[0])
    assert cache_length == 7 + 3, (
        f"cache_lengths={cache_length}, expected processed_length(7) + "
        "accepted(3), independent of the larger verify width(6)"
    )


def _make_multi_kv_manager(
    *,
    page_size: int = 128,
    total_num_pages: int = 8,
    max_batch_size: int = 128,
    enable_prefix_caching: bool = False,
    session: InferenceSession | None = None,
    num_devices: int = 1,
    data_parallel_degree: int = 1,
) -> PagedKVCacheManager:
    """Creates a multi-cache manager with two caches (primary + secondary)."""
    devices = [DeviceRef.CPU() for _ in range(num_devices)]
    primary = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=128,
        num_layers=10,
        page_size=page_size,
        devices=devices,
        data_parallel_degree=data_parallel_degree,
        enable_prefix_caching=enable_prefix_caching,
    )
    secondary = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=64,
        num_layers=10,
        page_size=page_size,
        devices=devices,
        data_parallel_degree=data_parallel_degree,
        enable_prefix_caching=enable_prefix_caching,
    )
    multi_params = MultiKVCacheParams.from_params(
        {"primary": primary, "secondary": secondary}
    )
    if session is None:
        session = InferenceSession(devices=[CPU()])
    return PagedKVCacheManager(
        params=multi_params,
        session=session,
        total_num_pages=total_num_pages,
        max_batch_size=max_batch_size,
    )


@pytest.mark.asyncio
async def test_multi_cache_runtime_inputs_combined() -> None:
    """runtime_inputs returns combined inputs for all caches."""
    kv_manager = _make_multi_kv_manager(total_num_pages=16)
    kv_params = kv_manager.params
    assert isinstance(kv_params, MultiKVCacheParams)
    assert len(kv_params.children) == 2

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))
    kv_manager.claim(ctx)
    kv_manager.alloc(ctx)

    inputs = kv_manager.runtime_inputs([[ctx]])

    # With 1 device and 2 caches, the tree has one leaf per cache.
    assert isinstance(inputs, MultiKVCacheInputs)
    leaf0, leaf1 = inputs.children.values()
    assert isinstance(leaf0, KVCacheInputs)
    assert isinstance(leaf1, KVCacheInputs)

    # Both caches share the same cache_lengths and lookup_table buffers.
    assert leaf0.inputs[0].cache_lengths is leaf1.inputs[0].cache_lengths
    assert leaf0.inputs[0].lookup_table is leaf1.inputs[0].lookup_table

    # But they have different block buffers (different caches).
    assert leaf0.inputs[0].kv_blocks is not leaf1.inputs[0].kv_blocks


@pytest.mark.asyncio
async def test_multi_cache_lifecycle() -> None:
    """claim/alloc/step/release work across all caches with one call each."""
    kv_manager = _make_multi_kv_manager(total_num_pages=16)

    ctx = create_text_context(np.array([1, 2, 3], dtype=np.int64))

    # Single claim covers all caches.
    kv_manager.claim(ctx)
    assert kv_manager.contains(ctx)

    # Single alloc covers all caches.
    kv_manager.alloc(ctx)
    kv_manager.runtime_inputs([[ctx]])
    ctx.update(42)

    # Single step covers all caches.
    kv_manager.step(ctx)

    # Single release covers all caches.
    kv_manager.release(ctx)
    assert not kv_manager.contains(ctx)


def test_alloc_dummy_uses_null_block_without_refcount() -> None:
    """Dummy requests map to the null block."""
    kv_manager = _make_kv_manager(total_num_pages=8)
    pool = kv_manager._replica[0].block_manager.device_block_pool
    assert pool.null_block.is_null
    assert pool.null_block.bid == 8
    assert 8 not in pool.free_blocks
    assert pool.num_free_blocks == 8

    dummy_ctx = create_text_context(np.zeros(1, dtype=np.int64))
    kv_manager.alloc_dummy(dummy_ctx)
    assert pool.num_free_blocks == 8
    assert kv_manager.get_req_blocks(dummy_ctx) == [8]


def test_lut_tail_padding_sentinel_is_total_num_pages() -> None:
    """Regression test: LUT tail-padding must be filled with total_num_pages.

    The SIMD populate path in PagedKVCache multiplies every LUT entry by
    page_stride with no sentinel guard. Tail-padding columns (past a request's
    last real block) and dummy-request rows must contain total_num_pages so that
    the multiply produces the null-block address rather than an out-of-bounds
    GPU address (CUDA_ERROR_ILLEGAL_ADDRESS).
    """
    total_num_pages = 16
    page_size = 4
    # Real request spans 3 pages; dummy has 1 null-block entry.
    # LUT row width = padded_lut_cols(3) >> 3, so columns 3.. are tail-padding.
    num_real_pages = 3
    kv_manager = _make_kv_manager(
        total_num_pages=total_num_pages,
        page_size=page_size,
    )

    real_ctx = create_text_context(
        np.zeros(page_size * num_real_pages, dtype=np.int64)
    )
    kv_manager.claim(real_ctx)
    kv_manager.alloc(real_ctx)

    dummy_ctx = create_text_context(np.zeros(1, dtype=np.int64))
    kv_manager.alloc_dummy(dummy_ctx)

    lut = (
        kv_manager.runtime_inputs_for_leaf([[real_ctx, dummy_ctx]])
        .inputs[0]
        .lookup_table.to_numpy()
    )

    # No cell should contain the poison value 0xCCCCCCCC — that value times any
    # realistic page_stride overflows into unmapped GPU memory.
    assert not np.any(lut == 0xCCCCCCCC), (
        "LUT contains 0xCCCCCCCC — SIMD over-reads would compute illegal GPU"
        " addresses; fill must be total_num_pages"
    )

    # Tail-padding on the real request's row (columns past num_real_pages)
    # must be total_num_pages so populate's multiply lands on the null block.
    assert np.all(lut[0, num_real_pages:] == total_num_pages), (
        f"Real-request tail-padding should be {total_num_pages},"
        f" got: {lut[0, num_real_pages:]}"
    )

    # Dummy request: column 0 is the null block (== total_num_pages), and all
    # remaining columns are tail-padding fill — also total_num_pages.
    assert np.all(lut[1, :] == total_num_pages), (
        f"Dummy-request LUT row should be all {total_num_pages},"
        f" got: {lut[1, :]}"
    )
