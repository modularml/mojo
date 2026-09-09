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
"""Tests for mla_indexer_ragged_float8_paged."""

from max.gpu.host import DeviceContext, HostBuffer
from kv_cache.types import (
    KVCacheStaticParams,
    KVCollectionT,
    PagedKVCacheCollection,
)
from nn.attention.gpu.mla_index_fp8 import (
    _SCORES_BUDGET_BYTES,
    mla_indexer_ragged_float8_paged,
)
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
)
from nn.attention.mha_operand import (
    KVCacheMHAOperand,
    KVCacheScalesMHAOperand,
)
from nn.attention.mha_mask import MaskName
from std.math import ceildiv, clamp
from std.random import rand, random_ui64
from std.sys.info import _has_blackwell_tcgen05
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.utils.index import IndexList
from std.testing import assert_almost_equal, assert_true
from std.collections import Set


def _score_paged_sm100[
    num_heads: Int,
    depth: Int,
    KCollectionT: KVCollectionT,
](
    output: TileTensor[.float32, ...],
    q: TileTensor[mut=False, .float8_e4m3fn, ...],
    q_s: TileTensor[mut=False, .float32, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    k_collection: KCollectionT,
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    ctx: DeviceContext,
) raises:
    # Mirrors the production op's scorer call: with `k_collection` a parameter
    # its origins are provably disjoint from the mutable `output`, so the
    # scorer call passes exclusivity checking. An inline call in the test body
    # (local collection, MutAnyOrigin) trips a false aliasing error.
    var k_cache = k_collection.get_key_cache(0)
    var k_op = KVCacheMHAOperand(k_cache)
    var ks_op = KVCacheScalesMHAOperand(k_cache)
    fp8_index_score_sm100[
        DType.float8_e4m3fn,
        type_of(k_op),
        type_of(ks_op),
        num_heads,
        depth,
        _is_cache_length_accurate=False,
        # Load-bearing, not decorative: the alternate N-tile is chosen from this
        # hint, so omitting it scores the DEFAULT tile and the host-reference
        # check below then proves nothing about the tile production runs.
        N_TOKENS_ALT=SPEC_DECODE_N_TOKENS_ALT,
    ](
        output,
        q,
        q_s,
        k_op,
        ks_op,
        input_row_offsets,
        batch_size,
        max_seq_len,
        max_num_keys,
        False,
        ctx,
    )


def test_mla_index_fp8_paged_variable_lengths[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    top_k: Int,
    mask_name: StaticString = MaskName.NULL.name,
    strict_complete: Bool = False,
    check_scores: Bool = False,
    kpool: Int = 1,
](
    seq_lens: List[Int],
    cache_lens: List[Int],
    ctx: DeviceContext,
    metadata_cache_len: Int = 0,
) raises:
    """Test mla_indexer_ragged_float8_paged with variable-length sequences.

    Parameters:
        num_heads: Number of attention heads.
        depth: Head dimension.
        page_size: Page size for paged KV cache.
        top_k: Number of top indices to return.
        mask_name: Mask type name (NULL or CAUSAL).
        strict_complete: When True, additionally assert that every token
            selects its *complete* set of valid keys (exactly `num_keys`
            distinct indices covering `[0, num_keys)`).  Only valid in the
            dense regime where `top_k >= num_keys` for every token, so the
            indexer is expected to return all valid keys (no real sparsity).
            This is the strong invariant that the lenient default check
            (which permits -1 at any position) does NOT enforce; it is what
            catches the topk_gpu out_vals/out_idxs row-stride desync bug,
            where higher query rows collapsed to all -1 (see the regression
            cases in main()).
        check_scores: When True (B200 only, NULL mask), run the SM100 tensor-core
            scorer on the paged KV cache and compare every (token, key) logit
            against a host reference computed over the paged layout. A wrong
            paged TMA row mapping (page_size / LUT) reads the wrong K rows, so
            this catches it -- coverage the index-only checks and
            `test_index_fp8` (page_size == 0) never exercise.
        kpool: Tokens per pooled cache row. With `kpool > 1` the cache rows are
            pooled keys, so the indexer selects pool ids and each token's
            candidate count is its visible-token count floored by `kpool`.

    Args:
        seq_lens: Length of each sequence (new tokens) per batch item.
        cache_lens: Length of cached tokens per batch item.
        ctx: Device context.
        metadata_cache_len: When nonzero, the `max_cache_length` metadata the
            collection reports, in place of the batch's real maximum. Captured
            decode device graphs bake this at capture time, where it can sit
            far above every row's real length; the op must produce identical
            indices while touching only live score slots. Must be at least
            the real maximum. Incompatible with `check_scores`.
    """
    comptime use_causal_mask = mask_name != MaskName.NULL.name
    var batch_size = len(seq_lens)
    assert (
        len(cache_lens) == batch_size
    ), "cache_lens must have same length as seq_lens"

    # Compute totals and max lengths
    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache_len = 0
    for i in range(batch_size):
        total_seq_len += seq_lens[i]
        max_seq_len = max(max_seq_len, seq_lens[i])
        max_cache_len = max(max_cache_len, cache_lens[i])

    print(
        "test_mla_index_fp8_paged_variable_lengths with params:",
        "num_heads:",
        num_heads,
        "depth:",
        depth,
        "page_size:",
        page_size,
        "mask:",
        mask_name,
        "batch_size:",
        batch_size,
        "total_seq_len:",
        total_seq_len,
        "max_seq_len:",
        max_seq_len,
        "max_cache_len:",
        max_cache_len,
        "top_k:",
        top_k,
    )

    assert (
        metadata_cache_len == 0 or metadata_cache_len >= max_cache_len
    ), "metadata_cache_len must be 0 or >= the batch's real maximum"
    comptime if check_scores:
        assert (
            metadata_cache_len == 0
        ), "check_scores assumes the metadata equals the real maximum"

    comptime kv_params = KVCacheStaticParams(
        num_heads=1,  # MLA uses single head for K
        head_size=depth,
        is_mla=True,
    )
    comptime num_layers = 1

    # Calculate number of pages needed (based on max sequence)
    var total_num_keys_max = max_cache_len + max_seq_len
    var pages_per_seq = (total_num_keys_max + page_size - 1) // page_size
    var num_blocks = batch_size * pages_per_seq + 10  # Extra blocks

    # Q tensor: [total_seq_len, num_heads, depth]
    var q_size = total_seq_len * num_heads * depth
    var q_ptr = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    rand(q_ptr.as_span())
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    ctx.enqueue_copy(q_device, q_ptr)

    # Q scales: [total_seq_len, num_heads]
    var qs_size = total_seq_len * num_heads
    var qs_ptr = ctx.enqueue_create_host_buffer[.float32](qs_size)
    rand(qs_ptr.as_span())
    var qs_device = ctx.enqueue_create_buffer[.float32](qs_size)
    ctx.enqueue_copy(qs_device, qs_ptr)

    # Input row offsets: [batch_size + 1] for ragged indexing (variable lengths)
    var input_row_offsets_ptr = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    input_row_offsets_ptr[0] = UInt32(0)
    for i in range(batch_size):
        input_row_offsets_ptr[i + 1] = input_row_offsets_ptr[i] + UInt32(
            seq_lens[i]
        )
    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_ptr)

    # Cache lengths: [batch_size] - variable cached tokens per sequence
    var cache_lengths_ptr = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_ptr[i] = UInt32(cache_lens[i])
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_ptr)

    # K blocks: [num_blocks, 1, num_layers, page_size, num_heads, head_size]
    var k_shape = IndexList[6](
        num_blocks,
        1,  # MLA uses single kv
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime k_block_layout = Layout.row_major[6]()
    var k_block_runtime_layout = RuntimeLayout[k_block_layout].row_major(
        k_shape
    )
    var k_block_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        k_shape.flattened_length()
    )
    with k_block_device.map_to_host() as k_block_host:
        rand(k_block_host.as_span())

    # K scale blocks
    comptime head_dim_granularity = 1
    var ks_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_dim_granularity,
    )
    var ks_block_device = ctx.enqueue_create_buffer[.float32](
        ks_shape.flattened_length()
    )
    with ks_block_device.map_to_host() as ks_block_host:
        rand(ks_block_host.as_span())

    # Page lookup tables
    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_shape = IndexList[2](batch_size, pages_per_seq)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    var k_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )

    var paged_lut_set = Set[Int]()
    with k_lut_device.map_to_host() as k_lut_host:
        for bs in range(batch_size):
            for page_idx in range(pages_per_seq):
                var block_idx = Int(random_ui64(0, UInt64(num_blocks - 1)))
                while block_idx in paged_lut_set:
                    block_idx = Int(random_ui64(0, UInt64(num_blocks - 1)))
                paged_lut_set.add(block_idx)
                k_lut_host[bs * pages_per_seq + page_idx] = UInt32(block_idx)

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](batch_size)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)

    comptime ks_block_layout = Layout.row_major[6]()
    var ks_block_runtime_layout = RuntimeLayout[ks_block_layout].row_major(
        ks_shape
    )
    var k_collection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        kv_params,
        page_size,
        scale_dtype_=DType.float32,
        quantization_granularity_=128,
    ](
        LayoutTensor[.float8_e4m3fn, k_block_layout](
            k_block_device,
            k_block_runtime_layout,
        ),
        LayoutTensor[mut=False, .uint32, cache_lengths_layout](
            cache_lengths_device,
            cache_lengths_runtime_layout,
        ),
        LayoutTensor[mut=False, .uint32, paged_lut_layout](
            k_lut_device,
            paged_lut_runtime_layout,
        ),
        UInt32(max_seq_len),  # max_seq_length (new tokens)
        # max_cache_length (cached tokens), optionally frozen far above the
        # real maximum as a captured decode graph would bake it.
        UInt32(metadata_cache_len if metadata_cache_len > 0 else max_cache_len),
        LayoutTensor[.float32, ks_block_layout](
            ks_block_device,
            ks_block_runtime_layout,
        ),
    )

    # Dense output: [total_seq_len, top_k]
    var total_output_size = total_seq_len * top_k

    var o_ptr = ctx.enqueue_create_host_buffer[.int32](total_output_size)
    var o_device = ctx.enqueue_create_buffer[.int32](total_output_size)

    var q_tile = TileTensor(
        q_device,
        row_major(total_seq_len, num_heads, depth),
    )

    var qs_tile = TileTensor(
        qs_device,
        row_major(total_seq_len, num_heads),
    )

    var input_row_offsets_tile = TileTensor(
        input_row_offsets_device,
        row_major(
            batch_size + 1,
        ),
    )

    var o_tile = TileTensor(
        o_device,
        row_major(total_seq_len, top_k),
    )

    mla_indexer_ragged_float8_paged[
        DType.float8_e4m3fn,
        type_of(k_collection),
        num_heads,
        depth,
        top_k,
        mask_name,
        kpool=kpool,
    ](
        o_tile,
        q_tile,
        qs_tile,
        input_row_offsets_tile,
        k_collection,
        UInt32(0),  # layer_idx
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(o_ptr, o_device)
    ctx.synchronize()

    # Build a mapping from global token index to its valid candidate range.
    # With causal mask: num_keys = cache_len + local_seq_idx + 1
    # Without mask (NULL): num_keys = cache_len + seq_len
    # With kpool > 1 a candidate is a pool of `kpool` tokens, and a pool is only
    # selectable once its last member is visible, so the count is floored.
    var token_to_num_keys = List[Int]()
    for batch_idx in range(batch_size):
        var cache_len = cache_lens[batch_idx]
        var seq_len = seq_lens[batch_idx]

        comptime if use_causal_mask:
            for local_seq_idx in range(seq_len):
                var num_keys = (cache_len + local_seq_idx + 1) // kpool
                token_to_num_keys.append(num_keys)
        else:
            var num_keys = (cache_len + seq_len) // kpool
            for _ in range(seq_len):
                token_to_num_keys.append(num_keys)

    # Verify output:
    # - For k_idx < num_keys: index must be valid [0, num_keys)
    # - For k_idx >= num_keys: index must be -1 (invalid/padded)
    var global_token_idx = 0
    for batch_idx in range(batch_size):
        for _ in range(seq_lens[batch_idx]):
            var num_keys = token_to_num_keys[global_token_idx]
            var valid_count = 0
            for k_idx in range(top_k):
                var output_idx = global_token_idx * top_k + k_idx
                var idx_int = Int(o_ptr[output_idx])

                if idx_int >= 0:
                    valid_count += 1

                if k_idx < num_keys:
                    # Valid position: index should be in range or -1 if masked
                    assert_true(
                        idx_int == -1 or (idx_int >= 0 and idx_int < num_keys),
                        "Invalid index "
                        + String(idx_int)
                        + " at k_idx "
                        + String(k_idx)
                        + " for token "
                        + String(global_token_idx)
                        + " with num_keys "
                        + String(num_keys),
                    )
                else:
                    # Beyond valid range: must be -1
                    assert_true(
                        idx_int == -1,
                        "Expected -1 at k_idx "
                        + String(k_idx)
                        + " >= num_keys "
                        + String(num_keys)
                        + " for token "
                        + String(global_token_idx)
                        + ", got "
                        + String(idx_int),
                    )

            comptime if strict_complete:
                # Dense regime (top_k >= num_keys): the indexer must select
                # ALL num_keys valid keys, never drop or collapse any.  The
                # topk_gpu row-stride desync bug corrupted this for higher
                # query rows (valid_count fell below num_keys, reaching 0 for
                # the last prefill tokens), so this exact-count check is the
                # regression guard.  The default (non-strict) check above
                # permits -1 anywhere and would NOT catch that collapse.
                assert_true(
                    valid_count == num_keys,
                    "Incomplete top-k for token "
                    + String(global_token_idx)
                    + ": selected "
                    + String(valid_count)
                    + " valid keys but expected "
                    + String(num_keys)
                    + " (dense causal set). Indicates dropped/collapsed keys"
                    + " (e.g. topk out_vals/out_idxs row-stride desync).",
                )
            global_token_idx += 1

    comptime if check_scores:
        # tcgen05-only, so B200 only; on H100 this case still ran the scalar
        # fallback + the index checks above.
        comptime if _has_blackwell_tcgen05():
            var sc_size = total_seq_len * total_num_keys_max
            var sc_buf = ctx.enqueue_create_buffer[.float32](sc_size)
            sc_buf.enqueue_fill(-Float32.MAX)
            var sc_tile = TileTensor(
                sc_buf, row_major(total_seq_len, total_num_keys_max)
            )

            _score_paged_sm100[num_heads, depth, type_of(k_collection)](
                sc_tile,
                q_tile.as_immut(),
                qs_tile.as_immut(),
                input_row_offsets_tile.as_immut(),
                k_collection,
                batch_size,
                max_seq_len,
                total_num_keys_max,
                ctx,
            )
            ctx.synchronize()
            var sc_host = ctx.enqueue_create_host_buffer[.float32](sc_size)
            ctx.enqueue_copy(sc_host, sc_buf)
            ctx.synchronize()

            # Host reference over the paged layout: page = key // page_size,
            # offset = key % page_size, block = LUT[batch, page]. A wrong TMA
            # row mapping in the scorer reads different K rows -> mismatch.
            with k_block_device.map_to_host() as k_host:
                with ks_block_device.map_to_host() as ks_host:
                    with k_lut_device.map_to_host() as lut_host:
                        var g = 0
                        for b in range(batch_size):
                            var nk = cache_lens[b] + seq_lens[b]
                            for _ in range(seq_lens[b]):
                                for key in range(nk):
                                    var page = key // page_size
                                    var off = key % page_size
                                    var blk = Int(
                                        lut_host[b * pages_per_seq + page]
                                    )
                                    var kbase = (blk * page_size + off) * depth
                                    var kscale = ks_host[blk * page_size + off]
                                    var score = Float32(0)
                                    for h in range(num_heads):
                                        var dot = Float32(0)
                                        for d in range(depth):
                                            var qd = q_ptr[
                                                (g * num_heads + h) * depth + d
                                            ].cast[.float32]()
                                            var kd = k_host[kbase + d].cast[
                                                DType.float32
                                            ]()
                                            dot += qd * kd
                                        score += (
                                            max(dot, Float32(0))
                                            * qs_ptr[g * num_heads + h]
                                        )
                                    assert_almost_equal(
                                        sc_host[g * total_num_keys_max + key],
                                        score * kscale,
                                        atol=1e-2,
                                        rtol=1e-2,
                                    )
                                g += 1
            _ = sc_buf

    print("  Test passed!")

    # Cleanup
    _ = k_block_device
    _ = k_lut_device
    _ = cache_lengths_device
    _ = ks_block_device
    _ = q_device
    _ = qs_device
    _ = input_row_offsets_device
    _ = o_device


def _assert_same_selection(
    expected: HostBuffer[.int32],
    actual: HostBuffer[.int32],
    total_seq_len: Int,
    top_k: Int,
    changed_by: StaticString,
) raises:
    """Assert two runs of the op selected the same keys for every token.

    Compares the count of valid (non -1) slots and the SET of selected indices.
    Order is deliberately not compared: tie order among equal scores is a
    per-kernel implementation detail (the bitonic sort and the histogram-select
    rank ties differently) and fp8 inputs make exact score ties routine.

    Args:
        expected: Output of the reference run, `[total_seq_len, top_k]` int32.
        actual: Output of the run under test, same shape.
        total_seq_len: Number of token rows.
        top_k: Row stride of both buffers.
        changed_by: What differed between the runs, for the failure message.
    """
    for token in range(total_seq_len):
        var ref_set = Set[Int]()
        var other_set = Set[Int]()
        var ref_valid = 0
        var other_valid = 0
        for k in range(top_k):
            var e = Int(expected[token * top_k + k])
            var a = Int(actual[token * top_k + k])
            if e >= 0:
                ref_valid += 1
                ref_set.add(e)
            if a >= 0:
                other_valid += 1
                other_set.add(a)
        assert_true(
            ref_valid == other_valid,
            String(
                changed_by,
                " changed the valid-slot count at token ",
                token,
                ": ref=",
                ref_valid,
                " other=",
                other_valid,
            ),
        )
        for idx in ref_set:
            assert_true(
                idx in other_set,
                String(
                    changed_by,
                    " changed the selection at token ",
                    token,
                    ": ref selected ",
                    idx,
                    ", the other run did not",
                ),
            )
        for idx in other_set:
            assert_true(
                idx in ref_set,
                String(
                    changed_by,
                    " changed the selection at token ",
                    token,
                    ": the other run selected ",
                    idx,
                    ", ref did not",
                ),
            )


def test_mla_index_frozen_metadata_equivalence[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    top_k: Int,
    mask_name: StaticString,
](
    seq_lens: List[Int],
    cache_lens: List[Int],
    frozen_cache_len: Int,
    ctx: DeviceContext,
) raises:
    """The metadata bound must not change which indices are selected.

    This is the central invariant of the bounded-top-k / no-fill / streaming-
    scorer stack: the `max_cache_length` metadata (which a captured decode
    graph bakes at its capture-time upper bound) may only affect cost, never
    results. Chosen shapes make the two runs take different dispatch routes
    (K-resident scorer + one top-k variant at the runtime-length bound; the
    K-streaming scorer + another top-k variant at the frozen bound), so
    equality also cross-checks the routes' numerics against each other —
    include rows with more valid keys than `top_k` (real sparsity) so a score
    divergence near the selection boundary would change the set.

    Compared per token: the count of valid (non -1) slots, and the SET of
    selected indices. Order is not compared: tie order among equal scores is
    a per-kernel implementation detail (the bitonic sort and the
    histogram-select rank ties differently), and fp8 inputs make exact score
    ties routine.
    """
    var batch_size = len(seq_lens)
    assert len(cache_lens) == batch_size

    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache_len = 0
    for i in range(batch_size):
        total_seq_len += seq_lens[i]
        max_seq_len = max(max_seq_len, seq_lens[i])
        max_cache_len = max(max_cache_len, cache_lens[i])
    assert frozen_cache_len >= max_cache_len

    print(
        "test_mla_index_frozen_metadata_equivalence with params:",
        "num_heads:",
        num_heads,
        "mask:",
        mask_name,
        "batch_size:",
        batch_size,
        "max_cache_len:",
        max_cache_len,
        "frozen_cache_len:",
        frozen_cache_len,
        "top_k:",
        top_k,
    )

    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=depth, is_mla=True
    )
    comptime num_layers = 1

    # One shared LUT, wide enough for the frozen metadata; the reference run only
    # dereferences the real-page prefix. Tail slots point at block 0.
    var real_pages_per_seq = (max_cache_len + max_seq_len + page_size - 1) // (
        page_size
    )
    var lut_pages_per_seq = (
        frozen_cache_len + max_seq_len + page_size - 1
    ) // page_size
    var num_blocks = batch_size * real_pages_per_seq + 1

    var q_size = total_seq_len * num_heads * depth
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    with q_device.map_to_host() as q_host:
        rand(q_host.as_span())

    var qs_size = total_seq_len * num_heads
    var qs_device = ctx.enqueue_create_buffer[.float32](qs_size)
    with qs_device.map_to_host() as qs_host:
        rand(qs_host.as_span())

    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    with input_row_offsets_device.map_to_host() as iro_host:
        iro_host[0] = UInt32(0)
        for i in range(batch_size):
            iro_host[i + 1] = iro_host[i] + UInt32(seq_lens[i])

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for i in range(batch_size):
            cl_host[i] = UInt32(cache_lens[i])

    var k_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime k_block_layout = Layout.row_major[6]()
    var k_block_runtime_layout = RuntimeLayout[k_block_layout].row_major(
        k_shape
    )
    var k_block_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        k_shape.flattened_length()
    )
    with k_block_device.map_to_host() as k_block_host:
        rand(k_block_host.as_span())

    comptime head_dim_granularity = 1
    var ks_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_dim_granularity,
    )
    comptime ks_block_layout = Layout.row_major[6]()
    var ks_block_runtime_layout = RuntimeLayout[ks_block_layout].row_major(
        ks_shape
    )
    var ks_block_device = ctx.enqueue_create_buffer[.float32](
        ks_shape.flattened_length()
    )
    with ks_block_device.map_to_host() as ks_block_host:
        rand(ks_block_host.as_span())

    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_shape = IndexList[2](batch_size, lut_pages_per_seq)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    var k_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    with k_lut_device.map_to_host() as k_lut_host:
        for bs in range(batch_size):
            for page_idx in range(lut_pages_per_seq):
                var block_idx = 0
                if page_idx < real_pages_per_seq:
                    block_idx = 1 + bs * real_pages_per_seq + page_idx
                k_lut_host[bs * lut_pages_per_seq + page_idx] = UInt32(
                    block_idx
                )

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](batch_size)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)

    var total_output_size = total_seq_len * top_k
    var o_ref_device = ctx.enqueue_create_buffer[.int32](total_output_size)
    var o_frozen_device = ctx.enqueue_create_buffer[.int32](total_output_size)

    var q_tile = TileTensor(
        q_device, row_major(total_seq_len, num_heads, depth)
    )
    var qs_tile = TileTensor(qs_device, row_major(total_seq_len, num_heads))
    var input_row_offsets_tile = TileTensor(
        input_row_offsets_device, row_major(batch_size + 1)
    )

    for run in range(2):
        var metadata_cache = max_cache_len if run == 0 else frozen_cache_len
        var k_collection = PagedKVCacheCollection[
            DType.float8_e4m3fn,
            kv_params,
            page_size,
            scale_dtype_=DType.float32,
            quantization_granularity_=128,
        ](
            LayoutTensor[.float8_e4m3fn, k_block_layout](
                k_block_device,
                k_block_runtime_layout,
            ),
            LayoutTensor[mut=False, .uint32, cache_lengths_layout](
                cache_lengths_device,
                cache_lengths_runtime_layout,
            ),
            LayoutTensor[mut=False, .uint32, paged_lut_layout](
                k_lut_device,
                paged_lut_runtime_layout,
            ),
            UInt32(max_seq_len),
            UInt32(metadata_cache),
            LayoutTensor[.float32, ks_block_layout](
                ks_block_device,
                ks_block_runtime_layout,
            ),
        )
        var o_tile = TileTensor(
            o_ref_device if run == 0 else o_frozen_device,
            row_major(total_seq_len, top_k),
        )
        mla_indexer_ragged_float8_paged[
            DType.float8_e4m3fn,
            type_of(k_collection),
            num_heads,
            depth,
            top_k,
            mask_name,
        ](
            o_tile,
            q_tile,
            qs_tile,
            input_row_offsets_tile,
            k_collection,
            UInt32(0),
            ctx,
        )
        ctx.synchronize()

    var o_ref_host = ctx.enqueue_create_host_buffer[.int32](total_output_size)
    var o_frozen_host = ctx.enqueue_create_host_buffer[.int32](
        total_output_size
    )
    ctx.enqueue_copy(o_ref_host, o_ref_device)
    ctx.enqueue_copy(o_frozen_host, o_frozen_device)
    ctx.synchronize()

    _assert_same_selection(
        o_ref_host, o_frozen_host, total_seq_len, top_k, "metadata"
    )

    print("  Test passed!")

    _ = q_device
    _ = qs_device
    _ = input_row_offsets_device
    _ = cache_lengths_device
    _ = k_block_device
    _ = ks_block_device
    _ = k_lut_device
    _ = o_ref_device
    _ = o_frozen_device


def test_mla_index_chunked_equivalence[
    num_heads: Int,
    depth: Int,
    page_size: Int,
    top_k: Int,
    mask_name: StaticString,
    budget_bytes: Int,
](seq_lens: List[Int], cache_lens: List[Int], ctx: DeviceContext,) raises:
    """Chunking the score matrix must not change which indices are selected.

    The op materializes `total_seq_len x max_num_keys` f32 scores, which is what
    caps `--max-batch-input-tokens`; past a byte budget it scores one row window
    at a time into a single reused buffer. That is a cost change only, so the
    unchunked run IS the oracle here: same inputs, same dispatch route (routing
    reads `max_seq_len`, which the window deliberately does not touch), and every
    row keeps the scores and live-key bound it had.

    Compared per token: the count of valid (non -1) slots and the SET of selected
    indices. Order is not compared, for the reason
    `test_mla_index_frozen_metadata_equivalence` gives.

    The chunk count is ASSERTED, not assumed. `budget_bytes` interacts with
    `max_num_keys`, so a budget that happens to admit every row would leave this
    test passing while exercising nothing -- the failure mode a chunking test is
    least able to notice.
    """
    # Same trap from the other side, and the one that actually bit: only the
    # SM100 scorers take a row window, so a shape that routes to the scalar
    # fallback runs as one chunk however small the budget is, and every
    # assertion below then compares an unchunked run against itself. `page_size`
    # is the clause that is easy to get wrong -- 64 is a legal paged size the
    # rest of this file uses everywhere, and it fails `% _BM_KEY`. Mirrors
    # `use_sm100_scorer` in `mla_index_fp8.mojo`.
    comptime assert (
        depth == 128
        and num_heads in (64, 32, 8, 4)
        and (page_size == 0 or page_size % _BM_KEY == 0)
    ), (
        "chunking is SM100-only: this shape routes to the scalar scorer, which"
        " takes no row window, so the comparison would be vacuous"
    )

    var batch_size = len(seq_lens)
    assert len(cache_lens) == batch_size

    var total_seq_len = 0
    var max_seq_len = 0
    var max_cache_len = 0
    for i in range(batch_size):
        total_seq_len += seq_lens[i]
        max_seq_len = max(max_seq_len, seq_lens[i])
        max_cache_len = max(max_cache_len, cache_lens[i])

    # Mirrors the launcher's own sizing, so the assert below sees what it sees.
    var max_num_keys = max_cache_len + max_seq_len
    var rows_per_chunk = clamp(
        budget_bytes // (max_num_keys * 4), 1, total_seq_len
    )
    var num_chunks = ceildiv(total_seq_len, rows_per_chunk)

    print(
        "test_mla_index_chunked_equivalence with params:",
        "num_heads:",
        num_heads,
        "mask:",
        mask_name,
        "batch_size:",
        batch_size,
        "total_seq_len:",
        total_seq_len,
        "max_seq_len:",
        max_seq_len,
        "max_cache_len:",
        max_cache_len,
        "top_k:",
        top_k,
        "rows_per_chunk:",
        rows_per_chunk,
        "chunks:",
        num_chunks,
    )

    assert_true(
        num_chunks >= 2,
        String(
            "budget_bytes ",
            budget_bytes,
            " admits all ",
            total_seq_len,
            " rows at max_num_keys ",
            max_num_keys,
            ": this case would compare an unchunked run against itself",
        ),
    )

    comptime kv_params = KVCacheStaticParams(
        num_heads=1, head_size=depth, is_mla=True
    )
    comptime num_layers = 1

    var pages_per_seq = (max_cache_len + max_seq_len + page_size - 1) // (
        page_size
    )
    var num_blocks = batch_size * pages_per_seq + 1

    var q_size = total_seq_len * num_heads * depth
    var q_device = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    with q_device.map_to_host() as q_host:
        rand(q_host.as_span())

    var qs_size = total_seq_len * num_heads
    var qs_device = ctx.enqueue_create_buffer[.float32](qs_size)
    with qs_device.map_to_host() as qs_host:
        rand(qs_host.as_span())

    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    with input_row_offsets_device.map_to_host() as iro_host:
        iro_host[0] = UInt32(0)
        for i in range(batch_size):
            iro_host[i + 1] = iro_host[i] + UInt32(seq_lens[i])

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    with cache_lengths_device.map_to_host() as cl_host:
        for i in range(batch_size):
            cl_host[i] = UInt32(cache_lens[i])

    var k_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime k_block_layout = Layout.row_major[6]()
    var k_block_runtime_layout = RuntimeLayout[k_block_layout].row_major(
        k_shape
    )
    var k_block_device = ctx.enqueue_create_buffer[.float8_e4m3fn](
        k_shape.flattened_length()
    )
    with k_block_device.map_to_host() as k_block_host:
        rand(k_block_host.as_span())

    comptime head_dim_granularity = 1
    var ks_shape = IndexList[6](
        num_blocks,
        1,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_dim_granularity,
    )
    comptime ks_block_layout = Layout.row_major[6]()
    var ks_block_runtime_layout = RuntimeLayout[ks_block_layout].row_major(
        ks_shape
    )
    var ks_block_device = ctx.enqueue_create_buffer[.float32](
        ks_shape.flattened_length()
    )
    with ks_block_device.map_to_host() as ks_block_host:
        rand(ks_block_host.as_span())

    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_shape = IndexList[2](batch_size, pages_per_seq)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    var k_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    with k_lut_device.map_to_host() as k_lut_host:
        for bs in range(batch_size):
            for page_idx in range(pages_per_seq):
                k_lut_host[bs * pages_per_seq + page_idx] = UInt32(
                    1 + bs * pages_per_seq + page_idx
                )

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](batch_size)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)

    var total_output_size = total_seq_len * top_k
    var o_ref_device = ctx.enqueue_create_buffer[.int32](total_output_size)
    var o_chunked_device = ctx.enqueue_create_buffer[.int32](total_output_size)

    var q_tile = TileTensor(
        q_device, row_major(total_seq_len, num_heads, depth)
    )
    var qs_tile = TileTensor(qs_device, row_major(total_seq_len, num_heads))
    var input_row_offsets_tile = TileTensor(
        input_row_offsets_device, row_major(batch_size + 1)
    )

    var k_collection = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        kv_params,
        page_size,
        scale_dtype_=DType.float32,
        quantization_granularity_=128,
    ](
        LayoutTensor[.float8_e4m3fn, k_block_layout](
            k_block_device,
            k_block_runtime_layout,
        ),
        LayoutTensor[mut=False, .uint32, cache_lengths_layout](
            cache_lengths_device,
            cache_lengths_runtime_layout,
        ),
        LayoutTensor[mut=False, .uint32, paged_lut_layout](
            k_lut_device,
            paged_lut_runtime_layout,
        ),
        UInt32(max_seq_len),
        UInt32(max_cache_len),
        LayoutTensor[.float32, ks_block_layout](
            ks_block_device,
            ks_block_runtime_layout,
        ),
    )

    var o_ref_tile = TileTensor(o_ref_device, row_major(total_seq_len, top_k))
    mla_indexer_ragged_float8_paged[
        DType.float8_e4m3fn,
        type_of(k_collection),
        num_heads,
        depth,
        top_k,
        mask_name,
    ](
        o_ref_tile,
        q_tile,
        qs_tile,
        input_row_offsets_tile,
        k_collection,
        UInt32(0),
        ctx,
    )
    ctx.synchronize()

    var o_chunked_tile = TileTensor(
        o_chunked_device, row_major(total_seq_len, top_k)
    )
    mla_indexer_ragged_float8_paged[
        DType.float8_e4m3fn,
        type_of(k_collection),
        num_heads,
        depth,
        top_k,
        mask_name,
        budget_bytes,
    ](
        o_chunked_tile,
        q_tile,
        qs_tile,
        input_row_offsets_tile,
        k_collection,
        UInt32(0),
        ctx,
    )
    ctx.synchronize()

    var o_ref_host = ctx.enqueue_create_host_buffer[.int32](total_output_size)
    var o_chunked_host = ctx.enqueue_create_host_buffer[.int32](
        total_output_size
    )
    ctx.enqueue_copy(o_ref_host, o_ref_device)
    ctx.enqueue_copy(o_chunked_host, o_chunked_device)
    ctx.synchronize()

    _assert_same_selection(
        o_ref_host, o_chunked_host, total_seq_len, top_k, "chunking"
    )

    print("  Test passed!")

    _ = q_device
    _ = qs_device
    _ = input_row_offsets_device
    _ = cache_lengths_device
    _ = k_block_device
    _ = ks_block_device
    _ = k_lut_device
    _ = o_ref_device
    _ = o_chunked_device


def main() raises:
    with DeviceContext() as ctx:
        print("Testing mla_indexer_ragged_float8_paged...")

        # ===== Tests with NULL mask (no causal masking) =====
        print("\n--- NULL mask tests ---")

        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=16,
            mask_name=MaskName.NULL.name,
        ](
            seq_lens=[16, 32, 8, 64],
            cache_lens=[64, 128, 32, 96],
            ctx=ctx,
        )

        # Test with very short sequences (edge case: some num_keys < top_k)
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=64,
            page_size=32,
            top_k=32,
            mask_name=MaskName.NULL.name,
        ](
            seq_lens=[4, 8, 2],
            cache_lens=[4, 8, 2],
            ctx=ctx,
        )

        # ===== GLM indexer geometry (num_heads=64, depth=128): routes through
        # the SM100 tensor-core scorer (fp8_index_score_sm100) =====
        print("\n--- SM100 tensor-core scorer (num_heads=64, depth=128) ---")

        # Dense NULL + strict_complete: the full valid set must be selected, so
        # this asserts the tensor-core scores rank correctly vs the scalar ref.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=128,
            top_k=256,
            mask_name=MaskName.NULL.name,
            strict_complete=True,
        ](
            seq_lens=[6, 4, 2, 1],
            cache_lens=[64, 128, 32, 96],
            ctx=ctx,
        )

        # CAUSAL MTP decode: SM100 scorer + the separate causal mask launch.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=128,
            top_k=64,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[6, 1, 4, 1],
            cache_lens=[128, 64, 200, 50],
            ctx=ctx,
        )

        # strict_complete guard on the grid.z-split + causal path: max_seq_len=6
        # keeps out of the prefill gate (ceildiv(6, 2) = 3 < 16) and base_ctas=8
        # < sm_count forces num_slices=2 (split kernel), while top_k=256 covers
        # every token's causal key set (max 204) so the full set must be
        # selected. strict_complete on split otherwise only runs under NULL, and
        # on causal only via the prefill kernel.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=128,
            top_k=256,
            mask_name=MaskName.CAUSAL.name,
            strict_complete=True,
        ](
            seq_lens=[6, 1, 4, 1],
            cache_lens=[128, 64, 200, 50],
            ctx=ctx,
        )

        # page_size == BM_key exactly (one K tile per page): must stay on the
        # SM100 tensor-core path.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=128,
            top_k=256,
            mask_name=MaskName.NULL.name,
            strict_complete=True,
        ](
            seq_lens=[6, 4, 2, 1],
            cache_lens=[192, 128, 200, 96],
            ctx=ctx,
        )

        # Paged score check (B200 only): the SM100 scorer's TMA row mapping is
        # compared logit-by-logit against a host reference, for both a
        # single-tile page (128 == BM_key) and a multi-tile page (256).  Both
        # carry caches deep enough to span several pages, so a wrong
        # `key // page_size` -> LUT step cannot pass.  On H100 these run the
        # scalar fallback + index checks only.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=128,
            top_k=64,
            mask_name=MaskName.NULL.name,
            check_scores=True,
        ](
            seq_lens=[4, 2],
            cache_lens=[300, 160],
            ctx=ctx,
        )

        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=256,
            top_k=64,
            mask_name=MaskName.NULL.name,
            check_scores=True,
        ](
            seq_lens=[3, 2],
            cache_lens=[600, 300],
            ctx=ctx,
        )

        # page_size=32 (not a multiple of BM_key): the dispatch guard must
        # fall back to the scalar kernel, which must still rank correctly.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=32,
            top_k=256,
            mask_name=MaskName.NULL.name,
            strict_complete=True,
        ](
            seq_lens=[6, 4, 2, 1],
            cache_lens=[64, 96, 32, 128],
            ctx=ctx,
        )

        # ===== TP-head-sharded indexer geometry (num_heads=8, depth=128):
        # SM100 scorer with N_TOKENS = 16 tokens per tile. Sharded head
        # counts (< 16) exist only on the tensor-core path — the scalar
        # fallback's [16, 8] copier thread layout silently stages nothing
        # below 16 heads — so these are compile-gated to Blackwell. =====
        comptime if _has_blackwell_tcgen05():
            print("\n--- SM100 tensor-core scorer (num_heads=8, depth=128) ---")

            # Dense NULL + strict_complete across the 16-token tile boundary
            # (seq_len 17 -> two tiles with a 1-token partial; 16 -> one).
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=8,
                depth=128,
                page_size=128,
                top_k=256,
                mask_name=MaskName.NULL.name,
                strict_complete=True,
            ](
                seq_lens=[17, 16, 6, 1],
                cache_lens=[64, 128, 32, 96],
                ctx=ctx,
            )

            # CAUSAL MTP decode at the sharded-head count.
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=8,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.CAUSAL.name,
            ](
                seq_lens=[6, 1, 4, 1],
                cache_lens=[128, 64, 200, 50],
                ctx=ctx,
            )

            # Paged score check: logit-by-logit vs the host reference,
            # exercising both Q buffers (seq_len 18 -> 2 tiles) at N_TOKENS=16.
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=8,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.NULL.name,
                check_scores=True,
            ](
                seq_lens=[18, 2],
                cache_lens=[300, 160],
                ctx=ctx,
            )

        # ===== GLM 5.x replicated indexer geometry (num_heads=32,
        # depth=128): SM100 scorer with N_TOKENS = 4 tokens per tile =====
        print("\n--- SM100 tensor-core scorer (num_heads=32, depth=128) ---")

        test_mla_index_fp8_paged_variable_lengths[
            num_heads=32,
            depth=128,
            page_size=128,
            top_k=256,
            mask_name=MaskName.NULL.name,
            strict_complete=True,
        ](
            seq_lens=[5, 4, 2, 1],
            cache_lens=[64, 128, 32, 96],
            ctx=ctx,
        )

        test_mla_index_fp8_paged_variable_lengths[
            num_heads=32,
            depth=128,
            page_size=128,
            top_k=64,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[6, 1, 4, 1],
            cache_lens=[128, 64, 200, 50],
            ctx=ctx,
        )

        # Score check across the 4-token tile boundary (5 -> 2 tiles).
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=32,
            depth=128,
            page_size=128,
            top_k=64,
            mask_name=MaskName.NULL.name,
            check_scores=True,
        ](
            seq_lens=[5, 2],
            cache_lens=[300, 160],
            ctx=ctx,
        )

        # ===== Key-split route: few token blocks over a deep cache, so
        # `_KEYSPLIT_MAX_TOKEN_TILES`/`_KEYSPLIT_MIN_KEY_TILES` send these to the
        # K-streaming kernel with grid.z splitting the key range. This is the
        # decode/MTP geometry, and it is the ONLY value-level coverage of a
        # split key window -- the two long-prefill cases below check top-k
        # indices only. Each case targets a different window shape; the counts
        # assume B200 (sm_count=148) and `_ctas_per_sm() == 2`. =====

        # 94 key tiles over 74 parts: ~1 tile per CTA, so the load warp runs its
        # prologue only and never reaches the k_empty refill loop.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=32,
            depth=128,
            page_size=128,
            top_k=64,
            mask_name=MaskName.NULL.name,
            check_scores=True,
        ](
            seq_lens=[6, 4],
            cache_lens=[12000, 12000],
            ctx=ctx,
        )

        # Same split, ragged cache: entry 1 has only 2 key tiles against 74
        # parts, so its trailing parts get empty windows and must take the
        # `n_tiles_local <= 0` bail while entry 0 runs a full window alongside.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=32,
            depth=128,
            page_size=128,
            top_k=64,
            mask_name=MaskName.NULL.name,
            check_scores=True,
        ](
            seq_lens=[6, 4],
            cache_lens=[12000, 200],
            ctx=ctx,
        )

        # Ragged, and TILE-ALIGNED on purpose. `_MIN_TILES_PER_PART` narrows the
        # launcher's 74 parts per entry: entry 0 (94 tiles) to 24, entry 1 (1
        # tile) to 1, so 73 of its 74 grid.z CTAs must take the new
        # `block_idx.z >= p_eff` bail. Both entries have `cache_len + seq_len` an
        # exact multiple of BM_key (12032 = 94*128, 128 = 1*128), so a window
        # that drops or double-counts a tile moves 128 whole key columns instead
        # of hiding in a partial tail that the score tolerance would absorb.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=32,
            depth=128,
            page_size=128,
            top_k=64,
            mask_name=MaskName.NULL.name,
            check_scores=True,
        ](
            seq_lens=[6, 4],
            cache_lens=[12026, 124],
            ctx=ctx,
        )

        # 260 key tiles over 37 parts: ~7 tiles per CTA, which is past the
        # `_k_ring_stages` prologue, so the refill loop issues K TMAs at a
        # NON-ZERO tile offset. Windowing the load warp's two loops
        # inconsistently with the MMA/consumer trip counts hangs rather than
        # returning wrong values, so this case is the deadlock net.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=32,
            depth=128,
            page_size=128,
            top_k=64,
            mask_name=MaskName.NULL.name,
            check_scores=True,
        ](
            seq_lens=[6, 4, 6, 2],
            cache_lens=[33200, 33200, 33200, 33200],
            ctx=ctx,
        )

        # Causal over a split key range. `check_scores` cannot cover this: it
        # scores through `_score_paged_sm100`, which passes causal=False.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=32,
            depth=128,
            page_size=128,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[6, 1, 4, 1],
            cache_lens=[12000, 12000, 12000, 12000],
            ctx=ctx,
        )

        # Long nh=32 pure-prefill routes to the K-streaming prefill kernel:
        # seq=1792 (ceildiv(1792, 4) = 448 tiles >= _PREFILL_MIN_TOKEN_TILES_NH32)
        # with causal + cache=0 clears the prefill gate. strict_complete asserts
        # the prefill kernel selects every token's full causal key set.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=32,
            depth=128,
            page_size=128,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
            strict_complete=True,
        ](
            seq_lens=[1792],
            cache_lens=[0],
            ctx=ctx,
        )

        # ===== GLM 32 heads sharded over 8 ranks (num_heads=4, depth=128):
        # N_TOKENS = 32. The scalar fallback tiles heads by 8, so this count
        # only compiles where the SM100 tensor-core path is taken. =====
        comptime if _has_blackwell_tcgen05():
            print("\n--- SM100 tensor-core scorer (num_heads=4, depth=128) ---")

            test_mla_index_fp8_paged_variable_lengths[
                num_heads=4,
                depth=128,
                page_size=128,
                top_k=256,
                mask_name=MaskName.NULL.name,
                strict_complete=True,
            ](
                seq_lens=[33, 32, 6, 1],
                cache_lens=[64, 128, 32, 96],
                ctx=ctx,
            )

            test_mla_index_fp8_paged_variable_lengths[
                num_heads=4,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.CAUSAL.name,
            ](
                seq_lens=[6, 1, 4, 1],
                cache_lens=[128, 64, 200, 50],
                ctx=ctx,
            )

            # Score check across the 32-token tile boundary (34 -> 2 tiles).
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=4,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.NULL.name,
                check_scores=True,
            ](
                seq_lens=[34, 2],
                cache_lens=[300, 160],
                ctx=ctx,
            )

        # ===== Tests with CAUSAL mask =====
        print("\n--- CAUSAL mask tests ---")

        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=16,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[16, 32, 8, 64],
            cache_lens=[64, 128, 32, 96],
            ctx=ctx,
        )

        # Test with mixed prefill/decode (some seq_len=1, some larger)
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=16,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[1, 1, 32, 1],  # Mix of decode (1) and prefill
            cache_lens=[100, 50, 0, 200],  # Varied cache sizes
            ctx=ctx,
        )

        # Test causal mask with very short sequences
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=64,
            page_size=32,
            top_k=32,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[4, 8, 2],
            cache_lens=[4, 8, 2],
            ctx=ctx,
        )

        # ===== Regression: large top_k (2048) + long context =====
        # These cover two bugs that only appear at production scale:
        #   (A) topk_gpu stage-2 dynamic shared memory exceeded the device
        #       per-block limit once max_k = min(top_k, ctx) reached ~2000,
        #       crashing the launch with CUDA_ERROR_INVALID_VALUE.
        #   (B) fill_invalid_topk_kernel only covered the first 1024 output
        #       columns, leaving columns [1024, top_k) as garbage when
        #       top_k > 1024.
        # Each case mixes a long sequence (drives max_num_keys past the old
        # smem cliff -> exercises A) with a short sequence whose token needs
        # -1 padding spanning columns >1024 (-> exercises B).
        print("\n--- regression: top_k=2048, long context ---")

        # Decode, causal: long seq (cache 2100) + short seq (cache 50, so its
        # token needs -1 across columns [51, 2048), including the >1024 range).
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[1, 1, 1, 1],
            cache_lens=[2100, 1990, 1500, 50],
            ctx=ctx,
        )

        # Decode, NULL mask: long + short seq.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.NULL.name,
        ](
            seq_lens=[1, 1],
            cache_lens=[2100, 100],
            ctx=ctx,
        )

        # Prefill, causal: 200 new tokens over a 1900-token cache
        # (max_num_keys=2100, past the old cliff; early tokens need -1 padding).
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[200],
            cache_lens=[1900],
            ctx=ctx,
        )

        # Long-context (16000-token cache) exercises the N > 2048
        # streaming top-k path end-to-end.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=128,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[1],
            cache_lens=[16000],
            ctx=ctx,
        )

        # ===== Regression: topk out_vals/out_idxs row-stride desync =====
        # GLM-5.1 / DSv3.2 prefix-cached prefill: a multi-token chunk on top of
        # a cached prefix where max_num_keys < top_k, so effective_k =
        # min(top_k, max_num_keys) < top_k.  topk_gpu indexes both of its
        # outputs by effective_k, so out_vals (effective_k stride) and out_idxs
        # MUST share that stride; aliasing out_idxs onto the top_k-strided
        # output desynced them, scattering each query row's indices r*(top_k -
        # effective_k) elements off -> higher rows collapsed to all -1.  Row 0
        # always looked fine (offset 0), so the lenient check missed it; the
        # earlier top_k=2048 cases had max_num_keys>=2048 (effective_k==top_k)
        # so they never triggered it.  These cases force max_num_keys < top_k
        # AND use strict_complete to require every token's full causal set.
        print("\n--- regression: topk stride desync (max_num_keys < top_k) ---")

        # GLM geometry: 64 heads, depth 128, top_k=2048, ~900-token cached
        # prefix + 179 fresh tokens => max_num_keys=1075 < 2048.  Last fresh
        # token must still select all 1075 causal keys.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=128,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
            strict_complete=True,
        ](
            seq_lens=[179],
            cache_lens=[896],
            ctx=ctx,
        )

        # Multi-batch mix of cached prefixes + multi-token chunks, all with
        # max_num_keys < top_k.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=128,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
            strict_complete=True,
        ](
            seq_lens=[64, 200, 32],
            cache_lens=[300, 500, 100],
            ctx=ctx,
        )

        # ===== Capture-frozen metadata (max_cache_length >> real lengths) ====
        # Captured decode graphs bake `max_cache_length` at capture time, far
        # above every row's real length at replay; the op must produce the
        # same indices while touching only live score slots.
        print("\n--- capture-frozen metadata tests ---")

        # Decode/MTP shape, causal, strict: every token must still select its
        # complete causal key set with the stride frozen at 65536. Covers both
        # bounded-histsel select rows (num_keys > 2048 impossible here — all
        # <= 2048, so every row takes the select-all path) and the -1 tails.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
            strict_complete=True,
        ](
            seq_lens=[6, 1, 4, 1],
            cache_lens=[500, 2000, 128, 64],
            ctx=ctx,
            metadata_cache_len=65536,
        )

        # Same shape under NULL mask (bounds = cache + seq for every row).
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.NULL.name,
            strict_complete=True,
        ](
            seq_lens=[6, 1, 4, 1],
            cache_lens=[500, 2000, 128, 64],
            ctx=ctx,
            metadata_cache_len=65536,
        )

        # Rows above and below K with a frozen stride: long rows exercise the
        # bounded threshold rounds, the 50-token row the select-all + -1 tail.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[1, 1, 1, 1],
            cache_lens=[4000, 2500, 1500, 50],
            ctx=ctx,
            metadata_cache_len=65536,
        )

        # Small frozen stride (4102) lands in the register-resident histsel;
        # top_k=256 keeps strict_complete applicable.
        test_mla_index_fp8_paged_variable_lengths[
            num_heads=64,
            depth=128,
            page_size=64,
            top_k=256,
            mask_name=MaskName.CAUSAL.name,
            strict_complete=True,
        ](
            seq_lens=[6, 4, 2, 1],
            cache_lens=[64, 128, 32, 96],
            ctx=ctx,
            metadata_cache_len=4096,
        )

        # GLM 5.2 TP8 decode geometry (4 local heads) with the frozen stride;
        # Blackwell-only, as the scalar fallback does not compile at 4 heads.
        comptime if _has_blackwell_tcgen05():
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=4,
                depth=128,
                page_size=128,
                top_k=2048,
                mask_name=MaskName.CAUSAL.name,
                strict_complete=True,
            ](
                seq_lens=[6, 6, 6, 1],
                cache_lens=[500, 2000, 128, 64],
                ctx=ctx,
                metadata_cache_len=65536,
            )

            # Deep rows route the scorer through the
            # decode-streaming (key-split) kernel; check its logits against
            # the host reference — the small-shape check_scores cases above
            # all stay on the K-resident path.
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=4,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.NULL.name,
                check_scores=True,
            ](
                seq_lens=[1, 1, 1, 1, 1, 1, 1, 1],
                cache_lens=[
                    32800,
                    32800,
                    32800,
                    32800,
                    32800,
                    32800,
                    32800,
                    32800,
                ],
                ctx=ctx,
            )

        # ===== Metadata must not change results =====
        # Rationale and comparison semantics live on
        # `test_mla_index_frozen_metadata_equivalence`.
        print("\n--- frozen-metadata equivalence tests ---")

        test_mla_index_frozen_metadata_equivalence[
            num_heads=64,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[6, 1, 4, 1],
            cache_lens=[4000, 2500, 1500, 50],
            frozen_cache_len=65536,
            ctx=ctx,
        )

        test_mla_index_frozen_metadata_equivalence[
            num_heads=64,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.NULL.name,
        ](
            seq_lens=[6, 1, 4, 1],
            cache_lens=[4000, 2500, 1500, 50],
            frozen_cache_len=65536,
            ctx=ctx,
        )

        # Frozen stride in the register-resident histsel range (N <= 8192).
        test_mla_index_frozen_metadata_equivalence[
            num_heads=64,
            depth=128,
            page_size=64,
            top_k=2048,
            mask_name=MaskName.CAUSAL.name,
        ](
            seq_lens=[6, 1, 4, 1],
            cache_lens=[4000, 2500, 1500, 50],
            frozen_cache_len=8000,
            ctx=ctx,
        )

        # ===== Chunked score matrix (SM100 only -- the scalar fallback and
        # its mask pass do not take a row window, so there the op is always one
        # chunk and these cases would compare a run against itself) =====
        comptime if _has_blackwell_tcgen05():
            print("\n--- chunked score-matrix equivalence tests ---")

            # K-streaming prefill route (token_tiles = 20 >= 16), chunk
            # boundaries falling mid-request. top_k below the key count so the
            # selection is genuinely sparse and a score divergence would move
            # it.
            test_mla_index_chunked_equivalence[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=128,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=23040,
            ](
                seq_lens=[40, 33, 24, 17],
                cache_lens=[128, 192, 256, 320],
                ctx=ctx,
            )

            test_mla_index_chunked_equivalence[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=128,
                mask_name=MaskName.NULL.name,
                budget_bytes=23040,
            ](
                seq_lens=[40, 33, 24, 17],
                cache_lens=[128, 192, 256, 320],
                ctx=ctx,
            )

            # Key-split prefill route: few token blocks (3 <= 4) against a cache
            # deep enough to open the split (65 key tiles >= 64).
            test_mla_index_chunked_equivalence[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=2048,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=131168,
            ](
                seq_lens=[6, 5, 4, 3],
                cache_lens=[8192, 8192, 8192, 8192],
                ctx=ctx,
            )

            # K-resident route: 7 token tiles is above the key-split ceiling and
            # below the prefill floor, and the cache is too shallow to split.
            test_mla_index_chunked_equivalence[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=5380,
            ](
                seq_lens=[13, 11, 9, 7],
                cache_lens=[128, 256, 128, 256],
                ctx=ctx,
            )

            # Degenerate window: one row per chunk, so every chunk starts inside
            # a request and no token block is ever full.
            test_mla_index_chunked_equivalence[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=16,
            ](
                seq_lens=[13, 11, 9, 7],
                cache_lens=[128, 256, 128, 256],
                ctx=ctx,
            )

            # A single request whose rows alone exceed the budget -- the case a
            # request-grouped chunker could not split at all.
            test_mla_index_chunked_equivalence[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=32,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=6144,
            ](
                seq_lens=[64],
                cache_lens=[128],
                ctx=ctx,
            )

            # ---- Production scale. Everything above is a toy shape whose
            # chunking is forced by an artificially small budget; these are the
            # shapes the ticket is about, and the first one runs at the SHIPPED
            # default so the path production takes is the path under test.
            #
            # They also cross a structural boundary the toy cases cannot: at
            # small sizes `max_seq_len` is the binding term in
            # `min(max_seq_len, chunk_rows)` (which sizes grid.y), while here
            # `chunk_rows` binds -- the other side of that branch.

            # 2048 prefill tokens over KERN-3467's context distribution, at the
            # real 512 MB budget: 2 chunks of ~1754 rows.
            test_mla_index_chunked_equivalence[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=2048,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=_SCORES_BUDGET_BYTES,
            ](
                seq_lens=[512, 512, 512, 512],
                cache_lens=[76000, 60000, 45000, 30000],
                ctx=ctx,
            )

            # One long sequence: every chunk boundary falls deep inside a single
            # request, at a scale where a chunk spans hundreds of token blocks.
            test_mla_index_chunked_equivalence[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=2048,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=64 * 1024 * 1024,
            ](
                seq_lens=[2048],
                cache_lens=[32768],
                ctx=ctx,
            )

            # nh=32 PREFILL route, unreachable at toy sizes: its gate wants
            # `token_tiles >= 448` (seq >= 1792) plus causal and no cached
            # prefix, so this is the only case that chunks it.
            test_mla_index_chunked_equivalence[
                num_heads=32,
                depth=128,
                page_size=128,
                top_k=2048,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=4 * 1024 * 1024,
            ](
                seq_lens=[2048],
                cache_lens=[0],
                ctx=ctx,
            )

            # 2048 requests of one token: the ragged metadata at batch scale,
            # where `fill_invalid`/`row_bounds` walk 2048 entries per token.
            var many_lens = List[Int]()
            var many_cache = List[Int]()
            for i in range(2048):
                many_lens.append(1)
                many_cache.append(512 + (i % 4) * 512)
            test_mla_index_chunked_equivalence[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=2048,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=8 * 1024 * 1024,
            ](
                seq_lens=many_lens,
                cache_lens=many_cache,
                ctx=ctx,
            )

            # Head-sharded indexer: N_TOKENS is 32 here, so a 4-row window is a
            # fraction of one token block.
            test_mla_index_chunked_equivalence[
                num_heads=4,
                depth=128,
                page_size=128,
                top_k=1024,
                mask_name=MaskName.CAUSAL.name,
                budget_bytes=32096,
            ](
                seq_lens=[6, 5, 4, 3],
                cache_lens=[2000, 2000, 2000, 2000],
                ctx=ctx,
            )

        # ===== Pooled candidates (kpool > 1) =====
        # A cache row is one pooled key per `kpool` tokens, so each token's
        # candidate count is its visible-token count floored by `kpool`.
        # kpool > 1 requires the SM100 tensor-core scorer, so B200 only.
        comptime if _has_blackwell_tcgen05():
            print("\n--- pooled candidates (kpool=4) ---")

            # GLM indexer geometry. `nh=32` routes to the warp-specialized
            # prefill scorer. Every pool count here is under `top_k`, so each
            # token must select its whole pool set.
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=32,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.NULL.name,
                strict_complete=True,
                kpool=4,
            ](
                seq_lens=[6, 4, 2, 1],
                cache_lens=[64, 128, 32, 96],
                ctx=ctx,
            )

            # Same geometry under causality. The bound moves per token, so a
            # token whose visible count is not a multiple of 4 must not see the
            # pool its tail sits in.
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=32,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.CAUSAL.name,
                strict_complete=True,
                kpool=4,
            ](
                seq_lens=[7, 5, 3, 1],
                cache_lens=[64, 130, 33, 97],
                ctx=ctx,
            )

            # `nh=64` takes the resident-key scorer, which is the other
            # epilogue carrying the pooled store guard.
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=64,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.CAUSAL.name,
                strict_complete=True,
                kpool=4,
            ](
                seq_lens=[6, 1, 4, 1],
                cache_lens=[128, 64, 200, 50],
                ctx=ctx,
            )

            # Fewer tokens than one pool: no candidate exists, so every
            # output slot must be the invalid sentinel rather than garbage.
            # `effective_k` is 0 on this path, which no other case reaches.
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=32,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.CAUSAL.name,
                kpool=4,
            ](
                seq_lens=[2, 1, 3],
                cache_lens=[0, 0, 0],
                ctx=ctx,
            )

            # Sparse: 2048 tokens of context is 512 pools, well above top_k, so
            # the selection is a real top-k over pools rather than everything.
            test_mla_index_fp8_paged_variable_lengths[
                num_heads=32,
                depth=128,
                page_size=128,
                top_k=64,
                mask_name=MaskName.CAUSAL.name,
                kpool=4,
            ](
                seq_lens=[4, 2],
                cache_lens=[2048, 1024],
                ctx=ctx,
            )

        print("\nAll tests passed!")
