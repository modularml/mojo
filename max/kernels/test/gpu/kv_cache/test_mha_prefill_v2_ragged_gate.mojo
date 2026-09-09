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

"""MHA prefill ragged gate-firing test.

Exercises the long-context CDNA ragged prefill gate that routes eligible
BF16 long-sequence blocks through `MhaPrefillV2.run` (via
`mha_prefill_v2_ragged`) instead of FA2. The gate is mask-agnostic: any
`MHAMask` whose `status` + `mask` interface matches the generic mask
path in `MhaPrefillV2`'s `_maybe_apply_mask` works.

Eligibility (must all hold):
  comptime: BF16 + AMD CDNA + depth in {64,128} + not sink + page_size
            in {0, >=64}
  runtime:  max_prompt_len >= 4096 (perf gate, not correctness)

The paged-vs-continuous comparison is NOT an independent correctness
reference; both routes hit the same `MhaPrefillV2` kernel. The signal
this test provides:
  - gate compiles and dispatches without crashing through ragged
  - per-sequence rank-3 -> rank-4 BSHD Q-view construction is well-formed
  - output is finite (no NaN/Inf) at the gate-firing length
  - paged and continuous agree on the same `MhaPrefillV2` kernel
An independent gpu_naive-based correctness check is a follow-up.
"""

from std.math import ceildiv, rsqrt
from std.random import seed
from layout._utils import ManagedLayoutTensor
from max.gpu.host import DeviceContext
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from kv_cache_test_utils import (
    assert_no_nan_inf,
    padded_lut_cols,
    random_distinct,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from std.memory import unsafe_memcpy, unsafe_memset_zero
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import (
    CausalMask,
    ChunkedCausalMask,
    MHAMask,
    NullMask,
    SlidingWindowCausalMask,
)
from std.testing import assert_almost_equal

from std.utils import IndexList


def _run_ragged_at[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    mask_t: MHAMask,
    pass_kv_input_row_offsets: Bool = False,
    pass_sink: Bool = False,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    mask: mask_t,
    ctx: DeviceContext,
) raises:
    """Run the ragged MHA prefill gate at the given shape.

    `pass_kv_input_row_offsets=True` forces the dispatcher's
    cross-attention branch (`mha_prefill_v2_ragged[cross_attention=True]`).
    The kv-side offsets are set equal to the Q-side `input_row_offsets`,
    so this is a self-consistency check — `num_keys` derives to the
    same value as in the self-attention branch, and the two paths
    must produce identical output. Catches regressions in the new
    Phase-10 cross_attention plumbing without needing a separate
    encoder/decoder fixture.

    `pass_sink=True` forces the Phase-5b sink branch
    (`mha_prefill_v2_ragged[sink=True]`). Allocates a per-q-head
    `sink_weights` tensor with small random values and passes it
    through. Self-consistency check between paged and continuous
    paths catches regressions in the seeded `(max_vec, norm_vec)`
    init state.
    """
    # Trimmed clone of `execute_ragged_flash_attention` from
    # `test_batch_kv_cache_flash_attention_causal_mask_ragged_paged.mojo`:
    # paged + continuous at one shape with no NaN/Inf + paged-vs-continuous
    # agreement. The 16-repeat reproducibility loop is omitted — gate-firing
    # smoke test, not stress test.
    comptime page_size = 256

    var batch_size = len(valid_lengths)
    assert len(valid_lengths) == len(
        cache_lengths
    ), "expected valid_lengths and cache_lengths size to be equal"

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime lookup_table_layout = Layout(UNKNOWN_VALUE)
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_6d_layout = Layout.row_major[6]()

    var row_offsets_shape = IndexList[1](batch_size + 1)
    var cache_lengths_shape = IndexList[1](batch_size)
    var q_ragged_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )
    var output_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )

    var row_offsets_runtime_layout = RuntimeLayout[
        row_offsets_layout
    ].row_major(row_offsets_shape)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)
    var q_ragged_runtime_layout = RuntimeLayout[q_ragged_layout].row_major(
        q_ragged_shape
    )
    var output_runtime_layout = RuntimeLayout[output_layout].row_major(
        output_shape
    )
    var lookup_table_runtime_layout = RuntimeLayout[
        lookup_table_layout
    ].row_major(cache_lengths_shape)

    var input_row_offsets = ManagedLayoutTensor[.uint32, row_offsets_layout](
        row_offsets_runtime_layout, ctx
    )
    var cache_lengths_managed = ManagedLayoutTensor[
        .uint32, cache_lengths_layout
    ](cache_lengths_runtime_layout, ctx)
    var q_ragged = ManagedLayoutTensor[dtype, q_ragged_layout](
        q_ragged_runtime_layout, ctx
    )
    var test_output = ManagedLayoutTensor[dtype, output_layout](
        output_runtime_layout, ctx
    )
    var ref_output = ManagedLayoutTensor[dtype, output_layout](
        output_runtime_layout, ctx
    )

    var input_row_offsets_host = input_row_offsets.tensor[update=False]()
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()

    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = running_offset
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        running_offset += UInt32(valid_lengths[i])
    input_row_offsets_host[batch_size] = running_offset

    var q_ragged_tensor = q_ragged.tensor()
    random(q_ragged_tensor)

    var num_continuous_blocks = batch_size + 2
    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size
    )

    var kv_block_continuous_shape = IndexList[6](
        num_continuous_blocks,
        2,
        num_layers,
        max_full_context_length,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var paged_lut_shape = IndexList[2](
        batch_size,
        padded_lut_cols(ceildiv(max_full_context_length, page_size)),
    )

    var kv_block_continuous_runtime_layout = RuntimeLayout[
        kv_block_6d_layout
    ].row_major(kv_block_continuous_shape)
    var kv_block_paged_runtime_layout = RuntimeLayout[
        kv_block_6d_layout
    ].row_major(kv_block_paged_shape)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    var kv_block_continuous = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        kv_block_continuous_runtime_layout, ctx
    )
    var kv_block_paged = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged_runtime_layout, ctx
    )
    var lookup_table = ManagedLayoutTensor[.uint32, lookup_table_layout](
        lookup_table_runtime_layout, ctx
    )
    var paged_lut = ManagedLayoutTensor[.uint32, paged_lut_layout](
        paged_lut_runtime_layout, ctx
    )

    var kv_block_continuous_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_continuous.tensor[update=False]().ptr,
        kv_block_continuous_runtime_layout,
    )
    random(kv_block_continuous_tensor)
    var lookup_table_host = lookup_table.tensor[update=False]()

    # Assign each batch entry a distinct continuous block. `random_ui64` is
    # inclusive, so the draw range `[0, num_continuous_blocks - 1]` is a
    # population of `num_continuous_blocks` blocks.
    var continuous_blocks = random_distinct(num_continuous_blocks, batch_size)
    for idx in range(batch_size):
        lookup_table_host[idx] = UInt32(continuous_blocks[idx])

    var kv_block_continuous_lt = kv_block_continuous.device_tensor()
    var cache_lengths_lt = cache_lengths_managed.device_tensor()
    var lookup_table_lt = lookup_table.device_tensor()

    var kv_collection_continuous_device = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ](
        kv_block_continuous_lt,
        cache_lengths_lt,
        lookup_table_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var kv_block_paged_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged.tensor[update=False]().ptr,
        kv_block_paged_runtime_layout,
    )
    var paged_lut_tensor = paged_lut.tensor[update=False]()

    # Sample one distinct paged block per page across the whole batch up
    # front, then hand them out in iteration order. Total pages needed is
    # <= num_paged_blocks by construction.
    var total_pages = 0
    for bs in range(batch_size):
        total_pages += ceildiv(cache_lengths[bs] + valid_lengths[bs], page_size)
    var paged_blocks = random_distinct(num_paged_blocks, total_pages)

    var page_pos = 0
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        var continuous_idx = Int(lookup_table_host[bs])

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = paged_blocks[page_pos]
            page_pos += 1
            paged_lut_tensor[bs, block_idx] = UInt32(randval)
            var block_sz = min(page_size, seq_len - block_idx * page_size)

            for kv_idx in range(2):
                var paged_offset = (
                    randval
                    * kv_block_paged_shape[1]
                    * kv_block_paged_shape[2]
                    * kv_block_paged_shape[3]
                    * kv_block_paged_shape[4]
                    * kv_block_paged_shape[5]
                    + kv_idx
                    * kv_block_paged_shape[2]
                    * kv_block_paged_shape[3]
                    * kv_block_paged_shape[4]
                    * kv_block_paged_shape[5]
                    + layer_idx
                    * kv_block_paged_shape[3]
                    * kv_block_paged_shape[4]
                    * kv_block_paged_shape[5]
                )
                var continuous_offset = (
                    continuous_idx
                    * kv_block_continuous_shape[1]
                    * kv_block_continuous_shape[2]
                    * kv_block_continuous_shape[3]
                    * kv_block_continuous_shape[4]
                    * kv_block_continuous_shape[5]
                    + kv_idx
                    * kv_block_continuous_shape[2]
                    * kv_block_continuous_shape[3]
                    * kv_block_continuous_shape[4]
                    * kv_block_continuous_shape[5]
                    + layer_idx
                    * kv_block_continuous_shape[3]
                    * kv_block_continuous_shape[4]
                    * kv_block_continuous_shape[5]
                    + block_idx
                    * page_size
                    * kv_block_continuous_shape[4]
                    * kv_block_continuous_shape[5]
                )
                var n_cpy = block_sz * kv_params.num_heads * kv_params.head_size
                unsafe_memcpy(
                    dest=kv_block_paged_tensor.ptr + paged_offset,
                    src=kv_block_continuous_tensor.ptr + continuous_offset,
                    count=n_cpy,
                )
                if block_sz < page_size:
                    unsafe_memset_zero(
                        kv_block_paged_tensor.ptr + paged_offset + n_cpy,
                        (page_size - block_sz)
                        * kv_params.num_heads
                        * kv_params.head_size,
                    )

    var kv_block_paged_lt = kv_block_paged.device_tensor()
    var paged_lut_lt = paged_lut.device_tensor()

    var kv_collection_paged_device = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        kv_block_paged_lt,
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var q_ragged_lt = q_ragged.device_tensor()
    var ref_output_lt = ref_output.device_tensor()
    var test_output_lt = test_output.device_tensor()

    # Continuous-KV ragged path. On AMD CDNA + BF16 + depth in {64,128}
    # + any MHAMask + not-sink + seq_len>=4096 this routes through the
    # ragged prefill gate in `flash_attention_dispatch`
    # (k_t.page_size == 0 branch).
    #
    # `pass_kv_input_row_offsets=True` routes through the Phase-10
    # cross_attention branch of the dispatcher; the offsets are
    # equal to `input_row_offsets`, so the kernel arrives at the
    # same `num_keys` as the self-attention path. Output must
    # match the paged-vs-continuous reference under the same
    # tolerance.
    #
    # `kv_input_row_offsets` dispatcher contract is
    # `OptionalReg[LayoutTensor[uint32, Layout.row_major(UNKNOWN_VALUE),
    # ImmutAnyOrigin]]`. The test-side managed tensor has a different
    # layout/origin, so we rebuild a typed view (mirrors
    # `kv_cache_ragged.mojo:3492-3501`).
    var input_row_offsets_dt = input_row_offsets.device_tensor()
    var kv_input_row_offsets_view = LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        input_row_offsets_dt.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            input_row_offsets_dt.runtime_layout.shape.value.canonicalize()
        ),
    )

    # Sink-path setup (Phase 5b). Allocated only when `pass_sink=True`
    # so non-sink cases pay zero overhead. Per-q-head weights with small
    # random values keep the seeded `(max_vec, norm_vec)` invariant
    # exercised without dominating the rowmax across all tiles.
    comptime sink_layout = Layout.row_major(UNKNOWN_VALUE)
    var sink_runtime_layout = RuntimeLayout[sink_layout].row_major(
        IndexList[1](num_q_heads)
    )
    var sink_managed = ManagedLayoutTensor[dtype, sink_layout](
        sink_runtime_layout, ctx
    )
    comptime if pass_sink:
        var sink_host = sink_managed.tensor[update=False]()
        # Small fixed sink weights, one per q-head. Range matches the
        # `test_mha_sink_weights.mojo` adversarial seed (within ~[-1, 1]).
        for h in range(num_q_heads):
            sink_host[h] = Scalar[dtype](0.1) * Scalar[dtype](h % 7 - 3)
    var sink_device_t = sink_managed.device_tensor()
    var sink_device_view = LayoutTensor[
        dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ](
        sink_device_t.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            sink_device_t.runtime_layout.shape.value.canonicalize()
        ),
    )
    comptime if pass_sink:
        flash_attention[ragged=True, sink=True](
            ref_output_lt,
            q_ragged_lt,
            kv_collection_continuous_device.get_key_cache(layer_idx),
            kv_collection_continuous_device.get_value_cache(layer_idx),
            mask,
            input_row_offsets_dt,
            rsqrt(Float32(kv_params.head_size)),
            ctx,
            sink_weights=sink_device_view,
        )
    elif pass_kv_input_row_offsets:
        flash_attention[ragged=True](
            ref_output_lt,
            q_ragged_lt,
            kv_collection_continuous_device.get_key_cache(layer_idx),
            kv_collection_continuous_device.get_value_cache(layer_idx),
            mask,
            input_row_offsets_dt,
            rsqrt(Float32(kv_params.head_size)),
            ctx,
            kv_input_row_offsets=kv_input_row_offsets_view,
        )
    else:
        flash_attention[ragged=True](
            ref_output_lt,
            q_ragged_lt,
            kv_collection_continuous_device.get_key_cache(layer_idx),
            kv_collection_continuous_device.get_value_cache(layer_idx),
            mask,
            input_row_offsets_dt,
            rsqrt(Float32(kv_params.head_size)),
            ctx,
        )

    # Paged-KV ragged path. Same gate, page_size>=64 branch.
    comptime if pass_sink:
        flash_attention[ragged=True, sink=True](
            test_output_lt,
            q_ragged_lt,
            kv_collection_paged_device.get_key_cache(layer_idx),
            kv_collection_paged_device.get_value_cache(layer_idx),
            mask,
            input_row_offsets_dt,
            rsqrt(Float32(kv_params.head_size)),
            ctx,
            sink_weights=sink_device_view,
        )
    elif pass_kv_input_row_offsets:
        flash_attention[ragged=True](
            test_output_lt,
            q_ragged_lt,
            kv_collection_paged_device.get_key_cache(layer_idx),
            kv_collection_paged_device.get_value_cache(layer_idx),
            mask,
            input_row_offsets_dt,
            rsqrt(Float32(kv_params.head_size)),
            ctx,
            kv_input_row_offsets=kv_input_row_offsets_view,
        )
    else:
        flash_attention[ragged=True](
            test_output_lt,
            q_ragged_lt,
            kv_collection_paged_device.get_key_cache(layer_idx),
            kv_collection_paged_device.get_value_cache(layer_idx),
            mask,
            input_row_offsets_dt,
            rsqrt(Float32(kv_params.head_size)),
            ctx,
        )

    assert_no_nan_inf(ref_output, "ref_output_continuous")
    assert_no_nan_inf(test_output, "test_output_paged")

    var ref_out = ref_output.tensor()
    var test_out = test_output.tensor()
    var input_row_offsets_tensor = input_row_offsets.tensor()
    for bs in range(batch_size):
        var prompt_len = valid_lengths[bs]
        var ragged_offset = Int(input_row_offsets_tensor[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        # Paged-vs-continuous tolerance: 2e-2 accommodates
                        # BF16 accumulation-order differences over the
                        # longer multi-seq shapes (seq_len up to ~5K vs
                        # the upstream test's ~1K). Both paths exercise
                        # the same kernel; this is a self-consistency
                        # check, not an independent correctness reference.
                        assert_almost_equal(
                            ref_out[ragged_offset + s, h, hd],
                            test_out[ragged_offset + s, h, hd],
                            atol=2e-2,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            ref_out[ragged_offset + s, h, hd],
                            test_out[ragged_offset + s, h, hd],
                        )
                        raise e^


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        # Case 1: single sequence, seq_len = 4096 (BM-aligned). Smoke
        # baseline — the gate fires, block 15 is the last and fully
        # valid. Llama-3.1 8B GQA shape (32 Q heads / 8 KV heads, d=128).
        print(
            "[1/9] ragged Causal seq_len=4096 (aligned, full last tile):",
        )
        _run_ragged_at[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=8, head_size=128),
        ](
            [4096],
            [0],
            2,
            1,
            CausalMask(),
            ctx,
        )

        # Case 2: single sequence, seq_len = 4097 (NOT BM-aligned). The
        # gate fires (seq_len >= 4096); the last tile has 1 valid Q row
        # and 255 OOB rows. Exercises the partial-Q-tile writeback skip
        # in `_store_o_to_gmem`: OOB rows would otherwise corrupt the
        # output buffer (or get garbage from buffer_load returning 0).
        print(
            "[2/9] ragged Causal seq_len=4097 (unaligned, partial last tile):",
        )
        _run_ragged_at[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=8, head_size=128),
        ](
            [4097],
            [0],
            2,
            1,
            CausalMask(),
            ctx,
        )

        # Case 3: multi-sequence ragged with ALIGNED lengths.
        # Exercises the multi-seq dispatch (block_idx.z varies) with
        # `ragged=True` forcing the kernel's Q/O batch coord to 0 so the
        # per-sequence pre-offset pointer is selected.
        print(
            "[3/9] ragged Causal multi-seq, ALIGNED lengths:",
        )
        _run_ragged_at[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=8, head_size=128),
        ](
            [4096, 5120, 4352],
            [0, 0, 0],
            2,
            1,
            CausalMask(),
            ctx,
        )

        # Case 4: multi-sequence ragged with MIXED-LENGTH sequences,
        # none aligned to BM=256. Combines multi-seq dispatch + per-
        # sequence partial-Q writeback skip.
        print(
            "[4/9] ragged Causal multi-seq, mixed unaligned lengths:",
        )
        _run_ragged_at[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=8, head_size=128),
        ](
            [4097, 5333, 4200],
            [0, 0, 0],
            2,
            1,
            CausalMask(),
            ctx,
        )

        # Case 5: NullMask through the kernel at seq_len=8192. The
        # generic mask path comptime-elides for NullMask (status is
        # always NO_MASK), so this is effectively a "no mask" run.
        print(
            "[5/9] ragged NullMask seq_len=8192:",
        )
        _run_ragged_at[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=8, head_size=128),
        ](
            [8192],
            [0],
            2,
            1,
            NullMask(),
            ctx,
        )

        # Case 6: SlidingWindowCausalMask[4096] through the kernel at
        # seq_len=8192. Previously produced Inf — root-caused to a
        # stale `scale_vec` from the lazy rescale getting re-applied
        # in `_tail_softmax_unconditional` during the epilogue. Fixed
        # by resetting `scale_vec=1` in `_pv_strip_with_partial_softmax`'s
        # else-branch (no-rescale path), so the epilogue's
        # unconditional multiply is identity when no rescale fired.
        print(
            "[6/9] ragged SlidingWindowCausalMask[4096] seq_len=8192:",
        )
        _run_ragged_at[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=8, head_size=128),
        ](
            [8192],
            [0],
            2,
            1,
            SlidingWindowCausalMask[4096](),
            ctx,
        )

        # Case 7: ChunkedCausalMask[2048] through the kernel at
        # seq_len=8192. Chunked == causal within chunks; same
        # generic-mask path through the kernel as SlidingWindow.
        # TODO(KERN-3133): This test was disabled because the v2 kernel also was
        # disabled for KERN-3053. The v1 kernel hit asserts running this, so
        # disable to understand if the problem is the test or the v1 kernel.
        # Even if the v2 kernel is fixed, this should still be investigated as
        # that kernel is only selected for some shapes.
        # print(
        #     "[7/9] ragged ChunkedCausalMask[2048] seq_len=8192:",
        # )
        # _run_ragged_at[
        #     32,
        #     DType.bfloat16,
        #     KVCacheStaticParams(num_heads=8, head_size=128),
        # ](
        #     [8192],
        #     [0],
        #     2,
        #     1,
        #     ChunkedCausalMask[2048](),
        #     ctx,
        # )

        # Case 8: Phase-10 cross-attention plumbing smoke. Calls
        # `flash_attention[ragged=True]` with `kv_input_row_offsets`
        # set equal to `input_row_offsets`, exercising the
        # dispatcher's `if kv_input_row_offsets:` branch and the
        # `mha_prefill_v2_ragged[cross_attention=True]` launcher.
        # Because the kv-side offsets match the Q-side, `num_keys`
        # derives identically to the self-attention path and the
        # output must match the paged-vs-continuous reference at
        # the same tolerance (2e-2) used by the other cases.
        print(
            (
                "[8/9] ragged Causal seq_len=4096 + Phase-10"
                " kv_input_row_offsets (self-consistency):"
            ),
        )
        _run_ragged_at[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=8, head_size=128),
            CausalMask,
            pass_kv_input_row_offsets=True,
        ](
            [4096],
            [0],
            2,
            1,
            CausalMask(),
            ctx,
        )

        # Case 9: Phase-5b sink plumbing smoke. Calls
        # `flash_attention[ragged=True, sink=True]` with per-q-head
        # `sink_weights`. Exercises the dispatcher's
        # `comptime if sink:` branch and the
        # `mha_prefill_v2_ragged[sink=True]` launcher. The kernel's
        # `comptime if sink:` init seeds `max_vec / max_vec_prev` to
        # `log2e * sink_weight[head_idx]` and `norm_vec = 1` —
        # equivalent to a virtual sink token contributing to the
        # softmax denominator. Paged-vs-continuous self-consistency
        # catches regressions in the seeded init.
        print(
            (
                "[9/9] ragged Causal seq_len=4096 + Phase-5b"
                " sink_weights (self-consistency):"
            ),
        )
        _run_ragged_at[
            32,
            DType.bfloat16,
            KVCacheStaticParams(num_heads=8, head_size=128),
            CausalMask,
            pass_sink=True,
        ](
            [4096],
            [0],
            2,
            1,
            CausalMask(),
            ctx,
        )

        print("OK")
