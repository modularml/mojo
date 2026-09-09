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

"""Regression test for PAQ-2333: GPU attention kernel hang with inflight
batching on Gemma3 27B.

Production hit a TMA transaction barrier deadlock
(SYNCS.PHASECHK.TRANS64.TRYWAIT) in the SM100 FA4 attention kernel on a mixed
TG+CE batch. This file drives that batch -- 230 TG (seq_len 1, cache 100-1000)
plus 40 CE (seq_len 50-1000) over 270 requests -- and gates on the kernel both
returning and producing no NaN/Inf.

It does not, however, pin the deadlock. As first committed the batch
oversubscribed its own KV block pool, so it died in host setup before any
launch and never reached FA4 at any page size (see `num_blocks` below). A pass
here therefore means "this shape mix completes", not "the TMA deadlock class is
fixed".
"""

from std.math import ceildiv, rsqrt
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int
from layout._utils import ManagedLayoutTensor
from max.gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from kv_cache_test_utils import (
    assert_no_nan_inf,
    padded_lut_cols,
    random_distinct,
)
from nn.attention.gpu.mha import flash_attention
from nn.attention.mha_mask import CausalMask
from std.utils import IndexList


def test_paged_ragged_attention[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    num_paged_blocks: Int,
    ctx: DeviceContext,
) raises:
    comptime page_size = get_defined_int["page_size", 256]()
    var batch_size = len(valid_lengths)

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    # Pages the batch needs: one per `page_size` keys per request, rounded up
    # per request (pages are never shared between requests here).
    var total_pages = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]
        total_pages += ceildiv(cache_lengths[i] + valid_lengths[i], page_size)

    # `num_paged_blocks` is the production pool size the repro quotes, but the
    # randomized batch is not sized against it: per-request round-up costs up
    # to `page_size - 1` keys per request, and at 270 requests that overshoots
    # the pool at every page size in the sweep (695 / 1275 / 2406 blocks needed
    # vs 647).
    #
    # The LUT below hands out DISTINCT blocks, so an undersized pool is
    # unsatisfiable, and it fails LOUDLY but in the wrong place:
    # `random_distinct(n, k)` takes the `k`-prefix of `randperm(n)`, and
    # `List.shrink` calls `abort()` when `k > n`. So the repro dies in host
    # setup before any launch -- which is exactly what happened here for its
    # whole history, turning a kernel hang repro into a host-side failure that
    # still looked like a timeout. Grow the pool to what the batch needs.
    #
    # `total_pages` is the EXACT requirement, not a bound: it is accumulated by
    # the loop above with the same `ceildiv(cache + valid, page_size)`
    # expression the LUT loop below consumes, so `k <= n` holds by
    # construction. Sizing from `max_full_context_length` instead (as
    # `test_batch_kv_cache_flash_attention_causal_mask_ragged_paged.mojo` does)
    # is also correct but ~1.7x looser here, and being page-size-invariant it
    # would hide the fact that the requirement GROWS as `page_size` shrinks --
    # which is the reason this repro never ran.
    var num_blocks = max(num_paged_blocks, total_pages)

    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
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

    var row_offsets_rl = RuntimeLayout[row_offsets_layout].row_major(
        row_offsets_shape
    )
    var cache_lengths_rl = RuntimeLayout[cache_lengths_layout].row_major(
        cache_lengths_shape
    )
    var q_ragged_rl = RuntimeLayout[q_ragged_layout].row_major(q_ragged_shape)
    var output_rl = RuntimeLayout[output_layout].row_major(output_shape)

    var input_row_offsets = ManagedLayoutTensor[.uint32, row_offsets_layout](
        row_offsets_rl, ctx
    )
    var cache_lengths_managed = ManagedLayoutTensor[
        .uint32, cache_lengths_layout
    ](cache_lengths_rl, ctx)
    var q_ragged = ManagedLayoutTensor[dtype, q_ragged_layout](q_ragged_rl, ctx)
    var test_output = ManagedLayoutTensor[dtype, output_layout](output_rl, ctx)

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

    var kv_block_paged_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    # Pad LUT inner dim to honor `PagedKVCache.populate`'s SIMD padding
    # invariant — see `padded_lut_cols`.
    var paged_lut_shape = IndexList[2](
        batch_size,
        padded_lut_cols(ceildiv(max_full_context_length, page_size)),
    )

    var kv_block_paged_rl = RuntimeLayout[kv_block_6d_layout].row_major(
        kv_block_paged_shape
    )
    var paged_lut_rl = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    var kv_block_paged = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged_rl, ctx
    )
    var paged_lut = ManagedLayoutTensor[.uint32, paged_lut_layout](
        paged_lut_rl, ctx
    )

    var kv_block_paged_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged.tensor[update=False]().ptr,
        kv_block_paged_rl,
    )
    random(kv_block_paged_tensor)

    var paged_lut_tensor = paged_lut.tensor[update=False]()
    # Sample one distinct paged block per page across the whole batch up
    # front, then hand them out in iteration order. `num_blocks >= total_pages`
    # by construction above.
    var paged_blocks = random_distinct(num_blocks, total_pages)
    var page_pos = 0
    for bs in range(batch_size):
        var seq_len = cache_lengths[bs] + valid_lengths[bs]
        for block_idx in range(ceildiv(seq_len, page_size)):
            paged_lut_tensor[bs, block_idx] = UInt32(paged_blocks[page_pos])
            page_pos += 1

    var kv_block_paged_lt = kv_block_paged.device_tensor()
    var cache_lengths_lt = cache_lengths_managed.device_tensor()
    var paged_lut_lt = paged_lut.device_tensor()

    var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        kv_block_paged_lt,
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(max_prompt_length),
        UInt32(max_full_context_length),
    )

    var q_ragged_lt = q_ragged.device_tensor()
    var test_output_lt = test_output.device_tensor()

    var ce_count = 0
    var tg_count = 0
    for i in range(batch_size):
        if valid_lengths[i] == 1:
            tg_count += 1
        else:
            ce_count += 1

    print(
        "Running: batch_size=",
        batch_size,
        "total_tokens=",
        total_length,
        "CE=",
        ce_count,
        "TG=",
        tg_count,
        "max_prompt_len=",
        max_prompt_length,
        "max_context_len=",
        max_full_context_length,
        "num_pages=",
        num_blocks,
        "pages_needed=",
        total_pages,
    )

    flash_attention[ragged=True](
        test_output_lt,
        q_ragged_lt,
        kv_collection.get_key_cache(layer_idx),
        kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        input_row_offsets.device_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.synchronize()
    assert_no_nan_inf(test_output, "gemma3_hang_output")
    print("  -> OK")


def main() raises:
    seed(42)

    # Gemma3 27B config: 32 Q heads, 16 KV heads, head_dim=128
    comptime kv_params = KVCacheStaticParams(num_heads=16, head_size=128)
    comptime num_q_heads = 32
    comptime num_layers = 2
    comptime layer_idx = 1

    with DeviceContext() as ctx:
        # Mixed TG+CE batch with randomized shapes that triggers TMA deadlock.
        # 230 TG requests (seq_len=1) with cache_lens 100-1000
        # 40 CE requests with seq_lens 50-1000
        #
        # 647 is the Gemma3 27B server pool size this repro quotes, but it is
        # only a floor: the batch needs more than that at every swept page size,
        # so the pool grows to what it actually needs. The run line below prints
        # the pool that was allocated (`num_pages=`) and the requirement it was
        # sized from (`pages_needed=`).
        print("=== PAQ-2333 repro: 230 TG + 40 CE, seed=42 ===")
        var active_lens = List[Int]()
        var cache_lens = List[Int]()
        for _ in range(230):
            active_lens.append(1)
            cache_lens.append(Int(random_ui64(100, 1000)))
        for _ in range(40):
            active_lens.append(Int(random_ui64(50, 1000)))
            cache_lens.append(0)

        test_paged_ragged_attention[num_q_heads, DType.bfloat16, kv_params](
            active_lens, cache_lens, num_layers, layer_idx, 647, ctx
        )
