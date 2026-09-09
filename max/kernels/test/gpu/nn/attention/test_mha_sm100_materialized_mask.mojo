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

"""Compile-time regression guard: SM100 FA4 dispatch must not instantiate
split-K candidates for UNKNOWN_MASK masks (MaterializedMask).

## What this guards

Before the fix in dispatch.mojo, `SPLITK_CANDIDATES = [10, 4, 2]` was
unconditional. For any mask whose `nonfull_sets()[0] == UNKNOWN_MASK`
(MaterializedMask, AndMask, OrMask), the comptime-for loop over
SPLITK_CANDIDATES instantiated split-K kernels whose `fa4_softmax` hit a
comptime assert:

    "split-K (M2) supports only check_mask==False masks"

This caused a COMPILE FAILURE for any model using MaterializedMask on B200.
The motivating production failure: FLUX.2's padded Qwen3 text encoder, which
uses a causal+padding bias tensor via MaterializedMask.

The fix: `SPLITK_CANDIDATES = ([10, 4, 2] if splitk_mask_ok else List[Int]())`
where `splitk_mask_ok` gates on `nonfull_sets()[0] != UNKNOWN_MASK`.

With the fix, this file compiles and produces output matching the naive
reference within bf16 tolerance. Without the fix, it fails to compile.

Target hardware family: NVIDIA SM100 (B200) only.
Tested shape: depth=128, cache_length=1024 (-> by_cache=2 -> P=2 without fix),
valid_length=64 (-> stays in 1Q territory, not pair-CTA).
"""

from std.math import ceildiv, rsqrt
from std.random import random_ui64, seed

from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import MaterializedMask

from std.utils import IndexList

from std.collections import Set


comptime _LUT_TAIL_PAD = 16


def padded_lut_cols(cols: Int) -> Int:
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def execute_materialized_mask_test(ctx: DeviceContext) raises:
    # Shape: GQA 64q / 8kv (group=8), depth=128 -- same family as the
    # splitk_lse template. With cache_length=1024, by_cache=2, the pre-fix
    # dispatch would pick P=2 and instantiate the split-K kernel, hitting the
    # comptime assert. Post-fix, SPLITK_CANDIDATES is empty for MaterializedMask
    # (UNKNOWN_MASK), so dispatch falls through to the plain 1Q single-partition
    # path instead.
    comptime num_kv_heads = 8
    comptime group = 8
    comptime num_q_heads = num_kv_heads * group
    comptime head_size = 128
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_kv_heads, head_size=head_size
    )
    comptime dtype = DType.bfloat16
    comptime page_size = 128
    comptime kv_heads = kv_params.num_heads
    comptime num_layers = 1
    comptime layer_idx = 0

    # valid_length > 32 keeps this in the 1Q regime (not WS BM=32) and well
    # below a 2Q tile (BM=256), so the 1Q split-K scan triggers pre-fix.
    comptime valid_length = 64
    # cache_length=1024: by_cache = 1024//512 = 2; pre-fix dispatch would pick
    # P=2 and compile a split-K kernel for MaterializedMask -> compile error.
    comptime cache_length = 1024
    comptime num_keys = cache_length + valid_length
    var total_length = valid_length
    var batch_size = 1

    # --- Mask tensor ---
    # Shape: [batch=1, q=valid_length, k=num_keys], rank 3. The rank-3 path in
    # MaterializedMask.mask() drops the head coordinate, broadcasting one bias
    # plane across all heads (rank 4 would index the head dim unclamped).
    # Values encode an additive causal-style attention bias:
    #   0.0       for visible (causal: key <= query position)
    #   large_neg for masked  (future keys, i.e. key > query position)
    # This is the bias shape FLUX.2's Qwen3 text encoder produces for its
    # padded sequence attention: fully-visible past, fully-masked future.
    comptime mask_rows = valid_length
    comptime mask_cols = num_keys
    comptime large_neg = Float32(-10000.0)
    comptime mask_layout = Layout.row_major(1, mask_rows, mask_cols)

    var mask_size = 1 * mask_rows * mask_cols
    var mask_host = ctx.enqueue_create_host_buffer[dtype](mask_size)
    for qi in range(mask_rows):
        for ki in range(mask_cols):
            # Causal mask: key position relative to (cache_length + qi).
            # The query token at prompt position qi attends to keys
            # [0, cache_length+qi] (inclusive) -- the full cached prefix plus
            # its own position. Keys beyond that are future tokens.
            var visible = ki <= cache_length + qi
            var val = Float32(0.0) if visible else large_neg
            mask_host[qi * mask_cols + ki] = Scalar[dtype](val)
    var mask_dev = ctx.enqueue_create_buffer[dtype](mask_size)
    ctx.enqueue_copy(mask_dev, mask_host)

    # The kernels pass the mask a GLOBAL query row (cache_length + qi, see
    # mha_gpu_naive's `score_row = y + cur_cache_len - cur_query_len`). The
    # mask tensor only has `valid_length` prompt rows, so the row lookup must
    # be shifted back by cache_length. That is exactly MaterializedMask's
    # DEFAULT start_pos (mask_cols - mask_rows == cache_length), so no
    # explicit start_pos is passed, matching the production construction in
    # gpu/mha.mojo.
    var mask_lt = LayoutTensor[mut=False, dtype, mask_layout](
        mask_dev,
        RuntimeLayout[mask_layout].row_major(
            IndexList[3](1, mask_rows, mask_cols)
        ),
    )
    var mat_mask = MaterializedMask(mask_lt)

    print(
        "test_mha_sm100_materialized_mask: depth=",
        head_size,
        " cache=",
        cache_length,
        " valid=",
        valid_length,
        " num_keys=",
        num_keys,
        sep="",
    )

    comptime row_offsets_layout = Layout(UNKNOWN_VALUE)
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_size
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, head_size
    )
    comptime paged_lut_layout = Layout.row_major[2]()
    comptime kv_block_6d_layout = Layout.row_major[6]()

    var scale = rsqrt(Float32(head_size))

    seed(0x9F3A)

    # --- Row offsets ---
    var input_row_offsets = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    input_row_offsets[0] = 0
    input_row_offsets[1] = UInt32(valid_length)
    var input_row_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_dev, input_row_offsets)
    var input_row_offsets_lt = LayoutTensor[
        mut=False, .uint32, row_offsets_layout
    ](
        input_row_offsets_dev,
        RuntimeLayout[row_offsets_layout].row_major(
            IndexList[1](batch_size + 1)
        ),
    )

    # --- Q ---
    var q_size = total_length * num_q_heads * head_size
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    var q_host_tt = LayoutTensor[dtype, q_ragged_layout](
        q_host.unsafe_ptr(),
        RuntimeLayout[q_ragged_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )
    random(q_host_tt)
    var q_dev = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_dev, q_host)
    var q_lt = LayoutTensor[mut=False, dtype, q_ragged_layout](
        q_dev,
        RuntimeLayout[q_ragged_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )

    # --- Paged KV blocks ---
    var num_paged_blocks = ceildiv(num_keys, page_size) * batch_size + 4
    var kv_block_paged_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        head_size,
    )
    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * head_size
    )
    var kv_block_host = ctx.enqueue_create_host_buffer[dtype](kv_block_size)
    var kv_block_host_tt = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_host.unsafe_ptr(),
        RuntimeLayout[kv_block_6d_layout].row_major(kv_block_paged_shape),
    )
    random(kv_block_host_tt)
    var kv_block_dev = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(kv_block_dev, kv_block_host)
    var kv_block_paged_lt = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_dev,
        RuntimeLayout[kv_block_6d_layout].row_major(kv_block_paged_shape),
    )

    # --- Lookup table ---
    var full_pages = ceildiv(num_keys, page_size)
    var lut_cols = padded_lut_cols(full_pages)
    var paged_lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * lut_cols
    )
    var lut_set = Set[Int]()
    for blk in range(full_pages):
        var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
        while randval in lut_set:
            randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
        lut_set.add(randval)
        paged_lut_host[blk] = UInt32(randval)

    # --- Cache lengths ---
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    cache_lengths_host[0] = UInt32(cache_length)
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    var cache_lengths_lt = LayoutTensor[
        mut=False, .uint32, cache_lengths_layout
    ](
        cache_lengths_dev,
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](batch_size)),
    )

    var paged_lut_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size * lut_cols
    )
    ctx.enqueue_copy(paged_lut_dev, paged_lut_host)
    var paged_lut_lt = LayoutTensor[mut=False, .uint32, paged_lut_layout](
        paged_lut_dev,
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](batch_size, lut_cols)
        ),
    )

    var kv_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        # K and V views are disjoint kv_idx halves sharing one buffer origin;
        # declare UnsafeAnyOrigin to opt out of exclusivity checking (same
        # pattern as test_mha_sm100_1q_splitk_lse.mojo).
        kv_block_paged_lt.as_unsafe_any_origin(),
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(valid_length),
        UInt32(num_keys),
    )
    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)

    # ============ Run 1: flash_attention with MaterializedMask ============
    # This is the elaboration that FAILS TO COMPILE without the fix: the
    # dispatch's comptime-for over SPLITK_CANDIDATES instantiates split-K
    # kernels for MaterializedMask, which hit "split-K (M2) supports only
    # check_mask==False masks" in fa4_softmax.
    var test_out_size = total_length * num_q_heads * head_size
    var test_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var test_out_lt = LayoutTensor[dtype, output_layout](
        test_out_dev.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )

    flash_attention[ragged=True](
        test_out_lt,
        q_lt,
        k_cache,
        v_cache,
        mat_mask,
        input_row_offsets_lt,
        scale,
        ctx,
    )

    # ============ Run 2: mha_gpu_naive with the SAME MaterializedMask ========
    var ref_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var ref_out_lt = LayoutTensor[dtype, output_layout](
        ref_out_dev.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            IndexList[3](total_length, num_q_heads, head_size)
        ),
    )

    mha_gpu_naive[ragged=True](
        q_lt,
        k_cache,
        v_cache,
        mat_mask,
        ref_out_lt,
        input_row_offsets_lt,
        scale,
        batch_size,
        valid_length,
        num_keys,
        num_q_heads,
        head_size,
        group,
        ctx,
    )

    ctx.synchronize()

    var test_out_host = ctx.enqueue_create_host_buffer[dtype](test_out_size)
    var ref_out_host = ctx.enqueue_create_host_buffer[dtype](test_out_size)
    ctx.enqueue_copy(test_out_host, test_out_dev)
    ctx.enqueue_copy(ref_out_host, ref_out_dev)
    ctx.synchronize()

    var max_abs_diff: Float32 = 0.0
    var argmax_idx = 0
    for i in range(total_length):
        for h in range(num_q_heads):
            for d in range(head_size):
                var flat = (i * num_q_heads + h) * head_size + d
                var a = test_out_host[flat].cast[.float32]()
                var expected = ref_out_host[flat].cast[.float32]()
                var diff = abs(a - expected)
                if diff > max_abs_diff:
                    max_abs_diff = diff
                    argmax_idx = flat

    print(
        "  max-abs diff(flash_attention, mha_gpu_naive) =",
        max_abs_diff,
        " at flat idx",
        argmax_idx,
    )

    _ = q_dev^
    _ = kv_block_dev^
    _ = paged_lut_dev^
    _ = input_row_offsets_dev^
    _ = cache_lengths_dev^
    _ = test_out_dev^
    _ = ref_out_dev^
    _ = mask_dev^

    comptime atol = Float32(0.04)
    if max_abs_diff > atol:
        raise Error(
            "MaterializedMask attention diverged from naive reference:"
            " max-abs diff "
            + String(max_abs_diff)
            + " > atol "
            + String(atol)
        )

    print("test_mha_sm100_materialized_mask: PASSED")


def main() raises:
    with DeviceContext() as ctx:
        execute_materialized_mask_test(ctx)
