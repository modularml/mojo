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
"""MhaPrefillV2 with `KVCacheMHAOperand` (paged K/V) correctness test.

Same analytical setup as `test_mha_prefill_v2_causal.mojo`:

  K = Q = 1
  V[k, m] = (k + 1) / 512
  pre-mask att[k, q] = depth (= 128 here)
  After causal mask + softmax: o[q, m] = (q + 2) / 1024.

The novel surface is the `KVCacheMHAOperand` wrapping a
`PagedKVCacheCollection`: the kernel's `k.block_paged_tile[KV_BLOCK=64]`
call must resolve through the page LUT to land on the same K/V bytes
as the contiguous test. The LUT is a non-sequential permutation of
page indices to actually exercise the indirection.

Shape: BATCH=1, SEQ_LEN=256, NUM_KEYS=512, page_size=128, 4 pages.
depth tested at {64, 128}.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_almost_equal
from std.utils import IndexList, StaticTuple

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
)
from layout.coord import Coord, Idx
from layout.tile_layout import row_major

from nn.attention.gpu.amd_structured.mha_prefill_v2 import (
    MhaConfigV2,
    mha_prefill_v2,
)
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import KVCacheMHAOperand


comptime Q_BLOCK_SIZE = 32
comptime NUM_WARPS = 8
comptime BM = NUM_WARPS * Q_BLOCK_SIZE  # 256
comptime KV_BLOCK = 64
comptime NUM_HEADS = 1
comptime NUM_KV_HEADS = 1
comptime NUM_TILES = 8  # 8 * KV_BLOCK = 512 keys
comptime SEQ_LEN = BM
comptime NUM_KEYS = NUM_TILES * KV_BLOCK  # 512
comptime BATCH = 1
comptime PAGE_SIZE = 128  # matches production; 4 pages cover NUM_KEYS=512
comptime PAGES_PER_SEQ = NUM_KEYS // PAGE_SIZE  # 4
comptime NUM_PAGES = BATCH * PAGES_PER_SEQ  # 4
comptime NUM_LAYERS = 1
comptime LAYER_IDX = 0

# Non-sequential LUT to actually exercise the page-table indirection.
# Logical block_idx 0..3 maps to physical page [3, 1, 0, 2].
comptime _LUT_PERM = StaticTuple[Int, PAGES_PER_SEQ](3, 1, 0, 2)


def test_v2_causal_paged[depth: Int](ctx: DeviceContext) raises:
    comptime SIZE_Q = BM * depth
    comptime SIZE_OUT = BM * depth
    # kv_block layout: (NUM_PAGES, 2 [K|V], NUM_LAYERS, PAGE_SIZE, NUM_KV_HEADS, depth)
    comptime SIZE_KV_BLOCK = (
        NUM_PAGES * 2 * NUM_LAYERS * PAGE_SIZE * NUM_KV_HEADS * depth
    )

    comptime CONFIG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=KV_BLOCK,
        depth=depth,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_warps=NUM_WARPS,
    )

    print(
        "--- MhaPrefillV2 CAUSAL paged (BM=",
        BM,
        " KV=",
        KV_BLOCK,
        " D=",
        depth,
        " tiles=",
        NUM_TILES,
        " page_size=",
        PAGE_SIZE,
        " pages=",
        NUM_PAGES,
        ") ---",
    )

    var dev_q = ctx.enqueue_create_buffer[.bfloat16](SIZE_Q)
    var dev_out = ctx.enqueue_create_buffer[.float32](SIZE_OUT)
    var dev_kv_block = ctx.enqueue_create_buffer[.bfloat16](SIZE_KV_BLOCK)
    var dev_cache_lengths = ctx.enqueue_create_buffer[.uint32](BATCH)
    var dev_paged_lut = ctx.enqueue_create_buffer[.uint32](
        BATCH * PAGES_PER_SEQ
    )

    # Q = 1. Build LUT permutation. cache_lengths = 0.
    with dev_q.map_to_host() as host_q:
        for i in range(SIZE_Q):
            host_q[i] = BFloat16(1)
    with dev_cache_lengths.map_to_host() as host_cl:
        for b in range(BATCH):
            host_cl[b] = UInt32(0)
    with dev_paged_lut.map_to_host() as host_lut:
        for block_idx in range(PAGES_PER_SEQ):
            host_lut[block_idx] = UInt32(_LUT_PERM[block_idx])

    # KV block: zero everything first, then write K=1 / V=(k+1)/512 only at
    # the pages the LUT actually points to. The contiguous logical position
    # k=0..NUM_KEYS-1 maps to (page=LUT[k // PAGE_SIZE], tok=k % PAGE_SIZE).
    with dev_kv_block.map_to_host() as host_kv:
        for i in range(SIZE_KV_BLOCK):
            host_kv[i] = BFloat16(0)
        for k in range(NUM_KEYS):
            var block_idx = k // PAGE_SIZE
            var tok_in_page = k % PAGE_SIZE
            var page = _LUT_PERM[block_idx]
            var v_val = Float32(k + 1) / Float32(512)
            for m in range(depth):
                # Linear index into (NUM_PAGES, 2, NUM_LAYERS, PAGE_SIZE, NUM_KV_HEADS, depth).
                # K plane (idx 0)
                var k_off = (
                    (
                        ((page * 2 + 0) * NUM_LAYERS + LAYER_IDX) * PAGE_SIZE
                        + tok_in_page
                    )
                    * NUM_KV_HEADS
                    * depth
                    + 0 * depth
                    + m
                )
                # V plane (idx 1)
                var v_off = (
                    (
                        ((page * 2 + 1) * NUM_LAYERS + LAYER_IDX) * PAGE_SIZE
                        + tok_in_page
                    )
                    * NUM_KV_HEADS
                    * depth
                    + 0 * depth
                    + m
                )
                host_kv[k_off] = BFloat16(1)
                host_kv[v_off] = BFloat16(v_val)

    var q_tt = TileTensor(
        dev_q,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(SEQ_LEN),
                Idx[NUM_HEADS],
                Idx[depth],
            )
        ),
    )
    var o_tt = TileTensor(
        dev_out,
        row_major(
            Coord(
                Int32(BATCH),
                Int32(SEQ_LEN),
                Idx[NUM_HEADS],
                Idx[depth],
            )
        ),
    )
    # PagedKVCacheCollection. LayoutTensor wrappers over the device buffers.
    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_tensor = LayoutTensor[
        .bfloat16,
        kv_block_layout,
    ](
        dev_kv_block,
        RuntimeLayout[kv_block_layout].row_major(
            IndexList[6](
                NUM_PAGES, 2, NUM_LAYERS, PAGE_SIZE, NUM_KV_HEADS, depth
            )
        ),
    )

    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_tensor = LayoutTensor[
        mut=False,
        .uint32,
        cache_lengths_layout,
    ](
        dev_cache_lengths,
        RuntimeLayout[cache_lengths_layout].row_major(IndexList[1](BATCH)),
    )

    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_tensor = LayoutTensor[
        mut=False,
        .uint32,
        paged_lut_layout,
    ](
        dev_paged_lut.unsafe_ptr(),
        RuntimeLayout[paged_lut_layout].row_major(
            IndexList[2](BATCH, PAGES_PER_SEQ)
        ),
    )

    var kv_collection = PagedKVCacheCollection[
        DType.bfloat16,
        KVCacheStaticParams(num_heads=NUM_KV_HEADS, head_size=depth),
        PAGE_SIZE,
    ](
        # `mha_prefill_v2` reads both the `k` and `v` cache views, which are disjoint
        # kv_idx halves of one `blocks` buffer sharing its origin, so the
        # nested-origin exclusivity check rejects passing both. Declare the
        # kv_block_tensor origin as UnsafeAnyOrigin to opt out of exclusivity checking.
        kv_block_tensor.as_unsafe_any_origin(),
        cache_lengths_tensor,
        paged_lut_tensor,
        UInt32(SEQ_LEN),  # max_seq_length
        UInt32(NUM_KEYS),  # max_context_length
    )

    var k_operand = KVCacheMHAOperand(kv_collection.get_key_cache(LAYER_IDX))
    var v_operand = KVCacheMHAOperand(kv_collection.get_value_cache(LAYER_IDX))

    mha_prefill_v2[CONFIG](
        q_tt,
        k_operand,
        v_operand,
        o_tt,
        CausalMask(),
        Float32(1.0),
        NUM_KEYS,
        0,  # start_pos
        ctx,
    )

    var mismatches = 0
    var max_diff: Float32 = 0
    with dev_out.map_to_host() as host_out:
        for q in range(BM):
            var expected = Float32(q + 2) / Float32(1024)
            for d in range(depth):
                var got = host_out[q * depth + d]
                var diff = abs(got - expected)
                if diff > max_diff:
                    max_diff = diff
                if diff > 0.05:
                    mismatches += 1
                    if mismatches <= 5:
                        print(
                            "MISMATCH q=",
                            q,
                            " d=",
                            d,
                            " got=",
                            got,
                            " expected=",
                            expected,
                        )

    print("  mismatches=", mismatches, " max_diff=", max_diff)
    assert_almost_equal(Float32(mismatches), Float32(0))
    print("  PASSED")


def main() raises:
    print("=" * 60)
    print("MhaPrefillV2 CAUSAL paged-K/V GPU test")
    print("=" * 60)

    with DeviceContext() as ctx:
        test_v2_causal_paged[128](ctx)
        test_v2_causal_paged[64](ctx)

    print("=" * 60)
    print("ALL CAUSAL PAGED TESTS PASSED!")
    print("=" * 60)
