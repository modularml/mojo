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

"""Direct unit tests for `PagedKVCache.populate` / `_simd_load_lut`.

Drives `populate[BN, base_alignment]` over the production-relevant
`(BN, page_size, base_alignment)` matrix, with LUT padding entries set
to a sentinel block ID. Asserts no sentinel leaks through into the
populated `rows[]` — catching SIMD-chunk derivation bugs and
mis-aligned LUT loads regardless of whether they currently corrupt
softmax downstream.

Related: KERN-2861 (NaN at page_size=128 in Gemma-3/4). Complements
`test_mha_paged_kv_oob_canary.mojo` which catches OOB into KV-data
memory via a poisoned-padding stress test.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import unsafe_memset_zero
from std.sys.defines import get_defined_int
from std.utils import IndexList

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from kv_cache_test_utils import padded_lut_cols


# Sentinel block ID written to LUT padding entries. Larger than any
# `num_used` we'll ever set, so any returned block_idx >= num_used implies
# a load reached the padding region.
comptime _SENTINEL = UInt32(999_999)


def _populate_kernel[
    cache_t: KVCacheT,
    BN: Int,
    base_alignment: Int,
    num_pages: Int,
](
    kv: cache_t,
    output_ptr: MutPointer[UInt32, MutAnyOrigin],
    base_kv_row: UInt32,
):
    """Single-thread kernel: write `populate[BN, base_alignment]` rows[] to
    `output_ptr`. Comptime-specialized per (BN, base_alignment, page_size).
    """
    if global_idx.x != 0:
        return
    var rows = kv.populate[BN, base_alignment](
        batch_idx=UInt32(0), base_kv_row=base_kv_row
    )
    comptime for i in range(num_pages):
        output_ptr[i] = rows.rows[i]


def run_one[
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
    BN: Int,
    base_alignment: Int,
](base_kv_row: Int, ctx: DeviceContext) raises:
    """Exercise `populate[BN, base_alignment]` with a poisoned LUT.

    `num_used` is sized so the kernel-requested LUT indices
    `[base_kv_row/page_size, +num_pages)` all hit valid entries; everything
    past that is `_SENTINEL`. Asserts every returned `rows[i]` matches the
    expected `(first_lut_idx+i) * stride` (catches sentinel leaks AND
    wrong-but-valid block_idx — i.e., off-by-one or transposed loads).
    """
    # `eff_page = kv_sub_tile_rows(BN, page_size)`. For `page_size >= BN`
    # this collapses to BN (single-page case); otherwise eff_page == page_size.
    comptime eff_page = page_size if 0 < page_size < BN else BN
    comptime num_pages = BN // eff_page

    # Size `num_used` so the kernel's SIMD load reads only valid (non-
    # sentinel) entries. The kernel reads `[first_lut_idx, +num_pages)`
    # where `first_lut_idx = base_kv_row / page_size`. Add a 4-entry
    # sentinel tail past that.
    var first_lut_idx_runtime = base_kv_row // page_size
    var num_used = first_lut_idx_runtime + num_pages + 4

    # Pad LUT to honor `populate`'s SIMD invariants: row stride must
    # be chunk-aligned (chunk capped at 8) AND the row must hold a
    # 16-wide load past any valid `first_lut_idx`. `padded_lut_cols`
    # rounds to a multiple of 8 plus a 16-element tail pad.
    var lut_columns = padded_lut_cols(num_used)

    # LUT [batch=1, lut_columns] uint32. Fill: valid IDs then sentinel.
    comptime lut_layout = Layout.row_major[2]()
    var lut_shape = IndexList[2](1, lut_columns)
    var lut_runtime = RuntimeLayout[lut_layout].row_major(lut_shape)
    var lut = ManagedLayoutTensor[.uint32, lut_layout](lut_runtime, ctx)
    var lut_host = lut.tensor[update=False]()
    for c in range(lut_columns):
        if c < num_used:
            lut_host[0, c] = UInt32(c)
        else:
            lut_host[0, c] = _SENTINEL

    # cache_lengths [batch=1] uint32. Cover all valid blocks so the kernel
    # treats every `base_kv_row` as in-cache.
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](1)
    var cache_lengths_runtime = RuntimeLayout[cache_lengths_layout].row_major(
        cache_lengths_shape
    )
    var cache_lengths = ManagedLayoutTensor[.uint32, cache_lengths_layout](
        cache_lengths_runtime, ctx
    )
    var cache_lengths_host = cache_lengths.tensor[update=False]()
    cache_lengths_host[0] = UInt32(num_used * page_size)

    # Minimum-size blocks tensor (we never read it; populate is index-only).
    comptime blocks_layout = Layout.row_major[6]()
    var num_paged_blocks = max(num_used, 1)
    var blocks_shape = IndexList[6](
        num_paged_blocks,
        2,
        1,  # num_layers
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var blocks_runtime = RuntimeLayout[blocks_layout].row_major(blocks_shape)
    var blocks = ManagedLayoutTensor[dtype, blocks_layout](blocks_runtime, ctx)
    var blocks_host = blocks.tensor[update=False]()
    unsafe_memset_zero(blocks_host.ptr, blocks_runtime.size())

    # Output buffer: enough for the largest `num_pages` we'll ever request.
    comptime _MAX_PAGES = 16
    var output_buf = ctx.enqueue_create_buffer[.uint32](_MAX_PAGES)
    var output_init = ctx.enqueue_create_host_buffer[.uint32](_MAX_PAGES)
    for i in range(_MAX_PAGES):
        output_init[i] = UInt32(0xCDCDCDCD)
    ctx.enqueue_copy(output_buf, output_init)

    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths.device_tensor(),
        lut.device_tensor(),
        UInt32(num_used * page_size),
        UInt32(num_used * page_size),
    )
    var key_cache = collection.get_key_cache(0)

    ctx.enqueue_function[
        _populate_kernel[type_of(key_cache), BN, base_alignment, num_pages]
    ](
        key_cache,
        output_buf,
        UInt32(base_kv_row),
        grid_dim=1,
        block_dim=1,
    )

    var output_host = ctx.enqueue_create_host_buffer[.uint32](_MAX_PAGES)
    ctx.enqueue_copy(output_host, output_buf)
    ctx.synchronize()

    # `populate` returns `rows[i] = block_idx * stride` (with `tok_in_block ==
    # 0` on the SIMD path because `base_kv_row` is page-aligned). The K-view
    # `stride = 2 * num_layers * page_size` (PagedKVCache 4D-view of the 6D
    # collection at the K slot — see `PagedKVCache._stride()`). With
    # `num_layers=1` here, `stride = 2 * page_size`. We placed valid block
    # IDs `0..num_used-1` in the LUT and `_SENTINEL` in the padding, so the
    # expected SIMD-loaded LUT values for `populate(base_kv_row)` are
    # `[base_kv_row/page_size, +1, +2, ...]` clipped to valid range. Anything
    # else means the load picked up a sentinel or stale value.
    var stride_expected = UInt32(2 * page_size)
    var first_lut_idx_expected = UInt32(base_kv_row // page_size)
    for i in range(num_pages):
        var raw = output_host[i]
        var lut_value_observed = raw // stride_expected
        var lut_value_expected = first_lut_idx_expected + UInt32(i)
        if lut_value_observed != lut_value_expected:
            raise Error(
                String("populate returned wrong block_idx at rows[")
                + String(i)
                + "]: got "
                + String(lut_value_observed)
                + " (raw="
                + String(raw)
                + "), expected "
                + String(lut_value_expected)
                + " (num_used="
                + String(num_used)
                + ", BN="
                + String(BN)
                + ", page_size="
                + String(page_size)
                + ", base_alignment="
                + String(base_alignment)
                + ", base_kv_row="
                + String(base_kv_row)
                + ")"
            )


def main() raises:
    # Each bazel target picks one `page_size` via `-D page_size=N`; the
    # BN / base_alignment / base_kv_row sweep below covers all
    # production-relevant combinations.
    comptime page_size = get_defined_int["page_size", 128]()

    comptime kv_params = KVCacheStaticParams(num_heads=1, head_size=64)
    comptime dtype = DType.bfloat16

    with DeviceContext() as ctx:
        # BN=64 path (sub-tile when page_size <= 32, single-page when
        # page_size >= 64).
        comptime if 64 % page_size == 0 or page_size >= 64:
            run_one[dtype, kv_params, page_size, 64, 64](0, ctx)
            run_one[dtype, kv_params, page_size, 64, 64](64, ctx)
        # BN=128 path: covers gemma4 local layer's tile size; both
        # base_alignment=128 (gcd(SWA, BN)) and =page_size (chunked-mask
        # contract) — the two values production masks generate.
        comptime if 128 % page_size == 0 or page_size >= 128:
            run_one[dtype, kv_params, page_size, 128, 128](0, ctx)
            run_one[dtype, kv_params, page_size, 128, 128](128, ctx)
            comptime if page_size <= 128:
                run_one[dtype, kv_params, page_size, 128, page_size](0, ctx)
                run_one[dtype, kv_params, page_size, 128, page_size](128, ctx)
        # BN=256 path: covers the larger MLA-style tile; same alignment
        # variations as BN=128.
        comptime if 256 % page_size == 0 or page_size >= 256:
            run_one[dtype, kv_params, page_size, 256, 256](0, ctx)
            run_one[dtype, kv_params, page_size, 256, 256](256, ctx)
            comptime if page_size <= 256:
                run_one[dtype, kv_params, page_size, 256, page_size](0, ctx)
                run_one[dtype, kv_params, page_size, 256, page_size](256, ctx)

        print("PASS")
