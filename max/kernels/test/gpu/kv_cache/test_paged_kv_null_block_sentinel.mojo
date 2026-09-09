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

"""Regression test: LUT tail-padding sentinel `total_num_pages` is in-bounds.

`PagedKVCacheManager` must fill the LUT with `total_num_pages` (N), not
`0xCCCCCCCC`. The `populate` SIMD path multiplies every LUT entry —
including over-read tail-padding — by `page_stride`. With the wrong fill,
that multiplication produced a GPU address hundreds of GB past the buffer
(`CUDA_ERROR_ILLEGAL_ADDRESS`, SERVOPT-1456). With fill=N and an
`(N+1)`-page allocation, `N * stride` lands on the null block page, which
is always in-bounds.

This test verifies that contract directly on GPU: `populate` on a dummy
null-block row returns `N * stride` for every entry, and that value lies
within the `(N+1)`-page buffer.

Complements the Python-layer regression tests for the KV cache manager.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import unsafe_memset_zero
from std.utils import IndexList

from layout import Layout, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from kv_cache_test_utils import padded_lut_cols


def _null_block_populate_kernel[
    cache_t: KVCacheT,
    BN: Int,
    base_alignment: Int,
    num_pages: Int,
](
    kv: cache_t,
    output_ptr: MutPointer[UInt32, MutAnyOrigin],
    base_kv_row: UInt32,
):
    """Single-thread kernel: call populate on a dummy (null-block) row."""
    if global_idx.x != 0:
        return
    var rows = kv.populate[BN, base_alignment](
        batch_idx=UInt32(0), base_kv_row=base_kv_row
    )
    comptime for i in range(num_pages):
        output_ptr[i] = rows.rows[i]


def run_null_block_test[
    dtype: DType,
    kv_params: KVCacheStaticParams,
    page_size: Int,
    BN: Int,
    base_alignment: Int,
](ctx: DeviceContext) raises:
    """Verify `populate` with null-block sentinel fill produces in-bounds row
    indices.

    Sets up N real pages + 1 null block page (total N+1 allocated). Fills the
    LUT with N (the null block index) for all columns — mimicking what
    `PagedKVCacheManager` does for dummy/padding requests after the fix.
    Asserts that every returned `rows[i]` equals `N * stride`, and that this
    value is strictly less than `(N+1) * pages_bytes` (i.e., in-bounds).
    """
    # Use a small pool: N=4 real pages + 1 null block = 5 allocated slots.
    var total_num_real_pages = 4  # = N; null block is at index N
    var total_allocated = total_num_real_pages + 1  # N+1

    comptime eff_page = page_size if 0 < page_size < BN else BN
    comptime num_pages = BN // eff_page

    # LUT columns: fill all with `total_num_real_pages` (= N, the null block
    # index). This models both the tail-padding fill and a dummy request whose
    # only real entry is the null block.
    var lut_columns = padded_lut_cols(total_num_real_pages + 4)
    comptime lut_layout = Layout.row_major[2]()
    var lut_shape = IndexList[2](1, lut_columns)
    var lut_runtime = RuntimeLayout[lut_layout].row_major(lut_shape)
    var lut = ManagedLayoutTensor[.uint32, lut_layout](lut_runtime, ctx)
    var lut_host = lut.tensor[update=False]()
    for c in range(lut_columns):
        # Fill every column with N — the null block index.
        lut_host[0, c] = UInt32(total_num_real_pages)

    # cache_lengths: set to page_size so `base_kv_row=0` is in-cache.
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](1)
    var cache_lengths_runtime = RuntimeLayout[cache_lengths_layout].row_major(
        cache_lengths_shape
    )
    var cache_lengths = ManagedLayoutTensor[.uint32, cache_lengths_layout](
        cache_lengths_runtime, ctx
    )
    var cache_lengths_host = cache_lengths.tensor[update=False]()
    cache_lengths_host[0] = UInt32(page_size)

    # Allocate N+1 pages worth of block storage (null block at slot N).
    comptime blocks_layout = Layout.row_major[6]()
    var blocks_shape = IndexList[6](
        total_allocated,  # N+1 slots — null block at index N
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

    comptime _MAX_PAGES = 16
    var output_buf = ctx.enqueue_create_buffer[.uint32](_MAX_PAGES)
    var output_init = ctx.enqueue_create_host_buffer[.uint32](_MAX_PAGES)
    for i in range(_MAX_PAGES):
        output_init[i] = UInt32(0xDEADBEEF)
    ctx.enqueue_copy(output_buf, output_init)

    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths.device_tensor(),
        lut.device_tensor(),
        UInt32(page_size),
        UInt32(page_size),
    )
    var key_cache = collection.get_key_cache(0)

    ctx.enqueue_function[
        _null_block_populate_kernel[
            type_of(key_cache), BN, base_alignment, num_pages
        ]
    ](
        key_cache,
        output_buf,
        UInt32(0),  # base_kv_row=0 (first token position)
        grid_dim=1,
        block_dim=1,
    )

    var output_host = ctx.enqueue_create_host_buffer[.uint32](_MAX_PAGES)
    ctx.enqueue_copy(output_host, output_buf)
    ctx.synchronize()

    # stride = 2 * num_layers * page_size (K-view of the 6D block tensor with
    # num_layers=1). Expected row index = N * stride (null block offset).
    var stride = UInt32(2 * page_size)
    var expected_row = UInt32(total_num_real_pages) * stride
    # Buffer holds N+1 pages, so the null block at N*stride is in-bounds if:
    #   N * stride < (N+1) * stride  (always true for N >= 0)
    var buffer_end = UInt32(total_allocated) * stride

    for i in range(num_pages):
        var got = output_host[i]
        # Every entry should equal N * stride (null block offset).
        if got != expected_row:
            raise Error(
                String("populate[rows[")
                + String(i)
                + "]] = "
                + String(got)
                + ", expected null-block offset "
                + String(expected_row)
                + " (BN="
                + String(BN)
                + " page_size="
                + String(page_size)
                + ")"
            )
        # Null block offset must be within the N+1 allocation.
        if got >= buffer_end:
            raise Error(
                String("populate returned out-of-bounds row index ")
                + String(got)
                + " >= buffer_end "
                + String(buffer_end)
                + " (CUDA_ERROR_ILLEGAL_ADDRESS regression, SERVOPT-1456)"
            )


def main() raises:
    comptime kv_params = KVCacheStaticParams(num_heads=1, head_size=64)
    comptime dtype = DType.bfloat16

    with DeviceContext() as ctx:
        # page_size=128 is the primary production value for Kimi K2.5.
        run_null_block_test[dtype, kv_params, 128, 128, 128](ctx)
        run_null_block_test[dtype, kv_params, 128, 256, 128](ctx)
        # Also cover page_size=64 (Gemma/other models).
        run_null_block_test[dtype, kv_params, 64, 64, 64](ctx)
        run_null_block_test[dtype, kv_params, 64, 128, 64](ctx)

        print("PASS")
