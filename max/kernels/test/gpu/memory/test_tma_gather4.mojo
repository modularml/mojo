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
"""Tests for TMA gather4 instruction (SM100/Blackwell).

Test groups:

1. raw smoke test -- a single `cp_async_bulk_tensor_2d_gather4` call
   to verify the intrinsic works (4 rows, one dtype, minimal code).

2. While-loop tests using actual KV cache types -- constructs real
   `PagedKVCacheCollection` / `ContinuousBatchingKVCacheCollection` objects,
   calls `kv_cache.create_gather4_tma_tile[tile_width](ctx)` to create the
   TMA descriptor, and uses `kv_cache.row_idx(batch, tok)` to compute
   physical row indices for gather4.

   Variants:
   - Paged BF16 (PagedKVCacheCollection)
   - Paged FP8 (PagedKVCacheCollection, float8_e4m3fn)
   - Paged INT64-packed FP8 (PagedKVCacheCollection with dtype=int64,
     simulating the DeepSeek V3.2 576-byte row packing)
   - Contiguous BF16 (ContinuousBatchingKVCacheCollection)

Note: Production dispatch code accesses gather4 through the MHAOperand trait
layer, i.e. ``k.create_gather4_tma_tile[tile_width](ctx)`` where ``k`` is a
``KVCacheMHAOperand``, ``LayoutTensorMHAOperand``, or ``RaggedMHAOperand``.
The MHAOperand implementations delegate to the underlying cache or buffer.
See ``nn/mha_operand.mojo`` for the trait definition and implementations.
"""

from std.math import ceildiv
from std.sys.info import size_of

from max.gpu import block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    TMADescriptor,
    create_tma_descriptor,
    prefetch_tma_descriptor,
)
from max.gpu.memory import (
    cp_async_bulk_tensor_2d_gather4,
    external_memory,
)
from max.gpu.sync import (
    barrier,
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_try_wait_parity_shared,
)
from std.memory import unsafe_stack_allocation
from std.random import rand, randn, seed
from std.utils.index import IndexList

from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile_gather4,
)
from layout.swizzle import make_swizzle
from nn.attention.mha_operand import KVCacheMHAOperand
from std.testing import assert_equal


# ===========================================================================
# GPU kernels
# ===========================================================================


@__llvm_arg_metadata(descriptor, `nvvm.grid_constant`)
def gather4_raw_smoke_kernel[
    dtype: DType, row_width: Int
](
    descriptor: TMADescriptor,
    d_out: MutPointer[Scalar[dtype], MutAnyOrigin],
    row0: Int32,
    row1: Int32,
    row2: Int32,
    row3: Int32,
):
    """Minimal kernel: single gather4 via raw intrinsic, 4 rows."""
    var shmem = external_memory[
        Scalar[dtype],
        address_space=.SHARED,
        alignment=128,
    ]()

    var mbar = unsafe_stack_allocation[1, Int64, address_space=.SHARED]()
    var descriptor_ptr = Pointer(to=descriptor).bitcast[NoneType]()
    mbarrier_init(mbar, 1)

    if thread_idx.x == 0:
        var expected_bytes = 4 * row_width * size_of[dtype]()
        mbarrier_arrive_expect_tx_shared(mbar, Int32(expected_bytes))
        cp_async_bulk_tensor_2d_gather4(
            shmem,
            descriptor_ptr,
            mbar,
            col_idx=Int32(0),
            row0=row0,
            row1=row1,
            row2=row2,
            row3=row3,
        )

    mbarrier_try_wait_parity_shared(mbar, 0, 10_000_000)
    barrier()

    var total_elems = 4 * row_width
    var tid = thread_idx.x
    var num_threads = block_dim.x
    for i in range(tid, total_elems, num_threads):
        d_out[i] = shmem[i]


# ===========================================================================
# Host-side verification
# ===========================================================================


def _verify_gathered_rows[
    dtype: DType, row_width: Int
](
    h_out: ImmPointer[Scalar[dtype], _],
    h_source: ImmPointer[Scalar[dtype], _],
    h_indices: ImmPointer[Int32, _],
    num_rows: Int,
) raises:
    """Verifies gathered output rows match the source buffer at the expected
    physical row offsets."""
    for gather_idx in range(num_rows):
        var src_row = Int(h_indices[gather_idx])
        for col in range(row_width):
            var got = h_out[gather_idx * row_width + col]
            var expected = h_source[src_row * row_width + col]
            assert_equal(
                got.cast[.float32](),
                expected.cast[.float32](),
                msg=String(
                    "Mismatch at gathered row ",
                    gather_idx,
                    " (source row ",
                    src_row,
                    "), col ",
                    col,
                ),
            )


# ===========================================================================
# Host-side test drivers
# ===========================================================================


def test_raw_smoke[
    dtype: DType, row_width: Int, num_tokens: Int
](ctx: DeviceContext) raises:
    """Single gather4 via raw intrinsic -- minimal smoke test."""
    print(
        "== test_raw_smoke [",
        dtype,
        ", row_width=",
        row_width,
        ", num_tokens=",
        num_tokens,
        "]",
    )

    var num_elems = num_tokens * row_width
    var h_data = ctx.enqueue_create_host_buffer[dtype](num_elems)
    rand(h_data.unsafe_ptr(), num_elems)

    var d_data = ctx.enqueue_create_buffer[dtype](num_elems)
    ctx.enqueue_copy(d_data, h_data)

    var tma_desc = create_tma_descriptor[dtype, 2](
        d_data,
        (num_tokens, row_width),
        (row_width, 1),
        (1, row_width),
    )

    var r0 = Int32(3)
    var r1 = Int32(min(500, num_tokens - 1))
    var r2 = Int32(17)
    var r3 = Int32(min(999, num_tokens - 1))

    var output_elems = 4 * row_width
    var h_out = ctx.enqueue_create_host_buffer[dtype](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var shared_mem_bytes = 4 * row_width * size_of[dtype]()

    comptime kernel = gather4_raw_smoke_kernel[dtype, row_width]
    ctx.enqueue_function[kernel](
        tma_desc,
        d_out,
        r0,
        r1,
        r2,
        r3,
        grid_dim=1,
        block_dim=128,
        shared_mem_bytes=shared_mem_bytes,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # Build a 4-element index array for the unified verification helper.
    var h_indices = ctx.enqueue_create_host_buffer[.int32](4)
    h_indices[0] = r0
    h_indices[1] = r1
    h_indices[2] = r2
    h_indices[3] = r3
    _verify_gathered_rows[dtype, row_width](
        h_out.unsafe_ptr(), h_data.unsafe_ptr(), h_indices.unsafe_ptr(), 4
    )
    print("  PASSED: all 4 gathered rows match expected data")

    _ = d_data
    _ = d_out


def _run_paged_gather4_test[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    use_mha_operand: Bool,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext) raises:
    """Shared implementation for paged KV cache gather4 tests.

    When `use_mha_operand` is True, the TMA tile is created through
    KVCacheMHAOperand; otherwise it is created directly from the cache.
    """
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime assert kv_dim == 1 or kv_dim == 2, "kv_dim must be 1 or 2"
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads,
        head_size=head_size,
    )
    comptime row_width = num_heads * head_size

    # Build the 6D blocks tensor.
    # Shape: [num_blocks, kv_dim, num_layers, page_size, num_heads, head_size]
    comptime shape_6d = IndexList[6](
        num_blocks, kv_dim, num_layers, page_size, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    # Fill entire buffer with random data.
    var block_elems = (
        num_blocks * kv_dim * num_layers * page_size * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table with shuffled page assignments.
    comptime lut_layout = Layout.row_major[2]()
    var max_pages_per_seq = (tokens_per_seq + page_size - 1) // page_size
    var lut_managed = ManagedLayoutTensor[.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
        ctx,
    )
    var lut_host = lut_managed.tensor[update=False]()
    var lut_ptr = lut_host.ptr
    for s in range(batch_size):
        for p in range(max_pages_per_seq):
            var blk = ((s * max_pages_per_seq + p) * 37 + 13) % num_blocks
            lut_ptr[s * num_blocks + p] = UInt32(blk)

    # Construct the PagedKVCacheCollection and extract key cache for layer 0.
    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lut_managed.device_tensor(),
        UInt32(tokens_per_seq),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)

    # Create the TMA tile -- either directly or through MHAOperand.
    # The tile type encodes box_width in tile_shape[1]; no need to
    # compute it separately.
    var kv_tile = kv_cache.create_gather4_tma_tile[
        tile_width=row_width, swizzle_mode=swizzle_mode
    ](ctx)
    comptime if use_mha_operand:
        var operand = KVCacheMHAOperand(kv_cache)
        kv_tile = operand.create_gather4_tma_tile[
            tile_width=row_width, swizzle_mode=swizzle_mode
        ](ctx)

    # Build gather indices on host.
    # Physical row = phys_block * stride + offset_in_page
    # where stride = kv_dim * num_layers * page_size.
    comptime paged_stride = kv_dim * num_layers * page_size
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var page_within_seq = tok_idx // page_size
        var offset_in_page = tok_idx % page_size
        var phys_block = Int(lut_ptr[seq_idx * num_blocks + page_within_seq])
        var phys_row = phys_block * paged_stride + offset_in_page
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * row_width
    var h_out = ctx.enqueue_create_host_buffer[dtype](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # Verify by reading expected data directly from the host blocks buffer.
    # The 6D tensor is row-major, so for kv_idx=0, layer_idx=0:
    #   offset(block, tok, h, d) = block*s0 + tok*s3 + h*s4 + d
    # which equals phys_row * row_width + col.
    _verify_gathered_rows[dtype, row_width](
        h_out.unsafe_ptr(), blocks_host.ptr, h_indices.unsafe_ptr(), topk
    )

    comptime if use_mha_operand:
        print(
            "  PASSED: all",
            topk,
            "rows from KVCacheMHAOperand (",
            topk // 4,
            "iterations, page_size=",
            page_size,
            ") match",
        )
    else:
        print(
            "  PASSED: all",
            topk,
            "rows from PagedKVCache (",
            topk // 4,
            "iterations, page_size=",
            page_size,
            ") match",
        )

    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lut_managed


def test_paged_kv_cache[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext) raises:
    """Tests gather4 with a real PagedKVCacheCollection."""
    _run_paged_gather4_test[
        dtype,
        num_heads,
        head_size,
        page_size,
        num_blocks,
        num_layers,
        batch_size,
        tokens_per_seq,
        topk,
        kv_dim,
        use_mha_operand=False,
        swizzle_mode=swizzle_mode,
    ](ctx)


def test_continuous_kv_cache[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    max_seq_len: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext) raises:
    """Tests gather4 with a real ContinuousBatchingKVCacheCollection."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_size
    )
    comptime row_width = num_heads * head_size

    # Build the 6D blocks tensor.
    # Shape: [num_blocks, 2, num_layers, max_seq_len, num_heads, head_size]
    comptime shape_6d = IndexList[6](
        num_blocks, 2, num_layers, max_seq_len, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    # Fill entire buffer with random data.
    var block_elems = (
        num_blocks * 2 * num_layers * max_seq_len * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table (1D: one block per batch entry).
    var lookup_managed = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var lookup_host = lookup_managed.tensor[update=False]()
    for i in range(batch_size):
        lookup_host[i] = UInt32(i)

    # Construct the ContinuousBatchingKVCacheCollection.
    var collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lookup_managed.device_tensor(),
        UInt32(max_seq_len),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)
    var kv_tile = kv_cache.create_gather4_tma_tile[
        tile_width=row_width, swizzle_mode=swizzle_mode
    ](ctx)

    # Build gather indices on host.
    # Physical row = block_id * stride + tok_idx
    # where stride = 2 * num_layers * max_seq_len.
    comptime cont_stride = 2 * num_layers * max_seq_len
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    var lookup_ptr = lookup_host.ptr
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var block_id = Int(lookup_ptr[seq_idx])
        var phys_row = block_id * cont_stride + tok_idx
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * row_width
    var h_out = ctx.enqueue_create_host_buffer[dtype](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, row_width](
        h_out.unsafe_ptr(), blocks_host.ptr, h_indices.unsafe_ptr(), topk
    )
    print(
        "  PASSED: all",
        topk,
        "rows from ContinuousBatchingKVCache (",
        topk // 4,
        "iterations) match",
    )

    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lookup_managed


def test_device_buffer_overload[
    dtype: DType, row_width: Int, num_tokens: Int, topk: Int
](ctx: DeviceContext) raises:
    """Tests the DeviceBuffer overload of create_tma_tile_gather4."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"

    print(
        "== test_device_buffer_overload [",
        dtype,
        ", row_width=",
        row_width,
        ", num_tokens=",
        num_tokens,
        ", topk=",
        topk,
        "]",
    )

    var num_elems = num_tokens * row_width
    var h_data = ctx.enqueue_create_host_buffer[dtype](num_elems)
    rand(h_data.unsafe_ptr(), num_elems)

    var d_data = ctx.enqueue_create_buffer[dtype](num_elems)
    ctx.enqueue_copy(d_data, h_data)

    # Use the DeviceBuffer overload of create_tma_tile_gather4.
    var kv_tile = create_tma_tile_gather4[dtype, tile_width=row_width](
        ctx, d_data, num_tokens
    )

    # Build gather indices on host.
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    for i in range(topk):
        h_indices[i] = Int32((i * 7 + 3) % num_tokens)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch while-loop kernel.
    var output_elems = topk * row_width
    var h_out = ctx.enqueue_create_host_buffer[dtype](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        TensorMapSwizzle.SWIZZLE_NONE,
    ]
    ctx.enqueue_function[kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, row_width](
        h_out.unsafe_ptr(), h_data.unsafe_ptr(), h_indices.unsafe_ptr(), topk
    )
    print("  PASSED: all", topk, "rows from DeviceBuffer overload match")

    _ = d_data
    _ = d_out
    _ = d_indices


def test_mha_operand_gather4[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext) raises:
    """Tests gather4 through the MHAOperand trait via KVCacheMHAOperand."""
    _run_paged_gather4_test[
        dtype,
        num_heads,
        head_size,
        page_size,
        num_blocks,
        num_layers,
        batch_size,
        tokens_per_seq,
        topk,
        kv_dim,
        use_mha_operand=True,
        swizzle_mode=swizzle_mode,
    ](ctx)


@__llvm_arg_metadata(kv_tile, `nvvm.grid_constant`)
def gather4_kernel[
    dtype: DType,
    tile_width: Int,
    tile_rank: Int,
    tile_shape_param: IndexList[tile_rank],
    desc_shape_param: IndexList[tile_rank],
    swizzle_mode: TensorMapSwizzle,
](
    kv_tile: TMATensorTile[
        dtype, tile_rank, tile_shape_param, desc_shape_param
    ],
    d_out: MutPointer[Scalar[dtype], MutAnyOrigin],
    d_indices: MutPointer[Int32, MutAnyOrigin],
    num_tiles: Int32,
):
    """While-loop gather4 loader using the level 3 TMATensorTile API.

    Handles both narrow (tile_width == box_width, loop=1) and wide
    (tile_width > box_width, loop>1) rows via a comptime for-loop
    over column groups.  The box width and number of column groups are
    derived from the tile's compile-time shape ``tile_shape_param[1]``.
    """
    comptime box_width = tile_shape_param[1]
    comptime num_col_groups = ceildiv(tile_width, box_width)
    comptime smem_layout = Layout.row_major(4, box_width)
    var smem_tile = LayoutTensor[
        dtype,
        smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init(1)
        var desc_ptr = Pointer(to=kv_tile.descriptor).bitcast[NoneType]()
        prefetch_tma_descriptor(desc_ptr)
    barrier()

    comptime swizzle = make_swizzle[dtype, swizzle_mode]()

    var tid = thread_idx.x
    var num_threads = block_dim.x
    var phase = UInt32(0)

    var tile_idx: Int32 = 0
    while tile_idx < num_tiles:
        var idx_base = Int(tile_idx) * 4
        var row0 = d_indices[idx_base + 0]
        var row1 = d_indices[idx_base + 1]
        var row2 = d_indices[idx_base + 2]
        var row3 = d_indices[idx_base + 3]

        # Load each column group separately.
        comptime for cg in range(num_col_groups):
            if thread_idx.x == 0:
                mbar[0].expect_bytes(Int32(4 * box_width * size_of[dtype]()))
                kv_tile.async_copy_gather4(
                    smem_tile,
                    mbar[0],
                    col_idx=Int32(cg * box_width),
                    row0=row0,
                    row1=row1,
                    row2=row2,
                    row3=row3,
                )

            mbar[0].wait(phase)

            # Copy from SMEM to output, applying de-swizzle.
            # Output is laid out as [tile_idx * 4 rows, tile_width].
            var out_tile_base = Int(tile_idx) * 4 * tile_width
            for i in range(tid, 4 * box_width, num_threads):
                var row_in_tile = i // box_width
                var col_in_group = i % box_width
                var out_idx = (
                    out_tile_base
                    + row_in_tile * tile_width
                    + cg * box_width
                    + col_in_group
                )
                d_out[out_idx] = smem_tile.ptr[Int(swizzle(i))]

            barrier()
            phase ^= 1

        tile_idx += 1


def test_wide_gather4_device_buffer[
    dtype: DType,
    tile_width: Int,
    num_tokens: Int,
    topk: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests wide gather4 with DeviceBuffer: tile_width > box_width."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"

    print(
        "== test_wide_gather4_device_buffer [",
        dtype,
        ", tile_width=",
        tile_width,
        ", num_tokens=",
        num_tokens,
        ", topk=",
        topk,
        "]",
    )

    var num_elems = num_tokens * tile_width
    var h_data = ctx.enqueue_create_host_buffer[dtype](num_elems)
    rand(h_data.unsafe_ptr(), num_elems)

    var d_data = ctx.enqueue_create_buffer[dtype](num_elems)
    ctx.enqueue_copy(d_data, h_data)

    # Create the TMA tile -- box_width is encoded in tile_shape[1].
    var kv_tile = create_tma_tile_gather4[
        dtype, tile_width=tile_width, swizzle_mode=swizzle_mode
    ](ctx, d_data, num_tokens)

    # Build gather indices.
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    for i in range(topk):
        h_indices[i] = Int32((i * 7 + 3) % num_tokens)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * tile_width
    var h_out = ctx.enqueue_create_host_buffer[dtype](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        tile_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # Verify: output row i should match source row h_indices[i].
    _verify_gathered_rows[dtype, tile_width](
        h_out.unsafe_ptr(), h_data.unsafe_ptr(), h_indices.unsafe_ptr(), topk
    )
    print(
        "  PASSED: all",
        topk,
        "wide rows match expected data",
    )

    _ = d_data
    _ = d_out
    _ = d_indices


def test_wide_gather4_paged_kv[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests wide gather4 through PagedKVCache.create_gather4_tma_tile."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime assert kv_dim == 1 or kv_dim == 2, "kv_dim must be 1 or 2"
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads,
        head_size=head_size,
    )
    comptime tile_width = num_heads * head_size

    print(
        "== test_wide_gather4_paged_kv [",
        dtype,
        ", tile_width=",
        tile_width,
        "]",
    )

    # Build the 6D blocks tensor.
    comptime shape_6d = IndexList[6](
        num_blocks, kv_dim, num_layers, page_size, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    var block_elems = (
        num_blocks * kv_dim * num_layers * page_size * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table.
    comptime lut_layout = Layout.row_major[2]()
    var max_pages_per_seq = (tokens_per_seq + page_size - 1) // page_size
    var lut_managed = ManagedLayoutTensor[.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
        ctx,
    )
    var lut_host = lut_managed.tensor[update=False]()
    var lut_ptr = lut_host.ptr
    for s in range(batch_size):
        for p in range(max_pages_per_seq):
            var blk = ((s * max_pages_per_seq + p) * 37 + 13) % num_blocks
            lut_ptr[s * num_blocks + p] = UInt32(blk)

    # Construct the PagedKVCacheCollection and extract key cache.
    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lut_managed.device_tensor(),
        UInt32(tokens_per_seq),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)

    # Use the unified gather4 API (wide rows handled automatically).
    var kv_tile = kv_cache.create_gather4_tma_tile[
        tile_width=tile_width, swizzle_mode=swizzle_mode
    ](ctx)

    # Build gather indices.
    comptime paged_stride = kv_dim * num_layers * page_size
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var page_within_seq = tok_idx // page_size
        var offset_in_page = tok_idx % page_size
        var phys_block = Int(lut_ptr[seq_idx * num_blocks + page_within_seq])
        var phys_row = phys_block * paged_stride + offset_in_page
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * tile_width
    var h_out = ctx.enqueue_create_host_buffer[dtype](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        tile_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, tile_width](
        h_out.unsafe_ptr(), blocks_host.ptr, h_indices.unsafe_ptr(), topk
    )
    print(
        "  PASSED: all",
        topk,
        "wide rows from PagedKVCache match",
    )

    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lut_managed


def test_wide_gather4_continuous_kv[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    max_seq_len: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests wide gather4 through ContinuousBatchingKVCache.create_gather4_tma_tile.
    """
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads,
        head_size=head_size,
    )
    comptime tile_width = num_heads * head_size

    print(
        "== test_wide_gather4_continuous_kv [",
        dtype,
        ", tile_width=",
        tile_width,
        "]",
    )

    # Build the 6D blocks tensor.
    # Shape: [num_blocks, 2, num_layers, max_seq_len, num_heads, head_size]
    comptime shape_6d = IndexList[6](
        num_blocks, 2, num_layers, max_seq_len, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    var block_elems = (
        num_blocks * 2 * num_layers * max_seq_len * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table (1D: one block per batch entry).
    var lookup_managed = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var lookup_host = lookup_managed.tensor[update=False]()
    for i in range(batch_size):
        lookup_host[i] = UInt32(i)

    # Construct the ContinuousBatchingKVCacheCollection.
    var collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lookup_managed.device_tensor(),
        UInt32(max_seq_len),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)

    # Use the unified gather4 API (wide rows handled automatically).
    var kv_tile = kv_cache.create_gather4_tma_tile[
        tile_width=tile_width, swizzle_mode=swizzle_mode
    ](ctx)

    # Build gather indices.
    comptime cont_stride = 2 * num_layers * max_seq_len
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    var lookup_ptr = lookup_host.ptr
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var block_id = Int(lookup_ptr[seq_idx])
        var phys_row = block_id * cont_stride + tok_idx
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * tile_width
    var h_out = ctx.enqueue_create_host_buffer[dtype](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        tile_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, tile_width](
        h_out.unsafe_ptr(), blocks_host.ptr, h_indices.unsafe_ptr(), topk
    )
    print(
        "  PASSED: all",
        topk,
        "wide rows from ContinuousBatchingKVCache match",
    )

    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lookup_managed


def test_wide_gather4_mha_operand[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests wide gather4 through KVCacheMHAOperand wrapping PagedKVCache."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime assert kv_dim == 1 or kv_dim == 2, "kv_dim must be 1 or 2"
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads,
        head_size=head_size,
    )
    comptime tile_width = num_heads * head_size

    print(
        "== test_wide_gather4_mha_operand [",
        dtype,
        ", tile_width=",
        tile_width,
        "]",
    )

    # Build the 6D blocks tensor.
    comptime shape_6d = IndexList[6](
        num_blocks, kv_dim, num_layers, page_size, num_heads, head_size
    )
    comptime layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, layout_6d](
        RuntimeLayout[layout_6d].row_major(shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    var block_elems = (
        num_blocks * kv_dim * num_layers * page_size * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # Build cache_lengths.
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # Build lookup_table.
    comptime lut_layout = Layout.row_major[2]()
    var max_pages_per_seq = (tokens_per_seq + page_size - 1) // page_size
    var lut_managed = ManagedLayoutTensor[.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
        ctx,
    )
    var lut_host = lut_managed.tensor[update=False]()
    var lut_ptr = lut_host.ptr
    for s in range(batch_size):
        for p in range(max_pages_per_seq):
            var blk = ((s * max_pages_per_seq + p) * 37 + 13) % num_blocks
            lut_ptr[s * num_blocks + p] = UInt32(blk)

    # Construct the PagedKVCacheCollection and extract key cache.
    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lut_managed.device_tensor(),
        UInt32(tokens_per_seq),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)

    # Create TMA tile through MHAOperand trait.
    var operand = KVCacheMHAOperand(kv_cache)
    var kv_tile = operand.create_gather4_tma_tile[
        tile_width=tile_width, swizzle_mode=swizzle_mode
    ](ctx)

    # Build gather indices.
    comptime paged_stride = kv_dim * num_layers * page_size
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    for i in range(topk):
        var seq_idx = i % batch_size
        var tok_idx = (i * 3 + 7) % tokens_per_seq
        var page_within_seq = tok_idx // page_size
        var offset_in_page = tok_idx % page_size
        var phys_block = Int(lut_ptr[seq_idx * num_blocks + page_within_seq])
        var phys_row = phys_block * paged_stride + offset_in_page
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Launch kernel.
    var output_elems = topk * tile_width
    var h_out = ctx.enqueue_create_host_buffer[dtype](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    comptime kernel = gather4_kernel[
        dtype,
        tile_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    _verify_gathered_rows[dtype, tile_width](
        h_out.unsafe_ptr(), blocks_host.ptr, h_indices.unsafe_ptr(), topk
    )
    print(
        "  PASSED: all",
        topk,
        "wide rows from KVCacheMHAOperand match",
    )

    _ = d_indices
    _ = d_out
    _ = blocks
    _ = cache_lengths_managed
    _ = lut_managed


def test_non_divisible_width[
    dtype: DType,
    tile_width: Int,
    num_tokens: Int,
    topk: Int,
    swizzle_mode: TensorMapSwizzle,
](ctx: DeviceContext) raises:
    """Tests gather4 where tile_width is not a multiple of box_width.

    TMA hardware zero-fills out-of-bounds elements in the last column group.
    We verify that in-bounds elements match and out-of-bounds elements are zero.
    """
    comptime assert topk % 4 == 0, "topk must be divisible by 4"

    # Allocate global memory with the actual (non-padded) row width.
    var num_elems = num_tokens * tile_width
    var h_data = ctx.enqueue_create_host_buffer[dtype](num_elems)
    rand(h_data.unsafe_ptr(), num_elems)

    var d_data = ctx.enqueue_create_buffer[dtype](num_elems)
    ctx.enqueue_copy(d_data, h_data)

    # Create the TMA tile -- works even with non-divisible width.
    # box_width and num_col_groups are derived from the tile's type.
    var kv_tile = create_tma_tile_gather4[
        dtype, tile_width=tile_width, swizzle_mode=swizzle_mode
    ](ctx, d_data, num_tokens)

    comptime box_width = type_of(kv_tile).tile_shape[1]
    comptime num_col_groups = ceildiv(tile_width, box_width)
    comptime padded_row_width = num_col_groups * box_width

    print(
        "== test_non_divisible_width [",
        dtype,
        ", tile_width=",
        tile_width,
        ", box_width=",
        box_width,
        ", num_col_groups=",
        num_col_groups,
        ", padded_row_width=",
        padded_row_width,
        ", num_tokens=",
        num_tokens,
        ", topk=",
        topk,
        "]",
    )

    # Build gather indices.
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    for i in range(topk):
        h_indices[i] = Int32((i * 7 + 3) % num_tokens)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # Output uses padded_row_width so the kernel writes full column groups.
    var output_elems = topk * padded_row_width
    var h_out = ctx.enqueue_create_host_buffer[dtype](output_elems)
    var d_out = ctx.enqueue_create_buffer[dtype](output_elems)
    ctx.enqueue_memset(d_out, 0)

    var num_tiles = Int32(topk // 4)

    # Use the gather4_kernel which iterates over column groups.
    # NOTE: we pass padded_row_width as tile_width to the kernel so
    # that its output indexing accounts for the full padded layout.
    comptime kernel = gather4_kernel[
        dtype,
        padded_row_width,
        type_of(kv_tile).rank,
        type_of(kv_tile).tile_shape,
        type_of(kv_tile).desc_shape,
        swizzle_mode,
    ]
    ctx.enqueue_function[kernel](
        kv_tile,
        d_out,
        d_indices,
        num_tiles,
        grid_dim=1,
        block_dim=128,
    )

    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # Verify: for each gathered row, in-bounds columns must match source,
    # and out-of-bounds columns (tile_width..padded_row_width) must
    # be zero (TMA zero-fill).
    for gather_idx in range(topk):
        var src_row = Int(h_indices[gather_idx])
        # Check in-bounds columns.
        for col in range(tile_width):
            var got = h_out[gather_idx * padded_row_width + col]
            var expected = h_data[src_row * tile_width + col]
            assert_equal(
                got.cast[.float32](),
                expected.cast[.float32](),
                msg=String(
                    "Mismatch at gathered row ",
                    gather_idx,
                    " (source row ",
                    src_row,
                    "), col ",
                    col,
                ),
            )
        # Check out-of-bounds columns are zero-filled by TMA.
        for col in range(tile_width, padded_row_width):
            var got = h_out[gather_idx * padded_row_width + col]
            assert_equal(
                got.cast[.float32](),
                Float32(0),
                msg=String(
                    "Expected zero at OOB col ",
                    col,
                    " of gathered row ",
                    gather_idx,
                ),
            )

    print(
        "  PASSED: all",
        topk,
        "rows verified (",
        num_col_groups,
        "col groups, last group zero-fills",
        padded_row_width - tile_width,
        "elements)",
    )

    _ = d_data
    _ = d_out
    _ = d_indices


# ===========================================================================
# Gather4 Tile API Validation Kernel
# ===========================================================================


@__llvm_arg_metadata(g4t_tma, `nvvm.grid_constant`)
def gather4_tile_api_kernel[
    dtype: DType,
    bn: Int,
    cols: Int,
    num_threads: Int,
    tile_rank: Int,
    tile_shape: IndexList[tile_rank],
    desc_shape: IndexList[tile_rank],
](
    g4t_tma: TMATensorTile[dtype, tile_rank, tile_shape, desc_shape],
    d_indices: MutPointer[Int32, MutAnyOrigin],
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
):
    """Loads bn rows via async_copy_gather4_tile, then copies SMEM to
    global output for verification."""
    comptime smem_bytes = bn * cols * size_of[dtype]()
    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()
    var smem_ptr = smem_base.bitcast[Scalar[dtype]]()
    var mbar = (smem_base + smem_bytes).bitcast[SharedMemBarrier]()

    var tid = thread_idx.x
    var elect_one_thread = tid == 0

    if elect_one_thread:
        mbar[0].init()

    barrier()

    if elect_one_thread:
        mbar[0].expect_bytes(g4t_tma.gather4_tile_bytes[cols]())
        g4t_tma.async_copy_gather4_tile[tile_width=cols](
            smem_ptr, mbar[0], d_indices
        )

    barrier()
    mbar[0].wait()
    barrier()

    # Copy SMEM to global output (each thread copies a portion).
    comptime total_elems = bn * cols
    comptime elems_per_thread = total_elems // num_threads
    var base = tid * elems_per_thread
    for i in range(elems_per_thread):
        output[base + i] = smem_ptr[base + i]


def test_gather4_tile_api[
    dtype: DType, bn: Int, cols: Int, total_tokens: Int
](ctx: DeviceContext) raises:
    """Tests async_copy_gather4_tile with a flat DeviceBuffer."""
    comptime num_threads = 128
    comptime smem_bytes = bn * cols * size_of[dtype]()
    comptime metadata_bytes = 16  # mbar (8B) + padding (8B)
    comptime total_smem_bytes = smem_bytes + metadata_bytes

    print(
        "== test_gather4_tile_api [",
        dtype,
        ", bn=",
        bn,
        ", cols=",
        cols,
        ", total_tokens=",
        total_tokens,
        "]",
    )

    # ---- Allocate full K buffer [total_tokens, cols] ----
    seed(42)
    var k_full = ManagedLayoutTensor[
        dtype, Layout.row_major(total_tokens, cols)
    ](ctx)
    var k_full_host = k_full.tensor[update=False]()
    randn[dtype](k_full_host.ptr, total_tokens * cols)

    # ---- Build bn non-contiguous indices ----
    var h_indices = ctx.enqueue_create_host_buffer[.int32](bn)
    for i in range(bn):
        h_indices[i] = Int32((i * 37 + 13) % total_tokens)

    var d_indices = ctx.enqueue_create_buffer[.int32](bn)
    ctx.enqueue_copy(d_indices, h_indices)

    # ---- Build reference K_gathered on host ----
    var k_ref = ctx.enqueue_create_host_buffer[dtype](bn * cols)
    for i in range(bn):
        var src_row = Int(h_indices[i])
        for c in range(cols):
            k_ref[i * cols + c] = k_full_host.ptr[src_row * cols + c]

    # ---- Allocate output buffer ----
    var out_device = ctx.enqueue_create_buffer[dtype](bn * cols)

    # ---- Create gather4 TMA tile with tile_height=bn, SWIZZLE_NONE ----
    _ = k_full.device_tensor()
    var g4t_tma = create_tma_tile_gather4[
        dtype, tile_height=bn, tile_width=cols
    ](ctx, k_full.device_data.value(), total_tokens)

    # ---- Launch kernel ----
    comptime kernel = gather4_tile_api_kernel[
        dtype,
        bn,
        cols,
        num_threads,
        type_of(g4t_tma).rank,
        type_of(g4t_tma).tile_shape,
        type_of(g4t_tma).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        g4t_tma,
        d_indices,
        out_device,
        grid_dim=(1, 1),
        block_dim=num_threads,
        shared_mem_bytes=total_smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(total_smem_bytes)
        ),
    )
    ctx.synchronize()

    # ---- Verify output matches reference ----
    var out_host = ctx.enqueue_create_host_buffer[dtype](bn * cols)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var max_err: Float32 = 0.0
    var mismatches = 0
    for r in range(bn):
        for c in range(cols):
            var got = out_host[r * cols + c]
            var expected = k_ref[r * cols + c]
            if got != expected:
                var err = abs(Float32(got) - Float32(expected)) / max(
                    abs(Float32(expected)), Float32(1.0)
                )
                if err > max_err:
                    max_err = err
                if err > 0.001:
                    mismatches += 1
                    if mismatches <= 5:
                        print(
                            "  MISMATCH r=",
                            r,
                            " c=",
                            c,
                            " got=",
                            got,
                            " expected=",
                            expected,
                        )

    if mismatches > 0:
        print("  FAILED:", mismatches, "mismatches, max_err=", max_err)
    else:
        print("  gather4_tile_api max_err:", max_err)
        print("  gather4_tile_api PASSED")

    # Validate byte count helper.
    comptime expected_bytes = bn * cols * size_of[dtype]()
    var actual_bytes = g4t_tma.gather4_tile_bytes[cols]()
    if Int(actual_bytes) != expected_bytes:
        print(
            "  gather4_tile_bytes MISMATCH: got=",
            actual_bytes,
            " expected=",
            expected_bytes,
        )
    else:
        print("  gather4_tile_bytes correct:", actual_bytes, "bytes")

    _ = out_device
    _ = d_indices
    _ = k_full^


def test_gather4_tile_api_paged[
    dtype: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    tokens_per_seq: Int,
    topk: Int,
    kv_dim: Int,
](ctx: DeviceContext) raises:
    """Tests async_copy_gather4_tile with a paged KV cache."""
    comptime assert topk % 4 == 0, "topk must be divisible by 4"
    comptime assert kv_dim == 1 or kv_dim == 2, "kv_dim must be 1 or 2"
    comptime row_width = num_heads * head_size
    comptime num_threads = 128
    comptime smem_bytes = topk * row_width * size_of[dtype]()
    comptime metadata_bytes = 16  # mbar (8B) + padding (8B)
    comptime total_smem_bytes = smem_bytes + metadata_bytes
    comptime paged_stride = kv_dim * num_layers * page_size

    print(
        "== test_gather4_tile_api_paged [",
        dtype,
        ", row_width=",
        row_width,
        ", topk=",
        topk,
        ", page_size=",
        page_size,
        "]",
    )

    # ---- Build the 6D blocks tensor ----
    comptime pg_shape_6d = IndexList[6](
        num_blocks, kv_dim, num_layers, page_size, num_heads, head_size
    )
    comptime pg_layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[dtype, pg_layout_6d](
        RuntimeLayout[pg_layout_6d].row_major(pg_shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()
    seed(42)
    var block_elems = (
        num_blocks * kv_dim * num_layers * page_size * num_heads * head_size
    )
    rand[dtype](blocks_host.ptr, block_elems)

    # ---- Build cache_lengths ----
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(tokens_per_seq)

    # ---- Build lookup_table ----
    comptime lut_layout = Layout.row_major[2]()
    var max_pages_per_seq = (tokens_per_seq + page_size - 1) // page_size
    var lut_managed = ManagedLayoutTensor[.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
        ctx,
    )
    var lut_host = lut_managed.tensor[update=False]()
    var lut_ptr = lut_host.ptr
    for s in range(batch_size):
        for p in range(max_pages_per_seq):
            var blk = ((s * max_pages_per_seq + p) * 37 + 13) % num_blocks
            lut_ptr[s * num_blocks + p] = UInt32(blk)

    # ---- Construct PagedKVCacheCollection ----
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads,
        head_size=head_size,
    )
    var collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lut_managed.device_tensor(),
        UInt32(tokens_per_seq),
        UInt32(tokens_per_seq),
    )
    var kv_cache = collection.get_key_cache(0)

    # ---- Build gather indices from the paged cache ----
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    for i in range(topk):
        var tok_idx = (i * 37 + 13) % tokens_per_seq
        var page_within_seq = tok_idx // page_size
        var offset_in_page = tok_idx % page_size
        var phys_block = Int(lut_ptr[0 * num_blocks + page_within_seq])
        var phys_row = phys_block * paged_stride + offset_in_page
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # ---- Build reference K_gathered on host ----
    var k_ref = ctx.enqueue_create_host_buffer[dtype](topk * row_width)
    for i in range(topk):
        var src_row = Int(h_indices[i])
        for c in range(row_width):
            k_ref[i * row_width + c] = blocks_host.ptr[src_row * row_width + c]

    # ---- Allocate output buffer ----
    var out_device = ctx.enqueue_create_buffer[dtype](topk * row_width)

    # ---- Create gather4 TMA tile with tile_height=topk from paged cache ----
    var g4t_tma = kv_cache.create_gather4_tma_tile[
        tile_height=topk, tile_width=row_width
    ](ctx)

    # ---- Launch kernel ----
    comptime kernel = gather4_tile_api_kernel[
        dtype,
        topk,
        row_width,
        num_threads,
        type_of(g4t_tma).rank,
        type_of(g4t_tma).tile_shape,
        type_of(g4t_tma).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        g4t_tma,
        d_indices,
        out_device,
        grid_dim=(1, 1),
        block_dim=num_threads,
        shared_mem_bytes=total_smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(total_smem_bytes)
        ),
    )
    ctx.synchronize()

    # ---- Verify output matches reference ----
    var out_host = ctx.enqueue_create_host_buffer[dtype](topk * row_width)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var max_err: Float32 = 0.0
    var mismatches = 0
    for r in range(topk):
        for c in range(row_width):
            var got = out_host[r * row_width + c]
            var expected = k_ref[r * row_width + c]
            if got != expected:
                var err = abs(Float32(got) - Float32(expected)) / max(
                    abs(Float32(expected)), Float32(1.0)
                )
                if err > max_err:
                    max_err = err
                if err > 0.001:
                    mismatches += 1
                    if mismatches <= 5:
                        print(
                            "  MISMATCH r=",
                            r,
                            " c=",
                            c,
                            " got=",
                            got,
                            " expected=",
                            expected,
                        )

    if mismatches > 0:
        print("  FAILED:", mismatches, "mismatches, max_err=", max_err)
    else:
        print("  gather4_tile_api_paged max_err:", max_err)
        print("  gather4_tile_api_paged PASSED")

    _ = out_device
    _ = d_indices
    _ = blocks^
    _ = cache_lengths_managed^
    _ = lut_managed^


# ===========================================================================
# Main
# ===========================================================================


def main() raises:
    seed()

    with DeviceContext() as ctx:
        # Level 2 raw smoke test.
        print("--- Kernel 1: Level 2 raw smoke test ---")
        test_raw_smoke[.bfloat16, 128, 1024](ctx)

        # DeviceBuffer overload of create_tma_tile_gather4.
        print(
            "\n--- DeviceBuffer overload of create_tma_tile_gather4:"
            " bfloat16 ---"
        )
        test_device_buffer_overload[
            DType.bfloat16,
            128,  # row_width
            1024,  # num_tokens
            64,  # topk
        ](ctx)

        # Paged KV cache tests.
        print("\n--- Paged KV Cache (PagedKVCacheCollection): bfloat16 ---")
        test_paged_kv_cache[
            DType.bfloat16,
            1,  # num_heads
            256,  # head_size (row_width = 1*256 = 256)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
        ](ctx)

        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): bfloat16,"
            " kv_dim=2 ---"
        )
        test_paged_kv_cache[
            DType.bfloat16,
            1,  # num_heads
            256,  # head_size (row_width = 1*256 = 256)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            64,  # topk
            2,  # kv_dim (standard separate K/V layout)
        ](ctx)

        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): float8_e4m3fn ---"
        )
        test_paged_kv_cache[
            DType.float8_e4m3fn,
            1,  # num_heads
            128,  # head_size (row_width = 128)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
        ](ctx)

        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): int64-packed"
            " FP8 (72 int64 = 576 bytes) ---"
        )
        test_paged_kv_cache[
            DType.int64,
            1,  # num_heads
            72,  # head_size (72 INT64 = 576 bytes)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
        ](ctx)

        # Contiguous KV cache test.
        print(
            "\n--- Contiguous KV Cache"
            " (ContinuousBatchingKVCacheCollection): bfloat16 ---"
        )
        test_continuous_kv_cache[
            DType.bfloat16,
            1,  # num_heads
            256,  # head_size (row_width = 256)
            512,  # max_seq_len
            4,  # num_blocks (one per batch entry)
            1,  # num_layers
            4,  # batch_size
            256,  # tokens_per_seq
            64,  # topk
        ](ctx)

        # Swizzled paged KV cache tests (FlashMLA-style swizzle modes).
        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): bfloat16,"
            " SWIZZLE_128B ---"
        )
        test_paged_kv_cache[
            DType.bfloat16,
            1,  # num_heads
            64,  # head_size (row_width = 64, 128 bytes = SWIZZLE_128B)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        print(
            "\n--- Paged KV Cache (PagedKVCacheCollection): float8_e4m3fn,"
            " SWIZZLE_64B ---"
        )
        test_paged_kv_cache[
            DType.float8_e4m3fn,
            1,  # num_heads
            64,  # head_size (row_width = 64, 64 bytes = SWIZZLE_64B)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # MHAOperand trait layer test.
        print(
            "\n--- MHAOperand trait layer"
            " (KVCacheMHAOperand wrapping PagedKVCache): bfloat16 ---"
        )
        test_mha_operand_gather4[
            DType.bfloat16,
            1,  # num_heads
            256,  # head_size (row_width = 1*256 = 256)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
        ](ctx)

        # Wide gather4 tests: tile_width > box_width with swizzle.
        print(
            "\n--- Wide gather4: DeviceBuffer, bfloat16,"
            " tile_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_device_buffer[
            DType.bfloat16,
            512,  # tile_width
            1024,  # num_tokens
            16,  # topk (4 tiles of 4 rows)
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        print(
            "\n--- Wide gather4: DeviceBuffer, float8_e4m3fn,"
            " tile_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_device_buffer[
            DType.float8_e4m3fn,
            512,  # tile_width (512 bytes; box_width = 128)
            1024,  # num_tokens
            16,  # topk
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        print(
            "\n--- Wide gather4: DeviceBuffer, bfloat16,"
            " tile_width=256, SWIZZLE_64B ---"
        )
        test_wide_gather4_device_buffer[
            DType.bfloat16,
            256,  # tile_width
            1024,  # num_tokens
            16,  # topk
            TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # Wide gather4 through PagedKVCache.
        print(
            "\n--- Wide gather4: PagedKVCache, bfloat16,"
            " tile_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_paged_kv[
            DType.bfloat16,
            1,  # num_heads
            512,  # head_size (tile_width = 1*512 = 512)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            16,  # topk
            1,  # kv_dim
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        # --- Test 1: BF16 + SWIZZLE_64B + tile_width=64 (box=32, 2 col groups).
        print(
            "\n--- Paged KV Cache: bfloat16, SWIZZLE_64B,"
            " tile_width=64 (box=32, 2 col groups) ---"
        )
        test_wide_gather4_paged_kv[
            DType.bfloat16,
            1,  # num_heads
            64,  # head_size (tile_width = 1*64 = 64)
            128,  # page_size
            16,  # num_blocks
            1,  # num_layers
            4,  # batch_size
            384,  # tokens_per_seq
            16,  # topk
            1,  # kv_dim
            TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # --- Test 2: FP8 + SWIZZLE_64B + wide via DeviceBuffer (tile_width=256).
        print(
            "\n--- Wide gather4: DeviceBuffer, float8_e4m3fn,"
            " tile_width=256, SWIZZLE_64B ---"
        )
        test_wide_gather4_device_buffer[
            DType.float8_e4m3fn,
            256,  # tile_width (box=64, 4 col groups)
            1024,  # num_tokens
            16,  # topk
            TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        # --- Test 3: Wide via ContinuousBatchingKVCache.
        print(
            "\n--- Wide gather4: ContinuousBatchingKVCache, bfloat16,"
            " tile_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_continuous_kv[
            DType.bfloat16,
            1,  # num_heads
            512,  # head_size (tile_width = 1*512 = 512)
            512,  # max_seq_len
            4,  # num_blocks (one per batch entry)
            1,  # num_layers
            4,  # batch_size
            256,  # tokens_per_seq
            16,  # topk
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        # --- Test 4: Wide via MHAOperand.
        print(
            "\n--- Wide gather4: KVCacheMHAOperand, bfloat16,"
            " tile_width=512, SWIZZLE_128B ---"
        )
        test_wide_gather4_mha_operand[
            DType.bfloat16,
            1,  # num_heads
            512,  # head_size (tile_width = 1*512 = 512)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            2,  # batch_size
            256,  # tokens_per_seq
            16,  # topk
            1,  # kv_dim
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        # Non-divisible width test: tile_width=520, box_width=64,
        # ceildiv(520, 64)=9 col groups, last group extends 56 elements past
        # the boundary. TMA hardware zero-fills those 56 elements.
        # NOTE: tile_width must be 8-element aligned for BF16 with
        # swizzle modes (16-byte stride alignment required by TMA hardware).
        print(
            "\n--- Non-divisible width: DeviceBuffer, bfloat16,"
            " tile_width=520, SWIZZLE_128B ---"
        )
        test_non_divisible_width[
            DType.bfloat16,
            520,  # tile_width (not a multiple of 64, but 8-aligned)
            1024,  # num_tokens
            16,  # topk
            TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        # Gather4 tile API tests (SWIZZLE_NONE, flat buffer and paged).
        print("\n--- Gather4 tile API: flat buffer, bf16, SWIZZLE_NONE ---")
        test_gather4_tile_api[
            DType.bfloat16,
            64,  # bn
            64,  # cols (64 BF16 = 128 bytes)
            256,  # total_tokens
        ](ctx)

        print("\n--- Gather4 tile API: paged KV cache, bf16, SWIZZLE_NONE ---")
        test_gather4_tile_api_paged[
            DType.bfloat16,
            1,  # num_heads
            64,  # head_size (tile_width = 1*64 = 64)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            1,  # batch_size
            256,  # tokens_per_seq
            64,  # topk
            1,  # kv_dim
        ](ctx)
