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

from std.collections import Set
from std.random import random_ui64, seed
from std.sys import size_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, thread_idx
from max.gpu.memory import fence_async_view_proxy
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    lt_to_tt,
    row_major,
    UNKNOWN_VALUE,
)
from layout._fillers import random
from layout.tma_async import SharedMemBarrier, TMATensorTile, _idx_product
from std.memory import unsafe_stack_allocation
from nn.attention.mha_operand import (
    KVCacheMHAOperand,
    MHAOperand,
    LayoutTensorMHAOperand,
    RaggedMHAOperand,
)
from nn.attention.gpu.nvidia.sm90.attention import kv_coord
from std.testing import assert_equal

from std.utils import IndexList


@__llvm_arg_metadata(src_tma_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(dst_tma_tile, `nvvm.grid_constant`)
def mha_operand_tma_copy_kernel[
    rank: Int,
    tile_shape: IndexList[rank],
    desc_shape: IndexList[rank],
    smem_layout: Layout,
    kv_t: MHAOperand,
    swizzle_mode: TensorMapSwizzle,
    head_size: Int,
](
    src_tma_tile: TMATensorTile[
        kv_t.dtype,
        rank,
        tile_shape,
        desc_shape,
    ],
    dst_tma_tile: TMATensorTile[
        kv_t.dtype,
        rank,
        tile_shape,
        desc_shape,
    ],
    src_operand: kv_t,
    dst_operand: kv_t,
):
    # Map block indices to MHA parameters
    var batch_idx = UInt32(block_idx.z)
    var head_idx = UInt32(block_idx.y)

    var num_keys = src_operand.cache_length(Int(batch_idx))

    # Allocate shared memory tile
    var smem_tile = LayoutTensor[
        kv_t.dtype,
        smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    # Initialize barrier
    ref mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()[0]

    if thread_idx.x == 0:
        mbar.init()

    var phase: UInt32 = 0

    # Calculate col coordinates
    # Declare row coordinates

    comptime tile_m = tile_shape[0]
    comptime elements = _idx_product[rank, tile_shape]()
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[kv_t.dtype]()
    # Loop over columns to copy full head size
    for kv_tile_start_row in range(0, num_keys, tile_m):
        if thread_idx.x == 0:
            var src_row = src_operand.row_idx(
                batch_idx, UInt32(kv_tile_start_row)
            )
            mbar.expect_bytes(Int32(elements * size_of[kv_t.dtype]()))

            # Initiate TMA load
            src_tma_tile.async_copy(
                smem_tile,
                mbar,
                kv_coord[depth=head_size](src_row, head_idx),
            )

        # Synchronize all threads
        barrier()
        mbar.wait(phase)
        phase ^= 1

        # Ensure data is visible before store
        barrier()

        # Ensures all previous shared memory stores are completed.
        fence_async_view_proxy()
        # Store to destination
        if thread_idx.x == 0:
            var dst_row = dst_operand.row_idx(
                batch_idx, UInt32(kv_tile_start_row)
            )

            # Initiate TMA store
            dst_tma_tile.async_store(
                smem_tile,
                kv_coord[depth=head_size](dst_row, head_idx),
            )

            dst_tma_tile.commit_group()
            dst_tma_tile.wait_group()


def mha_operand_copy[
    kv_t: MHAOperand,
    //,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
    *,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    ctx: DeviceContext, src: kv_t, dst: kv_t, batch_size: Int, max_seq_len: Int
) raises:
    # Each kernel call copies the entire array, iterating over `tile_m` chunks
    # along the rows. `head_size` is the continugous axis.
    # For `Q @ K'`, `is_k_major` is True.
    # For `P @ V`, `is_k_major` is False.
    # We emulate both, using the appropriate smem layout for the (wg/u)mma instructions.
    comptime head_size = kv_params.head_size

    # Create TMA tiles
    var src_tma = src.create_tma_tile[swizzle_mode, BN=tile_m, depth=head_size](
        ctx
    )
    var dst_tma = dst.create_tma_tile[swizzle_mode, BN=tile_m, depth=head_size](
        ctx
    )

    # Calculate grid dimensions
    # NOTE: In context encoding, we would have grid_x = ceildiv(max_prompt_len, BM)
    # Each of these, as well as `q_num_heads // kv_num_heads` represent multicast
    # opportunities.
    var grid_x = 1
    comptime grid_y = kv_params.num_heads
    var grid_z = batch_size

    comptime kernel = mha_operand_tma_copy_kernel[
        type_of(src_tma).rank,
        type_of(src_tma).tile_shape,
        type_of(src_tma).desc_shape,
        Layout.row_major(type_of(src_tma).tile_shape),
        kv_t,
        swizzle_mode,
        head_size,
    ]

    # Launch kernel with block_dim=32
    ctx.enqueue_function[kernel](
        src_tma,
        dst_tma,
        src,
        dst,
        grid_dim=(grid_x, grid_y, grid_z),
        block_dim=(32,),
    )

    ctx.synchronize()


def test_mha_host_operand[
    kv_t: MHAOperand,
    //,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
](src: kv_t, dst: kv_t, batch_size: Int) raises:
    """Test function that compares two MHAOperands using block_paged_ptr."""
    comptime kv_row_stride = kv_params.head_size * kv_params.num_heads
    # Iterate over all batch entries and tokens
    for b in range(batch_size):
        var seq_len = src.cache_length(b)
        for s in range(0, seq_len, tile_m):
            var actual_tokens = min(tile_m, seq_len - s)
            for h in range(kv_params.num_heads):
                # Get pointers using block_paged_ptr
                var src_ptr = src.block_paged_ptr[tile_m](
                    UInt32(b), UInt32(s), UInt32(h), UInt32(0)
                )
                var dst_ptr = dst.block_paged_ptr[tile_m](
                    UInt32(b), UInt32(s), UInt32(h), UInt32(0)
                )

                # Compare values for the actual number of tokens
                for tok in range(actual_tokens):
                    for hd in range(kv_params.head_size):
                        var offset = tok * kv_row_stride + hd
                        var src_val = src_ptr[offset]
                        var dst_val = dst_ptr[offset]
                        if src_val != dst_val:
                            print(b, s, h, tok, hd, src_val, dst_val)
                        assert_equal(src_val, dst_val)


def test_continuous_kv_cache[
    dtype: DType,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
](
    ctx: DeviceContext, batch_size: Int, max_seq_len: Int, num_layers: Int
) raises:
    comptime msg = "  Testing ContinuousBatchingKVCache with tile_m=" + String(
        tile_m
    ) + ", head_size=" + String(kv_params.head_size)
    print(msg)

    # Initialize cache blocks
    var num_blocks = batch_size + 2
    var dyn_shape = IndexList[6](
        num_blocks,
        2,  # key and value
        num_layers,
        max_seq_len,
        kv_params.num_heads,
        kv_params.head_size,
    )

    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        dyn_shape
    )

    # Create device buffer for kv blocks
    var kv_block_device = ctx.enqueue_create_buffer[dtype](
        dyn_shape.flattened_length()
    )

    # Initialize with random data
    with kv_block_device.map_to_host() as kv_block_host:
        var kv_block_host_tensor = LayoutTensor[dtype, kv_block_layout](
            kv_block_host, kv_block_runtime_layout
        )
        random(kv_block_host_tensor)

    # Set up lookup table and cache lengths
    comptime lookup_layout = Layout(UNKNOWN_VALUE)
    var lookup_shape = IndexList[1](batch_size)
    var lookup_runtime_layout = RuntimeLayout[lookup_layout].row_major(
        lookup_shape
    )

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)

    with lookup_table_device.map_to_host() as lookup_host:
        for i in range(batch_size):
            lookup_host[i] = UInt32(i)

    with cache_lengths_device.map_to_host() as cache_lengths_host:
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(
                max_seq_len // 2
            )  # Half filled caches

    # Create source collection on device
    var src_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            kv_block_device, kv_block_runtime_layout
        ),
        LayoutTensor[.uint32, lookup_layout, ImmutAnyOrigin](
            cache_lengths_device, lookup_runtime_layout
        ),
        LayoutTensor[.uint32, lookup_layout, ImmutAnyOrigin](
            lookup_table_device, lookup_runtime_layout
        ),
        UInt32(max_seq_len),
        UInt32(max_seq_len),
    )

    var src_key = KVCacheMHAOperand(src_collection.get_key_cache(0))

    # Create destination buffer
    var dst_block_device = ctx.enqueue_create_buffer[dtype](
        dyn_shape.flattened_length()
    )

    # Initialize destination with zeros
    with dst_block_device.map_to_host() as dst_host:
        for i in range(len(dst_host)):
            dst_host[i] = 0

    var dst_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            dst_block_device, kv_block_runtime_layout
        ),
        LayoutTensor[.uint32, lookup_layout, ImmutAnyOrigin](
            cache_lengths_device, lookup_runtime_layout
        ),
        LayoutTensor[.uint32, lookup_layout, ImmutAnyOrigin](
            lookup_table_device, lookup_runtime_layout
        ),
        UInt32(max_seq_len),
        UInt32(max_seq_len),
    )
    var dst_key = KVCacheMHAOperand(dst_collection.get_key_cache(0))

    mha_operand_copy[tile_m, kv_params](
        ctx,
        src_key,
        dst_key,
        batch_size,
        max_seq_len,
    )

    # Verify results by comparing on host
    with kv_block_device.map_to_host() as src_host:
        with dst_block_device.map_to_host() as dst_host:
            with lookup_table_device.map_to_host() as lookup_host:
                with cache_lengths_device.map_to_host() as cache_lengths_host:
                    var src_host_tensor = LayoutTensor[dtype, kv_block_layout](
                        src_host, kv_block_runtime_layout
                    )
                    var dst_host_tensor = LayoutTensor[dtype, kv_block_layout](
                        dst_host, kv_block_runtime_layout
                    )
                    var lookup_host_tensor = LayoutTensor[
                        .uint32, lookup_layout
                    ](lookup_host, lookup_runtime_layout)
                    var cache_lengths_host_tensor = LayoutTensor[
                        .uint32, lookup_layout
                    ](cache_lengths_host, lookup_runtime_layout)

                    var src_host_collection = ContinuousBatchingKVCacheCollection[
                        dtype, kv_params
                    ](
                        src_host_tensor.as_unsafe_any_origin(),
                        cache_lengths_host_tensor.as_unsafe_any_origin().as_imm(),
                        lookup_host_tensor.as_unsafe_any_origin().as_imm(),
                        UInt32(max_seq_len),
                        UInt32(max_seq_len),
                    )
                    var dst_host_collection = ContinuousBatchingKVCacheCollection[
                        dtype, kv_params
                    ](
                        dst_host_tensor.as_unsafe_any_origin(),
                        cache_lengths_host_tensor.as_unsafe_any_origin().as_imm(),
                        lookup_host_tensor.as_unsafe_any_origin().as_imm(),
                        UInt32(max_seq_len),
                        UInt32(max_seq_len),
                    )

                    var src_host_key = KVCacheMHAOperand(
                        src_host_collection.get_key_cache(0)
                    )
                    var dst_host_key = KVCacheMHAOperand(
                        dst_host_collection.get_key_cache(0)
                    )

                    test_mha_host_operand[tile_m, kv_params](
                        src_host_key, dst_host_key, batch_size
                    )

    print("    ContinuousBatchingKVCache test passed!")


def test_paged_kv_cache[
    dtype: DType,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    ctx: DeviceContext, batch_size: Int, max_seq_len: Int, num_layers: Int
) raises:
    comptime msg = "  Testing PagedKVCache with tile_m=" + String(
        tile_m
    ) + ", head_size=" + String(kv_params.head_size)
    print(msg)

    # Calculate number of pages needed
    var pages_per_seq = (max_seq_len + page_size - 1) // page_size
    var num_blocks = batch_size * pages_per_seq + 10  # Extra blocks

    # Initialize paged cache blocks
    var dyn_shape = IndexList[6](
        num_blocks,
        2,  # key and value
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )

    comptime kv_block_layout = Layout.row_major[6]()
    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        dyn_shape
    )

    # Create device buffer for kv blocks
    var kv_block_device = ctx.enqueue_create_buffer[dtype](
        dyn_shape.flattened_length()
    )

    # Initialize with random data
    with kv_block_device.map_to_host() as kv_block_host:
        var kv_block_host_tensor = LayoutTensor[dtype, kv_block_layout](
            kv_block_host, kv_block_runtime_layout
        )
        random(kv_block_host_tensor)

    # Set up page lookup table
    comptime paged_lut_layout = Layout.row_major[2]()
    var paged_lut_shape = IndexList[2](batch_size, pages_per_seq)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    var paged_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )

    var paged_lut_set = Set[Int]()
    with paged_lut_device.map_to_host() as paged_lut_host:
        var paged_lut_tensor = LayoutTensor[.uint32, paged_lut_layout](
            paged_lut_host, paged_lut_runtime_layout
        )
        for bs in range(batch_size):
            for page_idx in range(pages_per_seq):
                var block_idx = Int(random_ui64(0, UInt64(num_blocks - 1)))
                while block_idx in paged_lut_set:
                    block_idx = Int(random_ui64(0, UInt64(num_blocks - 1)))
                paged_lut_set.add(block_idx)
                paged_lut_tensor[bs, page_idx] = UInt32(block_idx)

    # Set up cache lengths
    comptime cache_lengths_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_shape = IndexList[1](batch_size)
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)

    with cache_lengths_device.map_to_host() as cache_lengths_host:
        for i in range(batch_size):
            cache_lengths_host[i] = UInt32(max_seq_len // 2)  # Half filled

    # Create source collection on device
    var src_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            kv_block_device, kv_block_runtime_layout
        ),
        LayoutTensor[.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_device, cache_lengths_runtime_layout
        ),
        LayoutTensor[.uint32, paged_lut_layout, ImmutAnyOrigin](
            paged_lut_device, paged_lut_runtime_layout
        ),
        UInt32(max_seq_len),
        UInt32(max_seq_len),
    )
    var src_key = KVCacheMHAOperand(src_collection.get_key_cache(0))

    # Create destination buffer
    var dst_block_device = ctx.enqueue_create_buffer[dtype](
        dyn_shape.flattened_length()
    )

    # Initialize destination with zeros
    with dst_block_device.map_to_host() as dst_host:
        for i in range(len(dst_host)):
            dst_host[i] = 0

    var dst_collection = PagedKVCacheCollection[dtype, kv_params, page_size](
        LayoutTensor[dtype, kv_block_layout, MutAnyOrigin](
            dst_block_device, kv_block_runtime_layout
        ),
        LayoutTensor[.uint32, cache_lengths_layout, ImmutAnyOrigin](
            cache_lengths_device, cache_lengths_runtime_layout
        ),
        LayoutTensor[.uint32, paged_lut_layout, ImmutAnyOrigin](
            paged_lut_device, paged_lut_runtime_layout
        ),
        UInt32(max_seq_len),
        UInt32(max_seq_len),
    )
    var dst_key = KVCacheMHAOperand(dst_collection.get_key_cache(0))

    mha_operand_copy[tile_m, kv_params](
        ctx,
        src_key,
        dst_key,
        batch_size,
        max_seq_len,
    )

    # Verify results by comparing on host
    with kv_block_device.map_to_host() as src_host:
        with dst_block_device.map_to_host() as dst_host:
            with paged_lut_device.map_to_host() as paged_lut_host:
                with cache_lengths_device.map_to_host() as cache_lengths_host:
                    var src_host_tensor = LayoutTensor[dtype, kv_block_layout](
                        src_host, kv_block_runtime_layout
                    )
                    var dst_host_tensor = LayoutTensor[dtype, kv_block_layout](
                        dst_host, kv_block_runtime_layout
                    )
                    var paged_lut_host_tensor = LayoutTensor[
                        .uint32, paged_lut_layout
                    ](paged_lut_host, paged_lut_runtime_layout)
                    var cache_lengths_host_tensor = LayoutTensor[
                        .uint32, cache_lengths_layout
                    ](cache_lengths_host, cache_lengths_runtime_layout)

                    var src_host_collection = PagedKVCacheCollection[
                        dtype, kv_params, page_size
                    ](
                        src_host_tensor.as_unsafe_any_origin(),
                        cache_lengths_host_tensor.as_unsafe_any_origin().as_imm(),
                        paged_lut_host_tensor.as_unsafe_any_origin().as_imm(),
                        UInt32(max_seq_len),
                        UInt32(max_seq_len),
                    )
                    var dst_host_collection = PagedKVCacheCollection[
                        dtype, kv_params, page_size
                    ](
                        dst_host_tensor.as_unsafe_any_origin(),
                        cache_lengths_host_tensor.as_unsafe_any_origin().as_imm(),
                        paged_lut_host_tensor.as_unsafe_any_origin().as_imm(),
                        UInt32(max_seq_len),
                        UInt32(max_seq_len),
                    )

                    var src_host_key = KVCacheMHAOperand(
                        src_host_collection.get_key_cache(0)
                    )
                    var dst_host_key = KVCacheMHAOperand(
                        dst_host_collection.get_key_cache(0)
                    )

                    test_mha_host_operand[tile_m, kv_params](
                        src_host_key, dst_host_key, batch_size
                    )

    print("    PagedKVCache test passed!")


def test_layout_tensor[
    dtype: DType,
    tile_m: Int,
    kv_params: KVCacheStaticParams,
](ctx: DeviceContext, batch_size: Int, max_seq_len: Int) raises:
    comptime msg = "  Testing LayoutTensor with tile_m=" + String(
        tile_m
    ) + ", head_size=" + String(kv_params.head_size)
    print(msg)

    # Create source and destination buffers with BSHD layout
    comptime num_heads = kv_params.num_heads
    comptime head_size = kv_params.head_size
    var total_elems = batch_size * max_seq_len * num_heads * head_size

    # Create device buffer for source
    var src_device = ctx.enqueue_create_buffer[dtype](total_elems)

    # Initialize with random data
    with src_device.map_to_host() as src_host:
        var src_host_tt = TileTensor(
            src_host,
            row_major(
                (
                    batch_size,
                    max_seq_len,
                    Idx[num_heads],
                    Idx[head_size],
                )
            ),
        )
        random(src_host_tt)

    # Create MHAOperand for source
    var src_tt = TileTensor(
        src_device,
        row_major(
            (
                batch_size,
                max_seq_len,
                Idx[num_heads],
                Idx[head_size],
            )
        ),
    )
    var src_operand = LayoutTensorMHAOperand(
        src_tt.as_immut().as_unsafe_any_origin()
    )

    for is_k_major in range(2):
        # Create destination buffer
        var dst_device = ctx.enqueue_create_buffer[dtype](total_elems)

        # Initialize destination with zeros
        with dst_device.map_to_host() as dst_host:
            for i in range(len(dst_host)):
                dst_host[i] = 0

        var dst_tt = TileTensor(
            dst_device,
            row_major(
                (
                    batch_size,
                    max_seq_len,
                    Idx[num_heads],
                    Idx[head_size],
                )
            ),
        )
        var dst_operand = LayoutTensorMHAOperand(
            dst_tt.as_immut().as_unsafe_any_origin()
        )

        mha_operand_copy[tile_m, kv_params](
            ctx,
            src_operand,
            dst_operand,
            batch_size,
            max_seq_len,
        )

        # Verify results by comparing on host
        with src_device.map_to_host() as src_host:
            with dst_device.map_to_host() as dst_host:
                var src_host_tt = TileTensor(
                    src_host,
                    row_major(
                        (
                            batch_size,
                            max_seq_len,
                            Idx[num_heads],
                            Idx[head_size],
                        )
                    ),
                )
                var dst_host_tt = TileTensor(
                    dst_host,
                    row_major(
                        (
                            batch_size,
                            max_seq_len,
                            Idx[num_heads],
                            Idx[head_size],
                        )
                    ),
                )

                var src_host_operand = LayoutTensorMHAOperand(
                    lt_to_tt(
                        src_host_tt.to_layout_tensor().as_unsafe_any_origin()
                    )
                )
                var dst_host_operand = LayoutTensorMHAOperand(
                    lt_to_tt(
                        dst_host_tt.to_layout_tensor().as_unsafe_any_origin()
                    )
                )

                test_mha_host_operand[tile_m, kv_params](
                    src_host_operand, dst_host_operand, batch_size
                )

    print("    LayoutTensor test passed!")


def test_ragged[
    dtype: DType, tile_m: Int, kv_params: KVCacheStaticParams
](ctx: DeviceContext, batch_size: Int) raises:
    comptime msg = "  Testing RaggedTensor with tile_m=" + String(
        tile_m
    ) + ", head_size=" + String(kv_params.head_size)
    print(msg)

    # Create variable length sequences
    var total_tokens = 0

    # Create cache row offsets
    var cache_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )

    # First pass: calculate sequence lengths and total tokens
    var seq_lens = List[Int](capacity=batch_size)
    for _ in range(batch_size):
        var seq_len = Int(random_ui64(100, 500))
        seq_lens.append(seq_len)
        total_tokens += seq_len

    # Initialize offsets on device
    with cache_row_offsets_device.map_to_host() as offsets_host:
        var offset = 0
        for i in range(batch_size):
            offsets_host[i] = UInt32(offset)
            offset += seq_lens[i]
        offsets_host[batch_size] = UInt32(offset)

    # Create ragged buffers
    comptime num_heads = kv_params.num_heads
    comptime head_size = kv_params.head_size
    var total_elems = total_tokens * num_heads * head_size

    # Create device buffer for source
    var src_device = ctx.enqueue_create_buffer[dtype](total_elems)

    # Initialize with random data
    with src_device.map_to_host() as src_host:
        var src_host_tt = TileTensor(
            src_host,
            row_major(
                (
                    total_tokens,
                    Idx[num_heads],
                    Idx[head_size],
                )
            ),
        )
        random(src_host_tt)

    # Create MHAOperand for source
    var src_tt = TileTensor(
        src_device,
        row_major(
            (
                total_tokens,
                Idx[num_heads],
                Idx[head_size],
            )
        ),
    )
    var cro_tt = TileTensor(
        cache_row_offsets_device, row_major(Coord(batch_size + 1))
    )
    var src_operand = RaggedMHAOperand(
        src_tt.as_unsafe_any_origin(),
        cro_tt.as_unsafe_any_origin(),
    )

    # Find max sequence length for grid calculation
    var max_seq_len = 0
    for i in range(batch_size):
        max_seq_len = max(max_seq_len, seq_lens[i])

    # Create destination buffer
    var dst_device = ctx.enqueue_create_buffer[dtype](total_elems)

    # Initialize destination with zeros
    with dst_device.map_to_host() as dst_host:
        for i in range(len(dst_host)):
            dst_host[i] = 0

    var dst_tt = TileTensor(
        dst_device,
        row_major(
            (
                total_tokens,
                Idx[num_heads],
                Idx[head_size],
            )
        ),
    )
    var dst_operand = RaggedMHAOperand(
        dst_tt.as_unsafe_any_origin(),
        cro_tt.as_unsafe_any_origin(),
    )

    mha_operand_copy[tile_m, kv_params](
        ctx,
        src_operand,
        dst_operand,
        batch_size,
        max_seq_len,
    )

    # Verify results by comparing on host
    with src_device.map_to_host() as src_host:
        with dst_device.map_to_host() as dst_host:
            with cache_row_offsets_device.map_to_host() as offsets_host:
                var src_host_tt = TileTensor(
                    src_host,
                    row_major(
                        (
                            total_tokens,
                            Idx[num_heads],
                            Idx[head_size],
                        )
                    ),
                )
                var dst_host_tt = TileTensor(
                    dst_host,
                    row_major(
                        (
                            total_tokens,
                            Idx[num_heads],
                            Idx[head_size],
                        )
                    ),
                )
                var offsets_host_tt = TileTensor(
                    offsets_host, row_major(Coord(batch_size + 1))
                )

                var src_host_operand = RaggedMHAOperand(
                    src_host_tt.as_unsafe_any_origin(),
                    offsets_host_tt.as_unsafe_any_origin(),
                )
                var dst_host_operand = RaggedMHAOperand(
                    dst_host_tt.as_unsafe_any_origin(),
                    offsets_host_tt.as_unsafe_any_origin(),
                )

                test_mha_host_operand[tile_m, kv_params](
                    src_host_operand, dst_host_operand, batch_size
                )

    print("    RaggedTensor test passed!")


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        comptime batch_size = 4
        comptime max_seq_len = 1024
        comptime num_layers = 2
        comptime page_size = 512
        comptime dtype = DType.bfloat16

        print("Testing TMA copy with different tile configurations")

        comptime for i in range(6, 9):
            comptime head_size = 1 << i  # 64, 128, 256
            comptime kv_params = KVCacheStaticParams(
                num_heads=8, head_size=head_size
            )

            comptime for j in range(6, 15 - i):
                comptime block_m = 1 << j  # 64, ..., (64 * 256) // block_m

                comptime msg = "\nTesting block_m=" + String(
                    block_m
                ) + ", head_size=" + String(head_size)
                print(msg)

                test_continuous_kv_cache[dtype, block_m, kv_params](
                    ctx, batch_size, max_seq_len, num_layers
                )
                test_paged_kv_cache[dtype, block_m, kv_params, page_size](
                    ctx,
                    batch_size,
                    max_seq_len,
                    num_layers,
                )
                test_layout_tensor[dtype, block_m, kv_params](
                    ctx, batch_size, max_seq_len
                )
                test_ragged[dtype, block_m, kv_params](ctx, batch_size)
