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
from std.math import ceildiv
from std.random import random_ui64

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._fillers import random
from std.memory import unsafe_memcpy

from nn.fused_qk_rope import fused_qk_rope_ragged
from std.testing import assert_almost_equal

from std.utils import Index, IndexList


def execute_fused_qk_rope_ragged(
    ctx: DeviceContext,
) raises:
    comptime num_q_heads = 32
    comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    comptime dtype = DType.float32
    comptime num_paged_blocks = 32
    comptime page_size = 128
    var num_layers = 1
    var layer_idx = 0

    comptime max_seq_len = 1024

    var true_ce_prompt_lens = [100, 200, 300, 400]
    var mixed_ce_prompt_lens = [50, 100, 150, 100]

    var true_ce_cache_lens = [0, 0, 0, 0]
    var mixed_ce_cache_lens = [50, 100, 150, 300]

    var batch_size = len(true_ce_prompt_lens)

    var true_ce_total_length = 0
    var mixed_ce_total_length = 0
    var true_ce_max_cache_length = 0
    var mixed_ce_max_cache_length = 0
    var true_ce_max_full_context_length = 0
    var mixed_ce_max_full_context_length = 0
    var true_ce_max_prompt_length = 0
    var mixed_ce_max_prompt_length = 0
    for i in range(batch_size):
        true_ce_max_cache_length = max(
            true_ce_max_cache_length, true_ce_cache_lens[i]
        )
        mixed_ce_max_cache_length = max(
            mixed_ce_max_cache_length, mixed_ce_cache_lens[i]
        )
        true_ce_max_full_context_length = max(
            true_ce_max_full_context_length,
            true_ce_cache_lens[i] + true_ce_prompt_lens[i],
        )
        mixed_ce_max_full_context_length = max(
            mixed_ce_max_full_context_length,
            mixed_ce_cache_lens[i] + mixed_ce_prompt_lens[i],
        )
        true_ce_max_prompt_length = max(
            true_ce_max_prompt_length, true_ce_prompt_lens[i]
        )
        mixed_ce_max_prompt_length = max(
            mixed_ce_max_prompt_length, mixed_ce_prompt_lens[i]
        )
        true_ce_total_length += true_ce_prompt_lens[i]
        mixed_ce_total_length += mixed_ce_prompt_lens[i]

    # Define layouts for LayoutTensor (used for KV cache)
    comptime cache_lengths_layout = Layout.row_major(UNKNOWN_VALUE)
    comptime kv_block_layout = Layout.row_major(
        UNKNOWN_VALUE,
        2,
        UNKNOWN_VALUE,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime paged_lut_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    # Define TileTensor layouts (compile-time static where possible)
    var row_offsets_tile_layout = row_major(batch_size + 1)
    comptime freqs_tile_layout = row_major[max_seq_len, kv_params.head_size]()

    # Create shapes
    var true_ce_row_offsets_shape = Index(batch_size + 1)
    var mixed_ce_row_offsets_shape = Index(batch_size + 1)
    var true_ce_cache_lengths_shape = Index(batch_size)
    var mixed_ce_cache_lengths_shape = Index(batch_size)
    var true_ce_q_ragged_shape = IndexList[3](
        true_ce_total_length, num_q_heads, kv_params.head_size
    )
    var mixed_ce_q_ragged_shape = IndexList[3](
        mixed_ce_total_length, num_q_heads, kv_params.head_size
    )
    var true_ce_output_shape = IndexList[3](
        true_ce_total_length, num_q_heads, kv_params.head_size
    )
    var mixed_ce_output_shape = IndexList[3](
        mixed_ce_total_length, num_q_heads, kv_params.head_size
    )
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var paged_lut_shape = IndexList[2](
        batch_size, ceildiv(true_ce_max_full_context_length, page_size)
    )
    var freqs_shape = IndexList[2](max_seq_len, kv_params.head_size)

    # Create runtime layouts for LayoutTensor
    var true_ce_cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(true_ce_cache_lengths_shape)
    var mixed_ce_cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(mixed_ce_cache_lengths_shape)
    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    # Create device buffers
    var true_ce_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var mixed_ce_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var true_ce_cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
        batch_size
    )
    var mixed_ce_cache_lengths_device = ctx.enqueue_create_buffer[.uint32](
        batch_size
    )
    var true_ce_q_ragged_device = ctx.enqueue_create_buffer[dtype](
        true_ce_q_ragged_shape.flattened_length()
    )
    var mixed_ce_q_ragged_device = ctx.enqueue_create_buffer[dtype](
        mixed_ce_q_ragged_shape.flattened_length()
    )
    var true_ce_output_device = ctx.enqueue_create_buffer[dtype](
        true_ce_output_shape.flattened_length()
    )
    var mixed_ce_output_device = ctx.enqueue_create_buffer[dtype](
        mixed_ce_output_shape.flattened_length()
    )
    var true_ce_kv_block_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var mixed_ce_kv_block_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var paged_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    var freqs_device = ctx.enqueue_create_buffer[dtype](
        freqs_shape.flattened_length()
    )

    # Allocate host pointers for row offsets (need to keep for verification)
    var true_ce_row_offsets_host_ptr = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](batch_size + 1)
    var mixed_ce_row_offsets_host_ptr = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](batch_size + 1)
    var true_ce_cache_lengths_host_ptr = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](batch_size)
    var mixed_ce_cache_lengths_host_ptr = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](batch_size)

    # Initialize row offsets and cache lengths
    var true_ce_offset = 0
    var mixed_ce_offset = 0
    for i in range(batch_size):
        true_ce_row_offsets_host_ptr[i] = UInt32(true_ce_offset)
        mixed_ce_row_offsets_host_ptr[i] = UInt32(mixed_ce_offset)
        true_ce_cache_lengths_host_ptr[i] = UInt32(true_ce_cache_lens[i])
        mixed_ce_cache_lengths_host_ptr[i] = UInt32(mixed_ce_cache_lens[i])
        true_ce_offset += true_ce_prompt_lens[i]
        mixed_ce_offset += mixed_ce_prompt_lens[i]
    true_ce_row_offsets_host_ptr[batch_size] = UInt32(true_ce_offset)
    mixed_ce_row_offsets_host_ptr[batch_size] = UInt32(mixed_ce_offset)

    ctx.enqueue_copy(true_ce_row_offsets_device, true_ce_row_offsets_host_ptr)
    ctx.enqueue_copy(mixed_ce_row_offsets_device, mixed_ce_row_offsets_host_ptr)
    ctx.enqueue_copy(
        true_ce_cache_lengths_device, true_ce_cache_lengths_host_ptr
    )
    ctx.enqueue_copy(
        mixed_ce_cache_lengths_device, mixed_ce_cache_lengths_host_ptr
    )

    # Define runtime layouts for q_ragged and output (these have UNKNOWN_VALUE dims)
    comptime q_ragged_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    comptime output_layout = Layout.row_major(
        UNKNOWN_VALUE, num_q_heads, kv_params.head_size
    )
    var true_ce_q_ragged_runtime_layout = RuntimeLayout[
        q_ragged_layout
    ].row_major(true_ce_q_ragged_shape)
    var mixed_ce_q_ragged_runtime_layout = RuntimeLayout[
        q_ragged_layout
    ].row_major(mixed_ce_q_ragged_shape)
    var true_ce_output_runtime_layout = RuntimeLayout[output_layout].row_major(
        true_ce_output_shape
    )
    var mixed_ce_output_runtime_layout = RuntimeLayout[output_layout].row_major(
        mixed_ce_output_shape
    )

    # Initialize true_ce_q_ragged with random data
    with true_ce_q_ragged_device.map_to_host() as true_ce_q_ragged_host:
        var true_ce_q_ragged_tensor = LayoutTensor[dtype, q_ragged_layout](
            true_ce_q_ragged_host, true_ce_q_ragged_runtime_layout
        )
        random(true_ce_q_ragged_tensor)

        # Initialize mixed_ce_q_ragged by copying from true_ce
        with mixed_ce_q_ragged_device.map_to_host() as mixed_ce_q_ragged_host:
            var mixed_ce_q_ragged_tensor = LayoutTensor[dtype, q_ragged_layout](
                mixed_ce_q_ragged_host, mixed_ce_q_ragged_runtime_layout
            )
            for bs_idx in range(batch_size):
                var mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]
                var true_ce_row_offset = Int(
                    true_ce_row_offsets_host_ptr[bs_idx]
                )
                var mixed_ce_row_offset = Int(
                    mixed_ce_row_offsets_host_ptr[bs_idx]
                )
                var mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

                var true_ce_src_offset = (
                    (true_ce_row_offset + mixed_ce_cache_len)
                    * num_q_heads
                    * kv_params.head_size
                )
                var mixed_ce_dest_offset = (
                    mixed_ce_row_offset * num_q_heads * kv_params.head_size
                )

                unsafe_memcpy(
                    dest=mixed_ce_q_ragged_tensor.ptr + mixed_ce_dest_offset,
                    src=true_ce_q_ragged_tensor.ptr + true_ce_src_offset,
                    count=mixed_ce_prompt_len
                    * num_q_heads
                    * kv_params.head_size,
                )

    # Fill the whole freqs_cis buffer: the kernel reads row
    # `cache_len + token`, not just the first rows. Unwritten rows are zero
    # under the default allocator but NaN under `poison-all`. (KERN-3088)
    comptime freqs_layout = Layout.row_major(max_seq_len, kv_params.head_size)
    var freqs_runtime_layout = RuntimeLayout[freqs_layout].row_major(
        freqs_shape
    )
    with freqs_device.map_to_host() as freqs_host:
        var freqs_init_tensor = LayoutTensor[dtype, freqs_layout](
            freqs_host, freqs_runtime_layout
        )
        random(freqs_init_tensor)

    # Initialize KV blocks with random data using regular host memory
    # (not host-pinned memory via map_to_host) to avoid exhausting
    # the limited host-pinned memory buffer cache
    var kv_block_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var kv_block_host_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_host_ptr.unsafe_ptr(), kv_block_runtime_layout
    )
    random(kv_block_host_tensor)
    ctx.enqueue_copy(true_ce_kv_block_device, kv_block_host_ptr)
    # Copy same data to mixed_ce for consistency
    ctx.enqueue_copy(mixed_ce_kv_block_device, kv_block_host_ptr)
    ctx.synchronize()

    # Initialize paged_lut
    with paged_lut_device.map_to_host() as paged_lut_host:
        var paged_lut_tensor = LayoutTensor[.uint32, paged_lut_layout](
            paged_lut_host, paged_lut_runtime_layout
        )
        var paged_lut_set = Set[Int]()
        for bs in range(batch_size):
            var seq_len = true_ce_cache_lens[bs] + true_ce_prompt_lens[bs]
            for block_idx in range(0, ceildiv(seq_len, page_size)):
                var randval = Int(random_ui64(0, num_paged_blocks - 1))
                while randval in paged_lut_set:
                    randval = Int(random_ui64(0, num_paged_blocks - 1))
                paged_lut_set.add(randval)
                paged_lut_tensor[bs, block_idx] = UInt32(randval)

    # Create TileTensors for row offsets, q_ragged, output, and freqs
    var true_ce_row_offsets_tensor = TileTensor(
        true_ce_row_offsets_device, row_offsets_tile_layout
    )
    var mixed_ce_row_offsets_tensor = TileTensor(
        mixed_ce_row_offsets_device, row_offsets_tile_layout
    )
    var true_ce_q_ragged_tensor = TileTensor(
        true_ce_q_ragged_device,
        row_major(
            (
                true_ce_total_length,
                Idx[num_q_heads],
                Idx[kv_params.head_size],
            )
        ),
    )
    var mixed_ce_q_ragged_tensor = TileTensor(
        mixed_ce_q_ragged_device,
        row_major(
            (
                mixed_ce_total_length,
                Idx[num_q_heads],
                Idx[kv_params.head_size],
            )
        ),
    )
    var true_ce_output_tensor = TileTensor(
        true_ce_output_device,
        row_major(
            (
                true_ce_total_length,
                Idx[num_q_heads],
                Idx[kv_params.head_size],
            )
        ),
    )
    var mixed_ce_output_tensor = TileTensor(
        mixed_ce_output_device,
        row_major(
            (
                mixed_ce_total_length,
                Idx[num_q_heads],
                Idx[kv_params.head_size],
            )
        ),
    )
    var freqs_tensor = TileTensor(freqs_device, freqs_tile_layout)

    # Create LayoutTensors for KV cache (still uses LayoutTensor)
    var true_ce_cache_lengths_tensor = LayoutTensor[
        mut=False, .uint32, Layout(UNKNOWN_VALUE)
    ](
        true_ce_cache_lengths_device,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            true_ce_cache_lengths_shape
        ),
    )
    var mixed_ce_cache_lengths_tensor = LayoutTensor[
        mut=False, .uint32, Layout(UNKNOWN_VALUE)
    ](
        mixed_ce_cache_lengths_device,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
            mixed_ce_cache_lengths_shape
        ),
    )
    var true_ce_kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        true_ce_kv_block_device, kv_block_runtime_layout
    )
    var mixed_ce_kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        mixed_ce_kv_block_device, kv_block_runtime_layout
    )
    var paged_lut_tensor = LayoutTensor[
        mut=False, .uint32, Layout.row_major[2]()
    ](
        paged_lut_device,
        RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape),
    )

    var true_ce_k_cache_collection = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            true_ce_kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                true_ce_kv_block_tensor.runtime_layout.shape.value.canonicalize(),
                true_ce_kv_block_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        true_ce_cache_lengths_tensor,
        paged_lut_tensor,
        UInt32(true_ce_max_prompt_length),
        UInt32(true_ce_max_cache_length),
    )

    var mixed_ce_k_cache_collection = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            mixed_ce_kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                mixed_ce_kv_block_tensor.runtime_layout.shape.value.canonicalize(),
                mixed_ce_kv_block_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        mixed_ce_cache_lengths_tensor,
        paged_lut_tensor,
        UInt32(mixed_ce_max_prompt_length),
        UInt32(mixed_ce_max_cache_length),
    )

    # "true CE" execution
    print("true")
    fused_qk_rope_ragged[
        mixed_ce_k_cache_collection.CacheType, interleaved=False, target="gpu"
    ](
        q_proj=true_ce_q_ragged_tensor,
        input_row_offsets=true_ce_row_offsets_tensor,
        kv_collection=true_ce_k_cache_collection,
        freqs_cis=freqs_tensor,
        position_ids=None,
        layer_idx=UInt32(layer_idx),
        output=true_ce_output_tensor,
        context=ctx,
    )

    # "mixed CE" execution
    print("mixed")
    fused_qk_rope_ragged[
        mixed_ce_k_cache_collection.CacheType, interleaved=False, target="gpu"
    ](
        q_proj=mixed_ce_q_ragged_tensor,
        input_row_offsets=mixed_ce_row_offsets_tensor,
        kv_collection=mixed_ce_k_cache_collection,
        freqs_cis=freqs_tensor,
        position_ids=None,
        layer_idx=UInt32(layer_idx),
        output=mixed_ce_output_tensor,
        context=ctx,
    )

    ctx.synchronize()

    # Verify results
    with mixed_ce_output_device.map_to_host() as mixed_ce_output_host:
        with true_ce_output_device.map_to_host() as true_ce_output_host:
            var mixed_ce_out_tensor = LayoutTensor[dtype, output_layout](
                mixed_ce_output_host, mixed_ce_output_runtime_layout
            )
            var true_ce_out_tensor = LayoutTensor[dtype, output_layout](
                true_ce_output_host, true_ce_output_runtime_layout
            )

            print("comparing Q")
            for bs_idx in range(batch_size):
                var true_ce_batch_start_idx = Int(
                    true_ce_row_offsets_host_ptr[bs_idx]
                )
                var mixed_ce_batch_start_idx = Int(
                    mixed_ce_row_offsets_host_ptr[bs_idx]
                )
                var mixed_ce_cache_len = Int(
                    mixed_ce_cache_lengths_host_ptr[bs_idx]
                )

                for tok_idx in range(mixed_ce_prompt_lens[bs_idx]):
                    for head_idx in range(num_q_heads):
                        for head_dim in range(kv_params.head_size):
                            assert_almost_equal(
                                mixed_ce_out_tensor[
                                    mixed_ce_batch_start_idx + tok_idx,
                                    head_idx,
                                    head_dim,
                                ],
                                true_ce_out_tensor[
                                    true_ce_batch_start_idx
                                    + mixed_ce_cache_len
                                    + tok_idx,
                                    head_idx,
                                    head_dim,
                                ],
                            )

    # Copy KV blocks back to host for K comparison
    ctx.enqueue_copy(kv_block_host_ptr, true_ce_kv_block_device)
    var mixed_kv_block_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    ctx.enqueue_copy(mixed_kv_block_host_ptr, mixed_ce_kv_block_device)

    # Also need paged_lut on host for K cache comparison
    var paged_lut_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    ctx.enqueue_copy(paged_lut_host_ptr, paged_lut_device)
    ctx.synchronize()

    var true_ce_kv_block_host_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_host_ptr.unsafe_ptr(), kv_block_runtime_layout
    )
    var mixed_ce_kv_block_host_tensor = LayoutTensor[dtype, kv_block_layout](
        mixed_kv_block_host_ptr.unsafe_ptr(), kv_block_runtime_layout
    )
    var paged_lut_host_tensor = LayoutTensor[.uint32, paged_lut_layout](
        paged_lut_host_ptr.unsafe_ptr(), paged_lut_runtime_layout
    )

    var true_ce_k_cache_collection_host = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            true_ce_kv_block_host_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                true_ce_kv_block_host_tensor.runtime_layout.shape.value.canonicalize(),
                true_ce_kv_block_host_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        LayoutTensor[mut=False, .uint32, Layout(UNKNOWN_VALUE)](
            true_ce_cache_lengths_host_ptr.unsafe_ptr(),
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                true_ce_cache_lengths_shape
            ),
        ),
        LayoutTensor[mut=False, .uint32, Layout.row_major[2]()](
            paged_lut_host_tensor.ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape),
        ),
        UInt32(true_ce_max_prompt_length),
        UInt32(true_ce_max_cache_length),
    )
    var true_ce_k_cache = true_ce_k_cache_collection_host.get_key_cache(
        layer_idx
    )

    var mixed_ce_k_cache_collection_host = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            mixed_ce_kv_block_host_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                mixed_ce_kv_block_host_tensor.runtime_layout.shape.value.canonicalize(),
                mixed_ce_kv_block_host_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        LayoutTensor[mut=False, .uint32, Layout(UNKNOWN_VALUE)](
            mixed_ce_cache_lengths_host_ptr.unsafe_ptr(),
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                mixed_ce_cache_lengths_shape
            ),
        ),
        LayoutTensor[mut=False, .uint32, Layout.row_major[2]()](
            paged_lut_host_tensor.ptr,
            RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape),
        ),
        UInt32(mixed_ce_max_prompt_length),
        UInt32(mixed_ce_max_cache_length),
    )
    var mixed_ce_k_cache = mixed_ce_k_cache_collection_host.get_key_cache(
        layer_idx
    )

    print("comparing K")
    for bs_idx in range(batch_size):
        var mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        for tok_idx in range(mixed_ce_prompt_lens[bs_idx]):
            for head_idx in range(kv_params.num_heads):
                for head_dim in range(kv_params.head_size):
                    assert_almost_equal(
                        true_ce_k_cache.load[width=1](
                            bs_idx,
                            head_idx,
                            mixed_ce_cache_len + tok_idx,
                            head_dim,
                        ),
                        mixed_ce_k_cache.load[width=1](
                            bs_idx,
                            head_idx,
                            mixed_ce_cache_len + tok_idx,
                            head_dim,
                        ),
                    )

    # Explicitly free device buffers to return memory to the buffer cache
    _ = true_ce_row_offsets_device^
    _ = mixed_ce_row_offsets_device^
    _ = true_ce_cache_lengths_device^
    _ = mixed_ce_cache_lengths_device^
    _ = true_ce_q_ragged_device^
    _ = mixed_ce_q_ragged_device^
    _ = true_ce_output_device^
    _ = mixed_ce_output_device^
    _ = true_ce_kv_block_device^
    _ = mixed_ce_kv_block_device^
    _ = paged_lut_device^
    _ = freqs_device^


# We test the fused_qk_rope_ragged kernel with rope_dim = 64 and q_head_size = 192
# and kv_params.head_size = 576 (shapes are chosen based on Deepseek models).
# For Q, we confirm that the only the last 64 elements in each head are correctly roped,
# and the first 128 elements in each head are simply copied from the input Q tensor.
# For KV cache, we confirm that the only the last 64 elements in each head are correctly roped,
# and the first 512 elements are left unchanged.
def execute_fused_qk_rope_ragged_mla(ctx: DeviceContext) raises:
    comptime num_q_heads = 16
    comptime q_head_size = 192
    comptime kv_params = KVCacheStaticParams(num_heads=1, head_size=576)
    comptime kv_params_64 = KVCacheStaticParams(num_heads=1, head_size=64)
    comptime dtype = DType.bfloat16
    comptime num_paged_blocks = 2
    comptime page_size = 128
    comptime rope_dim = 64
    comptime max_seq_len = 256
    comptime num_layers = 1
    comptime layer_idx = 0

    comptime seq_len = 200
    comptime batch_size = 1

    # Define layouts for LayoutTensor (used for KV cache)
    comptime kv_block_layout = Layout.row_major(
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    comptime kv_block_64_layout = Layout.row_major(
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        rope_dim,
    )
    comptime paged_lut_layout = Layout.row_major(batch_size, 2)
    comptime cache_lengths_layout = Layout.row_major(UNKNOWN_VALUE)

    # Define TileTensor layouts
    comptime q_ragged_tile_layout = row_major[
        seq_len, num_q_heads, q_head_size
    ]()
    comptime q_ragged_64_tile_layout = row_major[
        seq_len, num_q_heads, rope_dim
    ]()
    comptime freqs_tile_layout = row_major[max_seq_len, rope_dim]()
    comptime output_tile_layout = row_major[seq_len, num_q_heads, q_head_size]()
    comptime output_64_tile_layout = row_major[seq_len, num_q_heads, rope_dim]()
    comptime row_offsets_tile_layout = row_major[batch_size + 1]()

    # Create shapes
    var q_ragged_shape = IndexList[3](seq_len, num_q_heads, q_head_size)
    var q_ragged_64_shape = IndexList[3](seq_len, num_q_heads, rope_dim)
    var kv_block_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var kv_block_64_shape = IndexList[6](
        num_paged_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        rope_dim,
    )
    var freqs_shape = IndexList[2](max_seq_len, rope_dim)
    var output_shape = IndexList[3](seq_len, num_q_heads, q_head_size)
    var output_64_shape = IndexList[3](seq_len, num_q_heads, rope_dim)
    var row_offsets_shape = Index(batch_size + 1)
    var paged_lut_shape = IndexList[2](batch_size, 2)
    var cache_lengths_shape = Index(batch_size)

    # Create runtime layouts for LayoutTensor
    var kv_block_runtime_layout = RuntimeLayout[kv_block_layout].row_major(
        kv_block_shape
    )
    var kv_block_64_runtime_layout = RuntimeLayout[
        kv_block_64_layout
    ].row_major(kv_block_64_shape)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )
    var cache_lengths_runtime_layout = RuntimeLayout[
        cache_lengths_layout
    ].row_major(cache_lengths_shape)

    # Create device buffers
    var q_ragged_device = ctx.enqueue_create_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var q_ragged_device_64 = ctx.enqueue_create_buffer[dtype](
        q_ragged_64_shape.flattened_length()
    )
    var kv_block_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var kv_block_device_64 = ctx.enqueue_create_buffer[dtype](
        kv_block_64_shape.flattened_length()
    )
    var freqs_device = ctx.enqueue_create_buffer[dtype](
        freqs_shape.flattened_length()
    )
    var output_device = ctx.enqueue_create_buffer[dtype](
        output_shape.flattened_length()
    )
    var output_device_ref = ctx.enqueue_create_buffer[dtype](
        output_64_shape.flattened_length()
    )
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var paged_lut_device = ctx.enqueue_create_buffer[.uint32](
        paged_lut_shape.flattened_length()
    )
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)

    # Define runtime layouts for q_ragged (used for random initialization)
    comptime q_ragged_layout = Layout.row_major(
        seq_len, num_q_heads, q_head_size
    )
    comptime q_ragged_64_layout = Layout.row_major(
        seq_len, num_q_heads, rope_dim
    )
    var q_ragged_runtime_layout = RuntimeLayout[q_ragged_layout].row_major(
        q_ragged_shape
    )
    var q_ragged_64_runtime_layout = RuntimeLayout[
        q_ragged_64_layout
    ].row_major(q_ragged_64_shape)

    # Allocate host pointer for q_ragged (needed for verification)
    var q_ragged_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        q_ragged_shape.flattened_length()
    )
    var q_ragged_host_tensor = LayoutTensor[dtype, q_ragged_layout](
        q_ragged_host_ptr.unsafe_ptr(), q_ragged_runtime_layout
    )
    random(q_ragged_host_tensor)
    ctx.enqueue_copy(q_ragged_device, q_ragged_host_ptr)

    # Initialize q_ragged_64 by copying last 64 elements from q_ragged
    with q_ragged_device_64.map_to_host() as q_ragged_64_host:
        var q_ragged_64_tensor = LayoutTensor[dtype, q_ragged_64_layout](
            q_ragged_64_host, q_ragged_64_runtime_layout
        )
        for seq_idx in range(seq_len):
            for head_idx in range(num_q_heads):
                var src_offset = (
                    seq_idx * num_q_heads * q_head_size
                    + head_idx * q_head_size
                    + q_head_size
                    - rope_dim
                )
                var dest_offset = (
                    seq_idx * num_q_heads * rope_dim + head_idx * rope_dim
                )
                unsafe_memcpy(
                    dest=q_ragged_64_tensor.ptr + dest_offset,
                    src=q_ragged_host_ptr.unsafe_ptr() + src_offset,
                    count=rope_dim,
                )

    # Allocate host pointer for kv_block (needed for verification)
    var kv_block_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    var kv_block_host_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_host_ptr.unsafe_ptr(), kv_block_runtime_layout
    )
    random(kv_block_host_tensor)
    ctx.enqueue_copy(kv_block_device, kv_block_host_ptr)

    # Initialize kv_block_64 by copying last 64 elements from kv_block
    with kv_block_device_64.map_to_host() as kv_block_64_host:
        var kv_block_64_tensor = LayoutTensor[dtype, kv_block_64_layout](
            kv_block_64_host, kv_block_64_runtime_layout
        )
        for page_idx in range(num_paged_blocks):
            for kv_idx in range(2):
                for layer_idx in range(num_layers):
                    for tok_idx in range(page_size):
                        for head_idx in range(kv_params.num_heads):
                            var src_offset = (
                                page_idx
                                * 2
                                * num_layers
                                * page_size
                                * kv_params.num_heads
                                * Int(kv_params.head_size)
                                + kv_idx
                                * num_layers
                                * page_size
                                * kv_params.num_heads
                                * kv_params.head_size
                                + layer_idx
                                * page_size
                                * kv_params.num_heads
                                * kv_params.head_size
                                + tok_idx
                                * kv_params.num_heads
                                * kv_params.head_size
                                + head_idx * kv_params.head_size
                                + kv_params.head_size
                                - rope_dim
                            )
                            var dest_offset = (
                                page_idx
                                * 2
                                * num_layers
                                * page_size
                                * kv_params.num_heads
                                * rope_dim
                                + kv_idx
                                * num_layers
                                * page_size
                                * kv_params.num_heads
                                * rope_dim
                                + layer_idx
                                * page_size
                                * kv_params.num_heads
                                * rope_dim
                                + tok_idx * kv_params.num_heads * rope_dim
                                + head_idx * rope_dim
                            )
                            unsafe_memcpy(
                                dest=kv_block_64_tensor.ptr + dest_offset,
                                src=kv_block_host_ptr.unsafe_ptr() + src_offset,
                                count=rope_dim,
                            )

    # Initialize freqs_cis with random data
    comptime freqs_layout = Layout.row_major(max_seq_len, rope_dim)
    var freqs_runtime_layout = RuntimeLayout[freqs_layout].row_major(
        freqs_shape
    )
    with freqs_device.map_to_host() as freqs_host:
        var freqs_tensor = LayoutTensor[dtype, freqs_layout](
            freqs_host, freqs_runtime_layout
        )
        random(freqs_tensor)

    # Initialize row_offsets
    with row_offsets_device.map_to_host() as row_offsets_host:
        row_offsets_host[0] = 0
        row_offsets_host[1] = seq_len

    # Initialize paged_lut
    with paged_lut_device.map_to_host() as paged_lut_host:
        paged_lut_host[0] = 0
        paged_lut_host[1] = 1

    # Initialize cache_lengths
    with cache_lengths_device.map_to_host() as cache_lengths_host:
        cache_lengths_host[0] = 0

    var max_prompt_length = Int(seq_len)
    var max_cache_length = Int(0)

    # Create TileTensors for q_ragged, output, row_offsets, and freqs
    var q_ragged_tensor = TileTensor(q_ragged_device, q_ragged_tile_layout)
    var q_ragged_64_tensor = TileTensor(
        q_ragged_device_64, q_ragged_64_tile_layout
    )
    var output_tensor = TileTensor(output_device, output_tile_layout)
    var output_ref_tensor = TileTensor(output_device_ref, output_64_tile_layout)
    var row_offsets_tensor = TileTensor(
        row_offsets_device, row_offsets_tile_layout
    )
    var freqs_tensor = TileTensor(freqs_device, freqs_tile_layout)

    # Create LayoutTensors for KV cache (still uses LayoutTensor)
    var kv_block_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_device, kv_block_runtime_layout
    )
    var kv_block_64_tensor = LayoutTensor[dtype, kv_block_64_layout](
        kv_block_device_64, kv_block_64_runtime_layout
    )
    var cache_lengths_tensor = LayoutTensor[
        mut=False, .uint32, Layout(UNKNOWN_VALUE)
    ](
        cache_lengths_device,
        RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(cache_lengths_shape),
    )
    var paged_lut_tensor = LayoutTensor[
        mut=False, .uint32, Layout.row_major[2]()
    ](
        paged_lut_device,
        RuntimeLayout[Layout.row_major[2]()].row_major(paged_lut_shape),
    )

    var k_cache_collection = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            kv_block_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_tensor.runtime_layout.shape.value.canonicalize(),
                kv_block_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tensor,
        paged_lut_tensor,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )

    var k_cache_collection_64 = PagedKVCacheCollection[
        dtype, kv_params_64, page_size
    ](
        LayoutTensor[dtype, Layout.row_major[6]()](
            kv_block_64_tensor.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_64_tensor.runtime_layout.shape.value.canonicalize(),
                kv_block_64_tensor.runtime_layout.stride.value.canonicalize(),
            ),
        ),
        cache_lengths_tensor,
        paged_lut_tensor,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )

    fused_qk_rope_ragged[
        k_cache_collection.CacheType, interleaved=True, target="gpu"
    ](
        q_proj=q_ragged_tensor,
        input_row_offsets=row_offsets_tensor,
        kv_collection=k_cache_collection,
        freqs_cis=freqs_tensor,
        position_ids=None,
        layer_idx=layer_idx,
        output=output_tensor,
        context=ctx,
    )

    # Execute the kernel for 64
    fused_qk_rope_ragged[
        k_cache_collection_64.CacheType, interleaved=True, target="gpu"
    ](
        q_proj=q_ragged_64_tensor,
        input_row_offsets=row_offsets_tensor,
        kv_collection=k_cache_collection_64,
        freqs_cis=freqs_tensor,
        position_ids=None,
        layer_idx=layer_idx,
        output=output_ref_tensor,
        context=ctx,
    )

    ctx.synchronize()

    # Define output layouts for verification
    comptime output_layout = Layout.row_major(seq_len, num_q_heads, q_head_size)
    comptime output_64_layout = Layout.row_major(seq_len, num_q_heads, rope_dim)
    var output_runtime_layout = RuntimeLayout[output_layout].row_major(
        output_shape
    )
    var output_64_runtime_layout = RuntimeLayout[output_64_layout].row_major(
        output_64_shape
    )

    # Verify Q output
    with output_device.map_to_host() as output_host:
        with output_device_ref.map_to_host() as output_ref_host:
            var out_tensor = LayoutTensor[dtype, output_layout](
                output_host, output_runtime_layout
            )
            var out_ref_tensor = LayoutTensor[dtype, output_64_layout](
                output_ref_host, output_64_runtime_layout
            )

            for seq_idx in range(seq_len):
                for head_idx in range(num_q_heads):
                    # First 128 elements should match input
                    for head_dim_idx in range(q_head_size - rope_dim):
                        assert_almost_equal(
                            out_tensor[seq_idx, head_idx, head_dim_idx],
                            q_ragged_host_tensor[
                                seq_idx, head_idx, head_dim_idx
                            ],
                        )

                    # Last 64 elements should match reference
                    for head_dim_idx in range(rope_dim):
                        assert_almost_equal(
                            out_tensor[
                                seq_idx,
                                head_idx,
                                q_head_size - rope_dim + head_dim_idx,
                            ],
                            out_ref_tensor[seq_idx, head_idx, head_dim_idx],
                        )

    # Verify KV cache
    var kv_block_out_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        kv_block_shape.flattened_length()
    )
    ctx.enqueue_copy(kv_block_out_host_ptr, kv_block_device)
    var kv_block_64_out_host_ptr = ctx.enqueue_create_host_buffer[dtype](
        kv_block_64_shape.flattened_length()
    )
    ctx.enqueue_copy(kv_block_64_out_host_ptr, kv_block_device_64)
    ctx.synchronize()

    var kv_block_out_tensor = LayoutTensor[dtype, kv_block_layout](
        kv_block_out_host_ptr.unsafe_ptr(), kv_block_runtime_layout
    )
    var kv_block_64_out_tensor = LayoutTensor[dtype, kv_block_64_layout](
        kv_block_64_out_host_ptr.unsafe_ptr(), kv_block_64_runtime_layout
    )

    for page_idx in range(num_paged_blocks):
        # Only compare the K cache
        for kv_idx in range(1):
            for layer_idx in range(num_layers):
                for tok_idx in range(page_size):
                    if tok_idx + page_idx * page_size < seq_len:
                        for head_idx in range(kv_params.num_heads):
                            # First 512 elements should match input
                            for head_dim_idx in range(
                                kv_params.head_size - rope_dim
                            ):
                                assert_almost_equal(
                                    kv_block_out_tensor[
                                        page_idx,
                                        kv_idx,
                                        layer_idx,
                                        tok_idx,
                                        head_idx,
                                        head_dim_idx,
                                    ],
                                    kv_block_host_tensor[
                                        page_idx,
                                        kv_idx,
                                        layer_idx,
                                        tok_idx,
                                        head_idx,
                                        head_dim_idx,
                                    ],
                                )
                            # Last 64 elements should match reference
                            for head_dim_idx in range(rope_dim):
                                assert_almost_equal(
                                    kv_block_out_tensor[
                                        page_idx,
                                        kv_idx,
                                        layer_idx,
                                        tok_idx,
                                        head_idx,
                                        kv_params.head_size
                                        - rope_dim
                                        + head_dim_idx,
                                    ],
                                    kv_block_64_out_tensor[
                                        page_idx,
                                        kv_idx,
                                        layer_idx,
                                        tok_idx,
                                        head_idx,
                                        head_dim_idx,
                                    ],
                                )

    # FIXME(MSTDL-2742): HostBuffer is origin incorrect.
    _ = Pointer(to=kv_block_out_host_ptr).as_unsafe_any_origin()[]

    # Explicitly free device buffers to return memory to the buffer cache
    _ = q_ragged_device^
    _ = q_ragged_device_64^
    _ = kv_block_device^
    _ = kv_block_device_64^
    _ = freqs_device^
    _ = output_device^
    _ = output_device_ref^
    _ = row_offsets_device^
    _ = paged_lut_device^
    _ = cache_lengths_device^


def main() raises:
    with DeviceContext() as ctx:
        execute_fused_qk_rope_ragged(ctx)
        execute_fused_qk_rope_ragged_mla(ctx)
