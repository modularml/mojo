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

from std.math import ceildiv, rsqrt
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int
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
from nn.attention.mha_mask import CausalMask, MHAMask, SlidingWindowCausalMask
from std.testing import assert_almost_equal, assert_equal
from std.sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator

from std.utils import IndexList


def execute_ragged_flash_attention[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
    mask_t: MHAMask,
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    mask: mask_t,
    ctx: DeviceContext,
) raises:
    # TODO(KERN-2666): float32 + depth=256 exceeds shared memory on H100/B200.
    comptime if dtype == .float32 and kv_params.head_size == 256:
        return

    comptime page_size = get_defined_int["page_size", 256]()

    var batch_size = len(valid_lengths)
    assert len(valid_lengths) == len(
        cache_lengths
    ), "expected valid_lengths and cache_lengths size to be equal"

    # Compute dimensions
    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]

    # Define layouts
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

    # Create shapes
    var row_offsets_shape = IndexList[1](batch_size + 1)
    var cache_lengths_shape = IndexList[1](batch_size)
    var q_ragged_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )
    var output_shape = IndexList[3](
        total_length, num_q_heads, kv_params.head_size
    )

    # Create runtime layouts
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

    # Create device buffers
    var input_row_offsets = ManagedLayoutTensor[.uint32, row_offsets_layout](
        row_offsets_runtime_layout,
        ctx,
    )
    var cache_lengths_managed = ManagedLayoutTensor[
        .uint32, cache_lengths_layout
    ](
        cache_lengths_runtime_layout,
        ctx,
    )
    var q_ragged = ManagedLayoutTensor[dtype, q_ragged_layout](
        q_ragged_runtime_layout, ctx
    )
    var test_output = ManagedLayoutTensor[dtype, output_layout](
        output_runtime_layout, ctx
    )
    var ref_output = ManagedLayoutTensor[dtype, output_layout](
        output_runtime_layout, ctx
    )

    # Populate metadata tensors from host views.
    var input_row_offsets_host = input_row_offsets.tensor[update=False]()
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()

    var running_offset: UInt32 = 0
    for i in range(batch_size):
        input_row_offsets_host[i] = running_offset
        cache_lengths_host[i] = UInt32(cache_lengths[i])
        running_offset += UInt32(valid_lengths[i])
    input_row_offsets_host[batch_size] = running_offset

    # Initialize q_ragged with random data
    var q_ragged_tensor = q_ragged.tensor()
    random(q_ragged_tensor)

    var num_continuous_blocks = batch_size + 2
    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size
    )

    # KV block shapes
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
    # Pad LUT inner dim to honor `PagedKVCache.populate`'s SIMD padding
    # invariant — see `padded_lut_cols`.
    var paged_lut_shape = IndexList[2](
        batch_size,
        padded_lut_cols(ceildiv(max_full_context_length, page_size)),
    )

    # KV block runtime layouts
    var kv_block_continuous_runtime_layout = RuntimeLayout[
        kv_block_6d_layout
    ].row_major(kv_block_continuous_shape)
    var kv_block_paged_runtime_layout = RuntimeLayout[
        kv_block_6d_layout
    ].row_major(kv_block_paged_shape)
    var paged_lut_runtime_layout = RuntimeLayout[paged_lut_layout].row_major(
        paged_lut_shape
    )

    # Build managed tensors for continuous and paged KV state.
    var kv_block_continuous = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        kv_block_continuous_runtime_layout, ctx
    )
    var kv_block_paged = ManagedLayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged_runtime_layout, ctx
    )
    var lookup_table = ManagedLayoutTensor[.uint32, lookup_table_layout](
        lookup_table_runtime_layout,
        ctx,
    )
    var paged_lut = ManagedLayoutTensor[.uint32, paged_lut_layout](
        paged_lut_runtime_layout,
        ctx,
    )

    # Initialize kv_block_continuous with random data
    var kv_block_continuous_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_continuous.tensor[update=False]().ptr,
        kv_block_continuous_runtime_layout,
    )
    random(kv_block_continuous_tensor)
    var lookup_table_host = lookup_table.tensor[update=False]()

    # Assign each batch entry a distinct continuous block. `random_ui64` is
    # inclusive, so the original draw range `[0, num_continuous_blocks - 1]`
    # is a population of `num_continuous_blocks` blocks.
    var continuous_blocks = random_distinct(num_continuous_blocks, batch_size)
    for idx in range(batch_size):
        lookup_table_host[idx] = UInt32(continuous_blocks[idx])

    # Create LayoutTensors for KV collection
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

    # Initialize paged KV cache by copying from continuous
    var kv_block_paged_tensor = LayoutTensor[dtype, kv_block_6d_layout](
        kv_block_paged.tensor[update=False]().ptr,
        kv_block_paged_runtime_layout,
    )
    var paged_lut_tensor = paged_lut.tensor[update=False]()

    # Sample one distinct paged block per page across the whole batch up
    # front, then hand them out in iteration order. Total pages needed is the
    # sum of per-sequence page counts and is <= num_paged_blocks by
    # construction.
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
                # Calculate offsets manually for the 6D tensors
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

    # Create LayoutTensors for paged KV collection
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

    # Create LayoutTensors for flash attention calls
    var q_ragged_lt = q_ragged.device_tensor()
    var ref_output_lt = ref_output.device_tensor()
    var test_output_lt = test_output.device_tensor()

    # continuous execution
    flash_attention[ragged=True](
        ref_output_lt,
        q_ragged_lt,
        kv_collection_continuous_device.get_key_cache(layer_idx),
        kv_collection_continuous_device.get_value_cache(layer_idx),
        mask,
        input_row_offsets.device_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )

    # paged execution
    flash_attention[ragged=True](
        test_output_lt,
        q_ragged_lt,
        kv_collection_paged_device.get_key_cache(layer_idx),
        kv_collection_paged_device.get_value_cache(layer_idx),
        mask,
        input_row_offsets.device_tensor(),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    # Catch NaN/Inf before the tolerance comparison — gives a clear, named
    # failure if either path produces non-finite output.
    assert_no_nan_inf(ref_output, "ref_output_continuous")
    assert_no_nan_inf(test_output, "test_output_paged")

    # Fetch host views for verification.
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
                        assert_almost_equal(
                            ref_out[ragged_offset + s, h, hd],
                            test_out[ragged_offset + s, h, hd],
                            atol=1e-2,
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

    # check reproducibility
    for repeat in range(16):
        flash_attention[ragged=True](
            ref_output_lt,
            q_ragged_lt,
            kv_collection_paged_device.get_key_cache(layer_idx),
            kv_collection_paged_device.get_value_cache(layer_idx),
            mask,
            input_row_offsets.device_tensor(),
            rsqrt(Float32(kv_params.head_size)),
            ctx,
        )
        ref_out = ref_output.tensor()
        input_row_offsets_tensor = input_row_offsets.tensor()
        for bs in range(batch_size):
            var prompt_len = valid_lengths[bs]
            var ragged_offset = Int(input_row_offsets_tensor[bs])
            for s in range(prompt_len):
                for h in range(num_q_heads):
                    for d in range(kv_params.head_size):
                        var rep = ref_out[ragged_offset + s, h, d]
                        var orig = test_out[ragged_offset + s, h, d]
                        if rep != orig:
                            print("repeat s h d =", repeat, s, h, d)
                        assert_equal(rep, orig)
                        ref_out[ragged_offset + s, h, d] = 123.4567


def execute_flash_attention_suite[
    num_q_heads: Int,
    kv_params: KVCacheStaticParams,
    mask_t: MHAMask,
](mask: mask_t, ctx: DeviceContext) raises:
    comptime types = (DType.bfloat16,)

    for bs in [1, 4, 16]:
        comptime for type_idx in range(len(types)):
            comptime type = rebind[DType](types[type_idx])
            var ce_cache_sizes = List[Int]()
            var ce_seq_lens = List[Int]()
            var tg_cache_sizes = List[Int]()
            var tg_seq_lens = List[Int]()

            for _ in range(bs):
                tg_seq_lens.append(1)
                tg_cache_sizes.append(Int(random_ui64(1, 1024)))
                ce_seq_lens.append(Int(random_ui64(1, 1024)))
                ce_cache_sizes.append(0)

            print(
                "CE",
                bs,
                type,
                "depth=",
                kv_params.head_size,
            )
            execute_ragged_flash_attention[num_q_heads, type, kv_params](
                ce_seq_lens, ce_cache_sizes, 2, 1, mask, ctx
            )

            print(
                "TG",
                bs,
                type,
                "depth=",
                kv_params.head_size,
            )
            execute_ragged_flash_attention[num_q_heads, type, kv_params](
                tg_seq_lens, tg_cache_sizes, 2, 0, mask, ctx
            )

            # GQA group=16 (405B TP=8 shape)
            print(
                "TG",
                bs,
                type,
                "q_heads//kv_heads = 16//1",
                "cache_sizes=",
                tg_cache_sizes,
            )
            execute_ragged_flash_attention[
                16,
                type,
                KVCacheStaticParams(num_heads=1, head_size=kv_params.head_size),
            ](tg_seq_lens, tg_cache_sizes, 2, 0, mask, ctx)
            # GQA group=16 (32Q/2KV variant)
            print("TG", bs, type, "q_heads//kv_heads = 32//2")
            execute_ragged_flash_attention[
                32,
                type,
                KVCacheStaticParams(num_heads=2, head_size=kv_params.head_size),
            ](tg_seq_lens, tg_cache_sizes, 2, 0, mask, ctx)

    # edge cases
    print("CE", 1, DType.bfloat16, "depth=", kv_params.head_size)
    for len in [2, 27]:
        var short_ce_seq_len: List = [len]
        var short_ce_cache_size: List = [0]
        execute_ragged_flash_attention[num_q_heads, DType.bfloat16, kv_params](
            short_ce_seq_len, short_ce_cache_size, 2, 1, mask, ctx
        )

    print("TG", 2, DType.bfloat16, "depth=", kv_params.head_size)
    var tg_seq_lens: List = [1, 1]
    var tg_variable_cache_lens: List = [1024, 11]
    execute_ragged_flash_attention[num_q_heads, DType.bfloat16, kv_params](
        tg_seq_lens, tg_variable_cache_lens, 2, 0, mask, ctx
    )


def main() raises:
    comptime head_size_val = get_defined_int["head_size", 128]()
    comptime num_q_heads = 32 if head_size_val <= 128 else 8
    comptime kv_num_heads = 8 if head_size_val <= 128 else 4

    with DeviceContext() as ctx:
        # Stress test paged decode with many seeds
        var fail_count = 0
        print(
            "Paged Decode Stress [num_q_heads=",
            num_q_heads,
            ", num_heads=",
            kv_num_heads,
            ", depth=",
            head_size_val,
            "]:",
            sep="",
        )
        for s in range(20):
            seed(s)
            var tg_cache_sizes = List[Int]()
            var tg_seq_lens = List[Int]()
            for _ in range(4):
                tg_seq_lens.append(1)
                tg_cache_sizes.append(Int(random_ui64(1, 1024)))
            try:
                execute_ragged_flash_attention[
                    num_q_heads,
                    DType.bfloat16,
                    KVCacheStaticParams(
                        num_heads=kv_num_heads, head_size=head_size_val
                    ),
                ](tg_seq_lens, tg_cache_sizes, 2, 0, CausalMask(), ctx)
            except e:
                print("FAIL seed=", s, "caches=", tg_cache_sizes)
                fail_count += 1
        print("Results:", fail_count, "failures out of 20 seeds")
        if fail_count > 0:
            raise Error(String(fail_count) + " failures")

        # Then run the full suite
        seed(42)
        execute_flash_attention_suite[
            num_q_heads,
            KVCacheStaticParams(
                num_heads=kv_num_heads, head_size=head_size_val
            ),
        ](CausalMask(), ctx)

        # Sliding-window coverage at the same head_size as above. 1024
        # matches the Gemma 3 / Gemma 4 production window; 32 stresses the
        # in-page mask boundary since it is well below page_size=256.
        comptime sliding_windows = (32, 1024)
        comptime for ws_idx in range(len(sliding_windows)):
            comptime window_size = rebind[Int](sliding_windows[ws_idx])
            seed(42 + window_size)
            print(
                "Sliding-window suite [window_size=",
                window_size,
                "]:",
                sep="",
            )
            execute_flash_attention_suite[
                num_q_heads,
                KVCacheStaticParams(
                    num_heads=kv_num_heads, head_size=head_size_val
                ),
            ](SlidingWindowCausalMask[window_size](), ctx)
