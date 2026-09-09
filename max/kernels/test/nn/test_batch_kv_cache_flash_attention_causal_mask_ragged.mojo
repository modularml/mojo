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
from std.math import rsqrt
from std.random import random_ui64, seed

from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from std.memory import unsafe_memcpy
from nn.attention.cpu.mha import flash_attention_kv_cache
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal

from std.utils import IndexList

comptime kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
comptime llama_num_q_heads = 32


def execute_ragged_flash_attention[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths_list: List[Int],
    max_seq_len_cache: Int,
    cache_lengths_list: List[Int],
    num_layers: Int,
    layer_idx: Int,
) raises:
    comptime num_blocks = 32
    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype, kv_params, ...
    ]

    var batch_size = len(valid_lengths_list)
    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )
    assert len(valid_lengths_list) == len(
        cache_lengths_list
    ), "expected valid_lengths and cache_lengths size to be equal"

    comptime layout_1d = Layout.row_major[1]()
    var input_row_offsets_buf = List(length=batch_size + 1, fill=UInt32(0))
    var input_row_offsets = LayoutTensor[.uint32, layout_1d](
        input_row_offsets_buf,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    )
    var cache_lengths_buf = List(length=batch_size, fill=UInt32(0))
    var cache_lengths = LayoutTensor[.uint32, layout_1d](
        cache_lengths_buf,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )
    var valid_lengths_buf = List(length=batch_size, fill=UInt32(0))
    var valid_lengths = LayoutTensor[.uint32, layout_1d](
        valid_lengths_buf,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )

    var total_length = 0
    var max_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets[i] = UInt32(total_length)
        cache_lengths[i] = UInt32(cache_lengths_list[i])
        valid_lengths[i] = UInt32(valid_lengths_list[i])
        max_context_length = max(
            max_context_length, cache_lengths_list[i] + valid_lengths_list[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths_list[i])
        total_length += valid_lengths_list[i]
    input_row_offsets[batch_size] = UInt32(total_length)

    comptime layout_3d = Layout.row_major[3]()
    var q_ragged_buf = List(
        length=total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[dtype](0),
    )
    var q_ragged = LayoutTensor[dtype, layout_3d](
        q_ragged_buf,
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](total_length, num_q_heads, kv_params.head_size)
        ),
    )
    random(q_ragged)

    comptime layout_4d = Layout.row_major[4]()
    var q_padded_buf = List(
        length=batch_size
        * max_prompt_length
        * num_q_heads
        * kv_params.head_size,
        fill=Scalar[dtype](0),
    )
    var q_padded = LayoutTensor[dtype, layout_4d](
        q_padded_buf,
        RuntimeLayout[layout_4d].row_major(
            IndexList[4](
                batch_size,
                max_prompt_length,
                num_q_heads,
                kv_params.head_size,
            )
        ),
    )

    # copy over the ragged values to the padded tensor.
    # Don't worry about padded values, we won't read them.
    for bs in range(batch_size):
        var unpadded_seq_len = valid_lengths_list[bs]
        var ragged_start_idx = Int(input_row_offsets[bs])
        var padded_ptr = q_padded.ptr + q_padded._offset(
            IndexList[4](bs, 0, 0, 0)
        )
        var ragged_ptr = q_ragged.ptr + q_ragged._offset(
            IndexList[3](ragged_start_idx, 0, 0)
        )
        unsafe_memcpy(
            dest=padded_ptr,
            src=ragged_ptr,
            count=unpadded_seq_len * num_q_heads * kv_params.head_size,
        )

    # initialize reference output
    var ref_output_buf = List(
        length=batch_size
        * max_prompt_length
        * num_q_heads
        * kv_params.head_size,
        fill=Scalar[dtype](0),
    )
    var ref_output = LayoutTensor[dtype, layout_4d](
        ref_output_buf,
        RuntimeLayout[layout_4d].row_major(
            IndexList[4](
                batch_size,
                max_prompt_length,
                num_q_heads,
                kv_params.head_size,
            )
        ),
    )

    var test_output_buf = List(
        length=total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[dtype](0),
    )
    var test_output = LayoutTensor[dtype, layout_3d](
        test_output_buf,
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](total_length, num_q_heads, kv_params.head_size)
        ),
    )

    # initialize our KVCache
    comptime layout_6d = Layout.row_major[6]()
    var kv_block_buf = List(
        length=num_blocks
        * 2
        * num_layers
        * max_seq_len_cache
        * kv_params.num_heads
        * kv_params.head_size,
        fill=Scalar[dtype](0),
    )
    var kv_block = LayoutTensor[dtype, layout_6d](
        kv_block_buf,
        RuntimeLayout[layout_6d].row_major(
            IndexList[6](
                num_blocks,
                2,
                num_layers,
                max_seq_len_cache,
                kv_params.num_heads,
                kv_params.head_size,
            )
        ),
    )
    random(kv_block)
    var lookup_table_buf = List(length=batch_size, fill=UInt32(0))
    var lookup_table = LayoutTensor[.uint32, layout_1d](
        lookup_table_buf,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )

    # hacky way to select random blocks.
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = Int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table[idx] = UInt32(randval)
        idx += 1

    var kv_collection = CollectionType(
        LayoutTensor[kv_block.dtype, Layout.row_major[6]()](
            kv_block.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block.runtime_layout.shape.value,
                kv_block.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, cache_lengths.dtype, Layout(UNKNOWN_VALUE)](
            cache_lengths.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                cache_lengths.runtime_layout.shape.value,
                cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, lookup_table.dtype, Layout(UNKNOWN_VALUE)](
            lookup_table.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                lookup_table.runtime_layout.shape.value,
                lookup_table.runtime_layout.stride.value,
            ),
        ),
        UInt32(max_prompt_length),
        UInt32(max_context_length),
    )

    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)

    # ragged execution
    flash_attention_kv_cache(
        q_ragged,
        input_row_offsets,
        # Assume self attention: Q and KV sequence lengths are equal.
        input_row_offsets,
        k_cache,
        v_cache,
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        test_output,
    )
    # padded execution
    flash_attention_kv_cache(
        q_padded,
        k_cache,
        v_cache,
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        ref_output,
    )

    var ref_out = ref_output
    var test_out = test_output
    for bs in range(batch_size):
        var prompt_len = Int(valid_lengths[bs])
        var ragged_offset = Int(input_row_offsets[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            ref_out[bs, s, h, hd][0],
                            test_out[ragged_offset + s, h, hd][0],
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            ref_out[bs, s, h, hd][0],
                            test_out[ragged_offset + s, h, hd][0],
                        )
                        raise e^


comptime dtype = DType.float32


def execute_flash_attention_suite() raises:
    for bs in [1, 16]:
        var ce_cache_sizes = List[Int]()
        var ce_seq_lens = List[Int]()
        var tg_cache_sizes = List[Int]()
        var tg_seq_lens = List[Int]()
        for _ in range(bs):
            tg_seq_lens.append(1)
            tg_cache_sizes.append(Int(random_ui64(1, 100)))
            ce_seq_lens.append(Int(random_ui64(1, 100)))
            ce_cache_sizes.append(0)
        print("CE", bs, dtype)
        execute_ragged_flash_attention[
            llama_num_q_heads, dtype, kv_params_llama3
        ](ce_seq_lens, 110, ce_cache_sizes, 2, 1)

        print("TG", bs, dtype)
        execute_ragged_flash_attention[
            llama_num_q_heads, dtype, kv_params_llama3
        ](tg_seq_lens, 110, tg_cache_sizes, 2, 0)

    # Edge-case specific tests
    # Case 0: token gen in one batch, context encoding in another
    var c0_seq_lens: List[Int] = [25, 1]
    var c0_cache_sizes: List[Int] = [0, 25]

    execute_ragged_flash_attention[llama_num_q_heads, dtype, kv_params_llama3](
        c0_seq_lens, 110, c0_cache_sizes, 2, 0
    )


def main() raises:
    seed(42)
    execute_flash_attention_suite()
