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
from std.math import ceildiv, rsqrt
from std.random import random_ui64

from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from std.memory import unsafe_memcpy
from nn.attention.cpu.mha import flash_attention_kv_cache
from nn.attention.mha_mask import CausalMask
from std.testing import assert_almost_equal

from std.utils import IndexList


def execute_ragged_flash_attention() raises:
    comptime num_q_heads = 32
    comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    comptime type = DType.float32
    comptime num_paged_blocks = 32
    comptime page_size = 512
    comptime PagedCollectionType = PagedKVCacheCollection[
        type, kv_params, page_size, ...
    ]
    var num_layers = 1
    var layer_idx = 0

    var true_ce_prompt_lens = [100, 200, 300, 400]
    var mixed_ce_prompt_lens = [50, 100, 150, 100]

    var true_ce_cache_lens = [0, 0, 0, 0]
    var mixed_ce_cache_lens = [50, 100, 150, 300]

    var batch_size = len(true_ce_prompt_lens)

    comptime layout_1d = Layout.row_major[1]()
    var true_ce_row_offsets_buf = List(length=batch_size + 1, fill=UInt32(0))
    var true_ce_row_offsets = LayoutTensor[.uint32, layout_1d](
        true_ce_row_offsets_buf,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    )
    var true_ce_cache_lengths_buf = List(length=batch_size, fill=UInt32(0))
    var true_ce_cache_lengths = LayoutTensor[.uint32, layout_1d](
        true_ce_cache_lengths_buf,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )
    var mixed_ce_row_offsets_buf = List(length=batch_size + 1, fill=UInt32(0))
    var mixed_ce_row_offsets = LayoutTensor[.uint32, layout_1d](
        mixed_ce_row_offsets_buf,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size + 1)),
    )
    var mixed_ce_cache_lengths_buf = List(length=batch_size, fill=UInt32(0))
    var mixed_ce_cache_lengths = LayoutTensor[.uint32, layout_1d](
        mixed_ce_cache_lengths_buf,
        RuntimeLayout[layout_1d].row_major(IndexList[1](batch_size)),
    )

    var true_ce_total_length = 0
    var mixed_ce_total_length = 0
    var true_ce_max_full_context_length = 0
    var mixed_ce_max_full_context_length = 0
    var true_ce_max_prompt_length = 0
    var mixed_ce_max_prompt_length = 0
    for i in range(batch_size):
        true_ce_row_offsets[i] = UInt32(true_ce_total_length)
        mixed_ce_row_offsets[i] = UInt32(mixed_ce_total_length)
        true_ce_cache_lengths[i] = UInt32(true_ce_cache_lens[i])
        mixed_ce_cache_lengths[i] = UInt32(mixed_ce_cache_lens[i])

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

    true_ce_row_offsets[batch_size] = UInt32(true_ce_total_length)
    mixed_ce_row_offsets[batch_size] = UInt32(mixed_ce_total_length)
    comptime layout_3d = Layout.row_major[3]()
    var true_ce_q_ragged_buf = List(
        length=true_ce_total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[type](0),
    )
    var true_ce_q_ragged = LayoutTensor[type, layout_3d](
        true_ce_q_ragged_buf,
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size)
        ),
    )
    random(true_ce_q_ragged)

    var mixed_ce_q_ragged_buf = List(
        length=mixed_ce_total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[type](0),
    )
    var mixed_ce_q_ragged = LayoutTensor[type, layout_3d](
        mixed_ce_q_ragged_buf,
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](
                mixed_ce_total_length, num_q_heads, kv_params.head_size
            )
        ),
    )
    for bs_idx in range(batch_size):
        var mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]

        var true_ce_row_offset = true_ce_row_offsets[bs_idx]
        var mixed_ce_row_offset = mixed_ce_row_offsets[bs_idx]

        var mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        var true_ce_offset = true_ce_q_ragged.ptr + true_ce_q_ragged._offset(
            IndexList[3](
                Int(true_ce_row_offset + UInt32(mixed_ce_cache_len)), 0, 0
            )
        )
        var mixed_ce_offset = mixed_ce_q_ragged.ptr + mixed_ce_q_ragged._offset(
            IndexList[3](Int(mixed_ce_row_offset), 0, 0)
        )

        unsafe_memcpy(
            dest=mixed_ce_offset,
            src=true_ce_offset,
            count=mixed_ce_prompt_len * num_q_heads * kv_params.head_size,
        )

    # initialize reference output
    var mixed_ce_output_buf = List(
        length=mixed_ce_total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[type](0),
    )
    var mixed_ce_output = LayoutTensor[type, layout_3d](
        mixed_ce_output_buf,
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](
                mixed_ce_total_length, num_q_heads, kv_params.head_size
            )
        ),
    )
    var true_ce_output_buf = List(
        length=true_ce_total_length * num_q_heads * kv_params.head_size,
        fill=Scalar[type](0),
    )
    var true_ce_output = LayoutTensor[type, layout_3d](
        true_ce_output_buf,
        RuntimeLayout[layout_3d].row_major(
            IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size)
        ),
    )

    # initialize our KVCache
    comptime layout_6d = Layout.row_major[6]()
    var kv_block_paged_buf = List(
        length=num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * kv_params.head_size,
        fill=Scalar[type](0),
    )
    var kv_block_paged = LayoutTensor[type, layout_6d](
        kv_block_paged_buf,
        RuntimeLayout[layout_6d].row_major(
            IndexList[6](
                num_paged_blocks,
                2,
                num_layers,
                page_size,
                kv_params.num_heads,
                kv_params.head_size,
            )
        ),
    )
    random(kv_block_paged)

    comptime layout_2d = Layout.row_major[2]()
    var paged_lut_buf = List(
        length=batch_size * ceildiv(true_ce_max_full_context_length, page_size),
        fill=UInt32(0),
    )
    var paged_lut = LayoutTensor[.uint32, layout_2d](
        paged_lut_buf,
        RuntimeLayout[layout_2d].row_major(
            IndexList[2](
                batch_size,
                ceildiv(true_ce_max_full_context_length, page_size),
            )
        ),
    )
    var paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        var seq_len = true_ce_cache_lens[bs] + true_ce_prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut[bs, block_idx] = UInt32(randval)

    var true_ce_kv_collection = PagedCollectionType(
        LayoutTensor[
            kv_block_paged.dtype,
            Layout.row_major[6](),
        ](
            kv_block_paged.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_paged.runtime_layout.shape.value,
                kv_block_paged.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[
            mut=False, true_ce_cache_lengths.dtype, Layout(UNKNOWN_VALUE)
        ](
            true_ce_cache_lengths.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                true_ce_cache_lengths.runtime_layout.shape.value,
                true_ce_cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, paged_lut.dtype, Layout.row_major[2]()](
            paged_lut.ptr,
            RuntimeLayout[Layout.row_major[2]()](
                paged_lut.runtime_layout.shape.value,
                paged_lut.runtime_layout.stride.value,
            ),
        ),
        UInt32(true_ce_max_prompt_length),
        UInt32(true_ce_max_full_context_length),
    )

    var mixed_ce_kv_collection = PagedCollectionType(
        LayoutTensor[
            kv_block_paged.dtype,
            Layout.row_major[6](),
        ](
            kv_block_paged.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                kv_block_paged.runtime_layout.shape.value,
                kv_block_paged.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[
            mut=False, mixed_ce_cache_lengths.dtype, Layout(UNKNOWN_VALUE)
        ](
            mixed_ce_cache_lengths.ptr,
            RuntimeLayout[Layout(UNKNOWN_VALUE)](
                mixed_ce_cache_lengths.runtime_layout.shape.value,
                mixed_ce_cache_lengths.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, paged_lut.dtype, Layout.row_major[2]()](
            paged_lut.ptr,
            RuntimeLayout[Layout.row_major[2]()](
                paged_lut.runtime_layout.shape.value,
                paged_lut.runtime_layout.stride.value,
            ),
        ),
        UInt32(mixed_ce_max_prompt_length),
        UInt32(mixed_ce_max_full_context_length),
    )

    # "true CE" execution
    print("true")
    flash_attention_kv_cache(
        true_ce_q_ragged,
        true_ce_row_offsets,
        true_ce_row_offsets,
        true_ce_kv_collection.get_key_cache(layer_idx),
        true_ce_kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        true_ce_output,
    )

    # "mixed CE" execution
    print("mixed")
    flash_attention_kv_cache(
        mixed_ce_q_ragged,
        mixed_ce_row_offsets,
        mixed_ce_row_offsets,
        mixed_ce_kv_collection.get_key_cache(layer_idx),
        mixed_ce_kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        mixed_ce_output,
    )

    var true_ce_out = true_ce_output
    var mixed_ce_out = mixed_ce_output
    for bs in range(batch_size):
        var mixed_ce_prompt_len = mixed_ce_prompt_lens[bs]
        var mixed_ce_row_offset = mixed_ce_row_offsets[bs]
        var true_ce_row_offset = true_ce_row_offsets[bs]
        var mixed_ce_cache_len = mixed_ce_cache_lens[bs]

        var true_ce_ragged_offset = Int(
            true_ce_row_offset + UInt32(mixed_ce_cache_len)
        )
        var mixed_ce_ragged_offset = Int(mixed_ce_row_offset)
        for s in range(mixed_ce_prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            true_ce_out[true_ce_ragged_offset + s, h, hd][0],
                            mixed_ce_out[mixed_ce_ragged_offset + s, h, hd][0],
                            atol=1e-3,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            true_ce_out[true_ce_ragged_offset + s, h, hd][0],
                            mixed_ce_out[mixed_ce_ragged_offset + s, h, hd][0],
                        )
                        raise e^


def main() raises:
    execute_ragged_flash_attention()
