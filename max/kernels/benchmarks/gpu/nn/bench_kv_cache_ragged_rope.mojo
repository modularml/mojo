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
from std.sys import get_defined_dtype, get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
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
from nn.fused_qk_rope import fused_qk_rope_ragged

from std.utils.index import IndexList


def _get_run_name[
    dtype: DType,
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
](batch_size: Int, seq_len: Int, use_random_seq_lengths: Bool) -> String:
    # fmt: off
    return String(
        "fused_qkv_ragged_rope(", dtype, ") : ",

        # head_info
        "num_q_heads=", num_q_heads, ", ",
        "num_kv_heads=", num_kv_heads, ", ",
        "head_dim=", head_dim, " : ",

        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "use_random_seq_lengths=", use_random_seq_lengths, ", ",
    )
    # fmt: on


def execute_kv_cache_ragged_rope[
    dtype: DType, head_dim: Int, num_q_heads: Int, num_kv_heads: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    use_random_seq_lengths: Bool,
) raises:
    comptime max_seq_len = 2048
    var num_blocks = batch_size * 2
    var num_layers = 1

    comptime CollectionType = ContinuousBatchingKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=num_kv_heads, head_size=head_dim),
        ...,
    ]
    var input_row_offsets_device = ctx.enqueue_create_buffer[dtype.uint32](
        batch_size + 1
    )
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var max_prompt_length = 0
    var total_seq_len: UInt32 = 0
    var cache_len: UInt32 = 10

    var max_context_length: UInt32
    with cache_lengths_device.map_to_host() as cache_lengths_host:
        with input_row_offsets_device.map_to_host() as input_row_offsets_host:
            for i in range(batch_size):
                var curr_seq_length: UInt32
                if use_random_seq_lengths:
                    curr_seq_length = random_ui64(1, UInt64(seq_len)).cast[
                        DType.uint32
                    ]()
                else:
                    curr_seq_length = UInt32(seq_len)

                input_row_offsets_host[i] = curr_seq_length
                if curr_seq_length > UInt32(max_prompt_length):
                    max_prompt_length = Int(curr_seq_length)

                cache_lengths_host[i] = cache_len
                total_seq_len += curr_seq_length

            max_context_length = UInt32(max_prompt_length) + cache_len

            input_row_offsets_host[batch_size] = total_seq_len

    var q_device = ctx.enqueue_create_buffer[dtype](
        Int(total_seq_len) * num_q_heads * head_dim
    )
    var output_device = ctx.enqueue_create_buffer[dtype](len(q_device))
    var q_layout = row_major((total_seq_len, Idx[num_q_heads], Idx[head_dim]))
    with q_device.map_to_host() as q_host:
        var q_tensor = TileTensor(q_host, q_layout)
        random(q_tensor)
    ctx.enqueue_copy(output_device, q_device)
    var output_device_tensor = TileTensor(
        output_device,
        row_major(total_seq_len, Idx[num_q_heads], Idx[head_dim]),
    )

    var kv_block_shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        Int(UInt32(max_prompt_length) + cache_len),
        num_kv_heads,
        head_dim,
    )
    var kv_block_device = ctx.enqueue_create_buffer[dtype](
        kv_block_shape.flattened_length()
    )

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](batch_size)

    # hacky way to select random blocks.
    var block_idx_set = Set[Int]()
    with lookup_table_device.map_to_host() as lookup_table_host:
        var idx = 0
        while idx < batch_size:
            var randval = Int(random_ui64(0, UInt64(num_blocks - 1)))
            if randval in block_idx_set:
                continue

            block_idx_set.add(randval)
            lookup_table_host[idx] = UInt32(randval)
            idx += 1

    var kv_collection_device = CollectionType(
        LayoutTensor[
            kv_block_device.dtype, Layout.row_major[6](), MutAnyOrigin
        ](
            kv_block_device,
            RuntimeLayout[Layout.row_major[6]()].row_major(kv_block_shape),
        ),
        LayoutTensor[
            cache_lengths_device.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            cache_lengths_device,
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                IndexList[1](batch_size)
            ),
        ),
        LayoutTensor[
            lookup_table_device.dtype, Layout(UNKNOWN_VALUE), ImmutAnyOrigin
        ](
            lookup_table_device,
            RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                IndexList[1](batch_size)
            ),
        ),
        max_context_length,
        max_context_length,
    )

    comptime freqs_cis_table_layout = row_major[max_seq_len, head_dim]()
    var freqs_cis_table_device = ctx.enqueue_create_buffer[dtype](
        freqs_cis_table_layout.static_product
    )

    var num_flops_per_elem = 6
    var num_elems = (
        Int(total_seq_len) * num_q_heads * num_kv_heads * head_dim // 2
    )
    var flop_count = num_flops_per_elem * num_elems

    @always_inline
    def bench_func(
        mut b: Bencher,
    ) raises {
        var q_device,
        var kv_collection_device,
        var input_row_offsets_device,
        var freqs_cis_table_device,
        var output_device_tensor,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            fused_qk_rope_ragged[
                kv_collection_device.CacheType,
                interleaved=False,
                target="gpu",
            ](
                TileTensor(q_device, q_layout),
                TileTensor(input_row_offsets_device, row_major(batch_size + 1)),
                kv_collection_device,
                TileTensor(freqs_cis_table_device, freqs_cis_table_layout),
                None,
                0,
                output_device_tensor,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    m.bench_function(
        bench_func,
        BenchId(
            _get_run_name[dtype, num_q_heads, num_kv_heads, head_dim](
                batch_size,
                seq_len,
                use_random_seq_lengths,
            )
        ),
        [ThroughputMeasure(BenchMetric.flops, flop_count)],
    )


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()

    comptime head_dim = get_defined_int["head_dim", 128]()
    comptime num_q_heads = get_defined_int["num_q_heads", 32]()
    comptime num_kv_heads = get_defined_int["num_kv_heads", 8]()

    var batch_size = arg_parse("batch_size", 1)
    var use_random_seq_lengths = arg_parse("use_random_lengths", False)
    var seq_len = arg_parse("seq_len", 1)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        # benchmarking flash attention
        execute_kv_cache_ragged_rope[
            dtype,
            head_dim,
            num_q_heads,
            num_kv_heads,
        ](
            ctx,
            m,
            batch_size,
            seq_len,
            use_random_seq_lengths,
        )

    m.dump_report()
