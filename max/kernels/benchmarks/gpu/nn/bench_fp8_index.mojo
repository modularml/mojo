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

from std.random import seed
from std.sys import get_defined_int

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
from layout import Idx, TileTensor, row_major
from layout._fillers import random
from nn.index_fp8 import fp8_index


def _get_run_name[
    num_heads: Int,
    depth: Int,
](batch_size: Int, seq_len: Int, num_keys: Int) -> String:
    # fmt: off
    return String(
        "fp8_index : ",
        "num_heads=", num_heads, ", ",
        "depth=", depth, " : ",
        "batch_size=", batch_size, ", ",
        "seq_len=", seq_len, ", ",
        "num_keys=", num_keys,
    )
    # fmt: on


def execute_fp8_index[
    num_heads: Int,
    depth: Int,
](
    ctx: DeviceContext,
    mut m: Bench,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    run_benchmark: Bool,
) raises:
    var q_size = batch_size * seq_len * num_heads * depth
    var qs_size = batch_size * seq_len * num_heads
    var k_size = batch_size * num_keys * depth
    var ks_size = batch_size * num_keys
    var o_size = batch_size * seq_len * num_keys

    var q_device_ptr = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    var qs_device_ptr = ctx.enqueue_create_buffer[.float32](qs_size)
    var k_device_ptr = ctx.enqueue_create_buffer[.float8_e4m3fn](k_size)
    var ks_device_ptr = ctx.enqueue_create_buffer[.float32](ks_size)
    var input_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var cache_row_offsets_device_ptr = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var o_device_ptr = ctx.enqueue_create_buffer[.float32](o_size)

    var q_layout = row_major((batch_size * seq_len, Idx[num_heads], Idx[depth]))
    var qs_layout = row_major((batch_size * seq_len, Idx[num_heads]))
    var k_layout = row_major((batch_size * num_keys, Idx[1], Idx[depth]))
    var ks_layout = row_major(batch_size * num_keys)
    var o_layout = row_major((batch_size * seq_len, num_keys))
    var iro_layout = row_major(batch_size + 1)

    with q_device_ptr.map_to_host() as q_host:
        random(TileTensor(q_host, q_layout))
    with qs_device_ptr.map_to_host() as qs_host:
        random(TileTensor(qs_host, qs_layout))
    with k_device_ptr.map_to_host() as k_host:
        random(TileTensor(k_host, k_layout))
    with ks_device_ptr.map_to_host() as ks_host:
        random(TileTensor(ks_host, ks_layout))

    with input_row_offsets_device_ptr.map_to_host() as iro_host:
        for i in range(batch_size):
            iro_host[i] = UInt32(i * seq_len)
        iro_host[batch_size] = UInt32(batch_size * seq_len)

    with cache_row_offsets_device_ptr.map_to_host() as cro_host:
        for i in range(batch_size):
            cro_host[i] = UInt32(i * num_keys)
        cro_host[batch_size] = UInt32(batch_size * num_keys)

    var q_device = TileTensor(q_device_ptr, q_layout)
    var qs_device = TileTensor(qs_device_ptr, qs_layout)
    var k_device = TileTensor(k_device_ptr, k_layout)
    var ks_device = TileTensor(ks_device_ptr, ks_layout)
    var o_device = TileTensor(o_device_ptr, o_layout)
    var input_row_offsets_device = TileTensor(
        input_row_offsets_device_ptr, iro_layout
    )
    var cache_row_offsets_device = TileTensor[mut=False](
        cache_row_offsets_device_ptr, iro_layout
    )

    # FLOPs: dominant work is the per-(seq, key, head) FP8 dot product over
    # `depth` (mul+add = 2 flops), plus a relu and scale per (seq, key, head)
    # and a final scale+sum per (seq, key). Approximate with the GEMV term.
    var flop_count = batch_size * seq_len * num_keys * num_heads * depth * 2

    if run_benchmark:

        @always_inline
        def bench_func(
            mut b: Bencher,
        ) raises {
            var q_device,
            var qs_device,
            var k_device,
            var ks_device,
            var o_device,
            var input_row_offsets_device,
            var cache_row_offsets_device,
            imm,
        }:
            @always_inline
            def kernel_launch(ctx: DeviceContext) raises {imm}:
                fp8_index[num_heads, depth](
                    o_device,
                    q_device,
                    qs_device,
                    k_device,
                    ks_device,
                    input_row_offsets_device,
                    cache_row_offsets_device,
                    batch_size,
                    seq_len,
                    num_keys,
                    ctx,
                )

            bencher_iter_custom(b, kernel_launch, ctx)

        m.bench_function(
            bench_func,
            BenchId(
                _get_run_name[num_heads, depth](batch_size, seq_len, num_keys)
            ),
            [ThroughputMeasure(BenchMetric.flops, flop_count)],
        )
    else:
        fp8_index[num_heads, depth](
            o_device,
            q_device,
            qs_device,
            k_device,
            ks_device,
            input_row_offsets_device,
            cache_row_offsets_device,
            batch_size,
            seq_len,
            num_keys,
            ctx,
        )

    _ = q_device_ptr
    _ = qs_device_ptr
    _ = k_device_ptr
    _ = ks_device_ptr
    _ = input_row_offsets_device_ptr
    _ = cache_row_offsets_device_ptr
    _ = o_device_ptr


def main() raises:
    comptime num_heads = get_defined_int["num_heads", 128]()
    comptime depth = get_defined_int["depth", 128]()

    var batch_size = arg_parse("batch_size", 2)
    var seq_len = arg_parse("seq_len", 128)
    var num_keys = arg_parse("num_keys", 128)
    var run_benchmark = arg_parse("run_benchmark", True)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        execute_fp8_index[num_heads, depth](
            ctx, m, batch_size, seq_len, num_keys, run_benchmark
        )

    m.dump_report()
