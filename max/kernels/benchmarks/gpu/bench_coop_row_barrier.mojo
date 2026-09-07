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
"""Benchmarks `CoopRow.combine` at sampling launch shapes.

    ./bazelw run //max/kernels/benchmarks:gpu/bench_coop_row_barrier -- \
        --rows=24 --group_size=8 --iters=100

Divide the reported time by `iters` to get the cost of one combine.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext, Dim
from max.gpu import block_idx, thread_idx
from std.memory import unsafe_stack_allocation
from internal_utils import arg_parse

from nn.sampling.coop_row import COOP_SLOT_FLOATS, CoopRow, coop_row_words

comptime BLOCK_SIZE = 1024


@always_inline
@__parameter
def _sum(x: SIMD, y: type_of(x)) -> type_of(x):
    return x + y


def coop_combine_kernel[
    group_size: Int
](
    workspace: UnsafePointer[Int32, MutAnyOrigin],
    sink: UnsafePointer[Float32, MutAnyOrigin],
    iters: Int32,
):
    var rank = Int(block_idx.x) % group_size
    var row = Int(block_idx.x) // group_size
    var coop = CoopRow[group_size](row, rank)
    var table = unsafe_stack_allocation[
        group_size * COOP_SLOT_FLOATS,
        Float32,
        alignment=16,
        address_space=.SHARED,
    ]()

    var acc = SIMD[.float32, 8](Float32(rank))
    for _ in range(Int(iters)):
        acc = coop.combine[8, _sum](workspace, table, acc)
    if thread_idx.x == 0 and rank == 0:
        sink[row] = acc[0]


def main() raises:
    var rows = Int(arg_parse("rows", 24))
    var group_size = Int(arg_parse("group_size", 8))
    var iters = Int(arg_parse("iters", 100))

    var bench = Bench(BenchConfig(num_repetitions=3, max_iters=20))
    with DeviceContext() as ctx:
        var workspace = ctx.enqueue_create_buffer[.int32](
            coop_row_words(rows, group_size)
        )
        var sink = ctx.enqueue_create_buffer[.float32](rows)
        ctx.enqueue_memset(workspace, 0)
        ctx.synchronize()

        @always_inline
        def launch(ctx: DeviceContext) raises {mut workspace, mut sink, imm}:
            @__parameter
            def go[g: Int]() raises:
                ctx.enqueue_function[coop_combine_kernel[g]](
                    workspace.unsafe_ptr(),
                    sink.unsafe_ptr(),
                    Int32(iters),
                    grid_dim=Dim(rows * g),
                    block_dim=Dim(BLOCK_SIZE),
                )

            comptime for g in [1, 2, 4, 8, 16]:
                if group_size == g:
                    return go[g]()

        @always_inline
        def bench_fn(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, launch, ctx)

        bench.bench_function(
            bench_fn,
            BenchId(
                "coop-row-combine",
                input_id=String(
                    "rows=",
                    rows,
                    "/group_size=",
                    group_size,
                    "/iters=",
                    iters,
                ),
            ),
        )
        _ = workspace^
        _ = sink^
    bench.dump_report()
