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

from std.sys import get_defined_dtype, get_defined_int

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from internal_utils import (
    Mode,
    arg_parse,
    get_defined_shape,
    int_list_to_tuple,
    update_bench_config_args,
)

from std.time import sleep

# mojo build sample.mojo
# mpirun -n 8 ./sample -o output.csv


def bench_func[
    dtype: DType, M: Int, N: Int, K: Int, stages: Int
](mut m: Bench, mode: Mode, pe_rank: Int) raises:
    @always_inline
    def bench_iter(mut b: Bencher):
        @always_inline
        def call_fn():
            sleep(0.01)

        b.iter(call_fn)

    var name = String(
        "gemm/dtype=", dtype, "/m=", M, "/n=", N, "/k=", N, "/stages=", stages
    )

    if Mode.BENCHMARK == mode:
        m.bench_function(
            bench_iter,
            BenchId(
                name, input_id=String("1st-metric (pe_rank=", pe_rank, ")")
            ),
        )
        # TODO: enable the following line after adding support for multi-output to kplot and kprofile.
        # m.bench_function(bench_iter, BenchId(name, input_id=String("2nd-metric (pe_rank=",pe_rank,")")))
    if Mode.VERIFY == mode:
        print("verifying dummy results...PASS")
    if Mode.RUN == mode:
        print("pretending to run the kernel...PASS")


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .float16]()
    comptime shape_int_list = get_defined_shape["shape", "1024x1024x1024"]()
    comptime shape = int_list_to_tuple[shape_int_list]()
    comptime stages = get_defined_int["stages", 0]()

    var runtime_x = arg_parse("x", 0)

    # define benchmark mode: [run, benchmark, verify] or a combo (run+benchmark)
    var mode = Mode(arg_parse("mode", "benchmark"))

    print("mode=" + String(mode))

    if Mode.RUN == mode:
        print("-- mode: run kernel once")
    if Mode.BENCHMARK == mode:
        print("-- mode: run kernel benchmark")
    if Mode.VERIFY == mode:
        print("-- mode: verify kernel")

    var m = Bench(BenchConfig(max_iters=1, max_batch_size=1))
    var pe_rank = m.check_mpirun()
    update_bench_config_args(m)
    bench_func[dtype, shape[0], shape[1], shape[2], stages](m, mode, pe_rank)

    m.dump_report()
