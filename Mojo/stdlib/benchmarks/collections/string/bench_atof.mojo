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

from std.pathlib import _dir_of_current_file

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)


# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#
def bench_parsing_all_floats_in_file[
    origin: Origin
](mut b: Bencher, items_to_parse: List[StringSlice[origin]]) raises:
    @always_inline
    def call_fn() raises {imm items_to_parse}:
        for item in items_to_parse:
            var res = atof(item)
            keep(res)

    b.iter(call_fn)


def add_atof_benchmark[
    origin: Origin
](
    mut bench: Bench,
    items_to_parse: List[StringSlice[origin]],
    name: String,
    nb_of_bytes: Int,
) raises:
    def bench_items(mut b: Bencher) raises {imm items_to_parse}:
        bench_parsing_all_floats_in_file(b, items_to_parse)

    bench.bench_function(
        bench_items,
        BenchId("atof", name),
        [
            ThroughputMeasure(BenchMetric.elements, len(items_to_parse)),
            ThroughputMeasure(BenchMetric.bytes, nb_of_bytes),
        ],
    )


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#


def main() raises:
    var bench = Bench()
    comptime files = ["canada", "mesh"]

    comptime for filename in files:
        var file_path = _dir_of_current_file() / "data" / (filename + ".txt")
        var file_contents = file_path.read_text()
        var items_to_parse = file_contents.splitlines()
        var nb_of_bytes = 0
        for item2 in items_to_parse:
            nb_of_bytes += item2.byte_length()

        add_atof_benchmark(
            bench,
            items_to_parse,
            String(filename),
            nb_of_bytes,
        )

    print(bench)
