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
# RUN: %mojo %s -r 2 -o %t.csv | FileCheck %s
# RUN: cat %t.csv | FileCheck %s --check-prefix=CHECK-OUT
# RUN: %mojo %s -t | FileCheck %s --check-prefix=CHECK-TEST

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    Format,
    ThroughputMeasure,
)
from std.testing import TestSuite


def bench1(mut b: Bencher):
    def to_bench():
        print("hello")

    b.iter(to_bench)


def bench2(mut b: Bencher, mystr: String) raises:
    def to_bench() {imm}:
        print(mystr)

    b.iter(to_bench)


def test_mojobench() raises:
    var m = Bench(BenchConfig(max_iters=10_000))
    m.bench_function(
        bench1,
        BenchId("bench1"),
        [
            ThroughputMeasure(BenchMetric.elements, 0),
            ThroughputMeasure(BenchMetric.flops, 0),
        ],
    )

    var inputs = List[String]()
    inputs.append("input1")
    inputs.append("input2")
    for i, input_val in enumerate(inputs):
        m.bench_with_input(
            bench2,
            BenchId("bench2", String(i)),
            input_val,
            [
                ThroughputMeasure(
                    BenchMetric.elements, input_val.byte_length()
                ),
                ThroughputMeasure(BenchMetric.flops, input_val.byte_length()),
            ],
        )

    m.config.verbose_timing = True

    # Check default print format
    # CHECK: | name              | met (ms)
    # CHECK: | ----------------- | -
    # CHECK: | bench1            |
    # CHECK: | bench2/input_id:0 |
    # CHECK: | bench2/input_id:1 |
    print(m)

    # CHECK: name,met (ms),iters,throughput (GElems/s),Arithmetic (GFLOPS/s),min (ms),mean (ms),max (ms),duration (ms)
    # CHECK: "bench1",
    # CHECK: "bench2/input_id:0",
    # CHECK: "bench2/input_id:1",
    m.config.format = Format.csv
    print(m)

    # CHECK: bench1
    # CHECK-NEXT: bench1
    # CHECK-NEXT: bench2/input_id:0
    # CHECK-NEXT: bench2/input_id:0
    # CHECK-NEXT: bench2/input_id:1
    # CHECK-NEXT: bench2/input_id:1
    # CHECK-OUT: bench1
    m.config.format = Format.tabular
    m.dump_report()

    # CHECK-TEST-COUNT-1: hello


def main() raises:
    # NOTE: we pass an empty list since the benchmark infra also tries to parse
    # the arguments for its own purposes.
    TestSuite.discover_tests[__functions_in_module()](
        cli_args=List[StaticString]()
    ).run()
