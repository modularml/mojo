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

# Covers the `List` mutation paths whose cost is dominated by reallocation and
# element movement: growth from empty, repeated `extend` of an owned list,
# front insertion, back removal, iteration, reversal, a byte fill via `resize`,
# and repeated `resize` growth.
#
# Element counts go through `black_box` so the compiler sees each length as a
# runtime value; a compile-time-constant count lets it hoist or unroll the work
# and stop measuring the dynamic behavior these paths actually have. Every
# closure builds its `List` inside the body — the list must be destroyed each
# iteration, otherwise growth is measured against an ever-larger allocation —
# and ends in `keep` so the stores are not dead-code eliminated.

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep

comptime SIZES = [64, 1024, 16384]


def bench_append[size: Int](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var n = black_box(size)
        var list = List[Int]()
        for i in range(n):
            list.append(i)
        keep(list.unsafe_ptr())

    b.iter(call_fn)


def bench_extend_owned[size: Int](mut b: Bencher) raises:
    # Extending in a loop is the shape that turns an exact-reserve `extend`
    # into quadratic work.
    @always_inline
    def call_fn():
        var n = black_box(size)
        var list = List[Int]()
        for i in range(n // 4):
            var chunk: List[Int] = [i, i + 1, i + 2, i + 3]
            list.extend(chunk^)
        keep(list.unsafe_ptr())

    b.iter(call_fn)


def bench_insert_front[size: Int](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var n = black_box(size)
        var list = List[Int]()
        for i in range(n):
            list.insert(0, i)
        keep(list.unsafe_ptr())

    b.iter(call_fn)


def bench_pop_back[size: Int](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var n = black_box(size)
        var list = List[Int](length=n, fill=0)
        var total = 0
        for _ in range(n):
            total += list.pop()
        keep(total)

    b.iter(call_fn)


def bench_iterate_sum[size: Int](mut b: Bencher) raises:
    var list = List[Int](length=black_box(size), fill=1)

    @always_inline
    def call_fn() {imm list}:
        var total = 0
        for element in list:
            total += element
        keep(total)

    b.iter(call_fn)
    _ = list^


def bench_reverse[size: Int](mut b: Bencher) raises:
    # Reversal walks the elements through `self._data`, so it is the clearest
    # measure of whether that load stays hoisted out of the loop.
    var list = List[Int](length=black_box(size), fill=1)

    @always_inline
    def call_fn() {mut list}:
        list.reverse()
        keep(list.unsafe_ptr())

    b.iter(call_fn)
    _ = list^


def bench_resize_grow[size: Int](mut b: Bencher) raises:
    # Growing by a fixed increment is the `resize` shape an exact reserve turns
    # quadratic. `bench_byte_resize_fill` reaches its length in one step, where
    # an exact reserve and an amortized one allocate the same amount, so it
    # cannot see that difference.
    @always_inline
    def call_fn():
        var n = black_box(size)
        var list = List[Int]()
        for length in range(4, n + 1, 4):
            list.resize(length, 0)
        keep(list.unsafe_ptr())

    b.iter(call_fn)


def bench_byte_resize_fill[size: Int](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var list = List[Byte]()
        list.resize(black_box(size), 0)
        keep(list.unsafe_ptr())

    b.iter(call_fn)


def main() raises:
    var m = Bench(BenchConfig(min_runtime_secs=0.01, max_iters=1_000_000))
    comptime for size in SIZES:
        m.bench_function(
            bench_append[size],
            BenchId(String(t"append/{size}")),
        )
        m.bench_function(
            bench_extend_owned[size],
            BenchId(String(t"extend_owned/{size}")),
        )
        m.bench_function(
            bench_insert_front[size],
            BenchId(String(t"insert_front/{size}")),
        )
        m.bench_function(
            bench_pop_back[size],
            BenchId(String(t"pop_back/{size}")),
        )
        m.bench_function(
            bench_iterate_sum[size],
            BenchId(String(t"iterate_sum/{size}")),
        )
        m.bench_function(
            bench_reverse[size],
            BenchId(String(t"reverse/{size}")),
        )
        m.bench_function(
            bench_resize_grow[size],
            BenchId(String(t"resize_grow/{size}")),
        )
        m.bench_function(
            bench_byte_resize_fill[size],
            BenchId(String(t"byte_resize_fill/{size}")),
        )
    m.dump_report()
