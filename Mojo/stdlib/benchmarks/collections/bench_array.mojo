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

import std.time
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep

# Purposefully mixes power-of-two with non-power-of-two values.
comptime SIZES = [5, 8, 33, 64, 173, 256, 2111, 4096, 65536]


struct NonTrivial(Copyable, Defaultable):
    var value: Int

    def __init__(out self):
        self.value = 1

    def __init__(out self, *, copy: Self):
        self.value = copy.value + 1

    def __init__(out self, *, deinit move: Self):
        self.value = move.value + 1


def bench_move[
    T: Copyable & Defaultable & Deinitable, size: Int
](mut b: Bencher) raises:
    def benchmark(var array: Array[T, size]):
        var moved = array^
        keep(moved)

    b._iter_setup(
        setup=lambda () -> Array[T, size]: {fill = T()}, benchmark=benchmark
    )


def bench_copy[
    T: Copyable & Defaultable & Deinitable, size: Int
](mut b: Bencher) raises:
    def benchmark(var array: Array[T, size]):
        var copied = array.copy()
        keep(copied)

    b._iter_setup(
        setup=lambda () -> Array[T, size]: {fill = T()}, benchmark=benchmark
    )


def bench_fill[
    T: Copyable & Defaultable & Deinitable, size: Int
](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var array = Array[T, size](fill=T())
        keep(array)

    b.iter(call_fn)


def bench_default_init[
    T: Copyable & Defaultable & Deinitable, size: Int
](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var array = Array[T, size]()
        keep(array)

    b.iter(call_fn)


def bench_eq[size: Int](mut b: Bencher) raises:
    """Compares equal arrays, which never short-circuit."""
    var lhs = Array[Int, size](fill=0)
    var rhs = Array[Int, size](fill=0)

    @always_inline
    def call_fn() {imm lhs, imm rhs}:
        var res = black_box(lhs) == black_box(rhs)
        keep(res)

    b.iter(call_fn)


def bench_lt[size: Int](mut b: Bencher) raises:
    """Orders equal arrays, which never short-circuit."""
    var lhs = Array[Int, size](fill=0)
    var rhs = Array[Int, size](fill=0)

    @always_inline
    def call_fn() {imm lhs, imm rhs}:
        var res = black_box(lhs) < black_box(rhs)
        keep(res)

    b.iter(call_fn)


def bench_contains_miss[size: Int](mut b: Bencher) raises:
    """Searches for an absent value, which never short-circuits."""
    var array = Array[Int, size](fill=0)

    @always_inline
    def call_fn() {imm array}:
        var res = black_box(1) in black_box(array)
        keep(res)

    b.iter(call_fn)


def bench_iter[size: Int](mut b: Bencher) raises:
    var array = Array[Int, size](fill=1)

    @always_inline
    def call_fn() {imm array}:
        var total = 0
        for el in black_box(array):
            total += el
        keep(total)

    b.iter(call_fn)


def main() raises:
    var m = Bench(
        BenchConfig(
            min_runtime_secs=0.5, max_runtime_secs=2.0, max_iters=100_000
        )
    )
    comptime for size in SIZES:
        m.bench_function(
            bench_move[Int, size],
            BenchId(String(t"array_move/trivial/{size}")),
        )
        m.bench_function(
            bench_move[NonTrivial, size],
            BenchId(String(t"array_move/nontrivial/{size}")),
        )
        m.bench_function(
            bench_copy[Int, size],
            BenchId(String(t"array_copy/trivial/{size}")),
        )
        m.bench_function(
            bench_copy[NonTrivial, size],
            BenchId(String(t"array_copy/nontrivial/{size}")),
        )
        m.bench_function(
            bench_fill[Byte, size],
            BenchId(String(t"array_fill/byte/{size}")),
        )
        m.bench_function(
            bench_fill[Int, size],
            BenchId(String(t"array_fill/trivial/{size}")),
        )
        m.bench_function(
            bench_fill[NonTrivial, size],
            BenchId(String(t"array_fill/nontrivial/{size}")),
        )
        m.bench_function(
            bench_default_init[Byte, size],
            BenchId(String(t"array_default_init/byte/{size}")),
        )
        m.bench_function(
            bench_default_init[Int, size],
            BenchId(String(t"array_default_init/trivial/{size}")),
        )
        m.bench_function(
            bench_eq[size],
            BenchId(String(t"array_eq/trivial/{size}")),
        )
        m.bench_function(
            bench_lt[size],
            BenchId(String(t"array_lt/trivial/{size}")),
        )
        m.bench_function(
            bench_contains_miss[size],
            BenchId(String(t"array_contains_miss/trivial/{size}")),
        )
        m.bench_function(
            bench_iter[size],
            BenchId(String(t"array_iter/trivial/{size}")),
        )
    m.dump_report()
