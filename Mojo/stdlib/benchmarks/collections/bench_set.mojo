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

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
def make_int_set[size: Int]() -> Set[Int]:
    var s = Set[Int]()
    for i in range(size):
        s.add(i)
    return s^


def make_string_set[size: Int]() -> Set[String]:
    var s = Set[String]()
    for i in range(size):
        s.add(String("key_") + String(i))
    return s^


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__eq__ (Int keys)
# ===-----------------------------------------------------------------------===#
def bench_set_eq_int[size: Int](mut b: Bencher) raises:
    """Benchmark equality check of two equal Int sets."""
    var s1 = make_int_set[size]()
    var s2 = make_int_set[size]()

    @always_inline
    def call_fn() {imm}:
        keep(black_box(s1) == black_box(s2))

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__eq__ (String keys - expensive hash)
# ===-----------------------------------------------------------------------===#
def bench_set_eq_string[size: Int](mut b: Bencher) raises:
    """Benchmark equality check of two equal String sets."""
    var s1 = make_string_set[size]()
    var s2 = make_string_set[size]()

    @always_inline
    def call_fn() {imm}:
        keep(black_box(s1) == black_box(s2))

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__eq__ early exit (different sizes)
# ===-----------------------------------------------------------------------===#
def bench_set_eq_diff_size[size: Int](mut b: Bencher) raises:
    """Benchmark equality fast-path rejection when sizes differ."""
    var s1 = make_int_set[size]()
    var s2 = make_int_set[size + 1]()

    @always_inline
    def call_fn() {imm}:
        keep(black_box(s1) == black_box(s2))

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__eq__ early exit (same size, different elements)
# ===-----------------------------------------------------------------------===#
def bench_set_eq_diff_elems[size: Int](mut b: Bencher) raises:
    """Benchmark equality when sets have same size but different elements."""
    var s1 = make_int_set[size]()
    var s2 = Set[Int]()
    for i in range(size):
        s2.add(i + size)

    @always_inline
    def call_fn() {imm}:
        keep(black_box(s1) == black_box(s2))

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__contains__
# ===-----------------------------------------------------------------------===#
def bench_set_contains[size: Int](mut b: Bencher) raises:
    """Benchmark membership check for 10 elements."""
    var s = make_int_set[size]()

    @always_inline
    def call_fn() {imm}:
        ref int_set = black_box(s)
        for key in range(10):
            keep(black_box(key) in int_set)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.add
# ===-----------------------------------------------------------------------===#
def bench_set_add[size: Int](mut b: Bencher) raises:
    """Benchmark adding 10 existing elements (duplicate check) to a set."""
    var s = make_int_set[size]()

    @always_inline
    def call_fn() {mut s}:
        ref int_set = black_box(s)
        for key in range(10):
            int_set.add(black_box(key))

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.union
# ===-----------------------------------------------------------------------===#
def bench_set_union[size: Int](mut b: Bencher) raises:
    """Benchmark union of two sets with 50% overlap."""
    var s1 = make_int_set[size]()
    var s2 = Set[Int]()
    var half = size // 2
    for i in range(half, half + size):
        s2.add(i)

    @always_inline
    def call_fn() {imm}:
        keep(black_box(s1) | black_box(s2))

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.intersection
# ===-----------------------------------------------------------------------===#
def bench_set_intersection[size: Int](mut b: Bencher) raises:
    """Benchmark intersection of two sets with 50% overlap."""
    var s1 = make_int_set[size]()
    var s2 = Set[Int]()
    var half = size // 2
    for i in range(half, half + size):
        s2.add(i)

    @always_inline
    def call_fn() {imm}:
        keep(black_box(s1) & black_box(s2))

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.difference
# ===-----------------------------------------------------------------------===#
def bench_set_difference[size: Int](mut b: Bencher) raises:
    """Benchmark difference of two sets with 50% overlap."""
    var s1 = make_int_set[size]()
    var s2 = Set[Int]()
    var half = size // 2
    for i in range(half, half + size):
        s2.add(i)

    @always_inline
    def call_fn() {imm}:
        keep(black_box(s1) - black_box(s2))

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.intersection_update (50% overlap, destructive)
# ===-----------------------------------------------------------------------===#
def bench_set_intersection_update[size: Int](mut b: Bencher) raises:
    """Benchmark in-place intersection with 50% overlap.

    Uses iter_preproc to reset s1 between iterations since
    intersection_update is destructive.
    """
    var half = size // 2
    var s1 = make_int_set[size]()
    var s1_orig = s1.copy()
    var s2 = Set[Int]()
    for i in range(half, half + size):
        s2.add(i)

    @always_inline
    def reset(mut s1: Set[Int]) {imm s1_orig}:
        s1 = s1_orig.copy()

    @always_inline
    def call_fn(mut s1: Set[Int]) {imm s2}:
        black_box(s1).intersection_update(black_box(s2))
        keep(len(s1))

    b.iter_preproc(s1, call_fn, reset)


# ===-----------------------------------------------------------------------===#
# Benchmark Set.intersection_update (asymmetric: large self, small other)
# ===-----------------------------------------------------------------------===#
def bench_set_intersection_update_asymmetric[
    size: Int
](mut b: Bencher,) raises:
    """Benchmark in-place intersection where self >> other.

    self has `size` elements, other has 10 elements (all in self).
    Exercises the iterate-smaller-side optimization.
    """
    var s1 = make_int_set[size]()
    var s1_orig = s1.copy()
    var s2 = Set[Int]()
    for i in range(10):
        s2.add(i)

    @always_inline
    def reset(mut s1: Set[Int]) {imm s1_orig}:
        s1 = s1_orig.copy()

    @always_inline
    def call_fn(mut s1: Set[Int]) {s2}:
        black_box(s1).intersection_update(black_box(s2))
        keep(s1)

    b.iter_preproc(s1, call_fn, reset)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    var m = Bench(BenchConfig(num_repetitions=10))
    comptime sizes = (10, 100, 1000, 10_000)

    comptime for i in range(len(sizes)):
        comptime size = rebind[Int](sizes[i])

        # Equality benchmarks
        m.bench_function(
            bench_set_eq_int[size],
            BenchId(String("bench_set_eq_int[", size, "]")),
        )
        m.bench_function(
            bench_set_eq_string[size],
            BenchId(String("bench_set_eq_string[", size, "]")),
        )
        m.bench_function(
            bench_set_eq_diff_size[size],
            BenchId(String("bench_set_eq_diff_size[", size, "]")),
        )
        m.bench_function(
            bench_set_eq_diff_elems[size],
            BenchId(String("bench_set_eq_diff_elems[", size, "]")),
        )

        # Basic operations
        m.bench_function(
            bench_set_contains[size],
            BenchId(String("bench_set_contains[", size, "]")),
        )
        m.bench_function(
            bench_set_add[size],
            BenchId(String("bench_set_add[", size, "]")),
        )

        # Set algebra
        m.bench_function(
            bench_set_union[size],
            BenchId(String("bench_set_union[", size, "]")),
        )
        m.bench_function(
            bench_set_intersection[size],
            BenchId(String("bench_set_intersection[", size, "]")),
        )
        m.bench_function(
            bench_set_difference[size],
            BenchId(String("bench_set_difference[", size, "]")),
        )
        m.bench_function(
            bench_set_intersection_update[size],
            BenchId(String("bench_set_intersection_update[", size, "]")),
        )
        m.bench_function(
            bench_set_intersection_update_asymmetric[size],
            BenchId(
                String("bench_set_intersection_update_asymmetric[", size, "]")
            ),
        )

    print(m)
