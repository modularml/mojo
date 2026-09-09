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
# tests.mojo
# Tests for lambda.mdx code examples.
#
# Not tested (no runnable behavior to assert from this file):
#   - the bare lambda expression `lambda (x: Int) -> Int: x + 1`, shown
#     without a call site
#   - the inline-lambda call fragment `transform(lambda ..., values)`, shown
#     without surrounding context
#   - the two behavior fragments under "Delegating behavior"
#     (`lambda (value: Int) -> c_int: c_int(value)` and the doubler)
#   - `inplace_transform[lambda (x: Int) -> Int: x ** factor](numbers)`, which
#     the page presents as failing to compile: capturing `factor` makes the
#     lambda a closure, so it can't bind the `thin` function pointer parameter
from std.ffi import external_call, c_int, c_size_t
from std.sys import size_of
from std.testing import assert_almost_equal, assert_equal

# --- Higher-order helpers defined on the page ---


def bubble_sort[
    T: ImplicitlyCopyable & Deinitable, F: def(T, T) -> Bool, //
](compare_fn: F, mut values: List[T]):
    for end in reversed(range(len(values))):
        for i in range(end):
            if compare_fn(values[i], values[i + 1]):
                values[i], values[i + 1] = values[i + 1], values[i]


def inplace_transform[
    T: ImplicitlyCopyable & Deinitable, //, f: def(T) thin -> T
](mut list: List[T]):
    for index in range(len(list)):
        list[index] = f(list[index])


def transform[
    T: Copyable, U: Copyable, F: def(T) -> U, //
](f: F, list: List[T]) -> List[U]:
    return [f(item) for item in list]


def apply[T: Copyable, F: def(T) -> None, //](f: F, i: List[T]):
    for item in i:
        f(item)


def increment[
    Key: ImplicitlyCopyable & Hashable & Equatable & Deinitable
](mut d: Dict[Key, Int], key: Key):
    d[key] = d.get(key, 0) + 1


# --- Creating a lambda: a lambda and the named function it mirrors ---


def inc_named(x: Int) -> Int:
    return x + 1


def test_creating_a_lambda() raises:
    var inc = lambda (x: Int) -> Int: x + 1
    assert_equal(inc(4), 5)
    # The page states the two forms produce the same behavior.
    assert_equal(inc(4), inc_named(4))


# --- Supplying behavior: a named comparator and an inline lambda agree ---


def ascending(x: Int, y: Int) -> Bool:
    return x > y


def test_bubble_sort_named_comparator() raises:
    var values: List[Int] = [3, 1, 4, 1, 5, 9]
    bubble_sort(ascending, values)
    assert_equal(values, [1, 1, 3, 4, 5, 9])


def test_bubble_sort_inline_lambda() raises:
    var values: List[Int] = [3, 1, 4, 1, 5, 9]
    bubble_sort(lambda (a: Int, b: Int) -> Bool: a > b, values)
    assert_equal(values, [1, 1, 3, 4, 5, 9])


def test_bubble_sort_flipped_comparison() raises:
    # The page notes that flipping the comparison reverses the order.
    var values: List[Int] = [3, 1, 4, 1, 5, 9]
    bubble_sort(lambda (a: Int, b: Int) -> Bool: a < b, values)
    assert_equal(values, [9, 5, 4, 3, 1, 1])


# --- Thin lambdas: a thin lambda binds a `thin` function pointer parameter ---


def test_inplace_transform_thin_lambda() raises:
    var numbers: List[Int] = [1, 2, 3, 4, 5]
    inplace_transform[lambda (x: Int) -> Int: x * 2](numbers)
    assert_equal(numbers, [2, 4, 6, 8, 10])


# --- Closures with direct calls: the closure reads current values ---


def test_closure_direct_call() raises:
    var x, y = 3.0, 4.5
    var magnitude = lambda -> Float64: (x**2 + y**2) ** 0.5
    assert_almost_equal(magnitude(), 5.408326913175031)

    # An `imm` capture holds a reference, so a later call sees the new values.
    x, y = -2.5, 1.5
    assert_almost_equal(magnitude(), 2.9154759474226504)


# --- Runtime arguments: a capturing lambda passes where a thin one can't ---


def test_transform_capturing_lambda() raises:
    var numbers: List[Int] = [2, 4, 6, 8, 10]

    var factor = 3
    var transformed = transform(lambda (x: Int) -> Int: x**factor, numbers)
    assert_equal(transformed, [8, 64, 216, 512, 1000])

    factor = 2
    transformed = transform(lambda (x: Int) -> Int: x**factor, numbers)
    assert_equal(transformed, [4, 16, 36, 64, 100])


# --- Capturing and mutating state: the word-length histogram ---


def test_histogram() raises:
    var words = (
        String(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed fringilla nons sapien quis pharetra."
        )
        .replace(",", "")
        .replace(".", "")
    )
    var word_list = [String(w) for w in words.split(" ")]

    comptime counter = lambda (x: String) -> Int: x.count_codepoints()
    var counts = transform(counter, word_list)
    assert_equal(counts, [5, 5, 5, 3, 4, 11, 10, 4, 3, 9, 4, 6, 4, 8])

    var histogram: Dict[Int, Int] = {}
    var collector = lambda (n: Int) {mut histogram}: increment(histogram, n)
    apply(collector, counts)
    assert_equal(len(histogram), 8)
    assert_equal(histogram[5], 3)
    assert_equal(histogram[4], 4)
    assert_equal(histogram[3], 2)
    assert_equal(histogram[11], 1)
    assert_equal(histogram[10], 1)
    assert_equal(histogram[9], 1)
    assert_equal(histogram[6], 1)
    assert_equal(histogram[8], 1)

    var stars = lambda (n: Int) -> String: "*" * n
    assert_equal(stars(histogram.get(5, 0)), "***")
    assert_equal(stars(histogram.get(4, 0)), "****")


# --- FFI interop: a thin `abi("C")` lambda as a qsort comparator ---


def test_ffi_qsort() raises:
    var values: List[Int] = [5, 3, 11, 10, 9, 6, 4]

    var c_values: List[c_int] = transform(
        lambda (v: Int) -> c_int: c_int(v), values
    )

    external_call["qsort", NoneType](
        c_values.unsafe_ptr(),
        c_size_t(len(c_values)),
        c_size_t(size_of[c_int]()),
        lambda (
            a: MutOpaquePointer[MutUntrackedOrigin],
            b: MutOpaquePointer[MutUntrackedOrigin],
        ) abi("C") -> c_int: a.unsafe_bitcast[c_int]()[]
        - b.unsafe_bitcast[c_int]()[],
    )

    assert_equal(c_values, [3, 4, 5, 6, 9, 10, 11])


def main() raises:
    test_creating_a_lambda()
    test_bubble_sort_named_comparator()
    test_bubble_sort_inline_lambda()
    test_bubble_sort_flipped_comparison()
    test_inplace_transform_thin_lambda()
    test_closure_direct_call()
    test_transform_capturing_lambda()
    test_histogram()
    test_ffi_qsort()
