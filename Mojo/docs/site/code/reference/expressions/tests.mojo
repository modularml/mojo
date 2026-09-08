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
# Tests for expressions.mdx code examples.
#
# Not tested (intentional errors, type-only forms, or interactive
# constructs):
#   - Set/dict mix errors (`{"a": 1, 2}`, `{1, "b": 2}`)
#   - Positional-after-keyword call error
#   - Duplicate keyword call error
#   - `comptime` without parentheses (intentional error)
#   - `input()` walrus example (interactive)
#   - Function type expressions (`def() -> Int`, etc.) are type-level
#     forms with no standalone runtime behavior to assert
#   - `type_of`, `conforms_to`, `origin_of` return compiler-internal
#     types per the doc; they're exercised via `where` clauses
#     elsewhere in the test corpus
#   - Identifier expressions section (bare names, not standalone code)
from std.collections import Set
from std.testing import assert_equal


# --- Parenthesized expressions ---


def test_paren_precedence() raises:
    var a = 2
    var b = 3
    var c = 4
    assert_equal((a + b) * c, 20)


def test_paren_multiline() raises:
    var first_value = 1
    var second_value = 2
    var third_value = 3
    var result = first_value + second_value + third_value
    assert_equal(result, 6)


# --- Tuples ---


def test_tuple_by_comma() raises:
    var a = 2, 3
    var b = (2, 3)
    assert_equal(a[0], 2)
    assert_equal(a[1], 3)
    assert_equal(b[0], 2)
    assert_equal(b[1], 3)


def test_tuple_destructuring() raises:
    var b = (2, 3)
    var x, y = b
    assert_equal(x, 2)
    assert_equal(y, 3)


def test_tuple_one_element() raises:
    var one_tup = (1,)
    var three_tup = (1, 2, 3)
    assert_equal(one_tup[0], 1)
    assert_equal(three_tup[0], 1)
    assert_equal(three_tup[2], 3)


def test_tuple_indexing() raises:
    var point = (10, 20)
    assert_equal(point[0], 10)
    assert_equal(point[1], 20)


# --- Collection displays ---


def test_list_display() raises:
    var numbers = [1, 2, 3]
    var strings = [
        "one",
        "two",
        "three",
    ]  # trailing comma allowed
    var empty: List[Float32] = []
    assert_equal(len(numbers), 3)
    assert_equal(numbers[0], 1)
    assert_equal(strings[2], "three")
    assert_equal(len(empty), 0)


def test_list_display_with_expressions() raises:
    var values = [1, 1 + 1, 1 + 1 + 1]
    assert_equal(values[0], 1)
    assert_equal(values[1], 2)
    assert_equal(values[2], 3)


def test_dict_display() raises:
    var ages = {"Alice": 30, "Bob": 25}
    var empty: Dict[String, Int] = {}
    assert_equal(ages["Alice"], 30)
    assert_equal(ages["Bob"], 25)
    assert_equal(len(empty), 0)


def test_set_display() raises:
    var primes = {2, 3, 5, 7}
    assert_equal(len(primes), 4)
    assert_equal(2 in primes, True)
    assert_equal(4 in primes, False)


def test_set_explicit_type() raises:
    # Set displays are built-in syntax, but naming the `Set` type needs the
    # import at the top of this file.
    var display_set = {1, 2, 3}  # set display, no import needed
    assert_equal(len(display_set), 3)

    var empty_set = Set[Int]()  # naming Set[Int] requires the import
    empty_set.add(4)
    empty_set.add(4)  # adding a duplicate is a no-op
    assert_equal(len(empty_set), 1)


# --- Initializer lists ---


@fieldwise_init
struct Pair(Copyable, Movable):
    var a: Int
    var b: String


def process(p: Pair) -> String:
    return p.b


def test_initializer_list_call() raises:
    # `{1, "hello"}` is sugar for `Pair(1, "hello")` when type is inferred.
    assert_equal(process({1, "hello"}), "hello")


def test_initializer_list_keyword() raises:
    var p: Pair = {a = 7, b = "kw"}
    assert_equal(p.a, 7)
    assert_equal(p.b, "kw")


# --- Member access ---


@fieldwise_init
struct Point2(Copyable, Movable):
    var x: Int
    var y: Int


def test_member_access() raises:
    var p = Point2(10, 20)
    assert_equal(p.x, 10)
    assert_equal(p.y, 20)


# --- Calls (positional and keyword arguments) ---


def greet(name: String, loud: Bool = False) -> String:
    return String("HELLO, ").upper() if loud else String("Hello, ") + name


def add(a: Int, b: Int) -> Int:
    return a + b


def test_call_positional() raises:
    assert_equal(add(2, 3), 5)


def test_call_keyword() raises:
    assert_equal(add(a=2, b=3), 5)


def test_call_mixed_with_default() raises:
    assert_equal(greet("Alice"), "Hello, Alice")
    assert_equal(greet("Alice", loud=True), "HELLO, ")
    assert_equal(greet(name="Bob"), "Hello, Bob")


# --- Subscripts and slices ---


def test_subscript_list() raises:
    var items = [10, 20, 30]
    assert_equal(items[0], 10)
    assert_equal(items[2], 30)


def test_subscript_dict() raises:
    var mapping = {"key": 42}
    assert_equal(mapping["key"], 42)


def test_slice_start_stop() raises:
    var items: List = [0, 1, 2, 3, 4, 5]
    var first_three = items[0:3]
    assert_equal(len(first_three), 3)
    assert_equal(first_three[0], 0)
    assert_equal(first_three[2], 2)


def test_slice_open_stop() raises:
    var items: List = [0, 1, 2, 3, 4, 5]
    var from_three = items[3:]
    assert_equal(len(from_three), 3)
    assert_equal(from_three[0], 3)
    assert_equal(from_three[2], 5)


def test_slice_stride() raises:
    var items: List = [0, 1, 2, 3, 4, 5]
    var every_other = items[::2]
    assert_equal(len(every_other), 3)
    assert_equal(every_other[0], 0)
    assert_equal(every_other[1], 2)
    assert_equal(every_other[2], 4)


def test_slice_negative_stride() raises:
    var items: List = [0, 1, 2, 3, 4, 5]
    var reversed_items = items[::-1]
    assert_equal(len(reversed_items), 6)
    assert_equal(reversed_items[0], 5)
    assert_equal(reversed_items[5], 0)


# --- Ternary conditional ---


def test_ternary_basic() raises:
    var x = 4
    var label = "even" if x % 2 == 0 else "odd"
    assert_equal(label, "even")
    x = 5
    label = "even" if x % 2 == 0 else "odd"
    assert_equal(label, "odd")


def test_ternary_chained_right_assoc() raises:
    # Parses as: "small" if n < 10 else ("large" if n > 100 else "medium")
    def size(n: Int) -> String:
        return "small" if n < 10 else "large" if n > 100 else "medium"

    assert_equal(size(5), "small")
    assert_equal(size(50), "medium")
    assert_equal(size(500), "large")


# --- Walrus operator ---


def test_walrus_in_if() raises:
    var items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    var captured: Int = -1
    var n: Int
    if (n := len(items)) > 10:
        captured = n
    assert_equal(captured, 11)


def test_walrus_in_while() raises:
    var i = 0
    var last_seen: Int = -1
    var j: Int
    while (j := i) < 3:
        last_seen = j
        i += 1
    assert_equal(last_seen, 2)


# --- Compile-time expressions ---


def heavy_calculation() -> Int:
    var sum = 0
    for i in range(1_000_000):
        sum += i
    return sum


def test_comptime_expression() raises:
    var x = comptime (heavy_calculation())
    assert_equal(x, 499_999_500_000)


# --- Comprehensions ---


def test_list_comprehension_with_filter() raises:
    var squares = [x * x for x in [0, 1, 2, 3, 4] if x % 2 == 0]
    assert_equal(len(squares), 3)
    assert_equal(squares[0], 0)
    assert_equal(squares[1], 4)
    assert_equal(squares[2], 16)


def test_list_comprehension_over_range() raises:
    var positive = [x for x in range(-3, 3) if x > 0]
    assert_equal(len(positive), 2)
    assert_equal(positive[0], 1)
    assert_equal(positive[1], 2)


def fib(n: Int) -> Int:
    # fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, fib(5)=8
    if n < 2:
        return 1
    var a = 2
    var b = 3
    for _ in range(n - 2):
        (a, b) = (b, a + b)
    return a


def test_set_comprehension() raises:
    var fibs = {fib(x) for x in range(6)}
    # 6 iterations produce {1, 2, 3, 5, 8} after dedup
    assert_equal(len(fibs), 5)
    assert_equal(1 in fibs, True)
    assert_equal(8 in fibs, True)


def test_dict_comprehension_basic() raises:
    var dict_squares = {x: x * x for x in range(3)}
    assert_equal(len(dict_squares), 3)
    assert_equal(dict_squares[0], 0)
    assert_equal(dict_squares[1], 1)
    assert_equal(dict_squares[2], 4)


def test_dict_comprehension_string_keys() raises:
    var lengths: Dict[String, Int] = {
        k: k.byte_length() for k in ["one", "two", "three", "four"]
    }
    assert_equal(lengths["one"], 3)
    assert_equal(lengths["three"], 5)
    assert_equal(lengths["four"], 4)


def test_multi_clause_comprehension() raises:
    var products = [
        (x, y, x * y) for x in range(3) for y in range(3) if (x + y) % 2 == 0
    ]
    # Expected: [(0,0,0), (0,2,0), (1,1,1), (2,0,0), (2,2,4)]
    assert_equal(len(products), 5)


def main() raises:
    test_paren_precedence()
    test_paren_multiline()
    test_tuple_by_comma()
    test_tuple_destructuring()
    test_tuple_one_element()
    test_tuple_indexing()
    test_list_display()
    test_list_display_with_expressions()
    test_dict_display()
    test_set_display()
    test_set_explicit_type()
    test_initializer_list_call()
    test_initializer_list_keyword()
    test_member_access()
    test_call_positional()
    test_call_keyword()
    test_call_mixed_with_default()
    test_subscript_list()
    test_subscript_dict()
    test_slice_start_stop()
    test_slice_open_stop()
    test_slice_stride()
    test_slice_negative_stride()
    test_ternary_basic()
    test_ternary_chained_right_assoc()
    test_walrus_in_if()
    test_walrus_in_while()
    test_comptime_expression()
    test_list_comprehension_with_filter()
    test_list_comprehension_over_range()
    test_set_comprehension()
    test_dict_comprehension_basic()
    test_dict_comprehension_string_keys()
    test_multi_clause_comprehension()
