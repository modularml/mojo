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
# test_function_declarations.mojo
# Tests for function-declarations.mdx code examples.
# Skip: inferred_type[Int](5) positional error, keyword type
#        mismatch error, div keyword error, configure positional
#        error, sum missing keyword error, bad default ordering
#        error, mut default error, out + return type error,
#        variadic default error, out variadic error, __init__
#        without out error, where on runtime arg error,
#        Resource/__deinit__ (undefined _release), parse/parse_strict
#        (signature-only stubs), copy/move constructor snippets
#        (no enclosing struct), configure keyword-only (pass-only
#        body), overload-set import-shadowing snippet (fictional
#        `from somewhere import add` - hand tested), ambiguous overload
#        call `foo("Hello")` (negative example, exercised by the
#        disambiguation test instead), return-type-only overload
#        of `parse` (negative example: `redefinition of function
#        'parse': cannot overload on return type only`),
#        raises-only overload of `maybe_raise` (negative example:
#        `redefinition of function 'maybe_raise' with identical
#        signature` - hand tested), literal-conversion ambiguity
#        of `take_param[1, 2]()` against `[Int, Int]` vs
#        `[Int, Float64]` (negative example; the runnable
#        `take_param` test uses `[Int, String]` instead),
#        unguarded `var self` on a `TrivialRegisterPassable` type
#        (sharp edge: `^` is a no-op for TRP, so `(c^).which()`
#        returns the `ref self` value - hand verified; the runnable
#        test instead uses `where not conforms_to(Self,
#        TrivialRegisterPassable)` to filter the overload out).
from std.testing import assert_equal
from std.reflection import reflect


# --- Basic function ---


def greet(name: String) -> String:
    return "Hello, " + name


def test_greet() raises:
    assert_equal(greet("Mojo"), "Hello, Mojo")


# --- Minimal function ---


def do_nothing():
    pass


def test_do_nothing():
    do_nothing()


# --- Backtick-escaped name ---


def `import`():
    print("In `import`")


def test_backtick_name():
    `import`()  # In `import`


# --- Parameterized function ---


def clamp[
    T: Comparable & ImplicitlyCopyable
](val: T, lo: T, hi: T,) -> T:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def test_clamp() raises:
    assert_equal(clamp(5, 1, 10), 5)
    assert_equal(clamp(-3, 1, 10), 1)
    assert_equal(clamp(99, 1, 10), 10)


# --- Infer-only marker (//) ---


def inferred_type[T: Writable, //](value: T):
    print(t"Value is {value}. Type is {reflect[T].name()}.")


def test_inferred_type():
    inferred_type(5)  # Value is 5. Type is Int.
    inferred_type("Hello")  # Value is Hello. Type is String.


# --- Keyword bypass for infer-only ---


def test_inferred_type_keyword():
    inferred_type[T=Int](5)  # Value is 5. Type is Int.


# --- Parameter inference ---


def add[T: Intable](a: T, b: T) -> Int:
    return Int(a) + Int(b)


def test_add() raises:
    assert_equal(add[Int](1, 2), 3)
    assert_equal(add(1, 2), 3)
    assert_equal(add[Float64](4.5, 1.2), 5)
    assert_equal(add(4.5, 1.2), 5)


# --- Positional-only marker (/) ---


def div(a: Int, b: Int, /) -> Int:
    return a // b


def test_positional_only() raises:
    assert_equal(div(10, 3), 3)


# --- Variadic with keyword-only ---


def sum_kw(*values: Int, name: String) -> Int:
    print(name, end=": ")
    var total = 0
    for value in values:
        total += value
    return total


def test_variadic_keyword() raises:
    assert_equal(sum_kw(1, 2, 3, name="total"), 6)  # total: 6


# --- Default values ---


def connect(
    host: String = "www.modular.com",
    port: Int = 80,
):
    print(t"Connecting to {host}:{port}")


def test_defaults():
    connect()  # Connecting to www.modular.com:80
    connect(port=8080)  # Connecting to www.modular.com:8080


# --- Defaults with return ---


def my_function(x: Int, y: Int = 0, z: Int = 0) -> Int:
    return x + y + z


def test_defaults_return() raises:
    assert_equal(my_function(1), 1)
    assert_equal(my_function(1, 2), 3)
    assert_equal(my_function(1, 2, 3), 6)


# --- Where clause on parameter ---


def process[
    n: Int
](data: SIMD[.float32, n]) -> Float32 where (
    n == 1 or n == 2 or n == 4 or n == 8 or n == 16 or n == 32
):
    var sum: Float32 = 0.0
    for i in range(n):
        sum += data[i]
    return sum


def test_where_clause() raises:
    var data = SIMD[.float32, 16](255.0)
    var sum = process[n=16](data)
    assert_equal(sum, 4080.0)


# --- Where with conforms_to ---


comptime LESS_THAN: Int32 = -1
comptime EQUAL: Int32 = 0
comptime GREATER_THAN: Int32 = 1


def compare[
    T: AnyType
](x: T, y: T,) -> Int32 where conforms_to(T, Comparable):
    if x < y:
        return LESS_THAN
    elif x > y:
        return GREATER_THAN
    else:
        return EQUAL


def test_compare() raises:
    assert_equal(compare(5, 10), -1)
    assert_equal(compare(7, 7), 0)
    assert_equal(compare("Z", "A"), 1)


# --- mut convention ---


def double_it(mut x: Int):
    x *= 2


def test_mut() raises:
    var x = 5
    double_it(x)
    assert_equal(x, 10)


# --- var convention ---


def consume(var s: String):
    s += "!"
    print(s)


def test_var_convention() raises:
    var greeting = "Hello"
    consume(greeting)  # Hello! (copied)
    assert_equal(greeting, "Hello")

    consume(greeting^)  # Hello! (moved)
    # greeting is now inaccessible


# --- out convention ---


def make_int(out result: Int):
    result = 42


def test_out() raises:
    var x = make_int()
    assert_equal(x, 42)


# --- ref convention ---


def get_first[T: Copyable](ref data: List[T]) -> ref[data[0]] T:
    return data[0]


def test_ref() raises:
    var data: List = ["one", "two", "three"]
    ref first = get_first(data)
    assert_equal(first, "one")
    first = "Первый"
    assert_equal(data[0], "Первый")


# --- Default convention ---


def length[T: Copyable](s: List[T]) -> Int:
    return len(s)


def test_default_convention() raises:
    assert_equal(length(["M", "o", "j", "o"]), 4)


# --- Homogeneous variadics ---


def sum_all(*values: Int) -> Int:
    var total = 0
    for v in values:
        total += v
    return total


def test_homogeneous_variadics() raises:
    assert_equal(sum_all(1, 2, 3), 6)
    assert_equal(sum_all(10, 20), 30)


# --- Variadic packs ---


def print_all[*Ts: Writable](*args: *Ts):
    comptime for idx in range(args.__len__()):
        print(args[idx], end=" ")
    print()


def test_variadic_packs():
    print_all("Hello", 42, 3.14)  # Hello 42 3.14


# --- thin function effect ---


def map_thin[
    T: Copyable, U: Copyable
](f: def(T) thin -> U, input: List[T]) -> List[U]:
    var result: List[U] = []
    for item in input:
        result.append(f(item))
    return result^


def square_thin(x: Int) -> Int:
    return x * x


def test_thin() raises:
    var nums: List[Int] = [1, 2, 3]
    var squares = map_thin(square_thin, nums)
    assert_equal(squares[0], 1)
    assert_equal(squares[1], 4)
    assert_equal(squares[2], 9)


# --- Return type ---


def square(x: Int) -> Int:
    return x * x


def test_square() raises:
    assert_equal(square(5), 25)


# --- Nested functions ---


def outer(x: Int) -> Int:
    def inner() {imm} -> Int:
        return x + 1

    return inner()


def test_nested() raises:
    assert_equal(outer(5), 6)


# --- Static methods ---


struct MathUtils:
    comptime pi: Float64 = 3.141592653589793

    @staticmethod
    def square(x: Int) -> Int:
        return x * x


def test_static_methods() raises:
    assert_equal(MathUtils.square(5), 25)


# --- Function overloads: basic ---


def add_pair(x: Int, y: Int) -> Int:
    return x + y


def add_pair(x: String, y: String) -> String:
    return x + y


def test_add_pair() raises:
    assert_equal(add_pair(1, 2), 3)
    assert_equal(add_pair("Hi, ", "Mojo"), "Hi, Mojo")


# --- Overloading parameters ---


def take_param[a: Int, b: Int]() -> String:
    return "take_param[a: Int, b: Int]"


def take_param[a: Int, b: String]() -> String:
    return "take_param[a: Int, b: String]"


def test_take_param() raises:
    assert_equal(take_param[1, 2](), "take_param[a: Int, b: Int]")
    assert_equal(take_param[1, "hi"](), "take_param[a: Int, b: String]")


# --- Overloading the `self` convention ---


@fieldwise_init
struct Counter(Copyable):
    var n: Int

    def which(ref self) -> Int:
        return 1

    # Constrain `var self` to types where `^` actually transfers.
    # For `TrivialRegisterPassable` types, `^` is a no-op, so this
    # overload would otherwise be silently unreachable.
    def which(
        var self,
    ) -> Int where not conforms_to(Self, TrivialRegisterPassable):
        return 2


@fieldwise_init
struct CounterRegister(ImplicitlyCopyable, TrivialRegisterPassable):
    var n: Int

    def which(ref self) -> Int:
        return 1

    # Same constraint here: the `where` clause filters out this
    # overload entirely for `TrivialRegisterPassable` types.
    def which(
        var self,
    ) -> Int where not conforms_to(Self, TrivialRegisterPassable):
        return 2


def test_self_convention_overload() raises:
    # Non-TRP: both overloads reachable
    var c1 = Counter(0)
    assert_equal(c1.which(), 1)  # default call borrows -> ref self

    var c2 = Counter(0)
    assert_equal((c2^).which(), 2)  # transfer at call site -> var self

    # TRP: the `var self` overload is excluded by `where`, so only
    # `ref self` matches any call.
    var r1 = CounterRegister(0)
    assert_equal(r1.which(), 1)


# --- Instance methods beat static methods ---


struct StaticOverload:
    def __init__(out self):
        pass

    def foo(mut self) -> String:
        return "instance method"

    @staticmethod
    def foo() -> String:
        return "static method"


def test_instance_beats_static() raises:
    var a = StaticOverload()
    assert_equal(a.foo(), "instance method")
    assert_equal(StaticOverload.foo(), "static method")


# --- Variadic candidates lose ties ---


def take_one_or_many(x: Int) -> String:
    return "take(x: Int)"


def take_one_or_many(*xs: Int) -> String:
    return "take(*xs: Int)"


def test_variadic_ties() raises:
    assert_equal(take_one_or_many(1), "take(x: Int)")
    assert_equal(take_one_or_many(1, 2, 3), "take(*xs: Int)")


# --- Ambiguity, resolved by cast ---


struct MyString:
    @implicit
    def __init__(out self, s: String):
        pass


struct YourString:
    @implicit
    def __init__(out self, s: String):
        pass


def overloaded_str(name: MyString) -> String:
    return "MyString"


def overloaded_str(name: YourString) -> String:
    return "YourString"


def test_ambiguity_resolved_by_cast() raises:
    # overloaded_str("Hello") would be ambiguous: skipped intentionally.
    assert_equal(overloaded_str(MyString("Hello")), "MyString")
    assert_equal(overloaded_str(YourString("Hello")), "YourString")


def main() raises:
    test_greet()
    test_do_nothing()
    test_backtick_name()
    test_clamp()
    test_inferred_type()
    test_inferred_type_keyword()
    test_add()
    test_positional_only()
    test_variadic_keyword()
    test_defaults()
    test_defaults_return()
    test_where_clause()
    test_compare()
    test_mut()
    test_var_convention()
    test_out()
    test_ref()
    test_default_convention()
    test_homogeneous_variadics()
    test_variadic_packs()
    test_thin()
    test_square()
    test_nested()
    test_static_methods()
    test_add_pair()
    test_take_param()
    test_self_convention_overload()
    test_instance_beats_static()
    test_variadic_ties()
    test_ambiguity_resolved_by_cast()
