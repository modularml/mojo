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
# Tests for generics.mdx code examples.
#
# Compile-only (no runtime assertions, exercised by main()):
#   - my_parameterized_fn, the `Some` function-type and variadic before/after
#     pairs, the Foo composition struct, and the fixed/dynamic
#     value-vs-argument examples are signature-only on the page; main()
#     calls them to confirm they build.
#
# Not tested (intentional):
#   - The erroring `function` that prints an `AnyType` directly, the
#     `Some` struct field that can't infer a concrete type, and the
#     zero-capacity SizedListWrapper are negative examples on the page.
#   - The prose-only section ("Parameterized declarations and
#     explicit destruction") has no code.
#
# Renamed to coexist in one file (the page names both `Pair`):
#   - Parameterized types -> `Pair`; parts conformance -> `HashablePair`.
from std.testing import assert_equal, assert_false, assert_true
from std.sys.info import is_64bit

# --- Intro: value parameter alongside a type parameter ---
# The page bounds `T` as `Comparable & ImplicitlyCopyable & Deinitable`
# so the compile-time `threshold` value can be used in the runtime comparison.


def count_above[
    T: Comparable & ImplicitlyCopyable & Deinitable, threshold: T
](values: List[T]) -> Int:
    var count = 0
    for v in values:
        if v > threshold:
            count += 1
    return count


def test_count_above() raises:
    assert_equal(count_above[Int, 3]([1, 2, 3, 4, 5]), 2)


# --- Basic parameterized declarations: compare two lists ---


def all_equal_int(ref lhs: List[Int], ref rhs: List[Int]) -> Bool:
    if len(lhs) != len(rhs):
        return False

    for left, right in zip(lhs, rhs):
        if left != right:
            return False
    return True


def all_equal[
    T: Equatable & Copyable & Deinitable
](ref lhs: List[T], ref rhs: List[T]) -> Bool:
    if len(lhs) != len(rhs):
        return False

    for left, right in zip(lhs, rhs):
        if left != right:
            return False
    return True


def test_all_equal() raises:
    assert_true(all_equal_int([1, 2, 3], [1, 2, 3]))
    assert_false(all_equal_int([1, 2, 3], [4, 5, 6]))

    # The compiler infers T from each call site.
    assert_true(all_equal([1, 2, 3], [1, 2, 3]))
    assert_false(all_equal([1, 2, 3], [4, 5, 6]))
    assert_true(all_equal(["hello", "world"], ["hello", "world"]))
    assert_false(all_equal(["hello", "world"], ["goodbye", "world"]))


# --- Parameterized types: AnyType (compile-only) ---


def my_parameterized_fn[T: AnyType](value: T):
    pass


# --- Printing under-specified parameterized values ---


@fieldwise_init
struct SomeStruct:
    var x: Int


def represent[T: AnyType](v: T) -> String:
    comptime if conforms_to(T, Writable):
        return String(v)
    else:
        return String(t"{reflect[T].name()}")


def function[Ts: AnyType](*args: Ts) raises:
    for arg in args:
        assert_true(represent(arg).byte_length() > 0)


def test_printing_under_specified_values() raises:
    function(1, 2, 3)
    function(SomeStruct(1), SomeStruct(2))


# --- Simplified conformance with `Some`: arguments ---


def as_int_param[T: Intable, //](x: T) -> Int:
    return x.__int__()


def as_int_some(x: Some[Intable]) -> Int:
    return x.__int__()


def test_some_arguments() raises:
    assert_equal(as_int_param(42.1), 42)
    assert_equal(as_int_some(42.1), 42)


# --- Simplified conformance with `Some`: function types (compile-only) ---


def sync_parallelize_param[
    FuncType: def(Int) -> None,
](func: FuncType):
    pass


def sync_parallelize_some(func: Some[def(Int) -> None]):
    pass


def int_to_none(x: Int) -> None:
    pass


# --- Simplified conformance with `Some`: variadics (compile-only) ---


def show_param[*Ts: Writable](*pack: *Ts):
    pass


def show_some(*pack: *SomeTypeList[Writable]):
    pass


# --- Simplified conformance with `Some`: operator overloads ---


@fieldwise_init
struct IndexParam[T: Copyable & Deinitable]:
    var x: Self.T

    def __getitem__[I: Indexer, //](self, idx: I) -> ref[self.x] Self.T:
        return self.x


@fieldwise_init
struct IndexSome[T: Copyable & Deinitable]:
    var x: Self.T

    def __getitem__(self, idx: Some[Indexer]) -> ref[self.x] Self.T:
        return self.x


def test_some_operator_overloads() raises:
    assert_equal(IndexParam(42)[0], 42)
    assert_equal(IndexSome(42)[0], 42)


# --- Where `Some` won't work: move conformance back to a type parameter ---
# The page names this `Struct`; the broken `Some` field version that
# precedes it is a negative example (it can't infer a concrete type).


@fieldwise_init
struct FixedStruct[T: Copyable & Deinitable & Writable](Writable):
    var x: Self.T


def test_where_some_wont_work() raises:
    var s = FixedStruct(1)
    assert_true("1" in String(s))  # FixedStruct[Int](x=1)


# --- Parameterized types ---

comptime ComparableValue = Equatable & ImplicitlyCopyable & Deinitable


@fieldwise_init
struct Pair[T: ComparableValue](ComparableValue):
    var left: Self.T
    var right: Self.T

    def __eq__(self, other: Pair[Self.T]) -> Bool:
        return self.left == other.left and self.right == other.right


def test_parameterized_types() raises:
    assert_true(Pair(1, 2) == Pair(1, 2))
    assert_false(Pair(1, 2) == Pair(1, 3))


# --- Mixing type and value parameters ---


struct ExampleStruct:
    def __init__(out self):
        pass

    def example[
        T: Writable & Copyable,  # type parameter
        count: Int,  # value parameter
    ](
        self,
        data: String,  # argument
        init_value: T,  # generic argument
    ) -> String:
        var result = String(data)
        for _ in range(count):
            result += String(init_value)
        return result


def test_mixing_type_and_value() raises:
    assert_equal(ExampleStruct().example[Int, 3]("x:", 7), "x:777")


# --- Type refinement: `conforms_to` ---
# The page prints the result; this returns a String so it can be asserted.


def process[T: AnyType](value: T) -> String:
    comptime if conforms_to(T, Writable & ImplicitlyCopyable & Deinitable):
        return String(value)
    else:
        return "<not writable>"


def test_type_refinement() raises:
    assert_equal(process(42), "42")
    assert_equal(process("Hello, Mojo!"), "Hello, Mojo!")
    assert_equal(process(3.14), "3.14")
    assert_equal(process([1, 2, 3]), "<not writable>")
    assert_equal(process({"key": "value"}), "<not writable>")


# --- Value parameters: basic example ---

comptime MyCollectionElement = ImplicitlyCopyable & Deinitable


def make_filled[T: MyCollectionElement, size: Int](splat_value: T) -> List[T]:
    var result = List[T](capacity=size)
    for _ in range(size):
        result.append(splat_value)
    return result^


def test_make_filled() raises:
    var three_zeros = make_filled[Int, 3](0)
    var five_hellos = make_filled[String, 5]("hello")
    assert_equal(three_zeros, [0, 0, 0])
    assert_equal(five_hellos, ["hello", "hello", "hello", "hello", "hello"])


# --- Value parameters vs runtime arguments (compile-only) ---


def fixed[size: Int]():
    var buf = Array[Int, size](fill=0)
    _ = buf


def dynamic(size: Int):
    var buf = List[Int](capacity=size)
    _ = buf


# --- Conditional trait conformance: derived conformance and method access ---
# This single `Wrapper` combines the page's "derived conformance" example
# (conditional `Writable`) and its "conditional method access" example
# (conditional `Boolable` gating `__bool__()`).

comptime BaseTraits = Copyable & Deinitable


@fieldwise_init
struct Wrapper[T: BaseTraits](
    Boolable where conforms_to(T, Boolable),
    Writable where conforms_to(T, Writable),
):
    var value: Self.T

    def __bool__(self) -> Bool where conforms_to(Self.T, Boolable):
        return self.value.__bool__()


@fieldwise_init
struct NotWritable(BaseTraits):
    var data: Int


def test_conditional_conformance() raises:
    # Int and String are Writable, so Wrapper gains Writable.
    var w_int = Wrapper[Int](42)
    assert_true("42" in String(w_int))

    var w_str = Wrapper[String]("Hello")
    assert_true("Hello" in String(w_str))

    # Boolable gates __bool__(): non-empty is truthy, empty is falsy.
    assert_true(Bool(w_str))
    var w_empty_str = Wrapper[String]("")
    assert_false(Bool(w_empty_str))

    # NotWritable wrappers are constructible but gain neither conformance.
    var w_not_writable = Wrapper[NotWritable](NotWritable(10))
    _ = w_not_writable


# --- Conditional trait conformance: parts conformance ---
# The page names this `Pair`; renamed here to coexist with the parameterized
# `Pair` from the "Parameterized types" section.


@fieldwise_init
struct HashablePair[L: BaseTraits, R: BaseTraits](
    Hashable where conforms_to(L, Hashable) and conforms_to(R, Hashable)
):
    var left: Self.L
    var right: Self.R


@fieldwise_init
struct NotHashable(BaseTraits):
    var data: Int


def test_parts_conformance() raises:
    # Both parts are Hashable, so the pair gains hash().
    var pair = HashablePair[Int, String](left=1, right="one")
    _ = hash(pair)

    # Constructible, but hashing is unavailable when a part isn't Hashable.
    var pair2 = HashablePair[Int, NotHashable](left=1, right=NotHashable(10))
    _ = pair2


# --- Conditional trait composition (compile-only) ---


@fieldwise_init
struct Foo[T: AnyType](Copyable, Writable where conforms_to(T, Writable)):
    pass


# --- Conditional conformance with value parameters ---
# The page bounds `T` as `Writable & Copyable & Deinitable` (via the
# `ElementTraits` alias) so `write_to()` type-checks; the conditional behavior
# demonstrated here is the value condition `capacity > 0`.


struct SizedListWrapper[capacity: Int, T: Writable & Copyable & Deinitable](
    Sized, Writable where conforms_to(T, Writable) and capacity > 0
):
    var data: List[Self.T]

    def __init__(out self, value: Self.T):
        self.data = List[Self.T](capacity=Self.capacity)
        for _ in range(Self.capacity):
            self.data.append(value.copy())

    def __len__(self) -> Int:
        return len(self.data)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(repr(self.data))

    def first(self) -> Self.T where Self.capacity > 0:
        return self.data[0].copy()


def test_value_param_conformance() raises:
    var s = SizedListWrapper[5, Int](42)
    assert_equal(len(s), 5)
    assert_equal(s.first(), 42)
    assert_true(conforms_to(type_of(s), Writable))


# Gate a function on a trait with a where clause.
def describe[T: AnyType](value: T) -> String where conforms_to(T, Writable):
    return String(value)


# Combine checks with `and`.
def describe_copyable[
    T: AnyType
](value: T) -> String where conforms_to(T, Writable) and conforms_to(
    T, Copyable
):
    return String(value)


# A numeric fact on a parameter: a power-of-two width.
def block[w: Int]() -> Int where w.is_power_of_two():
    return w


# A DType kind predicate on a parameter.
def float_only[dt: DType]() -> Bool where dt.is_floating_point():
    return True


def test_where_clauses() raises:
    assert_equal(describe(2), "2")  # Int is Writable
    assert_equal(describe("hello"), "hello")  # String is Writable
    assert_equal(describe_copyable(7), "7")  # Writable and Copyable
    assert_equal(block[8](), 8)  # 8 is a power of two
    assert_true(float_only[DType.float32]())  # float32 is floating point
    # process(non_writable) would be a COMPILE error (fails the where);
    # a runtime test can't assert a compile error.


# Contrast: is_64bit() works in `comptime if`, NOT in a `where` clause.
def test_comptime_if_machine() raises:
    var w: Int
    comptime if is_64bit():
        w = 64
    else:
        w = 32
    assert_equal(w, 64)


def main() raises:
    test_count_above()
    test_all_equal()
    test_printing_under_specified_values()
    test_some_arguments()
    test_some_operator_overloads()
    test_where_some_wont_work()
    test_parameterized_types()
    test_mixing_type_and_value()
    test_type_refinement()
    test_make_filled()
    test_conditional_conformance()
    test_parts_conformance()
    test_value_param_conformance()
    test_where_clauses()
    test_comptime_if_machine()

    # Compile-only examples: confirm the signature-only snippets build.
    my_parameterized_fn(42)
    sync_parallelize_param(int_to_none)
    sync_parallelize_some(int_to_none)
    show_param(1, 2, 3)
    show_some(1, 2, 3)
    fixed[4]()
    dynamic(4)
    _ = Foo[Int]()
