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

from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)
from test_utils import ExplicitDelOnly

from std.builtin.variadics import _call_with_dynamic_pack_pointers


def test_variadic_iterator() raises:
    def helper(*args: Int) raises:
        var n = 5
        for e in args:
            assert_equal(e, n)
            n -= 1

    helper(5, 4, 3, 2, 1)


def test_variadic_reverse_empty() raises:
    var _tup = ()
    comptime ReversedVariadic = type_of(_tup).Ts.reverse()
    assert_equal(_tup.Ts.length, 0)
    assert_equal(ReversedVariadic.length, 0)


def test_variadic_reverse_odd() raises:
    var _tup = (String("hi"), Int(42), Float32(3.14))
    comptime ReversedVariadic = type_of(_tup).Ts.reverse()
    assert_equal(_tup.Ts.length, 3)
    assert_equal(ReversedVariadic.length, 3)
    assert_true(ReversedVariadic[0] == Float32)
    assert_true(ReversedVariadic[1] == Int)
    assert_true(ReversedVariadic[2] == String)


def test_variadic_reverse_even() raises:
    var _tup = (Int(1), String("a"))
    comptime ReversedVariadic3 = type_of(_tup).Ts.reverse()
    assert_equal(_tup.Ts.length, 2)
    assert_equal(ReversedVariadic3.length, 2)
    assert_true(ReversedVariadic3[0] == String)
    assert_true(ReversedVariadic3[1] == Int)


def test_variadic_concat_empty() raises:
    var _tup = ()
    comptime ConcattedVariadic = TypeList._concat[
        type_of(_tup).Ts.values, type_of(_tup).Ts.values
    ]()
    assert_equal(_tup.Ts.length, 0)
    assert_equal(ConcattedVariadic.length, 0)


def test_variadic_concat_singleton() raises:
    var _tup = (String("hi"), Int(42), Float32(3.14))
    var _tup2 = (Bool(True),)
    comptime ConcattedVariadic = TypeList._concat[
        type_of(_tup).Ts.values, type_of(_tup2).Ts.values
    ]()
    assert_equal(_tup.Ts.length, 3)
    assert_equal(ConcattedVariadic.length, 4)
    assert_true(ConcattedVariadic[0] == String)
    assert_true(ConcattedVariadic[1] == Int)
    assert_true(ConcattedVariadic[2] == Float32)
    assert_true(ConcattedVariadic[3] == Bool)


def test_variadic_concat_identity() raises:
    var _tup = (Int(1), String("a"))
    var _tup2 = ()
    comptime ConcattedVariadic = TypeList._concat[
        type_of(_tup).Ts.values, type_of(_tup2).Ts.values
    ]()
    assert_equal(_tup.Ts.length, 2)
    assert_equal(ConcattedVariadic.length, 2)
    assert_true(ConcattedVariadic[0] == Int)
    assert_true(ConcattedVariadic[1] == String)


trait HasStaticValue:
    comptime STATIC_VALUE: Int


@fieldwise_init
struct WithValue[value: Int](HasStaticValue, ImplicitlyCopyable):
    comptime STATIC_VALUE = Self.value


comptime _IntToWithValueMapper[
    value: Int,
]: HasStaticValue = WithValue[value]

comptime IntToWithValue[*values: Int] = values.map_to_type[
    _IntToWithValueMapper
]()


comptime _HasStaticToIntMapper[T: HasStaticValue]: Int = T.STATIC_VALUE


comptime _WithValuePlusIdx[
    T: HasStaticValue, idx: Int
]: HasStaticValue = WithValue[T.STATIC_VALUE + idx]


def test_type_list_map_to_values() raises:
    comptime pl = TypeList.of[
        Trait=HasStaticValue, WithValue[10], WithValue[20], WithValue[30]
    ]().map_to_values[_HasStaticToIntMapper]()
    assert_equal(pl.size, 3)
    assert_equal(pl[0], 10)
    assert_equal(pl[1], 20)
    assert_equal(pl[2], 30)


def test_type_list_map_to_values_empty() raises:
    comptime pl = TypeList.of[Trait=HasStaticValue]().map_to_values[
        _HasStaticToIntMapper
    ]()
    assert_equal(pl.size, 0)


comptime _KeepEvenIdx[T: HasStaticValue, idx: Int]: Bool = (idx & 1) == 0


comptime _KeepAllIdx[T: HasStaticValue, idx: Int]: Bool = True


def test_type_list_filter_idx_even_positions() raises:
    comptime filtered = TypeList.of[
        Trait=HasStaticValue,
        WithValue[10],
        WithValue[20],
        WithValue[30],
        WithValue[40],
    ]().filter_idx[_KeepEvenIdx]()
    assert_equal(filtered.length, 2)
    assert_true(filtered[0] == WithValue[10])
    assert_true(filtered[1] == WithValue[30])


def test_type_list_filter_idx_keep_all() raises:
    comptime filtered = TypeList.of[
        Trait=HasStaticValue, WithValue[1], WithValue[2]
    ]().filter_idx[_KeepAllIdx]()
    assert_equal(filtered.length, 2)
    assert_true(filtered[0] == WithValue[1])
    assert_true(filtered[1] == WithValue[2])


def test_type_list_filter_idx_empty() raises:
    comptime filtered = TypeList.of[Trait=HasStaticValue]().filter_idx[
        _KeepEvenIdx
    ]()
    assert_equal(filtered.length, 0)


def test_type_list_filter_idx_even_positions_pair() raises:
    comptime filtered = TypeList.of[
        Trait=HasStaticValue, WithValue[7], WithValue[8]
    ]().filter_idx[_KeepEvenIdx]()
    assert_equal(filtered.length, 1)
    assert_true(filtered[0] == WithValue[7])


comptime _KeepFirstTwo[T: HasStaticValue, idx: Int]: Bool = idx < 2


def test_type_list_filter_idx_by_index() raises:
    comptime filtered = TypeList.of[
        Trait=HasStaticValue,
        WithValue[100],
        WithValue[200],
        WithValue[300],
    ]().filter_idx[_KeepFirstTwo]()
    assert_equal(filtered.length, 2)
    assert_true(filtered[0] == WithValue[100])
    assert_true(filtered[1] == WithValue[200])


comptime _SumEltAndIdx[prev: Int, T: HasStaticValue, idx: Int]: Int = (
    prev + T.STATIC_VALUE + idx
)


def test_type_list_reduce_idx() raises:
    comptime folded = TypeList.of[
        Trait=HasStaticValue,
        WithValue[1],
        WithValue[2],
        WithValue[3],
    ]().reduce_idx[
        Int(10),
        _SumEltAndIdx,
    ]
    assert_equal(folded, 19)


def test_type_list_reduce_idx_empty() raises:
    comptime folded = TypeList.of[Trait=HasStaticValue]().reduce_idx[
        Int(7),
        _SumEltAndIdx,
    ]
    assert_equal(folded, 7)


def test_variadic_value_reducer() raises:
    comptime mapped_values = IntToWithValue[1, 2, 3]
    assert_true(mapped_values[0] == WithValue[1])
    assert_true(mapped_values[1] == WithValue[2])
    assert_true(mapped_values[2] == WithValue[3])
    assert_equal(mapped_values.length, 3)


def test_variadic_value_reducer_empty() raises:
    comptime mapped_values = IntToWithValue[*ParameterList.empty_of[Int]()]
    assert_equal(mapped_values.length, 0)


def test_variadic_splatted() raises:
    comptime splatted_variadic = TypeList.splat[3, String]()
    assert_equal(splatted_variadic.length, 3)
    assert_true(splatted_variadic[0] == String)
    assert_true(splatted_variadic[1] == String)
    assert_true(splatted_variadic[2] == String)


def test_variadic_splatted_zero() raises:
    comptime splatted_variadic = TypeList.splat[0, Float64]()
    assert_equal(splatted_variadic.length, 0)


comptime _TabulateTimesThree[index: Int]: Int = index * 3


def test_parameter_list_tabulate() raises:
    comptime pl = ParameterList.tabulate[4, _TabulateTimesThree]()
    assert_equal(pl.size, 4)
    assert_equal(pl[0], 0)
    assert_equal(pl[1], 3)
    assert_equal(pl[2], 6)
    assert_equal(pl[3], 9)


def test_parameter_list_splat() raises:
    comptime pl = ParameterList.splat[3, 42]()
    assert_equal(pl.size, 3)
    assert_equal(pl[0], 42)
    assert_equal(pl[1], 42)
    assert_equal(pl[2], 42)


def test_parameter_list_splat_zero() raises:
    comptime pl = ParameterList.splat[0, 1]()
    assert_equal(pl.size, 0)


def test_variadic_contains() raises:
    comptime types = TypeList.of[Int, String, Float32]()
    assert_equal(types.length, 3)
    assert_true(types.contains[Int]())
    assert_true(types.contains[String]())
    assert_true(types.contains[Float32]())
    assert_false(types.contains[Bool]())


def test_variadic_contains_empty() raises:
    comptime types = TypeList.of[Trait=Writable]()
    assert_equal(types.length, 0)
    assert_false(types.contains[Bool]())


def test_variadic_contains_value() raises:
    comptime list = ParameterList.of[1, 2, 3]()
    assert_equal(list.size, 3)
    assert_true(list.contains[1]())
    assert_true(list.contains[2]())
    assert_true(list.contains[3]())
    assert_false(list.contains[4]())


def test_variadic_contains_value_empty() raises:
    comptime list = ParameterList.empty_of[Int]()
    assert_equal(list.size, 0)
    assert_false(list.contains[1]())


def test_slice_types_empty() raises:
    comptime types = TypeList.of[Trait=Writable]().slice[start=0, end=0]()
    assert_equal(types.length, 0)


def test_slice_types() raises:
    comptime types = TypeList.of[Int, String, Float32]().slice[start=0, end=2]()
    assert_equal(types.length, 2)
    assert_true(types[0] == Int)
    assert_true(types[1] == String)


def test_map_types_to_types_empty() raises:
    comptime mapper[T: AnyType] = Int
    comptime types = TypeList.of[Trait=AnyType]().map[mapper]()
    assert_equal(types.length, 0)


trait TestErrable:
    comptime ErrorType: AnyType


struct Foo(TestErrable):
    comptime ErrorType = Int


struct Baz(TestErrable):
    comptime ErrorType = String


def test_map_types_to_types() raises:
    comptime Mapper[T: TestErrable] = T.ErrorType
    # TODO: to remove `Trait=TestErrable`, we need to better support generator type conversion.
    comptime types = TypeList.of[Trait=TestErrable, Foo, Baz]().map[Mapper]()
    assert_equal(types.length, 2)
    assert_true(types[0] == Int)
    assert_true(types[1] == String)


def test_variadic_list_linear_type() raises:
    """Test owned variadics with a linear type (ExplicitDelOnly)."""

    def destroy_elem(idx: Int, var arg: ExplicitDelOnly):
        arg^.destroy()

    def take_owned_linear(var *args: ExplicitDelOnly):
        args^.consume_elements(destroy_elem)

    take_owned_linear(ExplicitDelOnly(5), ExplicitDelOnly(10))


def test_variadic_list_write_to() raises:
    def check_three(*args: Int) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(1, 2, 3)")

    def check_single(*args: Int) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(42)")

    def check_empty(*args: Int) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "()")

    check_three(1, 2, 3)
    check_single(42)
    check_empty()


def test_variadic_list_write_repr_to() raises:
    def check_three(*args: Int) raises:
        var s = String()
        args.write_repr_to(s)
        assert_equal(
            s, "VariadicList[SIMD[DType.int, 1]]((Int(1), Int(2), Int(3)))"
        )

    def check_single(*args: Int) raises:
        var s = String()
        args.write_repr_to(s)
        assert_equal(s, "VariadicList[SIMD[DType.int, 1]]((Int(42)))")

    check_three(1, 2, 3)
    check_single(42)


def test_variadic_list_mem_write_to() raises:
    def check_two(*args: String) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(hello, world)")

    def check_single(*args: String) raises:
        var s = String()
        args.write_to(s)
        assert_equal(s, "(hi)")

    check_two("hello", "world")
    check_single("hi")


def test_variadic_list_mem_write_repr_to() raises:
    def check_two(*args: String) raises:
        var s = String()
        args.write_repr_to(s)
        assert_equal(s, "VariadicList[String](('hello', 'world'))")

    check_two("hello", "world")


def test_variadic_pack_write_to() raises:
    def helper[*Ts: Writable](*args: *Ts) raises:
        comptime assert Ts.all_conforms_to[Writable]()
        var s = String()
        args.write_to(s)
        assert_equal(s, "(1, hello, True)")

    helper(1, "hello", True)


def test_variadic_pack_write_repr_to() raises:
    def helper[*Ts: Writable](*args: *Ts) raises:
        comptime assert Ts.all_conforms_to[Writable]()
        var s = String()
        args.write_repr_to(s)
        assert_equal(s, "(Int(1), 'hello', True)")

    helper(1, "hello", True)


def test_variadic_pack_forwarding() raises:
    """Test that variadic packs can be forwarded with *pack syntax."""

    def callee[*Ts: Writable](*args: *Ts) raises:
        comptime assert Ts.all_conforms_to[Writable]()
        var s = String()
        args.write_to(s)
        assert_equal(s, "(1, hello, 3.14)")

    def forwarder[*Ts: Writable](*args: *Ts) raises:
        callee(*args)

    forwarder(1, "hello", 3.14)


def test_variadic_pack_forwarding_single_element() raises:
    """Test forwarding a single-element variadic pack."""

    def callee[*Ts: Writable](*args: *Ts) raises:
        comptime assert Ts.all_conforms_to[Writable]()
        var s = String()
        args.write_to(s)
        assert_equal(s, "(42)")

    def forwarder[*Ts: Writable](*args: *Ts) raises:
        callee(*args)

    forwarder(42)


def test_variadic_pack_forwarding_empty() raises:
    """Test forwarding an empty variadic pack."""

    def callee[*Ts: Writable](*args: *Ts) raises:
        comptime assert Ts.all_conforms_to[Writable]()
        var s = String()
        args.write_to(s)
        assert_equal(s, "()")

    def forwarder[*Ts: Writable](*args: *Ts) raises:
        callee(*args)

    forwarder()


def test_variadic_pack_forwarding_through_two_levels() raises:
    """Test forwarding a variadic pack through two levels of indirection."""

    def callee[*Ts: Writable](*args: *Ts) raises:
        comptime assert Ts.all_conforms_to[Writable]()
        var s = String()
        args.write_to(s)
        assert_equal(s, "(a, True)")

    def middle[*Ts: Writable](*args: *Ts) raises:
        callee(*args)

    def outer[*Ts: Writable](*args: *Ts) raises:
        middle(*args)

    outer("a", True)


def test_variadic_pack_some() raises:
    """Test using SomeTypeList in a variadic pack."""

    def foo(*args: *SomeTypeList[Writable]) raises:
        comptime assert args.Ts.all_conforms_to[Writable]()
        var s = String()
        args.write_to(s)
        assert_equal(s, "(a, True)")

    foo("a", True)


# ===-----------------------------------------------------------------------===#
# TypeList tests
# ===-----------------------------------------------------------------------===#


def test_typelist_length() raises:
    assert_equal(TypeList.of[Trait=AnyType, Int, String, Float64].length, 3)
    assert_equal(TypeList.of[Bool].length, 1)


def test_typelist_getitem() raises:
    comptime TL = TypeList.of[Trait=AnyType, Int, String, Float64]()
    assert_true(TL[0] == Int)
    assert_true(TL[1] == String)
    assert_true(TL[2] == Float64)


def test_typelist_reversed() raises:
    comptime rev = TypeList.of[Trait=AnyType, Int, String, Float64]().reverse()
    assert_equal(rev.length, 3)
    assert_true(rev[0] == Float64)
    assert_true(rev[1] == String)
    assert_true(rev[2] == Int)


def test_typelist_contains() raises:
    comptime TL = TypeList.of[Trait=AnyType, Int, String, Float64]()
    comptime assert TL.contains[Int]()
    comptime assert TL.contains[String]()
    comptime assert TL.contains[Float64]()
    comptime assert not TL.contains[Bool]()


def test_typelist_reduce() raises:
    comptime TL = TypeList.of[Trait=AnyType, Int, String, Float64]()

    comptime CountTypes[accum: Int, element: AnyType] = accum + 1
    comptime count = TL.reduce[0, CountTypes]
    assert_equal(count, 3)


def test_typelist_any_all() raises:
    comptime TL = TypeList.of[Trait=AnyType, Int, String, Float64]()

    comptime IsInt[T: AnyType] = T == Int
    comptime IsNumber[T: AnyType] = T == Int or T == Float64

    comptime assert TL.any[IsInt]()
    comptime assert TL.any[IsNumber]()

    comptime IsBool[T: AnyType] = T == Bool
    comptime assert not TL.any[IsBool]()

    comptime AlwaysTrue[T: AnyType] = True
    comptime assert TL.all[AlwaysTrue]()

    comptime IsIntOrString[T: AnyType] = T == Int or T == String
    comptime assert not TL.all[IsIntOrString]()


def test_typelist_all_empty() raises:
    comptime TL = TypeList.of[Trait=AnyType]()
    comptime AlwaysFalse[T: AnyType] = False
    comptime assert TL.all[AlwaysFalse]()


def test_typelist_slice() raises:
    comptime TL = TypeList.of[Trait=AnyType, Int, String, Float64, Bool]()

    # Slice middle
    comptime middle = TL.slice[start=1, end=3]()
    assert_equal(middle.length, 2)
    assert_true(middle[0] == String)
    assert_true(middle[1] == Float64)

    # Slice from start
    comptime first2 = TL.slice[start=0, end=2]()
    assert_equal(first2.length, 2)
    assert_true(first2[0] == Int)
    assert_true(first2[1] == String)

    # Slice to end (using default)
    comptime last2 = TL.slice[start=2]()
    assert_equal(last2.length, 2)
    assert_true(last2[0] == Float64)
    assert_true(last2[1] == Bool)

    # Full slice (defaults)
    comptime full = TL.slice[]
    assert_equal(full.length, 4)


def test_typelist_map() raises:
    comptime TL = TypeList.of[Trait=Copyable, Int, String, Float64]()

    comptime ToList[T: Copyable]: Copyable = List[T]
    comptime mapped = TL.map[ToList]()
    assert_equal(mapped.length, 3)
    comptime assert mapped[0] == List[Int]
    comptime assert mapped[1] == List[String]
    comptime assert mapped[2] == List[Float64]


def test_typelist_map_identity() raises:
    comptime TL = TypeList.of[Trait=AnyType, Int, Bool]()

    comptime Identity[T: AnyType] = T
    comptime mapped = TL.map[Identity]()
    assert_equal(mapped.length, 2)
    comptime assert mapped[0] == Int
    comptime assert mapped[1] == Bool


def test_typelist_map_empty() raises:
    comptime TL = TypeList.of[Trait=Copyable]()

    comptime ToList[T: Copyable]: Copyable = List[T]
    comptime mapped = TL.map[ToList]()
    assert_equal(mapped.length, 0)


def format_args(
    buffer: Pointer[String, MutAnyOrigin],
    x: Int,
    y: String,
    z: List[Int],
):
    buffer[].write(x, " ", y, " ", z)


def return_from_dynamic_pack(
    buffer: Pointer[String, MutAnyOrigin],
    x: Int,
    y: String,
    z: List[Int],
) -> Int:
    return x


def raise_from_dynamic_pack(
    buffer: Pointer[String, MutAnyOrigin],
    x: Int,
    y: String,
    z: List[Int],
) raises StopIteration:
    raise StopIteration()


def test_dynamic_variadic_pack() raises:
    var buffer = String()
    var buffer_ptr = Pointer(to=buffer).as_unsafe_any_origin()
    var a1 = Int(5)
    var a2 = String("hello")
    var a3: List[Int] = [1, 20, 300]

    comptime Args = TypeList.of[
        Trait=AnyType,
        type_of(buffer_ptr),
        Int,
        String,
        List[Int],
    ]()

    def make_elem_ptr[
        idx: Int
    ]() {imm} -> Pointer[Args[idx], MutUnsafeAnyOrigin]:
        comptime ArgPtr = Pointer[Args[idx], MutUnsafeAnyOrigin]
        comptime if idx == 0:
            return rebind_var[ArgPtr](
                Pointer(to=buffer_ptr).as_unsafe_any_origin()
            )
        elif idx == 1:
            return rebind_var[ArgPtr](Pointer(to=a1).as_unsafe_any_origin())
        elif idx == 2:
            return rebind_var[ArgPtr](Pointer(to=a2).as_unsafe_any_origin())
        else:
            comptime assert idx == 3
            return rebind_var[ArgPtr](Pointer(to=a3).as_unsafe_any_origin())

    _call_with_dynamic_pack_pointers[ArgTrait=AnyType, format_args](
        make_elem_ptr
    )

    assert_equal(buffer, "5 hello [1, 20, 300]")

    # Test that dynamic pack call works with callees that return a value
    assert_equal(
        _call_with_dynamic_pack_pointers[
            ArgTrait=AnyType, return_from_dynamic_pack
        ](make_elem_ptr),
        5,
    )

    # Test that dynamic pack call works with callees that `raises`
    with assert_raises(contains="StopIteration"):
        _call_with_dynamic_pack_pointers[
            ArgTrait=AnyType, raise_from_dynamic_pack
        ](make_elem_ptr)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
