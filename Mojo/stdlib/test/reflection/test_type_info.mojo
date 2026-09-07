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
"""Tests for type and function name introspection APIs."""

from std.sys.info import CompilationTarget, _current_target, is_64bit

from std.collections import List, Optional

from std.reflection import (
    get_linkage_name,
    get_function_name,
    reflect,
)
from std.reflection.type_info import _unqualified_type_name
from std.testing import assert_equal, assert_true, assert_false
from std.testing import TestSuite


def my_func() -> Int:
    return 0


def test_get_linkage_name() raises:
    var name = get_linkage_name[my_func]()
    assert_equal(name, "test_type_info::my_func()")


def test_get_linkage_name_nested() raises:
    def _nested_func(x: Int) {var} -> Int:
        return x

    var name = get_linkage_name[type_of(_nested_func).__call__]()
    assert_equal(
        name,
        (
            "test_type_info::test_get_linkage_name_nested()::_nested_func::__storage::_nested_func(::SIMD[DType.int,"
            " 1])`"
        ),
    )


def your_func[x: Int]() raises -> Int:
    return x


def test_get_linkage_name_parameterized() raises:
    var name = get_linkage_name[your_func[7]]()
    assert_equal(
        name,
        "test_type_info::your_func[::SIMD[DType.int, 1]](),x=7",
    )


def test_get_linkage_name_on_itself() raises:
    var name = get_linkage_name[_current_target]()
    assert_equal(name, "std::sys::info::_current_target()")


@export("export_as_other_name")
def export_func() abi("Mojo"):
    pass


def test_get_linkage_name_export() raises:
    var name = get_linkage_name[export_func]()
    assert_equal(name, "export_as_other_name")


def test_get_function_name() raises:
    var name = get_function_name[my_func]()
    assert_equal(name, "my_func")


def test_get_function_name_nested() raises:
    def nested_func(x: Int) -> Int:
        return x

    var name2 = get_function_name[nested_func]()
    assert_equal(name2, "nested_func")


def test_get_function_name_parameterized() raises:
    var name = get_function_name[your_func]()
    assert_equal(name, "your_func")

    var name2 = get_function_name[your_func[7]]()
    assert_equal(name2, "your_func")


def test_get_function_name_export() raises:
    var name = get_function_name[export_func]()
    assert_equal(name, "export_func")


struct WhatsMyName:
    pass


struct ImGeneric[T: AnyType]:
    pass


def test_unqualified_type_name() raises:
    assert_equal(_unqualified_type_name[WhatsMyName](), "WhatsMyName")
    assert_equal(_unqualified_type_name[Int](), "SIMD[DType.int, 1]")
    assert_equal(_unqualified_type_name[String](), "String")
    assert_equal(_unqualified_type_name[Int8](), "SIMD[DType.int8, 1]")

    # TODO: strip the module name from the inner type name
    assert_equal(
        _unqualified_type_name[ImGeneric[WhatsMyName]](),
        "ImGeneric[test_type_info.WhatsMyName]",
    )


def test_get_type_name() raises:
    var name = reflect[Int].name()
    assert_equal(name, "SIMD[DType.int, 1]")

    name = reflect[Int].name[qualified_builtins=True]()
    assert_equal(name, "std.builtin.simd.SIMD[std.builtin.dtype.DType.int, 1]")


def test_get_type_name_nested() raises:
    def nested_func[T: AnyType]() -> StaticString:
        return reflect[T].name()

    var name = nested_func[String]()
    assert_equal(name, "String")


def test_get_type_name_simd() raises:
    var name = reflect[Float32].name()
    assert_equal(name, "SIMD[DType.float32, 1]")

    name = reflect[SIMD[.uint16, 4]].name[qualified_builtins=True]()
    assert_equal(
        name, "std.builtin.simd.SIMD[std.builtin.dtype.DType.uint16, 4]"
    )


@fieldwise_init
struct Bar[x: Int, f: Float32 = 1.3](Intable):
    def __int__(self) -> Int:
        return self.x

    var y: Int
    var z: Float64


@fieldwise_init
struct Foo[
    T: Intable, //, b: T, c: Bool, d: NoneType = None, e: StaticString = "hello"
]:
    pass


def test_get_type_name_non_scalar_simd_value() raises:
    var name = reflect[Foo[SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0), True]].name()
    assert_equal(
        name,
        (
            "test_type_info.Foo[SIMD[DType.float32, 4], [1, 2, 3, 4] :"
            ' SIMD[DType.float32, 4], True, None, {"hello\0", 5 :'
            " SIMD[DType.int, 1]}]"
        ),
    )

    name = reflect[Foo[SIMD[.bool, 4](True, False, True, False), True]].name()
    assert_equal(
        name,
        (
            "test_type_info.Foo[SIMD[DType.bool, 4], "
            "[True, False, True, False] : SIMD[DType.bool, 4], "
            'True, None, {"hello\0", 5 : SIMD[DType.int, 1]}]'
        ),
    )


def test_get_type_name_struct() raises:
    var name = reflect[Foo[Bar[2](y=3, z=4.1), True]].name()
    assert_equal(
        name,
        (
            "test_type_info.Foo[test_type_info.Bar[2 : SIMD[DType.int, 1],"
            " 1.29999995 : SIMD[DType.float32, 1]], {3 : SIMD[DType.int, 1],"
            " 4.0999999999999996 : SIMD[DType.float64, 1]}, True, None,"
            ' {"hello\0", 5 : SIMD[DType.int, 1]}]'
        ),
    )


def test_get_type_name_unprintable() raises:
    var name = reflect[CompilationTarget[_current_target()]].name()
    assert_equal(name, "std.sys.info.CompilationTarget[<unprintable>]")


# Generic struct for testing types with constructor calls in parameters
struct WrapperWithValue[T: AnyType, //, value: T]:
    pass


@fieldwise_init
struct SimpleParam(TrivialRegisterPassable):
    var b: Bool


def test_get_type_name_ctor_param_direct() raises:
    """Test that direct usage of types with constructor calls works."""
    var name = reflect[WrapperWithValue[SimpleParam(True)]].name()
    assert_equal(
        name,
        "test_type_info.WrapperWithValue[test_type_info.SimpleParam, True]",
    )


# Generic struct for testing nested parametric types
struct GenericWrapper[T: AnyType]:
    pass


# Generic struct with multiple type parameters
struct Pair[T: AnyType, U: AnyType]:
    pass


def test_get_type_name_nested_parametric_direct() raises:
    """Test that directly using nested parametric types works."""
    # Direct usage works fine
    var name = reflect[GenericWrapper[List[String]]].name()
    assert_equal(name, "test_type_info.GenericWrapper[List[String]]")

    # Deeply nested direct usage also works
    name = reflect[GenericWrapper[GenericWrapper[Int]]].name()
    assert_equal(
        name,
        (
            "test_type_info.GenericWrapper[test_type_info.GenericWrapper[SIMD[DType.int,"
            " 1]]]"
        ),
    )


def test_get_type_name_alias() raises:
    comptime T = Bar[5]
    var name = reflect[T].name()
    assert_equal(
        name,
        (
            "test_type_info.Bar[5 : SIMD[DType.int, 1], 1.29999995 :"
            " SIMD[DType.float32, 1]]"
        ),
    )


def _get_type_name_generic[T: AnyType]() -> StaticString:
    """Helper function to test name through generic parameter."""
    return reflect[T].name()


def test_get_type_name_through_generic() raises:
    """Test reflect[T].name() through a generic function."""
    assert_equal(_get_type_name_generic[Int](), "SIMD[DType.int, 1]")
    assert_equal(_get_type_name_generic[String](), "String")


struct UIndexParam[value: UInt]:
    pass


struct IndexParam[value: Int]:
    pass


def test_get_type_name_uint_int_simd_value() raises:
    """Test that DType.uint and DType.int SIMD values are printed correctly."""

    # Test unsigned uindex value - should print as large positive number
    comptime uint_max: UInt = UInt.MAX
    var name = reflect[UIndexParam[uint_max]].name()
    if is_64bit():
        assert_equal(
            name,
            (
                "test_type_info.UIndexParam[18446744073709551615 :"
                " SIMD[DType.uint, 1]]"
            ),
        )
    else:
        assert_equal(
            name,
            "test_type_info.UIndexParam[4294967295 : SIMD[DType.uint, 1]]",
        )

    # Test signed index value for comparison - should print as -1
    comptime neg_one: Int = -1
    name = reflect[IndexParam[neg_one]].name()
    assert_equal(
        name,
        "test_type_info.IndexParam[-1 : SIMD[DType.int, 1]]",
    )


# ===----------------------------------------------------------------------=== #
# Base Type Reflection Tests (Issue #5735)
# ===----------------------------------------------------------------------=== #


def test_get_base_type_name_basic() raises:
    """Test base_name with simple and parameterized types."""
    # For non-parameterized types, base_name returns the type's name
    assert_equal(reflect[Int].base_name(), "SIMD")

    # For parameterized types, base_name returns the base type name
    assert_equal(reflect[List[Int]].base_name(), "List")

    # Different parameterizations of the same type have the same base name
    assert_equal(
        reflect[List[Int]].base_name(), reflect[List[String]].base_name()
    )


def test_get_base_type_name_user_defined() raises:
    """Test base_name with user-defined generic types."""
    # User-defined generics should have the same base name
    assert_equal(
        reflect[GenericWrapper[Int]].base_name(),
        reflect[GenericWrapper[String]].base_name(),
    )

    # Types with multiple parameters
    assert_equal(reflect[Pair[Int, String]].base_name(), "Pair")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
