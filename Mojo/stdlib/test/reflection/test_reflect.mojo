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
"""Tests for the unified `reflect[T]` / `Reflected[T]` reflection API."""

from std.reflection import Reflected
from std.sys.info import size_of
from std.testing import TestSuite, assert_equal, assert_false, assert_true


# ===----------------------------------------------------------------------=== #
# Helper structs
# ===----------------------------------------------------------------------=== #


struct SimpleStruct:
    var x: Int
    var y: Float64


struct Inner:
    var a: Int
    var b: Int


struct Outer:
    var name: String
    var inner: Inner
    var count: Int


struct EmptyStruct:
    pass


struct StructWithMLIRField(TrivialRegisterPassable):
    var mojo_field: Int
    var mlir_field: __mlir_type.index


@fieldwise_init
struct OffsetTestStruct(TrivialRegisterPassable):
    var a: Int8  # offset 0
    var b: Int64  # offset 8
    var c: Int8  # offset 16
    var d: Int32  # offset 20


@fieldwise_init
struct SimpleOffsetStruct(TrivialRegisterPassable):
    var x: Int64
    var y: Int64


# Non-copyable type for ref tests.
struct NonCopyableValue:
    var data: Int

    def __init__(out self, data: Int):
        self.data = data

    def __init__(out self, *, copy: Self):
        # If this fires, `field_ref` is copying instead of returning a ref.
        print("ERROR: NonCopyableValue was copied!")
        self.data = copy.data


struct ContainerWithNonCopyable:
    var id: Int
    var resource: NonCopyableValue
    var count: Int

    def __init__(out self, id: Int, value: Int, count: Int):
        self.id = id
        self.resource = NonCopyableValue(value)
        self.count = count


# Struct with a wrapped (`Optional[T]`) field, for inner-type peeling by index.
struct WrappedFields:
    var a: Int64
    var b: Optional[Int64]


# ===----------------------------------------------------------------------=== #
# `reflect` and `Reflected.T`
# ===----------------------------------------------------------------------=== #


def test_reflect_returns_handle() raises:
    comptime r = reflect[SimpleStruct]
    # The handle's T parameter is the reflected type.
    assert_equal(reflect[r.T].name(), "test_reflect.SimpleStruct")
    assert_equal(r.field_count(), 2)


def test_reflected_t_is_usable_as_type() raises:
    comptime r = reflect[Int]
    # Reflected[Int].T is Int -> can be used in declarations.
    var v: r.T = 42
    assert_equal(v, 42)


def test_reflected_is_zero_sized() raises:
    """`Reflected[T]` is a tag handle: it must not occupy any storage."""
    assert_equal(size_of[Reflected[Int]](), 0)
    assert_equal(size_of[Reflected[String]](), 0)
    assert_equal(size_of[Reflected[SimpleStruct]](), 0)


# ===----------------------------------------------------------------------=== #
# `is_struct`
# ===----------------------------------------------------------------------=== #


def test_is_struct_user_struct() raises:
    assert_true(reflect[SimpleStruct].is_struct())
    assert_true(reflect[Outer].is_struct())
    assert_true(reflect[EmptyStruct].is_struct())


def test_is_struct_stdlib() raises:
    assert_true(reflect[Int].is_struct())
    assert_true(reflect[String].is_struct())
    assert_true(reflect[Float64].is_struct())


def _is_struct_generic[T: AnyType]() -> Bool:
    return reflect[T].is_struct()


def test_is_struct_through_generic() raises:
    assert_true(_is_struct_generic[Int]())
    assert_true(_is_struct_generic[String]())


def test_is_struct_with_mlir_primitive() raises:
    """When an MLIR primitive is reached via field_types, is_struct is False."""
    comptime types = reflect[StructWithMLIRField].field_types()
    assert_true(_is_struct_generic[types[0]]())  # Int
    assert_false(_is_struct_generic[types[1]]())  # __mlir_type.index


def test_is_struct_as_guard() raises:
    """Use `comptime if is_struct()` to guard reflection on mixed field types.
    """
    var struct_count = 0
    var non_struct_count = 0

    comptime r = reflect[StructWithMLIRField]
    comptime for i in range(r.field_count()):
        comptime field_type = r.field_types()[i]
        comptime if reflect[field_type].is_struct():
            struct_count += 1
        else:
            non_struct_count += 1

    assert_equal(struct_count, 1)
    assert_equal(non_struct_count, 1)


# ===----------------------------------------------------------------------=== #
# `field_count`, `field_names`, `field_types`
# ===----------------------------------------------------------------------=== #


def test_field_count_simple() raises:
    assert_equal(reflect[SimpleStruct].field_count(), 2)


def test_field_count_nested_returns_unflattened() raises:
    # Nested struct counts the inner field as one entry, not its inner fields.
    assert_equal(reflect[Outer].field_count(), 3)
    assert_equal(reflect[Inner].field_count(), 2)


def test_field_count_empty() raises:
    assert_equal(reflect[EmptyStruct].field_count(), 0)


def test_field_names_simple() raises:
    var names = reflect[SimpleStruct].field_names()
    assert_equal(names[0], "x")
    assert_equal(names[1], "y")


def test_field_types_match_get_type_name() raises:
    comptime types = reflect[SimpleStruct].field_types()
    assert_equal(reflect[types[0]].name(), "SIMD[DType.int, 1]")
    assert_equal(reflect[types[1]].name(), "SIMD[DType.float64, 1]")


def test_field_iteration() raises:
    var count = 0
    comptime r = reflect[SimpleStruct]
    comptime for i in range(r.field_count()):
        comptime field_name = r.field_names()[i]
        comptime field_type = r.field_types()[i]
        _ = field_name
        _ = field_type
        count += 1
    assert_equal(count, 2)


def _count_fields_generic[T: AnyType]() -> Int:
    return reflect[T].field_count()


def test_field_count_through_generic() raises:
    assert_equal(_count_fields_generic[SimpleStruct](), 2)
    assert_equal(_count_fields_generic[Outer](), 3)
    assert_equal(_count_fields_generic[EmptyStruct](), 0)


# ===----------------------------------------------------------------------=== #
# `field_index` / `field` (lookup by name)
# ===----------------------------------------------------------------------=== #


def test_field_index_simple() raises:
    comptime r = reflect[SimpleStruct]
    assert_equal(r.field_index["x"](), 0)
    assert_equal(r.field_index["y"](), 1)


def test_field_index_nested() raises:
    comptime r = reflect[Outer]
    assert_equal(r.field_index["name"](), 0)
    assert_equal(r.field_index["inner"](), 1)
    assert_equal(r.field_index["count"](), 2)


def test_field_by_name_simple() raises:
    comptime r = reflect[SimpleStruct]
    comptime x_type = r.field["x"]
    assert_equal(x_type.name(), "SIMD[DType.int, 1]")

    comptime y_type = r.field["y"]
    assert_equal(y_type.name(), "SIMD[DType.float64, 1]")


def test_field_returns_reflected_handle() raises:
    """`field[name]` returns a `Reflected[FieldT]`, fully composable."""
    comptime r = reflect[Outer]
    comptime inner_handle = r.field["inner"]
    # The returned handle is itself a Reflected, with its own field_count etc.
    assert_equal(inner_handle.field_count(), 2)
    assert_equal(inner_handle.field_names()[0], "a")
    assert_equal(inner_handle.field_names()[1], "b")


def test_field_usable_as_type_annotation() raises:
    comptime y_type = reflect[SimpleStruct].field["y"]
    var v: y_type.T = 3.14
    assert_true(v > 3.0)


def test_field_matches_field_types_by_index() raises:
    comptime r = reflect[SimpleStruct]
    comptime idx = r.field_index["x"]()
    comptime by_name = r.field["x"]
    comptime by_idx = r.field_types()[idx]
    assert_equal(by_name.name(), reflect[by_idx].name())


def test_field_at_simple() raises:
    comptime r = reflect[SimpleStruct]
    comptime x_type = r.field_at[0]
    assert_equal(x_type.name(), "SIMD[DType.int, 1]")

    comptime y_type = r.field_at[1]
    assert_equal(y_type.name(), "SIMD[DType.float64, 1]")


def test_field_at_matches_by_name() raises:
    comptime r = reflect[SimpleStruct]
    comptime by_name = r.field["y"]
    comptime by_idx = r.field_at[r.field_index["y"]()]
    assert_equal(by_idx.name(), by_name.name())


def test_field_at_returns_reflected_handle() raises:
    """`field_at[idx]` returns a `Reflected[FieldT]`, fully composable."""
    comptime r = reflect[Outer]
    comptime inner_handle = r.field_at[1]
    assert_equal(inner_handle.field_count(), 2)
    assert_equal(inner_handle.field_names()[0], "a")
    assert_equal(inner_handle.field_names()[1], "b")


def test_field_at_usable_as_type_annotation() raises:
    comptime y_type = reflect[SimpleStruct].field_at[1]
    var v: y_type.T = 3.14
    assert_true(v > 3.0)


def _field_base_names_generic[T: AnyType]() -> String:
    """Recover each field's concrete type by index through a generic `T`."""
    comptime r = reflect[T]
    var seen = String("")
    comptime for i in range(r.field_count()):
        comptime FT = r.field_at[i]
        seen += FT.base_name()
        seen += ","
    return seen^


def test_field_at_through_generic() raises:
    # `Int` is `SIMD[DType.int, 1]`, so its base name is `SIMD`.
    assert_equal(_field_base_names_generic[Outer](), "String,Inner,SIMD,")


def test_field_at_peels_wrapper() raises:
    """Find a wrapped field by index and peel its inner concrete type."""
    comptime opt = reflect[WrappedFields].field_at[1]
    assert_equal(opt.base_name(), "Optional")
    # `opt.T` is `Optional[Int64]`; peel its inner type to `Int64`.
    comptime Inner = opt.T.T
    assert_equal(reflect[Inner].name(), "SIMD[DType.int64, 1]")


def test_field_at_first_index() raises:
    comptime first = reflect[Outer].field_at[0]
    assert_equal(first.name(), "String")


# ===----------------------------------------------------------------------=== #
# `field_ref`
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct PointForRef:
    var x: Int
    var y: Int


def test_field_ref_basic_read() raises:
    var p = PointForRef(10, 20)
    comptime r = reflect[PointForRef]
    ref x_ref = r.field_ref[0](p)
    ref y_ref = r.field_ref[1](p)
    assert_equal(x_ref, 10)
    assert_equal(y_ref, 20)


def test_field_ref_mutation() raises:
    var p = PointForRef(1, 2)
    comptime r = reflect[PointForRef]
    r.field_ref[0](p) = 100
    r.field_ref[1](p) = 200
    assert_equal(p.x, 100)
    assert_equal(p.y, 200)


def test_field_ref_non_copyable() raises:
    """Verify `field_ref` does not copy non-copyable fields."""
    var c = ContainerWithNonCopyable(42, 100, 5)
    comptime r = reflect[ContainerWithNonCopyable]

    ref id_ref = r.field_ref[0](c)
    ref resource_ref = r.field_ref[1](c)
    ref count_ref = r.field_ref[2](c)

    assert_equal(id_ref, 42)
    assert_equal(resource_ref.data, 100)
    assert_equal(count_ref, 5)

    r.field_ref[1](c).data = 999
    assert_equal(c.resource.data, 999)


def _print_fields_generic[T: AnyType](ref s: T):
    """Generic example: works with parametric idx in a comptime for loop."""
    comptime r = reflect[T]
    comptime for i in range(r.field_count()):
        _ = r.field_ref[i](s)


def test_field_ref_parametric_index() raises:
    var p = PointForRef(10, 20)
    var c = ContainerWithNonCopyable(1, 2, 3)
    _print_fields_generic(p)
    _print_fields_generic(c)


# ===----------------------------------------------------------------------=== #
# `field_offset`
# ===----------------------------------------------------------------------=== #


def test_field_offset_first_is_zero() raises:
    comptime r = reflect[SimpleOffsetStruct]
    assert_equal(r.field_offset[name="x"](), 0)
    assert_equal(r.field_offset[index=0](), 0)


def test_field_offset_with_padding() raises:
    comptime r = reflect[OffsetTestStruct]
    assert_equal(r.field_offset[name="a"](), 0)
    assert_equal(r.field_offset[name="b"](), 8)  # aligned to 8
    assert_equal(r.field_offset[name="c"](), 16)
    assert_equal(r.field_offset[name="d"](), 20)  # aligned to 4


def test_field_offset_by_index_matches_by_name() raises:
    comptime r = reflect[OffsetTestStruct]
    assert_equal(r.field_offset[index=0](), r.field_offset[name="a"]())
    assert_equal(r.field_offset[index=1](), r.field_offset[name="b"]())
    assert_equal(r.field_offset[index=2](), r.field_offset[name="c"]())
    assert_equal(r.field_offset[index=3](), r.field_offset[name="d"]())


def test_field_offset_iteration() raises:
    var offsets = Array[Int, 4](uninitialized=True)
    comptime r = reflect[OffsetTestStruct]
    comptime for i in range(r.field_count()):
        offsets[i] = r.field_offset[index=i]()
    assert_equal(offsets[0], 0)
    assert_equal(offsets[1], 8)
    assert_equal(offsets[2], 16)
    assert_equal(offsets[3], 20)


# ===----------------------------------------------------------------------=== #
# Closure capture reflection
# ===----------------------------------------------------------------------=== #


struct StructToCapture(ImplicitlyCopyable):
    var x: Int
    var y: Int

    def __init__(out self: Self):
        self.x = 27
        self.y = 42

    def __init__(out self, *, copy: Self):
        self.x = copy.x
        self.y = copy.y


def test_closure_capture_struct_reflection() raises:
    """A closure captures a struct."""
    var s = StructToCapture()

    def closure() {var s} -> None:
        pass

    closure()

    # The closure type is the storage struct; its fields are the captures.
    comptime captures = reflect[type_of(closure)]
    # closure captures s — one field.
    assert_equal(captures.field_count(), 1)

    # capture s is itself a struct containing x and y.
    comptime s_type = captures.field_types()[0]
    assert_true(reflect[s_type].is_struct())
    assert_equal(reflect[s_type].field_count(), 2)


def test_imm_capture_field_type_name() raises:
    """`{imm}`/`{mut}` capture fields are refs; `name()` prints the pointee."""
    var a: Int = 0
    var b: Int = 1
    var s = String("x")

    def closure() {imm a, imm b, mut s}:
        _ = a
        _ = b
        s += "!"

    closure()

    comptime captures = reflect[type_of(closure)]
    assert_equal(captures.field_count(), 3)
    comptime a_type = captures.field_types()[0]
    assert_false(reflect[a_type].is_struct())
    assert_equal(reflect[a_type].name(), "ref[SIMD[DType.int, 1]]")
    comptime s_type = captures.field_types()[2]
    assert_false(reflect[s_type].is_struct())
    assert_equal(reflect[s_type].name(), "ref[String]")


def test_nested_closure_capture_reflection() raises:
    """A closure captured by another closure is reflectable as a struct."""
    var x = UInt32(1)
    var y = UInt32(2)
    var z = UInt32(3)

    def inner() {var x} -> None:
        pass

    def outer() {var y, var z, var inner} -> None:
        pass

    outer()

    # The outer closure type is its storage struct.
    comptime outer_captures = reflect[type_of(outer)]
    # outer captures y, z, inner — three fields.
    assert_equal(outer_captures.field_count(), 3)

    # inner is the third capture; its type must be a reflectable struct.
    comptime inner_type = outer_captures.field_types()[2]
    assert_true(reflect[inner_type].is_struct())

    # inner's storage has one capture field: x (UInt32).
    comptime inner_r = reflect[inner_type]
    assert_equal(inner_r.field_count(), 1)
    assert_true(reflect[inner_r.field_types()[0]].is_struct())


def test_deeply_nested_closure_capture_reflection() raises:
    """Reflection works transitively: A captured by B captured by C."""
    var x = UInt32(1)
    var y = UInt32(2)
    var z = UInt32(3)
    var w = UInt32(4)

    def a() {var x} -> None:
        pass

    def b() {var y, var a} -> None:
        pass

    def c() {var z, var w, var b} -> None:
        pass

    c()

    # Navigate into c's storage struct.
    comptime c_captures = reflect[type_of(c)]
    # c captures z, w, b — three fields.
    assert_equal(c_captures.field_count(), 3)

    # b is c's third capture.
    comptime b_type = reflect[c_captures.field_types()[2]]
    assert_true(b_type.is_struct())
    # b's storage has two capture fields: y and a.
    assert_equal(b_type.field_count(), 2)

    # a is b's second capture and is a struct.
    comptime a_type = reflect[b_type.field_types()[1]]
    assert_true(a_type.is_struct())
    # a's storage has one capture field: x.
    assert_equal(a_type.field_count(), 1)
    assert_true(reflect[a_type.field_types()[0]].is_struct())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
