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
# test_struct_declarations.mojo
# Tests for struct-declarations.mdx code examples.
# Skip: nested struct error, field without type error, dynamic
#        trait field error, fieldwise_init synthesis failure
#        (Alpha/Color), Incomplete(Sized) missing method,
#        Box without Deinitable abandoned error,
#        Unsound(Copyable) with SomeMoveOnlyType, default
#        method conflicts (struct S(A & B) constructed to fail),
#        missing self error, __init__ without out self error,
#        recursive Node error, __deinit__ snippet (no standalone
#        example), @staticmethod snippet (error pair only),
#        NotWritable example (constructed to fail), conditional
#        conformance struct snippets (hypothetical trait names
#        like GPUComputable/FloatMath/Printable, no def main()).
from std.testing import assert_equal
from std.math import sqrt


# --- Point with distance ---


struct Point_1:
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y

    def distance(self) -> Float64:
        return sqrt(Float64(self.x * self.x + self.y * self.y))


def test_point_distance() raises:
    var p = Point_1(3, 4)
    assert_equal(p.distance(), 5.0)


# --- Minimal struct ---


@fieldwise_init
struct ValidationError:
    pass


def test_validation_error():
    _ = ValidationError()


# --- Color with manual __init__ ---


struct Color_1:
    var r: UInt8
    var g: UInt8
    var b: UInt8

    def __init__(out self, r: UInt8, g: UInt8, b: UInt8):
        (self.r, self.g, self.b) = (r, g, b)


def test_color_manual() raises:
    var c = Color_1(255, 0, 0)
    assert_equal(c.r, 255)
    assert_equal(c.g, 0)
    assert_equal(c.b, 0)


# --- Field type from parameter ---


@fieldwise_init
struct Sound[T: Writable & Copyable & Deinitable]:
    var item: Self.T


def test_field_type_parameter() raises:
    var g = Sound[Int](item=42)
    assert_equal(g.item, 42)


# --- Color with @fieldwise_init ---


@fieldwise_init
struct Color_2:
    var r: UInt8
    var g: UInt8
    var b: UInt8


def test_color_fieldwise() raises:
    var color = Color_2(255, 0, 0)
    assert_equal(color.r, 255)
    assert_equal(color.g, 0)
    assert_equal(color.b, 0)


# --- Parameterized Pair ---


@fieldwise_init
struct Pair_1[T: Copyable & Deinitable]:
    var first: Self.T
    var second: Self.T


def test_parameterized_pair() raises:
    var p = Pair_1[Int](first=1, second=2)
    assert_equal(p.first, 1)
    assert_equal(p.second, 2)


# --- Defaulted parameters ---


struct SplatList[
    T: ImplicitlyCopyable & Deinitable,
    *,
    fill: T,
    length: Int = 5,
]:
    var items: List[Self.T]

    def __init__(out self):
        self.items = List[Self.T](length=Self.length, fill=Self.fill)


def test_defaulted_parameters() raises:
    var l = SplatList[Int, fill=42]()
    assert_equal(len(l.items), 5)
    assert_equal(l.items[0], 42)
    assert_equal(l.items[4], 42)


# --- Trait conformance ---


@fieldwise_init
struct MyInt(Copyable, Writable):
    var value: Int

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.value)


def test_trait_conformance() raises:
    var my_int = MyInt(42)
    assert_equal(String(my_int), "42")


# --- Conformance with where clause ---


@fieldwise_init
struct Pair_2[T: Copyable & Deinitable](
    Equatable where conforms_to(T, Equatable)
):
    var first: Self.T
    var second: Self.T


def test_conformance_where() raises:
    var a = Pair_2[Int](first=1, second=2)
    var b = Pair_2[Int](first=1, second=2)
    var c = Pair_2[Int](first=3, second=4)
    assert_equal(a == b, True)
    assert_equal(a == c, False)


# --- Mixed trait list: Equatable + Writable conditional, Copyable plain ---


@fieldwise_init
struct Pair_Mixed[T: Copyable & Deinitable](
    Copyable,
    Equatable where conforms_to(T, Equatable),
    Writable where conforms_to(T, Writable),
):
    var first: Self.T
    var second: Self.T


def test_mixed_trait_list() raises:
    var pair1 = Pair_Mixed(first=1, second=2)
    var pair2 = Pair_Mixed(first=1, second=2)
    var pair3 = Pair_Mixed(first=3, second=4)
    assert_equal(pair1 == pair2, True)
    assert_equal(pair1 == pair3, False)
    var pair4 = pair1.copy()
    _ = pair4


# --- Box with Deinitable ---


@fieldwise_init
struct Box[T: Copyable & Deinitable](Equatable where conforms_to(T, Equatable)):
    var item: Self.T


def test_box_implicit_destructible() raises:
    var box = Box(42)
    assert_equal(box.item, 42)


# --- Point with Float64 ---


struct Point_2:
    var x: Float64
    var y: Float64

    def __init__(out self, x: Float64, y: Float64):
        (self.x, self.y) = (x, y)


def test_point_float() raises:
    var p = Point_2(3.0, 4.0)
    assert_equal(p.x, 3.0)
    assert_equal(p.y, 4.0)


# --- comptime members: constants, type aliases, parameterized aliases ---


@fieldwise_init
struct Matrix2D[dtype: DType, w: Int, h: Int]:
    pass


@fieldwise_init
struct CompTest[dtype: DType]:
    comptime default_size = 1024
    comptime DefaultMatrixType = Matrix2D[
        Self.dtype, Self.default_size, Self.default_size
    ]
    comptime SquareMatrixType[size: Int] = Matrix2D[Self.dtype, size, size]


def test_comptime_members() raises:
    # Access constants on the type and on an instance.
    assert_equal(CompTest[.int32].default_size, 1024)
    assert_equal(CompTest[.int32]().default_size, 1024)
    # Type aliases via comptime resolve to concrete types.
    var _default: CompTest[.int32].DefaultMatrixType = Matrix2D[
        DType.int32, 1024, 1024
    ]()
    # Parameterized comptime members specialize at use site.
    var _square: CompTest[.int32].SquareMatrixType[10] = Matrix2D[
        DType.int32, 10, 10
    ]()


def main() raises:
    test_point_distance()
    test_validation_error()
    test_color_manual()
    test_field_type_parameter()
    test_color_fieldwise()
    test_parameterized_pair()
    test_defaulted_parameters()
    test_trait_conformance()
    test_conformance_where()
    test_mixed_trait_list()
    test_box_implicit_destructible()
    test_point_float()
    test_comptime_members()
