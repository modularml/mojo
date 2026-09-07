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
# test_trait_declarations.mojo
# Tests for trait-declarations.mdx code examples.
# Skip: nested trait error, parameterized trait error, var field
#        error, where clause error, pass vs ... error, comptime
#        without initializer errors, missing method errors, missing
#        required member errors, type mismatch errors, default
#        method conflict errors, non-conforming argument errors.
# Print-only tests: test_required_methods, test_provided_methods,
#        test_able_to_say_hello, test_parameter_level_bounds, and
#        test_fieldwise_bounds use print() rather than assertions
#        because asserting on string representations is fragile
#        (reflection-based output format can change). These tests
#        still verify compilation, trait constraint satisfaction, and
#        dispatch — successful execution without raising is the signal.
from std.testing import assert_equal


# --- Required methods ---


trait RequiredMethods:
    def required_method(self):
        ...

    @staticmethod
    def required_static_method():
        ...


@fieldwise_init
struct SampleStruct_1(RequiredMethods):
    def required_method(self):
        print("Required method")

    @staticmethod
    def required_static_method():
        print("Required static method")


def test_required_methods():
    var s = SampleStruct_1()
    s.required_method()  # Required method
    SampleStruct_1.required_static_method()  # Required static method


# --- Provided methods ---


trait ProvidedMethods:
    def provided_method(self):
        print("Provided method")

    @staticmethod
    def provided_static_method():
        print("Provided static method")


@fieldwise_init
struct SampleStruct_2(ProvidedMethods):
    def provided_method(self):
        print("Overridden provided method")


def test_provided_methods():
    var sample = SampleStruct_2()
    sample.provided_method()  # Overridden provided method
    SampleStruct_2.provided_static_method()  # Provided static method


# --- Default return value ---


trait Describable:
    def describe(self) -> String:
        return "no description"


@fieldwise_init
struct Undescribed(Describable):
    pass


def test_describable() raises:
    var u = Undescribed()
    assert_equal(u.describe(), "no description")


# --- Empty struct with default methods ---


trait AbleToSayHello:
    def say_hello(self):
        print("Hello!")


@fieldwise_init
struct SampleStruct_3(AbleToSayHello):
    pass


def test_able_to_say_hello():
    var sample = SampleStruct_3()
    sample.say_hello()  # Hello!


# --- Associated types ---


trait Boxable:
    comptime Associated: Writable & Copyable & Deinitable

    def unbox(self) -> Self.Associated:
        ...


@fieldwise_init
struct ConcreteBox(Boxable):
    comptime Associated = String
    var value: Self.Associated

    def unbox(self) -> Self.Associated:
        return self.value.copy()


def test_associated_types() raises:
    var box = ConcreteBox(value="Hello")
    var unboxed = box.unbox()
    assert_equal(unboxed, "Hello")
    _ = unboxed^


# --- Associated type ---


comptime Base = Copyable & Deinitable & Writable


@fieldwise_init
struct Box_1[T: Base](Boxable):
    comptime Associated = Self.T
    var value: Self.Associated

    def unbox(self) -> Self.Associated:
        return self.value.copy()


def test_associated_type() raises:
    var box = Box_1[Int](value=42)
    var unboxed = box.unbox()
    assert_equal(unboxed, 42)


# --- Comptime constants ---


from std.sys.info import bit_width_of

comptime BaseElement = Copyable & Deinitable & Writable


trait Test:
    comptime Element: RegisterPassable
    comptime element_bitwidth = bit_width_of[Self.Element]()


@fieldwise_init
struct SampleStruct_4[T: BaseElement](Test):
    comptime Element = Int64
    var x: Self.T

    def show_element_bitwidth(self) raises:
        assert_equal(Self.element_bitwidth, 64)


def test_comptime_constants() raises:
    var s = SampleStruct_4[Int64](x=42)
    s.show_element_bitwidth()
    assert_equal(s.x, 42)


# --- Required values ---


trait Measurable:
    comptime unit: StaticString
    comptime always_positive: Bool

    def get_value(self) -> Float64:
        ...


def validate[T: Measurable](measurement: T) raises:
    comptime if T.always_positive:
        if Float64(measurement.get_value()) < 0.0:
            raise Error(t"{T.unit} cannot be negative")


@fieldwise_init
struct Pascals(Measurable):
    comptime unit: StaticString = "Pa"
    comptime always_positive: Bool = True
    var value: Float64

    def get_value(self) -> Float64:
        return self.value


def test_required_values() raises:
    validate(Pascals(value=101325.0))


# --- Trait refinement ---


trait Printable:
    def to_string(self) -> String:
        ...


trait PrettyPrintable(Printable):
    def to_pretty_string(self) -> String:
        ...


@fieldwise_init
struct Box_2[T: Copyable & Writable & Deinitable](PrettyPrintable):
    var value: Self.T

    def to_string(self) -> String:
        return String(t"Box({self.value})")

    def to_pretty_string(self) -> String:
        return String(t"Box with value: {self.value}")


def render[T: PrettyPrintable](item: T):
    print(item.to_string(), "-", item.to_pretty_string())


def test_trait_refinement() raises:
    render(Box_2(1))  # Box(1) - Box with value: 1
    render(Box_2("hello"))  # Box(hello) - Box with value: hello
    assert_equal(Box_2(1).to_string(), "Box(1)")
    assert_equal(Box_2(1).to_pretty_string(), "Box with value: 1")
    assert_equal(Box_2("hello").to_string(), "Box(hello)")
    assert_equal(Box_2("hello").to_pretty_string(), "Box with value: hello")


# --- Parameter-level trait constraints ---


def print_twice[T: Writable & Copyable](value: T):
    print(value, value)


def test_parameter_level_bounds():
    print_twice(42)  # 42 42
    print_twice("hi")  # hi hi


# --- Struct-level trait constraints ---


@fieldwise_init
struct Pair_1[T: Writable & Copyable & Deinitable](Writable):
    var first: Self.T
    var second: Self.T

    def write_to[W: Writer](self, mut writer: W):
        writer.write(t"({self.first}, {self.second})")


def test_struct_level_bounds() raises:
    var p = Pair_1(1, 2)
    assert_equal(String(p), "(1, 2)")


# --- Fieldwise trait constraints ---


@fieldwise_init
struct Pair_2[T: Copyable & Deinitable](
    Writable where conforms_to(T, Writable)
):
    var first: Self.T
    var second: Self.T


def test_fieldwise_bounds() raises:
    var pair = Pair_2(1, 2)
    print(pair)  # Pair_2[Int](first=1, second=2)


def main() raises:
    test_required_methods()
    test_provided_methods()
    test_describable()
    test_able_to_say_hello()
    test_associated_types()
    test_associated_type()
    test_comptime_constants()
    test_required_values()
    test_trait_refinement()
    test_parameter_level_bounds()
    test_struct_level_bounds()
    test_fieldwise_bounds()
