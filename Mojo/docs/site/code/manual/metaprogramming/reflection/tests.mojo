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
# Tests for reflection.mdx code examples.
# Skip: show_type/show_layout/log print output (path-dependent),
#        origin_of abort path (aborts by design),
#        get_linkage_name exact value (mangled symbol),
#        "no runtime cost" and "resolves at compile time" (codegen claims),
#        "name-based lookup requires a concrete type" (needs a
#        compile-failure test, not a runtime one),
#        the claim that a non-`@always_inline` caller reports its own line
#        (does not reproduce: a plain `def` also reports the call site, so
#        the behavior is optimizer-dependent and not safe to pin),
#        "print cleanly, without extra code" (compiling `Sensor` with a
#        `Writable` conformance and no `write_to` is the proof; the derived
#        rendering itself belongs to the compiler),
#        the wording of `prefix()` output (stdlib owns that format string).
#
# `reflect`, `conforms_to`, `materialize`, `type_of`, and `origin_of` are
# deliberately used without imports: the page claims they are built in, so
# adding imports here would silently retire that check.
from std.collections import Set
from std.testing import assert_equal, assert_true
from std.reflection import (
    source_location,
    call_location,
    get_function_name,
    get_linkage_name,
)
from std.os import abort


# --- Why reflection? ---


@fieldwise_init
struct Sensor(Equatable, Hashable, Writable):
    var id: Int
    var label: String
    var reading: Float64


def test_sensor_equality() raises:
    """Equatable auto-implemented via fieldwise reflection."""
    var a = Sensor(1, "temp", 98.6)
    var b = Sensor(1, "temp", 98.6)
    var c = Sensor(2, "pressure", 14.7)
    assert_true(a == b)
    assert_true(a != c)


def test_sensor_as_dict_key() raises:
    """`Hashable` plus `Equatable` lets `Sensor` key a `Dict`."""
    var d = Dict[Sensor, Int]()
    d[Sensor(1, "temp", 98.6)] = 10
    assert_equal(d[Sensor(1, "temp", 98.6)], 10)


def test_sensor_as_set_element() raises:
    """Equal sensors collapse to a single `Set` element."""
    var s = Set[Sensor]()
    s.add(Sensor(1, "temp", 98.6))
    s.add(Sensor(1, "temp", 98.6))
    s.add(Sensor(2, "pressure", 14.7))
    assert_equal(len(s), 2)


# --- Inspect a type ---


def test_base_name() raises:
    """Check `base_name()` strips parameters and module path."""
    assert_equal(reflect[List[Int]].base_name(), "List")
    assert_equal(reflect[Dict[String, Int]].base_name(), "Dict")


def test_name_is_qualified_where_base_name_is_not() raises:
    """`name()` adds the module path that `base_name()` strips."""
    comptime full = reflect[Optional[Float64]].name()
    comptime base = reflect[Optional[Float64]].base_name()
    assert_true(full != base)
    assert_true(full.find(base) >= 0)
    assert_true(full.find(".") >= 0)


def test_name_applies_parameters() raises:
    """Distinct parameters yield distinct names; `base_name()` drops them."""
    comptime int_name = reflect[Optional[Int]].name()
    comptime float_name = reflect[Optional[Float64]].name()
    assert_true(int_name != float_name)
    assert_equal(
        reflect[Optional[Int]].base_name(),
        reflect[Optional[Float64]].base_name(),
    )


@fieldwise_init
struct Config(Equatable):
    var host: String
    var port: Int
    var verbose: Bool
    var timeout: Float64


def test_type_name() raises:
    """Check `name()` returns the compiler-resolved name."""
    comptime name = reflect[Config].name()
    assert_true(name.find("Config") >= 0)


def test_field_count() raises:
    """Check `field_count()` counts the declared fields."""
    assert_equal(reflect[Config].field_count(), 4)


def test_field_names() raises:
    """Check `field_names()` returns names in declaration order."""
    var names = materialize[reflect[Config].field_names()]()
    assert_equal(String(names[0]), "host")
    assert_equal(String(names[3]), "timeout")


def test_field_types() raises:
    """Check each `field_types()` entry is usable as the field's own type."""
    comptime types = reflect[Config].field_types()
    comptime PortType = types[1]
    var port: PortType = 8080
    assert_equal(port, 8080)


def test_field_by_name() raises:
    """Check `field["host"]` gives a handle whose .T is usable."""
    # `field[name]` returns a `Reflected` handle itself (here `Reflected[String]`),
    # not the bare field type, so you can keep chaining reflection methods on it.
    comptime host_handle = reflect[Config].field["host"]
    var default_host: host_handle.T = "localhost"
    assert_equal(default_host, "localhost")


comptime DefaultItemCount = 10


# `Movable & Deinitable` is the minimal bound `List[Self.T]` accepts: bare
# `Movable` leaves the field non-`Deinitable`, bare `Deinitable` leaves it
# non-`Movable`.
struct ParameterizedStruct[
    T: Movable & Deinitable, item_count: Int = DefaultItemCount
]:
    var list: List[Self.T]

    def __init__(out self):
        self.list = List[Self.T](capacity=Self.item_count)


def test_reflect_parameterized_struct() raises:
    """Index-based iteration works on a type that takes parameters."""
    comptime P = ParameterizedStruct[String, item_count=5]
    assert_equal(reflect[P].field_count(), 1)
    var names = materialize[reflect[P].field_names()]()
    assert_equal(String(names[0]), "list")


def test_parameterized_name_applies_parameters() raises:
    """Changing a parameter value changes the resolved name."""
    comptime five = reflect[ParameterizedStruct[String, item_count=5]].name()
    comptime ten = reflect[ParameterizedStruct[String, item_count=10]].name()
    assert_true(five != ten)


# --- Detect field-level changes ---


def diff_fields[T: AnyType](a: T, b: T) -> List[String]:
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    var diffs = List[String]()

    comptime for idx in range(reflect[T].field_count()):
        comptime if conforms_to(types[idx], Equatable):
            ref a_val = reflect[T].field_ref[idx](a)
            ref b_val = reflect[T].field_ref[idx](b)
            if a_val != b_val:
                diffs.append(String(comptime (names[idx])))

    return diffs^


def test_diff_fields_detects_changes() raises:
    var old = Config("localhost", 8080, False, 30.0)
    var new_cfg = Config("localhost", 9090, True, 30.0)
    var changes = diff_fields(old, new_cfg)
    assert_equal(len(changes), 2)
    assert_equal(changes[0], "port")
    assert_equal(changes[1], "verbose")


def test_diff_fields_identical() raises:
    var a = Config("localhost", 8080, False, 30.0)
    var b = Config("localhost", 8080, False, 30.0)
    assert_equal(len(diff_fields(a, b)), 0)


def test_diff_fields_all_different() raises:
    var a = Config("localhost", 8080, False, 30.0)
    var b = Config("remote", 9090, True, 60.0)
    assert_equal(len(diff_fields(a, b)), 4)


def test_diff_fields_single_field() raises:
    var a = Config("localhost", 8080, False, 30.0)
    var b = Config("localhost", 8080, False, 99.0)
    var changes = diff_fields(a, b)
    assert_equal(len(changes), 1)
    assert_equal(changes[0], "timeout")


# `Opaque` conforms to neither `Equatable` nor `Copyable`, so it exercises
# the `conforms_to` guards that every all-conforming struct leaves dormant.
@fieldwise_init
struct Opaque(Movable):
    var tag: Int


@fieldwise_init
struct Mixed(Movable):
    var n: Int
    var blob: Opaque


def test_diff_fields_skips_non_equatable() raises:
    """Non-`Equatable` fields drop out of the comparison entirely."""
    var a = Mixed(1, Opaque(100))
    var b = Mixed(2, Opaque(999))
    var changes = diff_fields(a, b)
    assert_equal(len(changes), 1)
    assert_equal(changes[0], "n")


# --- Write once reuse everywhere ---


trait MakeCopyable:
    def copy_to(self, mut other: Self):
        comptime field_count = reflect[Self].field_count()
        comptime field_types = reflect[Self].field_types()

        comptime Usable = Copyable & Deinitable
        comptime for idx in range(field_count):
            comptime field_type = field_types[idx]
            comptime if conforms_to(field_type, Usable):
                reflect[Self].field_ref[idx](other) = (
                    reflect[Self].field_ref[idx](self).copy()
                )


@fieldwise_init
struct MultiType(MakeCopyable, Writable):
    var w: String
    var x: Int
    var y: Bool
    var z: Float64

    def write_to[W: Writer](self, mut writer: W):
        writer.write(String(t"[{self.w}, {self.x}, {self.y}, {self.z}]"))


def test_copy_to_transfers_values() raises:
    """Check `copy_to` duplicates all Copyable fields."""
    var original = MultiType("Hello", 1, True, 2.5)
    var target = MultiType("", 0, False, 0.0)
    original.copy_to(target)
    assert_equal(target.w, "Hello")
    assert_equal(target.x, 1)
    assert_equal(target.y, True)
    assert_equal(target.z, 2.5)


def test_copy_to_independent() raises:
    """Modifying original after copy_to doesn't affect target."""
    var original = MultiType("Hello", 1, True, 2.5)
    var target = MultiType("", 0, False, 0.0)
    original.copy_to(target)
    original.x = 999
    assert_equal(target.x, 1)


def test_multitype_write_to() raises:
    """`write_to` reaches every field, so the rendering is not empty."""
    var rendered = String(MultiType("Hello", 1, True, 2.5))
    assert_true(rendered.startswith("["))
    assert_true(rendered.endswith("]"))
    assert_true(rendered.find("Hello") >= 0)


@fieldwise_init
struct Pair(MakeCopyable):
    var a: Int
    var b: String


def test_copy_to_on_a_second_struct() raises:
    """A differently shaped struct gets `copy_to` with no implementation."""
    var src = Pair(7, "seven")
    var dst = Pair(0, "")
    src.copy_to(dst)
    assert_equal(dst.a, 7)
    assert_equal(dst.b, "seven")


@fieldwise_init
struct PartlyCopyable(MakeCopyable):
    var n: Int
    var blob: Opaque


def test_copy_to_skips_non_copyable_field() raises:
    """Fields outside `Copyable & Deinitable` are left untouched."""
    var src = PartlyCopyable(5, Opaque(100))
    var dst = PartlyCopyable(0, Opaque(999))
    src.copy_to(dst)
    assert_equal(dst.n, 5)
    assert_equal(dst.blob.tag, 999)


# --- Field layout ---


struct Packet:
    var flags: UInt8
    var id: UInt32
    var payload: UInt64


def test_field_offset_zero() raises:
    """First field is always at offset 0."""
    assert_equal(reflect[Packet].field_offset[index=0](), 0)


def test_field_offsets_exact() raises:
    """Offsets are 0, 4, 8: three bytes of padding follow the UInt8."""
    assert_equal(reflect[Packet].field_offset[index=0](), 0)
    assert_equal(reflect[Packet].field_offset[index=1](), 4)
    assert_equal(reflect[Packet].field_offset[index=2](), 8)


def test_field_offset_alignment() raises:
    """UInt32 field is at offset >= 4 due to alignment padding after UInt8."""
    comptime off = reflect[Packet].field_offset[index=1]()
    assert_true(off >= 4)


def test_field_offset_by_name() raises:
    """Check `field_offset` accepts `name=` and agrees with `index=`."""
    comptime by_name = reflect[Packet].field_offset[name="id"]()
    comptime by_index = reflect[Packet].field_offset[index=1]()
    assert_equal(by_name, by_index)


# --- type_of ---


def make_default[T: Defaultable]() -> T:
    return T()


def test_type_of() raises:
    """Check `type_of` captures the compile-time type of an expression."""
    var x = 42
    var y = make_default[type_of(x)]()
    assert_equal(y, 0)
    _ = x


# --- origin_of ---


def first_ref[T: Movable](ref list: List[T]) -> ref[list[0]] T:
    if not list:
        abort("empty list")
    return list[0]


def test_origin_of_first_ref() raises:
    """Check `first_ref` returns a reference tied to the list's origin."""
    var l: List = [1, 2, 3]
    ref x = first_ref(l)
    assert_equal(x, 1)
    x += 10
    assert_equal(l[0], 11)
    assert_equal(l[1], 2)
    assert_equal(l[2], 3)


# --- Source locations ---


def test_source_location_line() raises:
    """Check `source_location` returns a positive line number."""
    var loc = source_location()
    assert_true(loc.line() > 0)


def test_source_location_column() raises:
    """Check `source_location` returns a positive column number."""
    var loc = source_location()
    assert_true(loc.column() > 0)


def test_source_location_file_name() raises:
    """Check `source_location` returns a non-empty file name."""
    var loc = source_location()
    assert_true(loc.file_name())


def test_source_location_prefix() raises:
    """Check `prefix()` carries the message and prepends location context."""
    var loc = source_location()
    var msg = loc.prefix("test message")
    # A position past 0 means location context precedes the message.
    assert_true(msg.find("test message") > 0)


def helper_source_line() -> Int:
    return source_location().line()


def test_source_location_ignores_call_site() raises:
    """`source_location` reports its own line, so two call sites agree."""
    var first = helper_source_line()
    var second = helper_source_line()
    assert_equal(first, second)


@always_inline
def get_caller_line() -> Int:
    return call_location().line()


def test_call_location() raises:
    """Check `call_location` captures the immediate caller's line, not its own.
    """
    var line = get_caller_line()
    assert_true(line > 0)


def test_call_location_reports_caller_line() raises:
    """The reported line is the call site, not the line inside the helper."""
    var here = source_location().line()
    var caller = get_caller_line()
    assert_equal(caller, here + 1)


def test_call_location_tracks_each_call_site() raises:
    """Two call sites one line apart report lines one apart."""
    var first = get_caller_line()
    var second = get_caller_line()
    assert_equal(second, first + 1)


@always_inline
def one_level_caller_line() -> Int:
    return call_location().line()


@always_inline
def wrap_one_level() -> Int:
    return one_level_caller_line()


@always_inline
def two_level_caller_line() -> Int:
    return call_location[inline_count=2]().line()


@always_inline
def wrap_two_levels() -> Int:
    return two_level_caller_line()


def test_inline_count_default_stops_at_immediate_caller() raises:
    """The default (1) reports the line inside the wrapper, so distinct call
    sites agree."""
    var first = wrap_one_level()
    var second = wrap_one_level()
    assert_equal(first, second)


def test_inline_count_two_skips_a_level() raises:
    """`inline_count=2` skips the wrapper and reports this call site."""
    var here = source_location().line()
    var line = wrap_two_levels()
    assert_equal(line, here + 1)


@always_inline
def require(cond: Bool, msg: String = "requirement failed") raises:
    if not cond:
        raise Error(call_location().prefix(msg))


def test_require_carries_the_caller_message() raises:
    """`require` raises with the caller's message.

    That the location is the caller's, not `require`'s, is checked directly
    in `test_call_location_reports_caller_line` rather than by matching the
    formatted error text.
    """
    var caught = String("")
    try:
        require(False, "x must be > 10")
    except e:
        caught = String(e)
    assert_true(caught.find("x must be > 10") >= 0)


# --- Function names ---


def process_data():
    pass


def test_get_function_name() raises:
    """Check `get_function_name` returns the source-level name."""
    comptime name = get_function_name[process_data]()
    assert_equal(name, "process_data")


def test_get_linkage_name() raises:
    """Check `get_linkage_name` returns a non-empty mangled symbol."""
    comptime linkage = get_linkage_name[process_data]()
    assert_true(linkage)


def test_linkage_name_is_mangled() raises:
    """The linkage name is mangled, so it differs from the source name."""
    comptime source = get_function_name[process_data]()
    comptime linkage = get_linkage_name[process_data]()
    assert_true(source != linkage)


# --- Additional methods: conforms_to and where clause ---


def eq[T: AnyType](a: T, b: T) -> Bool where conforms_to(T, Equatable):
    return a == b


def test_eq_where_clause() raises:
    """Check `where conforms_to(T, Equatable)` enables `==` operator."""
    assert_true(eq(1, 1))
    assert_true(not eq(1, 2))
    assert_true(eq("hello", "hello"))


def test_conforms_to() raises:
    """Check `conforms_to` checks compile-time trait conformance."""
    assert_true(conforms_to(Int, Equatable))
    assert_true(conforms_to(String, Equatable))


def main() raises:
    # Why reflection?
    test_sensor_equality()
    test_sensor_as_dict_key()
    test_sensor_as_set_element()

    # Inspect a type
    test_base_name()
    test_name_is_qualified_where_base_name_is_not()
    test_name_applies_parameters()
    test_type_name()
    test_field_count()
    test_field_names()
    test_field_types()
    test_field_by_name()
    test_reflect_parameterized_struct()
    test_parameterized_name_applies_parameters()

    # Detect field-level changes
    test_diff_fields_detects_changes()
    test_diff_fields_identical()
    test_diff_fields_all_different()
    test_diff_fields_single_field()
    test_diff_fields_skips_non_equatable()

    # Write once reuse everywhere
    test_copy_to_transfers_values()
    test_copy_to_independent()
    test_multitype_write_to()
    test_copy_to_on_a_second_struct()
    test_copy_to_skips_non_copyable_field()

    # Field layout
    test_field_offset_zero()
    test_field_offsets_exact()
    test_field_offset_alignment()
    test_field_offset_by_name()

    # type_of
    test_type_of()

    # origin_of
    test_origin_of_first_ref()

    # Source locations
    test_source_location_line()
    test_source_location_column()
    test_source_location_file_name()
    test_source_location_prefix()
    test_source_location_ignores_call_site()
    test_call_location()
    test_call_location_reports_caller_line()
    test_call_location_tracks_each_call_site()
    test_inline_count_default_stops_at_immediate_caller()
    test_inline_count_two_skips_a_level()
    test_require_carries_the_caller_message()

    # Function names
    test_get_function_name()
    test_get_linkage_name()
    test_linkage_name_is_mangled()

    # Additional methods
    test_eq_where_clause()
    test_conforms_to()
