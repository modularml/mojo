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
# Tests for initialization.mdx code examples and prose claims.
#
# Page claims, and where each is verified:
#   1. Calling `Person("Alice", 30)` is syntactic sugar for calling the
#      initializer directly, `Person.__init__("Alice", 30)`.
#      -> test_initializer_call_is_sugar (both spellings produce the same value)
#   2. A struct needs both fieldwise and logical initialization before use.
#      -> test_construct_establishes_both, test_declare_then_construct
#   3. An initializer must populate every field.
#      -> not runtime-testable; see "Compile errors" below, claim C2
#   4. The initializer signature doesn't have to mirror the struct's fields:
#      parameters, constants, and external values can initialize them.
#      -> test_parameter_initialization, test_constant_initialization,
#         test_external_value_initialization
#   5. Methods can't be called until all fields are initialized.
#      -> not runtime-testable; see claim C3
#   6. Field initialization is limited to `__init__()` methods; regular methods
#      can't initialize individual fields of an `out` argument.
#      -> not runtime-testable; see claims C4 and C5
#
# Compile errors the page describes. Each is a compile failure, so it can't run
# here. Verified against the compiler on 2026-08-21, Mojo 1.1.0.dev0, and
# recorded with the diagnostic that fires:
#   C1. Fields assigned individually, without an `__init__()` call:
#       "'me' used with all fields manually initialized but without calling an
#       '__init__' method"
#   C2. A field left unset in `__init__()`:
#       "'self.age' is uninitialized at the implicit return from this function"
#   C3. A method called before every field is set:
#       "use of uninitialized value 'self.name'"
#   C4. `out self` on a method other than `__init__()`:
#       "'self' argument must have type 'Self'; use a 'where' clause to
#       constrain the 'Self' type instead"
#   C5. A regular method assigning every field of an `out` argument: the same
#       diagnostic as C1, reported against that argument.
#
# The page shows two structs, both named `Person`: one declares `__init__()`,
# the other uses `@fieldwise_init`. One file can't hold two types of the same
# name, so the second is `FieldwisePerson` here.
from std.math import pi
from std.testing import assert_equal, assert_true

# --- The basics ---


struct Person:
    var name: String
    var age: Int

    def __init__(out self, name: String, age: Int):
        self.name = name
        self.age = age


def test_construct_establishes_both() raises:
    var me = Person("Alice", 30)
    assert_equal(me.name, "Alice")
    assert_equal(me.age, 30)


def test_initializer_call_is_sugar() raises:
    var sugared = Person("Alice", 30)

    var direct: Person
    direct = Person.__init__("Alice", 30)  # identical to the line above

    assert_equal(sugared.name, direct.name)
    assert_equal(sugared.age, direct.age)


# --- Fieldwise vs logical initialization ---


@fieldwise_init
struct FieldwisePerson(Writable):
    var name: String
    var age: Int


def test_declare_then_construct() raises:
    var me: FieldwisePerson  # declared, not initialized
    me = FieldwisePerson("Alice", 30)  # logically and fieldwise initialized
    assert_equal(me.name, "Alice")
    assert_equal(me.age, 30)

    # `print(me)` on the page relies on the `Writable` default that reflection
    # supplies, since the struct writes no `write_to()` of its own. The default
    # leads with the struct's own name, so the page's `Person` renders as
    # "Person(name=Alice, age=30)". The rendering is a stdlib default, not a
    # page claim.
    assert_equal(String(me), "FieldwisePerson(name=Alice, age=30)")


# --- Inside __init__() ---


@fieldwise_init
struct Greeter:
    var name: String

    def greeting(self) -> String:
        return String("hello ", self.name)


def test_methods_after_fields_are_set() raises:
    # The page's `greet()` example calls a method only after every field is
    # set; that ordering is what compiles.
    var g = Greeter("Alice")
    assert_equal(g.greeting(), "hello Alice")


# --- Initializing fields from parameters ---


struct Buffer[T: Copyable & Deinitable, Count: Int]:
    var _store: List[Self.T]

    def __init__(out self):
        self._store = List[Self.T](capacity=Self.Count)

    def append(mut self, var value: Self.T):
        self._store.append(value^)


def test_parameter_initialization() raises:
    var b = Buffer[Int, 8]()
    assert_equal(len(b._store), 0)
    b.append(1)
    assert_equal(len(b._store), 1)


# --- Initializing fields from constants and external values ---


struct Config:
    var string: String
    var default_angle: Float64
    var uuid: String

    def __init__(out self):
        self.string = ""  # constant
        self.default_angle = pi / 2.0  # external value
        self.uuid = Self._uuid()

    @staticmethod
    def _uuid() -> String:
        return String("00000000-0000-0000-0000-000000000000")


def test_constant_initialization() raises:
    var c = Config()
    assert_equal(c.string, "")


def test_external_value_initialization() raises:
    var c = Config()
    assert_equal(c.default_angle, pi / 2.0)
    assert_true(c.default_angle > 1.5707 and c.default_angle < 1.5708)
    assert_equal(c.uuid.byte_length(), 36)


def main() raises:
    test_construct_establishes_both()
    test_initializer_call_is_sugar()
    test_declare_then_construct()
    test_methods_after_fields_are_set()
    test_parameter_initialization()
    test_constant_initialization()
    test_external_value_initialization()
