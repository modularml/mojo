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
# Tests for death.mdx code examples and prose claims.
#
# The page demonstrates destruction by printing from `__deinit__()`. Printing
# can't be asserted, so the types here record into a list instead and the tests
# assert the recorded order. The recorded strings match the page's output text,
# so a change in either the order or the wording fails a test.
#
# Page claims, and where each is verified:
#   1. Destruction is as-soon-as-possible: a value dies after its last use,
#      not at the end of a block or expression.
#      -> test_asap_within_expression, test_asap_before_block_end
#   2. The documented destruction order for `a + (b - c) * d`, including the
#      three unnamed intermediates.
#      -> test_documented_destruction_order (asserts all seven entries in order)
#   3. Every value is initialized once and deinitialized once.
#      -> test_documented_destruction_order (seven values, seven records)
#   4. `__deinit__()` uses the `deinit` convention, and the value and its
#      fields stay valid while it runs.
#      -> test_fields_valid_during_deinit
#   5. Mojo generates `__deinit__()` for every struct with deinitializable
#      fields.
#      -> test_generated_deinitializer (conformance, no explicit method)
#   6. A deinitializer can move values out of fields without reinitializing
#      them.
#      -> test_move_out_of_field
#   7. Struct fields use ASAP destruction within `__deinit__()`, so a field
#      whose last use is earlier dies earlier.
#      -> test_field_asap_order
#   8. A custom deinitializer can close an owned resource.
#      -> test_custom_deinitializer_closes_file
#   9. `_ = value` marks a last use, so the deinitializer runs after that
#      statement, which extends the value's lifetime to that point.
#      -> test_lifetime_extension_orders_release
#  10. An RAII guard is released too early without the extension, and neither
#      case is a compiler error.
#      -> test_guard_released_early, test_guard_held_to_discard
#  11. Discarding a move-only value needs no transfer sigil.
#      -> test_discard_without_transfer_sigil
#
# Not tested:
#   - "In general, don't call `__deinit__()` directly" (advice, not a compiler
#     rule: the direct call is accepted, which is what the page's own carve-out
#     for wrapping a deinitializer in parameterized code depends on, and what
#     explicit-destroy.mdx's `implicit_consumer` lambda uses)
#   - the use-after-free half of the origin-erasure hazard: reading through a
#     pointer whose owner has already died is undefined behavior, so the test
#     asserts only the extended case, where the read is well defined
from std.os import remove
from std.os.path import exists
from std.tempfile import NamedTemporaryFile
from std.testing import assert_equal, assert_false, assert_true
from std.traits import Deinitable
from std.memory import Allocation, Layout, alloc, dealloc

# --- Recording helper: an origin-erased handle to a list of events ---

comptime Recorder = Pointer[List[String], MutUntrackedOrigin]


def recorder(mut events: List[String]) -> Recorder:
    return Pointer(to=events).unsafe_origin_cast[MutUntrackedOrigin]()


def record(into: Recorder, var event: String):
    into[].append(event^)


# --- When Mojo destroys values ---


@fieldwise_init
struct Number(Copyable, Writable):
    var value: Int
    var events: Recorder

    def __deinit__(deinit self):
        record(self.events, String("Destroying Number(value=", self.value, ")"))

    def __add__(self, other: Self) -> Self:
        return Number(self.value + other.value, self.events)

    def __sub__(self, other: Self) -> Self:
        return Number(self.value - other.value, self.events)

    def __mul__(self, other: Self) -> Self:
        return Number(self.value * other.value, self.events)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Number(value=", self.value, ")")


def test_documented_destruction_order() raises:
    var events = List[String]()
    var log = recorder(events)

    var a = Number(1, log)
    var b = Number(2, log)
    var c = Number(3, log)
    var d = Number(4, log)

    #                1    2   3    4
    var rendered = String(a + (b - c) * d)
    assert_equal(rendered, "Number(value=-3)")

    # The page's documented order: the parenthesized subtraction first, then
    # the multiplication, then the addition, then the result.
    var expected = [
        "Destroying Number(value=3)",  # subtraction
        "Destroying Number(value=2)",  # subtraction
        "Destroying Number(value=-1)",  # multiplication, intermediate
        "Destroying Number(value=4)",  # multiplication
        "Destroying Number(value=-4)",  # addition, intermediate
        "Destroying Number(value=1)",  # addition
        "Destroying Number(value=-3)",  # result
    ]
    assert_equal(len(events), len(expected))
    for i in range(len(expected)):
        assert_equal(events[i], expected[i])


def test_asap_within_expression() raises:
    # An intermediate dies while the enclosing expression is still evaluating,
    # so an event is already recorded before the statement ends.
    var events = List[String]()
    var log = recorder(events)
    var a = Number(1, log)
    var b = Number(2, log)

    # The intermediate from `a + b` is destroyed before `len(events)` is read
    # later in the same expression, so the count is already non-zero.
    var sum_plus_count = (a + b).value + len(events)
    assert_true(sum_plus_count > 3)

    # Three values in total: the intermediate, then `a` and `b`.
    assert_equal(len(events), 3)


def test_asap_before_block_end() raises:
    # `n`'s last use is the assert below, not the end of the function.
    var events = List[String]()
    var log = recorder(events)
    var counted: Int

    var n = Number(7, log)
    counted = n.value
    assert_equal(len(events), 1)  # already destroyed, mid-block

    assert_equal(counted, 7)


# --- Deinitializer behavior ---


@fieldwise_init
struct Fields(Movable):
    var name: String
    var events: Recorder

    def __deinit__(deinit self):
        # The value and its fields are still valid here.
        record(self.events, String("deinit saw ", self.name))


def test_fields_valid_during_deinit() raises:
    var events = List[String]()
    var log = recorder(events)
    _ = Fields("intact", log)
    assert_equal(len(events), 1)
    assert_equal(events[0], "deinit saw intact")


@fieldwise_init
struct NoExplicitDeinit(Movable):
    var owned_string: String  # a deinitializable field


def test_generated_deinitializer() raises:
    # No `__deinit__()` is written, yet the type is `Deinitable`.
    assert_true(conforms_to(NoExplicitDeinit, Deinitable))
    _ = NoExplicitDeinit("released by the generated deinitializer")


# --- Moving values out of fields ---


@fieldwise_init
struct Holder(Movable):
    var name: String
    var events: Recorder

    def __deinit__(deinit self):
        var name = self.name^  # OK: can take ownership of fields
        record(self.events, String("moved out ", name))


def test_move_out_of_field() raises:
    var events = List[String]()
    var log = recorder(events)
    _ = Holder("payload", log)
    assert_equal(events[0], "moved out payload")


@fieldwise_init
struct Marker(Movable):
    var name: String
    var events: Recorder

    def use(self):
        record(self.events, String("use ", self.name))

    def __deinit__(deinit self):
        record(self.events, String("destroy ", self.name))


@fieldwise_init
struct S(Movable):
    var a: Marker
    var b: Marker
    var events: Recorder

    def __deinit__(deinit self):
        # `a` has no use in this body, so Mojo destroys it here.
        self.b.use()
        # `b`'s last use was the line above, so Mojo destroys it there.


def test_field_asap_order() raises:
    var events = List[String]()
    var log = recorder(events)
    _ = S(Marker("a", log), Marker("b", log), log)

    var expected = ["destroy a", "use b", "destroy b"]
    assert_equal(len(events), len(expected))
    for i in range(len(expected)):
        assert_equal(events[i], expected[i])


# --- Custom deinitializers ---


struct QuickLogger:
    var temporary_file: NamedTemporaryFile
    var path: String

    def __init__(out self) raises:
        self.temporary_file = NamedTemporaryFile(mode="w", delete=False)
        self.path = self.temporary_file.name

    def log(mut self, message: String) raises:
        self.temporary_file.write(message + "\n")

    def __deinit__(deinit self):
        try:
            self.temporary_file.close()
        except:
            pass


def test_custom_deinitializer_closes_file() raises:
    var path: String

    var ql = QuickLogger()
    path = ql.path
    ql.log("This is a test log message.")
    ql.log("This is the last use of 'ql'")

    # `ql` is destroyed above, at its last use, so the file is closed and
    # (with delete=False) still on disk.
    assert_true(exists(path))
    remove(path)
    assert_false(exists(path))


# --- Explicit lifetime extension ---


@fieldwise_init
struct Guard(Movable):
    var name: String
    var events: Recorder

    def __deinit__(deinit self):
        record(self.events, String("released ", self.name))


def test_guard_released_early() raises:
    # No extension: the guard's last use is the length check, so the lock is
    # released before the work it was meant to protect. This compiles, and the
    # compiler reports nothing.
    var events = List[String]()
    var log = recorder(events)

    var g = Guard("early", log)
    assert_equal(g.name.byte_length(), 5)  # last use of `g`
    record(log, "work")

    assert_equal(events[0], "released early")
    assert_equal(events[1], "work")


def test_guard_held_to_discard() raises:
    # With the extension, the release lands after the work.
    var events = List[String]()
    var log = recorder(events)

    var g = Guard("held", log)
    record(log, "work")
    _ = g  # last use of `g` moves here

    assert_equal(events[0], "work")
    assert_equal(events[1], "released held")


def test_lifetime_extension_orders_release() raises:
    # The page's `s` and `t` pair: `s` dies at its last use, while `t` is held
    # until the discard line.
    var events = List[String]()
    var log = recorder(events)

    var s = Guard("s", log)
    assert_equal(s.name, "s")  # last use of `s`

    var t = Guard("t", log)
    assert_equal(t.name, "t")

    record(log, "some time later")
    _ = t  # `t` is destroyed after this line

    var expected = ["released s", "some time later", "released t"]
    for i in range(len(expected)):
        assert_equal(events[i], expected[i])


struct Owner(Movable):
    var storage: Allocation[Int64]

    def __init__(out self):
        self.storage = alloc(Layout[Int64](count=1))
        self.storage.unsafe_span().fill(7)

    def __deinit__(deinit self):
        dealloc(self.storage^)


def test_origin_erased_pointer_needs_extension() raises:
    # `unsafe_ptr()` alone borrows the owner, so origins keep it alive. Erasing
    # the origin removes that protection, and the discard is what keeps the
    # storage valid for the read.
    var o = Owner()
    var p = o.storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    assert_equal(p[], 7)
    _ = o  # without this, the read above is a use-after-free


def test_discard_without_transfer_sigil() raises:
    # A move-only value: no `^` is needed to discard it, since the compiler
    # doesn't move the discarded value.
    var events = List[String]()
    var log = recorder(events)
    var only_movable = Guard("move-only", log)
    _ = only_movable
    assert_equal(events[0], "released move-only")


def main() raises:
    test_documented_destruction_order()
    test_asap_within_expression()
    test_asap_before_block_end()
    test_fields_valid_during_deinit()
    test_generated_deinitializer()
    test_move_out_of_field()
    test_field_asap_order()
    test_custom_deinitializer_closes_file()
    test_guard_released_early()
    test_guard_held_to_discard()
    test_lifetime_extension_orders_release()
    test_origin_erased_pointer_needs_extension()
    test_discard_without_transfer_sigil()
