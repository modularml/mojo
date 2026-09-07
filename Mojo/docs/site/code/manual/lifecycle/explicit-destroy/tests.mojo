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
# Tests for explicit-destroy.mdx code examples and prose claims.
#
# The page demonstrates destruction by printing. Printing can't be asserted, so
# the types here record into a list and the tests assert the recorded strings
# against the page's documented output, so a change in either the order or the
# wording fails a test.
#
# Page claims, and where each is verified:
#   1. `Deinitable where False` opts a type out of automatic destruction, and
#      its values must be consumed by a named deinitializer or transferred.
#      -> test_named_deinitializer_consumes, test_transfer_defers_destruction;
#         the failure to consume is claim C1 below
#   2. A type can offer several cleanup paths, one per named deinitializer.
#      -> test_multiple_cleanup_paths
#   3. A named deinitializer can raise; `__deinit__()` can't.
#      -> test_raising_deinitializer; `__deinit__()` is claim C2
#   4. `deinit self` consumes the value at the call, whether the call returns
#      or raises, so the caller can't invoke a second deinitializer.
#      -> test_raising_deinitializer_consumes (the caller needs no cleanup on
#         the handler path); the second call is claim C3
#   5. Explicitly destroyed fields must be disposed before a deinitializer
#      raises; implicitly destructible fields are cleaned up during unwinding.
#      -> test_dispose_before_raise, test_implicit_field_cleaned_on_unwind;
#         raising with a live linear field is claim C4
#   6. A custom `where` message improves the diagnostic.
#      -> not runtime-testable; claim C1 carries the message text
#   7. Parameterized code over `AnyType`/`Movable` accepts explicitly destroyed
#      values but can't destroy them, except by returning them or by receiving
#      a deinitializer.
#      -> test_pass_through, test_consuming_method, test_deinitable_function;
#         the abandoning form is claim C5
#   8. The documented output of the `Tally` and `Basic` examples.
#      -> test_documented_tally_output, test_documented_basic_output
#   9. A lambda supplies the type-specific deinitializer a parametric consumer
#      can't name.
#      -> test_consuming_method
#  10. One generic lambda covers every `Deinitable` type.
#      -> test_implicit_consumer_covers_deinitable; its limit is claim C7
#
# Compile errors the page describes, or that these claims imply. Each is a
# compile failure, so it can't run here. Verified against the compiler on
# 2026-08-21, Mojo 1.1.0.dev0, and recorded with the diagnostic that fires:
#   C1. A value left unconsumed: "'tally' abandoned without being explicitly
#       destroyed: type 'Tally' does not conform to 'Deinitable' and must be
#       explicitly destroyed". With a `where` message, that message replaces
#       the trailing explanation.
#   C2. `raises` on `__deinit__()`: "destructor must not declare 'raises';
#       remove the 'raises' keyword"
#   C3. A second deinitializer after a raising one, including in the `except`
#       block: "use of uninitialized value 't'"
#   C4. A deinitializer that raises while an explicitly destroyed field is
#       still live: "'self' abandoned without being explicitly destroyed: An
#       `Allocation` owns heap storage and must be consumed before it goes out
#       of scope."
#   C5. `def owning_function[T: AnyType](var value: T)`: "'value' abandoned
#       without being explicitly destroyed: unhandled explicitly destroyed
#       type 'AnyType'", with the note "consider adding trait conformance to
#       Deinitable". The error is reported at the function's own definition,
#       not at a call site.
#   C6. A function type with a `deinit` argument: "function types do not
#       support 'deinit'; replace with 'var'"
#   C7. `implicit_consumer[Tally]`, instantiating the `Deinitable` lambda for
#       an explicitly destroyed type: "parameter 'T' has 'Deinitable' type,
#       but value has type 'AnyStruct[Tally]'"
#
# Mechanics worth knowing, confirmed while writing these tests:
#   M1. A runtime argument can't carry an inline function type. The page's
#       `consuming_function`/`consuming_method` signatures are right because
#       they name the type as a parameter (`Cleanup: def(var T)`); spelling it
#       `consume: def(var T)` instead fails inside the body and at the call.
#   M2. A parametric lambda stays parametric, so `Cleanup` binds only after
#       instantiation: `implicit_consumer[Basic]`, not `implicit_consumer`.
#       Binding the lambda with `var` instead of `comptime` also fails, since
#       it decays to a thin function pointer, which is a value, not a type.
#   M3. Calling a named deinitializer on a temporary needs no transfer sigil:
#       `Transaction(log, 0).rollback()` compiles, and adding `^` warns
#       "transfer from an owned value has no effect and can be removed".
#
from std.memory import Allocation, Layout, alloc, dealloc
from std.reflection import reflect
from std.testing import assert_equal, assert_raises

# Testing constraint worth knowing: any raising call, including `assert_equal`,
# made while an explicitly destroyed value is still live leaves that value
# unconsumed on the unwind path, which the compiler rejects with "value was not
# consumed when an error is thrown". These tests read what they need into a
# local, consume the value, and assert afterward.

# --- Recording helper: an origin-erased handle to a list of events ---

comptime Recorder = Pointer[List[String], MutUntrackedOrigin]


def recorder(mut events: List[String]) -> Recorder:
    return Pointer(to=events).unsafe_origin_cast[MutUntrackedOrigin]()


def record(into: Recorder, var event: String):
    into[].append(event^)


# --- Opting out of automatic destruction ---


struct Example(Deinitable where False):
    var events: Recorder

    def __init__(out self, events: Recorder):
        self.events = events

    def cleanup(deinit self):
        record(self.events, "cleaned up")


def test_named_deinitializer_consumes() raises:
    var events = List[String]()
    var log = recorder(events)

    var value = Example(log)
    value^.cleanup()

    assert_equal(len(events), 1)
    assert_equal(events[0], "cleaned up")


def hand_back(var value: Example) -> Example:
    return value^


def test_transfer_defers_destruction() raises:
    # Transferring the value out of scope satisfies the requirement at this
    # point; the destruction obligation moves with the value.
    var events = List[String]()
    var log = recorder(events)

    var value = hand_back(Example(log))
    var before_cleanup = len(events)  # nothing destroyed yet
    value^.cleanup()

    assert_equal(before_cleanup, 0)
    assert_equal(len(events), 1)


# --- When to use explicit destruction: multiple cleanup paths ---


struct Transaction(
    Deinitable where (False, "call 'commit()' or 'rollback()'"), Movable
):
    var events: Recorder
    var rows: Int

    def __init__(out self, events: Recorder, rows: Int):
        self.events = events
        self.rows = rows

    def commit(deinit self) raises:
        if self.rows == 0:
            raise Error("nothing to commit")
        record(self.events, String("committed ", self.rows))

    def rollback(deinit self):
        record(self.events, "rolled back")


struct MutexGuard(Deinitable where False, Movable):
    var events: Recorder

    def __init__(out self, events: Recorder):
        self.events = events
        record(self.events, "locked")

    def unlock(deinit self):
        record(self.events, "unlocked")


def test_multiple_cleanup_paths() raises:
    var events = List[String]()
    var log = recorder(events)

    Transaction(log, 3).commit()
    Transaction(log, 0).rollback()

    assert_equal(events[0], "committed 3")
    assert_equal(events[1], "rolled back")


def test_cleanup_order_with_a_guard() raises:
    var events = List[String]()
    var log = recorder(events)

    var guard = MutexGuard(log)
    record(log, "critical section")
    guard^.unlock()  # the lock releases exactly here

    var expected = ["locked", "critical section", "unlocked"]
    for i in range(len(expected)):
        assert_equal(events[i], expected[i])


# --- Raising deinitializers ---


def test_raising_deinitializer() raises:
    var events = List[String]()
    var log = recorder(events)

    with assert_raises(contains="nothing to commit"):
        Transaction(log, 0).commit()

    assert_equal(len(events), 0)  # the failing path recorded nothing


def test_raising_deinitializer_consumes() raises:
    # The value is consumed at the call, so the handler has no cleanup left to
    # do and the compiler asks for none.
    var events = List[String]()
    var log = recorder(events)
    var caught = String("")

    var t = Transaction(log, 0)
    try:
        t^.commit()
    except e:
        caught = String(e)

    assert_equal(caught, "nothing to commit")


struct Counters(
    Deinitable where (False, "call 'flush()' or 'drop()'"), Movable
):
    var counts: Allocation[Int64]
    var events: Recorder

    def __init__(out self, buckets: Int, events: Recorder):
        self.counts = alloc(Layout[Int64](count=buckets))
        self.counts.unsafe_span().fill(0)
        self.events = events

    def bump(mut self, bucket: Int):
        self.counts.unsafe_span()[bucket] += 1

    # The linear field is released first, so the raise happens with no field
    # left to dispose of. Raising before the `dealloc` doesn't compile: see C4.
    def flush(deinit self) raises:
        var total = Int(self.counts.unsafe_span()[0])
        dealloc(self.counts^)
        record(self.events, "storage released")
        if total == 0:
            raise Error("nothing recorded")
        record(self.events, String("flushed ", total))

    def drop(deinit self):
        dealloc(self.counts^)


def test_dispose_before_raise() raises:
    var events = List[String]()
    var log = recorder(events)

    with assert_raises(contains="nothing recorded"):
        Counters(2, log).flush()

    # The storage was freed before the failure was reported.
    assert_equal(events[0], "storage released")
    assert_equal(len(events), 1)


def test_dispose_before_raise_success_path() raises:
    var events = List[String]()
    var log = recorder(events)

    var counters = Counters(2, log)
    counters.bump(0)
    counters^.flush()

    assert_equal(events[0], "storage released")
    assert_equal(events[1], "flushed 1")


struct Draft(Deinitable where (False, "call 'save()' or 'discard()'"), Movable):
    var data: String  # implicitly destructible

    def __init__(out self, var data: String):
        self.data = data^

    def save(deinit self) raises:
        if self.data.byte_length() == 0:
            raise Error("nothing to save")  # raises with the field still live
        print(self.data)

    def discard(deinit self):
        pass


def test_implicit_field_cleaned_on_unwind() raises:
    # Unlike a linear field, an implicitly destructible field can still be live
    # when the deinitializer raises. It's released during unwinding.
    with assert_raises(contains="nothing to save"):
        Draft("").save()


# --- Parameterized code and explicit destruction ---


def deinitable_function[T: Deinitable](var value: T):
    pass  # value.__deinit__() called automatically


def pass_through[T: Movable](var value: T) -> T:
    # perform work with value
    return value^


def consuming_method[
    T: Movable, //, Cleanup: def(var T)
](var value: T, consume: Cleanup) -> String:
    var name = String(reflect[T].name())
    consume(value^)
    return name


# --- Example types for parameterized destruction ---


@fieldwise_init
struct Basic(Movable):
    var string: String
    var events: Recorder

    def __deinit__(deinit self):
        record(self.events, String("Destroying Basic: ", self.string))


struct Tally(
    Deinitable where (False, "call 'destroy()' to free the counters"), Movable
):
    var counts: Allocation[Int64]
    var events: Recorder

    def __init__(out self, buckets: Int, events: Recorder):
        self.counts = alloc(Layout[Int64](count=buckets))
        self.counts.unsafe_span().fill(0)
        self.events = events

    def record_bucket(mut self, bucket: Int):
        self.counts.unsafe_span()[bucket] += 1

    def destroy(deinit self):
        record(
            self.events,
            String("Destroying Tally: ", self.counts.unsafe_span()),
        )
        dealloc(self.counts^)


def test_deinitable_function() raises:
    var events = List[String]()
    var log = recorder(events)

    deinitable_function(Basic("implicit", log))
    assert_equal(events[0], "Destroying Basic: implicit")


def test_documented_tally_output() raises:
    var events = List[String]()
    var log = recorder(events)

    var tally = Tally(3, log)
    tally.record_bucket(2)
    tally = pass_through(tally^)
    tally^.destroy()

    assert_equal(events[0], "Destroying Tally: [0, 0, 1]")


def test_documented_basic_output() raises:
    var events = List[String]()
    var log = recorder(events)

    var basic = Basic("Hello", log)
    _ = pass_through(basic^)

    assert_equal(events[0], "Destroying Basic: Hello")


def test_pass_through() raises:
    # The function doesn't need to know the destruction model of either type.
    var events = List[String]()
    var log = recorder(events)

    var tally = pass_through(Tally(2, log))
    tally^.destroy()
    assert_equal(events[0], "Destroying Tally: [0, 0]")


def test_consuming_method() raises:
    var events = List[String]()
    var log = recorder(events)

    var tally = Tally(2, log)
    tally.record_bucket(1)

    comptime tally_consumer = lambda (var t: Tally): t^.destroy()
    var name = consuming_method(tally^, tally_consumer)

    # `reflect` reports the module-qualified name.
    assert_equal(name, "tests.Tally")
    assert_equal(events[0], "Destroying Tally: [0, 1]")


def test_implicit_consumer_covers_deinitable() raises:
    var events = List[String]()
    var log = recorder(events)

    # Works across all Deinitable types. The instantiation is required: a
    # parametric lambda stays parametric, and `Cleanup` is one concrete
    # function type.
    comptime implicit_consumer = lambda [T: Deinitable](
        var value: T
    ): T.__deinit__(value^)

    var name = consuming_method(Basic("World", log), implicit_consumer[Basic])
    assert_equal(name, "tests.Basic")
    assert_equal(events[0], "Destroying Basic: World")


def main() raises:
    test_named_deinitializer_consumes()
    test_transfer_defers_destruction()
    test_multiple_cleanup_paths()
    test_cleanup_order_with_a_guard()
    test_raising_deinitializer()
    test_raising_deinitializer_consumes()
    test_dispose_before_raise()
    test_dispose_before_raise_success_path()
    test_implicit_field_cleaned_on_unwind()
    test_deinitable_function()
    test_documented_tally_output()
    test_documented_basic_output()
    test_pass_through()
    test_consuming_method()
    test_implicit_consumer_covers_deinitable()
