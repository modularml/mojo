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

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | FileCheck %s
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics

from std.builtin._coroutine import Coroutine, RaisingCoroutine, AnyCoroutine


@explicit_destroy("Must use consume!")
struct EmptyExplicit(Deinitable where False, Movable where False):
    def __init__(out self):
        pass

    def consume(deinit self):
        pass


def errorExample():
    # expected-error @below {{Must use consume!}}
    _ = EmptyExplicit()


@explicit_destroy("Must use consume!")
struct DeinitableContainerOfExplicitWithAutoDel(
    Deinitable where False, Movable where False
):
    var m: EmptyExplicit

    def __init__(out self):
        self.m = EmptyExplicit()


def testDeinitableContainerOfExplicitWithAutoDel():
    # expected-error @below {{Must use consume!}}
    _ = DeinitableContainerOfExplicitWithAutoDel()


struct DeinitableContainerOfExplicitWithIncompleteDel(Movable where False):
    var m: EmptyExplicit

    def __init__(out self):
        self.m = EmptyExplicit()

    # expected-error @below {{Must use consume!}}
    def __deinit__(deinit self):
        pass


# CHECK-LABEL: @"test_any_type_error
# expected-error @below {{unhandled explicitly destroyed type 'AnyType'}}
# expected-note @below {{consider adding trait conformance to Deinitable}}
def test_any_type_error[T: AnyType](var x: T):
    pass


# TODO(MOCO-2363): re-enable the test below
#
# # TODO(MOCO-1468): Require error message for @explicit_destroy
# @explicit_destroy
# trait LinearCopyable(ImplicitlyCopyable):
#     pass


# # C_HECK-LABEL: @"receiveLinearCopyable
# # e_xpected-error @below {{Unhandled explicit_destroy type LinearCopyable}}
# def receiveLinearCopyable[T: LinearCopyable](var x: T):
#     pass


# @explicit_destroy
# struct LinearCopyableStruct(LinearCopyable):
#     def __init__(out self, *, copy: Self):
#         pass


# # C_HECK-LABEL: @"upcastLinearCopyable
# def upcastLinearCopyable(var x: LinearCopyableStruct):
#     receiveLinearCopyable(x)


# CHECK-LABEL: lit.fn @"callsWith
def callsWith():
    # expected-error @below {{type 'Coroutine' does not conform to 'Deinitable' and must be explicitly destroy}}
    _ = testAsyncVoid()
    # CHECK-NOT: lit.call {{.*}}__deinit__


# CHECK-LABEL: lit.fn @"testAsyncVoid
async def testAsyncVoid():
    pass


# CHECK-LABEL: lit.struct.decl @ExplicitWithDeinit


# MOCO-2787 - Linear types do not error if they contain an explicit deinit
@explicit_destroy("must use __deinit__() explicitly")
struct ExplicitWithDeinit(Deinitable where False, Movable where False):
    def __init__(out self):
        pass

    # Presence of a deinit shouldn't override @explicit_destroy.
    def __deinit__(deinit self):
        pass

    def method(self):
        pass


def testExplicitWithDeinit():
    var a = ExplicitWithDeinit()
    a.method()
    a^.__deinit__()  # ok

    var b = ExplicitWithDeinit()
    b.method()  # expected-error {{'b' abandoned without being explicitly destroyed: must use __deinit__() explicitly}}


# This comes from stubs library.
# CHECK-LABEL: lit.struct.decl @Coroutine
# CHECK-NOT: destructor :!lit.generator


# ===----------------------------------------------------------------------=== #
# Trait with custom @explicit_destroy error message
# ===----------------------------------------------------------------------=== #


# Trait without @explicit_destroy and without Deinitable
trait PlainTrait:
    def do_something(self):
        ...


# A plain trait (no @explicit_destroy, no Deinitable) is linear and
# uses the synthesized default message.
trait LinearNoMessage:
    def destroy_no_msg(deinit self):
        ...


@explicit_destroy("Use `destroy()` method.")
trait ExplicitDestroyWithMessage:
    def destroy(deinit self):
        ...


# Test: Plain trait without @explicit_destroy
# expected-error @below {{unhandled explicitly destroyed type 'PlainTrait'}}
# expected-note @below {{consider adding trait conformance to Deinitable}}
def take_plain_trait[T: PlainTrait](var value: T):
    pass


# Test: Plain linear trait with no custom message uses the default message
# expected-error @below {{unhandled explicitly destroyed type 'LinearNoMessage'}}
# expected-note @below {{consider adding trait conformance to Deinitable}}
def take_generic_linear_no_message[T: LinearNoMessage](var value: T):
    pass


# Test: Trait with @explicit_destroy("...") custom message
def take_generic_linear_with_message[
    T: ExplicitDestroyWithMessage
    # expected-error @below {{Use `destroy()` method.}}
    # expected-note @below {{consider adding trait conformance to Deinitable}}
](var value: T):
    pass


# ===----------------------------------------------------------------------=== #
# Trait composition with @explicit_destroy
# ===----------------------------------------------------------------------=== #


@explicit_destroy("Use custom_foo_destroy().")
trait LinearFoo:
    def custom_foo_destroy(deinit self):
        ...


@explicit_destroy("Use custom_bar_destroy().")
trait LinearBar:
    def custom_bar_destroy(deinit self):
        ...


# Test: First trait has custom message - uses that message
# expected-error @below {{Use custom_bar_destroy().}}
# expected-note @below {{consider adding trait conformance to Deinitable}}
def take_foo_and_bar[T: LinearFoo & LinearBar](var value: T):
    pass


# Test: Composing a default-message linear trait with a custom-message one -
# the trait carrying an explicit @explicit_destroy message provides the
# diagnostic.
# expected-error @below {{Use custom_bar_destroy().}}
# expected-note @below {{consider adding trait conformance to Deinitable}}
def take_no_msg_first[T: LinearNoMessage & LinearBar](var value: T):
    pass


# ===----------------------------------------------------------------------=== #
# @explicit_destroy("") on linear traits (valid but poor form)
# ===----------------------------------------------------------------------=== #


# Test: Empty string message is valid on a linear trait (trait without
# Deinitable). The empty message will be used as the error.
@explicit_destroy("")
trait LinearWithEmptyMessage:
    def consume(deinit self):
        ...


# expected-error @below {{abandoned without being explicitly destroyed: }}
# expected-note @below {{consider adding trait conformance to Deinitable}}
def take_linear_empty_message[T: LinearWithEmptyMessage](var value: T):
    pass


# ===----------------------------------------------------------------------=== #
# Conditional Deinitable without @explicit_destroy
# ===----------------------------------------------------------------------=== #


# A struct may conditionally conform to Deinitable via a where-clause
# without using @explicit_destroy. When the where-clause is not satisfied for a
# given instantiation the type is linear, and the compiler emits a synthesized
# default linear-type error message.
struct CondDeinitableDefault[cond: Bool](
    Deinitable where cond, Movable where False
):
    def __init__(out self):
        pass


def testConditionalDeinitableDefaultMessage():
    # cond=True satisfies the where-clause, no error.
    _ = CondDeinitableDefault[True]()

    # expected-error @below {{abandoned without being explicitly destroyed: type 'CondDeinitableDefault' does not conditionally conform to 'Deinitable' for these parameters}}
    _ = CondDeinitableDefault[False]()


# ===----------------------------------------------------------------------=== #
# `Deinitable where False` opts out of implicit deletability (MOCO-4235)
# ===----------------------------------------------------------------------=== #


# An unsatisfiable where-clause opts the struct out of implicit deletability
# entirely. A default linear-type error message is used when no custom one is
# provided by `@explicit_destroy`.
struct WhereFalseLinear(Deinitable where False, Movable where False):
    def __init__(out self):
        pass


def testWhereFalseLinear():
    # expected-error @below {{abandoned without being explicitly destroyed: type 'WhereFalseLinear' does not conform to 'Deinitable' and must be explicitly destroyed}}
    _ = WhereFalseLinear()


# A custom message via @explicit_destroy is used in preference to the default.
@explicit_destroy("use consume()")
struct WhereFalseCustom(Deinitable where False, Movable where False):
    def __init__(out self):
        pass


def testWhereFalseCustomMessage():
    # expected-error @below {{abandoned without being explicitly destroyed: use consume()}}
    _ = WhereFalseCustom()


# A conditional Deinitable slot that the struct's own where-clause
# renders unsatisfiable (`where not (n > 0)` under `where n > 0`) is an opt-out
# indistinguishable from the literal `where False` above: for any instantiable
# `n`, the struct is linear and reports the *same* default message. This is
# invariant 2 (semantic == literal unsatisfiability) applied to the message
# text, not just to synthesis.
struct ContradictedLinear[n: Int](
    Deinitable where not (n > 0), Movable where False
) where n > 0:
    def __init__(out self):
        pass


def testContradictedLinear():
    # expected-error @below {{abandoned without being explicitly destroyed: type 'ContradictedLinear' does not conform to 'Deinitable' and must be explicitly destroyed}}
    _ = ContradictedLinear[1]()


# ===----------------------------------------------------------------------=== #
# Indirect mutable assignment of a linear type
# ===----------------------------------------------------------------------=== #


# An indirect mutable assignment `ptr[] = linear^` overwrites the value at
# the reference target, which must first be implicitly destroyed. When the
# target's type is linear the reference has no user-visible name, so the
# diagnostic describes the abandoned value generically and relies on the
# source location to point at the offending expression.
@explicit_destroy("use consume()")
struct IndirectAssignLinear(Movable, Deinitable where False):
    def __init__(out self):
        pass


def testIndirectAssignLinear(p: UnsafePointer[IndirectAssignLinear, MutAnyOrigin]):
    # expected-error @+1 {{value abandoned without being explicitly destroyed: use consume()}}
    p[] = IndirectAssignLinear()^
