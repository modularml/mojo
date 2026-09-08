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

# Tests for @unavailable decorator error emission.

# RUN: %parse-mojo-isolated -verify-diagnostics %s -I=%S/inputs


# ===----------------------------------------------------------------------=== #
# Test declarations
# ===----------------------------------------------------------------------=== #


@unavailable("unavailable function")
# expected-note @below {{'unavailable_fn' declared here}}
def unavailable_fn():
    ...


def normal_fn():
    pass


struct StructWithUnavailableMembers(Movable where False):
    def __init__(out self):
        pass

    @unavailable("unavailable method")
    # expected-note @below {{'unavailable_method' declared here}}
    def unavailable_method(self):
        ...

    def normal_method(self):
        pass


# ===----------------------------------------------------------------------=== #
# Test: Unavailable function call
# ===----------------------------------------------------------------------=== #


def test_unavailable_function_call():
    # expected-error @below {{unavailable function}}
    unavailable_fn()

    # No error expected.
    normal_fn()


# ===----------------------------------------------------------------------=== #
# Test: Unavailable function reference (not call)
# ===----------------------------------------------------------------------=== #


def takes_fn_ref(f: def() thin -> None):
    f()


def test_unavailable_function_reference():
    # Taking a reference to an unavailable function also emits the error.
    # expected-error @below {{unavailable function}}
    var f = unavailable_fn
    takes_fn_ref(f)

    # No error expected.
    var g = normal_fn
    takes_fn_ref(g)


# ===----------------------------------------------------------------------=== #
# Test: Unavailable method call
# ===----------------------------------------------------------------------=== #


def test_unavailable_method_call():
    var obj = StructWithUnavailableMembers()

    # expected-error @below {{unavailable method}}
    obj.unavailable_method()

    # No error expected.
    obj.normal_method()


# ===----------------------------------------------------------------------=== #
# Test: Unavailable method reference (not call)
# ===----------------------------------------------------------------------=== #


def test_unavailable_method_reference():
    # Taking a reference to an unavailable method also emits the error.
    # expected-error @below {{unavailable method}}
    _ = StructWithUnavailableMembers.unavailable_method

    # No error expected.
    _ = StructWithUnavailableMembers.normal_method


# ===----------------------------------------------------------------------=== #
# Test: @unavailable is rejected on non-function/method declarations
# ===----------------------------------------------------------------------=== #


# expected-error @below {{@unavailable can only be applied to functions and methods}}
@unavailable("not allowed on structs")
struct StructNotAllowed(Movable where False):
    pass


# expected-error @below {{@unavailable can only be applied to functions and methods}}
@unavailable("not allowed on traits")
trait TraitNotAllowed:
    pass


# expected-error @below {{@unavailable can only be applied to functions and methods}}
@unavailable("not allowed on aliases")
comptime alias_not_allowed = 42


# `use=` form is also rejected on non-function/method decls.
def replacement_for_struct():
    pass


# expected-error @below {{@unavailable can only be applied to functions and methods}}
@unavailable(use=replacement_for_struct)
struct StructUseNotAllowed(Movable where False):
    pass


# ===----------------------------------------------------------------------=== #
# Test: @unavailable decorator errors
# ===----------------------------------------------------------------------=== #


# expected-error @below {{@unavailable requires a reason message}}
@unavailable
def no_message():
    ...


comptime NOT_A_STRING = 123


# expected-error @below {{'reason' argument must be a string literal}}
@unavailable(NOT_A_STRING)
def unavailable_with_non_string_reason():
    ...


# A `reason=` keyword is accepted (mirrors @deprecated).
@unavailable(reason="reason keyword works")
# expected-note @below {{'unavailable_with_reason_keyword' declared here}}
def unavailable_with_reason_keyword():
    ...


def test_reason_keyword():
    # expected-error @below {{reason keyword works}}
    unavailable_with_reason_keyword()


# `reason=` still requires a string literal.
# expected-error @below {{'reason' argument must be a string literal}}
@unavailable(reason=NOT_A_STRING)
def unavailable_with_non_string_reason_keyword():
    ...


# Multiple positional args are rejected.
# expected-error @below {{@unavailable accepts either a reason message or a replacement symbol (with 'use')}}
@unavailable("msg1", "msg2")
def unavailable_with_too_many_args():
    ...


# Unrecognized keywords are rejected.
# expected-error @below {{unavailable must specify either a message or a symbol (with the 'use' argument)}}
@unavailable(message="x")
def unavailable_with_unknown_keyword():
    ...


# `use=` requires a symbol reference, not a literal.
@unavailable(
    # expected-error @below {{'use' must reference a symbol}}
    use="not_a_symbol"
)
def unavailable_with_string_use():
    ...


# ===----------------------------------------------------------------------=== #
# Test: @unavailable mutual exclusivity with @deprecated and @stable
# ===----------------------------------------------------------------------=== #


@unavailable("don't use this")
# expected-error @below {{@unavailable and @deprecated cannot be used together}}
@deprecated("use something else")
def unavailable_and_deprecated():
    ...


# Order does not matter - still an error.
@deprecated("use something else")
# expected-error @below {{@unavailable and @deprecated cannot be used together}}
@unavailable("don't use this")
def deprecated_and_unavailable():
    ...


@unavailable("don't use this")
# expected-error @below {{@unavailable and @stable cannot be used together}}
@stable
def unavailable_and_stable():
    ...


# Order does not matter - still an error.
@stable
# expected-error @below {{@unavailable and @stable cannot be used together}}
@unavailable("don't use this")
def stable_and_unavailable():
    ...


# ===----------------------------------------------------------------------=== #
# Test: @unavailable(use=...) for methods
# ===----------------------------------------------------------------------=== #


def some_top_level_func():
    pass


struct InstanceMethodRefTopLevel(Movable where False):
    # Instance methods cannot reference top-level functions as replacements.
    # expected-error @below {{cannot reference unknown value 'some_top_level_func'}}
    @unavailable(use=some_top_level_func)
    def instance_method(self):
        ...


# ===----------------------------------------------------------------------=== #
# Test: Cross-module unavailability errors
# ===----------------------------------------------------------------------=== #


from imported_module import unavailable_in_another_module


def use_unavailable_import():
    # expected-error @below {{use of unavailable function in another module}}
    unavailable_in_another_module()


# ===----------------------------------------------------------------------=== #
# Test: @unavailable method satisfying a trait requirement
#
# A struct cannot use an @unavailable method to satisfy a trait requirement.
# The error fires at the conformance declaration itself, which in turn prevents
# the type from being used through a generic constrained on the trait (e.g.
# `calls_trait_method[UnavailableTraitImpl]`): there is no way to reach a use
# site without first declaring the (rejected) conformance.
# ===----------------------------------------------------------------------=== #


trait Fooable:
    def foo(self):
        ...


# expected-error @below {{unavailable trait method 'foo'}}
struct UnavailableTraitImpl(Fooable, Movable where False):
    def __init__(out self):
        pass

    @unavailable("unavailable trait method 'foo'")
    # expected-note @below {{'foo' declared here}}
    def foo(self):
        ...


def calls_trait_method[T: Fooable](t: T):
    # Inside the generic, `t.foo()` binds to the trait requirement `Fooable.foo`
    # (which is available), so there is no error here. It is the conformance
    # check above that rejects `UnavailableTraitImpl` as a `Fooable`.
    t.foo()


def use_unavailable_via_trait():
    var obj = UnavailableTraitImpl()
    calls_trait_method(obj)
