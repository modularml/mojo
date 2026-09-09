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

# Tests for @deprecated decorator warning emission.

# RUN: %parse-mojo-isolated -verify-diagnostics %s -I=%S/inputs


# ===----------------------------------------------------------------------=== #
# Test declarations
# ===----------------------------------------------------------------------=== #


@deprecated("deprecated struct")
# expected-note @below {{'DeprecatedStruct' declared here}}
struct DeprecatedStruct(Movable where False):
    pass


@deprecated("deprecated function")
# expected-note @below {{'deprecated_fn' declared here}}
def deprecated_fn():
    pass


def normal_fn():
    pass


@deprecated("deprecated trait")
# expected-note @below {{'DeprecatedTrait' declared here}}
trait DeprecatedTrait:
    pass


trait NormalTrait:
    pass


@deprecated("deprecated alias")
# Note: The note format for aliases includes a suffix like `0x, so we use
# expected-note-re to match it. See MOCO-3108.
# expected-note-re @below {{'deprecated_alias{{.*}}' declared here}}
comptime deprecated_alias = 42


comptime normal_alias = 42


struct StructWithDeprecatedMembers(Movable where False):
    # Note: @deprecated on fields is not supported - decorators cannot be applied
    # to `var` statements. This is a parser-level limitation.

    def __init__(out self):
        pass

    @deprecated("deprecated method")
    # expected-note @below {{'deprecated_method' declared here}}
    def deprecated_method(self):
        pass

    def normal_method(self):
        pass


# ===----------------------------------------------------------------------=== #
# Test: Deprecated struct as type annotation
# ===----------------------------------------------------------------------=== #


# expected-warning @below {{deprecated struct}}
def use_deprecated_struct_in_signature(value: DeprecatedStruct):
    pass


# ===----------------------------------------------------------------------=== #
# Test: Deprecated function call
# ===----------------------------------------------------------------------=== #


def test_deprecated_function_call():
    # expected-warning @below {{deprecated function}}
    deprecated_fn()

    # No warning expected.
    normal_fn()


# ===----------------------------------------------------------------------=== #
# Test: Deprecated function reference (not call)
# ===----------------------------------------------------------------------=== #


def takes_fn_ref(f: def() thin -> None):
    f()


def test_deprecated_function_reference():
    # Taking a reference to a deprecated function also emits the deprecation warning.
    # expected-warning @below {{deprecated function}}
    var f = deprecated_fn
    takes_fn_ref(f)

    # No warning expected.
    var g = normal_fn
    takes_fn_ref(g)


# ===----------------------------------------------------------------------=== #
# Test: Deprecated trait conformance
# ===----------------------------------------------------------------------=== #


# expected-warning @below {{deprecated trait}}
struct StructConformingToDeprecatedTrait(DeprecatedTrait, Movable where False):
    pass


struct StructConformingToNormalTrait(NormalTrait, Movable where False):
    # No warning expected.
    pass


# ===----------------------------------------------------------------------=== #
# Test: Deprecated alias usage
# ===----------------------------------------------------------------------=== #


def test_deprecated_alias():
    # expected-warning @below {{deprecated alias}}
    _ = deprecated_alias

    # No warning expected.
    _ = normal_alias


# ===----------------------------------------------------------------------=== #
# Test: Deprecated method call
# ===----------------------------------------------------------------------=== #


def test_deprecated_method_call():
    var obj = StructWithDeprecatedMembers()

    # expected-warning @below {{deprecated method}}
    obj.deprecated_method()

    # No warning expected.
    obj.normal_method()


# ===----------------------------------------------------------------------=== #
# Test: Deprecated method reference (not call)
# ===----------------------------------------------------------------------=== #


def test_deprecated_method_reference():
    # Taking a reference to a deprecated method emits the deprecation warning.
    # expected-warning @below {{deprecated method}}
    _ = StructWithDeprecatedMembers.deprecated_method

    # No warning expected.
    _ = StructWithDeprecatedMembers.normal_method


# ===----------------------------------------------------------------------=== #
# Test: Deprecated function overload
# ===----------------------------------------------------------------------=== #


@deprecated("deprecated overload")
# expected-note @below {{'overloaded_fn' declared here}}
def overloaded_fn():
    pass


# expected-warning @below {{deprecated struct}}
def overloaded_fn(value: DeprecatedStruct):
    pass


def test_deprecated_function_overload():
    # expected-warning @below {{deprecated overload}}
    overloaded_fn()


# ===----------------------------------------------------------------------=== #
# Test: @deprecated decorator errors
# ===----------------------------------------------------------------------=== #


# expected-error @below {{@deprecated requires a reason message}}
@deprecated
def no_message():
    pass


comptime NOT_A_STRING = 123


# expected-error @below {{'reason' argument must be a string literal}}
@deprecated(NOT_A_STRING)
def deprecated_with_non_string_reason():
    pass


# expected-error @below {{'reason' argument must be a string literal}}
@deprecated(reason=NOT_A_STRING)
def deprecated_with_non_string_keyword_reason():
    pass


# ===----------------------------------------------------------------------=== #
# Test: @deprecated and @stable mutual exclusivity
# ===----------------------------------------------------------------------=== #


# Both decorators cannot be used together.
@deprecated("use something else")
# expected-error @below {{@deprecated and @stable cannot be used together}}
@stable
def deprecated_and_stable():
    pass


# Order does not matter - still an error.
@stable
# expected-error @below {{@deprecated and @stable cannot be used together}}
@deprecated("use something else")
def stable_and_deprecated():
    pass


# Another decorator in between does not matter - still an error.
@deprecated("use something else")
@no_inline
# expected-error @below {{@deprecated and @stable cannot be used together}}
@stable
def deprecated_other_stable():
    pass


# ===----------------------------------------------------------------------=== #
# Test: @deprecated(use=...) for methods
# ===----------------------------------------------------------------------=== #


def some_top_level_func():
    pass


struct InstanceMethodRefTopLevel(Movable where False):
    # Instance methods cannot reference top-level functions as replacements.
    # expected-error @below {{cannot reference unknown value 'some_top_level_func'}}
    @deprecated(use=some_top_level_func)
    def instance_method(self):
        pass


# ===----------------------------------------------------------------------=== #
# Test: Cross-module deprecation warnings
# ===----------------------------------------------------------------------=== #


from imported_module import DeprecatedInAnotherModule


# expected-warning @below {{use of deprecated struct 'DeprecatedInAnotherModule'}}
def use_deprecated_import(value: DeprecatedInAnotherModule):
    pass
