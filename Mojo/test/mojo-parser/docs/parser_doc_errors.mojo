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

"""Tests for doc string validation diagnostics."""

# RUN: %parse-mojo-isolated -o /dev/null -mojo-diagnose-missing-doc-strings -verify-diagnostics %s


# expected-warning @below {{public symbol 'ArgStruct' is missing a doc string}}
struct ArgStruct(Movable where False):
    pass


struct _ParamStruct_Private_Missing[_type: __mlir_type.`!kgen.dtype`](Movable where False):
    """This is a private struct doc string.

    It doesn't need to include a `Parameters:` section.
    """

    pass


# expected-warning @below {{struct takes parameters, but has no 'Parameters' in doc string}}
struct ParamStruct_Missing[_type: __mlir_type.`!kgen.dtype`](Movable where False):
    """This doc string is missing a `Parameters:` section."""

    pass


struct ParamStruct_Invalid[_type: __mlir_type.`!kgen.dtype`](Movable where False):
    """This is a class summary.

    # expected-warning-re @below {{parameter '{{.*}}_type' is not documented}}
    Parameters:
        invalid_param: This is an invalid parameter.
          # expected-warning @above {{unknown parameter 'invalid_param' in doc string}}
    """

    pass


struct ParamStruct_Duplicates[_type: __mlir_type.`!kgen.dtype`](Movable where False):
    """This is a class summary.

    # expected-note @below {{see previous definition here}}
    Parameters:
        _type: Summary.
        _type: Summary.
          # expected-warning @above {{duplicate parameter '_type' in doc string}}


    # expected-warning @below {{duplicate 'Parameters' section found in doc string}}
    Parameters:
        _type: Summary.
    """

    pass


struct ParamStruct_Order[
    param1: __mlir_type.`!kgen.dtype`, param2: __mlir_type.`!kgen.dtype`
](Movable where False):
    """This is a class summary.

    Parameters:
        param2: Summary.
        param1: Summary.
    """

    # expected-warning @-4 {{'param2' is defined at index 1, but specified in doc string at index 0}}
    pass


struct StructWithMissingMethod(Movable where False):
    """This defines methods with missing doc strings."""

    # expected-warning @below {{public symbol 'method_with_missing_doc_string' is missing a doc string}}
    def method_with_missing_doc_string(self):
        pass

    def _private_method_with_no_doc_string(self):
        pass

    # expected-warning @below {{public symbol '__init__' is missing a doc string}}
    def __init__(out self):
        pass


# expected-warning @below {{public symbol 'fn_missing_doc_string' is missing a doc string}}
def fn_missing_doc_string():
    pass


def _fn_private_no_doc_string():
    pass


def fn_poor_style():
    """this summary should be capitalized and end with a period"""
    # expected-warning @above {{doc string summary should begin with a capital letter or non-alpha character, but this begins with 't'}}
    # expected-warning @above {{doc string summary should end with a period '.', exclamation mark '!', question mark '?', or backtick '`', but this ends with 'd'}}
    pass


def _fn_private_args_missing(arg: ArgStruct):
    """This is a private function doc string.

    It doesn't need to include an `Args:` section.
    """
    pass


# expected-warning @below {{function takes arguments, but has no 'Args' in doc string}}
def fn_args_missing(arg: ArgStruct):
    """This doc string is missing an `Args:` section."""
    return


def fn_args_invalid(arg: ArgStruct):
    """This is a function summary.

    # expected-warning @below {{argument 'arg' is not documented}}
    Args:
        unknown_arg: This is an argument.
          # expected-warning @above {{unknown argument 'unknown_arg' in doc string}}
    """
    return


def fn_args_overindent(arg: ArgStruct):
    """This is a function summary.

    Description.

        # expected-warning @below {{section tag 'Args' is overindented}}
        Args:
            arg: This is an argument.
    """
    return


def fn_args_duplicates(arg: ArgStruct):
    """This is a function summary.

    # expected-note @below {{see previous definition here}}
    Args:
        arg: This is an argument.
        arg: This is an argument.
          # expected-warning @above {{duplicate argument 'arg' in doc string}}

    # expected-warning @below {{duplicate 'Args' section found in doc string}}
    Args:
        arg: This is an argument.
    """
    return


def fn_args_order(arg: ArgStruct, arg2: ArgStruct):
    """This is a function summary.

    Args:
        arg2: This is an argument.
        arg: This is an argument.
    """
    # expected-warning @-3 {{'arg2' is defined at index 1, but specified in doc string at index 0}}
    return


def fn_args_empty(arg: ArgStruct, arg2: ArgStruct):
    """This function contains empty argument descriptions.

    Args:
        arg:
        arg2:
    """
    # expected-warning @-3 {{'arg' does not have a description}}
    # expected-warning @-3 {{'arg2' does not have a description}}
    pass


def fn_args_poor_style(arg: ArgStruct, arg2: ArgStruct):
    """This function contains arguments with poor style.

    Args:
        arg: `arg` starts with a valid character but doesn't end with a period
        arg2: this should start with a capital letter.
    """
    # expected-warning @-3 {{'arg' description should end with a period '.', exclamation mark '!', question mark '?', or backtick '`', but this ends with 'd'}}
    # expected-warning @-3 {{'arg2' description should begin with a capital letter or non-alpha character, but this begins with 't'}}
    pass


def fn_args_return():
    """This is a function summary.

    # expected-warning @below {{unexpected 'Returns' in doc string for function with no results}}
    # expected-note @below {{see previous definition here}}
    Returns:
      This returns nothing.

    # expected-warning @below {{duplicate 'Returns' section found in doc string}}
    Returns:
      This returns nothing.
    """
    return


def fn_raises():
    """This is a function summary.

    # expected-warning @below {{unexpected 'Raises' in doc string for function that does not throw}}
    # expected-note @below {{see previous definition here}}
    Raises:
      This raises nothing.

    # expected-warning @below {{duplicate 'Raises' section found in doc string}}
    Raises:
      This raises nothing.
    """
    return


# expected-warning @below {{function can throw errors, but has no 'Raises' in doc string}}
def fn_missing_raises_section() raises:
    """This doc string is missing a `Raises:` section."""
    pass


# expected-warning @below {{function has results, but has no 'Returns' in doc string}}
def fn_args_missing_return() -> Int:
    """This doc string is missing a `Returns:` section."""
    return 0


def fn_returns_section_empty() -> Int:
    """This doc string includes a `Returns:` section, but it's empty.

    # expected-warning @below {{'Returns' section is empty}}
    Returns:
    """
    return 0


def fn_returns_section_poor_style() -> Int:
    """This doc string has a `Returns:` section with poor style.

    Returns:
        doesn't start with a capital letter and doesn't end with punctuation
    """
    # expected-warning @-2 {{section body should begin with a capital letter or non-alpha character, but this begins with 'd'}}
    # expected-warning @-3 {{section body should end with a period '.', exclamation mark '!', question mark '?', or backtick '`', but this ends with 'n'}}
    return 0


def fn_nested_fn():
    """This is a function that defines a nested function.

    The nested function does not include a doc string, but it should not be
    reported as invalid.
    """

    def nested_fn():
        pass

    return


def fn_unified_thin_closure():
    """This function defines a thin unified closure (no captures).

    The closure is lifted to file scope during compilation, but docstring
    rules must not be enforced on it.
    """

    def thin_closure(x: Int) -> Int:
        return x


struct Error(Movable where False):
    """Error type stub to allow decoupling from the builtins."""

    pass


def fn_raises_with_return_type(x: Int) raises -> Int:
    """This is a function that raises, with an explicit return type.

    Because it raises, it implicitly has a memory-only `__result__` argument.
    However, this doc string should not document this hidden argument.

    Args:
        x: An explicit argument.

    Returns:
        `0`.

    Raises:
        Exception description goes here.
    """
    return 0


def def_implicit_return_type(x: Int) raises:
    """This is a `def` function with no explicit return type.

    Args:
        x: An explicit argument.

    Raises:
        Exception: if an error occurs.
    """
    pass


# ===----------------------------------------------------------------------=== #
# Alias doc string validation
# ===----------------------------------------------------------------------=== #


# expected-warning-re @below {{public symbol 'AliasWithParams_MissingDocString{{.*}}' is missing a doc string}}
comptime AliasWithParams_MissingDocString[T: AnyType] = T


comptime _AliasWithParams_Private[T: AnyType] = T


# expected-warning @below {{comptime value has parameters, but has no 'Parameters' in doc string}}
comptime AliasWithParams_MissingParamsSection[T: AnyType] = T
"""This alias doc string is missing a Parameters section."""


comptime AliasWithParams_InvalidParam[T: AnyType] = T
"""This alias has an invalid parameter documented.

# expected-warning-re @below {{parameter '{{.*}}T' is not documented}}
Parameters:
    invalid_param: This is an invalid parameter.
      # expected-warning @above {{unknown parameter 'invalid_param' in doc string}}
"""


comptime AliasWithParams_Valid[T: AnyType] = T
"""This alias has valid parameter documentation.

Parameters:
    T: The type parameter.
"""


# expected-warning @+5 {{'U' is defined at index 1, but specified in doc string at index 0}}
comptime AliasWithMultipleParams_Order[T: AnyType, U: AnyType] = T
"""This alias has parameters in wrong order.

Parameters:
    U: Second parameter documented first.
    T: First parameter documented second.
"""


# Test that autoparams (compiler-generated parameters with mangled names like
# `a`1`) are correctly ignored in docstring validation. When doing partial
# binding, the compiler generates autoparams for the unbound parameters.
struct _AutoParamTest[
    a: __mlir_type.`!kgen.dtype`, b: __mlir_type.`!kgen.dtype`
](Movable where False):
    pass


def _fn_with_params[x: Int](s: _AutoParamTest):
    pass


# Partial binding generates autoparams like `a`1`, `b`2` for the struct's params.
# These mangled names should NOT require documentation.
comptime AliasWithAutoParams = _fn_with_params[1]
"""This alias has compiler-generated autoparams that should be ignored."""


# ===----------------------------------------------------------------------=== #
# @doc_hidden suppresses missing doc string warnings (MOTO-957)
# ===----------------------------------------------------------------------=== #


# @doc_hidden on a comptime alias suppresses the missing doc string warning.
# (MOTO-957: previously @doc_hidden caused a compiler error on aliases)
@doc_hidden
comptime doc_hidden_alias = 42


# @doc_hidden on a struct field suppresses the missing doc string warning.
# (MOTO-957: previously @doc_hidden caused a compiler error on struct fields)
struct StructWithDocHiddenField(Movable where False):
    """A struct testing @doc_hidden on fields."""

    @doc_hidden
    var hidden_field: Int


# ===----------------------------------------------------------------------=== #
# Inferred parameter doc string validation (MOTO-485)
# ===----------------------------------------------------------------------=== #

# Inferred parameters (before //) do not require documentation.
# No warning should be emitted for undocumented inferred param 'T'.
def fn_with_inferred_param[T: AnyType, //, U: AnyType]():
    """This function has both inferred and regular parameters.

    Parameters:
        U: The user-visible parameter.
    """
    pass


# Inferred parameters may also be documented without triggering "unknown
# parameter" warnings, since they appear in the generated API docs.
def fn_with_documented_inferred_param[T: AnyType, //, U: AnyType]():
    """This function documents both its inferred and regular parameters.

    Parameters:
        T: The inferred parameter — documenting it is valid but not required.
        U: The user-visible parameter.
    """
    pass


# When all compile-time parameters are inferred, no 'Parameters' section is needed.
def fn_with_only_inferred_params[T: AnyType, //](arg: ArgStruct):
    """This function's only compile-time parameters are inferred.

    Args:
        arg: An argument.
    """
    pass


# Structs: inferred parameters do not require documentation.
struct StructWithInferredParam[T: AnyType, //, U: AnyType](Movable where False):
    """A struct with an inferred and a regular parameter.

    Parameters:
        U: The user-visible parameter.
    """

    pass


# Aliases: inferred parameters do not require documentation.
comptime AliasWithInferredParam[T: AnyType, //, U: AnyType] = U
"""An alias with an inferred and a regular parameter.

Parameters:
    U: The user-visible parameter.
"""


# When all compile-time parameters are inferred, no 'Parameters' section is needed.
comptime AliasWithOnlyInferredParams[T: AnyType, //] = T
"""This alias's only compile-time parameter is inferred."""
