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

# RUN: %parse-mojo-isolated --verify-diagnostics %s

# =============================================================================
# Empty and malformed expressions
# =============================================================================

def test_empty_braces():
    # Empty expression (just braces)
    # expected-error @below {{t-string expression must not be empty; add an expression or remove these braces}}
    _ = t"Hello {}"

# =============================================================================
# Unterminated errors
# =============================================================================

def test_newline_in_single_quote():
    # expected-error @below {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
    _ = t"Hello
    # expected-error @below {{unterminated string}}
    World"

# =============================================================================
# Lone closing braces
# =============================================================================
# Note: With single-token lexing, a lone `}` at brace depth 0 is included as
# literal text (not an error). Use `}}` to write a literal brace character.

def test_lone_brace():
    # Lone closing brace is treated as literal text (no error).
    _ = t"Hello }"


def test_triple_quote_lone_brace():
    # Lone closing brace in triple-quoted t-string is literal text.
    _ = t"""Hello }"""


def test_multiple_lone_braces():
    # Multiple lone closing braces are literal text.
    _ = t"Hello } World }"


def test_escaped_brace_after_expression():
    var value = 42
    # `}}` after expression close is an escaped literal brace (not an error).
    _ = t"Value: {value}}"


def test_lone_brace_with_nested_quotes():
    # Lone `}` with nested quotes is literal text.
    _ = t"Hello } 'nested string' end"


# =============================================================================
# Format specs (not yet supported)
# =============================================================================

def test_format_spec_with_precision():
    var x = 42
    # expected-error @below {{format specifiers are not supported in t-strings; format the value manually before interpolating}}
    _ = t"{x:.2}"


def test_format_spec_empty():
    var x = 42
    # expected-error @below {{format specifiers are not supported in t-strings; format the value manually before interpolating}}
    _ = t"{x:}"


def test_format_spec_in_multiple_interpolations():
    var x = 42
    var y = 3.14159
    # expected-error @below {{format specifiers are not supported in t-strings; format the value manually before interpolating}}
    _ = t"x={x:}, y={y:.3f}"


def test_format_spec_with_complex_expression():
    var x = 10
    # expected-error @below {{format specifiers are not supported in t-strings; format the value manually before interpolating}}
    _ = t"{x + 10:.2}"


# =============================================================================
# expression errors
# =============================================================================

def test_undefined_variable():
    # Test: Undefined variable in interpolation
    # expected-error @below {{use of unknown declaration 'undefined_var'}}
    var s1 = t"Hello, {undefined_var}!"


def test_invalid_expression():
    # Test: Invalid expression in interpolation (bare operator)
    # expected-error @below {{unexpected token in expression}}
    var s2 = t"Result: {+}"


def test_incomplete_expression():
    # Test: Incomplete expression in interpolation
    # expected-error @below {{unexpected token in expression}}
    var s3 = t"Value: {1 +}"


def test_multiple_undefined_vars():
    # Test: Multiple undefined variables in one t-string (only first error reported)
    # expected-error @below {{use of unknown declaration 'x'}}
    var s4 = t"Values: {x} and {y}"

# =============================================================================
# comptime errors
# =============================================================================

def test_unmaterializable_value_in_tstring():
    comptime l: List[Int] = [1, 2, 3]
    # expected-error @below {{cannot materialize comptime value of type 'List[Int]' to runtime because it is not 'ImplicitlyCopyable'}}
    # expected-note @below {{use 'materialize' to explicitly materialize the value}}
    _ = t"List: {l}"

def test_runtime_value_in_comptime_tstring():
    var x = 42
    # expected-error @below {{cannot use a dynamic value in comptime initializer}}
    comptime tstring = t"Value: {x}"

# =============================================================================
# Raw t-string errors
# =============================================================================

def test_raw_empty_braces():
    # Empty expression in raw t-string (rt prefix)
    # expected-error @below {{t-string expression must not be empty; add an expression or remove these braces}}
    _ = rt"Hello {}"


def test_raw_empty_braces_tr():
    # Empty expression in raw t-string (tr prefix)
    # expected-error @below {{t-string expression must not be empty; add an expression or remove these braces}}
    _ = tr"Hello {}"


def test_raw_newline_in_single_quote():
    # Raw t-string cannot contain a literal (source-level) newline either.
    # expected-error @below {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
    _ = rt"Hello
    # expected-error @below {{unterminated string}}
    World"


def test_raw_format_spec():
    var x = 42
    # expected-error @below {{format specifiers are not supported in t-strings; format the value manually before interpolating}}
    _ = rt"{x:.2}"


def test_raw_format_spec_empty():
    var x = 42
    # expected-error @below {{format specifiers are not supported in t-strings; format the value manually before interpolating}}
    _ = rt"{x:}"


def test_raw_undefined_variable():
    # expected-error @below {{use of unknown declaration 'undefined_var'}}
    _ = rt"Hello, {undefined_var}!"


def test_raw_invalid_expression():
    # expected-error @below {{unexpected token in expression}}
    _ = rt"Result: {+}"


def test_raw_incomplete_expression():
    # expected-error @below {{unexpected token in expression}}
    _ = rt"Value: {1 +}"


def test_raw_runtime_value_in_comptime_tstring():
    var x = 42
    # expected-error @below {{cannot use a dynamic value in comptime initializer}}
    comptime tstring = rt"Value: {x}"


# =============================================================================
# Unicode escape errors in t-string literal segments
# =============================================================================

def test_tstring_unicode_little_u_too_few_digits():
    var x = 42
    # expected-error @below {{\u requires exactly four hex digits}}
    _ = t"\u006{x}"


def test_tstring_unicode_big_u_too_few_digits():
    var x = 42
    # expected-error @below {{\U requires exactly eight hex digits}}
    _ = t"\U0001F60{x}"


def test_tstring_unicode_out_of_range():
    var x = 42
    # expected-error @below {{value must not exceed U+10FFFF}}
    _ = t"\U00110000{x}"


def test_tstring_unicode_surrogate():
    var x = 42
    # expected-error @below {{unicode escape sequences do not support surrogate code points (U+D800 to U+DFFF); use '\U' with the full code point (not a UTF-16 surrogate pair)}}
    _ = t"\uD800{x}"


# =============================================================================
# Catastrophic errors (these must be last - they break parser recovery)
# =============================================================================

def test_unclosed_expression():
    # Unclosed expression (missing closing })
    # With single-token lexing, the lexer sees the quote inside the expression
    # as starting a nested string, ultimately leading to an unterminated t-string.
    # expected-error @below {{unterminated t-string (missing closing quote)}}
    _ = t"Hello {name"
