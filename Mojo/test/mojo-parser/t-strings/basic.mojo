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

# RUN: %parse-mojo-isolated %s | FileCheck %s


# Helper function for testing t-strings as function arguments
# CHECK-LABEL: lit.fn @"dummy_function(::String,::String)"
def dummy_function(arg1: String, arg2: String):
    pass


# CHECK-LABEL: lit.fn @"test_t_strings()"
def test_t_strings():
    # Basic t-string literal with no interpolations - now uses .make_tstring() for consistency
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Hello, World!"
    var s1 = t"Hello, World!"

    # T-string with single interpolation
    var name = "Alice"
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Hello, {}!"
    var s2 = t"Hello, {name}!"

    # T-string with multiple interpolations
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Sum: {} = {}"
    var s3 = t"Sum: {1 + 2} = {3}"

    # T-string with escaped braces - now uses .make_tstring() for consistent escape handling
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Use {{[{][{]}}braces{{[}][}]}} like this"
    var s4 = t"Use {{braces}} like this"

    # Empty t-string (valid) - now uses .make_tstring() for consistency
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string ""
    var s5 = t""

    # T-string with only an expression
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{}"
    var s6 = t"{42}"

    # Triple-quoted t-string - now uses .make_tstring() for consistency
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "\0A    Multi-line\0A    t-string\0A    "
    var s7 = t"""
    Multi-line
    t-string
    """

    # T-string with nested braces in expression (escaped braces) - now uses .make_tstring()
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Dict: {{.*}}1, 2, 3{{.*}}"
    var s8 = t"Dict: {{1, 2, 3}}"

    # T-string with expression using variables
    var x = 10
    var y = 3
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Result: {}"
    var s9 = t"Result: {x + y}"

    # T-string with expression using integer literals (constant-folded to !Int)
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Calculation: {}"
    var s10 = t"Calculation: {5 * 4 + 10}"

    # T-string with mixed literal and variable expression
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Mixed: {}"
    var s11 = t"Mixed: {x * 2 + 5}"

    # Adjacent interpolations (no literal text between)
    var a = "A"
    var b = "B"
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{}{}"
    var s12 = t"{a}{b}"

    # Mix of escaped braces and interpolation
    var val = 123
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Value {{.*}}x{{.*}} = {}"
    var s13 = t"Value {{x}} = {val}"

    # Boolean values
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Bool: {} and {}"
    var s14 = t"Bool: {True} and {False}"

    # Same variable used multiple times
    var num = 5
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{} * {} = {}"
    var s15 = t"{num} * {num} = {num * num}"

    # Floating point literal
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Pi: {}"
    var s16 = t"Pi: {3.14159}"

    # Empty with escaped braces
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{{.*}}{{.*}}"
    var s17 = t"{{}}"

    # T-string as direct function argument (tests keyword arg lookahead - cursor workaround)
    # This specifically tests the bug where t-strings with ! after interpolations
    # fail when used as function arguments (which triggers keyword argument lookahead)
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Arg: {}!"
    # CHECK: lit.call @{{.*}}dummy_function
    dummy_function(t"Arg: {name}!", "test")

    # Nested t-string
    var val1 = 42
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{}"
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "hello, {}"
    var s18 = t"hello, {t'{val1}'}"

    # Triple-nested t-strings with same quote delimiter
    var val2 = 3
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "C {}"
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "B {}"
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "A {}"
    var s20 = t"A {t"B {t"C {val2}"}"}"
