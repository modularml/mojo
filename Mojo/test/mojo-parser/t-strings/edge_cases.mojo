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

# Test edge cases and corner cases for t-strings


# CHECK-LABEL: lit.fn @"test_edge_cases()"
def test_edge_cases():
    # Test 1: Multiple consecutive escaped braces
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{{[{][{][{][{][}][}][}][}]}}"
    var s1 = t"{{{{}}}}"

    # Test 2: Escaped braces surrounding interpolation
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{{[{][{][{][}][}][}]}}"
    var s2 = t"{{{42}}}"

    # Test 3: Dict literal in interpolation (braces inside expression)
    # This should work - the braces are part of the expression, not t-string syntax
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Dict: {}"
    var dict_expr = String("{key: value}")  # Simplified dict representation
    var s3 = t"Dict: {dict_expr}"

    # Test 4: Multiple interpolations with no space between
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{}{}{}"
    var s4 = t"{1}{2}{3}"

    # Test 5: Interpolation at start
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{} end"
    var s5 = t"{String('start')} end"

    # Test 6: Interpolation at end
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "start {}"
    var s6 = t"start {String('end')}"

    # Test 7: Complex nested expression
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Result: {}"
    var s7 = t"Result: {((1 + 2) * 3) + (4 * 5)}"

    # Test 8: Empty t-string
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string ""
    var s8 = t""

    # Test 9: Only escaped braces (no interpolations)
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{{[{][{][}][}]}}"
    var s9 = t"{{}}"

    # Test 10: Long string with many interpolations
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "a{}b{}c{}d{}e{}f{}g{}h{}i{}j{}"
    var s10 = t"a{1}b{2}c{3}d{4}e{5}f{6}g{7}h{8}i{9}j{10}"

    # Test 11: Triple-quoted t-string with newlines
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "\0ALine 1\0ALine 2\0A"
    var s11 = t"""
Line 1
Line 2
"""

    # Test 12: Triple-quoted with interpolation
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "\0AValue: {}\0A"
    var s12 = t"""
Value: {42}
"""

    # Test 13: Single quote t-string (if supported)
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "single"
    var s13 = t'single'

    # Test 14: Escaped braces before interpolation
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{{[{][{][{][}]}}"
    var s14 = t"{{{1}"

    # Test 15: Escaped braces after interpolation
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{{[{][}][}][}]}}"
    var s15 = t"{1}}}"

    # Test 16: Arithmetic in interpolation
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Sum: {}"
    var s16 = t"Sum: {5 + 10}"

    # Test 17: Comparison expression in interpolation
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{}"
    var s17 = t"{5 > 3}"

    # Test 18: Multiple types in same t-string
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "{} and {}"
    var s18 = t"{42} and {True}"

    # Test 19: Escaped braces representing dict-like syntax (not actual dict literal)
    # The {{ and }} are escaped braces in the t-string, producing literal { } in output
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Dict: {{[{][{]'key': 'value'[}][}]}}"
    var s19 = t"Dict: {{'key': 'value'}}"

    # Test 20: Expression with parentheses that could look like function call
    # Validates expressionBraceDepth tracks {} in expressions correctly
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "Result: {}"
    var s20 = t"Result: {(1 + 2) * 3}"

    # Test 21: String literal with same quote as t-string (double quotes)
    # This should work - the inner string is a separate expression
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "hello {}"
    var s21 = t"hello {"world"}"

    # Test 22: String literal with same quote as t-string (single quotes)
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "hello {}"
    var s22 = t'hello {'world'}'

    # Test 23: Multiple nested strings with same quotes
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "a {} c {}"
    var s23 = t"a {"b"} c {"d"}"

    # Test 24: Triple-quoted t-string with double-quoted string inside
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "hello {}"
    var s24 = t"""hello {"world"}"""

    # Test 25: Mixed quotes - t-string with double, inner with single
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "outer {}"
    var s25 = t"outer {'inner'}"

    # Test 26: Mixed quotes - t-string with single, inner with double
    # CHECK: lit.call @{{.*}}__make_tstring{{.*}}:string "outer {}"
    var s26 = t'outer {"inner"}'
