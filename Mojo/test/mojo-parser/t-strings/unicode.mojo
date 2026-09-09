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

# Verify that \u and \U unicode escape sequences inside t-string literal
# segments are processed correctly. The decoded code point is UTF-8 encoded
# in the format template string, exactly as in regular string literals.


# CHECK-LABEL: lit.fn @"test_tstring_unicode()"
def test_tstring_unicode():
    var x = 42

    # U+0068 'h' — single-byte UTF-8; literal part before interpolation
    # CHECK: __make_tstring{{.*}}:string "h{}"
    var s1 = t"\u0068{x}"

    # U+00E9 'é' — two-byte UTF-8 (C3 A9); literal part after interpolation
    # CHECK: __make_tstring{{.*}}:string "{}\C3\A9"
    var s2 = t"{x}\u00E9"

    # U+4E2D '中' — three-byte UTF-8 (E4 B8 AD); around an interpolation
    # CHECK: __make_tstring{{.*}}:string "\E4\B8\AD{}\E4\B8\AD"
    var s3 = t"\u4E2D{x}\u4E2D"

    # U+1F600 '😀' — four-byte UTF-8 (F0 9F 98 80)
    # CHECK: __make_tstring{{.*}}:string "\F0\9F\98\80{}"
    var s4 = t"\U0001F600{x}"

    # Multiple unicode escapes in one literal segment, no interpolation
    # CHECK: __make_tstring{{.*}}:string "h\C3\A9"
    var s5 = t"\u0068\u00E9"

    # \U with a small code point (same result as equivalent \u)
    # CHECK: __make_tstring{{.*}}:string "A{}"
    var s6 = t"\U00000041{x}"

    # Unicode embedded between ASCII characters — verifies surrounding bytes
    # are not disturbed by the multi-byte encoding.
    # CHECK: __make_tstring{{.*}}:string "abc\C3\A9def{}"
    var s7 = t"abc\u00E9def{x}"

    # Three-byte encoding between ASCII characters
    # CHECK: __make_tstring{{.*}}:string "abc\E4\B8\ADdef{}"
    var s8 = t"abc\u4E2Ddef{x}"

    # Four-byte encoding between ASCII characters
    # CHECK: __make_tstring{{.*}}:string "abc\F0\9F\98\80def{}"
    var s9 = t"abc\U0001F600def{x}"

    # Multiple unicode escapes interleaved with ASCII characters
    # CHECK: __make_tstring{{.*}}:string "a\C3\A9b\E4\B8\ADc{}"
    var s10 = t"a\u00E9b\u4E2Dc{x}"

    # Double backslash before u/U is a literal backslash, not a unicode escape.
    # \\u0068 produces the 7 literal bytes \u0068, not 'h'.
    # CHECK: __make_tstring{{.*}}:string "\\u0068{}"
    var s11 = t"\\u0068{x}"

    # \\U with an out-of-range value must not trigger a false-positive error.
    # CHECK: __make_tstring{{.*}}:string "\\U00110000{}"
    var s12 = t"\\U00110000{x}"
