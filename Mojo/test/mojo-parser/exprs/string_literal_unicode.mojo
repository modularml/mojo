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

# Verify that \u and \U unicode escape sequences produce the correct UTF-8
# bytes in the compiled IR. The IR printer emits non-printable and non-ASCII
# bytes as uppercase \XX hex escapes, so we can check each encoding width.


# CHECK-LABEL: lit.fn @"test_unicode_escapes
def test_unicode_escapes():
    # U+0068 'h' — single-byte UTF-8 (ASCII printable, emitted as-is)
    # CHECK: #StringLiteral <:string "h">
    comptime ascii = "\u0068"

    # U+0000 NUL — single-byte UTF-8 (non-printable, emitted as \XX)
    # CHECK: #StringLiteral <:string "\00">
    comptime nul = "\u0000"

    # U+00E9 'é' — two-byte UTF-8 (C3 A9)
    # CHECK: #StringLiteral <:string "\C3\A9">
    comptime latin = "\u00E9"

    # U+4E2D '中' — three-byte UTF-8 (E4 B8 AD)
    # CHECK: #StringLiteral <:string "\E4\B8\AD">
    comptime cjk = "\u4E2D"

    # U+FFFF — three-byte UTF-8, last BMP code point (EF BF BF)
    # CHECK: #StringLiteral <:string "\EF\BF\BF">
    comptime bmp_max = "\uFFFF"

    # U+1F600 '😀' — four-byte UTF-8 (F0 9F 98 80)
    # CHECK: #StringLiteral <:string "\F0\9F\98\80">
    comptime emoji = "\U0001F600"

    # U+10FFFF — four-byte UTF-8, maximum valid code point (F4 8F BF BF)
    # CHECK: #StringLiteral <:string "\F4\8F\BF\BF">
    comptime max_cp = "\U0010FFFF"

    # \U with a small code point produces the same bytes as \u
    # CHECK: #StringLiteral <:string "A">
    comptime big_u_small = "\U00000041"

    # Unicode embedded between ASCII characters — verifies surrounding bytes
    # are not disturbed by the multi-byte encoding.
    # CHECK: #StringLiteral <:string "abc\C3\A9def">
    comptime prefix_suffix_2byte = "abc\u00E9def"

    # Three-byte encoding between ASCII characters
    # CHECK: #StringLiteral <:string "abc\E4\B8\ADdef">
    comptime prefix_suffix_3byte = "abc\u4E2Ddef"

    # Four-byte encoding between ASCII characters
    # CHECK: #StringLiteral <:string "abc\F0\9F\98\80def">
    comptime prefix_suffix_4byte = "abc\U0001F600def"

    # Multiple unicode escapes interleaved with ASCII characters
    # CHECK: #StringLiteral <:string "a\C3\A9b\E4\B8\ADc">
    comptime interleaved = "a\u00E9b\u4E2Dc"

    # Raw strings must not decode \u/\U — the backslash is literal.
    # CHECK: #StringLiteral <:string "\\u0068">
    comptime raw_u = r"\u0068"
    # CHECK: #StringLiteral <:string "\\U00000068">
    comptime raw_U = r"\U00000068"
