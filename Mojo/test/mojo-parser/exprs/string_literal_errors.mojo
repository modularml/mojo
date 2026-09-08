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

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s


# The octal escape sequence in string literals \ooo can have variable length.
def testOctal():
    comptime x = "A\0"
    comptime y = "A\01"
    comptime z = "A\012"


# // -----


def testTripleQuote():
    # expected-error @below {{invalid escape sequence}}
    var x = """$\s$"""


# // -----


# Unicode \u escape requires exactly four hex digits.
def testUnicodeLittleUTooFewDigits():
    # expected-error @below {{\u requires exactly four hex digits}}
    var x = "\u006"


# // -----


# Unicode \U escape requires exactly eight hex digits.
def testUnicodeBigUTooFewDigits():
    # expected-error @below {{\U requires exactly eight hex digits}}
    var x = "\U0001F60"


# // -----


# \U value exceeding U+10FFFF is rejected.
def testUnicodeBigUOutOfRange():
    # expected-error @below {{value must not exceed U+10FFFF}}
    var x = "\U00110000"


# // -----


# Surrogates are not valid Unicode scalar values.
def testUnicodeSurrogate():
    # expected-error @below {{unicode escape sequences do not support surrogate code points (U+D800 to U+DFFF); use '\U' with the full code point (not a UTF-16 surrogate pair)}}
    var x = "\uD800"


# // -----


# Named unicode escapes \N{name} are not supported.
def testUnicodeNamedEscape():
    # expected-error @below {{invalid escape sequence}}
    var x = "\N{SNOWMAN}"
