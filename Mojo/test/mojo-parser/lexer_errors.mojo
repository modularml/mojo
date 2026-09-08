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

# expected-error @+1 {{unterminated backtick identifier}}
`

# // -----

# expected-error @+1 {{unexpected character}}
!

# // -----

# expected-error @+1 {{decimal integer literals must not use leading zeros; add '0o' for octal literals}}
0123

# // -----

# expected-error @+1 {{no digits specified for octal literal}}
0o_

# // -----

# expected-error @+1 {{expected a digit after the exponent}}
1e+*

# // -----

# expected-error @+1 {{expected a digit after the exponent}}
1e*

# // -----

# expected-error @+1 {{unterminated string}}
"Hello'

# // -----

# expected-error @+1 {{unterminated string}}
'Hello"

# // -----

# expected-error @+1 {{unterminated string}}
"Hello

# // -----

# expected-error @+1 {{unterminated string}}
'Hello

# // -----

# expected-error @+1 {{invalid hex escape sequence: exactly two hex digits needed}}
"A\x4"

# // -----

# expected-error @+1 {{invalid hex escape sequence: exactly two hex digits needed}}
"A\x"

# // -----

# expected-error @+1 {{invalid escape sequence}}
"A\zB"

# // -----

# expected-error @+1 {{unterminated string}}
"AB\"

# // -----

# expected-error @+1 {{unterminated string}}
r"AB\"

# // -----

# Issue #12818
def inconsistent_indent():
    var x = __mlir_attr.`1 : index`
   	var y = __mlir_attr.`2 : index`  # expected-error {{indentation must not mix tabs and spaces; select one style}}

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
t"Hello
t"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
t'Hello
t'Hello'

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"""Hello

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t'''Hello

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"Hello{name

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
t"Hello
World"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
t'Hello
World'

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
T"Hello
T"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
T'Hello
T'Hello'

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"Unclosed {expr
t"Unclosed {expr}"

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"Missing closing brace {expr"
t"Missing closing brace {expr}"

# // -----

# expected-error @+1 {{unterminated t-string (missing closing quote)}}
t"Nested {t"inner} incomplete"
t"Nested {t"inner"} incomplete"

# // -----

# Raw t-string: unescaped newline with double quotes (rt prefix)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rt"Hello
rt"Hello"

# // -----

# Raw t-string: unescaped newline with single quotes (rt prefix)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rt'Hello
rt'Hello'

# // -----

# Raw t-string: unescaped newline with double quotes (tr prefix)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
tr"Hello
tr"Hello"

# // -----

# Raw t-string: unescaped newline with single quotes (tr prefix)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
tr'Hello
tr'Hello'

# // -----

# Raw t-string: unterminated triple-quoted (rt prefix)
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"""Hello

# // -----

# Raw t-string: unterminated triple-quoted single quotes (rt prefix)
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt'''Hello

# // -----

# Raw t-string: unclosed expression
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"Hello{name

# // -----

# Raw t-string: newline in middle of string (double quotes)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rt"Hello
World"

# // -----

# Raw t-string: newline in middle of string (single quotes)
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rt'Hello
World'

# // -----

# Raw t-string: uppercase prefix variants
# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
Rt"Hello
Rt"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
rT"Hello
rT"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
RT"Hello
RT"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
tR"Hello
tR"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
Tr"Hello
Tr"Hello"

# // -----

# expected-error @+1 {{t-string cannot contain unescaped newline (use triple quotes or escape as \n)}}
TR"Hello
TR"Hello"

# // -----

# Raw t-string: unclosed expression with recovery line
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"Unclosed {expr
rt"Unclosed {expr}"

# // -----

# Raw t-string: missing closing brace
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"Missing closing brace {expr"
rt"Missing closing brace {expr}"

# // -----

# Raw t-string: nested raw t-string incomplete
# expected-error @+1 {{unterminated t-string (missing closing quote)}}
rt"Nested {rt"inner} incomplete"

# // -----

# Docstring (triple-quoted): incomplete \x escape — only one hex digit before
# the closing quotes. The lexer must reject this just as it does in single-
# quoted strings.
# expected-error @+1 {{invalid hex escape sequence: exactly two hex digits needed}}
"""\xA"""

# // -----

# Docstring: no hex digits after \x.
# expected-error @+1 {{invalid hex escape sequence: exactly two hex digits needed}}
"""\x"""

# // -----

# Docstring: non-hex character after \x (G is not a hex digit).
# expected-error @+1 {{invalid hex escape sequence: exactly two hex digits needed}}
"""\xGG"""

# // -----

# Docstring: invalid escape sequence in triple-quoted string.
# expected-error @+1 {{invalid escape sequence}}
"""\8"""
rt"Nested {rt"inner"} incomplete"
