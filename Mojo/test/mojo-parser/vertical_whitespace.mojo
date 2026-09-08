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

# This tests for vertical whitespace tolerance; mostly covering negative cases.

# RUN: %parse-mojo-isolated -split-input-file %s -verify-diagnostics

def
# expected-error @+1 {{identifier may not appear at the start of the line}}
foo():
    pass

# // -----

async
# expected-error @+1 {{'def' keyword may not appear at the start of the line}}
def foo():
    pass

# // -----

# The argument list must begin on the function name's line.
def with_args
# expected-error @+1 {{argument list may not appear at the start of the line}}
():
    pass

# // -----

# The parameter list must begin on the function name's line.
def with_params
# expected-error @+1 {{parameter list may not appear at the start of the line}}
[T: AnyType](x: T):
    pass

# // -----

struct
# expected-error @+1 {{identifier may not appear at the start of the line}}
Foo:
    pass

# // -----

# The parameter list must begin on the struct name's line.
struct WithParams
# expected-error @+1 {{parameter list may not appear at the start of the line}}
[T: AnyType]:
    pass

# // -----

trait
# expected-error @+1 {{identifier may not appear at the start of the line}}
Bar:
    pass

# // -----

comptime
# expected-error @+1 {{identifier may not appear at the start of the line}}
MyInt = Int

# // -----

struct HasField:
    var
    # expected-error @+1 {{identifier may not appear at the start of the line}}
    x: Int

# // -----

struct Spaceship:
    pass

__extension
# expected-error @+1 {{identifier may not appear at the start of the line}}
Spaceship:
    pass

# // -----

import std
# expected-error @+1 {{module path may not appear at the start of the line}}
.builtin._coroutine

# // -----

import std.builtin._coroutine
# expected-error @+1 {{'as' keyword may not appear at the start of the line}}
as Coroutine

# // -----

import std.builtin._coroutine as
# expected-error @+1 {{bound import name may not appear at the start of the line}}
Coroutine

# // -----

import std
# expected-error @+1 {{comma may not appear at the start of the line}}
, std.builtin._coroutine

# // -----

import std as mystd
# expected-error @+1 {{comma may not appear at the start of the line}}
, std.builtin._coroutine

# // -----

from
# expected-error @+1 {{module path may not appear at the start of the line}}
std.builtin._coroutine import Coroutine

# // -----

from std
# expected-error @+1 {{module path may not appear at the start of the line}}
.builtin._coroutine import Coroutine

# // -----

from std.
# expected-error @+1 {{module path may not appear at the start of the line}}
builtin._coroutine import Coroutine

# // -----

from std.builtin._coroutine
# expected-error @+1 {{'import' statement may not appear at the start of the line}}
import Coroutine

# // -----

from std.builtin._coroutine import
# expected-error @+1 {{wildcard import may not appear at the start of the line}}
*

# // -----

from std.builtin._coroutine import
# expected-error @+1 {{construct name to import may not appear at the start of the line}}
Coroutine

# // -----

from std.builtin._coroutine import
# expected-error @+1 {{beginning of tuple import may not appear at the start of the line}}
(Coroutine)

# // -----

from std.builtin._coroutine import Coroutine
# expected-error @+1 {{'as' keyword may not appear at the start of the line}}
as C

# // -----

from std.builtin._coroutine import Coroutine as
# expected-error @+1 {{bound import name may not appear at the start of the line}}
C

# // -----

from std.builtin._coroutine import (
        Coroutine as C
# expected-error @+1 {{comma may not appear at the start of the line}}
        ,
        Coroutine as D)

# // -----

from std.builtin._coroutine import (
        Coroutine as C, # ok
        Coroutine as
# expected-error @+1 {{bound import name may not appear at the start of the line}}
        D)

# // -----

# A dangling comma with no following name reports the specific "expected
# construct name" diagnostic, not the start-of-line one.
# expected-error @+1 {{expected construct name to import}}
from std.builtin._coroutine import Coroutine,

# // -----

# Positive cases: these must parse without a diagnostic.

# A parenthesized import list may span multiple lines.
from std.builtin._coroutine import (
    Coroutine,
    RaisingCoroutine,
)


from std.builtin._coroutine import (
        Coroutine as C,
        Coroutine as D,
        )

# A multi-line argument list is fine; only the name is restricted.
def multiline_args(
    x: Int,
    y: Int,
):
    pass

# // -----

# Backslash line-continuation is an escape hatch: a token on a physically
# continued line is not treated as being at the start of the line, so the
# restriction does not fire. These must parse without a diagnostic.
def \
backslash_decl():
    pass


from std.builtin._coroutine import \
    Coroutine
