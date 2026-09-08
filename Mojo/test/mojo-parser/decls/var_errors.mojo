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

# RUN: %parse-mojo-isolated -verify-diagnostics %s

struct SomeStruct(Movable where False):
    def __init__(out self): pass

struct SomeOtherStruct(Movable where False): pass


def var_decl_without_type():
    # expected-error @+1 {{cannot declare 'x' without a contextual type from its initializer}}
    var x

    # expected-error @+1 {{invalid implicit conversion to 'SomeOtherStruct': no constructors found}}
    var y : SomeOtherStruct = SomeStruct()

    # expected-error @+1 {{invalid implicit conversion to 'SomeOtherStruct': no constructors found}}
    var z: SomeOtherStruct = SomeStruct()

def fudge_int(x: Int): pass

def var_decl():
    var x = "x"  # expected-note {{previous definition here}}
    var x : Int   # expected-error {{invalid redefinition of 'x'}}
    fudge_int(x)  # No follow-on error.

def bad_type_error_message():
    var localVar = 42
    var y : localVar  # expected-error {{cannot use a dynamic value in type specification}}

    var x: Int
    var ptr: fudge_int(x)  # expected-error {{cannot use a dynamic value in type specification}}

def missing_type_on_var_decl():
    var abc :       # This line break is intentional.
    pass            # expected-error {{unexpected token in expression}}
    fudge_int(abc)  # No follow-on error.

def bad_stmt_list(cond: Bool):
    # expected-error @+1 {{'if' statement must be on its own line}}
    var abc = 42; if cond: pass

def cannot_fwd_declare_plus_equal():
    # expected-error @+1 {{use of unknown declaration 'x'}}
    x += 42

def test_var_let_type_literal_value():
    # expected-error @below {{expected a type, not a value}}
    var c: 42

def use_before_def():
    # expected-error @below {{use of unknown declaration 'x'}}
    var y = x
    var x = 42

# Issue #18150: https://github.com/modularml/modular/issues/18150
def self_reference():
    # expected-error @+1 {{cannot implicitly convert 'None' value to 'Int'}}
    var num: Int = fudge_int(1)

# Doesn't reject empty identifier name
# https://github.com/modular/mojo/issues/1232
def empty_name():
  # expected-error @+1 {{backtick identifier must not be empty; add content between the backticks}}
  var `` = 42

# COM: Issue #957 https://github.com/modular/mojo/issues/957
struct MemoryStruct(Movable where False):
    @implicit
    def __init__(out self, s: Int): pass

def take_variadic(*elements: MemoryStruct): pass

def test_var_let_type_variadic_func():
  # expected-error @below {{expected a type, not a value}}
  var a: take_variadic(42)
