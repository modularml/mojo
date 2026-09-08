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

# RUN: %parse-mojo-isolated %s -verify-diagnostics

##===----------------------------------------------------------------------===##
# Lexical Issues
##===----------------------------------------------------------------------===##

# https://github.com/modularml/modular/issues/4181
struct Issue4181IndentWeirdness[dt: DType](Movable where False):
  var b : Int
    # expected-error @+1 {{definition isn't on its own line at the correct indentation}}
    def f() raises:
      pass

# Failed to parse due to indentation.
def issue_6291(
    val: __mlir_type.index
) -> __mlir_type.index:
    return val

# This file contains parsing related bugs.

def bracketError1():
  _ = ] # expected-error {{unexpected token in expression}}

def bracketError2():
  _ = [[1, 2], }# expected-error {{unexpected token in expression}}


# Indentation errors
def nothing(): pass

def test_indentation1():
  nothing()
    nothing() # expected-error {{statement indentation must match the rest of the block; adjust to align}}

def test_indentation2(p: Bool):
  nothing()
  if p:
      nothing()
   nothing() # expected-error {{statement indentation must match the rest of the block; adjust to align}}

# Decorator processing.
# https://github.com/modular/mojo/issues/1655
@ : # expected-error {{unexpected token in expression}}
    def a  # expected-error {{expected '(' for argument list}}

# https://github.com/modular/mojo/issues/1230
# Parser crashes on incomplete decorator
@ # expected-error {{found stray '@'; '@' must be followed by a decorator name}}
def m # expected-error {{expected '(' for argument list}}

# Issue #6909
# expected-error @below {{invalid comptime declaration: expected an identifier or '_'}}
comptime True = 42
