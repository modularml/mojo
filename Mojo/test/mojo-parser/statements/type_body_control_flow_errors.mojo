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

# Test that control flow statements in struct/trait/extension bodies emit
# errors instead of crashing the compiler (MOCO-3870).

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Control flow statements in struct body
##===----------------------------------------------------------------------===##

struct StructWithIf(Movable where False):
    var x: Int
    # expected-error @below {{'if' must be contained in a function}}
    if True:
        pass

struct StructWithFor(Movable where False):
    var x: Int
    # expected-error @below {{'for' must be contained in a function}}
    for i in range(10):
        pass

struct StructWithWhile(Movable where False):
    var x: Int
    # expected-error @below {{'while' must be contained in a function}}
    while True:
        pass

struct StructWithComptimeIf(Movable where False):
    var x: Int
    # expected-error @below {{'comptime if' must be contained in a function}}
    comptime if True:
        pass

struct StructWithComptimeFor(Movable where False):
    var x: Int
    # expected-error @below {{'comptime for' must be contained in a function}}
    comptime for i in range(10):
        pass

struct StructWithWith(Movable where False):
    var x: Int
    # expected-error @below {{'with' must be contained in a function}}
    with foo:
        pass

##===----------------------------------------------------------------------===##
# Control flow statements in trait body
##===----------------------------------------------------------------------===##

trait TraitWithIf:
    # expected-error @below {{'if' must be contained in a function}}
    if True:
        pass

trait TraitWithFor:
    # expected-error @below {{'for' must be contained in a function}}
    for i in range(10):
        pass

trait TraitWithWhile:
    # expected-error @below {{'while' must be contained in a function}}
    while True:
        pass

trait TraitWithWith:
    # expected-error @below {{'with' must be contained in a function}}
    with foo:
        pass

##===----------------------------------------------------------------------===##
# Control flow statements in extension body
##===----------------------------------------------------------------------===##

struct ExtendedStruct(Movable where False):
    var x: Int

__extension ExtendedStruct:
    # expected-error @below {{'if' must be contained in a function}}
    if True:
        pass

struct ExtendedStruct2(Movable where False):
    var x: Int

__extension ExtendedStruct2:
    # expected-error @below {{'for' must be contained in a function}}
    for i in range(10):
        pass

struct ExtendedStruct3(Movable where False):
    var x: Int

__extension ExtendedStruct3:
    # expected-error @below {{'while' must be contained in a function}}
    while True:
        pass

struct ExtendedStruct4(Movable where False):
    var x: Int

__extension ExtendedStruct4:
    # expected-error @below {{'comptime if' must be contained in a function}}
    comptime if True:
        pass

struct ExtendedStruct5(Movable where False):
    var x: Int

__extension ExtendedStruct5:
    # expected-error @below {{'comptime for' must be contained in a function}}
    comptime for i in range(10):
        pass

struct ExtendedStruct6(Movable where False):
    var x: Int

__extension ExtendedStruct6:
    # expected-error @below {{'with' must be contained in a function}}
    with foo:
        pass
