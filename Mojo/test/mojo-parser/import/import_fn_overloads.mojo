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

# Tests importing function overloads through a package. An overload set
# resolves from a single origin: the package __init__.mojo re-exports
# module1's `foo`, and module2's same-named `foo` is reachable only via
# qualified access or its own import - importing both under one name is
# deprecated (see decls/conflicting_imports_errors.mojo).

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

import fn_overload_package

def main():
    _ = fn_overload_package.foo(42)      # ok
    # expected-error @+1 {{invalid call to 'foo': value passed to 'x' cannot be converted from 'StringLiteral["hello"]' to 'Int'}}
    _ = fn_overload_package.foo("hello")

# // -----

import fn_overload_package.module1
import fn_overload_package.module2

def main():
    _ = fn_overload_package.module1.foo(42)      # ok
    _ = fn_overload_package.module2.foo("hello") # ok

# // -----

from fn_overload_package import foo

def main():
    _ = foo(42)      # ok
    # expected-error @+1 {{invalid call to 'foo': value passed to 'x' cannot be converted from 'StringLiteral["hello"]' to 'Int'}}
    _ = foo("hello")

# // -----

from fn_overload_package.module1 import foo

def main():
    _ = foo(42)
    # expected-error @+1 {{invalid call to 'foo': value passed to 'x' cannot be converted from 'StringLiteral["hello"]' to 'Int'}}
    _ = foo("hello")

# // -----

from fn_overload_package.module2 import foo

def main():
    # expected-error @+1 {{invalid call to 'foo': value passed to 'x' cannot be converted from 'IntLiteral[42]' to 'String'}}
    _ = foo(42)
    _ = foo("hello")
