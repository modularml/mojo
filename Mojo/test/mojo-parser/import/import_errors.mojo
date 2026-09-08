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

# Test the various error cases of imports. The run line also checks that we
# properly handle the case of an invalid import directory.

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=unknownincludedir -I=%S/inputs -I=%S/inputs/test_package %s

# expected-error @+1 {{expected module name}}
import --

# // -----

# expected-error @+1 {{expected name to bind import}}
import foo as --

# // -----

# expected-error @+1 {{unable to locate module 'there_cant_be_a_module_named_this'}}
import there_cant_be_a_module_named_this

# // -----

# expected-error @+1 {{expected module name}}
from --

# // -----

# expected-error @+1 {{expected 'import' after module name}}
from foo --

# // -----

# expected-error @+1 {{expected construct name to import}}
from imported_module import ()

# // -----

# expected-error @+1 {{expected name to import 'foo' as}}
from imported_module import (foo as --)

# // -----

# Tuple import allows an optional trailing comma.
from imported_module import (imported_fn,)

# expected-error @+1 {{expected construct name to import}}
from imported_module import imported_fn,

# // -----

# expected-error @+1 {{expected ')' after import list}}
from imported_module import (imported_fn --

# // -----

# expected-error @below {{cannot import relative to a top-level package}}
from .module import foo

# // -----

# expected-error @below {{unable to locate module 'unknown_nested_module'}}
from test_package.unknown_nested_module import bar

# // -----

import test_package

def assignPackageModule():
  # expected-error @+1 {{expression must be mutable in assignment}}
  test_package = 42

# // -----

# Check that we can't directly import `test_package.module_in_package` just
# because `test_package` is in the path.

# expected-error @below {{unable to locate module 'module_in_package'}}
import module_in_package

# Check that we don't crash on an invalid use of the missing imported decl.
def test():
  _ = module_in_package

# // -----

# expected-error @below {{unable to locate module 'does_not_exist'}}
import imported_module.does_not_exist

from imported_module import *

def test_import():
  imported_fn()

  # expected-error @below {{use of unknown declaration '_ignored_wildcard_fn'}}
  _ignored_wildcard_fn()

from imported_module import (
  # expected-error @below {{module 'imported_module' does not contain 'there_cant_be_a_decl_named_this'}}
  there_cant_be_a_decl_named_this
)

# expected-error @below {{unable to locate module 'there_cant_be_a_module_named_this'}}
from there_cant_be_a_module_named_this import there_cant_be_another_decl_named_this

# expected-note @below {{previous definition here}}
def already_defined_fn():
  return

# expected-error @below {{invalid redefinition of 'already_defined_fn'}}
from imported_module import imported_fn as already_defined_fn

# // -----

import imported_module
import test_package

def baz():
    # expected-error @below {{'imported_module' does not implement the '__call__' method}}
    imported_module()
    # expected-error @below {{'imported_module' is not subscriptable, it does not implement the `__getitem__`/`__setitem__` methods}}
    imported_module[`0`]
    # expected-error @below {{'test_package' does not implement the '__call__' method}}
    test_package()
    # expected-error @below {{'test_package' is not subscriptable, it does not implement the `__getitem__`/`__setitem__` methods}}
    test_package[`0`]

# // -----

import test_package

# expected-error @below {{'S' parameter 'a' has 'AnyType' type, but value has type 'test_package'}}
comptime x = S[test_package]()

struct S[a: AnyType]: # expected-note {{'S' declared here}}
    pass

# // -----

# Import statements are only supported at module or function scope; they should
# be rejected with a clear diagnostic in struct, trait, and extension bodies.

struct StructWithFromImport:
    # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
    from std.collections import Dict

    def method(self):
        pass

# // -----

struct StructWithImport:
    # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
    import std.collections

    def method(self):
        pass

# // -----

trait TraitWithFromImport:
    # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
    from std.collections import Dict

    def method(self):
        pass

# // -----

trait TraitWithImport:
    # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
    import std.collections

    def method(self):
        pass

# // -----

struct Foo:
    pass

__extension Foo:
    # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
    from std.collections import Dict

    def method(self):
        pass

# // -----

struct Bar:
    pass

__extension Bar:
    # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
    import std.collections

    def method(self):
        pass

# // -----

# Imports inside runtime control flow (if/while) within a function are rejected.

def importInIf(x: Int):
    if x > 0:
        # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
        from std.collections import Dict

# // -----

def importInWhile():
    while True:
        # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
        from std.collections import Dict

# // -----

# Imports inside try blocks are rejected like other runtime control flow; a
# Mojo import cannot fail at runtime, so Python's try/except-ImportError
# pattern does not apply.

def importInTry() raises:
    try:
        # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
        from std.collections import Dict
    except e:
        pass

# // -----

# Imports inside with blocks are rejected as well.

@fieldwise_init
struct _Ctx(TrivialRegisterPassable):
    def __enter__(self) -> Int:
        return 0

    def __exit__(self):
        pass

def importInWith():
    with _Ctx() as _val:
        # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
        from std.collections import Dict

# // -----

# Imports inside comptime control flow are allowed, since comptime blocks are
# treated as part of the enclosing function scope for import placement. The
# binding is only visible within the comptime block; see
# import_fn_scope_comptime_blocks.mojo.

def importInComptimeIf():
    comptime if True:
        from std.collections import Dict

# // -----

def importInComptimeIfFalse():
    comptime if False:
        from std.collections import Dict

# // -----

# Runtime for loop: rejected.

@fieldwise_init
struct _Range(TrivialRegisterPassable, Iterator):
    comptime Element = Int
    def __iter__(self) -> Self: return self
    def __next__(mut self) raises StopIteration -> Int: raise StopIteration()
    def __len__(self) -> Int: return 0

def importInFor():
    for _ in _Range():
        # expected-error @below {{'import' statements must be at module or function scope; move this to a valid location}}
        from std.collections import Dict

# // -----

# Comptime for loop: allowed.

@fieldwise_init
struct _CRange(TrivialRegisterPassable, Iterator):
    comptime Element = Int
    def __iter__(self) -> Self: return self
    def __next__(mut self) raises StopIteration -> Int: raise StopIteration()
    def __len__(self) -> Int: return 0

def importInComptimeFor():
    comptime for _ in _CRange():
        from std.collections import Dict

# // -----

# Imports inside a nested function (closure) are allowed: the nested def
# body is its own function scope.

def importInNestedFn():
    def inner() capturing:
        from std.collections import Dict
    inner()

# // -----

# Verify that importing from a package does not make the package itself
# implicitly available for unrelated member access.
from test_package.module import function

def test_package_not_leaking():
    function()
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    _ = test_package
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    _ = test_package.module

# // -----

# Verify that importing from a package does not make a sibling package
# implicitly available for unrelated member access.
import test_package.test_nested_package

def test_package_not_leaking():
    _ = test_package  # OK: reference to parent package
    _ = test_package.test_nested_package  # OK: reference to child package
    # expected-error @+1 {{package 'test_package' has no declaration 'module'}}
    test_package.module.function()

# // -----

# expected-error @+1 {{relative imports must use 'from'}}
import ....

# // -----

# expected-error @+1 {{relative imports must use 'from'; did you mean 'from . import test_package'?}}
import .test_package

# // -----

# expected-error @+1 {{relative imports must use 'from'; did you mean 'from .. import test_package.test_nested_package'?}}
import ..test_package.test_nested_package

# // -----

# expected-error @+1 {{relative imports must use 'from'; did you mean 'from ... import test_package'?}}
import ...test_package

# // -----

# expected-error @+1 {{unable to locate module 'package_with'}}
import package_with

# // -----

# expected-error @+1 {{unable to locate module 'nested_package_with'}}
import package_without_dots.nested_package_with
