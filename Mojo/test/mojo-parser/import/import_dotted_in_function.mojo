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

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s

# Test that imports inside a function body work.


def test_package_import_in_function():
    import test_package

    test_package.method_defined_in_init()  # ok


# // -----


def test_dotted_module_import_in_function():
    import test_package.module

    test_package.module.function()  # ok


# // -----


def test_dotted_package_import_in_function():
    import test_package.test_nested_package

    test_package.test_nested_package.nested_function()  # ok


# // -----


def test_from_package_import_module_in_function():
    from test_package import module

    module.function()  # ok


# // -----


def test_from_package_import_package_in_function():
    from test_package import test_nested_package

    test_nested_package.nested_function()  # ok


# // -----


def test_from_nested_package_import_symbol_in_function():
    from test_package.test_nested_package.module import nested_function

    nested_function()  # ok


# // -----


def test_from_nested_package_import_symbol_in_function():
    from test_package.test_nested_package.deep_package.leaf import deep_function

    deep_function()  # ok


# // -----

# Imports inside struct methods work: method bodies are function scopes.


struct MethodImports:
    var x: Int

    def __init__(out self):
        import test_package

        test_package.method_defined_in_init()  # ok
        self.x = 0

    def method(self):
        from test_package.test_nested_package.module import nested_function

        nested_function()  # ok


# // -----

# Parenthesized and renamed import lists work in functions.


def test_list_forms_in_function():
    from test_package import (module, test_nested_package)
    from test_package.module import function as renamed_function

    module.function()  # ok
    test_nested_package.nested_function()  # ok
    renamed_function()  # ok


# // -----

# Comma-separated imports work in functions.


def test_comma_import_in_function():
    import test_package, plain_mod

    test_package.method_defined_in_init()  # ok
    _ = plain_mod.from_plain()  # ok


# // -----

# A relative import (`from .util import util_fn`) inside a function body of a
# packaged module works; rel_fn's body lives in inputs/fn_scope_rel_pkg.


from fn_scope_rel_pkg.user import rel_fn


def test_relative_import_in_function():
    _ = rel_fn()  # ok
