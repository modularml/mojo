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

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

# Importing a package gates access to submodules that were not imported: you
# reach exactly the path you imported, not arbitrarily deep sub-packages.

import test_package


def main():
    # A submodule of test_package that was not imported is not an attribute.
    # expected-error @+1 {{package 'test_package' has no declaration 'test_nested_package'}}
    _ = test_package.test_nested_package

    # expected-error @+1 {{package 'test_package' has no declaration 'test_nested_package'}}
    test_package.test_nested_package.module.nested_function()

    # expected-error @+1 {{package 'test_package' has no declaration 'test_nested_package'}}
    test_package.test_nested_package.deep_package.leaf.deep_function()


# // -----

# Importing one level: that submodule is reachable, along with everything it
# re-exports (test_nested_package re-exports module and deep_package).

import test_package.test_nested_package


def main():
    _ = test_package

    # On the imported path — reachable.
    _ = test_package.test_nested_package

    # 'module' and 'deep_package' are re-exported by test_nested_package, so
    # they (and deep_package.leaf) are reachable.
    test_package.test_nested_package.module.nested_function()
    test_package.test_nested_package.deep_package.leaf.deep_function()

    # deep_package re-exports `leaf` but not `hidden`, so the non-re-exported
    # submodule stays gated even when deep_package is reached via re-export.
    # expected-error @+1 {{package 'deep_package' has no declaration 'hidden'}}
    test_package.test_nested_package.deep_package.hidden.hidden_fn()


# // -----

# Same but importing two extra levels; deep_package stays reachable because
# test_nested_package re-exports it.

import test_package.test_nested_package.module


def main():
    test_package.test_nested_package.deep_package.leaf.deep_function()


# // -----

# Same but importing one extra level as an alias

import test_package.test_nested_package as test_nested


def main():
    # 2 levels deep
    _ = test_nested

    # 3 levels deep
    test_nested.module.nested_function()

    # 4 levels deep
    test_nested.deep_package.leaf.deep_function()


# // -----

# Aliased import should NOT leak the parent package.

import test_package.test_nested_package as test_nested


def main():
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    _ = test_package.test_nested_package

    # 3 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    test_package.test_nested_package.module.nested_function()

    # 4 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    test_package.test_nested_package.deep_package.leaf.deep_function()


# // -----

# 'from' import of a subpackage: the imported name works, but the parent
# package is not accessible.

from test_package import test_nested_package


def main():
    # The imported name is in scope and works.
    _ = test_nested_package

    # 3 levels deep through the imported name
    test_nested_package.module.nested_function()

    # 4 levels deep through the imported name
    test_nested_package.deep_package.leaf.deep_function()


# // -----

# 'from' import should NOT leak the parent package.

from test_package import test_nested_package


def main():
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    _ = test_package

    # 2 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    _ = test_package.test_nested_package

    # 3 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    test_package.test_nested_package.module.nested_function()

    # 4 levels deep
    # expected-error @+1 {{use of unknown declaration 'test_package'}}
    test_package.test_nested_package.deep_package.leaf.deep_function()
