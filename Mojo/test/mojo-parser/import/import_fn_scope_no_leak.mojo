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

# A function-scoped import binds a name only within that function. It must not
# be visible to sibling functions, to the enclosing function of a nested def,
# or through the shared package decl of an unrelated import of the same
# package.

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s

# An import in one function is not visible in a sibling function, even when
# the importer is parsed first.


def importer():
    from wildcard_shadow_a import shadowed_fn

    _ = shadowed_fn()


def user():
    # expected-error @below {{use of unknown declaration 'shadowed_fn'}}
    _ = shadowed_fn()


# // -----

# An import inside a nested def is not visible in the enclosing function.


def outer():
    def inner():
        from wildcard_shadow_a import shadowed_fn

        _ = shadowed_fn()

    inner()
    # expected-error @below {{use of unknown declaration 'shadowed_fn'}}
    _ = shadowed_fn()


# // -----

# A function-scoped WILDCARD import is not visible in a sibling function.


def wildcard_importer():
    from wildcard_shadow_a import *

    _ = shadowed_fn()


def wildcard_user():
    # expected-error @below {{use of unknown declaration 'shadowed_fn'}}
    _ = shadowed_fn()


# // -----

# A deep chain import (`import a.b`) in one function must not make `b`
# reachable through the package decl in a sibling that only imported `a`.


def deep_importer():
    import test_package.test_nested_package

    test_package.test_nested_package.nested_function()


def shallow_importer():
    import test_package

    # expected-error @below {{package 'test_package' has no declaration 'test_nested_package'}}
    test_package.test_nested_package.nested_function()


# // -----

# A function-scoped deep chain import must not pollute a MODULE-level import
# of the same package used by an innocent sibling function.

import test_package


def deep_importer():
    import test_package.test_nested_package

    test_package.test_nested_package.nested_function()


def innocent():
    # expected-error @below {{package 'test_package' has no declaration 'test_nested_package'}}
    test_package.test_nested_package.nested_function()
