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

# Test that `import a.b` emits errors the leaf name `b` is used unqualified.
# Also verify that `import a.b as b` and `from a import b` do NOT emit any
# diagnostic.

# Leaf import without 'as': should error

import test_package.module


def test_leaf_import():
    # expected-error @below {{use of unknown declaration 'module'}}
    module.function()
    # should not emit a diagnostic
    test_package.module.function()


# // -----

# Import with 'as': should NOT warn

import test_package.module as mod


def test_as_import():
    mod.function()


# 'from' import: should NOT warn

from test_package import module as mod2


def test_from_import():
    mod2.function()
