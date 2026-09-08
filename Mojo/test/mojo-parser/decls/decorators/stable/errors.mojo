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

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -verify-diagnostics %s

# Test error cases for @stable decorator.
#
# Note: Tests for "@stable member in unstable struct/trait" are located in
# test_std_mock/__init__.mojo since those errors only apply in opted-in packages.

# Error: @stable with a positional (non-keyword) argument is not supported.
# expected-error @+1 {{@stable requires a keyword argument ('since' or 'recursive'), not a positional argument}}
@stable("since 1.0")
struct StableWithArg(Movable where False):
    pass


# Error: @stable on local variable is not supported.
def test_local_var():
    # expected-error @+2 {{'var' statement in function body does not support decorators; remove the decorator}}
    @stable
    var x = 1


# @stable on comptime is now supported (as of escape hatches feature).
# Verify it doesn't error - this alias will be verified in a separate test.
@stable
comptime MY_STABLE_ALIAS = 42


# Verify that @stable members in stable structs are allowed (no error).
@stable
struct LocalStableStruct(Movable where False):
    @stable
    def stable_method_in_stable(self):
        pass


# Verify that @stable members in stable traits are allowed (no error).
@stable
trait StableTrait:
    @stable
    def stable_method_in_stable_trait(self): ...


# Verify that @stable members in non-opted-in package types are allowed.
# This file is not in an opted-in package, so structs/traits here are stable
# by default, and @stable members should be allowed.
struct StructInNonOptedInPackage(Movable where False):
    @stable
    def stable_method(self):
        pass


trait TraitInNonOptedInPackage:
    @stable
    def stable_method(self): ...


# Error: @stable on wildcard import is not supported.
# expected-error @+2 {{@stable(recursive=True) is not supported on wildcard imports}}
@stable(recursive=True)
from test_std_mock import *

# Error: bare @stable on import requires recursive=True argument.
# expected-error @+1 {{@stable on import requires 'recursive=True' argument}}
@stable
from test_std_mock import StableStruct

# Error: @stable(recursive=False) is not supported.
# expected-error @+1 {{'recursive' argument to @stable must be True}}
@stable(recursive=False)
from test_std_mock import StableStruct

# Error: @stable(recursive=<non-bool>) is not supported.
# expected-error @+1 {{'recursive' argument to @stable must be True}}
@stable(recursive="yes")
from test_std_mock import StableStruct
