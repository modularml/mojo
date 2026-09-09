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

# Test that API author warnings are emitted without --warn-on-unstable-apis.
# These warnings help API authors catch mistakes when authoring stable APIs.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S %s 2>&1 | FileCheck %s

# This test imports from test_std_mock which contains:
# 1. StableStructWithUnstableImpl - stable struct with unstable trait impl
# 2. stable_fn_returning_unstable - stable function returning unstable type
# 3. StableTraitWithUnstableParent - stable trait inheriting from unstable trait
#
# All should emit warnings when the package is parsed.

# CHECK-DAG: warning: stable trait 'StableTraitWithUnstableParent' cannot inherit from unstable trait 'UnstableTrait'
# CHECK-DAG: warning: stable function 'stable_fn_returning_unstable' returns unstable type 'UnstableStruct'
# CHECK-DAG: warning: stable struct 'StableStructWithUnstableImpl' implements stable trait method 'stable_required_method' with unstable implementation

from test_std_mock import (
    StableStructWithStableImpl,
    StableStructWithUnstableImpl,
    StableTraitWithUnstableParent,
    StableTraitWithStableParent,
    stable_fn_returning_stable,
    stable_fn_returning_unstable,
)


def main():
    # Just using these to trigger the imports.
    # The warnings should be emitted during parsing of test_std_mock,
    # not during use here.
    pass
