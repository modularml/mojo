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

# Test that --warn-on-unstable-apis emits warnings for unstable struct usage
# from opted-in packages.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

from test_std_mock import StableStruct, UnstableStruct, stable_fn, unstable_fn


def test_stable_struct():
    # Using a stable struct should not trigger a warning.
    var x: StableStruct
    pass


def test_unstable_struct():
    # Using an unstable struct should trigger a warning.
    # CHECK: warning: use of unstable API 'UnstableStruct'
    var y: UnstableStruct
    pass


def test_stable_fn():
    # Calling a stable function should not trigger a warning.
    stable_fn()


def test_unstable_fn():
    # Calling an unstable function should trigger a warning.
    # CHECK: warning: use of unstable API 'unstable_fn'
    unstable_fn()
