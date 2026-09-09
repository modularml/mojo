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

# Test that --warn-on-unstable-apis correctly handles alias stability.
# A @stable alias can re-export unstable types without warning when the alias
# itself is referenced. Note: hierarchical propagation (suppressing warnings
# for members accessed through a stable alias) is not yet implemented.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

from test_std_mock import (
    StableAliasToUnstable,
    UnstableAlias,
    STABLE_CONSTANT,
    UNSTABLE_CONSTANT,
)


def test_stable_alias_to_unstable():
    # Using a stable alias that wraps an unstable struct should NOT warn.
    # The user is using the stable alias, not the underlying unstable type.
    # CHECK-NOT: warning{{.*}}StableAliasToUnstable
    var x: StableAliasToUnstable
    pass


def test_unstable_alias():
    # Using an unstable alias should trigger a warning, even though it wraps
    # a stable struct.
    # Note: Alias names include a hex suffix (e.g., 'UnstableAlias`0x1'). See MOCO-3108.
    # CHECK: warning: use of unstable API 'UnstableAlias{{.*}}'
    var y: UnstableAlias
    pass


def test_stable_constant():
    # Using a stable constant alias should NOT warn.
    # CHECK-NOT: warning{{.*}}STABLE_CONSTANT
    var a = STABLE_CONSTANT
    _ = a


def test_unstable_constant():
    # Using an unstable constant alias should trigger a warning.
    # Note: Alias names include a hex suffix (e.g., 'UNSTABLE_CONSTANT`0x3'). See MOCO-3108.
    # CHECK: warning: use of unstable API 'UNSTABLE_CONSTANT{{.*}}'
    var b = UNSTABLE_CONSTANT
    _ = b
