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

# Test that @stable(recursive=True) on an import suppresses unstable API
# warnings for the imported name and its members.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

# Import with @stable(recursive=True) — warnings for these names are suppressed.
@stable(recursive=True)
from test_std_mock import UnstableStruct
@stable(recursive=True)
from test_std_mock import unstable_fn
@stable(recursive=True)
from test_std_mock import UnstableTraitWithMembers


def test_stable_import_suppresses_type():
    var _x: UnstableStruct


def test_stable_import_suppresses_fn():
    unstable_fn()


def test_member_access_suppressed():
    # Method and comptime member accesses on a struct imported with
    # @stable(recursive=True) should also have their warnings suppressed.
    var x = UnstableStruct()
    _ = x.unstable_method()
    _ = UnstableStruct.UNSTABLE_CONST


struct LocalImpl(UnstableTraitWithMembers, Movable):
    # Using an unstable trait as a parent in the current file should not warn.
    def __init__(out self):
        pass


def test_trait_default_method_suppressed():
    # Calling the default-impl method from an unstable trait should not warn.
    var x = LocalImpl()
    _ = x.default_method()


def test_trait_assoc_type_suppressed():
    # Using the associated type fulfilled by LocalImpl should not warn.
    var _x: LocalImpl.ASSOC_TYPE


# The single CHECK-NOT below covers all test functions above.
# CHECK-NOT: warn_import.mojo:{{.*}}warning: use of unstable API
