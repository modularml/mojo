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

# Demonstrates a known limitation of @stable(recursive=True) on imports:
# the override covers the imported name and its direct members, but NOT other
# unstable types exposed through those members. Accessing a comptime alias
# member that resolves to a different unstable type, then calling a static
# method on it, still warns.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

@stable(recursive=True)
from test_std_mock import UnstableStruct

def test_hole():
    # Accessing a comptime alias member is suppressed — it's a member of
    # UnstableStruct, which is in the override set.
    # CHECK-NOT: warning{{.*}}AliasedType
    comptime B = UnstableStruct.AliasedType
    # Calling a static method on the aliased type (AnotherUnstableStruct) is
    # NOT suppressed — AnotherUnstableStruct is not in the stable override set.
    # CHECK: warning: use of unstable API 'static_method'
    _ = B.static_method()
