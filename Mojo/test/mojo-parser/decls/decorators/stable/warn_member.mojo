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

# Test that --warn-on-unstable-apis emits warnings for unstable method access
# from opted-in packages.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

from test_std_mock import StructWithMethods


def test_stable_method():
    # Calling a stable method should not trigger a warning.
    var s = StructWithMethods()
    _ = s.stable_method()


def test_unstable_method():
    # Calling an unstable method should trigger a warning.
    var s = StructWithMethods()
    # CHECK: warning: use of unstable API 'unstable_method'
    _ = s.unstable_method()
