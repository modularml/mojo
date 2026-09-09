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

# Test that @stable(recursive=True) used in a dependency does NOT suppress
# warnings in the importing file. The override is strictly file-scoped.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

# test_std_mock_reexporter imports UnstableStruct with @stable(recursive=True).
# That override lives in the reexporter's file scope only.
from test_std_mock_reexporter import UnstableStruct


def test_type_ref_still_warns():
    # CHECK: warning: use of unstable API 'UnstableStruct'
    var _x: UnstableStruct


def test_constructor_still_warns():
    # CHECK: warning: use of unstable API 'UnstableStruct'
    var x = UnstableStruct()
    # CHECK: warning: use of unstable API 'unstable_method'
    _ = x.unstable_method()
