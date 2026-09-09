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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Test that @__allow_legacy_any_origin_fields is recognized on struct fields
# and sets the allowLegacyAnyOrigin attribute in the IR.


# CHECK-LABEL: lit.struct.decl @WithLegacyField
struct WithLegacyField(Movable where False):
    # CHECK: lit.struct.field legacy_field {allowLegacyAnyOrigin}
    @__allow_legacy_any_origin_fields
    var legacy_field: Int

    # CHECK: lit.struct.field normal_field
    # CHECK-NOT: allowLegacyAnyOrigin
    var normal_field: Int
