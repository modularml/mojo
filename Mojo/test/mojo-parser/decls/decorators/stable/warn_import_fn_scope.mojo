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

# Test that @stable(recursive=True) on a function-level import suppresses
# unstable API warnings within that function, and does not bleed to sibling
# functions that import the same name without the decorator.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

# CHECK-NOT: warning: use of unstable API

def test_fn_level_suppression():
    # @stable(recursive=True) on an import inside a function body suppresses
    # warnings for the imported name and its members within this function only.
    @stable(recursive=True)
    from test_std_mock import UnstableStruct
    var x = UnstableStruct()
    _ = x.unstable_method()
    _ = UnstableStruct.UNSTABLE_CONST


def test_no_bleed_to_sibling():
    # The stable override from the function above is scoped to that function's
    # ASTDecl; it does not suppress warnings here.
    from test_std_mock import UnstableStruct
    # CHECK: warning: use of unstable API 'UnstableStruct'
    var x = UnstableStruct()
    # CHECK: warning: use of unstable API 'unstable_method'
    _ = x.unstable_method()
