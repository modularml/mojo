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

# Test that intra-package usage across sub-packages does NOT warn.
#
# Code in test_std_mock.subpkg uses unstable APIs from test_std_mock.
# Both are under the same opted-in package ("test_std_mock"), so the
# compiler must recognize them as the same package and suppress warnings.
# This requires walking the full ancestor PackageOp chain, not just
# checking the nearest (leaf) package name.

# RUN: %parse-mojo-isolated -mojo-search-paths=%S -warn-on-unstable-apis %s 2>&1 | FileCheck %s

from test_std_mock.subpkg import subpkg_stable_fn


def test():
    # The function itself is @stable, so no warning for calling it.
    # Internally it uses unstable APIs from a sibling sub-package,
    # which should also not warn (intra-package).
    #
    # CHECK-NOT: no_warn_subpackage.mojo:{{.*}}warning: use of unstable API
    _ = subpkg_stable_fn()
