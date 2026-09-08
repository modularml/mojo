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

# Sub-package of test_std_mock for testing intra-package stability checks.
# Code here uses unstable APIs from the parent package (test_std_mock).
# Since both are sub-packages of the same opted-in package, no warnings
# should be emitted.

from test_std_mock import UnstableStruct


@stable
def subpkg_stable_fn() -> Int:
    # This uses UnstableStruct from the parent package.  Both this sub-package
    # and the parent are under the opted-in "test_std_mock" package, so this
    # is intra-package usage and should NOT warn.
    var s = UnstableStruct()
    return s.unstable_method()
