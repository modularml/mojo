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


from std.testing import assert_equal, TestSuite
from test_utils import ObservableDel, Pinned


def test_deinit_results_in_deinitializer_being_run() raises:
    var is_deinitialized = False
    var observe = ObservableDel(Pointer(to=is_deinitialized))
    assert_equal(is_deinitialized, False)
    deinit(observe^)
    assert_equal(is_deinitialized, True)


def test_deinit_works_with_non_movable_types() raises:
    var pinned = Pinned()
    deinit(pinned^)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
