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

from std.testing import TestSuite


# CHECK-LABEL: test_fail_list_index
def test_fail_list_index() raises:
    print("== test_fail_list_index")
    # CHECK: index 4 is out of bounds, valid range is 0 to 2
    var nums: List = [1, 2, 3]
    print(nums[4])

    # CHECK-NOT: is never reached
    print("is never reached")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
