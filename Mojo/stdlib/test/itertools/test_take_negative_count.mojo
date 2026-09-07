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

from std.itertools import take
from std.testing import TestSuite


# CHECK-LABEL: test_take_negative_count
def test_take_negative_count() raises:
    print("== test_take_negative_count")
    var nums = [1, 2, 3]
    # CHECK: The `count` argument must be non-negative
    _ = take(nums, -1)

    # CHECK-NOT: is never reached
    print("is never reached")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
