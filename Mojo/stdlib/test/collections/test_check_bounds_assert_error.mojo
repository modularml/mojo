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
#
# This file only tests the check_bounds function
#
# ===----------------------------------------------------------------------=== #

from std.collections import check_bounds


struct Collection:
    var size: Int

    def __init__(out self, size: Int):
        self.size = size

    @always_inline
    def __getitem__(self, idx: Int) -> Int:
        check_bounds(idx, self.size)
        return 0


# CHECK-LABEL: test_fail
def main():
    print("== test_fail")

    var collection = Collection(2)

    # CHECK: test_check_bounds_assert_error.mojo:40:19: Assert Error: index 2 is out of bounds, valid range is 0 to 1
    _ = collection[2]

    # CHECK-NOT: is never reached
    print("is never reached")
