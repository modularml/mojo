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
# Regression test: `check_bounds` called directly from an ordinary (non
# `@always_inline`) function used to fail to lower under `-g` because its
# `call_location[inline_count=2]` could not be inlined the requested number of
# times. It must now compile and degrade the assert to the best-available
# location.
#
# ===----------------------------------------------------------------------=== #

from std.collections import check_bounds


def index_directly(idx: Int, size: Int):
    check_bounds(idx, size)


# CHECK-LABEL: test_direct
def main():
    print("== test_direct")

    # CHECK: test_check_bounds_direct_call.mojo:26:17: Assert Error: index 5 is out of bounds, valid range is 0 to 2
    index_directly(5, 3)

    # CHECK-NOT: is never reached
    print("is never reached")
