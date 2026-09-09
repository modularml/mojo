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
# This file only tests the debug_assert function
#
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: test_fail
def main():
    print("== test_fail")
    # CHECK: formatted failure message: 2, 4
    debug_assert[assert_mode="safe"](
        False, "formatted failure message: ", 2, ", ", UInt8(4)
    )
    # CHECK-NOT: is never reached
    print("is never reached")
