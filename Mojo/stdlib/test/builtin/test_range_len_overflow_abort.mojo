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
# `range(...).__len__()` asserts when an unsigned count exceeds `Int.MAX`. The
# assert aborts, so verify it with FileCheck rather than a regular unit test.
#
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: test_range_len_overflow_aborts
def main():
    print("== test_range_len_overflow_aborts")
    # CHECK: range length exceeds Int.MAX
    var n = range(UInt(0), UInt.MAX).__len__()
    # CHECK-NOT: is never reached
    print("is never reached", n)
