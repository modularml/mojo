# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# This file only tests that conversion of negative IntLiteral to UInt fails.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_not
<<<<<<<< HEAD:stdlib/test/builtin/test_uint_error.mojo
# RUN: not mojo %s 2>&1 | FileCheck %s


fn main():
    # CHECK: integer value -1 is negative, but is being converted to an unsigned type
    print(UInt(-1))
    # CHECK-NOT: is never reached
========
# RUN: not --crash %bare-mojo -D BUILD_TYPE=debug %s 2>&1 | FileCheck %s -check-prefix=CHECK-FAIL


# CHECK-FAIL-LABEL: test_fail_list_index
fn main():
    print("== test_fail_list_index")
    # CHECK-FAIL: index: 4 is out of bounds for `List` of size: 3
    nums = List[Int](1, 2, 3)
    print(nums[4])

    # CHECK-FAIL-NOT: is never reached
>>>>>>>> origin/nightly:stdlib/test/collections/test_list_getitem_invalid_index.mojo
    print("is never reached")
