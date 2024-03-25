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
# RUN: %mojo -debug-level full %s | FileCheck %s


# CHECK-LABEL: test_list
fn test_list():
    print("== test_list")
    # CHECK: 4
    print(len([1, 2.0, 3.14, [-1, -2]]))


# CHECK-LABEL: test_variadic_list
fn test_variadic_list():
    print("== test_variadic_list")

    @parameter
    fn print_list(*nums: Int):
        # CHECK: 5
        # CHECK: 8
        # CHECK: 6
        for num in nums:
            print(num)

        # CHECK: 3
        print(len(nums))

    print_list(5, 8, 6)


fn main():
    test_list()
    test_variadic_list()
