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
# RUN: %mojo %s | FileCheck %s


# CHECK-LABEL: test_range_len
fn test_range_len():
    print("== test_range_len")

    # CHECK: 10
    print(range(10).__len__())

    # CHECK: 10
    print(range(0, 10).__len__())

    # CHECK: 5
    print(range(5, 10).__len__())

    # CHECK: 10
    print(range(10, 0, -1).__len__())

    # CHECK: 5
    print(range(0, 10, 2).__len__())

    # CHECK: 3
    print(range(38, -13, -23).__len__())


# CHECK-LABEL: test_range_getitem
fn test_range_getitem():
    print("== test_range_getitem")

    # CHECK: 5
    print(range(10)[5])

    # CHECK: 3
    print(range(0, 10)[3])

    # CHECK: 8
    print(range(5, 10)[3])

    # CHECK: 8
    print(range(10, 0, -1)[2])

    # CHECK: 8
    print(range(0, 10, 2)[4])


fn main():
    test_range_len()
    test_range_getitem()
