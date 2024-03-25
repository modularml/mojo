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

from sys.info import (
    alignof,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    sizeof,
)


# CHECK-LABEL: test_sizeof
fn test_sizeof():
    print("== test_sizeof")

    # CHECK: 2
    print(sizeof[__mlir_type.i16]())

    # CHECK: 2
    print(sizeof[__mlir_type.ui16]())

    # CHECK: 2
    print(sizeof[DType.int16]())

    # CHECK: 2
    print(sizeof[DType.uint16]())

    # CHECK: 4
    print(sizeof[SIMD[DType.int16, 2]]())


# CHECK-LABEL: test_alignof
fn test_alignof():
    print("== test_alignof")

    # CHECK: True
    print(alignof[__mlir_type.i16]() > 0)

    # CHECK: True
    print(alignof[__mlir_type.ui16]() > 0)

    # CHECK: True
    print(alignof[DType.int16]() > 0)

    # CHECK: True
    print(alignof[DType.uint16]() > 0)

    # CHECK: True
    print(alignof[SIMD[DType.int16, 2]]() > 0)


fn test_cores():
    # CHECK: True
    print(num_logical_cores() > 0)
    # CHECK: True
    print(num_physical_cores() > 0)
    # CHECK: True
    print(num_performance_cores() > 0)


fn main():
    test_sizeof()
    test_alignof()
    test_cores()
