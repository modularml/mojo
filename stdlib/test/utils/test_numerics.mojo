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

from utils._numerics import FPUtils

alias FPU64 = FPUtils[DType.float64]


# CHECK-LABEL: test_numerics
fn test_numerics():
    print("== test_numerics")

    # CHECK: 23
    print(FPUtils[DType.float32].mantissa_width())

    # CHECK: 52
    print(FPUtils[DType.float64].mantissa_width())

    # CHECK: 127
    print(FPUtils[DType.float32].exponent_bias())

    # CHECK: 1023
    print(FPUtils[DType.float64].exponent_bias())

    # CHECK: 2
    print(FPU64.get_exponent(FPU64.set_exponent(1, 2)))
    # CHECK-NEXT: 3
    print(FPU64.get_mantissa(FPU64.set_mantissa(1, 3)))
    # CHECK-NEXT: 4
    print(FPU64.get_exponent(FPU64.set_exponent(-1, 4)))
    # CHECK-NEXT: 5
    print(FPU64.get_mantissa(FPU64.set_mantissa(-1, 5)))
    # CHECK-NEXT: True
    print(FPU64.get_sign(FPU64.set_sign(0, True)))
    # CHECK-NEXT: False
    print(FPU64.get_sign(FPU64.set_sign(0, False)))
    # CHECK-NEXT: True
    print(FPU64.get_sign(FPU64.set_sign(-0, True)))
    # CHECK-NEXT: False
    print(FPU64.get_sign(FPU64.set_sign(-0, False)))
    # CHECK-NEXT: False
    print(FPU64.get_sign(1))
    # CHECK-NEXT: True
    print(FPU64.get_sign(-1))
    # CHECK-NEXT: False
    print(FPU64.get_sign(FPU64.pack(False, 6, 12)))
    # CHECK-NEXT: 6
    print(FPU64.get_exponent(FPU64.pack(False, 6, 12)))
    # CHECK-NEXT: 12
    print(FPU64.get_mantissa(FPU64.pack(False, 6, 12)))
    # CHECK-NEXT: True
    print(FPU64.get_sign(FPU64.pack(True, 6, 12)))
    # CHECK-NEXT: 6
    print(FPU64.get_exponent(FPU64.pack(True, 6, 12)))
    # CHECK-NEXT: 12
    print(FPU64.get_mantissa(FPU64.pack(True, 6, 12)))


fn main():
    test_numerics()
