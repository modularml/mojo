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
# This file tests the Neon matmul intrinsics
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: neon_matmul
# RUN: %mojo-no-debug %s | FileCheck %s

from std.sys.info import CompilationTarget

from linalg.arch.cpu.neon_intrinsics import _neon_matmul


# CHECK-LABEL: test_has_neon_int8_matmul
def test_has_neon_int8_matmul():
    print("== test_has_neon_int8_matmul")

    # CHECK: True
    print(CompilationTarget.has_neon_int8_matmul())


# CHECK-LABEL: test_s8s8_matmul
def test_s8s8_matmul():
    print("== test_s8s8_matmul")

    var a = SIMD[.int8, 16](
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1
    )
    var b = SIMD[.int8, 16](
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
    )
    var c = SIMD[.int32, 4](10000, 20000, 30000, 40000)

    # CHECK: [9916, 20140, 30204, 39916]
    print(_neon_matmul(c, a, b))


# CHECK-LABEL: test_u8u8_matmul
def test_u8u8_matmul():
    print("== test_u8u8_matmul")

    var a = SIMD[.uint8, 16](
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )
    var b = SIMD[.uint8, 16](
        0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240
    )
    var c = SIMD[.int32, 4](10000, 20000, 30000, 40000)

    # CHECK: [12240, 25824, 35824, 57600]
    print(_neon_matmul(c, a, b))


# CHECK-LABEL: test_u8s8_matmul
def test_u8s8_matmul():
    print("== test_u8s8_matmul")

    var a = SIMD[.uint8, 16](
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )
    var b = SIMD[.int8, 16](
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
    )
    var c = SIMD[.int32, 4](10000, 20000, 30000, 40000)

    # CHECK: [9916, 20140, 29628, 40364]
    print(_neon_matmul(c, a, b))


def main():
    test_has_neon_int8_matmul()
    test_s8s8_matmul()
    test_u8u8_matmul()
    test_u8s8_matmul()
