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
# This file tests the Neon dotprod intrinsics
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: neon_dotprod
# RUN: %mojo-no-debug %s | FileCheck %s

from std.sys.info import CompilationTarget

from linalg.arch.cpu.neon_intrinsics import _neon_dotprod, _neon_dotprod_lane


# CHECK-LABEL: test_has_neon_int8_dotprod
def test_has_neon_int8_dotprod():
    print("== test_has_neon_int8_dotprod")

    # CHECK: True
    print(CompilationTarget.has_neon_int8_dotprod())


# CHECK-LABEL: test_int8_dotprod
def test_int8_dotprod():
    print("== test_int8_dotprod")

    var a = SIMD[.int8, 16](
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )
    var b = SIMD[.int8, 16](
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
    )
    var c = SIMD[.int32, 4](10000, 20000, 30000, 40000)

    # CHECK: [9966, 19950, 30062, 40302]
    print(_neon_dotprod(c, a, b))

    # CHECK: [10014, 20038, 30062, 40086]
    print(_neon_dotprod_lane[2](c, a, b))

    var b8 = SIMD[.int8, 8](-8, -6, -4, -2, 0, 2, 4, 6)

    # CHECK: [10028, 20076, 30124, 40172]
    print(_neon_dotprod_lane[1](c, a, b8))


# CHECK-LABEL: test_uint8_dotprod
def test_uint8_dotprod():
    print("== test_uint8_dotprod")

    var a = SIMD[.uint8, 16](
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )
    var b = SIMD[.uint8, 16](
        0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240
    )
    var c = SIMD[.int32, 4](10000, 20000, 30000, 40000)

    # CHECK: [10224, 22016, 35856, 51744]
    print(_neon_dotprod(c, a, b))

    # CHECK: [10608, 22016, 33424, 44832]
    print(_neon_dotprod_lane[1](c, a, b))

    var b8 = SIMD[.uint8, 8](1, 3, 5, 7, 9, 11, 13, 15)

    # CHECK: [10034, 20098, 30162, 40226]
    print(_neon_dotprod_lane[0](c, a, b8))


def main():
    test_has_neon_int8_dotprod()
    test_int8_dotprod()
    test_uint8_dotprod()
