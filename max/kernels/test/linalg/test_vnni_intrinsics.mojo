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
# This file tests the vnni intrinsics
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: avx2
# RUN: %mojo-no-debug %s

from std.sys.info import CompilationTarget

from std.memory import unsafe_stack_allocation
from linalg.arch.cpu.vnni_intrinsics import (
    dot_i8_to_i32_AVX2,
    dot_i8_to_i32_saturated_AVX2,
    dot_i16_to_i32_AVX2,
    dot_i16_to_i32_x86,
)
from std.testing import assert_equal


def test_i8_to_i32() raises:
    var a = unsafe_stack_allocation[16 * 64, DType.uint8, alignment=64]()
    var asat = unsafe_stack_allocation[16 * 64, DType.uint8, alignment=64]()
    var b = unsafe_stack_allocation[64 * 16, DType.int8, alignment=64]()

    var c = unsafe_stack_allocation[16 * 16, DType.int32, alignment=64]()
    var csat = unsafe_stack_allocation[16 * 16, DType.int32, alignment=64]()

    for i in range(16 * 64):
        a[unsafe_offset=i] = UInt8(i & 255)
        asat[unsafe_offset=i] = UInt8(i & 127)
        b[unsafe_offset=i] = Int8((i & 255) - 128)

    for i in range(16 * 16):
        c[unsafe_offset=i] = Int32(i)
        csat[unsafe_offset=i] = c[unsafe_offset=i]

    var av16u = (
        a.unsafe_offset(128 + 64)
        .unsafe_bitcast[Int32]()
        .unsafe_load[width=16]()
    )
    var av16s = (
        asat.unsafe_offset(128 + 64)
        .unsafe_bitcast[Int32]()
        .unsafe_load[width=16]()
    )
    var bv16 = b.unsafe_bitcast[Int32]().unsafe_load[width=16]()
    var cv16u: SIMD[.int32, 16]
    var cv16s: SIMD[.int32, 16]
    if CompilationTarget.has_avx512f():
        cv16u = dot_i8_to_i32_AVX2[16](c.unsafe_load[width=16](), av16u, bv16)
        cv16s = dot_i8_to_i32_saturated_AVX2[16](
            c.unsafe_load[width=16](), av16s, bv16
        )
    else:
        # split the vectors into high and low
        var cv8ul = dot_i8_to_i32_AVX2[8](
            c.unsafe_load[width=8](), av16u.slice[8](), bv16.slice[8]()
        )
        var cv8sl = dot_i8_to_i32_saturated_AVX2[8](
            c.unsafe_load[width=8](), av16s.slice[8](), bv16.slice[8]()
        )
        var cv8uh = dot_i8_to_i32_AVX2[8](
            c.unsafe_offset(8).unsafe_load[width=8](),
            av16u.slice[8, offset=8](),
            bv16.slice[8, offset=8](),
        )
        var cv8sh = dot_i8_to_i32_saturated_AVX2[8](
            c.unsafe_offset(8).unsafe_load[width=8](),
            av16s.slice[8, offset=8](),
            bv16.slice[8, offset=8](),
        )
        cv16u = cv8ul.join(cv8uh)
        cv16s = cv8sl.join(cv8sh)

    assert_equal(
        cv16u,
        SIMD[.int32, 16](
            -97906,
            -96769,
            -95504,
            -94111,
            -92590,
            -90941,
            -89164,
            -87259,
            -85226,
            -83065,
            -80776,
            -78359,
            -75814,
            -73141,
            -70340,
            -67411,
        ),
    )
    assert_equal(
        cv16s,
        SIMD[.int32, 16](
            -33138,
            -34049,
            -34832,
            -35487,
            -36014,
            -36413,
            -36684,
            -36827,
            -36842,
            -36729,
            -36488,
            -36119,
            -35622,
            -34997,
            -34244,
            -33363,
        ),
    )

    var av8u = (
        a.unsafe_offset(128 + 64).unsafe_bitcast[Int32]().unsafe_load[width=8]()
    )
    var av8s = (
        asat.unsafe_offset(128 + 64)
        .unsafe_bitcast[Int32]()
        .unsafe_load[width=8]()
    )
    var bv8 = b.unsafe_bitcast[Int32]().unsafe_load[width=8]()
    var cv8u = dot_i8_to_i32_AVX2[8](
        c.unsafe_bitcast[Int32]().unsafe_load[width=8](), av8u, bv8
    )
    var cv8s = dot_i8_to_i32_saturated_AVX2[8](
        c.unsafe_bitcast[Int32]().unsafe_load[width=8](), av8s, bv8
    )

    assert_equal(
        cv8u,
        SIMD[.int32, 8](
            -97906, -96769, -95504, -94111, -92590, -90941, -89164, -87259
        ),
    )
    assert_equal(
        cv8s,
        SIMD[.int32, 8](
            -33138, -34049, -34832, -35487, -36014, -36413, -36684, -36827
        ),
    )

    var av4u = (
        a.unsafe_offset(128 + 64).unsafe_bitcast[Int32]().unsafe_load[width=4]()
    )
    var av4s = (
        asat.unsafe_offset(128 + 64)
        .unsafe_bitcast[Int32]()
        .unsafe_load[width=4]()
    )
    var bv4 = b.unsafe_bitcast[Int32]().unsafe_load[width=4]()
    var cv4u = dot_i8_to_i32_AVX2[4](
        c.unsafe_bitcast[Int32]().unsafe_load[width=4](), av4u, bv4
    )
    var cv4s = dot_i8_to_i32_saturated_AVX2[4](
        c.unsafe_bitcast[Int32]().unsafe_load[width=4](), av4s, bv4
    )

    assert_equal(cv4u, SIMD[.int32, 4](-97906, -96769, -95504, -94111))
    assert_equal(cv4s, SIMD[.int32, 4](-33138, -34049, -34832, -35487))


def test_i16_to_i32() raises:
    def test_simd_width[width: Int]() raises:
        var a = SIMD[.int16, SIMDLength(width) * 2]()
        var b = SIMD[.int16, SIMDLength(width) * 2]()
        var c_start = SIMD[.int32, width]()
        var c_golden = SIMD[.int32, width]()

        comptime for i in range(width * 2):
            a[i] = Int16(i * 17 - 191)
            b[i] = Int16(i * 19 + 155)

        comptime for i in range(width):
            c_start[i] = Int32(i * 233 - 322)

        comptime for i in range(width):
            c_golden[i] = c_start[i]

            comptime for j in range(2):
                var a_val = a[i * 2 + j].cast[.int32]()
                var b_val = b[i * 2 + j].cast[.int32]()
                c_golden[i] += a_val * b_val

        var c_avx2 = dot_i16_to_i32_AVX2(c_start, a, b)
        assert_equal(c_golden, c_avx2)

        var c_x86 = dot_i16_to_i32_x86(c_start, a, b)
        assert_equal(c_golden, c_x86)

    comptime if CompilationTarget.has_avx512f():
        test_simd_width[16]()

    test_simd_width[8]()
    test_simd_width[4]()


def main() raises:
    test_i8_to_i32()
    test_i16_to_i32()
