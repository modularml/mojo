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

from std.math import gcd
from std.sys import (
    compressed_store,
    masked_load,
    masked_store,
    strided_load,
    strided_store,
)
from std.sys.intrinsics import assume, likely, unlikely

from std.testing import assert_equal
from std.testing import TestSuite

comptime F32x4 = SIMD[.float32, 4]
comptime F32x8 = SIMD[.float32, 8]
comptime iota_4 = F32x4(0.0, 1.0, 2.0, 3.0)
comptime iota_8 = F32x8(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def test_intrinsic_comp_eval() raises:
    comptime res = gcd(5, 4)
    assert_equal(res, gcd(5, 4))


def test_compressed_store() raises:
    var vector = List(length=5, fill=Float32(0))

    compressed_store(iota_4, vector.unsafe_ptr(), iota_4.ge(2))
    assert_equal(
        vector.unsafe_ptr().unsafe_load[width=4](0), F32x4(2.0, 3.0, 0.0, 0.0)
    )

    # Just clear the buffer.
    vector.unsafe_ptr().unsafe_store(0, SIMD[.float32, 4](0))

    var val = F32x4(0.0, 1.0, 3.0, 0.0)
    compressed_store(val, vector.unsafe_ptr(), val.ne(0))
    assert_equal(
        vector.unsafe_ptr().unsafe_load[width=4](0), F32x4(1.0, 3.0, 0.0, 0.0)
    )


def test_masked_load() raises:
    var vector = List(length=5, fill=Float32(0))
    for i in range(5):
        vector[i] = 1

    assert_equal(
        masked_load[4](vector.unsafe_ptr(), iota_4.lt(5), 0),
        F32x4(1.0, 1.0, 1.0, 1.0),
    )

    assert_equal(
        masked_load[8](vector.unsafe_ptr(), iota_8.lt(5), 0),
        F32x8(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    )

    assert_equal(
        masked_load[8](
            vector.unsafe_ptr(),
            iota_8.lt(5),
            F32x8(43, 321, 12, 312, 323, 15, 9, 3),
        ),
        F32x8(1.0, 1.0, 1.0, 1.0, 1.0, 15.0, 9.0, 3.0),
    )

    assert_equal(
        masked_load[8](
            vector.unsafe_ptr(),
            iota_8.lt(2),
            F32x8(43, 321, 12, 312, 323, 15, 9, 3),
        ),
        F32x8(1.0, 1.0, 12.0, 312.0, 323.0, 15.0, 9.0, 3.0),
    )


def test_masked_store() raises:
    var vector = List(length=5, fill=Float32(0))

    masked_store[4](iota_4, vector.unsafe_ptr(), iota_4.lt(5))
    assert_equal(
        vector.unsafe_ptr().unsafe_load[width=4](0), F32x4(0.0, 1.0, 2.0, 3.0)
    )

    masked_store[8](iota_8, vector.unsafe_ptr(), iota_8.lt(5))
    assert_equal(
        masked_load[8](vector.unsafe_ptr(), iota_8.lt(5), 33),
        F32x8(0.0, 1.0, 2.0, 3.0, 4.0, 33.0, 33.0, 33.0),
    )


def test_strided_load() raises:
    comptime size = 16
    var vector = List(length=size, fill=Float32(0))

    for i in range(size):
        vector[i] = Float32(i)

    var s = strided_load[4](vector.unsafe_ptr(), 4)
    assert_equal(s, SIMD[.float32, 4](0, 4, 8, 12))


def test_strided_store() raises:
    comptime size = 8
    var vector = List(length=size, fill=Float32(0))

    strided_store(SIMD[.float32, 4](99, 12, 23, 56), vector.unsafe_ptr(), 2)
    assert_equal(vector[0], 99.0)
    assert_equal(vector[1], 0.0)
    assert_equal(vector[2], 12.0)
    assert_equal(vector[3], 0.0)
    assert_equal(vector[4], 23.0)
    assert_equal(vector[5], 0.0)
    assert_equal(vector[6], 56.0)
    assert_equal(vector[7], 0.0)


def test_likely_unlikely() raises:
    assert_equal(likely(True), True)
    assert_equal(unlikely(True), True)


def test_assume() raises:
    assume(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
