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
# RUN: %mojo %s

from math import gcd
from sys import (
    compressed_store,
    masked_load,
    masked_store,
    strided_load,
    strided_store,
)
from sys.intrinsics import assume, likely, unlikely

from memory import UnsafePointer, memset_zero
from testing import assert_equal

alias F32x4 = SIMD[DType.float32, 4]
alias F32x8 = SIMD[DType.float32, 8]
alias iota_4 = F32x4(0.0, 1.0, 2.0, 3.0)
alias iota_8 = F32x8(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def test_intrinsic_comp_eval():
    alias res = gcd(5, 4)
    assert_equal(res, gcd(5, 4))


def test_compressed_store():
    var vector = UnsafePointer[Float32]().alloc(5)
    memset_zero(vector, 5)

    compressed_store(iota_4, vector, iota_4 >= 2)
    assert_equal(vector.load[width=4](0), F32x4(2.0, 3.0, 0.0, 0.0))

    # Just clear the buffer.
    vector.store(0, SIMD[DType.float32, 4](0))

    var val = F32x4(0.0, 1.0, 3.0, 0.0)
    compressed_store(val, vector, val != 0)
    assert_equal(vector.load[width=4](0), F32x4(1.0, 3.0, 0.0, 0.0))
    vector.free()


def test_masked_load():
    var vector = UnsafePointer[Float32]().alloc(5)
    for i in range(5):
        vector[i] = 1

    assert_equal(
        masked_load[4](vector, iota_4 < 5, 0), F32x4(1.0, 1.0, 1.0, 1.0)
    )

    assert_equal(
        masked_load[8](vector, iota_8 < 5, 0),
        F32x8(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    )

    assert_equal(
        masked_load[8](
            vector, iota_8 < 5, F32x8(43, 321, 12, 312, 323, 15, 9, 3)
        ),
        F32x8(1.0, 1.0, 1.0, 1.0, 1.0, 15.0, 9.0, 3.0),
    )

    assert_equal(
        masked_load[8](
            vector, iota_8 < 2, F32x8(43, 321, 12, 312, 323, 15, 9, 3)
        ),
        F32x8(1.0, 1.0, 12.0, 312.0, 323.0, 15.0, 9.0, 3.0),
    )
    vector.free()


def test_masked_store():
    var vector = UnsafePointer[Float32]().alloc(5)
    memset_zero(vector, 5)

    masked_store[4](iota_4, vector, iota_4 < 5)
    assert_equal(vector.load[width=4](0), F32x4(0.0, 1.0, 2.0, 3.0))

    masked_store[8](iota_8, vector, iota_8 < 5)
    assert_equal(
        masked_load[8](vector, iota_8 < 5, 33),
        F32x8(0.0, 1.0, 2.0, 3.0, 4.0, 33.0, 33.0, 33.0),
    )
    vector.free()


fn test_strided_load() raises:
    alias size = 16
    var vector = UnsafePointer[Float32]().alloc(size)

    for i in range(size):
        vector[i] = i

    var s = strided_load[4](vector, 4)
    assert_equal(s, SIMD[DType.float32, 4](0, 4, 8, 12))

    vector.free()


fn test_strided_store() raises:
    alias size = 8
    var vector = UnsafePointer[Float32]().alloc(size)
    memset_zero(vector, size)

    strided_store(SIMD[DType.float32, 4](99, 12, 23, 56), vector, 2)
    assert_equal(vector[0], 99.0)
    assert_equal(vector[1], 0.0)
    assert_equal(vector[2], 12.0)
    assert_equal(vector[3], 0.0)
    assert_equal(vector[4], 23.0)
    assert_equal(vector[5], 0.0)
    assert_equal(vector[6], 56.0)
    assert_equal(vector[7], 0.0)

    vector.free()


def test_likely_unlikely():
    assert_equal(likely(True), True)
    assert_equal(unlikely(True), True)


def test_assume():
    assume(True)


def main():
    test_intrinsic_comp_eval()
    test_compressed_store()
    test_masked_load()
    test_masked_store()
    test_strided_load()
    test_strided_store()
    test_likely_unlikely()
    test_assume()
