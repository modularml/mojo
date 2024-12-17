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

from hashlib.fnv1a import Fnv1a
from testing import assert_equal, assert_not_equal, assert_true


def same_low_bits(i1: UInt64, i2: UInt64, bits: Int = 5) -> UInt8:
    var mask = (1 << bits) - 1
    return int(not (i1 ^ i2) & mask)


def test_hash_byte_array():
    # Test that values hash deterministically
    assert_equal(hash("a".unsafe_ptr(), 1), hash("a".unsafe_ptr(), 1))
    assert_equal(hash("b".unsafe_ptr(), 1), hash("b".unsafe_ptr(), 1))
    assert_equal(hash("c".unsafe_ptr(), 1), hash("c".unsafe_ptr(), 1))
    assert_equal(hash("d".unsafe_ptr(), 1), hash("d".unsafe_ptr(), 1))

    # Test that low bits are different
    var num_same: UInt8 = 0
    num_same += same_low_bits(
        hash("a".unsafe_ptr(), 1), hash("b".unsafe_ptr(), 1)
    )
    num_same += same_low_bits(
        hash("a".unsafe_ptr(), 1), hash("c".unsafe_ptr(), 1)
    )
    num_same += same_low_bits(
        hash("a".unsafe_ptr(), 1), hash("d".unsafe_ptr(), 1)
    )
    num_same += same_low_bits(
        hash("b".unsafe_ptr(), 1), hash("c".unsafe_ptr(), 1)
    )
    num_same += same_low_bits(
        hash("b".unsafe_ptr(), 1), hash("d".unsafe_ptr(), 1)
    )
    num_same += same_low_bits(
        hash("c".unsafe_ptr(), 1), hash("d".unsafe_ptr(), 1)
    )

    assert_true(num_same < 6, "too little entropy in hash fn low bits")


def _test_hash_int_simd[type: DType](bits: Int = 4, max_num_same: Int = 2):
    var a = Scalar[type](0)
    var b = Scalar[type](1)
    var c = Scalar[type](2)
    var d = Scalar[type](-1)

    # Test that values hash deterministically
    assert_equal(hash(a), hash(a))
    assert_equal(hash(b), hash(b))
    assert_equal(hash(c), hash(c))
    assert_equal(hash(d), hash(d))

    # Test that low bits are different
    var num_same: UInt8 = 0
    num_same += same_low_bits(hash(a), hash(b), bits)
    num_same += same_low_bits(hash(a), hash(c), bits)
    num_same += same_low_bits(hash(a), hash(d), bits)
    num_same += same_low_bits(hash(b), hash(c), bits)
    num_same += same_low_bits(hash(b), hash(d), bits)
    num_same += same_low_bits(hash(c), hash(d), bits)

    assert_true(
        num_same < max_num_same, "too little entropy in hash fn low bits"
    )


def test_hash_simd():
    _test_hash_int_simd[DType.int8]()
    _test_hash_int_simd[DType.int16]()
    _test_hash_int_simd[DType.int32]()
    _test_hash_int_simd[DType.int64]()
    # float32 currently has low entropy in the low bits for these test examples.
    # this could affect performance of small dicts some. Let's punt and see
    # if this is an issue in practice, if so we can specialize the float
    # hash implementation.
    _test_hash_int_simd[DType.float32](max_num_same=7)
    # TODO: test hashing different NaNs.

    # Test a couple other random things
    assert_not_equal(
        hash(Float32(3.14159)),
        hash(Float32(1e10)),
    )
    assert_equal(
        hash(Scalar[DType.bool](True)),
        hash(Scalar[DType.bool](True)),
    )
    assert_equal(
        hash(Scalar[DType.bool](False)),
        hash(Scalar[DType.bool](False)),
    )
    assert_not_equal(
        hash(Scalar[DType.bool](True)),
        hash(Scalar[DType.bool](False)),
    )
    assert_equal(
        hash(SIMD[DType.bool, 2](True)),
        hash(SIMD[DType.bool, 2](True)),
    )
    assert_equal(
        hash(SIMD[DType.bool, 2](False)),
        hash(SIMD[DType.bool, 2](False)),
    )
    assert_not_equal(
        hash(SIMD[DType.bool, 2](True)),
        hash(SIMD[DType.bool, 2](False)),
    )


fn test_issue_31111():
    _ = hash(Int(1))


def test_hash_comptime():
    alias hash_123 = hash[HasherType=Fnv1a]("123")
    assert_equal(hash_123, hash[HasherType=Fnv1a]("123"))

    alias hash_22 = hash[HasherType=Fnv1a](22)
    assert_equal(hash_22, hash[HasherType=Fnv1a](22))


def main():
    test_hash_byte_array()
    test_hash_simd()
    test_issue_31111()
    test_hash_comptime()
