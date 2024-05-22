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
# RUN: %mojo  -O0 %s

# Issue #31111 -- run this test with -O0 also.

# These tests aren't _great_. They're platform specific, and implementation
# specific. But for now they test behavior and reproducibility.

from builtin.hash import _hash_simd
from testing import assert_equal, assert_not_equal, assert_true


def same_low_bits(i1: Int, i2: Int, bits: Int = 5) -> UInt8:
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

    # This test is just really bad. We really need to re-evaluate the
    # right way to test these. Hash function behavior varies a bit  based
    # on architecture, so these tests as-is end up being really flaky.
    # Making this _much_ more relaxed for now, but at least still testing
    # that at least the hash function returns _some_ different things.

    # TODO(MSTDL-472): fix this flaky check
    # assert_true(num_same < 6, "too little entropy in hash fn low bits")


def _test_hash_int_simd[type: DType](bits: Int = 4, max_num_same: Int = 2):
    var a = Scalar[type](0)
    var b = Scalar[type](1)
    var c = Scalar[type](2)
    var d = Scalar[type](-1)

    # Test that values hash deterministically
    assert_equal(_hash_simd(a), _hash_simd(a))
    assert_equal(_hash_simd(b), _hash_simd(b))
    assert_equal(_hash_simd(c), _hash_simd(c))
    assert_equal(_hash_simd(d), _hash_simd(d))

    # Test that low bits are different
    var num_same: UInt8 = 0
    num_same += same_low_bits(_hash_simd(a), _hash_simd(b), bits)
    num_same += same_low_bits(_hash_simd(a), _hash_simd(c), bits)
    num_same += same_low_bits(_hash_simd(a), _hash_simd(d), bits)
    num_same += same_low_bits(_hash_simd(b), _hash_simd(c), bits)
    num_same += same_low_bits(_hash_simd(b), _hash_simd(d), bits)
    num_same += same_low_bits(_hash_simd(c), _hash_simd(d), bits)

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
        _hash_simd(Float32(3.14159)),
        _hash_simd(Float32(1e10)),
    )
    assert_equal(
        _hash_simd(Scalar[DType.bool](True)),
        _hash_simd(Scalar[DType.bool](True)),
    )
    assert_equal(
        _hash_simd(Scalar[DType.bool](False)),
        _hash_simd(Scalar[DType.bool](False)),
    )
    assert_not_equal(
        _hash_simd(Scalar[DType.bool](True)),
        _hash_simd(Scalar[DType.bool](False)),
    )
    assert_equal(
        _hash_simd(SIMD[DType.bool, 2](True)),
        _hash_simd(SIMD[DType.bool, 2](True)),
    )
    assert_equal(
        _hash_simd(SIMD[DType.bool, 2](False)),
        _hash_simd(SIMD[DType.bool, 2](False)),
    )
    assert_not_equal(
        _hash_simd(SIMD[DType.bool, 2](True)),
        _hash_simd(SIMD[DType.bool, 2](False)),
    )


fn test_issue_31111():
    _ = hash(Int(1))


def main():
    test_hash_byte_array()
    test_hash_simd()
    test_issue_31111()
