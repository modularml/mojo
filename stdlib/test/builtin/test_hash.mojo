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

from testing import assert_equal, assert_not_equal, assert_true


def same_low_bits(i1: UInt64, i2: UInt64, bits: Int = 5) -> Int:
    var mask = (1 << bits) - 1
    return int(not (i1 ^ i2) & mask)


def test_hash_byte_array():
    # Test that values hash deterministically
    assert_equal(hash("a"), hash("a"))
    assert_equal(hash("b"), hash("b"))
    assert_equal(hash("c"), hash("c"))
    assert_equal(hash("d"), hash("d"))

    # Test that low bits are different
    var num_same = 0
    num_same += same_low_bits(hash("a"), hash("b"))
    num_same += same_low_bits(hash("a"), hash("c"))
    num_same += same_low_bits(hash("a"), hash("d"))
    num_same += same_low_bits(hash("b"), hash("c"))
    num_same += same_low_bits(hash("b"), hash("d"))
    num_same += same_low_bits(hash("c"), hash("d"))

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
    assert_equal(hash(a), hash(a))
    assert_equal(hash(b), hash(b))
    assert_equal(hash(c), hash(c))
    assert_equal(hash(d), hash(d))

    # Test that low bits are different
    var num_same = 0
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


def main():
    test_hash_byte_array()
    test_hash_simd()
    test_issue_31111()
