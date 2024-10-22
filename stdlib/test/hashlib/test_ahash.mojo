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

from bit import pop_count
from builtin._location import __call_location

from hashlib.ahash import AHasher
from testing import assert_equal, assert_not_equal, assert_true
from memory import memset_zero, stack_allocation
from words import *


def dif_bits(i1: UInt64, i2: UInt64) -> Int:
    return int(pop_count(i1 ^ i2))


@always_inline
def assert_dif_hashes(hashes: List[UInt64], upper_bound: Int):
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            var diff = dif_bits(hashes[i], hashes[j])
            assert_true(
                diff > upper_bound,
                str("Index: {}:{}, diff between: {} and {} is: {}").format(
                    i, j, hashes[i], hashes[j], diff
                ),
                location=__call_location(),
            )


alias hasher0 = AHasher[SIMD[DType.uint64, 4](0, 0, 0, 0)]
alias hasher1 = AHasher[SIMD[DType.uint64, 4](1, 0, 0, 0)]


def test_hash_byte_array():
    assert_equal(hash[HasherType=hasher0]("a"), hash[HasherType=hasher0]("a"))
    assert_equal(hash[HasherType=hasher1]("a"), hash[HasherType=hasher1]("a"))
    assert_not_equal(
        hash[HasherType=hasher0]("a"), hash[HasherType=hasher1]("a")
    )
    assert_equal(hash[HasherType=hasher0]("b"), hash[HasherType=hasher0]("b"))
    assert_equal(hash[HasherType=hasher1]("b"), hash[HasherType=hasher1]("b"))
    assert_not_equal(
        hash[HasherType=hasher0]("b"), hash[HasherType=hasher1]("b")
    )
    assert_equal(hash[HasherType=hasher0]("c"), hash[HasherType=hasher0]("c"))
    assert_equal(hash[HasherType=hasher1]("c"), hash[HasherType=hasher1]("c"))
    assert_not_equal(
        hash[HasherType=hasher0]("c"), hash[HasherType=hasher1]("c")
    )
    assert_equal(hash[HasherType=hasher0]("d"), hash[HasherType=hasher0]("d"))
    assert_equal(hash[HasherType=hasher1]("d"), hash[HasherType=hasher1]("d"))
    assert_not_equal(
        hash[HasherType=hasher0]("d"),
        hash[HasherType=hasher1]("d"),
    )


def test_avalanche():
    # test that values which differ just in one bit,
    # produce significatly different hash values
    var data = stack_allocation[256, UInt8]()
    memset_zero(data, 256)
    var hashes0 = List[UInt64]()
    var hashes1 = List[UInt64]()
    hashes0.append(hash[HasherType=hasher0](data, 256))
    hashes1.append(hash[HasherType=hasher1](data, 256))

    for i in range(256):
        memset_zero(data, 256)
        var v = 1 << (i & 7)
        data[i >> 3] = v
        hashes0.append(hash[HasherType=hasher0](data, 256))
        hashes1.append(hash[HasherType=hasher1](data, 256))

    for i in range(len(hashes0)):
        var diff = dif_bits(hashes0[i], hashes1[i])
        assert_true(
            diff > 16,
            str("Index: {}, diff between: {} and {} is: {}").format(
                i, hashes0[i], hashes1[i], diff
            ),
        )

    assert_dif_hashes(hashes0, 12)
    assert_dif_hashes(hashes1, 12)


def test_trailing_zeros():
    # checks that a value with different amount of trailing zeros,
    # results in significantly different hash values
    var data = stack_allocation[8, UInt8]()
    memset_zero(data, 8)
    data[0] = 23
    var hashes0 = List[UInt64]()
    var hashes1 = List[UInt64]()
    for i in range(1, 9):
        hashes0.append(hash[HasherType=hasher0](data, i))
        hashes1.append(hash[HasherType=hasher1](data, i))

    for i in range(len(hashes0)):
        var diff = dif_bits(hashes0[i], hashes1[i])
        assert_true(
            diff > 18,
            str("Index: {}, diff between: {} and {} is: {}").format(
                i, hashes0[i], hashes1[i], diff
            ),
        )

    assert_dif_hashes(hashes0, 18)
    assert_dif_hashes(hashes1, 18)


@always_inline
def assert_fill_factor[
    label: String
](words: List[String], num_buckets: Int, lower_bound: Float64):
    # A perfect hash function is when the number of buckets is equal to number of words
    # and the fill factor results in 1.0
    var buckets = List[Int](0) * num_buckets
    for w in words:
        var h = hash[HasherType=hasher0](w[])
        buckets[int(h) % num_buckets] += 1
    var unfilled = 0
    for v in buckets:
        if v[] == 0:
            unfilled += 1

    var fill_factor = 1 - unfilled / num_buckets
    assert_true(
        fill_factor >= lower_bound,
        str("Fill factor for {} is {}, provided lower boound was {}").format(
            label, fill_factor, lower_bound
        ),
        location=__call_location(),
    )


def test_fill_factor():
    var words = List[String]()

    words = gen_word_pairs[words_ar]()
    assert_fill_factor["AR"](words, len(words), 0.63)
    assert_fill_factor["AR"](words, len(words) // 2, 0.86)
    assert_fill_factor["AR"](words, len(words) // 4, 0.98)
    assert_fill_factor["AR"](words, len(words) // 13, 1.0)

    words = gen_word_pairs[words_el]()
    assert_fill_factor["EL"](words, len(words), 0.63)
    assert_fill_factor["EL"](words, len(words) // 2, 0.86)
    assert_fill_factor["EL"](words, len(words) // 4, 0.98)
    assert_fill_factor["EL"](words, len(words) // 13, 1.0)

    words = gen_word_pairs[words_en]()
    assert_fill_factor["EN"](words, len(words), 0.63)
    assert_fill_factor["EN"](words, len(words) // 2, 0.85)
    assert_fill_factor["EN"](words, len(words) // 4, 0.98)
    assert_fill_factor["EN"](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_he]()
    assert_fill_factor["HE"](words, len(words), 0.63)
    assert_fill_factor["HE"](words, len(words) // 2, 0.86)
    assert_fill_factor["HE"](words, len(words) // 4, 0.98)
    assert_fill_factor["HE"](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_lv]()
    assert_fill_factor["LV"](words, len(words), 0.63)
    assert_fill_factor["LV"](words, len(words) // 2, 0.86)
    assert_fill_factor["LV"](words, len(words) // 4, 0.98)
    assert_fill_factor["LV"](words, len(words) // 13, 0.99)

    words = gen_word_pairs[words_pl]()
    assert_fill_factor["PL"](words, len(words), 0.63)
    assert_fill_factor["PL"](words, len(words) // 2, 0.86)
    assert_fill_factor["PL"](words, len(words) // 4, 0.98)
    assert_fill_factor["PL"](words, len(words) // 13, 1.0)

    words = gen_word_pairs[words_ru]()
    assert_fill_factor["RU"](words, len(words), 0.63)
    assert_fill_factor["RU"](words, len(words) // 2, 0.86)
    assert_fill_factor["RU"](words, len(words) // 4, 0.98)
    assert_fill_factor["RU"](words, len(words) // 13, 1.0)


def test_hash_simd_values():
    fn hash(value: SIMD) -> UInt64:
        hasher = hasher0()
        hasher._update_with_simd(value)
        return hasher^.finish()

    assert_equal(hash(SIMD[DType.float16, 1](1.5)), 17170570249477226196)
    assert_equal(hash(SIMD[DType.float32, 1](1.5)), 13132675331935815936)
    assert_equal(hash(SIMD[DType.float64, 1](1.5)), 5099020174652265565)
    assert_equal(hash(SIMD[DType.float16, 1](1)), 2425465491952508383)
    assert_equal(hash(SIMD[DType.float32, 1](1)), 7268380206556411294)
    assert_equal(hash(SIMD[DType.float64, 1](1)), 1824371972732385641)

    assert_equal(hash(SIMD[DType.int8, 1](1)), 7121024052126637824)
    assert_equal(hash(SIMD[DType.int16, 1](1)), 7121024052126637824)
    assert_equal(hash(SIMD[DType.int32, 1](1)), 7121024052126637824)
    assert_equal(hash(SIMD[DType.int64, 1](1)), 7121024052126637824)
    assert_equal(hash(SIMD[DType.bool, 1](True)), 7121024052126637824)

    assert_equal(hash(SIMD[DType.int8, 1](-1)), 3480139124131340807)
    assert_equal(hash(SIMD[DType.int16, 1](-1)), 3480139124131340807)
    assert_equal(hash(SIMD[DType.int32, 1](-1)), 3480139124131340807)
    assert_equal(hash(SIMD[DType.int64, 1](-1)), 3480139124131340807)

    assert_equal(hash(SIMD[DType.int8, 1](0)), 14824966480498192933)
    assert_equal(hash(SIMD[DType.int8, 2](0)), 4666323910194780317)
    assert_equal(hash(SIMD[DType.int8, 4](0)), 17500360606583578786)
    assert_equal(hash(SIMD[DType.int8, 8](0)), 17090405845262319422)
    assert_equal(hash(SIMD[DType.int8, 16](0)), 14743323736766031385)
    assert_equal(hash(SIMD[DType.int8, 32](0)), 10765559911200264018)
    assert_equal(hash(SIMD[DType.int8, 64](0)), 810077408472869726)

    assert_equal(hash(SIMD[DType.int32, 1](0)), 14824966480498192933)
    assert_equal(hash(SIMD[DType.int32, 2](0)), 4666323910194780317)
    assert_equal(hash(SIMD[DType.int32, 4](0)), 17500360606583578786)
    assert_equal(hash(SIMD[DType.int32, 8](0)), 17090405845262319422)
    assert_equal(hash(SIMD[DType.int32, 16](0)), 14743323736766031385)
    assert_equal(hash(SIMD[DType.int32, 32](0)), 10765559911200264018)
    assert_equal(hash(SIMD[DType.int32, 64](0)), 810077408472869726)


def main():
    test_hash_byte_array()
    test_avalanche()
    test_trailing_zeros()
    test_fill_factor()
    test_hash_simd_values()
