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

from std.hashlib._ahash import AHasher, hash_seeded, hash_seeded_bytes

from std.memory import unsafe_memset_zero
from test_utils import (
    assert_dif_hashes,
    assert_fill_factor,
    dif_bits,
    gen_word_pairs,
    words_ar,
    words_el,
    words_en,
    words_he,
    words_lv,
    words_pl,
    words_ru,
)
from std.testing import assert_equal, assert_not_equal, assert_true, TestSuite

comptime hasher0 = AHasher[SIMD[.uint64, 4](0, 0, 0, 0)]
comptime hasher1 = AHasher[SIMD[.uint64, 4](1, 0, 0, 0)]

comptime seed0 = SIMD[.uint64, 4](0, 0, 0, 0)
comptime seed1 = SIMD[.uint64, 4](1, 0, 0, 0)


def test_hash_byte_array() raises:
    comptime a = StaticString("a")
    comptime b = StaticString("b")
    comptime c = StaticString("c")
    comptime d = StaticString("d")

    assert_equal(hash[hasher0](a), hash[hasher0](a))
    assert_equal(hash[hasher1](a), hash[hasher1](a))
    assert_not_equal(hash[hasher0](a), hash[hasher1](a))
    assert_equal(hash[hasher0](b), hash[hasher0](b))
    assert_equal(hash[hasher1](b), hash[hasher1](b))
    assert_not_equal(hash[hasher0](b), hash[hasher1](b))
    assert_equal(hash[hasher0](c), hash[hasher0](c))
    assert_equal(hash[hasher1](c), hash[hasher1](c))
    assert_not_equal(hash[hasher0](c), hash[hasher1](c))
    assert_equal(hash[hasher0](d), hash[hasher0](d))
    assert_equal(hash[hasher1](d), hash[hasher1](d))
    assert_not_equal(
        hash[hasher0](d),
        hash[hasher1](d),
    )


def test_avalanche() raises:
    # test that values which differ just in one bit,
    # produce significatly different hash values
    var data = Array[UInt8, 256](fill={})
    var hashes0 = List[UInt64]()
    var hashes1 = List[UInt64]()
    hashes0.append(hash[hasher0](data.unsafe_ptr(), 256))
    hashes1.append(hash[hasher1](data.unsafe_ptr(), 256))

    for i in range(256):
        unsafe_memset_zero(data.unsafe_ptr(), 256)
        var v = 1 << (i & 7)
        data[i >> 3] = UInt8(v)
        hashes0.append(hash[hasher0](data.unsafe_ptr(), 256))
        hashes1.append(hash[hasher1](data.unsafe_ptr(), 256))

    for i in range(len(hashes0)):
        var diff = dif_bits(hashes0[i], hashes1[i])
        assert_true(
            diff > 16,
            "Index: {}, diff between: {} and {} is: {}".format(
                i, hashes0[i], hashes1[i], diff
            ),
        )

    assert_dif_hashes(hashes0, 12)
    assert_dif_hashes(hashes1, 12)


def test_trailing_zeros() raises:
    # checks that a value with different amount of trailing zeros,
    # results in significantly different hash values
    var data = Array[UInt8, 8](fill={})
    data[0] = 23
    var hashes0 = List[UInt64]()
    var hashes1 = List[UInt64]()
    for i in range(1, 9):
        hashes0.append(hash[hasher0](data.unsafe_ptr(), i))
        hashes1.append(hash[hasher1](data.unsafe_ptr(), i))

    for i in range(len(hashes0)):
        var diff = dif_bits(hashes0[i], hashes1[i])
        assert_true(
            diff > 18,
            "Index: {}, diff between: {} and {} is: {}".format(
                i, hashes0[i], hashes1[i], diff
            ),
        )

    assert_dif_hashes(hashes0, 18)
    assert_dif_hashes(hashes1, 18)


def test_seeded_zero_matches_unseeded() raises:
    # An all-zero seed must reproduce the unseeded hash exactly, so
    # existing callers that never pass a seed see no behavior change.
    comptime a = StaticString("a")
    var data = Array[UInt8, 256](fill={})

    assert_equal(
        hash_seeded_bytes(data.unsafe_ptr(), 256, seed0),
        hash[hasher0](data.unsafe_ptr(), 256),
    )
    assert_equal(hash_seeded(a, seed0), hash[hasher0](a))


def test_seeded_diffusion() raises:
    # Two distinct seeds hashing identical data must diffuse into
    # significantly different hash values (mirrors test_avalanche).
    var data = Array[UInt8, 256](fill={})
    var hashes0 = List[UInt64]()
    var hashes1 = List[UInt64]()
    hashes0.append(hash_seeded_bytes(data.unsafe_ptr(), 256, seed0))
    hashes1.append(hash_seeded_bytes(data.unsafe_ptr(), 256, seed1))

    for i in range(256):
        unsafe_memset_zero(data.unsafe_ptr(), 256)
        var v = 1 << (i & 7)
        data[i >> 3] = UInt8(v)
        hashes0.append(hash_seeded_bytes(data.unsafe_ptr(), 256, seed0))
        hashes1.append(hash_seeded_bytes(data.unsafe_ptr(), 256, seed1))

    for i in range(len(hashes0)):
        var diff = dif_bits(hashes0[i], hashes1[i])
        assert_true(
            diff > 16,
            "Index: {}, diff between: {} and {} is: {}".format(
                i, hashes0[i], hashes1[i], diff
            ),
        )

    assert_dif_hashes(hashes0, 12)
    assert_dif_hashes(hashes1, 12)


def test_fill_factor() raises:
    var words: List[String] = gen_word_pairs[words_ar]()
    assert_fill_factor["AR", hasher0](words, len(words), 0.63)
    assert_fill_factor["AR", hasher0](words, len(words) // 2, 0.86)
    assert_fill_factor["AR", hasher0](words, len(words) // 4, 0.98)
    assert_fill_factor["AR", hasher0](words, len(words) // 13, 1.0)

    words = gen_word_pairs[words_el]()
    assert_fill_factor["EL", hasher0](words, len(words), 0.63)
    assert_fill_factor["EL", hasher0](words, len(words) // 2, 0.86)
    assert_fill_factor["EL", hasher0](words, len(words) // 4, 0.98)
    assert_fill_factor["EL", hasher0](words, len(words) // 13, 1.0)

    words = gen_word_pairs[words_en]()
    assert_fill_factor["EN", hasher0](words, len(words), 0.63)
    assert_fill_factor["EN", hasher0](words, len(words) // 2, 0.85)
    assert_fill_factor["EN", hasher0](words, len(words) // 4, 0.98)
    assert_fill_factor["EN", hasher0](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_he]()
    assert_fill_factor["HE", hasher0](words, len(words), 0.63)
    assert_fill_factor["HE", hasher0](words, len(words) // 2, 0.86)
    assert_fill_factor["HE", hasher0](words, len(words) // 4, 0.98)
    assert_fill_factor["HE", hasher0](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_lv]()
    assert_fill_factor["LV", hasher0](words, len(words), 0.63)
    assert_fill_factor["LV", hasher0](words, len(words) // 2, 0.86)
    assert_fill_factor["LV", hasher0](words, len(words) // 4, 0.98)
    assert_fill_factor["LV", hasher0](words, len(words) // 13, 0.99)

    words = gen_word_pairs[words_pl]()
    assert_fill_factor["PL", hasher0](words, len(words), 0.63)
    assert_fill_factor["PL", hasher0](words, len(words) // 2, 0.86)
    assert_fill_factor["PL", hasher0](words, len(words) // 4, 0.98)
    assert_fill_factor["PL", hasher0](words, len(words) // 13, 1.0)

    words = gen_word_pairs[words_ru]()
    assert_fill_factor["RU", hasher0](words, len(words), 0.63)
    assert_fill_factor["RU", hasher0](words, len(words) // 2, 0.86)
    assert_fill_factor["RU", hasher0](words, len(words) // 4, 0.98)
    assert_fill_factor["RU", hasher0](words, len(words) // 13, 1.0)


def test_hash_simd_values() raises:
    def hash(value: SIMD) -> UInt64:
        var hasher = hasher0()
        hasher._update_with_simd(value)
        return hasher^.finish()

    assert_equal(hash(Float16(1.5)), 17170570249477226196)
    assert_equal(hash(Float32(1.5)), 13132675331935815936)
    assert_equal(hash(Float64(1.5)), 5099020174652265565)
    assert_equal(hash(Float16(1)), 2425465491952508383)
    assert_equal(hash(Float32(1)), 7268380206556411294)
    assert_equal(hash(Float64(1)), 1824371972732385641)

    assert_equal(hash(Scalar[.bool](True)), 7121024052126637824)
    assert_equal(hash(Int8(1)), 7121024052126637824)
    assert_equal(hash(Int16(1)), 7121024052126637824)
    assert_equal(hash(Int32(1)), 7121024052126637824)
    assert_equal(hash(Int64(1)), 7121024052126637824)
    assert_equal(hash(UInt8(1)), 7121024052126637824)
    assert_equal(hash(Int128(1)), 5122900632109575720)
    assert_equal(hash(SIMD[.int64, 2](1, 0)), 5122900632109575720)
    assert_equal(hash(Int256(1)), 1160009272114074316)
    assert_equal(hash(SIMD[.int64, 4](1, 0, 0, 0)), 1160009272114074316)
    assert_equal(hash(SIMD[.int256, 2](1)), 8329308917989271970)
    assert_equal(
        hash(SIMD[.int64, 8](1, 0, 0, 0, 1, 0, 0, 0)), 8329308917989271970
    )
    assert_equal(hash(SIMD[.uint256, 2](1)), 8329308917989271970)
    assert_equal(
        hash(SIMD[.uint64, 8](1, 0, 0, 0, 1, 0, 0, 0)), 8329308917989271970
    )

    assert_equal(hash(Int8(-1)), 14422269892667380249)
    assert_equal(hash(Int16(-1)), 11448219905088272650)
    assert_equal(hash(Int32(-1)), 3690585083486137738)
    assert_equal(hash(Int64(-1)), 3480139124131340807)
    assert_equal(hash(Int128(-1)), 11199890586389833974)
    assert_equal(hash(SIMD[.int64, 2](-1)), 11199890586389833974)
    assert_equal(hash(Int256(-1)), 17522107111403053621)
    assert_equal(hash(SIMD[.int64, 4](-1)), 17522107111403053621)

    assert_equal(hash(Int8(0)), 14824966480498192933)
    assert_equal(hash(SIMD[.int8, 2](0)), 4666323910194780317)
    assert_equal(hash(SIMD[.int8, 4](0)), 17500360606583578786)
    assert_equal(hash(SIMD[.int8, 8](0)), 17090405845262319422)
    assert_equal(hash(SIMD[.int8, 16](0)), 14743323736766031385)
    assert_equal(hash(SIMD[.int8, 32](0)), 10765559911200264018)
    assert_equal(hash(SIMD[.int8, 64](0)), 810077408472869726)

    assert_equal(hash(Int32(0)), 14824966480498192933)
    assert_equal(hash(SIMD[.int32, 2](0)), 4666323910194780317)
    assert_equal(hash(SIMD[.int32, 4](0)), 17500360606583578786)
    assert_equal(hash(SIMD[.int32, 8](0)), 17090405845262319422)
    assert_equal(hash(SIMD[.int32, 16](0)), 14743323736766031385)
    assert_equal(hash(SIMD[.int32, 32](0)), 10765559911200264018)
    assert_equal(hash(SIMD[.int32, 64](0)), 810077408472869726)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
