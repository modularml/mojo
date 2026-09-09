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

from std.hashlib._fnv1a import Fnv1a

from std.memory import unsafe_memset_zero
from test_utils import (
    assert_dif_hashes,
    assert_fill_factor,
    gen_word_pairs,
    words_ar,
    words_el,
    words_en,
    words_he,
    words_lv,
    words_pl,
    words_ru,
)
from std.testing import assert_equal, TestSuite


def test_hash_byte_array() raises:
    assert_equal(
        hash[Fnv1a]("a"),
        hash[Fnv1a]("a"),
    )
    assert_equal(
        hash[Fnv1a]("b"),
        hash[Fnv1a]("b"),
    )

    assert_equal(
        hash[Fnv1a]("c"),
        hash[Fnv1a]("c"),
    )

    assert_equal(
        hash[Fnv1a]("d"),
        hash[Fnv1a]("d"),
    )
    assert_equal(
        hash[Fnv1a]("d"),
        hash[Fnv1a]("d"),
    )


def test_avalanche() raises:
    # test that values which differ just in one bit,
    # produce significantly different hash values
    var buffer = Array[UInt8, 256](fill=0)
    var hashes = List[UInt64]()
    hashes.append(hash[Fnv1a](buffer.unsafe_ptr(), 256))

    for i in range(256):
        unsafe_memset_zero(buffer.unsafe_ptr(), 256)
        var v = 1 << (i & 7)
        buffer[i >> 3] = UInt8(v)
        hashes.append(hash[Fnv1a](buffer.unsafe_ptr(), 256))

    assert_dif_hashes(hashes, 15)


def test_trailing_zeros() raises:
    # checks that a value with different amount of trailing zeros,
    # results in significantly different hash values
    var buffer = Array[UInt8, 8](fill=0)
    buffer[0] = 23
    var hashes = List[UInt64]()
    for i in range(1, 9):
        hashes.append(hash[Fnv1a](buffer.unsafe_ptr(), i))

    assert_dif_hashes(hashes, 21)


def test_fill_factor() raises:
    var words = gen_word_pairs[words_ar]()
    assert_fill_factor["AR", Fnv1a](words, len(words), 0.63)
    assert_fill_factor["AR", Fnv1a](words, len(words) // 2, 0.86)
    assert_fill_factor["AR", Fnv1a](words, len(words) // 4, 0.98)
    assert_fill_factor["AR", Fnv1a](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_el]()
    assert_fill_factor["EL", Fnv1a](words, len(words), 0.63)
    assert_fill_factor["EL", Fnv1a](words, len(words) // 2, 0.86)
    assert_fill_factor["EL", Fnv1a](words, len(words) // 4, 0.98)
    assert_fill_factor["EL", Fnv1a](words, len(words) // 13, 1.0)

    words = gen_word_pairs[words_en]()
    assert_fill_factor["EN", Fnv1a](words, len(words), 0.63)
    assert_fill_factor["EN", Fnv1a](words, len(words) // 2, 0.86)
    assert_fill_factor["EN", Fnv1a](words, len(words) // 4, 0.98)
    assert_fill_factor["EN", Fnv1a](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_he]()
    assert_fill_factor["HE", Fnv1a](words, len(words), 0.63)
    assert_fill_factor["HE", Fnv1a](words, len(words) // 2, 0.86)
    assert_fill_factor["HE", Fnv1a](words, len(words) // 4, 0.98)
    assert_fill_factor["HE", Fnv1a](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_lv]()
    assert_fill_factor["LV", Fnv1a](words, len(words), 0.63)
    assert_fill_factor["LV", Fnv1a](words, len(words) // 2, 0.86)
    assert_fill_factor["LV", Fnv1a](words, len(words) // 4, 0.98)
    assert_fill_factor["LV", Fnv1a](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_pl]()
    assert_fill_factor["PL", Fnv1a](words, len(words), 0.63)
    assert_fill_factor["PL", Fnv1a](words, len(words) // 2, 0.86)
    assert_fill_factor["PL", Fnv1a](words, len(words) // 4, 0.98)
    assert_fill_factor["PL", Fnv1a](words, len(words) // 14, 1.0)

    words = gen_word_pairs[words_ru]()
    assert_fill_factor["RU", Fnv1a](words, len(words), 0.63)
    assert_fill_factor["RU", Fnv1a](words, len(words) // 2, 0.86)
    assert_fill_factor["RU", Fnv1a](words, len(words) // 4, 0.98)
    assert_fill_factor["RU", Fnv1a](words, len(words) // 14, 1.0)


def test_hash_simd_values() raises:
    def hash(value: SIMD) -> UInt64:
        var hasher = Fnv1a()
        hasher._update_with_simd(value)
        return hasher^.finish()

    assert_equal(hash(Float16(1.5)), 12636464265834235359)
    assert_equal(hash(Float32(1.5)), 8026467504136239071)
    assert_equal(hash(Float64(1.5)), 15000291120250992607)
    assert_equal(hash(Float16(1)), 12637027215787879391)
    assert_equal(hash(Float32(1)), 3414781483884328927)
    assert_equal(hash(Float64(1)), 14020758201297909727)

    assert_equal(hash(Scalar[.bool](True)), 12638152016183539244)
    assert_equal(hash(Int8(1)), 12638152016183539244)
    assert_equal(hash(Int16(1)), 12638152016183539244)
    assert_equal(hash(Int32(1)), 12638152016183539244)
    assert_equal(hash(Int64(1)), 12638152016183539244)
    assert_equal(hash(Int128(1)), 589727492704079044)
    assert_equal(hash(SIMD[.int64, 2](1, 0)), 589727492704079044)
    assert_equal(hash(Int256(1)), 12478008331234465636)
    assert_equal(hash(SIMD[.int64, 4](1, 0, 0, 0)), 12478008331234465636)

    assert_equal(hash(Int8(-1)), 12638352127299873646)
    assert_equal(hash(Int16(-1)), 12690424998011946606)
    assert_equal(hash(Int32(-1)), 7718450949043144302)
    assert_equal(hash(Int64(-1)), 5808589858502755950)
    assert_equal(hash(Int128(-1)), 591639543425159523)
    assert_equal(hash(SIMD[.int64, 2](-1)), 591639543425159523)
    assert_equal(hash(Int256(-1)), 16463485165778154321)
    assert_equal(hash(SIMD[.int64, 4](-1)), 16463485165778154321)

    assert_equal(hash(Int8(0)), 12638153115695167455)
    assert_equal(hash(SIMD[.int8, 2](0)), 590684067820433389)
    assert_equal(hash(SIMD[.int8, 4](0)), 5558979605539197941)
    assert_equal(hash(SIMD[.int8, 8](0)), 12161962213042174405)
    assert_equal(hash(SIMD[.int8, 16](0)), 9808874869469701221)
    assert_equal(hash(SIMD[.int8, 32](0)), 901300984310592933)
    assert_equal(hash(SIMD[.int8, 64](0)), 13380826962402805797)

    assert_equal(hash(Int32(0)), 12638153115695167455)
    assert_equal(hash(SIMD[.int32, 2](0)), 590684067820433389)
    assert_equal(hash(SIMD[.int32, 4](0)), 5558979605539197941)
    assert_equal(hash(SIMD[.int32, 8](0)), 12161962213042174405)
    assert_equal(hash(SIMD[.int32, 16](0)), 9808874869469701221)
    assert_equal(hash(SIMD[.int32, 32](0)), 901300984310592933)
    assert_equal(hash(SIMD[.int32, 64](0)), 13380826962402805797)


def test_hash_at_compile_time() raises:
    comptime h = hash[Fnv1a]("hello")
    assert_equal(h, 11831194018420276491)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
