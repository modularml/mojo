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
# RUN: %mojo -debug-level full %s


from testing import assert_equal, assert_false, assert_raises, assert_true

from time import time

from datetime.fast import DateTime64, DateTime32, DateTime16, DateTime8
from datetime.dt_str import IsoFormat


fn test_add64() raises:
    # test february leapyear
    var result = DateTime64(2024, 3, 1) + DateTime64(0, 0, 1)
    var offset_0 = DateTime64(2024, 2, 29)
    var offset_p_1 = DateTime64(2024, 2, 29)
    var offset_n_1 = DateTime64(2024, 3, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    var add_seconds = DateTime64(2024, 3, 1).add(seconds=24 * 3600)
    assert_true(result.m_second == add_seconds.m_second)

    # test february not leapyear
    result = DateTime64(2023, 3, 1) + DateTime64(0, 0, 1)
    offset_0 = DateTime64(2023, 2, 28)
    offset_p_1 = DateTime64(2023, 2, 28)
    offset_n_1 = DateTime64(2023, 3, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime64(2023, 3, 1).add(seconds=24 * 3600)
    assert_true(result.m_second == add_seconds.m_second)

    # test normal month
    result = DateTime64(2024, 5, 31) + DateTime64(0, 0, 1)
    offset_0 = DateTime64(2024, 6, 1)
    offset_p_1 = DateTime64(2024, 6, 1)
    offset_n_1 = DateTime64(2024, 5, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime64(2024, 5, 31).add(seconds=24 * 3600)
    assert_true(result.m_second == add_seconds.m_second)

    # test december
    result = DateTime64(2024, 12, 31) + DateTime64(0, 0, 1)
    offset_0 = DateTime64(2025, 1, 1)
    offset_p_1 = DateTime64(2025, 1, 1)
    offset_n_1 = DateTime64(2024, 12, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime64(2024, 12, 31).add(seconds=24 * 3600)
    assert_true(result.m_second == add_seconds.m_second)

    # test year and month add
    result = DateTime64(2022, 6, 1) + DateTime64(3, 6, 31)
    offset_0 = DateTime64(2025, 1, 1)
    offset_p_1 = DateTime64(2025, 1, 1)
    offset_n_1 = DateTime64(2024, 12, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)

    # test positive overflow pycal
    result = DateTime64(9999, 12, 31) + DateTime64(0, 0, 1)
    offset_0 = DateTime64(1, 1, 1)
    offset_p_1 = DateTime64(1, 1, 1)
    offset_n_1 = DateTime64(1, 1, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime64(9999, 12, 31).add(seconds=24 * 3600)
    assert_true(result.m_second == add_seconds.m_second)

    # test positive overflow unixcal
    result = DateTime64(9999, 12, 31) + DateTime64(0, 0, 1)
    offset_0 = DateTime64(1970, 1, 1)
    offset_p_1 = DateTime64(1970, 1, 1)
    offset_n_1 = DateTime64(1970, 1, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime64(9999, 12, 31).add(seconds=24 * 3600)
    assert_true(result.m_second == add_seconds.m_second)


fn test_add32() raises:
    # test february leapyear
    var result = DateTime32(2024, 3, 1) + DateTime32(0, 0, 1)
    var offset_0 = DateTime32(2024, 2, 29)
    var offset_p_1 = DateTime32(2024, 2, 29)
    var offset_n_1 = DateTime32(2024, 3, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    var add_seconds = DateTime32(2024, 3, 1).add(seconds=24 * 3600)
    assert_true(result.minute == add_seconds.minute)

    # test february not leapyear
    result = DateTime32(2023, 3, 1) + DateTime32(0, 0, 1)
    offset_0 = DateTime32(2023, 2, 28)
    offset_p_1 = DateTime32(2023, 2, 28)
    offset_n_1 = DateTime32(2023, 3, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime32(2023, 3, 1).add(seconds=24 * 3600)
    assert_true(result.minute == add_seconds.minute)

    # test normal month
    result = DateTime32(2024, 5, 31) + DateTime32(0, 0, 1)
    offset_0 = DateTime32(2024, 6, 1)
    offset_p_1 = DateTime32(2024, 6, 1)
    offset_n_1 = DateTime32(2024, 5, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime32(2024, 5, 31).add(seconds=24 * 3600)
    assert_true(result.minute == add_seconds.minute)

    # test december
    result = DateTime32(2024, 12, 31) + DateTime32(0, 0, 1)
    offset_0 = DateTime32(2025, 1, 1)
    offset_p_1 = DateTime32(2025, 1, 1)
    offset_n_1 = DateTime32(2024, 12, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime32(2024, 12, 31).add(seconds=24 * 3600)
    assert_true(result.minute == add_seconds.minute)

    # test year and month add
    result = DateTime32(2022, 6, 1) + DateTime32(3, 6, 31)
    offset_0 = DateTime32(2025, 1, 1)
    offset_p_1 = DateTime32(2025, 1, 1)
    offset_n_1 = DateTime32(2024, 12, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)

    # test positive overflow pycal
    result = DateTime32(9999, 12, 31) + DateTime32(0, 0, 1)
    offset_0 = DateTime32(1, 1, 1)
    offset_p_1 = DateTime32(1, 1, 1)
    offset_n_1 = DateTime32(1, 1, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime32(9999, 12, 31).add(seconds=24 * 3600)
    assert_true(result.minute == add_seconds.minute)

    # test positive overflow unixcal
    result = DateTime32(9999, 12, 31) + DateTime32(0, 0, 1)
    offset_0 = DateTime32(1970, 1, 1)
    offset_p_1 = DateTime32(1970, 1, 1)
    offset_n_1 = DateTime32(1970, 1, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    add_seconds = DateTime32(9999, 12, 31).add(seconds=24 * 3600)
    assert_true(result.minute == add_seconds.minute)


fn test_subtract64() raises:
    # test february leapyear
    var result = DateTime64(2024, 3, 1) - DateTime64(0, 0, 1)
    var offset_0 = DateTime64(2024, 2, 29)
    var offset_p_1 = DateTime64(2024, 2, 29)
    var offset_n_1 = DateTime64(2024, 3, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    var sub_seconds = DateTime64(2024, 3, 1).subtract(seconds=1)
    assert_true(result.m_second == sub_seconds.m_second)

    # test february not leapyear
    result = DateTime64(2023, 3, 1) - DateTime64(0, 0, 1)
    offset_0 = DateTime64(2023, 2, 28)
    offset_p_1 = DateTime64(2023, 2, 28)
    offset_n_1 = DateTime64(2023, 3, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime64(2023, 3, 1).subtract(seconds=1)
    assert_true(result.m_second == sub_seconds.m_second)

    # test normal month
    result = DateTime64(2024, 6, 1) - DateTime64(0, 0, 1)
    offset_0 = DateTime64(2024, 5, 31)
    offset_p_1 = DateTime64(2024, 5, 31)
    offset_n_1 = DateTime64(2024, 6, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime64(2024, 6, 1).subtract(seconds=1)
    assert_true(result.m_second == sub_seconds.m_second)

    # test december
    result = DateTime64(2025, 1, 1) - DateTime64(0, 0, 1)
    offset_0 = DateTime64(2024, 12, 31)
    offset_p_1 = DateTime64(2024, 12, 31)
    offset_n_1 = DateTime64(2025, 1, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime64(2025, 1, 1).subtract(seconds=1)
    assert_true(result.m_second == sub_seconds.m_second)

    # test year and month subtract
    result = DateTime64(2025, 1, 1) - DateTime64(3, 6, 31)
    offset_0 = DateTime64(2022, 6, 1)
    offset_p_1 = DateTime64(2022, 6, 1)
    offset_n_1 = DateTime64(2022, 5, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)

    # test negative overflow pycal
    result = DateTime64(1, 1, 1) - DateTime64(0, 0, 1)
    offset_0 = DateTime64(9999, 12, 31)
    offset_p_1 = DateTime64(9999, 12, 31)
    offset_n_1 = DateTime64(9999, 12, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime64(1, 1, 1).subtract(seconds=1)
    assert_true(result.m_second == sub_seconds.m_second)

    # test negative overflow unixcal
    result = DateTime64(1970, 1, 1) - DateTime64(0, 0, 1)
    offset_0 = DateTime64(9999, 12, 31)
    offset_p_1 = DateTime64(9999, 12, 31)
    offset_n_1 = DateTime64(9999, 12, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime64(1970, 1, 1).subtract(seconds=1)
    assert_true(result.m_second == sub_seconds.m_second)


fn test_subtract32() raises:
    # test february leapyear
    var result = DateTime32(2024, 3, 1) - DateTime32(0, 0, 1)
    var offset_0 = DateTime32(2024, 2, 29)
    var offset_p_1 = DateTime32(2024, 2, 29)
    var offset_n_1 = DateTime32(2024, 3, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    var sub_seconds = DateTime32(2024, 3, 1).subtract(seconds=1)
    assert_true(result.minute == sub_seconds.minute)

    # test february not leapyear
    result = DateTime32(2023, 3, 1) - DateTime32(0, 0, 1)
    offset_0 = DateTime32(2023, 2, 28)
    offset_p_1 = DateTime32(2023, 2, 28)
    offset_n_1 = DateTime32(2023, 3, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime32(2023, 3, 1).subtract(seconds=1)
    assert_true(result.minute == sub_seconds.minute)

    # test normal month
    result = DateTime32(2024, 6, 1) - DateTime32(0, 0, 1)
    offset_0 = DateTime32(2024, 5, 31)
    offset_p_1 = DateTime32(2024, 5, 31)
    offset_n_1 = DateTime32(2024, 6, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime32(2024, 6, 1).subtract(seconds=1)
    assert_true(result.minute == sub_seconds.minute)

    # test december
    result = DateTime32(2025, 1, 1) - DateTime32(0, 0, 1)
    offset_0 = DateTime32(2024, 12, 31)
    offset_p_1 = DateTime32(2024, 12, 31)
    offset_n_1 = DateTime32(2025, 1, 1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime32(2025, 1, 1).subtract(seconds=1)
    assert_true(result.minute == sub_seconds.minute)

    # test year and month subtract
    result = DateTime32(2025, 1, 1) - DateTime32(3, 6, 31)
    offset_0 = DateTime32(2022, 6, 1)
    offset_p_1 = DateTime32(2022, 6, 1)
    offset_n_1 = DateTime32(2022, 5, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)

    # test negative overflow pycal
    result = DateTime32(1, 1, 1) - DateTime32(0, 0, 1)
    offset_0 = DateTime32(9999, 12, 31)
    offset_p_1 = DateTime32(9999, 12, 31)
    offset_n_1 = DateTime32(9999, 12, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime32(1, 1, 1).subtract(seconds=1)
    assert_true(result.minute == sub_seconds.minute)

    # test negative overflow unixcal
    result = DateTime32(1970, 1, 1) - DateTime32(0, 0, 1)
    offset_0 = DateTime32(9999, 12, 31)
    offset_p_1 = DateTime32(9999, 12, 31)
    offset_n_1 = DateTime32(9999, 12, 31)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)
    sub_seconds = DateTime32(1970, 1, 1).subtract(seconds=1)
    assert_true(result.minute == sub_seconds.minute)


fn test_logic64() raises:
    var ref = DateTime64(1970, 1, 1)
    assert_true(ref == DateTime64(1970, 1, 1))
    assert_true(ref == DateTime64(1970, 1, 1))
    assert_true(ref == DateTime64(1969, 12, 31))

    assert_true(ref < DateTime64(1970, 1, 2))
    assert_true(ref <= DateTime64(1970, 1, 2))
    assert_true(ref > DateTime64(1969, 12, 31))
    assert_true(ref >= DateTime64(1969, 12, 31))


fn test_logic32() raises:
    var ref = DateTime32(1970, 1, 1)
    assert_true(ref == DateTime32(1970, 1, 1))
    assert_true(ref == DateTime32(1970, 1, 1))
    assert_true(ref == DateTime32(1969, 12, 31))

    assert_true(ref < DateTime32(1970, 1, 2))
    assert_true(ref <= DateTime32(1970, 1, 2))
    assert_true(ref > DateTime32(1969, 12, 31))
    assert_true(ref >= DateTime32(1969, 12, 31))


fn test_logic16() raises:
    var ref = DateTime16(1970, 1, 1)
    assert_true(ref == DateTime16(1970, 1, 1))
    assert_true(ref == DateTime16(1970, 1, 1))
    assert_true(ref == DateTime16(1969, 12, 31))

    assert_true(ref < DateTime16(1970, 1, 2))
    assert_true(ref <= DateTime16(1970, 1, 2))
    assert_true(ref > DateTime16(1969, 12, 31))
    assert_true(ref >= DateTime16(1969, 12, 31))


fn test_logic8() raises:
    var ref = DateTime8(1970, 1, 1)
    assert_true(ref == DateTime8(1970, 1, 1))
    assert_true(ref == DateTime8(1970, 1, 1))
    assert_true(ref == DateTime8(1969, 12, 31))

    assert_true(ref < DateTime8(1970, 1, 2))
    assert_true(ref <= DateTime8(1970, 1, 2))
    assert_true(ref > DateTime8(1969, 12, 31))
    assert_true(ref >= DateTime8(1969, 12, 31))


fn test_bitwise64() raises:
    var ref = DateTime64(1970, 1, 1)
    assert_true(ref & DateTime64(1970, 1, 1) == 0)
    assert_true(ref & DateTime64(1970, 1, 1) == 0)
    assert_true(ref & DateTime64(1969, 12, 31) == 0)

    assert_true(ref ^ DateTime64(1970, 1, 2) != 0)
    assert_true(ref | (DateTime64(1970, 1, 2) & 0) == ref)
    assert_true(ref & ~ref == 0)
    assert_true(ref ^ ~ref == UInt64.MAX_FINITE)


fn test_bitwise32() raises:
    var ref = DateTime32(1970, 1, 1)
    assert_true(ref & DateTime32(1970, 1, 1) == 0)
    assert_true(ref & DateTime32(1970, 1, 1) == 0)
    assert_true(ref & DateTime32(1969, 12, 31) == 0)

    assert_true(ref ^ DateTime32(1970, 1, 2) != 0)
    assert_true(ref | (DateTime32(1970, 1, 2) & 0) == ref)
    assert_true(ref & ~ref == 0)
    assert_true(ref ^ ~ref == UInt32.MAX_FINITE)


fn test_bitwise16() raises:
    var ref = DateTime16(1970, 1, 1)
    assert_true(ref & DateTime16(1970, 1, 1) == 0)
    assert_true(ref & DateTime16(1970, 1, 1) == 0)
    assert_true(ref & DateTime16(1969, 12, 31) == 0)

    assert_true(ref ^ DateTime16(1970, 1, 2) != 0)
    assert_true(ref | (DateTime16(1970, 1, 2) & 0) == ref)
    assert_true(ref & ~ref == 0)
    assert_true(ref ^ ~ref == UInt16.MAX_FINITE)


fn test_bitwise8() raises:
    var ref = DateTime8(1970, 1, 1)
    assert_true(ref & DateTime8(1970, 1, 1) == 0)
    assert_true(ref & DateTime8(1970, 1, 1) == 0)
    assert_true(ref & DateTime8(1969, 12, 31) == 0)

    assert_true(ref ^ DateTime8(1970, 1, 2) != 0)
    assert_true(ref | (DateTime8(1970, 1, 2) & 0) == ref)
    assert_true(ref & ~ref == 0)
    assert_true(ref ^ ~ref == UInt32.MAX_FINITE)


fn test_iso64() raises:
    var ref = DateTime64(1970, 1, 1)
    var iso_str = "1970-01-01T00:00:00+00:00"
    alias fmt1 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS_TZD)
    assert_equal(ref, DateTime64.from_iso[fmt1]())
    assert_equal(iso_str, ref.to_iso[fmt1]())

    iso_str = "1970-01-01 00:00:00+00:00"
    alias fmt2 = IsoFormat(IsoFormat.YYYY_MM_DD___HH_MM_SS_TZD)
    assert_equal(ref, DateTime64.from_iso[fmt2]())
    assert_equal(iso_str, ref.to_iso[fmt2]())

    iso_str = "1970-01-01T00:00:00"
    alias fmt3 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS)
    assert_equal(ref, DateTime64.from_iso[fmt3]())
    assert_equal(iso_str, ref.to_iso[fmt3]())

    iso_str = "19700101T000000"
    alias fmt4 = IsoFormat(IsoFormat.YYYYMMDDHHMMSS)
    assert_equal(ref, Date.from_iso[fmt4]())
    assert_equal(iso_str, ref.to_iso[fmt4]())

    iso_str = "00:00:00"
    alias fmt5 = IsoFormat(IsoFormat.HH_MM_SS)
    assert_equal(ref, DateTime64.from_iso[fmt5]())
    assert_equal(iso_str, ref.to_iso[fmt5]())

    iso_str = "000000"
    alias fmt6 = IsoFormat(IsoFormat.HHMMSS)
    assert_equal(ref, DateTime64.from_iso[fmt6]())
    assert_equal(iso_str, ref.to_iso[fmt6]())


fn test_iso32() raises:
    var ref = DateTime32(1970, 1, 1)
    var iso_str = "1970-01-01T00:00:00+00:00"
    alias fmt1 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS_TZD)
    assert_equal(ref, DateTime32.from_iso[fmt1]())
    assert_equal(iso_str, ref.to_iso[fmt1]())

    iso_str = "1970-01-01 00:00:00+00:00"
    alias fmt2 = IsoFormat(IsoFormat.YYYY_MM_DD___HH_MM_SS_TZD)
    assert_equal(ref, DateTime32.from_iso[fmt2]())
    assert_equal(iso_str, ref.to_iso[fmt2]())

    iso_str = "1970-01-01T00:00:00"
    alias fmt3 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS)
    assert_equal(ref, DateTime32.from_iso[fmt3]())
    assert_equal(iso_str, ref.to_iso[fmt3]())

    iso_str = "19700101T000000"
    alias fmt4 = IsoFormat(IsoFormat.YYYYMMDDHHMMSS)
    assert_equal(ref, Date.from_iso[fmt4]())
    assert_equal(iso_str, ref.to_iso[fmt4]())

    iso_str = "00:00:00"
    alias fmt5 = IsoFormat(IsoFormat.HH_MM_SS)
    assert_equal(ref, DateTime32.from_iso[fmt5]())
    assert_equal(iso_str, ref.to_iso[fmt5]())

    iso_str = "000000"
    alias fmt6 = IsoFormat(IsoFormat.HHMMSS)
    assert_equal(ref, DateTime32.from_iso[fmt6]())
    assert_equal(iso_str, ref.to_iso[fmt6]())


fn test_iso16() raises:
    var ref = DateTime16(1970, 1, 1)
    var iso_str = "1970-01-01T00:00:00+00:00"
    alias fmt1 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS_TZD)
    assert_equal(ref, DateTime16.from_iso[fmt1]())
    assert_equal(iso_str, ref.to_iso[fmt1]())

    iso_str = "1970-01-01 00:00:00+00:00"
    alias fmt2 = IsoFormat(IsoFormat.YYYY_MM_DD___HH_MM_SS_TZD)
    assert_equal(ref, DateTime16.from_iso[fmt2]())
    assert_equal(iso_str, ref.to_iso[fmt2]())

    iso_str = "1970-01-01T00:00:00"
    alias fmt3 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS)
    assert_equal(ref, DateTime16.from_iso[fmt3]())
    assert_equal(iso_str, ref.to_iso[fmt3]())

    iso_str = "19700101T000000"
    alias fmt4 = IsoFormat(IsoFormat.YYYYMMDDHHMMSS)
    assert_equal(ref, Date.from_iso[fmt4]())
    assert_equal(iso_str, ref.to_iso[fmt4]())

    iso_str = "00:00:00"
    alias fmt5 = IsoFormat(IsoFormat.HH_MM_SS)
    assert_equal(ref, DateTime16.from_iso[fmt5]())
    assert_equal(iso_str, ref.to_iso[fmt5]())

    iso_str = "000000"
    alias fmt6 = IsoFormat(IsoFormat.HHMMSS)
    assert_equal(ref, DateTime16.from_iso[fmt6]())
    assert_equal(iso_str, ref.to_iso[fmt6]())


fn test_iso8() raises:
    var ref = DateTime8(1970, 1, 1)
    var iso_str = "1970-01-01T00:00:00+00:00"
    alias fmt1 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS_TZD)
    assert_equal(ref, DateTime8.from_iso[fmt1]())
    assert_equal(iso_str, ref.to_iso[fmt1]())

    iso_str = "1970-01-01 00:00:00+00:00"
    alias fmt2 = IsoFormat(IsoFormat.YYYY_MM_DD___HH_MM_SS_TZD)
    assert_equal(ref, DateTime8.from_iso[fmt2]())
    assert_equal(iso_str, ref.to_iso[fmt2]())

    iso_str = "1970-01-01T00:00:00"
    alias fmt3 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS)
    assert_equal(ref, DateTime8.from_iso[fmt3]())
    assert_equal(iso_str, ref.to_iso[fmt3]())

    iso_str = "19700101T000000"
    alias fmt4 = IsoFormat(IsoFormat.YYYYMMDDHHMMSS)
    assert_equal(ref, Date.from_iso[fmt4]())
    assert_equal(iso_str, ref.to_iso[fmt4]())

    iso_str = "00:00:00"
    alias fmt5 = IsoFormat(IsoFormat.HH_MM_SS)
    assert_equal(ref, DateTime8.from_iso[fmt5]())
    assert_equal(iso_str, ref.to_iso[fmt5]())

    iso_str = "000000"
    alias fmt6 = IsoFormat(IsoFormat.HHMMSS)
    assert_equal(ref, DateTime8.from_iso[fmt6]())
    assert_equal(iso_str, ref.to_iso[fmt6]())


fn test_time64() raises:
    var start = DateTime64.now()
    time.sleep(1e-3)  # milisecond resolution
    var end = DateTime64.now()
    assert_true(start != end)


fn test_hash64() raises:
    var ref = DateTime64(1970, 1, 1)
    var data = hash(ref)
    var parsed = DateTime64.from_hash(data)
    assert_true(ref == parsed)


fn test_hash32() raises:
    var ref = DateTime32(1970, 1, 1)
    var data = hash(ref)
    var parsed = DateTime32.from_hash(data)
    assert_true(ref == parsed)


fn test_hash16() raises:
    var ref = DateTime16(1970, 1, 1)
    var data = hash(ref)
    var parsed = DateTime16.from_hash(data)
    assert_true(ref == parsed)


fn test_hash8() raises:
    var ref = DateTime8(0, 23)
    var data = hash(ref)
    var parsed = DateTime8.from_hash(data)
    assert_true(ref == parsed)


fn main() raises:
    test_add64()
    test_subtract64()
    test_logic64()
    test_bitwise64()
    test_iso64()
    test_hash64()
    test_time64()
    test_add32()
    test_subtract32()
    test_logic32()
    test_bitwise32()
    test_iso32()
    test_hash32()
    test_logic16()
    test_bitwise16()
    test_iso16()
    test_hash16()
    test_logic8()
    test_bitwise8()
    test_iso8()
    test_hash8()
