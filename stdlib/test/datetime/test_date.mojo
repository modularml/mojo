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


from testing import assert_equal, assert_false, assert_raises, assert_true

from time import time

from datetime.date import Date as dt
from datetime.calendar import Calendar, PythonCalendar, UTCCalendar
from datetime.timezone import TimeZone
from datetime.dt_str import IsoFormat

alias TZ = TimeZone[iana=None, pyzoneinfo=False, native=False]
alias Date = dt[iana=None, pyzoneinfo=False, native=False]


fn test_add(pycal: Calendar, unixcal: Calendar, tz1_: TZ, tz_0_: TZ, tz_1: TZ):
    # test february leapyear
    var result = Date(2024, 3, 1, tz_0_, pycal) + Date(0, 0, 1, tz_0_, unixcal)
    var offset_0 = Date(2024, 2, 29, tz_0_, unixcal)
    var offset_p_1 = Date(2024, 2, 29, tz_1, unixcal)
    var offset_n_1 = Date(2024, 3, 1, tz1_, unixcal)
    var add_seconds = Date(2024, 3, 1, tz_0_, unixcal).add(seconds=24 * 3600)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test february not leapyear
    result = Date(2023, 3, 1, tz_0_, pycal) + Date(0, 0, 1, tz_0_, unixcal)
    offset_0 = Date(2023, 2, 28, tz_0_, unixcal)
    offset_p_1 = Date(2023, 2, 28, tz_1, unixcal)
    offset_n_1 = Date(2023, 3, 1, tz1_, unixcal)
    add_seconds = Date(2023, 3, 1, tz_0_, unixcal).add(seconds=24 * 3600)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test normal month
    result = Date(2024, 5, 31, tz_0_, pycal) + Date(0, 0, 1, tz_0_, unixcal)
    offset_0 = Date(2024, 6, 1, tz_0_, unixcal)
    offset_p_1 = Date(2024, 6, 1, tz_1, unixcal)
    offset_n_1 = Date(2024, 5, 31, tz1_, unixcal)
    add_seconds = Date(2024, 5, 31, tz_0_, unixcal).add(seconds=24 * 3600)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test december
    result = Date(2024, 12, 31, tz_0_, pycal) + Date(0, 0, 1, tz_0_, unixcal)
    offset_0 = Date(2025, 1, 1, tz_0_, unixcal)
    offset_p_1 = Date(2025, 1, 1, tz_1, unixcal)
    offset_n_1 = Date(2024, 12, 31, tz1_, unixcal)
    add_seconds = Date(2024, 12, 31, tz_0_, unixcal).add(seconds=24 * 3600)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test year and month add
    result = Date(2022, 6, 1, tz_0_, pycal) + Date(3, 6, 31, tz_0_, unixcal)
    offset_0 = Date(2025, 1, 1, tz_0_, unixcal)
    offset_p_1 = Date(2025, 1, 1, tz_1, unixcal)
    offset_n_1 = Date(2024, 12, 31, tz1_, unixcal)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)

    # test positive overflow pycal
    result = Date(9999, 12, 31, tz_0_, pycal) + Date(0, 0, 1, tz_0_, pycal)
    offset_0 = Date(1, 1, 1, tz_0_, pycal)
    offset_p_1 = Date(1, 1, 1, tz_1, pycal)
    offset_n_1 = Date(1, 1, 1, tz1_, pycal)
    add_seconds = Date(9999, 12, 31, tz_0_, pycal).add(seconds=24 * 3600)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test positive overflow unixcal
    result = Date(9999, 12, 31, tz_0_, unixcal) + Date(0, 0, 1, tz_0_, unixcal)
    offset_0 = Date(1970, 1, 1, tz_0_, unixcal)
    offset_p_1 = Date(1970, 1, 1, tz_1, unixcal)
    offset_n_1 = Date(1970, 1, 1, tz1_, unixcal)
    add_seconds = Date(9999, 12, 31, tz_0_, unixcal).add(seconds=24 * 3600)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)


fn test_subtract(
    pycal: Calendar, unixcal: Calendar, tz1_: TZ, tz_0_: TZ, tz_1: TZ
):
    # test february leapyear
    var result = Date(2024, 3, 1, tz_0_, pycal) - Date(0, 0, 1, tz_0_, unixcal)
    var offset_0 = Date(2024, 2, 29, tz_0_, unixcal)
    var offset_p_1 = Date(2024, 2, 29, tz_1, unixcal)
    var offset_n_1 = Date(2024, 3, 1, tz1_, unixcal)
    var sub_seconds = Date(2024, 3, 1, tz_0_, unixcal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test february not leapyear
    result = Date(2023, 3, 1, tz_0_, pycal) - Date(0, 0, 1, tz_0_, unixcal)
    offset_0 = Date(2023, 2, 28, tz_0_, unixcal)
    offset_p_1 = Date(2023, 2, 28, tz_1, unixcal)
    offset_n_1 = Date(2023, 3, 1, tz1_, unixcal)
    sub_seconds = Date(2023, 3, 1, tz_0_, unixcal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test normal month
    result = Date(2024, 6, 1, tz_0_, pycal) - Date(0, 0, 1, tz_0_, unixcal)
    offset_0 = Date(2024, 5, 31, tz_0_, unixcal)
    offset_p_1 = Date(2024, 5, 31, tz_1, unixcal)
    offset_n_1 = Date(2024, 6, 1, tz1_, unixcal)
    sub_seconds = Date(2024, 6, 1, tz_0_, unixcal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test december
    result = Date(2025, 1, 1, tz_0_, pycal) - Date(0, 0, 1, tz_0_, unixcal)
    offset_0 = Date(2024, 12, 31, tz_0_, unixcal)
    offset_p_1 = Date(2024, 12, 31, tz_1, unixcal)
    offset_n_1 = Date(2025, 1, 1, tz1_, unixcal)
    sub_seconds = Date(2025, 1, 1, tz_0_, unixcal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test year and month subtract
    result = Date(2025, 1, 1, tz_0_, pycal) - Date(3, 6, 31, tz_0_, unixcal)
    offset_0 = Date(2022, 6, 1, tz_0_, unixcal)
    offset_p_1 = Date(2022, 6, 1, tz_1, unixcal)
    offset_n_1 = Date(2022, 5, 31, tz1_, unixcal)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)

    # test negative overflow pycal
    result = Date(1, 1, 1, tz_0_, pycal) - Date(0, 0, 1, tz_0_, pycal)
    offset_0 = Date(9999, 12, 31, tz_0_, pycal)
    offset_p_1 = Date(9999, 12, 31, tz_1, pycal)
    offset_n_1 = Date(9999, 12, 31, tz1_, pycal)
    sub_seconds = Date(1, 1, 1, tz_0_, pycal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test negative overflow unixcal
    result = Date(1970, 1, 1, tz_0_, unixcal) - Date(0, 0, 1, tz_0_, unixcal)
    offset_0 = Date(9999, 12, 31, tz_0_, unixcal)
    offset_p_1 = Date(9999, 12, 31, tz_1, unixcal)
    offset_n_1 = Date(9999, 12, 31, tz1_, unixcal)
    sub_seconds = Date(1970, 1, 1, tz_0_, unixcal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)


fn test_logic(
    pycal: Calendar, unixcal: Calendar, tz1_: TZ, tz_0_: TZ, tz_1: TZ
):
    var ref = Date(1970, 1, 1, tz_0_, pycal)
    assert_true(ref == Date(1970, 1, 1, tz_0_, unixcal))
    assert_true(ref == Date(1970, 1, 1, tz_1, unixcal))
    assert_true(ref == Date(1969, 12, 31, tz1_, pycal))

    assert_true(ref < Date(1970, 1, 2, tz_0_, pycal))
    assert_true(ref <= Date(1970, 1, 2, tz_0_, pycal))
    assert_true(ref > Date(1969, 12, 31, tz_0_, pycal))
    assert_true(ref >= Date(1969, 12, 31, tz_0_, pycal))


fn test_bitwise(
    pycal: Calendar, unixcal: Calendar, tz1_: TZ, tz_0_: TZ, tz_1: TZ
):
    var ref = Date(1970, 1, 1, tz_0_, pycal)
    assert_true(ref & Date(1970, 1, 1, tz_0_, unixcal) == 0)
    assert_true(ref & Date(1970, 1, 1, tz_1, unixcal) == 0)
    assert_true(ref & Date(1969, 12, 31, tz1_, pycal) == 0)

    assert_true(ref ^ Date(1970, 1, 2, tz_0_, pycal) != 0)
    assert_true(ref | (Date(1970, 1, 2, tz_0_, pycal) & 0) == ref)
    assert_true(ref & ~ref == 0)
    assert_true(ref ^ ~ref == UInt32.MAX_FINITE)


fn test_iso(pycal: Calendar, tz_0_: TZ):
    var ref = Date(1970, 1, 1, tz_0_, pycal)
    var iso_str = "1970-01-01T00:00:00+00:00"
    var iso_format = IsoFormat(IsoFormat.YYYY_MM_DD_T_MM_HH_SS_TZD)
    assert_equal(ref, Date.from_iso[iso_format]())
    assert_equal(iso_str, ref.to_iso[iso_format]())

    iso_str = "1970-01-01 00:00:00+00:00"
    iso_format = IsoFormat(IsoFormat.YYYY_MM_DD___MM_HH_SS_TZD)
    assert_equal(ref, Date.from_iso[iso_format]())
    assert_equal(iso_str, ref.to_iso[iso_format]())

    iso_str = "1970-01-01T00:00:00"
    iso_format = IsoFormat(IsoFormat.YYYY_MM_DD_T_MM_HH_SS)
    assert_equal(ref, Date.from_iso[iso_format]())
    assert_equal(iso_str, ref.to_iso[iso_format]())

    iso_str = "19700101000000"
    iso_format = IsoFormat(IsoFormat.YYYYMMDDMMHHSS)
    assert_equal(ref, Date.from_iso[iso_format]())
    assert_equal(iso_str, ref.to_iso[iso_format]())

    iso_str = "00:00:00"
    iso_format = IsoFormat(IsoFormat.MM_HH_SS)
    assert_equal(ref, Date.from_iso[iso_format]())
    assert_equal(iso_str, ref.to_iso[iso_format]())

    iso_str = "000000"
    iso_format = IsoFormat(IsoFormat.MMHHSS)
    assert_equal(ref, Date.from_iso[iso_format]())
    assert_equal(iso_str, ref.to_iso[iso_format]())


fn test_time():
    try:
        var start = Date.now()
        time.sleep(0.1)
        var end = Date.now()
        assert_equal(start, end)
    except:
        assert_true(False)


fn test_hash():
    var ref = Date(1970, 1, 1)
    var data = hash(ref)
    var parsed = Date.from_hash(data)
    assert_true(ref == parsed)


fn main():
    var tz_0_ = TZ("", 0, 0)
    var tz_1 = TZ("", 1, 0)
    var tz1_ = TZ("", 1, 0, 1)
    # when using python and unix calendar there
    # should be no difference in results
    var pycal = PythonCalendar
    var unixcal = UTCCalendar
    test_add(pycal, unixcal, tz1_, tz_0_, tz_1)
    test_subtract(pycal, unixcal, tz1_, tz_0_, tz_1)
    test_logic(pycal, unixcal, tz1_, tz_0_, tz_1)
    test_bitwise(pycal, unixcal, tz1_, tz_0_, tz_1)
    test_iso(pycal, tz_0_)
    test_time()
    test_hash()
