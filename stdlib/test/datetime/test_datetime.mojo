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

from datetime.datetime import DateTime
from datetime.calendar import Calendar, PythonCalendar, UTCCalendar
from datetime.timezone import TimeZone
from datetime.dt_str import IsoFormat


fn test_add() raises:
    # when using python and unix calendar there should be no difference in results
    alias pycal = PythonCalendar
    alias unixcal = UTCCalendar
    alias dt = DateTime[iana=False, pyzoneinfo=False, native=False]
    alias TZ = dt._tz
    alias tz_0_ = TZ("", 0, 0)
    alias tz_1 = TZ("", 1, 0)
    alias tz1_ = TZ("", 1, 0, 1)

    # test february leapyear
    var result = dt(2024, 3, 1, tz=tz_0_, calendar=pycal) + dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    var offset_0 = dt(2024, 2, 29, tz=tz_0_, calendar=unixcal)
    var offset_p_1 = dt(2024, 2, 29, tz=tz_1, calendar=unixcal)
    var offset_n_1 = dt(2024, 3, 1, tz=tz1_, calendar=unixcal)
    var add_seconds = dt(2024, 3, 1, tz=tz_0_, calendar=unixcal).add(
        seconds=24 * 3600
    )
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test february not leapyear
    result = dt(2023, 3, 1, tz=tz_0_, calendar=pycal) + dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(2023, 2, 28, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(2023, 2, 28, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(2023, 3, 1, tz=tz1_, calendar=unixcal)
    add_seconds = dt(2023, 3, 1, tz=tz_0_, calendar=unixcal).add(
        seconds=24 * 3600
    )
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test normal month
    result = dt(2024, 5, 31, tz=tz_0_, calendar=pycal) + dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(2024, 6, 1, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(2024, 6, 1, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(2024, 5, 31, tz=tz1_, calendar=unixcal)
    add_seconds = dt(2024, 5, 31, tz=tz_0_, calendar=unixcal).add(
        seconds=24 * 3600
    )
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test december
    result = dt(2024, 12, 31, tz=tz_0_, calendar=pycal) + dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(2025, 1, 1, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(2025, 1, 1, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(2024, 12, 31, tz=tz1_, calendar=unixcal)
    add_seconds = dt(2024, 12, 31, tz=tz_0_, calendar=unixcal).add(
        seconds=24 * 3600
    )
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test year and month add
    result = dt(2022, 6, 1, tz=tz_0_, calendar=pycal) + dt(
        3, 6, 31, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(2025, 1, 1, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(2025, 1, 1, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(2024, 12, 31, tz=tz1_, calendar=unixcal)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)

    # test positive overflow pycal
    result = dt(9999, 12, 31, tz=tz_0_, calendar=pycal) + dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(1, 1, 1, tz=tz_0_, calendar=pycal)
    offset_p_1 = dt(1, 1, 1, tz=tz_1, calendar=pycal)
    offset_n_1 = dt(1, 1, 1, tz=tz1_, calendar=pycal)
    add_seconds = dt(9999, 12, 31, tz=tz_0_, calendar=pycal).add(
        seconds=24 * 3600
    )
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)

    # test positive overflow unixcal
    result = dt(9999, 12, 31, tz=tz_0_, calendar=unixcal) + dt(
        0, 0, 1, tz=tz_0_, calendar=unixcal
    )
    offset_0 = dt(1970, 1, 1, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(1970, 1, 1, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(1970, 1, 1, tz=tz1_, calendar=unixcal)
    add_seconds = dt(9999, 12, 31, tz=tz_0_, calendar=unixcal).add(
        seconds=24 * 3600
    )
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == add_seconds)


fn test_subtract() raises:
    # when using python and unix calendar there should be no difference in results
    alias pycal = PythonCalendar
    alias unixcal = UTCCalendar
    alias dt = DateTime[iana=False, pyzoneinfo=False, native=False]
    alias TZ = dt._tz
    alias tz_0_ = TZ("", 0, 0)
    alias tz_1 = TZ("", 1, 0)
    alias tz1_ = TZ("", 1, 0, 1)

    # test february leapyear
    var result = dt(2024, 3, 1, tz=tz_0_, calendar=pycal) - dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    var offset_0 = dt(2024, 2, 29, tz=tz_0_, calendar=unixcal)
    var offset_p_1 = dt(2024, 2, 29, tz=tz_1, calendar=unixcal)
    var offset_n_1 = dt(2024, 3, 1, tz=tz1_, calendar=unixcal)
    var sub_seconds = dt(2024, 3, 1, tz=tz_0_, calendar=unixcal).subtract(
        seconds=1
    )
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test february not leapyear
    result = dt(2023, 3, 1, tz=tz_0_, calendar=pycal) - dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(2023, 2, 28, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(2023, 2, 28, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(2023, 3, 1, tz=tz1_, calendar=unixcal)
    sub_seconds = dt(2023, 3, 1, tz=tz_0_, calendar=unixcal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test normal month
    result = dt(2024, 6, 1, tz=tz_0_, calendar=pycal) - dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(2024, 5, 31, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(2024, 5, 31, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(2024, 6, 1, tz=tz1_, calendar=unixcal)
    sub_seconds = dt(2024, 6, 1, tz=tz_0_, calendar=unixcal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test december
    result = dt(2025, 1, 1, tz=tz_0_, calendar=pycal) - dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(2024, 12, 31, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(2024, 12, 31, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(2025, 1, 1, tz=tz1_, calendar=unixcal)
    sub_seconds = dt(2025, 1, 1, tz=tz_0_, calendar=unixcal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test year and month subtract
    result = dt(2025, 1, 1, tz=tz_0_, calendar=pycal) - dt(
        3, 6, 31, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(2022, 6, 1, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(2022, 6, 1, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(2022, 5, 31, tz=tz1_, calendar=unixcal)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1)

    # test negative overflow pycal
    result = dt(1, 1, 1, tz=tz_0_, calendar=pycal) - dt(
        0, 0, 1, tz=tz_0_, calendar=pycal
    )
    offset_0 = dt(9999, 12, 31, tz=tz_0_, calendar=pycal)
    offset_p_1 = dt(9999, 12, 31, tz=tz_1, calendar=pycal)
    offset_n_1 = dt(9999, 12, 31, tz=tz1_, calendar=pycal)
    sub_seconds = dt(1, 1, 1, tz=tz_0_, calendar=pycal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)

    # test negative overflow unixcal
    result = dt(1970, 1, 1, tz=tz_0_, calendar=unixcal) - dt(
        0, 0, 1, tz=tz_0_, calendar=unixcal
    )
    offset_0 = dt(9999, 12, 31, tz=tz_0_, calendar=unixcal)
    offset_p_1 = dt(9999, 12, 31, tz=tz_1, calendar=unixcal)
    offset_n_1 = dt(9999, 12, 31, tz=tz1_, calendar=unixcal)
    sub_seconds = dt(1970, 1, 1, tz=tz_0_, calendar=unixcal).subtract(seconds=1)
    assert_true(result == offset_0 and result == offset_p_1)
    assert_true(result == offset_n_1 and result == sub_seconds)


fn test_logic() raises:
    # when using python and unix calendar there should be no difference in results
    alias pycal = PythonCalendar
    alias unixcal = UTCCalendar
    alias dt = DateTime[iana=False, pyzoneinfo=False, native=False]
    alias TZ = dt._tz
    alias tz_0_ = TZ("", 0, 0)
    alias tz_1 = TZ("", 1, 0)
    alias tz1_ = TZ("", 1, 0, 1)

    var ref = dt(1970, 1, 1, tz=tz_0_, calendar=pycal)
    assert_true(ref == dt(1970, 1, 1, tz=tz_0_, calendar=unixcal))
    assert_true(ref == dt(1970, 1, 1, tz=tz_1, calendar=unixcal))
    assert_true(ref == dt(1969, 12, 31, tz=tz1_, calendar=pycal))

    assert_true(ref < dt(1970, 1, 2, tz=tz_0_, calendar=pycal))
    assert_true(ref <= dt(1970, 1, 2, tz=tz_0_, calendar=pycal))
    assert_true(ref > dt(1969, 12, 31, tz=tz_0_, calendar=pycal))
    assert_true(ref >= dt(1969, 12, 31, tz=tz_0_, calendar=pycal))


fn test_bitwise() raises:
    # when using python and unix calendar there should be no difference in results
    alias pycal = PythonCalendar
    alias unixcal = UTCCalendar
    alias dt = DateTime[iana=False, pyzoneinfo=False, native=False]
    alias TZ = dt._tz
    alias tz_0_ = TZ("", 0, 0)
    alias tz_1 = TZ("", 1, 0)
    alias tz1_ = TZ("", 1, 0, 1)

    var ref = dt(1970, 1, 1, tz=tz_0_, calendar=pycal)
    assert_true(ref & dt(1970, 1, 1, tz=tz_0_, calendar=unixcal) == 0)
    assert_true(ref & dt(1970, 1, 1, tz=tz_1, calendar=unixcal) == 0)
    assert_true(ref & dt(1969, 12, 31, tz=tz1_, calendar=pycal) == 0)

    assert_true((ref ^ dt(1970, 1, 2, tz=tz_0_, calendar=pycal)) != 0)
    assert_true(
        (ref | (dt(1970, 1, 2, tz=tz_0_, calendar=pycal) & 0)) == hash(ref)
    )
    assert_true((ref & ~ref) == 0)
    assert_true(~(ref ^ ~ref) == 0)


fn test_iso() raises:
    alias pycal = PythonCalendar
    alias dt = DateTime[iana=False, pyzoneinfo=False, native=False]
    alias TZ = dt._tz
    alias tz_0_ = TZ("", 0, 0)

    var ref = dt(1970, 1, 1, tz=tz_0_, calendar=pycal)
    var iso_str = "1970-01-01T00:00:00+00:00"
    alias fmt1 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS_TZD)
    assert_true(ref == dt.from_iso[fmt1](iso_str).value())
    assert_equal(iso_str, ref.to_iso[fmt1]())

    iso_str = "1970-01-01 00:00:00+00:00"
    alias fmt2 = IsoFormat(IsoFormat.YYYY_MM_DD___HH_MM_SS)
    assert_true(ref == dt.from_iso[fmt2](iso_str).value())
    assert_equal(iso_str, ref.to_iso[fmt2]())

    iso_str = "1970-01-01T00:00:00"
    alias fmt3 = IsoFormat(IsoFormat.YYYY_MM_DD_T_HH_MM_SS)
    assert_true(ref == dt.from_iso[fmt3](iso_str).value())
    assert_equal(iso_str, ref.to_iso[fmt3]())

    iso_str = "19700101000000"
    alias fmt4 = IsoFormat(IsoFormat.YYYYMMDDHHMMSS)
    assert_true(ref == dt.from_iso[fmt4](iso_str).value())
    assert_equal(iso_str, ref.to_iso[fmt4]())

    iso_str = "00:00:00"
    alias fmt5 = IsoFormat(IsoFormat.HH_MM_SS)
    assert_true(ref == dt.from_iso[fmt5](iso_str).value())
    assert_equal(iso_str, ref.to_iso[fmt5]())

    iso_str = "000000"
    alias fmt6 = IsoFormat(IsoFormat.HHMMSS)
    assert_true(ref == dt.from_iso[fmt6](iso_str).value())
    assert_equal(iso_str, ref.to_iso[fmt6]())


fn test_time() raises:
    alias dt = DateTime[iana=False, pyzoneinfo=False, native=False]

    var start = dt.now()
    time.sleep(1e-9)  # nanosecond resolution
    var end = dt.now()
    assert_true(start.n_second != end.n_second)


fn test_hash() raises:
    alias dt = DateTime[iana=False, pyzoneinfo=False, native=False]

    var ref = dt(1970, 1, 1)
    var data = hash(ref)
    var parsed = dt.from_hash(data)
    assert_true(ref == parsed)


fn main() raises:
    test_add()
    test_subtract()
    test_logic()
    test_bitwise()
    test_iso()
    test_time()
    test_hash()
