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

from datetime.zoneinfo import (
    Offset,
    TzDT,
    ZoneDST,
    ZoneInfoFile32,
    ZoneInfoFile8,
    ZoneInfoMem32,
    ZoneInfoMem8,
    get_zoneinfo,
    get_leapsecs,
    # _parse_iana_leapsecs,
    # _parse_iana_zonenow,
    # _parse_iana_dst_transitions,
)


fn _test_offset(i: Int, j: Int, k: Int):
    var minutes = (0, 30, 45)
    var sign = 1 if k == 0 else -1
    var of = Offset(i, minutes[j], sign)
    assert_equal(of.hour, i)
    assert_equal(of.minute, minutes[j])
    assert_equal(of.sign, sign)


fn test_offset() raises:
    for k in range(2):
        for j in range(3):
            for i in range(16):
                _test_offset(i, j, k)


fn test_tzdst() raises:
    # TODO
    pass


fn test_zonedst() raises:
    # TODO
    pass


fn test_zoneinfomem32() raises:
    var storage = ZoneInfoMem32()
    var tz0 = "tz0"
    var tz1 = "tz1"
    var tz2 = "tz2"
    var tz30 = "tz2"
    var tz45 = "tz2"
    var tz0_of = Offset(0, 0, 1)
    var tz1_of = Offset(1, 0, 1)
    var tz2_of = Offset(2, 0, 1)
    var tz30_of = Offset(0, 30, 1)
    var tz45_of = Offset(0, 45, 1)
    storage.add(tz0, tz0_of)
    storage.add(tz1, tz1_of)
    storage.add(tz2, tz2_of)
    storage.add(tz30, tz30_of)
    storage.add(tz45, tz45_of)
    var tz0_read = storage.get(tz0)
    var tz1_read = storage.get(tz1)
    var tz2_read = storage.get(tz2)
    var tz30_read = storage.get(tz3)
    var tz45_read = storage.get(tz4)
    assert_true(tz0_read.hour == tz0_of.hour)
    assert_true(tz1_read.hour == tz1_of.hour)
    assert_true(tz2_read.hour == tz2_of.hour)
    assert_true(tz30_read.hour == tz30_of.hour)
    assert_true(tz45_read.hour == tz45_of.hour)
    assert_true(tz0_read.minute == tz0_of.minute)
    assert_true(tz1_read.minute == tz1_of.minute)
    assert_true(tz2_read.minute == tz2_of.minute)
    assert_true(tz30_read.minute == tz30_of.minute)
    assert_true(tz45_read.minute == tz45_of.minute)
    assert_true(tz0_read.sign == tz0_of.sign)
    assert_true(tz1_read.sign == tz1_of.sign)
    assert_true(tz2_read.sign == tz2_of.sign)
    assert_true(tz30_read.sign == tz30_of.sign)
    assert_true(tz45_read.sign == tz45_of.sign)
    assert_true(tz0_read.buf == tz0_of.buf)
    assert_true(tz1_read.buf == tz1_of.buf)
    assert_true(tz2_read.buf == tz2_of.buf)
    assert_true(tz30_read.buf == tz30_of.buf)
    assert_true(tz45_read.buf == tz45_of.buf)


fn test_zoneinfomem8() raises:
    var storage = ZoneInfoMem8()
    var tz0 = "tz0"
    var tz1 = "tz1"
    var tz2 = "tz2"
    var tz30 = "tz2"
    var tz45 = "tz2"
    var tz0_of = Offset(0, 0, 1)
    var tz1_of = Offset(1, 0, 1)
    var tz2_of = Offset(2, 0, 1)
    var tz30_of = Offset(0, 30, 1)
    var tz45_of = Offset(0, 45, 1)
    storage.add(tz0, tz0_of)
    storage.add(tz1, tz1_of)
    storage.add(tz2, tz2_of)
    storage.add(tz30, tz30_of)
    storage.add(tz45, tz45_of)
    var tz0_read = storage.get(tz0)
    var tz1_read = storage.get(tz1)
    var tz2_read = storage.get(tz2)
    var tz30_read = storage.get(tz3)
    var tz45_read = storage.get(tz4)
    assert_true(tz0_read.hour == tz0_of.hour)
    assert_true(tz1_read.hour == tz1_of.hour)
    assert_true(tz2_read.hour == tz2_of.hour)
    assert_true(tz30_read.hour == tz30_of.hour)
    assert_true(tz45_read.hour == tz45_of.hour)
    assert_true(tz0_read.minute == tz0_of.minute)
    assert_true(tz1_read.minute == tz1_of.minute)
    assert_true(tz2_read.minute == tz2_of.minute)
    assert_true(tz30_read.minute == tz30_of.minute)
    assert_true(tz45_read.minute == tz45_of.minute)
    assert_true(tz0_read.sign == tz0_of.sign)
    assert_true(tz1_read.sign == tz1_of.sign)
    assert_true(tz2_read.sign == tz2_of.sign)
    assert_true(tz30_read.sign == tz30_of.sign)
    assert_true(tz45_read.sign == tz45_of.sign)
    assert_true(tz0_read.buf == tz0_of.buf)
    assert_true(tz1_read.buf == tz1_of.buf)
    assert_true(tz2_read.buf == tz2_of.buf)
    assert_true(tz30_read.buf == tz30_of.buf)
    assert_true(tz45_read.buf == tz45_of.buf)


fn test_zoneinfofile32() raises:
    var storage = ZoneInfoFile32()
    var tz0 = "tz0"
    var tz1 = "tz1"
    var tz2 = "tz2"
    var tz30 = "tz2"
    var tz45 = "tz2"
    var tz0_of = Offset(0, 0, 1)
    var tz1_of = Offset(1, 0, 1)
    var tz2_of = Offset(2, 0, 1)
    var tz30_of = Offset(0, 30, 1)
    var tz45_of = Offset(0, 45, 1)
    storage.add(tz0, tz0_of)
    storage.add(tz1, tz1_of)
    storage.add(tz2, tz2_of)
    storage.add(tz30, tz30_of)
    storage.add(tz45, tz45_of)
    var tz0_read = storage.get(tz0)
    var tz1_read = storage.get(tz1)
    var tz2_read = storage.get(tz2)
    var tz30_read = storage.get(tz3)
    var tz45_read = storage.get(tz4)
    assert_true(tz0_read.hour == tz0_of.hour)
    assert_true(tz1_read.hour == tz1_of.hour)
    assert_true(tz2_read.hour == tz2_of.hour)
    assert_true(tz30_read.hour == tz30_of.hour)
    assert_true(tz45_read.hour == tz45_of.hour)
    assert_true(tz0_read.minute == tz0_of.minute)
    assert_true(tz1_read.minute == tz1_of.minute)
    assert_true(tz2_read.minute == tz2_of.minute)
    assert_true(tz30_read.minute == tz30_of.minute)
    assert_true(tz45_read.minute == tz45_of.minute)
    assert_true(tz0_read.sign == tz0_of.sign)
    assert_true(tz1_read.sign == tz1_of.sign)
    assert_true(tz2_read.sign == tz2_of.sign)
    assert_true(tz30_read.sign == tz30_of.sign)
    assert_true(tz45_read.sign == tz45_of.sign)
    assert_true(tz0_read.buf == tz0_of.buf)
    assert_true(tz1_read.buf == tz1_of.buf)
    assert_true(tz2_read.buf == tz2_of.buf)
    assert_true(tz30_read.buf == tz30_of.buf)
    assert_true(tz45_read.buf == tz45_of.buf)


fn test_zoneinfofile8() raises:
    var storage = ZoneInfoFile8()
    var tz0 = "tz0"
    var tz1 = "tz1"
    var tz2 = "tz2"
    var tz30 = "tz2"
    var tz45 = "tz2"
    var tz0_of = Offset(0, 0, 1)
    var tz1_of = Offset(1, 0, 1)
    var tz2_of = Offset(2, 0, 1)
    var tz30_of = Offset(0, 30, 1)
    var tz45_of = Offset(0, 45, 1)
    storage.add(tz0, tz0_of)
    storage.add(tz1, tz1_of)
    storage.add(tz2, tz2_of)
    storage.add(tz30, tz30_of)
    storage.add(tz45, tz45_of)
    var tz0_read = storage.get(tz0)
    var tz1_read = storage.get(tz1)
    var tz2_read = storage.get(tz2)
    var tz30_read = storage.get(tz3)
    var tz45_read = storage.get(tz4)
    assert_true(tz0_read.hour == tz0_of.hour)
    assert_true(tz1_read.hour == tz1_of.hour)
    assert_true(tz2_read.hour == tz2_of.hour)
    assert_true(tz30_read.hour == tz30_of.hour)
    assert_true(tz45_read.hour == tz45_of.hour)
    assert_true(tz0_read.minute == tz0_of.minute)
    assert_true(tz1_read.minute == tz1_of.minute)
    assert_true(tz2_read.minute == tz2_of.minute)
    assert_true(tz30_read.minute == tz30_of.minute)
    assert_true(tz45_read.minute == tz45_of.minute)
    assert_true(tz0_read.sign == tz0_of.sign)
    assert_true(tz1_read.sign == tz1_of.sign)
    assert_true(tz2_read.sign == tz2_of.sign)
    assert_true(tz30_read.sign == tz30_of.sign)
    assert_true(tz45_read.sign == tz45_of.sign)
    assert_true(tz0_read.buf == tz0_of.buf)
    assert_true(tz1_read.buf == tz1_of.buf)
    assert_true(tz2_read.buf == tz2_of.buf)
    assert_true(tz30_read.buf == tz30_of.buf)
    assert_true(tz45_read.buf == tz45_of.buf)


fn test_get_zoneinfo() raises:
    # TODO
    pass


fn test_get_leapsecs() raises:
    # TODO
    pass


fn test_parse_iana_leapsecs() raises:
    # TODO
    pass


fn test_parse_iana_zonenow() raises:
    # TODO
    pass


fn test_parse_iana_dst_transitions() raises:
    # TODO
    pass


fn main() raises:
    test_zoneinfomem32()
    test_zoneinfomem8()
    test_zoneinfofile32()
    test_zoneinfofile8()
    test_get_zoneinfo()
    test_get_leapsecs()
    test_parse_iana_leapsecs()
    test_parse_iana_zonenow()
    test_parse_iana_dst_transitions()
