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

from datetime.timezone import TimeZone, ZoneInfo, ZoneInfoMem32, ZoneInfoMem8


fn test_tz_no_iana() raises:
    alias tz = TimeZone[iana=False, pyzoneinfo=False, native=False]
    var tz0 = tz("Etc/UTC", 0, 0)
    var tz_1 = tz("Etc/UTC-1", 1, 0)
    var tz_2 = tz("Etc/UTC-2", 2, 30)
    var tz_3 = tz("Etc/UTC-3", 3, 45)
    var tz1_ = tz("Etc/UTC+1", 1, 0, -1)
    var tz2_ = tz("Etc/UTC+2", 2, 30, -1)
    var tz3_ = tz("Etc/UTC+3", 3, 45, -1)
    assert_true(tz0 == tz())
    assert_true(tz1_ != tz_1 and tz2_ != tz_2 and tz3_ != tz_3)
    var d = (1970, 1, 1, 0, 0, 0)
    var tz0_of = tz0.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz_1_of = tz_1.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz_2_of = tz_2.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz_3_of = tz_3.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz1__of = tz1_.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz2__of = tz2_.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz3__of = tz3_.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    assert_true(tz0_of.hour == 0 and tz0_of.minute == 0 and tz0_of.sign == 1)
    assert_true(tz_1_of.hour == 1 and tz_1_of.minute == 0 and tz_1_of.sign == 1)
    assert_true(
        tz_2_of.hour == 2 and tz_2_of.minute == 30 and tz_2_of.sign == 1
    )
    assert_true(
        tz_3_of.hour == 3 and tz_3_of.minute == 45 and tz_3_of.sign == 1
    )
    assert_true(
        tz1__of.hour == 1 and tz1__of.minute == 0 and tz1__of.sign == -1
    )
    assert_true(
        tz2__of.hour == 2 and tz2__of.minute == 30 and tz2__of.sign == -1
    )
    assert_true(
        tz3__of.hour == 3 and tz3__of.minute == 45 and tz3__of.sign == -1
    )


fn test_tz_iana_dst() raises:
    # TODO: test from positive and negative UTC
    # TODO: test transitions to and from DST
    # TODO: test for Australia/Lord_Howe and Antarctica/Troll base
    pass


fn test_tz_iana_no_dst() raises:
    # TODO: test from positive and negative UTC
    pass


fn main() raises:
    # TODO: more thorough tests
    test_tz_no_iana()
    test_tz_iana_dst()
    test_tz_iana_no_dst()
