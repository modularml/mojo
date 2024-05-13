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

from datetime.timezone import TimeZone


fn test_tz_no_iana() raises:
    alias tz = TimeZone[None, False, False]
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
    var tz0_offset = tz0.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz_1_offset = tz_1.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz_2_offset = tz_2.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz_3_offset = tz_3.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz1__offset = tz1_.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz2__offset = tz2_.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    var tz3__offset = tz3_.offset_at(d[0], d[1], d[2], d[3], d[4], d[5])
    assert_true(tz0_offset == (0, 0, 0))
    assert_true(tz_1_offset == (1, 0, 1))
    assert_true(tz_2_offset == (2, 30, 1))
    assert_true(tz_3_offset == (3, 45, 1))
    assert_true(tz1__offset == (1, 0, -1))
    assert_true(tz2__offset == (2, 30, -1))
    assert_true(tz3__offset == (3, 45, -1))


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
    test_tz_iana_no_dst()
