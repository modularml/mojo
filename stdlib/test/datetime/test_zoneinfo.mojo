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
    _parse_iana_leapsecs,
    # _parse_iana_zonenow,
    # _parse_iana_dst_transitions,
)


fn test_offset() raises:
    # TODO
    pass


fn test_tzdst() raises:
    # TODO
    pass


fn test_zonedst() raises:
    # TODO
    pass


fn test_zoneinfomem32() raises:
    # TODO
    pass


fn test_zoneinfomem8() raises:
    # TODO
    pass


fn test_zoneinfofile32() raises:
    # TODO
    pass


fn test_zoneinfofile8() raises:
    # TODO
    pass


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
