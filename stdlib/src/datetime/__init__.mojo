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
"""The datetime package.

- `DateTime`
    - A structure aware of TimeZone, Calendar, and leap days and seconds.
    - Nanosecond resolution, though when using dunder methods (e.g. dt1 == dt2) 
        it has only Microsecond resolution.
- `Date`
    - A structure aware of TimeZone, Calendar, and leap days and seconds.
- `TimeZone`
    - By default UTC, highly customizable and options for full or partial
        IANA timezones support.
- `DateTime64`, `DateTime32`, `DateTime16`, `DateTime8`
    - Fast implementations of DateTime, no leap seconds or years,
        and some have much lower resolutions but better performance.
- Notes:
    - The caveats of each implementation are better explained in each struct's docstrings.
"""

from .zoneinfo import get_zoneinfo
from .timezone import TimeZone
from .dt_str import IsoFormat
from .datetime import DateTime
from .date import Date
from .fast import DateTime64, DateTime32, DateTime16, DateTime8
