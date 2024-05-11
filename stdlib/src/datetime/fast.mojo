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
"""Fast implementations of `DateTime` module. All assume no leap seconds or
years.

- `DateTime64`:
    - This is a "normal" `DateTime` with milisecond resolution.
- `DateTime32`:
    - This is a "normal" `DateTime` with minute resolution.
- `DateTime16`:
    - This is a `DateTime` with hour resolution, it can be used as a 
    year, dayofyear, hour representation.
- `DateTime8`:
    - This is a `DateTime` with hour resolution, it can be used as a 
    dayofweek, hour representation.
- Notes:
    - The caveats of each implementation are better explained in 
    each struct's docstrings.
"""
from time import time

from .calendar import Calendar, UTCFastCal, CalendarHashes
import .dt_str

alias _calendar = UTCFastCal
alias _cal_h64 = CalendarHashes(64)
alias _cal_h32 = CalendarHashes(32)
alias _cal_h16 = CalendarHashes(16)
alias _cal_h8 = CalendarHashes(8)


@register_passable("trivial")
struct DateTime64(Hashable, Stringable):
    """Fast `DateTime64` struct. This is a "normal"
    `DateTime` with milisecond resolution. Uses
    given calendar's epoch and other params at
    build time. Assumes all instances have the same
    timezone and epoch and that there are no leap
    seconds or days. UTCCalendar is the default.

    - Hash Resolution:
        - year: Up to year 134_217_728.
        - month: Up to month 32.
        - day: Up to day 32.
        - hour: Up to hour 32.
        - minute: Up to minute 64.
        - second: Up to second 64.
        - m_second: Up to m_second 1024.

    - Milisecond Resolution (Gregorian as reference):
        - year: Up to year 584_942_417 since calendar epoch.

    - Notes:
        - Once methods that alter the underlying m_seconds
            are used, the hash and binary operations thereof
            shouldn't be used since they are invalid.
    """

    var m_second: UInt64
    """Milisecond."""
    var hash: UInt64
    """Hash."""

    fn __init__(
        inout self,
        year: Optional[Int] = None,
        month: Optional[Int] = None,
        day: Optional[Int] = None,
        hour: Optional[Int] = None,
        minute: Optional[Int] = None,
        second: Optional[Int] = None,
        m_second: Optional[Int] = None,
        calendar: Calendar = _calendar,
        hash_val: Optional[UInt64] = None,
    ):
        """Construct a `DateTime64` from valid values.
        UTCCalendar is the default.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.
            calendar: Calendar.
            hash_val: Hash_val.
        """
        var y = year.or_else(int(calendar.min_year))
        var mon = month.or_else(int(calendar.min_month))
        var d = day.or_else(int(calendar.min_day))
        var h = hour.or_else(int(calendar.min_hour))
        var m = minute.or_else(int(calendar.min_minute))
        var s = second.or_else(int(calendar.min_second))
        var ms = day.or_else(int(calendar.min_second))
        self.m_second = m_second.or_else(
            int(calendar.m_seconds_since_epoch(y, mon, d, h, m, s, ms))
        )
        self.hash = hash_val.or_else(
            calendar.hash[_cal_h64](y, mon, d, h, m, s, ms)
        )

    fn get_year(self) -> UInt64:
        """Get the year assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_y) >> _cal_h64.shift_64_y

    fn get_month(self) -> UInt64:
        """Get the month assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_mon) >> _cal_h64.shift_64_mon

    fn get_day(self) -> UInt64:
        """Get the day assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_d) >> _cal_h64.shift_64_d

    fn get_hour(self) -> UInt64:
        """Get the hour assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_h) >> _cal_h64.shift_64_h

    fn get_minute(self) -> UInt64:
        """Get the minute assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_m) >> _cal_h64.shift_64_m

    fn get_second(self) -> UInt64:
        """Get the second assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_s) >> _cal_h64.shift_64_s

    fn get_m_second(self) -> UInt64:
        """Get the m_second assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h64.mask_64_ms) >> _cal_h64.shift_64_ms

    fn get_attrs(
        self,
    ) -> (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64):
        """Get the year, month, day, hour, minute, second, milisecond
        assuming the hash is valid.

        Returns:
            The items.
        """
        return (
            self.get_year(),
            self.get_month(),
            self.get_day(),
            self.get_hour(),
            self.get_minute(),
            self.get_second(),
            self.get_m_second(),
        )

    @always_inline
    fn replace(
        owned self,
        *,
        owned year: Optional[Int] = None,
        owned month: Optional[Int] = None,
        owned day: Optional[Int] = None,
        owned hour: Optional[Int] = None,
        owned minute: Optional[Int] = None,
        owned second: Optional[Int] = None,
        owned m_second: Optional[Int] = None,
    ) -> Self:
        """Replace values inside the hash.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.

        Returns:
            Self.
        """
        var s = self
        if year:
            s.hash = (s.hash & ~_cal_h64.mask_64_y) | (
                year.unsafe_take() << _cal_h64.shift_64_y
            )
        if month:
            s.hash = (s.hash & ~_cal_h64.mask_64_mon) | (
                month.unsafe_take() << _cal_h64.shift_64_mon
            )
        if day:
            s.hash = (s.hash & ~_cal_h64.mask_64_d) | (
                day.unsafe_take() << _cal_h64.shift_64_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h64.mask_64_h) | (
                day.unsafe_take() << _cal_h64.shift_64_h
            )
        if minute:
            s.hash = (s.hash & ~_cal_h64.mask_64_m) | (
                day.unsafe_take() << _cal_h64.shift_64_m
            )
        if second:
            s.hash = (s.hash & ~_cal_h64.mask_64_s) | (
                day.unsafe_take() << _cal_h64.shift_64_s
            )
        if m_second:
            s.hash = (s.hash & ~_cal_h64.mask_64_ms) | (
                day.unsafe_take() << _cal_h64.shift_64_ms
            )
        return s

    @always_inline
    fn seconds_since_epoch(self) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds since epoch.
        """
        return self.m_second // 1000

    @always_inline
    fn m_seconds_since_epoch(self) -> UInt64:
        """Miliseconds since the begining of the calendar's epoch.

        Returns:
            Miliseconds since epoch.
        """
        return self.m_second

    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.m_second += other.m_second
        return self

    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.m_second -= other.m_second
        return self

    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.m_second += other.m_second

    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.m_second -= other.m_second

    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second == other.m_second

    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second != other.m_second

    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second > other.m_second

    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second >= other.m_second

    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second <= other.m_second

    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_second < other.m_second

    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    fn __and__(self, other: Self) -> UInt64:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & other.hash

    fn __or__(self, other: Self) -> UInt64:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | other.hash

    fn __xor__(self, other: Self) -> UInt64:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ other.hash

    fn __int__(self) -> UInt64:
        """Int.

        Returns:
            Result.
        """
        return self.m_second

    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    fn add(owned self, seconds: Int = 0, m_seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            seconds: Seconds.
            m_seconds: Miliseconds.

        Returns:
            Self.
        """
        self.m_second += seconds * 1000 + m_seconds
        return self

    fn subtract(owned self, seconds: Int = 0, m_seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            seconds: Seconds.
            m_seconds: Miliseconds.

        Returns:
            Self.
        """
        self.m_second -= seconds * 1000 + m_seconds
        return self

    @staticmethod
    @always_inline
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime64` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime64().add(seconds=seconds)

    @staticmethod
    fn now() -> Self:
        """Construct a `DateTime64` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        var ms = time.now() // 1_000_000
        var s = ms // 1_000
        return DateTime64.from_unix_epoch(s).add(m_seconds=ms)

    @parameter
    fn to_iso(self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        e.g. `1970-01-01T00:00:00` .

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var s = self.get_attrs()
        var time = dt_str.to_iso(
            int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4]), int(s[5])
        )
        return time[:19]

    @parameter
    fn to_iso_compact[
        iso: dt_str.IsoFormat = dt_str.IsoFormat()
    ](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYYMMDDMMHHSS`.
        e.g. `19700101000000` .

        Parameters:
            iso: The chosen IsoFormat.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var s = self.get_attrs()
        var time = dt_str.to_iso_compact(
            int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4]), int(s[5])
        )
        return time[:14]

    @staticmethod
    @parameter
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a dateTime64time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The IsoFormat to parse.
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = DateTime64(
                int(p[0]),
                int(p[1]),
                int(p[2]),
                int(p[3]),
                int(p[4]),
                int(p[5]),
                calendar=calendar,
            )
            return dt
        except:
            return None

    @staticmethod
    fn from_hash(value: UInt64, calendar: Calendar = _calendar) -> Self:
        """Construct a `DateTime64` from a hash made by it.

        Args:
            value: The hash.
            calendar: The Calendar to parse the hash with.

        Returns:
            Self.
        """
        var d = calendar.from_hash[_cal_h64](int(value))
        return DateTime64(
            int(d[0]),
            int(d[1]),
            int(d[2]),
            int(d[3]),
            int(d[4]),
            int(d[5]),
            calendar=calendar,
            hash_val=value,
        )


@register_passable("trivial")
struct DateTime32(Hashable, Stringable):
    """Fast `DateTime32 ` struct. This is a "normal" `DateTime`
    with minute resolution. Uses given calendar's epoch
    and other params at build time. Assumes all instances have
    the same timezone and epoch and that there are no leap
    seconds or days. UTCCalendar is the default.

    - Hash Resolution:
        - year: Up to year 4_096.
        - month: Up to month 16.
        - day: Up to day 32.
        - hour: Up to hour 32.
        - minute: Up to minute 64.

    - Minute Resolution (Gregorian as reference):
        - year: Up to year 8_171 since calendar epoch.

    - Notes:
        - Once methods that alter the underlying minutes
            are used, the hash and binary operations thereof
            shouldn't be used since they are invalid.
    """

    var minute: UInt32
    """Minute."""
    var hash: UInt32
    """Hash."""

    fn __init__(
        inout self,
        year: Optional[Int] = None,
        month: Optional[Int] = None,
        day: Optional[Int] = None,
        hour: Optional[Int] = None,
        minute: Optional[Int] = None,
        calendar: Calendar = _calendar,
        hash_val: Optional[UInt32] = None,
    ):
        """Construct a `DateTime32 ` from valid values.
        UTCCalendar is the default.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            calendar: Calendar.
            hash_val: Hash_val.
        """
        var y = year.or_else(int(calendar.min_year))
        var mon = month.or_else(int(calendar.min_month))
        var d = day.or_else(int(calendar.min_day))
        var h = hour.or_else(int(calendar.min_hour))
        var m = minute.or_else(int(calendar.min_minute))
        self.minute = (
            minute.or_else(int(calendar.min_minute))
            + (calendar.seconds_since_epoch(y, mon, d, h, m, 0) // 60).cast[
                DType.uint32
            ]()
        )
        self.hash = hash_val.or_else(calendar.hash[_cal_h32](y, mon, d, h, m))

    fn get_year(self) -> UInt32:
        """Get the year assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_y) >> _cal_h32.shift_32_y

    fn get_month(self) -> UInt32:
        """Get the month assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_mon) >> _cal_h32.shift_32_mon

    fn get_day(self) -> UInt32:
        """Get the day assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_d) >> _cal_h32.shift_32_d

    fn get_hour(self) -> UInt32:
        """Get the hour assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_h) >> _cal_h32.shift_32_h

    fn get_minute(self) -> UInt32:
        """Get the minute assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h32.mask_32_m) >> _cal_h32.shift_32_m

    fn get_attrs(
        self,
    ) -> (UInt32, UInt32, UInt32, UInt32, UInt32):
        """Get the year, month, day, hour, minute, second
        assuming the hash is valid.

        Returns:
            The items.
        """
        return (
            self.get_year(),
            self.get_month(),
            self.get_day(),
            self.get_hour(),
            self.get_minute(),
        )

    @always_inline
    fn seconds_since_epoch(self) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds since epoch.
        """
        return self.minute.cast[DType.uint64]() * 60

    @always_inline
    fn replace(
        owned self,
        *,
        owned year: Optional[Int] = None,
        owned month: Optional[Int] = None,
        owned day: Optional[Int] = None,
        owned hour: Optional[Int] = None,
        owned minute: Optional[Int] = None,
    ) -> Self:
        """Replace values inside the hash.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.

        Returns:
            Self.
        """
        var s = self
        if year:
            s.hash = (s.hash & ~_cal_h32.mask_32_y) | (
                year.unsafe_take() << _cal_h32.shift_32_y
            )
        if month:
            s.hash = (s.hash & ~_cal_h32.mask_32_mon) | (
                month.unsafe_take() << _cal_h32.shift_32_mon
            )
        if day:
            s.hash = (s.hash & ~_cal_h32.mask_32_d) | (
                day.unsafe_take() << _cal_h32.shift_32_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h32.mask_32_h) | (
                day.unsafe_take() << _cal_h32.shift_32_h
            )
        if minute:
            s.hash = (s.hash & ~_cal_h32.mask_32_m) | (
                day.unsafe_take() << _cal_h32.shift_32_m
            )
        return s

    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.minute += other.minute
        return self

    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.minute -= other.minute
        return self

    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.minute += other.minute

    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.minute -= other.minute

    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute == other.minute

    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute != other.minute

    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute > other.minute

    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute >= other.minute

    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute <= other.minute

    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minute < other.minute

    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    fn __and__(self, other: Self) -> UInt32:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & other.hash

    fn __or__(self, other: Self) -> UInt32:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | other.hash

    fn __xor__(self, other: Self) -> UInt32:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ other.hash

    fn __int__(self) -> UInt32:
        """Int.

        Returns:
            Result.
        """
        return self.minute

    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    fn add(owned self, minutes: Int = 0, seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            minutes: Minutes.
            seconds: Seconds.

        Returns:
            Self.
        """

        self.minute += minutes + seconds * 60
        return self

    fn subtract(owned self, minutes: Int = 0, seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            minutes: Minutes.
            seconds: Seconds.

        Returns:
            Self.
        """
        self.minute -= minutes + seconds * 60
        return self

    @staticmethod
    @always_inline
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime32 ` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime32().add(seconds=seconds)

    @staticmethod
    fn now() -> Self:
        """Construct a `DateTime32 ` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime32.from_unix_epoch(time.now() // 1_000_000_000)

    @parameter
    fn to_iso(self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        e.g. `1970-01-01T00:00:00` .

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso(
            int(self.get_year()),
            int(self.get_month()),
            int(self.get_day()),
            int(self.get_hour()),
            int(self.get_minute()),
            int(_calendar.min_second),
        )
        return time[:19]

    @parameter
    fn to_iso_compact[
        iso: dt_str.IsoFormat = dt_str.IsoFormat()
    ](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYYMMDDMMHHSS`.
        e.g. `19700101000000` .

        Parameters:
            iso: The chosen IsoFormat.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso_compact(
            int(self.get_year()),
            int(self.get_month()),
            int(self.get_day()),
            int(self.get_hour()),
            int(self.get_minute()),
            int(_calendar.min_second),
        )
        return time[:14]

    @staticmethod
    @parameter
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a `DateTime32` time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The IsoFormat to parse.
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = DateTime32(
                int(p[0]),
                int(p[1]),
                int(p[2]),
                int(p[3]),
                int(p[4]),
                calendar=calendar,
            )
            return dt
        except:
            return None

    @staticmethod
    fn from_hash(value: UInt32, calendar: Calendar = _calendar) -> Self:
        """Construct a `DateTime32 ` from a hash made by it.

        Args:
            value: The hash.
            calendar: The Calendar to parse the hash with.

        Returns:
            Self.
        """
        var d = calendar.from_hash[_cal_h32](int(value))
        return DateTime32(
            int(d[0]),
            int(d[1]),
            int(d[2]),
            int(d[3]),
            int(d[4]),
            calendar=calendar,
            hash_val=value,
        )


@register_passable("trivial")
struct DateTime16(Hashable, Stringable):
    """Fast `DateTime16 ` struct. This is a `DateTime` with
    hour resolution, it can be used as a year, dayofyear,
    hour representation. uses given calendar's epoch and
    other params at build time. Assumes all instances have
    the same timezone and epoch and that there are no leap
    seconds or days. UTCCalendar is the default.

    - Hash Resolution:
        year: Up to year 4.
        day: Up to day 512.
        hour: Up to hour 32.

    - Hour Resolution (Gregorian as reference):
        - year: Up to year 7 since calendar epoch.

    - Notes:
        - Once methods that alter the underlying m_seconds
            are used, the hash and binary operations thereof
            shouldn't be used since they are invalid.
    """

    var hour: UInt16
    """Hour."""
    var hash: UInt16
    """Hash."""

    fn __init__(
        inout self,
        year: Optional[Int] = None,
        month: Optional[Int] = None,
        day: Optional[Int] = None,
        hour: Optional[Int] = None,
        minute: Optional[Int] = None,
        second: Optional[Int] = None,
        calendar: Calendar = _calendar,
        hash_val: Optional[UInt16] = None,
    ):
        """Construct a `DateTime16` from valid values.
        UTCCalendar is the default.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            calendar: Calendar.
            hash_val: Hash_val.
        """
        var y = year.or_else(int(calendar.min_year))
        var mon = month.or_else(int(calendar.min_month))
        var d = day.or_else(int(calendar.min_day))
        var h = hour.or_else(int(calendar.min_hour))
        var m = minute.or_else(int(calendar.min_minute))
        var s = second.or_else(int(calendar.min_second))
        self.hour = (
            hour.or_else(int(calendar.min_hour))
            + (
                calendar.seconds_since_epoch(y, mon, d, h, m, 0) // (60 * 60)
            ).cast[DType.uint16]()
        )
        self.hash = hash_val.or_else(
            calendar.hash[_cal_h16](y, mon, d, h, m, s)
        )

    fn get_year(self) -> UInt16:
        """Get the year assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h16.mask_16_y) >> _cal_h16.shift_16_y

    fn get_day(self) -> UInt16:
        """Get the day assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h16.mask_16_d) >> _cal_h16.shift_16_d

    fn get_hour(self) -> UInt16:
        """Get the hour assuming the hash is valid.

        Returns:
            The item.
        """
        return (self.hash & _cal_h16.mask_16_h) >> _cal_h16.shift_16_h

    fn get_attrs(self) -> (UInt16, UInt16, UInt16):
        """Get the year, month, day, hour, minute, second
        assuming the hash is valid.

        Returns:
            The items.
        """
        return (self.get_year(), self.get_day(), self.get_hour())

    @always_inline
    fn replace(
        owned self,
        *,
        owned day: Optional[Int] = None,
        owned hour: Optional[Int] = None,
    ) -> Self:
        """Replace values inside the hash.

        Args:
            day: Day.
            hour: Hour.

        Returns:
            Self.
        """
        var s = self
        if day:
            s.hash = (s.hash & ~_cal_h16.mask_16_d) | (
                day.unsafe_take() << _cal_h16.shift_16_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h16.mask_16_h) | (
                day.unsafe_take() << _cal_h16.shift_16_h
            )
        return s

    @always_inline
    fn seconds_since_epoch(self) -> UInt32:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds.
        """
        return self.hour.cast[DType.uint32]() * 60 * 60

    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hour += other.hour
        return self

    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hour -= other.hour
        return self

    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.hour += other.hour

    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.hour -= other.hour

    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour == other.hour

    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour != other.hour

    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour > other.hour

    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour >= other.hour

    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour <= other.hour

    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour < other.hour

    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    fn __and__(self, other: Self) -> UInt16:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & other.hash

    fn __or__(self, other: Self) -> UInt16:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | other.hash

    fn __xor__(self, other: Self) -> UInt16:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ other.hash

    fn __int__(self) -> UInt16:
        """Int.

        Returns:
            Result.
        """
        return self.hour

    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    fn add(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """

        self.hour += hours + seconds * 60 * 60
        return self

    fn subtract(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """
        self.hour -= hours + seconds * 60 * 60
        return self

    @staticmethod
    @always_inline
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime16 ` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime16().add(seconds=seconds)

    @staticmethod
    fn now() -> Self:
        """Construct a `DateTime16 ` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime16.from_unix_epoch(time.now() // 1_000_000_000)

    @parameter
    fn to_iso(self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        e.g. `1970-01-01T00:00:00` .

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso(
            int(self.get_year()),
            int(_calendar.min_month),
            int(self.get_day()),
            int(self.get_hour()),
            int(_calendar.min_minute),
            int(_calendar.min_second),
        )
        return time[:19]

    @parameter
    fn to_iso_compact[
        iso: dt_str.IsoFormat = dt_str.IsoFormat()
    ](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYYMMDDMMHHSS`.
        e.g. `19700101000000` .

        Parameters:
            iso: The chosen IsoFormat.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso_compact(
            int(self.get_year()),
            int(_calendar.min_month),
            int(self.get_day()),
            int(self.get_hour()),
            int(_calendar.min_minute),
            int(_calendar.min_second),
        )
        return time[:14]

    @staticmethod
    @parameter
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a `DateTime16` time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The IsoFormat to parse.
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = DateTime16(
                int(p[0]),
                int(p[1]),
                int(p[2]),
                int(p[3]),
                int(p[4]),
                int(p[5]),
                calendar=calendar,
            )
            return dt
        except:
            return None

    @staticmethod
    fn from_hash(value: UInt16, calendar: Calendar = _calendar) -> Self:
        """Construct a `DateTime16 ` from a hash made by it.

        Args:
            value: The hash.
            calendar: The Calendar to parse the hash with.

        Returns:
            Self.
        """
        var d = calendar.from_hash[_cal_h16](int(value))
        return DateTime16(
            year=int(d[0]),
            day=int(d[2]),
            hour=int(d[3]),
            calendar=calendar,
            hash_val=value,
        )


@register_passable("trivial")
struct DateTime8(Hashable, Stringable):
    """Fast `DateTime8 ` struct. This is a `DateTime`
    with hour resolution, it can be used as a dayofweek,
    hour representation. uses given calendar's epoch and
    other params at build time. Assumes all instances have
    the same timezone and epoch and that there are no leap
    seconds or days. UTCCalendar is the default.

    - Hash Resolution:
        - day: Up to day 8.
        - hour: Up to hour 32.

    - Hour Resolution (Gregorian as reference):
        - hour: Up to hour 256 (~ 10 days) since calendar epoch.

    - Notes:
        - Once methods that alter the underlying m_seconds
            are used, the hash and binary operations thereof
            shouldn't be used since they are invalid.
    """

    var hour: UInt8
    """Hour."""
    var hash: UInt8
    """Hash."""

    fn __init__(
        inout self,
        year: Optional[Int] = None,
        month: Optional[Int] = None,
        day: Optional[Int] = None,
        hour: Optional[Int] = None,
        minute: Optional[Int] = None,
        second: Optional[Int] = None,
        calendar: Calendar = _calendar,
        hash_val: Optional[UInt8] = None,
    ):
        """Construct a `DateTime8 ` from valid values.
        UTCCalendar is the default.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            calendar: Calendar.
            hash_val: Hash_val.
        """
        var y = year.or_else(int(calendar.min_year))
        var mon = month.or_else(int(calendar.min_month))
        var d = day.or_else(int(calendar.min_day))
        var h = hour.or_else(int(calendar.min_hour))
        var m = minute.or_else(int(calendar.min_minute))
        var s = second.or_else(int(calendar.min_second))
        self.hour = (
            hour.or_else(int(calendar.min_hour))
            + (
                calendar.seconds_since_epoch(y, mon, d, h, m, 0) // (60 * 60)
            ).cast[DType.uint8]()
        )
        self.hash = hash_val.or_else(calendar.hash[_cal_h8](y, mon, d, h, m, s))

    fn get_day(self) -> UInt8:
        """Get the day assuming the hash is valid.

        Returns:
            The day.
        """
        return (self.hash & _cal_h8.mask_8_d) >> _cal_h8.shift_8_d

    fn get_hour(self) -> UInt8:
        """Get the hour assuming the hash is valid.

        Returns:
            The hour.
        """
        return (self.hash & _cal_h8.mask_8_h) >> _cal_h8.shift_8_h

    fn get_attrs(
        self,
    ) -> (UInt8, UInt8):
        """Get the year, month, day, hour, minute, second
        assuming the hash is valid.

        Returns:
            The items.
        """
        return (self.get_day(), self.get_hour())

    @always_inline
    fn replace(
        owned self,
        *,
        owned day: Optional[Int] = None,
        owned hour: Optional[Int] = None,
    ) -> Self:
        """Replace values inside the hash.

        Args:
            day: Day.
            hour: Hour.

        Returns:
            Self.
        """
        var s = self
        if day:
            s.hash = (s.hash & ~_cal_h8.mask_8_d) | (
                day.unsafe_take() << _cal_h8.shift_8_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h8.mask_8_h) | (
                day.unsafe_take() << _cal_h8.shift_8_h
            )
        return s

    @always_inline
    fn seconds_since_epoch(self) -> UInt16:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds.
        """
        return self.hour.cast[DType.uint16]() * 60 * 60

    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hour += other.hour
        return self

    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hour -= other.hour
        return self

    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.hour += other.hour

    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.hour -= other.hour

    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour == other.hour

    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour != other.hour

    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour > other.hour

    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour >= other.hour

    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour <= other.hour

    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hour < other.hour

    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    fn __and__(self, other: Self) -> UInt8:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & other.hash

    fn __or__(self, other: Self) -> UInt8:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | other.hash

    fn __xor__(self, other: Self) -> UInt8:
        """And.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ other.hash

    fn __int__(self) -> UInt8:
        """Int.

        Returns:
            Result.
        """
        return self.hour

    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    fn add(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """

        self.hour += hours + seconds * 60 * 60
        return self

    fn subtract(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """
        self.hour -= hours + seconds * 60 * 60
        return self

    @staticmethod
    @always_inline
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime8 ` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime8().add(seconds=seconds)

    @staticmethod
    fn now() -> Self:
        """Construct a `DateTime8 ` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return DateTime8.from_unix_epoch(time.now() // 1_000_000_000)

    @parameter
    fn to_iso(self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        e.g. `1970-01-01T00:00:00` .

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso(
            int(_calendar.min_year),
            int(_calendar.min_month),
            int(self.get_day()),
            int(self.get_hour()),
            int(_calendar.min_minute),
            int(_calendar.min_second),
        )
        return time[:19]

    @parameter
    fn to_iso_compact[
        iso: dt_str.IsoFormat = dt_str.IsoFormat()
    ](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYYMMDDMMHHSS`.
        e.g. `19700101000000` .

        Parameters:
            iso: The chosen IsoFormat.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso_compact(
            int(_calendar.min_year),
            int(_calendar.min_month),
            int(self.get_day()),
            int(self.get_hour()),
            int(_calendar.min_minute),
            int(_calendar.min_second),
        )
        return time[:14]

    @staticmethod
    @parameter
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a `DateTime8` time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The IsoFormat to parse.
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = DateTime8(
                int(p[0]),
                int(p[1]),
                int(p[2]),
                int(p[3]),
                int(p[4]),
                int(p[5]),
                calendar=calendar,
            )
            return dt
        except:
            return None

    @staticmethod
    fn from_hash(value: UInt8, calendar: Calendar = _calendar) -> Self:
        """Construct a `DateTime8` from a hash made by it.

        Args:
            value: The hash.
            calendar: The Calendar to parse the hash with.

        Returns:
            Self.
        """
        var d = calendar.from_hash[_cal_h8](int(value))
        return DateTime8(
            day=int(d[2]), hour=int(d[3]), calendar=calendar, hash_val=value
        )
