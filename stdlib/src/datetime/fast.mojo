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
from collections.optional import OptionalReg

from .calendar import Calendar, UTCFastCal, CalendarHashes
import .dt_str

alias _calendar = UTCFastCal
alias _cal_h64 = CalendarHashes(64)
alias _cal_h32 = CalendarHashes(32)
alias _cal_h16 = CalendarHashes(16)
alias _cal_h8 = CalendarHashes(8)


# trait _IntCollect(Intable, CollectionElement):
#     ...


@register_passable("trivial")
struct DateTime64(Hashable, Stringable):
    """Fast `DateTime64` struct. This is a "normal"
    `DateTime` with milisecond resolution. Uses
    UTCFastCal epoch [1970-01-01, 9999-12-31] and other
    params at build time. Assumes all instances have
    the same timezone and epoch and that there are no
    leap seconds or days.

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

    var m_seconds: UInt64
    """Miliseconds since epoch."""
    var hash: UInt64
    """Hash."""

    fn __init__(
        inout self,
        year: OptionalReg[Int] = None,
        month: OptionalReg[Int] = None,
        day: OptionalReg[Int] = None,
        hour: OptionalReg[Int] = None,
        minute: OptionalReg[Int] = None,
        second: OptionalReg[Int] = None,
        m_second: OptionalReg[Int] = None,
        hash_val: OptionalReg[Int] = None,
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
            hash_val: Hash_val.
        """
        var y = int(year.value()) if year else int(_calendar.min_year)
        var mon = int(month.value()) if month else int(_calendar.min_month)
        var d = int(day.value()) if day else int(_calendar.min_day)
        var h = int(hour.value()) if hour else int(_calendar.min_hour)
        var m = int(minute.value()) if minute else int(_calendar.min_minute)
        var s = int(second.value()) if second else int(_calendar.min_second)
        var ms = int(m_second.value()) if day else int(_calendar.min_milisecond)
        self.m_seconds = _calendar.m_seconds_since_epoch(y, mon, d, h, m, s, ms)
        self.hash = int(hash_val.value()) if hash_val else int(
            _calendar.hash[_cal_h64](y, mon, d, h, m, s, ms)
        )

    @always_inline("nodebug")
    fn __getattr__[name: StringLiteral](self) -> UInt64:
        """Get the attribute.

        Parameters:
            name: The name of the attribute.

        Returns:
            The attribute value assuming the hash is valid.
        """

        @parameter
        if name == "year":
            return (self.hash & _cal_h64.mask_64_y) >> _cal_h64.shift_64_y
        elif name == "month":
            return (self.hash & _cal_h64.mask_64_mon) >> _cal_h64.shift_64_mon
        elif name == "day":
            return (self.hash & _cal_h64.mask_64_d) >> _cal_h64.shift_64_d
        elif name == "hour":
            return (self.hash & _cal_h64.mask_64_h) >> _cal_h64.shift_64_h
        elif name == "minute":
            return (self.hash & _cal_h64.mask_64_m) >> _cal_h64.shift_64_m
        elif name == "second":
            return (self.hash & _cal_h64.mask_64_s) >> _cal_h64.shift_64_s
        elif name == "m_second":
            return (self.hash & _cal_h64.mask_64_ms) >> _cal_h64.shift_64_ms
        else:
            constrained[name == "b", "that attr does not exist"]()
            return 0

    @always_inline
    fn replace(
        owned self,
        *,
        owned year: OptionalReg[Int] = None,
        owned month: OptionalReg[Int] = None,
        owned day: OptionalReg[Int] = None,
        owned hour: OptionalReg[Int] = None,
        owned minute: OptionalReg[Int] = None,
        owned second: OptionalReg[Int] = None,
        owned m_second: OptionalReg[Int] = None,
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
                year.value() << _cal_h64.shift_64_y
            )
        if month:
            s.hash = (s.hash & ~_cal_h64.mask_64_mon) | (
                month.value() << _cal_h64.shift_64_mon
            )
        if day:
            s.hash = (s.hash & ~_cal_h64.mask_64_d) | (
                day.value() << _cal_h64.shift_64_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h64.mask_64_h) | (
                day.value() << _cal_h64.shift_64_h
            )
        if minute:
            s.hash = (s.hash & ~_cal_h64.mask_64_m) | (
                day.value() << _cal_h64.shift_64_m
            )
        if second:
            s.hash = (s.hash & ~_cal_h64.mask_64_s) | (
                day.value() << _cal_h64.shift_64_s
            )
        if m_second:
            s.hash = (s.hash & ~_cal_h64.mask_64_ms) | (
                day.value() << _cal_h64.shift_64_ms
            )
        return s

    @always_inline("nodebug")
    fn seconds_since_epoch(self) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds since epoch.
        """
        return self.m_seconds // 1000

    @always_inline("nodebug")
    fn m_seconds_since_epoch(self) -> UInt64:
        """Miliseconds since the begining of the calendar's epoch.

        Returns:
            Miliseconds since epoch.
        """
        return self.m_seconds

    @always_inline("nodebug")
    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.m_seconds += other.m_seconds
        return self

    @always_inline("nodebug")
    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.m_seconds -= other.m_seconds
        return self

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.m_seconds += other.m_seconds

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.m_seconds -= other.m_seconds

    @always_inline("nodebug")
    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_seconds == other.m_seconds

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_seconds != other.m_seconds

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_seconds > other.m_seconds

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_seconds >= other.m_seconds

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_seconds <= other.m_seconds

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.m_seconds < other.m_seconds

    @always_inline("nodebug")
    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    @always_inline("nodebug")
    fn __and__[T: Intable](self, other: T) -> UInt64:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & int(other)

    @always_inline("nodebug")
    fn __or__[T: Intable](self, other: T) -> UInt64:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | int(other)

    @always_inline("nodebug")
    fn __xor__[T: Intable](self, other: T) -> UInt64:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ int(other)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Result.
        """
        return int(self.m_second)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    @always_inline("nodebug")
    fn add(owned self, seconds: Int = 0, m_seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            seconds: Seconds.
            m_seconds: Miliseconds.

        Returns:
            Self.
        """
        self.m_seconds += seconds * 1000 + m_seconds
        return self

    @always_inline("nodebug")
    fn subtract(owned self, seconds: Int = 0, m_seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            seconds: Seconds.
            m_seconds: Miliseconds.

        Returns:
            Self.
        """
        self.m_seconds -= seconds * 1000 + m_seconds
        return self

    @staticmethod
    @always_inline("nodebug")
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
        return Self().add(seconds=seconds)

    @staticmethod
    @always_inline("nodebug")
    fn now() -> Self:
        """Construct a `DateTime64` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        var ms = time.now() // 1_000_000
        var s = ms // 1_000
        return Self.from_unix_epoch(s).add(m_seconds=ms)

    @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` e.g. `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        -> `1970-01-01T00:00:00` .

        Parameters:
            iso: The `IsoFormat`.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var s = self
        var time = dt_str.to_iso[iso](
            int(s.year),
            int(s.month),
            int(s.day),
            int(s.hour),
            int(s.minute),
            int(s.second),
        )
        return time[:19]

    @staticmethod
    @always_inline("nodebug")
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
    ](s: String) -> OptionalReg[Self]:
        """Construct a `DateTime64` from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The `IsoFormat` to parse.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An OptionalReg[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = Self(
                int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])
            )
            return dt
        except:
            return None

    @staticmethod
    @always_inline("nodebug")
    fn from_hash(value: UInt64) -> Self:
        """Construct a `DateTime64` from a hash made by it.

        Args:
            value: The hash.

        Returns:
            Self.
        """
        var d = _calendar.from_hash[_cal_h64](int(value))
        return Self(
            int(d[0]),
            int(d[1]),
            int(d[2]),
            int(d[3]),
            int(d[4]),
            int(d[5]),
            hash_val=int(value),
        )


@register_passable("trivial")
struct DateTime32(Hashable, Stringable):
    """Fast `DateTime32 ` struct. This is a "normal" `DateTime`
    with minute resolution. Uses UTCFastCal epoch
    [1970-01-01, 9999-12-31] and other params at build time.
    Assumes all instances have the same timezone and epoch
    and that there are no leap seconds or days.

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

    var minutes: UInt32
    """Minutes since epoch."""
    var hash: UInt32
    """Hash."""

    fn __init__(
        inout self,
        year: OptionalReg[Int] = None,
        month: OptionalReg[Int] = None,
        day: OptionalReg[Int] = None,
        hour: OptionalReg[Int] = None,
        minute: OptionalReg[Int] = None,
        hash_val: OptionalReg[Int] = None,
    ):
        """Construct a `DateTime32` from valid values.
        UTCCalendar is the default.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            hash_val: Hash_val.
        """
        var y = int(year.value()) if year else int(_calendar.min_year)
        var mon = int(month.value()) if month else int(_calendar.min_month)
        var d = int(day.value()) if day else int(_calendar.min_day)
        var h = int(hour.value()) if hour else int(_calendar.min_hour)
        var m = int(minute.value()) if minute else int(_calendar.min_minute)
        self.minutes = (
            _calendar.seconds_since_epoch(
                y, mon, d, h, m, int(_calendar.min_second)
            )
            // 60
        ).cast[DType.uint32]()
        self.hash = int(hash_val.value()) if hash_val else int(
            _calendar.hash[_cal_h32](y, mon, d, h, m)
        )

    @always_inline("nodebug")
    fn __getattr__[name: StringLiteral](self) -> UInt32:
        """Get the attribute.

        Parameters:
            name: The name of the attribute.

        Returns:
            The attribute value assuming the hash is valid.
        """

        @parameter
        if name == "year":
            return (self.hash & _cal_h32.mask_32_y) >> _cal_h32.shift_32_y
        elif name == "month":
            return (self.hash & _cal_h32.mask_32_mon) >> _cal_h32.shift_32_mon
        elif name == "day":
            return (self.hash & _cal_h32.mask_32_d) >> _cal_h32.shift_32_d
        elif name == "hour":
            return (self.hash & _cal_h32.mask_32_h) >> _cal_h32.shift_32_h
        elif name == "minute":
            return (self.hash & _cal_h32.mask_32_m) >> _cal_h32.shift_32_m
        else:
            constrained[name == "b", "that attr does not exist"]()
            return 0

    @always_inline("nodebug")
    fn seconds_since_epoch(self) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds since epoch.
        """
        return self.minutes.cast[DType.uint64]() * 60

    @always_inline
    fn replace(
        owned self,
        *,
        owned year: OptionalReg[Int] = None,
        owned month: OptionalReg[Int] = None,
        owned day: OptionalReg[Int] = None,
        owned hour: OptionalReg[Int] = None,
        owned minute: OptionalReg[Int] = None,
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
                year.value() << _cal_h32.shift_32_y
            )
        if month:
            s.hash = (s.hash & ~_cal_h32.mask_32_mon) | (
                month.value() << _cal_h32.shift_32_mon
            )
        if day:
            s.hash = (s.hash & ~_cal_h32.mask_32_d) | (
                day.value() << _cal_h32.shift_32_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h32.mask_32_h) | (
                day.value() << _cal_h32.shift_32_h
            )
        if minute:
            s.hash = (s.hash & ~_cal_h32.mask_32_m) | (
                day.value() << _cal_h32.shift_32_m
            )
        return s

    @always_inline("nodebug")
    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.minutes += other.minutes
        return self

    @always_inline("nodebug")
    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.minutes -= other.minutes
        return self

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.minutes += other.minutes

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.minutes -= other.minutes

    @always_inline("nodebug")
    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minutes == other.minutes

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minutes != other.minutes

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minutes > other.minutes

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minutes >= other.minutes

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minutes <= other.minutes

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.minutes < other.minutes

    @always_inline("nodebug")
    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    @always_inline("nodebug")
    fn __and__[T: Intable](self, other: T) -> UInt32:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & int(other)

    @always_inline("nodebug")
    fn __or__[T: Intable](self, other: T) -> UInt32:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | int(other)

    @always_inline("nodebug")
    fn __xor__[T: Intable](self, other: T) -> UInt32:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ int(other)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Result.
        """
        return int(self.minutes)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    @always_inline("nodebug")
    fn add(owned self, minutes: Int = 0, seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            minutes: Minutes.
            seconds: Seconds.

        Returns:
            Self.
        """

        self.minutes += minutes + seconds * 60
        return self

    @always_inline("nodebug")
    fn subtract(owned self, minutes: Int = 0, seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            minutes: Minutes.
            seconds: Seconds.

        Returns:
            Self.
        """
        self.minutes -= minutes + seconds * 60
        return self

    @staticmethod
    @always_inline("nodebug")
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime32` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return Self().add(seconds=seconds)

    @staticmethod
    @always_inline("nodebug")
    fn now() -> Self:
        """Construct a `DateTime32` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return Self.from_unix_epoch(time.now() // 1_000_000_000)

    @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` e.g. `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        -> `1970-01-01T00:00:00` .

        Parameters:
            iso: The `IsoFormat`.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso(
            int(self.year),
            int(self.month),
            int(self.day),
            int(self.hour),
            int(self.minute),
            int(_calendar.min_second),
        )
        return time[:19]

    @staticmethod
    @always_inline("nodebug")
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
    ](s: String) -> OptionalReg[Self]:
        """Construct a `DateTime32` time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The `IsoFormat` to parse.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An OptionalReg[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = Self(int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]))
            return dt
        except:
            return None

    @staticmethod
    @always_inline("nodebug")
    fn from_hash(value: UInt32) -> Self:
        """Construct a `DateTime32` from a hash made by it.

        Args:
            value: The hash.

        Returns:
            Self.
        """
        var d = _calendar.from_hash[_cal_h32](int(value))
        return Self(
            int(d[0]),
            int(d[1]),
            int(d[2]),
            int(d[3]),
            int(d[4]),
            hash_val=int(value),
        )


@register_passable("trivial")
struct DateTime16(Hashable, Stringable):
    """Fast `DateTime16` struct. This is a `DateTime` with
    hour resolution, it can be used as a year, dayofyear,
    hour representation. Uses UTCFastCal epoch
    [1970-01-01, 9999-12-31] and other params at build time.
    Assumes all instances have the same timezone and epoch
    and that there are no leap seconds or days.

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

    var hours: UInt16
    """Hours since epoch."""
    var hash: UInt16
    """Hash."""

    fn __init__(
        inout self,
        year: OptionalReg[Int] = None,
        month: OptionalReg[Int] = None,
        day: OptionalReg[Int] = None,
        hour: OptionalReg[Int] = None,
        hash_val: OptionalReg[Int] = None,
    ):
        """Construct a `DateTime16` from valid values.
        UTCCalendar is the default.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            hash_val: Hash_val.
        """
        var y = int(year.value()) if year else int(_calendar.min_year)
        var mon = int(month.value()) if month else int(_calendar.min_month)
        var d = int(day.value()) if day else int(_calendar.min_day)
        var h = int(hour.value()) if hour else int(_calendar.min_hour)
        var m = int(_calendar.min_minute)
        var s = int(_calendar.min_second)
        self.hours = (
            _calendar.seconds_since_epoch(y, mon, d, h, m, s) // (60 * 60)
        ).cast[DType.uint16]()
        self.hash = int(hash_val.value()) if hash_val else int(
            _calendar.hash[_cal_h16](y, mon, d, h, m, s)
        )

    @always_inline("nodebug")
    fn __getattr__[name: StringLiteral](self) -> UInt16:
        """Get the attribute.

        Parameters:
            name: The name of the attribute.

        Returns:
            The attribute value assuming the hash is valid.
        """

        @parameter
        if name == "year":
            return (self.hash & _cal_h16.mask_16_y) >> _cal_h16.shift_16_y
        elif name == "day":
            return (self.hash & _cal_h16.mask_16_d) >> _cal_h16.shift_16_d
        elif name == "hour":
            return (self.hash & _cal_h16.mask_16_h) >> _cal_h16.shift_16_h
        else:
            constrained[name == "b", "that attribute does not exist"]()
            return 0

    @always_inline
    fn replace(
        owned self,
        *,
        owned day: OptionalReg[Int] = None,
        owned hour: OptionalReg[Int] = None,
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
                day.value() << _cal_h16.shift_16_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h16.mask_16_h) | (
                day.value() << _cal_h16.shift_16_h
            )
        return s

    @always_inline
    fn seconds_since_epoch(self) -> UInt32:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds.
        """
        return self.hours.cast[DType.uint32]() * 60 * 60

    @always_inline("nodebug")
    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hours += other.hours
        return self

    @always_inline("nodebug")
    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hours -= other.hours
        return self

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.hours += other.hours

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.hours -= other.hours

    @always_inline("nodebug")
    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours == other.hours

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours != other.hours

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours > other.hours

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours >= other.hours

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours <= other.hours

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours < other.hours

    @always_inline("nodebug")
    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    @always_inline("nodebug")
    fn __and__[T: Intable](self, other: T) -> UInt16:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & int(other)

    @always_inline("nodebug")
    fn __or__[T: Intable](self, other: T) -> UInt16:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | int(other)

    @always_inline("nodebug")
    fn __xor__[T: Intable](self, other: T) -> UInt16:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ int(other)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Result.
        """
        return int(self.hours)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    @always_inline("nodebug")
    fn add(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """

        self.hours += hours + seconds * 60 * 60
        return self

    @always_inline("nodebug")
    fn subtract(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """
        self.hours -= hours + seconds * 60 * 60
        return self

    @staticmethod
    @always_inline("nodebug")
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime16` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return Self().add(seconds=seconds)

    @staticmethod
    @always_inline("nodebug")
    fn now() -> Self:
        """Construct a `DateTime16` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return Self.from_unix_epoch(time.now() // 1_000_000_000)

    @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` e.g. `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        -> `1970-01-01T00:00:00` .

        Parameters:
            iso: The `IsoFormat`.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso[iso](
            int(self.year),
            int(_calendar.min_month),
            int(self.day),
            int(self.hour),
            int(_calendar.min_minute),
            int(_calendar.min_second),
        )
        return time[:19]

    @staticmethod
    @always_inline("nodebug")
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
    ](s: String) -> OptionalReg[Self]:
        """Construct a `DateTime16` time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The `IsoFormat` to parse.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An OptionalReg[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = Self(int(p[0]), int(p[1]), int(p[2]), int(p[3]))
            return dt
        except:
            return None

    @staticmethod
    @always_inline("nodebug")
    fn from_hash(value: UInt16) -> Self:
        """Construct a `DateTime16` from a hash made by it.

        Args:
            value: The hash.

        Returns:
            Self.
        """
        var d = _calendar.from_hash[_cal_h16](int(value))
        return Self(
            year=int(d[0]), day=int(d[2]), hour=int(d[3]), hash_val=int(value)
        )


@register_passable("trivial")
struct DateTime8(Hashable, Stringable):
    """Fast `DateTime8` struct. This is a `DateTime`
    with hour resolution, it can be used as a dayofweek,
    hour representation. Uses UTCFastCal epoch
    [1970-01-01, 9999-12-31] and other params at build time.
    Assumes all instances have the same timezone and epoch
    and that there are no leap seconds or days.

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

    var hours: UInt8
    """Hours since epoch."""
    var hash: UInt8
    """Hash."""

    fn __init__(
        inout self,
        year: OptionalReg[Int] = None,
        month: OptionalReg[Int] = None,
        day: OptionalReg[Int] = None,
        hour: OptionalReg[Int] = None,
        hash_val: OptionalReg[Int] = None,
    ):
        """Construct a `DateTime8` from valid values.
        UTCCalendar is the default.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            hash_val: Hash_val.
        """
        var y = int(year.value()) if year else int(_calendar.min_year)
        var mon = int(month.value()) if month else int(_calendar.min_month)
        var d = int(day.value()) if day else int(_calendar.min_day)
        var h = int(hour.value()) if hour else int(_calendar.min_hour)
        var m = int(_calendar.min_minute)
        var s = int(_calendar.min_second)
        self.hours = (
            _calendar.seconds_since_epoch(y, mon, d, h, m, s) // (60 * 60)
        ).cast[DType.uint8]()
        self.hash = int(hash_val.value()) if hash_val else int(
            _calendar.hash[_cal_h8](y, mon, d, h, m, s)
        )

    @always_inline("nodebug")
    fn __getattr__[name: StringLiteral](self) -> UInt8:
        """Get the attribute.

        Parameters:
            name: The name of the attribute.

        Returns:
            The attribute value assuming the hash is valid.
        """

        @parameter
        if name == "day":
            return (self.hash & _cal_h8.mask_8_d) >> _cal_h8.shift_8_d
        elif name == "hour":
            return (self.hash & _cal_h8.mask_8_h) >> _cal_h8.shift_8_h
        else:
            constrained[name == "b", "that attr does not exist"]()
            return 0

    @always_inline
    fn replace(
        owned self,
        *,
        owned day: OptionalReg[Int] = None,
        owned hour: OptionalReg[Int] = None,
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
                day.value() << _cal_h8.shift_8_d
            )
        if hour:
            s.hash = (s.hash & ~_cal_h8.mask_8_h) | (
                day.value() << _cal_h8.shift_8_h
            )
        return s

    @always_inline("nodebug")
    fn seconds_since_epoch(self) -> UInt16:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            Seconds.
        """
        return self.hour.cast[DType.uint16]() * 60 * 60

    @always_inline("nodebug")
    fn __add__(owned self, owned other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hours += other.hours
        return self

    @always_inline("nodebug")
    fn __sub__(owned self, owned other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Self.
        """
        self.hours -= other.hours
        return self

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self.hours += other.hours

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self.hours -= other.hours

    @always_inline("nodebug")
    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            Result.
        """
        return int(self.hash)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours == other.hours

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours != other.hours

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours > other.hours

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours >= other.hours

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours <= other.hours

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return self.hours < other.hours

    @always_inline("nodebug")
    fn __invert__(owned self) -> Self:
        """Invert.

        Returns:
            Self.
        """
        self.hash = ~self.hash
        return self

    @always_inline("nodebug")
    fn __and__[T: Intable](self, other: T) -> UInt8:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash & int(other)

    @always_inline("nodebug")
    fn __or__[T: Intable](self, other: T) -> UInt8:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash | int(other)

    @always_inline("nodebug")
    fn __xor__[T: Intable](self, other: T) -> UInt8:
        """And.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.hash ^ int(other)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Result.
        """
        return int(self.hours)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Str.

        Returns:
            String.
        """
        return self.to_iso()

    @always_inline("nodebug")
    fn add(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Add to self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """

        self.hours += hours + seconds * 60 * 60
        return self

    @always_inline("nodebug")
    fn subtract(owned self, hours: Int = 0, seconds: Int = 0) -> Self:
        """Subtract from self.

        Args:
            hours: Hours.
            seconds: Seconds.

        Returns:
            Self.
        """
        self.hours -= hours + seconds * 60 * 60
        return self

    @staticmethod
    @always_inline("nodebug")
    fn from_unix_epoch(seconds: Int) -> Self:
        """Construct a `DateTime8` from the seconds since the Unix Epoch
        1970-01-01.

        Args:
            seconds: Seconds.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return Self().add(seconds=seconds)

    @staticmethod
    @always_inline("nodebug")
    fn now() -> Self:
        """Construct a `DateTime8` from `time.now()`.

        Returns:
            Self.

        Notes:
            This builds an instance with a hash set to default UTC epoch start.
        """
        return Self.from_unix_epoch(time.now() // 1_000_000_000)

    @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` e.g. `IsoFormat.YYYY_MM_DD_T_MM_HH_SS`.
        -> `1970-01-01T00:00:00` .

        Parameters:
            iso: The `IsoFormat`.

        Returns:
            String.

        Notes:
            This is done assuming the current hash is valid.
        """
        var time = dt_str.to_iso[iso](
            int(_calendar.min_year),
            int(_calendar.min_month),
            int(self.day),
            int(self.hour),
            int(_calendar.min_minute),
            int(_calendar.min_second),
        )
        return time[:19]

    @staticmethod
    @always_inline("nodebug")
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
    ](s: String) -> OptionalReg[Self]:
        """Construct a `DateTime8` time from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The `IsoFormat` to parse.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An OptionalReg[Self].
        """
        try:
            var p = dt_str.from_iso[iso](s)
            var dt = Self(int(p[0]), int(p[1]), int(p[2]), int(p[3]))
            return dt
        except:
            return None

    @staticmethod
    @always_inline("nodebug")
    fn from_hash(value: UInt8) -> Self:
        """Construct a `DateTime8` from a hash made by it.

        Args:
            value: The hash.

        Returns:
            Self.
        """
        var d = _calendar.from_hash[_cal_h8](int(value))
        return Self(day=int(d[2]), hour=int(d[3]), hash_val=int(value))
