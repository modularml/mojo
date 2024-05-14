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
"""`Date` module.

- Notes:
    - IANA is supported: [`TimeZone` and DST data sources](
        http://www.iana.org/time-zones/repository/tz-link.html).
        [List of TZ identifiers (`tz_str`)](
        https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).
"""
from time import time

from .timezone import TimeZone, ZoneInfo
from .calendar import Calendar, UTCCalendar, PythonCalendar, CalendarHashes
import .dt_str


alias _calendar = PythonCalendar
alias _cal_hash = CalendarHashes(32)


trait _IntCollect(Intable, CollectionElement):
    ...


# @value
@register_passable("trivial")
struct Date[
    iana: Bool = True,
    pyzoneinfo: Bool = True,
    native: Bool = False,
](Hashable, Stringable):
    """Custom `Calendar` and `TimeZone` may be passed in.
    By default uses `PythonCalendar` which is a proleptic
    Gregorian calendar with its given epoch and max years:
    from [0001-01-01, 9999-12-31]. Default `TimeZone`
    is UTC.

    Parameters:
        iana: What timezones from the [IANA database](
            http://www.iana.org/time-zones/repository/tz-link.html)
            are used. [List of TZ identifiers (`tz_str`)](
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
            ). If None, defaults to only using the offsets
            as is, no daylight saving or special exceptions.
        pyzoneinfo: Whether to use python's zoneinfo and
            datetime to get full IANA support.
        native: (fast, partial IANA support) Whether to use a native Dict
            with the current timezones from the [List of TZ identifiers](
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
            at the time of compilation (for now they're hardcoded
            at stdlib release time, in the future it should get them
            from the OS). If it fails at compile time, it defaults to
            using the given offsets when the timezone was constructed.

    - Max Resolution:
        - year: Up to year 65_536.
        - month: Up to month 256.
        - day: Up to day 256.
        - hash: 32 bits.

    - Notes:
        - By default, PythonCalendar has min_hour set to 0,
            that means if you have timezones that have one or more
            hours less than UTC they will be set a day before
            in most calculations. If that is a problem, a custom
            Gregorian calendar with min_hour=12 can be passed
            in the constructor and most timezones will be inside
            the same day.
    """

    var year: UInt16
    """Year."""
    var month: UInt8
    """Month."""
    var day: UInt8
    """Day."""
    # TODO: tz and calendar should be references
    alias _tz = TimeZone[iana, pyzoneinfo, native]
    var tz: Self._tz
    """Tz."""
    var calendar: Calendar
    """Calendar."""

    fn __init__[
        T: _IntCollect = UInt16, A: _IntCollect = UInt8
    ](
        inout self,
        year: Optional[T] = None,
        month: Optional[A] = None,
        day: Optional[A] = None,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ):
        """Construct a `Date` from valid values.

        Parameters:
            T: Any Intable Collectable type.
            A: Any Intable Collectable type.

        Args:
            year: Year.
            month: Month.
            day: Day.
            tz: Tz.
            calendar: Calendar.
        """
        self.year = int(year.value()[]) if year else int(calendar.min_year)
        self.month = int(month.value()[]) if month else int(calendar.min_month)
        self.day = int(day.value()[]) if day else int(calendar.min_day)
        self.tz = tz
        self.calendar = calendar

    @staticmethod
    fn _from_overflow(
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from possibly overflowing values."""
        var d = Self._from_days(days, tz, calendar)
        var mon = Self._from_months(months, tz, calendar)
        var y = Self._from_years(years, tz, calendar)

        y.year = 0 if years == 0 else y.year

        for dt in List(mon, d):
            if dt[].year != calendar.min_year:
                y.year += dt[].year
        y.month = mon.month
        if d.month != calendar.min_month:
            y.month += d.month
        y.day = d.day
        return y

    @always_inline
    fn replace(
        owned self,
        *,
        owned year: Optional[UInt16] = None,
        owned month: Optional[UInt8] = None,
        owned day: Optional[UInt8] = None,
        owned tz: Optional[Self._tz] = None,
        owned calendar: Optional[Calendar] = None,
    ) -> Self:
        """Replace with give value/s.

        Args:
            year: Year.
            month: Month.
            day: Day.
            tz: Tz.
            calendar: Calendar.

        Returns:
            Self.
        """
        var new_self = self
        if year:
            new_self.year = year.unsafe_take()
        if month:
            new_self.month = month.unsafe_take()
        if day:
            new_self.day = day.unsafe_take()
        if tz:
            new_self.tz = tz.unsafe_take()
        if calendar:
            var cal = calendar.unsafe_take()
            new_self.year -= self.calendar.min_year
            new_self.year += cal.min_year
            new_self.month -= self.calendar.min_month
            new_self.month += cal.min_month
            new_self.day -= self.calendar.min_day
            new_self.day += cal.min_day
            new_self.calendar = cal
        return new_self

    fn to_utc(owned self) -> Self:
        """Returns a new instance of `Self` transformed to UTC. If
        `self.tz` is UTC it returns early.

        Returns:
            Self with tz casted to UTC.
        """
        alias TZ_UTC = Self._tz()
        if self.tz == TZ_UTC:
            return self
        var new_self = self
        var offset = self.tz.offset_at(self.year, self.month, self.day, 0, 0, 0)
        var of_h = int(offset[0])
        var of_m = int(offset[1])
        var maxmin = self.calendar.max_minute
        var maxsec = self.calendar.max_typical_second + int(
            self.calendar.leapsecs_since_epoch(self.year, self.month, self.day)
        )
        var amnt = int(of_h * maxmin * maxsec + of_m * maxsec)
        if offset[2] == -1:
            new_self = self.add(seconds=amnt)
        else:
            new_self = self.subtract(seconds=amnt)
        return new_self.replace(tz=TZ_UTC)

    fn from_utc(owned self, tz: Self._tz) -> Self:
        """Translate `TimeZone` from UTC. If `self.tz` is UTC
        it returns early.

        Args:
            tz: `TimeZone` to cast to.

        Returns:
            Self with tz casted to given tz.
        """
        if tz == Self._tz():
            return self
        var maxmin = self.calendar.max_minute
        var maxsec = self.calendar.max_typical_second
        var offset = tz.offset_at(self.year, self.month, self.day, 0, 0, 0)
        var of_h = int(offset[0])
        var of_m = int(offset[1])
        var amnt = int(of_h * maxmin * maxsec + of_m * maxsec)
        var new_self = self
        if offset[2] == 1:
            new_self = self.add(seconds=amnt)
        else:
            new_self = self.subtract(seconds=amnt)
        var leapsecs = int(
            new_self.calendar.leapsecs_since_epoch(
                new_self.year, new_self.month, new_self.day
            )
        )
        return new_self.add(seconds=leapsecs).replace(tz=tz)

    fn seconds_since_epoch(self) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Returns:
            The amount.
        """
        return self.calendar.seconds_since_epoch(
            self.year, self.month, self.day, 0, 0, 0
        )

    fn delta_s(self, other: Self) -> (UInt64, UInt64):
        """Calculates the seconds for `self` and other, using
        a reference calendar.

        Args:
            other: Other.

        Returns:
            - self_ns: Seconds from `self` to reference calendar.
            - other_ns: Seconds from other to reference calendar.
        """
        var s = self
        var o = other.replace(calendar=self.calendar)

        if self.tz != other.tz:
            s = self.to_utc()
            o = other.to_utc()
        var self_s = s.seconds_since_epoch()
        var other_s = o.seconds_since_epoch()
        return self_s, other_s

    fn add(
        owned self,
        *,
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        seconds: Int = 0,
    ) -> Self:
        """Recursively evaluated function to build a valid `Date`
        according to its calendar.

        Args:
            years: Amount of years to add.
            months: Amount of months to add.
            days: Amount of days to add.
            seconds: It is assumed that each day has always `24*60*60` seconds.

        Returns:
            Self.

        Notes:
            On overflow, the `Date` starts from the beginning of the
            calendar's epoch and keeps evaluating until valid.
        """
        var dt = self._from_overflow(
            int(self.year) + years,
            int(self.month) + months,
            int(self.day)
            + days
            + seconds
            // int(
                (self.calendar.max_hour + 1)
                * (self.calendar.max_minute + 1)
                * (self.calendar.max_typical_second + 1)
            ),
            self.tz,
            self.calendar,
        )

        var maxyear = dt.calendar.max_year
        if dt.year > maxyear:
            dt = dt.replace(year=dt.calendar.min_year).add(
                years=int(dt.year - maxyear)
            )
        var maxmon = dt.calendar.max_month
        if dt.month > maxmon:
            dt = dt.replace(month=maxmon).add(months=int(dt.month - maxmon))
        var maxday = dt.calendar.max_days_in_month(dt.year, dt.month)
        if dt.day > maxday:
            dt = dt.replace(day=maxday).add(months=1, days=int(dt.day - maxday))
        return dt

    fn subtract(
        owned self,
        *,
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        seconds: Int = 0,
    ) -> Self:
        """Recursively evaluated function to build a valid `Date`
        according to its calendar.

        Args:
            years: Amount of years to add.
            months: Amount of months to add.
            days: Amount of days to add.
            seconds: Amount of seconds to add.

        Returns:
            Self.

        Notes:
            On overflow, the `Date` goes to the end of the
            calendar's epoch and keeps evaluating until valid.
        """
        var dt = self._from_overflow(
            int(self.year) - years,
            int(self.month) - months,
            int(self.day)
            - days
            - seconds
            // int(
                (self.calendar.max_hour + 1)
                * (self.calendar.max_minute + 1)
                * (self.calendar.max_typical_second + 1)
            ),
            self.tz,
            self.calendar,
        )
        var minyear = dt.calendar.min_year
        if dt.year < minyear:
            dt = dt.replace(year=dt.calendar.max_year).subtract(
                years=int(minyear - dt.year)
            )
        var minmonth = dt.calendar.min_month
        if dt.month < minmonth:
            dt = dt.replace(month=dt.calendar.max_month).subtract(
                years=1, months=int(minmonth - dt.month)
            )
        var minday = dt.calendar.min_day
        if dt.day < minday:
            var prev_day = dt.calendar.max_days_in_month(dt.year, dt.month - 1)
            if dt.month - 1 < dt.calendar.min_month:
                var ref = dt.calendar
                prev_day = ref.max_days_in_month(ref.min_year, ref.max_month)
            dt = dt.replace(day=prev_day).subtract(
                months=1, days=int(minday - dt.day)
            )
        return dt

    @always_inline("nodebug")
    fn add(owned self, other: Self) -> Self:
        """Adds another `Date`.

        Args:
            other: Other.

        Returns:
            A `Date` with the `TimeZone` and `Calendar` of `self`.
        """
        return self.add(
            years=int(other.year), months=int(other.month), days=int(other.day)
        )

    @always_inline("nodebug")
    fn subtract(owned self, other: Self) -> Self:
        """Subtracts another `Date`.

        Args:
            other: Other.

        Returns:
            A `Date` with the `TimeZone` and `Calendar` of `self`.
        """
        return self.subtract(
            years=int(other.year), months=int(other.month), days=int(other.day)
        )

    @always_inline("nodebug")
    fn __add__(owned self, other: Self) -> Self:
        """Add.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.add(other)

    @always_inline("nodebug")
    fn __sub__(owned self, other: Self) -> Self:
        """Subtract.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return self.subtract(other)

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Add Immediate.

        Args:
            other: Other.
        """
        self = self.add(other)

    @always_inline("nodebug")
    fn __isub__(inout self, owned other: Self):
        """Subtract Immediate.

        Args:
            other: Other.
        """
        self = self.subtract(other)

    @always_inline("nodebug")
    fn dayofweek(self) -> UInt8:
        """Calculates the day of the week for a `Date`.

        Returns:
            - day: Day of the week: [0, 6] (monday - sunday) (default).
        """
        return self.calendar.dayofweek(self.year, self.month, self.day)

    fn dayofyear(self) -> UInt16:
        """Calculates the day of the year for a `Date`.

        Returns:
            - day: Day of the year: [1, 366] (for gregorian calendar).
        """
        return self.calendar.dayofyear(self.year, self.month, self.day)

    fn leapsec_since_epoch(self) -> UInt32:
        """Cumulative leap seconds since the calendar's epoch start.

        Returns:
            The amount.
        """
        var dt = self.to_utc()
        return dt.calendar.leapsecs_since_epoch(dt.year, dt.month, dt.day)

    fn __hash__(self) -> Int:
        """Hash.

        Returns:
            The hash.
        """
        return self.calendar.hash[_cal_hash](self.year, self.month, self.day)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return hash(self) == hash(other)

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return hash(self) != hash(other)

    @always_inline("nodebug")
    fn __gt__(self, other: Self) -> Bool:
        """Gt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return hash(self) > hash(other)

    @always_inline("nodebug")
    fn __ge__(self, other: Self) -> Bool:
        """Ge.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return hash(self) >= hash(other)

    @always_inline("nodebug")
    fn __le__(self, other: Self) -> Bool:
        """Le.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return hash(self) <= hash(other)

    @always_inline("nodebug")
    fn __lt__(self, other: Self) -> Bool:
        """Lt.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return hash(self) < hash(other)

    @always_inline("nodebug")
    fn __invert__(owned self) -> UInt32:
        """Invert.

        Returns:
            Self.
        """
        return ~hash(self)

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
        return hash(self) & int(other)

    @always_inline("nodebug")
    fn __or__[T: Intable](self, other: T) -> UInt32:
        """Or.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return hash(self) | int(other)

    @always_inline("nodebug")
    fn __xor__[T: Intable](self, other: T) -> UInt32:
        """Xor.

        Parameters:
            T: Any Intable type.

        Args:
            other: Other.

        Returns:
            Result.
        """
        return hash(self) ^ int(other)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Int.

        Returns:
            Int.
        """
        return hash(self)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """str.

        Returns:
            String.
        """
        return self.to_iso()

    @staticmethod
    fn _from_years(
        years: UInt16,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from years."""
        var delta = calendar.max_year - years
        if delta > 0:
            if years > calendar.min_year:
                return Self(year=years, tz=tz, calendar=calendar)
            return Self._from_years(delta)
        return Self._from_years(calendar.max_year - delta)

    @staticmethod
    fn _from_months(
        months: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from months."""
        if months <= int(calendar.max_month):
            return Self(month=UInt8(months), tz=tz, calendar=calendar)
        var y = months // int(calendar.max_month)
        var rest = months % int(calendar.max_month)
        var dt = Self._from_years(y, tz, calendar)
        dt.month = rest
        return dt

    @staticmethod
    fn _from_days[
        add_leap: Bool = False
    ](
        days: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from days."""
        var minyear = calendar.min_year
        var dt = Self(minyear, tz=tz, calendar=calendar)
        var maxtdays = int(calendar.max_typical_days_in_year)
        var maxposdays = int(calendar.max_possible_days_in_year)
        var years = days // maxtdays
        if years > int(minyear):
            dt = Self._from_years(years, tz, calendar)
        var maxydays = maxposdays if calendar.is_leapyear(dt.year) else maxtdays
        var day = days
        if add_leap:
            var leapdays = calendar.leapdays_since_epoch(
                dt.year, dt.month, dt.day
            )
            day += int(leapdays)
        if day > maxydays:
            var y = day // maxydays
            day = day % maxydays
            var dt2 = Self._from_years(UInt16(y), tz, calendar)
            dt.year += dt2.year
        var maxmondays = int(calendar.max_days_in_month(dt.year, dt.month))
        while day > maxmondays:
            day -= maxmondays
            dt.month += 1
            maxmondays = int(calendar.max_days_in_month(dt.year, dt.month))
        dt.day = day
        return dt

    @staticmethod
    fn _from_hours[
        add_leap: Bool = False
    ](
        hours: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from hours."""
        var h = int(calendar.max_hour)
        if hours <= h:
            return Self(calendar.min_year, tz=tz, calendar=calendar)
        var d = hours // (h + 1)
        return Self._from_days[add_leap](d, tz, calendar)

    @staticmethod
    fn _from_minutes[
        add_leap: Bool = False
    ](
        minutes: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from minutes."""
        var m = int(calendar.max_minute)
        if minutes < m:
            return Self(calendar.min_year, tz=tz, calendar=calendar)
        var h = minutes // (m + 1)
        return Self._from_hours[add_leap](h, tz, calendar)

    @staticmethod
    fn from_seconds[
        add_leap: Bool = False
    ](
        seconds: Int,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from seconds.

        Parameters:
            add_leap: Whether to add the leap seconds and leap days
                since the start of the calendar's epoch.

        Args:
            seconds: Seconds.
            tz: Tz.
            calendar: Calendar.

        Returns:
            Self.
        """
        var minutes = seconds // (int(calendar.max_typical_second) + 1)
        var dt = Self._from_minutes(minutes, tz, calendar)
        if not add_leap:
            return dt
        var max_second = calendar.max_second(
            dt.year, dt.month, dt.day, calendar.min_hour, calendar.min_minute
        )
        var numerator = seconds
        if add_leap:
            var leapsecs = calendar.leapsecs_since_epoch(
                dt.year, dt.month, dt.day
            )
            numerator += int(leapsecs)
        var m = numerator // (int(max_second) + 1)
        return Self._from_minutes(m, tz, calendar)

    @staticmethod
    fn from_unix_epoch[
        add_leap: Bool = False
    ](seconds: Int, tz: Self._tz = Self._tz(),) -> Self:
        """Construct a `Date` from the seconds since the Unix Epoch
        1970-01-01. Adding the cumulative leap seconds since 1972
        to the given date.

        Parameters:
            add_leap: Whether to add the leap seconds and leap days
                since the start of the calendar's epoch.

        Args:
            seconds: Seconds.
            tz: Tz.

        Returns:
            Self.
        """
        return Self.from_seconds[add_leap](seconds, tz=tz, calendar=UTCCalendar)

    @staticmethod
    fn now(
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a date from `time.now()`.

        Args:
            tz: The tz to cast the result to.
            calendar: The Calendar to cast the result to.

        Returns:
            Self.
        """
        var s = time.now() // 1_000_000_000
        return Date.from_unix_epoch(s, tz).replace(calendar=calendar)

    fn strftime[format_str: StringLiteral](self) -> String:
        """Formats time into a `String`.

        Parameters:
            format_str: The chosen format.

        Returns:
            Self.

        - TODO
            - localization.
        """
        return dt_str.strftime[format_str](
            self.year, self.month, self.day, 0, 0, 0
        )

    fn strftime(self, fmt: String) -> String:
        """Formats time into a `String`.

        Args:
            fmt: The chosen format.

        Returns:
            String.

        - TODO
            - localization.
        """
        return dt_str.strftime(fmt, self.year, self.month, self.day, 0, 0, 0)

    @always_inline("nodebug")
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant formatted`String` e.g. `IsoFormat.YYYY_MM_DD` ->
         `1970-01-01` . The `Date` is first converted to UTC.

        Parameters:
            iso: The IsoFormat.

        Returns:
            String.
        """
        var date = (int(self.year), int(self.month), int(self.day))
        var time = dt_str.to_iso[iso](date[0], date[1], date[2], 0, 0, 0)
        return time[:10]

    @staticmethod
    @parameter
    fn strptime[
        format_str: StringLiteral,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Parse a `Date` from a  `String`.

        Parameters:
            format_str: The format string.
            tz: The `TimeZone` to cast the result to.
            calendar: The Calendar to cast the result to.

        Args:
            s: The string.

        Returns:
            An Optional Self.
        """
        var parsed = dt_str.strptime[format_str](s)
        if not parsed:
            return None
        var p = parsed.unsafe_take()
        return Self(p[0], p[1], p[2], tz=tz, calendar=calendar)

    @staticmethod
    @parameter
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        tz: Optional[Self._tz] = None,
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a date from an
        [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601) compliant
        `String`.

        Parameters:
            iso: The IsoFormat to parse.
            tz: Optional timezone to transform the result into
                (taking into account that the format may return with a `TimeZone`).
            calendar: The calendar to which the result will belong.

        Args:
            s: The `String` to parse; it's assumed that it is properly formatted
                i.e. no leading whitespaces or anything different to the selected
                IsoFormat.

        Returns:
            An Optional Self.
        """
        try:
            var p = dt_str.from_iso[iso, iana, pyzoneinfo, native](s)
            var dt = Self(p[0], p[1], p[2], tz=p[6], calendar=calendar)
            if tz:
                var t = tz.value()[]
                if t != dt.tz:
                    return dt.to_utc().from_utc(t)
            return dt
        except:
            return None

    @staticmethod
    fn from_hash(
        value: UInt32,
        tz: Self._tz = Self._tz(),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from a hash made by it.

        Args:
            value: The value to parse.
            tz: The `TimeZone` to designate to the result.
            calendar: The Calendar to designate to the result.

        Returns:
            Self.
        """
        var d = calendar.from_hash[_cal_hash](int(value))
        return Self(d[0], d[1], d[2], tz=tz, calendar=calendar)
