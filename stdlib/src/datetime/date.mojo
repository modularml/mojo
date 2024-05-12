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

from .timezone import TimeZone, ZoneInfo, all_zones
from .calendar import Calendar, UTCCalendar, PythonCalendar, CalendarHashes
import .dt_str


alias _calendar = PythonCalendar
alias _cal_hash = CalendarHashes(32)


@register_passable("trivial")
struct Date[iana: Optional[ZoneInfo] = all_zones](Hashable, Stringable):
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
    var tz: TimeZone[iana]
    """Tz."""
    var calendar: Calendar
    """Calendar."""

    fn __init__(
        inout self,
        year: Optional[Int] = None,
        month: Optional[Int] = None,
        day: Optional[Int] = None,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ):
        """Construct a `Date` from valid values.

        Args:
            year: Year.
            month: Month.
            day: Day.
            tz: Tz.
            calendar: Calendar.
        """
        self.year = year.or_else(int(calendar.min_year))
        self.month = month.or_else(int(calendar.min_month))
        self.day = day.or_else(int(calendar.min_day))
        self.tz = tz
        self.calendar = calendar

    fn __init__(
        inout self,
        year: Optional[UInt16] = None,
        month: Optional[UInt8] = None,
        day: Optional[UInt8] = None,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ):
        """Construct a `Date` from valid values.

        Args:
            year: Year.
            month: Month.
            day: Day.
            tz: Tz.
            calendar: Calendar.

        Notes:
            Optional extra: [`TimeZone` and DST data sources](
            http://www.iana.org/time-zones/repository/tz-link.html).
        """
        self.year = year.or_else(calendar.min_year)
        self.month = month.or_else(calendar.min_month)
        self.day = day.or_else(calendar.min_day)
        self.tz = tz
        self.calendar = calendar

    @staticmethod
    fn _from_overflow(
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from possibly overflowing values."""
        var d = Date[iana]._from_days(days, tz, calendar)
        var mon = Date[iana]._from_months(months, tz, calendar)
        var y = Date[iana]._from_years(years, tz, calendar)

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
        owned tz: Optional[TimeZone[iana]] = None,
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
            if not (
                self.calendar.max_possible_days_in_year
                == cal.max_possible_days_in_year
                and self.calendar.max_hour == cal.max_hour
                and self.calendar.max_possible_second == cal.max_possible_second
            ):
                if (
                    self.calendar.max_possible_days_in_year
                    < cal.max_possible_days_in_year
                    or self.calendar.max_hour < cal.max_hour
                    or self.calendar.max_possible_second
                    < cal.max_possible_second
                ):
                    new_self.calendar = cal
                    new_self = new_self.subtract()
            new_self.calendar = cal
        return new_self

    fn to_utc(owned self) -> Self:
        """Returns a new instance of `Self` transformed to UTC. If
        `self.tz` is UTC it returns early.

        Returns:
            Self with tz casted to UTC.
        """
        alias TZ_UTC = TimeZone[iana]()
        if self.tz == TZ_UTC:
            return self
        var new_self: Date[iana]
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

    fn from_utc(owned self, tz: TimeZone[iana]) -> Self:
        """Translate `TimeZone` from UTC. If `self.tz` is UTC
        it returns early.

        Args:
            tz: `TimeZone` to cast to.

        Returns:
            Self with tz casted to given tz.
        """
        if tz == TimeZone[iana]():
            return self
        var maxmin = self.calendar.max_minute
        var maxsec = self.calendar.max_typical_second
        var offset = tz.offset_at(self.year, self.month, self.day, 0, 0, 0)
        var of_h = int(offset[0])
        var of_m = int(offset[1])
        var amnt = int(of_h * maxmin * maxsec + of_m * maxsec)
        var new_self: Date[iana]
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
            int(self.year + years),
            int(self.month + months),
            int(self.day + days) + seconds * 24 * 60 * 60,
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
        var delta = self.delta_s(other)
        var result = int(delta[0] + delta[1])
        return self.add(seconds=result)

    @always_inline("nodebug")
    fn subtract(owned self, other: Self) -> Self:
        """Subtracts another `Date`.

        Args:
            other: Other.

        Returns:
            A `Date` with the `TimeZone` and `Calendar` of `self`.
        """
        var delta = self.delta_s(other)
        var result = int(delta[0] - delta[1])
        return self.subtract(seconds=result)

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
    fn __and__(self, other: Self) -> UInt32:
        """And.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return hash(self) & hash(other)

    @always_inline("nodebug")
    fn __or__(self, other: Self) -> UInt32:
        """Or.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return hash(self) | hash(other)

    @always_inline("nodebug")
    fn __xor__(self, other: Self) -> UInt32:
        """Xor.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return hash(self) ^ hash(other)

    @always_inline("nodebug")
    fn __int__(self) -> UInt32:
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
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from years."""
        var delta = calendar.max_year - years
        if delta > 0:
            if years > calendar.min_year:
                return Date[iana](year=years, tz=tz, calendar=calendar)
            return Date[iana]._from_years(delta)
        return Date[iana]._from_years(calendar.min_year - delta)

    @staticmethod
    fn _from_months(
        months: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from months."""
        if months <= int(calendar.max_month):
            return Date[iana](month=UInt8(months), tz=tz, calendar=calendar)
        var mon = calendar.max_month
        var dt_y = Date[iana]._from_years(
            months // int(calendar.max_month), tz, calendar
        )
        var rest = months - dt_y.year * calendar.max_month.cast[DType.uint16]()
        var dt = Date[iana](
            year=dt_y.year,
            month=mon,
            tz=tz,
            calendar=calendar,
        )
        dt += Date[iana](month=UInt8(rest), tz=tz, calendar=calendar)
        return dt

    @staticmethod
    fn _from_days[
        add_leap: Bool = False
    ](
        days: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from days."""
        var maxtdays = int(calendar.max_typical_days_in_year)
        var maxposdays = int(calendar.max_possible_days_in_year)
        var years = days // maxtdays
        var dt = Date[iana]._from_years(years, tz, calendar)
        var maxdays = maxtdays if calendar.is_leapyear(dt.year) else maxposdays
        if days <= maxdays:
            var mindays = calendar.max_days_in_month(
                calendar.min_year, calendar.min_month
            )
            if days <= int(mindays):
                return Date[iana](day=days, tz=tz, calendar=calendar)
            return Date[iana](calendar.min_year, tz=tz, calendar=calendar).add(
                days=days
            )
        var numerator = (days - maxdays)
        if add_leap:
            var leapdays = calendar.leapdays_since_epoch(
                dt.year, dt.month, dt.day
            )
            numerator += int(leapdays)
        var y = numerator // maxdays
        return Date[iana]._from_years(UInt16(y), tz, calendar)

    @staticmethod
    fn _from_hours[
        add_leap: Bool = False
    ](
        hours: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from hours."""
        var h = int(calendar.max_hour)
        if hours <= h:
            return Date[iana](calendar.min_year, tz=tz, calendar=calendar)
        var d = (hours - h) // (h + 1)
        return Date[iana]._from_days[add_leap](d, tz, calendar)

    @staticmethod
    fn _from_minutes[
        add_leap: Bool = False
    ](
        minutes: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `Date` from minutes."""
        var m = int(calendar.max_minute)
        if minutes < m:
            return Date[iana](calendar.min_year, tz=tz, calendar=calendar)
        var h = (minutes - m) // (m + 1)
        return Date[iana]._from_hours[add_leap](h, tz, calendar)

    @staticmethod
    fn from_seconds[
        add_leap: Bool = False
    ](
        seconds: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
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
        var minutes = seconds // int(calendar.max_typical_second + 1)
        var dt = Date[iana]._from_minutes(minutes, tz, calendar)
        if not add_leap:
            return dt
        var max_second = calendar.max_second(
            dt.year, dt.month, dt.day, calendar.min_hour, calendar.min_minute
        )
        var leapsecs = calendar.leapsecs_since_epoch(dt.year, dt.month, dt.day)
        var numerator = (seconds + int(leapsecs) - max_second)
        var m = int(numerator // (max_second + 1))
        return Date[iana]._from_minutes(m, tz, calendar)

    @staticmethod
    fn from_unix_epoch[
        add_leap: Bool = False
    ](seconds: Int, tz: TimeZone[iana] = TimeZone[iana]()) -> Self:
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
        return Date[iana].from_seconds[add_leap](
            seconds, tz=tz, calendar=UTCCalendar
        )

    @staticmethod
    fn now(
        tz: TimeZone[iana] = TimeZone[iana](), calendar: Calendar = _calendar
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
        var time = dt_str.to_iso(date[0], date[1], date[2], 0, 0, 0)
        return time[:10]

    @staticmethod
    @parameter
    fn strptime[
        format_str: StringLiteral,
        tz: TimeZone[iana] = TimeZone[iana](),
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
        return Date[iana](p[0], p[1], p[2], tz=tz, calendar=calendar)

    @staticmethod
    @parameter
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        tz: Optional[TimeZone[iana]] = None,
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
            var p = dt_str.from_iso[iso, iana](s)
            var dt = Date[iana](p[0], p[1], p[2], tz=p[6], calendar=calendar)
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
        tz: TimeZone[iana] = TimeZone[iana](),
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
        return Date[iana](d[0], d[1], d[2], tz=tz, calendar=calendar)
