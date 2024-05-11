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
"""Nanosecond resolution `DateTime` module.

Notes:
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
alias _cal_hash = CalendarHashes(64)

alias _max_delta = UInt16(
    UInt64.MAX_FINITE // (365 * 24 * 60 * 60 * 1_000_000_000)
)
"""Maximum year delta that fits in a UInt64 for a 
Gregorian calendar with year = 365 d * 24 h, 60 min, 60 s, 10^9 ns"""


@register_passable("trivial")
struct DateTime[iana: Optional[ZoneInfo] = all_zones](Hashable, Stringable):
    """Custom `Calendar` and `TimeZone` may be passed in.
    By default, it uses `PythonCalendar` which is a Gregorian
    calendar with its given epoch and max year:
    [0001-01-01, 9999-12-31]. Default `TimeZone` is UTC.

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
        - hour: Up to hour 256.
        - minute: Up to minute 256.
        - second: Up to second 256.
        - m_second: Up to m_second 65_536.
        - u_second: Up to u_second 65_536.
        - n_second: Up to n_second 65_536.
        - hash: 64 bits.

    - Notes:
        - The default hash that is used for logical and bitwise
            operations has only microsecond resolution.
        - The Default `DateTime` hash has only Microsecond resolution.
    """

    var year: UInt16
    var month: UInt8
    var day: UInt8
    var hour: UInt8
    var minute: UInt8
    var second: UInt8
    var m_second: UInt16
    var u_second: UInt16
    var n_second: UInt16
    # TODO: tz and calendar should be references
    var tz: TimeZone[iana]
    var calendar: Calendar

    fn __init__(
        inout self,
        year: Optional[Int] = None,
        month: Optional[Int] = None,
        day: Optional[Int] = None,
        hour: Optional[Int] = None,
        minute: Optional[Int] = None,
        second: Optional[Int] = None,
        m_second: Optional[Int] = None,
        u_second: Optional[Int] = None,
        n_second: Optional[Int] = None,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ):
        """Construct a `DateTime` from valid values."""
        self.year = year.or_else(int(calendar.min_year))
        self.month = month.or_else(int(calendar.min_month))
        self.day = day.or_else(int(calendar.min_day))
        self.hour = hour.or_else(int(calendar.min_hour))
        self.minute = minute.or_else(int(calendar.min_minute))
        self.second = second.or_else(int(calendar.min_second))
        self.m_second = m_second.or_else(int(calendar.min_milisecond))
        self.u_second = u_second.or_else(int(calendar.min_microsecond))
        self.n_second = n_second.or_else(int(calendar.min_nanosecond))
        self.tz = tz
        self.calendar = calendar

    fn __init__(
        inout self,
        year: Optional[UInt16] = None,
        month: Optional[UInt8] = None,
        day: Optional[UInt8] = None,
        hour: Optional[UInt8] = None,
        minute: Optional[UInt8] = None,
        second: Optional[UInt8] = None,
        m_second: Optional[UInt16] = None,
        u_second: Optional[UInt16] = None,
        n_second: Optional[UInt16] = None,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ):
        """Construct a `DateTime` from valid values."""
        self.year = year.or_else(calendar.min_year)
        self.month = month.or_else(calendar.min_month)
        self.day = day.or_else(calendar.min_day)
        self.hour = hour.or_else(calendar.min_hour)
        self.minute = minute.or_else(calendar.min_minute)
        self.second = second.or_else(calendar.min_second)
        self.m_second = m_second.or_else(calendar.min_milisecond)
        self.u_second = u_second.or_else(calendar.min_microsecond)
        self.n_second = n_second.or_else(calendar.min_nanosecond)
        self.tz = tz
        self.calendar = calendar

    @staticmethod
    fn _from_overflow(
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        hours: Int = 0,
        minutes: Int = 0,
        seconds: Int = 0,
        m_seconds: Int = 0,
        u_seconds: Int = 0,
        n_seconds: Int = 0,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from possibly overflowing values."""
        var ns = DateTime[iana]._from_n_seconds(n_seconds, tz, calendar)
        var us = DateTime[iana]._from_u_seconds(u_seconds, tz, calendar)
        var ms = DateTime[iana]._from_m_seconds(m_seconds, tz, calendar)
        var s = DateTime[iana].from_seconds(seconds, tz, calendar)
        var m = DateTime[iana]._from_minutes(minutes, tz, calendar)
        var h = DateTime[iana]._from_hours(hours, tz, calendar)
        var d = DateTime[iana]._from_days(days, tz, calendar)
        var mon = DateTime[iana]._from_months(months, tz, calendar)
        var y = DateTime[iana]._from_years(years, tz, calendar)

        y.year = 0 if years == 0 else y.year

        for dt in List(mon, d, h, m, s, ms, us, ns):
            if dt[].year != calendar.min_year:
                y.year += dt[].year
        y.month = mon.month
        for dt in List(d, h, m, s, ms, us, ns):
            if dt[].month != calendar.min_month:
                y.month += dt[].month
        y.day = d.day
        for dt in List(h, m, s, ms, us, ns):
            if dt[].day != calendar.min_day:
                y.day += dt[].day
        y.hour = h.hour
        for dt in List(m, s, ms, us, ns):
            if dt[].hour != calendar.min_hour:
                y.hour += dt[].hour
        y.minute = m.minute
        for dt in List(s, ms, us, ns):
            if dt[].minute != calendar.min_minute:
                y.minute += dt[].minute
        y.second = s.second
        for dt in List(ms, us, ns):
            if dt[].second != calendar.min_second:
                y.second += dt[].second
        y.m_second = ms.m_second
        for dt in List(us, ns):
            if dt[].m_second != calendar.min_milisecond:
                y.m_second += dt[].m_second
        y.u_second = us.u_second
        if ns.u_second != calendar.min_microsecond:
            y.u_second += ns.u_second
        y.n_second = ns.n_second
        return y

    @always_inline
    fn replace(
        owned self,
        *,
        owned year: Optional[UInt16] = None,
        owned month: Optional[UInt8] = None,
        owned day: Optional[UInt8] = None,
        owned hour: Optional[UInt8] = None,
        owned minute: Optional[UInt8] = None,
        owned second: Optional[UInt8] = None,
        owned m_second: Optional[UInt16] = None,
        owned u_second: Optional[UInt16] = None,
        owned n_second: Optional[UInt16] = None,
        tz: Optional[TimeZone[iana]] = None,
        calendar: Optional[Calendar] = None,
    ) -> Self:
        var new_self = self
        if year:
            new_self.year = year.unsafe_take()
        if month:
            new_self.month = month.unsafe_take()
        if day:
            new_self.day = day.unsafe_take()
        if hour:
            new_self.hour = hour.unsafe_take()
        if minute:
            new_self.minute = minute.unsafe_take()
        if second:
            new_self.second = second.unsafe_take()
        if m_second:
            new_self.m_second = m_second.unsafe_take()
        if u_second:
            new_self.u_second = u_second.unsafe_take()
        if n_second:
            new_self.n_second = n_second.unsafe_take()
        if tz:
            new_self.tz = tz.value()[]
        if calendar:
            var cal = calendar.value()[]
            if not (
                self.calendar.max_typical_days_in_year
                == cal.max_typical_days_in_year
                and self.calendar.max_hour == cal.max_hour
                and self.calendar.max_typical_second == cal.max_typical_second
            ):
                if (
                    self.calendar.max_typical_days_in_year
                    < cal.max_typical_days_in_year
                    or self.calendar.max_hour < cal.max_hour
                    or self.calendar.max_typical_second < cal.max_typical_second
                ):
                    new_self.calendar = cal
                    new_self = new_self.subtract()
            new_self.calendar = cal
        return new_self

    fn to_utc(owned self) -> Self:
        """Returns a new instance of `Self` transformed to UTC. If
        `self.tz` is UTC it returns early."""
        alias TZ_UTC = TimeZone[iana]()
        if self.tz == TZ_UTC:
            return self
        var new_self = self
        var offset = self.tz.offset_at(
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
        var of_h = int(offset[0])
        var of_m = int(offset[1])
        if offset[2] == -1:
            new_self = self.add(hours=of_h, minutes=of_m)
        else:
            new_self = self.subtract(hours=of_h, minutes=of_m)
        return new_self.replace(tz=TZ_UTC)

    fn from_utc(owned self, tz: TimeZone[iana]) -> Self:
        """Translate `TimeZone` from UTC. If `self.tz` is UTC
        it returns early."""
        if tz == TimeZone[iana]():
            return self
        var offset = tz.offset_at(
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
        var h = int(offset[0])
        var m = int(offset[1])
        var new_self: DateTime[iana]
        if offset[2] == 1:
            new_self = self.add(hours=h, minutes=m)
        else:
            new_self = self.subtract(hours=h, minutes=m)
        var leapsecs = int(
            new_self.calendar.leapsecs_since_epoch(
                new_self.year, new_self.month, new_self.day
            )
        )
        return new_self.add(seconds=leapsecs).replace(tz=tz)

    @always_inline
    fn n_seconds_since_epoch(self) -> UInt64:
        """Nanoseconds since the begining of the calendar's epoch.
        Can only represent up to ~ 580 years since epoch start."""
        return self.calendar.n_seconds_since_epoch(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.m_second,
            self.u_second,
            self.n_second,
        )

    @always_inline
    fn seconds_since_epoch(self) -> UInt64:
        """Seconds since the begining of the calendar's epoch."""
        return self.calendar.seconds_since_epoch(
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )

    fn delta_ns(self, other: Self) -> (UInt64, UInt64, UInt16, UInt8):
        """Calculates the nanoseconds for `self` and other, creating
        a reference calendar to keep nanosecond resolution.

        Returns:
            - self_ns: Nanoseconds from `self` to created temp calendar.
            - other_ns: Nanoseconds from other to created temp calendar.
            - overflow: the amount of years added / subtracted from `self`
                to make the temp calendar. This occurs if the difference
                in years is bigger than ~ 580 (Gregorian years).
            - sign: {1, -1} if the overflow was added or subtracted.
        """
        var s = self
        var o = other.replace(calendar=self.calendar)

        var overflow: UInt16 = 0
        var sign: UInt8 = 1
        var year = s.year
        if s.year < o.year:
            sign = -1
            while o.year - year > _max_delta:
                year -= _max_delta
                overflow += _max_delta
        else:
            while year - o.year > _max_delta:
                year -= _max_delta
                overflow += _max_delta

        if s.tz != o.tz:
            s = s.to_utc()
            o = o.to_utc()
        var cal = Calendar.from_year(year)
        var self_ns = s.replace(calendar=cal).n_seconds_since_epoch()
        var other_ns = o.replace(calendar=cal).n_seconds_since_epoch()
        return self_ns, other_ns, overflow, sign

    fn add(
        owned self,
        *,
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        hours: Int = 0,
        minutes: Int = 0,
        seconds: Int = 0,
        m_seconds: Int = 0,
        u_seconds: Int = 0,
        n_seconds: Int = 0,
    ) -> Self:
        """Recursively evaluated function to build a valid `DateTime`
        according to its calendar.

        Notes:
            On overflow, the `DateTime` starts from the beginning of the
            calendar's epoch and keeps evaluating until valid
        """
        var dt = self._from_overflow(
            int(self.year + years),
            int(self.month + months),
            int(self.day + days),
            int(self.hour + hours),
            int(self.minute + minutes),
            int(self.second + seconds),
            int(self.m_second + m_seconds),
            int(self.u_second + u_seconds),
            int(self.n_second + n_seconds),
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
        var maxsec = dt.calendar.max_second(
            dt.year, dt.month, dt.day, dt.hour, dt.minute
        )
        if dt.second > maxsec:
            dt = dt.replace(second=maxsec).add(
                days=1, seconds=int(dt.second - maxsec)
            )
        return dt

    fn subtract(
        owned self,
        *,
        years: Int = 0,
        months: Int = 0,
        days: Int = 0,
        hours: Int = 0,
        minutes: Int = 0,
        seconds: Int = 0,
        m_seconds: Int = 0,
        u_seconds: Int = 0,
        n_seconds: Int = 0,
    ) -> Self:
        """Recursively evaluated function to build a valid `DateTime`
        according to its calendar.

        Notes:
            On overflow, the `DateTime` goes to the end of the
            calendar's epoch and keeps evaluating until valid
        """
        var dt = self._from_overflow(
            int(self.year - years),
            int(self.month - months),
            int(self.day - days),
            int(self.hour - hours),
            int(self.minute - minutes),
            int(self.second - seconds),
            int(self.m_second - m_seconds),
            int(self.u_second - u_seconds),
            int(self.n_second - n_seconds),
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
        var minhour = dt.calendar.min_hour
        if dt.hour < minhour:
            dt = dt.replace(hour=dt.calendar.max_hour).subtract(
                days=1, hours=int(minhour - dt.hour)
            )
        var minmin = dt.calendar.min_minute
        if dt.minute < minmin:
            dt = dt.replace(minute=dt.calendar.max_minute).subtract(
                days=1, minutes=int(minmin - dt.minute)
            )
        var minsec = dt.calendar.min_second
        if dt.second < minsec:
            var sec = dt.calendar.max_second(
                dt.year, dt.month, dt.day, dt.hour, dt.minute
            )
            dt = dt.replace(second=sec).subtract(
                days=1, seconds=int(minsec - dt.second)
            )
        var minmsec = dt.calendar.min_milisecond
        if dt.m_second < minmsec:
            dt = dt.replace(m_second=UInt16(999)).subtract(
                days=1, m_seconds=int(minmsec - dt.m_second)
            )
        var minusec = dt.calendar.min_microsecond
        if dt.u_second < minusec:
            dt = dt.replace(u_second=UInt16(999)).subtract(
                days=1, u_seconds=int(minusec - dt.u_second)
            )
        var minnsec = dt.calendar.min_nanosecond
        if dt.n_second < minnsec:
            dt = dt.replace(n_second=UInt16(999)).subtract(
                days=1, n_seconds=int(minnsec - dt.n_second)
            )
        return dt

    fn add(owned self, other: Self) -> Self:
        """Adds another `DateTime`.

        Returns:
            A `DateTime` with the `TimeZone` and `Calendar` of `self`.
        """
        var delta = self.delta_ns(other)
        var result = int(delta[0] + delta[1])
        return self.add(years=int(delta[2]), seconds=result)

    fn subtract(owned self, other: Self) -> Self:
        """Subtracts another `DateTime`.

        Returns:
            A `DateTime` with the `TimeZone` and `Calendar` of `self`.
        """
        var delta = self.delta_ns(other)
        var result = int(delta[0] - delta[1])
        return self.subtract(years=int(delta[2]), seconds=result)

    fn __add__(owned self, other: Self) -> Self:
        return self.add(other)

    fn __sub__(owned self, other: Self) -> Self:
        return self.subtract(other)

    fn __iadd__(inout self, owned other: Self):
        self = self.add(other)

    fn __isub__(inout self, owned other: Self):
        self = self.subtract(other)

    fn dayofweek(self) -> UInt8:
        """Calculates the day of the week for a `DateTime`.

        Returns:
            - day: Day of the week: [0, 6] (monday - sunday) (default).
        """
        return self.calendar.dayofweek(self.year, self.month, self.day)

    fn dayofyear(self) -> UInt16:
        """Calculates the day of the year for a `DateTime`.

        Returns:
            - day: Day of the year: [1, 366] (for gregorian calendar).
        """
        return self.calendar.dayofyear(self.year, self.month, self.day)

    fn leapsecs_since_epoch(self) -> UInt32:
        """Cumulative leap seconds since the calendar's epoch start."""
        var dt = self.to_utc()
        return dt.calendar.leapsecs_since_epoch(dt.year, dt.month, dt.day)

    fn __hash__(self) -> Int:
        return self.calendar.hash[_cal_hash](
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.m_second,
            self.u_second,
            self.n_second,
        )

    fn __eq__(self, other: Self) -> Bool:
        return hash(self) == hash(other)

    fn __ne__(self, other: Self) -> Bool:
        return hash(self) != hash(other)

    fn __gt__(self, other: Self) -> Bool:
        return hash(self) > hash(other)

    fn __ge__(self, other: Self) -> Bool:
        return hash(self) >= hash(other)

    fn __le__(self, other: Self) -> Bool:
        return hash(self) <= hash(other)

    fn __lt__(self, other: Self) -> Bool:
        return hash(self) < hash(other)

    fn __and__(self, other: Self) -> UInt64:
        return hash(self) & hash(other)

    fn __or__(self, other: Self) -> UInt64:
        return hash(self) | hash(other)

    fn __xor__(self, other: Self) -> UInt64:
        return hash(self) ^ hash(other)

    fn __int__(self) -> UInt64:
        return hash(self)

    fn __str__(self) -> String:
        return self.to_iso()

    @staticmethod
    fn _from_years(
        years: UInt16,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from years."""
        var delta = calendar.max_year - years
        if delta > 0:
            if years > calendar.min_year:
                return DateTime[iana](year=years, tz=tz, calendar=calendar)
            return DateTime[iana]._from_years(delta)
        return DateTime[iana]._from_years(calendar.min_year - delta)

    @staticmethod
    fn _from_months(
        months: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from months."""
        if months <= int(calendar.max_month):
            return DateTime[iana](month=UInt8(months), tz=tz, calendar=calendar)
        var mon = calendar.max_month
        var dt_y = DateTime[iana]._from_years(
            months // int(calendar.max_month), tz, calendar
        )
        var rest = months - dt_y.year * calendar.max_month.cast[DType.uint16]()
        var dt = DateTime[iana](
            year=dt_y.year,
            month=mon,
            tz=tz,
            calendar=calendar,
        )
        dt += DateTime[iana](month=UInt8(rest), tz=tz, calendar=calendar)
        return dt

    @staticmethod
    fn _from_days[
        add_leap: Bool = False
    ](
        days: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from days."""
        var maxtdays = int(calendar.max_typical_days_in_year)
        var maxposdays = int(calendar.max_possible_days_in_year)
        var years = days // maxtdays
        var dt = DateTime[iana]._from_years(years, tz, calendar)
        var maxdays = maxtdays if calendar.is_leapyear(dt.year) else maxposdays
        if days <= maxdays:
            var mindays = calendar.max_days_in_month(
                calendar.min_year, calendar.min_month
            )
            if days <= int(mindays):
                return DateTime[iana](day=days, tz=tz, calendar=calendar)
            return DateTime[iana](
                calendar.min_year, tz=tz, calendar=calendar
            ).add(days=days)
        var numerator = (days - maxdays)
        if add_leap:
            var leapdays = calendar.leapdays_since_epoch(
                dt.year, dt.month, dt.day
            )
            numerator += int(leapdays)
        var y = numerator // maxdays
        var rest = numerator % maxdays
        dt = DateTime[iana]._from_years(UInt16(y), tz, calendar)
        return dt.replace(day=UInt8(rest))

    @staticmethod
    fn _from_hours[
        add_leap: Bool = False
    ](
        hours: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from hours."""
        var h = int(calendar.max_hour)
        if hours <= h:
            return DateTime[iana](hour=UInt8(hours), tz=tz, calendar=calendar)
        var d = (hours - h) // (h + 1)
        var rest = (hours - h) % (h + 1)
        var dt = DateTime[iana]._from_days[add_leap](d, tz, calendar)
        return dt.replace(hour=UInt8(rest))

    @staticmethod
    fn _from_minutes[
        add_leap: Bool = False
    ](
        minutes: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from minutes."""
        var m = int(calendar.max_minute)
        if minutes < m:
            return DateTime[iana](minute=minutes, tz=tz, calendar=calendar)
        var h = (minutes - m) // (m + 1)
        var rest = (minutes - m) % (m + 1)
        var dt = DateTime[iana]._from_hours[add_leap](h, tz, calendar)
        return dt.replace(minute=UInt8(rest))

    @staticmethod
    fn from_seconds[
        add_leap: Bool = False
    ](
        seconds: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from seconds.

        Parameters:
            add_leap: Whether to add the leap seconds and leap days
                since the start of the calendar's epoch.
        """
        var minutes = seconds // int(calendar.max_typical_second + 1)
        var dt = DateTime[iana]._from_minutes(minutes, tz, calendar)
        var max_second = calendar.max_second(
            dt.year, dt.month, dt.day, dt.hour, dt.minute
        )
        if dt.second <= max_second:
            return dt.replace(second=UInt8(seconds))
        var numerator = (seconds - max_second)
        if add_leap:
            var leapsecs = calendar.leapsecs_since_epoch(
                dt.year, dt.month, dt.day
            )
            numerator += int(leapsecs)
        var m = numerator // (max_second + 1)
        var rest = numerator % (max_second + 1)
        dt = DateTime[iana]._from_minutes(int(m), tz, calendar)
        return dt.replace(second=rest)

    @staticmethod
    fn _from_m_seconds(
        m_seconds: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from miliseconds."""
        var ms = int(calendar.max_milisecond)
        if m_seconds <= ms:
            return DateTime[iana](
                m_second=UInt16(m_seconds), tz=tz, calendar=calendar
            )
        var s = (m_seconds - ms) // (ms + 1)
        var rest = (m_seconds - ms) % (ms + 1)
        var dt = DateTime[iana].from_seconds(s, tz, calendar)
        return dt.replace(m_second=UInt16(rest))

    @staticmethod
    fn _from_u_seconds(
        u_seconds: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from microseconds."""
        var us = int(calendar.max_microsecond)
        if u_seconds <= us:
            return DateTime[iana](
                u_second=UInt16(u_seconds), tz=tz, calendar=calendar
            )
        var ms = (u_seconds - us) // (us + 1)
        var rest = (u_seconds - us) % (us + 1)
        var dt = DateTime[iana]._from_m_seconds(ms, tz, calendar)
        return dt.replace(u_second=UInt16(rest))

    @staticmethod
    fn _from_n_seconds(
        n_seconds: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from nanoseconds."""
        var ns = int(calendar.max_nanosecond)
        if n_seconds <= ns:
            return DateTime[iana](
                n_second=UInt16(n_seconds), tz=tz, calendar=calendar
            )
        var us = (n_seconds - ns) // (ns + 1)
        var rest = (n_seconds - ns) % (ns + 1)
        var dt = DateTime[iana]._from_u_seconds(us, tz, calendar)
        return dt.replace(n_second=UInt16(rest))

    @staticmethod
    fn from_unix_epoch[
        add_leap: Bool = False
    ](seconds: Int, tz: TimeZone[iana] = TimeZone[iana]()) -> Self:
        """Construct a `DateTime` from the seconds since the Unix Epoch
        1970-01-01. Adding the cumulative leap seconds since 1972
        to the given date.

        Parameters:
            add_leap: Whether to add the leap seconds and leap days
                since the start of the calendar's epoch.
        """
        return DateTime[iana].from_seconds[add_leap](
            seconds, tz=tz, calendar=UTCCalendar
        )

    @staticmethod
    fn now(
        tz: TimeZone[iana] = TimeZone[iana](), calendar: Calendar = _calendar
    ) -> Self:
        """Construct a datetime from `time.now()`."""
        var ns = time.now()
        var us: UInt16 = ns // 1_000
        var ms: UInt16 = ns // 1_000_000
        var s = ns // 1_000_000_000
        var dt = DateTime[iana].from_unix_epoch(s, tz).replace(
            calendar=calendar
        )
        return dt.replace(m_second=ms, u_second=us, n_second=UInt16(ns))

    fn strftime[format_str: StringLiteral](self) -> String:
        """Formats time into a `String`.

        - TODO
            - localization.
        """
        return dt_str.strftime[format_str](
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )

    fn strftime(self, fmt: String) -> String:
        """Formats time into a `String`.

        - TODO
            - localization.
        """
        return dt_str.strftime(
            fmt,
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
        )

    fn __format__(self, fmt: String) -> String:
        return self.strftime(fmt)

    @parameter
    fn to_iso[iso: dt_str.IsoFormat = dt_str.IsoFormat()](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYY_MM_DD_T_MM_HH_SSTimeZone[iana]()D`.
        e.g. `1970-01-01T00:00:00+00:00` ."""
        var date = (int(self.year), int(self.month), int(self.day))
        var hour = (int(self.hour), int(self.minute), int(self.second))
        var time = dt_str.to_iso(
            date[0], date[1], date[2], hour[0], hour[1], hour[2]
        )
        return time + self.tz.to_iso()

    @parameter
    fn to_iso_compact[
        iso: dt_str.IsoFormat = dt_str.IsoFormat()
    ](self) -> String:
        """Return an [ISO 8601](https://es.wikipedia.org/wiki/ISO_8601)
        compliant `String` in the form `IsoFormat.YYYYMMDDMMHHSS`.
        e.g. `19700101000000` . The `DateTime` is first converted to UTC."""
        var utc_s = self.to_utc()
        var date = (int(utc_s.year), int(utc_s.month), int(utc_s.day))
        var hour = (int(utc_s.hour), int(utc_s.minute), int(utc_s.second))
        var time = dt_str.to_iso_compact(
            date[0], date[1], date[2], hour[0], hour[1], hour[2]
        )
        return time

    @staticmethod
    fn strptime[
        format_str: StringLiteral,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ](s: String,) -> Optional[Self]:
        """Parse a `DateTime` from a  `String`."""
        var parsed = dt_str.strptime[format_str](s)
        if not parsed:
            return None
        var p = parsed.unsafe_take()
        return DateTime[iana](
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
            p[6],
            p[7],
            p[8],
            tz=tz,
            calendar=calendar,
        )

    @staticmethod
    @parameter
    fn from_iso[
        iso: dt_str.IsoFormat = dt_str.IsoFormat(),
        tz: Optional[TimeZone[iana]] = None,
        calendar: Calendar = _calendar,
    ](s: String) -> Optional[Self]:
        """Construct a datetime from an
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
        """
        try:
            var p = dt_str.from_iso[iso, iana](s)
            var dt = DateTime[iana](
                p[0], p[1], p[2], p[3], p[4], p[5], tz=p[6], calendar=calendar
            )
            if tz:
                var t = tz.value()[]
                if t != dt.tz:
                    return dt.to_utc().from_utc(t)
            return dt
        except:
            return None

    @staticmethod
    fn from_hash(
        value: Int,
        tz: TimeZone[iana] = TimeZone[iana](),
        calendar: Calendar = _calendar,
    ) -> Self:
        """Construct a `DateTime` from a hash made by it.
        Nanoseconds are set to the calendar's minimum."""
        var d = calendar.from_hash(value)
        return DateTime[iana](
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
            d[5],
            d[6],
            d[7],
            tz=tz,
            calendar=calendar,
        )
