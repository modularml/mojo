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
"""`Calendar` module."""

from utils import Variant

# TODO: other calendars besides Gregorian
alias PythonCalendar = Calendar()
"""The default Python proleptic Gregorian calendar, goes from [0001-01-01, 9999-12-31]"""
alias UTCCalendar = Calendar(Gregorian(min_year=1970))
"""The leap year and leap second aware UTC calendar, goes from [1970-01-01, 9999-12-31]"""
alias UTCFastCal = Calendar(UTCFast())
"""UTC calendar for the fast module."""
alias _date = (UInt16, UInt8, UInt8, UInt8, UInt8, UInt8, UInt16, UInt16)
"""Alias for the date type. Up to microsecond resolution."""


struct CalendarHashes:
    """Hashing definitions. Up to microsecond resolution for
    the 64bit hash. Each calendar implementation can still
    override with its own definitions."""

    alias UINT8 = 8
    """Hash width UINT8"""
    alias UINT16 = 16
    """Hash width UINT16"""
    alias UINT32 = 32
    """Hash width UINT32"""
    alias UINT64 = 64
    """Hash width UINT64"""
    var selected: Int
    """What hash width was selected."""

    alias _10b = 0b1111111111
    alias _9b = 0b111111111
    alias _6b = 0b111111
    alias _5b = 0b11111
    alias _4b = 0b1111
    alias _2b = 0b11

    alias shift_64_y = (5 + 5 + 5 + 6 + 6 + 10 + 10)
    """Up to 131_072 years"""
    alias shift_64_mon = (5 + 5 + 6 + 6 + 10 + 10)
    """Up to 32 months"""
    alias shift_64_d = (5 + 6 + 6 + 10 + 10)
    """Up to 32 days"""
    alias shift_64_h = (6 + 6 + 10 + 10)
    """Up to 32 hours"""
    alias shift_64_m = (6 + 10 + 10)
    """Up to 64 minutes"""
    alias shift_64_s = (10 + 10)
    """Up to 64 seconds"""
    alias shift_64_ms = 10
    """Up to 1024 m_seconds"""
    alias shift_64_us = 0
    """Up to 1024 u_seconds"""
    alias mask_64_y: UInt64 = CalendarHashes._10b << CalendarHashes.shift_64_y
    alias mask_64_mon: UInt64 = CalendarHashes._5b << CalendarHashes.shift_64_mon
    alias mask_64_d: UInt64 = CalendarHashes._5b << CalendarHashes.shift_64_d
    alias mask_64_h: UInt64 = CalendarHashes._5b << CalendarHashes.shift_64_h
    alias mask_64_m: UInt64 = CalendarHashes._6b << CalendarHashes.shift_64_m
    alias mask_64_s: UInt64 = CalendarHashes._6b << CalendarHashes.shift_64_s
    alias mask_64_ms: UInt64 = CalendarHashes._10b << CalendarHashes.shift_64_ms
    alias mask_64_us: UInt64 = CalendarHashes._10b << CalendarHashes.shift_64_us

    alias shift_32_y = (4 + 5 + 5 + 6)
    """Up to 4096 years"""
    alias shift_32_mon = (5 + 5 + 6)
    """Up to 16 months"""
    alias shift_32_d = (5 + 6)
    """Up to 32 days"""
    alias shift_32_h = 6
    """Up to 32 hours"""
    alias shift_32_m = 0
    """Up to 64 minutes"""
    alias mask_32_y: UInt32 = CalendarHashes._10b << CalendarHashes.shift_32_y
    alias mask_32_d: UInt32 = CalendarHashes._4b << CalendarHashes.shift_32_mon
    alias mask_32_mon: UInt32 = CalendarHashes._5b << CalendarHashes.shift_32_d
    alias mask_32_h: UInt32 = CalendarHashes._5b << CalendarHashes.shift_32_h
    alias mask_32_m: UInt32 = CalendarHashes._6b << CalendarHashes.shift_32_m

    alias shift_16_y = (9 + 5)
    """Up to 4 years"""
    alias shift_16_d = 5
    """Up to 512 days"""
    alias shift_16_h = 0
    """Up to 32 hours"""
    alias mask_16_y: UInt16 = CalendarHashes._2b << CalendarHashes.shift_16_y
    alias mask_16_d: UInt16 = CalendarHashes._9b << CalendarHashes.shift_16_d
    alias mask_16_h: UInt16 = CalendarHashes._5b << CalendarHashes.shift_16_h

    alias shift_8_d = 5
    """Up to 8 days"""
    alias shift_8_h = 0
    """Up to 32 hours"""
    alias mask_8_d: UInt8 = CalendarHashes._5b << CalendarHashes.shift_8_d
    alias mask_8_h: UInt8 = CalendarHashes._5b << CalendarHashes.shift_8_h

    fn __init__(inout self, selected: Int = 64):
        """Construct a `CalendarHashes`.

        Args:
            selected: The selected hash bit width.
        """
        debug_assert(
            selected == self.UINT8
            or selected == self.UINT16
            or selected == self.UINT32
            or selected == self.UINT64,
            msg="there is no such hash size",
        )
        self.selected = selected


# TODO: once traits with attributes and impl are ready Calendar will replace
# a bunch of this file
trait _Calendarized:
    fn is_leapyear(self, year: UInt16) -> Bool:
        ...

    fn is_leapsec(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
    ) -> Bool:
        ...

    fn dayofweek(self, year: UInt16, month: UInt8, day: UInt8) -> UInt8:
        ...

    fn dayofyear(self, year: UInt16, month: UInt8, day: UInt8) -> UInt16:
        ...

    fn max_second(
        self, year: UInt16, month: UInt8, day: UInt8, hour: UInt8, minute: UInt8
    ) -> UInt8:
        ...

    fn max_days_in_month(self, year: UInt16, month: UInt8) -> UInt8:
        ...

    fn monthrange(self, year: UInt16, month: UInt8) -> (UInt8, UInt8):
        ...

    fn leapsecs_since_epoch(
        self, year: UInt16, month: UInt8, day: UInt8
    ) -> UInt32:
        ...

    fn leapdays_since_epoch(
        self, year: UInt16, month: UInt8, day: UInt8
    ) -> UInt32:
        ...

    fn seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
    ) -> UInt64:
        ...

    fn m_seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
        m_second: UInt8,
    ) -> UInt64:
        ...

    fn n_seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
        m_second: UInt16,
        u_second: UInt16,
        n_second: UInt16,
    ) -> UInt64:
        ...

    @staticmethod
    fn from_year(year: UInt16) -> Self:
        ...

    fn hash[
        cal_hash: CalendarHashes
    ](
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
        m_second: UInt16,
        u_second: UInt16,
        n_second: UInt16,
    ) -> Int:
        ...

    fn from_hash[cal_hash: CalendarHashes](self, value: Int) -> _date:
        ...


@register_passable("trivial")
struct Calendar(_Calendarized):
    """`Calendar` interface."""

    var max_year: UInt16
    """Maximum value of years."""
    var max_typical_days_in_year: UInt16
    """Maximum typical value of days in a year (no leaps)."""
    var max_possible_days_in_year: UInt16
    """Maximum possible value of days in a year (with leaps)."""
    var max_month: UInt8
    """Maximum value of months in a year."""
    var max_hour: UInt8
    """Maximum value of hours in a day."""
    var max_minute: UInt8
    """Maximum value of minutes in an hour."""
    var max_typical_second: UInt8
    """Maximum typical value of seconds in a minute (no leaps)."""
    var max_possible_second: UInt8
    """Maximum possible value of seconds in a minute (with leaps)."""
    var max_milisecond: UInt8
    """Maximum value of miliseconds in a second."""
    var max_microsecond: UInt8
    """Maximum value of microseconds in a second."""
    var max_nanosecond: UInt8
    """Maximum value of nanoseconds in a second."""
    var min_year: UInt16
    """Default minimum year in the calendar."""
    var min_month: UInt8
    """Default minimum month."""
    var min_day: UInt8
    """Default minimum day."""
    var min_hour: UInt8
    """Default minimum hour."""
    var min_minute: UInt8
    """Default minimum minute."""
    var min_second: UInt8
    """Default minimum second."""
    var min_milisecond: UInt16
    """Default minimum milisecond."""
    var min_microsecond: UInt16
    """Default minimum microsecond."""
    var min_nanosecond: UInt16
    """Default minimum nanosecond."""
    alias _monthdays = List[UInt8]()
    """An array with the amount of days each month contains without 
    leap values. It's assumed that `len(monthdays) == max_month`."""
    var _implementation: Gregorian

    fn __init__(
        inout self, owned impl: Variant[Gregorian, UTCFast] = Gregorian()
    ):
        """Construct a `Calendar`.

        Args:
            impl: Calendar implementation.
        """
        var imp = impl.unsafe_take[Gregorian]()
        self.max_year = imp.max_year
        self.max_typical_days_in_year = imp.max_typical_days_in_year
        self.max_possible_days_in_year = imp.max_possible_days_in_year
        self.max_month = imp.max_month
        self.max_hour = imp.max_hour
        self.max_minute = imp.max_minute
        self.max_typical_second = imp.max_typical_second
        self.max_possible_second = imp.max_possible_second
        self.max_milisecond = imp.max_milisecond
        self.max_microsecond = imp.max_microsecond
        self.max_nanosecond = imp.max_nanosecond
        self.min_year = imp.min_year
        self.min_month = imp.min_month
        self.min_day = imp.min_day
        self.min_hour = imp.min_hour
        self.min_minute = imp.min_minute
        self.min_second = imp.min_second
        self.min_milisecond = imp.min_milisecond
        self.min_microsecond = imp.min_microsecond
        self.min_nanosecond = imp.min_nanosecond
        self._implementation = imp

    @always_inline
    fn dayofweek(self, year: UInt16, month: UInt8, day: UInt8) -> UInt8:
        """Calculates the day of the week for a given date.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            - day: Day of the week: [0, 6] (monday - sunday).
        """
        return self._implementation.dayofweek(year, month, day)

    @always_inline
    fn dayofyear(self, year: UInt16, month: UInt8, day: UInt8) -> UInt16:
        """Calculates the day of the year for a given date.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            - day: Day of the year: [1, 366] (for gregorian calendar).
        """
        return self._implementation.dayofyear(year, month, day)

    @always_inline
    fn max_days_in_month(self, year: UInt16, month: UInt8) -> UInt8:
        """The maximum amount of days in a given month.

        Args:
            year: Year.
            month: Month.

        Returns:
            The amount of days.
        """
        return self._implementation.max_days_in_month(year, month)

    @always_inline
    fn monthrange(self, year: UInt16, month: UInt8) -> (UInt8, UInt8):
        """Calculates the day of the week and the day of the month
        that a month in a given year ends.

        Args:
            year: Year.
            month: Month.

        Returns:
            - dayofweek: Day of the week.
            - dayofmonth: Day of the month.
        """
        return self._implementation.monthrange(year, month)

    @always_inline
    fn max_second(
        self, year: UInt16, month: UInt8, day: UInt8, hour: UInt8, minute: UInt8
    ) -> UInt8:
        """The maximum amount of seconds in a minute (usually 59).

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.

        Returns:
            The amount.
        """
        return self._implementation.max_second(year, month, day, hour, minute)

    @always_inline
    fn is_leapyear(self, year: UInt16) -> Bool:
        """Whether the year is a leap year.

        Args:
            year: Year.

        Returns:
            Bool.
        """
        return self._implementation.is_leapyear(year)

    @always_inline
    fn is_leapsec(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
    ) -> Bool:
        """Whether the second is a leap second.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.

        Returns:
            Bool.
        """
        return self._implementation.is_leapsec(
            year, month, day, hour, minute, second
        )

    @always_inline
    fn leapsecs_since_epoch(
        self, year: UInt16, month: UInt8, day: UInt8
    ) -> UInt32:
        """Cumulative leap seconds since the calendar's epoch start.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            The amount.
        """
        return self._implementation.leapsecs_since_epoch(year, month, day)

    @always_inline
    fn leapdays_since_epoch(
        self, year: UInt16, month: UInt8, day: UInt8
    ) -> UInt32:
        """Cumulative leap days since the calendar's epoch start.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            The amount.
        """
        return self._implementation.leapdays_since_epoch(year, month, day)

    @always_inline
    fn seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
    ) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.

        Returns:
            The amount.
        """
        return self._implementation.seconds_since_epoch(
            year, month, day, hour, minute, second
        )

    fn m_seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
        m_second: UInt8,
    ) -> UInt64:
        """Miliseconds since the begining of the calendar's epoch.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: Milisecond.

        Returns:
            The amount.
        """
        _ = m_second
        return self._implementation.seconds_since_epoch(
            year, month, day, hour, minute, second
        )

    @always_inline
    fn n_seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
        m_second: UInt16,
        u_second: UInt16,
        n_second: UInt16,
    ) -> UInt64:
        """Nanoseconds since the begining of the calendar's epoch.
        Can only represent up to ~ 580 years since epoch start.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.
            u_second: U_second.
            n_second: N_second.

        Returns:
            The amount.
        """
        return self._implementation.n_seconds_since_epoch(
            year, month, day, hour, minute, second, m_second, u_second, n_second
        )

    @staticmethod
    @always_inline
    fn from_year(year: UInt16) -> Self:
        """Get a Calendar with min_year=year.

        Args:
            year: The year.

        Returns:
            Self.
        """
        return Calendar.from_year[Gregorian](year)

    @staticmethod
    @always_inline
    fn from_year[
        T: Variant[Gregorian, UTCFast] = Gregorian
    ](year: UInt16) -> Self:
        """Get a Calendar with min_year=year.

        Parameters:
            T: The type of Calendar.

        Args:
            year: The year.

        Returns:
            Self.
        """
        if T.isa[Gregorian]():
            return Self(Gregorian().from_year(year))
        elif T.isa[UTCFast]():
            return Self(UTCFast().from_year(year))
        return Self()

    fn hash[
        cal_hash: CalendarHashes = CalendarHashes()
    ](
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8 = 0,
        minute: UInt8 = 0,
        second: UInt8 = 0,
        m_second: UInt16 = 0,
        u_second: UInt16 = 0,
        n_second: UInt16 = 0,
    ) -> Int:
        """Hash the given values according to the calendar's component
        lengths bitshifted, BigEndian (i.e. yyyymmdd...).

        Parameters:
            cal_hash: The hashing schema (CalendarHashes).

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.
            u_second: U_second.
            n_second: N_second.

        Returns:
            The hash.
        """
        return self._implementation.hash[cal_hash](
            year, month, day, hour, minute, second, m_second, u_second, n_second
        )

    @always_inline
    fn from_hash[
        cal_hash: CalendarHashes = CalendarHashes()
    ](self, value: Int) -> _date:
        """Build a date from a hashed value.

        Parameters:
            cal_hash: The hashing schema (CalendarHashes).

        Args:
            value: The Hash.

        Returns:
            Tuple containing date data.
        """
        return self._implementation.from_hash[cal_hash](value)


@register_passable("trivial")
struct Gregorian(_Calendarized):
    """`Gregorian` Calendar."""

    var max_year: UInt16
    """Maximum value of years."""
    alias max_typical_days_in_year: UInt16 = 365
    """Maximum typical value of days in a year (no leaps)."""
    alias max_possible_days_in_year: UInt16 = 366
    """Maximum possible value of days in a year (with leaps)."""
    alias max_month: UInt8 = 12
    """Maximum value of months in a year."""
    alias max_hour: UInt8 = 23
    """Maximum value of hours in a day."""
    alias max_minute: UInt8 = 59
    """Maximum value of minutes in an hour."""
    alias max_typical_second: UInt8 = 59
    """Maximum typical value of seconds in a minute (no leaps)."""
    alias max_possible_second: UInt8 = 60
    """Maximum possible value of seconds in a minute (with leaps)."""
    alias max_milisecond: UInt8 = 999
    """Maximum value of miliseconds in a second."""
    alias max_microsecond: UInt8 = 999
    """Maximum value of microseconds in a second."""
    alias max_nanosecond: UInt8 = 999
    """Maximum value of nanoseconds in a second."""
    var min_year: UInt16
    """Default minimum year in the calendar."""
    alias min_month: UInt8 = 1
    """Default minimum month."""
    alias min_day: UInt8 = 1
    """Default minimum day."""
    alias min_hour: UInt8 = 0
    """Default minimum hour."""
    alias min_minute: UInt8 = 0
    """Default minimum minute."""
    alias min_second: UInt8 = 0
    """Default minimum second."""
    alias min_milisecond: UInt16 = 0
    """Default minimum milisecond."""
    alias min_microsecond: UInt16 = 0
    """Default minimum microsecond."""
    alias min_nanosecond: UInt16 = 0
    """Default minimum nanosecond."""
    alias _monthdays: List[UInt8] = List[UInt8](
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    )
    """An array with the amount of days each month contains without 
    leap values. It's assumed that `len(monthdays) == max_month`."""

    fn __init__(
        inout self,
        *,
        max_year: UInt16 = 9999,
        min_year: UInt16 = 1,
    ):
        """Construct a `Gregorian` Calendar from values.
        
        Args:
            max_year: Max year (epoch end).
            min_year: Min year (epoch start).
        """
        self.max_year = max_year
        self.min_year = min_year

    @always_inline
    fn monthrange(self, year: UInt16, month: UInt8) -> (UInt8, UInt8):
        """Calculates the day of the week and the day of the month
        that a month in a given year ends.

        Args:
            year: Year.
            month: Month.

        Returns:
            - dayofweek: Day of the week.
            - dayofmonth: Day of the month.
        """
        _ = self, year, month
        # TODO
        return UInt8(0), UInt8(0)

    @always_inline
    fn max_second(
        self, year: UInt16, month: UInt8, day: UInt8, hour: UInt8, minute: UInt8
    ) -> UInt8:
        """The maximum amount of seconds that a minute lasts (usually 59).
        Some years its 60 when a leap second is added. The spec also lists
        the posibility of 58 seconds but it stil hasn't ben done.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.

        Returns:
            The amount.
        """
        if self.is_leapsec(year, month, day, hour, minute, 60):
            return 60
        return 59

    @always_inline
    fn max_days_in_month(self, year: UInt16, month: UInt8) -> UInt8:
        """The maximum amount of days in a given month.

        Args:
            year: Year.
            month: Month.

        Returns:
            The amount of days.
        """
        var days = self._monthdays[int(month) - 1]
        if month == 2 and self.is_leapyear(year):
            return days + 1
        return days

    @always_inline
    fn dayofweek(self, year: UInt16, month: UInt8, day: UInt8) -> UInt8:
        """Calculates the day of the week for a given date.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            - day: Day of the week: [0, 6] (monday - sunday).
        """
        var days = int(self.dayofyear(year, month, day))
        var y = int(year - 1)
        var days_before_year = y * 365 + y // 4 - y // 100 + y // 400
        return (days_before_year + days + 6) % 7

    @always_inline
    fn dayofyear(self, year: UInt16, month: UInt8, day: UInt8) -> UInt16:
        """Calculates the day of the year for a given date.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            - day: Day of the year: [1, 366] (for gregorian calendar).
        """
        var total: UInt16 = 1 if self.is_leapyear(year) else 0
        for i in range(month):
            var amnt_days = self._monthdays[i].cast[DType.uint16]()
            total += (
                amnt_days if (i + 1) != int(month) else day.cast[DType.uint16]()
            )
        return total

    @always_inline
    fn is_leapyear(self, year: UInt16) -> Bool:
        """Whether the year is a leap year.

        Args:
            year: Year.

        Returns:
            Bool.
        """
        _ = self
        return Gregorian.is_leapyear(year)

    @always_inline
    fn is_leapsec(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
    ) -> Bool:
        """Whether the second is a leap second.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.

        Returns:
            Bool.
        """
        _ = self, year, month, day, hour, minute, second
        # TODO: use hardcoded list in _lists ?
        return False

    fn leapsecs_since_epoch(
        self, year: UInt16, month: UInt8, day: UInt8
    ) -> UInt32:
        """Cumulative leap seconds since the calendar's epoch start.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            The amount.
        """
        _ = self, month, day
        if year < 1972:
            return 0
        # TODO: use hardcoded list in _lists ?
        return 27

    fn leapdays_since_epoch(
        self, year: UInt16, month: UInt8, day: UInt8
    ) -> UInt32:
        """Cumulative leap days since the calendar's epoch start.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            The amount.
        """
        var leapdays: UInt32 = 0
        for i in range(self.min_year, year):
            if self.is_leapyear(i):
                leapdays += 1
        if self.is_leapyear(year) and month >= 2:
            if not (month == 2 and day != 29):
                leapdays += 1
        return leapdays

    fn seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
    ) -> UInt64:
        """Seconds since the begining of the calendar's epoch.
        Takes leap seconds added to UTC up to the given datetime into
        account.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.

        Returns:
            The amount.
        """
        alias min_to_sec: UInt64 = 60
        alias hours_to_sec: UInt64 = 60 * min_to_sec
        alias days_to_sec: UInt64 = 24 * hours_to_sec
        alias years_to_sec: UInt64 = 365 * days_to_sec

        var leapsecs = self.leapsecs_since_epoch(year, month, day)
        var leapdays = self.leapdays_since_epoch(year, month, day).cast[
            DType.uint64
        ]()
        var leaps = leapsecs.cast[DType.uint64]() + leapdays * days_to_sec

        var y_d = (year - self.min_year).cast[DType.uint64]() * years_to_sec

        var days: UInt64 = 0
        for i in range(self.min_month, month + 1):
            days += self._monthdays[i].cast[DType.uint64]()
        var mon_d = days * days_to_sec

        var d_d = (day - self.min_day).cast[DType.uint64]() * days_to_sec
        var h_d = (hour - self.min_hour).cast[DType.uint64]() * hours_to_sec
        var min_d = (minute - self.min_minute).cast[DType.uint64]() * min_to_sec
        var s_d = (second - self.min_second).cast[DType.uint64]()
        return y_d + mon_d + d_d + h_d + min_d + s_d + leaps

    fn m_seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
        m_second: UInt8,
    ) -> UInt64:
        """Miliseconds since the begining of the calendar's epoch.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: Milisecond.

        Returns:
            The amount.
        """
        _ = m_second
        alias sec_to_mili = 1000
        alias min_to_mili = 60 * sec_to_mili
        alias hours_to_mili = 60 * min_to_mili
        alias days_to_mili = 24 * hours_to_mili
        alias years_to_mili = 365 * days_to_mili

        var leapsecs = self.leapsecs_since_epoch(year, month, day).cast[
            DType.uint64
        ]()
        var leapdays = self.leapdays_since_epoch(year, month, day).cast[
            DType.uint64
        ]()
        var leaps = (leapsecs * sec_to_mili + leapdays * days_to_mili).cast[
            DType.uint64
        ]()

        var y_d = ((year - self.min_year) * years_to_mili).cast[DType.uint64]()

        var days: UInt64 = 0
        for i in range(self.min_month, month + 1):
            days += self._monthdays[i].cast[DType.uint64]()
        var mon_d = days * days_to_mili

        var d_d = (day - UInt8(self.min_day)).cast[
            DType.uint64
        ]() * days_to_mili
        var h_d = (hour - UInt8(self.min_hour)).cast[
            DType.uint64
        ]() * hours_to_mili
        var min_d = (minute - UInt8(self.min_minute)).cast[
            DType.uint64
        ]() * min_to_mili
        var s_d = (second - UInt8(self.min_second)).cast[
            DType.uint64
        ]() * sec_to_mili
        return y_d + mon_d + d_d + h_d + min_d + s_d + leaps

    fn n_seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
        m_second: UInt16,
        u_second: UInt16,
        n_second: UInt16,
    ) -> UInt64:
        """Nanoseconds since the begining of the calendar's epoch.
        unsafe_Takes leap seconds added to UTC up to the given datetime into
        account. Can only represent up to ~ 580 years since epoch start.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.
            u_second: U_second.
            n_second: N_second.

        Returns:
            The amount.
        """
        alias sec_to_nano = 1000_000_000
        alias min_to_nano = 60 * sec_to_nano
        alias hours_to_nano = 60 * min_to_nano
        alias days_to_nano = 24 * hours_to_nano
        alias years_to_nano = 365 * days_to_nano

        var leapsecs = self.leapsecs_since_epoch(year, month, day).cast[
            DType.uint64
        ]()
        var leapdays = self.leapdays_since_epoch(year, month, day).cast[
            DType.uint64
        ]()
        var leaps = (leapsecs * sec_to_nano + leapdays * days_to_nano).cast[
            DType.uint64
        ]()

        var y_d = ((year - self.min_year) * years_to_nano).cast[DType.uint64]()

        var days: UInt64 = 0
        for i in range(self.min_month, month + 1):
            days += self._monthdays[i].cast[DType.uint64]()
        var mon_d = days * days_to_nano

        var d_d = (day - UInt8(self.min_day)).cast[
            DType.uint64
        ]() * days_to_nano
        var h_d = (hour - UInt8(self.min_hour)).cast[
            DType.uint64
        ]() * hours_to_nano
        var min_d = (minute - UInt8(self.min_minute)).cast[
            DType.uint64
        ]() * min_to_nano
        var s_d = (second - UInt8(self.min_second)).cast[
            DType.uint64
        ]() * sec_to_nano
        var ms_d = (m_second - UInt16(self.min_milisecond)).cast[
            DType.uint64
        ]() * 1_000_000
        var us_d = (u_second - UInt16(self.min_microsecond)).cast[
            DType.uint64
        ]() * 1_000
        var ns_d = (n_second - UInt16(self.min_nanosecond)).cast[DType.uint64]()
        return (
            y_d + mon_d + d_d + h_d + min_d + s_d + ms_d + us_d + ns_d + leaps
        )

    @always_inline
    fn hash[
        cal_h: CalendarHashes = CalendarHashes()
    ](
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8 = 0,
        minute: UInt8 = 0,
        second: UInt8 = 0,
        m_second: UInt16 = 0,
        u_second: UInt16 = 0,
        n_second: UInt16 = 0,
    ) -> Int:
        """Hash the given values according to the calendar's component
        lengths bitshifted, BigEndian (i.e. yyyymmdd...).

        Parameters:
            cal_h: The hashing schema (CalendarHashes).

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.
            u_second: U_second.
            n_second: N_second.

        Returns:
            The hash.
        """
        _ = self, n_second
        if cal_h.selected == cal_h.UINT8:
            pass
        elif cal_h.selected == cal_h.UINT16:
            pass
        elif cal_h.selected == cal_h.UINT32:  # hash for `Date`
            return int(
                (UInt32(year) << (5 + 5)) | (UInt32(month) << 5) | UInt32(day)
            )
        elif cal_h.selected == cal_h.UINT64:  # hash for `DateTime`
            return int(
                UInt64(year) << cal_h.shift_64_y
                | UInt64(month) << cal_h.shift_64_mon
                | UInt64(day) << cal_h.shift_64_d
                | UInt64(hour) << cal_h.shift_64_h
                | UInt64(minute) << cal_h.shift_64_m
                | UInt64(second) << cal_h.shift_64_s
                | UInt64(m_second) << cal_h.shift_64_ms
                | UInt64(u_second) << cal_h.shift_64_us
            )
        return 0

    @always_inline
    fn from_hash[
        cal_h: CalendarHashes = CalendarHashes()
    ](self, value: Int) -> _date:
        """Build a date from a hashed value.

        Parameters:
            cal_h: The hashing schema (CalendarHashes).

        Args:
            value: The Hash.

        Returns:
            Tuple containing date data.
        """
        _ = self
        var num8 = UInt8(0)
        var num16 = UInt16(0)
        var result = (num16, num8, num8, num8, num8, num8, num16, num16)

        if cal_h.selected == cal_h.UINT8:
            pass
        elif cal_h.selected == cal_h.UINT16:
            pass
        elif cal_h.selected == cal_h.UINT32:  # hash for `Date`
            result[0] = value >> (5 + 5)
            result[1] = (value >> 5) & 5
            result[2] = value & 5
        elif cal_h.selected == cal_h.UINT64:  # hash for `DateTime`
            result[0] = int(((value & cal_h.mask_64_y) >> cal_h.shift_64_y))
            result[1] = int(((value & cal_h.mask_64_mon) >> cal_h.shift_64_mon))
            result[2] = int(((value & cal_h.mask_64_d) >> cal_h.shift_64_d))
            result[3] = int(((value & cal_h.mask_64_h) >> cal_h.shift_64_h))
            result[4] = int(((value & cal_h.mask_64_m) >> cal_h.shift_64_m))
            result[5] = int(((value & cal_h.mask_64_s) >> cal_h.shift_64_s))
            result[6] = int(((value & cal_h.mask_64_ms) >> cal_h.shift_64_ms))
            result[7] = int(((value & cal_h.mask_64_us) >> cal_h.shift_64_us))
        return result

    @always_inline
    @staticmethod
    fn is_leapyear(year: UInt16) -> Bool:
        """Whether the year is a leap year.

        Args:
            year: Year.

        Returns:
            Bool.
        """
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @staticmethod
    @always_inline
    fn from_year(year: UInt16) -> Self:
        """Get a Calendar with min_year=year.

        Args:
            year: The year.

        Returns:
            Self.
        """
        return Self(min_year=year)


@register_passable("trivial")
struct UTCFast(_Calendarized):
    """`UTCFast` Calendar."""

    var max_year: UInt16
    """Maximum value of years."""
    alias max_typical_days_in_year: UInt16 = 365
    """Maximum typical value of days in a year (no leaps)."""
    alias max_possible_days_in_year: UInt16 = 365
    """Maximum possible value of days in a year (with leaps)."""
    alias max_month: UInt8 = 12
    """Maximum value of months in a year."""
    alias max_hour: UInt8 = 23
    """Maximum value of hours in a day."""
    alias max_minute: UInt8 = 59
    """Maximum value of minutes in an hour."""
    alias max_typical_second: UInt8 = 59
    """Maximum typical value of seconds in a minute (no leaps)."""
    alias max_possible_second: UInt8 = 59
    """Maximum possible value of seconds in a minute (with leaps)."""
    alias max_milisecond: UInt8 = 999
    """Maximum value of miliseconds in a second."""
    alias max_microsecond: UInt8 = 999
    """Maximum value of microseconds in a second."""
    alias max_nanosecond: UInt8 = 999
    """Maximum value of nanoseconds in a second."""
    var min_year: UInt16
    """Default minimum year in the calendar."""
    alias min_month: UInt8 = 1
    """Default minimum month."""
    alias min_day: UInt8 = 1
    """Default minimum day."""
    alias min_hour: UInt8 = 0
    """Default minimum hour."""
    alias min_minute: UInt8 = 0
    """Default minimum minute."""
    alias min_second: UInt8 = 0
    """Default minimum second."""
    alias min_milisecond: UInt16 = 0
    """Default minimum milisecond."""
    alias min_microsecond: UInt16 = 0
    """Default minimum microsecond."""
    alias min_nanosecond: UInt16 = 0
    """Default minimum nanosecond."""
    alias _monthdays: List[UInt8] = List[UInt8]()
    """An array with the amount of days each month contains without 
    leap values. It's assumed that `len(monthdays) == max_month`."""

    fn __init__(
        inout self, *, max_year: UInt16 = 9999, min_year: UInt16 = 1970
    ):
        """Construct a `UTCFast` Calendar from values.
        
        Args:
            max_year: Max year (epoch end).
            min_year: Min year (epoch start).
        """
        self.max_year = max_year
        self.min_year = min_year

    @always_inline
    fn monthrange(self, year: UInt16, month: UInt8) -> (UInt8, UInt8):
        """Calculates the day of the week and the day of the month
        that a month in a given year ends.

        Args:
            year: Year.
            month: Month.

        Returns:
            - dayofweek: Day of the week.
            - dayofmonth: Day of the month.
        """
        _ = self, year, month
        return UInt8(0), UInt8(0)

    @always_inline
    fn max_second(
        self, year: UInt16, month: UInt8, day: UInt8, hour: UInt8, minute: UInt8
    ) -> UInt8:
        """The maximum amount of seconds in a minute (usually 59).

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.

        Returns:
            The amount.
        """
        _ = self, year, month, day, hour, minute
        return 59

    @always_inline
    fn max_days_in_month(self, year: UInt16, month: UInt8) -> UInt8:
        """The maximum amount of days in a given month.

        Args:
            year: Year.
            month: Month.

        Returns:
            The amount of days.
        """
        _ = self, year, month
        return 0

    @always_inline
    fn dayofweek(self, year: UInt16, month: UInt8, day: UInt8) -> UInt8:
        """Calculates the day of the week for a given date.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            - day: Day of the week: [0, 6] (monday - sunday).
        """
        _ = self, year, month, day
        return 0

    @always_inline
    fn dayofyear(self, year: UInt16, month: UInt8, day: UInt8) -> UInt16:
        """Calculates the day of the year for a given date.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            - day: Day of the year: [1, 366] (for gregorian calendar).
        """
        _ = self, year, month, day
        return 0

    @always_inline
    fn is_leapyear(self, year: UInt16) -> Bool:
        """Whether the year is a leap year.

        Args:
            year: Year.

        Returns:
            Bool.
        """
        _ = self, year
        return False

    @always_inline
    fn is_leapsec(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
    ) -> Bool:
        """Whether the second is a leap second.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.

        Returns:
            Bool.
        """
        _ = self, year, month, day, hour, minute, second
        return False

    fn leapsecs_since_epoch(
        self, year: UInt16, month: UInt8, day: UInt8
    ) -> UInt32:
        """Cumulative leap seconds since the calendar's epoch start.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            The amount.
        """
        _ = self, year, month, day
        return 0

    fn leapdays_since_epoch(
        self, year: UInt16, month: UInt8, day: UInt8
    ) -> UInt32:
        """Cumulative leap days since the calendar's epoch start.

        Args:
            year: Year.
            month: Month.
            day: Day.

        Returns:
            The amount.
        """
        _ = self, year, month, day
        return 0

    fn seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
    ) -> UInt64:
        """Seconds since the begining of the calendar's epoch.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.

        Returns:
            The amount.
        """
        _ = self
        alias min_to_sec: UInt64 = 60
        alias hours_to_sec: UInt64 = 60 * min_to_sec
        alias days_to_sec: UInt64 = 24 * hours_to_sec
        alias years_to_sec: UInt64 = 365 * days_to_sec

        return (
            UInt64(year) * years_to_sec
            + UInt64(month) * 30
            + UInt64(day) * days_to_sec
            + UInt64(hour) * hours_to_sec
            + UInt64(minute) * min_to_sec
            + UInt64(second)
        )

    fn m_seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
        m_second: UInt8,
    ) -> UInt64:
        """Miliseconds since the begining of the calendar's epoch.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: Milisecond.

        Returns:
            The amount.
        """
        _ = self
        alias sec_to_mili = 1000
        alias min_to_mili = 60 * sec_to_mili
        alias hours_to_mili = 60 * min_to_mili
        alias days_to_mili = 24 * hours_to_mili
        alias years_to_mili = 365 * days_to_mili

        return (
            UInt64(year) * years_to_mili
            + UInt64(month) * 30
            + UInt64(day) * days_to_mili
            + UInt64(hour) * hours_to_mili
            + UInt64(minute) * min_to_mili
            + UInt64(second) * sec_to_mili
            + UInt64(m_second)
        )

    fn n_seconds_since_epoch(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
        m_second: UInt16,
        u_second: UInt16,
        n_second: UInt16,
    ) -> UInt64:
        """Nanoseconds since the begining of the calendar's epoch.
        Assumes every year has 365 days and all months have 30 days.
        Can only represent up to ~ 580 years since epoch start.

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.
            u_second: U_second.
            n_second: N_second.

        Returns:
            The amount.
        """
        _ = self
        alias sec_to_nano = 1000_000_000
        alias min_to_nano = 60 * sec_to_nano
        alias hours_to_nano = 60 * min_to_nano
        alias days_to_nano = 24 * hours_to_nano
        alias years_to_nano = 365 * days_to_nano

        return (
            UInt64(year) * years_to_nano
            + UInt64(month) * 30
            + UInt64(day) * days_to_nano
            + UInt64(hour) * hours_to_nano
            + UInt64(minute) * min_to_nano
            + UInt64(second) * sec_to_nano
            + UInt64(m_second) * 1_000_000
            + UInt64(u_second) * 1_000
            + UInt64(n_second)
        )

    @always_inline
    fn hash[
        cal_h: CalendarHashes = CalendarHashes()
    ](
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8 = 0,
        minute: UInt8 = 0,
        second: UInt8 = 0,
        m_second: UInt16 = 0,
        u_second: UInt16 = 0,
        n_second: UInt16 = 0,
    ) -> Int:
        """Hash the given values according to the calendar's component
        lengths bitshifted, BigEndian (i.e. yyyymmdd...).

        Parameters:
            cal_h: The hashing schema (CalendarHashes).

        Args:
            year: Year.
            month: Month.
            day: Day.
            hour: Hour.
            minute: Minute.
            second: Second.
            m_second: M_second.
            u_second: U_second.
            n_second: N_second.

        Returns:
            The hash.
        """
        _ = self, u_second, n_second
        if cal_h.selected == cal_h.UINT8:
            return int(
                (UInt8(day) << cal_h.shift_8_d)
                | (UInt8(hour) << cal_h.shift_8_h)
            )
        elif cal_h.selected == cal_h.UINT16:
            return int(
                (UInt16(year) << cal_h.shift_16_y)
                | (UInt16(day) << cal_h.shift_16_d)
                | (UInt16(hour) << cal_h.shift_16_h)
            )
        elif cal_h.selected == cal_h.UINT32:
            return int(
                (UInt32(year) << cal_h.shift_32_y)
                | (UInt32(month) << cal_h.shift_32_mon)
                | (UInt32(day) << cal_h.shift_32_d)
                | (UInt32(hour) << cal_h.shift_32_h)
                | (UInt32(minute) << cal_h.shift_32_m)
            )
        elif cal_h.selected == cal_h.UINT64:
            return int(
                (UInt64(year) << (cal_h.shift_64_y - cal_h.shift_64_ms))
                | (UInt64(month) << (cal_h.shift_64_mon - cal_h.shift_64_ms))
                | (UInt64(day) << (cal_h.shift_64_d - cal_h.shift_64_ms))
                | (UInt64(hour) << (cal_h.shift_64_h - cal_h.shift_64_ms))
                | (UInt64(minute) << (cal_h.shift_64_m - cal_h.shift_64_ms))
                | (UInt64(second) << (cal_h.shift_64_s - cal_h.shift_64_ms))
                | UInt64(m_second)
            )
        return 0

    @always_inline
    fn from_hash[
        cal_h: CalendarHashes = CalendarHashes()
    ](self, value: Int) -> _date:
        """Build a date from a hashed value.

        Parameters:
            cal_h: The hashing schema (CalendarHashes).

        Args:
            value: The Hash.

        Returns:
            Tuple containing date data.
        """
        _ = self
        var num8 = UInt8(0)
        var num16 = UInt16(0)
        var result = (num16, num8, num8, num8, num8, num8, num16, num16)

        if cal_h.selected == cal_h.UINT8:
            result[2] = int(((value & cal_h.mask_8_d) >> cal_h.shift_8_d))
            result[3] = int(((value & cal_h.mask_8_h) >> cal_h.shift_8_h))
        elif cal_h.selected == cal_h.UINT16:
            result[0] = int(((value & cal_h.mask_16_y) >> cal_h.shift_16_y))
            result[2] = int(((value & cal_h.mask_16_d) >> cal_h.shift_16_d))
            result[3] = int(((value & cal_h.mask_16_h) >> cal_h.shift_16_h))
        elif cal_h.selected == cal_h.UINT32:
            result[0] = int(((value & cal_h.mask_32_y) >> cal_h.shift_32_y))
            result[1] = int(((value & cal_h.mask_32_mon) >> cal_h.shift_32_mon))
            result[2] = int(((value & cal_h.mask_32_d) >> cal_h.shift_32_d))
            result[3] = int(((value & cal_h.mask_32_h) >> cal_h.shift_32_h))
            result[4] = int(((value & cal_h.mask_32_m) >> cal_h.shift_32_m))
        elif cal_h.selected == cal_h.UINT64:
            result[0] = int(
                (value & cal_h.mask_64_y)
                >> (cal_h.shift_64_y - cal_h.shift_64_ms)
            )
            result[1] = int(
                (value & cal_h.mask_64_mon)
                >> (cal_h.shift_64_mon - cal_h.shift_64_ms)
            )
            result[2] = int(
                (value & cal_h.mask_64_d)
                >> (cal_h.shift_64_d - cal_h.shift_64_ms)
            )
            result[3] = int(
                (value & cal_h.mask_64_h)
                >> (cal_h.shift_64_h - cal_h.shift_64_ms)
            )
            result[4] = int(
                (value & cal_h.mask_64_m)
                >> (cal_h.shift_64_m - cal_h.shift_64_ms)
            )
            result[5] = int(
                (value & cal_h.mask_64_s)
                >> (cal_h.shift_64_s - cal_h.shift_64_ms)
            )
            result[6] = int(value & cal_h.mask_64_ms)
        return result

    @always_inline
    @staticmethod
    fn is_leapyear(year: UInt16) -> Bool:
        """Whether the year is a leap year.

        Args:
            year: Year.

        Returns:
            Bool.
        """
        _ = year
        return False

    @staticmethod
    @always_inline
    fn from_year(year: UInt16) -> Self:
        """Get a Calendar with min_year=year.

        Args:
            year: The year.

        Returns:
            Self.
        """
        return Self(min_year=year)
