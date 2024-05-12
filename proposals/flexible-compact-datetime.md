# Create a set of flexible and useful tools in the datetime package

## Motivation
Python's `datetime` and `timedelta` are expensive to instantiate and use.

## Proposal
In this document a proposal for a datetime package linked to PR [#2623](https://github.com/modularml/mojo/pull/2623) is presented.

Highlights of the proposal are:
- Default DateTime with most pythonic APIs
- DateTime64, DateTime32, DateTime16, DateTime8
    - Very fast implementations that take no timezone into account.
    - Each has a buffer of its uint size (i.e. 64, 32, 16, 8) with its current value for its base unit of measurement and another buffer with its hash representing its initial state.
- ZoneInfo
    - Materialized in memory (ZoneInfoMem):
        - fits all ~ 418 timezones into 7.96 kB.
    - Materialized as a File (ZoneInfoFile):
        - fits all ~ 418 timezones into 418 B in memory, 4.9kB in storage.
- Calendar
    - Being able to inject a custom Calendar for calculations will be very powerful for those wanting to do them relative to a date/datetime.


## calendar module
### Calendar struct
Should handle standard logic that one would expect. It defines:
```mojo
# TODO: once traits with attributes and impl are ready Calendar will replace
# a bunch of this file
trait _Calendarized:
    ...

@register_passable("trivial")
struct Calendar(_Calendarized):
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
        var imp = impl.unsafe_take[Gregorian]()
        ...

@register_passable("trivial")
struct Gregorian(_Calendarized):
    ...

@register_passable("trivial")
struct UTCFast(_Calendarized):
    ...
```
The most used Calendar will be the Gregorian, it has capabilities to deal with leap seconds and years since epoch start (min_year)
The default calendar is PythonCalendar, which is an instantiation of the Gregorian calendar (proleptic Gregorian starts year 1 month 1 day 1).
There is a UTCCalendar which is Gregorian instanciated with min_year=1970
There is a second type of Calendar which is UTCFast, this one is a naive implementation since the Unix epoch that doesn't have any leap seconds or years and has it's own hashing implementations for the `fast` module.

## date module
### `Date` struct
Has all of the basic interfaces and uses the given calendar's 32bit hash for logical and bitwise operations.
```mojo
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
    """
    ...
```

## datetime module
### `DateTime` struct
has all of the basic interfaces and uses the given calendar's 64bit hash for logical and bitwise operations. Its hash has only microsecond resolution so it loses its nanoseconds if it's hashed and parsed back and comparisons don't take nanoseconds into account.
```mojo
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
```
## fast module
Fast implementations of `DateTime` module. All assume no leap seconds or
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

## dt_str module
`DateTime` and `Date` String parsing module.
## zoneinfo module
Now this one was a big challenge. Who would've thought that there is no single list of timezones and their respective STD, DST and transition rules. IANA has a list of tz and their current offsets but it's stil marked as experimental, still no transition rules. I decided to try and have several ways to go about getting the data.
```mojo
fn get_leapsecs() -> Optional[List[(UInt8, UInt8, UInt16)]]:
    try:
        # TODO: maybe some policy that only if x amount
        # of years have passed since latest hardcoded value
        from python import Python

        var requests = Python.import_module("requests")
        var secs = requests.get(
            "https://raw.githubusercontent.com/eggert/tz/main/leap-seconds.list"
        )
        var leapsecs = parse_iana_leapsecs(secs.text)
        return leapsecs
    except:
        pass
        # TODO: fallback to hardcoded
        # from ._lists import leapsecs

        # return List[(UInt8, UInt8, UInt16)](
        #     unsafe_pointer=leapsecs.data.address,
        #     capacity=leapsecs.capacity,
        #     size=leapsecs.size,
        # )
    return List[(UInt8, UInt8, UInt16)]()


fn get_zoneinfo() -> Optional[ZoneInfo]:
    """Get all zoneinfo available. First tries to get it
    from the OS, then from the internet, then falls back
    on hardcoded values.

    - TODO: this should get zoneinfo from the OS it's compiled in
    - TODO: should have a fallback to hardcoded
    - TODO: this should use IANA's https://raw.githubusercontent.com/eggert/tz/main/zonenow.tab
        - but "# The format of this table is experimental, and may change in future versions."
        - Excerpt:
        ```text
        # -10
        XX	-1732-14934	Pacific/Tahiti	Tahiti; Cook Islands
        #
        # -10/-09 - HST / HDT (North America DST)
        XX	+515248-1763929	America/Adak	western Aleutians in Alaska ("HST/HDT")
        #
        # -09:30
        XX	-0900-13930	Pacific/Marquesas	Marquesas
        ```
        Meanwhile 2 public APIs can be used https://worldtimeapi.org/api
        and https://timeapi.io/swagger/index.html .
    """
```

ZoneInfo is a Tuple of tzs_with_dst, tzs_without_dst
```mojo
struct ZoneInfoFile:
    """Zoneinfo that lives in a file. Smallest memory footprint
    but only supports 256 timezones (there are ~ 418)."""
    ...


alias ZoneInfoFile32 = ZoneInfoFile(32, 0xFFFFFFFF)
"""ZoneInfoFile to store Offset of tz with DST"""
alias ZoneInfoFile8 = ZoneInfoFile(8, 0xFFFF)
"""ZoneInfoFile to store Offset of tz with no DST"""


struct ZoneInfoMem32:
    """`ZoneInfo` that lives in memory. For zones that have DST."""
    ...
struct ZoneInfoMem8:
    """`ZoneInfo` that lives in memory. For zones that have no DST."""
    ...

# TODO: get_zoneinfo should be able to return a ZoneInfoMem
# or ZoneInfoFile according to parameter
alias ZoneInfo = (ZoneInfoMem32, ZoneInfoMem8)
"""ZoneInfo."""
```

![image](https://github.com/modularml/mojo/assets/110240700/88b9f920-fd13-400f-8b18-71c030af3ceb)

When using all timezones:
`ZoneInfoMem`:
- If tz_str is assumed to be about 18 bytes, the total memory footprint for a ZoneInfoMem32 for 70 timezones with dst + ZoneInfoMem8 for (418 - 70) timezones would be around `18 * 8 * 418 + 70 * 32 + (418 - 70) * 8 = 63.7 kb = 7.96 kB`
 
`ZoneInfoFile`:
- If a microcontroller application needs to have an even smaller memory footprint and uses ZoneInfoFile (basically an index): `418 * 8 = 3.26 kb  = 418 B` in memory, `70 * 32 + (418 - 70) * 8 = 5024 b = 4.9 kB`. The user will most likely only use a very small subset, but it's still nice to have it so compact.

## timezone module
The `TimeZone` struct is the actual entrypoint to set the timezone for any given datelike struct.

```mojo
from .zoneinfo import all_zones, ZoneInfo, offset_at, offset_no_dst_tz


@register_passable("trivial")
struct TimeZone[
    iana: Optional[ZoneInfo] = all_zones,
    pyzoneinfo: Bool = True,
    native: Bool = False,
]:
    """`TimeZone` struct. Because of a POSIX standard, if you set
    the tz_str e.g. Etc/UTC-4 it means 4 hours east of UTC
    which is UTC + 4 in numbers. That is:
    `TimeZone("Etc/UTC-4", offset_h=4, offset_m=0, sign=1)`. If
    `TimeZone[iana=True]("Etc/UTC-4")`, the correct offsets are
    returned for the calculations, but the attributes offset_h,
    offset_m and sign will remain the default 0, 0, 1 respectively.

    Parameters:
        iana: What timezones from the [IANA database](
            http://www.iana.org/time-zones/repository/tz-link.html)
            are used. It defaults to using all available timezones,
            if getting them fails at compile time, it tries using
            python's zoneinfo if pyzoneinfo is set to True, otherwise
            it uses the offsets as is, no daylight saving or
            special exceptions. [List of TZ identifiers](
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).
        pyzoneinfo: Whether to use python's zoneinfo and
            datetime to get full IANA support.
        native: (fast, partial IANA support) Whether to use a native Dict
            with the current timezones from the [List of TZ identifiers](
            https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
            at the time of compilation (for now they're hardcoded
            at stdlib release time, in the future it should get them
            from the OS). If it fails at compile time, it defaults to
            using the given offsets when the timezone was constructed.
    """

    var tz_str: StringLiteral
    """[`TZ identifier`](
        https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)."""
    var offset_h: UInt8
    """Offset for the hour."""
    var offset_m: UInt8
    """Offset for the minute."""
    var sign: UInt8
    """Sign: {1, -1}."""
    var has_dst: Bool
    """Whether the `TimeZone` has Daylight Saving Time."""

    fn __init__(
        inout self,
        tz_str: StringLiteral = "Etc/UTC",
        offset_h: UInt8 = 0,
        offset_m: UInt8 = 0,
        sign: UInt8 = 1,
        has_dst: Bool = True,
    ):
        """Construct a `TimeZone`."""
        debug_assert(
            offset_h < 100 and offset_m < 100 and (sign == 1 or sign == -1),
            msg=(
                "utc offsets can't have a member bigger than 100, "
                "and sign must be either 1 or -1"
            ),
        )
        if iana:
            debug_assert(
                iana.value()[][0].get(tz_str) or iana.value()[][1].get(tz_str),
                msg="that timezone is not in the given IANA ZoneInfo",
            )
        self.tz_str = tz_str
        self.offset_h = offset_h
        self.offset_m = offset_m
        self.sign = sign
        self.has_dst = has_dst

        if iana and not has_dst:
            var tz = iana.value()[][1].get(tz_str)
            var val = offset_no_dst_tz(tz)
            if not val:
                return
            var offset = val.unsafe_take()
            self.offset_h = offset[0]
            self.offset_m = offset[1]
            self.sign = offset[2]


    fn offset_at(
        self,
        year: UInt16,
        month: UInt8,
        day: UInt8,
        hour: UInt8,
        minute: UInt8,
        second: UInt8,
    ) -> (UInt8, UInt8, UInt8):
        """Return the UTC offset for the `TimeZone` at the given date.

        Returns:
            - offset_h: Offset for the hour: [0, 15].
            - offset_m: Offset for the minute: {0, 30, 45}.
            - sign: Sign of the offset: {1, -1}.
        """
        if iana and native and self.has_dst:
            var dst = iana.value()[][0].get(self.tz_str)
            var offset = offset_at(dst, year, month, day, hour, minute, second)
            if offset:
                return offset.unsafe_take()
        elif iana and pyzoneinfo:
            try:
                from python import Python

                var zoneinfo = Python.import_module("zoneinfo")
                var dt = Python.import_module("datetime")
                var zone = zoneinfo.ZoneInfo(self.tz_str)
                var local = dt.datetime(year, month, day, hour, tzinfo=zone)
                var offset = local.utcoffset()
                var sign = 1 if offset.days == -1 else -1
                var hours = offset.seconds // (60 * 60) - hour
                var minutes = offset.seconds % 60
                return UInt8(hours), UInt8(minutes), UInt8(sign)
            except:
                pass
        return self.offset_h, self.offset_m, self.sign


    @staticmethod
    fn from_offset(
        year: UInt16,
        month: UInt8,
        day: UInt8,
        offset_h: UInt8,
        offset_m: UInt8,
        sign: UInt8,
    ) -> Self:
        # TODO: it should create an Etc/UTC-X TimeZone
        return TimeZone[iana, pyzoneinfo, native]()
```

## changes to `time.time()`
I needed to add the functionality to use the realtime clock, not just the monotonic clock from the OS.
```mojo
@always_inline
fn _realtime_nanoseconds() -> Int:
    """Returns the current realtime time in nanoseconds"""

    @parameter
    if os_is_windows():
        var ft = _FILETIME()
        external_call["GetSystemTimePreciseAsFileTime", NoneType](
            UnsafePointer.address_of(ft)
        )
        return ft.as_nanoseconds()
    else:
        return _gettime_as_nsec_unix(_CLOCK_REALTIME)


@always_inline
@parameter
fn now[monotonic: Bool = True]() -> Int:
    """Returns the current time in nanoseconds. This function
    queries the current platform's monotonic (default) or 
    realtime clock, making it useful for measuring time 
    differences, but the significance of the returned value 
    varies depending on the underlying implementation.
    Parameters:
        monotonic: Whether the monotonic clock or the realtime clock is used.
    Returns:
        The current time in ns.
    """
    if monotonic:
        return _monotonic_nanoseconds()
    else:
        return _realtime_nanoseconds()
```



 ## TODOs
There is a lot to do yet. There is no localization, strftime is all currently handled by Python libraries, `Date` and `DateTime` should have references to tz and calendar, not values. And many other `#TODO`s thrown around the code.

