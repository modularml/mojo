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
"""`ZoneInfo` module."""

from pathlib import Path, cwd
from utils import Variant


@register_passable("trivial")
struct Offset:
    """Only supports hour offsets less than 16 hours and minute offsets
    that are in (00, 30, 45). Offset sign and minute are assumed
    to be equal in DST and STD and DST adds 1 hour to STD hour,
    unless 2 reserved bits are unequal to 0 which means the offset is
    one of 2 weird time zones (this was added because of literally [one
    small island](https://en.wikipedia.org/wiki/Lord_Howe_Island)
    and an [Antartica research station](
        https://es.wikipedia.org/wiki/Base_Troll ))."""

    var buf: UInt8
    """Buffer."""

    fn __init__(inout self, buf: UInt8):
        """Construct an `Offset` from a buffer.

        Args:
            buf: The buffer.
        """
        self.buf = buf

    fn __init__(inout self, hour: Int, minute: Int, sign: Int):
        """Construct an `Offset` from values.

        Args:
            hour: Hour.
            minute: Minute.
            sign: Sign.
        """
        self.buf = (sign << 7) | (hour << 3) | (minute << 1) | 0

    fn __init__(
        inout self,
        iso_tzd_std: String = "+00:00",
        iso_tzd_dst: String = "+00:00",
    ):
        """Construct an `Offset` (8 bits total) for DST start/end.

        Args:
            iso_tzd_std: String with the full ISO8601 TZD (i.e. +00:00).
            iso_tzd_dst: String with the full ISO8601 TZD (i.e. +00:00).
        """
        try:
            var sign = (0 if iso_tzd_std[0] == "+" else 1)

            var std_h: UInt8 = atol(iso_tzd_std[1:2])
            var dst_h: UInt8 = atol(iso_tzd_dst[1:2])

            var std_m: UInt8 = atol(iso_tzd_std[4:6])
            var dst_m: UInt8 = atol(iso_tzd_std[4:6])
            var weird: UInt8 = 0
            if std_m - dst_m == 30:
                std_m = 3
            elif (dst_h - std_h) ^ 0b10 == 0:
                weird = 1
            else:
                if std_m == 30:
                    std_m = 1
                elif std_m == 45:
                    std_m = 2
            self.buf = (sign << 7) | (std_h << 3) | (std_m << 1) | weird
        except:
            self.buf = 0

    fn from_hash(self) -> (UInt8, UInt8, UInt8, UInt8):
        """Get the values from hash.

        Returns:
            - negative: (1 bit) Whether the offset is negative.
            - std_hour: (4 bits).
            - minute: (2 bits) Values: {0, 1, 2}, 0 means 0 minutes
                1 means 30 minutes, 2 means 45 minutes. 3 means
                it is a weird offset and the nex bit is read.
            - weird: (1 bit) Whether its one of two weird tzs.
                0 means dst adds 30 minutes, 1 means adds 2 hours.
                This is for [Lord_Howe_Island](
                    https://en.wikipedia.org/wiki/Lord_Howe_Island)
                and [Base_Troll](
                    https://es.wikipedia.org/wiki/Base_Troll)
                respectively.
        """
        return (
            (self.buf >> 7) & 0b1,
            (self.buf >> 3) & 0b1111,
            (self.buf >> 1) & 0b11,
            self.buf & 0b1,
        )


@register_passable("trivial")
struct TzDT:
    """`TzDT` stores the rules for DST start/end."""

    var buf: UInt16
    """Buffer."""

    fn __init__(inout self, buf: UInt16):
        """Construct a `TzDT` from a buffer.

        Args:
            buf: The buffer.
        """
        self.buf = buf

    fn __init__(
        inout self,
        month: UInt16 = 1,
        dow: UInt16 = 0,
        eomon: UInt16 = 0,
        week: UInt16 = 0,
        hour: UInt16 = 0,
    ):
        """Construct a `TzDT` buffer (12 bits total) for DST start/end.

        Args:
            month: Month: [1, 12].
            dow: Day of week: [0, 6] (monday - sunday).
            eomon: End of month: {0, 1} Whether to count from the
                beginning of the month or the end.
            week: {0, 1} If week=0 -> first week of the month,
                if it's week=1 -> second week. In the case that
                eomon=1, fw=0 -> last week of the month
                and fw=1 -> second to last.
            hour: {20, 21, 22, 23, 0, 1, 2, 3} Hour at which DST starts/ends.
        """
        var h = 0
        var i = 0
        for item in List(20, 21, 22, 23, 0, 1, 2, 3):
            if hour == item[]:
                h = i
                break
            i += 1

        self.buf = (month << 8) | (dow << 5) | (eomon << 4) | (week << 3) | h

    fn from_hash(self) -> (UInt8, UInt8, UInt8, UInt8, UInt8):
        """Get the values from hash.

        Returns:
            - month: (4 bits) Month: [1, 12].
            - dow: (3 bits) Day of week: [0, 6].
            - eomon: (1 bit) End of month: {0, 1}.
            - fw: (1 bit) First week: {0, 1}.
            - hour: (3 bits) Hour: [0, 8] represents:
                {20, 21, 22, 23, 0, 1, 2, 3}.
        """
        return (
            ((self.buf >> 8) & 0b1111).cast[DType.uint8](),
            ((self.buf >> 5) & 0b111).cast[DType.uint8](),
            ((self.buf >> 4) & 0b1).cast[DType.uint8](),
            ((self.buf >> 3) & 0b1).cast[DType.uint8](),
            (self.buf & 0b111).cast[DType.uint8](),
        )

    fn __bool__(self) -> Bool:
        return self.buf == 0

    fn __eq__(self, other: Self) -> Bool:
        return self.buf == other.buf


@register_passable("trivial")
struct ZoneDST:
    """`ZoneDST` stores both start and end dates, and
    the offset for a timezone with DST."""

    var buf: UInt32
    """Buffer."""

    fn __init__(inout self, dst_start: TzDT, dst_end: TzDT, offset: Offset):
        """Construct a `ZoneDST` from values.

        Args:
            dst_start: TzDT.
            dst_end: TzDT.
            offset: Offset.
        """
        self.buf = (
            (dst_start.buf.cast[DType.uint32]() << 20)
            | (dst_end.buf.cast[DType.uint32]() << 12)
            | offset.buf.cast[DType.uint32]()
        )

    fn __init__(inout self, buf: UInt32):
        """Construct a `ZoneDST` from a buffer.

        Args:
            buf: The buffer.
        """
        self.buf = buf

    fn from_hash(self) -> (TzDT, TzDT, Offset):
        """Get the values from hash.

        Returns:
            - dst_start: TzDT hash (12 bits in a UInt16 buffer).
            - dst_end: TzDT hash (12 bits in a UInt16 buffer).
            - offset: Offset hash (8 bits in a UInt16 buffer).
        """
        return (
            TzDT(buf=((self.buf >> 20) & 0b11111111).cast[DType.uint16]()),
            TzDT(buf=((self.buf >> 12) & 0b11111111).cast[DType.uint16]()),
            Offset((self.buf & 0b111111111111).cast[DType.uint8]()),
        )


@value
struct ZoneInfoFile:
    """Zoneinfo that lives in a file. Smallest memory footprint
    but only supports 256 timezones (there are ~ 418)."""

    var _index: UInt8
    var _BIT_WIDTH: UInt8
    var _BIT_MASK: UInt32
    var _file: Path

    fn __init__(inout self, BIT_WIDTH: UInt8, BIT_MASK: UInt32):
        """Construct a `ZoneInfoFile`.

        Args:
            BIT_WIDTH: Bit width of the values.
            BIT_MASK: Bit mask for the values.
        """
        try:
            self._file = Path(cwd()) / "zoneinfo_dump"
        except:
            self._file = "./zoneinfo_dump"
        self._index = 0
        self._BIT_WIDTH = BIT_WIDTH
        self._BIT_MASK = BIT_MASK

    fn add(inout self, key: StringLiteral, buf: UInt32) raises -> UInt8:
        """Add a value to the file.

        Args:
            key: The tz_str.
            buf: The buffer with the hash.

        Returns:
            The index in the file.
        """
        _ = key
        var b_width64 = self._BIT_WIDTH.cast[DType.uint64]()
        var b_width32 = self._BIT_WIDTH.cast[DType.uint32]()
        with open(self._file, "rb") as f:
            _ = f.seek(b_width64 * (self._index).cast[DType.uint64]())
            f.write(buf << (32 - b_width32))
        if self._index > UInt8.MAX_FINITE:
            self._index = 0
        else:
            self._index += 1
        return self._index

    fn get(self, index: UInt8) raises -> Optional[ZoneDST]:
        """Get a value from the file.

        Args:
            index: The index in the file.
        """
        if self._index > UInt8.MAX_FINITE:
            return None
        var value: UInt32
        with open(self._file, "rb") as f:
            _ = f.seek(
                self._BIT_WIDTH.cast[DType.uint64]()
                * index.cast[DType.uint64]()
            )
            var bufs = f.read_bytes(4)
            value = (
                (bufs[0].cast[DType.uint32]() << 24)
                | (bufs[1].cast[DType.uint32]() << 16)
                | (bufs[2].cast[DType.uint32]() << 8)
                | bufs[3].cast[DType.uint32]()
            )
        return ZoneDST(value & self._BIT_MASK)

    fn __del__(owned self):
        """Delete the file."""
        try:
            import os

            os.remove(self._file)
        except:
            pass


alias ZoneInfoFile32 = ZoneInfoFile(32, 0xFFFFFFFF)
"""ZoneInfoFile to store Offset of tz with DST"""
alias ZoneInfoFile8 = ZoneInfoFile(8, 0xFF)
"""ZoneInfoFile to store Offset of tz with no DST"""


@value
struct ZoneInfoMem32:
    """`ZoneInfo` that lives in memory. For zones that have DST."""

    var _zones: Dict[StringLiteral, UInt32]

    fn __init__(inout self):
        """Construct a `ZoneInfoMem32`."""

        self._zones = Dict[StringLiteral, UInt32]()

    fn add(inout self, key: StringLiteral, value: ZoneDST):
        """Add a value to `ZoneInfoMem32`.

        Args:
            key: The tz_str.
            value: Offset.
        """

        self._zones[key] = value.buf

    fn get(self, key: StringLiteral) -> Optional[ZoneDST]:
        """Get value from `ZoneInfoMem32`.

        Args:
            key: The tz_str.
        """

        var value = self._zones.find(key)
        if not value:
            return None
        return ZoneDST(value.unsafe_take())


@value
struct ZoneInfoMem8:
    """`ZoneInfo` that lives in memory. For zones that have no DST."""

    var _zones: Dict[StringLiteral, UInt8]

    fn __init__(inout self):
        """Construct a `ZoneInfoMem8`."""
        self._zones = Dict[StringLiteral, UInt8]()

    fn add(inout self, key: StringLiteral, value: Offset):
        """Add a value to `ZoneInfoMem8`.

        Args:
            key: The tz_str.
            value: Offset.
        """
        self._zones[key] = value.buf

    fn get(self, key: StringLiteral) -> Optional[Offset]:
        """Get value from `ZoneInfoMem8`.

        Args:
            key: The tz_str.
        """
        var value = self._zones.find(key)
        if not value:
            return None
        return Offset(value.unsafe_take())


fn _parse_iana_leapsecs(
    text: PythonObject,
) raises -> List[(UInt8, UInt8, UInt16)]:
    var leaps = List[(UInt8, UInt8, UInt16)]()
    var index = 0
    while True:
        var found = text.find("      #", index)
        if found == -1:
            break

        var endday = text.find(" ", found + 2)
        var day: UInt8 = atol(text.__getitem__(found + 2, endday))

        var month: UInt8 = 0
        if text.__getitem__(endday, endday + 3) == "Jan":
            month = 1
        elif text.__getitem__(endday, endday + 3) == "Jul":
            month = 7
        if month == 0:
            raise Error("month not found")

        var year: UInt16 = atol(text.__getitem__(endday + 3, endday + 7))
        leaps.append((day, month, year))
    return leaps


fn get_leapsecs() -> Optional[List[(UInt8, UInt8, UInt16)]]:
    """Get the leap seconds added to UTC.

    Returns:
        A list of tuples (day, month, year) of leapseconds.
    """
    try:
        # TODO: maybe some policy that only if x amount
        # of years have passed since latest hardcoded value
        from python import Python

        var requests = Python.import_module("requests")
        var secs = requests.get(
            "https://raw.githubusercontent.com/eggert/tz/main/leap-seconds.list"
        )
        var leapsecs = _parse_iana_leapsecs(secs.text)
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


# TODO: get_zoneinfo should be able to return a ZoneInfoMem
# or ZoneInfoFile according to parameter
alias ZoneInfo = (ZoneInfoMem32, ZoneInfoMem8)
"""ZoneInfo."""


fn get_zoneinfo() -> Optional[ZoneInfo]:
    """Get all zoneinfo available. First tries to get it
    from the OS, then from the internet, then falls back
    on hardcoded values.

    Returns:
        Optional ZoneInfo.

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
    try:
        # TODO: this should get zoneinfo from the OS it's compiled in
        # for Linux the files are under /usr/share/zoneinfo
        # no idea where they're for Windows or MacOS
        raise Error()
    except:
        pass
    try:
        # TODO
        pass
        var dst_zones = ZoneInfoMem32()
        var no_dst_zones = ZoneInfoMem8()
        from python import Python

        var json = Python.import_module("json")
        var requests = Python.import_module("requests")
        var datetime = Python.import_module("datetime")
        var text = requests.get("https://worldtimeapi.org/api/timezone").text
        var tz_list = json.loads(text)

        for item in tz_list:
            var tz = requests.get(
                "https://timeapi.io/TimeZone/" + String(item[])
            ).text
            var data = json.loads(tz)
            var utc_offset = data["standardUtcOffset"]["seconds"] // 60
            var h = int(utc_offset // 60)
            var m = int(utc_offset % 60)
            var sign = 1 if utc_offset >= 0 else -1

            var dst_start: PythonObject = ""
            var dst_end: PythonObject = ""
            if not data["hasDayLightSaving"]:
                _ = h, m, sign
                # TODO: somehow force cast python object to StringLiteral
                # no_dst_zones.add(
                #     str(item[]), Offset(abs(h), abs(m), sign)
                # )
                continue
            # -1 is to avoid Z timezone designation that
            # python's datetime doesn't like
            dst_start = data["dstInterval"]["dstStart"].__getitem__(0, -1)
            dst_end = data["dstInterval"]["dstEnd"].__getitem__(0, -1)

            var dt_start = datetime.datetime(dst_start)
            var month_start = UInt16(dst_start.month)
            var dow_start = UInt16(dt_start.weekday())
            var eom_start = UInt16(0 if dt_start <= 15 else 1)
            var week_start = 0  # TODO
            var h_start = UInt16(dt_start.hour)
            var dt_end = datetime.datetime(dst_end)
            var month_end = UInt16(dst_end.month)
            var week_end = 0  # TODO
            var h_end = UInt16(dt_end.hour)
            var dow_end = UInt16(dt_end.weekday())
            var eom_end = UInt16(0 if dt_end <= 15 else 1)

            _ = (
                dt_start,
                month_start,
                dow_start,
                eom_start,
                week_start,
                h_start,
                dt_end,
                month_end,
                week_end,
                h_end,
                dow_end,
                eom_end,
            )

            # TODO: somehow force cast python object to StringLiteral
            # dst_zones.add(
            #     item[],
            #     ZoneDST(
            #         TzDT(
            #             month_start, dow_start, eom_start, week_start, h_start
            #         ),
            #         TzDT(month_end, dow_end, eom_end, week_end, h_end),
            #         Offset(abs(h), abs(m), sign),
            #     ),
            # )
        return dst_zones, no_dst_zones
    except:
        pass
    # TODO: fallback to hardcoded
    # from ._lists import tz_list
    return None


from .calendar import PythonCalendar

alias _cal = PythonCalendar
alias all_zones = get_zoneinfo()
"""All timezones available at compile time."""


fn offset_no_dst_tz(
    owned no_dst: Optional[Offset],
) -> Optional[(UInt8, UInt8, UInt8)]:
    """Return the UTC offset for the `TimeZone` if it has no DST.

    Args:
        no_dst: Optional zone with no dst.

    Returns:
        - offset_h: Offset for the hour: [0, 15].
        - offset_m: Offset for the minute: {0, 30, 45}.
        - sign: Sign of the offset: {1, -1}.
    """

    if no_dst:
        var zone = no_dst.unsafe_take()
        var offset = zone.from_hash()
        var offset_h = offset[1]
        var offset_m = UInt8(
            0 if offset[2] == 0 else (30 if offset[2] == 1 else 45)
        )
        var sign: UInt8 = 1 if offset[0] == 0 else -1
        return offset_h, offset_m, sign
    return None


fn offset_at(
    owned with_dst: Optional[ZoneDST],
    year: UInt16,
    month: UInt8,
    day: UInt8,
    hour: UInt8,
    minute: UInt8,
    second: UInt8,
) -> Optional[(UInt8, UInt8, UInt8)]:
    """Return the UTC offset for the `TimeZone` at the given date
    if it has DST.

    Args:
        with_dst: Optional zone with DST.
        year: Year.
        month: Month.
        day: Day.
        hour: Hour.
        minute: Minute.
        second: Second.

    Returns:
        - offset_h: Offset for the hour: [0, 15].
        - offset_m: Offset for the minute: {0, 30, 45}.
        - sign: Sign of the offset: {1, -1}.
    """
    if with_dst:
        var zone = with_dst.unsafe_take()
        var items = zone.from_hash()
        var dst_start = items[0].from_hash()
        var dst_end = items[1].from_hash()
        var offset = items[2].from_hash()
        var sign: UInt8 = 1 if offset[0] == 0 else -1
        var m = UInt8(0 if offset[2] == 0 else (30 if offset[2] == 1 else 45))
        var dst_h = offset[1] + 1
        var std_m = m
        var dst_m = m
        # if it's a weird tz
        if offset[2] == 3:
            if offset[3] == 0:  # "Australia/Lord_Howe"
                dst_h = 0
                std_m = 0
                dst_m = 30
            elif offset[3] == 1:  # "Antarctica/Troll"
                dst_h += 1
                std_m = 0
                dst_m = 0

        var std = offset[1], std_m, sign
        var dst = dst_h, dst_m, sign

        fn eval_dst(
            dst_st: Bool, data: (UInt8, UInt8, UInt8, UInt8, UInt8)
        ) -> (UInt8, UInt8, UInt8):
            var is_end_mon = data[2] == 1
            var maxdays = _cal.max_days_in_month(year, month)
            var iterable = range(0, maxdays, step=1)
            if is_end_mon:
                iterable = range(maxdays - 1, -1, step=-1)

            var dow_target = data[1]
            var dow = _cal.dayofweek(year, month, day)
            var amnt_weeks_target = data[3]
            var is_later = hour > data[4] and minute > 0 and second >= 0
            var accum: UInt8 = 0
            for i in iterable:
                if _cal.dayofweek(year, month, i) == dow_target:
                    if accum != amnt_weeks_target:
                        accum += 1
                        continue
                var is_less = dow < dow_target
                var is_more = dow > dow_target
                var is_start_mon = not is_end_mon
                var is_dst_start_and_dow_is_less = dst_st and is_less
                var is_dst_end_and_dow_is_more = not dst_st and is_more

                if is_start_mon and is_dst_start_and_dow_is_less:
                    return std
                elif is_start_mon and is_dst_end_and_dow_is_more:
                    return std
                elif is_end_mon and is_dst_start_and_dow_is_less:
                    return std
                elif is_end_mon and is_dst_end_and_dow_is_more:
                    return std
                elif is_start_mon and not is_later:
                    return std
                elif is_end_mon and is_later:
                    return std
                break
            return dst

        if month == dst_start[0]:
            return eval_dst(True, dst_start)
        elif month == dst_end[0]:
            return eval_dst(False, dst_end)
        elif month > dst_start[0] and month < dst_end[0]:
            return dst
        return std
    return None
