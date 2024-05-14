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
"""`TimeZone` module.

- Notes:
    - IANA is supported: [`TimeZone` and DST data sources](
        http://www.iana.org/time-zones/repository/tz-link.html).
        [List of TZ identifiers (`tz_str`)](
        https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).
"""

from .zoneinfo import ZoneInfo, offset_at, offset_no_dst_tz


alias _all_zones = get_zoneinfo()


# @value
@register_passable("trivial")
struct TimeZone[
    iana: Bool = True, pyzoneinfo: Bool = True, native: Bool = False
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
        """Construct a `TimeZone`.

        Args:
            tz_str: The [`TZ identifier`](
                https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).
            offset_h: Offset for the hour.
            offset_m: Offset for the minute.
            sign: Sign: {1, -1}.
            has_dst: Whether the `TimeZone` has Daylight Saving Time.
        """
        debug_assert(
            offset_h < 100
            and offset_h >= 0
            and offset_m < 100
            and offset_m >= 0
            and (sign == 1 or sign == -1),
            msg=(
                "utc offsets can't have a member bigger than 100, "
                "and sign must be either 1 or -1"
            ),
        )

        # @parameter
        # if iana:
        #     debug_assert(
        #         iana.value()[][0].get(tz_str) or iana.value()[][1].get(tz_str),
        #         msg="that timezone is not in the given IANA ZoneInfo",
        #     )

        self.tz_str = tz_str
        self.offset_h = offset_h
        self.offset_m = offset_m
        self.sign = sign
        self.has_dst = has_dst

        @parameter
        if iana:
            if has_dst or not _all_zones:
                return
            var tz = _all_zones.value()[][1].get(tz_str)
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

        Args:
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

        @parameter
        if iana and native:
            if self.has_dst:
                var dst = _all_zones.value()[][0].get(self.tz_str)
                var offset = offset_at(
                    dst, year, month, day, hour, minute, second
                )
                if offset:
                    return offset.value()[]
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

    @always_inline("nodebug")
    fn __str__(self) -> StringLiteral:
        """Str.

        Returns:
            String.
        """
        return self.tz_str

    @always_inline("nodebug")
    fn __repr__(self) -> StringLiteral:
        """Repr.

        Returns:
            String.
        """
        return self.__str__()

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Eq.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return (
            self.tz_str == other.tz_str
            and self.offset_h == other.offset_h
            and self.offset_m == other.offset_m
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Ne.

        Args:
            other: Other.

        Returns:
            Bool.
        """
        return not (self == other)

    @always_inline("nodebug")
    fn to_iso(self) -> String:
        """Return the Offset's ISO8601 representation.

        Returns:
            String.
        """
        var sign = "+" if self.sign == 1 else "-"
        var hh = self.offset_h if self.offset_h > 9 else "0" + str(
            self.offset_h
        )
        var mm = self.offset_m if self.offset_m > 9 else "0" + str(
            self.offset_m
        )
        return sign + hh + ":" + mm

    @staticmethod
    fn from_offset(
        year: UInt16,
        month: UInt8,
        day: UInt8,
        offset_h: UInt8,
        offset_m: UInt8,
        sign: UInt8,
    ) -> Self:
        """Build a UTC TZ string from the offset.

        Args:
            year: Year.
            month: Month.
            day: Day.
            offset_h: Offset for the hour.
            offset_m: Offset for the minute.
            sign: Sign: {1, -1}.

        Returns:
            Self.
        """
        _ = year, month, day, offset_h, offset_m, sign
        # TODO: it should create an Etc/UTC-X TimeZone
        return Self()
