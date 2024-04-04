# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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

from builtin.divmod import divmod


fn _divmod_no_raise(a: Int, b: Int) -> Tuple[Int, Int]:
    debug_assert(b!=0, "We should never try to divide by 0")
    try:
        return divmod(a, b)
    except:
        # this should never happen
        return Tuple(0, 0)


struct timedelta(CollectionElement):
    """Represents a time difference. Is accurate to the microsecond.

    The internal representation is (days, seconds, microseconds).

    You can get access to some useful constants:
    ```mojo
    import datetime as dt

    dt.timedelta.min
    # timedelta(-999999999)
    # The lowest value which can be represented by dt.timedelta

    dt.timedelta.max
    # timedelta(days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999)
    # The highest value which can be represented by dt.timedelta

    dt.timedelta.resolution
    # timedelta(microseconds=1)
    # The smallest duration which can be represented by timedelta.
    ```

    You can read the following attributes:
    ```mojo
    import datetime as dt

    duration = dt.timedelta(milliseconds=123456789)
    print(duration.days)
    # 1
    print(duration.seconds)
    # 37056
    print(durations.microseconds)
    # 789000
    ```

    Those attributes must NOT be set manually.
    You can consider a `dt.timedelta` object to be immutable.

    The internal representation ensures that:
    ```mojo
    0 <= microseconds < 1_000_000
    0 <= seconds < 24 * 60 * 60
    -99999999 <= days <= 999999999
    ```
    """

    var days: Int
    var seconds: Int
    var microseconds: Int

    alias min = Self(-999999999)
    alias max = Self(
        days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999
    )
    alias resolution = Self(microseconds=1)

    fn __init__(
        inout self,
        owned days: Int = 0,
        owned seconds: Int = 0,
        owned microseconds: Int = 0,
        milliseconds: Int = 0,
        minutes: Int = 0,
        hours: Int = 0,
        weeks: Int = 0,
    ):
        """Creates a `dt.timedelta` struct from integer values.

        Each of the values provided can be positive or negative.

        The durations provided by each of the argument are summed together
        to provide the final `dt.timedelta`. For example

        ```mojo
        import datetime as dt

        print(dt.timedelta(minutes=2, seconds=5))
        # datetime.timedelta(seconds=125)
        ```

        Args:
            days: The number of days.
            seconds: The number of seconds.
            microseconds: The number of microseconds.
            milliseconds: The number of milliseconds.
            minutes: The number of minutes.
            hours: The number of hours.
            weeks: The number of weeks.
        """
        # We keep only days, seconds, microseconds
        microseconds += milliseconds * 1000
        seconds += minutes * 60 + hours * 3600
        days += weeks * 7
        self.__init__(days, seconds, microseconds, are_normalized=False)

    fn __init__(
        inout self,
        owned days: Int = 0,
        owned seconds: Int = 0,
        owned microseconds: Int = 0,
        *,
        are_normalized: Bool,
    ):
        """This constructor is similar to the constructor described above, but is faster.

        1) Becase there is no need to convert all quantities to (days, seconds, microseconds)
        2) Because if you know the values are normalized, you can also skip the normalization.

        The value are normalized if they are in the following range:
        ```mojo
        0 <= microseconds < 1_000_000
        0 <= seconds < 24 * 60 * 60
        -99999999 <= days <= 999999999
        ```

        If `are_normalized=False`, the constructor will perform the normalization.

        Args:
            days: The number of days.
            seconds: The number of seconds.
            microseconds: The number of microseconds.
            are_normalized: Set this argument to `True` if you know that the previous
                arguments are alread in the correct range.
        """

        if not are_normalized:
            var extra_seconds: Int
            extra_seconds, microseconds = _divmod_no_raise(
                microseconds, 1000000
            )
            seconds += extra_seconds

            var extra_days: Int
            extra_days, seconds = _divmod_no_raise(seconds, 24 * 60 * 60)
            days += extra_days

        debug_assert(
            0 <= microseconds < 1000000,
            "microseconds should be in the range [0, 1000000[",
        )
        debug_assert(
            0 <= seconds < 24 * 60 * 60,
            "seconds should be in the range [0, 24 * 60 * 60[",
        )
        debug_assert(
            -99999999 <= days <= 999999999,
            "days should be in the range -999999999 to 999999999",
        )

        self.days = days
        self.seconds = seconds
        self.microseconds = microseconds

    # TODO: Use @value when https://github.com/modularml/mojo/issues/1705 is fixed
    fn __copyinit__(inout self, existing: Self):
        self.days = existing.days
        self.seconds = existing.seconds
        self.microseconds = existing.microseconds

    fn __moveinit__(inout self, owned existing: Self):
        self.days = existing.days
        self.seconds = existing.seconds
        self.microseconds = existing.microseconds
