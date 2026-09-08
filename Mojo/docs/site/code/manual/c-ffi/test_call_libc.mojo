# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
# test_call_libc.mojo
# Tests for c-ffi.mdx: "Call libc functions", including the clock_gettime
# struct-by-pointer example and the div struct-by-value example.
from std.ffi import (
    OwnedDLHandle,
    c_char,
    c_int,
    c_long,
    c_size_t,
    external_call,
)
from std.sys.info import platform_map
from std.testing import assert_equal, assert_true


# --- external_call: rand ---


def random_int() -> c_int:
    return external_call["rand", c_int]()


def test_rand() raises:
    # rand() returns a non-negative int; 1 + (r % 10) lands in [1, 10].
    var raw = random_int()
    assert_true(raw >= 0)
    var bounded = 1 + raw % 10
    assert_true(bounded >= 1)
    assert_true(bounded <= 10)


# --- external_call: struct passed by pointer (clock_gettime) ---


@fieldwise_init
struct CTimeSpec(RegisterPassable):  # Matches C's struct timespec.
    # CLOCK_MONOTONIC differs by platform
    comptime monotonic = c_int(
        platform_map["CLOCK_MONOTONIC", linux=1, macos=6]()
    )

    var tv_sec: c_long
    var tv_nsec: c_long

    @staticmethod
    def monotonic_nanos() raises -> c_long:
        var time_spec = Self(0, 0)
        if (
            external_call["clock_gettime", c_int](
                Self.monotonic,
                Pointer(to=time_spec),
            )
            != 0
        ):
            raise Error("clock_gettime failed")
        return time_spec.tv_sec * 1_000_000_000 + time_spec.tv_nsec


def test_clock_gettime() raises:
    var first = CTimeSpec.monotonic_nanos()
    assert_true(first > 0)
    var second = CTimeSpec.monotonic_nanos()
    assert_true(second >= first)  # monotonic clock never goes backward


# --- get_function: struct returned by value (div) ---


@fieldwise_init
struct DivT(RegisterPassable):  # Mirrors C div_t: two ints, 8 bytes.
    var quot: c_int
    var rem: c_int


def test_div_struct_by_value() raises:
    var proc = OwnedDLHandle()  # no path: opens the current process
    var div = proc.get_function[DivT]("div")
    var d = div(c_int(7), c_int(3))
    assert_equal(d.quot, c_int(2))
    assert_equal(d.rem, c_int(1))


# --- external_call: variadic callee needs num_fixed_args (snprintf) ---


def test_snprintf_variadic() raises:
    var buf = Array[c_char, 64](uninitialized=True)
    var written = external_call["snprintf", c_int, num_fixed_args=3](
        buf.unsafe_ptr(),
        c_size_t(64),
        "score: %d/%d".as_c_string_slice(),
        c_int(7),
        c_int(10),
    )
    assert_equal(written, c_int(11))
    assert_equal(
        String(unsafe_from_utf8_ptr=buf.unsafe_ptr()), String("score: 7/10")
    )


def main() raises:
    test_rand()
    test_clock_gettime()
    test_div_struct_by_value()
    test_snprintf_variadic()
