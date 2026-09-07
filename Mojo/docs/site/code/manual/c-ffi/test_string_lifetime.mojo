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
# test_string_lifetime.mojo
# Tests for c-ffi.mdx: keeping a Mojo string alive across a C call, and
# reusing one buffer across repeated calls while its origin stays valid.
from std.ffi import OwnedDLHandle, c_char, c_size_t, external_call
from std.testing import assert_equal


# --- Keep a string alive for the duration of the C call ---


def test_keep_alive() raises:
    var s = String("x")
    var length = external_call["strlen", c_size_t](s.as_c_string_slice())
    assert_equal(Int(length), 1)
    _ = s  # manual last use keeps `s` alive until here


# --- Pin the parameter's origin to a string, refill across calls ---


def test_origin_pinned_refill() raises:
    var proc = OwnedDLHandle()  # no path: opens the current process

    var line = String("Hello")
    var c_strlen = proc.get_function[c_size_t]("strlen")

    # The pointer carries `line`'s origin, so `line` stays alive across
    # the call without a manual keep-alive.
    var length = c_strlen(line.as_c_string_slice())
    assert_equal(Int(length), 5)

    # Refill the same buffer; the origin stays valid.
    line = "Hello, Mojo!"
    length = c_strlen(line.as_c_string_slice())
    assert_equal(Int(length), 12)


def main() raises:
    test_keep_alive()
    test_origin_pinned_refill()
