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
# test_c_number_types.mojo
# Tests for c-ffi.mdx: "C number types" table.
#
# Verifies the std.ffi C type aliases have the widths the table documents.
from std.ffi import (
    c_char,
    c_double,
    c_float,
    c_int,
    c_long,
    c_long_long,
    c_short,
    c_size_t,
    c_ssize_t,
    c_uchar,
    c_uint,
    c_ulong,
    c_ushort,
)
from std.sys import size_of
from std.testing import assert_equal


# --- Fixed-width aliases ---


def test_fixed_width_aliases() raises:
    assert_equal(size_of[c_char](), 1)
    assert_equal(size_of[c_uchar](), 1)
    assert_equal(size_of[c_short](), 2)
    assert_equal(size_of[c_ushort](), 2)
    assert_equal(size_of[c_int](), 4)
    assert_equal(size_of[c_uint](), 4)
    assert_equal(size_of[c_long_long](), 8)
    assert_equal(size_of[c_float](), 4)
    assert_equal(size_of[c_double](), 8)


# --- Pointer-width aliases (64-bit Linux and macOS targets) ---


def test_pointer_width_aliases() raises:
    assert_equal(size_of[c_long](), 8)
    assert_equal(size_of[c_ulong](), 8)
    assert_equal(size_of[c_size_t](), 8)
    assert_equal(size_of[c_ssize_t](), 8)


def main() raises:
    test_fixed_width_aliases()
    test_pointer_width_aliases()
