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
# test_shared_libraries.mojo
# Tests for c-ffi.mdx: "Use shared libraries" and "Retrieve functions by name".
#
# Not tested at runtime (intentional):
#   - The libncursesw and libcurl examples load libraries that aren't
#     guaranteed present in the test environment. This test exercises the
#     same OwnedDLHandle + get_function path against a guaranteed libc
#     symbol (abs), which stands in for those examples.
from std.ffi import OwnedDLHandle, c_int
from std.testing import assert_equal


# --- get_function, parameterized on the C return type ---


def test_get_function() raises:
    var proc = OwnedDLHandle()  # no path: opens the current process
    var c_abs = proc.get_function[c_int]("abs")
    assert_equal(c_abs(c_int(-5)), c_int(5))
    assert_equal(c_abs(c_int(5)), c_int(5))


def main() raises:
    test_get_function()
