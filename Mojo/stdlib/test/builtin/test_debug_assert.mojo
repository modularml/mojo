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
#
# This file only tests the debug_assert function
#
# ===----------------------------------------------------------------------=== #

from std.testing import TestSuite


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# CHECK-LABEL: test_debug_assert
def test_debug_assert() raises:
    print("== test_debug_assert")
    debug_assert(True, "ok")
    debug_assert(Bool(3), Error("also ok"))
    # CHECK: is reached
    print("is reached")


# CHECK-LABEL: test_debug_assert_multiple_args
def test_debug_assert_multiple_args() raises:
    print("== test_debug_assert_multiple_args")
    debug_assert(True, "passing multiple args: ", 42, ", ", 4.2)
    # CHECK: is reached
    print("is reached")


# CHECK-LABEL: test_debug_assert_writable
def test_debug_assert_writable() raises:
    print("== test_debug_assert_writable")
    debug_assert(True, WritableOnly("failed with Writable arg"))
    # CHECK: is reached
    print("is reached")


@fieldwise_init
struct WritableOnly(Writable):
    var message: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.message)
