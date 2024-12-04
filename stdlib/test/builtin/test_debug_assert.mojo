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
#
# This file only tests the debug_assert function
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -D DEBUG -debug-level full %s | FileCheck %s -check-prefix=CHECK-OK


def main():
    test_debug_assert()
    test_debug_assert_multiple_args()
    test_debug_assert_writable()


# CHECK-OK-LABEL: test_debug_assert
def test_debug_assert():
    print("== test_debug_assert")
    debug_assert(True, "ok")
    debug_assert(3, Error("also ok"))
    # CHECK-OK: is reached
    print("is reached")


# CHECK-OK-LABEL: test_debug_assert_multiple_args
def test_debug_assert_multiple_args():
    print("== test_debug_assert_multiple_args")
    debug_assert(True, "passing mutliple args: ", 42, ", ", 4.2)
    # CHECK-OK: is reached
    print("is reached")


# CHECK-OK-LABEL: test_debug_assert_writable
def test_debug_assert_writable():
    print("== test_debug_assert_writable")
    debug_assert(True, WritableOnly("failed with Writable arg"))
    # CHECK-OK: is reached
    print("is reached")


@value
struct WritableOnly:
    var message: String

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.message)
