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
<<<<<<<< HEAD:stdlib/test/utils/test_format_to_stdout.mojo
# RUN: %mojo -debug-level full %s

from utils import Formattable, Formatter


fn main() raises:
    test_write_to_stdout()


@value
struct Point(Formattable):
    var x: Int
    var y: Int

    fn format_to(self, inout writer: Formatter):
        writer.write("Point(", self.x, ", ", self.y, ")")


# CHECK-LABEL: test_write_to_stdout
fn test_write_to_stdout():
    print("== test_write_to_stdout")

    var stdout = Formatter.stdout()

    # CHECK: Hello, World!
    stdout.write("Hello, World!")

    # CHECK: point = Point(1, 1)
    var point = Point(1, 1)
    stdout.write("point = ", point)
========
#
# This file only tests the debug_assert function
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s -check-prefix=CHECK-OK


def main():
    test_debug_assert_mode_all_true()


# CHECK-OK-LABEL: test_debug_assert_mode_all_true
def test_debug_assert_mode_all_true():
    print("== test_debug_assert_mode_all_true")
    debug_assert(True, "ok")
    debug_assert[assert_mode="safe"](True, "ok")
    debug_assert[assert_mode="safe", cpu_only=True](True, "ok")
    # CHECK-OK: is reached
    print("is reached")
>>>>>>>> origin/nightly:stdlib/test/builtin/test_debug_assert_mode_all.mojo
