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

import std.sys
from std.testing import TestSuite


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


@fieldwise_init
struct Point(Writable):
    var x: Int
    var y: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Point(", self.x, ", ", self.y, ")")


# CHECK-LABEL: test_write_to_stdout
def test_write_to_stdout() raises:
    print("== test_write_to_stdout")

    var stdout = std.sys.stdout

    # CHECK: Hello, World!
    stdout.write("Hello, World!")

    # CHECK: point = Point(1, 1)
    var point = Point(1, 1)
    stdout.write("point = ", point)
