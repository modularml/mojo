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
# RUN: %mojo -debug-level full %s

from builtin.io import _print_fmt

from utils._format import Formattable, Formatter, write_to


fn main() raises:
    test_write_to_stdout()


@value
struct Point(Formattable):
    var x: Int
    var y: Int

    fn format_to(self, inout writer: Formatter):
        write_to(writer, "Point(", self.x, ", ", self.y, ")")


# CHECK-LABEL: test_write_to_stdout
fn test_write_to_stdout():
    print("== test_write_to_stdout")

    var stdout = Formatter.stdout()

    # CHECK: Hello, World!
    write_to(stdout, "Hello, World!")

    # CHECK: point = Point(1, 1)
    var point = Point(1, 1)
    write_to(stdout, "point = ", point)


# CHECK-LABEL: test_print_fmt()
fn test_print_fmt():
    print("== test_print_fmt")

    # CHECK: The quick brown fox...
    _print_fmt("The quick brown fox...")

    # CHECK: ...jumps over the lazy ..Point(2, 5) ???
    _print_fmt("...jumps over the lazy ..", Point(2, 5), " ???")
