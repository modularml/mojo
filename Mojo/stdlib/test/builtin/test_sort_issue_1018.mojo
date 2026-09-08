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

from std.random import rand

from std.testing import TestSuite


def _sort_test[dtype: DType, name: StaticString](size: Int, max: Int) raises:
    var p = List(length=size, fill=Scalar[dtype](0))
    rand(p)
    sort(p)
    for i in range(1, size - 1):
        if p[i] < p[i - 1]:
            print(name, "size:", size, "max:", max, "incorrect sort")
            print("p[", end="")
            print(i - 1, end="")
            print("] =", p[i - 1])
            print("p[", end="")
            print(i, end="")
            print("] =", p[i])
            print()
            raise Error("Failed")


def test_sort_issue_1018() raises:
    _sort_test[.int8, "int8"](300, 3_000)
    _sort_test[.float32, "float32"](3_000, 3_000)
    _sort_test[.float64, "float64"](300_000, 3_000_000_000)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
