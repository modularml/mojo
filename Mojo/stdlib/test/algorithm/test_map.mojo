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

from std.algorithm import map
from std.testing import assert_equal
from std.testing import TestSuite


def test_map() raises:
    var vector_stack: Array[Float32, 5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var vector = Span(vector_stack)

    def add_two(idx: Int) {var}:
        vector[idx] = vector[idx] + 2

    map(len(vector), add_two)

    assert_equal(vector[0], 3.0)
    assert_equal(vector[1], 4.0)
    assert_equal(vector[2], 5.0)
    assert_equal(vector[3], 6.0)
    assert_equal(vector[4], 7.0)

    def add(idx: Int) {var}:
        vector[idx] = vector[idx] + vector[idx]

    map(len(vector), add)

    assert_equal(vector[0], 6.0)
    assert_equal(vector[1], 8.0)
    assert_equal(vector[2], 10.0)
    assert_equal(vector[3], 12.0)
    assert_equal(vector[4], 14.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
