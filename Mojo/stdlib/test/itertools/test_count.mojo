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

from std.itertools import count
from std.testing import TestSuite, assert_equal


def test_count() raises:
    var it = count()
    var elem = next(it)
    assert_equal(elem, 0)
    elem = next(it)
    assert_equal(elem, 1)


def test_count_strided() raises:
    var it = count(start=10, step=20)
    var elem = next(it)
    assert_equal(elem, 10)
    elem = next(it)
    assert_equal(elem, 30)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
