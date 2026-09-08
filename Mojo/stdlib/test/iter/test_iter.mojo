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


from std.collections.list import _ListIter
from std.testing import TestSuite, assert_true


def test_iter() raises:
    var l: List = [1, 2, 3]
    var it = iter(l)
    assert_true(type_of(it) == _ListIter[Int, origin_of(l)])
    # Check that iter(iter(l)) is the same as iter(l)
    var _it2 = iter(it)
    assert_true(type_of(_it2) == _ListIter[Int, origin_of(l)])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
