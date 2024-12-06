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
# RUN: %mojo %s

from testing import assert_equal
from utils import StaticTuple


def test_getitem():
    # Should be constructible from a single element
    # as well as a variadic list of elements.
    var tup1 = StaticTuple[Int, 1](1)
    assert_equal(tup1[0], 1)

    var tup2 = StaticTuple[Int, 2](1, 1)
    assert_equal(tup2[0], 1)
    assert_equal(tup2[1], 1)

    var tup3 = StaticTuple[Int, 3](1, 2, 3)
    assert_equal(tup3[0], 1)
    assert_equal(tup3[1], 2)
    assert_equal(tup3[2], 3)

    assert_equal(tup1[Int(0)], 1)


def test_setitem():
    var t = StaticTuple[Int, 3](1, 2, 3)

    t[0] = 100
    assert_equal(t[0], 100)

    t[1] = 200
    assert_equal(t[1], 200)

    t[2] = 300
    assert_equal(t[2], 300)

    alias idx: Int = 0
    t.__setitem__[idx](400)
    assert_equal(t[0], 400)


def main():
    test_getitem()
    test_setitem()
