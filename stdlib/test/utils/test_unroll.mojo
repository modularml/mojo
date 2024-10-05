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

from testing import assert_equal, assert_raises

from utils import IndexList, unroll


def test_unroll():
    var indexes_seen = List[Int]()

    @parameter
    fn func[idx: Int]():
        indexes_seen.append(idx)

    unroll[func, 4]()

    assert_equal(indexes_seen[0], 0)
    assert_equal(indexes_seen[1], 1)
    assert_equal(indexes_seen[2], 2)
    assert_equal(indexes_seen[3], 3)
    assert_equal(len(indexes_seen), 4)


def test_unroll2():
    var static_tuples_seen = List[IndexList[2]]()

    @parameter
    fn func[idx0: Int, idx1: Int]():
        static_tuples_seen.append(IndexList[2](idx0, idx1))

    unroll[func, 2, 2]()

    assert_equal(static_tuples_seen[0], IndexList[2](0, 0))
    assert_equal(static_tuples_seen[1], IndexList[2](0, 1))
    assert_equal(static_tuples_seen[2], IndexList[2](1, 0))
    assert_equal(static_tuples_seen[3], IndexList[2](1, 1))
    assert_equal(len(static_tuples_seen), 4)


def test_unroll3():
    var static_tuples_seen = List[IndexList[3]]()

    @parameter
    fn func[idx0: Int, idx1: Int, idx2: Int]():
        static_tuples_seen.append(IndexList[3](idx0, idx1, idx2))

    unroll[func, 4, 2, 3]()
    assert_equal(static_tuples_seen[0], IndexList[3](0, 0, 0))
    assert_equal(static_tuples_seen[1], IndexList[3](0, 0, 1))
    assert_equal(static_tuples_seen[2], IndexList[3](0, 0, 2))
    assert_equal(static_tuples_seen[3], IndexList[3](0, 1, 0))
    assert_equal(static_tuples_seen[4], IndexList[3](0, 1, 1))
    assert_equal(static_tuples_seen[5], IndexList[3](0, 1, 2))
    assert_equal(static_tuples_seen[6], IndexList[3](1, 0, 0))
    assert_equal(static_tuples_seen[7], IndexList[3](1, 0, 1))
    assert_equal(static_tuples_seen[8], IndexList[3](1, 0, 2))
    assert_equal(static_tuples_seen[9], IndexList[3](1, 1, 0))
    assert_equal(static_tuples_seen[10], IndexList[3](1, 1, 1))
    assert_equal(static_tuples_seen[11], IndexList[3](1, 1, 2))
    assert_equal(static_tuples_seen[12], IndexList[3](2, 0, 0))
    assert_equal(static_tuples_seen[13], IndexList[3](2, 0, 1))
    assert_equal(static_tuples_seen[14], IndexList[3](2, 0, 2))
    assert_equal(static_tuples_seen[15], IndexList[3](2, 1, 0))
    assert_equal(static_tuples_seen[16], IndexList[3](2, 1, 1))
    assert_equal(static_tuples_seen[17], IndexList[3](2, 1, 2))
    assert_equal(static_tuples_seen[18], IndexList[3](3, 0, 0))
    assert_equal(static_tuples_seen[19], IndexList[3](3, 0, 1))
    assert_equal(static_tuples_seen[20], IndexList[3](3, 0, 2))
    assert_equal(static_tuples_seen[21], IndexList[3](3, 1, 0))
    assert_equal(static_tuples_seen[22], IndexList[3](3, 1, 1))
    assert_equal(static_tuples_seen[23], IndexList[3](3, 1, 2))
    assert_equal(len(static_tuples_seen), 24)


fn test_unroll_raises() raises:
    var indexes_seen = List[Int]()

    @parameter
    fn func[idx: Int]() raises:
        indexes_seen.append(idx)

    unroll[func, 4]()
    assert_equal(indexes_seen[0], 0)
    assert_equal(indexes_seen[1], 1)
    assert_equal(indexes_seen[2], 2)
    assert_equal(indexes_seen[3], 3)
    assert_equal(len(indexes_seen), 4)

    indexes_seen = List[Int]()

    @parameter
    fn func2[idx: Int]() raises:
        indexes_seen.append(idx)
        raise "Exception"

    with assert_raises(contains="Exception"):
        unroll[func2, 4]()

    assert_equal(indexes_seen[0], 0)
    assert_equal(len(indexes_seen), 1)


def test_unroll_zero_starting_range():
    var indexes_seen = List[Int]()

    @parameter
    fn func[idx: Int]():
        indexes_seen.append(idx)

    unroll[func, range(6)]()

    assert_equal(indexes_seen[0], 0)
    assert_equal(indexes_seen[1], 1)
    assert_equal(indexes_seen[2], 2)
    assert_equal(indexes_seen[3], 3)
    assert_equal(indexes_seen[4], 4)
    assert_equal(indexes_seen[5], 5)
    assert_equal(len(indexes_seen), 6)


def test_unroll_sequential_range():
    var indexes_seen = List[Int]()

    @parameter
    fn func[idx: Int]():
        indexes_seen.append(idx)

    unroll[func, range(3, 6)]()

    assert_equal(indexes_seen[0], 3)
    assert_equal(indexes_seen[1], 4)
    assert_equal(indexes_seen[2], 5)
    assert_equal(len(indexes_seen), 3)


def test_unroll_strided_range():
    var indexes_seen = List[Int]()

    @parameter
    fn func[idx: Int]():
        indexes_seen.append(idx)

    unroll[func, range(0, 9, 3)]()

    assert_equal(indexes_seen[0], 0)
    assert_equal(indexes_seen[1], 3)
    assert_equal(indexes_seen[2], 6)
    assert_equal(len(indexes_seen), 3)


fn main() raises:
    test_unroll()
    test_unroll2()
    test_unroll3()
    test_unroll_raises()
    test_unroll_zero_starting_range()
    test_unroll_sequential_range()
    test_unroll_strided_range()
