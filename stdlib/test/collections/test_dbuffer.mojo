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

from collections import InlineArray, List
from testing import assert_equal, assert_true

from collections.dbuffer import DBuffer


def test_dbuffer_list_init_trivial():
    # test taking ownership
    var l1 = List[Int](1, 2, 3, 4, 5, 6, 7)
    var l1_copy = List(other=l1)
    var s1 = DBuffer[origin=MutableAnyOrigin].own(l1^)
    assert_true(s1.is_owner())
    assert_equal(len(s1), len(l1_copy))
    for i in range(len(s1)):
        assert_equal(l1_copy[i], s1[i])
    # subslice
    var slice_1 = s1[2:]
    assert_true(not slice_1.is_owner())
    assert_equal(slice_1[0], l1_copy[2])
    assert_equal(slice_1[1], l1_copy[3])
    assert_equal(slice_1[2], l1_copy[4])
    assert_equal(slice_1[3], l1_copy[5])
    assert_equal(s1[-1], l1_copy[-1])

    # test non owning Buffer
    var l2 = List[Int](1, 2, 3, 4, 5, 6, 7)
    var s2 = DBuffer(l2)
    assert_true(not s2.is_owner())
    assert_equal(len(s2), len(l2))
    for i in range(len(s2)):
        assert_equal(l2[i], s2[i])
    # subslice
    var slice_2 = s2[2:]
    assert_true(not slice_2.is_owner())
    assert_equal(slice_2[0], l2[2])
    assert_equal(slice_2[1], l2[3])
    assert_equal(slice_2[2], l2[4])
    assert_equal(slice_2[3], l2[5])
    assert_equal(s2[-1], l2[-1])

    # Test mutation
    s2[0] = 9
    assert_equal(s2[0], 9)
    assert_equal(l2[0], 9)

    s2[-1] = 0
    assert_equal(s2[-1], 0)
    assert_equal(l2[-1], 0)


def test_dbuffer_list_init_memory():
    # test taking ownership
    var l1 = List[String]("a", "b", "c", "d", "e", "f", "g")
    var l1_copy = List(other=l1)
    var s1 = DBuffer[origin=MutableAnyOrigin].own(l1^)
    assert_true(s1.is_owner())
    assert_equal(len(s1), len(l1_copy))
    for i in range(len(s1)):
        assert_equal(l1_copy[i], s1[i])
    # subslice
    var slice_1 = s1[2:]
    assert_true(not slice_1.is_owner())
    assert_equal(slice_1[0], l1_copy[2])
    assert_equal(slice_1[1], l1_copy[3])
    assert_equal(slice_1[2], l1_copy[4])
    assert_equal(slice_1[3], l1_copy[5])

    # test non owning Buffer
    var l2 = List[String]("a", "b", "c", "d", "e", "f", "g")
    var s2 = DBuffer(l2)
    assert_true(not s2.is_owner())
    assert_equal(len(s2), len(l2))
    for i in range(len(s2)):
        assert_equal(l2[i], s2[i])
    # subslice
    var slice_2 = s2[2:]
    assert_true(not slice_2.is_owner())
    assert_equal(slice_2[0], l2[2])
    assert_equal(slice_2[1], l2[3])
    assert_equal(slice_2[2], l2[4])
    assert_equal(slice_2[3], l2[5])

    # Test mutation
    s2[0] = "h"
    assert_equal(s2[0], "h")
    assert_equal(l2[0], "h")

    s2[-1] = "i"
    assert_equal(s2[-1], "i")
    assert_equal(l2[-1], "i")


def test_dbuffer_array_int():
    var l = InlineArray[Int, 7](1, 2, 3, 4, 5, 6, 7)
    var s = DBuffer[Int](array=l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])

    # Test mutation
    s[0] = 9
    assert_equal(s[0], 9)
    assert_equal(l[0], 9)

    s[-1] = 0
    assert_equal(s[-1], 0)
    assert_equal(l[-1], 0)


def test_dbuffer_array_str():
    var l = InlineArray[String, 7]("a", "b", "c", "d", "e", "f", "g")
    var s = DBuffer[String](array=l)
    assert_true(not s.is_owner())
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])

    # Test mutation
    s[0] = "h"
    assert_equal(s[0], "h")
    assert_equal(l[0], "h")

    s[-1] = "i"
    assert_equal(s[-1], "i")
    assert_equal(l[-1], "i")


def test_indexing():
    var l = InlineArray[Int, 7](1, 2, 3, 4, 5, 6, 7)
    var s = DBuffer[Int](array=l)
    assert_equal(s[True], 2)
    assert_equal(s[int(0)], 1)
    assert_equal(s[3], 4)


def test_dbuffer_slice():
    def compare(s: DBuffer[Int], l: List[Int]) -> Bool:
        if len(s) != len(l):
            return False
        for i in range(len(s)):
            if s[i] != l[i]:
                return False
        return True

    var l = List(1, 2, 3, 4, 5)
    var s = DBuffer(l)
    var res = s[1:2]
    assert_equal(res[0], 2)
    res = s[1:-1:1]
    assert_equal(res[0], 2)
    assert_equal(res[1], 3)
    assert_equal(res[2], 4)
    # Test slicing with negative step
    res = s[1::-1]
    assert_equal(res[0], 2)
    assert_equal(res[1], 1)
    res = s[2:1:-1]
    assert_equal(res[0], 3)
    assert_equal(len(res), 1)
    res = s[5:1:-2]
    assert_equal(res[0], 5)
    assert_equal(res[1], 3)


def test_bool():
    var l = InlineArray[String, 7]("a", "b", "c", "d", "e", "f", "g")
    var s = DBuffer[String](l)
    assert_true(s)
    assert_true(not s[0:0])


def test_equality():
    var l = InlineArray[String, 7]("a", "b", "c", "d", "e", "f", "g")
    var l2 = List[String]("a", "b", "c", "d", "e", "f", "g")
    var sp = DBuffer[String](l)
    var sp2 = DBuffer[String](l)
    var sp3 = DBuffer(l2)
    # same pointer
    assert_true(sp == sp2)
    # different pointer
    assert_true(sp == sp3)
    # different length
    assert_true(sp != sp3[:-1])
    # empty
    assert_true(sp[0:0] == sp3[0:0])


def test_fill():
    var l1 = List[Int](0, 1, 2, 3, 4, 5, 6, 7, 8)
    var s1 = DBuffer(l1)

    s1.fill(2)

    for i in range(len(l1)):
        assert_equal(l1[i], 2)
        assert_equal(s1[i], 2)

    var l2 = List[String]("a", "b", "c", "d", "e", "f", "g")
    var s2 = DBuffer(l2)

    s2.fill("hi")

    for i in range(len(s2)):
        assert_equal(l2[i], "hi")
        assert_equal(s2[i], "hi")


def main():
    test_dbuffer_list_init_trivial()
    test_dbuffer_list_init_memory()
    test_dbuffer_array_int()
    test_dbuffer_array_str()
    test_indexing()
    test_dbuffer_slice()
    test_bool()
    test_equality()
    test_fill()
