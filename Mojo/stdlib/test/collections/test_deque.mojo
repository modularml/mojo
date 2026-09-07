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

from std.collections import Deque

from test_utils import ExplicitDestroy, MoveOnly, Observable, check_write_to
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite

# ===-----------------------------------------------------------------------===#
# Implementation tests
# ===-----------------------------------------------------------------------===#


def test_impl_init_default() raises:
    var q = Deque[Int]()

    assert_equal(q._capacity, q.default_capacity)
    assert_equal(q._min_capacity, q.default_capacity)
    assert_equal(q._maxlen, -1)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 0)
    assert_equal(q._shrink, True)


def test_impl_init_capacity() raises:
    var q = Deque[Int](capacity=-10)
    assert_equal(q._capacity, q.default_capacity)
    assert_equal(q._min_capacity, q.default_capacity)

    q = Deque[Int](capacity=0)
    assert_equal(q._capacity, q.default_capacity)
    assert_equal(q._min_capacity, q.default_capacity)

    q = Deque[Int](capacity=10)
    assert_equal(q._capacity, 16)
    assert_equal(q._min_capacity, q.default_capacity)

    q = Deque[Int](capacity=100)
    assert_equal(q._capacity, 128)
    assert_equal(q._min_capacity, q.default_capacity)


def test_impl_init_min_capacity() raises:
    var q = Deque[Int](min_capacity=-10)
    assert_equal(q._min_capacity, q.default_capacity)
    assert_equal(q._capacity, q.default_capacity)

    q = Deque[Int](min_capacity=0)
    assert_equal(q._min_capacity, q.default_capacity)
    assert_equal(q._capacity, q.default_capacity)

    q = Deque[Int](min_capacity=10)
    assert_equal(q._min_capacity, 16)
    assert_equal(q._capacity, q.default_capacity)

    q = Deque[Int](min_capacity=100)
    assert_equal(q._min_capacity, 128)
    assert_equal(q._capacity, q.default_capacity)


def test_impl_init_maxlen() raises:
    var q = Deque[Int](maxlen=-10)
    assert_equal(q._maxlen, -1)
    assert_equal(q._capacity, q.default_capacity)

    q = Deque[Int](maxlen=0)
    assert_equal(q._maxlen, -1)
    assert_equal(q._capacity, q.default_capacity)

    q = Deque[Int](maxlen=10)
    assert_equal(q._maxlen, 10)
    assert_equal(q._capacity, 16)

    # has to allocate two times more capacity
    # when `maxlen` in a power of 2 because
    # tail should always point into a free space
    q = Deque[Int](maxlen=16)
    assert_equal(q._maxlen, 16)
    assert_equal(q._capacity, 32)

    q = Deque[Int](maxlen=100)
    assert_equal(q._maxlen, 100)
    assert_equal(q._capacity, q.default_capacity)


def test_impl_init_shrink() raises:
    var q = Deque[Int](shrink=False)
    assert_equal(q._shrink, False)
    assert_equal(q._capacity, q.default_capacity)


def test_impl_shrink_realloc_empty_deque() raises:
    # Regression test for issue #5635:
    # When capacity > min_capacity and deque is empty (head == tail),
    # _realloc must recognize this as empty (not full) and shrink correctly.
    var q = Deque[Int](capacity=8, min_capacity=4)

    # Fill to trigger growth: capacity 8 -> 16
    for i in range(8):
        q.append(i)
    assert_equal(q._capacity, 16)

    # Empty the deque
    for _ in range(8):
        _ = q.pop()
    assert_equal(len(q), 0)

    # Now capacity=16 > min_capacity=4, and deque is empty.
    # Shrink should work correctly, not copy garbage.
    assert_equal(q._capacity, 4)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 0)

    # Verify deque still works after shrink
    q.append(42)
    assert_equal(len(q), 1)
    assert_equal(q[0], 42)


def test_impl_init_list() raises:
    var q = Deque(elements=Optional(List([Int(0), 1, 2])))
    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)
    assert_equal(q._capacity, q.default_capacity)
    assert_equal(q._data[unsafe_offset=0], 0)
    assert_equal(q._data[unsafe_offset=1], 1)
    assert_equal(q._data[unsafe_offset=2], 2)

    _ = q^


def test_impl_init_list_args() raises:
    var q = Deque(elements=Optional(List([0, 1, 2])), maxlen=2, capacity=10)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 2)
    assert_equal(q._capacity, 4)
    assert_equal(q._data[unsafe_offset=0], 1)
    assert_equal(q._data[unsafe_offset=1], 2)

    _ = q^


def test_impl_init_variadic() raises:
    var q = Deque(0, 1, 2)

    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)
    assert_equal(q._capacity, q.default_capacity)
    assert_equal(q._data[unsafe_offset=0], 0)
    assert_equal(q._data[unsafe_offset=1], 1)
    assert_equal(q._data[unsafe_offset=2], 2)

    _ = q^


def test_impl_len() raises:
    var q = Deque[Int]()

    q._head = 0
    q._tail = 10
    assert_equal(len(q), 10)

    q._head = q.default_capacity - 5
    q._tail = 5
    assert_equal(len(q), 10)


def test_impl_bool() raises:
    var q = Deque[Int]()
    assert_false(q)

    q._tail = 1
    assert_true(q)


def test_impl_append() raises:
    var q = Deque[Int](capacity=2)

    q.append(0)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 1)
    assert_equal(q._capacity, 2)
    assert_equal(q._data[unsafe_offset=0], 0)

    q.append(1)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 2)
    assert_equal(q._capacity, 4)
    assert_equal(q._data[unsafe_offset=0], 0)
    assert_equal(q._data[unsafe_offset=1], 1)

    q.append(2)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)
    assert_equal(q._capacity, 4)
    assert_equal(q._data[unsafe_offset=0], 0)
    assert_equal(q._data[unsafe_offset=1], 1)
    assert_equal(q._data[unsafe_offset=2], 2)

    # simulate popleft()
    q._head += 1
    q.append(3)
    assert_equal(q._head, 1)
    # tail wrapped to the front
    assert_equal(q._tail, 0)
    assert_equal(q._capacity, 4)
    assert_equal(q._data[unsafe_offset=1], 1)
    assert_equal(q._data[unsafe_offset=2], 2)
    assert_equal(q._data[unsafe_offset=3], 3)

    q.append(4)
    # re-allocated buffer and moved all elements
    assert_equal(q._head, 0)
    assert_equal(q._tail, 4)
    assert_equal(q._capacity, 8)
    assert_equal(q._data[unsafe_offset=0], 1)
    assert_equal(q._data[unsafe_offset=1], 2)
    assert_equal(q._data[unsafe_offset=2], 3)
    assert_equal(q._data[unsafe_offset=3], 4)

    _ = q^


def test_impl_append_with_maxlen() raises:
    var q = Deque[Int](maxlen=3)

    assert_equal(q._maxlen, 3)
    assert_equal(q._capacity, 4)

    q.append(0)
    q.append(1)
    q.append(2)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)

    q.append(3)
    # first popped the leftmost element
    # so there was no re-allocation of buffer
    assert_equal(q._head, 1)
    assert_equal(q._tail, 0)
    assert_equal(q._capacity, 4)
    assert_equal(q._data[unsafe_offset=1], 1)
    assert_equal(q._data[unsafe_offset=2], 2)
    assert_equal(q._data[unsafe_offset=3], 3)

    _ = q^


def test_impl_appendleft() raises:
    var q = Deque[Int](capacity=2)

    q.appendleft(0)
    # head wrapped to the end of the buffer
    assert_equal(q._head, 1)
    assert_equal(q._tail, 0)
    assert_equal(q._capacity, 2)
    assert_equal(q._data[unsafe_offset=1], 0)

    q.appendleft(1)
    # re-allocated buffer and moved all elements
    assert_equal(q._head, 0)
    assert_equal(q._tail, 2)
    assert_equal(q._capacity, 4)
    assert_equal(q._data[unsafe_offset=0], 1)
    assert_equal(q._data[unsafe_offset=1], 0)

    q.appendleft(2)
    # head wrapped to the end of the buffer
    assert_equal(q._head, 3)
    assert_equal(q._tail, 2)
    assert_equal(q._capacity, 4)
    assert_equal(q._data[unsafe_offset=3], 2)
    assert_equal(q._data[unsafe_offset=0], 1)
    assert_equal(q._data[unsafe_offset=1], 0)

    # simulate pop()
    q._tail -= 1
    q.appendleft(3)
    assert_equal(q._head, 2)
    assert_equal(q._tail, 1)
    assert_equal(q._capacity, 4)
    assert_equal(q._data[unsafe_offset=2], 3)
    assert_equal(q._data[unsafe_offset=3], 2)
    assert_equal(q._data[unsafe_offset=0], 1)

    q.appendleft(4)
    # re-allocated buffer and moved all elements
    assert_equal(q._head, 0)
    assert_equal(q._tail, 4)
    assert_equal(q._capacity, 8)
    assert_equal(q._data[unsafe_offset=0], 4)
    assert_equal(q._data[unsafe_offset=1], 3)
    assert_equal(q._data[unsafe_offset=2], 2)
    assert_equal(q._data[unsafe_offset=3], 1)

    _ = q^


def test_impl_appendleft_with_maxlen() raises:
    var q = Deque[Int](maxlen=3)

    assert_equal(q._maxlen, 3)
    assert_equal(q._capacity, 4)

    q.appendleft(0)
    q.appendleft(1)
    q.appendleft(2)
    assert_equal(q._head, 1)
    assert_equal(q._tail, 0)

    q.appendleft(3)
    # first popped the rightmost element
    # so there was no re-allocation of buffer
    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)
    assert_equal(q._capacity, 4)
    assert_equal(q._data[unsafe_offset=0], 3)
    assert_equal(q._data[unsafe_offset=1], 2)
    assert_equal(q._data[unsafe_offset=2], 1)

    _ = q^


def test_impl_extend() raises:
    var q = Deque[Int](maxlen=4)
    var lst: List = [0, 1, 2]

    q.extend(lst.copy())
    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)
    assert_equal(q._capacity, 8)
    assert_equal(q._data[unsafe_offset=0], 0)
    assert_equal(q._data[unsafe_offset=1], 1)
    assert_equal(q._data[unsafe_offset=2], 2)

    q.extend(lst.copy())
    # has to popleft the first 2 elements
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 2)
    assert_equal(q._tail, 6)
    assert_equal(q._data[unsafe_offset=2], 2)
    assert_equal(q._data[unsafe_offset=3], 0)
    assert_equal(q._data[unsafe_offset=4], 1)
    assert_equal(q._data[unsafe_offset=5], 2)

    # turn off `maxlen` restriction
    q._maxlen = -1
    q.extend(lst.copy())
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 2)
    assert_equal(q._tail, 1)
    assert_equal(q._data[unsafe_offset=2], 2)
    assert_equal(q._data[unsafe_offset=3], 0)
    assert_equal(q._data[unsafe_offset=4], 1)
    assert_equal(q._data[unsafe_offset=5], 2)
    assert_equal(q._data[unsafe_offset=6], 0)
    assert_equal(q._data[unsafe_offset=7], 1)
    assert_equal(q._data[unsafe_offset=0], 2)

    # turn on `maxlen` and force to re-allocate
    q._maxlen = 8
    q.extend(lst.copy())
    assert_equal(q._capacity, 16)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 8)
    # has to popleft the first 2 elements
    assert_equal(q._data[unsafe_offset=0], 1)
    assert_equal(q._data[unsafe_offset=1], 2)
    assert_equal(q._data[unsafe_offset=6], 1)
    assert_equal(q._data[unsafe_offset=7], 2)

    # extend with the list that is longer than `maxlen`
    # has to pop all deque elements and some initial
    # elements from the list as well
    lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    q.extend(lst.copy())
    assert_equal(q._capacity, 16)
    assert_equal(q._head, 8)
    assert_equal(q._tail, 0)
    assert_equal(q._data[unsafe_offset=8], 2)
    assert_equal(q._data[unsafe_offset=9], 3)
    assert_equal(q._data[unsafe_offset=14], 8)
    assert_equal(q._data[unsafe_offset=15], 9)

    _ = q^


def test_impl_extendleft() raises:
    var q = Deque[Int](maxlen=4)
    var lst: List = [0, 1, 2]

    q.extendleft(lst.copy())
    # head wrapped to the end of the buffer
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 5)
    assert_equal(q._tail, 0)
    assert_equal(q._data[unsafe_offset=5], 2)
    assert_equal(q._data[unsafe_offset=6], 1)
    assert_equal(q._data[unsafe_offset=7], 0)

    q.extendleft(lst.copy())
    # popped the last 2 elements
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 2)
    assert_equal(q._tail, 6)
    assert_equal(q._data[unsafe_offset=2], 2)
    assert_equal(q._data[unsafe_offset=3], 1)
    assert_equal(q._data[unsafe_offset=4], 0)
    assert_equal(q._data[unsafe_offset=5], 2)

    # turn off `maxlen` restriction
    q._maxlen = -1
    q.extendleft(lst.copy())
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 7)
    assert_equal(q._tail, 6)
    assert_equal(q._data[unsafe_offset=7], 2)
    assert_equal(q._data[unsafe_offset=0], 1)
    assert_equal(q._data[unsafe_offset=1], 0)
    assert_equal(q._data[unsafe_offset=2], 2)
    assert_equal(q._data[unsafe_offset=3], 1)
    assert_equal(q._data[unsafe_offset=4], 0)
    assert_equal(q._data[unsafe_offset=5], 2)

    # turn on `maxlen` and force to re-allocate
    q._maxlen = 8
    q.extendleft(lst.copy())
    assert_equal(q._capacity, 16)
    assert_equal(q._head, 13)
    assert_equal(q._tail, 5)
    # has to popleft the last 2 elements
    assert_equal(q._data[unsafe_offset=13], 2)
    assert_equal(q._data[unsafe_offset=14], 1)
    assert_equal(q._data[unsafe_offset=3], 2)
    assert_equal(q._data[unsafe_offset=4], 1)

    # extend with the list that is longer than `maxlen`
    # has to pop all deque elements and some initial
    # elements from the list as well
    lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    q.extendleft(lst.copy())
    assert_equal(q._capacity, 16)
    assert_equal(q._head, 5)
    assert_equal(q._tail, 13)
    assert_equal(q._data[unsafe_offset=5], 9)
    assert_equal(q._data[unsafe_offset=6], 8)
    assert_equal(q._data[unsafe_offset=11], 3)
    assert_equal(q._data[unsafe_offset=12], 2)

    _ = q^


def test_impl_insert() raises:
    var q = Deque[Int](0, 1, 2, 3, 4, 5)

    q.insert(0, 6)
    assert_equal(q._head, q.default_capacity - 1)
    assert_equal(q._data[unsafe_offset=q._head], 6)
    assert_equal(q._data[unsafe_offset=0], 0)

    q.insert(1, 7)
    assert_equal(q._head, q.default_capacity - 2)
    assert_equal(q._data[unsafe_offset=q._head + 0], 6)
    assert_equal(q._data[unsafe_offset=q._head + 1], 7)

    q.insert(8, 8)
    assert_equal(q._tail, 7)
    assert_equal(q._data[unsafe_offset=q._tail - 1], 8)
    assert_equal(q._data[unsafe_offset=q._tail - 2], 5)

    q.insert(8, 9)
    assert_equal(q._tail, 8)
    assert_equal(q._data[unsafe_offset=q._tail - 1], 8)
    assert_equal(q._data[unsafe_offset=q._tail - 2], 9)

    _ = q^


def test_impl_pop() raises:
    var q = Deque[Int](capacity=2, min_capacity=2)
    with assert_raises():
        _ = q.pop()

    q.append(1)
    q.appendleft(2)
    assert_equal(q._capacity, 4)
    assert_equal(q.pop(), 1)
    assert_equal(len(q), 1)
    assert_equal(q[0], 2)
    assert_equal(q._capacity, 2)


def test_popleft() raises:
    var q = Deque[Int](capacity=2, min_capacity=2)
    assert_equal(q._capacity, 2)
    with assert_raises():
        _ = q.popleft()

    q.appendleft(1)
    q.append(2)
    assert_equal(q._capacity, 4)
    assert_equal(q.popleft(), 1)
    assert_equal(len(q), 1)
    assert_equal(q[0], 2)
    assert_equal(q._capacity, 2)


def test_impl_clear() raises:
    var q = Deque[Int](capacity=2)
    q.append(1)
    assert_equal(q._tail, 1)

    q.clear()
    assert_equal(q._head, 0)
    assert_equal(q._tail, 0)
    assert_equal(q._capacity, q._min_capacity)


def test_impl_add() raises:
    var l1: List = [1, 2, 3, 4, 5, 6, 7, 8]
    var l2: List = [9, 10, 11, 12, 13, 14, 15, 16]
    var q1 = Deque(elements=l1^, capacity=20, maxlen=30)
    var q2 = Deque(elements=l2^, min_capacity=200, shrink=False)

    assert_equal(q1._capacity, 32)
    assert_equal(q1._maxlen, 30)
    assert_equal(q2._capacity, 64)
    assert_equal(q2._min_capacity, 256)

    var q3 = q1 + q2
    # has to inherit q1 properties
    assert_equal(q3._capacity, 32)
    assert_equal(q3._min_capacity, 64)
    assert_equal(q3._maxlen, 30)
    assert_equal(q3._shrink, True)
    assert_equal(q3._head, 0)
    assert_equal(q3._tail, 16)
    for i, value in enumerate(q3):
        assert_equal(value, 1 + i)

    var q4 = q2 + q1
    # has to inherit q2 properties
    assert_equal(q4._capacity, 64)
    assert_equal(q4._min_capacity, 256)
    assert_equal(q4._maxlen, -1)
    assert_equal(q4._shrink, False)
    assert_equal(q4._head, 0)
    assert_equal(q4._tail, 16)
    var mid_len = len(q4) // 2
    for i in range(mid_len):
        assert_equal(q4._data[unsafe_offset=i], 9 + i)
    for i in range(mid_len, len(q4)):
        assert_equal(q4._data[unsafe_offset=i], i - 7)

    var q5 = q3 + q4
    # has to inherit q3 properties
    assert_equal(q5._capacity, 32)
    assert_equal(q5._min_capacity, 64)
    assert_equal(q5._maxlen, 30)
    assert_equal(q5._shrink, True)
    # has to obey to maxlen
    assert_equal(len(q5), 30)
    assert_equal(q5._head, 2)
    assert_equal(q5._tail, 0)
    assert_equal(q5._data[unsafe_offset=2], 3)
    assert_equal(q5._data[unsafe_offset=31], 8)
    _ = q5^

    var q6 = q4 + q3
    # has to inherit q4 properties
    assert_equal(q6._capacity, 64)
    assert_equal(q6._min_capacity, 256)
    assert_equal(q6._maxlen, -1)
    assert_equal(q6._shrink, False)
    # has to obey to maxlen
    assert_equal(len(q6), 32)
    assert_equal(q6._head, 0)
    assert_equal(q6._tail, 32)
    assert_equal(q6._data[unsafe_offset=0], 9)
    assert_equal(q6._data[unsafe_offset=31], 16)

    _ = q6^


def test_impl_iadd() raises:
    var l1: List = [1, 2, 3, 4, 5, 6, 7, 8]
    var l2: List = [9, 10, 11, 12, 13, 14, 15, 16]
    var q1 = Deque(elements=l1^, maxlen=10)
    var q2 = Deque(elements=l2^, min_capacity=200, shrink=False)

    q1 += q2
    # has to keep q1 properties
    assert_equal(q1._capacity, 16)
    assert_equal(q1._min_capacity, 64)
    assert_equal(q1._maxlen, 10)
    assert_equal(q1._shrink, True)
    # has to obey maxlen
    assert_equal(len(q1), 10)
    assert_equal(q1._head, 6)
    assert_equal(q1._tail, 0)
    for i, value in enumerate(q1):
        assert_equal(value, 7 + i)

    q2 += q1
    # has to keep q2 properties
    assert_equal(q2._capacity, 64)
    assert_equal(q2._min_capacity, 256)
    assert_equal(q2._maxlen, -1)
    assert_equal(q2._shrink, False)
    assert_equal(len(q2), 18)
    assert_equal(q2._head, 0)
    assert_equal(q2._tail, 18)
    assert_equal(q2._data[unsafe_offset=0], 9)
    assert_equal(q2._data[unsafe_offset=17], 16)

    _ = q2^


def test_impl_mul() raises:
    var l: List = [1, 2, 3]
    var q = Deque(
        elements=l^, capacity=3, min_capacity=2, maxlen=7, shrink=False
    )

    var q1 = q * 0
    assert_equal(q1._head, 0)
    assert_equal(q1._tail, 0)
    assert_equal(q1._capacity, q._min_capacity)
    assert_equal(q1._min_capacity, q._min_capacity)
    assert_equal(q1._maxlen, q._maxlen)
    assert_equal(q1._shrink, q._shrink)

    var q2 = q * 1
    assert_equal(q2._head, 0)
    assert_equal(q2._tail, len(q))
    assert_equal(q2._capacity, q._capacity)
    assert_equal(q2._min_capacity, q._min_capacity)
    assert_equal(q2._maxlen, q._maxlen)
    assert_equal(q2._shrink, q._shrink)
    assert_equal(q2._data[unsafe_offset=0], q._data[unsafe_offset=0])
    assert_equal(q2._data[unsafe_offset=1], q._data[unsafe_offset=1])
    assert_equal(q2._data[unsafe_offset=2], q._data[unsafe_offset=2])
    _ = q2^

    var q3 = q * 2
    assert_equal(q3._head, 0)
    assert_equal(q3._tail, 2 * len(q))
    assert_equal(q3._min_capacity, q._min_capacity)
    assert_equal(q3._maxlen, q._maxlen)
    assert_equal(q3._shrink, q._shrink)
    assert_equal(q3._data[unsafe_offset=0], q._data[unsafe_offset=0])
    assert_equal(q3._data[unsafe_offset=5], q._data[unsafe_offset=2])
    _ = q3^

    var q4 = q * 3
    # should obey maxlen
    assert_equal(q4._head, 2)
    assert_equal(q4._tail, 1)
    assert_equal(q4._capacity, 8)
    assert_equal(q4._min_capacity, q._min_capacity)
    assert_equal(q4._maxlen, q._maxlen)
    assert_equal(q4._shrink, q._shrink)
    assert_equal(q4._data[unsafe_offset=2], 3)
    assert_equal(q4._data[unsafe_offset=0], 3)
    _ = q4^


def test_impl_imul() raises:
    var l: List = [1, 2, 3]

    var q = Deque(
        elements=l.copy(), capacity=3, min_capacity=2, maxlen=7, shrink=False
    )
    q *= 0
    assert_equal(q._head, 0)
    assert_equal(q._tail, 0)
    # resets capacity to min_capacity
    assert_equal(q._capacity, 2)
    assert_equal(q._min_capacity, 2)
    assert_equal(q._maxlen, 7)
    assert_equal(q._shrink, False)
    _ = q^

    q = Deque(
        elements=l.copy(), capacity=3, min_capacity=2, maxlen=7, shrink=False
    )
    q *= 1
    assert_equal(q._head, 0)
    assert_equal(q._tail, len(q))
    assert_equal(q._capacity, 4)
    assert_equal(q._min_capacity, 2)
    assert_equal(q._maxlen, 7)
    assert_equal(q._shrink, False)
    assert_equal(q._data[unsafe_offset=0], 1)
    assert_equal(q._data[unsafe_offset=1], 2)
    assert_equal(q._data[unsafe_offset=2], 3)
    _ = q^

    q = Deque(
        elements=l.copy(), capacity=3, min_capacity=2, maxlen=7, shrink=False
    )
    q *= 2
    assert_equal(q._head, 0)
    assert_equal(q._tail, 6)
    assert_equal(q._capacity, 8)
    assert_equal(q._min_capacity, 2)
    assert_equal(q._maxlen, 7)
    assert_equal(q._shrink, False)
    assert_equal(q._data[unsafe_offset=0], 1)
    assert_equal(q._data[unsafe_offset=5], 3)
    _ = q^

    q = Deque(
        elements=l.copy(), capacity=3, min_capacity=2, maxlen=7, shrink=False
    )
    q *= 3
    # should obey maxlen
    assert_equal(q._head, 2)
    assert_equal(q._tail, 1)
    assert_equal(q._capacity, 8)
    assert_equal(q._min_capacity, 2)
    assert_equal(q._maxlen, 7)
    assert_equal(q._shrink, False)
    assert_equal(q._data[unsafe_offset=2], 3)
    assert_equal(q._data[unsafe_offset=0], 3)
    _ = q^


# ===-----------------------------------------------------------------------===#
# API Interface tests
# ===-----------------------------------------------------------------------===#


def test_init_variadic_list() raises:
    var lst1 = [0, 1]
    var lst2 = [2, 3]

    var q = Deque(lst1.copy(), lst2.copy())
    assert_equal(q[0], lst1)
    assert_equal(q[1], lst2)

    lst1[0] = 4
    assert_equal(q[0], [0, 1])

    var p = Deque(lst1^, lst2^)
    assert_equal(p[0], [4, 1])
    assert_equal(p[1], [2, 3])


def test_copy_trivial() raises:
    var q = Deque(1, 2, 3)

    var p = q.copy()
    assert_equal(p[0], q[0])

    p[0] = 3
    assert_equal(p[0], 3)
    assert_equal(q[0], 1)


def test_copy_list() raises:
    var q = Deque[List[Int]]()
    var lst1: List = [1, 2, 3]
    var lst2: List = [4, 5, 6]
    q.append(lst1.copy())
    q.append(lst2^)
    assert_equal(q[0], lst1)

    lst1[0] = 7
    assert_equal(q[0], [1, 2, 3])

    var p = q.copy()
    assert_equal(p[0], q[0])

    p[0][0] = 7
    assert_equal(p[0], [7, 2, 3])
    assert_equal(q[0], [1, 2, 3])


def test_move_list() raises:
    var q = Deque[List[Int]]()
    var lst1: List = [1, 2, 3]
    var lst2: List = [4, 5, 6]
    q.append(lst1.copy())
    q.append(lst2^)
    assert_equal(q[0], lst1)

    var p = q^
    assert_equal(p[0], lst1)

    lst1[0] = 7
    assert_equal(lst1[0], 7)
    assert_equal(p[0], [1, 2, 3])


def test_getitem() raises:
    var q = Deque(1, 2)
    assert_equal(q[0], 1)
    assert_equal(q[1], 2)
    assert_equal(q[len(q) - 1], 2)
    assert_equal(q[len(q) - 2], 1)


def test_setitem() raises:
    var q = Deque(1, 2)
    assert_equal(q[0], 1)

    q[0] = 3
    assert_equal(q[0], 3)

    q[len(q) - 1] = 4
    assert_equal(q[1], 4)


def test_eq() raises:
    var q = Deque[Int](1, 2, 3)
    var p = Deque[Int](1, 2, 3)

    assert_true(q == p)

    var r = Deque[Int](0, 1, 2, 3)
    q.appendleft(0)
    assert_true(q == r)


def test_ne() raises:
    var q = Deque[Int](1, 2, 3)
    var p = Deque[Int](3, 2, 1)

    assert_true(q != p)

    q.appendleft(0)
    p.append(0)
    assert_true(q != p)


def test_count() raises:
    var q = Deque(1, 2, 1, 2, 3, 1)

    assert_equal(q.count(1), 3)
    assert_equal(q.count(2), 2)
    assert_equal(q.count(3), 1)
    assert_equal(q.count(4), 0)

    q.appendleft(2)
    assert_equal(q.count(2), 3)


def test_contains() raises:
    var q = Deque[Int](1, 2, 3)

    assert_true(1 in q)
    assert_false(4 in q)


def test_index() raises:
    var q = Deque(1, 2, 1, 2, 3, 1)

    assert_equal(q.index(2), 1)
    assert_equal(q.index(2, 1), 1)
    assert_equal(q.index(2, 1, 3), 1)
    assert_equal(q.index(2, stop=4), 1)
    assert_equal(q.index(1, -12, 10), 0)
    assert_equal(q.index(1, -4), 2)
    assert_equal(q.index(1, -3), 5)
    with assert_raises():
        _ = q.index(4)


def test_insert() raises:
    var q = Deque[Int](capacity=4, maxlen=7)

    # index 0 (clamps to beginning)
    q.insert(0, 0)
    # Deque(0)
    assert_equal(q[0], 0)
    assert_equal(len(q), 1)

    # zero index
    q.insert(0, 1)
    # Deque(1, 0)
    assert_equal(q[0], 1)
    assert_equal(q[1], 0)
    assert_equal(len(q), 2)

    # positive index eq length
    q.insert(2, 2)
    # Deque(1, 0, 2)
    assert_equal(q[2], 2)
    assert_equal(q[1], 0)

    # positive index at end
    q.insert(len(q), 3)
    # Deque(1, 0, 2, 3)
    assert_equal(q[3], 3)
    assert_equal(q[2], 2)

    # assert deque buffer reallocated
    assert_equal(len(q), 4)
    assert_equal(q._capacity, 8)

    # positive index inbound
    q.insert(1, 4)
    # Deque(1, 4, 0, 2, 3)
    assert_equal(q[1], 4)
    assert_equal(q[0], 1)
    assert_equal(q[2], 0)

    # positive index inbound
    q.insert(3, 5)
    # Deque(1, 4, 0, 5, 2, 3)
    assert_equal(q[3], 5)
    assert_equal(q[2], 0)
    assert_equal(q[4], 2)

    # index from end
    q.insert(len(q) - 3, 6)
    # Deque(1, 4, 0, 6, 5, 2, 3)
    assert_equal(q[3], 6)
    assert_equal(q[2], 0)
    assert_equal(q[4], 5)

    # deque is at its maxlen
    assert_equal(len(q), 7)
    with assert_raises():
        q.insert(3, 7)


def test_remove() raises:
    var q = Deque[Int](min_capacity=32)
    q.extend([0, 1, 0, 2, 3, 0, 4, 5])
    assert_equal(len(q), 8)
    assert_equal(q._capacity, 64)

    # remove first
    q.remove(0)
    # Deque(1, 0, 2, 3, 0, 4, 5)
    assert_equal(len(q), 7)
    assert_equal(q[0], 1)
    # had to shrink its capacity
    assert_equal(q._capacity, 32)

    # remove last
    q.remove(5)
    # Deque(1, 0, 2, 3, 0, 4)
    assert_equal(len(q), 6)
    assert_equal(q[5], 4)
    # should not shrink further
    assert_equal(q._capacity, 32)

    # remove in the first half
    q.remove(0)
    # Deque(1, 2, 3, 0, 4)
    assert_equal(len(q), 5)
    assert_equal(q[1], 2)

    # remove in the last half
    q.remove(0)
    # Deque(1, 2, 3, 4)
    assert_equal(len(q), 4)
    assert_equal(q[3], 4)

    # assert raises when not found
    with assert_raises():
        q.remove(5)


def test_peek_and_peekleft() raises:
    var q = Deque[Int](capacity=4)
    assert_equal(q._capacity, 4)

    with assert_raises():
        _ = q.peek()
    with assert_raises():
        _ = q.peekleft()

    q.extend([1, 2, 3])
    assert_equal(q.peekleft(), 1)
    assert_equal(q.peek(), 3)

    _ = q.popleft()
    assert_equal(q.peekleft(), 2)
    assert_equal(q.peek(), 3)

    q.append(4)
    assert_equal(q._capacity, 4)
    assert_equal(q.peekleft(), 2)
    assert_equal(q.peek(), 4)

    q.append(5)
    assert_equal(q._capacity, 8)
    assert_equal(q.peekleft(), 2)
    assert_equal(q.peek(), 5)


def test_reverse() raises:
    var q = Deque(0, 1, 2, 3)

    q.reverse()
    assert_equal(q[0], 3)
    assert_equal(q[1], 2)
    assert_equal(q[2], 1)
    assert_equal(q[3], 0)

    q.appendleft(4)
    q.reverse()
    assert_equal(q[0], 0)
    assert_equal(q[4], 4)


def test_rotate() raises:
    var q = Deque(0, 1, 2, 3)

    q.rotate()
    assert_equal(q[0], 3)
    assert_equal(q[3], 2)

    q.rotate(-1)
    assert_equal(q[0], 0)
    assert_equal(q[3], 3)

    q.rotate(3)
    assert_equal(q[0], 1)
    assert_equal(q[3], 0)

    q.rotate(-3)
    assert_equal(q[0], 0)
    assert_equal(q[3], 3)


def test_iter() raises:
    var q = Deque(1, 2, 3)

    var i = 0
    for e in q:
        assert_equal(e, q[i])
        i += 1
    assert_equal(i, len(q))

    for ref e in q:
        if e == 1:
            e = 4
            assert_equal(e, 4)
    assert_equal(q[0], 4)


def test_iter_with_list() raises:
    var q = Deque[List[Int]]()
    var lst1: List = [1, 2, 3]
    var lst2: List = [4, 5, 6]
    q.append(lst1.copy())
    q.append(lst2.copy())
    assert_equal(len(q), 2)

    var i = 0
    for e in q:
        assert_equal(e, q[i])
        i += 1
    assert_equal(i, len(q))

    for ref e in q:
        if e == lst1:
            e[0] = 7
            assert_equal(e, [7, 2, 3])
    assert_equal(q[0], [7, 2, 3])

    for ref e in q:
        if e == lst2:
            e = [1, 2, 3]
            assert_equal(e, [1, 2, 3])
    assert_equal(q[1], [1, 2, 3])


def test_reversed_iter() raises:
    var q = Deque(1, 2, 3)

    var i = 0
    for e in reversed(q):
        assert_equal(e, q[len(q) - 1 - i])
        i += 1
    assert_equal(i, len(q))


def _test_deque_iter_bounds[
    I: Iterator
](var deque_iter: I, deque_len: Int) raises where conforms_to(
    I.Element, Deinitable
):
    var iter = deque_iter^

    for i in range(deque_len):
        var lower, upper = iter.bounds()
        assert_equal(deque_len - i, lower)
        assert_equal(deque_len - i, upper.value())
        _ = iter.__next__()

    var lower, upper = iter.bounds()
    assert_equal(0, lower)
    assert_equal(0, upper.value())


def test_deque_iter_bounds() raises:
    var deque = Deque(1, 2, 3)
    _test_deque_iter_bounds(iter(deque), len(deque))
    _test_deque_iter_bounds(reversed(deque), len(deque))


def test_deque_literal() raises:
    var q: Deque[Int] = [1, 2, 3]
    assert_equal(3, len(q))
    assert_equal(1, q[0])
    assert_equal(2, q[1])
    assert_equal(3, q[2])

    var q2: Deque[Float64] = [1, 2.5]
    assert_equal(2, len(q2))
    assert_equal(1.0, q2[0])
    assert_equal(2.5, q2[1])

    var q3: Deque[Int] = []
    assert_equal(0, len(q3))


def test_repr_wrap() raises:
    var s = Deque[String]("a", "b", "c")
    assert_equal(
        repr(s),
        "Deque[String](['a', 'b', 'c'])",
    )


def test_write_to() raises:
    """Test Writable trait implementation."""
    check_write_to(
        Deque[Int](10, 20, 30), expected="[10, 20, 30]", is_repr=False
    )
    check_write_to(
        Deque[String]("a", "b", "c"), expected="[a, b, c]", is_repr=False
    )
    check_write_to(Deque[Int](), expected="[]", is_repr=False)
    check_write_to(Deque[Int](42), expected="[42]", is_repr=False)


def test_write_repr_to() raises:
    """Test write_repr_to implementation."""
    check_write_to(
        Deque[Int](1, 2, 3),
        expected="Deque[SIMD[DType.int, 1]]([Int(1), Int(2), Int(3)])",
        is_repr=True,
    )
    check_write_to(
        Deque[Int](1),
        expected="Deque[SIMD[DType.int, 1]]([Int(1)])",
        is_repr=True,
    )
    check_write_to(
        Deque[Int](), expected="Deque[SIMD[DType.int, 1]]([])", is_repr=True
    )


struct NonEquatable(Copyable):
    pass


struct CopyableExplicitDestroy(Copyable, Deinitable where False):
    """Test type that is `Copyable` but must be explicitly destroyed."""

    var value: Int
    """Int data."""

    @implicit
    def __init__(out self, value: Int):
        """Constructs a new instance.

        Args:
            value: The integer value to store.
        """
        self.value = value

    def __init__(out self, *, copy: Self):
        """Copies from another instance.

        Args:
            copy: The instance being copied from.
        """
        self.value = copy.value

    def destroy(deinit self):
        """Destroys self."""
        pass


def test_deque_conditional_conformances() raises:
    assert_true(conforms_to(Deque[Int], Equatable))
    assert_false(conforms_to(Deque[NonEquatable], Equatable))

    assert_true(conforms_to(Deque[Int], Hashable))
    assert_false(conforms_to(Deque[NonEquatable], Hashable))

    # Verify equal deques produce equal hashes.
    var d1 = Deque[Int](1, 2, 3)
    var d2 = Deque[Int](1, 2, 3)
    assert_equal(hash(d1), hash(d2))

    assert_true(conforms_to(Deque[Int], Writable))
    assert_false(conforms_to(Deque[NonEquatable], Writable))

    # `Deinitable` is conditional on the element type.
    assert_true(conforms_to(Deque[Int], Deinitable))
    assert_false(conforms_to(Deque[ExplicitDestroy], Deinitable))

    # Owned iteration requires `Deinitable` elements; consuming
    # iteration moves elements out, so it no longer requires `Copyable`.
    assert_true(conforms_to(Deque[Int], IterableOwned))
    assert_false(conforms_to(Deque[ExplicitDestroy], IterableOwned))
    assert_true(conforms_to(Deque[MoveOnly[Int]], IterableOwned))

    # A `Copyable` but non-`Deinitable` element type makes the deque
    # `Copyable` (copying never destroys an element) yet still linear.
    assert_true(conforms_to(Deque[CopyableExplicitDestroy], Copyable))
    assert_false(conforms_to(Deque[CopyableExplicitDestroy], Deinitable))


def test_deque_with_explicit_destroy_type() raises:
    var deque = Deque[ExplicitDestroy](
        ExplicitDestroy(0), ExplicitDestroy(1), ExplicitDestroy(2)
    )

    var destroyed = List[Int]()

    def destroy_closure(var e: ExplicitDestroy) {mut}:
        destroyed.append(e.value)
        e^.destroy()

    deque^.deinit_with(destroy_closure)

    assert_equal(destroyed, [0, 1, 2])


def test_deque_copy_copyable_explicit_destroy_type() raises:
    # Copying requires only `Copyable`, not `Deinitable`: a deque of a
    # `Copyable` but linear element type can be copied, and both the original
    # and the copy must then be drained explicitly.
    var deque = Deque[CopyableExplicitDestroy](
        CopyableExplicitDestroy(0),
        CopyableExplicitDestroy(1),
        CopyableExplicitDestroy(2),
    )
    var deque_copy = deque.copy()

    var destroyed = List[Int]()

    def destroy_closure(var e: CopyableExplicitDestroy) {mut}:
        destroyed.append(e.value)
        e^.destroy()

    deque^.deinit_with(destroy_closure)
    deque_copy^.deinit_with(destroy_closure)

    assert_equal(destroyed, [0, 1, 2, 0, 1, 2])


def test_deque_empty_deinit_with() raises:
    var deque = Deque[ExplicitDestroy]()

    var destroyed = List[Int]()

    def destroy_closure(var e: ExplicitDestroy) {mut}:
        destroyed.append(e.value)
        e^.destroy()

    deque^.deinit_with(destroy_closure)

    assert_equal(len(destroyed), 0)


# ===-------------------------------------------------------------------===#
# Owned iteration tests
# ===-------------------------------------------------------------------===#

# We use `MutAnyOrigin` to bypass exclusivity checking
# otherwise we cannot construct a deque of Observables where
# all point to the same copy/move/deinit counter.
comptime ObservableElement = Observable[
    CopyOrigin=MutAnyOrigin,
    MoveOrigin=MutAnyOrigin,
    DelOrigin=MutAnyOrigin,
]


def make_observable_deque(
    *, mut copies: Int, mut moves: Int, mut dels: Int, length: Int
) -> Deque[ObservableElement]:
    var deque = Deque[ObservableElement]()
    for _i in range(length):
        deque.append(
            ObservableElement(
                copies=Pointer[Int, MutAnyOrigin](to=copies),
                moves=Pointer[Int, MutAnyOrigin](to=moves),
                dels=Pointer[Int, MutAnyOrigin](to=dels),
            )
        )
    return deque^


def test_deque_iter_owned() raises:
    var deque = Deque[Int](1, 2, 3, 4, 5)
    var result = List[Int]()
    for elem in deque^:
        result.append(elem)
    assert_equal(len(result), 5)
    assert_equal(result[0], 1)
    assert_equal(result[1], 2)
    assert_equal(result[2], 3)
    assert_equal(result[3], 4)
    assert_equal(result[4], 5)


def test_deque_iter_owned_destroys_elements_if_not_consumed() raises:
    var copies = 0
    var moves = 0
    var dels = 0

    var deque = make_observable_deque(
        copies=copies, moves=moves, dels=dels, length=2
    )
    var _ = deque^.__iter__()
    assert_equal(copies, 0)
    assert_equal(dels, 2)


def test_deque_iter_owned_destroys_elements_if_partially_consumed() raises:
    var copies = 0
    var moves = 0
    var dels = 0

    var deque = make_observable_deque(
        copies=copies, moves=moves, dels=dels, length=2
    )

    var iter = deque^.__iter__()
    assert_equal(copies, 0)
    assert_equal(dels, 0)

    var _ = iter.__next__()
    assert_equal(dels, 1)

    _ = iter^
    assert_equal(copies, 0)
    assert_equal(dels, 2)


def test_deque_iter_owned_move_only() raises:
    # Owned iteration moves elements out, so it works for move-only
    # (non-`Copyable`) element types.
    var d = Deque[MoveOnly[Int]]()
    d.append(MoveOnly[Int](1))
    d.append(MoveOnly[Int](2))
    d.append(MoveOnly[Int](3))

    var result = List[Int]()
    for elem in d^:
        result.append(elem.data)
    assert_equal(result, [1, 2, 3])


def test_deque_iter_owned_bounds() raises:
    var deque = Deque[Int](1, 2, 3)
    var iter = deque^.__iter__()
    for i in range(3, 0, -1):
        var lower, upper = iter.bounds()
        assert_equal(i, lower)
        assert_equal(i, upper.value())
        _ = iter.__next__()

    var lower, upper = iter.bounds()
    assert_equal(0, lower)
    assert_equal(0, upper.value())


def test_deque_move_only() raises:
    # `MoveOnly[Int]` is not `Copyable`; this exercises the conditional
    # conformance path of `Deque[T: Movable & Deinitable]`.
    assert_false(conforms_to(Deque[MoveOnly[Int]], Copyable))

    var d = Deque[MoveOnly[Int]]()
    d.append(MoveOnly[Int](0))
    d.append(MoveOnly[Int](1))
    d.appendleft(MoveOnly[Int](-1))
    assert_equal(len(d), 3)
    assert_equal(d[0], MoveOnly[Int](-1))
    assert_equal(d[1], MoveOnly[Int](0))
    assert_equal(d[2], MoveOnly[Int](1))

    # Methods that take/move don't require `Copyable`.
    var right = d.pop()
    assert_equal(right, MoveOnly[Int](1))
    var left = d.popleft()
    assert_equal(left, MoveOnly[Int](-1))
    assert_equal(len(d), 1)

    d.insert(0, MoveOnly[Int](42))
    assert_equal(d[0], MoveOnly[Int](42))
    assert_equal(d[1], MoveOnly[Int](0))

    d.clear()
    assert_equal(len(d), 0)


# ===-------------------------------------------------------------------===#
# main
# ===-------------------------------------------------------------------===#


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
