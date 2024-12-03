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

from collections import Deque

from testing import assert_equal, assert_false, assert_raises, assert_true

# ===-----------------------------------------------------------------------===#
# Implementation tests
# ===-----------------------------------------------------------------------===#


fn test_impl_init_default() raises:
    q = Deque[Int]()

    assert_equal(q._capacity, q.default_capacity)
    assert_equal(q._min_capacity, q.default_capacity)
    assert_equal(q._maxlen, -1)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 0)
    assert_equal(q._shrink, True)


fn test_impl_init_capacity() raises:
    q = Deque[Int](capacity=-10)
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


fn test_impl_init_min_capacity() raises:
    q = Deque[Int](min_capacity=-10)
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


fn test_impl_init_maxlen() raises:
    q = Deque[Int](maxlen=-10)
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


fn test_impl_init_shrink() raises:
    q = Deque[Int](shrink=False)
    assert_equal(q._shrink, False)
    assert_equal(q._capacity, q.default_capacity)


fn test_impl_init_list() raises:
    q = Deque(elements=List(0, 1, 2))
    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)
    assert_equal(q._capacity, q.default_capacity)
    assert_equal((q._data + 0)[], 0)
    assert_equal((q._data + 1)[], 1)
    assert_equal((q._data + 2)[], 2)


fn test_impl_init_list_args() raises:
    q = Deque(elements=List(0, 1, 2), maxlen=2, capacity=10)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 2)
    assert_equal(q._capacity, 4)
    assert_equal((q._data + 0)[], 1)
    assert_equal((q._data + 1)[], 2)


fn test_impl_init_variadic() raises:
    q = Deque(0, 1, 2)

    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)
    assert_equal(q._capacity, q.default_capacity)
    assert_equal((q._data + 0)[], 0)
    assert_equal((q._data + 1)[], 1)
    assert_equal((q._data + 2)[], 2)


fn test_impl_len() raises:
    q = Deque[Int]()

    q._head = 0
    q._tail = 10
    assert_equal(len(q), 10)

    q._head = q.default_capacity - 5
    q._tail = 5
    assert_equal(len(q), 10)


fn test_impl_bool() raises:
    q = Deque[Int]()
    assert_false(q)

    q._tail = 1
    assert_true(q)


fn test_impl_append() raises:
    q = Deque[Int](capacity=2)

    q.append(0)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 1)
    assert_equal(q._capacity, 2)
    assert_equal((q._data + 0)[], 0)

    q.append(1)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 2)
    assert_equal(q._capacity, 4)
    assert_equal((q._data + 0)[], 0)
    assert_equal((q._data + 1)[], 1)

    q.append(2)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)
    assert_equal(q._capacity, 4)
    assert_equal((q._data + 0)[], 0)
    assert_equal((q._data + 1)[], 1)
    assert_equal((q._data + 2)[], 2)

    # simulate popleft()
    q._head += 1
    q.append(3)
    assert_equal(q._head, 1)
    # tail wrapped to the front
    assert_equal(q._tail, 0)
    assert_equal(q._capacity, 4)
    assert_equal((q._data + 1)[], 1)
    assert_equal((q._data + 2)[], 2)
    assert_equal((q._data + 3)[], 3)

    q.append(4)
    # re-allocated buffer and moved all elements
    assert_equal(q._head, 0)
    assert_equal(q._tail, 4)
    assert_equal(q._capacity, 8)
    assert_equal((q._data + 0)[], 1)
    assert_equal((q._data + 1)[], 2)
    assert_equal((q._data + 2)[], 3)
    assert_equal((q._data + 3)[], 4)


fn test_impl_append_with_maxlen() raises:
    q = Deque[Int](maxlen=3)

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
    assert_equal((q._data + 1)[], 1)
    assert_equal((q._data + 2)[], 2)
    assert_equal((q._data + 3)[], 3)


fn test_impl_appendleft() raises:
    q = Deque[Int](capacity=2)

    q.appendleft(0)
    # head wrapped to the end of the buffer
    assert_equal(q._head, 1)
    assert_equal(q._tail, 0)
    assert_equal(q._capacity, 2)
    assert_equal((q._data + 1)[], 0)

    q.appendleft(1)
    # re-allocated buffer and moved all elements
    assert_equal(q._head, 0)
    assert_equal(q._tail, 2)
    assert_equal(q._capacity, 4)
    assert_equal((q._data + 0)[], 1)
    assert_equal((q._data + 1)[], 0)

    q.appendleft(2)
    # head wrapped to the end of the buffer
    assert_equal(q._head, 3)
    assert_equal(q._tail, 2)
    assert_equal(q._capacity, 4)
    assert_equal((q._data + 3)[], 2)
    assert_equal((q._data + 0)[], 1)
    assert_equal((q._data + 1)[], 0)

    # simulate pop()
    q._tail -= 1
    q.appendleft(3)
    assert_equal(q._head, 2)
    assert_equal(q._tail, 1)
    assert_equal(q._capacity, 4)
    assert_equal((q._data + 2)[], 3)
    assert_equal((q._data + 3)[], 2)
    assert_equal((q._data + 0)[], 1)

    q.appendleft(4)
    # re-allocated buffer and moved all elements
    assert_equal(q._head, 0)
    assert_equal(q._tail, 4)
    assert_equal(q._capacity, 8)
    assert_equal((q._data + 0)[], 4)
    assert_equal((q._data + 1)[], 3)
    assert_equal((q._data + 2)[], 2)
    assert_equal((q._data + 3)[], 1)


fn test_impl_appendleft_with_maxlen() raises:
    q = Deque[Int](maxlen=3)

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
    assert_equal((q._data + 0)[], 3)
    assert_equal((q._data + 1)[], 2)
    assert_equal((q._data + 2)[], 1)


fn test_impl_extend() raises:
    q = Deque[Int](maxlen=4)
    lst = List[Int](0, 1, 2)

    q.extend(lst)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 3)
    assert_equal(q._capacity, 8)
    assert_equal((q._data + 0)[], 0)
    assert_equal((q._data + 1)[], 1)
    assert_equal((q._data + 2)[], 2)

    q.extend(lst)
    # has to popleft the first 2 elements
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 2)
    assert_equal(q._tail, 6)
    assert_equal((q._data + 2)[], 2)
    assert_equal((q._data + 3)[], 0)
    assert_equal((q._data + 4)[], 1)
    assert_equal((q._data + 5)[], 2)

    # turn off `maxlen` restriction
    q._maxlen = -1
    q.extend(lst)
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 2)
    assert_equal(q._tail, 1)
    assert_equal((q._data + 2)[], 2)
    assert_equal((q._data + 3)[], 0)
    assert_equal((q._data + 4)[], 1)
    assert_equal((q._data + 5)[], 2)
    assert_equal((q._data + 6)[], 0)
    assert_equal((q._data + 7)[], 1)
    assert_equal((q._data + 0)[], 2)

    # turn on `maxlen` and force to re-allocate
    q._maxlen = 8
    q.extend(lst)
    assert_equal(q._capacity, 16)
    assert_equal(q._head, 0)
    assert_equal(q._tail, 8)
    # has to popleft the first 2 elements
    assert_equal((q._data + 0)[], 1)
    assert_equal((q._data + 1)[], 2)
    assert_equal((q._data + 6)[], 1)
    assert_equal((q._data + 7)[], 2)

    # extend with the list that is longer than `maxlen`
    # has to pop all deque elements and some initial
    # elements from the list as well
    lst = List(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    q.extend(lst)
    assert_equal(q._capacity, 16)
    assert_equal(q._head, 8)
    assert_equal(q._tail, 0)
    assert_equal((q._data + 8)[], 2)
    assert_equal((q._data + 9)[], 3)
    assert_equal((q._data + 14)[], 8)
    assert_equal((q._data + 15)[], 9)


fn test_impl_extendleft() raises:
    q = Deque[Int](maxlen=4)
    lst = List[Int](0, 1, 2)

    q.extendleft(lst)
    # head wrapped to the end of the buffer
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 5)
    assert_equal(q._tail, 0)
    assert_equal((q._data + 5)[], 2)
    assert_equal((q._data + 6)[], 1)
    assert_equal((q._data + 7)[], 0)

    q.extendleft(lst)
    # popped the last 2 elements
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 2)
    assert_equal(q._tail, 6)
    assert_equal((q._data + 2)[], 2)
    assert_equal((q._data + 3)[], 1)
    assert_equal((q._data + 4)[], 0)
    assert_equal((q._data + 5)[], 2)

    # turn off `maxlen` restriction
    q._maxlen = -1
    q.extendleft(lst)
    assert_equal(q._capacity, 8)
    assert_equal(q._head, 7)
    assert_equal(q._tail, 6)
    assert_equal((q._data + 7)[], 2)
    assert_equal((q._data + 0)[], 1)
    assert_equal((q._data + 1)[], 0)
    assert_equal((q._data + 2)[], 2)
    assert_equal((q._data + 3)[], 1)
    assert_equal((q._data + 4)[], 0)
    assert_equal((q._data + 5)[], 2)

    # turn on `maxlen` and force to re-allocate
    q._maxlen = 8
    q.extendleft(lst)
    assert_equal(q._capacity, 16)
    assert_equal(q._head, 13)
    assert_equal(q._tail, 5)
    # has to popleft the last 2 elements
    assert_equal((q._data + 13)[], 2)
    assert_equal((q._data + 14)[], 1)
    assert_equal((q._data + 3)[], 2)
    assert_equal((q._data + 4)[], 1)

    # extend with the list that is longer than `maxlen`
    # has to pop all deque elements and some initial
    # elements from the list as well
    lst = List(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    q.extendleft(lst)
    assert_equal(q._capacity, 16)
    assert_equal(q._head, 5)
    assert_equal(q._tail, 13)
    assert_equal((q._data + 5)[], 9)
    assert_equal((q._data + 6)[], 8)
    assert_equal((q._data + 11)[], 3)
    assert_equal((q._data + 12)[], 2)


fn test_impl_insert() raises:
    q = Deque[Int](0, 1, 2, 3, 4, 5)

    q.insert(0, 6)
    assert_equal(q._head, q.default_capacity - 1)
    assert_equal((q._data + q._head)[], 6)
    assert_equal((q._data + 0)[], 0)

    q.insert(1, 7)
    assert_equal(q._head, q.default_capacity - 2)
    assert_equal((q._data + q._head + 0)[], 6)
    assert_equal((q._data + q._head + 1)[], 7)

    q.insert(8, 8)
    assert_equal(q._tail, 7)
    assert_equal((q._data + q._tail - 1)[], 8)
    assert_equal((q._data + q._tail - 2)[], 5)

    q.insert(8, 9)
    assert_equal(q._tail, 8)
    assert_equal((q._data + q._tail - 1)[], 8)
    assert_equal((q._data + q._tail - 2)[], 9)


fn test_impl_pop() raises:
    q = Deque[Int](capacity=2, min_capacity=2)
    with assert_raises():
        _ = q.pop()

    q.append(1)
    q.appendleft(2)
    assert_equal(q._capacity, 4)
    assert_equal(q.pop(), 1)
    assert_equal(len(q), 1)
    assert_equal(q[0], 2)
    assert_equal(q._capacity, 2)


fn test_popleft() raises:
    q = Deque[Int](capacity=2, min_capacity=2)
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


fn test_impl_clear() raises:
    q = Deque[Int](capacity=2)
    q.append(1)
    assert_equal(q._tail, 1)

    q.clear()
    assert_equal(q._head, 0)
    assert_equal(q._tail, 0)
    assert_equal(q._capacity, q._min_capacity)


fn test_impl_add() raises:
    l1 = List(1, 2, 3, 4, 5, 6, 7, 8)
    l2 = List(9, 10, 11, 12, 13, 14, 15, 16)
    q1 = Deque(elements=l1, capacity=20, maxlen=30)
    q2 = Deque(elements=l2, min_capacity=200, shrink=False)

    assert_equal(q1._capacity, 32)
    assert_equal(q1._maxlen, 30)
    assert_equal(q2._capacity, 64)
    assert_equal(q2._min_capacity, 256)

    q3 = q1 + q2
    # has to inherit q1 properties
    assert_equal(q3._capacity, 32)
    assert_equal(q3._min_capacity, 64)
    assert_equal(q3._maxlen, 30)
    assert_equal(q3._shrink, True)
    assert_equal(q3._head, 0)
    assert_equal(q3._tail, 16)
    for i in range(len(q3)):
        assert_equal((q3._data + i)[], 1 + i)

    q4 = q2 + q1
    # has to inherit q2 properties
    assert_equal(q4._capacity, 64)
    assert_equal(q4._min_capacity, 256)
    assert_equal(q4._maxlen, -1)
    assert_equal(q4._shrink, False)
    assert_equal(q4._head, 0)
    assert_equal(q4._tail, 16)
    mid_len = len(q4) // 2
    for i in range(mid_len):
        assert_equal((q4._data + i)[], 9 + i)
    for i in range(mid_len, len(q4)):
        assert_equal((q4._data + i)[], i - 7)

    q5 = q3 + q4
    # has to inherit q3 properties
    assert_equal(q5._capacity, 32)
    assert_equal(q5._min_capacity, 64)
    assert_equal(q5._maxlen, 30)
    assert_equal(q5._shrink, True)
    # has to obey to maxlen
    assert_equal(len(q5), 30)
    assert_equal(q5._head, 2)
    assert_equal(q5._tail, 0)
    assert_equal((q5._data + 2)[], 3)
    assert_equal((q5._data + 31)[], 8)

    q6 = q4 + q3
    # has to inherit q4 properties
    assert_equal(q6._capacity, 64)
    assert_equal(q6._min_capacity, 256)
    assert_equal(q6._maxlen, -1)
    assert_equal(q6._shrink, False)
    # has to obey to maxlen
    assert_equal(len(q6), 32)
    assert_equal(q6._head, 0)
    assert_equal(q6._tail, 32)
    assert_equal((q6._data + 0)[], 9)
    assert_equal((q6._data + 31)[], 16)


fn test_impl_iadd() raises:
    l1 = List(1, 2, 3, 4, 5, 6, 7, 8)
    l2 = List(9, 10, 11, 12, 13, 14, 15, 16)
    q1 = Deque(elements=l1, maxlen=10)
    q2 = Deque(elements=l2, min_capacity=200, shrink=False)

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
    for i in range(len(q1)):
        assert_equal(q1[i], 7 + i)

    q2 += q1
    # has to keep q2 properties
    assert_equal(q2._capacity, 64)
    assert_equal(q2._min_capacity, 256)
    assert_equal(q2._maxlen, -1)
    assert_equal(q2._shrink, False)
    assert_equal(len(q2), 18)
    assert_equal(q2._head, 0)
    assert_equal(q2._tail, 18)
    assert_equal((q2._data + 0)[], 9)
    assert_equal((q2._data + 17)[], 16)


fn test_impl_mul() raises:
    l = List(1, 2, 3)
    q = Deque(elements=l, capacity=3, min_capacity=2, maxlen=7, shrink=False)

    q1 = q * 0
    assert_equal(q1._head, 0)
    assert_equal(q1._tail, 0)
    assert_equal(q1._capacity, q._min_capacity)
    assert_equal(q1._min_capacity, q._min_capacity)
    assert_equal(q1._maxlen, q._maxlen)
    assert_equal(q1._shrink, q._shrink)

    q2 = q * 1
    assert_equal(q2._head, 0)
    assert_equal(q2._tail, len(q))
    assert_equal(q2._capacity, q._capacity)
    assert_equal(q2._min_capacity, q._min_capacity)
    assert_equal(q2._maxlen, q._maxlen)
    assert_equal(q2._shrink, q._shrink)
    assert_equal((q2._data + 0)[], (q._data + 0)[])
    assert_equal((q2._data + 1)[], (q._data + 1)[])
    assert_equal((q2._data + 2)[], (q._data + 2)[])

    q3 = q * 2
    assert_equal(q3._head, 0)
    assert_equal(q3._tail, 2 * len(q))
    assert_equal(q3._min_capacity, q._min_capacity)
    assert_equal(q3._maxlen, q._maxlen)
    assert_equal(q3._shrink, q._shrink)
    assert_equal((q3._data + 0)[], (q._data + 0)[])
    assert_equal((q3._data + 5)[], (q._data + 2)[])

    q4 = q * 3
    # should obey maxlen
    assert_equal(q4._head, 2)
    assert_equal(q4._tail, 1)
    assert_equal(q4._capacity, 8)
    assert_equal(q4._min_capacity, q._min_capacity)
    assert_equal(q4._maxlen, q._maxlen)
    assert_equal(q4._shrink, q._shrink)
    assert_equal((q4._data + 2)[], 3)
    assert_equal((q4._data + 0)[], 3)


fn test_impl_imul() raises:
    l = List(1, 2, 3)

    q = Deque(elements=l, capacity=3, min_capacity=2, maxlen=7, shrink=False)
    q *= 0
    assert_equal(q._head, 0)
    assert_equal(q._tail, 0)
    # resets capacity to min_capacity
    assert_equal(q._capacity, 2)
    assert_equal(q._min_capacity, 2)
    assert_equal(q._maxlen, 7)
    assert_equal(q._shrink, False)

    q = Deque(elements=l, capacity=3, min_capacity=2, maxlen=7, shrink=False)
    q *= 1
    assert_equal(q._head, 0)
    assert_equal(q._tail, len(q))
    assert_equal(q._capacity, 4)
    assert_equal(q._min_capacity, 2)
    assert_equal(q._maxlen, 7)
    assert_equal(q._shrink, False)
    assert_equal((q._data + 0)[], 1)
    assert_equal((q._data + 1)[], 2)
    assert_equal((q._data + 2)[], 3)

    q = Deque(elements=l, capacity=3, min_capacity=2, maxlen=7, shrink=False)
    q *= 2
    assert_equal(q._head, 0)
    assert_equal(q._tail, 6)
    assert_equal(q._capacity, 8)
    assert_equal(q._min_capacity, 2)
    assert_equal(q._maxlen, 7)
    assert_equal(q._shrink, False)
    assert_equal((q._data + 0)[], 1)
    assert_equal((q._data + 5)[], 3)

    q = Deque(elements=l, capacity=3, min_capacity=2, maxlen=7, shrink=False)
    q *= 3
    # should obey maxlen
    assert_equal(q._head, 2)
    assert_equal(q._tail, 1)
    assert_equal(q._capacity, 8)
    assert_equal(q._min_capacity, 2)
    assert_equal(q._maxlen, 7)
    assert_equal(q._shrink, False)
    assert_equal((q._data + 2)[], 3)
    assert_equal((q._data + 0)[], 3)


# ===-----------------------------------------------------------------------===#
# API Interface tests
# ===-----------------------------------------------------------------------===#


fn test_init_variadic_list() raises:
    lst1 = List(0, 1)
    lst2 = List(2, 3)

    q = Deque(lst1, lst2)
    assert_equal(q[0], lst1)
    assert_equal(q[1], lst2)

    lst1[0] = 4
    assert_equal(q[0], List(0, 1))

    p = Deque(lst1^, lst2^)
    assert_equal(p[0], List(4, 1))
    assert_equal(p[1], List(2, 3))


fn test_copy_trivial() raises:
    q = Deque(1, 2, 3)

    p = Deque(q)
    assert_equal(p[0], q[0])

    p[0] = 3
    assert_equal(p[0], 3)
    assert_equal(q[0], 1)


fn test_copy_list() raises:
    q = Deque[List[Int]]()
    lst1 = List(1, 2, 3)
    lst2 = List(4, 5, 6)
    q.append(lst1)
    q.append(lst2)
    assert_equal(q[0], lst1)

    lst1[0] = 7
    assert_equal(q[0], List(1, 2, 3))

    p = Deque(q)
    assert_equal(p[0], q[0])

    p[0][0] = 7
    assert_equal(p[0], List(7, 2, 3))
    assert_equal(q[0], List(1, 2, 3))


fn test_move_list() raises:
    q = Deque[List[Int]]()
    lst1 = List(1, 2, 3)
    lst2 = List(4, 5, 6)
    q.append(lst1)
    q.append(lst2)
    assert_equal(q[0], lst1)

    p = q^
    assert_equal(p[0], lst1)

    lst1[0] = 7
    assert_equal(lst1[0], 7)
    assert_equal(p[0], List(1, 2, 3))


fn test_getitem() raises:
    q = Deque(1, 2)
    assert_equal(q[0], 1)
    assert_equal(q[1], 2)
    assert_equal(q[-1], 2)
    assert_equal(q[-2], 1)


fn test_setitem() raises:
    q = Deque(1, 2)
    assert_equal(q[0], 1)

    q[0] = 3
    assert_equal(q[0], 3)

    q[-1] = 4
    assert_equal(q[1], 4)


fn test_eq() raises:
    q = Deque[Int](1, 2, 3)
    p = Deque[Int](1, 2, 3)

    assert_true(q == p)

    r = Deque[Int](0, 1, 2, 3)
    q.appendleft(0)
    assert_true(q == r)


fn test_ne() raises:
    q = Deque[Int](1, 2, 3)
    p = Deque[Int](3, 2, 1)

    assert_true(q != p)

    q.appendleft(0)
    p.append(0)
    assert_true(q != p)


fn test_count() raises:
    q = Deque(1, 2, 1, 2, 3, 1)

    assert_equal(q.count(1), 3)
    assert_equal(q.count(2), 2)
    assert_equal(q.count(3), 1)
    assert_equal(q.count(4), 0)

    q.appendleft(2)
    assert_equal(q.count(2), 3)


fn test_contains() raises:
    q = Deque[Int](1, 2, 3)

    assert_true(1 in q)
    assert_false(4 in q)


fn test_index() raises:
    q = Deque(1, 2, 1, 2, 3, 1)

    assert_equal(q.index(2), 1)
    assert_equal(q.index(2, 1), 1)
    assert_equal(q.index(2, 1, 3), 1)
    assert_equal(q.index(2, stop=4), 1)
    assert_equal(q.index(1, -12, 10), 0)
    assert_equal(q.index(1, -4), 2)
    assert_equal(q.index(1, -3), 5)
    with assert_raises():
        _ = q.index(4)


fn test_insert() raises:
    q = Deque[Int](capacity=4, maxlen=7)

    # negative index outbound
    q.insert(-10, 0)
    # Deque(0)
    assert_equal(q[0], 0)
    assert_equal(len(q), 1)

    # zero index
    q.insert(0, 1)
    # Deque(1, 0)
    assert_equal(q[0], 1)
    assert_equal(q[1], 0)
    assert_equal(len(q), 2)

    # # positive index eq length
    q.insert(2, 2)
    # Deque(1, 0, 2)
    assert_equal(q[2], 2)
    assert_equal(q[1], 0)

    # # positive index outbound
    q.insert(10, 3)
    # Deque(1, 0, 2, 3)
    assert_equal(q[3], 3)
    assert_equal(q[2], 2)

    # assert deque buffer reallocated
    assert_equal(len(q), 4)
    assert_equal(q._capacity, 8)

    # # positive index inbound
    q.insert(1, 4)
    # Deque(1, 4, 0, 2, 3)
    assert_equal(q[1], 4)
    assert_equal(q[0], 1)
    assert_equal(q[2], 0)

    # # positive index inbound
    q.insert(3, 5)
    # Deque(1, 4, 0, 5, 2, 3)
    assert_equal(q[3], 5)
    assert_equal(q[2], 0)
    assert_equal(q[4], 2)

    # # negative index inbound
    q.insert(-3, 6)
    # Deque(1, 4, 0, 6, 5, 2, 3)
    assert_equal(q[3], 6)
    assert_equal(q[2], 0)
    assert_equal(q[4], 5)

    # deque is at its maxlen
    assert_equal(len(q), 7)
    with assert_raises():
        q.insert(3, 7)


fn test_remove() raises:
    q = Deque[Int](min_capacity=32)
    q.extend(List(0, 1, 0, 2, 3, 0, 4, 5))
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


fn test_peek_and_peekleft() raises:
    q = Deque[Int](capacity=4)
    assert_equal(q._capacity, 4)

    with assert_raises():
        _ = q.peek()
    with assert_raises():
        _ = q.peekleft()

    q.extend(List(1, 2, 3))
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


fn test_reverse() raises:
    q = Deque(0, 1, 2, 3)

    q.reverse()
    assert_equal(q[0], 3)
    assert_equal(q[1], 2)
    assert_equal(q[2], 1)
    assert_equal(q[3], 0)

    q.appendleft(4)
    q.reverse()
    assert_equal(q[0], 0)
    assert_equal(q[4], 4)


fn test_rotate() raises:
    q = Deque(0, 1, 2, 3)

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


fn test_iter() raises:
    q = Deque(1, 2, 3)

    i = 0
    for e in q:
        assert_equal(e[], q[i])
        i += 1
    assert_equal(i, len(q))

    for e in q:
        if e[] == 1:
            e[] = 4
            assert_equal(e[], 4)
    assert_equal(q[0], 4)


fn test_iter_with_list() raises:
    q = Deque[List[Int]]()
    lst1 = List(1, 2, 3)
    lst2 = List(4, 5, 6)
    q.append(lst1)
    q.append(lst2)
    assert_equal(len(q), 2)

    i = 0
    for e in q:
        assert_equal(e[], q[i])
        i += 1
    assert_equal(i, len(q))

    for e in q:
        if e[] == lst1:
            e[][0] = 7
            assert_equal(e[], List(7, 2, 3))
    assert_equal(q[0], List(7, 2, 3))

    for e in q:
        if e[] == lst2:
            e[] = List(1, 2, 3)
            assert_equal(e[], List(1, 2, 3))
    assert_equal(q[1], List(1, 2, 3))


fn test_reversed_iter() raises:
    q = Deque(1, 2, 3)

    i = 0
    for e in reversed(q):
        i -= 1
        assert_equal(e[], q[i])
    assert_equal(-i, len(q))


fn test_str_and_repr() raises:
    q = Deque(1, 2, 3)

    assert_equal(q.__str__(), "Deque(1, 2, 3)")
    assert_equal(q.__repr__(), "Deque(1, 2, 3)")

    s = Deque("a", "b", "c")

    assert_equal(s.__str__(), "Deque('a', 'b', 'c')")
    assert_equal(s.__repr__(), "Deque('a', 'b', 'c')")


# ===-------------------------------------------------------------------===#
# main
# ===-------------------------------------------------------------------===#


def main():
    test_impl_init_default()
    test_impl_init_capacity()
    test_impl_init_min_capacity()
    test_impl_init_maxlen()
    test_impl_init_shrink()
    test_impl_init_list()
    test_impl_init_list_args()
    test_impl_init_variadic()
    test_impl_len()
    test_impl_bool()
    test_impl_append()
    test_impl_append_with_maxlen()
    test_impl_appendleft()
    test_impl_appendleft_with_maxlen()
    test_impl_extend()
    test_impl_extendleft()
    test_impl_insert()
    test_impl_pop()
    test_popleft()
    test_impl_clear()
    test_impl_add()
    test_impl_iadd()
    test_impl_mul()
    test_impl_imul()
    test_init_variadic_list()
    test_copy_trivial()
    test_copy_list()
    test_move_list()
    test_getitem()
    test_setitem()
    test_eq()
    test_ne()
    test_count()
    test_contains()
    test_index()
    test_insert()
    test_remove()
    test_peek_and_peekleft()
    test_reverse()
    test_rotate()
    test_iter()
    test_iter_with_list()
    test_reversed_iter()
    test_str_and_repr()
