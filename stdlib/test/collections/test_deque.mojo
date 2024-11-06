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

from testing import assert_equal, assert_false, assert_true, assert_raises

from collections import Deque

# ===----------------------------------------------------------------------===#
# Implementation tests
# ===----------------------------------------------------------------------===#


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


# ===----------------------------------------------------------------------===#
# API Interface tests
# ===----------------------------------------------------------------------===#


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
    test_impl_extend()
    test_init_variadic_list()
    test_copy_trivial()
    test_copy_list()
    test_move_list()
    test_getitem()
