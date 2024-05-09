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

# RUN: %mojo %s | FileCheck %s
from collections import List
from random import random_si64, seed

from builtin.sort import _quicksort, _small_sort


# CHECK-LABEL: test_sort_small_3
fn test_sort_small_3():
    print("== test_sort_small_3")
    alias length = 3

    var list = List[Int]()

    list.append(9)
    list.append(1)
    list.append(2)

    @parameter
    fn _less_than_equal[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    var ptr = rebind[Pointer[Int]](list.data)
    _small_sort[length, Int, _less_than_equal](ptr)

    # CHECK: 1
    # CHECK: 2
    # CHECK: 9
    for i in range(length):
        print(list[i])


# CHECK-LABEL: test_sort_small_5
fn test_sort_small_5():
    print("== test_sort_small_5")
    alias length = 5

    var list = List[Int]()

    list.append(9)
    list.append(1)
    list.append(2)
    list.append(3)
    list.append(4)

    @parameter
    fn _less_than_equal[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    var ptr = rebind[Pointer[Int]](list.data)
    _small_sort[length, Int, _less_than_equal](ptr)

    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    # CHECK: 9
    for i in range(length):
        print(list[i])


# CHECK-LABEL: test_sort0
fn test_sort0():
    print("== test_sort0")

    var list = List[Int]()

    sort(list)


# CHECK-LABEL: test_sort2
fn test_sort2():
    print("== test_sort2")

    alias length = 2
    var list = List[Int]()

    list.append(-1)
    list.append(0)

    sort(list)

    # CHECK: -1
    # CHECK: 0
    for i in range(length):
        print(list[i])

    list[0] = 2
    list[1] = -2

    sort(list)

    # CHECK: -2
    # CHECK: 2
    for i in range(length):
        print(list[i])


# CHECK-LABEL: test_sort3
fn test_sort3():
    print("== test_sort3")

    alias length = 3
    var list = List[Int]()

    list.append(-1)
    list.append(0)
    list.append(1)

    sort(list)

    # CHECK: -1
    # CHECK: 0
    # CHECK: 1
    for i in range(length):
        print(list[i])

    list[0] = 2
    list[1] = -2
    list[2] = 0

    sort(list)

    # CHECK: -2
    # CHECK: 0
    # CHECK: 2
    for i in range(length):
        print(list[i])


# CHECK-LABEL test_sort3_dupe_elements
fn test_sort3_dupe_elements():
    print("== test_sort3_dupe_elements")

    alias length = 3

    fn test[
        cmp_fn: fn[type: AnyRegType] (type, type) capturing -> Bool,
    ]():
        var list = List[Int](capacity=3)
        list.append(5)
        list.append(3)
        list.append(3)

        var ptr = rebind[Pointer[Int]](list.data)
        _quicksort[Int, cmp_fn](ptr, len(list))

        # CHECK: 3
        # CHECK: 3
        # CHECK: 5
        for i in range(length):
            print(list[i])

    @parameter
    fn _lt[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) < rebind[Int](rhs)

    @parameter
    fn _leq[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    test[_lt]()
    test[_leq]()


# CHECK-LABEL: test_sort4
fn test_sort4():
    print("== test_sort4")

    alias length = 4
    var list = List[Int]()

    list.append(-1)
    list.append(0)
    list.append(1)
    list.append(2)

    sort(list)

    # CHECK: -1
    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    for i in range(length):
        print(list[i])

    list[0] = 2
    list[1] = -2
    list[2] = 0
    list[3] = -4

    sort(list)

    # CHECK: -4
    # CHECK: -2
    # CHECK: 0
    # CHECK: 2
    for i in range(length):
        print(list[i])


# CHECK-LABEL: test_sort5
fn test_sort5():
    print("== test_sort5")

    alias length = 5
    var list = List[Int]()

    for i in range(5):
        list.append(i)

    sort(list)

    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    for i in range(length):
        print(list[i])

    list[0] = 2
    list[1] = -2
    list[2] = 0
    list[3] = -4
    list[4] = 1

    sort(list)

    # CHECK: -4
    # CHECK: -2
    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    for i in range(length):
        print(list[i])


# CHECK-LABEL: test_sort_reverse
fn test_sort_reverse():
    print("== test_sort_reverse")

    alias length = 5
    var list = List[Int](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort(list)

    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    for i in range(length):
        print(list[i])


# CHECK-LABEL: test_sort_semi_random
fn test_sort_semi_random():
    print("== test_sort_semi_random")

    alias length = 8
    var list = List[Int](capacity=length)

    for i in range(length):
        if i % 2:
            list.append(-i)
        else:
            list.append(i)

    sort(list)

    # CHECK: 7
    # CHECK: 5
    # CHECK: 3
    # CHECK: 1
    # CHECK: 0
    # CHECK: 2
    # CHECK: 4
    # CHECK: 6
    for i in range(length):
        print(list[i])


# CHECK-LABEL: test_sort9
fn test_sort9():
    print("== test_sort9")

    alias length = 9
    var list = List[Int](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort(list)

    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    # CHECK: 3
    # CHECK: 4
    # CHECK: 5
    # CHECK: 6
    # CHECK: 7
    # CHECK: 8
    for i in range(length):
        print(list[i])


# CHECK-LABEL: test_sort103
fn test_sort103():
    print("== test_sort103")

    alias length = 103
    var list = List[Int](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort(list)

    # CHECK-NOT: unsorted
    for i in range(1, length):
        if list[i - 1] > list[i]:
            print("error: unsorted")


# CHECK-LABEL: test_sort_any_103
fn test_sort_any_103():
    print("== test_sort_any_103")

    alias length = 103
    var list = List[Float32](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort[DType.float32](list)

    # CHECK-NOT: unsorted
    for i in range(1, length):
        if list[i - 1] > list[i]:
            print("error: unsorted")


fn test_quick_sort_repeated_val():
    print("==  test_quick_sort_repeated_val")

    alias length = 36
    var list = List[Float32](capacity=length)

    for i in range(0, length // 4):
        list.append(i + 1)
        list.append(i + 1)
        list.append(i + 1)
        list.append(i + 1)

    @parameter
    fn _greater_than[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) > rebind[Float32](rhs)

    var ptr = rebind[Pointer[Float32]](list.data)
    _quicksort[Float32, _greater_than](ptr, len(list))

    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    for i in range(0, length):
        print(list[i])

    @parameter
    fn _less_than[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) < rebind[Float32](rhs)

    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 1.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 3.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 5.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 6.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 7.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 8.0
    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 9.0
    # CHECK: 9.0
    var sptr = rebind[Pointer[Float32]](list.data)
    _quicksort[Float32, _less_than](sptr, len(list))
    for i in range(0, length):
        print(list[i])


fn test_partition_top_k(length: Int, k: Int):
    print("== test_partition_top_k_", end="")
    print(length, end="")
    print("_", end="")
    print(k, end="")
    print("")

    var list = List[Float32](capacity=length)

    for i in range(0, length):
        list.append(i)

    @parameter
    fn _great_than_equal[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) >= rebind[Float32](rhs)

    var ptr = rebind[Pointer[Float32]](list.data)
    _ = partition[Float32, _great_than_equal](ptr, k, len(list))

    for i in range(0, k):
        if list[i] < length - k:
            print("error: incorrect top-k element", list[i])


# CHECK-LABEL: test_sort_stress
fn test_sort_stress():
    print("== test_sort_stress")
    var lens = VariadicList[Int](3, 100, 117, 223, 500, 1000, 1500, 2000, 3000)
    var random_seed = 0
    seed(random_seed)

    @__copy_capture(random_seed)
    @parameter
    fn test[
        cmp_fn: fn[type: AnyRegType] (type, type) capturing -> Bool,
        check_fn: fn[type: AnyRegType] (type, type) capturing -> Bool,
    ](length: Int):
        var list = List[Int](capacity=length)
        for i in range(length):
            list.append(int(random_si64(-length, length)))

        var ptr = rebind[Pointer[Int]](list.data)
        _quicksort[Int, cmp_fn](ptr, len(list))

        # CHECK-NOT: error
        for i in range(length - 1):
            if not check_fn[Int](list[i], list[i + 1]):
                print("error: unsorted, seed is", random_seed)
                return

    @parameter
    @always_inline
    fn _gt[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) > rebind[Int](rhs)

    @parameter
    @always_inline
    fn _geq[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) >= rebind[Int](rhs)

    @parameter
    @always_inline
    fn _lt[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) < rebind[Int](rhs)

    @parameter
    @always_inline
    fn _leq[type: AnyRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    for i in range(len(lens)):
        var length = lens[i]
        test[_gt, _geq](length)
        test[_geq, _geq](length)
        test[_lt, _leq](length)
        test[_leq, _leq](length)


@value
struct MyStruct:
    var val: Int


# CHECK-LABEL: test_sort_custom
fn test_sort_custom():
    print("== test_sort_custom")

    alias length = 103
    var list = List[MyStruct](capacity=length)

    for i in range(length):
        list.append(MyStruct(length - i - 1))

    @parameter
    fn compare_fn(lhs: MyStruct, rhs: MyStruct) -> Bool:
        return lhs.val <= rhs.val

    sort[MyStruct, compare_fn](list)

    # CHECK-NOT: unsorted
    for i in range(1, length):
        if list[i - 1].val > list[i].val:
            print("error: unsorted")


fn main():
    test_sort_small_3()
    test_sort_small_5()
    test_sort0()
    test_sort2()
    test_sort3()
    test_sort3_dupe_elements()
    test_sort4()
    test_sort5()
    test_sort_reverse()
    test_sort_semi_random()
    test_sort9()
    test_sort103()
    test_sort_any_103()
    test_quick_sort_repeated_val()

    test_sort_stress()

    test_sort_custom()

    # CHECK-LABEL: test_partition_top_k_7_5
    # CHECK-NOT: incorrect top-k
    test_partition_top_k(7, 5)
    # CHECK-LABEL: test_partition_top_k_11_2
    # CHECK-NOT: incorrect top-k
    test_partition_top_k(11, 2)
    # CHECK-LABEL: test_partition_top_k_4_1
    # CHECK-NOT: incorrect top-k
    test_partition_top_k(4, 1)
