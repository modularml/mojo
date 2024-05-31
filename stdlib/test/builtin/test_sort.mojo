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
# RUN: %mojo -D CURRENT_DIR=%S %s

from pathlib import Path
from sys import os_is_windows, env_get_string

alias CURRENT_DIR = env_get_string["CURRENT_DIR"]()
from testing import assert_true, assert_equal, assert_false
from random import random_si64, random_ui64, random_float64, seed

from builtin.sort import _quicksort, _small_sort


fn random_numbers[
    dtype: DType
](size: Int, max: Int = 3000) -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]](size)
    for _ in range(size):

        @parameter
        if (
            dtype == DType.int8
            or dtype == DType.int16
            or dtype == DType.int32
            or dtype == DType.int64
        ):
            result.append(random_si64(0, max).cast[dtype]())
        elif (
            dtype == DType.float16
            or dtype == DType.float32
            or dtype == DType.float64
        ):
            result.append(random_float64(0, max).cast[dtype]())
        else:
            result.append(random_ui64(0, max).cast[dtype]())
    return result


# fn assert_sorted[dtype: DType](inout list: List[Scalar[dtype]]) raises:
#     sort[dtype](list)
#     for i in range(1, len(list)):
#         assert_true(
#             list[i] >= list[i - 1], str(list[i - 1]) + " > " + str(list[i])
#         )


fn assert_sorted_string(inout list: List[String]) raises:
    for i in range(1, len(list)):
        assert_true(
            list[i] >= list[i - 1], str(list[i - 1]) + " > " + str(list[i])
        )


fn assert_sorted[
    type: ComparableCollectionElement
](inout list: List[type]) raises:
    for i in range(1, len(list)):
        assert_true(list[i] >= list[i - 1], "error at index: " + str(i))


fn test_sort_small_3() raises:
    alias length = 3

    var list = List[Int]()

    list.append(9)
    list.append(1)
    list.append(2)

    @parameter
    fn _less_than_equal[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    var ptr = rebind[Pointer[Int]](list.data)
    _small_sort[length, Int, _less_than_equal](ptr)

    var expected = List[Int](1, 2, 9)
    for i in range(length):
        assert_equal(expected[i], list[i])


fn test_sort_small_5() raises:
    alias length = 5

    var list = List[Int]()

    list.append(9)
    list.append(1)
    list.append(2)
    list.append(3)
    list.append(4)

    @parameter
    fn _less_than_equal[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    var ptr = rebind[Pointer[Int]](list.data)
    _small_sort[length, Int, _less_than_equal](ptr)

    var expected = List[Int](1, 2, 3, 4, 9)
    for i in range(length):
        assert_equal(expected[i], list[i])


fn test_sort0():
    var list = List[Int]()

    sort(list)


fn test_sort2() raises:
    alias length = 2
    var list = List[Int]()

    list.append(-1)
    list.append(0)

    sort(list)

    var expected = List[Int](-1, 0)
    for i in range(length):
        assert_equal(expected[i], list[i])

    list[0] = 2
    list[1] = -2

    sort(list)

    expected = List[Int](-2, 2)
    for i in range(length):
        assert_equal(expected[i], list[i])


fn test_sort3() raises:
    alias length = 3
    var list = List[Int]()

    list.append(-1)
    list.append(0)
    list.append(1)

    sort(list)

    var expected = List[Int](-1, 0, 1)
    for i in range(length):
        assert_equal(expected[i], list[i])

    list[0] = 2
    list[1] = -2
    list[2] = 0

    sort(list)

    expected = List[Int](-2, 0, 2)
    for i in range(length):
        assert_equal(expected[i], list[i])


fn test_sort3_dupe_elements() raises:
    alias length = 3

    fn test[
        cmp_fn: fn[type: AnyTrivialRegType] (type, type) capturing -> Bool,
    ]() raises:
        var list = List[Int](capacity=3)
        list.append(5)
        list.append(3)
        list.append(3)

        var ptr = rebind[Pointer[Int]](list.data)
        _quicksort[Int, cmp_fn](ptr, len(list))

        var expected = List[Int](3, 3, 5)
        for i in range(length):
            assert_equal(expected[i], list[i])

    @parameter
    fn _lt[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) < rebind[Int](rhs)

    @parameter
    fn _leq[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) <= rebind[Int](rhs)

    test[_lt]()
    test[_leq]()


fn test_sort4() raises:
    alias length = 4
    var list = List[Int]()

    list.append(-1)
    list.append(0)
    list.append(1)
    list.append(2)

    sort(list)

    var expected = List[Int](-1, 0, 1, 2)
    for i in range(length):
        assert_equal(expected[i], list[i])

    list[0] = 2
    list[1] = -2
    list[2] = 0
    list[3] = -4

    sort(list)

    expected = List[Int](-4, -2, 0, 2)
    for i in range(length):
        assert_equal(expected[i], list[i])


fn test_sort5() raises:
    alias length = 5
    var list = List[Int]()

    for i in range(5):
        list.append(i)

    sort(list)

    var expected = List[Int](0, 1, 2, 3, 4)
    for i in range(length):
        assert_equal(expected[i], list[i])

    list[0] = 2
    list[1] = -2
    list[2] = 0
    list[3] = -4
    list[4] = 1

    sort(list)

    expected = List[Int](-4, -2, 0, 1, 2)
    for i in range(length):
        assert_equal(expected[i], list[i])


fn test_sort_reverse() raises:
    alias length = 5
    var list = List[Int](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort(list)

    var expected = List[Int](0, 1, 2, 3, 4)
    for i in range(length):
        assert_equal(expected[i], list[i])


fn test_sort_semi_random() raises:
    alias length = 8
    var list = List[Int](capacity=length)

    for i in range(length):
        if i % 2:
            list.append(-i)
        else:
            list.append(i)

    sort(list)

    var expected = List[Int](-7, -5, -3, -1, 0, 2, 4, 6)
    for i in range(length):
        assert_equal(expected[i], list[i])


fn test_sort9() raises:
    alias length = 9
    var list = List[Int](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort(list)

    var expected = List[Int](0, 1, 2, 3, 4, 5, 6, 7, 8)
    for i in range(length):
        assert_equal(expected[i], list[i])


fn test_sort103() raises:
    alias length = 103
    var list = List[Int](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort(list)

    for i in range(1, length):
        assert_false(list[i - 1] > list[i])


fn test_sort_any_103() raises:
    alias length = 103
    var list = List[Float32](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort[DType.float32](list)

    for i in range(1, length):
        assert_false(list[i - 1] > list[i])


fn test_quick_sort_repeated_val() raises:
    alias length = 36
    var list = List[Float32](capacity=length)

    for i in range(0, length // 4):
        list.append(i + 1)
        list.append(i + 1)
        list.append(i + 1)
        list.append(i + 1)

    @parameter
    fn _greater_than[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) > rebind[Float32](rhs)

    var ptr = rebind[Pointer[Float32]](list.data)
    _quicksort[Float32, _greater_than](ptr, len(list))

    var expected = List[Float32](
        9.0,
        9.0,
        9.0,
        9.0,
        8.0,
        8.0,
        8.0,
        8.0,
        7.0,
        7.0,
        7.0,
        7.0,
        6.0,
        6.0,
        6.0,
        6.0,
        5.0,
        5.0,
        5.0,
        5.0,
        4.0,
        4.0,
        4.0,
        4.0,
        3.0,
        3.0,
        3.0,
        3.0,
        2.0,
        2.0,
        2.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    for i in range(0, length):
        assert_equal(expected[i], list[i])

    @parameter
    fn _less_than[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) < rebind[Float32](rhs)

    expected = List[Float32](
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
        3.0,
        4.0,
        4.0,
        4.0,
        4.0,
        5.0,
        5.0,
        5.0,
        5.0,
        6.0,
        6.0,
        6.0,
        6.0,
        7.0,
        7.0,
        7.0,
        7.0,
        8.0,
        8.0,
        8.0,
        8.0,
        9.0,
        9.0,
        9.0,
        9.0,
    )
    var sptr = rebind[Pointer[Float32]](list.data)
    _quicksort[Float32, _less_than](sptr, len(list))
    for i in range(0, length):
        assert_equal(expected[i], list[i])


fn test_partition_top_k(length: Int, k: Int) raises:
    var list = List[Float32](capacity=length)

    for i in range(0, length):
        list.append(i)

    @parameter
    fn _great_than_equal[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Float32](lhs) >= rebind[Float32](rhs)

    var ptr = rebind[Pointer[Float32]](list.data)
    _ = partition[Float32, _great_than_equal](ptr, k, len(list))

    for i in range(0, k):
        if list[i] < length - k:
            assert_true(False)


fn test_sort_stress() raises:
    var lens = List[Int](3, 100, 117, 223, 500, 1000, 1500, 2000, 3000)
    var random_seed = 0
    seed(random_seed)

    @__copy_capture(random_seed)
    @parameter
    fn test[
        cmp_fn: fn[type: AnyTrivialRegType] (type, type) capturing -> Bool,
        check_fn: fn[type: AnyTrivialRegType] (type, type) capturing -> Bool,
    ](length: Int) raises:
        var list = List[Int](capacity=length)
        for _ in range(length):
            list.append(int(random_si64(-length, length)))

        var ptr = rebind[Pointer[Int]](list.data)
        _quicksort[Int, cmp_fn](ptr, len(list))

        for i in range(length - 1):
            assert_true(check_fn[Int](list[i], list[i + 1]))

    @parameter
    @always_inline
    fn _gt[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) > rebind[Int](rhs)

    @parameter
    @always_inline
    fn _geq[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) >= rebind[Int](rhs)

    @parameter
    @always_inline
    fn _lt[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
        return rebind[Int](lhs) < rebind[Int](rhs)

    @parameter
    @always_inline
    fn _leq[type: AnyTrivialRegType](lhs: type, rhs: type) -> Bool:
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


fn test_sort_custom() raises:
    alias length = 103
    var list = List[MyStruct](capacity=length)

    for i in range(length):
        list.append(MyStruct(length - i - 1))

    @parameter
    fn compare_fn(lhs: MyStruct, rhs: MyStruct) -> Bool:
        return lhs.val <= rhs.val

    sort[MyStruct, compare_fn](list)

    for i in range(1, length):
        assert_false(list[i - 1].val > list[i].val)


def test_sort_string_small_list():
    var list = random_numbers[DType.int32](10)
    var string_list = List[String]()
    for n in list:
        string_list.append(str(int(n[])))
    sort(string_list)
    assert_sorted_string(string_list)


def test_sort_string_big_list():
    var list = random_numbers[DType.int32](1000)
    var string_list = List[String]()
    for n in list:
        string_list.append(str(int(n[])))
    sort(string_list)
    assert_sorted_string(string_list)


def test_sort_strings():
    var text = (Path(CURRENT_DIR) / "test_file_dummy_input.txt").read_text()
    var strings = text.split(" ")
    sort(strings)
    assert_sorted_string(strings)


@value
struct Person(ComparableCollectionElement):
    var name: String
    var age: Int

    fn __lt__(self, other: Self) -> Bool:
        if self.age < other.age:
            return True
        if self.age == other.age:
            return self.name < other.name
        return False

    fn __le__(self, other: Self) -> Bool:
        return not (other < self)

    fn __gt__(self, other: Self) -> Bool:
        return other < self

    fn __ge__(self, other: Self) -> Bool:
        return not (self < other)

    fn __eq__(self, other: Self) -> Bool:
        return self.age == other.age and self.name == other.name

    fn __ne__(self, other: Self) -> Bool:
        return self.age != other.age or self.name != other.name


def test_sort_comparamble_elements_list():
    var list = List[Person]()

    @parameter
    fn gen_list(count: Int):
        list = List[Person]()
        var ages = random_numbers[DType.uint8](count)
        var names = List[String]("Maxim", "Max", "Alex", "Bob", "Joe")
        for age in ages:
            var name = names[int(age[]) % len(names)]
            list.append(Person(name, int(age[])))

    gen_list(10)
    sort(list)
    assert_sorted(list)

    gen_list(100)
    sort(list)
    assert_sorted(list)

    gen_list(1000)
    sort(list)
    assert_sorted(list)


fn test_sort_empty_comparamble_elements_list() raises:
    var person_list = List[Person]()
    sort(person_list)
    insertion_sort(person_list)
    quick_sort(person_list)
    assert_true(len(person_list) == 0)


def main():
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

    test_partition_top_k(7, 5)
    test_partition_top_k(11, 2)
    test_partition_top_k(4, 1)

    test_sort_string_small_list()
    test_sort_string_big_list()
    test_sort_strings()
    test_sort_comparamble_elements_list()
    test_sort_empty_comparamble_elements_list()
