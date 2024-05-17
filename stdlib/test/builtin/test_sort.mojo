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
from testing import assert_true
from random import random_si64, random_ui64, random_float64


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


fn assert_sorted[dtype: DType](inout list: List[Scalar[dtype]]) raises:
    sort[dtype](list)
    for i in range(1, len(list)):
        assert_true(
            list[i] >= list[i - 1], str(list[i - 1]) + " > " + str(list[i])
        )


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


def test_sort_random_numbers():
    alias type_list = List[DType](
        DType.uint8,
        DType.int8,
        DType.uint16,
        DType.int16,
        DType.float16,
        DType.uint32,
        DType.int32,
        DType.float32,
        DType.uint64,
        DType.int64,
        DType.float64,
    )

    @parameter
    @always_inline
    fn perform_test[idx: Int]() raises:
        alias concrete_type = type_list[idx]
        var list = random_numbers[concrete_type](10)
        assert_sorted(list)
        list = random_numbers[concrete_type](100)
        assert_sorted(list)
        list = random_numbers[concrete_type](1000)
        assert_sorted(list)

    unroll[perform_test, len(type_list)]()


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


def test_sort_oder_comparamble_elements_list():
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


fn test_sort_empty_list() raises:
    var person_list = List[Person]()
    sort(person_list)
    insertion_sort(person_list)
    quick_sort(person_list)
    assert_true(len(person_list) == 0)

    var uint_list = List[UInt64]()
    sort[DType.uint64](uint_list)
    insertion_sort[DType.uint64](uint_list)
    quick_sort[DType.uint64](uint_list)
    assert_true(len(uint_list) == 0)


def main():
    test_sort_random_numbers()
    test_sort_string_small_list()
    test_sort_string_big_list()
    test_sort_strings()
    test_sort_oder_comparamble_elements_list()
    test_sort_empty_list()
