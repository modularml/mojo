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


fn random_numbers[D: DType](size: Int, max: Int = 3000) -> List[SIMD[D, 1]]:
    var result = List[SIMD[D, 1]](size)
    for _ in range(size):

        @parameter
        if (
            D == DType.int8
            or D == DType.int16
            or D == DType.int32
            or D == DType.int64
        ):
            result.append(random_si64(0, max).cast[D]())
        elif D == DType.float16 or D == DType.float32 or D == DType.float64:
            result.append(random_float64(0, max).cast[D]())
        else:
            result.append(random_ui64(0, max).cast[D]())
    return result


fn assert_sorted[D: DType](inout list: List[SIMD[D, 1]]) raises:
    sort[D](list)
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
    D: OrderComparableCollectionElement
](inout list: List[D]) raises:
    for i in range(1, len(list)):
        assert_true(list[i] >= list[i - 1], "error at index: " + str(i))


def test_sort_random_numbers_u8():
    var list = random_numbers[DType.uint8](10)
    assert_sorted(list)
    list = random_numbers[DType.uint8](100)
    assert_sorted(list)
    list = random_numbers[DType.uint8](1000)
    assert_sorted(list)


def test_sort_random_numbers_i8():
    var list = random_numbers[DType.int8](10)
    assert_sorted(list)
    list = random_numbers[DType.int8](100)
    assert_sorted(list)
    list = random_numbers[DType.int8](1000)
    assert_sorted(list)


def test_sort_random_numbers_u16():
    var list = random_numbers[DType.uint16](10)
    assert_sorted(list)
    list = random_numbers[DType.uint16](100)
    assert_sorted(list)
    list = random_numbers[DType.uint16](1000)
    assert_sorted(list)


def test_sort_random_numbers_i16():
    var list = random_numbers[DType.int16](10)
    assert_sorted(list)
    list = random_numbers[DType.int16](100)
    assert_sorted(list)
    list = random_numbers[DType.int16](1000)
    assert_sorted(list)


def test_sort_random_numbers_f16():
    var list = random_numbers[DType.float16](10)
    assert_sorted(list)
    list = random_numbers[DType.float16](100)
    assert_sorted(list)
    list = random_numbers[DType.float16](1000)
    assert_sorted(list)


def test_sort_random_numbers_u32():
    var list = random_numbers[DType.uint32](10)
    assert_sorted(list)
    list = random_numbers[DType.uint32](100)
    assert_sorted(list)
    list = random_numbers[DType.uint32](1000)
    assert_sorted(list)


def test_sort_random_numbers_i32():
    var list = random_numbers[DType.int32](10)
    assert_sorted(list)
    list = random_numbers[DType.int32](100)
    assert_sorted(list)
    list = random_numbers[DType.int32](1000)
    assert_sorted(list)


def test_sort_random_numbers_f32():
    var list = random_numbers[DType.float32](10)
    assert_sorted(list)
    list = random_numbers[DType.float32](100)
    assert_sorted(list)
    list = random_numbers[DType.float32](1000)
    assert_sorted(list)


def test_sort_random_numbers_u64():
    var list = random_numbers[DType.uint64](10)
    assert_sorted(list)
    list = random_numbers[DType.uint64](100)
    assert_sorted(list)
    list = random_numbers[DType.uint64](1000)
    assert_sorted(list)


def test_sort_random_numbers_i64():
    var list = random_numbers[DType.int64](10)
    assert_sorted(list)
    list = random_numbers[DType.int64](100)
    assert_sorted(list)
    list = random_numbers[DType.int64](1000)
    assert_sorted(list)


def test_sort_random_numbers_f64():
    var list = random_numbers[DType.float64](10)
    assert_sorted(list)
    list = random_numbers[DType.float64](100)
    assert_sorted(list)
    list = random_numbers[DType.float64](1000)
    assert_sorted(list)


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
struct Person(OrderComparableCollectionElement):
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


def main():
    test_sort_random_numbers_u8()
    test_sort_random_numbers_i8()
    test_sort_random_numbers_u16()
    test_sort_random_numbers_i16()
    test_sort_random_numbers_f16()
    test_sort_random_numbers_u32()
    test_sort_random_numbers_i32()
    test_sort_random_numbers_f32()
    test_sort_random_numbers_u64()
    test_sort_random_numbers_i64()
    test_sort_random_numbers_f64()
    test_sort_string_small_list()
    test_sort_string_big_list()
    test_sort_strings()
    test_sort_oder_comparamble_elements_list()
