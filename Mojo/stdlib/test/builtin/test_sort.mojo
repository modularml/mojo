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

from std.collections.string.string_span import _to_string_list
from std.pathlib import _dir_of_current_file
from std.random import rand, random_float64, random_si64, random_ui64, seed

from std.builtin.sort import (
    _heap_sort,
    _insertion_sort,
    _quicksort,
    _small_sort,
    _stable_sort,
)
from test_utils import CopyCounter
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def random_numbers[
    dtype: DType
](size: Int, max: Int = 3000) -> List[Scalar[dtype]]:
    var result = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        comptime if (
            dtype == .int8
            or dtype == .int16
            or dtype == .int32
            or dtype == .int64
        ):
            result.append(random_si64(0, Int64(max)).cast[dtype]())
        elif (dtype == .float16 or dtype == .float32 or dtype == .float64):
            result.append(random_float64(0, Float64(max)).cast[dtype]())
        else:
            result.append(random_ui64(0, UInt64(max)).cast[dtype]())
    return result^


def assert_sorted_string(mut list: List[String]) raises:
    for i in range(1, len(list)):
        assert_true(
            list[i] >= list[i - 1],
            String(list[i - 1], " > ", list[i]),
        )


def assert_sorted[T: Copyable & Comparable](mut list: List[T]) raises:
    for i in range(1, len(list)):
        assert_true(list[i] >= list[i - 1], String("error at index: ", i))


def test_sort_small_3() raises:
    comptime length = 3

    var list = List[Int]()

    list.append(9)
    list.append(1)
    list.append(2)

    def _less_than(lhs: Int, rhs: Int) -> Bool:
        return lhs < rhs

    _small_sort[length](list, _less_than)

    var expected = [1, 2, 9]
    for i in range(length):
        assert_equal(expected[i], list[i])


def test_sort_small_5() raises:
    comptime length = 5

    var list = List[Int]()

    list.append(9)
    list.append(1)
    list.append(2)
    list.append(3)
    list.append(4)

    def _less_than(lhs: Int, rhs: Int) -> Bool:
        return lhs < rhs

    _small_sort[length](list, _less_than)

    var expected = [1, 2, 3, 4, 9]
    for i in range(length):
        assert_equal(expected[i], list[i])


def test_sort0() raises:
    var list = List[Int]()

    sort(list)


def test_sort2() raises:
    comptime length = 2
    var list = List[Int]()

    list.append(-1)
    list.append(0)

    sort(list)

    var expected = [-1, 0]
    for i in range(length):
        assert_equal(expected[i], list[i])

    list[0] = 2
    list[1] = -2

    sort(list)

    expected = [-2, 2]
    for i in range(length):
        assert_equal(expected[i], list[i])


def test_sort3() raises:
    comptime length = 3
    var list = List[Int]()

    list.append(-1)
    list.append(0)
    list.append(1)

    sort(list)

    var expected = [-1, 0, 1]
    for i in range(length):
        assert_equal(expected[i], list[i])

    list[0] = 2
    list[1] = -2
    list[2] = 0

    sort(list)

    expected = [-2, 0, 2]
    for i in range(length):
        assert_equal(expected[i], list[i])


def test_sort3_dupe_elements() raises:
    comptime length = 3

    def test(cmp_fn: Some[def(Int, Int) -> Bool]) raises:
        var list = List[Int](capacity=3)
        list.append(5)
        list.append(3)
        list.append(3)

        _quicksort(list, cmp_fn)

        var expected = [3, 3, 5]
        for i in range(length):
            assert_equal(expected[i], list[i])

    def _lt(lhs: Int, rhs: Int) -> Bool:
        return lhs < rhs

    test(_lt)


def test_sort4() raises:
    comptime length = 4
    var list = List[Int]()

    list.append(-1)
    list.append(0)
    list.append(1)
    list.append(2)

    sort(list)

    var expected = [-1, 0, 1, 2]
    for i in range(length):
        assert_equal(expected[i], list[i])

    list[0] = 2
    list[1] = -2
    list[2] = 0
    list[3] = -4

    sort(list)

    expected = [-4, -2, 0, 2]
    for i in range(length):
        assert_equal(expected[i], list[i])


def test_sort5() raises:
    comptime length = 5
    var list = List[Int]()

    for i in range(5):
        list.append(i)

    sort(list)

    var expected = [0, 1, 2, 3, 4]
    for i in range(length):
        assert_equal(expected[i], list[i])

    list[0] = 2
    list[1] = -2
    list[2] = 0
    list[3] = -4
    list[4] = 1

    sort(list)

    expected = [-4, -2, 0, 1, 2]
    for i in range(length):
        assert_equal(expected[i], list[i])


def test_sort_reverse() raises:
    comptime length = 5
    var list = List[Int](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort(list)

    var expected = [0, 1, 2, 3, 4]
    for i in range(length):
        assert_equal(expected[i], list[i])


def test_sort_semi_random() raises:
    comptime length = 8
    var list = List[Int](capacity=length)

    for i in range(length):
        if i % 2:
            list.append(-i)
        else:
            list.append(i)

    sort(list)

    var expected = [-7, -5, -3, -1, 0, 2, 4, 6]
    for i in range(length):
        assert_equal(expected[i], list[i])


def test_sort9() raises:
    comptime length = 9
    var list = List[Int](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort(list)

    var expected = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(length):
        assert_equal(expected[i], list[i])


def test_sort103() raises:
    comptime length = 103
    var list = List[Int](capacity=length)

    for i in range(length):
        list.append(length - i - 1)

    sort(list)

    for i in range(1, length):
        assert_false(list[i - 1] > list[i])

    var list1 = List[Int](capacity=length)

    for i in range(length):
        list1.append(length - i - 1)

    sort[stable=True](list1)

    for i in range(1, length):
        assert_false(list1[i - 1] > list1[i])


def test_sort_any_103() raises:
    comptime length = 103
    var list = List[Float32](capacity=length)

    for i in range(length):
        list.append(Float32(length - i - 1))

    sort(list)

    for i in range(1, length):
        assert_false(list[i - 1] > list[i])


def test_quick_sort_repeated_val() raises:
    comptime length = 36
    var list = List[Float32](capacity=length)

    for i in range(0, length // 4):
        list.append(Float32(i + 1))
        list.append(Float32(i + 1))
        list.append(Float32(i + 1))
        list.append(Float32(i + 1))

    def _greater_than(lhs: Float32, rhs: Float32) -> Bool:
        return lhs > rhs

    _quicksort(list, _greater_than)

    var expected: List[Float32] = [
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
    ]
    for i in range(0, length):
        assert_equal(expected[i], list[i])

    def _less_than(lhs: Float32, rhs: Float32) -> Bool:
        return lhs < rhs

    expected: List[Float32] = [
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
    ]
    _quicksort(list, _less_than)
    for i in range(0, length):
        assert_equal(expected[i], list[i])


def _test_partition_top_k(length: Int, k: Int) raises:
    var list = List[Float32](capacity=length)

    for i in range(0, length):
        list.append(Float32(i))

    def _great_than(lhs: Float32, rhs: Float32) -> Bool:
        return lhs > rhs

    _ = partition(list, k, _great_than)

    for i in range(0, k):
        assert_false(list[i] < Float32(length - k))


def test_sort_stress() raises:
    var lens = [3, 100, 117, 223, 500, 1000, 1500, 2000, 3000]
    var random_seed = 0
    seed(random_seed)

    def test(
        length: Int,
        cmp_fn: Some[def(Int, Int) -> Bool],
        check_fn: Some[def(Int, Int) -> Bool],
    ) raises:
        var list = List[Int](capacity=length)
        for _ in range(length):
            list.append(Int(random_si64(Int64(-length), Int64(length))))

        _quicksort(list, cmp_fn)

        for i in range(length - 1):
            assert_true(check_fn(list[i], list[i + 1]))

    @always_inline
    def _gt(lhs: Int, rhs: Int) -> Bool:
        return lhs > rhs

    @always_inline
    def _geq(lhs: Int, rhs: Int) -> Bool:
        return lhs >= rhs

    @always_inline
    def _lt(lhs: Int, rhs: Int) -> Bool:
        return lhs < rhs

    @always_inline
    def _leq(lhs: Int, rhs: Int) -> Bool:
        return lhs <= rhs

    for length in lens:
        test(length, _gt, _geq)
        test(length, _lt, _leq)


@fieldwise_init
struct MyStruct(ImplicitlyCopyable):
    var val: Int


def test_sort_custom() raises:
    comptime length = 103

    var list = List[MyStruct](capacity=length)

    for i in range(length):
        list.append(MyStruct(length - i - 1))

    def compare_fn(lhs: MyStruct, rhs: MyStruct) -> Bool:
        return lhs.val < rhs.val

    sort(list, compare_fn)

    for i in range(1, length):
        assert_false(list[i - 1].val > list[i].val)


def test_sort_string_small_list() raises:
    var list = random_numbers[.int32](10)
    var string_list = List[String]()
    for n in list:
        string_list.append(String(Int(n)))
    sort(string_list)
    assert_sorted_string(string_list)


def test_sort_string_big_list() raises:
    var list = random_numbers[.int32](1000)
    var string_list = List[String]()
    for n in list:
        string_list.append(String(Int(n)))
    sort(string_list)
    assert_sorted_string(string_list)


def test_sort_strings() raises:
    var text = (
        _dir_of_current_file() / "test_file_dummy_input.txt"
    ).read_text()
    var strings = _to_string_list(text.split(" "))
    sort(strings)
    assert_sorted_string(strings)


@fieldwise_init
struct Person(Comparable, ImplicitlyCopyable):
    """Test struct using default reflection-based __eq__."""

    var name: String
    var age: Int

    def __lt__(self, other: Self) -> Bool:
        if self.age < other.age:
            return True
        if self.age == other.age:
            return self.name < other.name
        return False

    # Uses default reflection-based __eq__ from Equatable trait


def test_sort_comparamble_elements_list() raises:
    var list = List[Person]()

    def gen_list(count: Int) {mut list}:
        list = List[Person]()
        var ages = random_numbers[.uint8](count)
        var names = ["Maxim", "Max", "Alex", "Bob", "Joe"]
        for age in ages:
            var name = names[Int(age) % len(names)]
            list.append(Person(name, Int(age)))

    gen_list(10)
    sort(list)
    assert_sorted(list)

    gen_list(100)
    sort(list)
    assert_sorted(list)

    gen_list(1000)
    sort(list)
    assert_sorted(list)


def test_sort_empty_comparable_elements_list() raises:
    var person_list = List[Person]()
    sort(person_list)
    assert_true(len(person_list) == 0)


@fieldwise_init
struct IntPair(TrivialRegisterPassable):
    var x: Int
    var idx: Int


def test_stable_sort_stress() raises:
    var lens = [3, 100, 117, 223, 500, 1000, 1500, 2000, 3000]
    var random_seed = 0
    seed(random_seed)

    def test(
        length: Int,
        cmp_fn: Some[def(IntPair, IntPair) -> Bool],
        check_fn: Some[def(IntPair, IntPair) -> Bool],
    ) raises:
        var list = List[IntPair](capacity=length)
        for i in range(length):
            # make the range smaller so we can get more repeats
            list.append(IntPair(Int(random_si64(0, 100)), i))

        sort[stable=True](list, cmp_fn)

        for i in range(length - 1):
            assert_true(check_fn(list[i], list[i + 1]))

    # sort by only comparing the x value of the IntPair, then check the sort is
    # stable by making sure that for the same x value, the idx field is sorted.
    @always_inline
    def _lt(lhs: IntPair, rhs: IntPair) -> Bool:
        return lhs.x < rhs.x

    @always_inline
    def _lt_check(lhs: IntPair, rhs: IntPair) -> Bool:
        return lhs.idx < rhs.idx if lhs.x == rhs.x else lhs.x < rhs.x

    for length in lens:
        test(length, _lt, _lt_check)


def test_sort_scalar() raises:
    var listi32 = random_numbers[.int32](50, max=Int(Int32.MAX))
    sort(listi32)
    assert_sorted(listi32)

    # Note: We'd use Float32.MAX_FINITE here, but it doesn't fit in Int
    # (random_numbers takes Int max). Float32.MAX returns inf, so we use Int.MAX
    # which is safe and provides good coverage (9.2e18 >> typical float32 values)
    var listf32 = random_numbers[.float32](50, max=Int.MAX)
    sort(listf32)
    assert_sorted(listf32)


def test_ensure_no_copies() raises:
    def get_list() -> List[CopyCounter[UInt64]]:
        seed(0)
        var list = List[CopyCounter[UInt64]](capacity=50)
        for _ in range(50):
            list.append(CopyCounter[UInt64](random_ui64(min=0, max=UInt64.MAX)))
        return list^

    def verify_list(list: List[CopyCounter[UInt64]]) raises:
        for e in list:
            assert_true(e.copy_count == 0)

    def cmp_fn(lhs: CopyCounter[UInt64], rhs: CopyCounter[UInt64]) -> Bool:
        return lhs.value < rhs.value

    var list = get_list()
    _insertion_sort(list, cmp_fn)
    verify_list(list)

    list = get_list()
    _stable_sort(list, cmp_fn)
    verify_list(list)

    list = get_list()
    _quicksort(list, cmp_fn)
    verify_list(list)

    list = get_list()
    _heap_sort(list, cmp_fn)
    verify_list(list)

    list = get_list()
    sort(list, cmp_fn)
    verify_list(list)


def test_partition() raises:
    _test_partition_top_k(7, 5)
    _test_partition_top_k(11, 2)
    _test_partition_top_k(4, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
