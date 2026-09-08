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

from std.testing import TestSuite
from std.testing import assert_equal, assert_raises, assert_true, assert_false
from test_utils import MoveOnly, check_write_to
from std.math import iota
from std.hashlib import Hasher


def test_span_list_int() raises:
    var l: List = [1, 2, 3, 4, 5, 6, 7]
    var s = Span(list=l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])
    assert_equal(s[len(s) - 1], l[len(l) - 1])

    # Test mutation
    s[0] = 9
    assert_equal(s[0], 9)
    assert_equal(l[0], 9)

    s[len(s) - 1] = 0
    assert_equal(s[len(s) - 1], 0)
    assert_equal(l[len(l) - 1], 0)


def test_span_list_str() raises:
    var l = ["a", "b", "c", "d", "e", "f", "g"]
    var s = Span(l)
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

    s[len(s) - 1] = "i"
    assert_equal(s[len(s) - 1], "i")
    assert_equal(l[len(l) - 1], "i")


def test_span_array_int() raises:
    var l: Array[Int, 7] = [1, 2, 3, 4, 5, 6, 7]
    var s = Span(array=l)
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

    s[len(s) - 1] = 0
    assert_equal(s[len(s) - 1], 0)
    assert_equal(l[len(l) - 1], 0)


def test_span_array_str() raises:
    var l: Array[String, 7] = ["a", "b", "c", "d", "e", "f", "g"]
    var s = Span(array=l)
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

    s[len(s) - 1] = "i"
    assert_equal(s[len(s) - 1], "i")
    assert_equal(l[len(l) - 1], "i")


def test_indexing() raises:
    var l: Array[Int, 7] = [1, 2, 3, 4, 5, 6, 7]
    var s = Span(array=l)
    assert_equal(s[Int(0)], 1)
    assert_equal(s[3], 4)


def test_span_slice() raises:
    def compare(s: Span[Int, ...], l: List[Int]) raises -> Bool:
        if len(s) != len(l):
            return False
        for i in range(len(s)):
            if s[i] != l[i]:
                return False
        return True

    var l = [1, 2, 3, 4, 5]
    var s = Span(l)
    var res = s[1:2]
    assert_equal(res[0], 2)
    res = s[1 : len(l) - 1]
    assert_equal(res[0], 2)
    assert_equal(res[1], 3)
    assert_equal(res[2], 4)


def test_copy_from() raises:
    var a = [0, 1, 2, 3]
    var b = [4, 5, 6, 7, 8, 9, 10]
    var s = Span(a)
    var s2 = Span(b)
    s.copy_from(s2[: len(a)])
    for i, val in enumerate(a):
        assert_equal(val, b[i])
        assert_equal(s[i], s2[i])


def test_bool() raises:
    var l: Array[String, 7] = ["a", "b", "c", "d", "e", "f", "g"]
    var s = Span(l)
    assert_true(s)
    assert_true(not s[0:0])


def test_contains() raises:
    var items: List[Byte] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    var span = Span(items)
    assert_true(0 not in span)
    assert_true(16 not in span)
    for item in items:
        assert_true(item in span)


def test_contains_non_scalar() raises:
    var items = ["a", "b", "c", "d"]
    var span = Span(items)
    assert_true("a" in span)
    assert_true("d" in span)
    assert_true("e" not in span)
    assert_true("" not in span)


def test_equality() raises:
    var l: Array[String, 7] = ["a", "b", "c", "d", "e", "f", "g"]
    var l2 = [String("a"), "b", "c", "d", "e", "f", "g"]
    var sp = Span(l)
    var sp2 = Span(l)
    var sp3 = Span(l2)
    # same pointer
    assert_true(sp == sp2)
    # different pointer
    assert_true(sp == sp3)
    # different length
    assert_true(sp != sp3[0 : len(l2) - 1])
    # empty
    assert_true(sp[0:0] == sp3[0:0])


def test_fill() raises:
    var a = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    var s = Span(a)

    s.fill(2)

    for i, val in enumerate(a):
        assert_equal(val, 2)
        assert_equal(s[i], 2)


def test_fill_bytes() raises:
    # Exercises the single-byte `memset` fast path in `Span.fill`: a non-zero
    # pattern, a length past the SIMD width (tail handling), and zeroing.
    var a = Array[Byte, 37](fill=0)
    var s = Span(array=a)

    s.fill(0xAB)
    for i in range(len(s)):
        assert_equal(s[i], 0xAB)

    s.fill(0)
    for i in range(len(s)):
        assert_equal(s[i], 0)


def test_fill_empty() raises:
    var a = Array[Byte, 4](fill=0)
    var s = Span(array=a)
    var empty = s[0:0]
    # Filling a zero-length span is a no-op and must not touch memory.
    empty.fill(0xFF)
    assert_equal(len(empty), 0)
    for i in range(len(s)):
        assert_equal(s[i], 0)


def test_fill_multibyte() raises:
    # Exercises the element-wise (non-`memset`) branch for a wider element type.
    var a = Array[Int32, 5](fill=0)
    var s = Span(array=a)

    s.fill(-1)
    for i in range(len(s)):
        assert_equal(s[i], -1)


def test_ref() raises:
    var l: Array[Int, 3] = [1, 2, 3]
    var s = Span(array=l)
    assert_true(s.as_ref() == Pointer(to=l.unsafe_ptr()[]))


def test_reversed() raises:
    var forward: Array[Int, 3] = [1, 2, 3]
    var backward: Array[Int, 3] = [3, 2, 1]
    var s = Span(forward)
    var i = 0
    for num in reversed(s):
        assert_equal(num, backward[i])
        i += 1


# We don't actually need to call this test
# but we want to make sure it compiles
def test_span_coerce() raises:
    var l = [1, 2, 3]
    var a: Array[Int, 3] = [1, 2, 3]

    def takes_span(s: Span[Int, ...]):
        pass

    takes_span(l)
    takes_span(a)


def test_swap_elements() raises:
    var l = [1, 2, 3, 4, 5]
    var s = Span(l)
    s.swap_elements(1, 4)
    assert_equal(l[1], 5)
    assert_equal(l[4], 2)

    var l2 = ["hi", "hello", "hey"]
    var s2 = Span(l2)
    s2.swap_elements(0, 2)
    assert_equal(l2[0], "hey")
    assert_equal(l2[2], "hi")

    with assert_raises(contains="index out of bounds"):
        s2.swap_elements(0, 4)


def test_merge() raises:
    var a: List = [1, 2, 3]
    var b: List = [4, 5, 6]

    def inner(cond: Bool, mut a: List[Int], mut b: List[Int]):
        var either = Span(a) if cond else Span(b)
        either[0] = 0
        either[len(either) - 1] = 10

    inner(True, a, b)
    inner(False, a, b)

    assert_equal(a, [0, 2, 10])
    assert_equal(b, [0, 5, 10])


def test_reverse() raises:
    def _test_dtype[D: DType]() raises:
        var forward: Array[Scalar[D], 11] = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]
        var backward: Array[Scalar[D], 11] = [
            11,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
        ]
        var s = Span(forward)
        s.reverse()
        var i = 0
        for num in s:
            assert_equal(num, backward[i])
            i += 1

    _test_dtype[.uint8]()
    _test_dtype[.uint16]()
    _test_dtype[.uint32]()
    _test_dtype[.uint64]()
    _test_dtype[.int8]()
    _test_dtype[.int16]()
    _test_dtype[.int32]()
    _test_dtype[.int64]()
    _test_dtype[.float16]()
    _test_dtype[.float32]()
    _test_dtype[.float64]()


def test_apply() raises:
    def _test[D: DType]() raises:
        def _twice[w: SIMDLength](x: SIMD[D, w]) -> SIMD[D, w]:
            return x * 2

        def _where[w: SIMDLength](x: SIMD[D, w]) -> SIMD[.bool, w]:
            return (x % 2).eq(0)

        var items: List[Scalar[D]] = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ]
        var twice = items.copy()
        var span = Span(twice)
        span.apply(_twice)
        for i, item in enumerate(items):
            assert_true(span[i] == item * 2)

        # twice only even numbers
        twice = items.copy()
        span = Span(twice)
        span.apply(_twice, cond=_where)
        for i, item in enumerate(items):
            if item % 2 == 0:
                assert_true(span[i] == item * 2)
            else:
                assert_true(span[i] == item)

    _test[.uint8]()
    _test[.uint16]()
    _test[.uint32]()
    _test[.uint64]()
    _test[.int8]()
    _test[.int16]()
    _test[.int32]()
    _test[.int64]()
    _test[.float16]()
    _test[.float32]()
    _test[.float64]()


def test_count_func() raises:
    def is_2[w: SIMDLength](v: SIMD[.uint8, w]) -> SIMD[.bool, w]:
        return v.eq(2)

    var data = Span([Byte(0), 1, 2, 1, 2, 1, 2])
    assert_equal(3, Int(data.count(is_2)))
    assert_equal(2, Int(data[0 : len(data) - 1].count(is_2)))
    assert_equal(1, Int(data[:3].count(is_2)))


def test_unsafe_subspan() raises:
    var data = Span([0, 1, 2, 3, 4])

    var subspan1 = data.unsafe_subspan(offset=0, length=4)
    assert_equal(List(subspan1), [0, 1, 2, 3])

    var subspan2 = data.unsafe_subspan(offset=1, length=3)
    assert_equal(List(subspan2), [1, 2, 3])


def test_binary_search() raises:
    def _test[dtype: DType]() raises:
        comptime max_val = Int(Scalar[dtype].MAX)
        var data = List[Scalar[dtype]](unsafe_uninit_length=max_val + 1)
        iota(data)

        # make sure we aren't reading an empty pointer
        var view = Span(data)[:0]
        assert_true(view._binary_search_index(0) is None)
        view = Span(data)[:1]
        assert_true(view._binary_search_index(0))
        assert_equal(view._binary_search_index(0).value(), 0)
        view = Span(data)[: len(data) - 1]
        assert_true(view._binary_search_index(1))
        assert_equal(view._binary_search_index(1).value(), 1)
        view = Span(data)
        assert_true(view._binary_search_index(Scalar[dtype](max_val)))
        assert_equal(
            view._binary_search_index(Scalar[dtype](max_val)).value(),
            max_val,
        )
        view = Span(data)[: len(data) - 1]
        assert_true(view._binary_search_index(Scalar[dtype](max_val - 1)))
        assert_equal(
            view._binary_search_index(Scalar[dtype](max_val - 1)).value(),
            max_val - 1,
        )

    _test[.uint8]()
    _test[.int8]()
    _test[.uint16]()
    _test[.int16]()


def test_binary_search_by() raises:
    var data: List[Int] = [1, 3, 5, 7, 9, 11, 13]
    var span = Span(data)

    def cmp_7(x: Int) -> Int:
        return x - 7

    var result = span.binary_search_by[cmp_7]()
    assert_equal(3, result.value())

    def cmp_6(x: Int) -> Int:
        return x - 6

    var result2 = span.binary_search_by[cmp_6]()
    assert_true(not result2)

    def cmp_1(x: Int) -> Int:
        return x - 1

    var result3 = span.binary_search_by[cmp_1]()
    assert_equal(0, result3.value())

    def cmp_13(x: Int) -> Int:
        return x - 13

    var result4 = span.binary_search_by[cmp_13]()
    assert_equal(6, result4.value())


def test_binary_search_by_unified() raises:
    var data: List[Int] = [1, 3, 5, 7, 9, 11, 13]
    var span = Span(data)

    var seven = 7

    def cmp_7(x: Int) {var} -> Int:
        return x - seven

    var result = span.binary_search_by(cmp_7)
    assert_equal(3, result.value())

    var six = 6

    def cmp_6(x: Int) {var six} -> Int:
        return x - six

    var result2 = span.binary_search_by(cmp_6)
    assert_true(not result2)

    var one = 1

    def cmp_1(x: Int) {var one} -> Int:
        return x - one

    var result3 = span.binary_search_by(cmp_1)
    assert_equal(0, result3.value())

    var thirteen = 13

    def cmp_13(x: Int) {var thirteen} -> Int:
        return x - thirteen

    var result4 = span.binary_search_by(cmp_13)
    assert_equal(6, result4.value())


def test_iter() raises:
    var data = [1, 2, 3, 4, 5]
    var span = Span(data)
    var it = iter(span)
    assert_equal(len(it), len(span))
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    assert_equal(next(it), 5)
    assert_equal(len(it), 0)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_iter_empty() raises:
    var data: List[Int] = []
    var span = Span(data)
    var it = iter(span)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_mut_span_alias() raises:
    var data = [1, 2, 3, 4, 5]

    def fill_span(span: MutSpan[Int, _]):
        span.fill(42)

    fill_span(data)
    for val in data:
        assert_equal(val, 42)


def test_immut_span_alias() raises:
    var data: List[Int] = [1, 2, 3, 4, 5]

    def sum_span(span: ImmSpan[Int, _]) -> Int:
        var total = 0
        for i in range(len(span)):
            total += span[i]
        return total

    # ImmSpan works with both mutable and immutable data
    assert_equal(sum_span(data), 15)


def test_span_write_to() raises:
    check_write_to(Span([1, 2, 3]), expected="[1, 2, 3]", is_repr=False)
    check_write_to(Span(List[Int]()), expected="[]", is_repr=False)
    check_write_to(Span([42]), expected="[42]", is_repr=False)


def test_span_write_repr_to() raises:
    check_write_to(
        Span([1, 2, 3]),
        expected=(
            "Span[mut=False, SIMD[DType.int, 1]]([Int(1), Int(2), Int(3)])"
        ),
        is_repr=True,
    )
    check_write_to(
        Span(List[Int]()),
        expected="Span[mut=False, SIMD[DType.int, 1]]([])",
        is_repr=True,
    )
    check_write_to(
        Span([42]),
        expected="Span[mut=False, SIMD[DType.int, 1]]([Int(42)])",
        is_repr=True,
    )


def test_span_hashable() raises:
    var a = [1, 2, 3]
    var b = [1, 2, 3]
    var c = [3, 2, 1]

    # Same contents should produce same hash.
    assert_equal(hash(Span(a)), hash(Span(b)))

    # Different contents should (almost certainly) produce different hashes.
    assert_true(hash(Span(a)) != hash(Span(c)))

    # Different lengths with shared prefix should produce different hashes
    # (prefix-freedom).
    var d = [1, 2]
    assert_true(hash(Span(a)) != hash(Span(d)))

    # Empty spans should hash equally.
    var empty1 = List[Int]()
    var empty2 = List[Int]()
    assert_equal(hash(Span(empty1)), hash(Span(empty2)))


@fieldwise_init
struct HashableOnly(Deinitable, Hashable, Movable):
    var value: Int

    def __hash__(self, mut hasher: Some[Hasher]):
        self.value.__hash__(hasher)


def test_span_hashable_non_copyable() raises:
    var allocation = alloc[HashableOnly]({count = 2}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_write(HashableOnly(1))
    ptr.unsafe_offset(1).unsafe_write(HashableOnly(2))
    var span = Span(unsafe_ptr=ptr, length=2)
    _ = hash(span)
    ptr.unsafe_offset(1).unsafe_deinit_pointee()
    ptr.unsafe_deinit_pointee()


def test_span_with_move_only_type() raises:
    var allocation = alloc[MoveOnly[Int]]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_write(MoveOnly(42))
    var span = Span(unsafe_ptr=ptr, length=1)
    assert_equal(span[0].data, 42)
    ptr.unsafe_deinit_pointee()


struct NonMovable:
    var x: Int


def test_span_with_non_movable_type() raises:
    var allocation = alloc[NonMovable]({count = 1}).into_managed()
    var ptr = allocation.unsafe_ptr()
    var _span = Span(unsafe_ptr=ptr, length=0)


def test_span_iter_owned() raises:
    var list = [10, 20, 30]
    var result = List[Int]()
    for elem in Span(list):
        result.append(elem)

    assert_equal(len(result), 3)
    assert_equal(result[0], 10)
    assert_equal(result[1], 20)
    assert_equal(result[2], 30)
    # Original list is still intact (Span doesn't own data).
    assert_equal(len(list), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
