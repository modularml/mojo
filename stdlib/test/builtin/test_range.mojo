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

from testing import assert_equal


def test_range_len():
    # Usual cases
    assert_equal(range(10).__len__(), 10, "len(range(10))")
    assert_equal(range(0, 10).__len__(), 10, "len(range(0, 10))")
    assert_equal(range(5, 10).__len__(), 5, "len(range(5, 10))")
    assert_equal(range(10, 0, -1).__len__(), 10, "len(range(10, 0, -1))")
    assert_equal(range(0, 10, 2).__len__(), 5, "len(range(0, 10, 2))")
    assert_equal(range(38, -13, -23).__len__(), 3, "len(range(38, -13, -23))")

    # Edge cases
    assert_equal(range(0).__len__(), 0, "len(range(0))")
    assert_equal(range(-10).__len__(), 0, "len(range(-10))")
    assert_equal(range(0, 0).__len__(), 0, "len(range(0, 0))")
    assert_equal(range(10, 0).__len__(), 0, "len(range(10, 0))")
    assert_equal(range(0, 0, 1).__len__(), 0, "len(range(0, 0, 1))")

    assert_equal(range(5, 10, -1).__len__(), 0, "len(range(5, 10, -1))")
    assert_equal(range(10, 5, 1).__len__(), 0, "len(range(10, 5, 1))")
    assert_equal(range(5, 10, -10).__len__(), 0, "len(range(5, 10, -10))")
    assert_equal(range(10, 5, 10).__len__(), 0, "len(range(10, 5, 10))")
    assert_equal(range(5, 10, 20).__len__(), 1, "len(range(5, 10, 20))")
    assert_equal(range(10, 5, -20).__len__(), 1, "len(range(10, 5, -20))")


def test_range_len_uint_maxuint():
    assert_equal(
        range(UInt(0), UInt.MAX).__len__(), UInt.MAX, "len(range(0, UInt.MAX))"
    )
    assert_equal(
        range(UInt.MAX, UInt(0), UInt(1)).__len__(),
        0,
        "len(range(UInt.MAX, 0, 1))",
    )


def test_range_len_uint_empty():
    assert_equal(
        range(UInt(0), UInt(0), UInt(1)).__len__(), 0, "len(range(0, 0, 1))"
    )
    assert_equal(
        range(UInt(10), UInt(10), UInt(1)).__len__(), 0, "len(range(10, 10, 1))"
    )


def test_range_len_uint():
    assert_equal(range(UInt(10)).__len__(), 10, "len(range(10))")

    # start < end
    assert_equal(range(UInt(0), UInt(10)).__len__(), 10, "len(range(0, 10))")
    assert_equal(range(UInt(5), UInt(10)).__len__(), 5, "len(range(5, 10))")
    assert_equal(
        range(UInt(0), UInt(10), UInt(2)).__len__(), 5, "len(range(0, 10, 2))"
    )
    # start > end
    assert_equal(
        range(UInt(10), UInt(0), UInt(1)).__len__(), 0, "len(range(10, 0, 1))"
    )


def test_range_getitem():
    # Usual cases
    assert_equal(range(10)[3], 3, "range(10)[3]")
    assert_equal(range(0, 10)[3], 3, "range(0, 10)[3]")
    assert_equal(range(5, 10)[3], 8, "range(5, 10)[3]")
    assert_equal(range(5, 10)[4], 9, "range(5, 10)[4]")
    assert_equal(range(10, 0, -1)[2], 8, "range(10, 0, -1)[2]")
    assert_equal(range(0, 10, 2)[4], 8, "range(0, 10, 2)[4]")
    assert_equal(range(38, -13, -23)[1], 15, "range(38, -13, -23)[1]")


def test_range_getitem_uint():
    assert_equal(range(UInt(10))[3], 3, "range(10)[3]")

    assert_equal(range(UInt(0), UInt(10))[3], 3, "range(0, 10)[3]")
    assert_equal(range(UInt(5), UInt(10))[3], 8, "range(5, 10)[3]")
    assert_equal(range(UInt(5), UInt(10))[4], 9, "range(5, 10)[4]")

    # Specify the step size > 1
    assert_equal(range(UInt(0), UInt(10), UInt(2))[4], 8, "range(0, 10, 2)[4]")

    # start > end
    var bad_strided_uint_range = range(UInt(10), UInt(5), UInt(1))
    var bad_strided_uint_range_iter = bad_strided_uint_range.__iter__()
    assert_equal(UInt(0), bad_strided_uint_range_iter.__len__())


def test_range_reversed():
    # Zero starting
    assert_equal(
        range(10).__reversed__().start, 9, "range(10).__reversed__().start"
    )
    assert_equal(
        range(10).__reversed__().end, -1, "range(10).__reversed__().end"
    )
    assert_equal(
        range(10).__reversed__().step, -1, "range(10).__reversed__().step"
    )
    # Sequential
    assert_equal(
        range(5, 10).__reversed__().start, 9, "range(5,10).__reversed__().start"
    )
    assert_equal(
        range(5, 10).__reversed__().end, 4, "range(5,10).__reversed__().end"
    )
    assert_equal(
        range(5, 10).__reversed__().step, -1, "range(5,10).__reversed__().step"
    )
    # Strided
    assert_equal(
        range(38, -13, -23).__reversed__().start,
        -8,
        "range(38, -13, -23).__reversed__().start",
    )
    assert_equal(
        range(38, -13, -23).__reversed__().end,
        61,
        "range(38, -13, -23).__reversed__().end",
    )
    assert_equal(
        range(38, -13, -23).__reversed__().step,
        23,
        "range(38, -13, -23).__reversed__().step",
    )

    # Test a reversed range's sum and length compared to the original
    @parameter
    fn test_sum_reversed(start: Int, end: Int, step: Int) raises:
        var forward = range(start, end, step)
        var iforward = forward.__iter__()
        var ibackward = forward.__reversed__()
        var backward = range(ibackward.start, ibackward.end, ibackward.step)
        assert_equal(
            forward.__len__(), backward.__len__(), "len(forward), len(backward)"
        )
        var forward_sum = 0
        var backward_sum = 0
        for i in range(len(forward)):
            forward_sum += iforward.__next__()
            backward_sum += ibackward.__next__()
        assert_equal(forward_sum, backward_sum, "forward_sum, backward_sum")

    # Test using loops and reversed
    for end in range(10, 13):
        test_sum_reversed(1, end, 3)

    for end in range(10, 13).__reversed__():
        test_sum_reversed(20, end, -3)


def test_indexing():
    var r = range(10)
    assert_equal(r[True], 1)
    assert_equal(r[int(4)], 4)
    assert_equal(r[3], 3)


def test_range_bounds():
    var start = 0
    var end = 10

    # verify loop iteration
    var r = range(start, end)
    var last_seen = -1
    for x in r:
        last_seen = x
    assert_equal(last_seen, end - 1)

    # verify index lookup
    var ln = r.__len__()
    assert_equal(r[ln - 1], last_seen)


def test_scalar_range():
    r = range(UInt8(2), 16, 4)
    assert_equal(r.start, 2)
    assert_equal(r.end, 16)
    assert_equal(r.step, 4)

    fn append_many(mut list: List, *values: list.T):
        for value in values:
            list.append(value[])

    expected_elements = List[UInt8]()
    append_many(expected_elements, 2, 6, 10, 14)
    actual_elements = List[UInt8]()
    for e in r:
        actual_elements.append(e)
    assert_equal(actual_elements, expected_elements)


def main():
    test_range_len()
    test_range_len_uint()
    test_range_len_uint_maxuint()
    test_range_len_uint_empty()
    test_range_getitem()
    test_range_getitem_uint()
    test_range_reversed()
    test_indexing()
    test_range_bounds()
