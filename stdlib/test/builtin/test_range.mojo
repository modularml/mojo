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
    # FIXME(#38392)
    # assert_equal(range(5, 10, -1).__len__(), 0, "len(range(5, 10, -1))")
    # assert_equal(range(10, 5, 1).__len__(), 0, "len(range(10, 5, 1))")
    # assert_equal(range(5, 10, -10).__len__(), 0, "len(range(5, 10, -10))")
    # assert_equal(range(10, 5, 10).__len__(), 0, "len(range(10, 5, 10))")
    assert_equal(range(5, 10, 20).__len__(), 1, "len(range(5, 10, 20))")
    assert_equal(range(10, 5, -20).__len__(), 1, "len(range(10, 5, -20))")


def test_range_getitem():
    # Usual cases
    assert_equal(range(10)[3], 3, "range(10)[3]")
    assert_equal(range(0, 10)[3], 3, "range(0, 10)[3]")
    assert_equal(range(5, 10)[3], 8, "range(5, 10)[3]")
    assert_equal(range(5, 10)[4], 9, "range(5, 10)[4]")
    assert_equal(range(10, 0, -1)[2], 8, "range(10, 0, -1)[2]")
    assert_equal(range(0, 10, 2)[4], 8, "range(0, 10, 2)[4]")
    assert_equal(range(38, -13, -23)[1], 15, "range(38, -13, -23)[1]")


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


def main():
    test_range_len()
    test_range_getitem()
    test_range_reversed()
    test_indexing()
