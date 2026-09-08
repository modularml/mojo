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

from std.builtin.builtin_slice import ContiguousSlice, StridedSlice
from test_utils import check_write_to
from std.testing import assert_equal, assert_true, TestSuite


def test_none_end_folds() raises:
    var all_def_slice = slice(0, None, 1)
    assert_equal(all_def_slice.start.value(), 0)
    assert_true(all_def_slice.end is None)
    assert_equal(all_def_slice.step.value(), 1)


# This requires parameter inference of StartT.
@fieldwise_init
struct FunnySlice(ImplicitlyCopyable):
    var start: Int
    var upper: String
    var stride: Float64
    var __slice_literal__: NoneType


@fieldwise_init
struct BoringSlice(ImplicitlyCopyable):
    var a: Int
    var b: Int
    var c: String
    var __slice_literal__: NoneType


struct Sliceable:
    def __init__(out self):
        pass

    def __getitem__(self, a: FunnySlice) -> FunnySlice:
        return a

    def __getitem__(self, a: BoringSlice) -> BoringSlice:
        return a


def test_sliceable() raises:
    var sliceable = Sliceable()

    var new_slice = sliceable[1:"hello":4.0]
    assert_equal(new_slice.start, 1)
    assert_equal(new_slice.upper, "hello")
    assert_equal(new_slice.stride, 4.0)

    var boring_slice = sliceable[1:2:"foo"]
    assert_equal(boring_slice.a, 1)
    assert_equal(boring_slice.b, 2)
    assert_equal(boring_slice.c, "foo")


struct SliceStringable:
    def __init__(out self):
        pass

    def __getitem__(self, a: Slice) -> String:
        return String(a)


def test_slice_stringable() raises:
    var s = SliceStringable()
    assert_equal(s[2::-1], "Slice(2, None, -1)")
    assert_equal(s[1:-1:2], "Slice(1, -1, 2)")
    assert_equal(s[:-1], "Slice(None, -1, None)")
    assert_equal(s[::], "Slice(None, None, None)")
    assert_equal(s[::4], "Slice(None, None, 4)")
    assert_equal(repr(slice(None, 2, 3)), "Slice(start=None, end=2, step=3)")
    assert_equal(repr(slice(10)), "Slice(start=None, end=10, step=None)")


def test_strided_slice_write_to() raises:
    check_write_to(
        StridedSlice(1, 10, 2), expected="Slice(1, 10, 2)", is_repr=False
    )
    check_write_to(
        StridedSlice(2, None, -1), expected="Slice(2, None, -1)", is_repr=False
    )
    check_write_to(
        StridedSlice(1, -1, 2), expected="Slice(1, -1, 2)", is_repr=False
    )
    check_write_to(
        StridedSlice(None, 5, 2), expected="Slice(None, 5, 2)", is_repr=False
    )

    check_write_to(
        StridedSlice(1, 10, 2),
        expected="Slice(start=1, end=10, step=2)",
        is_repr=True,
    )
    check_write_to(
        StridedSlice(2, None, -1),
        expected="Slice(start=2, end=None, step=-1)",
        is_repr=True,
    )
    check_write_to(
        StridedSlice(None, None, 3),
        expected="Slice(start=None, end=None, step=3)",
        is_repr=True,
    )


def test_contiguous_slice_write_to() raises:
    check_write_to(
        ContiguousSlice(1, 5, None), expected="Slice(1, 5, None)", is_repr=False
    )
    check_write_to(
        ContiguousSlice(None, 3, None),
        expected="Slice(None, 3, None)",
        is_repr=False,
    )
    check_write_to(
        ContiguousSlice(None, None, None),
        expected="Slice(None, None, None)",
        is_repr=False,
    )

    check_write_to(
        ContiguousSlice(1, 5, None),
        expected="Slice(start=1, end=5, step=None)",
        is_repr=True,
    )
    check_write_to(
        ContiguousSlice(None, None, None),
        expected="Slice(start=None, end=None, step=None)",
        is_repr=True,
    )


def test_slice_eq() raises:
    assert_equal(slice(1, 2, 3), slice(1, 2, 3))
    assert_equal(slice(None, 1, None), slice(1))
    assert_true(slice(2, 3) != slice(4, 5))
    assert_equal(slice(1, None, None), slice(1, None, None))
    assert_equal(slice(1, 2), slice(1, 2, None))


def test_slice_indices() raises:
    var start: Int
    var end: Int
    var step: Int
    var s = slice(1, 10)
    start, end, step = s.indices(9)
    assert_equal(slice(start, end, step), slice(1, 9, 1))
    s = slice(1, None, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(1, 5, 1))
    s = slice(1, None, -1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(1, -1, -1))
    s = slice(-1, None, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(4, 5, 1))
    s = slice(None, 2, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(0, 2, 1))
    s = slice(None, 2, -1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(4, 2, -1))
    s = slice(0, -1, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(0, 4, 1))
    s = slice(None, None, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(0, 5, 1))
    s = slice(20)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(0, 5, 1))
    s = slice(10, -10, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(5, 0, 1))
    assert_equal(len(range(start, end, step)), 0)
    s = slice(-12, -10, -1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(-1, -1, -1))
    assert_equal(len(range(start, end, step)), 0)
    # TODO: Decide how to handle 0 step
    # s = slice(-10, -2, 0)
    # start, end, step = s.indices(5)
    # assert_equal(slice(start, end, step), slice(-1, 3, 0))
    # assert_equal(len(range(start, end, step)), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
