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

from testing import assert_equal, assert_false, assert_true


def test_none_end_folds():
    var all_def_slice = slice(0, None, 1)
    assert_equal(all_def_slice.start.value(), 0)
    assert_true(all_def_slice.end is None)
    assert_equal(all_def_slice.step.value(), 1)


# This requires parameter inference of StartT.
@value
struct FunnySlice:
    var start: Int
    var upper: String
    var stride: Float64


@value
struct BoringSlice:
    var a: Int
    var b: Int
    var c: String


struct Sliceable:
    fn __init__(out self):
        pass

    fn __getitem__(self, a: FunnySlice) -> FunnySlice:
        return a

    fn __getitem__(self, a: BoringSlice) -> BoringSlice:
        return a


def test_slicable():
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
    fn __init__(out self):
        pass

    fn __getitem__(self, a: Slice) -> String:
        return str(a)


def test_slice_stringable():
    var s = SliceStringable()
    assert_equal(s[2::-1], "slice(2, None, -1)")
    assert_equal(s[1:-1:2], "slice(1, -1, 2)")
    assert_equal(s[:-1], "slice(None, -1, None)")
    assert_equal(s[::], "slice(None, None, None)")
    assert_equal(s[::4], "slice(None, None, 4)")
    assert_equal(repr(slice(None, 2, 3)), "slice(None, 2, 3)")
    assert_equal(repr(slice(10)), "slice(None, 10, None)")


def test_slice_eq():
    assert_equal(slice(1, 2, 3), slice(1, 2, 3))
    assert_equal(slice(None, 1, None), slice(1))
    assert_true(slice(2, 3) != slice(4, 5))
    assert_equal(slice(1, None, None), slice(1, None, None))
    assert_equal(slice(1, 2), slice(1, 2, None))


def test_slice_indices():
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


def main():
    test_none_end_folds()
    test_slicable()
    test_slice_stringable()
    test_slice_eq()
    test_slice_indices()
