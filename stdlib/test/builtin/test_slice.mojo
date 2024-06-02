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
    alias all_def_slice = slice(0, None, 1)
    assert_equal(all_def_slice.start, 0)
    assert_equal(all_def_slice.end, int(Int32.MAX))
    assert_equal(all_def_slice.step, 1)


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


struct Slicable:
    fn __init__(inout self):
        pass

    fn __getitem__(self, a: FunnySlice) -> FunnySlice:
        return a

    fn __getitem__(self, a: BoringSlice) -> BoringSlice:
        return a


def test_slicable():
    var slicable = Slicable()

    var new_slice = slicable[1:"hello":4.0]
    assert_equal(new_slice.start, 1)
    assert_equal(new_slice.upper, "hello")
    assert_equal(new_slice.stride, 4.0)

    var boring_slice = slicable[1:2:"foo"]
    assert_equal(boring_slice.a, 1)
    assert_equal(boring_slice.b, 2)
    assert_equal(boring_slice.c, "foo")


def test_has_end():
    alias is_end = Slice(None, None, None)._has_end()
    assert_false(is_end)


struct SliceStringable:
    fn __init__(inout self):
        pass

    fn __getitem__(self, a: Slice) -> String:
        return str(a)


def test_slice_stringable():
    var s = SliceStringable()
    assert_equal(s[2::-1], "2::-1")
    assert_equal(s[1:-1:2], "1:-1:2")
    assert_equal(s[:-1], "0:-1:1")


def test_indexing():
    var s = slice(1, 10)
    assert_equal(s[True], 2)
    assert_equal(s[int(0)], 1)
    assert_equal(s[2], 3)


def test_slice_indices_new():
    var start: Int
    var end: Int
    var step: Int
    var s = slice_new(1, 10)
    start, end, step = s.indices(9)
    assert_equal(slice_new(start, end, step), slice_new(1, 9))
    s = slice_new(1, None, 1)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(1, 5))
    s = slice_new(1, None, -1)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(1, -1, -1))
    s = slice_new(-1, None, 1)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(4, 5, 1))
    s = slice_new(None, 2, 1)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(0, 2, 1))
    s = slice_new(None, 2, -1)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(4, 2, -1))
    s = slice_new(0, -1, 1)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(0, 4, 1))
    s = slice_new(None, None, 1)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(0, 5, 1))
    s = slice_new(20)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(0, 5, 1))
    s = slice_new(10, -10, 1)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(5, 0, 1))
    # FIXME(#38392)
    # assert_equal(len(range(start, end, step)), 0)
    s = slice_new(-12, -10, -1)
    start, end, step = s.indices(5)
    assert_equal(slice_new(start, end, step), slice_new(-1, -1, -1))
    assert_equal(len(range(start, end, step)), 0)


def test_slice_eq_new():
    assert_equal(slice_new(1, 2, 3), slice_new(1, 2, 3))
    assert_equal(slice_new(0, 1), slice_new(1))
    assert_true(slice_new(2, 3) != slice_new(4, 5))
    assert_equal(slice_new(1, None, 1), slice_new(1, None, None))


def test_none_end_folds_new():
    alias all_def_slice = slice_new(0, None, 1)
    assert_equal(all_def_slice.start.value(), 0)
    assert_true(all_def_slice.end is None)
    assert_equal(all_def_slice.step, 1)


def main():
    test_none_end_folds()
    test_slicable()
    test_has_end()
    test_slice_stringable()
    test_indexing()
    test_slice_indices_new()
    test_slice_eq_new()
    test_none_end_folds_new()
