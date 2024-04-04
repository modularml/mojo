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
# RUN: %mojo -debug-level full %s | FileCheck %s

from testing import assert_equal, assert_false


# CHECK-LABEL: test_none_end_folds
fn test_none_end_folds():
    print("== test_none_end_folds")
    alias all_def_slice = slice(0, None, 1)
    #      CHECK: 0
    # CHECK-SAME: 1
    print(all_def_slice.start, all_def_slice.end, all_def_slice.step)


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

    fn __getitem__(self, a: FunnySlice) -> Int:
        print(a.upper)
        print(a.stride)
        return a.start

    fn __getitem__(self, a: BoringSlice) -> String:
        print(a.a)
        print(a.b)
        return a.c


def test_slicable():
    # CHECK: Slicable
    print("Slicable")
    var slicable = Slicable()

    # CHECK: hello
    # CHECK: 4.0
    # CHECK: 1
    print(slicable[1:"hello":4.0])

    # CHECK: 1
    # CHECK: 2
    # CHECK: foo
    print(slicable[1:2:"foo"])


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


def main():
    test_none_end_folds()
    test_slicable()
    test_has_end()

    test_slice_stringable()
