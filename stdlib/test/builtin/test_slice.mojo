# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


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


fn main():
    test_none_end_folds()

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

    # CHECK: False
    alias is_end = Slice(None, None, None)._has_end()
    print(is_end)
