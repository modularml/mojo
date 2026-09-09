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

from std.sys.info import size_of

from std.compile import compile_info
from std.hashlib import hash
from std.memory import MaybeUninit
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)
from test_utils import (
    CopyCounter,
    DelRecorder,
    ExplicitDestroy,
    MoveCounter,
    MoveOnly,
    check_write_to,
)
from std.testing import assert_equal, assert_true, assert_false, TestSuite


def test_array_unsafe_get() raises:
    # Negative indexing is undefined behavior with unsafe_get
    # so there are not test cases for it.
    var arr: Array[Int, 3] = [0, 0, 0]

    assert_equal(arr.unsafe_get(0), 0)
    assert_equal(arr.unsafe_get(1), 0)
    assert_equal(arr.unsafe_get(2), 0)

    arr[0] = 1
    arr[1] = 2
    arr[2] = 3

    assert_equal(arr.unsafe_get(0), 1)
    assert_equal(arr.unsafe_get(1), 2)
    assert_equal(arr.unsafe_get(2), 3)


def test_array_int() raises:
    var arr: Array[Int, 3] = [0, 0, 0]

    assert_equal(arr[0], 0)
    assert_equal(arr[1], 0)
    assert_equal(arr[2], 0)

    arr[0] = 1
    arr[1] = 2
    arr[2] = 3

    assert_equal(arr[0], 1)
    assert_equal(arr[1], 2)
    assert_equal(arr[2], 3)

    # test indexing from end
    assert_equal(arr[len(arr) - 1], 3)
    assert_equal(arr[len(arr) - 2], 2)

    # test indexing from end with dynamic index
    var i = len(arr) - 1
    assert_equal(arr[i], 3)
    i -= 1
    assert_equal(arr[i], 2)

    var copy = arr.copy()
    assert_equal(arr[0], copy[0])
    assert_equal(arr[1], copy[1])
    assert_equal(arr[2], copy[2])

    var move = arr^
    assert_equal(copy[0], move[0])
    assert_equal(copy[1], move[1])
    assert_equal(copy[2], move[2])

    # fill element initializer
    var arr2 = Array[Int, 3](fill=5)
    assert_equal(arr2[0], 5)
    assert_equal(arr2[1], 5)
    assert_equal(arr2[2], 5)

    var arr3: Array[Int, 1] = [5]
    assert_equal(arr3[0], 5)

    def test_init_fill[
        size: Int, batch_size: Int, dt: DType
    ](arg: Scalar[dt]) raises {imm}:
        var arr = Array[Scalar[dt], size].__init__[batch_size=batch_size](
            fill=arg
        )
        for i in range(size):
            assert_equal(arr[i], arg)

    def test_init_fill_scalars[
        *dts: DType, sizes: List[Int], batch_sizes: List[Int]
    ]() raises {imm}:
        comptime for current_batch_size in range(len(batch_sizes)):
            comptime for current_size in range(len(sizes)):
                comptime for current_type in range(dts.size):
                    test_init_fill[
                        sizes[current_size], batch_sizes[current_batch_size]
                    ](Scalar[dts[current_type]].MAX)

    test_init_fill_scalars[
        Int64.dtype,
        Int8.dtype,
        sizes=[1, 32, 64, 129, 256, 512, 768, 1000],
        batch_sizes=[1, 8, 32, 64, 128],
    ]()

    test_init_fill[2048, 512](Int64.MAX)
    test_init_fill[2048, 1](Int64.MAX)


def test_array_fill_with() raises:
    var squares = Array[Int, 5](fill_with=lambda (i: Int) -> Int: i * i)
    for i in range(5):
        assert_equal(squares[i], i * i)

    var strs = Array[String, 3](fill_with=lambda (i: Int) -> String: String(i))
    for i in range(3):
        assert_equal(strs[i], String(i))


def test_array_fill_with_named_function() raises:
    def cube(i: Int) {imm} -> Int:
        return i * i * i

    var cubes = Array[Int, 4](fill_with=cube)
    for i in range(4):
        assert_equal(cubes[i], i * i * i)


def test_array_fill_with_non_movable() raises:
    # `fill_with=` constructs each element in place, so it works even for a
    # non-`Movable` element type -- no constraint on `T` at all.
    var arr = Array[NonMovable, 3](
        fill_with=lambda (i: Int) -> NonMovable: NonMovable(i * 10)
    )
    assert_equal(arr[0].value, 0)
    assert_equal(arr[1].value, 10)
    assert_equal(arr[2].value, 20)


def test_array_fill_with_explicit_batch_size() raises:
    var arr = Array[Int, 10].__init__[batch_size=4](
        fill_with=lambda (i: Int) -> Int: i * 2
    )
    for i in range(10):
        assert_equal(arr[i], i * 2)


def test_array_fill_with_crosses_batch_boundary() raises:
    # The default `batch_size` is 64: indices `>= 64` come from a runtime
    # loop rather than a `comptime for` unroll, so this exercises the
    # `batch_start + i` arithmetic on both sides of that boundary.
    comptime size = 130
    var arr = Array[Int, size](fill_with=lambda (i: Int) -> Int: i + 1)
    for i in range(size):
        assert_equal(arr[i], i + 1)


def test_array_String() raises:
    var arr: Array[String, 3] = ["hi", "hello", "hey"]

    assert_equal(arr[0], "hi")
    assert_equal(arr[1], "hello")
    assert_equal(arr[2], "hey")

    # Test mutating an array through its __getitem__
    arr[0] = "howdy"
    arr[1] = "morning"
    arr[2] = "wazzup"

    assert_equal(arr[0], "howdy")
    assert_equal(arr[1], "morning")
    assert_equal(arr[2], "wazzup")

    # test indexing from end
    assert_equal(arr[len(arr) - 1], "wazzup")
    assert_equal(arr[len(arr) - 2], "morning")

    var copy = arr.copy()
    assert_equal(arr[0], copy[0])
    assert_equal(arr[1], copy[1])
    assert_equal(arr[2], copy[2])

    var move = arr^
    assert_equal(copy[0], move[0])
    assert_equal(copy[1], move[1])
    assert_equal(copy[2], move[2])

    # fill element initializer
    var arr2 = Array[String, 3](fill="hi")
    assert_equal(arr2[0], "hi")
    assert_equal(arr2[1], "hi")
    assert_equal(arr2[2], "hi")

    # size 1 array to prevent regressions in the constructors
    var arr3: Array[String, 1] = ["hi"]
    assert_equal(arr3[0], "hi")


def test_array_default_init() raises:
    var arr = Array[Int, 5]()
    for i in range(5):
        assert_equal(arr[i], Int())

    var arr2 = Array[String, 3]()
    for i in range(3):
        assert_equal(arr2[i], String())


def test_array_int_pointer() raises:
    var arr: Array[Int, 3] = [0, 10, 20]

    var ptr = arr.unsafe_ptr()
    assert_equal(ptr[unsafe_offset=0], 0)
    assert_equal(ptr[unsafe_offset=1], 10)
    assert_equal(ptr[unsafe_offset=2], 20)

    ptr[unsafe_offset=0] = 0
    ptr[unsafe_offset=1] = 1
    ptr[unsafe_offset=2] = 2

    assert_equal(arr[0], 0)
    assert_equal(arr[1], 1)
    assert_equal(arr[2], 2)

    assert_equal(ptr[unsafe_offset=0], 0)
    assert_equal(ptr[unsafe_offset=1], 1)
    assert_equal(ptr[unsafe_offset=2], 2)

    # We make sure it lives long enough
    _ = arr


def test_array_unsafe_assume_initialized_constructor_string() raises:
    var maybe_uninitialized_arr = Array[MaybeUninit[String], 3](
        uninitialized=True
    )
    maybe_uninitialized_arr[0].unsafe_write("hello")
    maybe_uninitialized_arr[1].unsafe_write("mojo")
    maybe_uninitialized_arr[2].unsafe_write("world")

    var initialized_arr = Array[String, 3](
        unsafe_assume_initialized=maybe_uninitialized_arr^
    )

    assert_equal(initialized_arr[0], "hello")
    assert_equal(initialized_arr[1], "mojo")
    assert_equal(initialized_arr[2], "world")

    # trigger a move
    var initialized_arr2 = initialized_arr^

    assert_equal(initialized_arr2[0], "hello")
    assert_equal(initialized_arr2[1], "mojo")
    assert_equal(initialized_arr2[2], "world")

    # trigger a copy
    var initialized_arr3 = initialized_arr2.copy()

    assert_equal(initialized_arr3[0], "hello")
    assert_equal(initialized_arr3[1], "mojo")
    assert_equal(initialized_arr3[2], "world")

    # We assume the destructor was called correctly, but one
    # might want to add a test for that in the future.


def test_array_contains() raises:
    var arr: Array[String, 3] = ["hi", "hello", "hey"]
    assert_true("hi" in arr)
    assert_true(not "greetings" in arr)


def test_array_runs_destructors() raises:
    """Ensure we delete the right number of elements."""
    var destructor_recorder = List[Int]()
    var ptr = Pointer(to=destructor_recorder).as_imm()
    comptime capacity = 32
    var arr: Array[DelRecorder[ptr.origin], 4] = [
        DelRecorder(0, ptr),
        DelRecorder(10, ptr),
        DelRecorder(20, ptr),
        DelRecorder(30, ptr),
    ]
    _ = arr
    # This is the last use of the array, so it should be destroyed here,
    # along with each element.
    assert_equal(len(destructor_recorder), 4)
    assert_equal(destructor_recorder[0], 0)
    assert_equal(destructor_recorder[1], 10)
    assert_equal(destructor_recorder[2], 20)
    assert_equal(destructor_recorder[3], 30)


def test_unsafe_ptr() raises:
    comptime N = 10
    var arr = Array[Int, 10](fill=0)
    for i in range(N):
        arr[i] = i

    var ptr = arr.unsafe_ptr()
    for i in range(N):
        assert_equal(arr[i], ptr[unsafe_offset=i])


def _test_size_of_array[current_type: Copyable, capacity: Int]() raises:
    """Testing if `size_of` the array equals capacity * `size_of` current_type.

    Parameters:
        current_type: The type of the elements of the `Array`.
        capacity: The capacity of the `Array`.
    """
    comptime size_of_current_type = size_of[current_type]()
    assert_equal(
        size_of[Array[current_type, capacity]](),
        capacity * size_of_current_type,
    )


def test_size_of_array() raises:
    _test_size_of_array[Int, 32]()


def test_move() raises:
    """Test that moving an Array works correctly."""

    # === 1. Check that the move constructor is called correctly. ===

    var arr: Array[MoveCounter[Int], 3] = [{1}, {2}, {3}]
    var copied_arr = arr.copy()

    for i in range(len(arr)):
        # The elements were moved into the array
        assert_equal(arr[i].move_count, 1)

    var moved_arr = arr^

    for i in range(len(moved_arr)):
        # Check that the moved array has the same elements as the copied array
        assert_equal(copied_arr[i].value, moved_arr[i].value)
        # Check that the move constructor was called again for each element
        assert_equal(moved_arr[i].move_count, 2)

    # === 2. Check that the copy constructor is not called when moving. ===

    var arr2: Array[CopyCounter[], 3] = [{}, {}, {}]
    for i in range(len(arr2)):
        # The elements were moved into the array and not copied
        assert_equal(arr2[i].copy_count, 0)

    var moved_arr2 = arr2^

    for i in range(len(moved_arr2)):
        # Check that the copy constructor was not called
        assert_equal(moved_arr2[i].copy_count, 0)

    # === 3. Check that the destructor is not called when moving. ===

    var del_counter = List[Int]()
    var del_counter_ptr = Pointer(to=del_counter).as_imm()
    var del_recorder = DelRecorder[del_counter_ptr.origin](0, del_counter_ptr)
    var arr3: Array[DelRecorder[del_counter_ptr.origin], 1] = [del_recorder]

    assert_equal(len(del_counter_ptr[]), 0)

    var moved_arr3 = arr3^

    assert_equal(len(del_counter_ptr[]), 0)

    _ = moved_arr3

    # Double check that the destructor is called when the array is destroyed
    assert_equal(len(del_counter_ptr[]), 1)
    _ = del_recorder
    _ = del_counter


def test_different_types() raises:
    """Test with different element types."""
    # Test with single element
    var single: Array[Int, 1] = [42]
    var single_str = String(single)
    assert_equal(single_str, "[42]")

    # Test with default values
    var default_array = Array[Int, 3](fill=0)
    var default_str = String(default_array)
    assert_equal(default_str, "[0, 0, 0]")

    # Test with filled array
    var filled_array = Array[Int, 4](fill=99)
    var filled_str = String(filled_array)
    assert_equal(filled_str, "[99, 99, 99, 99]")


def test_write_to() raises:
    """Test Writable trait implementation."""
    var array: Array[Int, 3] = [10, 20, 30]
    check_write_to(array, expected="[10, 20, 30]", is_repr=False)

    var string_array: Array[String, 2] = ["a", "b"]
    check_write_to(string_array, expected="[a, b]", is_repr=False)

    var single_elem: Array[Int, 1] = [42]
    check_write_to(single_elem, expected="[42]", is_repr=False)


def test_write_repr_to() raises:
    """Test write_repr_to implementation."""
    var array: Array[Int, 3] = [1, 2, 3]
    check_write_to(
        array,
        expected="Array[SIMD[DType.int, 1], 3]([Int(1), Int(2), Int(3)])",
        is_repr=True,
    )

    var single: Array[Int, 1] = [1]
    check_write_to(
        single,
        expected="Array[SIMD[DType.int, 1], 1]([Int(1)])",
        is_repr=True,
    )


def test_array_triviality() raises:
    assert_true(IsTriviallyDeinitable[Array[Int, 1]])
    assert_true(IsTriviallyCopyable[Array[Int, 1]])
    assert_true(IsTriviallyMovable[Array[Int, 1]])

    assert_false(IsTriviallyDeinitable[Array[String, 1]])
    assert_false(IsTriviallyCopyable[Array[String, 1]])
    assert_true(IsTriviallyMovable[Array[String, 1]])


def _return_array[copy: Bool = False]() -> Array[Int32, 4]:
    var arr = Array[Int32, 4](fill=0)

    comptime if copy:
        return arr.copy()
    else:
        return arr^


def _return_batched_array[copy: Bool = False]() -> Array[Int32, 64]:
    var arr = Array[Int32, 64](fill=0)

    comptime if copy:
        return arr.copy()
    else:
        return arr^


def test_array_batched_copy_and_move_llvm_ir() raises:
    # 64 elements reaches `fill=`'s batched runtime loop, unlike the 4-element
    # case above which unrolls at compile time. A callsite marker is expected
    # for the live loop, so this checks only the range attribute.
    def _test(ir: StringSlice) raises:
        assert_true("initializes((0, 256))" in ir)

    var move_info = compile_info[
        _return_batched_array[copy=False], emission_kind="llvm-opt"
    ]()
    _test(move_info.asm)
    var copy_info = compile_info[
        _return_batched_array[copy=True], emission_kind="llvm-opt"
    ]()
    _test(copy_info.asm)


def test_array_copy_and_move_llvm_ir() raises:
    def _test(ir: StringSlice) raises:
        assert_true("initializes((0, 16))" in ir)
        assert_false('asm sideeffect "nop"' in ir)

    var move_info = compile_info[
        _return_array[copy=False], emission_kind="llvm-opt"
    ]()
    _test(move_info.asm)
    var copy_info = compile_info[
        _return_array[copy=True], emission_kind="llvm-opt"
    ]()
    _test(copy_info.asm)


def test_array_iter() raises:
    var arr: Array[Int, 3] = [0, 1, 2]
    var s = 0
    for el in arr:
        s += el
    assert_equal(s, 3)

    for el in reversed(arr):
        s -= el
    assert_equal(s, 0)


def test_array_iter_mut() raises:
    var arr: Array[Int, 3] = [0, 1, 2]
    for ref el in arr:
        el += 1

    var s = 0
    for el in arr:
        s += el
    assert_equal(s, 6)


def _test_array_iter_bounds[
    I: Iterator
](var array_iter: I, array_len: Int) raises where conforms_to(
    I.Element, Deinitable
):
    var iter = array_iter^

    for i in range(array_len):
        var lower, upper = iter.bounds()
        assert_equal(array_len - i, lower)
        assert_equal(array_len - i, upper.value())
        _ = iter.__next__()

    var lower, upper = iter.bounds()
    assert_equal(0, lower)
    assert_equal(0, upper.value())


struct NonWritable(Copyable):
    """A simple type that does not conform to Writable."""

    var value: Int


struct NonMovable(Movable where False):
    """A non-`Movable` (pinned) type; still implicitly deletable by default."""

    var value: Int

    def __init__(out self, value: Int):
        self.value = value


struct LinearNonMovable(Deinitable where False, Movable where False):
    """A fully linear type: neither `Movable` nor `Deinitable`."""

    var value: Int


struct NonDefaultable:
    pass


struct DefaultableNonMovable(Defaultable, Movable where False):
    """A `Defaultable` type that is not `Movable`."""

    var value: Int

    def __init__(out self):
        self.value = 0


def test_array_eq() raises:
    var a: Array[Int, 3] = [1, 2, 3]
    var b: Array[Int, 3] = [1, 2, 3]
    var c: Array[Int, 3] = [1, 2, 4]

    assert_true(a == b)
    assert_false(a == c)
    assert_false(a != b)
    assert_true(a != c)

    var x: Array[Int, 1] = [42]
    var y: Array[Int, 1] = [42]
    var z: Array[Int, 1] = [0]
    assert_true(x == y)
    assert_true(x != z)

    var s1: Array[String, 2] = ["hello", "world"]
    var s2: Array[String, 2] = ["hello", "world"]
    var s3: Array[String, 2] = ["hello", "mojo"]
    assert_true(s1 == s2)
    assert_true(s1 != s3)

    # Test arrays of different lengths.
    var d: Array[Int, 4] = [1, 2, 3, 4]
    var e: Array[Int, 4] = [1, 2, 3, 4]
    var f: Array[Int, 4] = [1, 2, 3, 5]
    assert_true(d == e)
    assert_true(d != f)

    var g: Array[Int, 5] = [10, 20, 30, 40, 50]
    var h: Array[Int, 5] = [10, 20, 30, 40, 50]
    var i: Array[Int, 5] = [10, 20, 99, 40, 50]
    assert_true(g == h)
    assert_true(g != i)


def test_array_ordering() raises:
    # Lexicographic ordering: the first differing element decides, so
    # `[1, 5] < [2, 3]` even though `5 > 3`. This is the key behavioral
    # difference from `IndexList`'s element-wise comparison.
    var a: Array[Int, 2] = [1, 5]
    var b: Array[Int, 2] = [2, 3]
    assert_true(a < b)
    assert_true(a <= b)
    assert_false(a > b)
    assert_false(a >= b)
    assert_false(b < a)
    assert_true(b > a)

    # Equal arrays: strict comparisons are False, non-strict are True.
    var c: Array[Int, 3] = [1, 2, 3]
    var d: Array[Int, 3] = [1, 2, 3]
    assert_false(c < d)
    assert_true(c <= d)
    assert_false(c > d)
    assert_true(c >= d)

    # First element decides when it differs.
    var e: Array[Int, 3] = [1, 9, 9]
    var f: Array[Int, 3] = [2, 0, 0]
    assert_true(e < f)
    assert_true(f > e)

    # Difference in the last element.
    var g: Array[Int, 3] = [1, 2, 3]
    var h: Array[Int, 3] = [1, 2, 4]
    assert_true(g < h)
    assert_true(g <= h)
    assert_true(h > g)
    assert_true(h >= g)

    # Single-element arrays.
    var x: Array[Int, 1] = [1]
    var y: Array[Int, 1] = [2]
    assert_true(x < y)
    assert_true(y > x)

    # Ordering also works for non-`Int` `Comparable` elements.
    var s1: Array[String, 2] = ["apple", "zebra"]
    var s2: Array[String, 2] = ["banana", "aardvark"]
    assert_true(s1 < s2)
    assert_true(s2 > s1)

    var s3: Array[String, 2] = ["hello", "world"]
    var s4: Array[String, 2] = ["hello", "world"]
    assert_true(s3 <= s4)
    assert_true(s3 >= s4)
    assert_false(s3 < s4)


def test_array_hash() raises:
    var a: Array[Int, 3] = [1, 2, 3]
    var b: Array[Int, 3] = [1, 2, 3]
    var c: Array[Int, 3] = [1, 2, 4]

    # Equal arrays must have equal hashes.
    assert_equal(hash(a), hash(b))

    # Different arrays should (almost certainly) have different hashes.
    assert_true(hash(a) != hash(c))


def test_array_conditional_conformances() raises:
    assert_true(conforms_to(Array[Int, 3], Writable))
    assert_true(conforms_to(Array[Int, 3], Equatable))
    assert_true(conforms_to(Array[Int, 3], Comparable))
    # `NonWritable` is `Equatable`-less and not `Comparable`, so the array
    # tracks the element and does not conform to `Comparable`.
    assert_false(conforms_to(Array[NonWritable, 3], Comparable))
    assert_true(conforms_to(Array[Int, 3], Hashable))
    assert_true(conforms_to(Array[Int, 3], Copyable))
    assert_true(conforms_to(Array[Int, 3], Deinitable))
    # An array of explicitly-destroyed elements is not implicitly deletable.
    assert_false(conforms_to(Array[ExplicitDestroy, 3], Deinitable))
    assert_false(conforms_to(Array[NonWritable, 3], Writable))
    # Owned iteration requires `Movable & Deinitable` elements, but
    # not `Copyable`: a consuming iterator moves elements out rather than
    # copying them.
    assert_true(conforms_to(Array[Int, 3], IterableOwned))
    # `MoveOnly[Int]` is movable and implicitly deletable but not copyable.
    assert_true(conforms_to(Array[MoveOnly[Int], 2], IterableOwned))
    # `ExplicitDestroy` is not implicitly deletable, so the consuming iterator
    # cannot destroy any unconsumed elements.
    assert_false(conforms_to(Array[ExplicitDestroy, 3], IterableOwned))

    # The element bound is `AnyType`, so a non-`Movable` element type is
    # accepted, and the array's `Movable` conformance follows the element.
    assert_true(conforms_to(Array[Int, 3], Movable))
    assert_false(conforms_to(Array[NonMovable, 3], Movable))
    assert_true(conforms_to(Array[NonMovable, 3], Deinitable))
    # A fully linear element (neither `Movable` nor `Deinitable`)
    # yields an array that is likewise neither.
    assert_false(conforms_to(Array[LinearNonMovable, 3], Movable))
    assert_false(conforms_to(Array[LinearNonMovable, 3], Deinitable))

    assert_true(conforms_to(Array[Int, 3], Defaultable))
    assert_true(conforms_to(Array[DefaultableNonMovable, 3], Defaultable))
    assert_false(conforms_to(Array[NonDefaultable, 3], Defaultable))


def test_array_iter_bounds() raises:
    var arr: Array[Int, 3] = [1, 2, 3]
    _test_array_iter_bounds(iter(arr), len(arr))
    _test_array_iter_bounds(reversed(arr), len(arr))


def test_array_iter_owned() raises:
    var arr: Array[Int, 3] = [10, 20, 30]
    var result = List[Int]()
    for elem in arr^:
        result.append(elem)

    assert_equal(len(result), 3)
    assert_equal(result[0], 10)
    assert_equal(result[1], 20)
    assert_equal(result[2], 30)


def test_array_iter_owned_move_only() raises:
    # Consuming iteration only requires `Movable & Deinitable`, not
    # `Copyable`: each element is moved out of the array, not copied.
    var arr: Array[MoveOnly[Int], 3] = [
        MoveOnly[Int](0),
        MoveOnly[Int](1),
        MoveOnly[Int](2),
    ]

    var total = 0
    for elem in arr^:
        total += elem.data

    assert_equal(total, 3)


def test_array_iter_owned_destroys_elements_if_not_consumed() raises:
    # Verify that creating and immediately dropping the iterator doesn't crash.
    var arr: Array[Int, 3] = [1, 2, 3]
    var _ = arr^.__iter__()


def test_array_iter_owned_destroys_elements_if_partially_consumed() raises:
    # Verify partial consumption followed by dropping doesn't crash.
    var arr: Array[Int, 3] = [1, 2, 3]
    var it = arr^.__iter__()
    _ = it.__next__()  # consume one element
    _ = it^  # drop iterator with remaining elements


def test_array_iter_owned_bounds() raises:
    var arr: Array[Int, 3] = [1, 2, 3]
    var it = arr^.__iter__()
    assert_equal(it.bounds()[0], 3)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 2)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 1)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 0)


def test_array_move_only() raises:
    # `MoveOnly[Int]` is not `Copyable`; this exercises the conditional
    # conformance path of `Array[T: Movable, size]`.
    assert_false(conforms_to(Array[MoveOnly[Int], 2], Copyable))

    var arr: Array[MoveOnly[Int], 3] = [
        MoveOnly[Int](0),
        MoveOnly[Int](1),
        MoveOnly[Int](2),
    ]
    assert_equal(arr[0], MoveOnly[Int](0))
    assert_equal(arr[1], MoveOnly[Int](1))
    assert_equal(arr[2], MoveOnly[Int](2))
    assert_equal(len(arr), 3)

    # `unsafe_get` is a non-copying accessor.
    assert_equal(arr.unsafe_get(2), MoveOnly[Int](2))


def test_array_literal_size_inference() raises:
    # The array length is inferred from the element count of the literal.
    var arr: Array[Int, _] = [1, 2, 3]
    comptime assert type_of(arr).length == 3
    assert_equal(arr[0], 1)
    assert_equal(arr[1], 2)
    assert_equal(arr[2], 3)

    var single: Array[Int, _] = [42]
    comptime assert type_of(single).length == 1
    assert_equal(single[0], 42)

    var strings: Array[String, _] = ["hi", "hello"]
    comptime assert type_of(strings).length == 2
    assert_equal(strings[0], "hi")
    assert_equal(strings[1], "hello")

    var move_only: Array[MoveOnly[Int], _] = [
        MoveOnly[Int](0),
        MoveOnly[Int](1),
    ]
    comptime assert type_of(move_only).length == 2
    assert_equal(move_only[0], MoveOnly[Int](0))
    assert_equal(move_only[1], MoveOnly[Int](1))


def test_array_with_explicit_destroy_type() raises:
    var arr: Array[ExplicitDestroy, 3] = [
        ExplicitDestroy(0),
        ExplicitDestroy(1),
        ExplicitDestroy(2),
    ]

    var destroyed = List[Int]()

    def destroy_closure(var e: ExplicitDestroy) {mut}:
        destroyed.append(e.value)
        e^.destroy()

    arr^.deinit_with(destroy_closure)

    assert_equal(destroyed, [0, 1, 2])


def test_array_concat() raises:
    var a: Array[Int, 2] = [1, 2]
    var b: Array[Int, 3] = [3, 4, 5]
    var c = a^.concat(b^)
    comptime assert type_of(c).length == 5
    for i in range(5):
        assert_equal(c[i], i + 1)

    var empty = Array[Int, 0](uninitialized=True)
    var d: Array[Int, 1] = [7]
    var e = empty^.concat(d^)
    comptime assert type_of(e).length == 1
    assert_equal(e[0], 7)


def test_array_concat_list_literal_rhs() raises:
    # A list literal on the right-hand side materializes as an `Array`.
    var a: Array[Int, 2] = [1, 2]
    var b = a^.concat([3, 4])
    comptime assert type_of(b).length == 4
    for i in range(4):
        assert_equal(b[i], i + 1)


def test_array_concat_list_literal_to_rvalue() raises:
    def returns_array() -> Array[Int, 2]:
        return [1, 2]

    # An rvalue array concatenates with a literal directly.
    var a = returns_array().concat([3, 4])
    comptime assert type_of(a).length == 4
    for i in range(4):
        assert_equal(a[i], i + 1)


def test_array_concat_list_literal_lhs() raises:
    def returns_array() -> Array[Int, 2]:
        return [3, 4]

    # A literal on the left-hand side also resolves to `Array.concat`.
    var a = [1, 2].concat(returns_array())
    comptime assert type_of(a).length == 4
    for i in range(4):
        assert_equal(a[i], i + 1)


def test_array_concat_two_list_literals() raises:
    # Two literals with an `Array`-typed binding.
    var a: Array[Int, 4] = [1, 2].concat([3, 4])
    for i in range(4):
        assert_equal(a[i], i + 1)


def test_array_concat_string() raises:
    var a: Array[String, 2] = ["hi", "hello"]
    var b: Array[String, 1] = ["hey"]
    var c = a^.concat(b^)
    assert_equal(c[0], "hi")
    assert_equal(c[1], "hello")
    assert_equal(c[2], "hey")


def test_array_concat_runs_destructors_once() raises:
    var destructor_recorder = List[Int]()
    var ptr = Pointer(to=destructor_recorder).as_imm()
    var a: Array[DelRecorder[ptr.origin], 2] = [
        DelRecorder(0, ptr),
        DelRecorder(1, ptr),
    ]
    var b: Array[DelRecorder[ptr.origin], 2] = [
        DelRecorder(2, ptr),
        DelRecorder(3, ptr),
    ]
    var c = a^.concat(b^)
    # The inputs' elements were moved into `c`, not destroyed.
    assert_equal(len(destructor_recorder), 0)
    _ = c^
    assert_equal(destructor_recorder, [0, 1, 2, 3])


def test_array_concat_explicit_destroy_type() raises:
    var a: Array[ExplicitDestroy, 1] = [ExplicitDestroy(0)]
    var b: Array[ExplicitDestroy, 2] = [
        ExplicitDestroy(1),
        ExplicitDestroy(2),
    ]
    var c = a^.concat(b^)

    var destroyed = List[Int]()

    def destroy_closure(var e: ExplicitDestroy) {mut}:
        destroyed.append(e.value)
        e^.destroy()

    c^.deinit_with(destroy_closure)
    assert_equal(destroyed, [0, 1, 2])


def test_array_repeat() raises:
    var a: Array[Int, 2] = [1, 2]
    var b = a^.repeat[3]()
    comptime assert type_of(b).length == 6
    for i in range(6):
        assert_equal(b[i], i % 2 + 1)


def test_array_repeat_by_one() raises:
    var a: Array[Int, 3] = [1, 2, 3]
    var b = a^.repeat[1]()
    comptime assert type_of(b).length == 3
    for i in range(3):
        assert_equal(b[i], i + 1)


def test_array_repeat_rvalue() raises:
    def returns_array() -> Array[Int, 2]:
        return [1, 2]

    var a = returns_array().repeat[2]()
    comptime assert type_of(a).length == 4
    for i in range(4):
        assert_equal(a[i], i % 2 + 1)


def test_array_repeat_string() raises:
    var a: Array[String, 2] = ["hi", "hello"]
    var b = a^.repeat[2]()
    assert_equal(b[0], "hi")
    assert_equal(b[1], "hello")
    assert_equal(b[2], "hi")
    assert_equal(b[3], "hello")


def test_array_repeat_runs_destructors_once() raises:
    var destructor_recorder = List[Int]()
    var ptr = Pointer(to=destructor_recorder).as_imm()
    var a: Array[DelRecorder[ptr.origin], 2] = [
        DelRecorder(0, ptr),
        DelRecorder(1, ptr),
    ]
    var b = a^.repeat[3]()
    # The input's elements were copied/moved into `b`, not destroyed.
    assert_equal(len(destructor_recorder), 0)
    _ = b^
    assert_equal(destructor_recorder, [0, 1, 0, 1, 0, 1])


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
