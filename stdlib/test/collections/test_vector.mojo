# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full -I %kernels_test_root %s

from collections.vector import DynamicVector, InlinedFixedVector

from testing import *
from test_utils import MoveCounter


def test_inlined_fixed_vector():
    var vector = InlinedFixedVector[Int, 5](10)

    for i in range(5):
        vector.append(i)

    # Verify it's iterable
    var index = 0
    for element in vector:
        assert_equal(vector[index], element)
        index += 1

    assert_equal(5, len(vector))

    # Can assign a specified index in static data range via `setitem`
    vector[2] = -2
    assert_equal(0, vector[0])
    assert_equal(1, vector[1])
    assert_equal(-2, vector[2])
    assert_equal(3, vector[3])
    assert_equal(4, vector[4])

    assert_equal(3, vector[-2])
    assert_equal(4, vector[-1])

    # Can assign past the static size into the regrowable dynamic data portion
    for j in range(5, 10):
        vector.append(j)

    assert_equal(10, len(vector))

    # Verify the dynamic data got properly assigned to from above
    assert_equal(5, vector[5])
    assert_equal(6, vector[6])
    assert_equal(7, vector[7])
    assert_equal(8, vector[8])
    assert_equal(9, vector[9])

    assert_equal(9, vector[-1])

    # Assign a specified index in the dynamic_data portion
    vector[5] = -2
    assert_equal(-2, vector[5])

    vector.clear()
    assert_equal(0, len(vector))

    # Free the memory since we manage it ourselves in `InlinedFixedVector` for now.
    vector._del_old()


def test_inlined_fixed_vector_with_default():
    var vector = InlinedFixedVector[Int](10)

    for i in range(5):
        vector.append(i)

    assert_equal(5, len(vector))

    vector[2] = -2

    assert_equal(0, vector[0])
    assert_equal(1, vector[1])
    assert_equal(-2, vector[2])
    assert_equal(3, vector[3])
    assert_equal(4, vector[4])

    for j in range(5, 10):
        vector.append(j)

    assert_equal(10, len(vector))

    assert_equal(5, vector[5])

    vector[5] = -2
    assert_equal(-2, vector[5])

    vector.clear()
    assert_equal(0, len(vector))

    vector._del_old()


def test_mojo_issue_698():
    var vector = DynamicVector[Float64]()
    for i in range(5):
        vector.push_back(i)

    assert_equal(0.0, vector[0])
    assert_equal(1.0, vector[1])
    assert_equal(2.0, vector[2])
    assert_equal(3.0, vector[3])
    assert_equal(4.0, vector[4])


def test_vector():
    var vector = DynamicVector[Int]()

    for i in range(5):
        vector.push_back(i)

    assert_equal(5, len(vector))
    assert_equal(0, vector[0])
    assert_equal(1, vector[1])
    assert_equal(2, vector[2])
    assert_equal(3, vector[3])
    assert_equal(4, vector[4])

    assert_equal(0, vector[-5])
    assert_equal(3, vector[-2])
    assert_equal(4, vector[-1])

    vector[2] = -2
    assert_equal(-2, vector[2])

    # pop_back shall return the last element
    # and adjust the size
    assert_equal(4, vector.pop_back())
    assert_equal(4, len(vector))

    # Verify that capacity shrinks as the vector goes smaller
    while vector.size > 1:
        _ = vector.pop_back()

    assert_equal(1, len(vector))
    assert_equal(
        1, vector.size
    )  # pedantically ensure len and size refer to the same thing
    assert_equal(4, vector.capacity)

    # Verify that capacity doesn't become 0 when the vector gets empty.
    _ = vector.pop_back()
    assert_equal(0, len(vector))

    # FIXME: revisit that pop_back is actually doing shrink_to_fit behavior
    # under the hood which will be surprising to users
    assert_equal(2, vector.capacity)

    vector.clear()
    assert_equal(0, len(vector))
    assert_equal(2, vector.capacity)


def test_vector_reverse():
    #
    # Test reversing the vector []
    #

    var vec = DynamicVector[Int]()

    assert_equal(len(vec), 0)

    vec.reverse()

    assert_equal(len(vec), 0)

    #
    # Test reversing the vector [123]
    #

    vec = DynamicVector[Int]()

    vec.push_back(123)

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    vec.reverse()

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    #
    # Test reversing the vector ["one", "two", "three"]
    #

    vec2 = DynamicVector[String]()
    vec2.push_back("one")
    vec2.push_back("two")
    vec2.push_back("three")

    assert_equal(len(vec2), 3)
    assert_equal(vec2[0], "one")
    assert_equal(vec2[1], "two")
    assert_equal(vec2[2], "three")

    vec2.reverse()

    assert_equal(len(vec2), 3)
    assert_equal(vec2[0], "three")
    assert_equal(vec2[1], "two")
    assert_equal(vec2[2], "one")

    #
    # Test reversing the vector [5, 10]
    #

    vec = DynamicVector[Int]()
    vec.push_back(5)
    vec.push_back(10)

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 5)
    assert_equal(vec[1], 10)

    vec.reverse()

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 10)
    assert_equal(vec[1], 5)

    #
    # Test reversing the vector [1, 2, 3, 4, 5] starting at the 3rd position
    # to produce [1, 2, 5, 4, 3]
    #

    vec = DynamicVector[Int]()
    vec.push_back(1)
    vec.push_back(2)
    vec.push_back(3)
    vec.push_back(4)
    vec.push_back(5)

    assert_equal(len(vec), 5)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)
    assert_equal(vec[3], 4)
    assert_equal(vec[4], 5)

    vec._reverse(start=2)

    assert_equal(len(vec), 5)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 5)
    assert_equal(vec[3], 4)
    assert_equal(vec[4], 3)

    #
    # Test edge case of reversing the vector [1, 2, 3] but starting after the
    # last element.
    #

    vec = DynamicVector[Int]()
    vec.push_back(1)
    vec.push_back(2)
    vec.push_back(3)

    vec._reverse(start=len(vec))

    assert_equal(len(vec), 3)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)


def test_vector_reverse_move_count():
    # Create this vec with enough capacity to avoid moves due to resizing.
    var vec = DynamicVector[MoveCounter[Int]](capacity=5)
    vec.push_back(MoveCounter(1))
    vec.push_back(MoveCounter(2))
    vec.push_back(MoveCounter(3))
    vec.push_back(MoveCounter(4))
    vec.push_back(MoveCounter(5))

    assert_equal(len(vec), 5)
    assert_equal(__get_address_as_lvalue((vec.data + 0).value).value, 1)
    assert_equal(__get_address_as_lvalue((vec.data + 1).value).value, 2)
    assert_equal(__get_address_as_lvalue((vec.data + 2).value).value, 3)
    assert_equal(__get_address_as_lvalue((vec.data + 3).value).value, 4)
    assert_equal(__get_address_as_lvalue((vec.data + 4).value).value, 5)

    assert_equal(__get_address_as_lvalue((vec.data + 0).value).move_count, 1)
    assert_equal(__get_address_as_lvalue((vec.data + 1).value).move_count, 1)
    assert_equal(__get_address_as_lvalue((vec.data + 2).value).move_count, 1)
    assert_equal(__get_address_as_lvalue((vec.data + 3).value).move_count, 1)
    assert_equal(__get_address_as_lvalue((vec.data + 4).value).move_count, 1)

    vec.reverse()

    assert_equal(len(vec), 5)
    assert_equal(__get_address_as_lvalue((vec.data + 0).value).value, 5)
    assert_equal(__get_address_as_lvalue((vec.data + 1).value).value, 4)
    assert_equal(__get_address_as_lvalue((vec.data + 2).value).value, 3)
    assert_equal(__get_address_as_lvalue((vec.data + 3).value).value, 2)
    assert_equal(__get_address_as_lvalue((vec.data + 4).value).value, 1)

    # NOTE:
    # Earlier elements went through 2 moves and later elements went through 3
    # moves because the implementation of DynamicVector.reverse arbitrarily
    # chooses to perform the swap of earlier and later elements by moving the
    # earlier element to a temporary (+1 move), directly move the later element
    # into the position the earlier element was in, and then move from the
    # temporary into the later position (+1 move).
    assert_equal(__get_address_as_lvalue((vec.data + 0).value).move_count, 2)
    assert_equal(__get_address_as_lvalue((vec.data + 1).value).move_count, 2)
    assert_equal(__get_address_as_lvalue((vec.data + 2).value).move_count, 1)
    assert_equal(__get_address_as_lvalue((vec.data + 3).value).move_count, 3)
    assert_equal(__get_address_as_lvalue((vec.data + 4).value).move_count, 3)

    # Keep vec alive until after we've done the last `vec.data + N` read.
    _ = vec ^


def test_vector_extend():
    #
    # Test extending the vector [1, 2, 3] with itself
    #

    vec = DynamicVector[Int]()
    vec.push_back(1)
    vec.push_back(2)
    vec.push_back(3)

    assert_equal(len(vec), 3)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)

    var copy = vec
    vec.extend(copy)

    # vec == [1, 2, 3, 1, 2, 3]
    assert_equal(len(vec), 6)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)
    assert_equal(vec[3], 1)
    assert_equal(vec[4], 2)
    assert_equal(vec[5], 3)

    vec._reverse(start=3)

    # vec == [1, 2, 3, 3, 2, 1]
    assert_equal(len(vec), 6)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)
    assert_equal(vec[3], 3)
    assert_equal(vec[4], 2)
    assert_equal(vec[5], 1)


def test_vector_extend_non_trivial():
    # Tests three things:
    #   - extend() for non-plain-old-data types
    #   - extend() with mixed-length self and other vectors
    #   - extend() using optimal number of __moveinit__() calls

    # Preallocate with enough capacity to avoid reallocation making the
    # move count checks below flaky.
    var v1 = DynamicVector[MoveCounter[String]](capacity=5)
    v1.push_back(MoveCounter[String]("Hello"))
    v1.push_back(MoveCounter[String]("World"))

    var v2 = DynamicVector[MoveCounter[String]](capacity=3)
    v2.push_back(MoveCounter[String]("Foo"))
    v2.push_back(MoveCounter[String]("Bar"))
    v2.push_back(MoveCounter[String]("Baz"))

    v1.extend(v2)

    assert_equal(len(v1), 5)
    assert_equal(v1[0].value, "Hello")
    assert_equal(v1[1].value, "World")
    assert_equal(v1[2].value, "Foo")
    assert_equal(v1[3].value, "Bar")
    assert_equal(v1[4].value, "Baz")

    assert_equal(__get_address_as_lvalue((v1.data + 0).value).move_count, 1)
    assert_equal(__get_address_as_lvalue((v1.data + 1).value).move_count, 1)
    assert_equal(__get_address_as_lvalue((v1.data + 2).value).move_count, 2)
    assert_equal(__get_address_as_lvalue((v1.data + 3).value).move_count, 2)
    assert_equal(__get_address_as_lvalue((v1.data + 4).value).move_count, 2)

    # Keep v1 alive until after we've done the last `vec.data + N` read.
    _ = v1 ^


def test_2d_dynamic_vector():
    var vector = DynamicVector[DynamicVector[Int]]()

    for i in range(2):
        var v = DynamicVector[Int]()
        for j in range(3):
            v.push_back(i + j)
        vector.push_back(v)

    assert_equal(0, vector[0][0])
    assert_equal(1, vector[0][1])
    assert_equal(2, vector[0][2])
    assert_equal(1, vector[1][0])
    assert_equal(2, vector[1][1])
    assert_equal(3, vector[1][2])

    assert_equal(2, len(vector))
    assert_equal(2, vector.capacity)

    assert_equal(3, len(vector[0]))

    vector[0].clear()
    assert_equal(0, len(vector[0]))
    assert_equal(4, vector[0].capacity)

    vector.clear()
    assert_equal(0, len(vector))
    assert_equal(2, vector.capacity)


# Ensure correct behavior of __copyinit__
# as reported in GH issue 27875 internally and
# https://github.com/modularml/mojo/issues/1493
def test_vector_copy_constructor():
    var vec = DynamicVector[Int](capacity=1)
    var vec_copy = vec
    vec_copy.push_back(1)  # Ensure copy constructor doesn't crash
    _ = vec ^  # To ensure previous one doesn't invoke move constuctor


def test_vector_iter():
    var vs = DynamicVector[Int]()
    vs.append(1)
    vs.append(2)
    vs.append(3)

    # Borrow immutably
    fn sum(vs: DynamicVector[Int]) -> Int:
        var sum = 0
        for v in vs:
            sum += v[]
        return sum

    assert_equal(6, sum(vs))


def test_vector_iter_mutable():
    var vs = DynamicVector[Int]()
    vs.append(1)
    vs.append(2)
    vs.append(3)

    for v in vs:
        v[] += 1

    var sum = 0
    for v in vs:
        sum += v[]

    assert_equal(9, sum)


def main():
    test_inlined_fixed_vector()
    test_inlined_fixed_vector_with_default()
    test_mojo_issue_698()
    test_vector()
    test_vector_reverse()
    test_vector_reverse_move_count()
    test_vector_extend()
    test_vector_extend_non_trivial()
    test_vector_copy_constructor()
    test_2d_dynamic_vector()
    test_vector_iter()
    test_vector_iter_mutable()
