# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from collections.vector import DynamicVector
from collections.vector import InlinedFixedVector

from testing import *


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
    let vec = DynamicVector[Int](1)
    var vec_copy = vec
    vec_copy.push_back(1)  # Ensure copy constructor doesn't crash
    _ = vec ^  # To ensure previous one doesn't invoke move constuctor


def main():
    test_inlined_fixed_vector()
    test_inlined_fixed_vector_with_default()
    test_mojo_issue_698()
    test_vector()
    test_vector_copy_constructor()
    test_2d_dynamic_vector()
