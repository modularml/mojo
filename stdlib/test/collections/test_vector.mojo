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

from collections.vector import InlinedFixedVector

from test_utils import MoveCounter
from memory import UnsafePointer
from testing import assert_equal


def test_inlined_fixed_vector_moves():
    var v1 = InlinedFixedVector[Int, 5](10)
    var v2 = InlinedFixedVector[Int, 5](10)

    # do one within the smallvec
    v2[3] = 99
    v1[3] = 42

    # plus one within the dynarray
    v2[7] = 9999
    v1[7] = 4242
    v2 = v1^  # moves

    assert_equal(v2[3], 42)
    assert_equal(v2[7], 4242)


def test_inlined_fixed_vector():
    var vector = InlinedFixedVector[Int, 5](10)

    for i in range(5):
        vector.append(i)

    # Verify it's iterable
    var index = 0
    for element in vector:
        assert_equal(vector[index], element[])
        index += 1

    assert_equal(5, len(vector))

    # Can assign a specified index in static data range via `setitem`
    vector[2] = -2
    assert_equal(0, vector[0])
    assert_equal(1, vector[1])
    assert_equal(-2, vector[2])
    assert_equal(3, vector[3])
    assert_equal(4, vector[4])

    assert_equal(0, vector[-5])
    assert_equal(3, vector[-2])
    assert_equal(4, vector[-1])

    vector[-5] = 5
    assert_equal(5, vector[-5])
    vector[-2] = 3
    assert_equal(3, vector[-2])
    vector[-1] = 7
    assert_equal(7, vector[-1])

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


def test_indexing():
    var vector = InlinedFixedVector[Int](10)
    for i in range(5):
        vector.append(i)
    assert_equal(0, vector[int(0)])
    assert_equal(1, vector[True])
    assert_equal(2, vector[2])


@value
struct MyStruct:
    var val: Int
    var is_copy: Bool
    var ondelete: UnsafePointer[Int]

    fn __copyinit__(inout self, other: Self):
        self.val = other.val
        self.is_copy = True
        self.ondelete = other.ondelete

    fn __del__(owned self):
        self.ondelete[] += 1


def test_collection_elements():
    # TODO: Parametrize the test into various InlinedFixedVector.size

    del_done = 0
    del_done_ptr = UnsafePointer.address_of(del_done)

    var vector = InlinedFixedVector[MyStruct, 16](1000)

    # assert del
    for i in range(1000):
        vector.append(MyStruct(i, False, del_done_ptr)^)
    assert_equal(len(vector), 1000)
    __type_of(vector).__del__(vector^)
    assert_equal(del_done, 1000)

    # assert copy and del
    del_done = 0
    vector = InlinedFixedVector[MyStruct, 16](1000)
    for i in range(1000):
        vector.append(MyStruct(i, False, del_done_ptr)^)
        assert_equal(vector[i].val, i)
        assert_equal(vector[i].is_copy, False)
        var cpy = vector[i]
        assert_equal(cpy.is_copy, True)
    assert_equal(del_done, 2000)

    # test move
    del_done = 0
    vector = InlinedFixedVector[MyStruct, 16](1000)
    for i in range(1000):
        vector.append(MyStruct(i, False, del_done_ptr)^)
    vector2 = vector^
    for i in range(1000):
        assert_equal(vector2[i].val, i)
    assert_equal(del_done, 1000)

    # test move no dynamic
    del_done = 0
    vector3 = InlinedFixedVector[MyStruct, 1000](1000)
    assert_equal(vector3.static_size, 1000)
    for i in range(1000):
        vector3.append(MyStruct(i, False, del_done_ptr)^)
    vector4 = vector3^
    assert_equal(del_done, 1000)

    # assert copy vector
    vector = InlinedFixedVector[MyStruct, 16](1000)
    del_done = 0
    for i in range(1000):
        vector.append(MyStruct(i*10, False, del_done_ptr)^)
    assert_equal(del_done, 0)
    vector_cpy = vector
    assert_equal(len(vector_cpy), 1000)
    assert_equal(len(vector), 1000)
    for i in range(len(vector)):
        assert_equal(vector[i].val, vector_cpy[i].val)
    assert_equal(del_done, 2000)
    
    # test iteration
    del_done = 0
    vector5 = InlinedFixedVector[MyStruct, 500](1000)
    assert_equal(vector5.static_size, 500)
    for i in range(1000):
        vector5.append(MyStruct(i, False, del_done_ptr)^)
    counter = 0
    for i in vector5:
        assert_equal(i[].val, counter)
        counter += 1
    assert_equal(del_done, 1000)

    # test mutate
    del_done = 0
    vector5 = InlinedFixedVector[MyStruct, 500](1000)
    for i in range(1000):
        vector5.append(MyStruct(i, False, del_done_ptr)^)
    assert_equal(len(vector5), 1000)

    for i in range(1000):
        vector5[i].val *= 2
        assert_equal(vector5[i].val, i * 2)

    assert_equal(del_done, 1000)

    # test clear
    del_done = 0
    vector5 = InlinedFixedVector[MyStruct, 500](1000)
    for i in range(1000):
        vector5.append(MyStruct(i, False, del_done_ptr)^)

    vector5.clear()
    assert_equal(del_done, 1000)

    # test mutable iteration
    del_done = 0
    vector5 = InlinedFixedVector[MyStruct, 500](1000)
    for i in range(1000):
        vector5.append(MyStruct(i, False, del_done_ptr)^)

    vector5_iterator = vector5.__iter__()
    while len(vector5_iterator):
        vector5_iterator.__next__()[].val *= 2

    counter = 0
    vector5_iterator = vector5.__iter__()
    while len(vector5_iterator):
        assert_equal(vector5_iterator.__next__()[].val, counter * 2)
        counter += 1
    assert_equal(counter, 1000)
    assert_equal(del_done, 1000)

    # test mutable iteration
    del_done = 0
    vector5 = InlinedFixedVector[MyStruct, 500](1000)
    for i in range(1000):
        vector5.append(MyStruct(i, False, del_done_ptr)^)

    counter = 0
    for e in vector5:
        e[].val *= 2

    counter = 0
    for e in vector5:
        assert_equal(e[].val, counter * 2)
        counter += 1

    assert_equal(del_done, 1000)

    # test __getitem__ with (index < 0)
    vector6 = InlinedFixedVector[Int, 16](16)

    vector6.append(123)
    assert_equal(vector6[-1], 123)
    vector6[-1] = 0
    assert_equal(vector6[-1], 0)

    for i in range(15):
        vector6.append(i + 1)
    assert_equal(vector6[-8], 8)


def main():
    test_inlined_fixed_vector()
    test_inlined_fixed_vector_with_default()
    test_indexing()
    test_collection_elements()
