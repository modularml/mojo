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
# RUN: %bare-mojo -D ASSERT=warn %s | FileCheck %s

from collections._index_normalization import normalize_index

from testing import assert_equal


def test_out_of_bounds_message():
    l = List[Int](1, 2)
    # CHECK: index out of bounds: 2
    _ = normalize_index["List"](2, l)
    # CHECK: index out of bounds: -3
    _ = normalize_index["List"](-3, l)

    l2 = List[Int]()
    # CHECK: indexing into a List that has 0 elements
    _ = normalize_index["List"](2, l2)


def test_normalize_index():
    alias t = "TestContainer"
    container = List[Int](1, 1, 1, 1)
    # test no cap
    alias no_cap = normalize_index[t, cap_to_container_length=False]
    assert_equal(no_cap(-4, container), 0)
    assert_equal(no_cap(-3, container), 1)
    assert_equal(no_cap(-2, container), 2)
    assert_equal(no_cap(-1, container), 3)
    assert_equal(no_cap(0, container), 0)
    assert_equal(no_cap(1, container), 1)
    assert_equal(no_cap(2, container), 2)
    assert_equal(no_cap(3, container), 3)
    # test cap to container length
    assert_equal(normalize_index[t](-4, container), 0)
    assert_equal(normalize_index[t](-3, container), 1)
    assert_equal(normalize_index[t](-2, container), 2)
    assert_equal(normalize_index[t](-1, container), 3)
    assert_equal(normalize_index[t](0, container), 0)
    assert_equal(normalize_index[t](1, container), 1)
    assert_equal(normalize_index[t](2, container), 2)
    assert_equal(normalize_index[t](3, container), 3)
    # test cap to container length overflow
    assert_equal(normalize_index[t](-8, container), 0)
    assert_equal(normalize_index[t](-7, container), 0)
    assert_equal(normalize_index[t](-6, container), 0)
    assert_equal(normalize_index[t](-5, container), 0)
    assert_equal(normalize_index[t](4, container), 3)
    assert_equal(normalize_index[t](5, container), 3)
    assert_equal(normalize_index[t](6, container), 3)
    assert_equal(normalize_index[t](7, container), 3)
    # test container with zero length
    container = List[Int]()
    assert_equal(normalize_index[t, ignore_zero_length=True](-8, container), 0)
    assert_equal(normalize_index[t, ignore_zero_length=True](-7, container), 0)
    assert_equal(normalize_index[t, ignore_zero_length=True](-6, container), 0)
    assert_equal(normalize_index[t, ignore_zero_length=True](-5, container), 0)
    assert_equal(normalize_index[t, ignore_zero_length=True](4, container), 0)
    assert_equal(normalize_index[t, ignore_zero_length=True](5, container), 0)
    assert_equal(normalize_index[t, ignore_zero_length=True](6, container), 0)
    assert_equal(normalize_index[t, ignore_zero_length=True](7, container), 0)
    # test container with zero length no cap
    alias ign_zero_no_cap = normalize_index[
        t, ignore_zero_length=True, cap_to_container_length=False
    ]
    assert_equal(ign_zero_no_cap(-8, container), 0)
    assert_equal(ign_zero_no_cap(-7, container), 0)
    assert_equal(ign_zero_no_cap(-6, container), 0)
    assert_equal(ign_zero_no_cap(-5, container), 0)
    assert_equal(ign_zero_no_cap(-4, container), 0)
    assert_equal(ign_zero_no_cap(-3, container), 0)
    assert_equal(ign_zero_no_cap(-2, container), 0)
    assert_equal(ign_zero_no_cap(-1, container), 0)
    assert_equal(ign_zero_no_cap(0, container), 0)
    assert_equal(ign_zero_no_cap(1, container), 0)
    assert_equal(ign_zero_no_cap(2, container), 0)
    assert_equal(ign_zero_no_cap(3, container), 0)
    assert_equal(ign_zero_no_cap(4, container), 0)
    assert_equal(ign_zero_no_cap(5, container), 0)
    assert_equal(ign_zero_no_cap(6, container), 0)
    assert_equal(ign_zero_no_cap(7, container), 0)


def main():
    test_out_of_bounds_message()
    test_normalize_index()
