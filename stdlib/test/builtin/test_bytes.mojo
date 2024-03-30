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
# RUN: %mojo -debug-level full %s

from testing import assert_equal


def test_int_constructor():
    some_bytes = Bytes(10)

    assert_equal(len(some_bytes), 10)

    for i in range(10):
        assert_equal(some_bytes[i], 0)

    assert_equal(some_bytes._data.capacity, 10)


def test_int_constructor_with_capacity():
    some_bytes = Bytes(10, capacity=20)

    assert_equal(len(some_bytes), 10)

    for i in range(10):
        assert_equal(some_bytes[i], 0)

    assert_equal(some_bytes._data.capacity, 20)


def test_list_constructor():
    some_bytes = Bytes(List[UInt8](10, 20, 30))

    assert_equal(len(some_bytes), 3)

    assert_equal(some_bytes[0], 10)
    assert_equal(some_bytes[1], 20)
    assert_equal(some_bytes[2], 30)


def test_bytes_setitem():
    some_bytes = Bytes(5)

    some_bytes[0] = 100
    some_bytes[2] = 102
    some_bytes[4] = 104

    assert_equal(some_bytes[0], 100)
    assert_equal(some_bytes[1], 0)
    assert_equal(some_bytes[2], 102)
    assert_equal(some_bytes[3], 0)
    assert_equal(some_bytes[4], 104)


def main():
    test_int_constructor()
    test_int_constructor_with_capacity()
    test_list_constructor()
    test_bytes_setitem()
