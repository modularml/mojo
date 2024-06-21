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

from collections.set import Set

from testing import assert_false, assert_true


def test_list_any():
    # List[Int]
    assert_true(any(List(-1, 2)))
    assert_true(any(List(-0, 2, 3)))
    assert_true(any(List(-0, 0, 3)))
    assert_false(any(List(0, 0, 0, 0)))
    assert_false(any(List[Int]()))

    # List[Float]
    assert_true(any(List(-1.0, 2.0, 3.0)))
    assert_true(any(List(-1.0, 0.0, 3.0)))
    assert_true(any(List(-0.0, 2.0, 0.0)))
    assert_false(any(List(0.0, 0.0, 0.0)))
    assert_false(any(List[Float64]()))

    # List[Bool]
    assert_true(any(List(True)))
    assert_true(any(List(True, True)))
    assert_true(any(List(True, False)))
    assert_true(any(List(False, True)))
    assert_false(any(List(False, False)))
    assert_false(any(List(False)))
    assert_false(any(List[Bool]()))


def test_list_all():
    # List[Int]
    assert_true(all(List(-1, 2, 3)))
    assert_false(all(List(1, 2, 0)))
    assert_false(all(List(1, 0, 0)))
    assert_false(all(List(0, 0, 0)))
    assert_true(all(List[Int]()))

    # List[Float]
    assert_true(all(List(-1.0, 2.0, 3.0, 4.0)))
    assert_false(all(List(1.0, 0.0, 3.0)))
    assert_false(all(List(0.0, 2.0, 0.0)))
    assert_false(all(List(0.0, 0.0)))
    assert_true(all(List[Float64]()))

    # List[Bool]
    assert_true(all(List(True)))
    assert_true(all(List(True, True)))
    assert_false(all(List(True, False)))
    assert_false(all(List(False, True)))
    assert_false(all(List(False, False)))
    assert_false(all(List(False)))
    assert_true(all(List[Bool]()))


def test_set_any():
    # Set[Int]
    assert_true(any(Set(-1)))
    assert_true(any(Set(-1, 0, 3)))
    assert_false(any(Set(0)))
    assert_false(any(Set[Int]()))

    # Set[String]
    assert_true(any(Set[String]("any")))
    assert_true(any(Set[String]("bleep", "bloop")))
    assert_true(any(Set[String]("", ":]")))
    assert_false(any(Set[String]("")))
    assert_false(any(Set[String]()))


def test_set_all():
    # Set[Int]
    assert_true(all(Set(-1)))
    assert_false(all(Set(0, 1, 3)))
    assert_false(all(Set(0)))
    assert_true(all(Set[Int]()))

    # Set[String]
    assert_true(all(Set[String]("all")))
    assert_true(all(Set[String]("0", "1")))
    assert_false(all(Set[String]("mojo", "")))
    assert_false(all(Set[String]("")))
    assert_true(all(Set[String]()))


def test_simd_any():
    @parameter
    def _test_dtype[type: DType]():
        assert_true(any(SIMD[type, 1](1)))
        assert_false(any(SIMD[type, 1](0)))
        assert_true(any(SIMD[type, 4](1, 2, 3, 4)))
        assert_true(any(SIMD[type, 4](0, 2, 3, 4)))
        assert_true(any(SIMD[type, 4](1, 2, 3, 0)))
        assert_true(any(SIMD[type, 4](0, 2, 3, 0)))
        assert_true(any(SIMD[type, 4](1, 0, 0, 4)))
        assert_false(any(SIMD[type, 4](0, 0, 0, 0)))

    _test_dtype[DType.bool]()
    _test_dtype[DType.int8]()
    _test_dtype[DType.int16]()
    _test_dtype[DType.int32]()
    _test_dtype[DType.int64]()
    _test_dtype[DType.uint8]()
    _test_dtype[DType.uint16]()
    _test_dtype[DType.uint32]()
    _test_dtype[DType.uint64]()
    _test_dtype[DType.float16]()
    _test_dtype[DType.float32]()
    _test_dtype[DType.float64]()


def test_simd_all():
    @parameter
    def _test_dtype[type: DType]():
        assert_true(all(SIMD[type, 1](1)))
        assert_false(all(SIMD[type, 1](0)))
        assert_true(all(SIMD[type, 4](1, 2, 3, 4)))
        assert_false(all(SIMD[type, 4](0, 2, 3, 4)))
        assert_false(all(SIMD[type, 4](1, 2, 3, 0)))
        assert_false(all(SIMD[type, 4](0, 2, 3, 0)))
        assert_false(all(SIMD[type, 4](1, 0, 0, 4)))
        assert_false(all(SIMD[type, 4](0, 0, 0, 0)))

    _test_dtype[DType.bool]()
    _test_dtype[DType.int8]()
    _test_dtype[DType.int16]()
    _test_dtype[DType.int32]()
    _test_dtype[DType.int64]()
    _test_dtype[DType.uint8]()
    _test_dtype[DType.uint16]()
    _test_dtype[DType.uint32]()
    _test_dtype[DType.uint64]()
    _test_dtype[DType.float16]()
    _test_dtype[DType.float32]()
    _test_dtype[DType.float64]()


def main():
    # any
    test_list_any()
    test_set_any()
    test_simd_any()

    # all
    test_list_all()
    test_set_all()
    test_simd_all()
