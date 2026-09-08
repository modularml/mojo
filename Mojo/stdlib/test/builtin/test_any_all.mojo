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

from std.collections.set import Set

from std.testing import assert_false, assert_true, TestSuite


def test_list_any() raises:
    # List[Int]
    assert_true(any([-1, 2]))
    assert_true(any([-0, 2, 3]))
    assert_true(any([-0, 0, 3]))
    assert_false(any([0, 0, 0, 0]))
    assert_false(any(List[Int]()))

    # List[Float]
    assert_true(any([-1.0, 2.0, 3.0]))
    assert_true(any([-1.0, 0.0, 3.0]))
    assert_true(any([-0.0, 2.0, 0.0]))
    assert_false(any([0.0, 0.0, 0.0]))
    assert_false(any(List[Float64]()))

    # List[Bool]
    assert_true(any([True]))
    assert_true(any([True, True]))
    assert_true(any([True, False]))
    assert_true(any([False, True]))
    assert_false(any([False, False]))
    assert_false(any([False]))
    assert_false(any(List[Bool]()))


def test_list_all() raises:
    # List[Int]
    assert_true(all([-1, 2, 3]))
    assert_false(all([1, 2, 0]))
    assert_false(all([1, 0, 0]))
    assert_false(all([0, 0, 0]))
    assert_true(all(List[Int]()))

    # List[Float]
    assert_true(all([-1.0, 2.0, 3.0, 4.0]))
    assert_false(all([1.0, 0.0, 3.0]))
    assert_false(all([0.0, 2.0, 0.0]))
    assert_false(all([0.0, 0.0]))
    assert_true(all(List[Float64]()))

    # List[Bool]
    assert_true(all([True]))
    assert_true(all([True, True]))
    assert_false(all([True, False]))
    assert_false(all([False, True]))
    assert_false(all([False, False]))
    assert_false(all([False]))
    assert_true(all(List[Bool]()))


def test_set_any() raises:
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


def test_set_all() raises:
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


def test_simd_any() raises:
    def _test_dtype[dtype: DType]() raises:
        assert_true(any(Scalar[dtype](1)))
        assert_false(any(Scalar[dtype](0)))
        assert_true(any(SIMD[dtype, 4](1, 2, 3, 4)))
        assert_true(any(SIMD[dtype, 4](0, 2, 3, 4)))
        assert_true(any(SIMD[dtype, 4](1, 2, 3, 0)))
        assert_true(any(SIMD[dtype, 4](0, 2, 3, 0)))
        assert_true(any(SIMD[dtype, 4](1, 0, 0, 4)))
        assert_false(any(SIMD[dtype, 4](0, 0, 0, 0)))

    _test_dtype[.bool]()
    _test_dtype[.int8]()
    _test_dtype[.int16]()
    _test_dtype[.int32]()
    _test_dtype[.int64]()
    _test_dtype[.uint8]()
    _test_dtype[.uint16]()
    _test_dtype[.uint32]()
    _test_dtype[.uint64]()
    _test_dtype[.float16]()
    _test_dtype[.float32]()
    _test_dtype[.float64]()


def test_simd_all() raises:
    def _test_dtype[dtype: DType]() raises:
        assert_true(all(Scalar[dtype](1)))
        assert_false(all(Scalar[dtype](0)))
        assert_true(all(SIMD[dtype, 4](1, 2, 3, 4)))
        assert_false(all(SIMD[dtype, 4](0, 2, 3, 4)))
        assert_false(all(SIMD[dtype, 4](1, 2, 3, 0)))
        assert_false(all(SIMD[dtype, 4](0, 2, 3, 0)))
        assert_false(all(SIMD[dtype, 4](1, 0, 0, 4)))
        assert_false(all(SIMD[dtype, 4](0, 0, 0, 0)))

    _test_dtype[.bool]()
    _test_dtype[.int8]()
    _test_dtype[.int16]()
    _test_dtype[.int32]()
    _test_dtype[.int64]()
    _test_dtype[.uint8]()
    _test_dtype[.uint16]()
    _test_dtype[.uint32]()
    _test_dtype[.uint64]()
    _test_dtype[.float16]()
    _test_dtype[.float32]()
    _test_dtype[.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
