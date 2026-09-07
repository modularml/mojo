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

from std.atomic import Atomic, Ordering, fence

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)


def test_ordering_equality_comparable() raises:
    var ordering = Ordering.SEQUENTIAL

    assert_not_equal(ordering, Ordering(42))
    assert_not_equal(ordering, Ordering.NOT_ATOMIC)
    assert_not_equal(ordering, Ordering.UNORDERED)
    assert_not_equal(ordering, Ordering.RELAXED)
    assert_not_equal(ordering, Ordering.ACQUIRE)
    assert_not_equal(ordering, Ordering.RELEASE)
    assert_not_equal(ordering, Ordering.ACQUIRE_RELEASE)
    assert_equal(ordering, Ordering.SEQUENTIAL)


def test_ordering_representable() raises:
    assert_equal(repr(Ordering(42)), "Ordering.UNKNOWN")
    assert_equal(repr(Ordering.NOT_ATOMIC), "Ordering.NOT_ATOMIC")
    assert_equal(repr(Ordering.UNORDERED), "Ordering.UNORDERED")
    assert_equal(repr(Ordering.RELAXED), "Ordering.RELAXED")
    assert_equal(repr(Ordering.ACQUIRE), "Ordering.ACQUIRE")
    assert_equal(repr(Ordering.RELEASE), "Ordering.RELEASE")
    assert_equal(repr(Ordering.ACQUIRE_RELEASE), "Ordering.ACQUIRE_RELEASE")
    assert_equal(repr(Ordering.SEQUENTIAL), "Ordering.SEQUENTIAL")


def test_ordering_stringable() raises:
    assert_equal(String(Ordering(42)), "UNKNOWN")
    assert_equal(String(Ordering.NOT_ATOMIC), "NOT_ATOMIC")
    assert_equal(String(Ordering.UNORDERED), "UNORDERED")
    assert_equal(String(Ordering.RELAXED), "RELAXED")
    assert_equal(String(Ordering.ACQUIRE), "ACQUIRE")
    assert_equal(String(Ordering.RELEASE), "RELEASE")
    assert_equal(String(Ordering.ACQUIRE_RELEASE), "ACQUIRE_RELEASE")
    assert_equal(String(Ordering.SEQUENTIAL), "SEQUENTIAL")


def _test_atomic[dtype: DType]() raises:
    comptime scalar = Scalar[dtype]
    var atom = Atomic[scalar](3)

    assert_equal(atom.load(), scalar(3))

    assert_equal(atom._value, scalar(3))

    atom += scalar(4)
    assert_equal(atom._value, scalar(7))

    atom -= scalar(4)
    assert_equal(atom._value, scalar(3))

    atom.max(scalar(0))
    assert_equal(atom._value, scalar(3))

    atom.max(scalar(42))
    assert_equal(atom._value, scalar(42))

    atom.min(scalar(3))
    assert_equal(atom._value, scalar(3))

    atom.min(scalar(0))
    assert_equal(atom._value, scalar(0))

    atom.store(scalar(99))
    assert_equal(atom.load(), scalar(99))


def test_atomic() raises:
    _test_atomic[.int32]()
    _test_atomic[.float64]()


def _test_compare_exchange[dtype: DType]() raises:
    comptime scalar = Scalar[dtype]
    var atom = Atomic[scalar](3)

    # Successful cmpxchg
    var expected = scalar(3)
    var success = atom.compare_exchange(expected, scalar(42))

    assert_true(success)
    assert_equal(expected, scalar(3))
    assert_equal(atom.load(), scalar(42))

    # Failure cmpxchg
    expected = scalar(3)
    var failure = atom.compare_exchange(expected, scalar(99))

    assert_false(failure)
    assert_equal(expected, scalar(42))
    assert_equal(atom.load(), scalar(42))


def test_compare_exchange() raises:
    _test_compare_exchange[.int32]()
    _test_compare_exchange[.float64]()


def test_comptime_atomic() raises:
    def comptime_fn() -> Int:
        var atom = Atomic[Int](3)
        atom += 4
        atom -= 4
        return Int(atom.load())

    comptime value = comptime_fn()
    assert_equal(value, 3)


def test_comptime_fence() raises:
    def comptime_fn() -> Int:
        fence()
        return 1

    comptime value = comptime_fn()
    assert_equal(value, 1)


def test_comptime_compare_exchange() raises:
    def comptime_fn(expected_in: Int32) -> Tuple[Bool, Int32, Int32]:
        var expected = expected_in
        var atom = Atomic[Int32](0)
        var success = atom.compare_exchange(expected, 42)
        return (success, expected, atom.load())

    comptime result_success = comptime_fn(0)
    assert_true(result_success[0])
    assert_equal(result_success[1], 0)
    assert_equal(result_success[2], 42)

    comptime result_failure = comptime_fn(1)
    assert_false(result_failure[0])
    assert_equal(result_failure[1], 0)
    assert_equal(result_failure[2], 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
