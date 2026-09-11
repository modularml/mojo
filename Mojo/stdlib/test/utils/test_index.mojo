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

from std.testing import TestSuite, assert_equal

from test_utils import check_write_to
from std.utils import Index, IndexList
from std.utils.index import product


def test_basics() raises:
    assert_equal(IndexList[2](1, 2), IndexList[2](1, 2))
    assert_equal(IndexList[3](1, 2, 3), IndexList[3](1, 2, 3))
    assert_equal(String(IndexList[3](1, 2, 3)), "(1, 2, 3)")
    assert_equal(IndexList[3](1, 2, 3)[2], 3)


def test_cast() raises:
    assert_equal(
        String(IndexList[1](1)),
        "(1,)",
    )
    assert_equal(
        String(IndexList[2](1, 2).cast[.int32]()),
        "(1, 2)",
    )
    assert_equal(
        String(IndexList[2, element_type=DType.int32](1, 2)),
        "(1, 2)",
    )
    assert_equal(
        String(IndexList[2, element_type=DType.int64](1, 2)),
        "(1, 2)",
    )
    assert_equal(
        String(IndexList[2, element_type=DType.int32](1, -2).cast[.int64]()),
        "(1, -2)",
    )
    assert_equal(
        String(IndexList[2, element_type=DType.int32](1, 2)),
        "(1, 2)",
    )
    comptime s = String(
        IndexList[2, element_type=DType.int32](1, 2).cast[.int64]()
    )
    assert_equal(s, "(1, 2)")


def test_index() raises:
    # 0-D
    assert_equal(String(Index()), "()")

    # 1-D
    assert_equal(String(Index(42)), "(42,)")

    # 2-D
    assert_equal(String(Index(1, 2)), "(1, 2)")

    # 3-D with explicit dtype
    assert_equal(String(Index[dtype=DType.int64](1, 2, 3)), "(1, 2, 3)")
    assert_equal(String(Index[dtype=DType.int32](1, 2, 3)), "(1, 2, 3)")
    assert_equal(String(Index[dtype=DType.uint32](1, 2, 3)), "(1, 2, 3)")

    # 4-D
    assert_equal(String(Index(1, 2, 3, 4)), "(1, 2, 3, 4)")

    # 5-D
    assert_equal(String(Index(1, 2, 3, 4, 5)), "(1, 2, 3, 4, 5)")


def test_list_literal() raises:
    var list: IndexList[3] = [1, 2, 3]
    assert_equal(list[0], 1)
    assert_equal(list[1], 2)
    assert_equal(list[2], 3)


def test_write_to() raises:
    check_write_to(IndexList[3](1, 2, 3), expected="(1, 2, 3)", is_repr=False)
    check_write_to(IndexList[1](42), expected="(42,)", is_repr=False)
    check_write_to(
        IndexList[2, element_type=DType.int32](1, 2),
        expected="(1, 2)",
        is_repr=False,
    )


def test_write_repr_to() raises:
    check_write_to(
        IndexList[3](1, 2, 3),
        expected="IndexList[3, int64]((1, 2, 3))",
        is_repr=True,
    )
    check_write_to(
        IndexList[1](42),
        expected="IndexList[1, int64]((42,))",
        is_repr=True,
    )
    check_write_to(
        IndexList[2, element_type=DType.int32](1, 2),
        expected="IndexList[2, int32]((1, 2))",
        is_repr=True,
    )


def test_product_basic() raises:
    """Computes the element product of an IndexList."""
    assert_equal(product(Index(2, 3, 4)), 24)
    assert_equal(product(Index(1, 1, 1)), 1)
    assert_equal(product(Index(5)), 5)


def test_product_with_zero() raises:
    """Returns 0 when any dimension is 0."""
    assert_equal(product(Index(2, 0, 4)), 0)


def test_product_partial_range() raises:
    """Computes a partial product over a start/end range."""
    assert_equal(product(Index(2, 3, 4), 0, 2), 6)
    assert_equal(product(Index(2, 3, 4), 1, 3), 12)


def test_product_end_idx() raises:
    """Stops at end_idx, not including it."""
    assert_equal(product(Index(2, 3, 4), 2), 6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
