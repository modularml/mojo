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

from testing import assert_equal


def test_min():
    assert_equal(0, min(0, 1))
    assert_equal(1, min(1, 42))

    var lhs = SIMD[DType.int32, 4](1, 2, 3, 4)
    var rhs = SIMD[DType.int32, 4](0, 1, 5, 7)
    var expected = SIMD[DType.int32, 4](0, 1, 3, 4)
    assert_equal(expected, lhs.min(rhs))
    assert_equal(expected, rhs.min(lhs))


def test_max():
    assert_equal(1, max(0, 1))
    assert_equal(2, max(1, 2))

    var lhs = SIMD[DType.int32, 4](1, 2, 3, 4)
    var rhs = SIMD[DType.int32, 4](0, 1, 5, 7)
    var expected = SIMD[DType.int32, 4](1, 2, 5, 7)
    assert_equal(expected, lhs.max(rhs))
    assert_equal(expected, rhs.max(lhs))


def main():
    test_min()
    test_max()
    # TODO: add tests for abs, divmod, round. These tests should be simple; they
    # test the free functions, so it's not needed to cover all corner cases of
    # the underlying implementations.
