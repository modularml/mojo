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


@value
struct Ind(Indexer):
    fn __index__(self) -> Int:
        return 1


def test_bin_scalar():
    assert_equal(bin(Int8(2)), "0b10")
    assert_equal(bin(Int32(123)), "0b1111011")
    assert_equal(bin(Int32(-123)), "-0b1111011")
    assert_equal(bin(Scalar[DType.bool](True)), "0b1")
    assert_equal(bin(Scalar[DType.bool](False)), "0b0")


def test_bin_int():
    assert_equal(bin(0), "0b0")
    assert_equal(bin(1), "0b1")
    assert_equal(bin(-1), "-0b1")
    assert_equal(bin(4), "0b100")
    assert_equal(bin(Int(-4)), "-0b100")
    assert_equal(bin(389703), "0b1011111001001000111")
    assert_equal(bin(-10), "-0b1010")


def test_bin_bool():
    assert_equal(bin(True), "0b1")
    assert_equal(bin(False), "0b0")


def test_indexer():
    assert_equal(bin(Ind()), "0b1")


def main():
    test_bin_scalar()
    test_bin_int()
    test_bin_bool()
    test_indexer()
