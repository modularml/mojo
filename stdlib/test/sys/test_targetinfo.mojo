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

from sys.info import (
    alignof,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    sizeof,
)

from testing import assert_equal, assert_true


fn test_sizeof() raises:
    assert_equal(sizeof[__mlir_type.i16](), 2)

    assert_equal(sizeof[__mlir_type.ui16](), 2)

    assert_equal(sizeof[DType.int16](), 2)

    assert_equal(sizeof[DType.uint16](), 2)

    assert_equal(sizeof[SIMD[DType.int16, 2]](), 4)


fn test_alignof() raises:
    assert_true(alignof[__mlir_type.i16]() > 0)

    assert_true(alignof[__mlir_type.ui16]() > 0)

    assert_true(alignof[DType.int16]() > 0)

    assert_true(alignof[DType.uint16]() > 0)

    assert_true(alignof[SIMD[DType.int16, 2]]() > 0)


fn test_cores() raises:
    assert_true(num_logical_cores() > 0)
    assert_true(num_physical_cores() > 0)
    assert_true(num_performance_cores() > 0)


def main():
    test_sizeof()
    test_alignof()
    test_cores()
