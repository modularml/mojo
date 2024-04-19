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

from sys import (
    compressed_store,
    masked_load,
    masked_store,
    strided_load,
    strided_store,
)

from testing import assert_equal
from memory import DTypePointer


fn test_strided_load() raises:
    alias size = 16
    var vector = DTypePointer[DType.float32]().alloc(size)

    for i in range(size):
        vector[i] = i

    var s = strided_load[DType.float32, 4](vector, 4)
    assert_equal(s, SIMD[DType.float32, 4](0, 4, 8, 12))

    vector.free()


fn test_strided_store() raises:
    alias size = 8
    var vector = DTypePointer[DType.float32]().alloc(size)
    memset_zero(vector, size)

    strided_store(SIMD[DType.float32, 4](99, 12, 23, 56), vector, 2)
    assert_equal(vector[0], 99.0)
    assert_equal(vector[1], 0.0)
    assert_equal(vector[2], 12.0)
    assert_equal(vector[3], 0.0)
    assert_equal(vector[4], 23.0)
    assert_equal(vector[5], 0.0)
    assert_equal(vector[6], 56.0)
    assert_equal(vector[7], 0.0)

    vector.free()


def main():
    test_strided_load()
    test_strided_store()
