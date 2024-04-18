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
# RUN: %mojo %s | FileCheck %s

from sys.intrinsics import (
    compressed_store,
    masked_load,
    masked_store,
    strided_load,
    strided_store,
)

from memory.unsafe import DTypePointer
from testing import assert_equal


def test_strided_load():
    alias size = 16
    var vector = DTypePointer[DType.float32]().alloc(size)

    for i in range(size):
        vector[i] = i

    # CHECK: [0.0, 4.0, 8.0, 12.0]
    var s = strided_load[DType.float32, 4](vector, 4)
    print(s)
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
