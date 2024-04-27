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

from memory import bitcast
from testing import assert_equal


def test_bitcast():
    assert_equal(
        bitcast[DType.int8, 8](SIMD[DType.int16, 4](1, 2, 3, 4)),
        SIMD[DType.int8, 8](1, 0, 2, 0, 3, 0, 4, 0),
    )

    assert_equal(
        bitcast[DType.int32, 1](SIMD[DType.int8, 4](0xFF, 0x00, 0xFF, 0x55)),
        Int32(1442775295),
    )


def main():
    test_bitcast()
