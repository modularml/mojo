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

# The TensorOps elementwise binary ops accept an rhs with a different storage
# engine than the destination. `StaticOffsetEngine` (the stub of the planned
# static-offset storage family) views its handle `static_offset` scalar
# elements ahead, so a rhs read through it skips a header region the plain
# pointer would see. The header is filled with a sentinel that would corrupt
# the results if the offset were ignored.

# RUN: %mojo %s | FileCheck %s

from layout import TileTensor, row_major
from layout.tensor_engine import DefaultEngine, StaticOffsetEngine

comptime _OFFSET = 3


def main():
    var dst_data = Array[Float32, 8](fill=0)
    var dst = TileTensor(dst_data, row_major[2, 4]())
    for i in range(2):
        for j in range(4):
            dst[i, j] = Float32(i * 4 + j)

    # A header of sentinels followed by the payload the offset exposes.
    var rhs_data = Array[Float32, _OFFSET + 8](fill=-1000)
    for i in range(8):
        rhs_data[_OFFSET + i] = Float32(10 * (i + 1))

    var dst_ptr: Pointer[Float32, origin_of(dst_data)] = dst_data.unsafe_ptr()
    var rhs_ptr: Pointer[Float32, origin_of(rhs_data)] = rhs_data.unsafe_ptr()
    DefaultEngine[element_width=1].iadd[
        dtype=DType.float32,
        OtherEngine=StaticOffsetEngine[static_offset=_OFFSET],
    ](
        (dst_ptr, row_major[2, 4]()),
        (rhs_ptr, row_major[2, 4]()),
    )
    # CHECK{LITERAL}: [[10.0, 21.0, 32.0, 43.0], [54.0, 65.0, 76.0, 87.0]]
    print(dst)

    # Rank-1 broadcast through the offset engine: bias[0] applies to row 0,
    # bias[1] to row 1.
    var bias_data = Array[Float32, _OFFSET + 2](fill=-1000)
    bias_data[_OFFSET] = 100.0
    bias_data[_OFFSET + 1] = 200.0

    var bias_ptr: Pointer[
        Float32, origin_of(bias_data)
    ] = bias_data.unsafe_ptr()
    DefaultEngine[element_width=1].iadd[
        dtype=DType.float32,
        OtherEngine=StaticOffsetEngine[static_offset=_OFFSET],
    ](
        (dst_ptr, row_major[2, 4]()),
        (bias_ptr, row_major[2]()),
    )
    # CHECK{LITERAL}: [[110.0, 121.0, 132.0, 143.0], [254.0, 265.0, 276.0, 287.0]]
    print(dst)
