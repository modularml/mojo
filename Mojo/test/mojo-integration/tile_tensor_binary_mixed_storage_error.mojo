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

# TileTensor's in-place binary operators require both operands to use the
# same engine: a `DefaultEngine`-backed destination rejects a
# `StaticOffsetEngine`-backed rhs at compile time. Mixed-engine ops are
# still available through the engine-level TensorOps API (see
# tile_tensor_binary_mixed_storage.mojo).

# RUN: not %mojo %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major
from layout.tensor_engine import StaticOffsetEngine


def main():
    var a_data = Array[Float32, 4](fill=1.0)
    var b_data = Array[Float32, 4](fill=1.0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor[
        .float32,
        type_of(row_major[2, 2]()),
        origin_of(b_data),
        Engine=StaticOffsetEngine[static_offset=0],
    ](b_data.unsafe_ptr(), row_major[2, 2]())

    # CHECK: in-place binary ops require operands with the same engine
    a += b
