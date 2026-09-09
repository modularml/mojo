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
#
# Verifies that dynamic split rejects negative partition indices before
# computing a pointer offset.
#
# ===----------------------------------------------------------------------=== #

from layout import TileTensor, row_major


# CHECK-LABEL: test_tile_tensor_split_dynamic_negative_idx
def main():
    print("== test_tile_tensor_split_dynamic_negative_idx")
    var data = Array[Int32, 8](fill=0)
    var tensor = TileTensor(data, row_major[4, 2]())

    # CHECK: Assert Error: split idx out of range
    _ = tensor.as_immut().split[axis=0](2, -1)
    # CHECK-NOT: is never reached
    print("is never reached")
