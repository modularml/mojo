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
# DOC: max/develop/dtypes.mdx

from max.dtype import DType


def calculate_memory(shape: list[int], dtype: DType) -> int:
    """Calculate memory usage in bytes for a tensor."""
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    bytes_used = num_elements * dtype.size_in_bytes
    return bytes_used


# Compare dtypes for same tensor
shape = [1024, 1024, 1024]  # 1B elements

float32_mb = calculate_memory(shape, DType.float32) / (1024**2)
float16_mb = calculate_memory(shape, DType.float16) / (1024**2)
int8_mb = calculate_memory(shape, DType.int8) / (1024**2)

print(f"float32: {float32_mb:.1f} MB")  # 4096.0 MB
print(f"float16: {float16_mb:.1f} MB")  # 2048.0 MB (50% reduction)
print(f"int8: {int8_mb:.1f} MB")  # 1024.0 MB (75% reduction)
