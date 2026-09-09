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

# Check memory size of different dtypes
print(f"float32 size: {DType.float32.size_in_bytes} bytes")  # 4
print(f"float32.is_float(): {DType.float32.is_float()}")  # True
print(f"int32.is_integral(): {DType.int32.is_integral()}")  # True
print(f"float8_e4m3fn.is_float8(): {DType.float8_e4m3fn.is_float8()}")  # True
