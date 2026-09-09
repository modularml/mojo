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

# DType is an enum that defines how numbers are stored in tensors
# Access dtypes as attributes of the DType class
print(DType.float32)  # 32-bit floating point
print(DType.int32)  # 32-bit integer
print(DType.bool)  # Boolean values
