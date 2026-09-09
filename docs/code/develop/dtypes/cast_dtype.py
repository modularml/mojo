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

from max.driver import CPU
from max.dtype import DType
from max.experimental.tensor import Tensor

# Create a float32 tensor
x = Tensor([1.7, 2.3, 3.9], dtype=DType.float32, device=CPU())
print(f"Original dtype: {x.dtype}")  # DType.float32

# Cast to int32 (truncates decimal values)
y = x.cast(DType.int32)
print(f"After cast to int32: {y.dtype}")  # DType.int32

# Cast to float64 for higher precision
z = x.cast(DType.float64)
print(f"After cast to float64: {z.dtype}")  # DType.float64
