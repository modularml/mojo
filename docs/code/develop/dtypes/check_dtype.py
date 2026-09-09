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

# Create tensors of different types
weights = Tensor.ones([3, 3], dtype=DType.float32, device=CPU())
indices = Tensor([0, 1, 2], dtype=DType.int64, device=CPU())

# Check the dtype of each tensor
print(f"Weights dtype: {weights.dtype}")  # DType.float32
print(f"Indices dtype: {indices.dtype}")  # DType.int64

# Compare dtypes directly
if weights.dtype == DType.float32:
    print("Weights are float32")
