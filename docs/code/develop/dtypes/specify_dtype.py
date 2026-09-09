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

# Create a tensor with float32 (default for most operations)
float_tensor = Tensor.ones([2, 3], dtype=DType.float32, device=CPU())
print(f"Float tensor dtype: {float_tensor.dtype}")

# Create a tensor with int32 for indices or counts
int_tensor = Tensor([1, 2, 3], dtype=DType.int32, device=CPU())
print(f"Int tensor dtype: {int_tensor.dtype}")
