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
# DOC: max/develop/eager-execution.mdx

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor

# Create input data
x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DType.float32, device=CPU())

# Create random weights
w = F.gaussian([2, 2], mean=0.0, std=0.1, dtype=DType.float32, device=CPU())

# Forward pass - each operation executes as you write it
z = x @ w  # Matrix multiply
h = F.relu(z)  # Activation
out = h.mean()  # Reduce to scalar

# Inspect intermediate results anytime
print(f"Input shape: {x.shape}")
print(f"After matmul: {z.shape}")
print(f"Output: {out}")
