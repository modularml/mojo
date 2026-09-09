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
# DOC: max/develop/basic-ops.mdx

import max.experimental.functional as F
from max.experimental.tensor import Tensor


def linear_layer(x: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    return F.relu(x @ weights + bias)


# input: (batch=2, features=4)
x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
weights = F.normal([4, 3])  # (features=4, output=3)
bias = Tensor.zeros([3])

output = linear_layer(x, weights, bias)
print(f"Output shape: {output.shape}")
