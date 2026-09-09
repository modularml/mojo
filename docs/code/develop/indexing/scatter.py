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
# DOC: max/develop/indexing.mdx

# scatter.py: route expert outputs back into a hidden state (CPU; scatter is CPU-only)
#
# Output:
#   [Dim(8), Dim(64)]

import max.experimental.functional as F
from max.dtype import DType
from max.experimental.tensor import Tensor

hidden = Tensor.zeros((8, 64))
expert_out = Tensor.ones((4, 64))

routes = Tensor([0, 2, 5, 7], dtype=DType.int64).reshape((4, 1))
indices = F.tile(routes, (1, 64))

result = F.scatter(hidden, expert_out, indices, axis=0)
print(result.shape)
