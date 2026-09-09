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

# where.py: causal mask with F.where (broadcasting mask over batch)
#
# Output:
#   [Dim(2), Dim(4), Dim(4)]

import max.experimental.functional as F
from max.dtype import DType
from max.experimental.tensor import Tensor

seq_len = 4

rows = Tensor.arange(seq_len, dtype=DType.int64).reshape((seq_len, 1))
cols = Tensor.arange(seq_len, dtype=DType.int64).reshape((1, seq_len))
mask = rows >= cols
scores = Tensor.ones((2, seq_len, seq_len))

masked = F.where(mask, scores, -1e9)
print(masked.shape)
