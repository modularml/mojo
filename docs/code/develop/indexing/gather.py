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

# gather.py: embedding lookup with F.gather
#
# Output:
#   [Dim(2), Dim(4), Dim(64)]

import max.experimental.functional as F
from max.dtype import DType
from max.experimental.tensor import Tensor

embeddings = Tensor.ones((32000, 64))
token_ids = Tensor(
    [[1, 42, 7, 100], [5, 13, 99, 200]],
    dtype=DType.int64,
)
output = F.gather(embeddings, token_ids, axis=0)
print(output.shape)
