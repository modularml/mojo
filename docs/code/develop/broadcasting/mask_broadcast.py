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
# DOC: max/develop/broadcasting.mdx

# mask_broadcast.py: broadcast a 2D attention mask over a batch dimension
#
# Output:
#   [Dim(2), Dim(8), Dim(8)]

from max.experimental.tensor import Tensor

mask = Tensor.zeros([8, 8])  # (seq_len=8, seq_len=8)
scores = Tensor.zeros([2, 8, 8])  # (batch=2, seq_len, seq_len)
result = scores + mask
print(result.shape)
