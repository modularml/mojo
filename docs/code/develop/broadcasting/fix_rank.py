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

# fix_rank.py: use unsqueeze to fix a rank mismatch before broadcasting
#
# Output:
#   [Dim(2), Dim(4), Dim(8)]

from max.experimental.tensor import Tensor

v = Tensor.ones([8])  # (features=8,)
v2d = v.unsqueeze(0)  # (1, features=8)
v3d = v2d.unsqueeze(0)  # (1, 1, features=8)
x = Tensor.zeros([2, 4, 8])  # (seq_len=2, batch=4, features=8)
result = x + v3d
print(result.shape)
