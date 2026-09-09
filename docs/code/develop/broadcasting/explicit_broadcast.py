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

# explicit_broadcast.py: expand a (1, features) tensor to (batch, features) with F.broadcast_to
#
# Output:
#   [Dim(4), Dim(8)]

from max.experimental import functional as F
from max.experimental.tensor import Tensor

weight = Tensor.ones([1, 8])  # (1, features=8)
expanded = F.broadcast_to(weight, [4, 8])  # (batch=4, features=8)
print(expanded.shape)
