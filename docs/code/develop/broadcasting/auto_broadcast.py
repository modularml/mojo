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

# auto_broadcast.py: add a bias vector to a batch of activations
#
# Output:
#   [Dim(4), Dim(8)]

from max.experimental.tensor import Tensor

activations = Tensor.zeros([4, 8])  # (batch=4, features=8)
bias = Tensor.ones([8])  # (features=8,)
result = activations + bias
print(result.shape)
