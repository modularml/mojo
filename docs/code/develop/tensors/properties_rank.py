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
# DOC: max/develop/tensors.mdx

from max.experimental.tensor import Tensor

scalar = Tensor([42])  # Rank 1 (it's a 1-element vector)
vector = Tensor([1, 2, 3])  # Rank 1
matrix = Tensor([[1, 2], [3, 4]])  # Rank 2
cube = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Rank 3

print(vector.rank)  # 1
print(matrix.rank)  # 2
print(cube.rank)  # 3
