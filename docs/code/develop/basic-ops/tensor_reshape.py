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

from max.experimental.tensor import Tensor

x = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(f"Original shape: {x.shape}")

matrix = x.reshape([3, 4])
print(f"Reshaped to 3x4: {matrix.shape}")
print(matrix)

cube = x.reshape([2, 2, 3])
print(f"Reshaped to 2x2x3: {cube.shape}")
