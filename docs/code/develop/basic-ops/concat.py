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

a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])

vertical = F.concat([a, b], axis=0)
horizontal = F.concat([a, b], axis=1)

print(f"Concatenated along axis 0: {vertical.shape}")
print(vertical)

print(f"Concatenated along axis 1: {horizontal.shape}")
print(horizontal)
