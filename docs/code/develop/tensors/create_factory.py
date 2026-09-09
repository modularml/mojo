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

from max.driver import CPU
from max.dtype import DType
from max.experimental.tensor import Tensor

# Tensor filled with ones
ones = Tensor.ones([3, 4], dtype=DType.float32, device=CPU())
print(ones)

# Tensor filled with zeros
zeros = Tensor.zeros([2, 3], dtype=DType.float32, device=CPU())
print(zeros)
