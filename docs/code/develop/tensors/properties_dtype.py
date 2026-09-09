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

from max.dtype import DType
from max.experimental.tensor import Tensor

# Float tensor (default for most operations)
floats = Tensor.ones([2, 2], dtype=DType.float32)

# Integer tensor
integers = Tensor.ones([2, 2], dtype=DType.int32)
