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
# DOC: max/develop/dtypes.mdx

import torch
from max.experimental.torch import torch_dtype_to_max

# PyTorch tensor
pt_tensor = torch.randn(10, 10, dtype=torch.float16)

# Convert PyTorch dtype to MAX dtype
max_dtype = torch_dtype_to_max(pt_tensor.dtype)
print(f"PyTorch {pt_tensor.dtype} → MAX {max_dtype}")  # float16 → DType.float16
