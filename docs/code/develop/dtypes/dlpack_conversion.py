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

import numpy as np
from max.experimental.tensor import Tensor

# Create a NumPy array
np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

# Convert to MAX tensor using DLPack (zero-copy when possible)
tensor = Tensor.from_dlpack(np_array)

print(f"NumPy dtype: {np_array.dtype}")  # float32
print(f"MAX tensor dtype: {tensor.dtype}")  # DType.float32
print(f"MAX tensor shape: {tensor.shape}")  # [2, 2]
