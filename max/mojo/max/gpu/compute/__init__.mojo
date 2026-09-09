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
"""GPU compute operations package - MMA and tensor core operations.

This package provides GPU tensor core and matrix multiplication operations:

- **mma**: Unified warp matrix-multiply-accumulate (WMMA) operations
- **mma_util**: Utility functions for loading/storing MMA operands
- **mma_operand_descriptor**: Operand descriptor types for MMA
- **tensor_ops**: Tensor core-based reductions and operations
- **arch/**: Architecture-specific MMA implementations (internal)
  - `mma_nvidia`: NVIDIA tensor cores (SM70-SM90)
  - `mma_nvidia_sm100`: NVIDIA Blackwell (SM100)
  - `mma_amd`: AMD Matrix Cores (CDNA2/3/4)
  - `mma_amd_rdna`: AMD WMMA (RDNA3/4)
  - `tcgen05`: 5th generation tensor core operations (Blackwell)

## Usage

Import compute operations directly:

```mojo
from max.gpu.compute import mma

# Usage: var result = mma.mma(a, b, c)
```

Architecture-specific implementations in `arch/` are internal and should not
be imported directly by user code.
"""
