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
"""SM100 Fused GEMM+SwiGLU kernel: BF16 matmul with SwiGLU in the epilogue.

The caller pre-permutes weight W on its N axis so adjacent output columns
(2i, 2i+1) carry (gate, up) pairs. The epilogue computes silu(gate)*up in
FP32 registers, writes BF16 results to double-buffered SMEM, and TMA-stores
to GMEM at half-N positions.
"""

from .config import swiglu_extra_fixed_smem
from .dispatch import (
    matmul_swiglu_dispatch_sm100,
    matmul_swiglu_dispatch_sm100_bf16,
)
