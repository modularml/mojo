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
"""Grouped block-scaled matmul with 1D-1D tensor layout for SM100.

This module provides a structured kernel implementation for grouped GEMM
operations in Mixture of Experts (MoE) layers, using contiguous token
buffers with offset-based addressing (the "1D-1D" layout).

Key characteristics:
- A tensor: Contiguous (total_tokens, K) with a_offsets for per-group access
- B tensor: Batched (num_experts, N, K) weights
- C tensor: Contiguous (total_tokens, N) output
- Per-expert output scaling via expert_scales tensor

This is a port of `max/kernels/src/linalg/grouped_matmul_sm100_1d1d.mojo`
to the structured kernels architecture.

See PORTING_PLAN.md for implementation details.
"""

from .grouped_1d1d_matmul import grouped_matmul_block_scaled
from .dispatch import (
    grouped_matmul_nvfp4_dispatch,
    grouped_matmul_mxfp8_dispatch,
    grouped_matmul_block_scaled_sm100_dispatch,
)
from .grouped_matmul_block_scaled_swiglu import (
    grouped_matmul_block_scaled_swiglu_sm100_dispatch,
)
from .grouped_matmul_swiglu_nvfp4 import grouped_matmul_swiglu_nvfp4_dispatch
from .grouped_matmul_swiglu_mxfp8 import grouped_matmul_swiglu_mxfp8_dispatch
