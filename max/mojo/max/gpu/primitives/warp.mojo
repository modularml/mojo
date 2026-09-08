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
"""GPU warp-level operations and utilities.

This module provides warp-level operations for NVIDIA and AMD GPUs, including:

- Shuffle operations to exchange values between threads in a warp:
  - shuffle_idx: Copy value from source lane to other lanes
  - shuffle_up: Copy from lower lane IDs
  - shuffle_down: Copy from higher lane IDs
  - shuffle_xor: Exchange values in butterfly pattern

- Warp-wide reductions:
  - sum: Compute sum across warp
  - max: Find maximum value across warp
  - min: Find minimum value across warp
  - broadcast: Broadcast value to all lanes

The module handles both NVIDIA and AMD GPU architectures through architecture-specific
implementations of the core operations. It supports various data types including
integers, floats, and half-precision floats, with SIMD vectorization.
"""

from std._gpu.globals import WARP_SIZE
from std._gpu.primitives.warp import (
    _ReduceFn,
    # Reached directly by the MSA top-k kernels and the SM100 attention
    # kernels; a private name needs an explicit re-export.
    _dpp_move,
    _vote_nvidia_helper,
)


@__doc_inline
from std._gpu.primitives.warp import (
    broadcast,
    lane_group_max,
    lane_group_min,
    lane_group_reduce,
    lane_group_sum,
    match_all,
    match_any,
    max,
    min,
    prefix_sum,
    reduce,
    shuffle_down,
    shuffle_idx,
    shuffle_up,
    shuffle_xor,
    sum,
    vote,
)
