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
"""CPU-side weight permutation for the MI355X small-M streaming matmul.

Mirrors the fragment-major layout of the ``smallm_preshuffle_b`` Mojo kernel
(``max/kernels/src/linalg/matmul/gpu/amd/smallm_streaming_matmul.mojo``):
16-column tile -> warp K-slice -> per-mfma 512-element chunk -> 16B per lane.
Run once at weight-load time; ``mo.smallm.streaming.matmul`` is the only
consumer of the result.

The layout is tile-major over contiguous 16-row groups, so a rowwise shard of
the shuffled tensor (at 16-row-aligned boundaries) is byte-identical to
shuffling the shard — sharding machinery needs no changes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def preshuffle_smallm_b(
    weight: npt.NDArray[np.uint16] | npt.NDArray[np.void],
    *,
    warps_per_block: int = 8,
) -> np.ndarray:
    """Permutes a row-major ``[n, k]`` bf16 weight into fragment-major order.

    Operates on the raw 2-byte elements (dtype-agnostic view), returning a new
    contiguous array with the same shape and dtype.

    Args:
        weight: Row-major ``[n, k]`` array of 2-byte elements.
        warps_per_block: Must match the matmul kernel's ``warps_per_block``.

    Returns:
        The permuted weight, same shape and dtype.
    """
    n, k = weight.shape
    mfma_k = 32
    if n % 16 != 0:
        raise ValueError(f"n must be a multiple of 16, got {n}")
    if k % (warps_per_block * mfma_k) != 0:
        raise ValueError(
            f"k must be a multiple of {warps_per_block * mfma_k}, got {k}"
        )
    if weight.dtype.itemsize != 2:
        raise ValueError(f"expected 2-byte elements, got {weight.dtype}")

    k_per_warp = k // warps_per_block
    iters = k_per_warp // mfma_k
    # Source k decomposes as (warp, chunk i, k-group g of 4, element e of 8);
    # the destination orders lanes g-major within a warp chunk and rows
    # r-minor: [tile, r, warp, i, g, e] -> [tile, warp, i, g, r, e].
    v = weight.reshape(n // 16, 16, warps_per_block, iters, 4, 8)
    v = v.transpose(0, 2, 3, 4, 1, 5)
    return np.ascontiguousarray(v).reshape(n, k)
