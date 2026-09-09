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

from __future__ import annotations

import numpy as np


# TODO(MODELS-1096): Determine whether materializing position IDs incurs a significant penalty
def compute_position_ids(
    grid_thws: list[tuple[int, int, int]], max_width: int
) -> np.ndarray:
    """Converts per-video (t, h, w) grids into flat position indices.

    Each video contributes t repetitions of its h*w grid. The 2D (row, col)
    coordinates are flattened to ``row * max_width + col``.

    Args:
        grid_thws: List of (temporal_frames, height, width) per video.
        max_width: Maximum width used for row-major flattening.

    Returns:
        1-D int64 array of position indices with length
        ``sum(t * h * w for t, h, w in grid_thws)``.
    """
    ids: list[np.ndarray] = []
    for t, h, w in grid_thws:
        row_ids = np.arange(h).reshape(-1, 1) * max_width
        col_ids = np.arange(w).reshape(1, -1)
        grid_ids = (row_ids + col_ids).flatten()  # (h*w,)
        ids.append(np.tile(grid_ids, t))  # (t*h*w,)
    return np.concatenate(ids).astype(np.int64)
