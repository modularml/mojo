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

"""Kimi K2.5 bespoke context object."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from max.pipelines.context.context import TextAndVisionContext


@dataclass(kw_only=True, slots=True)
class KimiK2_5TextAndVisionContext(TextAndVisionContext):
    """Context for Kimi K2.5 multimodal model processing.

    Extends ``TextAndVisionContext`` with the per-image grid dimensions
    (temporal, height, width) required by the vision encoder's positional
    embedding and RoPE computation, plus precomputed vision inputs that
    are populated by the tokenizer to avoid re-computation at model execute
    time.
    """

    grid_thws: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64)
    )
    """Per-image grid dimensions of shape ``(N, 3)`` where each row is
    ``(temporal_frames, height_patches, width_patches)``."""

    position_ids: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    """Flat 1-D vision RoPE position indices of length
    ``sum(t * h * w for t, h, w in grid_thws)``, precomputed by the
    tokenizer via ``compute_position_ids``."""

    image_token_indices: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    """Positions of the image-placeholder token within this context's token
    buffer. Offsets are relative to the start of this context and are
    adjusted to batch-absolute positions by the model at execute time."""

    max_h: int = 0
    """Maximum height (in patches) across all images in this context."""

    max_w: int = 0
    """Maximum width (in patches) across all images in this context."""
