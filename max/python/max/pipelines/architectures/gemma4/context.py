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

"""Gemma4-specific context for storing prompt state."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context import TextAndVisionContext


@dataclass(kw_only=True)
class Gemma4Context(TextAndVisionContext):
    """A context for storing prompt state for the Diancie model."""

    mm_token_type_ids: npt.NDArray[np.int64]
    pixel_position_ids: list[npt.NDArray[np.int32]]

    @classmethod
    def _padding_context_required_fields(cls) -> dict[str, Any]:
        # A DP padding dummy holds a single text token, so its per-token
        # type-id array marks that one token as text (0).
        return {
            **super()._padding_context_required_fields(),
            "mm_token_type_ids": np.zeros(1, dtype=np.int64),
            "pixel_position_ids": [],
        }
