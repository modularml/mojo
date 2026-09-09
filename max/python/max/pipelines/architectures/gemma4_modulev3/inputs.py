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

"""Model inputs for the Gemma4 ModuleV3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from max.driver import Buffer
from max.pipelines.lib import ModelInputs


@dataclass
class Gemma4Inputs(ModelInputs):
    """A class representing inputs for the Gemma4 model (ModuleV3).

    Image (and video-frame) embeddings arrive on the base
    ``vision_embeddings`` / ``vision_scatter_indices`` fields, populated by the
    pipeline-owned ``VisionEncoderCache``.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    return_n_logits: Buffer
    """Number of logits to return."""
