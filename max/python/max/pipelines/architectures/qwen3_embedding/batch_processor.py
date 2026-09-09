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
"""Input batching for Qwen3 embedding pipeline models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import Buffer
from max.pipelines.lib.interfaces.batch_processor import (
    SingleReplicaEmbeddingBatchProcessor,
)

if TYPE_CHECKING:
    from .model import Qwen3EmbeddingInputs


class Qwen3EmbeddingBatchProcessor(
    SingleReplicaEmbeddingBatchProcessor["Qwen3EmbeddingInputs"]
):
    """Ragged batching for Qwen3 embedding models (no KV cache, no signals)."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
    ) -> Qwen3EmbeddingInputs:
        from .model import Qwen3EmbeddingInputs

        return Qwen3EmbeddingInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
        )
