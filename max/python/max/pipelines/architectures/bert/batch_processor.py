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
"""Input batching for BERT embedding models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import Buffer
from max.pipelines.lib.interfaces.batch_processor import (
    PaddedEncoderBatchProcessor,
)

if TYPE_CHECKING:
    from .model import BertInputs


class BertBatchProcessor(PaddedEncoderBatchProcessor["BertInputs"]):
    """Fixed-shape padded batching for encoder-only BERT models."""

    def _make_inputs(
        self,
        *,
        next_tokens_batch: Buffer,
        attention_mask: Buffer,
    ) -> BertInputs:
        from .model import BertInputs

        return BertInputs(
            next_tokens_batch=next_tokens_batch,
            attention_mask=attention_mask,
        )
