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
"""Input batching for GPT OSS ModuleV3 pipeline models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.batch_processor import (
    ModuleV3SingleReplicaBatchProcessor,
)

if TYPE_CHECKING:
    from .model import GptOssInputs


class GptOssModuleV3BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, "GptOssInputs"]
):
    """Ragged batching for GPT OSS ModuleV3 models (single GPU, no signals)."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer],
    ) -> GptOssInputs:
        from .model import GptOssInputs

        return GptOssInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )
