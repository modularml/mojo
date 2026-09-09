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
"""Input batching for Gemma4 ModuleV3 pipeline models."""

from __future__ import annotations

from max.driver import Buffer
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.lib.interfaces.batch_processor import (
    ModuleV3SingleReplicaBatchProcessor,
    modulev3_gemma_multimodal_language_symbolic_inputs,
)

from .inputs import Gemma4Inputs


class Gemma4ModuleV3BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[Gemma4Context, Gemma4Inputs]
):
    """Ragged batching for Gemma4 ModuleV3 models (single GPU, no signals).

    Vision runs through the pipeline-owned ``VisionEncoderCache``, so this
    processor only builds tokens/offsets; the embeddings and scatter indices
    land on the base :class:`ModelInputs` vision fields.
    """

    def get_language_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_ref: DeviceRef,
        hidden_size: int,
        embedding_dtype: DType,
    ) -> list[TensorType | BufferType]:
        """Symbolic inputs for the ModuleV3 language-model ``compile()`` call.

        ``embedding_dtype`` must match the vision tower's output dtype:
        the runtime image-embedding buffers come
        either from the tower or from ``empty_vision_embeddings``, both of
        which use ``config.unquantized_dtype``.
        """
        return modulev3_gemma_multimodal_language_symbolic_inputs(
            kv_params=kv_params,
            device_ref=device_ref,
            hidden_size=hidden_size,
            embedding_dtype=embedding_dtype,
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer],
    ) -> Gemma4Inputs:
        return Gemma4Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )
