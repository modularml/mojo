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
"""Input batching for LFM2 pipeline models."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3BatchProcessor,
)
from max.pipelines.context import TextContext
from max.support.algorithm import flatten2d

if TYPE_CHECKING:
    from .model import ConvStateCache, LFM2Inputs


class LFM2BatchProcessor(Llama3BatchProcessor):
    """Ragged batching extended with per-request conv-state slots."""

    _conv_cache: ConvStateCache | None = None

    def bind_conv_cache(self, conv_cache: ConvStateCache) -> None:
        """Wires conv-state slots created during model ``__init__``."""
        self._conv_cache = conv_cache

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> LFM2Inputs:
        from .model import LFM2Inputs

        assert self._conv_cache is not None, (
            "LFM2 conv-state cache must be bound before prepare_initial_token_inputs()"
        )

        base = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
        context_batch = flatten2d(replica_batches)
        request_ids = [ctx.request_id for ctx in context_batch]
        for rid in request_ids:
            self._conv_cache.claim(rid)
        conv_states = self._conv_cache.get_states(request_ids)
        return LFM2Inputs(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
            conv_states=conv_states,
            request_ids=request_ids,
        )
