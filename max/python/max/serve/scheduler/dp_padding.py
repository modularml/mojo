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
"""DP batch padding: equalizes replica batch sizes for device graph capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.pipelines.context import (
    TextContext,
    TextGenerationContextType,
    TextGenerationOutput,
)
from max.pipelines.kv_cache import PagedKVCacheManagerInterface
from max.pipelines.modeling.types import (
    BatchType,
    Pipeline,
    TextGenerationInputs,
)


@dataclass
class DPPaddingInfo:
    """Padding metadata produced by `DPBatchPadder.pad_batch()`.

    Holds the dummy contexts allocated for this batch. The caller is
    responsible for releasing them via the KV cache manager and pipeline.
    """

    dummies: list[TextContext]
    """The dummy contexts padding out the short replicas."""


class DPBatchPadder:
    """Pads DP batches to equal replica sizes and removes padding from results.

    When device graph capture is active each replica must run with the
    same batch size.  This helper allocates fresh dummy contexts per
    batch and releases them after the GPU is done, so dummy KV blocks
    are not held permanently.

    Args:
        dp_size: The number of data-parallel replicas.
        kv_manager: The KV cache manager used to claim and allocate
            dummy entries.
        max_length: The maximum sequence length for dummy contexts.
        model_name: The model name passed to dummy context instances.
        pipeline: The pipeline used to release dummy entries.
        context_type: The architecture's concrete context class used to
            construct dummies. VLM batches require every context to be a
            `TextAndVisionContext`, so padding dummies must match the
            architecture's registered context type.
    """

    def __init__(
        self,
        *,
        dp_size: int,
        kv_manager: PagedKVCacheManagerInterface,
        max_length: int,
        model_name: str,
        pipeline: Pipeline[
            TextGenerationInputs[TextContext], TextGenerationOutput
        ],
        context_type: type[TextContext] = TextContext,
    ) -> None:
        self._kv_manager = kv_manager
        self._max_length = max_length
        self._model_name = model_name
        self._pipeline = pipeline
        self._context_type = context_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pad_batch(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> tuple[
        TextGenerationInputs[TextGenerationContextType],
        DPPaddingInfo | None,
    ]:
        """Pads short replicas with fresh dummy contexts.

        Returns `(inputs, None)` when no padding is needed.
        This can happen either because all replicas already have the same size
        or because the batch is not a TG batch. The returned `DPPaddingInfo` carries a
        `release()` callback that frees the dummy KV blocks and pipeline resources.
        Callers must invoke it after the GPU work is synchronized.

        Args:
            inputs: The batch inputs to pad.

        Returns:
            A tuple of the (possibly padded) inputs and a
            `DPPaddingInfo` instance, or `None` when no padding was
            needed.
        """
        if inputs.batch_type != BatchType.TG:
            return inputs, None

        real_sizes = [len(b) for b in inputs.batches]
        max_per_rank = max(real_sizes) if real_sizes else 0

        if not any(s < max_per_rank for s in real_sizes):
            return inputs, None

        # Allocate fresh dummies and track them for release.
        all_dummies: list[TextContext] = []
        padded_batches: list[list[TextGenerationContextType]] = []
        for rank, batch in enumerate(inputs.batches):
            pad_count = max_per_rank - len(batch)
            if pad_count > 0:
                dummies = self._alloc_dummies(rank, pad_count)
                all_dummies.extend(dummies)
                padded_batches.append(list(batch) + dummies)
            else:
                padded_batches.append(list(batch))

        padded_inputs: TextGenerationInputs[TextGenerationContextType] = (
            TextGenerationInputs(
                batches=padded_batches,
            )
        )
        # __post_init__ infers batch_type from contexts; override to
        # preserve the caller's original TG designation.
        padded_inputs.batch_type = inputs.batch_type

        return padded_inputs, DPPaddingInfo(dummies=all_dummies)

    def _alloc_dummies(
        self,
        replica_idx: int,
        count: int,
    ) -> list[Any]:
        """Allocates `count` fresh dummy contexts for `replica_idx`.

        Each dummy's `generated_length` is set to 1 so that
        downstream logic (e.g. EAGLE) treats the batch as a
        TG batch.

        Args:
            replica_idx: The data-parallel replica index.
            count: Number of dummy contexts to allocate.
        """
        dummies: list[Any] = []
        for _ in range(count):
            ctx = self._context_type.new_padding_context(
                max_length=self._max_length,
                model_name=self._model_name,
            )
            ctx.update(0)
            self._kv_manager.alloc_dummy(ctx, replica_idx=replica_idx)
            dummies.append(ctx)
        return dummies
