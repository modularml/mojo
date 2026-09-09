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

"""Provides functions for applying logits processors to model output batches before sampling."""

from __future__ import annotations

from max.driver import CPU, Buffer
from max.pipelines.context import TextGenerationContextType
from max.pipelines.context.logit_processors_type import (
    BatchLogitsProcessor,
    BatchProcessorInputs,
    ProcessorInputs,
)


def apply_logits_processors(
    context_batch: list[TextGenerationContextType],
    batch_logits: Buffer,
    batch_logit_offsets: Buffer | None,
    batch_processors: list[BatchLogitsProcessor] | None = None,
) -> None:
    """Applies logits processors to a batch of logits.

    Args:
        context_batch: The batch of contexts containing the inputs to the model.
        batch_logits: The model logits, a float32 tensor with shape `(N_batch, vocab_size)`.
        batch_logit_offsets: If the model returns multiple logits, this is a tensor with
            shape `(batch_size + 1, 1)` that contains the offsets of each sequence in
            the batch. Otherwise, this is `None`.
        logits_processors: List of logits processors to apply to the logits for
            each context in the batch. The length of this list must match the
            number of contexts in the batch.
        batch_processors: List of batch processors to apply to the batch logits.
            These are applied in order after the individual context-level
            processors.
    """
    batch_logit_offsets_cpu: Buffer | None = None
    for i, context in enumerate(context_batch):
        processors = context.sampling_params.logits_processors
        if processors is None:
            continue
        if batch_logit_offsets_cpu is None and batch_logit_offsets is not None:
            batch_logit_offsets_cpu = batch_logit_offsets.to(CPU())

        if batch_logit_offsets_cpu is not None:
            start_idx = batch_logit_offsets_cpu[i].item()
            end_idx = batch_logit_offsets_cpu[i + 1].item()
        else:
            start_idx = i
            end_idx = i + 1
        logits = batch_logits[start_idx:end_idx, :]
        assert isinstance(logits, Buffer)
        for processor in processors:
            processor(ProcessorInputs(logits=logits, context=context))

    if batch_processors is not None:
        for batch_processor in batch_processors:
            batch_processor(
                BatchProcessorInputs(
                    logits=batch_logits,
                    logit_offsets=batch_logit_offsets,
                    context_batch=context_batch,
                )
            )
