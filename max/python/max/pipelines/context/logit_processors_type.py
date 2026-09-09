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
"""Module for defining the `LogitsProcessor` type.

A callable for processing model logits, taking a `ProcessorInputs` object as
input. The `ProcessorInputs` dataclass contains the following fields:

    - logits: The model logits, a float32 tensor with shape `(N, vocab_size)`.
    `N` is the number of logits returned by the model.
    - context: The context containing the inputs to the model.

The function can update the logits in place if needed.

Examples:

.. code-block:: python

    from max.pipelines.context.logit_processors_type import ProcessorInputs

    # Example 1:
    def logits_processor(inputs: ProcessorInputs) -> None:
        logits = inputs.logits
        logits[-1, 0] = -10000

    # Example 2:
    class SuppressBeginToken:
        def __init__(self, tokens_to_suppress: list[int], steps: int):
            self.tokens_to_suppress = tokens_to_suppress
            self.steps = steps
            self.step_counter = 0

        def __call__(self, inputs: ProcessorInputs) -> None:
            logits = inputs.logits
            if self.step_counter < self.steps:
                logits[-1, self.tokens_to_suppress] = -10000
                self.step_counter += 1
    processor = SuppressBeginToken(tokens_to_suppress=[0, 1], steps=10)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import max.driver as md

if TYPE_CHECKING:
    from .context import TextContext


@dataclass
class ProcessorInputs:
    """Inputs passed to a logits processor callback."""

    logits: md.Buffer
    """The model logits buffer with shape ``(N, vocab_size)`` and ``float32`` dtype."""

    context: TextContext
    """The generation context containing request state and sampling configuration."""


LogitsProcessor: TypeAlias = Callable[[ProcessorInputs], None]


@dataclass
class BatchProcessorInputs:
    """Arguments for a batch logits processor.

    - logits: The model logits, a float32 tensor with shape ``(N_batch, vocab_size)``.
      ``N_batch`` is the number of logits returned by the model for each sequence in the batch.
    - logit_offsets: If the model returns multiple logits, this is a tensor with
      shape ``(batch_size + 1, 1)`` that contains the offsets of each sequence in
      the batch. Otherwise, this is ``None``.
    - context_batch: The batch of contexts containing the inputs to the model.
    """

    logits: md.Buffer
    """The model logits buffer with shape ``(N_batch, vocab_size)`` and ``float32`` dtype."""

    logit_offsets: md.Buffer | None
    """Offsets tensor with shape ``(batch_size + 1, 1)`` for multi-logit models, or ``None``."""

    context_batch: Sequence[TextContext]
    """The ordered sequence of generation contexts corresponding to each batch entry."""


BatchLogitsProcessor: TypeAlias = Callable[[BatchProcessorInputs], None]
