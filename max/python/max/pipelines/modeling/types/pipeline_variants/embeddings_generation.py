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
"""Interfaces and response structures for embedding generation in the MAX API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import msgspec
import numpy as np
import numpy.typing as npt
from max.pipelines.context import BaseContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.modeling.types.pipeline import PipelineInputs, PipelineOutput
from max.pipelines.request import RequestID


@runtime_checkable
class EmbeddingsContext(BaseContext, Protocol):
    """Protocol defining the interface for embeddings generation contexts.

    An ``EmbeddingsContext`` represents model inputs for embeddings generation pipelines,
    managing the state and parameters needed for generating embeddings from input text.
    Unlike text generation contexts, this focuses on single-step embedding generation
    without iterative token generation concerns.

    This protocol includes only the fields necessary for embeddings generation,
    excluding text generation specific features like:
    - End-of-sequence token handling (eos_token_ids)
    - Grammar matchers for structured output (matcher)
    - JSON schema constraints (json_schema)
    - Log probability tracking (log_probabilities)
    - Token generation iteration state
    """

    @property
    def tokens(self) -> TokenBuffer:
        """The input tokens to be embedded.

        Returns:
            A NumPy array of token IDs representing the input text to generate
            embeddings for.
        """
        ...

    @property
    def model_name(self) -> str:
        """The name of the embeddings model to use.

        Returns:
            A string identifying the specific embeddings model for this request.
        """
        ...


EmbeddingsGenerationContextType = TypeVar(
    "EmbeddingsGenerationContextType", bound=EmbeddingsContext
)


@dataclass(frozen=True)
class EmbeddingsGenerationInputs(PipelineInputs):
    """Batched inputs for an embeddings generation pipeline step."""

    batches: list[dict[RequestID, EmbeddingsContext]]

    @property
    def batch(self) -> dict[RequestID, EmbeddingsContext]:
        """Returns merged batches."""
        return {k: v for batch in self.batches for k, v in batch.items()}


class EmbeddingsGenerationOutput(msgspec.Struct, tag=True, omit_defaults=True):
    """Response structure for embedding generation.

    Configuration:
        embeddings: The generated embeddings as a NumPy array.
    """

    embeddings: npt.NDArray[np.floating[Any]]
    """The generated embeddings as a NumPy array."""

    @property
    def is_done(self) -> bool:
        """Indicates whether the embedding generation process is complete.

        Returns:
            Always ``True``, as embedding generation is a single-step operation.
        """
        return True


def _check_embeddings_output_implements_pipeline_output(
    x: EmbeddingsGenerationOutput,
) -> PipelineOutput:
    return x
