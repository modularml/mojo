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

"""Output types for pipeline generation operations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

from max.pipelines.request import RequestID
from max.pipelines.request.open_responses import OutputContent, Usage
from pydantic import BaseModel, ConfigDict

from .log_probabilities import LogProbabilities
from .status import GenerationStatus


@dataclass(kw_only=True)
class TextGenerationOutput:
    """Represents the output of a text generation operation.

    Combines token IDs, final generation status, request ID, and optional log
    probabilities for each token.
    """

    request_id: RequestID
    """The unique identifier for the generation request."""

    tokens: list[int]
    """List of generated token IDs."""

    final_status: GenerationStatus
    """The final status of the generation process."""

    log_probabilities: list[LogProbabilities] | None = None
    """Optional list of log probabilities for each token."""

    num_cached_tokens: int | None = None
    """Number of prompt tokens served from the KV prefix cache."""

    @property
    def is_done(self) -> bool:
        """Indicates whether the text generation process is complete.

        Returns:
            ``True`` if the generation is done, ``False`` otherwise.
        """
        return self.final_status.is_done

    @classmethod
    def merge(cls, outputs: list[TextGenerationOutput]) -> TextGenerationOutput:
        """Combine many TextGenerationOutput chunks into a single TextGenerationOutput."""
        if len(outputs) == 0:
            raise ValueError("Cannot combine empty list of chunks")
        if len(outputs) == 1:
            return outputs[0]

        if all(output.log_probabilities is not None for output in outputs):
            log_probabilities = list(
                chain.from_iterable(
                    output.log_probabilities or [] for output in outputs
                )
            )
        elif all(output.log_probabilities is None for output in outputs):
            log_probabilities = None
        else:
            raise ValueError(
                "Cannot combine TextGenerationOutput chunks with mixed None and non-None log_probabilities"
            )

        return cls(
            request_id=outputs[0].request_id,
            tokens=list(
                chain.from_iterable(output.tokens for output in outputs)
            ),
            log_probabilities=log_probabilities,
            final_status=outputs[-1].final_status,
            num_cached_tokens=outputs[0].num_cached_tokens,
        )


class GenerationOutput(BaseModel):
    """Output container for image generation pipeline operations.

    This class holds a list of generated images in OpenResponses API format,
    along with request tracking and status information. It implements the
    PipelineOutput protocol by providing the required ``is_done`` property.

    Example:

    .. code-block:: python

        import numpy as np
        from max.pipelines.context.outputs import GenerationOutput
        from max.pipelines.request import RequestID
        from max.pipelines.request.open_responses import OutputImageContent
        from max.pipelines.context.status import GenerationStatus

        img_array = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        result = GenerationOutput(
            request_id=RequestID(value="req-123"),
            final_status=GenerationStatus.END_OF_SEQUENCE,
            output=[OutputImageContent.from_numpy(img_array, format="png")],
        )
        if result.is_done:
            print(f"Generated {len(result.output)} images")
    """

    model_config = ConfigDict(frozen=True)

    request_id: RequestID
    """The unique identifier for the generation request."""

    final_status: GenerationStatus
    """The final status of the generation process."""

    output: list[OutputContent]
    """List of OutputContent objects (text, images, etc.) representing generated content."""

    usage: Usage | None = None
    """Usage reported in the API response. Image generation keeps token
    counts at 0 and reports metadata under ``image_generation_details``."""

    @property
    def is_done(self) -> bool:
        """Indicates whether the pipeline operation has completed.

        Returns:
            ``True`` if the generation is done (status is not ACTIVE),
            ``False`` otherwise.
        """
        return self.final_status.is_done
