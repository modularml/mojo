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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    GenerationStatus,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
    TokenSlice,
)
from max.pipelines.kv_cache import DummyKVCache, PagedKVCacheManager
from max.pipelines.lib import build_eos_tracker_for_request
from max.pipelines.modeling.types import (
    Pipeline,
    PipelineTokenizer,
    RequestID,
    TextGenerationInputs,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)


@dataclass
class EchoPipelineTokenizer(
    PipelineTokenizer[
        TextContext, npt.NDArray[np.integer[Any]], TextGenerationRequest
    ]
):
    """Echo tokenizer that creates TextContext instances.

    This tokenizer creates TextContext objects for echo generation without requiring
    a real HuggingFace tokenizer. It treats input characters as tokens for simplicity.
    """

    @property
    def eos_token_ids(self) -> set[int]:
        """Echo has no EOS; generation stops by length."""
        return set()

    @property
    def expects_content_wrapping(self) -> bool:
        """Echo tokenizer doesn't require content wrapping."""
        return False

    async def encode(
        self,
        prompt: str | Sequence[int],
        add_special_tokens: bool = False,
    ) -> TokenSlice:
        """Encode the prompt into token IDs.

        For simplicity, we convert string characters to their ASCII values as token IDs.
        If prompt is already a sequence of ints, we use it directly.
        """
        if isinstance(prompt, str):
            if not prompt:
                raise ValueError("Prompt cannot be empty")
            # Convert string characters to ASCII values as token IDs
            return np.array([ord(char) for char in prompt], dtype=np.int64)
        else:
            # Already a sequence of integers
            tokens = list(prompt)
            if not tokens:
                raise ValueError("Prompt cannot be empty")
            return np.array(tokens, dtype=np.int64)

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
    ) -> str:
        """Decode token IDs back to text.

        Convert ASCII values back to characters.
        """
        if isinstance(encoded, int):
            encoded = np.array([encoded])
        elif encoded.ndim == 0:
            encoded = np.array([encoded.item()])

        # Convert ASCII values back to characters
        try:
            return "".join(chr(int(token_id)) for token_id in encoded)
        except ValueError:
            # Fallback for non-ASCII values
            return "".join(str(int(token_id)) for token_id in encoded)

    async def new_context(self, request: TextGenerationRequest) -> TextContext:
        """Creates a new TextContext for echo generation."""

        # Extract prompt from request
        prompt: str | Sequence[int]
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages:
            prompt = "\n".join(
                [
                    str(message.content)
                    for message in cast(
                        list[TextGenerationRequestMessage], request.messages
                    )
                ]
            )
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        # Encode the prompt to get token array
        encoded_prompt = await self.encode(prompt, add_special_tokens=False)

        # Determine max tokens
        max_new_tokens = request.sampling_params.max_new_tokens or len(
            encoded_prompt
        )
        max_length = len(encoded_prompt) + max_new_tokens

        token_buffer = TokenBuffer(
            array=encoded_prompt.astype(np.int64, copy=False),
        )
        # Create TextContext manually
        context = TextContext(
            request_id=request.request_id,
            max_length=max_length,
            tokens=token_buffer,
            eos_tracker=await build_eos_tracker_for_request(
                self.eos_token_ids,
                request,
                self.encode,
            ),
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
        )

        return context


@dataclass
class EchoTokenGenerator(
    Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput]
):
    """Token generator that echoes the prompt tokens in their original order."""

    def __init__(self) -> None:
        # Track the echo index for each request (0-based, counts how many tokens we've echoed)
        self._echo_indices: dict[RequestID, int] = {}
        self._kv_manager = DummyKVCache()
        self._max_batch_size = 1

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the KV cache manager for this pipeline."""
        return self._kv_manager

    def execute(
        self,
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        responses = {}

        for context in inputs.flat_batch:
            request_id = context.request_id
            if request_id not in responses:
                responses[request_id] = TextGenerationOutput(
                    request_id=request_id,
                    tokens=[],
                    final_status=GenerationStatus.ACTIVE,
                    log_probabilities=None,
                )

            # Initialize echo index if not exists
            if request_id not in self._echo_indices:
                self._echo_indices[request_id] = 0

            echo_idx = self._echo_indices[request_id]
            prompt_tokens = context.tokens.prompt

            # Check if we have more tokens to echo and haven't reached max length
            if echo_idx >= len(prompt_tokens):
                responses[
                    request_id
                ].final_status = GenerationStatus.END_OF_SEQUENCE
                if request_id in self._echo_indices:
                    del self._echo_indices[request_id]
            elif len(context.tokens) >= context.max_length:
                responses[
                    request_id
                ].final_status = GenerationStatus.MAXIMUM_LENGTH
                if request_id in self._echo_indices:
                    del self._echo_indices[request_id]
            else:
                # Echo the next token in the original order
                next_token_id = int(prompt_tokens[echo_idx])

                # Update the context with the new token
                context.update(next_token_id)

                # Add to response
                responses[request_id].tokens.append(next_token_id)

                # Move to the next token
                self._echo_indices[request_id] += 1

        return responses

    def release(self, request_id: RequestID) -> None:
        """Clean up any state associated with the request."""
        # Clean up the echo index for this request if it exists
        if request_id in self._echo_indices:
            del self._echo_indices[request_id]
