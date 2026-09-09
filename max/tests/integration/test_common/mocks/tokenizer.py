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
"""Utilities for working with mock tokenizers for unit testing"""

import json
import random
import string
from collections.abc import Sequence

import numpy as np
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.modeling.types import (
    PipelineTokenizer,
    TextGenerationRequest,
)


class MockTextTokenizer(
    PipelineTokenizer[TextContext, np.ndarray, TextGenerationRequest]
):
    """Mock tokenizer for use in unit tests."""

    def __init__(
        self,
        model_path: str = "testing/testing",
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        seed: int = 42,
        vocab_size: int = 1000,
        **kwargs,
    ) -> None:
        self.i = 0
        self.vocab_size = vocab_size
        self.seed = seed
        self.max_length = max_length or 512
        self.max_new_tokens = max_new_tokens

        random.seed(self.seed)
        chars = list(string.printable)
        random.shuffle(chars)
        self.char_to_int = {}
        self.int_to_char = {}
        for idx, char in enumerate(chars):
            self.char_to_int[char] = idx
            self.int_to_char[idx] = char

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    @property
    def eos_token_ids(self) -> set[int]:
        return {self.vocab_size - 10}

    async def new_context(self, request: TextGenerationRequest) -> TextContext:
        self.i += 1

        if request.prompt is None and not request.messages:
            raise ValueError("either prompt or messages must be provided.")

        prompt: str | Sequence[int]
        if request.prompt is None and request.messages:
            prompt = ".".join(
                [str(message.content) for message in request.messages]
            )
        elif request.prompt is not None:
            assert request.prompt is not None
            prompt = request.prompt
        else:
            raise ValueError("either prompt or messages must be provided.")

        if isinstance(prompt, str):
            encoded = await self.encode(prompt)
        else:
            encoded = np.array(prompt)

        if self.max_length:
            if len(encoded) > self.max_length:
                raise PromptTooLongError(len(encoded), self.max_length)

        if request.sampling_params.max_new_tokens:
            max_length = len(encoded) + request.sampling_params.max_new_tokens
        elif self.max_new_tokens:
            max_length = len(encoded) + self.max_new_tokens
        else:
            max_length = self.max_length

        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format and request.response_format.json_schema
            else None
        )

        ctx = TextContext(
            request_id=request.request_id,
            max_length=max_length,
            tokens=TokenBuffer(encoded),
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json_schema,
            sampling_params=request.sampling_params,
        )
        return ctx

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        return np.array([self.char_to_int[c] for c in prompt])

    async def decode(self, encoded: np.ndarray, **kwargs) -> str:
        return "".join([self.int_to_char[c] for c in encoded])
