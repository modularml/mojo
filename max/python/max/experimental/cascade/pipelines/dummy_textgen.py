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
"""Dummy text-generation components for local cascade examples and tests."""

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from max.experimental.cascade.core import (
    Worker,
    worker_method,
)
from max.experimental.cascade.interfaces.gen_ai import (
    ChatMessage,
    GenAIChunk,
    GenAIRequest,
    Modality,
    TextGenOptions,
)
from max.experimental.cascade.interfaces.pipeline import GenAIPipeline
from max.experimental.cascade.pipelines.chat_parser import (
    ChatParserConfig,
    ChatParserWorker,
)
from max.experimental.cascade.workers.max_tokenizer import flatten_message

Int32Array = npt.NDArray[np.int32]


class AsciiTokenizer(Worker):
    """Encode and decode ASCII characters for toy text-generation tests."""

    def __init__(self) -> None:
        super().__init__(deploy_hints=["cpu"])

    @worker_method()
    async def encode(self, messages: list[ChatMessage]) -> Int32Array:
        """Convert a conversation to ASCII integer tokens."""
        text = "\n".join(
            flatten_message(message)["content"] for message in messages
        )
        return np.array([ord(char) for char in text], dtype=np.int32)

    @worker_method()
    async def decode(self, token: int) -> str:
        """Convert an integer token back into a single character."""
        return chr(token)

    @worker_method()
    async def decode_streaming(
        self, token_iter: AsyncIterable[int]
    ) -> AsyncIterator[str]:
        """Convert a token stream into a stream of single-character strings."""
        async for token in token_iter:
            yield chr(token)


class Transformer(Worker):
    """Yield a fixed token stream for deterministic tests."""

    def __init__(self) -> None:
        super().__init__(deploy_hints=["gpu"])

    @worker_method()
    async def decode(
        self, req: TextGenOptions, tokens: Int32Array
    ) -> AsyncIterator[int]:
        """Emit ``num_tokens`` copies of the token for ``"A"``."""
        del tokens
        for _ in range(req.num_tokens):
            yield ord("A")


@dataclass
class DummyTextGenPipeline(GenAIPipeline):
    """Cascade pipeline pairing the dummy tokenizer and transformer workers."""

    tokenizer: AsciiTokenizer
    transformer: Transformer
    parser: ChatParserWorker

    def supported_input_modalities(self) -> set[Modality]:
        """Accept text prompts."""
        return {Modality.TEXT}

    def supported_output_modalities(self) -> set[Modality]:
        """Emit assistant text, reasoning, and tool calls."""
        return {Modality.TEXT}

    async def _generate_iterator(
        self, req: GenAIRequest
    ) -> AsyncIterable[GenAIChunk]:
        """Run text generation from a conversation."""
        tokens = await self.tokenizer.encode(req.messages)
        gen_tokens = await self.transformer.decode(req.text, tokens)
        text = await self.tokenizer.decode_streaming(gen_tokens)
        return await self.parser.parse_stream(
            text, req.tools_enabled, req.tool_schemas()
        )


async def build_dummy_textgen_pipeline() -> DummyTextGenPipeline:
    """Build the dummy text pipeline."""
    return DummyTextGenPipeline(
        AsciiTokenizer(),
        Transformer(),
        ChatParserWorker(ChatParserConfig()),
    )
