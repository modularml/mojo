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
"""Common text-generation pipeline backed by real MAX workers.

Chains a :class:`MAXTokenizer`, a :class:`MAXModelWorker`, and a
:class:`ChatParserWorker` so a single :class:`PipelineConfig` drives
tokenization, decoding, and response parsing through the cascade runtime.
Because the config already names the model's tool and reasoning parsers, the
pipeline can produce structured chunks without anything above it knowing what
model is loaded.
"""

from __future__ import annotations

from collections.abc import AsyncIterable

from max.experimental.cascade.interfaces.gen_ai import (
    GenAIChunk,
    GenAIRequest,
    Modality,
)
from max.experimental.cascade.interfaces.pipeline import GenAIPipeline
from max.experimental.cascade.pipelines.chat_parser import (
    ChatParserConfig,
    ChatParserWorker,
)
from max.experimental.cascade.workers.max_model_worker import MAXModelWorker
from max.experimental.cascade.workers.max_tokenizer import (
    make_tokenizer_worker,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.pipeline_runtime_config import DISABLE_PARSER_SENTINEL
from max.pipelines.lib.reasoning import get_parser_cls as reasoning_parser_cls


def chat_parser_config(config: PipelineConfig) -> ChatParserConfig:
    """Derive a model's response format from its resolved pipeline config.

    ``runtime.tool_parser`` and ``runtime.reasoning_parser`` are already
    populated during config resolution, falling back to the architecture's
    declared defaults, so everything needed to parse the model's output is on
    the config by the time a pipeline is built.

    MAX Serve splits reasoning in the token domain; the cascade parser worker
    runs after detokenization and needs the same region as text, which it takes
    from the delimiters the named :class:`ReasoningParser` declares.

    A parser that declares neither delimiter has no text form, so reasoning
    stays in the assistant's content -- the same output MAX Serve produces for
    a model with reasoning parsing disabled.

    Raises:
        ValueError: If the named reasoning parser declares one delimiter but
            not the other. A span needs both ends, so half a declaration
            cannot be honored either way.
    """
    reasoning_start: str | None = None
    reasoning_end: str | None = None
    parser_name = config.runtime.reasoning_parser
    if parser_name and (parser_cls := reasoning_parser_cls(parser_name)):
        reasoning_start = parser_cls.REASONING_START
        reasoning_end = parser_cls.REASONING_END
        if (reasoning_start is None) != (reasoning_end is None):
            raise ValueError(
                f"Reasoning parser {parser_name!r} declares "
                f"REASONING_START={reasoning_start!r} and "
                f"REASONING_END={reasoning_end!r}: declare both delimiters or "
                f"neither. Fix {parser_cls.__name__}, or pass "
                f"--reasoning-parser={DISABLE_PARSER_SENTINEL} to serve "
                "without reasoning parsing."
            )

    return ChatParserConfig(
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        tool_parser=config.runtime.tool_parser,
    )


class CommonTextGenPipeline(GenAIPipeline):
    """Cascade pipeline chaining tokenizer, model worker, and response parser."""

    def __init__(
        self,
        config: PipelineConfig,
        tokenizer_impl: str | None = None,
    ) -> None:
        """Build the pipeline for *config*.

        Args:
            config: Fully-specified ``PipelineConfig``. Its ``model_path``
                seeds the tokenizer worker and the whole config drives the
                model worker.
            tokenizer_impl: Import path (``"module.path:ClassName"``) of the
                ``TokenizerWorker`` subclass to construct, or ``None`` to use
                the HuggingFace tokenizer.
        """
        self.tokenizer = make_tokenizer_worker(
            config.model.model_path, tokenizer_impl
        )
        self.model = MAXModelWorker(config)
        self.parser = ChatParserWorker(chat_parser_config(config))

    def supported_input_modalities(self) -> set[Modality]:
        """Accept text prompts."""
        return {Modality.TEXT}

    def supported_output_modalities(self) -> set[Modality]:
        """Emit assistant text, reasoning, and tool calls."""
        return {Modality.TEXT}

    async def _generate_iterator(
        self, req: GenAIRequest
    ) -> AsyncIterable[GenAIChunk]:
        """Tokenize, decode, detokenize, and parse a request end to end.

        The orchestrator only *wires stages together*: each worker's output
        handle is passed straight to the next worker, so tokens and text flow
        worker-to-worker (a ``ResultIter`` carries a runtime handle to the
        upstream worker) and the orchestrator never sits in the middle of every
        token doing per-chunk RPCs.
        """
        tokens = await self.tokenizer.encode(req.messages)
        gen_tokens = await self.model.decode(req.text, tokens)
        text = await self.tokenizer.decode_stream(gen_tokens, True)
        return await self.parser.parse_stream(
            text, req.tools_enabled, req.tool_schemas()
        )
