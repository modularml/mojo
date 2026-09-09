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
"""Echo text-generation pipeline: real tokenizer, no model compute.

Pairs the real :class:`MAXTokenizer` with an :class:`EchoTransformer` that
replays the prompt tokens back instead of running a model. It exercises the
whole cascade path -- HuggingFace tokenization, cross-pool worker-to-worker
token streaming, and incremental detokenization -- with the GPU model swapped
out, so a benchmark against it measures cascade framework overhead in isolation
(no model forward pass). Select it with an ``echo:`` prefix on
``--models.main.model-path``; the rest of the path is the tokenizer to load.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

import numpy as np
import numpy.typing as npt
from max.experimental.cascade.core import Worker, worker_method
from max.experimental.cascade.interfaces.gen_ai import (
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
from max.experimental.cascade.workers.max_tokenizer import (
    make_tokenizer_worker,
)

Int32Array = npt.NDArray[np.int32]


class EchoTransformer(Worker):
    """Replay prompt tokens as generated tokens, doing no model compute.

    Deployed to the ``gpu`` pool -- the same placement the real
    :class:`~max.experimental.cascade.workers.max_model_worker.MAXModelWorker`
    takes -- so the echoed token stream crosses the same worker/process boundary
    the production pipeline does.
    """

    def __init__(self) -> None:
        super().__init__(deploy_hints=["gpu"])

    @worker_method()
    async def decode(
        self, req: TextGenOptions, tokens: Int32Array
    ) -> AsyncIterator[Int32Array]:
        """Stream ``num_tokens`` tokens back, one per chunk, cycling the prompt.

        Yielding one token per chunk mirrors decode-phase streaming (one token
        per scheduler step), so incremental detokenization and per-token
        streaming overhead are exercised exactly as in the real pipeline. The
        prompt tokens are real vocabulary ids, so detokenization does real work.
        """
        prompt = np.asarray(tokens, dtype=np.int32).reshape(-1)
        if prompt.size == 0:
            return
        for i in range(int(req.num_tokens)):
            yield np.array([prompt[i % prompt.size]], dtype=np.int32)


class EchoTextGenPipeline(GenAIPipeline):
    """Cascade pipeline pairing ``MAXTokenizer`` with ``EchoTransformer``."""

    def __init__(
        self, model_path: str, tokenizer_impl: str | None = None
    ) -> None:
        """Build the pipeline for the tokenizer at *model_path*.

        Args:
            model_path: Hugging Face repo id (or local path) whose tokenizer the
                worker loads. No model worker is created, so the pipeline config
                is never resolved and no weights are downloaded.
            tokenizer_impl: Import path (``"module.path:ClassName"``) of the
                ``TokenizerWorker`` subclass to construct, or ``None`` to use
                the HuggingFace tokenizer.
        """
        self.tokenizer = make_tokenizer_worker(model_path, tokenizer_impl)
        self.transformer = EchoTransformer()
        # Echoed prompt tokens carry no reasoning or tool markers, so the
        # default (plain text) parser keeps the stage in the chain -- and its
        # overhead in the measurement -- without changing the output.
        self.parser = ChatParserWorker(ChatParserConfig())

    def supported_input_modalities(self) -> set[Modality]:
        """Accept text prompts."""
        return {Modality.TEXT}

    def supported_output_modalities(self) -> set[Modality]:
        """Emit assistant text, reasoning, and tool calls."""
        return {Modality.TEXT}

    async def _generate_iterator(
        self, req: GenAIRequest
    ) -> AsyncIterable[GenAIChunk]:
        """Tokenize, echo, detokenize, and parse a request end to end.

        Identical wiring to
        :class:`~max.experimental.cascade.pipelines.common_textgen.CommonTextGenPipeline`,
        with the model worker replaced by the echo worker: every stage's stream
        flows worker-to-worker.
        """
        tokens = await self.tokenizer.encode(req.messages)
        gen_tokens = await self.transformer.decode(req.text, tokens)
        text = await self.tokenizer.decode_stream(gen_tokens, True)
        return await self.parser.parse_stream(
            text, req.tools_enabled, req.tool_schemas()
        )
