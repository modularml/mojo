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
r"""Tokenizer-free echo pipeline that replays a prompt's text verbatim.

Unlike :class:`~max.experimental.cascade.pipelines.echo_textgen.EchoTransformer`
-- which echoes real prompt *token ids* through the real tokenizer (encode ->
replay tokens -> detokenize) and cycles them to ``num_tokens`` -- this worker
skips the tokenizer entirely and streams the prompt *text* straight back,
character by character, exactly once. It never produces token ids and never
loads a tokenizer.

That difference makes it useful in two ways the token echo can't be:

* **Hermetic, no network.** With no tokenizer to download it runs fully
  in-process, so it can drive the serve path end to end (reasoning + tool-call
  parsing and OpenAI framing) with no GPU model *and* no tokenizer. Feed it
  ``"<think>...</think>Sure!<tool_call>get_weather\n{...}</tool_call>"`` and the
  raw text is exactly the scripted "model output" the serve layer parses.
* **Tokenizer-skipping ablation.** Removing tokenization/detokenization from the
  path isolates cascade framework and serve-layer overhead from tokenizer cost,
  which is handy for ablation benchmarking runs.

Because it stays in the text domain, it can only exercise *text-domain* parsing
(e.g. tool-call extraction). It cannot exercise the *token-domain* reasoning
path, which needs real token ids -- use the token echo (``EchoTransformer`` with
a real tokenizer) for that.

The character-at-a-time replay is deliberate: it splits every delimiter across
chunk boundaries, exercising the streaming parser's partial-marker holdback the
way a real token-by-token decode would.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from max.experimental.cascade.core import Worker, worker_method
from max.experimental.cascade.interfaces.gen_ai import (
    ChatMessage,
    GenAIChunk,
    GenAIRequest,
    GenAITextChunk,
    Modality,
)
from max.experimental.cascade.interfaces.pipeline import GenAIPipeline
from max.experimental.cascade.pipelines.chat_parser import (
    ChatParserConfig,
    ChatParserWorker,
)


def _prompt_text(messages: list[ChatMessage]) -> str:
    """Pull the raw text to replay from a conversation.

    The last message's content is the scripted model output (a test sends it as
    the user turn).
    """
    if not messages:
        return ""
    return "".join(
        part.text
        for part in messages[-1].content
        if isinstance(part, GenAITextChunk)
    )


class TextEchoWorker(Worker):
    """Replay a string one character per chunk, doing no model compute.

    Text-domain counterpart to
    :class:`~max.experimental.cascade.pipelines.echo_textgen.EchoTransformer`:
    that worker replays token ids (and needs a tokenizer to detokenize them);
    this one replays the raw text directly, so no tokenizer is involved.

    Deployed to the ``gpu`` pool -- the placement a real model worker takes --
    so the replayed stream crosses the same worker/process boundary the
    production pipeline does.
    """

    def __init__(self) -> None:
        super().__init__(deploy_hints=["gpu"])

    @worker_method()
    async def replay(self, text: str) -> AsyncIterator[str]:
        """Stream ``text`` back one character at a time."""
        for char in text:
            yield char


class TextEchoPipeline(GenAIPipeline):
    """Cascade pipeline that echoes a prompt's text verbatim, skipping the tokenizer.

    The token-echo :class:`~max.experimental.cascade.pipelines.echo_textgen.EchoTextGenPipeline`
    pairs a real tokenizer with ``EchoTransformer``; this pipeline drops the
    tokenizer and pairs :class:`TextEchoWorker` with a parser worker, so the
    prompt text is the exact "model output" being parsed (see the module
    docstring for when to reach for which).

    The response format (reasoning delimiters, tool parser) is fixed at
    construction via a :class:`ChatParserConfig`, letting a single echo pipeline
    stand in for any reasoning / tool-calling model in serve-path tests.
    """

    def __init__(self, config: ChatParserConfig) -> None:
        self.transformer = TextEchoWorker()
        self.parser = ChatParserWorker(config)

    def supported_input_modalities(self) -> set[Modality]:
        """Accept text prompts."""
        return {Modality.TEXT}

    def supported_output_modalities(self) -> set[Modality]:
        """Emit assistant text, reasoning, and tool calls."""
        return {Modality.TEXT}

    async def _generate_iterator(
        self, req: GenAIRequest
    ) -> AsyncIterable[GenAIChunk]:
        """Replay the prompt's scripted output and parse it into chunks."""
        text = await self.transformer.replay(_prompt_text(req.messages))
        return await self.parser.parse_stream(
            text, req.tools_enabled, req.tool_schemas()
        )
