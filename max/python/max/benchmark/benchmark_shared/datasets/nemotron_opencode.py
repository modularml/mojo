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

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence
from typing import Any

import msgspec
from huggingface_hub import HfFileSystem
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ._tokenizer_pool import TokenizerPool
from .distribution import DistributionParameter
from .huggingface import HuggingFaceBenchmarkDataset
from .multiturn_distribution_fit import build_chat_samples_from_user_text_pool
from .types import (
    ChatMessage,
    ChatSamples,
    ChatSession,
    RequestSamples,
    SampledRequest,
    SessionMessage,
    TextContentBlock,
    estimate_num_tokens,
)

logger = logging.getLogger(__name__)


NEMOTRON_OPENCODE_REPO_ID = "nvidia/Nemotron-SFT-OpenCode-v1"

# Each subset is a separate ``<subset>/data.jsonl`` LFS blob (3-5 GB). The
# default is the smallest one so first-time runs do not block on downloading
# the entire 30 GB corpus when only a few thousand prompts are needed.
NEMOTRON_OPENCODE_SUBSETS: tuple[str, ...] = (
    "general",
    "bash_only_tool",
    "bash_only_tool_skills",
    "question_tool",
    "agent_skills",
    "agent_skills_question_tool",
)
DEFAULT_NEMOTRON_OPENCODE_SUBSET = "agent_skills_question_tool"

# Roles that are safe to forward as-is when ``enable_tool_calls=False``.
_CHAT_ONLY_ROLES = frozenset({"system", "user", "assistant"})

# Cap on rows we will stream before giving up trying to build a sample. The
# dataset has hundreds of thousands of rows; this is just defensive headroom
# so we do not loop forever if every row is filtered out by user options.
_MAX_STREAM_ROWS_PER_REQUEST = 50


class MessageContentPart(msgspec.Struct):
    type: str = ""
    text: str = ""


class Message(msgspec.Struct):
    role: str = ""
    content: str | list[MessageContentPart] | None = None


class AnthropicTool(msgspec.Struct):
    id: str = ""
    description: str = ""
    # ``inputSchema`` is either ``{"jsonSchema": {...}}`` (Anthropic's wrapper)
    # or the JSON Schema dict directly. The inner contents are an arbitrary
    # JSON Schema document, so the leaf is left as a raw dict.
    inputSchema: dict[str, Any] | None = None


class NemotronOpenCodeRow(msgspec.Struct):
    messages: list[Message] | None = None
    tools: list[AnthropicTool] | None = None


def _anthropic_tool_to_openai(tool: AnthropicTool) -> dict[str, Any]:
    """Translate an Anthropic-style tool schema to OpenAI function-tool shape.

    The nemotron-opencode corpus records OpenCode CLI sessions whose ``tools``
    field uses Anthropic's tool schema (``{id, description, inputSchema:
    {jsonSchema: ...}}``). The MAX OpenAI-compatible server validates against
    ``ChatCompletionFunctionToolParam`` which expects ``{type: "function",
    function: {name, description, parameters}}``. Without translation every
    request fails with ~120 Pydantic validation errors per call.
    """
    input_schema = tool.inputSchema
    if isinstance(input_schema, dict) and "jsonSchema" in input_schema:
        parameters = input_schema["jsonSchema"]
    elif isinstance(input_schema, dict):
        # Defensive fallback: some rows may place the JSON Schema directly
        # under ``inputSchema`` without the ``jsonSchema`` wrapper.
        parameters = input_schema
    else:
        parameters = {}
    return {
        "type": "function",
        "function": {
            "name": tool.id,
            "description": tool.description,
            "parameters": parameters,
        },
    }


class NemotronOpenCodeBenchmarkDataset(HuggingFaceBenchmarkDataset):
    """Benchmark dataset backed by ``nvidia/Nemotron-SFT-OpenCode-v1``.

    The corpus contains ~459K synthetic agentic-coding traces generated with
    Qwen3-Coder-480B inside the OpenCode CLI harness. Each row is a single
    multi-turn conversation laid out as:

    - ``messages``: list of ``{"role", "content"}`` entries covering the full
      system/user/assistant/tool exchange.
    - ``tools``: list of Anthropic-style tool schemas (``{id, description,
      inputSchema}``) the model was permitted to call during the trace.
      Translated to OpenAI ``{type, function}`` shape via
      :func:`_anthropic_tool_to_openai` before being forwarded on the wire.
    - ``enabled_tools``, ``agent_prompt``, ``skills_path``, ``uuid``,
      ``question_category``, ``complexity_level``, ``hf_split``: trace
      metadata.

    The data lives in six per-subset ``<subset>/data.jsonl`` LFS blobs
    (3-5 GB each). To avoid downloading 30 GB before the first prompt is
    issued, this dataset streams the blob over HTTP with
    :class:`huggingface_hub.HfFileSystem` and decodes one JSONL line at a time,
    pulling only enough rows to satisfy the requested sample count.

    Coverage gap vs. existing datasets:

    - ``instruct-coder`` is pure single-turn instruction/edit and has no tool
      messages.
    - ``agentic-code`` replays 22 recorded Claude Code sessions; it includes
      tool/assistant messages but ships only a few dozen distinct prompts.

    This dataset is ~3 orders of magnitude larger and exposes a much wider
    spread of agentic-coding prompt shapes. Single-turn requests
    (``sample_requests``) carry the row's tool definitions on
    :attr:`SampledRequest.tools`, which the chat-completions driver forwards
    as the OpenAI ``tools=[...]`` field — pass ``enable_tool_calls=False``
    to suppress this. The same schemas are also kept on
    :attr:`last_loaded_tool_schemas` for inspection.

    Multi-turn sessions still drop tool definitions because
    :class:`SessionMessage` does not yet carry a per-turn tools field.
    """

    #: Subset name (one of :data:`NEMOTRON_OPENCODE_SUBSETS`). Override before
    #: calling :meth:`sample_requests` to pull from a different split.
    subset: str = DEFAULT_NEMOTRON_OPENCODE_SUBSET

    #: Tool schemas captured from the most recent ``sample_requests`` /
    #: ``gen_multiturn_sessions`` call. One entry per emitted request/session.
    #: Empty until the first call. Exposed for follow-up work that wires
    #: ``tools`` through the request payload.
    last_loaded_tool_schemas: list[list[AnthropicTool]]

    def __init__(self) -> None:
        self.last_loaded_tool_schemas = []

    def fetch(self) -> None:
        # Streaming download happens lazily inside ``_iter_rows``; there is no
        # single file to materialise up front and we deliberately avoid eager
        # downloads since each subset is multiple GB.
        #
        # The streaming loader cannot be pointed at a local JSONL via
        # ``--dataset-path`` today: each row is several KB of nested JSON and
        # the row->SampledRequest pipeline reads the schema with HF column
        # names, so a hand-curated local file would have to match the upstream
        # layout exactly. Surface that explicitly rather than silently
        # ignoring the flag.
        if self.dataset_path is not None:
            raise ValueError(
                "nemotron-opencode does not support --dataset-path; the "
                "loader streams "
                f"{NEMOTRON_OPENCODE_REPO_ID} via "
                "huggingface_hub.HfFileSystem. Drop --dataset-path "
                "and let the dataset stream from HuggingFace."
            )

    # ------------------------------------------------------------------ utils

    def _iter_rows(self) -> Iterable[NemotronOpenCodeRow]:
        """Stream rows from the configured subset.

        Yields decoded ``NemotronOpenCodeRow`` structs in source order (the HF
        stream does not shuffle). Rows that fail msgspec decoding/validation are
        silently skipped so a single malformed entry does not abort the
        benchmark. Callers are responsible for breaking out of the iterator
        once they have enough samples.

        Each JSONL line is decoded independently with msgspec rather than
        through ``datasets.load_dataset``. The upstream loader infers a single
        Arrow schema across rows via pyarrow, which cannot represent
        ``messages[].content`` — that field is a plain ``str`` in some rows and
        a list of content blocks in others. pyarrow aborts with
        ``ArrowInvalid: Column(/messages/[]/content) changed from string to
        array`` and ``datasets`` then falls back to a whole-file
        ``pandas.read_json`` that chokes on the newline-delimited blob with
        ``ValueError: Trailing data``. Decoding line-by-line sidesteps the
        cross-row schema unification entirely, and ``NemotronOpenCodeRow``
        already models the union type.
        """
        decoder = msgspec.json.Decoder(NemotronOpenCodeRow)
        for line in self._stream_jsonl_lines():
            try:
                yield decoder.decode(line)
            except (msgspec.ValidationError, msgspec.DecodeError):
                continue

    def _stream_jsonl_lines(self) -> Iterable[bytes]:
        """Stream raw JSONL byte lines for the configured subset.

        Reads the per-subset ``<subset>/data.jsonl`` blob over HTTP via
        :class:`huggingface_hub.HfFileSystem` without materialising the whole
        multi-GB file, yielding one stripped, non-empty line at a time.
        """
        if self.subset not in NEMOTRON_OPENCODE_SUBSETS:
            raise ValueError(
                f"Unknown Nemotron-OpenCode subset {self.subset!r}; expected "
                f"one of {sorted(NEMOTRON_OPENCODE_SUBSETS)}"
            )
        # NOTE: ``self.dataset_name`` is the registry key (e.g.
        # "nemotron-opencode"), not the HuggingFace repo id, so we
        # deliberately use the module constant here. Dataset repos live under
        # the ``datasets/`` namespace on the HF filesystem.
        path = f"datasets/{NEMOTRON_OPENCODE_REPO_ID}/{self.subset}/data.jsonl"
        fs = HfFileSystem()
        with fs.open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    @staticmethod
    def _message_text(
        content: str | list[MessageContentPart] | None,
    ) -> str:
        """Flatten a message ``content`` field to a plain string."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return "".join(part.text for part in content)

    @classmethod
    def _to_chat_messages(
        cls,
        messages: Sequence[Message],
    ) -> list[ChatMessage]:
        """Convert raw dataset messages into ``ChatMessage`` objects.

        Tool/assistant messages with non-string content are flattened into a
        single ``TextContentBlock`` so they round-trip through the chat
        completions driver, which expects ``str`` or ``list[TextContentBlock]``.
        """
        chat_messages: list[ChatMessage] = []
        for msg in messages:
            if not msg.role:
                continue
            text = cls._message_text(msg.content)
            chat_messages.append(
                ChatMessage(
                    role=msg.role, content=[TextContentBlock(text=text)]
                )
            )
        return chat_messages

    @staticmethod
    def _split_prompt_and_completion(
        messages: Sequence[Message],
    ) -> tuple[list[Message], str] | None:
        """Split a conversation into (prompt-messages, last-assistant-text).

        Returns ``None`` if the trace does not end on a non-empty assistant
        reply (e.g. truncated traces or traces that only contain tool calls).
        """
        last_assistant_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.role != "assistant":
                continue
            text = NemotronOpenCodeBenchmarkDataset._message_text(msg.content)
            if text.strip():
                last_assistant_idx = i
                break
        if last_assistant_idx is None or last_assistant_idx == 0:
            return None
        prompt_messages = list(messages[:last_assistant_idx])
        completion = NemotronOpenCodeBenchmarkDataset._message_text(
            messages[last_assistant_idx].content
        )
        return prompt_messages, completion

    # ------------------------------------------------------------ single turn

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        enable_tool_calls: bool = True,
        min_prompt_len: int = 4,
        min_output_len: int = 4,
        **kwargs,
    ) -> RequestSamples:
        """Sample single-turn requests by collapsing each row.

        Each row in the source is treated as one sample: the message history
        up to (but excluding) the final assistant message is sent as the
        prompt, and the final assistant message's tokenised length is used as
        the reference ``output_len``.

        Args:
            num_requests: Number of samples to emit.
            tokenizer: Tokenizer for computing token lengths.
            output_lengths: Optional pinned per-request output lengths. When
                set, ``ignore_eos`` is ``True`` and the dataset's reference
                completion length is ignored.
            shuffle: Whether to shuffle the buffered candidates before
                sampling. Note that the HF stream itself is not random; this
                only shuffles within an oversampled buffer.
            enable_tool_calls: When ``False``, rows whose prompt history
                contains a ``tool`` or non-``user/assistant/system`` role are
                skipped. The corpus is heavily tool-flavoured so disabling
                this discards most rows.
            min_prompt_len: Floor on tokenised prompt length.
            min_output_len: Floor on tokenised completion length (ignored when
                ``output_lengths`` is provided).
        """
        if num_requests <= 0:
            self.last_loaded_tool_schemas = []
            return RequestSamples(requests=[])

        # Oversample so ``shuffle=True`` is not a no-op; cap defensively so we
        # do not iterate the full multi-GB stream when only a handful of rows
        # are needed.
        candidate_count = min(
            max(num_requests * 4, num_requests + 32),
            num_requests + _MAX_STREAM_ROWS_PER_REQUEST * num_requests,
        )

        candidates: list[tuple[list[Message], str, list[AnthropicTool]]]
        candidates = []
        for row in self._iter_rows():
            if len(candidates) >= candidate_count:
                break
            if not row.messages:
                continue
            if not enable_tool_calls and any(
                m.role not in _CHAT_ONLY_ROLES for m in row.messages
            ):
                continue
            split = self._split_prompt_and_completion(row.messages)
            if split is None:
                continue
            prompt_messages, completion = split
            tools_raw = row.tools or []
            candidates.append((prompt_messages, completion, tools_raw))

        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "Shuffling with pinned output_lengths is not supported"
                )
            random.shuffle(candidates)

        sampled: list[SampledRequest] = []
        sampled_tool_schemas: list[list[AnthropicTool]] = []
        for prompt_messages, completion, tools_raw in candidates:
            if len(sampled) >= num_requests:
                break
            chat_messages = self._to_chat_messages(prompt_messages)
            if not chat_messages:
                continue
            # Concatenated text token count is an approximation; the exact
            # wire length depends on the chat template, which is model-
            # specific. This is consistent with how other multi-message
            # datasets in this package estimate prompt_len.
            prompt_text = "\n".join(
                m.text for m in _iter_text_blocks(chat_messages)
            )
            prompt_len = estimate_num_tokens(tokenizer, prompt_text)
            if prompt_len < min_prompt_len:
                continue
            out_idx = len(sampled)
            if output_lengths is not None:
                output_len = output_lengths[out_idx]
            else:
                output_len = estimate_num_tokens(tokenizer, completion)
                if output_len < min_output_len:
                    continue
            sampled.append(
                SampledRequest(
                    prompt_formatted=chat_messages,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_images=[],
                    # ``output_len`` is always set on this path, so pin
                    # ``ignore_eos=True`` to match sharegpt / arxiv: decode the
                    # full target length instead of letting an early EOS skew
                    # throughput vs. instruct-coder baselines.
                    #
                    # Note: This plus the reference-length ``output_len`` stops
                    # generation mid-argument, so tool-call conformance errors
                    # are expected here. Removing both restrictions showed
                    # no tool call errors for Kimi K2.7.
                    ignore_eos=True,
                    tools=(
                        [_anthropic_tool_to_openai(t) for t in tools_raw]
                        if (enable_tool_calls and tools_raw)
                        else None
                    ),
                )
            )
            sampled_tool_schemas.append(tools_raw)

        self.last_loaded_tool_schemas = sampled_tool_schemas

        if len(sampled) < num_requests:
            logger.warning(
                "nemotron-opencode: only %d valid rows yielded after streaming "
                "%d candidates from subset %r; requested %d.",
                len(sampled),
                len(candidates),
                self.subset,
                num_requests,
            )
        return RequestSamples(requests=sampled)

    # --------------------------------------------------------------- multiturn

    def gen_multiturn_sessions(
        self,
        num_sessions: int,
        tokenizer: PreTrainedTokenizerBase,
        max_turns_per_session: int | None = None,
        shuffle: bool = True,
        *,
        pool: TokenizerPool | None = None,
        fit_length_distributions: bool = False,
        num_turns: DistributionParameter | None = None,
        input_len: DistributionParameter | None = None,
        output_len: DistributionParameter | None = None,
        delay_between_turns_dist: DistributionParameter | None = None,
        sys_prompt_ratio: float = 0.0,
        max_num_unique_sys_prompt: int = 1,
        min_input_len: int = 4,
        min_output_len: int = 1,
        enable_tool_calls: bool = True,
    ) -> ChatSamples:
        """Build :class:`ChatSamples` from streamed rows.

        Each row becomes one :class:`ChatSession`: consecutive user/assistant
        pairs from the recorded conversation become measured turns. Tool and
        system messages are dropped (the chat-session abstraction only models
        user/assistant alternation). Set ``enable_tool_calls=False`` to also
        skip any row whose conversation contains a non-user/assistant/system
        message before flattening.

        When ``fit_length_distributions=True``, behaves like the
        ``instruct-coder`` / ``agentic-code`` "fit" path: user texts are
        pooled and synthetic sessions are built to match
        ``num_turns`` / ``input_len`` / ``output_len`` distributions.
        """
        if fit_length_distributions:
            assert num_turns is not None, "num_turns required when fitting"
            assert input_len is not None, "input_len required when fitting"
            assert output_len is not None, "output_len required when fitting"
            if pool is None:
                raise ValueError(
                    "pool is required for nemotron-opencode "
                    "fit-distributions multiturn"
                )
            user_texts = self._collect_user_turn_texts(
                enable_tool_calls=enable_tool_calls,
                num_sessions=num_sessions,
            )
            if shuffle:
                random.shuffle(user_texts)
            return build_chat_samples_from_user_text_pool(
                pool=pool,
                user_text_pool=user_texts,
                num_sessions=num_sessions,
                num_turns=num_turns,
                input_len=input_len,
                output_len=output_len,
                delay_between_turns_dist=delay_between_turns_dist,
                sys_prompt_ratio=sys_prompt_ratio,
                max_num_unique_sys_prompt=max_num_unique_sys_prompt,
                min_input_len=min_input_len,
                min_output_len=min_output_len,
                shuffle_pool=False,
                log_prefix="nemotron-opencode",
            )

        # Oversample so shuffling is meaningful but bounded.
        candidate_count = min(
            max(num_sessions * 2, num_sessions + 8),
            num_sessions + _MAX_STREAM_ROWS_PER_REQUEST * num_sessions,
        )

        candidates: list[tuple[list[SessionMessage], list[AnthropicTool]]] = []
        for row in self._iter_rows():
            if len(candidates) >= candidate_count:
                break
            if not row.messages:
                continue
            if not enable_tool_calls and any(
                m.role not in _CHAT_ONLY_ROLES for m in row.messages
            ):
                continue
            session_messages = self._build_session_messages(
                row.messages,
                tokenizer=tokenizer,
                max_turns_per_session=max_turns_per_session,
                min_input_len=min_input_len,
                min_output_len=min_output_len,
            )
            if not session_messages:
                continue
            tools_raw = row.tools or []
            candidates.append((session_messages, tools_raw))

        if shuffle:
            random.shuffle(candidates)

        chat_sessions: list[ChatSession] = []
        session_tool_schemas: list[list[AnthropicTool]] = []
        for idx, (session_messages, tools_raw) in enumerate(candidates):
            if len(chat_sessions) >= num_sessions:
                break
            chat_sessions.append(ChatSession(idx, session_messages))
            session_tool_schemas.append(tools_raw)

        self.last_loaded_tool_schemas = session_tool_schemas

        if len(chat_sessions) < num_sessions:
            logger.warning(
                "nemotron-opencode: only %d valid sessions yielded from "
                "subset %r; requested %d.",
                len(chat_sessions),
                self.subset,
                num_sessions,
            )
        return ChatSamples(chat_sessions=chat_sessions)

    def _build_session_messages(
        self,
        messages: Sequence[Message],
        *,
        tokenizer: PreTrainedTokenizerBase,
        max_turns_per_session: int | None,
        min_input_len: int,
        min_output_len: int,
    ) -> list[SessionMessage]:
        """Project a recorded conversation onto alternating user/assistant turns."""
        session_messages: list[SessionMessage] = []
        pending_user: str | None = None
        turns = 0
        for msg in messages:
            role = msg.role
            text = self._message_text(msg.content)
            if not text.strip():
                continue
            if role == "user":
                # Coalesce consecutive user-side messages (e.g. tool outputs
                # surfaced back to the user) into one turn.
                pending_user = (
                    text if pending_user is None else f"{pending_user}\n{text}"
                )
            elif role == "assistant":
                if pending_user is None:
                    continue
                user_tokens = estimate_num_tokens(tokenizer, pending_user)
                assistant_tokens = estimate_num_tokens(tokenizer, text)
                if (
                    user_tokens < min_input_len
                    or assistant_tokens < min_output_len
                ):
                    pending_user = None
                    continue
                session_messages.append(
                    SessionMessage(
                        source="user",
                        content=pending_user,
                        num_tokens=user_tokens,
                    )
                )
                session_messages.append(
                    SessionMessage(
                        source="assistant",
                        content="",
                        num_tokens=assistant_tokens,
                    )
                )
                pending_user = None
                turns += 1
                if (
                    max_turns_per_session is not None
                    and turns >= max_turns_per_session
                ):
                    break
            # system / tool / other roles are dropped: they belong to the
            # transcript, not the user/assistant alternation that ChatSession
            # models.
        return session_messages

    def _collect_user_turn_texts(
        self,
        *,
        enable_tool_calls: bool,
        num_sessions: int,
    ) -> list[str]:
        """Flatten user-side messages across rows into a pool of texts."""
        target = max(num_sessions * 8, num_sessions + 32)
        texts: list[str] = []
        for row in self._iter_rows():
            if len(texts) >= target:
                break
            if not row.messages:
                continue
            if not enable_tool_calls and any(
                m.role not in _CHAT_ONLY_ROLES for m in row.messages
            ):
                continue
            for msg in row.messages:
                if msg.role != "user":
                    continue
                text = self._message_text(msg.content)
                if text.strip():
                    texts.append(text)
        return texts


def _iter_text_blocks(
    chat_messages: Sequence[ChatMessage],
) -> Iterable[TextContentBlock]:
    for msg in chat_messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, TextContentBlock):
                    yield part
        elif isinstance(msg.content, str):
            yield TextContentBlock(text=msg.content)
