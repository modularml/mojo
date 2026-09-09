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
from collections.abc import Sequence
from typing import Any, Literal

import msgspec
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ._hf_download import hf_hub_download_with_retry
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
)

logger = logging.getLogger(__name__)


class MessageContentPart(msgspec.Struct):
    type: Literal["text"]
    text: str = ""


class Message(msgspec.Struct):
    role: str
    content: str | list[MessageContentPart] | None = None


class Turn(msgspec.Struct):
    messages: list[Message] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: Any = None
    status_code: int = 200


class Session(msgspec.Struct):
    turns: list[Turn]


class AgenticCodeData(msgspec.Struct):
    sessions: list[Session]


def _to_chat_messages(messages: list[Message]) -> list[ChatMessage]:
    """Converts a list of dataset ``Message`` objects to ``ChatMessage`` objects.

    Flattens list-of-parts content into a list of ``TextContentBlock`` values;
    plain string content is passed through as-is (defaulting to ``""`` when
    ``None``).

    Args:
        messages: Raw messages decoded from the agentic-code dataset.

    Returns:
        A list of ``ChatMessage`` objects suitable for use in a
        ``SampledRequest``.
    """
    return [
        ChatMessage(
            role=msg.role,
            content=(
                [
                    TextContentBlock(type=part.type, text=part.text)
                    for part in msg.content
                ]
                if isinstance(msg.content, list)
                else (msg.content or "")
            ),
        )
        for msg in messages
    ]


class AgenticCodeBenchmarkDataset(HuggingFaceBenchmarkDataset):
    """Benchmark dataset from novita/agentic_code_dataset_22.

    22 real Claude Code sessions converted to OpenAI chat format.
    Each sample is a full turn with the complete message history,
    using pre-recorded input/output token counts from the original sessions.
    """

    def fetch(self) -> None:
        self.dataset_path = hf_hub_download_with_retry(
            repo_id="novita/agentic_code_dataset_22",
            filename="e22_sessions_openai.json",
            repo_type="dataset",
        )

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        enable_tool_calls: bool = True,
        **kwargs,
    ) -> RequestSamples:
        assert self.dataset_path is not None, (
            "dataset_path must be set; call fetch() first"
        )

        with open(self.dataset_path, "rb") as f:
            data = msgspec.json.decode(f.read(), type=AgenticCodeData)

        # Flatten all turns across all sessions into
        # (messages, input_tokens, output_tokens)
        turns: list[tuple[list[Message], int, int]] = []
        for session in data.sessions:
            for turn in session.turns:
                messages = turn.messages
                input_tokens = turn.input_tokens
                output_tokens = turn.output_tokens
                # Skip turns without messages or token counts
                if (
                    not messages
                    or input_tokens is None
                    or output_tokens is None
                ):
                    continue
                # Skip turns with errors
                if turn.error or turn.status_code != 200:
                    continue
                # Optionally skip turns that include tool calls
                if not enable_tool_calls:
                    allowed_roles = {"system", "user"}
                    if any(m.role not in allowed_roles for m in messages):
                        continue
                turns.append((messages, input_tokens, output_tokens))

        if output_lengths is not None and len(output_lengths) < len(turns):
            raise ValueError(
                f"output_lengths has {len(output_lengths)} entries but "
                f"there are {len(turns)} valid turns; "
                "output_lengths must be at least as long as the number of valid turns"
            )

        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "Shuffling with pinned output_lengths is not supported"
                )
            random.shuffle(turns)

        sampled: list[SampledRequest] = []
        for i, (messages, input_tokens, output_tokens) in enumerate(turns):
            if len(sampled) >= num_requests:
                break
            out_len = (
                output_tokens if output_lengths is None else output_lengths[i]
            )
            sampled.append(
                SampledRequest(
                    prompt_formatted=_to_chat_messages(messages),
                    prompt_len=input_tokens,
                    output_len=out_len,
                    encoded_images=[],
                    ignore_eos=(output_lengths is not None),
                )
            )

        if len(sampled) < num_requests:
            logger.warning(
                "agentic-code: only %d valid turns available, "
                "requested %d. Returning fewer requests.",
                len(sampled),
                num_requests,
            )

        return RequestSamples(requests=sampled)

    def _collect_user_turn_texts(self, enable_tool_calls: bool) -> list[str]:
        """Flatten all valid user messages into strings (one per recorded turn)."""
        assert self.dataset_path is not None, (
            "dataset_path must be set; call fetch() first"
        )
        with open(self.dataset_path, "rb") as f:
            data = msgspec.json.decode(f.read(), type=AgenticCodeData)

        texts: list[str] = []
        for session in data.sessions:
            for turn in session.turns:
                if turn.error or turn.status_code != 200:
                    continue
                input_tokens = turn.input_tokens
                output_tokens = turn.output_tokens
                if input_tokens is None or output_tokens is None:
                    continue
                turn_messages = turn.messages or []
                if not enable_tool_calls:
                    allowed_roles = {"system", "user"}
                    if any(m.role not in allowed_roles for m in turn_messages):
                        continue
                user_content = next(
                    (
                        m.content
                        for m in reversed(turn_messages)
                        if m.role == "user"
                    ),
                    None,
                )
                if user_content is None:
                    continue
                if isinstance(user_content, list):
                    user_text = " ".join(
                        part.text
                        for part in user_content
                        if part.type == "text"
                    )
                else:
                    user_text = user_content
                if not user_text.strip():
                    continue
                texts.append(user_text)
        return texts

    def gen_multiturn_sessions(
        self,
        num_sessions: int,
        tokenizer: PreTrainedTokenizerBase | None = None,
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
        """Generate multiturn ChatSessions from the agentic-code dataset.

        By default each session mirrors one recorded Claude Code session. With
        ``fit_length_distributions=True``, user turns are taken from a global
        pool (optionally shuffled) and grouped into synthetic sessions sized by
        ``num_turns``, with per-turn lengths and delays matching the random
        multiturn benchmark (same as ``--fit-distributions`` for instruct-coder).

        Args:
            num_sessions: Number of sessions to sample.
            tokenizer: Required when ``fit_length_distributions`` is True.
            max_turns_per_session: Cap on turns per session when not fitting
                distributions. Ignored when fitting.
            shuffle: Whether to shuffle before sampling (session order when not
                fitting; user-turn pool when fitting).
            fit_length_distributions: Build sessions from pooled turns and CLI
                distributions instead of replaying full recorded sessions.
            num_turns: Turns per session distribution when fitting.
            input_len: Target user token count distribution(s) when fitting.
            output_len: Target assistant token count distribution(s) when fitting.
            delay_between_turns_dist: Optional inter-turn delay (ms) when fitting.
            sys_prompt_ratio: Synthetic system prefix fraction when fitting.
            max_num_unique_sys_prompt: System-prefix variants when fitting.
            min_input_len: Floor for sampled input lengths when fitting.
            min_output_len: Floor for sampled output lengths when fitting.
            enable_tool_calls: When False, skip turns whose messages include
                non-system/user roles (same as ``sample_requests``).

        Returns:
            ChatSamples with one ChatSession per selected session.
        """
        assert self.dataset_path is not None, (
            "dataset_path must be set; call fetch() first"
        )

        if fit_length_distributions:
            if pool is None:
                raise ValueError(
                    "pool is required for agentic-code when "
                    "fit_length_distributions=True"
                )
            assert num_turns is not None
            assert input_len is not None
            assert output_len is not None
            user_texts = self._collect_user_turn_texts(enable_tool_calls)
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
                log_prefix="agentic-code",
            )

        with open(self.dataset_path, "rb") as f:
            data = msgspec.json.decode(f.read(), type=AgenticCodeData)

        # Build all valid sessions, then shuffle and slice.
        all_messages: list[list[SessionMessage]] = []
        for session in data.sessions:
            messages: list[SessionMessage] = []
            turns_added = 0
            for turn in session.turns:
                if (
                    max_turns_per_session is not None
                    and turns_added >= max_turns_per_session
                ):
                    break

                input_tokens = turn.input_tokens
                output_tokens = turn.output_tokens
                if input_tokens is None or output_tokens is None:
                    continue
                if turn.error or turn.status_code != 200:
                    continue

                # Extract the last user message from this turn's message list
                turn_messages = turn.messages or []
                if not enable_tool_calls:
                    allowed_roles = {"system", "user"}
                    if any(m.role not in allowed_roles for m in turn_messages):
                        continue
                user_content = next(
                    (
                        m.content
                        for m in reversed(turn_messages)
                        if m.role == "user"
                    ),
                    None,
                )
                if user_content is None:
                    continue

                # Flatten content to a plain string (handles str or list-of-parts)
                if isinstance(user_content, list):
                    user_text = " ".join(
                        part.text
                        for part in user_content
                        if part.type == "text"
                    )
                else:
                    user_text = user_content

                if not user_text.strip():
                    continue

                messages.append(
                    SessionMessage(
                        source="user",
                        content=user_text,
                        num_tokens=input_tokens,
                    )
                )
                messages.append(
                    SessionMessage(
                        source="assistant",
                        content="",  # filled by live model response
                        num_tokens=output_tokens,
                    )
                )
                turns_added += 1

            if len(messages) >= 2:
                all_messages.append(messages)

        if shuffle:
            random.shuffle(all_messages)

        chat_sessions: list[ChatSession] = [
            ChatSession(session_id, messages)
            for session_id, messages in enumerate(all_messages[:num_sessions])
        ]

        if len(chat_sessions) < num_sessions:
            logger.warning(
                "agentic-code: only %d valid sessions available, "
                "requested %d. Returning fewer sessions.",
                len(chat_sessions),
                num_sessions,
            )

        return ChatSamples(chat_sessions=chat_sessions)
