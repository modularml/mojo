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

import json
import logging
import random
from collections.abc import Sequence

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ._hf_download import hf_hub_download_with_retry
from ._tokenizer_pool import TokenizerPool
from .distribution import DistributionParameter
from .huggingface import HuggingFaceBenchmarkDataset
from .multiturn_distribution_fit import build_chat_samples_from_user_text_pool
from .types import (
    ChatSamples,
    ChatSession,
    RequestSamples,
    SampledRequest,
    SessionMessage,
    estimate_num_tokens,
)

logger = logging.getLogger(__name__)


class InstructCoderBenchmarkDataset(HuggingFaceBenchmarkDataset):
    """Benchmark dataset backed by the likaixin/InstructCoder HuggingFace dataset.

    The dataset contains instruction-input-output triplets for 20 distinct
    code-editing scenarios. Each entry has:

    - ``instruction``: a natural-language description of the editing task.
    - ``input``: the source code to be edited.
    - ``output``: the expected edited code.

    The prompt sent to the model concatenates the instruction and input code,
    while the output field determines the expected output length.

    Both single-turn (``sample_requests``) and multi-turn
    (``gen_multiturn_sessions``) benchmarking modes are supported.
    In multi-turn mode, consecutive code-editing tasks are grouped into
    chat sessions that simulate an iterative coding assistant conversation.

    With ``fit_length_distributions=True`` (CLI: ``--fit-distributions`` with
    ``--num-chat-sessions``), turns per session, per-turn input and output token
    targets, and inter-turn delays follow the same distribution parameters as the
    synthetic ``random`` multiturn workload. Prompts are expanded or truncated by
    repeating token sequences derived from each InstructCoder example (and
    optional synthetic system prefixes).
    """

    def fetch(self) -> None:
        if self.dataset_path is not None:
            return
        self.dataset_path = hf_hub_download_with_retry(
            repo_id="likaixin/InstructCoder",
            filename="train.json",
            repo_type="dataset",
        )

    def _load_pairs(self) -> list[tuple[str, str]]:
        """Load and return (prompt, completion) pairs from the dataset file."""
        assert self.dataset_path is not None, (
            "dataset_path must be set before loading"
        )

        with open(self.dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        pairs: list[tuple[str, str]] = []
        for entry in dataset:
            instruction = entry.get("instruction", "").strip()
            code_input = entry.get("input", "").strip()
            output = entry.get("output", "").strip()
            if not instruction or not output:
                continue
            if code_input:
                prompt = f"{instruction}\n\n{code_input}"
            else:
                prompt = instruction
            pairs.append((prompt, output))
        return pairs

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        **kwargs,
    ) -> RequestSamples:
        """Sample single-turn requests from the InstructCoder dataset.

        Args:
            num_requests: Number of requests to sample.
            tokenizer: Tokenizer for computing token lengths.
            output_lengths: Optional per-request output lengths. When
                ``None``, the actual tokenized length of the ``output`` field
                is used.
            shuffle: Whether to shuffle the dataset before sampling.
        """
        pairs = self._load_pairs()

        if shuffle:
            if output_lengths is not None:
                raise NotImplementedError(
                    "TODO: Add support for shuffling + pinned output lengths"
                )
            random.shuffle(pairs)

        filtered_dataset: list[SampledRequest] = []
        for i in range(len(pairs)):
            if len(filtered_dataset) == num_requests:
                break

            prompt, completion = pairs[i]
            prompt_token_ids = tokenizer.encode(prompt)
            completion_token_ids = tokenizer.encode(completion)
            prompt_len = len(prompt_token_ids)
            output_len = (
                len(completion_token_ids)
                if output_lengths is None
                else output_lengths[len(filtered_dataset)]
            )
            assert output_len is not None, "Unexpected null output length"

            # Filter degenerate samples to keep the benchmark workload realistic
            if prompt_len < 4:
                continue
            if output_lengths is None and output_len < 4:
                continue

            filtered_dataset.append(
                SampledRequest(
                    prompt_formatted=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    encoded_images=[],
                    ignore_eos=(output_len is not None),
                )
            )

        return RequestSamples(requests=filtered_dataset)

    def gen_multiturn_sessions(
        self,
        num_sessions: int,
        tokenizer: PreTrainedTokenizerBase,
        turns_per_session: int = 5,
        delay_between_chat_turns: float | None = None,
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
    ) -> ChatSamples:
        """Generate multi-turn chat sessions from InstructCoder entries.

        Consecutive code-editing tasks are grouped into chat sessions.
        Each turn sends the instruction (+input) as the user message and
        uses the reference output length for the expected assistant reply,
        unless ``fit_length_distributions`` is enabled (then lengths and delays
        are sampled from distributions).

        Args:
            num_sessions: Number of chat sessions to produce.
            tokenizer: Tokenizer for computing token lengths.
            turns_per_session: User/assistant round-trips per session when
                ``fit_length_distributions`` is False.
            delay_between_chat_turns: Delay in ms after each assistant message
                when ``fit_length_distributions`` is False (constant per turn).
            shuffle: Whether to shuffle entries before grouping.
            fit_length_distributions: If True, use ``num_turns``, ``input_len``,
                ``output_len``, and ``delay_between_turns_dist`` like the random
                multiturn dataset (including ``';'``-separated first vs remaining
                turn specs).
            num_turns: Distribution for turns per session when fitting.
            input_len: Distribution for user-side token targets per turn.
            output_len: Distribution for assistant ``max_tokens`` per turn.
            delay_between_turns_dist: Optional per-turn delay distribution (ms).
            sys_prompt_ratio: When fitting, fraction of each user message's
                token budget used for a synthetic system prefix.
            max_num_unique_sys_prompt: Number of system-prefix variants to cycle.
            min_input_len: Floor for sampled input lengths.
            min_output_len: Floor for sampled output lengths.

        Returns:
            ChatSamples containing the generated sessions.
        """
        pairs = self._load_pairs()

        if shuffle:
            random.shuffle(pairs)

        if fit_length_distributions:
            assert num_turns is not None, "num_turns is required when fitting"
            assert input_len is not None, "input_len is required when fitting"
            assert output_len is not None, "output_len is required when fitting"
            if pool is None:
                raise ValueError(
                    "pool is required for InstructCoder fit-distributions multiturn"
                )
            return self._gen_multiturn_sessions_from_distributions(
                num_sessions=num_sessions,
                pool=pool,
                pairs=pairs,
                num_turns=num_turns,
                input_len=input_len,
                output_len=output_len,
                delay_between_turns_dist=delay_between_turns_dist,
                sys_prompt_ratio=sys_prompt_ratio,
                max_num_unique_sys_prompt=max_num_unique_sys_prompt,
                min_input_len=min_input_len,
                min_output_len=min_output_len,
            )

        # Pre-tokenize to get lengths and filter unusable entries.
        tokenized: list[tuple[str, int, int]] = []
        for prompt, completion in pairs:
            prompt_len = estimate_num_tokens(tokenizer, prompt)
            output_len = estimate_num_tokens(tokenizer, completion)
            if prompt_len < 4 or output_len < 4:
                continue
            tokenized.append((prompt, prompt_len, output_len))

        sessions: list[ChatSession] = []
        idx = 0
        for session_id in range(num_sessions):
            if idx >= len(tokenized):
                break

            messages: list[SessionMessage] = []
            for _ in range(turns_per_session):
                if idx >= len(tokenized):
                    break

                prompt, prompt_len, output_len = tokenized[idx]
                idx += 1

                messages.append(
                    SessionMessage(
                        source="user",
                        content=prompt,
                        num_tokens=prompt_len,
                    )
                )
                messages.append(
                    SessionMessage(
                        source="assistant",
                        content="",
                        num_tokens=output_len,
                        delay_until_next_message=delay_between_chat_turns,
                    )
                )

            if len(messages) >= 2:
                sessions.append(ChatSession(session_id, messages))

        if len(sessions) < num_sessions:
            logger.warning(
                "instruct-coder: only %d sessions could be formed from "
                "available data, requested %d.",
                len(sessions),
                num_sessions,
            )

        return ChatSamples(chat_sessions=sessions)

    def _gen_multiturn_sessions_from_distributions(
        self,
        num_sessions: int,
        pool: TokenizerPool,
        pairs: list[tuple[str, str]],
        num_turns: DistributionParameter,
        input_len: DistributionParameter,
        output_len: DistributionParameter,
        delay_between_turns_dist: DistributionParameter | None,
        sys_prompt_ratio: float,
        max_num_unique_sys_prompt: int,
        min_input_len: int,
        min_output_len: int,
    ) -> ChatSamples:
        """Build multiturn sessions with sampled lengths (see ``gen_multiturn_sessions``)."""
        return build_chat_samples_from_user_text_pool(
            pool=pool,
            user_text_pool=[p for p, _ in pairs],
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
            log_prefix="instruct-coder",
        )
