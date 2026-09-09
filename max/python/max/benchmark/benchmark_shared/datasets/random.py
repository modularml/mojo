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
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import override

from ._hf_download import hf_hub_download_with_retry
from ._tokenizer_pool import TokenizerPool, _encode_ids
from .distribution import BaseDistribution, DistributionParameter
from .local import LocalBenchmarkDataset
from .types import (
    ChatSamples,
    ChatSession,
    OpenAIImage,
    RequestSamples,
    SampledRequest,
    SharedContext,
    build_chat_message,
    encode_image,
)

logger = logging.getLogger(__name__)

# Maximum ratio of model's max context length to use for random sequences.
# Set to 95% to leave buffer room for other overheads, like re-tokenization and special tokens.
MAX_CONTEXT_USAGE_RATIO = 0.95


@dataclass
class _SysPromptDraft:
    """A request's system-prompt slice prior to decode/encode resolution.

    `ids` is populated during the construction loop; `text` is filled by
    the post-loop batched `pool.decode_texts` call, and `encoded_len` by
    the subsequent batched `pool.encode_lens` call.
    """

    idx: int
    ids: list[int]
    text: str = ""
    encoded_len: int = 0


@dataclass
class _PendingRequest:
    """Per-request scratch built during the construction loop.

    `prompt_ids` and `images` are populated in the loop. `prompt` is
    filled by the post-loop batched `pool.decode_texts` call, and
    `prompt_encoded_len` by the subsequent batched `pool.encode_lens`.
    """

    prompt_ids: list[int]
    images: list[OpenAIImage] = field(default_factory=list)
    prompt: str = ""
    prompt_encoded_len: int = 0
    sys_prompt: _SysPromptDraft | None = None


def log_request_actual_length_percentiles(
    requests: Sequence[SampledRequest],
) -> None:
    """Log percentile statistics for prompts' actual lengths."""
    prompt_lens = [req.prompt_len for req in requests]
    percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]
    prompt_percentiles = np.percentile(prompt_lens, percentiles)
    logger.info(
        f"Prompt actual length percentiles {percentiles}: {prompt_percentiles.tolist()}"
    )


class RandomBenchmarkDataset(LocalBenchmarkDataset):
    @property
    def use_synthetic_tokens(self) -> bool:
        # Overridden in SyntheticBenchmarkDataset
        return False

    def fetch(self) -> None:
        """Fetch Random dataset.

        Random datasets are generated synthetically and don't require file fetching.
        """

    def gen_multiturn_random_requests(
        self,
        input_len: DistributionParameter,
        output_len: DistributionParameter,
        num_chat_sessions: int,
        num_turns: DistributionParameter,
        delay_between_chat_turns: DistributionParameter | None,
        pool: TokenizerPool,
        sys_prompt_ratio: float,
        max_num_unique_sys_prompt: int,
        min_input_len: int = 4,
        min_output_len: int = 1,
    ) -> ChatSamples:
        """Generate multiturn random chat requests.

        Args:
            input_len: Distribution parameter for input lengths. Use ';' to
                separate first-turn and remaining-turn distributions, e.g.
                "N(10000,2000);N(1024,200)". Without ';', all turns share the
                same distribution.
            output_len: Distribution parameter for output lengths. Same ';' format
                as input_len.
            num_chat_sessions: Number of chat sessions to generate.
            num_turns: Distribution parameter for number of turns per session.
            delay_between_chat_turns: Optional Distribution parameter for delay
                between turns.
            pool: Tokenizer process pool used for parallel encode/decode work.
            sys_prompt_ratio: Ratio of system prompt to input length.
            max_num_unique_sys_prompt: Max unique system prompts.
            min_input_len: Minimum input token length.
            min_output_len: Minimum output token length.
        """
        tokenizer = pool.tokenizer
        first_turn_input: DistributionParameter
        remaining_input: DistributionParameter
        if isinstance(input_len, str):
            input_parts = input_len.split(";")
            if len(input_parts) > 2:
                raise ValueError(
                    "Support more turns' input length distributions yet,"
                    " expected at most 2 input parts (first-turn and"
                    f" remaining-turn), got {len(input_parts)}"
                )
            first_turn_input = input_parts[0].strip()
            remaining_input = (
                input_parts[1].strip()
                if len(input_parts) == 2
                else first_turn_input
            )
        else:
            first_turn_input = input_len
            remaining_input = input_len

        first_turn_output: DistributionParameter
        remaining_output: DistributionParameter
        if isinstance(output_len, str):
            output_parts = output_len.split(";")
            if len(output_parts) > 2:
                raise ValueError(
                    "Support more turns' output length distributions yet,"
                    " expected at most 2 output parts (first-turn and"
                    f" remaining-turn), got {len(output_parts)}"
                )
            first_turn_output = output_parts[0].strip()
            remaining_output = (
                output_parts[1].strip()
                if len(output_parts) == 2
                else first_turn_output
            )
        else:
            first_turn_output = output_len
            remaining_output = output_len

        delay_dist = BaseDistribution.from_distribution_parameter(
            delay_between_chat_turns
        )

        model_max_length = min(
            tokenizer.model_max_length, np.iinfo(np.int64).max
        )
        max_context_length = int(model_max_length * MAX_CONTEXT_USAGE_RATIO)

        num_turns_dist = BaseDistribution.from_distribution_parameter(num_turns)
        assert num_turns_dist is not None
        num_turns_per_session = [
            max(round(num_turns_dist.sample_value()), 1)
            for _ in range(num_chat_sessions)
        ]

        first_turn_samples = self.sample_requests(
            num_requests=num_chat_sessions,
            tokenizer=tokenizer,
            pool=pool,
            input_len=first_turn_input,
            output_len=first_turn_output,
            sys_prompt_ratio=sys_prompt_ratio,
            max_num_unique_sys_prompt=max_num_unique_sys_prompt,
            min_input_len=min_input_len,
            min_output_len=min_output_len,
        )
        first_turns = first_turn_samples.requests

        follow_up_turn_samples = self.sample_requests(
            num_requests=(sum(num_turns_per_session) - num_chat_sessions),
            tokenizer=tokenizer,
            pool=pool,
            input_len=remaining_input,
            output_len=remaining_output,
            sys_prompt_ratio=0,
            max_num_unique_sys_prompt=1,
            min_input_len=min_input_len,
            min_output_len=min_output_len,
        )
        follow_up_turns = follow_up_turn_samples.requests
        follow_up_turn_idx_offset = 0

        sessions: list[ChatSession] = []
        for session_id, first_turn in enumerate(first_turns):
            assert isinstance(first_turn.prompt_formatted, str)
            messages = [
                build_chat_message(
                    "user",
                    first_turn.prompt_formatted,
                    tokenizer,
                    num_tokens=first_turn.prompt_len,
                ),
                build_chat_message(
                    "assistant",
                    "",
                    tokenizer,
                    first_turn.output_len,
                    delay_until_next_message=max(delay_dist.sample_value(), 0)
                    if delay_dist
                    else None,
                ),
            ]
            current_context_length = sum(msg.num_tokens for msg in messages)
            for i in range(num_turns_per_session[session_id] - 1):
                follow_up_turn = follow_up_turns[follow_up_turn_idx_offset + i]
                assert isinstance(follow_up_turn.prompt_formatted, str)
                user_msg = build_chat_message(
                    "user",
                    follow_up_turn.prompt_formatted,
                    tokenizer,
                    num_tokens=follow_up_turn.prompt_len,
                )
                turn_tokens = user_msg.num_tokens + (
                    follow_up_turn.output_len or 0
                )
                if current_context_length + turn_tokens > max_context_length:
                    logger.info(
                        f"Session {session_id}: stopping early at turn {i + 1}"
                        f"(original: {num_turns_per_session[session_id]}):"
                        f" context would be {current_context_length + turn_tokens}"
                        f" > max {max_context_length} at turn {i + 2}"
                    )
                    break
                current_context_length += turn_tokens
                messages.append(user_msg)
                messages.append(
                    build_chat_message(
                        "assistant",
                        "",
                        tokenizer,
                        follow_up_turn.output_len,
                        delay_until_next_message=max(
                            delay_dist.sample_value(), 0
                        )
                        if delay_dist
                        else None,
                    )
                )

            follow_up_turn_idx_offset += num_turns_per_session[session_id] - 1

            sessions.append(ChatSession(session_id, messages))

        return ChatSamples(
            chat_sessions=sessions,
            shared_contexts=first_turn_samples.shared_contexts,
        )

    def _load_sharegpt_prompts_limited(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pool: TokenizerPool,
        max_num_unique_sys_prompt: int,
        num_requests: int,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Load and tokenize a limited number of system and user prompts from ShareGPT dataset.

        Args:
            tokenizer: Tokenizer to encode prompts.
            max_num_unique_sys_prompt: Number of unique system prompts to load.
            num_requests: Number of user prompts to load.

        Returns:
            Tuple of (sys_prompt_pool, user_prompt_pool), each a list of
            tokenized prompts (list of token IDs).
        """
        dataset_path = hf_hub_download_with_retry(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
        )

        with open(dataset_path) as f:
            dataset = json.load(f)

        # Filter out any empty conversations
        dataset = [
            data
            for data in dataset
            if len(data.get("conversations", data.get("conversation", []))) > 0
        ]

        # Only keep the first turn (user prompt) of each conversation.
        # Prepend the conversation ID to each prompt to ensure uniqueness.
        prompts = [
            data.get("id", str(uuid.uuid4()))
            + ": "
            + data.get("conversations", data.get("conversation", []))[0][
                "value"
            ]
            for data in dataset
        ]

        # Shuffle the prompts.
        random.shuffle(prompts)

        # Fan the slow `tokenizer.encode` calls out to the shared spawn
        # pool. We trim to a comfortable headroom over `required_prompts`
        # (entries shorter than 4 tokens get filtered post-encode) so we
        # don't tokenize the full ~25k ShareGPT pool on every benchmark run.
        required_prompts = max_num_unique_sys_prompt + num_requests
        candidate_pool = prompts[: max(required_prompts * 4, 64)]
        encoded = pool.map(_encode_ids, candidate_pool)
        tokenized_prompts: list[list[int]] = [
            ids for ids in encoded if len(ids) >= 4
        ][:required_prompts]

        if len(tokenized_prompts) < required_prompts:
            raise ValueError(
                f"ShareGPT dataset has only {len(tokenized_prompts)} valid"
                f" prompts (after filtering) but {required_prompts} are required"
                f" (sys={max_num_unique_sys_prompt} + user={num_requests})"
            )

        logger.info(
            f"Loaded {len(tokenized_prompts)} ShareGPT prompts for"
            " synthetic tokens on"
            f" (sys={max_num_unique_sys_prompt} + user={num_requests})"
            " prompts."
        )
        sys_prompt_pool = tokenized_prompts[:max_num_unique_sys_prompt]
        user_prompt_pool = tokenized_prompts[
            max_num_unique_sys_prompt : max_num_unique_sys_prompt + num_requests
        ]
        return (sys_prompt_pool, user_prompt_pool)

    def _repeat_truncate_prompt_tokens(
        self,
        prompt_pool: list[list[int]],
        target_len: int,
        prompt_index: int,
    ) -> list[int]:
        """Select a prompt from the pool by index and repeat or truncate to target length.

        Args:
            prompt_pool: List of tokenized prompts (list of token IDs).
            target_len: Target number of tokens.
            prompt_index: Index into the pool to select which prompt to use.

        Returns:
            List of token IDs with exactly target_len tokens (or empty if target_len <= 0).
        """
        if target_len <= 0:
            return []

        prompt_token_ids = prompt_pool[prompt_index]
        prompt_len = len(prompt_token_ids)

        if prompt_len >= target_len:
            # Truncate to target length.
            return prompt_token_ids[:target_len]
        else:
            # Repeat tokens to reach target length.
            ratio = (target_len + prompt_len - 1) // prompt_len
            return (prompt_token_ids * ratio)[:target_len]

    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        *,
        pool: TokenizerPool | None = None,
        **kwargs,
    ) -> RequestSamples:
        """Sample random benchmark requests with configurable length distributions.

        Args:
            num_requests: Number of requests to generate.
            tokenizer: Tokenizer for encoding prompts. Must match
                ``pool.tokenizer``.
            output_lengths: Optional fixed output lengths per request.
            shuffle: Whether to shuffle requests.
            pool: Tokenizer process pool used to parallelize encode work.
                Required by this dataset; the abstract base declares it
                optional for datasets that don't tokenize via a pool.
            **kwargs: Additional parameters:
                input_len: Distribution spec for input lengths, e.g.
                    "1024", "N(1024, 200)", "U(500, 1500)", "G(2, 500)".
                output_len: Distribution spec for output lengths.
                sys_prompt_ratio: Ratio of system prompt to input.
                max_num_unique_sys_prompt: Max unique system prompts.
                min_input_len: Minimum input length (default: 4).
                min_output_len: Minimum output length (default: 1).
                image_size: Image dimensions as "width,height".
                image_count: Number of images per request.
        """
        if pool is None:
            raise ValueError(
                "RandomBenchmarkDataset.sample_requests requires `pool`"
            )
        assert pool.tokenizer is tokenizer, (
            "TokenizerPool must be constructed from the same tokenizer "
            "passed to sample_requests"
        )
        input_len = kwargs.get("input_len")
        output_len = kwargs.get("output_len")
        sys_prompt_ratio = kwargs.get("sys_prompt_ratio", 0.0)
        max_num_unique_sys_prompt = kwargs.get("max_num_unique_sys_prompt", 1)
        min_input_len = kwargs.get("min_input_len", 4)
        min_output_len = kwargs.get("min_output_len", 1)
        image_size = kwargs.get("image_size", "")
        image_count = kwargs.get("image_count", 0)
        model_max_length = min(
            tokenizer.model_max_length, np.iinfo(np.int64).max
        )

        if input_len is None:
            raise ValueError("input_len is required for RandomBenchmarkDataset")
        if sys_prompt_ratio > 0 and max_num_unique_sys_prompt <= 0:
            raise ValueError(
                "max_num_unique_sys_prompt must be greater than 0 when"
                " sys_prompt_ratio is greater than 0 for RandomBenchmarkDataset"
            )
        if output_len is None:
            raise ValueError(
                "output_len is required for RandomBenchmarkDataset"
            )
        if (image_size and not image_count) or (not image_size and image_count):
            raise ValueError(
                "both image_size and image_count are required when generating"
                " an image benchmark"
            )

        input_dist = BaseDistribution.from_distribution_parameter(input_len)
        output_dist = BaseDistribution.from_distribution_parameter(output_len)
        logger.info(
            f"Random samples: input_len={input_len}, output_len={output_len}"
        )

        assert input_dist is not None
        assert output_dist is not None
        input_lens = [
            max(round(input_dist.sample_value()), min_input_len)
            for _ in range(num_requests)
        ]
        output_lens = [
            max(round(output_dist.sample_value()), min_output_len)
            for _ in range(num_requests)
        ]

        image_width, image_height = None, None
        if image_size:
            if len(image_size.split(",")) == 2:
                image_width, image_height = map(int, image_size.split(","))
            else:
                raise ValueError(
                    "Expected image size to be 2 ints separated by a comma,"
                    f" instead got: {image_size}"
                )

        vocab_size = tokenizer.vocab_size

        # Build sys_prompt_pool and user_prompt_pool. For synthetic tokens we load
        # both from ShareGPT; for random tokens we only build a random sys_prompt_pool
        # and generate user prompts per request.
        sys_prompt_pool: list[list[int]] = []
        user_prompt_pool: list[list[int]] = []
        if self.use_synthetic_tokens:
            sys_prompt_pool, user_prompt_pool = (
                self._load_sharegpt_prompts_limited(
                    tokenizer, pool, max_num_unique_sys_prompt, num_requests
                )
            )
        elif not self.use_synthetic_tokens and max_num_unique_sys_prompt > 0:
            # Random path: create sys_prompt_pool with base length of p50 of
            # input_lens for random token sequences.
            sys_prompt_base_len = max(1, int(np.percentile(input_lens, 50)))
            sys_prompt_pool = [
                np.random.randint(
                    0, vocab_size, size=sys_prompt_base_len
                ).tolist()
                for _ in range(max_num_unique_sys_prompt)
            ]

        max_context_length = int(model_max_length * MAX_CONTEXT_USAGE_RATIO)

        # Track longest system prompt per unique idx for prefix-cache warmup.
        # Prompts with the same idx are repetitions of the same base token
        # sequence, so at the token-ID level the shortest is always a prefix of
        # the longest. Warming up with only the longest is therefore sufficient
        # to prime the cache for all shorter variants.
        # Note: this is best-effort — decode/re-tokenize round-trips can cause
        # minor token-boundary mismatches (~90-96% cache hit rate in practice).
        warmup_dict: dict[int, SharedContext] = {}

        input_requests: list[SampledRequest] = []
        # Per-request scratch captured during construction. The slow
        # tokenizer decode/encode round-trips are batched and fanned out to
        # the shared spawn Pool below; their results are written back into
        # these `_PendingRequest` rows.
        pending: list[_PendingRequest] = []
        # `image_token_len` is invariant across requests today; hoist it.
        image_token_len = image_count * 256
        for i in range(num_requests):
            input_len_cur = input_lens[i]
            output_len_cur = (
                int(output_lens[i]) if output_lens[i] is not None else 0
            )
            if input_len_cur + output_len_cur > max_context_length:
                # Cap over-length sequences.
                logger.info(
                    f"Capping too long sequences ({input_len_cur} + {output_len_cur})"
                    f" > {max_context_length})..."
                )
                input_len_cur = max_context_length - output_len_cur

            # Calculate per-request system prompt length based on this request's
            # input length multiplied by sys_prompt_ratio.
            sys_prompt_len_i = min(
                max_context_length,
                int(np.floor(input_lens[i] * sys_prompt_ratio)),
            )
            user_prompt_len = input_len_cur - sys_prompt_len_i

            # Generate system prompt tokens for this request.
            # Use sys_prompt_idx to select which ShareGPT prompt (for synthetic)
            # or random token sequence (for random tokens).
            sys_prompt_idx = np.random.randint(0, max_num_unique_sys_prompt)
            if self.use_synthetic_tokens:
                sys_prompt_ids = self._repeat_truncate_prompt_tokens(
                    sys_prompt_pool, sys_prompt_len_i, sys_prompt_idx
                )
                user_prompt_ids = self._repeat_truncate_prompt_tokens(
                    user_prompt_pool, user_prompt_len, i
                )
            else:
                sys_prompt_ids = self._repeat_truncate_prompt_tokens(
                    sys_prompt_pool, sys_prompt_len_i, sys_prompt_idx
                )
                # Generate random token IDs for user prompt.
                user_prompt_offset = np.random.randint(0, vocab_size)
                user_prompt_ids = [
                    (user_prompt_offset + i + j) % vocab_size
                    for j in range(user_prompt_len)
                ]

            prompt_ids = sys_prompt_ids + user_prompt_ids

            # Remove special tokens from the prompt.
            special_ids = set(tokenizer.all_special_ids)
            # Use common space prefix for replacement, pick the first valid one.
            # - " "      : plain space (some tokenizers use explicit space token)
            # - U+0120   : GPT-2/BPE style prefix (e.g. Llama 3, DeepSeek)
            # - U+2581   : SentencePiece style prefix (e.g. Llama 1/2, Mistral)
            replacement = next(
                tid
                for candidate in [" ", chr(0x0120), chr(0x2581)]
                if (tid := tokenizer.convert_tokens_to_ids(candidate))
                not in (None, tokenizer.unk_token_id)
            )
            prompt_ids = [
                replacement if id in special_ids else id for id in prompt_ids
            ]

            sys_prompt_draft: _SysPromptDraft | None = None
            if sys_prompt_ratio > 0:
                sys_prompt_ids = [
                    replacement if id in special_ids else id
                    for id in sys_prompt_ids
                ]
                sys_prompt_draft = _SysPromptDraft(
                    idx=sys_prompt_idx, ids=sys_prompt_ids
                )

            images: list[OpenAIImage] = []
            for _ in range(image_count):
                assert image_height is not None
                assert image_width is not None
                raw_image = self._generate_random_image(
                    image_height, image_width
                )
                # TODO: figure out how to account for image tokens and chat prompts in this length.
                # For now, just hardcoding to the internvl 512x512 image token count.
                images.append(encode_image(raw_image))

            pending.append(
                _PendingRequest(
                    prompt_ids=prompt_ids,
                    images=images,
                    sys_prompt=sys_prompt_draft,
                )
            )

        # Batch decode in one shared-pool fan-out, then batch the encode
        # round-trip in another, so the slow tokenizer's decode and encode
        # both run in parallel across all CPU cores. Prompts come first in
        # each batch; sys-prompt slices for requests that have one follow,
        # in pending order.
        id_lists = [p.prompt_ids for p in pending]
        id_lists.extend(
            p.sys_prompt.ids for p in pending if p.sys_prompt is not None
        )
        decoded = pool.decode_texts(id_lists)

        sys_decoded_iter = iter(decoded[len(pending) :])
        for p, prompt_text in zip(
            pending, decoded[: len(pending)], strict=True
        ):
            p.prompt = prompt_text
            if p.sys_prompt is not None:
                p.sys_prompt.text = next(sys_decoded_iter)

        texts = [p.prompt for p in pending]
        texts.extend(
            p.sys_prompt.text for p in pending if p.sys_prompt is not None
        )
        lens = pool.encode_lens(texts)

        sys_lens_iter = iter(lens[len(pending) :])
        for p, plen in zip(pending, lens[: len(pending)], strict=True):
            p.prompt_encoded_len = plen
            if p.sys_prompt is not None:
                p.sys_prompt.encoded_len = next(sys_lens_iter)

        for i, p in enumerate(pending):
            input_requests.append(
                SampledRequest(
                    prompt_formatted=p.prompt,
                    prompt_len=p.prompt_encoded_len + image_token_len,
                    output_len=int(output_lens[i]),
                    encoded_images=p.images,
                    ignore_eos=(output_lens[i] is not None),
                )
            )
            sys = p.sys_prompt
            if sys is None or sys.encoded_len <= 0:
                continue
            prev = warmup_dict.get(sys.idx)
            if prev is None or sys.encoded_len > prev.num_tokens:
                warmup_dict[sys.idx] = SharedContext(
                    text=sys.text, num_tokens=sys.encoded_len
                )

        log_request_actual_length_percentiles(input_requests)

        return RequestSamples(
            requests=input_requests,
            shared_contexts=list(warmup_dict.values()),
        )

    def _generate_random_image(self, height: int, width: int) -> Image.Image:
        # Truly random images end up being too large and incompressible.
        # Instead create a much more limited block based random image with limited color palette.
        block_size = 16
        colors = np.array([0, 64, 128, 192, 255], dtype=np.uint8)

        blocks_h = (height + block_size - 1) // block_size
        blocks_w = (width + block_size - 1) // block_size

        # Generate colors for all blocks
        block_colors = np.random.choice(
            len(colors), size=(blocks_h, blocks_w, 3)
        )
        block_array = colors[block_colors]

        # repeat blocks to create image
        array = np.repeat(
            np.repeat(block_array, block_size, axis=0), block_size, axis=1
        )

        # crop
        array = array[:height, :width]

        return Image.fromarray(array)


class SyntheticBenchmarkDataset(RandomBenchmarkDataset):
    @override
    @property
    def use_synthetic_tokens(self) -> bool:
        return True
