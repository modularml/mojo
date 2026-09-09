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

"""Build multiturn :class:`ChatSamples` from a pool of user texts and CLI distributions.

Used when ``--fit-distributions`` is set so real datasets (instruct-coder,
agentic-code, etc.) match ``--random-*`` and ``--delay-between-chat-turns``
the same way as :class:`RandomBenchmarkDataset` multiturn mode.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ._tokenizer_pool import TokenizerPool, worker_tokenizer
from .agentic_tools import ToolConfig
from .distribution import BaseDistribution, DistributionParameter
from .turn_profile import TurnProfile
from .types import (
    ChatSamples,
    ChatSession,
    SessionMessage,
    SharedContext,
    estimate_num_tokens,
)


@dataclass(frozen=True)
class _TurnSpec:
    """Pre-sampled inputs for one request: a user message and its reply.

    ``is_agentic`` describes the user half -- a tool result, not a question.
    """

    prompt_text: str
    target_in: int
    target_out: int
    sys_variant: int
    # The delay that precedes this turn: a tool's latency before a round,
    # the user's think-time before a question.
    preceding_delay_ms: float | None
    is_agentic: bool = False
    unique_marker: str = ""


@dataclass(frozen=True)
class _SessionArgs:
    """Per-session payload dispatched to a worker."""

    session_id: int
    turns: list[_TurnSpec]
    sys_prompt_ratio: float
    min_input_len: int
    max_context_length: int
    log_prefix: str


@dataclass
class _SessionResult:
    """What a `_build_session` worker returns to the parent."""

    session: ChatSession | None
    observations: list[tuple[int, SharedContext]]
    # Number of turns kept before context overflow forced an early stop, or
    # ``None`` when the session used all its planned turns. The parent
    # aggregates these into a single summary line instead of logging per
    # session.
    truncated_at_turn: int | None = None


logger = logging.getLogger(__name__)

# Match RandomBenchmarkDataset: headroom for template / special-token overhead.
MAX_CONTEXT_USAGE_RATIO = 0.95


def resolve_constant_delay_ms(
    param: DistributionParameter | None,
) -> float | None:
    """Resolve a CLI delay parameter to one non-negative delay in milliseconds.

    Used when per-turn distribution sampling is disabled: sample once (constants
    return themselves).

    Args:
        param: Raw ``delay_between_chat_turns`` from config, or None.

    Returns:
        Delay in ms, or None when unset.
    """
    if param is None:
        return None
    dist = BaseDistribution.from_distribution_parameter(param)
    if dist is None:
        return None
    return max(float(dist.sample_value()), 0.0)


def parse_two_part_distribution(
    param: DistributionParameter,
    label: str,
) -> tuple[DistributionParameter, DistributionParameter]:
    """Split ``first;rest`` specs the same way as :class:`RandomBenchmarkDataset`."""
    if isinstance(param, str):
        parts = param.split(";")
        if len(parts) > 2:
            raise ValueError(
                f"{label}: at most two segments separated by ';' "
                f"(first turn vs remaining turns), got {len(parts)}"
            )
        first = parts[0].strip()
        second = parts[1].strip() if len(parts) == 2 else first
        return first, second
    return param, param


def _replacement_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    return next(
        tid
        for candidate in [" ", chr(0x0120), chr(0x2581)]
        if (tid := tokenizer.convert_tokens_to_ids(candidate))
        not in (None, tokenizer.unk_token_id)
    )


def _sanitize_token_ids(
    tokenizer: PreTrainedTokenizerBase, ids: list[int]
) -> list[int]:
    special_ids = set(tokenizer.all_special_ids)
    replacement = _replacement_token_id(tokenizer)
    return [(replacement if (tid in special_ids) else tid) for tid in ids]


def _repeat_truncate_token_ids(
    token_ids: list[int], target_len: int
) -> list[int]:
    if target_len <= 0:
        return []
    assert token_ids, "Cannot scale an empty token sequence"
    prompt_len = len(token_ids)
    if prompt_len >= target_len:
        return token_ids[:target_len]
    ratio = (target_len + prompt_len - 1) // prompt_len
    return (token_ids * ratio)[:target_len]


def _text_scaled_to_token_length(
    tokenizer: PreTrainedTokenizerBase,
    base_text: str,
    target_len: int,
    min_len: int = 4,
) -> tuple[str, int]:
    target_len = max(target_len, min_len)
    ids = tokenizer.encode(base_text, add_special_tokens=False)
    ids = _sanitize_token_ids(tokenizer, ids)
    if not ids:
        rep = _replacement_token_id(tokenizer)
        ids = [rep] * min_len
    scaled_ids = _repeat_truncate_token_ids(ids, target_len)
    text = tokenizer.decode(scaled_ids, skip_special_tokens=False)
    return text, estimate_num_tokens(tokenizer, text)


def _split_sys_user_token_targets(
    target_total: int,
    sys_prompt_ratio: float,
    min_user_len: int,
) -> tuple[int, int]:
    target_total = max(target_total, min_user_len)
    if sys_prompt_ratio <= 0:
        return 0, target_total
    sys_len = int(np.floor(target_total * sys_prompt_ratio))
    user_len = target_total - sys_len
    if user_len < min_user_len:
        user_len = min_user_len
        sys_len = max(0, target_total - user_len)
    return sys_len, user_len


def _system_prompt_template(variant: int) -> str:
    return (
        "You are a helpful coding assistant. Follow the user's instructions "
        f"carefully (assistant profile {variant})."
    )


@dataclass(frozen=True)
class ScaledUserMessage:
    """User message scaled to a per-turn token budget, with optional sys prefix.

    `content` is the full text sent to the model
    (`sys_prefix.text + "\\n\\n" + user_body_scaled` when `sys_prefix` is set).
    `sys_prefix` describes the leading sys-prefix slice, or `None` when
    `sys_prompt_ratio <= 0`.
    """

    content: str
    num_tokens: int
    sys_prefix: SharedContext | None


def build_scaled_user_message(
    tokenizer: PreTrainedTokenizerBase,
    user_body: str,
    target_input_tokens: int,
    sys_prompt_ratio: float,
    sys_variant: int,
    min_input_len: int,
    *,
    unique_marker: str = "",
) -> ScaledUserMessage:
    """Optional synthetic system prefix + body scaled to the target token budget.

    When ``unique_marker`` is non-empty it is prepended to the scaled user
    body (after the system prefix) to make otherwise-identical pool entries
    cache-distinct across pool cycles. Marker tokens are reserved from the
    user-side budget so the on-the-wire input length stays close to
    ``target_input_tokens``.
    """
    sys_len, user_len = _split_sys_user_token_targets(
        target_input_tokens, sys_prompt_ratio, min_input_len
    )
    marker_tokens = (
        estimate_num_tokens(tokenizer, unique_marker) if unique_marker else 0
    )
    body_target_len = max(user_len - marker_tokens, min_input_len)
    parts: list[str] = []
    sys_prefix: SharedContext | None = None
    if sys_len > 0:
        sys_text, sys_tokens = _text_scaled_to_token_length(
            tokenizer, _system_prompt_template(sys_variant), sys_len, min_len=1
        )
        parts.append(sys_text)
        sys_prefix = SharedContext(text=sys_text, num_tokens=sys_tokens)
    user_text, _ = _text_scaled_to_token_length(
        tokenizer, user_body, body_target_len, min_len=min_input_len
    )
    parts.append(unique_marker + user_text if unique_marker else user_text)
    combined = "\n\n".join(parts)
    return ScaledUserMessage(
        content=combined,
        num_tokens=estimate_num_tokens(tokenizer, combined),
        sys_prefix=sys_prefix,
    )


def _register_longest_sys_prompt(
    warmup_dict: dict[int, SharedContext],
    variant_idx: int,
    sys_prefix: SharedContext,
) -> None:
    """Update warmup_dict[variant_idx] when sys_prefix.num_tokens exceeds the prior entry."""
    prev = warmup_dict.get(variant_idx)
    prev_tokens = prev.num_tokens if prev else 0
    if prev_tokens < sys_prefix.num_tokens:
        warmup_dict[variant_idx] = sys_prefix


def _build_session(args: _SessionArgs) -> _SessionResult:
    """Worker-side per-session build, run in spawn Pool processes.

    Uses the worker-local tokenizer loaded by `_tokenizer_pool._init_encoder`
    so the slow Kimi tokenizer is exercised in parallel across cores.

    Context overflow is reported back to the parent (``truncated_at_turn``)
    rather than logged here, so the parent can emit one aggregate summary
    instead of a per-session line for every truncated session.
    """
    tokenizer = worker_tokenizer()
    messages: list[SessionMessage] = []
    context_tokens = 0
    observations: list[tuple[int, SharedContext]] = []
    truncated_at_turn: int | None = None

    for turn_i, turn in enumerate(args.turns):
        sys_prompt_ratio = args.sys_prompt_ratio if turn_i == 0 else 0.0
        msg = build_scaled_user_message(
            tokenizer,
            turn.prompt_text,
            turn.target_in,
            sys_prompt_ratio,
            turn.sys_variant,
            args.min_input_len,
            unique_marker=turn.unique_marker,
        )
        if (
            context_tokens + msg.num_tokens + turn.target_out
            > args.max_context_length
        ):
            truncated_at_turn = turn_i
            break
        if msg.sys_prefix is not None:
            observations.append((turn.sys_variant, msg.sys_prefix))
        messages.append(
            SessionMessage(
                source="user",
                content=msg.content,
                num_tokens=msg.num_tokens,
                is_agentic=turn.is_agentic,
            )
        )
        # The driver sleeps after each reply, so this one carries the
        # think-time that precedes the turn following it.
        next_turn = (
            args.turns[turn_i + 1] if turn_i + 1 < len(args.turns) else None
        )
        messages.append(
            SessionMessage(
                source="assistant",
                content="",
                num_tokens=turn.target_out,
                delay_until_next_message=(
                    next_turn.preceding_delay_ms if next_turn else None
                ),
            )
        )
        context_tokens += msg.num_tokens + turn.target_out

    # Truncation can end a session on a reply that did have a successor, so
    # clear it here too: the driver sleeps on this before exiting, and a delay
    # on the last reply is dead wall-clock.
    if messages:
        messages[-1].delay_until_next_message = None

    session = (
        ChatSession(args.session_id, messages) if len(messages) >= 2 else None
    )
    return _SessionResult(
        session=session,
        observations=observations,
        truncated_at_turn=truncated_at_turn,
    )


def _build_sessions(
    *,
    pool: TokenizerPool,
    user_text_pool: Sequence[str],
    tool_text_pool: Sequence[str],
    session_turn_counts: Sequence[int],
    first_turn_profile: TurnProfile,
    later_turn_profile: TurnProfile,
    tools: Sequence[ToolConfig] | None,
    rounds: BaseDistribution | None,
    min_output_len: int,
    sys_prompt_ratio: float,
    max_num_unique_sys_prompt: int,
    min_input_len: int,
    shuffle_pool: bool,
    log_prefix: str,
) -> ChatSamples:
    """Draw each session's turns and build them from the pools.

    A human turn and an agent-loop round are the same draw from different
    distributions, so both go through one loop.
    """
    model_max_length = min(
        pool.tokenizer.model_max_length, np.iinfo(np.int64).max
    )
    max_context_length = int(model_max_length * MAX_CONTEXT_USAGE_RATIO)

    pool_lens = pool.encode_lens(list(user_text_pool))
    filtered_user_texts = [
        t
        for t, n in zip(user_text_pool, pool_lens, strict=False)
        if n >= min_input_len
    ]
    if not filtered_user_texts:
        logger.warning(
            "%s: no valid user texts in pool after filtering (min_input_len=%s).",
            log_prefix,
            min_input_len,
        )
        return ChatSamples(chat_sessions=[])

    num_sessions = len(session_turn_counts)
    total_turns_needed = sum(session_turn_counts)
    if total_turns_needed > len(filtered_user_texts):
        logger.info(
            "%s: pool has %d valid rows but %d turns planned; will cycle "
            "through the pool, with a per-draw unique marker (e.g. '[7] ') "
            "and distributions re-sampled on each turn.",
            log_prefix,
            len(filtered_user_texts),
            total_turns_needed,
        )

    # A real tool result can be a token or two, so no min-length filter here.
    # Datasets with no recorded tool messages fall back to the user pool; the
    # copy keeps the two shuffled and cycled independently below.
    tool_texts = list(tool_text_pool or filtered_user_texts)
    if shuffle_pool:
        random.shuffle(filtered_user_texts)
        random.shuffle(tool_texts)

    warmup_dict: dict[int, SharedContext] = {}
    max_variant = max(1, max_num_unique_sys_prompt)
    session_args_list: list[_SessionArgs] = []
    # One cycle per pool, so one wrapping does not disturb the other.
    user_body_cycle = itertools.cycle(filtered_user_texts)
    tool_body_cycle = itertools.cycle(tool_texts)
    draw_counter = itertools.count()
    tool_list = list(tools) if tools else []
    tool_weights = [tool.weight for tool in tool_list]

    def add_turn(
        turns: list[_TurnSpec],
        profile: TurnProfile,
        body_cycle: Iterator[str],
        *,
        session_id: int,
        turn_i: int,
        is_agentic: bool,
    ) -> None:
        """Draw one spec and append it, taking its body from ``body_cycle``.

        A spec is a human turn or an agent-loop round; both draw the same
        way, from different profiles.
        """
        targets = profile.draw(
            min_input_len=min_input_len, min_output_len=min_output_len
        )
        turns.append(
            _TurnSpec(
                prompt_text=next(body_cycle),
                target_in=targets.input_len,
                target_out=targets.output_len,
                sys_variant=(session_id + turn_i) % max_variant,
                preceding_delay_ms=targets.delay_ms,
                is_agentic=is_agentic,
                # Marked unconditionally: avoids unintended shared prefixes.
                unique_marker=f"[{next(draw_counter)}] ",
            )
        )

    for session_id, n_turns in enumerate(session_turn_counts):
        turn_specs: list[_TurnSpec] = []
        for turn_i in range(n_turns):
            block_start = len(turn_specs)
            add_turn(
                turn_specs,
                first_turn_profile if turn_i == 0 else later_turn_profile,
                user_body_cycle,
                session_id=session_id,
                turn_i=turn_i,
                is_agentic=False,
            )
            if tool_list and rounds is not None:
                for tool in random.choices(
                    tool_list,
                    weights=tool_weights,
                    k=max(round(rounds.sample_value()), 0),
                ):
                    add_turn(
                        turn_specs,
                        tool.profile,
                        tool_body_cycle,
                        session_id=session_id,
                        turn_i=turn_i,
                        is_agentic=True,
                    )

            # Each spec goes on the wire as a user message and a reply. In
            # the agent loop a reply is the call to the NEXT tool rather than
            # an answer to what just arrived, so every round displaces the
            # human's answer one slot further right. Rotating target_out one
            # slot left restores call-before-result, and the modulo wrap
            # leaves the human's answer on the block's last reply.
            block = turn_specs[block_start:]
            outs = [spec.target_out for spec in block]
            turn_specs[block_start:] = [
                replace(spec, target_out=outs[(i + 1) % len(outs)])
                for i, spec in enumerate(block)
            ]

        session_args_list.append(
            _SessionArgs(
                session_id=session_id,
                turns=turn_specs,
                sys_prompt_ratio=sys_prompt_ratio,
                min_input_len=min_input_len,
                max_context_length=max_context_length,
                log_prefix=log_prefix,
            )
        )

    sessions: list[ChatSession] = []
    truncated_turns: list[int] = []
    for result in pool.map(_build_session, session_args_list):
        if result.session is not None:
            sessions.append(result.session)
        if result.truncated_at_turn is not None:
            truncated_turns.append(result.truncated_at_turn)
        for sys_variant, sys_prefix in result.observations:
            _register_longest_sys_prompt(warmup_dict, sys_variant, sys_prefix)

    if truncated_turns:
        # One aggregate line instead of a per-session log for each truncated
        # session, which otherwise floods the terminal.
        logger.info(
            "%s: %d/%d sessions truncated to fit the model's max context "
            "(%d tokens); kept turns ranged %d-%d.",
            log_prefix,
            len(truncated_turns),
            num_sessions,
            max_context_length,
            min(truncated_turns),
            max(truncated_turns),
        )

    if len(sessions) < num_sessions:
        logger.warning(
            "%s: only %d sessions formed, requested %d.",
            log_prefix,
            len(sessions),
            num_sessions,
        )
    return ChatSamples(
        chat_sessions=sessions, shared_contexts=list(warmup_dict.values())
    )


def build_fitted_chat_samples(
    pool: TokenizerPool,
    user_text_pool: list[str],
    num_sessions: int,
    num_turns: DistributionParameter,
    input_len: DistributionParameter,
    output_len: DistributionParameter,
    delay_between_turns_dist: DistributionParameter | None,
    sys_prompt_ratio: float,
    max_num_unique_sys_prompt: int,
    min_input_len: int = 4,
    min_output_len: int = 1,
    *,
    shuffle_pool: bool = False,
    tools: Sequence[ToolConfig] | None = None,
    agentic_rounds_per_turn: DistributionParameter | None = None,
    tool_text_pool: Sequence[str] | None = None,
    log_prefix: str = "multiturn-fit",
) -> ChatSamples:
    """Assemble :class:`ChatSamples` by grouping pooled user strings into sessions.

    For each session, samples a turn count and per-turn input/output targets from
    the given distributions (with optional ``';'`` split for first vs remaining
    turns). Each user message is produced by token-space repeat/truncation of a
    pooled string, optionally prefixed with a scaled synthetic system block.

    Every user body is prefixed with a ``[N] `` marker, ``N`` counting draws
    across the run from zero, so cycled bodies stay cache-distinct. When the
    planned turn count exceeds the available pool it wraps back to the start;
    distribution samples are re-drawn on every turn, so a second pass over the
    pool produces a freshly fit workload rather than a replay.

    Args:
        pool: Tokenizer process pool used for batched pre-filter encodes
            and per-session worker dispatch.
        user_text_pool: Candidate user message bodies (one per underlying turn).
        num_sessions: Target number of chat sessions.
        num_turns: Distribution for turns per session (>= 1 after rounding).
        input_len: Per-turn user-side token target distribution(s).
        output_len: Per-turn assistant ``max_tokens`` distribution(s).
        delay_between_turns_dist: Optional per-assistant-message delay (ms).
        sys_prompt_ratio: Fraction of user message token budget for system prefix.
        max_num_unique_sys_prompt: Cycle count for system-prefix variants.
        min_input_len: Floor for sampled input lengths, human turns and
            agent-loop rounds alike. A round also carries the ``[N] `` marker,
            so a sub-floor tool profile cannot reach the wire either way.
        min_output_len: Floor for sampled output lengths.
        shuffle_pool: Whether to randomize the draw order of the text pools.
        tools: Optional tool profiles for an agent loop after each turn.
        agentic_rounds_per_turn: How many rounds follow each turn. Given with
            ``tools``, or neither.
        tool_text_pool: Optional body pool for agent-loop rounds; defaults to
            ``user_text_pool``.
        log_prefix: Logger prefix for warnings.

    Returns:
        Generated chat sessions. The pool is cycled with per-draw unique
        markers to satisfy ``num_sessions`` even when planned turns exceed the
        pool size; the result may still contain fewer sessions if any drop
        due to model max-context overflow.
    """
    first_in, rest_in = parse_two_part_distribution(input_len, "input_len")
    first_out, rest_out = parse_two_part_distribution(output_len, "output_len")

    first_turn_profile = TurnProfile.from_parameters(
        input_len=first_in,
        output_len=first_out,
        delay=delay_between_turns_dist,
    )
    later_turn_profile = TurnProfile.from_parameters(
        input_len=rest_in, output_len=rest_out, delay=delay_between_turns_dist
    )

    rounds_dist = BaseDistribution.from_distribution_parameter(
        agentic_rounds_per_turn
    )
    if (tools is None) != (rounds_dist is None):
        raise ValueError(
            "tools and agentic_rounds_per_turn must be given together: one"
            " says what a round costs, the other how many rounds run per"
            " turn."
        )

    # Turn counts are drawn up front, before any per-turn target, matching the
    # original sampling order so a fixed seed reproduces earlier runs.
    num_turns_dist = BaseDistribution.from_distribution_parameter_or_raise(
        num_turns
    )
    session_turn_counts = [
        max(round(num_turns_dist.sample_value()), 1)
        for _ in range(num_sessions)
    ]

    return _build_sessions(
        pool=pool,
        user_text_pool=user_text_pool,
        tool_text_pool=tool_text_pool or [],
        session_turn_counts=session_turn_counts,
        first_turn_profile=first_turn_profile,
        later_turn_profile=later_turn_profile,
        tools=tools,
        rounds=rounds_dist,
        min_input_len=min_input_len,
        min_output_len=min_output_len,
        sys_prompt_ratio=sys_prompt_ratio,
        max_num_unique_sys_prompt=max_num_unique_sys_prompt,
        shuffle_pool=shuffle_pool,
        log_prefix=log_prefix,
    )
