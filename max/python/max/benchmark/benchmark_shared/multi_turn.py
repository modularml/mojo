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

"""Multi-turn benchmark request generation and execution."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Sequence

try:
    from asyncio import TaskGroup  # type: ignore[attr-defined]  # added in 3.11
except ImportError:
    from taskgroup import TaskGroup  # Python < 3.11 backport

import numpy as np
from max.benchmark.benchmark_shared.config import SamplingConfig
from max.benchmark.benchmark_shared.datasets import ChatSession
from max.benchmark.benchmark_shared.datasets.types import TextContentBlock
from max.benchmark.benchmark_shared.lora_benchmark_manager import (
    LoRABenchmarkManager,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    BaseRequestFuncOutput,
    ChatMessage,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
    mark_cancelled_if_past_deadline,
    progressbar_request_driver,
)
from max.benchmark.benchmark_shared.utils import (
    deadline_passed,
    deadline_remaining_s,
    exceeds_deadline,
)
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _poisson_interval(request_rate: float, burstiness: float) -> float:
    """Gamma-distributed inter-arrival time for a target Poisson session rate."""
    theta = 1.0 / (request_rate * burstiness)
    return float(np.random.gamma(shape=burstiness, scale=theta))


async def chat_session_driver(
    model_id: str,
    api_url: str,
    request_driver: RequestDriver,
    request_counter: RequestCounter,
    chat_session: ChatSession,
    max_chat_len: int,
    sampling: SamplingConfig,
    ignore_first_turn_stats: bool = False,
    benchmark_should_end_time: int | None = None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
    est_ttft_ms: float = 0.0,
    est_tpot_ms: float = 0.0,
    use_session_id_as_cache_salt: bool = False,
) -> list[RequestFuncOutput]:
    request_func_input = RequestFuncInput(
        model=model_id,
        session_id=str(chat_session.id),
        cache_salt=str(chat_session.id)
        if use_session_id_as_cache_salt
        else None,
        sampling=sampling,
        prompt=[],
        images=[],
        api_url=api_url,
        prompt_len=0,
        max_tokens=0,
        ignore_eos=True,
    )
    content_idx = 0  # Assume user initiates the conversation

    session_outputs: list[RequestFuncOutput] = []
    message_history: list[ChatMessage] = []
    chat_len = 0

    messages = chat_session.messages
    prefix_end_idx = chat_session.prefix_turns * 2
    applied_initial_sleep = False

    # Build prefix turns locally (no server round-trips). The first
    # measured turn sends the full history for KV cache prefill.
    while content_idx < prefix_end_idx and content_idx + 1 < len(messages):
        chat_len += messages[content_idx].num_tokens
        output_len = messages[content_idx + 1].num_tokens
        if chat_len + output_len > max_chat_len:
            logger.warning(
                f"Session {chat_session.id}: prefix exceeded max chat"
                f" length {max_chat_len}, no measured turns possible"
            )
            break

        user_prompt = messages[content_idx].content
        message_history.append(
            ChatMessage(
                role="user",
                content=[TextContentBlock(text=user_prompt)],
            )
        )
        # Synthetic placeholder for the assistant response.
        assistant_content = messages[content_idx + 1].content
        if not assistant_content:
            assistant_content = " ".join(["token"] * max(output_len, 1))
        message_history.append(
            ChatMessage(
                role="assistant",
                content=[TextContentBlock(text=assistant_content)],
            )
        )
        chat_len += output_len
        content_idx += 2

    # If prefix exhausted the chat length budget, skip measured turns.
    if content_idx < prefix_end_idx:
        return session_outputs

    while content_idx + 1 < len(messages):
        chat_len += messages[content_idx].num_tokens
        if content_idx == 0 and run_prefix:
            chat_len += run_prefix_len
        output_len = messages[content_idx + 1].num_tokens
        if chat_len + output_len > max_chat_len:
            logger.warning(
                f"Ending conversation: hitting max chat length {max_chat_len}"
            )
            break

        advance_request = request_counter.advance_until_max()
        if not advance_request:  # reached max_requests
            break

        user_prompt = messages[content_idx].content
        if content_idx == 0 and run_prefix:
            user_prompt = run_prefix + user_prompt
        message_history.append(
            ChatMessage(
                role="user",
                content=[TextContentBlock(text=user_prompt)],
            )
        )
        request_func_input.prompt = message_history
        request_func_input.prompt_len = chat_len
        request_func_input.max_tokens = output_len

        if not applied_initial_sleep:
            applied_initial_sleep = True
            if (
                content_idx > 0
            ):  # pre-warmed: phase-spread across inter-turn window
                # Spread the first measured turn across the just-completed
                # turn's occupancy window R_{k} + D_{k} (generation + sleep).
                # A draw landing in the generation portion (u < R_k) means the
                # session would still be generating, so fire immediately;
                # otherwise sleep the remaining delay u - R_k. With zero runtime
                # estimates R_k = 0 and this is uniform[0, D_k] as before.
                prev_msg = messages[content_idx - 1]
                delay_ms = prev_msg.delay_until_next_message or 0.0
                runtime_ms = est_ttft_ms + est_tpot_ms * prev_msg.num_tokens
                window_ms = delay_ms + runtime_ms
                if window_ms > 0:
                    u = random.uniform(0, window_ms)
                    sleep_s = max(0.0, u - runtime_ms) / 1000
                    if sleep_s > 0:
                        if exceeds_deadline(sleep_s, benchmark_should_end_time):
                            return session_outputs
                        await asyncio.sleep(sleep_s)

        if deadline_passed(benchmark_should_end_time):
            response = RequestFuncOutput(
                cancelled=True, request_submit_time=time.perf_counter()
            )
        else:
            remaining_s = deadline_remaining_s(benchmark_should_end_time)
            try:
                raw_response = await asyncio.wait_for(
                    request_driver.request(request_func_input),
                    timeout=remaining_s,
                )
                if not isinstance(raw_response, RequestFuncOutput):
                    raise TypeError(
                        "Expected RequestFuncOutput in text-generation benchmark flow."
                    )
                response = raw_response
                mark_cancelled_if_past_deadline(
                    response, benchmark_should_end_time
                )
            except asyncio.TimeoutError:
                response = RequestFuncOutput(
                    cancelled=True, request_submit_time=time.perf_counter()
                )

        if not (ignore_first_turn_stats and content_idx == prefix_end_idx):
            # Tag with session + turn order so per-turn cache retention can
            # compare each measured turn against the previous one in-session.
            response.session_id = str(chat_session.id)
            response.turn_index = len(session_outputs)
            session_outputs.append(response)

        if not response.success:
            if not response.cancelled:
                logger.error(
                    f"Ending chat session {chat_session.id} due to server"
                    f" error response: {response.error}"
                )
            break

        message_history.append(
            ChatMessage(
                role="assistant",
                content=[TextContentBlock(text=response.generated_text)],
            )
        )
        chat_len += output_len

        if next_delay_ms := messages[content_idx + 1].delay_until_next_message:
            sleep_s = next_delay_ms / 1000
            if exceeds_deadline(sleep_s, benchmark_should_end_time):
                return session_outputs
            await asyncio.sleep(sleep_s)

        content_idx += 2

    return session_outputs


async def prerun_warmup_turns(
    sessions: Sequence[ChatSession],
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    max_chat_len: int,
    sampling: SamplingConfig,
    max_concurrency: int,
    disable_tqdm: bool = False,
    use_session_id_as_cache_salt: bool = False,
) -> None:
    """Send one warmup request per session with prefix_turns > 0.

    The request's prompt is the dataset history through user_N, where
    N is the session's prefix_turns. Processing it caches every
    earlier prefix in one pass, since turn k's prompt is a strict
    prefix of turn k+1's. Outputs are discarded.

    Inter-turn assistant text comes from the dataset, matching what
    chat_session_driver builds when it skips through prefix turns.
    The runner can resume the session at turn N+1 with a cache hit.

    Runs before benchmark_start_time so warmup time doesn't count.
    """
    sessions_with_prefix = [s for s in sessions if s.prefix_turns > 0]
    if not sessions_with_prefix:
        return

    requests_to_fire: list[RequestFuncInput] = []
    for session in sessions_with_prefix:
        messages = session.messages
        prefix_end_idx = session.prefix_turns * 2
        message_history: list[ChatMessage] = []
        chat_len = 0
        content_idx = 0
        pending_request: RequestFuncInput | None = None
        while content_idx < prefix_end_idx and content_idx + 1 < len(messages):
            chat_len += messages[content_idx].num_tokens
            output_len = messages[content_idx + 1].num_tokens
            if chat_len + output_len > max_chat_len:
                break
            message_history.append(
                ChatMessage(
                    role="user",
                    content=[
                        TextContentBlock(text=messages[content_idx].content)
                    ],
                )
            )
            # Overwrite each iteration; only the last fitting turn's
            # request fires. Its prompt covers every earlier prefix.
            pending_request = RequestFuncInput(
                model=model_id,
                session_id=str(session.id),
                cache_salt=str(session.id)
                if use_session_id_as_cache_salt
                else None,
                sampling=sampling,
                prompt=list(message_history),
                images=[],
                api_url=api_url,
                prompt_len=chat_len,
                max_tokens=output_len,
                ignore_eos=True,
            )
            assistant_content = messages[content_idx + 1].content
            if not assistant_content:
                assistant_content = " ".join(["token"] * max(output_len, 1))
            message_history.append(
                ChatMessage(
                    role="assistant",
                    content=[TextContentBlock(text=assistant_content)],
                )
            )
            chat_len += output_len
            content_idx += 2
        if pending_request is not None:
            requests_to_fire.append(pending_request)

    if not requests_to_fire:
        return

    logger.info(
        f"[warmup-prerun] Sending {len(requests_to_fire)} warmup"
        " requests to seed the prefix cache..."
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    warmup_results: list[BaseRequestFuncOutput | None] = [None] * len(
        requests_to_fire
    )

    with progressbar_request_driver(
        request_driver,
        len(requests_to_fire),
        disable_tqdm=disable_tqdm,
        desc="warmup",
    ) as driver:

        async def _fire(idx: int, req: RequestFuncInput) -> None:
            async with semaphore:
                warmup_results[idx] = await driver.request(req)

        await asyncio.gather(
            *(_fire(idx, r) for idx, r in enumerate(requests_to_fire))
        )

    for idx, result in enumerate(warmup_results):
        if result is None:
            raise RuntimeError(
                f"Warmup-prerun task {idx} did not produce a result"
                " (this is a bug)"
            )
        if not result.success:
            raise ValueError(
                f"Warmup-prerun request failed at index {idx}"
                f" (prompt_len: {requests_to_fire[idx].prompt_len}),"
                f" error: {result.error}"
            )
    logger.info("[warmup-prerun] complete.")


async def run_multiturn_benchmark(
    *,
    chat_sessions: Sequence[ChatSession],
    max_requests: int,
    semaphore: contextlib.AbstractAsyncContextManager[None],
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    tokenizer: PreTrainedTokenizerBase,
    ignore_first_turn_stats: bool,
    lora_manager: LoRABenchmarkManager | None,
    warmup_delay_ms: float,
    max_concurrency: int | None,
    sampling: SamplingConfig,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
    request_rate: float = float("inf"),
    burstiness: float = 1.0,
    est_ttft_ms: float = 0.0,
    est_tpot_ms: float = 0.0,
    use_session_id_as_cache_salt: bool = False,
) -> dict[str, list[RequestFuncOutput]]:
    """Run multi-turn chat benchmark scenario.

    chat_sessions is already reordered (warmup picks first, with
    prefix_turns set). The orchestrator runs warmup before the timer.
    """

    # Track total sent requests among chat sessions
    request_counter = RequestCounter(
        max_requests=max_requests,
        total_sent_requests=0,
    )

    # apply the semaphore at the session level
    # ex: with max_concurrency = 1,
    # the first session finishes before the second session starts
    async def limited_chat_session_driver(
        chat_session: ChatSession,
        session_idx: int,
    ) -> tuple[str, list[RequestFuncOutput]]:
        # Determine which LoRA to use for this chat session
        lora_id = None
        if lora_manager:
            lora_id = lora_manager.get_lora_for_request(session_idx)

        async with semaphore:
            outputs = await chat_session_driver(
                model_id=model_id if lora_id is None else lora_id,
                api_url=api_url,
                request_driver=request_driver,
                request_counter=request_counter,
                chat_session=chat_session,
                max_chat_len=tokenizer.model_max_length,
                sampling=sampling,
                ignore_first_turn_stats=ignore_first_turn_stats,
                benchmark_should_end_time=benchmark_should_end_time,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
                est_ttft_ms=est_ttft_ms,
                est_tpot_ms=est_tpot_ms,
                use_session_id_as_cache_salt=use_session_id_as_cache_salt,
            )
        session_id = (
            str(chat_session.id)
            if chat_session.id is not None
            else f"anonymous-{session_idx}"
        )
        return session_id, outputs

    # Pre-warmed sessions (prefix_turns > 0) get phase-spread jitter inside
    # chat_session_driver. Cold-start sessions are paced here: Poisson
    # inter-arrival when request_rate is finite, warmup_delay_ms stagger otherwise.
    use_rate_pacing = request_rate != float("inf")

    async def _pace_cold_start(idx: int) -> bool:
        """Pace a cold-start session launch. Returns True if the deadline was hit."""
        if use_rate_pacing and idx > 0:
            sleep_s = _poisson_interval(request_rate, burstiness)
        elif warmup_delay_ms > 0 and max_concurrency and idx < max_concurrency:
            sleep_s = warmup_delay_ms / 1000
        else:
            return False
        if exceeds_deadline(sleep_s, benchmark_should_end_time):
            return True
        await asyncio.sleep(sleep_s)
        return False

    tasks: list[asyncio.Task[tuple[str, list[RequestFuncOutput]]]] = []
    for idx, chat_session in enumerate(chat_sessions):
        if chat_session.prefix_turns == 0 and await _pace_cold_start(idx):
            break
        tasks.append(
            asyncio.create_task(limited_chat_session_driver(chat_session, idx))
        )

    outputs_by_session: dict[str, list[RequestFuncOutput]] = dict(
        await asyncio.gather(*tasks)
    )

    if benchmark_should_end_time is not None:
        if deadline_passed(benchmark_should_end_time):
            total_turns = sum(len(v) for v in outputs_by_session.values())
            cancelled = sum(
                1 for v in outputs_by_session.values() for o in v if o.cancelled
            )
            logger.info(
                "Benchmark stopped by the duration limit"
                " (--max-benchmark-duration-s):"
                f" {total_turns} turns dispatched across"
                f" {len(outputs_by_session)} sessions,"
                f" {cancelled} cancelled in flight."
            )
        else:
            logger.warning(
                "All chat sessions completed before the time limit. "
                "Consider increasing --num-chat-sessions for more stable load."
            )

    return outputs_by_session


class ConcurrentTurnsRequestDriver(RequestDriver):
    """Wraps a RequestDriver to cap the number of concurrent in-flight turns.

    Acquires a semaphore slot before issuing each turn request and releases it
    as soon as the response returns. Inter-turn delays (e.g. delay_until_next_message)
    fall outside the slot's hold window, so idle user-think-time does not consume
    concurrency capacity.

    With many concurrent conversations, a turn request may wait in the semaphore
    backlog long enough for the deadline to expire. Cancel it when stale.
    """

    def __init__(
        self,
        request_driver: RequestDriver,
        semaphore: contextlib.AbstractAsyncContextManager[None],
        benchmark_should_end_time: int | None = None,
    ) -> None:
        super().__init__(tokenizer=request_driver.tokenizer)
        self._request_driver = request_driver
        self._semaphore = semaphore
        self._benchmark_should_end_time = benchmark_should_end_time

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> BaseRequestFuncOutput:
        async with self._semaphore:
            if deadline_passed(self._benchmark_should_end_time):
                return request_func_input.get_output_type()(
                    cancelled=True, request_submit_time=time.perf_counter()
                )
            remaining_s = deadline_remaining_s(self._benchmark_should_end_time)
            try:
                output = await asyncio.wait_for(
                    self._request_driver.request(request_func_input),
                    timeout=remaining_s,
                )
            except asyncio.TimeoutError:
                return request_func_input.get_output_type()(
                    cancelled=True, request_submit_time=time.perf_counter()
                )
            return mark_cancelled_if_past_deadline(
                output, self._benchmark_should_end_time
            )


async def run_kv_cache_stress_benchmark(
    *,
    chat_sessions: Sequence[ChatSession],
    max_requests: int,
    max_concurrent_conversations: int,
    semaphore: contextlib.AbstractAsyncContextManager[None],
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    tokenizer: PreTrainedTokenizerBase,
    ignore_first_turn_stats: bool,
    lora_manager: LoRABenchmarkManager | None,
    warmup_delay_ms: float,
    sampling: SamplingConfig,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
    request_rate: float = float("inf"),
    burstiness: float = 1.0,
    est_ttft_ms: float = 0.0,
    est_tpot_ms: float = 0.0,
    use_session_id_as_cache_salt: bool = False,
) -> dict[str, list[RequestFuncOutput]]:
    """Run a KV-cache stress benchmark with independent conversation and turn concurrency.

    Two independent concurrency controls:

    - `max_concurrent_conversations`: at most this many chat sessions are
      driven at once. Workers pick up the next session from the queue when one
      finishes, growing the server's KV-cache footprint.
    - `semaphore` (`max_concurrency` in the CLI): caps the number of turn
      requests in-flight globally across all concurrent sessions. Workers that
      cannot acquire a turn slot block without sending a request; the session's
      `session_id` and client-side conversation state are preserved in the
      backlog until a slot becomes available.

    NOTE: TTFT reflects pure server-side cost (KV re-computation or reloading)
          since the timer starts only after the semaphore is acquired. Backlog
          wait reduces each session's firing cadence beyond what
          `delay_between_chat_turns` specifies — sessions are less frequent
          than configured.
    """
    request_counter = RequestCounter(
        max_requests=max_requests,
        total_sent_requests=0,
    )

    request_driver = ConcurrentTurnsRequestDriver(
        request_driver, semaphore, benchmark_should_end_time
    )

    # Queue holds (original_index, session) pairs so LoRA assignment is stable.
    # Pre-warmed sessions (prefix_turns > 0) are enqueued first so workers
    # always dequeue them before cold-start sessions. This ensures a warm
    # session is never blocked behind a cold session that is rate-gating.
    session_queue: asyncio.Queue[tuple[int, ChatSession]] = asyncio.Queue()
    indexed = sorted(
        enumerate(chat_sessions),
        key=lambda pair: 0 if pair[1].prefix_turns > 0 else 1,
    )
    for idx, session in indexed:
        await session_queue.put((idx, session))

    num_workers = min(max_concurrent_conversations, len(chat_sessions))
    worker_outputs: list[dict[str, list[RequestFuncOutput]]] = [
        {} for _ in range(num_workers)
    ]

    use_rate_pacing = request_rate != float("inf")
    session_rate_gate: asyncio.Queue[None] = asyncio.Queue()

    async def _emit_rate_tokens() -> None:
        """Produce session-start permits at the target Poisson/gamma rate."""
        while True:
            await asyncio.sleep(_poisson_interval(request_rate, burstiness))
            await session_rate_gate.put(None)

    async def _wait_for_rate_token() -> bool:
        """Block until a session-start permit arrives.

        Returns True if the benchmark deadline passed before a permit was
        granted. Polls with a bounded timeout so the worker can observe the
        deadline rather than blocking on the gate indefinitely.
        """
        while not deadline_passed(benchmark_should_end_time):
            remaining_s = deadline_remaining_s(benchmark_should_end_time)
            timeout = 1.0 if remaining_s is None else min(1.0, remaining_s)
            try:
                await asyncio.wait_for(session_rate_gate.get(), timeout=timeout)
                return False
            except asyncio.TimeoutError:
                continue
        return True

    async def _conversation_worker(worker_idx: int) -> None:
        if warmup_delay_ms > 0:
            sleep_s = worker_idx * warmup_delay_ms / 1000
            if exceeds_deadline(sleep_s, benchmark_should_end_time):
                return
            await asyncio.sleep(sleep_s)

        session_count = 0
        while True:
            if deadline_passed(benchmark_should_end_time):
                return

            try:
                idx, chat_session = session_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            # Pre-warmed sessions (prefix_turns > 0) have their KV cache
            # populated; start them immediately. Cold-start sessions
            # (prefix_turns == 0) are rate-gated like new arrivals.
            if use_rate_pacing and chat_session.prefix_turns == 0:
                if await _wait_for_rate_token():
                    return

            lora_id = (
                lora_manager.get_lora_for_request(idx) if lora_manager else None
            )
            outputs = await chat_session_driver(
                model_id=model_id if lora_id is None else lora_id,
                api_url=api_url,
                request_driver=request_driver,
                request_counter=request_counter,
                chat_session=chat_session,
                max_chat_len=tokenizer.model_max_length,
                sampling=sampling,
                ignore_first_turn_stats=ignore_first_turn_stats,
                benchmark_should_end_time=benchmark_should_end_time,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
                est_ttft_ms=est_ttft_ms,
                est_tpot_ms=est_tpot_ms,
                use_session_id_as_cache_salt=use_session_id_as_cache_salt,
            )
            session_id = (
                str(chat_session.id)
                if chat_session.id is not None
                else f"anonymous-w{worker_idx}-{session_count}"
            )
            session_count += 1
            worker_outputs[worker_idx].setdefault(session_id, []).extend(
                outputs
            )

    rate_emitter_task: asyncio.Task[None] | None = None
    if use_rate_pacing:
        rate_emitter_task = asyncio.create_task(_emit_rate_tokens())
    try:
        async with TaskGroup() as tg:
            for i in range(num_workers):
                tg.create_task(_conversation_worker(i))
    finally:
        if rate_emitter_task is not None:
            rate_emitter_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await rate_emitter_task

    outputs_by_session: dict[str, list[RequestFuncOutput]] = {}
    for worker_dict in worker_outputs:
        for sid, outs in worker_dict.items():
            outputs_by_session.setdefault(sid, []).extend(outs)

    if deadline_passed(benchmark_should_end_time):
        total_turns = sum(len(v) for v in outputs_by_session.values())
        cancelled = sum(
            1 for v in outputs_by_session.values() for o in v if o.cancelled
        )
        logger.info(
            "Benchmark stopped by the duration limit"
            " (--max-benchmark-duration-s):"
            f" {total_turns} turns dispatched across"
            f" {len(outputs_by_session)} sessions,"
            f" {cancelled} cancelled in flight."
        )

    return outputs_by_session


async def chat_judge_session_driver(
    model_id: str,
    api_url: str,
    request_driver: RequestDriver,
    request_counter: RequestCounter,
    chat_session: ChatSession,
    max_output_tokens: int,
    sampling: SamplingConfig,
    benchmark_should_end_time: int | None = None,
    use_session_id_as_cache_salt: bool = False,
) -> list[RequestFuncOutput]:
    """Drive one chat-judge session: every turn already has its full
    context inlined as text in the user message, so we send
    ``[system?, user]`` per turn without accumulating assistant responses.

    If the session's first message is a system prompt
    (``source="system"``), it is prepended to every user turn; otherwise
    user turns are sent alone. Turns within a session run sequentially.
    """
    session_outputs: list[RequestFuncOutput] = []

    messages = chat_session.messages
    system_message: ChatMessage | None = None
    system_num_tokens = 0
    if messages and messages[0].source == "system":
        system_message = ChatMessage(
            role="system",
            content=[TextContentBlock(text=messages[0].content)],
        )
        system_num_tokens = messages[0].num_tokens
        user_messages = messages[1:]
    else:
        user_messages = messages

    for message in user_messages:
        if not request_counter.advance_until_max():
            break

        prompt_messages: list[ChatMessage] = []
        if system_message is not None:
            prompt_messages.append(system_message)
        prompt_messages.append(
            ChatMessage(
                role="user",
                content=[TextContentBlock(text=message.content)],
            )
        )
        request_input = RequestFuncInput(
            model=model_id,
            session_id=str(chat_session.id),
            sampling=sampling,
            prompt=prompt_messages,
            images=[],
            api_url=api_url,
            prompt_len=system_num_tokens + message.num_tokens,
            max_tokens=max_output_tokens,
            ignore_eos=False,
            cache_salt=str(chat_session.id)
            if use_session_id_as_cache_salt
            else None,
        )

        if deadline_passed(benchmark_should_end_time):
            response = RequestFuncOutput(
                cancelled=True, request_submit_time=time.perf_counter()
            )
        else:
            remaining_s = deadline_remaining_s(benchmark_should_end_time)
            try:
                raw_response = await asyncio.wait_for(
                    request_driver.request(request_input),
                    timeout=remaining_s,
                )
                if not isinstance(raw_response, RequestFuncOutput):
                    raise TypeError(
                        "Expected RequestFuncOutput in chat-judge benchmark flow."
                    )
                response = raw_response
                mark_cancelled_if_past_deadline(
                    response, benchmark_should_end_time
                )
            except asyncio.TimeoutError:
                response = RequestFuncOutput(
                    cancelled=True, request_submit_time=time.perf_counter()
                )

        session_outputs.append(response)

        if not response.success:
            if not response.cancelled:
                logger.error(
                    f"Ending chat-judge session {chat_session.id} due to "
                    f"server error response: {response.error}"
                )
            break

        if next_delay_ms := message.delay_until_next_message:
            sleep_s = next_delay_ms / 1000
            if exceeds_deadline(sleep_s, benchmark_should_end_time):
                return session_outputs
            await asyncio.sleep(sleep_s)

    return session_outputs


async def run_chat_judge_benchmark(
    *,
    chat_sessions: Sequence[ChatSession],
    max_output_tokens: int,
    max_requests: int,
    semaphore: contextlib.AbstractAsyncContextManager[None],
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    lora_manager: LoRABenchmarkManager | None,
    warmup_delay_ms: float,
    max_concurrency: int | None,
    sampling: SamplingConfig,
    use_session_id_as_cache_salt: bool = False,
) -> dict[str, list[RequestFuncOutput]]:
    """Run the chat-judge multi-turn scenario."""
    request_counter = RequestCounter(
        max_requests=max_requests,
        total_sent_requests=0,
    )
    outputs_by_session: dict[str, list[RequestFuncOutput]] = {}

    async def limited_session_driver(
        chat_session: ChatSession,
        session_idx: int,
    ) -> None:
        lora_id = None
        if lora_manager:
            lora_id = lora_manager.get_lora_for_request(session_idx)

        async with semaphore:
            outputs = await chat_judge_session_driver(
                model_id=model_id if lora_id is None else lora_id,
                api_url=api_url,
                request_driver=request_driver,
                request_counter=request_counter,
                chat_session=chat_session,
                max_output_tokens=max_output_tokens,
                sampling=sampling,
                benchmark_should_end_time=benchmark_should_end_time,
                use_session_id_as_cache_salt=use_session_id_as_cache_salt,
            )
        session_id = (
            str(chat_session.id)
            if chat_session.id is not None
            else f"anonymous-{session_idx}"
        )
        outputs_by_session[session_id] = outputs

    async with TaskGroup() as tg:
        for idx, chat_session in enumerate(chat_sessions):
            if (
                warmup_delay_ms > 0
                and max_concurrency
                and idx < max_concurrency
            ):
                sleep_s = warmup_delay_ms / 1000
                if exceeds_deadline(sleep_s, benchmark_should_end_time):
                    break
                await asyncio.sleep(sleep_s)
            tg.create_task(limited_session_driver(chat_session, idx))

    if benchmark_should_end_time is not None and not deadline_passed(
        benchmark_should_end_time
    ):
        logger.warning(
            "All chat-judge sessions completed before the time limit. "
            "Consider increasing --num-chat-sessions for more stable load."
        )

    return outputs_by_session
