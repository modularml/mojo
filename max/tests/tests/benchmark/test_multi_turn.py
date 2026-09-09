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
"""Tests for benchmark_shared.multi_turn."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from max.benchmark.benchmark_shared.config import SamplingConfig
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSession,
    SessionMessage,
    TextContentBlock,
)
from max.benchmark.benchmark_shared.multi_turn import (
    ConcurrentTurnsRequestDriver,
    chat_judge_session_driver,
    chat_session_driver,
    prerun_warmup_turns,
    run_chat_judge_benchmark,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
)
from max.benchmark.benchmark_shared.warmup import (
    _prefix_delays_ms,
    _prefix_occupancy_ms,
    pick_warmup_population,
)


class _CapturingDriver(RequestDriver):
    """Request driver that records all requests and returns success."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[RequestFuncInput] = []

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        assert isinstance(request_func_input, RequestFuncInput)
        self.calls.append(request_func_input)
        return RequestFuncOutput(
            success=True,
            latency=0.1,
            ttft=0.05,
            prompt_len=request_func_input.prompt_len,
            generated_text="ok",
        )


def _make_4turn_session(
    prefix_turns: int = 0,
    delay_ms: float = 1000.0,
) -> ChatSession:
    """Create a 4-turn chat session for testing prefix_turns behavior."""
    return ChatSession(
        id=0,
        messages=[
            SessionMessage(source="user", content="Turn 1", num_tokens=5),
            SessionMessage(
                source="assistant",
                content="",
                num_tokens=5,
                delay_until_next_message=delay_ms,
            ),
            SessionMessage(source="user", content="Turn 2", num_tokens=5),
            SessionMessage(
                source="assistant",
                content="",
                num_tokens=5,
                delay_until_next_message=delay_ms,
            ),
            SessionMessage(source="user", content="Turn 3", num_tokens=5),
            SessionMessage(
                source="assistant",
                content="",
                num_tokens=5,
                delay_until_next_message=delay_ms,
            ),
            SessionMessage(source="user", content="Turn 4", num_tokens=5),
            SessionMessage(
                source="assistant",
                content="",
                num_tokens=5,
            ),
        ],
        prefix_turns=prefix_turns,
    )


def _make_session_with_id(session_id: int, prefix_turns: int) -> ChatSession:
    """Helper: 4-turn session with a specific id."""
    session = _make_4turn_session(prefix_turns=prefix_turns)
    return dataclasses.replace(session, id=session_id)


def test_chat_session_driver_forwards_sampling_params() -> None:
    """Test that chat_session_driver forwards temperature, top_p, top_k."""

    captured_inputs: list[RequestFuncInput] = []

    class CapturingDriver(RequestDriver):
        async def request(
            self, request_func_input: BaseRequestFuncInput
        ) -> RequestFuncOutput:
            assert isinstance(request_func_input, RequestFuncInput)
            captured_inputs.append(request_func_input)
            return RequestFuncOutput(
                success=True,
                latency=0.1,
                ttft=0.05,
                prompt_len=request_func_input.prompt_len,
                generated_text="Hello",
            )

    async def run_test() -> None:
        chat_session = ChatSession(
            id=0,
            messages=[
                SessionMessage(source="user", content="Hi", num_tokens=5),
                SessionMessage(
                    source="assistant", content="Hello", num_tokens=5
                ),
            ],
        )
        request_counter = RequestCounter(max_requests=10, total_sent_requests=0)

        await chat_session_driver(
            model_id="test-model",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=CapturingDriver(),
            request_counter=request_counter,
            chat_session=chat_session,
            max_chat_len=4096,
            sampling=SamplingConfig(temperature=0.7, top_p=0.9, top_k=50),
        )

    asyncio.run(run_test())

    assert len(captured_inputs) == 1
    assert captured_inputs[0].sampling.temperature == 0.7
    assert captured_inputs[0].sampling.top_p == 0.9
    assert captured_inputs[0].sampling.top_k == 50


def test_chat_session_driver_run_prefix_prepends_first_turn() -> None:
    """First user message gets the run prefix when run_prefix is set."""

    captured: list[RequestFuncInput] = []

    class CapturingDriver(RequestDriver):
        async def request(
            self, request_func_input: BaseRequestFuncInput
        ) -> RequestFuncOutput:
            assert isinstance(request_func_input, RequestFuncInput)
            captured.append(request_func_input)
            return RequestFuncOutput(
                success=True,
                latency=0.1,
                ttft=0.05,
                prompt_len=request_func_input.prompt_len,
                generated_text="Hello",
            )

    async def run_test() -> None:
        chat_session = ChatSession(
            id=0,
            messages=[
                SessionMessage(source="user", content="Hi", num_tokens=5),
                SessionMessage(
                    source="assistant", content="Hello", num_tokens=5
                ),
                SessionMessage(source="user", content="Again", num_tokens=5),
                SessionMessage(source="assistant", content="Hi", num_tokens=5),
            ],
        )
        request_counter = RequestCounter(max_requests=10, total_sent_requests=0)

        await chat_session_driver(
            model_id="test-model",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=CapturingDriver(),
            request_counter=request_counter,
            chat_session=chat_session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
            run_prefix="RUN-UUID: ",
            run_prefix_len=4,
        )

    asyncio.run(run_test())

    assert len(captured) == 2
    assert isinstance(captured[0].prompt, list)
    first_user_content_block = captured[0].prompt[0].content[0]
    assert isinstance(first_user_content_block, TextContentBlock)
    first_user_text = first_user_content_block.text
    assert first_user_text.endswith("Hi")
    assert first_user_text != "Hi"
    assert isinstance(captured[1].prompt, list)
    second_user_content_block = captured[1].prompt[2].content[0]
    assert isinstance(second_user_content_block, TextContentBlock)
    second_user_text = second_user_content_block.text
    assert second_user_text == "Again"


def test_prefix_turns_excluded_from_results() -> None:
    """With prefix_turns=2, a 4-turn session should return only 2 results."""

    async def run_test() -> list[RequestFuncOutput]:
        session = _make_4turn_session(prefix_turns=2)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        return await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )

    outputs = asyncio.run(run_test())
    assert len(outputs) == 2


def test_prefix_turns_dont_count_against_max_requests() -> None:
    """Prefix turns should not consume max_requests budget."""

    async def run_test() -> tuple[list[RequestFuncOutput], int, int]:
        session = _make_4turn_session(prefix_turns=2)
        counter = RequestCounter(max_requests=2)
        driver = _CapturingDriver()
        outputs = await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )
        return outputs, len(driver.calls), counter.total_sent_requests

    outputs, total_calls, counter_value = asyncio.run(run_test())
    # Prefix turns are built locally, so only measured turns hit the server.
    assert total_calls == 2
    assert counter_value == 2
    assert len(outputs) == 2


def test_prefix_turns_no_server_or_delays() -> None:
    """Prefix turns are built locally: no server calls. Pre-warmed sessions
    get a phase-spread jitter before turn 3, plus the turn 3→4 inter-turn delay."""
    sleep_calls: list[float] = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def run_test() -> int:
        session = _make_4turn_session(prefix_turns=2, delay_ms=5000.0)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await chat_session_driver(
                model_id="test",
                api_url="http://localhost:8000/v1/chat/completions",
                request_driver=driver,
                request_counter=counter,
                chat_session=session,
                max_chat_len=4096,
                sampling=SamplingConfig(),
            )
        return len(driver.calls)

    total_calls = asyncio.run(run_test())
    # Prefix turns don't hit the server.
    assert total_calls == 2
    # sleep[0]: phase-spread jitter in [0, 5.0] from messages[3] (last prefix asst)
    # sleep[1]: inter-turn delay after turn 3 (5.0 s)
    assert len(sleep_calls) == 2
    assert 0.0 <= sleep_calls[0] <= 5.0
    assert sleep_calls[1] == pytest.approx(5.0)


def test_prefix_turns_zero_is_noop() -> None:
    """prefix_turns=0 should behave identically to the old code."""

    async def run_test() -> list[RequestFuncOutput]:
        session = _make_4turn_session(prefix_turns=0)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        return await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
        )

    outputs = asyncio.run(run_test())
    assert len(outputs) == 4


def test_prerun_warmup_turns_one_request_per_session_with_prefix() -> None:
    """Each session with prefix_turns>0 produces exactly one warmup request."""
    sessions = [
        _make_session_with_id(0, prefix_turns=0),
        _make_session_with_id(1, prefix_turns=2),
        _make_session_with_id(2, prefix_turns=0),
        _make_session_with_id(3, prefix_turns=3),
    ]
    driver = _CapturingDriver()

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=sessions,
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            sampling=SamplingConfig(),
            max_concurrency=128,
        )

    asyncio.run(run())
    # Sessions 1 and 3 fire one request each; sessions with
    # prefix_turns=0 fire nothing.
    assert len(driver.calls) == 2
    assert {call.session_id for call in driver.calls} == {"1", "3"}
    assert all(call.ignore_eos is True for call in driver.calls)


def test_prerun_warmup_turns_raises_on_failed_request() -> None:
    """A failed warmup request fails the benchmark instead of being silently
    ignored (matches the check `prime_shared_contexts` has for single-turn).
    """
    sessions = [_make_session_with_id(0, prefix_turns=2)]

    class _FailingDriver(RequestDriver):
        async def request(
            self, request_func_input: BaseRequestFuncInput
        ) -> RequestFuncOutput:
            assert isinstance(request_func_input, RequestFuncInput)
            return RequestFuncOutput(
                success=False,
                error="simulated timeout",
                prompt_len=request_func_input.prompt_len,
            )

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=sessions,
            request_driver=_FailingDriver(),
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            sampling=SamplingConfig(),
            max_concurrency=128,
        )

    with pytest.raises(ValueError, match="simulated timeout"):
        asyncio.run(run())


def test_prerun_warmup_turns_request_prompt_is_last_turn_prefix() -> None:
    """For prefix_turns=N, the one request's prompt = dataset messages[0:2N-1]."""
    session = _make_session_with_id(0, prefix_turns=3)
    msgs = session.messages
    driver = _CapturingDriver()

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=[session],
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            sampling=SamplingConfig(),
            max_concurrency=128,
        )

    asyncio.run(run())
    assert len(driver.calls) == 1
    call = driver.calls[0]
    # prefix_turns=3 means the prompt is messages[0:5]:
    # [user_1, asst_1, user_2, asst_2, user_3].
    assert len(call.prompt) == 5
    last = call.prompt[-1]
    assert not isinstance(last, str)
    block = last.content[0]
    assert isinstance(block, TextContentBlock)
    assert block.text == msgs[4].content  # user_3
    assert call.max_tokens == msgs[5].num_tokens  # dataset turn-3 output


def test_prerun_warmup_turns_cross_session_parallelism() -> None:
    """Requests for different sessions fire concurrently."""

    enter_count = 0
    max_concurrent_seen = 0
    in_flight = 0
    release = asyncio.Event()

    class _SlowDriver(RequestDriver):
        async def request(
            self, request_func_input: BaseRequestFuncInput
        ) -> RequestFuncOutput:
            nonlocal enter_count, max_concurrent_seen, in_flight
            enter_count += 1
            in_flight += 1
            max_concurrent_seen = max(max_concurrent_seen, in_flight)
            try:
                if enter_count >= 3:
                    release.set()
                await release.wait()
            finally:
                in_flight -= 1
            assert isinstance(request_func_input, RequestFuncInput)
            return RequestFuncOutput(
                success=True,
                latency=0.0,
                ttft=0.0,
                prompt_len=request_func_input.prompt_len,
                generated_text="ok",
            )

    # 3 sessions, one request each. All 3 should overlap.
    sessions = [_make_session_with_id(i, prefix_turns=2) for i in range(3)]

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=sessions,
            request_driver=_SlowDriver(),
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            sampling=SamplingConfig(),
            max_concurrency=128,
        )

    asyncio.run(run())
    # If serialised, only one would be in flight before release fires.
    assert max_concurrent_seen == 3


def test_prerun_warmup_turns_noop_without_prefix_sessions() -> None:
    """When no session has prefix_turns > 0, no requests are issued."""
    sessions = [_make_session_with_id(idx, prefix_turns=0) for idx in range(3)]
    driver = _CapturingDriver()

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=sessions,
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            sampling=SamplingConfig(),
            max_concurrency=128,
        )

    asyncio.run(run())
    assert driver.calls == []


def test_prerun_warmup_turns_respects_chat_length_budget() -> None:
    """Turns that would overflow max_chat_len are dropped."""
    # Each turn = 5 user + 5 asst tokens. With max_chat_len=15, only
    # turn 1 fits.
    session = _make_session_with_id(0, prefix_turns=3)
    driver = _CapturingDriver()

    async def run() -> None:
        await prerun_warmup_turns(
            sessions=[session],
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=15,
            sampling=SamplingConfig(),
            max_concurrency=128,
        )

    asyncio.run(run())
    assert len(driver.calls) == 1


def _make_chat_judge_session(
    session_id: int,
    num_user_turns: int = 2,
) -> ChatSession:
    """Self-contained per-turn session with a system message at turn 0
    and `num_user_turns` user turns (no assistant turns)."""
    messages: list[SessionMessage] = [
        SessionMessage(
            source="system", content="You are a judge.", num_tokens=4
        )
    ]
    for i in range(num_user_turns):
        messages.append(
            SessionMessage(source="user", content=f"Item {i}", num_tokens=5)
        )
    return ChatSession(id=session_id, messages=messages)


@pytest.mark.asyncio
async def test_run_chat_judge_benchmark_smoke_with_system_role() -> None:
    """Orchestrator runs multiple sessions with system role end-to-end:
    outputs are collected per session and every call carries the
    system+user prompt without crashing through the TaskGroup wrapper."""
    driver = _CapturingDriver()
    sessions = [
        _make_chat_judge_session(0, num_user_turns=2),
        _make_chat_judge_session(1, num_user_turns=3),
    ]
    outputs_by_session = await run_chat_judge_benchmark(
        chat_sessions=sessions,
        max_output_tokens=16,
        max_requests=10,
        semaphore=asyncio.Semaphore(len(sessions)),
        benchmark_should_end_time=None,
        request_driver=driver,
        model_id="test-model",
        api_url="http://localhost:8000/v1/chat/completions",
        lora_manager=None,
        warmup_delay_ms=0.0,
        max_concurrency=None,
        sampling=SamplingConfig(),
    )

    assert set(outputs_by_session.keys()) == {"0", "1"}
    assert len(outputs_by_session["0"]) == 2
    assert len(outputs_by_session["1"]) == 3
    assert len(driver.calls) == 5
    for call in driver.calls:
        assert isinstance(call.prompt, list)
        assert len(call.prompt) == 2
        assert call.prompt[0].role == "system"
        assert call.prompt[1].role == "user"
        assert call.prompt_len == 4 + 5  # system + user tokens


def test_chat_judge_driver_sleeps_between_turns() -> None:
    """The driver sleeps each user message's delay between turns and skips
    the sleep after the final turn (which carries no delay)."""
    session = ChatSession(
        id=0,
        messages=[
            SessionMessage(source="system", content="judge", num_tokens=4),
            SessionMessage(
                source="user",
                content="i0",
                num_tokens=5,
                delay_until_next_message=250.0,
            ),
            SessionMessage(
                source="user",
                content="i1",
                num_tokens=5,
                delay_until_next_message=750.0,
            ),
            SessionMessage(source="user", content="i2", num_tokens=5),
        ],
    )
    sleep_calls: list[float] = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def run_test() -> list[RequestFuncOutput]:
        with patch("asyncio.sleep", side_effect=mock_sleep):
            return await chat_judge_session_driver(
                model_id="test",
                api_url="http://localhost:8000/v1/chat/completions",
                request_driver=_CapturingDriver(),
                request_counter=RequestCounter(max_requests=100),
                chat_session=session,
                max_output_tokens=16,
                sampling=SamplingConfig(),
                benchmark_should_end_time=None,
            )

    outputs = asyncio.run(run_test())
    assert len(outputs) == 3
    # Inter-turn delays (seconds) fire between turns only, not after the last.
    assert sleep_calls == pytest.approx([0.25, 0.75])


def test_chat_judge_driver_interturn_sleep_deadline_skip() -> None:
    """An inter-turn delay that would exceed the deadline returns early."""
    session = ChatSession(
        id=0,
        messages=[
            SessionMessage(
                source="user",
                content="i0",
                num_tokens=5,
                delay_until_next_message=60_000,  # 60 s
            ),
            SessionMessage(source="user", content="i1", num_tokens=5),
        ],
    )

    async def run_test() -> list[RequestFuncOutput]:
        deadline = time.perf_counter_ns() + 1_000_000  # 1 ms from now
        return await chat_judge_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=_CapturingDriver(),
            request_counter=RequestCounter(max_requests=100),
            chat_session=session,
            max_output_tokens=16,
            sampling=SamplingConfig(),
            benchmark_should_end_time=deadline,
        )

    outputs = asyncio.run(run_test())
    assert len(outputs) == 1  # only turn 1 completed before the deadline


def test_concurrent_turns_driver_expired_deadline_cancels_without_calling_base() -> (
    None
):
    """A turn that acquires the semaphore after the deadline is cancelled, not forwarded."""
    semaphore = asyncio.Semaphore(10)
    mock_driver = AsyncMock()
    mock_driver.tokenizer = None
    driver = ConcurrentTurnsRequestDriver(
        mock_driver,
        semaphore,
        benchmark_should_end_time=time.perf_counter_ns() - 1,
    )

    output_cls = MagicMock()
    output_cls.return_value = MagicMock()
    inp = MagicMock()
    inp.get_output_type.return_value = output_cls

    asyncio.run(driver.request(inp))

    assert output_cls.call_args.kwargs["cancelled"] is True
    mock_driver.request.assert_not_called()


def _make_prewarmed_session(
    prefix_turns: int = 2, delay_ms: float = 2000.0
) -> ChatSession:
    """4-turn session with delay_ms on the last prefix assistant turn."""
    messages = []
    for i in range(4):
        messages.append(
            SessionMessage(source="user", content=f"U{i}", num_tokens=5)
        )
        delay = delay_ms if i == prefix_turns - 1 else 0.0
        messages.append(
            SessionMessage(
                source="assistant",
                content=f"A{i}",
                num_tokens=5,
                delay_until_next_message=delay if delay > 0 else None,
            )
        )
    return ChatSession(id=0, messages=messages, prefix_turns=prefix_turns)


def test_prewarmed_phase_spread_uses_correct_message() -> None:
    """Phase-spread jitter draws from messages[content_idx-1], not messages[content_idx+1]."""
    sleep_calls: list[float] = []
    uniform_calls: list[tuple[float, float]] = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def mock_uniform(a: float, b: float) -> float:
        uniform_calls.append((a, b))
        return b * 0.5  # deterministic midpoint

    async def run_test() -> int:
        session = _make_prewarmed_session(prefix_turns=2, delay_ms=2000.0)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        with (
            patch("asyncio.sleep", side_effect=mock_sleep),
            patch("random.uniform", side_effect=mock_uniform),
        ):
            await chat_session_driver(
                model_id="test",
                api_url="http://localhost:8000/v1/chat/completions",
                request_driver=driver,
                request_counter=counter,
                chat_session=session,
                max_chat_len=4096,
                sampling=SamplingConfig(),
            )
        return len(driver.calls)

    total_calls = asyncio.run(run_test())
    assert total_calls == 2  # 2 measured turns
    # uniform was called with (0, 2000.0) — the delay on messages[3] (content_idx-1=3)
    assert len(uniform_calls) >= 1
    assert uniform_calls[0] == (0, 2000.0)
    # sleep was called with 0.5 * 2000/1000 = 1.0 before the first request
    assert sleep_calls[0] == pytest.approx(1.0)


def test_prewarmed_phase_spread_deadline_skip() -> None:
    """Phase-spread jitter returns early when the sleep would exceed the deadline."""

    async def run_test() -> list[RequestFuncOutput]:
        session = _make_prewarmed_session(prefix_turns=2, delay_ms=5000.0)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        deadline = time.perf_counter_ns()  # already at deadline
        return await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
            benchmark_should_end_time=deadline,
        )

    outputs = asyncio.run(run_test())
    assert outputs == []  # returned before any request


def test_interturn_sleep_deadline_skip() -> None:
    """Inter-turn sleep exceeding the deadline causes early return after turn 1."""

    async def run_test() -> list[RequestFuncOutput]:
        session = ChatSession(
            id=0,
            messages=[
                SessionMessage(source="user", content="Q1", num_tokens=5),
                SessionMessage(
                    source="assistant",
                    content="",
                    num_tokens=5,
                    delay_until_next_message=60_000,  # 60 s
                ),
                SessionMessage(source="user", content="Q2", num_tokens=5),
                SessionMessage(source="assistant", content="", num_tokens=5),
            ],
        )
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        deadline = time.perf_counter_ns() + 1_000_000  # 1 ms from now
        return await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            sampling=SamplingConfig(),
            benchmark_should_end_time=deadline,
        )

    outputs = asyncio.run(run_test())
    assert len(outputs) == 1  # only turn 1 completed before deadline


# ---------------------------------------------------------------------------
# warmup.py: delay-biased session and prefix-turn selection
# ---------------------------------------------------------------------------


def _make_session_with_delays(
    session_id: int,
    delays_ms: list[float],
) -> ChatSession:
    """Build a session whose inter-turn delays match ``delays_ms``.

    Each element in ``delays_ms`` is the delay on the assistant message of
    that turn.  The session has ``len(delays_ms)`` turns (one user+assistant
    pair per entry).
    """
    messages: list[SessionMessage] = []
    for i, d in enumerate(delays_ms):
        messages.append(
            SessionMessage(source="user", content=f"U{i}", num_tokens=5)
        )
        messages.append(
            SessionMessage(
                source="assistant",
                content=f"A{i}",
                num_tokens=5,
                delay_until_next_message=d if d > 0 else None,
            )
        )
    return ChatSession(id=session_id, messages=messages)


def test_prefix_delays_ms_returns_correct_array() -> None:
    """_prefix_delays_ms maps prefix position k to messages[2k-1]'s delay."""
    session = _make_session_with_delays(0, [100.0, 200.0, 300.0])
    # 3 turns → valid prefix_turns in {1, 2} → delays for turns 0 and 1
    delays = _prefix_delays_ms(session)
    assert list(delays) == pytest.approx([100.0, 200.0])


def test_prefix_delays_ms_single_turn_is_empty() -> None:
    """Single-turn sessions return an empty array (no valid prefix position)."""
    session = _make_session_with_delays(0, [500.0])
    assert len(_prefix_delays_ms(session)) == 0


def test_prefix_delays_ms_none_delay_stored_as_zero() -> None:
    """Turns with delay_until_next_message=None are stored as 0.0."""
    # 3-turn session: delays on turns 0 and 1 → array length 2
    session = _make_session_with_delays(0, [0.0, 400.0, 0.0])
    delays = _prefix_delays_ms(session)
    assert len(delays) == 2
    assert delays[0] == pytest.approx(0.0)  # turn 0 has no delay
    assert delays[1] == pytest.approx(400.0)  # turn 1 has 400 ms delay


def test_pick_warmup_population_delay_biased_never_picks_prefix_zero() -> None:
    """When delays are configured, prefix_turns is always >= 1 (session is mid-sleep)."""
    rng = np.random.default_rng(42)
    sessions = [
        _make_session_with_delays(i, [1000.0, 2000.0, 3000.0])
        for i in range(20)
    ]
    result, report = pick_warmup_population(
        sessions,
        warmup_count=10,
        warmup_to_steady_state=True,
        warmup_oversample_factor=2,
        main_pool_target=0,
        rng=rng,
        delay_biased=True,
    )
    warmup = result[:10]
    assert all(s.prefix_turns >= 1 for s in warmup)
    assert report is not None
    assert report.delay_biased is True
    assert report.weight_unit == "delay-ms"


def test_pick_warmup_population_delay_biased_prefix_proportional_to_delay() -> (
    None
):
    """prefix_turns selection is proportional to the delay at each position.

    Session has 3 turns with delays [100, 900] ms.  Over many draws the
    fraction of picks at prefix_turns=1 should be ~10% and at
    prefix_turns=2 should be ~90%.
    """
    # Single session so all picks come from it.
    session = _make_session_with_delays(0, [100.0, 900.0, 0.0])
    # Need a large pool so PPS doesn't cap the single session.
    sessions = [dataclasses.replace(session, id=i) for i in range(200)]

    rng = np.random.default_rng(0)
    picks_at_1 = 0
    picks_at_2 = 0
    n_trials = 100
    for _ in range(n_trials):
        result, _ = pick_warmup_population(
            sessions,
            warmup_count=1,
            warmup_to_steady_state=True,
            warmup_oversample_factor=2,
            main_pool_target=0,
            rng=rng,
            delay_biased=True,
        )
        pt = result[0].prefix_turns
        if pt == 1:
            picks_at_1 += 1
        elif pt == 2:
            picks_at_2 += 1

    # Expect ~10 picks at 1 and ~90 picks at 2 (±20 at 3-sigma for binomial).
    assert 0 <= picks_at_1 <= 30, f"too many/few prefix_turns=1: {picks_at_1}"
    assert 70 <= picks_at_2 <= 100, f"too many/few prefix_turns=2: {picks_at_2}"


def test_pick_warmup_population_no_delays_fallback_to_turn_counts() -> None:
    """With the flag on but no delays present, falls back to turn-count weighting."""
    rng = np.random.default_rng(7)
    sessions = [_make_session_with_delays(i, [0.0, 0.0]) for i in range(20)]
    _result, report = pick_warmup_population(
        sessions,
        warmup_count=10,
        warmup_to_steady_state=True,
        warmup_oversample_factor=2,
        main_pool_target=0,
        rng=rng,
        delay_biased=True,
    )
    assert report is not None
    assert report.delay_biased is False
    assert report.weight_unit == "turns"


def test_pick_warmup_population_delay_biased_off_by_default() -> None:
    """Delays present but flag off (the default): stays turn-based."""
    rng = np.random.default_rng(11)
    sessions = [
        _make_session_with_delays(i, [1000.0, 2000.0, 3000.0])
        for i in range(20)
    ]
    _result, report = pick_warmup_population(
        sessions,
        warmup_count=10,
        warmup_to_steady_state=True,
        warmup_oversample_factor=2,
        main_pool_target=0,
        rng=rng,
    )
    assert report is not None
    assert report.delay_biased is False
    assert report.weight_unit == "turns"


# ---------------------------------------------------------------------------
# Runtime-aware (occupancy) warmup weighting.
# ---------------------------------------------------------------------------


def _make_session_with_delays_and_outputs(
    session_id: int,
    delays_ms: list[float],
    output_lens: list[int],
) -> ChatSession:
    """Like ``_make_session_with_delays`` but with per-turn assistant output
    token counts so occupancy (``R_k + D_k``) can be exercised."""
    assert len(delays_ms) == len(output_lens)
    messages: list[SessionMessage] = []
    for i, (d, out_len) in enumerate(zip(delays_ms, output_lens, strict=False)):
        messages.append(
            SessionMessage(source="user", content=f"U{i}", num_tokens=5)
        )
        messages.append(
            SessionMessage(
                source="assistant",
                content=f"A{i}",
                num_tokens=out_len,
                delay_until_next_message=d if d > 0 else None,
            )
        )
    return ChatSession(id=session_id, messages=messages)


def test_prefix_occupancy_zero_estimates_equals_delays() -> None:
    """With zero runtime estimates, occupancy reduces to the inter-turn delay."""
    session = _make_session_with_delays_and_outputs(
        0, [100.0, 200.0, 300.0], [10, 20, 30]
    )
    occ = _prefix_occupancy_ms(session)
    assert list(occ) == pytest.approx(list(_prefix_delays_ms(session)))
    assert list(occ) == pytest.approx([100.0, 200.0])


def test_prefix_occupancy_adds_runtime() -> None:
    """Occupancy at position k is D_k + ttft + tpot * output_len_k."""
    # 3 turns → valid prefix positions for turns 0 and 1.
    session = _make_session_with_delays_and_outputs(
        0, [100.0, 200.0, 0.0], [10, 50, 7]
    )
    occ = _prefix_occupancy_ms(session, est_ttft_ms=5.0, est_tpot_ms=2.0)
    # turn 0: 100 + 5 + 2*10 = 125; turn 1: 200 + 5 + 2*50 = 305
    assert list(occ) == pytest.approx([125.0, 305.0])


def test_pick_warmup_population_occupancy_weight_unit_label() -> None:
    """Runtime estimates flip the diagnostic unit to occupancy-ms."""
    rng = np.random.default_rng(3)
    sessions = [
        _make_session_with_delays_and_outputs(
            i, [1000.0, 1000.0, 1000.0], [10, 10, 10]
        )
        for i in range(20)
    ]
    _result, report = pick_warmup_population(
        sessions,
        warmup_count=10,
        warmup_to_steady_state=True,
        warmup_oversample_factor=2,
        main_pool_target=0,
        rng=rng,
        delay_biased=True,
        est_ttft_ms=50.0,
        est_tpot_ms=10.0,
    )
    assert report is not None
    assert report.delay_biased is True
    assert report.weight_unit == "occupancy-ms"


def test_pick_warmup_population_occupancy_biases_long_output_position() -> None:
    """Equal delays but unequal output_len: high-runtime position dominates.

    Turn 0 has a short output, turn 1 a long one, with equal delays. Under
    delay-only weighting the two positions are equiprobable; with a large TPOT
    estimate the long-output position should be picked far more often.
    """
    # 3 turns, equal 100ms delays, outputs [10, 1000] at positions 0, 1.
    session = _make_session_with_delays_and_outputs(
        0, [100.0, 100.0, 0.0], [10, 1000, 7]
    )
    sessions = [dataclasses.replace(session, id=i) for i in range(200)]

    rng = np.random.default_rng(0)
    picks_at_1 = 0
    picks_at_2 = 0
    n_trials = 100
    for _ in range(n_trials):
        result, _ = pick_warmup_population(
            sessions,
            warmup_count=1,
            warmup_to_steady_state=True,
            warmup_oversample_factor=2,
            main_pool_target=0,
            rng=rng,
            delay_biased=True,
            est_ttft_ms=0.0,
            est_tpot_ms=10.0,  # R_1 = 10*1000 = 10000 >> R_0 = 100
        )
        pt = result[0].prefix_turns
        if pt == 1:
            picks_at_1 += 1
        elif pt == 2:
            picks_at_2 += 1

    # occupancy: pos0 = 100+100 = 200, pos1 = 100+10000 = 10100 → ~98% at pos1.
    assert picks_at_2 >= 90, (
        f"expected long-output position to dominate: {picks_at_2}"
    )
    assert picks_at_1 <= 10


def test_pick_warmup_population_runtime_engages_with_zero_delays() -> None:
    """No inter-turn delays but nonzero estimates: occupancy still biases.

    Without estimates this falls back to turn-count weighting (delay_biased
    False). With estimates, occupancy = R_k > 0 so biasing engages.
    """
    rng = np.random.default_rng(5)
    sessions = [
        _make_session_with_delays_and_outputs(i, [0.0, 0.0, 0.0], [10, 20, 30])
        for i in range(20)
    ]
    _result, report = pick_warmup_population(
        sessions,
        warmup_count=10,
        warmup_to_steady_state=True,
        warmup_oversample_factor=2,
        main_pool_target=0,
        rng=rng,
        delay_biased=True,
        est_ttft_ms=1.0,
        est_tpot_ms=1.0,
    )
    assert report is not None
    assert report.delay_biased is True
    assert report.weight_unit == "occupancy-ms"


def _jitter_trial_fired_immediately() -> bool:
    """Run one pre-warmed session and report whether the initial jitter slept.

    prefix_turns=2 → just-completed turn is messages[3] (output 5 tokens,
    delay 10ms). With tpot=1000ms/tok, R = 5000ms >> D = 10ms, so the phase
    draw lands in the generation portion (fire immediately) ~99.8% of the time.
    The turn 3→4 inter-turn delay (10ms = 0.01s) always fires regardless.
    """
    sleep_calls: list[float] = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def run_test() -> None:
        session = _make_4turn_session(prefix_turns=2, delay_ms=10.0)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await chat_session_driver(
                model_id="test",
                api_url="http://localhost:8000/v1/chat/completions",
                request_driver=driver,
                request_counter=counter,
                chat_session=session,
                max_chat_len=4096,
                sampling=SamplingConfig(),
                est_ttft_ms=0.0,
                est_tpot_ms=1000.0,
            )

    asyncio.run(run_test())
    # Only the inter-turn delay (0.01s) firing means the jitter collapsed to
    # fire-now; an extra sleep means the draw landed in the sleep portion.
    jitter_fired = [s for s in sleep_calls if s != pytest.approx(0.01)]
    return not jitter_fired


def test_jitter_collapse_fires_immediately_when_runtime_dominates() -> None:
    """When R_k >> D_k, most pre-warmed sessions fire immediately (no sleep)."""
    n_trials = 60
    fire_immediately = sum(
        _jitter_trial_fired_immediately() for _ in range(n_trials)
    )
    # R/(R+D) = 5000/5010 ≈ 0.998, so nearly every trial fires immediately.
    assert fire_immediately >= 55, (
        f"expected mostly fire-now: {fire_immediately}"
    )


def test_warmup_runtime_estimates_require_delay_biased() -> None:
    """Setting estimates without --warmup-delay-biased is a config error."""
    from max.benchmark.benchmark_shared.config import ServingBenchmarkConfig

    with pytest.raises(ValueError, match="warmup-delay-biased"):
        ServingBenchmarkConfig(warmup_delay_estimated_ttft_ms=10.0)

    with pytest.raises(ValueError, match="warmup-delay-biased"):
        ServingBenchmarkConfig(warmup_delay_estimated_tpot_ms=1.0)

    # Valid when paired with the flag.
    cfg = ServingBenchmarkConfig(
        warmup_delay_biased=True,
        warmup_delay_estimated_ttft_ms=10.0,
        warmup_delay_estimated_tpot_ms=1.0,
    )
    assert cfg.warmup_delay_estimated_ttft_ms == 10.0
