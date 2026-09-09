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

"""Single-turn benchmark request generation and execution."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, Sequence

try:
    from asyncio import TaskGroup  # type: ignore[attr-defined]  # added in 3.11
except ImportError:
    from taskgroup import TaskGroup  # Python < 3.11 backport

import numpy as np
from max.benchmark.benchmark_shared.config import (
    PIXEL_GENERATION_TASKS,
    BenchmarkTask,
    SamplingConfig,
)
from max.benchmark.benchmark_shared.datasets import SampledRequest
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSamples,
    PixelGenerationSampledRequest,
    Samples,
    TextContentBlock,
)
from max.benchmark.benchmark_shared.lora_benchmark_manager import (
    LoRABenchmarkManager,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    BaseRequestFuncOutput,
    ChatMessage,
    PixelGenerationRequestFuncInput,
    RequestDriver,
    RequestFuncInput,
    mark_cancelled_if_past_deadline,
    progressbar_request_driver,
)
from max.benchmark.benchmark_shared.utils import (
    deadline_passed,
    deadline_remaining_s,
    exceeds_deadline,
)

logger = logging.getLogger(__name__)


def _prepend_run_prefix_to_formatted_prompt(
    prompt: str | list[ChatMessage], run_prefix: str
) -> str | list[ChatMessage]:
    """Return a new prompt with `run_prefix` prepended to the first message."""
    if isinstance(prompt, str):
        return run_prefix + prompt

    # Chat format: prepend to the text content of the first message.
    # content may be a plain string or a list of typed content blocks.
    if not prompt:
        raise ValueError("run_prefix: empty prompt list")
    msg = prompt[0]
    content = msg.content
    if isinstance(content, str):
        new_msg = ChatMessage(role=msg.role, content=run_prefix + content)
    elif isinstance(content, list):
        text_block_idx = next(
            (
                idx
                for idx, block in enumerate(content)
                if isinstance(block, TextContentBlock)
            ),
            None,
        )
        if text_block_idx is None:
            raise ValueError(
                "run_prefix: no text block found in content list; cannot"
                " prepend run prefix"
            )
        text_block = content[text_block_idx]
        assert isinstance(text_block, TextContentBlock)
        new_block = TextContentBlock(text=run_prefix + text_block.text)
        new_content = [
            *content[:text_block_idx],
            new_block,
            *content[text_block_idx + 1 :],
        ]
        new_msg = ChatMessage(role=msg.role, content=new_content)
    else:
        raise ValueError(
            "run_prefix: unsupported prompt shape for first message"
        )
    return [new_msg, *prompt[1:]]


def build_single_turn_request_input(
    *,
    benchmark_task: BenchmarkTask,
    request: SampledRequest,
    model_id: str,
    lora_id: str | None,
    api_url: str,
    sampling: SamplingConfig,
    max_output_len: int | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> BaseRequestFuncInput:
    request_model_id = model_id if lora_id is None else lora_id
    if benchmark_task == "text-generation":
        max_tokens = min(
            filter(None, (request.output_len, max_output_len)),
            default=None,
        )
        prompt = request.prompt_formatted
        prompt_len = request.prompt_len
        if run_prefix:
            prompt = _prepend_run_prefix_to_formatted_prompt(prompt, run_prefix)
            prompt_len = prompt_len + run_prefix_len
        return RequestFuncInput(
            model=request_model_id,
            session_id=None,
            sampling=sampling,
            prompt=prompt,
            images=request.encoded_images,
            api_url=api_url,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            ignore_eos=request.ignore_eos,
            response_format=request.response_format,
            tools=request.tools,
        )
    if benchmark_task in PIXEL_GENERATION_TASKS:
        if not isinstance(request, PixelGenerationSampledRequest):
            raise TypeError(
                "pixel-generation benchmark requires PixelGenerationSampledRequest."
            )
        prompt = request.prompt_formatted
        if run_prefix and isinstance(prompt, str):
            prompt = run_prefix + prompt
        return PixelGenerationRequestFuncInput(
            model=request_model_id,
            session_id=None,
            prompt=prompt,
            input_image_paths=request.input_image_paths,
            api_url=api_url,
            image_options=request.image_options,
        )
    raise ValueError(f"Unsupported benchmark task: {benchmark_task}")


async def get_request(
    input_requests: Sequence[SampledRequest],
    request_rate: float,
    timing_data: dict[str, list[float]],
    burstiness: float = 1.0,
    benchmark_should_end_time: int | None = None,
) -> AsyncGenerator[SampledRequest, None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a SampledRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness:
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
        timing_data:
            Dictionary where timing data will be collected with keys:
            - 'intervals': List of actual time intervals between requests
    """

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)

    # Initialize timing data collection - always enabled
    timing_data.setdefault("intervals", [])

    start_time = time.perf_counter()
    last_request_time = start_time

    for request in input_requests:
        current_time = time.perf_counter()

        # Record timestamp when request is yielded
        if last_request_time != start_time:
            actual_interval = current_time - last_request_time
            timing_data["intervals"].append(actual_interval)

        yield request

        # Update last_request_time for next iteration
        last_request_time = current_time

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        if exceeds_deadline(interval, benchmark_should_end_time):
            return
        await asyncio.sleep(interval)


async def run_single_turn_benchmark(
    *,
    input_requests: Sequence[SampledRequest],
    benchmark_task: BenchmarkTask,
    request_rate: float,
    burstiness: float,
    timing_data: dict[str, list[float]] | None,
    semaphore: contextlib.AbstractAsyncContextManager[None],
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    max_output_len: int | None,
    sampling: SamplingConfig,
    lora_manager: LoRABenchmarkManager | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> list[BaseRequestFuncOutput]:
    """Run single-turn benchmark scenario."""
    if timing_data is None:
        timing_data = {}

    async def limited_request_func(
        request_func_input: BaseRequestFuncInput,
    ) -> BaseRequestFuncOutput:
        async with semaphore:
            if (
                benchmark_should_end_time is not None
                and time.perf_counter_ns() >= benchmark_should_end_time
            ):
                return request_func_input.get_output_type()(
                    cancelled=True, request_submit_time=time.perf_counter()
                )
            remaining_s = deadline_remaining_s(benchmark_should_end_time)
            try:
                output = await asyncio.wait_for(
                    request_driver.request(request_func_input),
                    timeout=remaining_s,
                )
            except asyncio.TimeoutError:
                return request_func_input.get_output_type()(
                    cancelled=True, request_submit_time=time.perf_counter()
                )
            return mark_cancelled_if_past_deadline(
                output, benchmark_should_end_time
            )

    tasks: list[asyncio.Task[BaseRequestFuncOutput]] = []
    request_idx = 0
    async for request in get_request(
        input_requests,
        request_rate,
        timing_data,
        burstiness,
        benchmark_should_end_time=benchmark_should_end_time,
    ):
        # If we've hit the time limit, then don't issue any more requests
        if benchmark_should_end_time is not None:
            if time.perf_counter_ns() >= benchmark_should_end_time:
                break

        # Determine which LoRA to use for this request
        lora_id = None
        if lora_manager:
            lora_id = lora_manager.get_lora_for_request(request_idx)

        request_func_input = build_single_turn_request_input(
            benchmark_task=benchmark_task,
            request=request,
            model_id=model_id,
            lora_id=lora_id,
            api_url=api_url,
            sampling=sampling,
            max_output_len=max_output_len,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
        )
        tasks.append(
            asyncio.create_task(limited_request_func(request_func_input))
        )
        request_idx += 1

    outputs = await asyncio.gather(*tasks)

    if deadline_passed(benchmark_should_end_time):
        cancelled = sum(1 for o in outputs if o.cancelled)
        logger.info(
            "Benchmark stopped by the duration limit"
            " (--max-benchmark-duration-s):"
            f" dispatched {len(outputs)} of {len(input_requests)} requests,"
            f" {cancelled} cancelled in flight."
        )

    return outputs


async def prime_shared_contexts(
    model_id: str,
    api_url: str,
    samples: Samples,
    request_driver: RequestDriver,
    sampling: SamplingConfig,
    max_concurrency: int,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
    disable_tqdm: bool = False,
) -> None:
    """Warm up prefix caching by sending each shared context for prefilling."""
    warmup_entries = samples.shared_contexts

    if not warmup_entries:
        logger.warning(
            "shared_contexts is empty; the prefix cache could not be primed."
            " Check that --random-sys-prompt-ratio > 0 (and"
            " --fit-distributions for instruct-coder/agentic-code) and that"
            " input lengths are sufficient to produce a non-trivial shared"
            " context."
        )
        return

    logger.info(
        f"Warming prefix cache with {len(warmup_entries)}"
        " unique shared context(s)..."
    )

    is_chat = isinstance(samples, ChatSamples)
    warmup_inputs: list[RequestFuncInput] = []
    for entry in warmup_entries:
        warmup_prompt: str | list[ChatMessage]
        if is_chat:
            warmup_prompt = [
                ChatMessage(
                    role="user",
                    content=[TextContentBlock(text=entry.text)],
                )
            ]
        else:
            warmup_prompt = entry.text

        if run_prefix:
            warmup_prompt = _prepend_run_prefix_to_formatted_prompt(
                warmup_prompt, run_prefix
            )

        warmup_inputs.append(
            RequestFuncInput(
                model=model_id,
                session_id=None,
                sampling=sampling,
                prompt=warmup_prompt,
                images=[],
                api_url=api_url,
                prompt_len=entry.num_tokens + run_prefix_len,
                max_tokens=1,
                ignore_eos=True,
            )
        )

    warmup_results: list[BaseRequestFuncOutput | None] = [None] * len(
        warmup_inputs
    )

    semaphore = asyncio.Semaphore(max_concurrency)

    warmup_start = time.perf_counter()
    with progressbar_request_driver(
        request_driver,
        len(warmup_inputs),
        disable_tqdm=disable_tqdm,
        desc="prefix-cache warmup",
    ) as driver:

        async def _run_warmup_index(idx: int, inp: RequestFuncInput) -> None:
            async with semaphore:
                warmup_results[idx] = await driver.request(inp)

        async with TaskGroup() as tg:
            for idx, inp in enumerate(warmup_inputs):
                tg.create_task(_run_warmup_index(idx, inp))
    warmup_elapsed_s = time.perf_counter() - warmup_start
    for sys_idx, inp in enumerate(warmup_inputs):
        result = warmup_results[sys_idx]
        if result is None:
            raise RuntimeError(
                f"Warmup task {sys_idx} did not produce a result (this is a bug)"
            )
        if not result.success:
            raise ValueError(
                f"Shared context warmup request failed at index {sys_idx}:"
                f" (prompt: (SKIPPED), prompt_len: {inp.prompt_len}),"
                f" error: {result.error}"
            )

    logger.info(
        "Prefix cache warmup completed and took %.2f seconds.",
        warmup_elapsed_s,
    )


async def run_single_test_prompt(
    benchmark_task: BenchmarkTask,
    model_id: str,
    api_url: str,
    samples: Samples,
    request_driver: RequestDriver,
    sampling: SamplingConfig,
    max_output_len: int | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> BaseRequestFuncOutput:
    if isinstance(samples, ChatSamples):
        test_question = samples.chat_sessions[0].messages[0]
        test_answer = samples.chat_sessions[0].messages[1]
        test_request = SampledRequest(
            prompt_formatted=[
                ChatMessage(
                    role="user",
                    content=[TextContentBlock(text=test_question.content)],
                )
            ],
            prompt_len=test_question.num_tokens,
            output_len=test_answer.num_tokens,
            encoded_images=[],
            ignore_eos=True,
        )
        # Chat samples define their own target output length per turn.
        test_max_output_len = None
    else:
        test_request = samples.requests[0]
        test_max_output_len = max_output_len

    test_input = build_single_turn_request_input(
        benchmark_task=benchmark_task,
        request=test_request,
        model_id=model_id,
        lora_id=None,
        api_url=api_url,
        sampling=sampling,
        max_output_len=test_max_output_len,
        run_prefix=run_prefix,
        run_prefix_len=run_prefix_len,
    )
    return await request_driver.request(test_input)
