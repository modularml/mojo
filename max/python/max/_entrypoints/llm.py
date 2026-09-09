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

"""Provides a high-level interface for interacting with LLMs built from MAX
pipelines.
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import TypeVar, cast

import tqdm
from max.pipelines.context import (
    SamplingParams,
    SamplingParamsInput,
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineArgs, PipelineConfig
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationRequest,
)
from max.serve.config import Settings
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.worker_interface._zmq_queue import generate_zmq_ipc_path
from max.serve.worker_interface.lora_queue import LoRAQueue
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerInterface

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class _Request:
    id: RequestID
    prompts: Sequence[str]
    max_new_tokens: int | None
    use_tqdm: bool


@dataclass
class _Response:
    complete_texts: Sequence[str]


@dataclass
class ThreadControl:
    ready: Event = field(default_factory=Event)
    cancel: Event = field(default_factory=Event)


# For now, the LLM class only supports the direct token generation use case.
# Long term, there are multiple other potential use cases to support.
# This class loosely mirrors vllm.LLM for offline inference: https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html
class LLM:
    """A high-level interface for interacting with LLMs.

    .. deprecated:: 26.5
        This is a legacy API that's now private. We'll introduce a new API for
        offline inference in a future release.

    Use this class for offline batch inference against a model loaded from a
    :class:`PipelineArgs`. Call :meth:`generate` with one or more prompts
    to receive completions.

    The following example loads ``LiquidAI/LFM2.5-350M`` and generates
    completions for a batch of prompts:

    .. code-block:: python

        from max._entrypoints.llm import LLM
        from max.pipelines.lib import PipelineArgs

        pipeline_args = PipelineArgs.from_flat_kwargs(model_path="LiquidAI/LFM2.5-350M")
        llm = LLM(pipeline_args)

        prompts = [
            "In the beginning, there was",
            "The meaning of life is",
        ]
        responses = llm.generate(prompts, max_new_tokens=50)
        for prompt, response in zip(prompts, responses, strict=True):
            print(prompt + response)

    Args:
        pipeline_args: The :class:`PipelineArgs` describing the model and
            runtime to load.
    """

    _pc: ThreadControl
    _async_runner: Thread
    _request_queue: queue.Queue[_Request]
    _pending_requests: dict[RequestID, queue.Queue[_Response]]

    def __init__(self, pipeline_args: PipelineArgs) -> None:
        settings = Settings(offline_inference=True)
        self._pc = ThreadControl()
        self._request_queue = queue.Queue()
        self._pending_requests = {}
        self._async_runner = Thread(
            target=_run_async_worker,
            args=(
                self._pc,
                pipeline_args,
                self._request_queue,
                self._pending_requests,
                settings,
            ),
        )
        self._async_runner.start()
        # TODO: set a timeout on wait
        self._pc.ready.wait()

    def __del__(self) -> None:
        # FIXME: refactor API to proper context manager
        self._pc.cancel.set()
        self._async_runner.join()

    def generate(
        self,
        prompts: str | Sequence[str],
        max_new_tokens: int | None = 100,
        use_tqdm: bool = True,
    ) -> Sequence[str]:
        """Generates text completions for the given prompts.

        This method is thread-safe and may be used on the same :class:`LLM`
        instance from multiple threads concurrently with no external
        synchronization.

        Args:
            prompts: The input string or list of strings to generate
                completions for.
            max_new_tokens: The maximum number of tokens to generate in the
                response. Defaults to ``100``.
            use_tqdm: Whether to display a progress bar during generation.
                Defaults to ``True``.

        Returns:
            A list of generated text completions corresponding to each input
            prompt.

        Raises:
            ValueError: If ``prompts`` is empty or contains invalid data.
            RuntimeError: If the model fails to generate completions.
        """
        if isinstance(prompts, str):
            # Handle the edge case where the user passes in a single string
            prompts = (prompts,)

        request = _Request(
            id=RequestID(),
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            use_tqdm=use_tqdm,
        )
        response_queue: queue.Queue[_Response] = queue.Queue()
        self._pending_requests[request.id] = response_queue

        try:
            self._request_queue.put_nowait(request)
            return response_queue.get().complete_texts
        finally:
            # Clean up the pending request mapping
            self._pending_requests.pop(request.id, None)


def _run_async_worker(
    pc: ThreadControl,
    pipeline_args: PipelineArgs,
    request_queue: queue.Queue[_Request],
    pending_requests: Mapping[RequestID, queue.Queue[_Response]],
    settings: Settings,
) -> None:
    asyncio.run(
        _async_worker(
            pc,
            pipeline_args,
            request_queue,
            pending_requests,
            settings,
        )
    )


async def _async_map(
    f: Callable[[T], Awaitable[U]],
    seq: Sequence[T],
    /,
    *,
    use_tqdm: bool = False,
) -> list[U]:
    outputs: list[U | None] = [None] * len(seq)

    async def task_wrapper(i: int) -> None:
        outputs[i] = await f(seq[i])
        if use_tqdm:
            pbar.update(1)

    if use_tqdm:
        with tqdm.tqdm(total=len(seq)) as pbar:
            await asyncio.gather(*map(task_wrapper, range(len(seq))))
    else:
        await asyncio.gather(*map(task_wrapper, range(len(seq))))
    return cast("list[U]", outputs)


async def _async_worker(
    pc: ThreadControl,
    pipeline_args: PipelineArgs,
    request_queue: queue.Queue[_Request],
    pending_requests: Mapping[RequestID, queue.Queue[_Response]],
    settings: Settings,
) -> None:
    pipeline_config = PipelineConfig.from_args(pipeline_args)
    retrieved = PIPELINE_REGISTRY.retrieve_factory(pipeline_config)
    tokenizer = retrieved.tokenizer
    model_factory = retrieved.factory
    model_name = pipeline_config.model.model_path

    # Start the model worker process.
    # Create dynamic and continuous batching workers and associated queues
    # to feed the model worker process.
    pipeline_task = PIPELINE_REGISTRY.retrieve_pipeline_task(
        pipeline_config.models.main_architecture_name,
    )
    zmq_endpoint_base = generate_zmq_ipc_path()
    lora_queue: LoRAQueue | None = (
        LoRAQueue(
            zmq_endpoint_base,
            pipeline_config.lora.lora_paths,
        )
        if pipeline_config.lora
        else None
    )
    # Create Queues
    model_worker_interface = ZmqModelWorkerInterface[
        TextAndVisionContext | TextContext, TextGenerationOutput
    ](
        pipeline_task,
        context_type=PIPELINE_REGISTRY.retrieve_context_type(pipeline_config),
    )
    async with (
        start_telemetry_consumer(settings) as metric_client,
        start_model_worker(
            model_factory=model_factory,
            pipeline_config=pipeline_config,
            settings=settings,
            metric_client=metric_client,
            model_worker_interface=model_worker_interface,
            zmq_endpoint_base=zmq_endpoint_base,
            memory_plan=retrieved.memory_plan,
        ) as model_worker,
    ):
        pipeline = TokenGeneratorPipeline(
            model_name=model_name,
            tokenizer=tokenizer,
            lora_queue=lora_queue,
            model_worker=model_worker,
        )

        pc.ready.set()
        while True:
            if pc.cancel.is_set():
                break
            try:
                request = request_queue.get(timeout=0.3)
            except queue.Empty:
                continue

            # Lambda to do a full text generation for a request.
            async def all_tokens(prompt: str) -> str:
                sampling_params = SamplingParams.from_input_and_generation_config(
                    SamplingParamsInput(
                        max_new_tokens=request.max_new_tokens  # noqa: B023
                    ),
                    sampling_params_defaults=pipeline_config.model.sampling_params_defaults,
                )
                gen_request = TextGenerationRequest(
                    request_id=RequestID(),
                    model_name=model_name,
                    prompt=prompt,
                    sampling_params=sampling_params,
                )

                # Generate this request until complete
                chunks = await pipeline.all_tokens(gen_request)
                # TODO: (MODELS-1120) determine whether to include reasoning tokens
                return "".join(
                    chunk.decoded_tokens
                    if chunk.decoded_tokens is not None
                    else ""
                    for chunk in chunks
                )

            responses = await _async_map(
                all_tokens, request.prompts, use_tqdm=request.use_tqdm
            )

            # Put the response in the specific queue for this request ID
            if response_queue := pending_requests.get(request.id):
                response_queue.put(_Response(complete_texts=responses))
