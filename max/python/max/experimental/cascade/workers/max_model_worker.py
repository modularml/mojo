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
"""Cascade worker that delegates decode work to a MAX Serve model worker.

This worker does **not** load the model in its own process. Instead it spawns a
``max.serve`` model-worker subprocess (the same one used by
``max.serve.api_server`` and ``max.entrypoints.LLM``) and communicates with it
over ZMQ (via :class:`ZmqModelWorkerInterface`). As in ``max.serve``, a single
fan-out task on the proxy side pulls ``SchedulerResult`` batches off the response
socket and routes individual outputs to per-request ``asyncio.Queue`` instances.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import AsyncIterator

import numpy as np
import numpy.typing as npt
from max.experimental.cascade.core import Worker, worker_method
from max.experimental.cascade.interfaces.gen_ai import TextGenOptions
from max.pipelines.architectures import register_all_models
from max.pipelines.context import (
    EOSTracker,
    SamplingParams,
    SamplingParamsInput,
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.lib import PIPELINE_REGISTRY, PipelineConfig
from max.pipelines.modeling.types import RequestID

logger = logging.getLogger(__name__)

Int32Array = npt.NDArray[np.int32]


def _sampling_params_input(req: TextGenOptions) -> SamplingParamsInput:
    """Map cascade :class:`TextGenOptions` onto ``SamplingParamsInput``.

    Forwards every request-configurable sampling field so a request routed
    through the cascade pipeline resolves the same parameters as one sent to
    ``max.serve``'s OpenAI routes. ``None`` fields fall back to the model's
    ``GenerationConfig`` defaults (then the ``SamplingParams`` class defaults).
    """
    return SamplingParamsInput(
        max_new_tokens=req.num_tokens,
        min_new_tokens=req.min_new_tokens,
        ignore_eos=req.ignore_eos,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        min_p=req.min_p,
        thinking_temperature=req.thinking_temperature,
        seed=req.seed,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        repetition_penalty=req.repetition_penalty,
        stop=req.stop,
        stop_token_ids=req.stop_token_ids,
    )


class MAXModelWorker(Worker):
    """Cascade worker backed by a MAX Serve model-worker subprocess.

    The worker is constructed with a :py:class:`PipelineConfig` (typically built
    once by the entrypoint/CLI and shared across the cascade pipeline's
    workers). It does **not** load the model in its own process; instead it
    spawns a ``max.serve`` model-worker subprocess (the same one used by
    ``max.serve.api_server`` and ``max.entrypoints.LLM``) and communicates with
    it over ZMQ. Per-request streaming is handled by the proxy's single
    response-fan-out task.
    """

    def __init__(self, pipeline_config: PipelineConfig) -> None:
        device_specs = pipeline_config.model.device_specs
        on_cpu = all(spec.device_type == "cpu" for spec in device_specs)
        super().__init__(deploy_hints=["cpu"] if on_cpu else ["gpu"])
        self.pipeline_config = pipeline_config

        retrieved = PIPELINE_REGISTRY.retrieve_factory(pipeline_config)
        # ``max_length`` is populated from the memory plan in open().
        self.max_length: int | None = None
        self._memory_plan = retrieved.memory_plan
        self._eos_token_ids: set[int] = set(retrieved.tokenizer.eos_token_ids)
        self._model_factory = retrieved.factory

        # lazy import to avoid circular imports when defining
        # CascadePipelines in model arch.py layers
        from max.serve.worker_interface import ModelWorkerProxy

        self._proxy: (
            ModelWorkerProxy[
                TextAndVisionContext | TextContext, TextGenerationOutput
            ]
            | None
        ) = None

    @contextlib.asynccontextmanager
    async def open(self) -> AsyncIterator[MAXModelWorker]:
        """Bring up the model-worker subprocess and proxy.

        Mirrors ``max.serve.api_server``: the config was already resolved and
        the factory built in the orchestrator (passed to the constructor), so
        here we only spawn the ``max.serve`` model-worker subprocess with that
        factory and resolved config -- the worker never resolves or mutates the
        config. ``retrieve_pipeline_task``/``retrieve_context_type`` are plain
        registry lookups (no config mutation, no network). The telemetry
        consumer, model-worker subprocess and proxy (with its single response
        fan-out task) stay alive for the worker's lifetime -- the same async
        context managers used by ``max.entrypoints.LLM`` and
        ``max.serve.api_server``.
        """
        # lazy import to avoid circular imports when defining
        # CascadePipelines in model arch.py layers
        from max.serve.config import MetricRecordingMethod, Settings
        from max.serve.pipelines.model_worker import start_model_worker
        from max.serve.pipelines.telemetry_worker import (
            start_telemetry_consumer,
        )
        from max.serve.worker_interface._zmq_queue import generate_zmq_ipc_path
        from max.serve.worker_interface.zmq_interface import (
            ZmqModelWorkerInterface,
        )

        assert self._model_factory is not None, (
            "MAXModelWorker needs a model_factory to deploy; construct it via "
            "build_pipeline, which resolves the config and builds the factory."
        )
        max_length = self._memory_plan.planned_max_length
        assert max_length is not None, (
            "memory plan must carry a planned_max_length"
        )
        self.max_length = max_length
        t0 = time.monotonic()
        register_all_models()

        pipeline_task = PIPELINE_REGISTRY.retrieve_pipeline_task(
            self.pipeline_config.models.main_architecture_name
        )
        context_type = PIPELINE_REGISTRY.retrieve_context_type(
            self.pipeline_config
        )

        settings = Settings(
            offline_inference=True,
            disable_telemetry=True,
            metric_recording=MetricRecordingMethod.NOOP,
        )

        model_worker_interface = ZmqModelWorkerInterface[
            TextAndVisionContext | TextContext, TextGenerationOutput
        ](
            pipeline_task,
            context_type=context_type,
        )

        async with contextlib.AsyncExitStack() as exit_stack:
            metric_client = await exit_stack.enter_async_context(
                start_telemetry_consumer(settings)
            )
            self._proxy = await exit_stack.enter_async_context(
                start_model_worker(
                    model_factory=self._model_factory,
                    pipeline_config=self.pipeline_config,
                    settings=settings,
                    metric_client=metric_client,
                    model_worker_interface=model_worker_interface,
                    zmq_endpoint_base=generate_zmq_ipc_path(),
                    memory_plan=self._memory_plan,
                )
            )
            logger.info("MAXModelWorker ready in %.1fs", time.monotonic() - t0)
            try:
                yield self
            finally:
                self._proxy = None

    @worker_method()
    async def decode(
        self, req: TextGenOptions, tokens: Int32Array
    ) -> AsyncIterator[Int32Array]:
        """Submit a decode request and stream generated token ids.

        The proxy's response-fan-out task delivers ``TextGenerationOutput``
        batches to a per-request ``asyncio.Queue``; we collect any tokens
        produced in a single fan-out delivery and forward them as one
        ``np.int32`` array to the cascade stream.
        """
        assert self._proxy is not None
        assert self.max_length is not None

        request_id = RequestID()

        prompt_tokens = tokens.astype(np.int64)
        # Layer the request's fields over the model's GenerationConfig defaults,
        # exactly as ``max.serve``'s OpenAI routes do, so the two paths resolve
        # sampling parameters identically.
        sampling_params = SamplingParams.from_input_and_generation_config(
            _sampling_params_input(req),
            sampling_params_defaults=self.pipeline_config.model.sampling_params_defaults,
        )
        request_max_length = min(
            self.max_length, int(prompt_tokens.shape[0]) + req.num_tokens
        )
        # Mirror max-serve's tokenizer behaviour: when ignore_eos is set,
        # strip the EOS triggers entirely. The scheduler terminates a request
        # as soon as it samples any token in ``eos_token_ids``, regardless of
        # ``sampling_params.ignore_eos`` -- so leaving the default EOS set
        # populated would silently cap generation early.
        ctx_eos_token_ids: set[int] = (
            set() if req.ignore_eos else set(self._eos_token_ids)
        )
        ctx = TextContext(
            request_id=request_id,
            max_length=request_max_length,
            tokens=TokenBuffer(prompt_tokens),
            eos_tracker=EOSTracker(eos_token_ids=ctx_eos_token_ids),
            sampling_params=sampling_params,
        )

        response_stream = await self._proxy.stream(request_id, ctx)
        async for outputs, _batch_id in response_stream:
            token_arrays = [
                np.asarray(output.tokens, dtype=np.int32)
                for output in outputs
                if output.tokens
            ]
            if token_arrays:
                yield np.concatenate(token_arrays)

    @worker_method()
    async def echo(self, tokens: Int32Array) -> AsyncIterator[np.int32]:
        """Echo the tokens back to the caller, for debugging."""
        for token in tokens:
            yield np.int32(token)
