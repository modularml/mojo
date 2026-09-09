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

import contextlib
import logging
from collections.abc import AsyncGenerator
from typing import Any, cast

_logger = logging.getLogger("max.pipelines")

from max.pipelines.audio.pipeline import AudioGenerationPipeline
from max.pipelines.context import (
    AudioContext,
    BaseContextType,
    PixelContext,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.diffusion.pipeline import (
    PixelGenerationPipeline,
)
from max.pipelines.lib import (
    EmbeddingsPipelineType,
    MemoryPlan,
    PipelineConfig,
    TextGenerationPipeline,
)
from max.pipelines.modeling.types import (
    AudioGenerationInputs,
    EmbeddingsContext,
    EmbeddingsGenerationOutput,
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
    PixelGenerationInputs,
    RequestID,
)
from max.serve.config import Settings
from max.serve.queue import MAXPullQueue, MAXPushQueue
from max.serve.scheduler.interface import Scheduler
from max.serve.scheduler_result import SchedulerResult
from max.serve.worker_interface import WorkerQueues

from .base import CancelRequest, PrefillRequest, PrefillResponse
from .config import TokenGenerationSchedulerConfig
from .decode_scheduler import load_decode_scheduler
from .embeddings_scheduler import EmbeddingsScheduler, EmbeddingsSchedulerConfig
from .one_shot_scheduler import OneShotScheduler
from .prefill_scheduler import load_prefill_scheduler
from .text_generation_scheduler import load_text_generation_scheduler

__all__ = [
    "CancelRequest",
    "EmbeddingsScheduler",
    "EmbeddingsSchedulerConfig",
    "OneShotScheduler",
    "PrefillRequest",
    "PrefillResponse",
    "TokenGenerationSchedulerConfig",
    "load_scheduler",
]


def load_scheduler(
    pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
    pipeline_config: PipelineConfig,
    settings: Settings,
    worker_queues: WorkerQueues[BaseContextType, PipelineOutputType],
    memory_plan: MemoryPlan | None,
) -> Scheduler:
    request_queue = worker_queues.request_queue
    response_queue = worker_queues.response_queue
    cancel_queue = worker_queues.cancel_queue

    _logger.info("max_batch_size: %d", pipeline.max_batch_size)

    if pipeline.__class__.__name__ == "PixelGenerationPipeline":
        pixel_pipeline = cast(PixelGenerationPipeline[Any], pipeline)

        def batch_constructor(
            context: PixelContext,
        ) -> PixelGenerationInputs[Any]:
            """Convert a single PixelContext into PixelGenerationInputs."""
            return PixelGenerationInputs(batch={context.request_id: context})

        return OneShotScheduler[
            PixelContext, PixelGenerationInputs[Any], GenerationOutput
        ](
            pipeline=pixel_pipeline,
            batch_constructor=batch_constructor,
            request_queue=cast(
                MAXPullQueue[PixelContext],
                request_queue,
            ),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[GenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
        )
    elif pipeline.__class__.__name__ == "AudioGenerationPipeline":
        audio_pipeline = cast(AudioGenerationPipeline[Any], pipeline)

        def audio_batch_constructor(
            context: AudioContext,
        ) -> AudioGenerationInputs[Any]:
            return AudioGenerationInputs(batch={context.request_id: context})

        return OneShotScheduler[
            AudioContext, AudioGenerationInputs[Any], GenerationOutput
        ](
            pipeline=audio_pipeline,
            batch_constructor=audio_batch_constructor,
            request_queue=cast(
                MAXPullQueue[AudioContext],
                request_queue,
            ),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[GenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
        )
    elif pipeline.__class__.__name__ == "EmbeddingsPipeline":
        embeddings_scheduler_config = EmbeddingsSchedulerConfig(
            max_batch_size=pipeline.max_batch_size
        )
        emb_pipeline = cast(EmbeddingsPipelineType, pipeline)
        return EmbeddingsScheduler(
            scheduler_config=embeddings_scheduler_config,
            pipeline=emb_pipeline,
            request_queue=cast(
                MAXPullQueue[EmbeddingsContext],
                request_queue,
            ),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[EmbeddingsGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
        )
    elif pipeline_config.runtime.pipeline_role == "prefill_and_decode":
        text_pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
        return load_text_generation_scheduler(
            text_pipeline,
            pipeline_config,
            request_queue=cast(MAXPullQueue[TextContext], request_queue),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[TextGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
            memory_plan=memory_plan,
            max_pending_requests=settings.max_pending_requests,
        )
    elif pipeline_config.runtime.pipeline_role == "decode_only":
        text_pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
        return load_decode_scheduler(
            text_pipeline,
            pipeline_config,
            request_queue=cast(MAXPullQueue[TextContext], request_queue),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[TextGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
            settings=settings,
            memory_plan=memory_plan,
        )
    elif pipeline_config.runtime.pipeline_role == "prefill_only":
        text_pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
        return load_prefill_scheduler(
            text_pipeline, pipeline_config, settings, memory_plan
        )
    else:
        raise ValueError(
            f"No scheduler support for pipeline_role ({pipeline_config.runtime.pipeline_role})."
        )
