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

"""Utilities for encoding text in the cli."""

from __future__ import annotations

import asyncio
import logging
from typing import cast

from max.pipelines import (
    PIPELINE_REGISTRY,
    EmbeddingsPipelineType,
)
from max.pipelines.context import TextContext
from max.pipelines.lib import PipelineArgs, PipelineConfig
from max.pipelines.modeling.types import (
    EmbeddingsGenerationInputs,
    EmbeddingsGenerationOutput,
    PipelineTask,
    PipelineTokenizer,
    RequestID,
    TextGenerationRequest,
)

from .metrics import EmbeddingsMetrics

logger = logging.getLogger("max._entrypoints")

MODEL_NAME = "model"


async def _run_pipeline_encode(
    pipeline: EmbeddingsPipelineType,
    tokenizer: PipelineTokenizer[TextContext, int, TextGenerationRequest],
    prompt: str,
    metrics: EmbeddingsMetrics | None = None,
) -> EmbeddingsGenerationOutput:
    context = await tokenizer.new_context(
        TextGenerationRequest(
            request_id=RequestID(), prompt=prompt, model_name=MODEL_NAME
        )
    )
    pipeline_request = EmbeddingsGenerationInputs(
        [{context.request_id: context}]
    )

    if metrics:
        metrics.prompt_size = len(context.tokens)
        metrics.signpost("begin_encoding")

    response = pipeline.execute(pipeline_request)

    if metrics:
        metrics.signpost("end_encoding")
    return response[context.request_id]


def pipeline_encode(
    pipeline_args: PipelineArgs,
    prompt: str,
    num_warmups: int = 0,
) -> None:
    # Run timed run & print results.
    with EmbeddingsMetrics(print_report=True) as metrics:
        tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(
            PipelineConfig.from_args(pipeline_args),
            task=PipelineTask.EMBEDDINGS_GENERATION,
        )

        # Cast pipeline to the expected type for embeddings generation
        embeddings_pipeline = cast(EmbeddingsPipelineType, pipeline)

        if num_warmups > 0:
            logger.info("Running warmup")
            for _ in range(num_warmups):
                asyncio.run(
                    _run_pipeline_encode(
                        embeddings_pipeline, tokenizer, prompt, metrics=None
                    )
                )

        # Run and print results.
        logger.info("Running model...")
        print("Encoding:", prompt)

        pipeline_output = asyncio.run(
            _run_pipeline_encode(
                embeddings_pipeline, tokenizer, prompt, metrics=metrics
            )
        )
        print("Embeddings:", pipeline_output.embeddings)
