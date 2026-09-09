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

"""Utilities for generating text in the cli."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections.abc import Iterable
from typing import Any

import requests
from max.pipelines import (
    PIPELINE_REGISTRY,
    GenerateMixin,
    TextAndVisionTokenizer,
    TextTokenizer,
)
from max.pipelines.context import (
    LogitsProcessor,
    ProcessorInputs,
    SamplingParams,
    TextContext,
)
from max.pipelines.lib import PipelineArgs, PipelineConfig
from max.pipelines.logging_utils import log_basic_config
from max.pipelines.modeling.types import (
    ImageContentPart,
    Pipeline,
    PipelineTokenizer,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.profiler import OneShotCapture, Tracer

from .metrics import TextGenerationMetrics

logger = logging.getLogger("max._entrypoints")

MODEL_NAME = "model"


class TrackMetrics:
    def __init__(self, metrics: TextGenerationMetrics):
        self.metrics = metrics
        self.first_token = True

    def __call__(self, inputs: ProcessorInputs) -> None:
        if self.first_token:
            self.first_token = False
            self.metrics.signpost("first_token")
            self.metrics.prompt_size = len(inputs.context.tokens)
        self.metrics.new_token()


async def stream_text_to_console(
    pipeline: Pipeline[Any, Any],
    tokenizer: PipelineTokenizer[TextContext, Any, TextGenerationRequest],
    prompt: str,
    images: list[bytes],
    sampling_params: SamplingParams,
    metrics: TextGenerationMetrics | None = None,
    print_tokens: bool = True,
) -> None:
    assert isinstance(tokenizer, TextTokenizer | TextAndVisionTokenizer)
    logits_processors: list[LogitsProcessor] = []
    if metrics:
        logits_processors.append(TrackMetrics(metrics))

    sampling_params = dataclasses.replace(
        sampling_params, logits_processors=logits_processors
    )

    # Base/completion models (e.g. GPT-2) have no chat template. For these, send the raw
    # prompt so the tokenizer encodes it directly instead of trying to apply a
    # chat template. Chat models (with a template) and multimodal requests
    # still go through the messages path.
    # Custom tokenizers (--custom-architectures) may have no HF delegate at
    # all; treat them like template-less tokenizers.
    has_chat_template = bool(
        getattr(getattr(tokenizer, "delegate", None), "chat_template", None)
    )
    if images or has_chat_template:
        request = TextGenerationRequest(
            request_id=RequestID(),
            messages=[
                TextGenerationRequestMessage(
                    role="user",
                    content=[
                        TextContentPart(text=prompt),
                        *(ImageContentPart() for _ in images),
                    ],
                )
            ],
            images=images,
            model_name=MODEL_NAME,
            sampling_params=sampling_params,
        )
    else:
        request = TextGenerationRequest(
            request_id=RequestID(),
            prompt=prompt,
            model_name=MODEL_NAME,
            sampling_params=sampling_params,
        )

    if metrics:
        metrics.signpost("begin_generation")
    try:
        assert isinstance(pipeline, GenerateMixin)
        async for outputs in pipeline.generate_async(request):
            if print_tokens:
                decoded = await tokenizer.decode(
                    outputs[0].tokens, skip_special_tokens=True
                )
                print(decoded, end="", flush=True)
    finally:
        if metrics:
            metrics.signpost("end_generation")


def generate_text_for_pipeline(
    pipeline_args: PipelineArgs,
    sampling_params: SamplingParams,
    prompt: str,
    image_urls: Iterable[str] = (),
    num_warmups: int = 0,
    profile: bool = False,
    profile_top_n: int = 15,
) -> None:
    # The capture handle is created outside `with TextGenerationMetrics` so
    # ``end_and_finalize`` can fire *after* the metrics report prints. Under
    # nsys, ``cudaProfilerStop`` triggers the ``.nsys-rep`` write — delaying
    # it past the metrics report keeps the normal generate output (text +
    # stats) from being buried inside nsys's file-writing progress lines.
    pipeline_config = PipelineConfig.from_args(pipeline_args)
    retrieved = PIPELINE_REGISTRY.retrieve_factory(pipeline_config)
    tokenizer = retrieved.tokenizer
    pipeline = retrieved.factory()
    assert isinstance(pipeline, Pipeline)
    log_basic_config(pipeline_config, memory_plan=retrieved.memory_plan)
    logger.info("max_batch_size: %d", pipeline.max_batch_size)

    capture = OneShotCapture(top_n=profile_top_n) if profile else None
    try:
        # Run timed run & print results.
        with TextGenerationMetrics(print_report=True) as metrics:
            if image_urls:
                logger.info("Downloading images")
                images = [requests.get(url).content for url in image_urls]
            else:
                images = []

            if num_warmups > 0:
                logger.info("Running warmup")
                warmup_params = dataclasses.replace(
                    sampling_params, max_new_tokens=num_warmups
                )
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        sampling_params=warmup_params,
                        metrics=None,
                        print_tokens=False,
                    )
                )

            # Run and print results.
            logger.info("Beginning text generation")

            with contextlib.ExitStack() as exit_stack:
                if capture is not None:
                    capture.start()
                    # ``Tracer`` adds an NVTX "inference" label visible in
                    # nsys-ui. It must be entered after ``capture.start``
                    # so its NVTX range sits inside the cuda profiler
                    # window.
                    exit_stack.enter_context(
                        Tracer("inference", color="modular_purple")
                    )
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        sampling_params=sampling_params,
                        metrics=metrics,
                        print_tokens=True,
                    )
                )
        # ``TextGenerationMetrics.__exit__`` has printed the report here.
        # End the capture *now*, so nsys's file-writing output follows.
    finally:
        if capture is not None:
            capture.end_and_finalize()
