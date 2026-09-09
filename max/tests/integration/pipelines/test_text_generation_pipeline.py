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
"""WIP Test Suite for Unit Testing the TextGenerationPipeline."""

import asyncio
import logging
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
from max.driver import DeviceSpec
from max.pipelines.context import (
    ImageMetadata,
    SamplingParams,
    TextContext,
)
from max.pipelines.lib import generate_local_model_path
from max.pipelines.lib.pipeline_variants import text_generation
from max.pipelines.lib.vision_encoder_cache import (
    VideoEncoderMetrics,
    VisionEncoderMetrics,
)
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationInputs,
    TextGenerationRequest,
)
from max.support.image import hash_image
from pytest import MonkeyPatch
from test_common.mocks import (
    MockTextTokenizer,
    retrieve_mock_text_generation_pipeline,
)

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
logger = logging.getLogger("max.pipelines")


def test_mock_text_tokenizer() -> None:
    tokenizer = MockTextTokenizer()
    test_prompt = "This is a test prompt"

    try:
        model_path = generate_local_model_path(REPO_ID)
    except FileNotFoundError:
        logger.warning(
            f"Model path does not exist: {REPO_ID}, falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        model_path = REPO_ID

    request = TextGenerationRequest(
        request_id=RequestID("request_0"),
        model_name=model_path,
        prompt=test_prompt,
    )

    new_context = asyncio.run(tokenizer.new_context(request))

    assert len(new_context.tokens) == len(test_prompt)

    decoded = asyncio.run(tokenizer.decode(new_context.tokens.active))
    assert test_prompt == decoded


def test_text_generation_image_metadata() -> None:
    image_metadata = ImageMetadata(
        start_idx=0,
        end_idx=2,
        pixel_values=np.array([99]),
    )
    assert image_metadata.image_hash is None

    # Prefix caching enabled by default
    image_metadata = ImageMetadata(
        start_idx=0,
        end_idx=2,
        pixel_values=np.array([99]),
        image_hash=hash_image(np.array([99])),
    )
    assert image_metadata.image_hash is not None


def test_batch_vision_metrics_falls_back_to_pooled_pipeline_model() -> None:
    """A pipeline model that owns its cache internally (no ``_encoder_cache``,
    e.g. MiniMax-M3) still surfaces vision/video metrics via
    ``SupportsPooledVisionMetrics`` (CLIN-1638)."""
    vision_sentinel = VisionEncoderMetrics(num_images_total=1)
    video_sentinel = VideoEncoderMetrics(num_clips_total=1)

    class _FakePooledMetricsModel:
        def pop_vision_metrics(self) -> VisionEncoderMetrics | None:
            return vision_sentinel

        def pop_video_metrics(self) -> VideoEncoderMetrics | None:
            return video_sentinel

    pipeline = cast(Any, object.__new__(text_generation.TextGenerationPipeline))
    pipeline._encoder_cache = None
    pipeline._pipeline_model = _FakePooledMetricsModel()

    assert pipeline.batch_vision_metrics() is vision_sentinel
    assert pipeline.batch_video_metrics() is video_sentinel


def test_batch_vision_metrics_none_without_protocol_or_cache() -> None:
    """Text-only models (no cache, no pooled-metrics protocol) get None."""
    pipeline = cast(Any, object.__new__(text_generation.TextGenerationPipeline))
    pipeline._encoder_cache = None
    pipeline._pipeline_model = object()

    assert pipeline.batch_vision_metrics() is None
    assert pipeline.batch_video_metrics() is None


def test_text_generation_pipeline(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        text_generation, "load_weights", MagicMock(return_value=None)
    )
    monkeypatch.setattr(
        text_generation, "weights_format", MagicMock(return_value=None)
    )
    monkeypatch.setattr(text_generation, "load_kv_manager", MagicMock())

    max_length = 512
    eos_token = 998

    try:
        model_path = generate_local_model_path(REPO_ID)
    except FileNotFoundError:
        logger.warning(
            f"Model path does not exist: {REPO_ID}, falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        model_path = REPO_ID

    with (
        retrieve_mock_text_generation_pipeline(
            vocab_size=1000,
            eos_token=eos_token,
            eos_prob=0.05,  # On average, one in every 20 tokens will be an eos token.
            max_length=max_length,
            device_specs=[DeviceSpec(device_type="cpu", id=0)],
        ) as (tokenizer, pipeline)
    ):
        prompts = [
            # These next two prompts should definitely generate at least 1 and 4 tokens.
            # Using them to ensure we return the correct number of new tokens.
            "The definition of hypothetical is ",
            "The definition of hypothetical is ",
            "This is a test prompt",
            "This is a slightly longer test prompt " * 2,
            "This is a very very long test prompt " * 4,
        ]
        _max_new_tokens = [1, 4, 25, 100, None]
        context_batch = {}
        max_new_tokens = {}
        for i, prompt in enumerate(prompts):
            id = RequestID(f"request_{i}")
            max_new_tokens[id] = _max_new_tokens[i]
            sampling_params = SamplingParams(max_new_tokens=max_new_tokens[id])
            request = TextGenerationRequest(
                request_id=id,
                model_name=model_path,
                prompt=prompt,
                sampling_params=sampling_params,
            )

            context_batch[id] = asyncio.run(tokenizer.new_context(request))

        length = {context.request_id: 0 for context in context_batch.values()}
        while True:
            # This will generate a list[dict[request_id, TextGenerationOutput]] for each step
            inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
                batches=[list(context_batch.values())]
            )
            output = pipeline.execute(inputs)
            assert len(output) == len(context_batch)

            for response in output.values():
                length[response.request_id] = len(response.tokens)
                # Check that we are not overrunning, the request max new tokens
                if _max := max_new_tokens[response.request_id]:
                    assert length[response.request_id] <= _max

                assert length[response.request_id] < max_length

                if response.is_done:
                    del context_batch[response.request_id]

            # Break
            if not context_batch:
                break

        # These two prompts should generate the full max new tokens.
        for response in output.values():
            if max_new_tokens[response.request_id] is not None:
                assert (
                    length[response.request_id]
                    == max_new_tokens[response.request_id]
                )
