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
"""MAX pipeline for audio generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
from max.driver import load_devices
from max.pipelines.context import AudioGenerationContextType, GenerationStatus
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.modeling.types import (
    AudioGenerationInputs,
    Pipeline,
    PipelineOutputsDict,
    RequestID,
)
from max.pipelines.request.open_responses import (
    AudioGenerationDetails,
    OutputAudioContent,
    Usage,
)

from .interface import AudioExecutor

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

_logger = logging.getLogger("max.pipelines")


class AudioGenerationPipeline(
    Pipeline[
        AudioGenerationInputs[AudioGenerationContextType], GenerationOutput
    ],
    Generic[AudioGenerationContextType],
):
    """Audio generation pipeline, driven by an :class:`AudioExecutor`.

    Unlike text generation, one request is one call: the executor runs every
    stage the model has -- autoregressive, denoising, vocoding -- and returns
    a finished waveform. The pipeline's own job is only to hand the batch to
    the executor and turn its samples into a response.

    Args:
        pipeline_config: Configuration for the pipeline and runtime behavior.
        pipeline_model: The audio executor class to instantiate.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[AudioExecutor[Any, Any, Any]],
    ) -> None:
        from max.engine import InferenceSession  # local import to avoid cycles

        self._pipeline_config = pipeline_config
        # Use the first component's device_specs for session initialization.
        first_config = next(iter(pipeline_config.models.values()))
        self._devices = load_devices(first_config.device_specs)

        session = InferenceSession(devices=[*self._devices])
        self._pipeline_config.configure_session(session)

        self._executor = pipeline_model(
            manifest=pipeline_config.models,
            session=session,
            runtime_config=pipeline_config.runtime,
        )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Returns the pipeline configuration."""
        return self._pipeline_config

    @property
    def max_batch_size(self) -> int:
        """Returns 1: audio generation runs one request at a time."""
        return 1

    def execute(
        self,
        inputs: AudioGenerationInputs[AudioGenerationContextType],
    ) -> PipelineOutputsDict[GenerationOutput]:
        """Generates one waveform per request in the batch."""
        model_inputs, flat_batch = self.prepare_batch(inputs.batch)
        if not flat_batch or model_inputs is None:
            return {}

        try:
            outputs = self._executor.execute(model_inputs)
        except Exception:
            _logger.error(
                "Encountered an exception while executing audio batch: "
                "batch_size=%d",
                len(flat_batch),
            )
            raise

        waveforms = np.from_dlpack(outputs.waveform)
        if waveforms.ndim != 3:
            raise ValueError(
                "Expected a (batch, channels, samples) waveform from the "
                f"executor, got shape {waveforms.shape}."
            )
        if waveforms.shape[0] != len(flat_batch):
            raise ValueError(
                "Unexpected number of waveforms returned from executor: "
                f"expected {len(flat_batch)}, got {waveforms.shape[0]}."
            )

        sample_rate = self._executor.sample_rate
        responses: dict[RequestID, GenerationOutput] = {}
        for index, (request_id, context) in enumerate(flat_batch):
            waveform = np.ascontiguousarray(waveforms[index], dtype=np.float32)
            responses[request_id] = GenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                output=[
                    OutputAudioContent.from_numpy_samples(
                        waveform,
                        sample_rate=sample_rate,
                        format=context.audio_format,
                    )
                ],
                # The token counters stay at zero, as image generation's do:
                # audio has no output tokens, and reporting a sample count as
                # one would make it look priceable per token. What was
                # actually produced goes in the audio details instead.
                usage=Usage(
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    audio_generation_details=AudioGenerationDetails.from_waveform(
                        waveform,
                        sample_rate=sample_rate,
                        steps=context.num_inference_steps,
                    ),
                ),
            )

        return responses

    def prepare_batch(
        self,
        batch: dict[RequestID, AudioGenerationContextType],
    ) -> tuple[Any, list[tuple[RequestID, AudioGenerationContextType]]]:
        """Delegates input preparation to the executor.

        Args:
            batch: Maps each request ID to its :class:`AudioContext`.

        Returns:
            The executor's inputs, and the batch flattened to
            ``(request_id, context)`` pairs in the order the executor sees
            them, which is the order its waveforms come back in.

        Raises:
            ValueError: If the batch holds more than one request.
        """
        if not batch:
            return None, []

        flat_batch = list(batch.items())
        if len(flat_batch) > 1:
            raise ValueError(
                "Batching of different requests is not supported yet."
            )

        contexts = [context for _, context in flat_batch]
        return self._executor.prepare_inputs(contexts), flat_batch

    def release(self, request_id: RequestID) -> None:
        """Releases resources held for a request.

        Nothing to do: an audio request holds no state between calls, since
        each one is generated in a single :meth:`execute`.
        """
