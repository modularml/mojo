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

import logging
from dataclasses import dataclass
from typing import Any

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import (
    GraphPipelineModel,
    KVCacheConfig,
    ModelInputs,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan

from .graph import build_graph

logger = logging.getLogger("max.pipelines")


@dataclass
class WhisperInputs(ModelInputs):
    """A class representing inputs for the Whisper model.

    input_features:
        Float values mel features extracted from the raw speech waveform.
        Raw speech waveform can be obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* viathe soundfile library (`pip install soundfile`).
        To prepare the array into `input_features`, the [`AutoFeatureExtractor`] from the transformers library should be used for extracting the mel features, padding and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        Shape = (batch_size, feature_size, sequence_length)

    decoder_input_ids:
        Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using [`WhisperTokenizer`].
        Whisper uses the `decoder_start_token_id` as the starting token for `decoder_input_ids` generation.
        Shape = (batch_size, target_sequence_length)
    """

    input_features: Buffer
    decoder_input_ids: Buffer


# TODO: Need specific Context type, not just this base type.
class Whisper(GraphPipelineModel[Any]):
    model: Model

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.model = self.load_model(session)

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session, model_config
        graph = build_graph(
            state_dict,
            self.huggingface_config,
            self.dtype,
            DeviceRef.from_device(self.devices[0]),
        )
        return graph, state_dict
