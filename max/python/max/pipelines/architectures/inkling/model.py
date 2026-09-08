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
"""Inkling pipeline model: input preparation, cache plumbing, execution."""

from __future__ import annotations

from typing import Any, ClassVar

from max.driver import Device, is_virtual_device_mode
from max.engine import InferenceSession, Model
from max.graph import Graph, Module
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
    SupportsSSMStateWarmup,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.modeling.types import RequestID
from typing_extensions import override

from .batch_processor import InklingBatchProcessor, InklingInputs
from .inkling import Inkling
from .layers.vision import InklingVisionModel
from .model_config import InklingConfig
from .state_cache import InklingConvStateCache
from .weight_adapters import VISION_PREFIX


class InklingModel(
    LogProbabilitiesMixin,
    MultiGraphPipelineModelWithKVCache[TextAndVisionContext],
    SupportsSSMStateWarmup,
):
    """Pipeline model for Inkling's decoder and its vision tower."""

    batch_processor_cls: ClassVar[type[InklingBatchProcessor]] = (
        InklingBatchProcessor
    )
    model_config_cls: ClassVar[type[Any]] = InklingConfig

    model: Model
    _nn_model: Inkling

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
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
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
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self._state_cache: InklingConvStateCache | None = None
        # The vision tower is only ever called through the batch processor,
        # which _wire_batch_processor hands it to.
        _, self.model = self.load_model(session)

    @override
    def _wire_batch_processor(
        self, model: Any = None, model_config: Any = None
    ) -> None:
        super()._wire_batch_processor(model, model_config)
        # Compile-only runs cannot allocate on a virtual device.
        if not is_virtual_device_mode():
            # The memory-plan-resolved batch size, not the runtime config's
            # (often None).
            max_batch_size = self.max_batch_size
            assert max_batch_size is not None
            self._state_cache = InklingConvStateCache(
                self._nn_model.conv_layout,
                max_slots=max_batch_size,
                devices=self.devices,
            )
        assert isinstance(model, Model)
        assert isinstance(self._batch_processor, InklingBatchProcessor)
        self._batch_processor.bind_runtime_state(self._state_cache, model)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        """Splits the checkpoint between the two graphs the towers compile to."""
        state_dict = super()._load_state_dict()
        self._vision_weights_dict = {
            name.removeprefix(VISION_PREFIX): data
            for name, data in state_dict.items()
            if name.startswith(VISION_PREFIX)
        }
        self._language_weights_dict = {
            name: data
            for name, data in state_dict.items()
            if not name.startswith(VISION_PREFIX)
        }
        return state_dict

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> InklingConfig:
        # Quantization is read off the loaded weights; the checkpoint config
        # may not declare it.
        model_config = InklingConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(self.huggingface_config, state_dict)
        return model_config

    @override
    def _build_vision_graph(
        self,
        model_config: InklingConfig,
        state_dict: dict[str, Any],
        module: Module,
    ) -> tuple[Graph, dict[str, Any]]:
        # The tower is small and runs once per image, so it stays on one device.
        vision_model = InklingVisionModel(
            model_config.vision_config,
            model_config.dtype,
            model_config.devices[0],
        )
        vision_model.load_state_dict(state_dict, weight_alignment=1)
        with Graph(
            "inkling_vision",
            input_types=vision_model.input_types(),
            module=module,
        ) as graph:
            graph.output(vision_model(graph.inputs[0].tensor))
        return graph, vision_model.state_dict()

    @override
    def _build_language_graph(
        self,
        model_config: InklingConfig,
        state_dict: dict[str, Any],
        module: Module,
    ) -> tuple[Graph, dict[str, Any]]:
        nn_model = Inkling(model_config, return_logits=self.return_logits)
        nn_model.load_state_dict(state_dict, weight_alignment=1)
        self._nn_model = nn_model

        with Graph(
            "inkling", input_types=nn_model.input_types(), module=module
        ) as graph:
            outputs = nn_model(*nn_model.unpack_inputs(graph.inputs))
            graph.output(*outputs)
        return graph, nn_model.state_dict()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, InklingInputs)
        model_outputs = list(self.model.execute(*model_inputs.buffers))

        assert self._batch_processor is not None
        return self._batch_processor.process_outputs(model_outputs)

    def release(self, request_id: RequestID) -> None:
        """Drops the request's convolution state, freeing its slot."""
        if self._state_cache is not None:
            self._state_cache.release(request_id)

    def release_warmup_state(self, request_ids: list[RequestID]) -> None:
        """Frees the slots a graph-capture warmup probe claimed.

        Without this the second probe finds no free slot and serving never
        starts; ``claim`` zeros a slot when a real request takes it.
        """
        for request_id in request_ids:
            self.release(request_id)
