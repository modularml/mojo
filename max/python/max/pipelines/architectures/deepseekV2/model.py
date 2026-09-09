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
"""Implements the DeepseekV2 nn.model."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.driver import Buffer, Device, DeviceSpec
from max.dtype import DType
from max.engine.api import InferenceSession, Model
from max.graph import BufferType, DeviceRef, Graph, TensorType, Value
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.comm import Signals
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import (
    BatchProcessor,
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from typing_extensions import override

from .batch_processor import DeepseekV2BatchProcessor
from .deepseekV2 import DeepseekV2
from .distributed_deepseekV2 import DistributedDeepseekV2
from .model_config import DeepseekV2Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV2Inputs(ModelInputs):
    """A class representing inputs for the DeepseekV2 model.

    This class encapsulates the input tensors required for the DeepseekV2 model execution:
    - tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    - return_n_logits: A tensor containing the number of logits to return
    """

    tokens: Buffer
    input_row_offsets: Buffer
    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer = field(kw_only=True)


class DeepseekV2Model(
    LogProbabilitiesMixin,
    GraphPipelineModelWithKVCache[TextContext],
):
    model_config_cls: ClassVar[type[Any]] = DeepseekV2Config
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]]] = (
        DeepseekV2BatchProcessor
    )

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
        return_logits: ReturnLogits = ReturnLogits.ALL,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        if pipeline_config.model.device_specs[0] == DeviceSpec.cpu():
            raise ValueError("DeepseekV2 currently only supported on gpu.")

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

        self.model = self.load_model(session)

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV2Inputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs.flatten(),
        )
        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    def graph_inputs(self) -> tuple[TensorType | BufferType, ...]:
        # Generate DeviceRef
        device_ref = DeviceRef.from_device(self.devices[0])

        # Construct general input types
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=device_ref
        )

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        if len(self.devices) > 1:
            signals = Signals(
                devices=(DeviceRef(d.label, d.id) for d in self.devices)
            )
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *signals.input_types(),
                *self.kv_params.flattened_kv_inputs(),
            )
        else:
            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                *self.kv_params.flattened_kv_inputs(),
            )

    def _unflatten_kv_inputs(
        self,
        kv_inputs_flat: Sequence[Value[Any]],
        kv_params: KVCacheParamInterface | None = None,
    ) -> list[PagedCacheValues]:
        kv_params = kv_params or self.get_kv_params(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            devices=[DeviceRef.from_device(d) for d in self.devices],
            kv_cache_config=self.kv_cache_config,
            cache_dtype=cache_dtype_for_encoding(
                _select_quantization_encoding(
                    self.pipeline_config.model,
                    DeepseekV2Config.DEFAULT_ENCODING,
                ),
                self.pipeline_config.model.kv_cache.kv_cache_format,
            ),
        )
        symbolic_inputs = kv_params.unflatten_kv_inputs(iter(kv_inputs_flat))
        assert isinstance(symbolic_inputs, KVCacheInputs)
        return list(symbolic_inputs.inputs)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights supported in DeepseekV2."
            )
        return super()._load_state_dict()

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        del state_dict
        model_config = DeepseekV2Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.max_batch_context_length = (
            self.planned_max_batch_total_tokens
            or model_config.max_batch_context_length
        )
        return model_config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, DeepseekV2Config)
        graph_inputs = self.graph_inputs()
        if len(self.devices) > 1:
            return self._build_tensor_parallel_graph_for_compile(
                state_dict, model_config, graph_inputs
            )
        return self._build_single_device_graph_for_compile(
            state_dict, model_config, graph_inputs
        )

    def _build_tensor_parallel_graph_for_compile(
        self,
        state_dict: dict[str, Any],
        model_config: Any,
        graph_inputs: tuple[TensorType | BufferType, ...],
    ) -> tuple[Graph, dict[str, Any]]:
        assert isinstance(model_config, DeepseekV2Config)
        nn_model = DistributedDeepseekV2(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=False)
        weights_registry = nn_model.state_dict()

        with Graph("deepseekV2", input_types=[*graph_inputs]) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )

            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]

            kv_caches_per_dev = self._unflatten_kv_inputs(
                variadic_args[len(self.devices) :]
            )

            outputs = nn_model(
                tokens.tensor,
                signal_buffers,
                kv_caches_per_dev,
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )

            graph.output(*outputs)
            return graph, weights_registry

    def _build_single_device_graph_for_compile(
        self,
        state_dict: dict[str, Any],
        model_config: Any,
        graph_inputs: tuple[TensorType | BufferType, ...],
    ) -> tuple[Graph, dict[str, Any]]:
        assert isinstance(model_config, DeepseekV2Config)
        nn_model = DeepseekV2(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1)
        weights_registry = nn_model.state_dict()

        with Graph("deepseekV2", input_types=[*graph_inputs]) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)
            outputs = nn_model(
                tokens.tensor,
                kv_collections[0],
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph, weights_registry
