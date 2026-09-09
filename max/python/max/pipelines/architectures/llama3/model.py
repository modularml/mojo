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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan

from .batch_processor import Llama3BatchProcessor
from .data_parallel_llama import create_graph as create_data_parallel_graph
from .distributed_llama import DistributedLlama3
from .llama3 import Llama3
from .model_config import Llama3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class Llama3Inputs(ModelInputs):
    """A class representing inputs for the Llama3 model.

    This class encapsulates the input tensors required for the Llama3 model
    execution.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer

    data_parallel_splits: Buffer | Sequence[Sequence[int]] | None = None
    """Tensor containing the data parallel splits."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        if self.data_parallel_splits is not None:
            if isinstance(self.data_parallel_splits, Buffer):
                splits_tensor = self.data_parallel_splits
            else:
                splits_array = np.concatenate(
                    [
                        np.array(split, dtype=np.int64)
                        for split in self.data_parallel_splits
                    ]
                )
                splits_tensor = Buffer.from_numpy(splits_array).to(
                    self.tokens.device
                )
            return (
                self.tokens,
                self.input_row_offsets,
                self.return_n_logits,
                splits_tensor,
                *(
                    self.kv_cache_inputs.flatten()
                    if self.kv_cache_inputs is not None
                    else ()
                ),
            )

        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
        )


class LlamaModelBase(
    LogProbabilitiesMixin,
    GraphPipelineModelWithKVCache[TextContext],
):
    """Base Llama pipeline model implementation."""

    model_config_cls: ClassVar[type[Any]] = Llama3Config
    batch_processor_cls: ClassVar[type[Llama3BatchProcessor]] = (
        Llama3BatchProcessor
    )

    model: Model
    """Compiled and initialized model ready for inference."""

    norm_method: Literal["rms_norm", "layer_norm"]
    """Normalization layer."""

    attention_bias: bool = False
    """Whether to use attention bias."""

    state_dict: dict[str, Any]
    """Weights to load into the model."""

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
        """
        Args:
            pipeline_config: The configuration for this pipeline.
            session: The container for the runtime for this model.
        """
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

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs:
        """Delegates to the batch processor and narrows to ``Llama3Inputs``."""
        inputs = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
        assert isinstance(inputs, Llama3Inputs)
        return inputs

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Llama3Inputs)
        assert model_inputs.kv_cache_inputs is not None
        model_outputs = self.model.execute(*model_inputs.buffers)

        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = Llama3Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        return model_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, Llama3Config)
        if model_config.data_parallel_degree > 1:
            return create_data_parallel_graph(
                model_config, self.kv_params, state_dict
            )
        if len(self.devices) > 1:
            return self._build_tensor_parallel_graph_for_compile(
                state_dict, model_config
            )
        return self._build_single_device_graph_for_compile(
            state_dict, model_config
        )

    def _build_tensor_parallel_graph_for_compile(
        self,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        assert isinstance(model_config, Llama3Config)
        dist_model: DistributedLlama3 = DistributedLlama3(model_config)

        dist_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,  # TODO(MODELS-550) `rope_freqs.weight` not used
        )

        weights_registry = dist_model.state_dict()

        with Graph(
            getattr(self.huggingface_config, "model_type", "llama3"),
            input_types=dist_model.input_types(self.kv_params),
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )

            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]

            kv_caches_per_dev = self._unflatten_kv_inputs(
                variadic_args[len(self.devices) :]
            )

            outputs = dist_model(
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
    ) -> tuple[Graph, dict[str, Any]]:
        assert isinstance(model_config, Llama3Config)
        single_model: Llama3 = Llama3(model_config)

        single_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,  # TODO(MODELS-550) `rope_freqs.weight` not used
        )
        weights_registry = single_model.state_dict()

        with Graph(
            "llama3",
            input_types=single_model.input_types(self.kv_params),
        ) as graph:
            (
                tokens,
                input_row_offsets,
                return_n_logits,
                *rest,
            ) = graph.inputs
            kv_collections = self._unflatten_kv_inputs(rest)
            outputs = single_model(
                tokens.tensor,
                kv_collections[0],
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )
            graph.output(*outputs)
            return graph, weights_registry


class Llama3Model(LlamaModelBase):
    """Llama 3 pipeline model implementation."""

    config_class: type[Llama3Config] = Llama3Config
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    """Normalization layer."""

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
