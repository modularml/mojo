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
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.comm import Signals
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan

from .batch_processor import GptOssBatchProcessor
from .gpt_oss import GptOss
from .model_config import GptOssConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class GptOssInputs(ModelInputs):
    """A class representing inputs for the GPT OSS model.

    This class encapsulates the input tensors required for the GPT OSS model
    execution.
    """

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: npt.NDArray[np.integer[Any]] | Buffer | list[Buffer]
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence. For distributed execution,
    this can be a list of tensors, one per device."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer
    """Number of logits to return."""


class GptOssModel(
    AlwaysSignalBuffersMixin, GraphPipelineModelWithKVCache[TextContext]
):
    """A GPT OSS pipeline model for text generation.

    This class integrates the GPT OSS architecture with the MAX Engine pipeline
    infrastructure, handling model loading, KV cache management, and input preparation
    for inference.
    """

    model_config_cls: ClassVar[type[Any]] = GptOssConfig
    batch_processor_cls: ClassVar[type[GptOssBatchProcessor]] = (
        GptOssBatchProcessor
    )

    model: Model
    """The compiled and initialized MAX Engine model ready for inference."""

    # For text-only models, we should be using all the weights.
    _strict_state_dict_loading = True

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
        """
        Args:
            pipeline_config: The configuration settings for the entire pipeline.
            session: The MAX Engine inference session managing the runtime.
            devices: A list of MAX Engine devices (:obj:`max.driver.Device`) to
                run the model on.
            kv_cache_config: Configuration settings for the Key-Value cache
                (:obj:`max.pipelines.max_config.KVCacheConfig`).
            weights: The model weights (:obj:`max.graph.weights.Weights`).
            adapter: An optional adapter to modify weights before loading
                (:obj:`max.graph.weights.WeightsAdapter`).
            return_logits: The number of top logits to return from the model
                execution.
        """
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

    def _create_model_config(self, state_dict: dict[str, Any]) -> GptOssConfig:
        model_config = GptOssConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        return model_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: GptOssConfig,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        # NOTE: input_row_offsets_len should be batch_size + 1.
        # Create input_row_offsets_type for each device
        input_row_offsets_types = [
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef(device.label, device.id),
            )
            for device in self.devices
        ]
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        nn_model = GptOss(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        weights_registry = nn_model.state_dict(auto_initialize=False)

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = kv_inputs.flatten()

        with Graph(
            getattr(self.huggingface_config, "model_type", "GptOss"),
            input_types=[
                tokens_type,
                return_n_logits_type,
                *input_row_offsets_types,
                *signals.input_types(),
                *flattened_kv_types,
            ],
        ) as graph:
            # Unpack inputs following InternVL pattern
            tokens, return_n_logits, *variadic_args = graph.inputs

            # Extract input_row_offsets (one per device)
            input_row_offsets = [
                v.tensor for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract signal buffers (one per device)
            signal_buffers = [
                v.buffer for v in variadic_args[: len(self.devices)]
            ]
            variadic_args = variadic_args[len(self.devices) :]

            # Extract KV cache inputs from the unified {sliding, global} tree.
            kv_cache_local, kv_cache_global = (
                self.kv_params.unflatten_basic_kv_tree(iter(variadic_args))
            )

            outputs = nn_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                sliding_kv_collections=kv_cache_local,
                global_kv_collections=kv_cache_global,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
            )
            graph.output(*outputs)
        return graph, weights_registry

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the GPT OSS model with the prepared inputs.

        Args:
            model_inputs: The prepared inputs for the model execution, typically including
                token IDs, attention masks/offsets, and KV cache inputs.

        Returns:
            An object containing the output logits from the model execution.
        """
        model_inputs = cast(GptOssInputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None

        # Check if input_row_offsets is a list or a single tensor
        if isinstance(model_inputs.input_row_offsets, list):
            input_row_offsets_list = model_inputs.input_row_offsets
        else:
            # For backward compatibility, distribute the single tensor to all devices
            if isinstance(model_inputs.input_row_offsets, np.ndarray):
                # Convert numpy array to tensor first
                tensor = Buffer.from_numpy(model_inputs.input_row_offsets)
                input_row_offsets_list = [
                    tensor.to(device) for device in self.devices
                ]
            else:
                # Already a tensor
                input_row_offsets_list = [
                    model_inputs.input_row_offsets.to(device)
                    for device in self.devices
                ]

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            *input_row_offsets_list,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs.flatten(),
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1]),
                next_token_logits=cast(Buffer, model_outputs[0]),
                logit_offsets=cast(Buffer, model_outputs[2]),
            )
        else:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[0]),
                next_token_logits=cast(Buffer, model_outputs[0]),
            )
