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
"""Implements the DeepseekV3 NextN pipeline model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, cast

from max.driver import Buffer, Device, DLPackArray, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, ops
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputs
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from typing_extensions import override

from ..deepseekV2.model import DeepseekV2Model
from ..deepseekV3.model import DeepseekV3Inputs, DeepseekV3Model
from .batch_processor import DeepseekV3NextNBatchProcessor
from .deepseekV3_nextn import DeepseekV3NextN
from .model_config import DeepseekV3NextNConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV3NextNInputs(DeepseekV3Inputs):
    """A class representing inputs for the DeepseekV3 NextN model.

    Inherits from DeepseekV3Inputs so that the target model's isinstance check
    passes during EAGLE verification (when draft_inputs is passed to the target).
    """

    hidden_states: Buffer | None = field(default=None, kw_only=True)


class DeepseekV3NextNModel(AlwaysSignalBuffersMixin, DeepseekV2Model):
    model_config_cls: ClassVar[type[Any]] = DeepseekV3NextNConfig
    batch_processor_cls: ClassVar[type[DeepseekV3NextNBatchProcessor]] = (
        DeepseekV3NextNBatchProcessor
    )

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
        shared_weights: dict[str, DLPackArray] | None = None,
        shared_ep_comm_initializer: EPCommInitializer | None = None,
        max_batch_size: int = 1,
    ) -> None:
        self._shared_weights = shared_weights
        self._shared_ep_comm_initializer = shared_ep_comm_initializer
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

    def _apply_shared_weights(self, state_dict: dict[str, WeightData]) -> None:
        if not self._shared_weights:
            return
        for key, value in self._shared_weights.items():
            # Replace existing weights or add missing ones.  The NextN
            # safetensors may omit weights shared with the target model
            # (e.g. embed_tokens, lm_head) so we must be able to add them.
            state_dict[key] = cast(WeightData, value)

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> DeepseekV3NextNConfig:
        """Create model configuration from huggingface config.

        The weight adapter converts keys to NextN format (decoder_layer.*),
        but the base model's _create_model_config expects base format (layers.0.*).
        We temporarily remap the key so the base implementation can extract dtype info.
        """
        # Temporarily add the base format key for norm dtype extraction
        # NextN format: "decoder_layer.self_attn.kv_a_layernorm.weight"
        # Base format:  "layers.0.self_attn.kv_a_layernorm.weight"
        nextn_key = "decoder_layer.self_attn.kv_a_layernorm.weight"
        base_key = "layers.0.self_attn.kv_a_layernorm.weight"

        if nextn_key not in state_dict:
            raise KeyError(
                f"Expected NextN norm key '{nextn_key}' not found in state_dict. "
                f"This may indicate the weights are not in the correct NextN format. "
                f"Available keys sample: {list(state_dict.keys())[:10]}..."
            )

        state_dict[base_key] = state_dict[nextn_key]

        # Call DeepseekV3Model's _create_model_config to compute
        # state-dict-dependent fields (ep_config, quant_config, etc.)
        base_config = DeepseekV3Model._create_model_config(self, state_dict)  # type: ignore[arg-type]

        # Remove temporary key
        if base_key in state_dict and nextn_key in state_dict:
            del state_dict[base_key]

        # The HF config may declare NVFP4 quantization (e.g. quantization_config
        # inherited from the target model), but the NextN weights may actually
        # be all BF16. Detect this by checking for weight_scale_2 keys which
        # are only present when weights are truly FP4-quantized.
        if (
            base_config.quant_config is not None
            and base_config.quant_config.is_nvfp4
            and not any("weight_scale_2" in key for key in state_dict)
        ):
            logger.info(
                "NextN weights are BF16 (no weight_scale_2 found); "
                "disabling NVFP4 config."
            )
            base_config.quant_config = None
            base_config.dtype = DType.bfloat16
            if base_config.ep_config is not None:
                base_config.ep_config.dispatch_dtype = DType.bfloat16
                base_config.ep_config.dispatch_quant_config = None

        # Build NextN config from the base config's fields, avoiding
        # asdict() which recursively converts nested dataclasses to dicts.
        model_config = DeepseekV3NextNConfig(
            **{
                f.name: getattr(base_config, f.name)
                for f in fields(base_config)
            }
        )
        return model_config

    @override
    def _load_state_dict(self) -> dict[str, WeightData]:
        state_dict = super()._load_state_dict()
        self._apply_shared_weights(state_dict)
        return state_dict

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: DeepseekV3NextNConfig,
    ) -> None:
        self.ep_comm_initializer = None
        # Skip EP initialization in virtual device mode (compilation-only)
        # since NVSHMEM functions cannot be linked without real GPU devices.
        # We still keep ep_config to generate the correct graph structure.
        if model_config.ep_config is None or is_virtual_device_mode():
            return
        if self._shared_ep_comm_initializer is not None:
            # Reuse target model's EP buffers (NVSHMEM symmetric memory).
            # Target and draft execute sequentially in the EAGLE pipeline,
            # so sharing is safe and avoids duplicating ~85 GiB of buffers.
            self.ep_comm_initializer = self._shared_ep_comm_initializer
            # Propagate node_id from the shared initializer since NextN
            # constructs its own EPConfig with node_id=-1.
            model_config.ep_config.node_id = (
                self._shared_ep_comm_initializer.config.node_id
            )
            logger.info("Reusing target model's EP communication buffers.")
        else:
            self.ep_comm_initializer = EPCommInitializer(model_config.ep_config)
            self.ep_comm_initializer.ep_init(session)
        if model_config.ep_config.node_id == -1:
            raise ValueError(
                "EP node ID is not set. Please check if the EP initialization is successful."
            )

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, WeightData],
        model_config: DeepseekV3NextNConfig,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        logger.info("Building DeepseekV3 NextN model...")
        nn_model = DeepseekV3NextN(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        weights_registry = nn_model.state_dict()

        num_devices = len(self.devices)

        with Graph(
            "deepseekV3_nextn_graph",
            input_types=nn_model.input_types(self.kv_params),
        ) as graph:
            graph_inputs_iter = iter(graph.inputs)

            tokens = next(graph_inputs_iter)
            hidden_states = next(graph_inputs_iter)
            device_input_row_offsets = next(graph_inputs_iter)
            host_input_row_offsets = next(graph_inputs_iter)
            return_n_logits = next(graph_inputs_iter)
            data_parallel_splits = next(graph_inputs_iter)

            signal_buffers = [
                next(graph_inputs_iter).buffer for _ in range(num_devices)
            ]

            kv_inputs = self.kv_params.unflatten_kv_inputs(graph_inputs_iter)
            assert isinstance(kv_inputs, KVCacheInputs)
            kv_caches_per_dev = list(kv_inputs.inputs)

            batch_context_lengths = [
                next(graph_inputs_iter).tensor for _ in range(num_devices)
            ]

            ep_model_inputs = list(graph_inputs_iter)

            hidden_states_per_dev = list(
                ops.distributed_broadcast(hidden_states.tensor, signal_buffers)
            )
            input_row_offsets_per_dev = list(
                ops.distributed_broadcast(
                    device_input_row_offsets.tensor, signal_buffers
                )
            )
            outputs = nn_model(
                tokens.tensor,
                hidden_states_per_dev,
                signal_buffers,
                kv_caches_per_dev,
                return_n_logits.tensor,
                input_row_offsets_per_dev,
                host_input_row_offsets.tensor,
                data_parallel_splits.tensor,
                batch_context_lengths,
                ep_model_inputs,
            )

            graph.output(*outputs)
            return graph, weights_registry

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV3NextNInputs)

        if model_inputs.hidden_states is None:
            raise ValueError(
                "hidden_states must be set before executing DeepSeekV3 NextN model. "
                "EAGLE pipeline should set this field after calling prepare_initial_token_inputs()."
            )

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None
        ep_inputs = (
            ()
            if self.ep_comm_initializer is None
            else self.ep_comm_initializer.model_inputs()
        )

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.hidden_states,
            model_inputs.input_row_offsets,
            model_inputs.host_input_row_offsets,
            model_inputs.return_n_logits,
            model_inputs.data_parallel_splits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs.flatten(),
            *model_inputs.batch_context_lengths,
            *ep_inputs,
        )

        if len(model_outputs) == 2:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
                hidden_states=model_outputs[1],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )


def maybe_build_deepseekv3_nextn_kwargs(
    target_model: PipelineModel[TextContext],
    draft_model_cls: type[PipelineModel[TextContext]],
) -> dict[str, Any]:
    """Builds kwargs needed to pass to DeepseekV3NextNModel constructor."""
    if not issubclass(draft_model_cls, DeepseekV3NextNModel):
        return {}
    if not isinstance(target_model, DeepseekV3Model):
        raise ValueError(
            "DeepseekV3NextNModel can only be used with DeepseekV3 target models"
        )

    required_prefixes = ("embed_tokens.", "lm_head.")
    shared_weights: dict[str, DLPackArray] = {}
    for name, value in target_model.state_dict.items():
        for prefix in required_prefixes:
            if name.startswith(prefix):
                shared_weights[name] = value

    if len(shared_weights) != len(required_prefixes):
        raise ValueError(
            f"Missing weight prefixes {required_prefixes} in target DeepseekV3 "
            f"state_dict. Cannot share weights with NextN draft model."
        )

    logger.info(
        "Sharing DeepseekV3 embedding and head weights with NextN draft model."
    )

    return {
        "shared_weights": shared_weights,
        # Share EP buffers between target and draft to avoid duplicating
        "shared_ep_comm_initializer": target_model.ep_comm_initializer,
    }
