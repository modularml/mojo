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
"""Implements the DeepseekV3 nn.model (ModuleV3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from max.driver import Buffer, Device, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.sharding import DeviceMesh
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.comm.ep import (
    EPBatchManager,
    EPCommInitializer,
    EPConfig,
    calculate_ep_max_tokens_per_rank,
)
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV2_modulev3.model import DeepseekV2Inputs, DeepseekV2Model
from .batch_processor import DeepseekV3ModuleV3BatchProcessor
from .deepseekV3 import DeepseekV3
from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV3Inputs(DeepseekV2Inputs):
    batch_context_lengths: list[Buffer] = field(kw_only=True)
    """Host (CPU) page-aligned KV context length, one per DP replica.

    Substituted for the planner's device-resident ``buffer_lengths`` so the
    per-layer ``.to(CPU())`` stays host-to-host and the graph is capturable.
    """

    data_parallel_splits: Buffer | None = field(default=None, kw_only=True)
    input_row_offsets_i64: Buffer | None = field(default=None, kw_only=True)
    ep_inputs: tuple[Buffer, ...] = field(default=(), kw_only=True)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Flat graph inputs in compile ABI order."""
        dp_inputs: tuple[Buffer, ...] = ()
        if self.data_parallel_splits is not None:
            assert self.input_row_offsets_i64 is not None
            dp_inputs = (self.data_parallel_splits, self.input_row_offsets_i64)
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *self.batch_context_lengths,
            *dp_inputs,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
            *self.ep_inputs,
        )


class DeepseekV3Model(DeepseekV2Model):
    """A DeepseekV3 model (ModuleV3), single- or multi-GPU (TP attention, EP)."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV3Config
    batch_processor_cls: ClassVar[type[DeepseekV3ModuleV3BatchProcessor]] = (
        DeepseekV3ModuleV3BatchProcessor
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
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        max_batch_size: int = 1,
    ) -> None:
        # Capture the session so _init_distributed_runtime() can initialize EP
        # communication, and default the EP buffers so execute() works without EP.
        self.session = session
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

    def _build_ep_config(
        self, ep_size: int, n_devices: int, model_config: DeepseekV3Config
    ) -> EPConfig:
        """Build the expert-parallel config from the pipeline ep_size."""
        if ep_size % n_devices != 0:
            raise ValueError(
                f"ep_size={ep_size} must be divisible by the number of GPUs on"
                f" this node ({n_devices}); for single-node set"
                f" ep_size={n_devices}."
            )
        ep_max_tokens_per_rank = calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
            ep_size=ep_size,
            data_parallel_degree=self.pipeline_config.model.data_parallel_degree,
            use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
        )
        # Only enable shared expert fusion if the shared expert is of
        # the same shape as routed experts.
        fused_shared_expert = model_config.n_shared_experts == 1

        # Set correct dispatch dtype, depending on the quant config.
        dispatch_dtype = DType.bfloat16
        dispatch_quant_config = None
        quant_config = model_config.quant_config
        if quant_config is not None and (
            self.dtype.is_float8() or quant_config.is_nvfp4
        ):
            dispatch_dtype = self.dtype
            dispatch_quant_config = quant_config
        return EPConfig(
            dispatch_dtype=dispatch_dtype,
            dispatch_quant_config=dispatch_quant_config,
            combine_dtype=DType.bfloat16,
            hidden_size=model_config.hidden_size,
            top_k=model_config.num_experts_per_tok,
            n_experts=model_config.n_routed_experts,
            max_tokens_per_rank=ep_max_tokens_per_rank,
            n_gpus_per_node=n_devices,
            n_nodes=ep_size // n_devices,
            fused_shared_expert=fused_shared_expert,
            use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: Any,
        devices: list[DeviceRef],
        kv_cache_config: Any,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        return DeepseekV3Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights supported in DeepseekV3."
            )
        return super()._load_state_dict()

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = DeepseekV3Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.max_batch_context_length = (
            self.planned_max_batch_total_tokens
            or model_config.max_batch_context_length
        )

        dtype = self.dtype
        quant_config = None
        if dtype in (
            DType.float8_e4m3fn,
            DType.uint8,
            DType.float4_e2m1fn,
        ):
            quant_config = parse_quant_config(
                self.huggingface_config, state_dict, dtype
            )
        model_config.quant_config = quant_config

        if model_config.topk_method == "noaux_tc":
            correction_bias_key = None
            for k in state_dict:
                if k.endswith("e_score_correction_bias"):
                    correction_bias_key = k
                    break
            if correction_bias_key is None:
                raise KeyError("Expected e_score_correction_bias in state_dict")
            model_config.correction_bias_dtype = state_dict[
                correction_bias_key
            ].dtype

        # Only 1-D TP/DP mesh configurations are supported.
        n_devices = len(self.devices)
        dp_degree = self.pipeline_config.model.data_parallel_degree
        if dp_degree > 1 and dp_degree != n_devices:
            raise NotImplementedError(
                f"data_parallel_degree={dp_degree} must equal the device "
                f"count ({n_devices}); dp x tp meshes are not supported yet."
            )
        axis_name = "dp" if dp_degree > 1 else "tp"
        model_config.mesh = DeviceMesh(
            tuple(self.devices), (n_devices,), (axis_name,)
        )
        return model_config

    @override
    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        del state_dict
        # When the weights are FP8, build the module with a bf16 default so
        # the non-quantized parameters (norms, biases, embeddings) match the
        # checkpoint's bf16 storage.
        if model_config.quant_config is not None:
            return DType.bfloat16
        return model_config.dtype

    @override
    def _init_distributed_runtime(self, model_config: Any) -> None:
        super()._init_distributed_runtime(model_config)
        self._ep_batch_manager = None

        ep_size = self.pipeline_config.runtime.ep_size
        if ep_size <= 1:
            return

        # Expert parallelism: ep_size > 1 distributes routed experts across the
        # devices via the NVSHMEM EPBatchManager. The communication buffers are
        # allocated once here and threaded through the graph as extra inputs.
        n_devices = len(self.devices)
        ep_config = self._build_ep_config(ep_size, n_devices, model_config)
        model_config.ep_config = ep_config
        self._ep_batch_manager = EPBatchManager(ep_config)
        self._modulev3_extra_input_types = self._ep_batch_manager.input_types()
        if not is_virtual_device_mode():
            self.ep_comm_initializer = EPCommInitializer(ep_config)
            self.ep_comm_initializer.ep_init(self.session)
            ep_config.node_id = self.ep_comm_initializer.config.node_id

    @override
    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = DeepseekV3(
            model_config, self.kv_params, self._ep_batch_manager
        )
        assert model_config.mesh is not None
        nn_model.to(model_config.mesh)
        return nn_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model."""
        model_outputs = self.model(*model_inputs.buffers)
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1].driver_tensor),
                next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
                logit_offsets=cast(Buffer, model_outputs[2].driver_tensor),
            )
        return ModelOutputs(
            logits=cast(Buffer, model_outputs[0].driver_tensor),
            next_token_logits=cast(Buffer, model_outputs[0].driver_tensor),
        )
