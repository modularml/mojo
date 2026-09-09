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
"""Implements the DeepseekV3 nn.model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from max.driver import Buffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, ops
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    ModelInputs,
    ModelOutputs,
)
from max.pipelines.weights.quant import parse_quant_config
from typing_extensions import override

from ..deepseekV2.model import DeepseekV2Inputs, DeepseekV2Model
from .batch_processor import DeepseekV3BatchProcessor
from .deepseekV3 import DeepseekV3
from .memory_planner import (
    _ep_max_rank_send_tokens_for_pipeline,
    _get_mtp_draft_ep_dispatch_dtype,
)
from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV3Inputs(DeepseekV2Inputs):
    """A class representing inputs for the DeepseekV3 model."""

    host_input_row_offsets: Buffer
    """Tensor containing the host input row offsets."""

    batch_context_lengths: list[Buffer]
    """List of tensors containing the context length of each batch."""

    data_parallel_splits: Buffer = field(kw_only=True)
    """Tensor containing the data parallel splits for the MLA layer."""

    ep_inputs: tuple[Buffer, ...] = field(kw_only=True, default=())
    """Expert parallel communication buffers (atomic counters and device pointers)."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
            *self.batch_context_lengths,
            *self.ep_inputs,
        )


class DeepseekV3Model(AlwaysSignalBuffersMixin, DeepseekV2Model):
    """A DeepseekV3 model."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV3Config
    batch_processor_cls: ClassVar[type[DeepseekV3BatchProcessor]] = (
        DeepseekV3BatchProcessor
    )

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        """Create model configuration from huggingface config."""
        config = self.huggingface_config

        # data_parallel_degree controls the attention strategy:
        #   == num_devices  ->  DP attention  (each device owns a batch shard)
        #   == 1            ->  TP attention  (heads sharded, tokens replicated)
        data_parallel_degree = self.pipeline_config.model.data_parallel_degree
        max_batch_total_tokens = self.planned_max_batch_total_tokens
        # PipelineConfig would automatically resolve it if not set by user.
        assert max_batch_total_tokens is not None, "max_length must be set"

        if self.pipeline_config.runtime.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif self.pipeline_config.runtime.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        dtype = self.dtype
        if dtype in (DType.float8_e4m3fn, DType.uint8, DType.float4_e2m1fn):
            quant_config = parse_quant_config(config, state_dict, dtype)
            if quant_config is None:
                # No quant config resolved, so these weights aren't
                # quantized -- e.g. a BF16 NextN draft that inherited the
                # target's NVFP4 dtype. Fall back to bfloat16: a quantized
                # dispatch dtype is only valid when paired with a
                # dispatch_quant_config, which we don't have here.
                dtype = DType.bfloat16
        else:
            quant_config = None

        # Check if EP should be configured
        ep_size = self.pipeline_config.runtime.ep_size
        if ep_size == 1:
            ep_config = None
        else:
            if ep_size % len(self.devices) != 0:
                raise ValueError(
                    f"ep_size={ep_size} is not divisible by the number of GPUs"
                    f" on this node ({len(self.devices)}). ep_size must equal"
                    f" n_gpus_per_node * n_nodes. For a single-node deployment"
                    f" set ep_size={len(self.devices)}."
                )

            n_nodes = ep_size // len(self.devices)

            ep_max_rank_send_tokens = _ep_max_rank_send_tokens_for_pipeline(
                self.pipeline_config
            )

            ep_kwargs: dict[str, Any] = dict(
                dispatch_dtype=dtype,
                combine_dtype=DType.bfloat16,
                hidden_size=config.hidden_size,
                top_k=config.num_experts_per_tok,
                n_experts=config.n_routed_experts,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_gpus_per_node=len(self.devices),
                n_nodes=n_nodes,
                dispatch_quant_config=None,
                use_allreduce=self.pipeline_config.runtime.ep_use_allreduce,
            )

            if config.n_shared_experts == 1:
                # Only enable shared expert fusion if the shared expert is of
                # the same shape as routed experts.
                ep_kwargs["fused_shared_expert"] = True

            if quant_config is not None:
                ep_kwargs["dispatch_quant_config"] = quant_config

            ep_config = EPConfig(**ep_kwargs)

        norm_dtype = state_dict[
            "layers.0.self_attn.kv_a_layernorm.weight"
        ].dtype

        # Extract gate dtype from actual weights (may differ from norm_dtype).
        gate_dtype_key = None
        for k in state_dict:
            if k.endswith("gate.gate_score.weight"):
                gate_dtype_key = k
                break
        gate_dtype = (
            state_dict[gate_dtype_key].dtype
            if gate_dtype_key is not None
            else None
        )

        if config.topk_method == "noaux_tc":
            correction_bias_key = None
            for k in state_dict:
                if k.endswith("e_score_correction_bias"):
                    correction_bias_key = k
                    break
            if correction_bias_key is None:
                raise KeyError("Expected e_score_correction_bias in state_dict")
            correction_bias_dtype = state_dict[correction_bias_key].dtype
        else:
            correction_bias_dtype = None

        # Initialize config with parameters from pipeline_config
        model_config = DeepseekV3Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )

        # Finalize config with state_dict-dependent parameters
        # `dtype` is downgraded to bfloat16 above when a quantized target has
        # no resolvable quant config (BF16 draft); otherwise it equals
        # self.dtype, so this is a no-op for the normal path.
        model_config.dtype = dtype
        model_config.norm_dtype = norm_dtype
        model_config.gate_dtype = gate_dtype
        model_config.correction_bias_dtype = correction_bias_dtype
        model_config.max_batch_context_length = max_batch_total_tokens
        model_config.quant_config = quant_config
        model_config.ep_config = ep_config
        model_config.graph_mode = graph_mode
        model_config.data_parallel_degree = data_parallel_degree
        model_config.return_logits = self.return_logits
        model_config.return_hidden_states = self.return_hidden_states

        num_devices = len(self.devices)
        if num_devices > 1:
            if ep_size > 1:
                attn_strategy = "TP" if data_parallel_degree == 1 else "DP"
                moe_strategy = "EP"
            else:
                attn_strategy = "TP"
                moe_strategy = "TP"
            logger.info(
                f"DeepSeekV3: data_parallel_degree={data_parallel_degree},"
                f" ep_size={ep_size}. Use {attn_strategy}-attention +"
                f" {moe_strategy}-MoE strategy."
            )

        return model_config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: Any,
    ) -> None:
        assert isinstance(model_config, DeepseekV3Config)
        self.ep_comm_initializer = None
        # Skip EP initialization in virtual device mode (compilation-only)
        # since NVSHMEM functions cannot be linked without real GPU devices.
        # We still keep ep_config to generate the correct graph structure.
        if model_config.ep_config is None or is_virtual_device_mode():
            return

        ep_alloc_config = model_config.ep_config
        # When EAGLE/MTP speculative decoding shares EP buffers between
        # target (FP4) and draft (BF16) models, allocate buffers
        # large enough for the draft model's dispatch dtype.
        draft_ep_dtype = _get_mtp_draft_ep_dispatch_dtype(self.pipeline_config)
        if draft_ep_dtype is not None:
            ep_alloc_config = replace(
                model_config.ep_config,
                dispatch_dtype=draft_ep_dtype,
                dispatch_quant_config=None,
            )
            logger.info(
                f"Upsizing EP buffers for draft model dispatch dtype: {draft_ep_dtype}"
            )
        self.ep_comm_initializer = EPCommInitializer(ep_alloc_config)
        self.ep_comm_initializer.ep_init(session)
        # ep_init() sets node_id on the initializer's config; propagate
        # it back to the model's ep_config (which may be a different
        # object when we created a copy above).
        model_config.ep_config.node_id = ep_alloc_config.node_id
        if model_config.ep_config.node_id == -1:
            raise ValueError(
                "EP node ID is not set. Please check if the EP initialization is successful."
            )

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, DeepseekV3Config)
        nn_model = DeepseekV3(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        weights_registry = nn_model.state_dict()

        # Create the graph
        with Graph(
            "deepseekV3_graph",
            input_types=nn_model.input_types(self.kv_params),
        ) as graph:
            (
                tokens,
                devices_input_row_offsets,
                host_input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                *variadic_args,
            ) = graph.inputs

            variadic_args_iter = iter(variadic_args)
            # Multi-GPU passes a signal buffer per device: unmarshal these.
            signal_buffers = [
                next(variadic_args_iter).buffer
                for _ in range(len(self.devices))
            ]

            # Unmarshal the KV cache arguments.
            kv_inputs = self.kv_params.unflatten_kv_inputs(variadic_args_iter)
            assert isinstance(kv_inputs, KVCacheInputs)
            kv_caches_per_dev = list(kv_inputs.inputs)

            # Unmarshal the batch context lengths
            batch_context_lengths = [
                next(variadic_args_iter).tensor
                for _ in range(len(self.devices))
            ]

            # all remaining arguments are for EP inputs
            ep_model_inputs = list(variadic_args_iter)

            # DeepseekV3.__call__ expects a per-device list for
            # input_row_offsets
            input_row_offsets_per_dev = list(
                ops.distributed_broadcast(
                    devices_input_row_offsets.tensor, signal_buffers
                )
            )
            outputs = nn_model(
                tokens.tensor,
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
        assert isinstance(model_inputs, DeepseekV3Inputs)

        model_outputs = self.model.execute(*model_inputs.buffers)
        num_outputs = len(model_outputs)

        # Possible output configurations:
        # - 4 outputs: next_token_logits, logits, logit_offsets + hidden_states
        # - 3 outputs: next_token_logits, logits, logit_offsets (variable logits)
        # - 2 outputs: next_token_logits + hidden_states
        # - 1 output: next_token_logits only

        if num_outputs == 4:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            assert isinstance(model_outputs[3], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
                hidden_states=model_outputs[3],
            )
        elif num_outputs == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        elif num_outputs == 2:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
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
