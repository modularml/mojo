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
"""Implements the DeepseekV3.2 PipelineModel."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.graph.weights import WeightData
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.pipelines.lib import PipelineConfig
from max.pipelines.weights.quant import parse_quant_config
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Model
from .deepseekV3_2 import DeepseekV3_2
from .model_config import DeepseekV3_2Config

logger = logging.getLogger("max.pipelines")


class DeepseekV3_2Model(DeepseekV3Model):
    """A DeepseekV3.2 model."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV3_2Config

    @classmethod
    def _ep_max_rank_send_tokens_for_pipeline(
        cls, pipeline_config: PipelineConfig
    ) -> int:
        """Each rank holds full-length activations before EP MoE (no RS like V3 TP_EP)."""
        return pipeline_config.runtime.max_batch_input_tokens

    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> DeepseekV3_2Config:
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
        else:
            quant_config = None

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

            ep_max_rank_send_tokens = (
                self._ep_max_rank_send_tokens_for_pipeline(self.pipeline_config)
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
                # Fuse into EP dispatch only when shared experts use the same
                # quantized layout as routed experts (modelopt ``*shared_experts*``
                # ignore leaves them bf16 → separate unfused path).
                if quant_config is None:
                    ep_kwargs["fused_shared_expert"] = True
                else:
                    ep_kwargs["fused_shared_expert"] = (
                        quant_config.shared_experts_dtype(DType.bfloat16)
                        == dtype
                    )

            if quant_config is not None:
                ep_kwargs["dispatch_quant_config"] = quant_config

            ep_config = EPConfig(**ep_kwargs)

        norm_dtype = state_dict[
            "layers.0.self_attn.kv_a_layernorm.weight"
        ].dtype

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
        model_config = self.arch_config

        # Finalize config with state_dict-dependent parameters
        model_config.norm_dtype = norm_dtype
        model_config.correction_bias_dtype = correction_bias_dtype
        model_config.max_batch_context_length = max_batch_total_tokens
        model_config.quant_config = quant_config
        model_config.ep_config = ep_config
        model_config.graph_mode = graph_mode
        model_config.data_parallel_degree = data_parallel_degree
        model_config.return_logits = self.return_logits
        model_config.return_hidden_states = self.return_hidden_states

        if ep_size > 1:
            attn_strategy = "TP" if data_parallel_degree == 1 else "DP"
            logger.info(
                f"DeepSeekV3.2: data_parallel_degree={data_parallel_degree},"
                f" ep_size={ep_size}. Use {attn_strategy}-attention + EP-MoE"
                f" strategy."
            )

        return model_config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: DeepseekV3_2Config,
    ) -> None:
        self.ep_comm_initializer = None
        if model_config.ep_config is None:
            return
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
        model_config: DeepseekV3_2Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        nn_model = DeepseekV3_2(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=True)
        weights_registry = nn_model.state_dict()

        with Graph(
            "deepseekV3_2_graph",
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
            signal_buffers = [
                next(variadic_args_iter).buffer
                for _ in range(len(self.devices))
            ]

            mla_kv_caches_per_dev, indexer_kv_caches_per_dev = (
                self.kv_params.unflatten_basic_kv_tree(variadic_args_iter)
            )

            batch_context_lengths = [
                next(variadic_args_iter).tensor
                for _ in range(len(self.devices))
            ]

            ep_model_inputs = list(variadic_args_iter)

            outputs = nn_model(
                tokens.tensor,
                signal_buffers,
                mla_kv_caches_per_dev,
                indexer_kv_caches_per_dev,
                return_n_logits.tensor,
                devices_input_row_offsets.tensor,
                host_input_row_offsets.tensor,
                data_parallel_splits.tensor,
                batch_context_lengths,
                ep_model_inputs,
            )

            graph.output(*outputs)
            return graph, weights_registry
