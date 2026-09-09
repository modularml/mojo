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
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from max._core.engine import Model
from max.driver import Buffer, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from typing_extensions import override

from ..llama3.model import Llama3Inputs, LlamaModelBase
from .batch_processor import MiniMaxM2BatchProcessor
from .minimax_m2 import MiniMaxM2
from .model_config import MiniMaxM2Config

logger = logging.getLogger("max.pipelines")


@dataclass
class MiniMaxM2Inputs(Llama3Inputs):
    """Inputs for MiniMax-M2 with EP and DP support."""

    ep_inputs: tuple[Buffer, ...] = field(kw_only=True, default=())
    host_input_row_offsets: Buffer | None = field(kw_only=True, default=None)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        # Must match MiniMaxM2.input_types / the graph-input unpacking:
        #   tokens, input_row_offsets, return_n_logits,
        #   [data_parallel_splits, host_input_row_offsets]  (DP attention only),
        #   *signals, *kv,
        #   *ep_inputs                                       (EP MoE only)
        assert self.kv_cache_inputs is not None
        kv_flat = self.kv_cache_inputs.flatten()
        result: tuple[Buffer, ...] = (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
        )

        data_parallel_splits = self.data_parallel_splits
        if data_parallel_splits is not None:
            # DP attention: include the batch-split tensors.
            host_input_row_offsets = self.host_input_row_offsets
            assert isinstance(data_parallel_splits, Buffer)
            assert host_input_row_offsets is not None
            result += (data_parallel_splits, host_input_row_offsets)

        result += (*self.signal_buffers, *kv_flat)

        if self.ep_inputs:
            # EP MoE (TP+EP or DP+EP): include the EP communication buffers.
            result += (*self.ep_inputs,)

        return result


class MiniMaxM2Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """MiniMax-M2 pipeline model for text generation.

    Uses AlwaysSignalBuffersMixin since VocabParallelEmbedding and
    ColumnParallelLinear always require signal buffers for allreduce.
    """

    model_config_cls: ClassVar[type[Any]] = MiniMaxM2Config
    batch_processor_cls: ClassVar[type[MiniMaxM2BatchProcessor]] = (
        MiniMaxM2BatchProcessor
    )

    model: Model
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = MiniMaxM2Config.initialize_from_config(
            self.pipeline_config,
            self.huggingface_config,
            max_seq_len=self.max_seq_len,
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        # Detect dtypes from state dict
        for k, v in state_dict.items():
            if k.endswith("mlp.gate.gate_score.weight"):
                model_config.gate_dtype = v.dtype
                break

        for k, v in state_dict.items():
            if k.endswith("mlp.gate.e_score_correction_bias"):
                model_config.correction_bias_dtype = v.dtype
                break

        for k, v in state_dict.items():
            if k.endswith("self_attn.q_proj.weight"):
                model_config.attn_dtype = v.dtype
                break

        num_devices = len(self.devices)
        data_parallel_degree = self.pipeline_config.model.data_parallel_degree
        ep_size = self.pipeline_config.runtime.ep_size
        ep_use_allreduce = self.pipeline_config.runtime.ep_use_allreduce
        # EP is enabled (for both DP+EP and TP+EP) whenever ep_size > 1 on a
        # multi-GPU node. The attention strategy is chosen separately by
        # data_parallel_degree (==1 -> TP attention, ==num_devices -> DP).
        ep_enabled = num_devices > 1 and ep_size > 1

        if ep_enabled:
            if ep_size % num_devices != 0:
                raise ValueError(
                    f"ep_size={ep_size} is not divisible by the number of GPUs"
                    f" on this node ({num_devices}). ep_size must equal"
                    f" n_gpus_per_node * n_nodes. For a single-node deployment"
                    f" set ep_size={num_devices}."
                )
            n_nodes = ep_size // num_devices
            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=data_parallel_degree,
                use_allreduce=ep_use_allreduce,
            )
            is_mxfp4 = (
                model_config.quant_config is not None
                and model_config.quant_config.is_mxfp4
            )
            model_config.ep_config = EPConfig(
                dispatch_dtype=DType.uint8 if is_mxfp4 else model_config.dtype,
                combine_dtype=DType.bfloat16,
                hidden_size=model_config.hidden_size,
                top_k=model_config.num_experts_per_tok,
                n_experts=model_config.num_local_experts,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_gpus_per_node=num_devices,
                n_nodes=n_nodes,
                dispatch_quant_config=model_config.quant_config,
                use_allreduce=ep_use_allreduce,
            )
        else:
            model_config.ep_config = None

        return model_config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: Any,
    ) -> None:
        assert isinstance(model_config, MiniMaxM2Config)
        self.ep_comm_initializer = None
        # Skip the EP comm allocation under a virtual device (compile-only graph
        # dumps, e.g. the kernel-model-map sweep): ep_init allocates NVSHMEM
        # buffers the virtual device can't service. ep_config still shapes the
        # graph, and execution never runs in virtual mode.
        if model_config.ep_config is None or is_virtual_device_mode():
            return
        self.ep_comm_initializer = EPCommInitializer(model_config.ep_config)
        self.ep_comm_initializer.ep_init(session)
        attn_strategy = (
            "TP"
            if self.pipeline_config.model.data_parallel_degree == 1
            else "DP"
        )
        logger.info(
            "MiniMax-M2 EP initialized (%s-attention + EP-MoE,"
            " use_allreduce=%s): node_id=%s, n_gpus=%s, n_nodes=%s, "
            "n_experts=%s, max_tokens_per_rank=%s",
            attn_strategy,
            model_config.ep_config.use_allreduce,
            model_config.ep_config.node_id,
            model_config.ep_config.n_gpus_per_node,
            model_config.ep_config.n_nodes,
            model_config.ep_config.n_experts,
            model_config.ep_config.max_tokens_per_rank,
        )

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, MiniMaxM2Config)
        nn_model = MiniMaxM2(model_config)
        graph_inputs = nn_model.input_types(self.kv_params)
        num_devices = len(self.devices)
        ep_enabled = model_config.ep_config is not None

        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=(
                not getattr(
                    self.huggingface_config, "tie_word_embeddings", False
                )
            ),
        )
        weights_registry = nn_model.state_dict()

        with Graph("minimax_m2", input_types=graph_inputs) as graph:
            # Unpack inputs in the exact order declared by
            # MiniMaxM2.input_types (and produced by MiniMaxM2Inputs.buffers):
            #   tokens, input_row_offsets, return_n_logits,
            #   [data_parallel_splits, host_input_row_offsets]  (DP only),
            #   *signals, *kv, *ep_inputs                       (EP only)
            graph_inputs_iter = iter(graph.inputs)
            tokens = next(graph_inputs_iter)
            input_row_offsets = next(graph_inputs_iter)
            return_n_logits = next(graph_inputs_iter)

            data_parallel_splits = None
            host_input_row_offsets = None
            # Gate on the same predicate input_types used to declare these, so
            # the declaration and this unpacking stay in lockstep (dp_attention
            # is equivalent to data_parallel_degree > 1).
            if nn_model.dp_attention:
                data_parallel_splits = next(graph_inputs_iter)
                host_input_row_offsets = next(graph_inputs_iter)

            signal_buffers = [
                next(graph_inputs_iter).buffer for _ in range(num_devices)
            ]
            num_kv_inputs = len(nn_model.kv_params.flattened_kv_inputs())
            kv_cache_inputs = [
                next(graph_inputs_iter) for _ in range(num_kv_inputs)
            ]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            # Remaining inputs (if any) are the EP communication buffers.
            ep_inputs = list(graph_inputs_iter) if ep_enabled else None

            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                ep_inputs,
                data_parallel_splits.tensor
                if data_parallel_splits is not None
                else None,
                host_input_row_offsets.tensor
                if host_input_row_offsets is not None
                else None,
            )

            graph.output(*outputs)
            return graph, weights_registry
