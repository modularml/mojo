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
from typing import Any, ClassVar, Literal

from max._core.engine import Model
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph
from max.nn.comm.ep import EPCommInitializer, EPConfig
from max.nn.comm.ep.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.pipelines.architectures.llama3.model import LlamaModelBase
from max.pipelines.lib import (
    PipelineConfig,
    supported_encoding_dtype,
)
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.weights.quant import parse_quant_config
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig
from typing_extensions import override

from .laguna import Laguna
from .model_config import LagunaConfig

logger = logging.getLogger("max.pipelines")


class LagunaModel(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Laguna pipeline model for text generation.

    Uses ``AlwaysSignalBuffersMixin`` since ``VocabParallelEmbedding`` and
    ``ColumnParallelLinear`` always require signal buffers for allreduce.
    """

    model_config_cls: ClassVar[type[Any]] = LagunaConfig

    model: Model
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        encoding = _select_quantization_encoding(
            pipeline_config.model, LagunaConfig.DEFAULT_ENCODING
        )
        n_gpus_per_node = len(pipeline_config.model.device_specs)
        num_experts = getattr(huggingface_config, "num_local_experts", 256)
        moe_dim = getattr(huggingface_config, "intermediate_size", 1536)
        hidden_size = getattr(huggingface_config, "hidden_size", 3072)
        top_k = getattr(huggingface_config, "num_experts_per_tok", 8)

        ep_buffer_memory = 0
        moe_activation_memory = 0
        ep_size = pipeline_config.runtime.ep_size
        if ep_size > 1:
            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
            )
            ep_dispatch_dtype = supported_encoding_dtype(encoding)

            # Worst-case tokens received per rank during all-to-all routing.
            max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
                num_experts,
                ep_size * top_k,
            )

            # Peak MoE activation: input to second grouped_matmul has shape
            # [max_recv_tokens_per_rank, moe_intermediate_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * moe_dim
                * ep_dispatch_dtype.size_in_bytes
            )
            # Output has shape [max_recv_tokens_per_rank, hidden_size] in
            # bfloat16.
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * hidden_size
                * DType.bfloat16.size_in_bytes
            )
            # 256MB per GPU for misc scalar buffers.
            moe_activation_memory += 256 * 1024 * 1024
            moe_activation_memory *= n_gpus_per_node

            n_nodes = max(ep_size // n_gpus_per_node, 1)
            per_device_ep_memory = estimate_ep_memory_usage(
                hidden_size=hidden_size,
                dispatch_dtype=ep_dispatch_dtype,
                combine_dtype=DType.bfloat16,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_experts=num_experts,
                n_nodes=n_nodes,
                n_gpus_per_node=n_gpus_per_node,
                top_k=top_k,
            )
            # EPCommInitializer double-buffers (NUM_GROUPS=2) the SHMEM
            # dispatch/combine buffers.
            ep_buffer_memory = per_device_ep_memory * n_gpus_per_node * 2

        activation_memory = moe_activation_memory + ep_buffer_memory

        if activation_memory != 0:
            logger.info(
                "Estimated activation memory: %s "
                "(ep_buffers=%s, moe_activation=%s)",
                to_human_readable_bytes(activation_memory),
                to_human_readable_bytes(ep_buffer_memory),
                to_human_readable_bytes(moe_activation_memory),
            )

        return activation_memory

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> LagunaConfig:
        model_config = LagunaConfig.initialize_from_config(
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
        self._resolve_nvfp4_quant_config(model_config, state_dict)
        self._detect_state_dict_dtypes(model_config, state_dict)
        self._setup_ep_config(model_config)
        return model_config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: LagunaConfig,
    ) -> None:
        self.ep_comm_initializer = None
        if model_config.ep_config is None:
            return
        self.ep_comm_initializer = EPCommInitializer(model_config.ep_config)
        self.ep_comm_initializer.ep_init(session)
        logger.info(
            f"EP initialized: node_id={model_config.ep_config.node_id}, "
            f"n_gpus={model_config.ep_config.n_gpus_per_node}, "
            f"n_nodes={model_config.ep_config.n_nodes}, "
            f"n_experts={model_config.ep_config.n_experts}, "
            f"max_tokens_per_rank={model_config.ep_config.max_tokens_per_rank}"
        )

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: LagunaConfig,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        nn_model = Laguna(model_config)
        graph_inputs = nn_model.input_types(self.kv_params)

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

        num_devices = len(self.devices)
        with Graph("laguna", input_types=graph_inputs) as graph:
            inputs_iter = iter(graph.inputs)
            tokens = next(inputs_iter)
            input_row_offsets = next(inputs_iter)
            return_n_logits = next(inputs_iter)
            if model_config.data_parallel_degree > 1:
                data_parallel_splits = next(inputs_iter).tensor
                host_input_row_offsets = next(inputs_iter).tensor
            else:
                data_parallel_splits = None
                host_input_row_offsets = None

            signal_buffers = [
                next(inputs_iter).buffer for _ in range(num_devices)
            ]

            num_kv_inputs = len(
                nn_model.kv_params.get_symbolic_inputs().flatten()
            )
            kv_cache_inputs = [next(inputs_iter) for _ in range(num_kv_inputs)]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            ep_inputs = list(inputs_iter)

            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                ep_inputs,  # type: ignore[arg-type]
                data_parallel_splits,
                host_input_row_offsets,
            )

            graph.output(*outputs)
            return graph, weights_registry

    def _resolve_nvfp4_quant_config(
        self, model_config: LagunaConfig, state_dict: dict[str, Any]
    ) -> None:
        """Builds the NVFP4 ``quant_config`` for the compressed-tensors checkpoint.

        The compute dtype is bf16 (embedding, attention, norms, lm_head), so the
        base ``finalize`` calls ``parse_quant_config`` with bf16 and gets
        ``None``. Laguna ships compressed-tensors NVFP4, which is numerically
        identical to modelopt NVFP4 (group_size 16, e4m3 per-group scales, fp32
        global scale) but is not recognised by MAX's FP4 parser, which only
        handles the modelopt flavor. Present a modelopt-shaped
        ``quantization_config`` to the parser so the dense/MoE Linears pack via
        ``quant_config.is_fp4``; the non-quant layers stay bf16 from
        ``config.dtype``.
        """
        hf_qc = getattr(self.huggingface_config, "quantization_config", None)
        if model_config.quant_config is not None or not hf_qc:
            return
        # The MAX parser reads only from ``huggingface_config``, so present the
        # modelopt-shaped config to it and restore the original afterwards.
        # compressed-tensors defines quant scope by ``targets`` (only MLP/MoE
        # gate/up/down_proj are FP4); the modelopt parser uses inverse
        # ``ignore`` semantics (quantize all Linears except ignore), so ignore
        # everything that is NOT a target: attention, the router gate, lm_head.
        orig_qc = self.huggingface_config.quantization_config
        try:
            self.huggingface_config.quantization_config = {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "ignore": [
                    "re:.*self_attn\\..*",
                    "re:.*\\.mlp\\.gate$",
                    "lm_head",
                ],
            }
            model_config.quant_config = parse_quant_config(
                self.huggingface_config, state_dict, DType.uint8
            )
        finally:
            self.huggingface_config.quantization_config = orig_qc

    @staticmethod
    def _detect_state_dict_dtypes(
        model_config: LagunaConfig, state_dict: dict[str, Any]
    ) -> None:
        """Reads the gate, correction-bias, and attention dtypes off the weights.

        These are kept higher-precision than the FP4 experts and vary by
        checkpoint, so detect them from the loaded tensors rather than assuming
        the compute dtype. Each dtype is uniform across layers.
        """
        for k, v in state_dict.items():
            if k.endswith("mlp.gate.gate_score.weight"):
                model_config.gate_dtype = v.dtype
            elif k.endswith("mlp.gate.e_score_correction_bias"):
                model_config.correction_bias_dtype = v.dtype
            elif k.endswith("self_attn.q_proj.weight"):
                model_config.attn_dtype = v.dtype

    def _setup_ep_config(self, model_config: LagunaConfig) -> None:
        """Sets ``model_config.ep_config``, or disables EP on a single GPU."""
        num_devices = len(self.devices)
        if num_devices <= 1:
            model_config.ep_config = None
            logger.info(
                "EP disabled (single-GPU); MoE runs all "
                f"{model_config.num_local_experts} experts locally"
            )
            return

        ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=self.pipeline_config.runtime.max_batch_input_tokens,
            ep_size=num_devices,
            data_parallel_degree=self.pipeline_config.model.data_parallel_degree,
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
            n_nodes=1,
            dispatch_quant_config=model_config.quant_config,
        )
