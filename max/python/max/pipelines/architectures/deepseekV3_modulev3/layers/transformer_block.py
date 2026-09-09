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

"""DeepSeek-V3 Transformer block (ModuleV3)."""

from __future__ import annotations

import enum
import functools
from typing import Any

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.multi_latent_attention import (
    MLAPrefillMetadata,
)
from max.experimental.nn.norm import RMSNorm
from max.experimental.sharding import Partial, PlacementMapping
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.comm.ep import EPBatchManager, EPCommBuffers

from ..model_config import DeepseekV3Config
from .moe_gate import DeepseekV3TopKRouter
from .quant_linear import QuantizedMLP, tensor_parallel_mlp
from .quant_mla import (
    QuantizedLatentAttentionWithRope,
    tensor_parallel_latent_attention_with_rope,
)
from .quant_moe import (
    ExpertParallelMoE,
    QuantizedMoE,
    TensorParallelMoE,
)


def _get_mlp(
    config: DeepseekV3Config,
    mode: ParallelismMode,
    layer_idx: int,
    ep_batch_manager: EPBatchManager | None = None,
) -> Module[[Tensor], Tensor]:
    """Returns either an MoE or MLP module for the given layer index.

    The MoE variant is selected by the parallelism strategy: expert parallelism
    (``ep_batch_manager`` set) → :class:`ExpertParallelMoE`; multi-device tensor
    parallelism → :class:`TensorParallelMoE`; single device → :class:`QuantizedMoE`.
    """
    use_moe = (
        config.n_routed_experts is not None
        and layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )
    gate_cls = functools.partial(
        DeepseekV3TopKRouter,
        routed_scaling_factor=config.routed_scaling_factor,
        scoring_func=config.scoring_func,
        topk_method=config.topk_method,
        n_group=config.n_group,
        topk_group=config.topk_group,
        norm_topk_prob=config.norm_topk_prob,
        correction_bias_dtype=config.correction_bias_dtype,
    )
    if use_moe:
        moe_kwargs: dict[str, Any] = dict(
            hidden_dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.moe_intermediate_size,
            gate_cls=gate_cls,
            has_shared_experts=True,
            shared_experts_dim=config.n_shared_experts
            * config.moe_intermediate_size,
            apply_router_weight_first=False,
            quant_config=config.quant_config,
        )
        if ep_batch_manager is not None:
            return ExpertParallelMoE(
                **moe_kwargs, ep_batch_manager=ep_batch_manager
            )
        if (
            mode is not ParallelismMode.DP_EP
            and config.mesh is not None
            and config.mesh.num_devices > 1
        ):
            return TensorParallelMoE(**moe_kwargs)
        # Single device or pure data parallelism: replicated expert set.
        return QuantizedMoE(**moe_kwargs)
    mlp = QuantizedMLP(
        hidden_dim=config.hidden_size,
        feed_forward_length=config.intermediate_size,
        quant_config=(
            None
            if layer_idx in config.dense_mlp_layers_without_quant
            else config.quant_config
        ),
    )
    if mode == ParallelismMode.TP_TP or (
        config.ep_config is not None and config.ep_config.use_allreduce
    ):
        return tensor_parallel_mlp(mlp)
    return mlp


class ParallelismMode(enum.Enum):
    """Parallelism strategy for a DeepseekV3 decoder layer.

    Each mode determines which attention/MoE implementations are used and which
    collective communication ops run after attention and after the MoE/MLP.
    """

    DP_EP = "dp_ep"
    """DP attention + EP MoE.  No inter-device collectives in the residual path."""

    TP_EP = "tp_ep"
    """TP attention (skip allreduce) + EP MoE.  Reduce-scatter after attention
    puts hidden states in sequence-parallel ``[S/P, H]`` form; allgather after
    MoE restores ``[S, H]``."""

    TP_TP = "tp_tp"
    """TP attention (with allreduce) + TP MoE.  Standard allreduce after MoE."""


class DeepseekV3TransformerBlock(Module[..., Tensor]):
    """Stack of MLA attention, MoE/MLP, and RMSNorm for DeepSeek V3."""

    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        attention_scale: float,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        self.config = config
        num_devices = len(config.devices)

        if num_devices <= 1:
            self.mode = ParallelismMode.TP_TP
        elif config.ep_config is not None:
            if config.data_parallel_degree == 1:
                self.mode = ParallelismMode.TP_EP
            else:
                self.mode = ParallelismMode.DP_EP
        elif config.data_parallel_degree == num_devices:
            # Pure data parallelism needs no residual collectives -- exactly
            # the DP_EP arms of the hooks below.
            self.mode = ParallelismMode.DP_EP
        else:
            self.mode = ParallelismMode.TP_TP

        self.self_attn = QuantizedLatentAttentionWithRope(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            layer_idx=layer_idx,
            scale=attention_scale,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            graph_mode=config.graph_mode,
            buffer_size=config.max_batch_context_length,
            quant_config=config.quant_config,
            quantize_o_proj=config.mla_o_proj_quantized,
        )
        tensor_parallel_latent_attention_with_rope(self.self_attn)
        self.mlp = _get_mlp(config, self.mode, layer_idx, ep_batch_manager)
        self.input_layernorm = RMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            dim=config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
        comm_buffers: EPCommBuffers | None = None,
    ) -> Tensor:
        residual = x
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attn(
            norm_x,
            kv_collection,
            freqs_cis,
            layer_idx,
            input_row_offsets,
            mla_prefill_metadata,
        )

        hidden_states = self._post_attention(residual, attn_out)
        norm_h = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, ExpertParallelMoE):
            assert comm_buffers is not None
            mlp_out = self.mlp(norm_h, comm_buffers)
        else:
            mlp_out = self.mlp(norm_h)
        hidden_states = self._post_mlp(hidden_states, mlp_out)
        return F.rebind(hidden_states, x.shape)

    def _post_attention(
        self,
        x: Tensor,
        attn_out: Tensor,
    ) -> Tensor:
        """Residual connection and collective after attention."""
        match self.mode:
            case ParallelismMode.TP_EP:
                assert self.config.ep_config is not None
                if self.config.ep_config.use_allreduce:
                    attn_out = F.allreduce_sum(attn_out)
                    return x + attn_out
                else:
                    # attn_outs[i] is device i's partial sum (allreduce was
                    # skipped).  Add the residual only on device 0 so it isn't
                    # counted P times after the reduce-scatter.
                    mesh = x.mesh
                    attn_shards = [
                        TensorValue(s) for s in attn_out.local_shards
                    ]
                    residual_shards = [TensorValue(s) for s in x.local_shards]
                    folded = [
                        residual_shards[0] + attn_shards[0],
                        *attn_shards[1:],
                    ]
                    partial = Tensor.from_shard_values(
                        folded, PlacementMapping(mesh, (Partial(),))
                    )

                    # Partial -> Sharded(0): real reduce-scatter collective.
                    return F.reduce_scatter(partial, scatter_axis=0)
            case ParallelismMode.TP_TP:
                attn_out = F.allreduce_sum(attn_out)
                return x + attn_out
            case ParallelismMode.DP_EP:
                return x + attn_out
            case _:
                raise ValueError(f"Unsupported parallelism mode: {self.mode}")

    def _post_mlp(
        self,
        h: Tensor,
        mlp_out: Tensor,
    ) -> Tensor:
        """Collective after MoE/MLP to restore the expected hidden-state layout."""
        match self.mode:
            case ParallelismMode.TP_EP:
                assert self.config.ep_config is not None
                if self.config.ep_config.use_allreduce:
                    mlp_out = F.allreduce_sum(mlp_out)
                    return h + mlp_out
                else:
                    h = h + mlp_out
                    return F.allgather(h, tensor_axis=0)
            case ParallelismMode.TP_TP:
                # Both the TP MoE (routed + shared) and the plain TP MLP return
                # each device's partial sum; one all-reduce after the layer
                # resolves it (matches the V2 ``nn.moe`` single-all-reduce
                # layout, where the shared expert is summed before this
                # collective).
                mlp_out = F.allreduce_sum(mlp_out)
                return h + mlp_out
            case ParallelismMode.DP_EP:
                return h + mlp_out
            case _:
                raise ValueError(f"Unsupported parallelism mode: {self.mode}")
