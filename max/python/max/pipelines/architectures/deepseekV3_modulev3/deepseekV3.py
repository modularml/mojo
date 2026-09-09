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
"""Implements the DeepseekV3 model using the ModuleV3 API."""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.embedding import VocabParallelEmbedding
from max.experimental.nn.common_layers.functional_kernels import local_map
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.mesh_axis import DP
from max.experimental.nn.module import subgraphable
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorValue, ops
from max.nn.comm.ep import EPBatchManager, EPCommBuffers
from max.nn.data_parallelism import split_batch_replicated
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParamInterface,
)
from max.nn.rotary_embedding import DeepseekYarnRopeScalingParams
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_moe import (
    QuantizedMoE,
)

from ..deepseekV2_modulev3.layers.rotary_embedding import (
    DeepseekYarnRotaryEmbedding,
)
from .layers.transformer_block import DeepseekV3TransformerBlock
from .model_config import DeepseekV3Config


def gather_last_tokens(h: Tensor, input_row_offsets: Tensor) -> Tensor:
    """Gathers the last token of each request from the ragged batch, per shard.

    Under data parallelism ``input_row_offsets`` is rebased into each replica's
    local frame, so its values only address that replica's own rows; the gather
    runs shard-locally via ``local_map``, never through the auto-sharded
    ``F.gather``.
    """

    def _body(h: Tensor, offsets: Tensor) -> TensorValue:
        return ops.gather(TensorValue(h), TensorValue(offsets)[1:] - 1, axis=0)

    if not h.is_distributed:
        return F.gather(h, input_row_offsets[1:] - 1, axis=0)
    outs = local_map(_body, {"h": h, "offsets": input_row_offsets}, {})
    # Placements-only mapping: the body changes the sharded extent (token
    # rows -> request rows), so a dim-carrying mapping would disagree.
    return Tensor.from_shard_values(
        outs, mapping=PlacementMapping(h.mesh, h.placements)
    )


def split_replicated_batch(
    h: Tensor,
    input_row_offsets: Tensor,
    input_row_offsets_i64: Tensor,
    data_parallel_splits: Tensor,
    mapping: DeviceMapping,
) -> tuple[Tensor, Tensor]:
    """Splits a replicated batch into data-parallel shards."""
    devices = mapping.mesh.devices

    h_shards, input_row_offsets_shards = split_batch_replicated(
        [DeviceRef.from_device(d) for d in devices],
        [TensorValue(shard) for shard in h.local_shards],
        [TensorValue(shard) for shard in input_row_offsets.local_shards],
        TensorValue(input_row_offsets_i64),
        TensorValue(data_parallel_splits),
    )
    return (
        Tensor.from_shard_values(h_shards, mapping),
        Tensor.from_shard_values(input_row_offsets_shards, mapping),
    )


class DeepseekV3TextModel(
    Module[
        [
            Tensor,
            PagedCacheValues,
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            EPCommBuffers | None,
        ],
        tuple[Tensor, ...],
    ]
):
    """The DeepseekV3 language model.

    Decoder-only Transformer with Multi-Latent Attention, MoE feed-forward
    (using a noaux_tc sigmoid router), and DeepSeek YaRN rotary embeddings.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        assert config.rope_scaling is not None
        self.ep_batch_manager = ep_batch_manager

        scaling_params = DeepseekYarnRopeScalingParams(
            scaling_factor=config.rope_scaling["factor"],
            original_max_position_embeddings=config.rope_scaling[
                "original_max_position_embeddings"
            ],
            beta_fast=config.rope_scaling["beta_fast"],
            beta_slow=config.rope_scaling["beta_slow"],
            mscale=config.rope_scaling["mscale"],
            mscale_all_dim=config.rope_scaling["mscale_all_dim"],
        )
        self.rope = DeepseekYarnRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            interleaved=config.rope_interleave,
            scaling_params=scaling_params,
        )

        # Override the tensor parallel axis if data parallelism is enabled.
        tp_axis = DP if config.data_parallel_degree > 1 else None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            dim=config.hidden_size,
            tp_axis=tp_axis,
        )

        self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = ColumnParallelLinear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
            tp_axis=tp_axis,
        )

        qk_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim
        scale = self.rope.compute_scale(math.sqrt(1.0 / qk_head_dim))
        layers = []
        for i in range(config.num_hidden_layers):
            layer = DeepseekV3TransformerBlock(
                config=config,
                layer_idx=i,
                attention_scale=scale,
                ep_batch_manager=ep_batch_manager,
            )

            # Subgraph the blocks with MoE.
            if isinstance(layer.mlp, QuantizedMoE):
                layer = subgraphable(layer, name="moe_block")
            layers.append(layer)

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList[DeepseekV3TransformerBlock](layers)
        self.kv_params = config.kv_params
        self.config = config
        self.mesh = config.mesh

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        batch_context_length: Tensor,
        data_parallel_splits: Tensor | None = None,
        input_row_offsets_i64: Tensor | None = None,
        comm_buffers: EPCommBuffers | None = None,
    ) -> tuple[Tensor, ...]:
        if self.mesh is not None:
            tokens = F.distributed_broadcast(tokens, self.mesh)
            input_row_offsets = F.distributed_broadcast(
                input_row_offsets, self.mesh
            )

        h = self.embed_tokens(tokens)
        if self.config.data_parallel_degree > 1:
            assert data_parallel_splits is not None
            assert input_row_offsets_i64 is not None
            assert self.mesh is not None
            batch_placement = tuple(
                Sharded(0) if name == "dp" else Replicated()
                for name in self.mesh.axis_names
            )
            h, input_row_offsets = split_replicated_batch(
                h,
                input_row_offsets,
                input_row_offsets_i64,
                data_parallel_splits,
                DeviceMapping(self.mesh, batch_placement),
            )

        freqs_cis = F.cast(self.rope.freqs_cis, h.dtype)
        if self.mesh is not None:
            freqs_cis = freqs_cis.to(self.mesh)
        else:
            freqs_cis = freqs_cis.to(h.device)

        # The MLA prefill plan depends only on the sequence layout (identical
        # across layers), so compute it once and thread it into every layer
        # instead of recomputing it per layer. Decode has no prefill plan.
        mla_prefill_metadata = None
        first_attn = self.layers[0].self_attn
        if first_attn.graph_mode in ("prefill", "auto"):
            mla_prefill_metadata = first_attn.create_mla_prefill_metadata(
                input_row_offsets, kv_collection
            )
            # Host-substitute the per-layer D2H buffer_length copies with the
            # CPU batch_context_length so the graph stays capturable.
            mla_prefill_metadata.buffer_lengths = batch_context_length

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=CPU())
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets,
                freqs_cis,
                mla_prefill_metadata,
                comm_buffers,
            )

        last_token_h = gather_last_tokens(h, input_row_offsets)
        if self.config.data_parallel_degree > 1:
            last_token_h = F.allgather(last_token_h)
        last_logits = self.lm_head(self.norm(last_token_h))
        if self.mesh is not None:
            last_logits = last_logits.to(self.mesh.devices[0])
        last_logits = F.cast(last_logits, DType.float32)
        return (last_logits,)


class DeepseekV3(Module[..., tuple[Tensor, ...]]):
    """Top-level DeepseekV3 wrapper that unflattens variadic KV cache args."""

    def __init__(
        self,
        config: DeepseekV3Config,
        kv_params: KVCacheParamInterface,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.language_model = DeepseekV3TextModel(config, ep_batch_manager)
        self.config = config
        self.kv_params = kv_params
        self.ep_batch_manager = ep_batch_manager

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args: Tensor,
    ) -> tuple[Tensor, ...]:
        mesh = self.config.mesh
        assert mesh is not None

        # Reconstruct inputs from variadic arguments.

        data_parallel_splits: Tensor | None = None
        input_row_offsets_i64: Tensor | None = None
        dp_degree = self.config.data_parallel_degree
        batch_context_lengths = variadic_args[:dp_degree]
        variadic_args = variadic_args[dp_degree:]
        if dp_degree > 1:
            assert dp_degree == mesh.num_devices

            cpu_mesh = DeviceMesh(
                tuple(CPU() for _ in range(dp_degree)), (dp_degree,), ("dp",)
            )
            batch_context_lengths_tensor = Tensor.from_shard_values(
                [TensorValue(shard) for shard in batch_context_lengths],
                PlacementMapping(cpu_mesh, (Replicated(),) * mesh.ndim),
            )

            data_parallel_splits, input_row_offsets_i64, *rest = variadic_args
            variadic_args = tuple(rest)
        else:
            batch_context_lengths_tensor = batch_context_lengths[0]

        kv_inputs = iter(x._graph_value for x in variadic_args)
        kv_collections = self.kv_params.unflatten_kv_inputs(kv_inputs)
        assert isinstance(kv_collections, KVCacheInputs)

        # Combine the per-device upstream KV collections into a single
        # mesh-distributed PagedCacheValues (one shard per device).
        if mesh is not None:
            kv_collection = PagedCacheValues.from_upstream(
                kv_collections.inputs,
                PlacementMapping(mesh, (Replicated(),) * mesh.ndim),
            )
        else:
            raise ValueError("Mesh must be define")

        comm_buffers: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            # Any variadic graph values left after the KV cache are the EP
            # communication buffers. Wrap them as an explicit forward argument
            # so they thread through each MoE block's subgraph boundary.
            ep_buffers = list(kv_inputs)
            comm_buffers = self.ep_batch_manager.comm_buffers(ep_buffers)
        return self.language_model(
            tokens,
            kv_collection,
            return_n_logits,
            input_row_offsets,
            batch_context_lengths_tensor,
            data_parallel_splits,
            input_row_offsets_i64,
            comm_buffers,
        )
