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
"""The Inkling text decoder, assembled."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Value,
    Weight,
    ops,
)
from max.nn.comm import Signals
from max.nn.embedding import Embedding, VocabParallelEmbedding
from max.nn.kv_cache import (
    KVCacheInputs,
    MHAKVCacheParams,
    MultiKVCacheInputs,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import LayerList, Module, SubgraphInput
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.moe import make_interleaved_gated_activation_fn
from max.nn.norm import RMSNorm
from max.nn.quant_config import QuantConfig
from max.nn.transformer import (
    ReturnHiddenStates,
    ReturnLogits,
    forward_sequential_layers,
    logits_postprocess,
)
from max.nn.transformer.distributed_transformer import (
    distributed_logits_postprocess,
    forward_sharded_layers,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from .layers.attention import InklingAttention, log_scaling_tau
from .layers.moe import InklingGate, InklingMoE
from .layers.short_convolution import ShortConvolution
from .model_config import (
    GLOBAL_ATTENTION,
    LOCAL_ATTENTION,
    InklingConfig,
    InklingTextConfig,
)
from .state_cache import ConvSite, InklingConvStateLayout


class InklingDecoderLayer(Module):
    """One decoder layer: attention and feed-forward, each with an output
    conv."""

    def __init__(
        self,
        *,
        text_config: InklingTextConfig,
        layer_idx: int,
        kv_params: MHAKVCacheParams,
        dtype: DType,
        devices: list[DeviceRef],
        quant_config: QuantConfig | None = None,
        is_local: bool | None = None,
        force_dense_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.num_devices = len(devices)
        tensor_parallel = ShardingStrategy.tensor_parallel(self.num_devices)
        replicate = ShardingStrategy.replicate(self.num_devices)
        device = devices[0]

        self.attn_norm = RMSNorm(
            text_config.hidden_size, dtype, eps=text_config.rms_norm_eps
        )
        self.attn_norm.sharding_strategy = replicate
        self.attn_norm_shards = list(self.attn_norm.shard(devices))

        self.attn = InklingAttention(
            text_config=text_config,
            layer_idx=layer_idx,
            kv_params=kv_params,
            dtype=dtype,
            devices=devices,
            is_local=is_local,
        )
        self.attn.sharding_strategy = tensor_parallel
        self.attn_shards = list(self.attn.shard(devices))

        self.attn_sconv = ShortConvolution(
            channels=text_config.hidden_size,
            kernel_size=text_config.sconv_kernel_size,
            dtype=dtype,
            device=device,
        )
        self.attn_sconv.sharding_strategy = tensor_parallel
        self.attn_sconv_shards = list(self.attn_sconv.shard(devices))

        self.mlp_norm = RMSNorm(
            text_config.hidden_size, dtype, eps=text_config.rms_norm_eps
        )
        self.mlp_norm.sharding_strategy = replicate
        self.mlp_norm_shards = list(self.mlp_norm.shard(devices))

        self.mlp: InklingMoE | MLP
        # Dense layers only: the learned scalar the output is scaled by.
        self.mlp_global_scale_shards: list[Weight] | None = None
        if text_config.is_moe_layer(layer_idx) and not force_dense_mlp:
            # Only routed experts quantize; sinks and gate stay at dtype.
            routed_quant = (
                quant_config
                if quant_config is not None
                and layer_idx in quant_config.mlp_quantized_layers
                else None
            )
            self.mlp = InklingMoE(
                devices=devices,
                hidden_dim=text_config.hidden_size,
                num_experts=text_config.n_routed_experts,
                num_experts_per_token=text_config.num_experts_per_tok,
                moe_dim=text_config.intermediate_size,
                gate_cls=functools.partial(
                    InklingGate,
                    n_shared_experts=text_config.n_shared_experts,
                    route_scale=text_config.route_scale,
                ),
                has_shared_experts=True,
                shared_experts_dim=(
                    text_config.n_shared_experts * text_config.intermediate_size
                ),
                # Packed FP4 stores as uint8; a differing shared_experts_dtype
                # is what keeps the sinks unquantized in max.nn.moe.
                dtype=DType.uint8 if routed_quant is not None else dtype,
                shared_experts_dtype=dtype,
                gated_activation_fn=make_interleaved_gated_activation_fn(
                    ops.silu
                ),
                quant_config=routed_quant,
            )
        else:
            self.mlp = MLP(
                dtype=dtype,
                quantization_encoding=None,
                hidden_dim=text_config.hidden_size,
                feed_forward_length=text_config.dense_intermediate_size,
                devices=devices,
            )
            # Dotted name matches the checkpoint's mlp namespace.
            self.mlp_global_scale = Weight(
                "mlp.global_scale", DType.float32, [1], device=device
            )
            self.mlp_global_scale.sharding_strategy = replicate
            self.mlp_global_scale_shards = list(
                self.mlp_global_scale.shard(devices)
            )
        self.mlp.sharding_strategy = tensor_parallel
        self.mlp_shards = list(self.mlp.shard(devices))

        self.mlp_sconv = ShortConvolution(
            channels=text_config.hidden_size,
            kernel_size=text_config.sconv_kernel_size,
            dtype=dtype,
            device=device,
        )
        self.mlp_sconv.sharding_strategy = tensor_parallel
        self.mlp_sconv_shards = list(self.mlp_sconv.shard(devices))

    def __call__(
        self,
        hs: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
        log_scaling: Sequence[TensorValue],
        conv_pools: Sequence[Sequence[BufferValue]],
        slot_idx: Sequence[TensorValue],
        cache_layer_idx: TensorValue,
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """Runs the layer over one ragged batch; compiled as a subgraph."""
        norm_xs = forward_sharded_layers(self.attn_norm_shards, hs)
        attention = [
            shard(
                norm_xs[rank],
                kv_collection=kv_collections[rank],
                input_row_offsets=input_row_offsets[rank],
                log_scaling=log_scaling[rank],
                k_conv_pool=conv_pools[rank][ConvSite.K],
                v_conv_pool=conv_pools[rank][ConvSite.V],
                slot_idx=slot_idx[rank],
                cache_layer_idx=cache_layer_idx,
            )
            for rank, shard in enumerate(self.attn_shards)
        ]
        # wo_ud is row-parallel: each rank holds a partial sum of the delta.
        convolved = self._branch_convolution(
            self.attn_sconv_shards,
            attention,
            [pools[ConvSite.ATTN_OUT] for pools in conv_pools],
            slot_idx,
            input_row_offsets,
            signal_buffers,
        )
        hs = [h + delta for h, delta in zip(hs, convolved, strict=True)]

        norm_outs = forward_sharded_layers(self.mlp_norm_shards, hs)
        convolved = self._branch_convolution(
            self.mlp_sconv_shards,
            self._feed_forward(norm_outs),
            [pools[ConvSite.MLP_OUT] for pools in conv_pools],
            slot_idx,
            input_row_offsets,
            signal_buffers,
        )
        return [h + delta for h, delta in zip(hs, convolved, strict=True)]

    def _feed_forward(
        self, norm_xs: Sequence[TensorValue]
    ) -> list[TensorValue]:
        """The dense scale applies before the reduce-scatter splits channels."""
        outs = [
            shard(norm_xs[rank]) for rank, shard in enumerate(self.mlp_shards)
        ]
        if self.mlp_global_scale_shards is None:
            return outs
        # Scaling in float32 keeps the scalar's low bits.
        return [
            ops.cast(
                ops.cast(out, DType.float32) * scale.to(out.device), out.dtype
            )
            for out, scale in zip(
                outs, self.mlp_global_scale_shards, strict=True
            )
        ]

    def _branch_convolution(
        self,
        convs: Sequence[ShortConvolution],
        partials: Sequence[TensorValue],
        pools: Sequence[BufferValue],
        slot_idx: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """Reduce-scatters onto each rank's channels, convolves, all-gathers."""
        if self.num_devices > 1:
            deltas = ops.reducescatter.sum(partials, signal_buffers, axis=-1)
        else:
            deltas = list(partials)
        outputs = [
            conv(
                deltas[rank],
                pools[rank],
                slot_idx[rank],
                input_row_offsets[rank],
            )
            for rank, conv in enumerate(convs)
        ]
        if self.num_devices > 1:
            outputs = ops.allgather(outputs, signal_buffers, axis=-1)
        return outputs


def _subgraph_layer_groups(
    text_config: InklingTextConfig, quant_config: QuantConfig | None
) -> tuple[list[str], list[list[int]]]:
    """Groups the decoder layers that can share one compiled subgraph."""
    groups: dict[tuple[bool, bool, bool], list[int]] = {}
    for layer_idx in range(text_config.num_hidden_layers):
        is_moe = text_config.is_moe_layer(layer_idx)
        key = (
            text_config.is_local_attention(layer_idx),
            is_moe,
            is_moe
            and quant_config is not None
            and layer_idx in quant_config.mlp_quantized_layers,
        )
        groups.setdefault(key, []).append(layer_idx)

    names = []
    shared = []
    for (local, is_moe, quantized), layer_indices in groups.items():
        if len(layer_indices) == 1:
            continue
        flavor = "local" if local else "global"
        mlp = "nvfp4_moe" if quantized else "moe" if is_moe else "dense"
        names.append(f"inkling_{flavor}_{mlp}_block")
        shared.append(layer_indices)
    return names, shared


def kv_collections_by_key(
    tree: MultiKVCacheInputs[TensorValue, BufferValue],
) -> dict[str, list[PagedCacheValues]]:
    """Groups an unflattened KV tree by attention flavor, then by rank."""
    collections: dict[str, list[PagedCacheValues]] = {}
    for key, child in tree.children.items():
        assert isinstance(child, KVCacheInputs)
        collections[key] = list(child.inputs)
    return collections


class Inkling(Module):
    """The Inkling text model: embedding, decoder layers, LM head."""

    def __init__(
        self,
        config: InklingConfig,
        *,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        super().__init__()
        text_config = config.text_config
        self.text_config = text_config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        self.dtype = config.dtype
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
        device = config.devices[0]
        replicate = ShardingStrategy.replicate(self.num_devices)

        # Sized for the decoder layers alone; MTP adds a row per draft depth.
        self.conv_layout = InklingConvStateLayout.from_config(
            text_config, tp_size=self.num_devices
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params: dict[str, MHAKVCacheParams] = {}
        for key, params in config.kv_params.children.items():
            assert isinstance(params, MHAKVCacheParams)
            kv_params[key] = params
        self.kv_params = config.kv_params

        # The embedding and the LM head are 2.4 GB each, so TP splits them.
        self.embed: Embedding | VocabParallelEmbedding
        if self.num_devices > 1:
            self.embed = VocabParallelEmbedding(
                text_config.vocab_size,
                text_config.hidden_size,
                config.dtype,
                config.devices,
            )
        else:
            self.embed = Embedding(
                text_config.vocab_size,
                text_config.hidden_size,
                config.dtype,
                device,
            )
        self.embed_norm: RMSNorm | None = None
        self.embed_norm_shards: list[RMSNorm] = []
        if text_config.use_embed_norm:
            self.embed_norm = RMSNorm(
                text_config.hidden_size,
                config.dtype,
                eps=text_config.rms_norm_eps,
            )
            self.embed_norm.sharding_strategy = replicate
            self.embed_norm_shards = list(self.embed_norm.shard(config.devices))

        layers = []
        self.layer_kv_keys: list[str] = []
        self.layer_cache_indices: list[int] = []
        layers_in_cache = {GLOBAL_ATTENTION: 0, LOCAL_ATTENTION: 0}
        for layer_idx in range(text_config.num_hidden_layers):
            kv_key = (
                LOCAL_ATTENTION
                if text_config.is_local_attention(layer_idx)
                else GLOBAL_ATTENTION
            )
            layers.append(
                InklingDecoderLayer(
                    text_config=text_config,
                    layer_idx=layer_idx,
                    kv_params=kv_params[kv_key],
                    dtype=config.dtype,
                    devices=config.devices,
                    quant_config=config.quant_config,
                )
            )
            self.layer_kv_keys.append(kv_key)
            self.layer_cache_indices.append(layers_in_cache[kv_key])
            layers_in_cache[kv_key] += 1
        self.layers = LayerList(layers)
        self.use_subgraphs = config.use_subgraphs
        self.subgraph_names, self.subgraph_groups = _subgraph_layer_groups(
            text_config, config.quant_config
        )

        self._padded_tail_mask: np.ndarray | None = None
        if text_config.unpadded_vocab_size < text_config.vocab_size:
            self._padded_tail_mask = np.zeros(
                text_config.vocab_size, dtype=np.float32
            )
            self._padded_tail_mask[text_config.unpadded_vocab_size :] = -np.inf

        self.norm = RMSNorm(
            text_config.hidden_size,
            config.dtype,
            eps=text_config.rms_norm_eps,
        )
        self.norm.sharding_strategy = replicate
        self.norm_shards = list(self.norm.shard(config.devices))
        self.unembed: Linear
        if self.num_devices > 1:
            self.unembed = ColumnParallelLinear(
                in_dim=text_config.hidden_size,
                out_dim=text_config.vocab_size,
                dtype=config.dtype,
                devices=config.devices,
                has_bias=False,
            )
        else:
            self.unembed = Linear(
                in_dim=text_config.hidden_size,
                out_dim=text_config.vocab_size,
                dtype=config.dtype,
                device=device,
                has_bias=False,
            )

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        positions: TensorValue,
        return_n_logits: TensorValue,
        image_embeddings: TensorValue,
        image_indices: TensorValue,
        signal_buffers: list[BufferValue],
        kv_collections: dict[str, list[PagedCacheValues]],
        slot_idx: list[TensorValue],
        conv_pools: list[list[BufferValue]],
    ) -> tuple[TensorValue, ...]:
        """Runs the whole text model over one ragged batch."""
        if self.num_devices > 1:
            assert isinstance(self.embed, VocabParallelEmbedding)
            hs = self.embed(tokens, signal_buffers)
            row_offsets = ops.distributed_broadcast(
                input_row_offsets, signal_buffers
            )
            log_scaling = ops.distributed_broadcast(
                self._log_scaling(positions), signal_buffers
            )
        else:
            assert isinstance(self.embed, Embedding)
            hs = [self.embed(tokens)]
            row_offsets = [input_row_offsets]
            log_scaling = [self._log_scaling(positions)]
        if self.embed_norm_shards:
            hs = forward_sharded_layers(self.embed_norm_shards, hs)
        # Merged after the embedding norm, so the tower's rows skip it.
        hs = self._merge_images(
            hs, image_embeddings, image_indices, signal_buffers
        )

        num_devices = self.num_devices

        def inputs_for_layer(
            layer_idx: int, previous: list[TensorValue]
        ) -> list[SubgraphInput]:
            return [
                previous,
                kv_collections[self.layer_kv_keys[layer_idx]],
                row_offsets,
                log_scaling,
                [
                    conv_pools[rank][
                        layer_idx * len(ConvSite) : (layer_idx + 1)
                        * len(ConvSite)
                    ]
                    for rank in range(num_devices)
                ],
                slot_idx,
                ops.constant(
                    self.layer_cache_indices[layer_idx],
                    DType.uint32,
                    device=DeviceRef.CPU(),
                ),
                signal_buffers,
            ]

        hs = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            initial_hidden_states=hs,
            subgraph_layer_groups=(
                self.subgraph_groups if self.use_subgraphs else None
            ),
            name_for_subgraph=lambda group: self.subgraph_names[group],
            weight_prefix_for_layer=lambda layer_idx: f"layers.{layer_idx}.",
        )

        if self.num_devices > 1:
            return distributed_logits_postprocess(
                hs,
                row_offsets,
                return_n_logits,
                lm_head=self._distributed_lm_head,
                signal_buffers=signal_buffers,
                return_logits=self.return_logits,
                device=self.devices[0],
                norm_shards=self.norm_shards,
                return_hidden_states=self.return_hidden_states,
                logits_scaling=self.text_config.logits_mup_width_multiplier,
            )
        return logits_postprocess(
            hs[0],
            input_row_offsets,
            return_n_logits,
            self.norm,
            self._lm_head,
            self.return_logits,
            return_hidden_states=self.return_hidden_states,
            logits_scaling=self.text_config.logits_mup_width_multiplier,
        )

    def _log_scaling(self, positions: TensorValue) -> TensorValue:
        return log_scaling_tau(
            positions,
            alpha=self.text_config.log_scaling_alpha,
            n_floor=self.text_config.log_scaling_n_floor,
        )

    def _merge_images(
        self,
        hs: list[TensorValue],
        embeddings: TensorValue,
        indices: TensorValue,
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """Overwrites each placeholder row with its vision-tower row."""
        if self.num_devices > 1:
            embeddings_per_rank = ops.distributed_broadcast(
                embeddings, signal_buffers
            )
            indices_per_rank = ops.distributed_broadcast(
                indices, signal_buffers
            )
        else:
            embeddings_per_rank = [embeddings]
            indices_per_rank = [indices]
        return [
            merge_multimodal_embeddings(h, rank_embeddings, rank_indices)
            for h, rank_embeddings, rank_indices in zip(
                hs, embeddings_per_rank, indices_per_rank, strict=True
            )
        ]

    def _lm_head(self, h: TensorValue) -> TensorValue:
        return self._mask_padded_tail(self.unembed(h))

    def _distributed_lm_head(
        self, hs: list[TensorValue], signal_buffers: Sequence[BufferValue]
    ) -> list[TensorValue]:
        assert isinstance(self.unembed, ColumnParallelLinear)
        return [
            self._mask_padded_tail(logits)
            for logits in self.unembed(hs, signal_buffers)
        ]

    def _mask_padded_tail(self, logits: TensorValue) -> TensorValue:
        """Sends the untrained tail of the vocabulary to negative infinity."""
        if self._padded_tail_mask is None:
            return logits
        # Materialized row: concatenated stride-zero broadcasts fault on B200.
        return logits + ops.cast(
            ops.constant(
                self._padded_tail_mask, DType.float32, device=logits.device
            ),
            logits.dtype,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Graph input types, in the order :meth:`__call__` consumes them."""
        device = self.devices[0]
        signals = (
            Signals(self.devices).input_types() if self.num_devices > 1 else []
        )
        return (
            TensorType(DType.int64, shape=["total_seq_len"], device=device),
            TensorType(
                DType.uint32, shape=["input_row_offsets_len"], device=device
            ),
            TensorType(DType.uint32, shape=["total_seq_len"], device=device),
            TensorType(
                DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
            ),
            TensorType(
                self.dtype,
                shape=["total_image_tokens", self.text_config.hidden_size],
                device=device,
            ),
            TensorType(
                DType.int32, shape=["total_image_tokens"], device=device
            ),
            *signals,
            *self.kv_params.flattened_kv_inputs(),
            *(
                TensorType(
                    DType.uint32, shape=["batch_size"], device=slot_device
                )
                for slot_device in self.devices
            ),
            *self.conv_layout.buffer_types(self.devices),
        )

    def unflatten_kv_inputs(
        self, kv_inputs: Sequence[object]
    ) -> dict[str, list[PagedCacheValues]]:
        """Groups the flattened KV inputs by attention flavor, then by rank."""
        return kv_collections_by_key(
            self.kv_params.unflatten_kv_inputs(iter(kv_inputs))
        )

    def unpack_inputs(
        self, inputs: Sequence[Value[Any]]
    ) -> tuple[
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
        TensorValue,
        list[BufferValue],
        dict[str, list[PagedCacheValues]],
        list[TensorValue],
        list[list[BufferValue]],
    ]:
        """Splits a graph's positional inputs into the arguments of ``__call__``."""
        num_devices = self.num_devices
        pools_per_device = len(self.conv_layout.layers) * len(ConvSite)
        num_pools = pools_per_device * num_devices
        num_signals = num_devices if num_devices > 1 else 0
        (
            tokens,
            input_row_offsets,
            positions,
            return_n_logits,
            image_embeddings,
            image_indices,
            *rest,
        ) = inputs
        signals = rest[:num_signals]
        rest = rest[num_signals:]
        kv_inputs = rest[: -(num_devices + num_pools)]
        slot_idx = rest[-(num_devices + num_pools) : -num_pools]
        pools = [value.buffer for value in rest[-num_pools:]]
        return (
            tokens.tensor,
            input_row_offsets.tensor,
            positions.tensor,
            return_n_logits.tensor,
            image_embeddings.tensor,
            image_indices.tensor,
            [value.buffer for value in signals],
            self.unflatten_kv_inputs(kv_inputs),
            [value.tensor for value in slot_idx],
            [
                pools[rank * pools_per_device : (rank + 1) * pools_per_device]
                for rank in range(num_devices)
            ],
        )
