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
"""Chained Inkling MTP draft depths: one full decoder block per speculative step."""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.embedding import Embedding, VocabParallelEmbedding
from max.nn.kv_cache import (
    MHAKVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.transformer.distributed_transformer import forward_sharded_layers

from ..inkling.inkling import InklingDecoderLayer
from ..inkling.layers.attention import log_scaling_tau
from ..inkling.model_config import (
    GLOBAL_ATTENTION,
    LOCAL_ATTENTION,
    InklingConfig,
)
from ..inkling.state_cache import ConvSite, InklingConvStateLayout


class InklingMTPDepthLayer(Module):
    """One MTP depth: dual RMSNorm, 2H->H fuse, then a dense Inkling block."""

    def __init__(
        self,
        config: InklingConfig,
        *,
        depth_idx: int,
        kv_params: MHAKVCacheParams,
        is_local: bool,
        cache_layer_idx: int,
        kv_key: str,
    ) -> None:
        super().__init__()
        text_config = config.text_config
        devices = config.devices
        dtype = config.dtype
        self.num_devices = len(devices)
        self.cache_layer_idx = cache_layer_idx
        self.kv_key = kv_key
        self.hidden_size = text_config.hidden_size
        replicate = ShardingStrategy.replicate(self.num_devices)

        self.hidden_norm = RMSNorm(
            text_config.hidden_size, dtype, eps=text_config.rms_norm_eps
        )
        self.hidden_norm.sharding_strategy = replicate
        self.hidden_norm_shards = list(self.hidden_norm.shard(devices))

        self.embed_norm = RMSNorm(
            text_config.hidden_size, dtype, eps=text_config.rms_norm_eps
        )
        self.embed_norm.sharding_strategy = replicate
        self.embed_norm_shards = list(self.embed_norm.shard(devices))

        self.input_proj = Linear(
            in_dim=text_config.hidden_size * 2,
            out_dim=text_config.hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        self.input_proj.sharding_strategy = replicate
        self.input_proj_shards = list(self.input_proj.shard(devices))

        self.decoder_layer = InklingDecoderLayer(
            text_config=text_config,
            layer_idx=depth_idx,
            kv_params=kv_params,
            dtype=dtype,
            devices=devices,
            quant_config=None,
            is_local=is_local,
            force_dense_mlp=True,
        )

    def __call__(
        self,
        token_embeds: Sequence[TensorValue],
        hidden: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
        log_scaling: Sequence[TensorValue],
        conv_pools: Sequence[Sequence[BufferValue]],
        slot_idx: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        hidden_states_first: bool,
    ) -> list[TensorValue]:
        """Fuses ``hidden`` with token embeddings and runs the decoder block."""
        norm_h = forward_sharded_layers(self.hidden_norm_shards, hidden)
        norm_e = forward_sharded_layers(self.embed_norm_shards, token_embeds)
        fused = []
        for h, e in zip(norm_h, norm_e, strict=True):
            e = e.rebind(h.shape)
            fused.append(
                ops.concat([h, e], axis=-1)
                if hidden_states_first
                else ops.concat([e, h], axis=-1)
            )
        hs = forward_sharded_layers(self.input_proj_shards, fused)
        cache_idx = ops.constant(
            self.cache_layer_idx, DType.uint32, device=DeviceRef.CPU()
        )
        return self.decoder_layer(
            hs,
            kv_collections,
            input_row_offsets,
            log_scaling,
            conv_pools,
            slot_idx,
            cache_idx,
            signal_buffers,
        )


class InklingMultiTokenPredictor(Module):
    """Stacked MTP depths sharing the target's embedding table and LM head."""

    def __init__(
        self,
        config: InklingConfig,
        n_depths: int,
        kv_params: MultiKVCacheParams,
    ) -> None:
        super().__init__()
        assert config.mtp is not None
        text_config = config.text_config
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        self.n_depths = n_depths
        self.kv_params = kv_params
        self.hidden_states_first = config.mtp.hidden_states_first
        self.is_local = config.mtp.local_flags(n_depths)
        self.conv_layout = InklingConvStateLayout.from_local_flags(
            text_config, self.is_local, tp_size=self.num_devices
        )

        kv_by_key: dict[str, MHAKVCacheParams] = {}
        for key, params in kv_params.children.items():
            assert isinstance(params, MHAKVCacheParams)
            kv_by_key[key] = params

        layers = []
        counts = {GLOBAL_ATTENTION: 0, LOCAL_ATTENTION: 0}
        for depth_idx, is_local in enumerate(self.is_local):
            kv_key = LOCAL_ATTENTION if is_local else GLOBAL_ATTENTION
            layers.append(
                InklingMTPDepthLayer(
                    config,
                    depth_idx=depth_idx,
                    kv_params=kv_by_key[kv_key],
                    is_local=is_local,
                    cache_layer_idx=counts[kv_key],
                    kv_key=kv_key,
                )
            )
            counts[kv_key] += 1
        self.layers = LayerList(layers)
        # Aliased to the target's embedding table, never owned here.
        self.embed: Embedding | VocabParallelEmbedding | None = None
        self.backbone_embed_norm_shards: list[RMSNorm] = []

        self.chain_norm_shards: list[RMSNorm] = []
        if config.mtp.chain_hidden_post_norm:
            chain_norm = RMSNorm(
                text_config.hidden_size,
                config.dtype,
                eps=text_config.rms_norm_eps,
            )
            chain_norm.sharding_strategy = ShardingStrategy.replicate(
                self.num_devices
            )
            self.chain_norm = chain_norm
            self.chain_norm_shards = list(chain_norm.shard(config.devices))

    def embed_tokens(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """Token embeddings with the backbone embed_norm applied first."""
        embed = self.embed
        assert embed is not None
        if self.num_devices > 1:
            assert isinstance(embed, VocabParallelEmbedding)
            hs = embed(tokens, signal_buffers)
        else:
            assert isinstance(embed, Embedding)
            hs = [embed(tokens)]
        if self.backbone_embed_norm_shards:
            hs = forward_sharded_layers(self.backbone_embed_norm_shards, hs)
        return hs

    def log_scaling(self, positions: TensorValue) -> TensorValue:
        text_config = self.config.text_config
        return log_scaling_tau(
            positions,
            alpha=text_config.log_scaling_alpha,
            n_floor=text_config.log_scaling_n_floor,
        )

    def depth_conv_pools(
        self,
        conv_pools: Sequence[Sequence[BufferValue]],
        depth_idx: int,
    ) -> list[list[BufferValue]]:
        """Slices the four conv sites of one depth out of the draft pool list."""
        n_sites = len(ConvSite)
        start = depth_idx * n_sites
        return [list(rank[start : start + n_sites]) for rank in conv_pools]

    def forward_depth(
        self,
        depth_idx: int,
        token_embeds: Sequence[TensorValue],
        hidden: Sequence[TensorValue],
        kv_collections: dict[str, list[PagedCacheValues]],
        input_row_offsets: Sequence[TensorValue],
        positions: TensorValue,
        conv_pools: Sequence[Sequence[BufferValue]],
        slot_idx: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """Runs MTP depth ``depth_idx`` and optionally applies chain_norm."""
        depth = self.layers[depth_idx]
        assert isinstance(depth, InklingMTPDepthLayer)
        if self.num_devices > 1:
            log_scaling = ops.distributed_broadcast(
                self.log_scaling(positions), signal_buffers
            )
        else:
            log_scaling = [self.log_scaling(positions)]
        hs = depth(
            token_embeds,
            hidden,
            kv_collections[depth.kv_key],
            input_row_offsets,
            log_scaling,
            self.depth_conv_pools(conv_pools, depth_idx),
            slot_idx,
            signal_buffers,
            self.hidden_states_first,
        )
        if self.chain_norm_shards:
            hs = forward_sharded_layers(self.chain_norm_shards, hs)
        return hs

    def __call__(self, *args: object, **kwargs: object) -> list[TensorValue]:
        raise TypeError(
            "InklingMultiTokenPredictor is invoked per depth via forward_depth"
        )
