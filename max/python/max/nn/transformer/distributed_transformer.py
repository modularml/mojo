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

from collections.abc import Callable, Sequence
from typing import Protocol

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn.comm.allreduce import Allreduce

from ..embedding import VocabParallelEmbedding
from ..kv_cache import KVCacheParams, PagedCacheValues
from ..layer import LayerList, Module, Shardable, SubgraphInput
from ..linear import ColumnParallelLinear
from ..rotary_embedding import RotaryEmbedding
from .transformer import (
    ReturnHiddenStates,
    ReturnLogits,
    extract_hs,
    forward_sequential_layers,
    forward_sharded_layers,
)


# NOTE: This should eventually be deleted once Weight & Linear are refactored to assume
# distributed by default.
class ShardableCallable(Shardable, Protocol):
    def __call__(self, x: TensorValue) -> TensorValue: ...


def distributed_logits_postprocess(
    h: Sequence[TensorValue],
    input_row_offsets: Sequence[TensorValue],
    return_n_logits: TensorValue,
    lm_head: Callable[
        [list[TensorValue], Sequence[BufferValue]], Sequence[TensorValue]
    ],
    signal_buffers: Sequence[BufferValue],
    return_logits: ReturnLogits,
    device: DeviceRef,
    norm_shards: Sequence[Callable[[TensorValue], TensorValue]] | None = None,
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    logits_scaling: float = 1.0,
    logit_softcapping: float | None = None,
    capture_hidden_states: list[list[TensorValue]] | None = None,
) -> tuple[TensorValue, ...]:
    """Common logits postprocessing for multi-device sharded models.

    Handles last-token gathering, logits computation (VARIABLE/ALL/LAST_TOKEN),
    logits scaling, and hidden states return for models that use per-device
    sharded hidden states.

    Args:
        h: Per-device hidden states from the final transformer layer.
        input_row_offsets: Per-device row offsets for ragged batching.
        return_n_logits: Number of logits to return per sequence.
        lm_head: Language model head (takes per-device inputs + signal buffers).
        signal_buffers: Signal buffers for collective operations.
        return_logits: Which logits to return.
        device: Primary device for scalar ops (e.g. ops.range).
        norm_shards: Per-device normalization functions. When None, hidden
            states are passed directly to lm_head without normalization.
        return_hidden_states: Which hidden states to return.
        logits_scaling: Scaling factor for logits.
        capture_hidden_states: For ``SELECTED_LAYERS`` mode, the per-layer
            captured hidden states gathered during the layer loop. See
            :func:`extract_hs`.

    Returns:
        Tuple of (last_logits, [logits, offsets], [hidden_states]).
    """

    def _maybe_norm(
        tensors: Sequence[TensorValue],
    ) -> list[TensorValue]:
        if norm_shards is not None:
            return forward_sharded_layers(norm_shards, tensors)
        return list(tensors)

    # Gather last tokens per device and compute last-token logits.
    last_token_indices = [offsets[1:] - 1 for offsets in input_row_offsets]
    last_token_h = [
        ops.gather(h_device, indices, axis=0)
        for h_device, indices in zip(h, last_token_indices, strict=True)
    ]
    last_logits = ops.cast(
        lm_head(_maybe_norm(last_token_h), signal_buffers)[0],
        DType.float32,
    )

    logits = None
    offsets = None

    if return_logits == ReturnLogits.VARIABLE and h:
        last_indices = [
            ops.reshape(
                ops.unsqueeze(row_offset[1:], -1)
                - ops.range(
                    start=return_n_logits[0],
                    stop=0,
                    step=-1,
                    out_dim="return_n_logits_range",
                    dtype=DType.int64,
                    device=row_offset.device,
                ),
                shape=(-1,),
            )
            for row_offset in input_row_offsets
        ]

        variable_tokens = _maybe_norm(
            [
                ops.gather(h_device, indices, axis=0)
                for h_device, indices in zip(h, last_indices, strict=True)
            ]
        )
        logits = ops.cast(
            lm_head(variable_tokens, signal_buffers)[0], DType.float32
        )
        offsets = ops.range(
            0,
            TensorValue(last_indices[0].shape[0]) + return_n_logits[0],
            return_n_logits[0],
            out_dim="logit_offsets",
            dtype=DType.int64,
            device=device,
        )
    elif return_logits == ReturnLogits.ALL and h:
        logits = ops.cast(
            lm_head(_maybe_norm(h), signal_buffers)[0], DType.float32
        )
        offsets = input_row_offsets[0]

    if logits_scaling != 1.0:
        last_logits = last_logits / logits_scaling
        if logits is not None:
            logits = logits / logits_scaling

    if logit_softcapping is not None:
        # Softcap: cap * tanh(logits / cap), bounding logits to (-cap, cap).
        last_logits = ops.tanh(last_logits / logit_softcapping) * (
            logit_softcapping
        )
        if logits is not None:
            logits = ops.tanh(logits / logit_softcapping) * logit_softcapping

    ret_val: tuple[TensorValue, ...] = (last_logits,)
    if offsets is not None:
        assert logits is not None
        ret_val += (logits, offsets)

    ret_val += extract_hs(
        return_hidden_states=return_hidden_states,
        last_token_hs_distributed=last_token_h,
        all_hs_distributed=h,
        normalizer=norm_shards,
        capture_hidden_states=capture_hidden_states,
    )
    return ret_val


class DistributedLogitsPostprocessMixin:
    """Mixin providing logits postprocessing for multi-device sharded models.

    Requires: self.norm_shards, self.lm_head, self.return_logits, self.devices.
    Optional: self.return_hidden_states, self.logits_scaling.
    """

    norm_shards: Sequence[Callable[[TensorValue], TensorValue]]
    lm_head: Callable[
        [list[TensorValue], Sequence[BufferValue]], Sequence[TensorValue]
    ]
    return_logits: ReturnLogits
    devices: list[DeviceRef]
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    logits_scaling: float = 1.0
    logit_softcapping: float | None = None

    def _postprocess_logits(
        self,
        h: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        return_n_logits: TensorValue,
        signal_buffers: Sequence[BufferValue],
        capture_hidden_states: list[list[TensorValue]] | None = None,
    ) -> tuple[TensorValue, ...]:
        return distributed_logits_postprocess(
            h,
            input_row_offsets,
            return_n_logits,
            norm_shards=self.norm_shards,
            lm_head=self.lm_head,
            signal_buffers=signal_buffers,
            return_logits=self.return_logits,
            device=self.devices[0],
            return_hidden_states=self.return_hidden_states,
            logits_scaling=self.logits_scaling,
            logit_softcapping=self.logit_softcapping,
            capture_hidden_states=capture_hidden_states,
        )


class DistributedTransformerBlock(Module):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    def __init__(
        self,
        attention: Module,
        mlp: ShardableCallable,
        attention_norm: ShardableCallable,
        mlp_norm: ShardableCallable,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()

        self.self_attn = attention
        self.mlp = mlp
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.mlp_shards = mlp.shard(devices)

        # Shard the norm layers
        self.input_layernorm = attention_norm
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            len(devices)
        )
        self.input_layernorm_shards = attention_norm.shard(devices)

        self.post_attention_layernorm = mlp_norm
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_attention_layernorm_shards = mlp_norm.shard(devices)

        self.devices = devices

        self.allreduce = Allreduce(num_accelerators=len(devices))

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
    ) -> list[TensorValue]:
        # Apply input layer norm to each shard
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        attn_outs = self.self_attn(
            layer_idx,
            norm_xs,
            signal_buffers,
            kv_collections,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
        )

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        # Apply post attention layer norm to each shard
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )
        mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class DistributedTransformer(DistributedLogitsPostprocessMixin, Module):
    """Transformer model consisting for TransformerBlock layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[DistributedTransformerBlock],
        norm: ShardableCallable,
        output: ColumnParallelLinear,
        embedding: VocabParallelEmbedding,
        kv_params: KVCacheParams,
        devices: list[DeviceRef],
        rope: RotaryEmbedding,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        use_subgraphs: bool = False,
        subgraph_layer_groups: list[list[int]] | None = None,
        logits_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        # Shard the final norm layer
        self.norm.sharding_strategy = ShardingStrategy.replicate(len(devices))
        self.norm_shards = norm.shard(devices)
        self.lm_head = output
        self.embed_tokens = embedding
        self.kv_params = kv_params
        self.return_logits = return_logits
        self.devices = devices
        self.rope = rope
        self.use_subgraphs = use_subgraphs
        if subgraph_layer_groups is None:
            # If no subgraph layer groups are provided, assume that all layers
            # are in a single group.
            subgraph_layer_groups = [[i for i in range(len(layers))]]
        self.subgraph_layer_groups = subgraph_layer_groups
        self.logits_scaling = logits_scaling

    def __call__(
        self,
        tokens: TensorValueLike,
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        freqs_cis = [self.rope.freqs_cis.to(device) for device in self.devices]

        input_row_offsets_per_device = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        def inputs_for_layer(
            idx: int, h: list[TensorValue]
        ) -> list[SubgraphInput]:
            return [
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                signal_buffers,
                kv_collections,
                freqs_cis,
                input_row_offsets_per_device,
            ]

        h = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=(
                self.subgraph_layer_groups if self.use_subgraphs else None
            ),
            name_for_subgraph=lambda g: f"dist_transformer_block_{g}",
            initial_hidden_states=h,
        )

        return self._postprocess_logits(
            h, input_row_offsets_per_device, return_n_logits, signal_buffers
        )
