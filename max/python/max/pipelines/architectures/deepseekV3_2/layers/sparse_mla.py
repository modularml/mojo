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
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.attention.multi_latent_attention import (
    LatentAttentionWithRope,
    MLAPrefillMetadata,
)
from max.nn.attention.multi_latent_attention_fp8 import (
    LatentAttentionWithRopeFp8,
)
from max.nn.comm import Allreduce
from max.nn.kernels import (
    mla_decode_graph,
    mla_prefill_decode_graph,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig
from max.nn.quant_ops import quantized_matmul
from max.nn.rotary_embedding import RotaryEmbedding

from .indexer import Indexer

logger = logging.getLogger("max.pipelines")


# Head counts that route prefill through the combined prefill/decode op
# unconditionally (landed behavior; the sparse-prefill kernel handles 128 or
# any multiple of 8 in (0, 64]).  Other counts fall back to decode rather
# than tripping the kernel's comptime assert.
_SPARSE_PREFILL_SUPPORTED_HEADS_BF16 = (64, 128)
_SPARSE_PREFILL_SUPPORTED_HEADS_FP8 = (64, 128)
# GLM 5.2's TP-sharded counts (64 // {8, 4, 2}) route to the combined op over a
# bfloat16 OR float8_e4m3fn latent cache.  The combined op's prefill arm now
# takes the sparse-prefill kernel for both cache dtypes: mla_graph.mojo
# dispatches an FP8 latent cache to mla_sm100_prefill_sparse_fp8 read at unit
# scale (scale_block_size=0), mirroring the sparse-decode kernel's read of the
# scale-less FP8 latent cache.  So the absorbed sparse path -- not the dense
# unabsorbed FP8 prefill (whose extra Q/K/V requantization cost accuracy) --
# runs for FP8-cache prefill at these head counts.
_SPARSE_PREFILL_TP_SHARDED_HEADS = (8, 16, 32)


# Master gate for the sparse-MLA *prefill* kernel. When False, prefill is
# routed through the sparse *decode* kernel (the same fallback used for
# unsupported head counts) instead of the sparse-prefill kernel, so the
# combined-op wiring can merge while the prefill kernel is still being
# optimized.
_ENABLE_SPARSE_MLA_PREFILL_KERNEL = True

# Dedup the one-time "prefill kernel gated off" notice (guard runs per layer).
_WARNED_PREFILL_KERNEL_DISABLED: set[str] = set()


def _warn_prefill_kernel_disabled() -> None:
    """Log once (deduped across layers) that the prefill kernel is off."""
    if "logged" in _WARNED_PREFILL_KERNEL_DISABLED:
        return
    _WARNED_PREFILL_KERNEL_DISABLED.add("logged")
    logger.info(
        "Sparse MLA prefill kernel disabled "
        "(_ENABLE_SPARSE_MLA_PREFILL_KERNEL=False); routing prefill "
        "through the sparse decode kernel."
    )


def _sparse_prefill_head_count_supported(
    n_heads: int,
    supported_heads: tuple[int, ...],
    cache_dtype: DType,
) -> bool:
    """Whether prefill may route through the combined prefill/decode op."""
    if n_heads in supported_heads:
        return True
    return n_heads in _SPARSE_PREFILL_TP_SHARDED_HEADS and cache_dtype in (
        DType.bfloat16,
        DType.float8_e4m3fn,
    )


# Head counts already warned about; the guard runs per layer, so dedup the log.
_WARNED_FALLBACK_HEADS: set[int] = set()


class SparseLatentAttentionWithRopeFp8(LatentAttentionWithRopeFp8):
    """FP8 latent attention with optional sparse decode (logical KV positions; MOGG remaps)."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        quant_config: QuantConfig,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        buffer_size: int = 16384,
        graph_mode: str | None = None,
        norm_dtype: DType = DType.bfloat16,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        skip_topk: bool = False,
        indexer_rope_interleave: bool = False,
    ):
        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            quant_config=quant_config,
            devices=devices,
            linear_cls=linear_cls,
            scale=scale,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            buffer_size=buffer_size,
            graph_mode=graph_mode,
            norm_dtype=norm_dtype,
        )

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.skip_topk = skip_topk
        self.indexer_rope_interleave = indexer_rope_interleave

        # ``shared`` layers carry no indexer weights, and instead reuse the
        # previous full layer's top-k selection.
        self.indexer: Indexer | None
        if skip_topk:
            self.indexer = None
        else:
            self.indexer = Indexer(
                dim=hidden_size,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                index_topk=index_topk,
                q_lora_rank=q_lora_rank,
                rope_interleaved=indexer_rope_interleave,
                devices=self.devices,
                quant_config=self.quant_config,
                k_norm_dtype=norm_dtype,
            )

    @LatentAttentionWithRopeFp8.sharding_strategy.setter  # type: ignore[attr-defined]
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Extends the base setter so the indexer participates in sharding."""
        LatentAttentionWithRopeFp8.sharding_strategy.fset(self, strategy)  # type: ignore[attr-defined]
        if self.indexer is None:
            return
        if strategy.is_replicate or strategy.is_tensor_parallel:
            rep = ShardingStrategy.replicate(strategy.num_devices)
            for linear in (
                self.indexer.wq_b,
                self.indexer.wk,
                self.indexer.weights_proj,
            ):
                linear.weight.sharding_strategy = rep
                if linear.weight_scale is not None:
                    linear.weight_scale.sharding_strategy = rep
                if linear.input_scale is not None:
                    linear.input_scale.sharding_strategy = rep
            self.indexer.k_norm.sharding_strategy = rep

    def _mla_impl(
        self,
        xq: TensorValue,
        kv: TensorValue,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
        freqs_cis: TensorValue,
        kv_a_proj_layernorm: TensorValue,
        _mla_prefill_metadata: MLAPrefillMetadata | None = None,
        *,
        sparse_indices: TensorValue | None = None,
        sparse_topk_lengths: TensorValue | None = None,
        sparse_attn_sink: TensorValue | None = None,
        sparse_indices_stride: int | None = None,
        index_share: bool = False,
    ) -> TensorValue:
        attn_kwargs: dict[str, Any] = {
            "q": xq,
            "kv": kv,
            "input_row_offsets": input_row_offsets,
            "freqs_cis": freqs_cis,
            "kv_norm_gamma": kv_a_proj_layernorm,
            "kv_params": self.kv_params,
            "kv_collection": kv_collection,
            "layer_idx": layer_idx,
            "epsilon": 1e-6,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": self.scale,
            "v_head_dim": self.v_head_dim,
            "quant_config": self.quant_config,
            "scale_granularity_override": self._b_scale_granularity,
        }

        w_k, w_k_scale = self.w_k
        w_uk, w_uk_scale = self.w_uk
        w_uv, w_uv_scale = self.w_uv

        effective_graph_mode = self.graph_mode
        if (
            effective_graph_mode != "decode"
            and not _ENABLE_SPARSE_MLA_PREFILL_KERNEL
        ):
            _warn_prefill_kernel_disabled()
            effective_graph_mode = "decode"
        elif (
            effective_graph_mode != "decode"
            and not _sparse_prefill_head_count_supported(
                self.n_heads,
                _SPARSE_PREFILL_SUPPORTED_HEADS_FP8,
                self.kv_params.dtype,
            )
        ):
            if self.n_heads not in _WARNED_FALLBACK_HEADS:
                _WARNED_FALLBACK_HEADS.add(self.n_heads)
                logger.warning(
                    "Sparse MLA prefill does not support %d query heads "
                    "(supported: %s); falling back to the slower decode path "
                    "for prefill. This usually means tensor-parallel attention "
                    "sharded the head count below a supported value.",
                    self.n_heads,
                    _SPARSE_PREFILL_SUPPORTED_HEADS_FP8,
                )
            effective_graph_mode = "decode"

        if effective_graph_mode in ["prefill", "auto"]:
            if _mla_prefill_metadata is None:
                mla_prefill_metadata = self.create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            else:
                mla_prefill_metadata = _mla_prefill_metadata

            attn_kwargs["buffer_row_offsets"] = (
                mla_prefill_metadata.buffer_row_offsets
            )
            attn_kwargs["cache_offsets"] = mla_prefill_metadata.cache_offsets
            attn_kwargs["buffer_length"] = (
                mla_prefill_metadata.buffer_lengths.to(DeviceRef.CPU())
            )
            attn_kwargs["w_k"] = w_k
            attn_kwargs["w_k_scale"] = w_k_scale
            attn_kwargs["w_uv"] = w_uv
            attn_kwargs["w_uv_scale"] = w_uv_scale

        if self.graph_mode in ["prefill", "decode", "auto"]:
            attn_kwargs["w_uk"] = w_uk
            attn_kwargs["w_uk_scale"] = w_uk_scale
            attn_kwargs["w_uv"] = w_uv
            attn_kwargs["w_uv_scale"] = w_uv_scale
            assert kv_collection.attention_dispatch_metadata is not None
            attn_kwargs["scalar_args"] = (
                kv_collection.attention_dispatch_metadata
            )
            assert kv_collection.mla_num_partitions is not None
            attn_kwargs["num_partitions_scalar"] = (
                kv_collection.mla_num_partitions
            )

        sparse_kw: dict[str, Any] = {}
        if sparse_indices is not None:
            sparse_kw = {
                "sparse_indices": sparse_indices,
                "sparse_topk_lengths": sparse_topk_lengths,
                "sparse_attn_sink": sparse_attn_sink,
                "sparse_indices_stride": sparse_indices_stride,
                # Read-once shared-KV fold (KERN-3141); only True when the
                # caller has a shared top-k across folded MTP positions.
                "index_share": index_share,
            }

        if effective_graph_mode == "decode":
            result = mla_decode_graph(**attn_kwargs, **sparse_kw)
        else:
            # TODO(KERN-3198): "prefill" uses the combined op because
            # mla_prefill_graph doesn't support sparse args yet.
            result = mla_prefill_decode_graph(**attn_kwargs, **sparse_kw)

        return result.reshape((-1, self.n_heads * self.v_head_dim))

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
        prev_topk_indices: TensorValue | None = None,
        reuse_prev_topk: bool = False,
    ) -> tuple[TensorValue, TensorValue]:
        wqkv, wqkv_scale = self.wqkv
        qkv = quantized_matmul(
            x=x,
            weight=wqkv,
            weight_scale=wqkv_scale,
            input_scale=None,
            quant_config=self.quant_config,
        )

        q_a_out, kv = ops.split(
            qkv, [self.q_lora_rank, self.cache_head_dim], axis=1
        )

        q_a_normed = self.q_a_layernorm(q_a_out)

        xq = quantized_matmul(
            x=q_a_normed,
            weight=self.q_b_proj,
            weight_scale=self.q_b_proj_scale,
            input_scale=None,
            quant_config=self.quant_config,
        )

        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))

        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)

        if self.indexer is not None and not reuse_prev_topk:
            # ``full`` layer: run the lightning indexer and select top-k keys.
            topk_indices = self.indexer(
                x,
                q_a_normed,
                freqs_cis,
                input_row_offsets,
                indexer_kv_collection,
                layer_idx,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
            )
        else:
            # ``shared`` layer: reuse the previous full layer's top-k selection
            # (cross-layer index sharing). The indices are sequence positions,
            # so they are valid for this layer's own MLA cache.
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared (skip_topk) sparse attention layers require top-k "
                    "indices from a previous full indexer layer."
                )
            topk_indices = prev_topk_indices

        sparse_topk_lengths = ops.broadcast_to(
            ops.constant(
                self.index_topk,
                dtype=DType.int32,
                device=xq.device,
            ),
            (xq.shape[0],),
        )
        sparse_attn_sink = ops.broadcast_to(
            ops.constant(-1.0e38, dtype=DType.float32, device=xq.device),
            (self.n_heads,),
        )

        # Read-once shared-index MTP fold (KERN-3141). Enable the fold only for
        # a *full* indexer layer (``skip_topk`` is False) that reuses a prior
        # selection: there the reused list is the single shared MTP top-k
        # (``index_share_for_mtp_iteration``), so every folded q position
        # attends one gathered pass. ``skip_topk`` (cross-layer) reuse keeps a
        # per-position list, so it must stay on the unfolded path. The decode
        # dispatch additionally self-gates on the fold shape (q_len in [2, 8]);
        # default-off remains the production behavior (see Phase 8: index_share
        # is only True at q_len=1 today, where the fold does not fire).
        index_share = reuse_prev_topk and not self.skip_topk
        attn_out = self._mla_impl(
            xq,
            kv,
            kv_collection,
            layer_idx,
            input_row_offsets,
            freqs_cis,
            self.kv_a_proj_layernorm,
            mla_prefill_metadata,
            sparse_indices=topk_indices,
            sparse_topk_lengths=sparse_topk_lengths,
            sparse_attn_sink=sparse_attn_sink,
            sparse_indices_stride=self.index_topk,
            index_share=index_share,
        )

        return self.o_proj(attn_out), topk_indices

    def shard(  # type: ignore[override]
        self, devices: Iterable[DeviceRef]
    ) -> list[SparseLatentAttentionWithRopeFp8]:
        """Creates sharded views of this module across devices.

        Supports `replicate` or `tensor_parallel` sharding strategies.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "SparseLatentAttentionWithRopeFp8 cannot be sharded because no "
                "sharding strategy was provided."
            )
        if not (
            self.sharding_strategy.is_replicate
            or self.sharding_strategy.is_tensor_parallel
        ):
            raise ValueError(
                "Only replicate or tensor parallel sharding strategies are "
                "supported for SparseLatentAttentionWithRopeFp8"
            )

        is_tp = self.sharding_strategy.is_tensor_parallel
        local_heads = (
            self.n_heads // self.sharding_strategy.num_devices
            if is_tp
            else self.n_heads
        )

        q_a_proj_shards = self.q_a_proj.shard(devices)
        q_a_proj_scale_shards = self.q_a_proj_scale.shard(devices)
        q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(devices)
        q_b_proj_shards = self.q_b_proj.shard(devices)
        q_b_proj_scale_shards = self.q_b_proj_scale.shard(devices)

        kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
        kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
        kv_a_proj_with_mqa_scale_shards = self.kv_a_proj_with_mqa_scale.shard(
            devices
        )
        kv_b_proj_shards = self.kv_b_proj.shard(devices)
        kv_b_proj_scale_shards = self.kv_b_proj_scale.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)

        if self.indexer is not None:
            indexer_wq_b_shards = self.indexer.wq_b.shard(devices)
            indexer_wk_shards = self.indexer.wk.shard(devices)
            indexer_weights_proj_shards = self.indexer.weights_proj.shard(
                devices
            )
            indexer_k_norm_shards = self.indexer.k_norm.shard(devices)

        replicas: list[SparseLatentAttentionWithRopeFp8] = []
        for shard_idx, device in enumerate(devices):
            replica = SparseLatentAttentionWithRopeFp8(
                rope=self.rope,
                num_attention_heads=local_heads,
                num_key_value_heads=self.num_key_value_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                quant_config=self.quant_config,
                devices=[device],
                graph_mode=self.graph_mode,
                linear_cls=self.linear_cls,
                scale=self._scale,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                buffer_size=self.BUFFER_TOK_SIZE,
                norm_dtype=self.norm_dtype,
                index_n_heads=self.index_n_heads,
                index_head_dim=self.index_head_dim,
                index_topk=self.index_topk,
                skip_topk=self.skip_topk,
                indexer_rope_interleave=self.indexer_rope_interleave,
            )

            replica.q_a_proj = q_a_proj_shards[shard_idx]
            replica.q_a_proj_scale = q_a_proj_scale_shards[shard_idx]
            replica.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                shard_idx
            ]
            replica.q_b_proj = q_b_proj_shards[shard_idx]
            replica.q_b_proj_scale = q_b_proj_scale_shards[shard_idx]

            replica.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[shard_idx]
            replica.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[shard_idx]
            replica.kv_a_proj_with_mqa_scale = kv_a_proj_with_mqa_scale_shards[
                shard_idx
            ]
            replica.kv_b_proj = kv_b_proj_shards[shard_idx]
            replica.kv_b_proj_scale = kv_b_proj_scale_shards[shard_idx]
            replica.o_proj = o_proj_shards[shard_idx]

            if self.indexer is not None:
                assert replica.indexer is not None
                replica.indexer.wq_b = indexer_wq_b_shards[shard_idx]
                replica.indexer.wk = indexer_wk_shards[shard_idx]
                replica.indexer.weights_proj = indexer_weights_proj_shards[
                    shard_idx
                ]
                replica.indexer.k_norm = indexer_k_norm_shards[shard_idx]

            replicas.append(replica)

        return replicas


class DataParallelSparseLatentAttentionWithRopeFp8(
    SparseLatentAttentionWithRopeFp8
):
    """Data-parallel sparse FP8 MLA: per-device optional sparse index tensors."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)
        self.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def create_mla_prefill_metadata(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[MLAPrefillMetadata]:
        """Creates per-device FP8 MLA prefill metadata for data-parallel execution."""
        multi_mla_prefill_metadata: list[MLAPrefillMetadata] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_prefill_metadata.append(
                super().create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            )

        return multi_mla_prefill_metadata

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        indexer_kv_collections: Sequence[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_prefill_metadata: list[MLAPrefillMetadata] | None = None,
        prev_topk_indices: Sequence[TensorValue] | None = None,
        reuse_prev_topk: bool = False,
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(indexer_kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, indexer_kv_collections, freqs_cis, and "
                f"input_row_offsets must all have length equal to number of devices "
                f"({n})"
            )

        outs: list[TensorValue] = []
        topk_indices: list[TensorValue] = []
        for i in range(n):
            # An empty list means no previous full layer has produced a
            # selection yet (the first layer); treat it as ``None``.
            prev_topk_i = prev_topk_indices[i] if prev_topk_indices else None
            if xs[i].shape[0] == 0:
                outs.append(xs[i])
                # No tokens on this device: carry the (empty) top-k forward.
                topk_indices.append(
                    prev_topk_i
                    if prev_topk_i is not None
                    else ops.broadcast_to(
                        ops.constant(0, dtype=DType.int32, device=xs[i].device),
                        (xs[i].shape[0], self.index_topk),
                    )
                )
                continue

            mla_prefill_metadata_i: MLAPrefillMetadata | None
            if (
                mla_prefill_metadata is not None
                and len(mla_prefill_metadata) == n
            ):
                mla_prefill_metadata_i = mla_prefill_metadata[i]
            else:
                assert (
                    mla_prefill_metadata is None
                    or len(mla_prefill_metadata) == 0
                )
                mla_prefill_metadata_i = None

            out_i, topk_i = self.list_of_attentions[i](
                layer_idx=layer_idx,
                x=xs[i],
                kv_collection=kv_collections[i],
                indexer_kv_collection=indexer_kv_collections[i],
                freqs_cis=freqs_cis[i],
                input_row_offsets=input_row_offsets[i],
                mla_prefill_metadata=mla_prefill_metadata_i,
                prev_topk_indices=prev_topk_i,
                reuse_prev_topk=reuse_prev_topk,
            )
            outs.append(out_i)
            topk_indices.append(topk_i)
        return outs, topk_indices


class TensorParallelSparseLatentAttentionWithRopeFp8(
    SparseLatentAttentionWithRopeFp8
):
    """Tensor-parallel sparse FP8 MLA."""

    def __init__(self, *, skip_allreduce: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)
        self.skip_allreduce = skip_allreduce
        self.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
        self.allreduce = Allreduce(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def create_mla_prefill_metadata(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[MLAPrefillMetadata]:
        """Creates per-device FP8 MLA prefill metadata for tensor-parallel execution."""
        multi_mla_prefill_metadata: list[MLAPrefillMetadata] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_prefill_metadata.append(
                super().create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            )

        return multi_mla_prefill_metadata

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        indexer_kv_collections: Sequence[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_prefill_metadata: list[MLAPrefillMetadata] | None = None,
        prev_topk_indices: Sequence[TensorValue] | None = None,
        reuse_prev_topk: bool = False,
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(indexer_kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, indexer_kv_collections, freqs_cis, and "
                f"input_row_offsets must all have length equal to number of devices "
                f"({n})"
            )

        outs: list[TensorValue] = []
        topk_indices: list[TensorValue] = []
        for i in range(n):
            mla_prefill_metadata_i: MLAPrefillMetadata | None
            if (
                mla_prefill_metadata is not None
                and len(mla_prefill_metadata) == n
            ):
                mla_prefill_metadata_i = mla_prefill_metadata[i]
            else:
                mla_prefill_metadata_i = None

            out_i, topk_i = self.list_of_attentions[i](
                layer_idx=layer_idx,
                x=xs[i],
                kv_collection=kv_collections[i],
                indexer_kv_collection=indexer_kv_collections[i],
                freqs_cis=freqs_cis[i],
                input_row_offsets=input_row_offsets[i],
                mla_prefill_metadata=mla_prefill_metadata_i,
                # An empty list means no previous full layer has produced a
                # selection yet (the first layer); treat it as ``None``.
                prev_topk_indices=(
                    prev_topk_indices[i] if prev_topk_indices else None
                ),
                reuse_prev_topk=reuse_prev_topk,
            )
            outs.append(out_i)
            topk_indices.append(topk_i)

        if self.skip_allreduce:
            return outs, topk_indices

        return (
            self.allreduce(inputs=outs, signal_buffers=signal_buffers),
            topk_indices,
        )


class SparseLatentAttentionWithRope(LatentAttentionWithRope):
    """BF16 latent attention with optional sparse decode (logical KV positions; MOGG remaps)."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        dtype: DType,
        indexer_quant_config: QuantConfig,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        buffer_size: int = 16384,
        graph_mode: str | None = None,
        norm_dtype: DType = DType.bfloat16,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        skip_topk: bool = False,
        indexer_rope_interleave: bool = False,
    ):
        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            dtype=dtype,
            devices=devices,
            linear_cls=linear_cls,
            scale=scale,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            buffer_size=buffer_size,
            graph_mode=graph_mode,
            norm_dtype=norm_dtype,
        )
        self.indexer_quant_config = indexer_quant_config
        self.skip_topk = skip_topk
        self.indexer_rope_interleave = indexer_rope_interleave
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        if skip_topk:
            self.indexer = None
        else:
            self.indexer = Indexer(
                dim=hidden_size,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                index_topk=index_topk,
                q_lora_rank=q_lora_rank,
                rope_interleaved=indexer_rope_interleave,
                devices=self.devices,
                quant_config=indexer_quant_config,
                k_norm_dtype=norm_dtype,
            )

    @LatentAttentionWithRope.sharding_strategy.setter  # type: ignore[attr-defined]
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Extends the base setter so the indexer participates in sharding."""
        LatentAttentionWithRope.sharding_strategy.fset(self, strategy)  # type: ignore[attr-defined]
        if self.indexer is None:
            return
        if strategy.is_replicate or strategy.is_tensor_parallel:
            rep = ShardingStrategy.replicate(strategy.num_devices)
            for linear in (
                self.indexer.wq_b,
                self.indexer.wk,
                self.indexer.weights_proj,
            ):
                linear.weight.sharding_strategy = rep
                if linear.weight_scale is not None:
                    linear.weight_scale.sharding_strategy = rep
                if linear.input_scale is not None:
                    linear.input_scale.sharding_strategy = rep
            self.indexer.k_norm.sharding_strategy = rep

    def _mla_impl(
        self,
        xq: TensorValue,
        kv: TensorValue,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
        freqs_cis: TensorValue,
        kv_norm_gamma: TensorValue,
        _mla_prefill_metadata: MLAPrefillMetadata | None = None,
        epsilon: float = 1e-6,
    ) -> TensorValue:
        return self._mla_impl_sparse(
            xq,
            kv,
            kv_collection,
            layer_idx,
            input_row_offsets,
            freqs_cis,
            kv_norm_gamma,
            _mla_prefill_metadata,
            epsilon,
        )

    def _mla_impl_sparse(
        self,
        xq: TensorValue,
        kv: TensorValue,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
        freqs_cis: TensorValue,
        kv_norm_gamma: TensorValue,
        _mla_prefill_metadata: MLAPrefillMetadata | None = None,
        epsilon: float = 1e-6,
        *,
        sparse_indices: TensorValue | None = None,
        sparse_topk_lengths: TensorValue | None = None,
        sparse_attn_sink: TensorValue | None = None,
        sparse_indices_stride: int | None = None,
    ) -> TensorValue:
        attn_kwargs: dict[str, Any] = {
            "q": xq,
            "kv": kv,
            "input_row_offsets": input_row_offsets,
            "freqs_cis": freqs_cis,
            "kv_norm_gamma": kv_norm_gamma,
            "kv_params": self.kv_params,
            "kv_collection": kv_collection,
            "layer_idx": layer_idx,
            "epsilon": epsilon,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": self.scale,
            "v_head_dim": self.v_head_dim,
        }

        effective_graph_mode = self.graph_mode
        if (
            effective_graph_mode != "decode"
            and not _ENABLE_SPARSE_MLA_PREFILL_KERNEL
        ):
            _warn_prefill_kernel_disabled()
            effective_graph_mode = "decode"
        elif (
            effective_graph_mode != "decode"
            and not _sparse_prefill_head_count_supported(
                self.n_heads,
                _SPARSE_PREFILL_SUPPORTED_HEADS_BF16,
                self.kv_params.dtype,
            )
        ):
            if self.n_heads not in _WARNED_FALLBACK_HEADS:
                _WARNED_FALLBACK_HEADS.add(self.n_heads)
                logger.warning(
                    "Sparse MLA prefill does not support %d query heads "
                    "(supported: %s); falling back to the slower decode path "
                    "for prefill. This usually means tensor-parallel attention "
                    "sharded the head count below a supported value.",
                    self.n_heads,
                    _SPARSE_PREFILL_SUPPORTED_HEADS_BF16,
                )
            effective_graph_mode = "decode"

        if effective_graph_mode in ["prefill", "auto"]:
            if _mla_prefill_metadata is None:
                mla_prefill_metadata = self.create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            else:
                mla_prefill_metadata = _mla_prefill_metadata

            attn_kwargs["buffer_row_offsets"] = (
                mla_prefill_metadata.buffer_row_offsets
            )
            attn_kwargs["cache_offsets"] = mla_prefill_metadata.cache_offsets
            attn_kwargs["buffer_length"] = (
                mla_prefill_metadata.buffer_lengths.to(DeviceRef.CPU())
            )
            attn_kwargs["w_k"] = self.w_k
            attn_kwargs["w_uv"] = self.w_uv

        if self.graph_mode in ["prefill", "decode", "auto"]:
            attn_kwargs["w_uk"] = self.w_uk
            attn_kwargs["w_uv"] = self.w_uv
            assert kv_collection.attention_dispatch_metadata is not None
            attn_kwargs["scalar_args"] = (
                kv_collection.attention_dispatch_metadata
            )
            assert kv_collection.mla_num_partitions is not None
            attn_kwargs["num_partitions_scalar"] = (
                kv_collection.mla_num_partitions
            )

        sparse_kw: dict[str, Any] = {}
        if sparse_indices is not None:
            sparse_kw = {
                "sparse_indices": sparse_indices,
                "sparse_topk_lengths": sparse_topk_lengths,
                "sparse_attn_sink": sparse_attn_sink,
                "sparse_indices_stride": sparse_indices_stride,
            }

        if effective_graph_mode == "decode":
            result = mla_decode_graph(**attn_kwargs, **sparse_kw)
        else:
            # TODO(KERN-3198): "prefill" uses the combined op because
            # mla_prefill_graph doesn't support sparse args yet.
            result = mla_prefill_decode_graph(**attn_kwargs, **sparse_kw)

        return result.reshape((-1, self.n_heads * self.v_head_dim))

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        indexer_kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
        prev_topk_indices: TensorValue | None = None,
        reuse_prev_topk: bool = False,
    ) -> tuple[TensorValue, TensorValue]:
        q_lora_rank = self.q_lora_rank
        if q_lora_rank is None:
            raise ValueError(
                "q_lora_rank is required for SparseLatentAttentionWithRope"
            )

        qkv = x @ self.wqkv.T
        q_a_out, kv = ops.split(qkv, [q_lora_rank, self.cache_head_dim], axis=1)

        q_a_normed = self.q_a_layernorm(q_a_out)
        xq = q_a_normed @ self.q_b_proj.T
        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))

        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)

        if self.indexer is not None and not reuse_prev_topk:
            topk_indices = self.indexer(
                x,
                q_a_normed,
                freqs_cis,
                input_row_offsets,
                indexer_kv_collection,
                layer_idx,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared (skip_topk) sparse attention layers require top-k "
                    "indices from a previous full indexer layer."
                )
            topk_indices = prev_topk_indices

        sparse_topk_lengths = ops.broadcast_to(
            ops.constant(
                self.index_topk,
                dtype=DType.int32,
                device=xq.device,
            ),
            (xq.shape[0],),
        )
        sparse_attn_sink = ops.broadcast_to(
            ops.constant(-1.0e38, dtype=DType.float32, device=xq.device),
            (self.n_heads,),
        )

        attn_out = self._mla_impl_sparse(
            xq,
            kv,
            kv_collection,
            layer_idx,
            input_row_offsets,
            freqs_cis,
            self.kv_a_proj_layernorm,
            mla_prefill_metadata,
            sparse_indices=topk_indices,
            sparse_topk_lengths=sparse_topk_lengths,
            sparse_attn_sink=sparse_attn_sink,
            sparse_indices_stride=self.index_topk,
        )

        return self.o_proj(attn_out), topk_indices

    def shard(  # type: ignore[override]
        self, devices: Iterable[DeviceRef]
    ) -> list[SparseLatentAttentionWithRope]:
        """Creates sharded views of this module across devices."""
        if not self.sharding_strategy:
            raise ValueError(
                "SparseLatentAttentionWithRope cannot be sharded because no "
                "sharding strategy was provided."
            )
        if not (
            self.sharding_strategy.is_replicate
            or self.sharding_strategy.is_tensor_parallel
        ):
            raise ValueError(
                "Only replicate or tensor parallel sharding strategies are "
                "supported for SparseLatentAttentionWithRope"
            )

        is_tp = self.sharding_strategy.is_tensor_parallel
        local_heads = (
            self.n_heads // self.sharding_strategy.num_devices
            if is_tp
            else self.n_heads
        )

        q_lora_rank = self.q_lora_rank
        if q_lora_rank is None:
            raise ValueError(
                "q_lora_rank is required for SparseLatentAttentionWithRope"
            )

        q_a_proj_shards = self.q_a_proj.shard(devices)
        q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(devices)
        q_b_proj_shards = self.q_b_proj.shard(devices)
        kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
        kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
        kv_b_proj_shards = self.kv_b_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)

        indexer_wq_b_shards = None
        indexer_wk_shards = None
        indexer_weights_proj_shards = None
        indexer_k_norm_shards = None
        if self.indexer is not None:
            indexer_wq_b_shards = self.indexer.wq_b.shard(devices)
            indexer_wk_shards = self.indexer.wk.shard(devices)
            indexer_weights_proj_shards = self.indexer.weights_proj.shard(
                devices
            )
            indexer_k_norm_shards = self.indexer.k_norm.shard(devices)

        replicas: list[SparseLatentAttentionWithRope] = []
        for shard_idx, device in enumerate(devices):
            replica = SparseLatentAttentionWithRope(
                rope=self.rope,
                num_attention_heads=local_heads,
                num_key_value_heads=self.num_key_value_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                dtype=self.dtype,
                indexer_quant_config=self.indexer_quant_config,
                devices=[device],
                graph_mode=self.graph_mode,
                linear_cls=self.linear_cls,
                scale=self._scale,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                buffer_size=self.BUFFER_TOK_SIZE,
                norm_dtype=self.norm_dtype,
                index_n_heads=self.index_n_heads,
                index_head_dim=self.index_head_dim,
                index_topk=self.index_topk,
                skip_topk=self.skip_topk,
                indexer_rope_interleave=self.indexer_rope_interleave,
            )

            replica.q_a_proj = q_a_proj_shards[shard_idx]
            replica.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                shard_idx
            ]
            replica.q_b_proj = q_b_proj_shards[shard_idx]
            replica.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[shard_idx]
            replica.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[shard_idx]
            replica.kv_b_proj = kv_b_proj_shards[shard_idx]
            replica.o_proj = o_proj_shards[shard_idx]

            if self.indexer is not None:
                assert indexer_wq_b_shards is not None
                assert indexer_wk_shards is not None
                assert indexer_weights_proj_shards is not None
                assert indexer_k_norm_shards is not None
                assert replica.indexer is not None
                replica.indexer.wq_b = indexer_wq_b_shards[shard_idx]
                replica.indexer.wk = indexer_wk_shards[shard_idx]
                replica.indexer.weights_proj = indexer_weights_proj_shards[
                    shard_idx
                ]
                replica.indexer.k_norm = indexer_k_norm_shards[shard_idx]

            replicas.append(replica)

        return replicas


class DataParallelSparseLatentAttentionWithRope(SparseLatentAttentionWithRope):
    """Data-parallel sparse BF16 MLA: per-device optional sparse index tensors."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)
        self.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def create_mla_prefill_metadata(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[MLAPrefillMetadata]:
        multi_mla_prefill_metadata: list[MLAPrefillMetadata] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_prefill_metadata.append(
                super().create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            )

        return multi_mla_prefill_metadata

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        indexer_kv_collections: Sequence[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_prefill_metadata: list[MLAPrefillMetadata] | None = None,
        prev_topk_indices: list[TensorValue] | None = None,
        reuse_prev_topk: bool = False,
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        del signal_buffers
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(indexer_kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, indexer_kv_collections, freqs_cis, and "
                f"input_row_offsets must all have length equal to number of devices "
                f"({n})"
            )

        outs: list[TensorValue] = []
        topk_outs: list[TensorValue] = []
        for i in range(n):
            prev_topk_i = prev_topk_indices[i] if prev_topk_indices else None
            if xs[i].shape[0] == 0:
                outs.append(xs[i])
                topk_outs.append(
                    prev_topk_i
                    if prev_topk_i is not None
                    else ops.broadcast_to(
                        ops.constant(0, dtype=DType.int32, device=xs[i].device),
                        (xs[i].shape[0], self.index_topk),
                    )
                )
                continue

            mla_prefill_metadata_i: MLAPrefillMetadata | None
            if (
                mla_prefill_metadata is not None
                and len(mla_prefill_metadata) == n
            ):
                mla_prefill_metadata_i = mla_prefill_metadata[i]
            else:
                assert (
                    mla_prefill_metadata is None
                    or len(mla_prefill_metadata) == 0
                )
                mla_prefill_metadata_i = None

            attn_out, topk_indices = self.list_of_attentions[i](
                layer_idx=layer_idx,
                x=xs[i],
                kv_collection=kv_collections[i],
                indexer_kv_collection=indexer_kv_collections[i],
                freqs_cis=freqs_cis[i],
                input_row_offsets=input_row_offsets[i],
                mla_prefill_metadata=mla_prefill_metadata_i,
                prev_topk_indices=prev_topk_i,
                reuse_prev_topk=reuse_prev_topk,
            )
            outs.append(attn_out)
            topk_outs.append(topk_indices)
        return outs, topk_outs


class TensorParallelSparseLatentAttentionWithRope(
    SparseLatentAttentionWithRope
):
    """Tensor-parallel sparse BF16 MLA."""

    def __init__(self, *, skip_allreduce: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)
        self.skip_allreduce = skip_allreduce
        self.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
        self.allreduce = Allreduce(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def create_mla_prefill_metadata(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[MLAPrefillMetadata]:
        multi_mla_prefill_metadata: list[MLAPrefillMetadata] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_prefill_metadata.append(
                super().create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            )

        return multi_mla_prefill_metadata

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        indexer_kv_collections: Sequence[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_prefill_metadata: list[MLAPrefillMetadata] | None = None,
        prev_topk_indices: list[TensorValue] | None = None,
        reuse_prev_topk: bool = False,
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(indexer_kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, indexer_kv_collections, freqs_cis, and "
                f"input_row_offsets must all have length equal to number of devices "
                f"({n})"
            )

        outs: list[TensorValue] = []
        topk_outs: list[TensorValue] = []
        for i in range(n):
            mla_prefill_metadata_i: MLAPrefillMetadata | None
            if (
                mla_prefill_metadata is not None
                and len(mla_prefill_metadata) == n
            ):
                mla_prefill_metadata_i = mla_prefill_metadata[i]
            else:
                mla_prefill_metadata_i = None

            prev_topk_i = prev_topk_indices[i] if prev_topk_indices else None
            attn_out, topk_indices = self.list_of_attentions[i](
                layer_idx=layer_idx,
                x=xs[i],
                kv_collection=kv_collections[i],
                indexer_kv_collection=indexer_kv_collections[i],
                freqs_cis=freqs_cis[i],
                input_row_offsets=input_row_offsets[i],
                mla_prefill_metadata=mla_prefill_metadata_i,
                prev_topk_indices=prev_topk_i,
                reuse_prev_topk=reuse_prev_topk,
            )
            outs.append(attn_out)
            topk_outs.append(topk_indices)

        if self.skip_allreduce:
            return outs, topk_outs

        return self.allreduce(
            inputs=outs, signal_buffers=signal_buffers
        ), topk_outs
