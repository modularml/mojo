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

"""An opaque KV Cache optimized attention mechanism with Rope."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weight import _compute_shard_range
from max.nn.quant_config import QuantConfig

from ..clamp import clamp
from ..comm import Allreduce
from ..kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul_quantized,
    rope_split_store_ragged,
    unfused_qkv_ragged_matmul_gguf_quantized,
)
from ..kv_cache import KVCacheParams, PagedCacheValues
from ..layer import Module, Shardable
from ..linear import Linear
from ..norm import RMSNorm
from ..rotary_embedding import RotaryEmbedding
from ..stacked_linear import StackedLinear
from .interfaces import DistributedAttentionImpl
from .mask_config import MHAMaskVariant


class AttentionWithRope(Module, Shardable):
    """Implementation of attention that uses Rotary Position Embedding (RoPE)."""

    # This class will not use the RotaryEmbedding to calculate rope, but it
    # already includes a freqs_cis calculation, which we will borrow.
    rope: RotaryEmbedding

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        sharding_strategy: ShardingStrategy | None = None,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        quant_config: QuantConfig | None = None,
        clip_qkv: float | None = None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
        sliding_window: int | None = None,
        _fuse_rope_and_store: bool = True,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            sharding_strategy: Optional initial sharding strategy.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache params, including number of kv heads, head dim, and dtype.
            dtype: DType of the QKV and output projection weights.
            devices: Device(s) on which to place the weights and run the computation. If multiple are
                provided, the first device is used for weight placement here.
            linear_cls: Linear class to use for projections.
            stacked_qkv: Whether Q/K/V weights are stacked in a single Weight.
            scale: Optional attention scale; defaults to sqrt(1/head_dim).
            has_bias: Whether Q/K/V have bias (stacked_qkv forbids bias).
            quant_config: Optional quantization config (dynamic or static).
            clip_qkv: If provided, clamp Q/K/V weights to [-clip_qkv, clip_qkv].
            use_qk_norm: Whether to use RMSNorm on Q/K.
            rms_norm_eps: Value to use for numerical stability in RMSNorm.

            _fuse_rope_and_store: If True (default), emit a single fused
                rope+split+store custom op. If False, emit separate rope, split,
                and store ops to test graph compiler fusion.
        """
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_params = kv_params
        self.hidden_size = hidden_size
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.clip_qkv = clip_qkv
        self.devices = devices or [DeviceRef.CPU()]
        self.quant_config = quant_config
        self.stacked_qkv = stacked_qkv
        self.use_qk_norm = use_qk_norm
        self.rms_norm_eps = rms_norm_eps
        self.mask_variant = mask_variant
        self.sliding_window = sliding_window
        self._fuse_rope_and_store = _fuse_rope_and_store
        self._sharding_strategy: ShardingStrategy | None = None

        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.kv_params.head_dim, dtype, eps=rms_norm_eps
            )
            self.k_norm = RMSNorm(
                self.kv_params.head_dim, dtype, eps=rms_norm_eps
            )
            num_devices = len(self.devices)
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_weight_dim

        self.qkv_proj = StackedLinear(
            in_dim=hidden_size,
            out_dims=[q_weight_dim, kv_weight_dim, kv_weight_dim],
            names=["q_proj", "k_proj", "v_proj"],
            dtype=dtype,
            device=self.devices[0],
            stacked=stacked_qkv,
            has_bias=has_bias,
            linear_cls=linear_cls,
            quant_config=quant_config,
            clip_weight=clip_qkv,
        )

        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
            quant_config=quant_config,
        )

        if sharding_strategy is not None:
            self.sharding_strategy = sharding_strategy

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the Module sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the Module sharding strategy and propagate to weights.

        We support both tensor-parallel and data-parallel (replicate) sharding strategies.
        """
        if strategy.is_tensor_parallel:
            self._sharding_strategy = strategy

            num_devices = strategy.num_devices
            if self.stacked_qkv:
                # Partition the [Q|K|V] block by heads.
                self.qkv_proj.sharding_strategy = ShardingStrategy.stacked_qkv(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            else:
                # Column-shard by output channels (heads) for each projection.
                self.qkv_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )

            # Row-shard o_proj.weight (standard tensor parallel o-proj).
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )
        elif strategy.is_replicate:
            self._sharding_strategy = strategy

            num_devices = strategy.num_devices
            self.qkv_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.o_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        else:
            raise ValueError(
                "Only tensor-parallel (rowwise) or data-parallel (replicate) sharding strategies are supported for AttentionWithRope"
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[AttentionWithRope]:
        """Create sharded views across `devices` (tensor-parallel).

        Returns one `AttentionWithRope` per device with appropriately sliced weights.
        """
        devices = list(devices)
        if not self.sharding_strategy:
            raise ValueError(
                "AttentionWithRope cannot be sharded: no sharding_strategy set. "
                "Set `self.sharding_strategy = ShardingStrategy.tensor_parallel(N)` first."
            )

        if self.sharding_strategy.is_tensor_parallel:
            if DeviceRef.CPU() in devices:
                raise ValueError(
                    "Tensor-parallel AttentionWithRope does not support CPU devices"
                )

            qkv_proj_shards = self.qkv_proj.shard(devices)
            o_proj_shards = self.o_proj.shard(devices)

            # Replicate Q/K RMSNorm gamma across devices (per-head gamma is shared).
            if self.use_qk_norm:
                # Ensure replication strategy is set before sharding.
                self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                    len(devices)
                )
                self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                    len(devices)
                )
                q_norm_replicas = self.q_norm.shard(devices)
                k_norm_replicas = self.k_norm.shard(devices)
            else:
                q_norm_replicas = None
                k_norm_replicas = None

            default_dtype = o_proj_shards[0].weight.dtype
            linear_cls = self.o_proj.__class__

            shards: list[AttentionWithRope] = []
            num_devices = len(devices)
            for n, device in enumerate(devices):
                # Compute this shard's number of Q and KV heads.
                head_start, head_end = _compute_shard_range(
                    self.n_heads, n, num_devices
                )
                device_num_heads = head_end - head_start

                kv_head_start, kv_head_end = _compute_shard_range(
                    self.num_key_value_heads, n, num_devices
                )
                device_num_kv_heads = kv_head_end - kv_head_start

                layer = AttentionWithRope(
                    rope=self.rope,
                    num_attention_heads=device_num_heads,
                    num_key_value_heads=device_num_kv_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    devices=[device],
                    dtype=default_dtype,
                    linear_cls=linear_cls,
                    stacked_qkv=self.stacked_qkv,
                    scale=self.scale,
                    has_bias=self.has_bias,
                    quant_config=self.quant_config,
                    clip_qkv=self.clip_qkv,
                    mask_variant=self.mask_variant,
                    sliding_window=self.sliding_window,
                )

                layer.qkv_proj = qkv_proj_shards[n]
                layer.o_proj = o_proj_shards[n]

                if self.use_qk_norm:
                    assert (
                        q_norm_replicas is not None
                        and k_norm_replicas is not None
                    )
                    layer.q_norm = q_norm_replicas[n]
                    layer.k_norm = k_norm_replicas[n]
                    layer.use_qk_norm = True
                    layer.rms_norm_eps = self.rms_norm_eps

                shards.append(layer)
            return shards

        elif self.sharding_strategy.is_replicate:
            # Replicate full weights to each device (no head split).
            qkv_proj_replicas = self.qkv_proj.shard(devices)
            o_proj_replicas = self.o_proj.shard(devices)

            # Replicate Q/K RMSNorm gamma as well if used.
            if self.use_qk_norm:
                q_norm_replicas = self.q_norm.shard(devices)
                k_norm_replicas = self.k_norm.shard(devices)
            else:
                q_norm_replicas = None
                k_norm_replicas = None

            default_dtype = o_proj_replicas[0].weight.dtype
            linear_cls = self.o_proj.__class__

            replicas: list[AttentionWithRope] = []
            for i, device in enumerate(devices):
                replica = AttentionWithRope(
                    rope=self.rope,
                    num_attention_heads=self.n_heads,  # DP keeps full heads
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    devices=[device],
                    dtype=default_dtype,
                    linear_cls=linear_cls,
                    stacked_qkv=self.stacked_qkv,
                    scale=self.scale,
                    has_bias=self.has_bias,
                    quant_config=self.quant_config,
                    clip_qkv=self.clip_qkv,
                    mask_variant=self.mask_variant,
                    sliding_window=self.sliding_window,
                )
                replica.qkv_proj = qkv_proj_replicas[i]
                replica.o_proj = o_proj_replicas[i]

                if self.use_qk_norm:
                    assert (
                        q_norm_replicas is not None
                        and k_norm_replicas is not None
                    )
                    replica.q_norm = q_norm_replicas[i]
                    replica.k_norm = k_norm_replicas[i]
                    replica.use_qk_norm = True
                    replica.rms_norm_eps = self.rms_norm_eps

                replicas.append(replica)
            return replicas

        else:
            # Should not happen due to setter validation.
            raise ValueError(
                "Unsupported sharding strategy for AttentionWithRope"
            )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        if self.stacked_qkv:
            return self.qkv_proj.weight
        else:
            wq: TensorValue = self.qkv_proj._child("q_proj").weight
            wk: TensorValue = self.qkv_proj._child("k_proj").weight
            wv: TensorValue = self.qkv_proj._child("v_proj").weight
            if self.clip_qkv:
                wq = clamp(wq, min=-self.clip_qkv, max=self.clip_qkv)
                wk = clamp(wk, min=-self.clip_qkv, max=self.clip_qkv)
                wv = clamp(wv, min=-self.clip_qkv, max=self.clip_qkv)

            wqkv = ops.concat((wq, wk, wv))
            return wqkv

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None
        # This was already checked in the constructor.
        assert not self.stacked_qkv

        # Access bias, which should all exist since has_bias=True.
        q_bias = self.qkv_proj._child("q_proj").bias
        k_bias = self.qkv_proj._child("k_proj").bias
        v_bias = self.qkv_proj._child("v_proj").bias
        assert q_bias is not None
        assert k_bias is not None
        assert v_bias is not None
        return ops.concat((q_bias, k_bias, v_bias))

    @property
    def qkv_input_scale(self) -> TensorValue | None:
        """The max of q, k, and v scale input vectors."""
        if not self.quant_config or self.quant_config.is_dynamic:
            return None

        if self.stacked_qkv:
            raise NotImplementedError(
                "QKV input scale not implemented for stacked_qkv=True"
            )

        q_is = self.qkv_proj._child("q_proj").input_scale
        k_is = self.qkv_proj._child("k_proj").input_scale
        v_is = self.qkv_proj._child("v_proj").input_scale
        assert q_is is not None
        assert k_is is not None
        assert v_is is not None

        return ops.max(
            ops.concat(
                (q_is.reshape((1,)), k_is.reshape((1,)), v_is.reshape((1,)))
            )
        ).reshape(())

    @property
    def qkv_weight_scale(self) -> TensorValue:
        """The max of q, k, and v scale weight vectors."""
        assert self.quant_config is not None

        if self.stacked_qkv:
            raise NotImplementedError(
                "QKV weight scale not implemented for stacked_qkv=True"
            )

        q_ws = self.qkv_proj._child("q_proj").weight_scale
        k_ws = self.qkv_proj._child("k_proj").weight_scale
        v_ws = self.qkv_proj._child("v_proj").weight_scale
        assert q_ws is not None
        assert k_ws is not None
        assert v_ws is not None
        q_scale: TensorValue = (
            q_ws.reshape((1,)) if len(q_ws.shape) == 0 else q_ws
        )
        k_scale: TensorValue = (
            k_ws.reshape((1,)) if len(k_ws.shape) == 0 else k_ws
        )
        v_scale: TensorValue = (
            v_ws.reshape((1,)) if len(v_ws.shape) == 0 else v_ws
        )

        weight_scale = ops.concat((q_scale, k_scale, v_scale))

        if self.quant_config.weight_scale.is_tensor:
            # Fused QKV: broadcast each projection's scalar scale to
            # [dim, 1] rowwise and concatenate so each output row keeps
            # its exact original per-projection scale.
            q_dim = self.n_heads * self.kv_params.head_dim
            kv_dim = self.num_key_value_heads * self.kv_params.head_dim
            q_row = ops.broadcast_to(q_scale.reshape([1, 1]), [q_dim, 1])
            k_row = ops.broadcast_to(k_scale.reshape([1, 1]), [kv_dim, 1])
            v_row = ops.broadcast_to(v_scale.reshape([1, 1]), [kv_dim, 1])
            return ops.concat((q_row, k_row, v_row))

        # Per-row / per-block scaling: each row (or block) of the fused QKV
        # weight has its own scale, so return the concatenated scales directly.
        return weight_scale

    @property
    def qkv_weight_scale_2(self) -> TensorValue | None:
        """The max of q, k, and v scale input vectors."""
        if (
            not self.quant_config
            or self.quant_config.is_dynamic
            or not self.quant_config.is_nvfp4
        ):
            return None

        if self.stacked_qkv:
            raise NotImplementedError(
                "QKV input scale not implemented for stacked_qkv=True"
            )

        q_ws2 = self.qkv_proj._child("q_proj").weight_scale_2
        k_ws2 = self.qkv_proj._child("k_proj").weight_scale_2
        v_ws2 = self.qkv_proj._child("v_proj").weight_scale_2
        assert q_ws2 is not None
        assert k_ws2 is not None
        assert v_ws2 is not None

        return ops.max(
            ops.concat(
                (
                    q_ws2.reshape((1,)),
                    k_ws2.reshape((1,)),
                    v_ws2.reshape((1,)),
                )
            )
        ).reshape(())

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # QKV matmul via StackedLinear (handles stacked vs unfused,
        # quantization, bias, and weight clipping).
        qkv = self.qkv_proj(x)

        if self.use_qk_norm:
            # QK norm must happen before rope. Split Q/K from the flat QKV
            # buffer, normalize per-head, then re-concat.
            head_dim = self.kv_params.head_dim
            q_dim = self.n_heads * head_dim
            kv_dim = self.num_key_value_heads * head_dim
            x_q, x_k, x_v = ops.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)
            # Per-head RMSNorm on Q and K before rope.
            x_q = self.q_norm(x_q.reshape((-1, head_dim))).reshape(x_q.shape)
            x_k = self.k_norm(x_k.reshape((-1, head_dim))).reshape(x_k.shape)
            qkv = ops.concat((x_q, x_k, x_v), axis=-1)

        # Fused rope + split + KV store.
        freqs_cis = ops.cast(freqs_cis, qkv.dtype).to(qkv.device)
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=self.rope.interleaved,
            fuse=self._fuse_rope_and_store,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        local_window_size = -1
        if self.sliding_window is not None:
            local_window_size = self.sliding_window

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=self.mask_variant,
            scale=self.scale,
            local_window_size=local_window_size,
        )
        attn_out = ops.reshape(
            attn_out, shape=[total_seq_len, self.q_weight_dim]
        )
        return self.o_proj(attn_out)

    def materialize_kv_from_hidden(
        self,
        layer_idx: TensorValue,
        hidden: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> None:
        """Project ``hidden`` to K/V and write into the paged KV cache.

        Used by speculative-decoding draft models that build their KV cache
        from external (e.g. target) hidden states.
        """
        qkv = self.qkv_proj(hidden)

        if self.use_qk_norm:
            head_dim = self.kv_params.head_dim
            q_dim = self.n_heads * head_dim
            kv_dim = self.num_key_value_heads * head_dim
            x_q, x_k, x_v = ops.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)
            x_q = self.q_norm(x_q.reshape((-1, head_dim))).reshape(x_q.shape)
            x_k = self.k_norm(x_k.reshape((-1, head_dim))).reshape(x_k.shape)
            qkv = ops.concat((x_q, x_k, x_v), axis=-1)

        freqs_cis_cast = ops.cast(freqs_cis, qkv.dtype).to(qkv.device)
        _ = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis_cast,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=self.rope.interleaved,
            fuse=self._fuse_rope_and_store,
        )


class GGUFQAttentionWithRope(AttentionWithRope):
    """Implementation of attention with GGUF quantized weights."""

    # This class will not use the RotaryEmbedding to calculate rope, but it
    # already includes a freqs_cis calculation, which we will borrow.
    rope: RotaryEmbedding

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        dtype: DType,
        quantization_encoding: QuantizationEncoding,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
        mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
    ) -> None:
        """Initializes the GGUF attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache params, including number of kv heads, head dim, and dtype.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the weights, should always be uint8.
            devices: Device(s) on which to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `TensorParallelAttentionWithRope` to use all devices during
                attention computation.
            quantization_encoding: Quantization encoding of the weights.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`
            mask_variant: Attention mask used by the flash-attention kernel.
                Defaults to ``MHAMaskVariant.CAUSAL_MASK``.
        """
        # Skip AttentionWithRope.__init__ because the weights are created differently.
        Module.__init__(self)
        self.mask_variant = mask_variant

        if dtype != DType.uint8:
            raise ValueError(
                f"GGUFQAttentionWithRope only supports uint8 dtype weights but got {dtype}"
            )
        if clip_qkv is not None:
            raise ValueError(
                "clip_qkv is not supported for GGUFQAttentionWithRope"
            )
        if has_bias:
            raise ValueError("GGUFQAttentionWithRope does not support bias")
        if not quantization_encoding.is_gguf:
            raise ValueError(
                f"Only GGUF quantization encoding is supported for GGUFQAttentionWithRope. Found: {quantization_encoding}"
            )

        self.quantization_encoding = quantization_encoding
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.devices = devices or [DeviceRef.CPU()]

        self.q_proj_weight = Weight(
            name="q_proj.weight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )
        self.k_proj_weight = Weight(
            name="k_proj.weight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )
        self.v_proj_weight = Weight(
            name="v_proj.weight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )
        self.o_proj = linear_cls(
            in_dim=1,  # Shape will be overridden at load_state_dict.
            out_dim=1,  # Shape will be overridden at load_state_dict.
            dtype=DType.uint8,
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )

    @property
    def wqkv(self) -> TensorValue:
        raise NotImplementedError(
            "wqkv is not implemented for unfused GGUFQAttentionWithRope"
        )

    @property
    def wqkv_bias(self) -> TensorValue | None:
        raise NotImplementedError(
            "wqkv_bias is not implemented for unfused GGUFQAttentionWithRope"
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        assert self.q_proj_weight.quantization_encoding is not None
        assert self.k_proj_weight.quantization_encoding is not None
        assert self.v_proj_weight.quantization_encoding is not None

        # Unfused GGUF path.
        xq = unfused_qkv_ragged_matmul_gguf_quantized(
            self.kv_params,
            input=x,
            input_row_offsets=input_row_offsets,
            n_heads=self.n_heads,
            q_weight=self.q_proj_weight,
            k_weight=self.k_proj_weight,
            v_weight=self.v_proj_weight,
            quantization_encoding_q=self.q_proj_weight.quantization_encoding,
            quantization_encoding_k=self.k_proj_weight.quantization_encoding,
            quantization_encoding_v=self.v_proj_weight.quantization_encoding,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
        )

        # Apply RoPE.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
        )

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=self.mask_variant,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)


class GPTQAttentionWithRope(AttentionWithRope):
    """Implementation of the GPTQ attention layer.

    Args:
        quantization_config: The GPTQ quantization configuration, including
            ``desc_act`` for activation-order permutation support.
        rope: The rope layer to borrow the ``freqs_cis`` value from.
        num_attention_heads: The number of attention heads.
        num_key_value_heads: The number of key/value heads.
        hidden_size: The dimension of the hidden states.
        kv_params: The KV cache parameters, including number of KV heads,
            head dim, and dtype.
        devices: The device or devices on which to place the weights and run
            the computation. If multiple are provided, the first device is used.
        dtype: The DType for the output projection weights.
        scale: Optional attention scale; defaults to ``sqrt(1/head_dim)``.
        linear_cls: The linear class to use for the output projection.
    """

    def __init__(
        self,
        quantization_config: QuantizationConfig,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: list[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        scale: float | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
    ) -> None:
        # Skip AttentionWithRope.__init__ because the weights are created differently.
        Module.__init__(self)
        self.quantization_config = quantization_config
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.hidden_size = hidden_size
        self.devices = devices or [DeviceRef.CPU()]
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.mask_variant = mask_variant

        self.kv_weight_dim = (
            hidden_size // num_attention_heads
        ) * num_key_value_heads

        self.q_proj_qweight = Weight(
            name="q_proj.qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.k_proj_qweight = Weight(
            name="k_proj.qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.v_proj_qweight = Weight(
            name="v_proj.qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )

        self.q_proj_scales = Weight(
            name="q_proj.scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.k_proj_scales = Weight(
            name="k_proj.scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.v_proj_scales = Weight(
            name="v_proj.scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.o_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )

        self.perm_idx = None
        if quantization_config.desc_act:
            self.perm_idx = Weight(
                name="q_proj.perm_idx",
                dtype=DType.int32,
                shape=[hidden_size],
                device=self.devices[0],
            )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors (packed + scales)."""
        # fmt: off
        # The `qweight` tensor for a QuantLinear is of type uint32. When allocated as bytes, we reshape the
        # uint8 tensor to [cols, rows * 4] so concatenating the uint8 tensors along axis=1 is equivalent to
        # concatenating the original uint32 tensors along axis=1.
        wq_qweight = ops.reshape(self.q_proj_qweight, (-1, self.hidden_size * 4))
        wk_qweight = ops.reshape(self.k_proj_qweight, (-1, self.kv_weight_dim * 4))
        wv_qweight = ops.reshape(self.v_proj_qweight, (-1, self.kv_weight_dim * 4))

        wqkv_qweight = ops.reshape(
            ops.concat((wq_qweight, wk_qweight, wv_qweight), axis=1),
            (-1, self.hidden_size + 2 * self.kv_weight_dim),
        )
        # `scales` tensor is in f16/bf16 type, so we reshape the uint8 tensor to [cols, rows * 2].
        wq_scales = ops.reshape(self.q_proj_scales, (-1, self.hidden_size * 2))
        wk_scales = ops.reshape(self.k_proj_scales, (-1, self.kv_weight_dim * 2))
        wv_scales = ops.reshape(self.v_proj_scales, (-1, self.kv_weight_dim * 2))

        wqkv_scales = ops.reshape(
            ops.concat((wq_scales, wk_scales, wv_scales), axis=1),
            (-1, self.hidden_size + 2 * self.kv_weight_dim),
        )
        # fmt: on
        return ops.concat((wqkv_qweight, wqkv_scales), axis=0)

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        wqkv = self.wqkv
        if self.devices:
            wqkv = wqkv.to(self.devices[0])

        xq = fused_qkv_ragged_matmul_quantized(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            perm_idx=self.perm_idx,
            quantization_config=self.quantization_config,
        )

        # Apply RoPE.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)

        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
        )

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=self.mask_variant,
            scale=self.scale,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.o_proj(attn_out)


class TensorParallelAttentionWithRope(
    AttentionWithRope, DistributedAttentionImpl
):
    """Tensor-parallel wrapper that delegates sharding to the base module."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        quant_config: QuantConfig | None = None,
        clip_qkv: float | None = None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
        sliding_window: int | None = None,
    ) -> None:
        """Initializes the distributed (tensor parallel) attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache params, including number of kv heads, head dim, and dtype.
            devices: Device(s) on which to place the weights and run the computation. Must
                provide at least 2 devices for tensor parallel attention.
            dtype: DType of the QKV and output projection weights.
            linear_cls: Linear class to use for the outputs dense layer.
            stacked_qkv: Whether the weights are stacked together.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            quant_config: Quantization configuration.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`.
            use_qk_norm: Whether to use RMSNorm on Q/K.
            rms_norm_eps: Value to use for numerical stability in RMSNorm.
            mask_variant: Attention mask used by the flash-attention kernel.
                Defaults to ``MHAMaskVariant.CAUSAL_MASK``.
        """
        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            devices=devices,
            dtype=dtype,
            linear_cls=linear_cls,
            stacked_qkv=stacked_qkv,
            scale=scale,
            has_bias=has_bias,
            quant_config=quant_config,
            clip_qkv=clip_qkv,
            use_qk_norm=use_qk_norm,
            rms_norm_eps=rms_norm_eps,
            mask_variant=mask_variant,
            sliding_window=sliding_window,
        )
        if DeviceRef.CPU() in self.devices:
            raise ValueError(
                "TensorParallelAttentionWithRope does not support CPU devices"
            )

        num_devices = len(self.devices)
        self.allreduce = Allreduce(num_devices)

        # Delegate: configure base sharding + create per-device modules.
        self.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        x: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")
        if len(input_row_offsets) != len(self.devices):
            raise ValueError(
                f"Expected {len(self.devices)} input_row_offsets, got {len(input_row_offsets)}"
            )
        if not all(isinstance(t, TensorValue) for t in input_row_offsets):
            raise TypeError(
                "All elements in input_row_offsets must be TensorValue instances"
            )
        if not all(isinstance(t, TensorValue) for t in freqs_cis):
            raise TypeError(
                "All elements in freqs_cis must be TensorValue instances"
            )

        attn_outputs = [
            self.list_of_attentions[i](
                layer_idx,
                x[i],
                kv_collections[i],
                freqs_cis[i],
                input_row_offsets[i],
            )
            for i in range(len(self.devices))
        ]

        return self.allreduce(
            inputs=attn_outputs, signal_buffers=signal_buffers
        )

    def materialize_kv_from_hidden(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        hiddens: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
    ) -> None:
        n = len(self.devices)
        if not (
            len(hiddens)
            == len(kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "All per-device inputs to materialize_kv_from_hidden must "
                f"have length equal to the number of devices ({n})."
            )
        for i in range(n):
            self.list_of_attentions[i].materialize_kv_from_hidden(
                layer_idx=layer_idx,
                hidden=hiddens[i],
                kv_collection=kv_collections[i],
                freqs_cis=freqs_cis[i],
                input_row_offsets=input_row_offsets[i],
            )


class DataParallelAttentionWithRope(AttentionWithRope):
    """Data-parallel implementation of Attention with RoPE.

    This replicates the attention module across devices and runs each replica on
    its local inputs (x, kv, freqs_cis, input_row_offsets). No collective ops
    are required; KV-cache remains local to each device.

    Notes:
      - Assumes the caller has already distributed `xs`, `kv_collections`,
        `freqs_cis`, and `input_row_offsets` so that index i corresponds to
        device i, with `input_row_offsets[i]` rebased to start at 0.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        quant_config: QuantConfig | None = None,
        clip_qkv: float | None = None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
        mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__(
            rope=rope,
            sharding_strategy=None,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            devices=devices,
            dtype=dtype,
            linear_cls=linear_cls,
            stacked_qkv=stacked_qkv,
            scale=scale,
            has_bias=has_bias,
            quant_config=quant_config,
            clip_qkv=clip_qkv,
            use_qk_norm=use_qk_norm,
            rms_norm_eps=rms_norm_eps,
            mask_variant=mask_variant,
            sliding_window=sliding_window,
        )
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)

        # Replicate component weights/modules to each device.
        self.qkv_proj.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        qkv_proj_replicas = self.qkv_proj.shard(self.devices)

        self.o_proj.sharding_strategy = ShardingStrategy.replicate(num_devices)
        if self.use_qk_norm:
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        o_proj_replicas = self.o_proj.shard(self.devices)

        # Replicate Q/K RMSNorm weights if enabled.
        if self.use_qk_norm:
            q_norm_replicas = self.q_norm.shard(self.devices)
            k_norm_replicas = self.k_norm.shard(self.devices)
        else:
            q_norm_replicas = None
            k_norm_replicas = None

        # Build one full copy per device (no head-splitting).
        self.replicated_attentions: list[AttentionWithRope] = []
        for i, device in enumerate(self.devices):
            replica = AttentionWithRope(
                rope=self.rope,
                sharding_strategy=None,
                num_attention_heads=self.n_heads,  # DP keeps full heads
                num_key_value_heads=self.num_key_value_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                devices=[device],
                dtype=dtype,
                linear_cls=linear_cls,
                stacked_qkv=self.stacked_qkv,
                scale=self.scale,
                has_bias=self.has_bias,
                quant_config=self.quant_config,
                clip_qkv=self.clip_qkv,
                mask_variant=self.mask_variant,
                sliding_window=self.sliding_window,
            )
            replica.qkv_proj = qkv_proj_replicas[i]
            replica.o_proj = o_proj_replicas[i]

            if self.use_qk_norm:
                assert (
                    q_norm_replicas is not None and k_norm_replicas is not None
                )
                replica.q_norm = q_norm_replicas[i]
                replica.k_norm = k_norm_replicas[i]
                replica.use_qk_norm = True
                replica.rms_norm_eps = self.rms_norm_eps

            self.replicated_attentions.append(replica)

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, freqs_cis, and input_row_offsets must all have "
                f"length equal to number of devices ({n})"
            )

        outs: list[TensorValue] = []
        for i in range(n):
            if xs[i].shape[0] == 0:
                outs.append(xs[i])
                continue

            outs.append(
                self.replicated_attentions[i](
                    layer_idx,
                    xs[i],
                    kv_collections[i],
                    freqs_cis=freqs_cis[i],
                    input_row_offsets=input_row_offsets[i],
                )
            )
        return outs

    def materialize_kv_from_hidden(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        hiddens: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
    ) -> None:
        """Data-parallel KV materialization from external hidden states."""
        n = len(self.devices)
        if not (
            len(hiddens)
            == len(kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "All per-device inputs to materialize_kv_from_hidden must "
                f"have length equal to the number of devices ({n})."
            )
        for i in range(n):
            self.replicated_attentions[i].materialize_kv_from_hidden(
                layer_idx=layer_idx,
                hidden=hiddens[i],
                kv_collection=kv_collections[i],
                freqs_cis=freqs_cis[i],
                input_row_offsets=input_row_offsets[i],
            )
