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

import os
from math import prod

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig
from max.nn.stacked_linear import StackedLinear

# Set to "1" to keep the FP8 cross-attn K/V as two separate matmuls instead
# of one fused (dequant -> concat -> requant) matmul. Mirrors the
# MAX_DISABLE_FUSED_SWIGLU_NVFP4 env toggle used for the NVFP4 MoE SwiGLU.
_DISABLE_FP8_KV_FUSION = os.environ.get("MAX_WAN_DISABLE_FP8_KV_FUSION") == "1"

from .embeddings import (
    TimestepEmbedding,
    Timesteps,
    apply_rotary_emb_fused,
)
from .model_config import WanConfigBase


def _make_linear(
    *,
    in_dim: int,
    out_dim: int,
    dtype: DType,
    device: DeviceRef,
    has_bias: bool,
    quant_config: QuantConfig | None,
) -> Linear:
    """Build a Linear that is FP8-quantized when ``quant_config`` is set.

    Under FP8 the weight is stored as ``float8_e4m3fn`` with scalar
    ``weight_scale`` / ``input_scale`` (static per-tensor); the bias stays
    in ``bfloat16`` (via ``quant_config.bias_dtype``). With no quant config
    the layer is a plain ``dtype`` Linear.
    """
    weight_dtype = DType.float8_e4m3fn if quant_config is not None else dtype
    return Linear(
        in_dim=in_dim,
        out_dim=out_dim,
        dtype=weight_dtype,
        device=device,
        has_bias=has_bias,
        quant_config=quant_config,
    )


class WanConv3d(Module):
    """3D conv for WAN patch embedding (NDHWC/QRSCF layout)."""

    def __init__(
        self,
        kernel_size: tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        d, h, w = kernel_size
        self.filter = Weight(
            "weight", dtype, [d, h, w, in_channels, out_channels], device
        )
        self.bias = (
            Weight("bias", dtype, [out_channels], device) if has_bias else None
        )
        self.stride = stride

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.conv3d(x, self.filter, stride=self.stride, bias=self.bias)


class WanLayerNorm(Module):
    """LayerNorm using the fused ``ops.layer_norm`` kernel.

    The fused kernel accumulates in f32 internally (via ``get_accum_type``)
    so bf16 inputs get full-precision normalization without graph-level f32
    scratchpad tensors.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        *,
        elementwise_affine: bool = True,
        use_bias: bool = True,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.has_weight = elementwise_affine
        self.has_bias = elementwise_affine and use_bias
        if elementwise_affine:
            self.weight = Weight("weight", dtype, [dim], device)
            if use_bias:
                self.bias = Weight("bias", dtype, [dim], device)

    def __call__(self, x: TensorValue) -> TensorValue:
        gamma = ops.cast(self.weight, x.dtype) if self.has_weight else None
        if self.has_bias:
            beta = ops.cast(self.bias, x.dtype)
        else:
            beta = ops.broadcast_to(
                ops.constant(0.0, x.dtype, x.device),
                shape=(x.shape[-1],),
            )
        if gamma is None:
            gamma = ops.broadcast_to(
                ops.constant(1.0, x.dtype, x.device),
                shape=(x.shape[-1],),
            )
        return ops.layer_norm(x, gamma=gamma, beta=beta, epsilon=self.eps)


class WanRMSNorm(Module):
    """RMSNorm using the fused ``ops.rms_norm`` kernel.

    The fused kernel accumulates in f32 internally (via ``get_accum_type``)
    so bf16 inputs get full-precision normalization without graph-level f32
    scratchpad tensors.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, [dim], device)
        self.eps = eps

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.rms_norm(x, ops.cast(self.weight, x.dtype), self.eps)


class WanTextProjection(Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.linear_1 = _make_linear(
            in_dim=in_features,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )
        self.linear_2 = _make_linear(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )

    def __call__(self, caption: TensorValue) -> TensorValue:
        hidden_states = self.linear_1(caption)
        hidden_states = ops.gelu(hidden_states, approximate="tanh")
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class WanImageEmbedder(Module):
    """Image embedding for Wan 2.1 I2V: LayerNorm → GEGLU FFN → LayerNorm.

    Matches diffusers' FeedForward(image_dim, dim, mult=1, activation_fn="gelu")
    with pre/post norms.  Weight keys::

        image_embedder.norm1.{weight,bias}
        image_embedder.ff.net.0.proj.{weight,bias}   (GEGLU gate+value)
        image_embedder.ff.net.2.{weight,bias}         (output linear)
        image_embedder.norm2.{weight,bias}
    """

    def __init__(
        self,
        image_dim: int,
        out_dim: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        # Matches diffusers FeedForward(image_dim, out_dim, mult=1, activation_fn="gelu"):
        #   norm1(image_dim) → Linear(image_dim→image_dim) → GELU →
        #   Linear(image_dim→out_dim) → norm2(out_dim)
        self.norm1 = WanLayerNorm(
            image_dim,
            elementwise_affine=True,
            use_bias=True,
            dtype=dtype,
            device=device,
        )
        self.ff_proj = _make_linear(
            in_dim=image_dim,
            out_dim=image_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )
        self.ff_out = _make_linear(
            in_dim=image_dim,
            out_dim=out_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )
        self.norm2 = WanLayerNorm(
            out_dim,
            elementwise_affine=True,
            use_bias=True,
            dtype=dtype,
            device=device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.norm1(x)
        x = ops.gelu(self.ff_proj(x))
        x = self.ff_out(x)
        return self.norm2(x)


class WanTimeTextImageEmbedding(Module):
    def __init__(
        self,
        dim: int,
        freq_dim: int,
        text_dim: int,
        num_layers: int,
        *,
        image_dim: int | None = None,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.timesteps_proj = Timesteps(
            num_channels=freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim,
            time_embed_dim=dim,
            dtype=dtype,
            device=device,
            quant_config=quant_config,
        )
        # Projects SiLU(temb) to 6 modulation params per block
        self.time_proj = _make_linear(
            in_dim=dim,
            out_dim=dim * 6,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )
        self.text_embedder = WanTextProjection(
            in_features=text_dim,
            hidden_size=dim,
            dtype=dtype,
            device=device,
            quant_config=quant_config,
        )
        # Optional image embedder (Wan 2.1 I2V)
        self.image_embedder: WanImageEmbedder | None = None
        if image_dim is not None:
            self.image_embedder = WanImageEmbedder(
                image_dim=image_dim,
                out_dim=dim,
                dtype=dtype,
                device=device,
                quant_config=quant_config,
            )

    def __call__(
        self, timestep: TensorValue, encoder_hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        # Sinusoidal timestep embedding (computed in float32 for precision).
        # Cast to the model's working dtype (bf16) for the MLP, matching
        # diffusers' behavior: float32 embedding → cast to weight dtype → MLP.
        timesteps_emb = self.timesteps_proj(timestep)  # [B, freq_dim] float32
        timesteps_emb = ops.cast(
            timesteps_emb, encoder_hidden_states.dtype
        )  # → bf16
        temb = self.time_embedder(timesteps_emb)  # [B, dim]

        # Timestep projection for modulation: SiLU then linear
        timestep_proj = self.time_proj(ops.silu(temb))  # [B, dim*6]
        # Reshape to [B, 6, dim] for per-block modulation
        timestep_proj = ops.reshape(
            timestep_proj,
            [timestep_proj.shape[0], 6, timestep_proj.shape[1] // 6],
        )

        # Text projection
        text_emb = self.text_embedder(encoder_hidden_states)  # [B, S, dim]

        return temb, timestep_proj, text_emb


class WanSelfAttention(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = dim

        self.to_q = _make_linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )
        self.to_k = _make_linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )
        self.to_v = _make_linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )
        self.norm_q = WanRMSNorm(dim, eps=eps, dtype=dtype, device=device)
        self.norm_k = WanRMSNorm(dim, eps=eps, dtype=dtype, device=device)
        self.to_out = _make_linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        rotary_emb: tuple[TensorValue, TensorValue],
    ) -> TensorValue:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # QK-norm applied across all heads (before reshape)
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape to multi-head: [B, S, D] -> [B, S, H, head_dim]
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        query = ops.reshape(
            query, [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        key = ops.reshape(
            key, [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        value = ops.reshape(
            value, [batch_size, seq_len, self.num_heads, self.head_dim]
        )

        # Apply RoPE via the fused ragged kernel (eliminates the per-token
        # fused_concat_inner_most_single_dim that ops.stack lowers to).
        query, key = apply_rotary_emb_fused(query, key, rotary_emb)

        # Flash attention
        scale = 1.0 / (self.head_dim**0.5)
        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )

        # Reshape back: [B, S, H, head_dim] -> [B, S, D]
        hidden_states = ops.reshape(
            hidden_states,
            [hidden_states.shape[0], hidden_states.shape[1], self.inner_dim],
        )
        return self.to_out(hidden_states)


class WanCrossAttention(Module):
    def __init__(
        self,
        dim: int,
        text_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
        *,
        added_kv_proj_dim: int | None = None,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = dim
        self._has_added_kv = added_kv_proj_dim is not None
        self._fp8 = quant_config is not None
        self._fuse_kv = not (self._fp8 and _DISABLE_FP8_KV_FUSION)

        self.to_q = _make_linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )
        if not self._fp8:
            # bfloat16: K+V fused into one weight at load time.
            self.to_kv = _make_linear(
                in_dim=text_dim,
                out_dim=dim * 2,
                dtype=dtype,
                device=device,
                has_bias=True,
                quant_config=quant_config,
            )
        elif self._fuse_kv:
            # FP8 default: StackedLinear holds per-scale to_k/to_v children and
            # fuses them (dequant -> concat -> requant) into one matmul in its
            # __call__. Weight names stay at attn2.to_k / attn2.to_v.
            self.kv_proj = StackedLinear(
                in_dim=text_dim,
                out_dims=[dim, dim],
                names=["to_k", "to_v"],
                dtype=DType.float8_e4m3fn,
                device=device,
                stacked=False,
                has_bias=True,
                quant_config=quant_config,
            )
        else:
            # FP8, fusion disabled: two standalone per-projection matmuls.
            self.to_k = _make_linear(
                in_dim=text_dim,
                out_dim=dim,
                dtype=dtype,
                device=device,
                has_bias=True,
                quant_config=quant_config,
            )
            self.to_v = _make_linear(
                in_dim=text_dim,
                out_dim=dim,
                dtype=dtype,
                device=device,
                has_bias=True,
                quant_config=quant_config,
            )
        self.norm_q = WanRMSNorm(dim, eps=eps, dtype=dtype, device=device)
        self.norm_k = WanRMSNorm(dim, eps=eps, dtype=dtype, device=device)
        self.to_out = _make_linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )

        # Optional added KV projections for image conditioning (Wan 2.1 I2V)
        if added_kv_proj_dim is not None:
            self.add_k_proj = _make_linear(
                in_dim=added_kv_proj_dim,
                out_dim=dim,
                dtype=dtype,
                device=device,
                has_bias=True,
                quant_config=quant_config,
            )
            self.add_v_proj = _make_linear(
                in_dim=added_kv_proj_dim,
                out_dim=dim,
                dtype=dtype,
                device=device,
                has_bias=True,
                quant_config=quant_config,
            )
            self.norm_added_q = WanRMSNorm(
                dim, eps=eps, dtype=dtype, device=device
            )
            self.norm_added_k = WanRMSNorm(
                dim, eps=eps, dtype=dtype, device=device
            )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        image_embeds: TensorValue | None = None,
    ) -> TensorValue:
        query = self.to_q(hidden_states)

        if not self._fp8:
            kv = self.to_kv(encoder_hidden_states)
            key = kv[:, :, : self.inner_dim]
            value = kv[:, :, self.inner_dim :]
        elif self._fuse_kv:
            # FP8 default: one fused matmul, sliced into K/V.
            kv = self.kv_proj(encoder_hidden_states)
            key = kv[:, :, : self.inner_dim]
            value = kv[:, :, self.inner_dim :]
        else:
            # FP8, fusion disabled: two per-projection matmuls.
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

        # QK-norm across all heads (before reshape)
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Added image KV (Wan 2.1 I2V)
        if self._has_added_kv and image_embeds is not None:
            added_key = self.norm_added_k(self.add_k_proj(image_embeds))
            added_value = self.add_v_proj(image_embeds)
            # Concatenate image KV with text KV along sequence dim
            key = ops.concat([key, added_key], axis=1)
            value = ops.concat([value, added_value], axis=1)

        # Reshape to multi-head
        batch_size = query.shape[0]
        q_seq_len = query.shape[1]
        kv_seq_len = key.shape[1]
        query = ops.reshape(
            query, [batch_size, q_seq_len, self.num_heads, self.head_dim]
        )
        key = ops.reshape(
            key, [batch_size, kv_seq_len, self.num_heads, self.head_dim]
        )
        value = ops.reshape(
            value, [batch_size, kv_seq_len, self.num_heads, self.head_dim]
        )

        # Flash attention (no RoPE for cross-attention)
        scale = 1.0 / (self.head_dim**0.5)
        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )

        # Reshape back: [B, S, H, head_dim] -> [B, S, D]
        hidden_states = ops.reshape(
            hidden_states,
            [hidden_states.shape[0], hidden_states.shape[1], self.inner_dim],
        )

        return self.to_out(hidden_states)


class WanFeedForward(Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        # WAN uses "gelu-approximate" (simple GELU), NOT GEGLU.
        # ffn_dim is the direct projection output size (no 2x expansion).
        self.proj = _make_linear(
            in_dim=dim,
            out_dim=ffn_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )
        self.linear_out = _make_linear(
            in_dim=ffn_dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=quant_config,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        hidden = self.proj(x)
        hidden = ops.gelu(hidden, approximate="tanh")
        return self.linear_out(hidden)


class WanTransformerBlockSequence(Module):
    """All transformer blocks as a single module for single-graph compilation.

    Wrapping all blocks in one module means ``session.load()`` is called
    once, so the runtime allocates a single shared workspace instead of
    40 independent ones -- reducing peak VRAM by ~40x.
    """

    def __init__(self, blocks: list[WanTransformerBlock]) -> None:
        super().__init__()
        self.blocks = LayerList(blocks)

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep_proj: TensorValue,
        rope_cos: TensorValue,
        rope_sin: TensorValue,
    ) -> TensorValue:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rope_cos,
                rope_sin,
            )
        return hidden_states


class WanTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        text_dim: int,
        cross_attn_norm: bool,
        eps: float,
        *,
        added_kv_proj_dim: int | None = None,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.scale_shift_table = Weight(
            "scale_shift_table", dtype, [1, 6, dim], device
        )
        self.norm1 = WanLayerNorm(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.attn1 = WanSelfAttention(
            dim,
            num_heads,
            head_dim,
            eps,
            dtype=dtype,
            device=device,
            quant_config=quant_config,
        )
        self.norm2 = WanLayerNorm(
            dim,
            eps=eps,
            elementwise_affine=cross_attn_norm,
            use_bias=cross_attn_norm,
            dtype=dtype,
            device=device,
        )
        self.attn2 = WanCrossAttention(
            dim,
            text_dim,
            num_heads,
            head_dim,
            eps,
            added_kv_proj_dim=added_kv_proj_dim,
            dtype=dtype,
            device=device,
            quant_config=quant_config,
        )
        self.norm3 = WanLayerNorm(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.ffn = WanFeedForward(
            dim, ffn_dim, dtype=dtype, device=device, quant_config=quant_config
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep_proj: TensorValue,
        rope_cos: TensorValue,
        rope_sin: TensorValue,
        image_embeds: TensorValue | None = None,
    ) -> TensorValue:
        rotary_emb = (rope_cos, rope_sin)

        # Modulation: scale_shift_table[1,6,D] + timestep_proj[B,6,D]
        mod = self.scale_shift_table + timestep_proj  # [B, 6, D]

        # Split into 6 modulation parameters
        shift_sa, scale_sa, gate_sa = (
            mod[:, 0:1, :],
            mod[:, 1:2, :],
            mod[:, 2:3, :],
        )
        shift_ff, scale_ff, gate_ff = (
            mod[:, 3:4, :],
            mod[:, 4:5, :],
            mod[:, 5:6, :],
        )

        # Self-attention
        x = self.norm1(hidden_states)
        x = x * (1 + scale_sa) + shift_sa
        x = self.attn1(x, rotary_emb)
        hidden_states = hidden_states + gate_sa * x

        # Cross-attention (with optional image KV for 2.1 I2V)
        x = self.norm2(hidden_states)
        x = self.attn2(x, encoder_hidden_states, image_embeds=image_embeds)
        hidden_states = hidden_states + x

        # Feed-forward
        x = self.norm3(hidden_states)
        x = x * (1 + scale_ff) + shift_ff
        x = self.ffn(x)
        hidden_states = hidden_states + gate_ff * x

        return hidden_states


class WanTransformerPreProcess(Module):
    """Patch embedding + condition embedding (compiled separately).

    Accepts f32 latents and an optional I2V condition tensor. The graph
    casts to model dtype and (for I2V) concatenates the condition along
    the channel axis before patch embedding.
    """

    def __init__(
        self,
        config: WanConfigBase,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> None:
        super().__init__()
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim
        self._dtype = dtype

        self.patch_embedding = WanConv3d(
            kernel_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=dim,
            stride=config.patch_size,
            dtype=dtype,
            device=device,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            num_layers=config.num_layers,
            image_dim=getattr(config, "image_dim", None),
            dtype=dtype,
            device=device,
            quant_config=config.quant_config,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        timestep: TensorValue,
        encoder_hidden_states: TensorValue,
        i2v_condition: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
        # Cast f32 latents to model dtype.
        hidden_states = ops.cast(hidden_states, self._dtype)

        # Broadcast latents/timestep to match encoder_hidden_states' batch
        # axis. For batched CFG the caller supplies B=1 latents/timestep
        # with a pre-concatenated [cond; uncond] B=2 text embedding, and
        # this fold avoids needing a separate pack graph. For non-CFG /
        # I2V the broadcast is an identity because all three inputs share
        # the same batch. ``ops.broadcast_to`` is used (not ``ops.tile``)
        # because tile currently falls back to CPU on GPU graphs.
        target_batch = encoder_hidden_states.shape[0]
        hidden_states = ops.broadcast_to(
            hidden_states,
            [
                target_batch,
                hidden_states.shape[1],
                hidden_states.shape[2],
                hidden_states.shape[3],
                hidden_states.shape[4],
            ],
        )
        timestep = ops.broadcast_to(timestep, [target_batch])

        # Concat I2V condition along channel axis if present.
        if i2v_condition is not None:
            hidden_states = ops.concat([hidden_states, i2v_condition], axis=1)

        batch_size = hidden_states.shape[0]
        hs = ops.permute(hidden_states, [0, 2, 3, 4, 1])
        hs = self.patch_embedding(hs)
        hs = ops.permute(hs, [0, 4, 1, 2, 3])
        seq_len = hs.shape[2] * hs.shape[3] * hs.shape[4]
        hs = ops.reshape(hs, [batch_size, self.inner_dim, seq_len])
        hs = ops.permute(hs, [0, 2, 1])

        temb, timestep_proj, text_emb = self.condition_embedder(
            timestep, encoder_hidden_states
        )
        return hs, temb, timestep_proj, text_emb


class WanTransformerPostProcess(Module):
    """Output modulation + unpatchify (compiled separately)."""

    def __init__(
        self,
        config: WanConfigBase,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> None:
        super().__init__()
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size

        self.scale_shift_table = Weight(
            "scale_shift_table", dtype, [1, 2, dim], device
        )
        self.norm_out = WanLayerNorm(
            dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.proj_out = _make_linear(
            in_dim=dim,
            out_dim=config.out_channels * prod(config.patch_size),
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=config.quant_config,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        temb: TensorValue,
        spatial_shape: TensorValue,
    ) -> TensorValue:
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.patch_size
        ppf = spatial_shape.shape[0]
        pph = spatial_shape.shape[1]
        ppw = spatial_shape.shape[2]

        mod = self.scale_shift_table + ops.reshape(
            temb, [batch_size, 1, self.inner_dim]
        )
        shift = mod[:, :1, :]
        scale = mod[:, 1:, :]
        hs = self.norm_out(hidden_states) * (1.0 + scale) + shift
        hs = self.proj_out(hs)
        hs = ops.rebind(
            hs,
            shape=[
                batch_size,
                ppf * pph * ppw,
                self.out_channels * p_t * p_h * p_w,
            ],
        )

        hs = ops.reshape(
            hs,
            [batch_size, ppf, pph, ppw, p_t, p_h, p_w, self.out_channels],
        )
        hs = ops.permute(hs, [0, 7, 1, 4, 2, 5, 3, 6])
        hs = ops.reshape(
            hs,
            [batch_size, self.out_channels, ppf * p_t, pph * p_h, ppw * p_w],
        )
        return ops.cast(hs, DType.bfloat16)


class WanTransformer3DModel(Module):
    """Full transformer (for reference / single-graph compilation)."""

    def __init__(
        self,
        config: WanConfigBase,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
    ) -> None:
        super().__init__()
        self.config = config
        dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size

        self.patch_embedding = WanConv3d(
            kernel_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=dim,
            stride=config.patch_size,
            dtype=dtype,
            device=device,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            num_layers=config.num_layers,
            image_dim=getattr(config, "image_dim", None),
            dtype=dtype,
            device=device,
            quant_config=config.quant_config,
        )
        self.blocks = LayerList(
            [
                WanTransformerBlock(
                    dim=dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    text_dim=dim,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                    dtype=dtype,
                    device=device,
                    quant_config=config.quant_config,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.scale_shift_table = Weight(
            "scale_shift_table", dtype, [1, 2, dim], device
        )
        self.norm_out = WanLayerNorm(
            dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.proj_out = _make_linear(
            in_dim=dim,
            out_dim=config.out_channels * prod(config.patch_size),
            dtype=dtype,
            device=device,
            has_bias=True,
            quant_config=config.quant_config,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        timestep: TensorValue,
        encoder_hidden_states: TensorValue,
        rope_cos: TensorValue,
        rope_sin: TensorValue,
    ) -> TensorValue:
        batch_size = hidden_states.shape[0]
        orig_T = hidden_states.shape[2]
        orig_H = hidden_states.shape[3]
        orig_W = hidden_states.shape[4]
        p_t, p_h, p_w = self.patch_size
        ppf = orig_T // p_t
        pph = orig_H // p_h
        ppw = orig_W // p_w

        hs = ops.permute(hidden_states, [0, 2, 3, 4, 1])
        hs = self.patch_embedding(hs)
        hs = ops.permute(hs, [0, 4, 1, 2, 3])
        hs = ops.reshape(hs, [batch_size, self.inner_dim, ppf * pph * ppw])
        hs = ops.permute(hs, [0, 2, 1])

        temb, timestep_proj, text_emb = self.condition_embedder(
            timestep, encoder_hidden_states
        )

        # Rebind RoPE to match the sequence length derived from spatial dims.
        seq_len = ppf * pph * ppw
        rope_cos = ops.rebind(rope_cos, shape=[seq_len, self.head_dim])
        rope_sin = ops.rebind(rope_sin, shape=[seq_len, self.head_dim])

        for block in self.blocks:
            hs = block(hs, text_emb, timestep_proj, rope_cos, rope_sin)

        mod = self.scale_shift_table + ops.reshape(
            temb, [batch_size, 1, self.inner_dim]
        )
        shift = mod[:, :1, :]
        scale = mod[:, 1:, :]
        hs = self.norm_out(hs) * (1.0 + scale) + shift
        hs = self.proj_out(hs)

        hs = ops.reshape(
            hs,
            [batch_size, ppf, pph, ppw, p_t, p_h, p_w, self.out_channels],
        )
        hs = ops.permute(hs, [0, 7, 1, 4, 2, 5, 3, 6])
        hs = ops.reshape(
            hs,
            [batch_size, self.out_channels, ppf * p_t, pph * p_h, ppw * p_w],
        )
        return ops.cast(hs, self.config.dtype)
