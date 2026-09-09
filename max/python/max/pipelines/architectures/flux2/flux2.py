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

from dataclasses import dataclass, field

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    ShardingStrategy,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.comm.allreduce import Allreduce
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig
from max.nn.transformer.transformer import forward_sharded_layers

from .layers.embeddings import TimestepEmbedding, Timesteps
from .layers.flux2_attention import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2ParallelSelfAttention,
    Flux2PosEmbed,
)
from .layers.normalizations import AdaLayerNormContinuous, LayerNorm
from .model_config import Flux2BlockQuant, Flux2Config


class Flux2TimestepGuidanceEmbeddings(Module):
    def __init__(
        self,
        *,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialize Flux2TimestepGuidanceEmbeddings.

        Args:
            in_channels: Number of sinusoidal channels.
            embedding_dim: Output embedding dimension.
            bias: Whether to use bias in MLP layers.
            guidance_embeds: If True, include guidance embedder.
            dtype: Weight dtype.
            device: Weight device.
        """
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
            dtype=dtype,
            device=device,
        )
        if guidance_embeds:
            self.guidance_embedder: TimestepEmbedding | None = (
                TimestepEmbedding(
                    in_channels=in_channels,
                    time_embed_dim=embedding_dim,
                    sample_proj_bias=bias,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            self.guidance_embedder = None

    def __call__(
        self, timestep: TensorValue, guidance: TensorValue
    ) -> TensorValue:
        """Compute combined timestep and guidance embeddings.

        Args:
            timestep: Timestep values of shape [B].
            guidance: Guidance scale values of shape [B].

        Returns:
            Combined embedding of shape [B, embedding_dim].
        """
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            ops.cast(timesteps_proj, timestep.dtype)
        )
        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(
                ops.cast(guidance_proj, guidance.dtype)
            )
            return timesteps_emb + guidance_emb
        return timesteps_emb


class Flux2Modulation(Module):
    def __init__(
        self,
        dim: int,
        *,
        dtype: DType,
        device: DeviceRef,
        mod_param_sets: int = 2,
        bias: bool = False,
    ) -> None:
        """Initialize Flux2Modulation.

        Args:
            dim: Input/output dimension.
            dtype: Weight dtype.
            device: Weight device.
            mod_param_sets: Number of modulation parameter sets.
            bias: Whether to use bias in the linear layer.
        """
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = Linear(
            in_dim=dim,
            out_dim=dim * 3 * mod_param_sets,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )

    def __call__(
        self, temb: TensorValue
    ) -> tuple[tuple[TensorValue, TensorValue, TensorValue], ...]:
        """Generate modulation parameters from timestep embedding.

        Args:
            temb: Timestep embedding of shape [B, dim] or [B, 1, dim].

        Returns:
            Tuple of modulation tuples, each containing (shift, scale, gate).
        """
        mod = self.linear(ops.silu(temb))
        if len(mod.shape) == 2:
            mod = ops.unsqueeze(mod, 1)
        mod_params = ops.split(
            mod,
            [temb.shape[-1]] * (3 * self.mod_param_sets),
            axis=-1,
        )
        return tuple(
            (mod_params[3 * i], mod_params[3 * i + 1], mod_params[3 * i + 2])
            for i in range(self.mod_param_sets)
        )


class Flux2TransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dtype: DType,
        devices: list[DeviceRef],
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        quant: Flux2BlockQuant = Flux2BlockQuant(),  # noqa: B008
    ) -> None:
        """Initialize Flux2TransformerBlock.

        Args:
            dim: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            dtype: Weight dtype.
            devices: Devices for placement and tensor parallelism. The
                un-sharded base lives on ``devices[0]``; sub-layers are
                sharded across the full list at construction time.
            mlp_ratio: Multiplier for feedforward hidden dimension.
            eps: Epsilon for layer normalization.
            bias: Whether to use bias in linear layers.
            quant: Per-Linear quant plan; default leaves every Linear BF16.
        """
        super().__init__()
        self.devices = list(devices)
        device = self.devices[0]
        num_devices = len(self.devices)

        # The Flux2 LayerNorm shim has no learnable parameters with
        # ``elementwise_affine=False`` and ``use_bias=False``, so calling it
        # on a tensor on any device runs the normalization on that device.
        # If either flag is ever flipped, this code must shard the norms too:
        # the parameters would live on ``devices[0]`` and silently mismatch
        # tensors on other devices.
        self.norm1 = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.norm1_context = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.norm2 = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )
        self.norm2_context = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )

        tp_strategy = (
            ShardingStrategy.tensor_parallel(num_devices)
            if num_devices > 1
            else ShardingStrategy.replicate(1)
        )

        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            dtype=dtype,
            devices=self.devices,
            quant=quant,
        )
        self.attn.sharding_strategy = tp_strategy
        self.attn_shards = self.attn.shard(self.devices)

        self.ff = Flux2FeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            devices=self.devices,
            quant_config=quant.ff,
        )
        self.ff.sharding_strategy = tp_strategy
        self.ff_shards = self.ff.shard(self.devices)

        self.ff_context = Flux2FeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            devices=self.devices,
            quant_config=quant.ff_context,
        )
        self.ff_context.sharding_strategy = tp_strategy
        self.ff_context_shards = self.ff_context.shard(self.devices)

        self.allreduce = Allreduce(num_accelerators=num_devices)

    def _replicate(self, t: TensorValue) -> list[TensorValue]:
        """Returns ``t`` as a list with one copy per device.

        For single-device blocks this is a no-op. For multi-device, each
        device gets its own copy via ``transfer_to``.
        """
        if len(self.devices) == 1:
            return [t]
        return [t.to(dev) for dev in self.devices]

    def __call__(
        self,
        hidden_states: list[TensorValue],
        encoder_hidden_states: list[TensorValue],
        temb_mod_params_img: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        temb_mod_params_txt: tuple[
            tuple[TensorValue, TensorValue, TensorValue],
            tuple[TensorValue, TensorValue, TensorValue],
        ],
        signal_buffers: list[BufferValue],
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        """Forward pass for dual-stream transformer block.

        Args:
            hidden_states: Per-device image tokens of shape [B, S_img, D].
            encoder_hidden_states: Per-device text tokens of shape
                [B, S_txt, D].
            temb_mod_params_img: Image-stream modulation parameters
                (computed once on device 0; replicated per-device internally).
            temb_mod_params_txt: Text-stream modulation parameters (same
                replication semantics).
            signal_buffers: Per-device communication buffers for allreduce.
                Pass ``[]`` when there is only one device.
            image_rotary_emb: Optional (cos, sin) tuple for rotary embeddings;
                replicated per-device internally.

        Returns:
            Tuple of per-device (encoder_hidden_states, hidden_states) lists.
        """
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = (
            temb_mod_params_img
        )
        (
            (c_shift_msa, c_scale_msa, c_gate_msa),
            (c_shift_mlp, c_scale_mlp, c_gate_mlp),
        ) = temb_mod_params_txt

        # Replicate modulation params across devices once.
        shift_msa_d = self._replicate(shift_msa)
        scale_msa_d = self._replicate(scale_msa)
        gate_msa_d = self._replicate(gate_msa)
        shift_mlp_d = self._replicate(shift_mlp)
        scale_mlp_d = self._replicate(scale_mlp)
        gate_mlp_d = self._replicate(gate_mlp)
        c_shift_msa_d = self._replicate(c_shift_msa)
        c_scale_msa_d = self._replicate(c_scale_msa)
        c_gate_msa_d = self._replicate(c_gate_msa)
        c_shift_mlp_d = self._replicate(c_shift_mlp)
        c_scale_mlp_d = self._replicate(c_scale_mlp)
        c_gate_mlp_d = self._replicate(c_gate_mlp)

        rope_d: list[tuple[TensorValue, TensorValue] | None]
        if image_rotary_emb is not None:
            cos_d = self._replicate(image_rotary_emb[0])
            sin_d = self._replicate(image_rotary_emb[1])
            rope_d = list(zip(cos_d, sin_d, strict=True))
        else:
            rope_d = [None] * len(self.devices)

        # Pre-attention norm + modulation, per-device.
        norm_hidden_states = [
            (1 + scale_msa_d[i]) * self.norm1(hidden_states[i]) + shift_msa_d[i]
            for i in range(len(self.devices))
        ]
        norm_encoder_hidden_states = [
            (1 + c_scale_msa_d[i])
            * self.norm1_context(encoder_hidden_states[i])
            + c_shift_msa_d[i]
            for i in range(len(self.devices))
        ]

        # Dual-stream attention per shard. Each shard returns
        # (image_partial, text_partial); both need to be allreduced.
        attn_pairs = [
            self.attn_shards[i](
                norm_hidden_states[i],
                norm_encoder_hidden_states[i],
                image_rotary_emb=rope_d[i],
            )
            for i in range(len(self.devices))
        ]
        attn_output_partials = []
        context_attn_output_partials = []
        for pair in attn_pairs:
            if not isinstance(pair, tuple):
                raise ValueError("Expected tuple from dual-stream attention")
            attn_output_partials.append(pair[0])
            context_attn_output_partials.append(pair[1])
        if len(self.devices) > 1:
            attn_outputs = self.allreduce(attn_output_partials, signal_buffers)
            context_attn_outputs = self.allreduce(
                context_attn_output_partials, signal_buffers
            )
        else:
            attn_outputs = attn_output_partials
            context_attn_outputs = context_attn_output_partials

        # Image stream: residual + gate * attn -> norm2 + modulation -> ff
        # -> allreduce -> residual + gate * ff.
        hidden_states = [
            hidden_states[i] + gate_msa_d[i] * attn_outputs[i]
            for i in range(len(self.devices))
        ]
        norm_hidden_states = [
            self.norm2(hidden_states[i]) * (1 + scale_mlp_d[i]) + shift_mlp_d[i]
            for i in range(len(self.devices))
        ]
        ff_outputs_partial = forward_sharded_layers(
            self.ff_shards, norm_hidden_states
        )
        ff_outputs = (
            self.allreduce(ff_outputs_partial, signal_buffers)
            if len(self.devices) > 1
            else ff_outputs_partial
        )
        hidden_states = [
            hidden_states[i] + gate_mlp_d[i] * ff_outputs[i]
            for i in range(len(self.devices))
        ]

        # Text stream: same shape as image stream.
        encoder_hidden_states = [
            encoder_hidden_states[i] + c_gate_msa_d[i] * context_attn_outputs[i]
            for i in range(len(self.devices))
        ]
        norm_encoder_hidden_states = [
            self.norm2_context(encoder_hidden_states[i])
            * (1 + c_scale_mlp_d[i])
            + c_shift_mlp_d[i]
            for i in range(len(self.devices))
        ]
        context_ff_outputs_partial = forward_sharded_layers(
            self.ff_context_shards, norm_encoder_hidden_states
        )
        context_ff_outputs = (
            self.allreduce(context_ff_outputs_partial, signal_buffers)
            if len(self.devices) > 1
            else context_ff_outputs_partial
        )
        encoder_hidden_states = [
            encoder_hidden_states[i] + c_gate_mlp_d[i] * context_ff_outputs[i]
            for i in range(len(self.devices))
        ]

        # Float16 saturation, per-device.
        if encoder_hidden_states[0].dtype == DType.float16:
            encoder_hidden_states = [
                ops.min(ops.max(h, -65504), 65504)
                for h in encoder_hidden_states
            ]

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dtype: DType,
        devices: list[DeviceRef],
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
        quant_config: QuantConfig | None = None,
    ) -> None:
        """Initialize Flux2SingleTransformerBlock.

        Args:
            dim: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension of each attention head.
            dtype: Weight dtype.
            devices: Devices for placement and tensor parallelism. The
                un-sharded base lives on ``devices[0]``; ``self.attn`` is
                sharded across the full list at construction time.
            mlp_ratio: Multiplier for feedforward hidden dimension.
            eps: Epsilon for layer normalization.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.devices = list(devices)
        device = self.devices[0]
        num_devices = len(self.devices)

        self.norm = LayerNorm(
            dim,
            dtype=dtype,
            device=device,
            eps=eps,
            elementwise_affine=False,
            use_bias=False,
        )

        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
            dtype=dtype,
            devices=self.devices,
            quant_config=quant_config,
        )
        self.attn.sharding_strategy = (
            ShardingStrategy.tensor_parallel(num_devices)
            if num_devices > 1
            else ShardingStrategy.replicate(1)
        )
        self.attn_shards = self.attn.shard(self.devices)

        self.allreduce = Allreduce(num_accelerators=num_devices)

    def _replicate(self, t: TensorValue) -> list[TensorValue]:
        """Returns ``t`` as a list with one copy per device (no-op for 1)."""
        if len(self.devices) == 1:
            return [t]
        return [t.to(dev) for dev in self.devices]

    def __call__(
        self,
        hidden_states: list[TensorValue],
        encoder_hidden_states: list[TensorValue] | None = None,
        temb_mod_params: tuple[
            TensorValue,
            TensorValue,
            TensorValue,
        ]
        | None = None,
        signal_buffers: list[BufferValue] | None = None,
        image_rotary_emb: tuple[TensorValue, TensorValue] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | Dim | None = None,
    ) -> list[TensorValue] | tuple[list[TensorValue], list[TensorValue]]:
        """Forward pass for single-stream transformer block.

        Args:
            hidden_states: Per-device image tokens, or per-device concatenated
                text+image tokens when ``encoder_hidden_states`` is None.
            encoder_hidden_states: Optional per-device text tokens to
                concatenate with ``hidden_states``.
            temb_mod_params: (shift, scale, gate) tuple of single tensors;
                replicated per-device internally.
            signal_buffers: Per-device communication buffers for allreduce.
                Pass ``[]`` (or omit) when there is only one device.
            image_rotary_emb: Optional (cos, sin) tuple for rotary embeddings;
                replicated per-device internally.
            split_hidden_states: If True, split output back into text and image.
            text_seq_len: Length of text sequence when splitting.

        Returns:
            Either per-device concatenated hidden states, or a per-device
            (encoder_hidden_states, hidden_states) tuple when splitting.
        """
        if signal_buffers is None:
            signal_buffers = []

        num_devices = len(self.devices)
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states[0].shape[1]
            hidden_states = [
                ops.concat([encoder_hidden_states[i], hidden_states[i]], axis=1)
                for i in range(num_devices)
            ]

        if temb_mod_params is None:
            raise ValueError("temb_mod_params cannot be None")
        mod_shift, mod_scale, mod_gate = temb_mod_params
        mod_shift_d = self._replicate(mod_shift)
        mod_scale_d = self._replicate(mod_scale)
        mod_gate_d = self._replicate(mod_gate)

        rope_d: list[tuple[TensorValue, TensorValue] | None]
        if image_rotary_emb is not None:
            cos_d = self._replicate(image_rotary_emb[0])
            sin_d = self._replicate(image_rotary_emb[1])
            rope_d = list(zip(cos_d, sin_d, strict=True))
        else:
            rope_d = [None] * num_devices

        norm_hidden_states = [
            (1 + mod_scale_d[i]) * self.norm(hidden_states[i]) + mod_shift_d[i]
            for i in range(num_devices)
        ]

        attn_partials = [
            self.attn_shards[i](
                norm_hidden_states[i],
                image_rotary_emb=rope_d[i],
            )
            for i in range(num_devices)
        ]
        # ``Flux2ParallelSelfAttention.__call__`` returns a single tensor; the
        # iterator above produces per-device partial sums.
        attn_outputs = (
            self.allreduce(attn_partials, signal_buffers)
            if num_devices > 1
            else attn_partials
        )

        hidden_states = [
            hidden_states[i] + mod_gate_d[i] * attn_outputs[i]
            for i in range(num_devices)
        ]

        if hidden_states[0].dtype == DType.float16:
            hidden_states = [
                ops.min(ops.max(h, -65504), 65504) for h in hidden_states
            ]

        if split_hidden_states:
            if text_seq_len is None:
                raise ValueError("text_seq_len is required when splitting")
            encoder_out = [h[:, :text_seq_len, :] for h in hidden_states]
            image_out = [h[:, text_seq_len:, :] for h in hidden_states]
            return encoder_out, image_out
        return hidden_states


@dataclass
class Flux2TransformerPreamble:
    """Intermediate state produced by :meth:`Flux2Transformer2DModel.forward_preamble`.

    Bundles the per-device streams and the shared conditioning tensors that
    every block stack and the postamble consume, so the transformer forward
    can be decomposed into (preamble, first-block, remaining-blocks,
    postamble) seams for First-Block-Cache without threading a dozen
    positional args through each seam.
    """

    hidden_states_d: list[TensorValue]
    """Per-device image latent streams after ``x_embedder`` (dim=inner_dim)."""
    encoder_hidden_states_d: list[TensorValue]
    """Per-device text streams after ``context_embedder`` (dim=inner_dim)."""
    temb: TensorValue
    """Timestep+guidance embedding, conditioning for modulation + norm_out."""
    double_stream_mod_img: tuple[
        tuple[TensorValue, TensorValue, TensorValue],
        tuple[TensorValue, TensorValue, TensorValue],
    ]
    """Image-side (shift, scale, gate) modulation for double-stream blocks."""
    double_stream_mod_txt: tuple[
        tuple[TensorValue, TensorValue, TensorValue],
        tuple[TensorValue, TensorValue, TensorValue],
    ]
    """Text-side (shift, scale, gate) modulation for double-stream blocks."""
    single_stream_mod: tuple[TensorValue, TensorValue, TensorValue]
    """Modulation for single-stream blocks."""
    image_rotary_emb: tuple[TensorValue, TensorValue]
    """Shared rope (cos, sin) over concatenated (txt, img) ids."""
    num_txt_tokens: Dim
    """Text token count, used by the postamble to slice off the text half."""
    signal_buffers: list[BufferValue] = field(default_factory=list)
    """Per-device allreduce scratch (empty on single device)."""


class Flux2Transformer2DModel(Module):
    def __init__(self, config: Flux2Config) -> None:
        """Initialize Flux2Transformer2DModel.

        Args:
            config: Flux2 configuration containing model dimensions,
                attention settings, and device/dtype information.
        """
        super().__init__()
        patch_size = config.patch_size
        in_channels = config.in_channels
        out_channels = config.out_channels
        num_layers = config.num_layers
        num_single_layers = config.num_single_layers
        attention_head_dim = config.attention_head_dim
        num_attention_heads = config.num_attention_heads
        joint_attention_dim = config.joint_attention_dim
        timestep_guidance_channels = config.timestep_guidance_channels
        mlp_ratio = config.mlp_ratio
        axes_dims_rope = config.axes_dims_rope
        rope_theta = config.rope_theta
        devices = config.devices
        device = devices[0]
        dtype = config.dtype
        eps = config.eps
        quant_config = config.quant_config
        nvfp4_layers_bfl: frozenset[str] = getattr(
            config, "nvfp4_layers_bfl", frozenset()
        )

        self.devices = devices
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.max_dtype = dtype
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim

        self.pos_embed = Flux2PosEmbed(
            theta=rope_theta, axes_dim=axes_dims_rope
        )
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=getattr(config, "guidance_embeds", True),
            dtype=dtype,
            device=device,
        )
        self.double_stream_modulation_img = Flux2Modulation(
            self.inner_dim,
            dtype=dtype,
            device=device,
            mod_param_sets=2,
            bias=False,
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            self.inner_dim,
            dtype=dtype,
            device=device,
            mod_param_sets=2,
            bias=False,
        )
        self.single_stream_modulation = Flux2Modulation(
            self.inner_dim,
            dtype=dtype,
            device=device,
            mod_param_sets=1,
            bias=False,
        )
        self.x_embedder = Linear(
            in_dim=in_channels,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.context_embedder = Linear(
            in_dim=joint_attention_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.transformer_blocks = LayerList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    devices=devices,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                    quant=Flux2BlockQuant.resolve(
                        i, quant_config, nvfp4_layers_bfl
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.single_transformer_blocks = LayerList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    devices=devices,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                    quant_config=quant_config,
                )
                for _ in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=self.inner_dim,
            conditioning_embedding_dim=self.inner_dim,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
            eps=eps,
            bias=False,
        )
        self.proj_out = Linear(
            in_dim=self.inner_dim,
            out_dim=patch_size * patch_size * self.out_channels,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model with symbolic shapes.

        Returns:
            Tuple of TensorType specifications for all model inputs.
        """
        primary = self.devices[0]
        return (
            TensorType(
                self.max_dtype,
                shape=["batch_size", "image_seq_len", self.in_channels],
                device=primary,
            ),
            TensorType(
                self.max_dtype,
                shape=["batch_size", "text_seq_len", self.joint_attention_dim],
                device=primary,
            ),
            TensorType(self.max_dtype, shape=["batch_size"], device=primary),
            TensorType(
                DType.int64,
                shape=["batch_size", "image_seq_len", 4],
                device=primary,
            ),
            TensorType(
                DType.int64,
                shape=["batch_size", "text_seq_len", 4],
                device=primary,
            ),
            TensorType(self.max_dtype, shape=["batch_size"], device=primary),
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        txt_ids: TensorValue,
        guidance: TensorValue,
        *,
        signal_buffers: list[BufferValue] | None = None,
    ) -> tuple[TensorValue]:
        """Forward pass through Flux2 Transformer.

        Args:
            hidden_states: Image latents of shape [B, H*W, in_channels].
            encoder_hidden_states: Text embeddings of shape [B, txt_len, joint_attention_dim].
            timestep: Denoising timestep of shape [B].
            img_ids: Image position IDs of shape [batch_size, image_seq_len, 4]
                or [image_seq_len, 4].
            txt_ids: Text position IDs of shape [batch_size, text_seq_len, 4]
                or [text_seq_len, 4].
            guidance: Guidance scale of shape [B].
            signal_buffers: Per-device communication buffers used by the
                transformer blocks for allreduce. Required when running
                multi-device; ignored on single-device (the blocks bypass
                allreduce when ``num_devices == 1``).

        Returns:
            Denoised output of shape [B, H*W, patch_size^2 * out_channels].
        """
        preamble = self.forward_preamble(
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            guidance,
            txt_ids,
            signal_buffers=signal_buffers,
        )
        state_after_first = self.run_first_block(preamble)
        hidden_states_d = self.run_remaining_blocks(preamble, state_after_first)
        output = self.forward_postamble(preamble, hidden_states_d)
        return (output,)

    # -- First-block-cache seams ---------------------------------------------
    #
    # The single fused ``__call__`` above is decomposed into four reusable
    # pieces so the executor's First-Block-Cache (FBCache) path can compute
    # the block-0 residual, then conditionally skip the remaining blocks +
    # postamble via ``ops.cond``.  The standard forward calls all four in
    # sequence, so the numerics are byte-identical to the pre-refactor path.

    def forward_preamble(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep: TensorValue,
        img_ids: TensorValue,
        guidance: TensorValue,
        txt_ids: TensorValue,
        *,
        signal_buffers: list[BufferValue] | None = None,
    ) -> Flux2TransformerPreamble:
        """Embed inputs and prepare per-device streams before block 0.

        Runs the timestep/guidance embedder, the three modulation heads,
        the ``x_embedder``/``context_embedder`` projections, and rope, then
        replicates the image and text streams across devices.  Returns
        everything the block stacks and the postamble need.
        """
        if signal_buffers is None:
            signal_buffers = []
        if img_ids.rank == 3:
            img_ids = img_ids[0]
        if txt_ids.rank == 3:
            txt_ids = txt_ids[0]

        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = ops.cast(timestep * 1000.0, hidden_states.dtype)
        guidance = ops.cast(guidance * 1000.0, hidden_states.dtype)
        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img_tuple = self.double_stream_modulation_img(temb)
        double_stream_mod_txt_tuple = self.double_stream_modulation_txt(temb)
        single_stream_mod_tuple = self.single_stream_modulation(temb)
        double_stream_mod_img = (
            double_stream_mod_img_tuple[0],
            double_stream_mod_img_tuple[1],
        )
        double_stream_mod_txt = (
            double_stream_mod_txt_tuple[0],
            double_stream_mod_txt_tuple[1],
        )
        single_stream_mod = single_stream_mod_tuple[0]

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        ids = ops.concat([txt_ids, img_ids], axis=0)
        image_rotary_emb = self.pos_embed(ids)

        # Replicate stream tensors across devices for the blocks. The
        # un-sharded head/tail (embedders, modulation, norm_out, proj_out)
        # still runs on ``self.devices[0]``; per-device collapse happens
        # after the block stacks via ``hidden_states_d[0]``.
        hidden_states_d: list[TensorValue] = [
            hidden_states.to(dev) for dev in self.devices
        ]
        encoder_hidden_states_d: list[TensorValue] = [
            encoder_hidden_states.to(dev) for dev in self.devices
        ]
        return Flux2TransformerPreamble(
            hidden_states_d=hidden_states_d,
            encoder_hidden_states_d=encoder_hidden_states_d,
            temb=temb,
            double_stream_mod_img=double_stream_mod_img,
            double_stream_mod_txt=double_stream_mod_txt,
            single_stream_mod=single_stream_mod,
            image_rotary_emb=image_rotary_emb,
            num_txt_tokens=num_txt_tokens,
            signal_buffers=list(signal_buffers),
        )

    def run_first_block(
        self, preamble: Flux2TransformerPreamble
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        """Run double-stream block 0.

        Returns the ``(encoder_hidden_states_d, hidden_states_d)`` streams
        after block 0.  The FBCache residual is derived from the image
        stream (``hidden_states_d``) delta across this block.
        """
        return self.transformer_blocks[0](
            hidden_states=preamble.hidden_states_d,
            encoder_hidden_states=preamble.encoder_hidden_states_d,
            temb_mod_params_img=preamble.double_stream_mod_img,
            temb_mod_params_txt=preamble.double_stream_mod_txt,
            signal_buffers=preamble.signal_buffers,
            image_rotary_emb=preamble.image_rotary_emb,
        )

    def run_remaining_blocks(
        self,
        preamble: Flux2TransformerPreamble,
        state_after_first: tuple[list[TensorValue], list[TensorValue]],
    ) -> list[TensorValue]:
        """Run double-stream blocks 1..N and all single-stream blocks.

        Consumes the streams produced by :meth:`run_first_block` and
        returns the concatenated per-device hidden states, ready for the
        postamble.
        """
        encoder_hidden_states_d, hidden_states_d = state_after_first
        for block in list(self.transformer_blocks)[1:]:
            encoder_hidden_states_d, hidden_states_d = block(
                hidden_states=hidden_states_d,
                encoder_hidden_states=encoder_hidden_states_d,
                temb_mod_params_img=preamble.double_stream_mod_img,
                temb_mod_params_txt=preamble.double_stream_mod_txt,
                signal_buffers=preamble.signal_buffers,
                image_rotary_emb=preamble.image_rotary_emb,
            )

        hidden_states_d = [
            ops.concat([encoder_hidden_states_d[i], hidden_states_d[i]], axis=1)
            for i in range(len(self.devices))
        ]

        for single_block in self.single_transformer_blocks:
            single_out = single_block(
                hidden_states=hidden_states_d,
                encoder_hidden_states=None,
                temb_mod_params=preamble.single_stream_mod,
                signal_buffers=preamble.signal_buffers,
                image_rotary_emb=preamble.image_rotary_emb,
                split_hidden_states=False,
            )
            if isinstance(single_out, tuple):
                raise ValueError("Expected concatenated hidden states")
            hidden_states_d = single_out
        return hidden_states_d

    def forward_postamble(
        self,
        preamble: Flux2TransformerPreamble,
        hidden_states_d: list[TensorValue],
    ) -> TensorValue:
        """Final norm + projection into the ``noise_pred`` output.

        Collapses the per-device streams onto ``devices[0]``, slices off
        the text tokens, applies ``norm_out`` (adaptive layer norm on
        ``temb``), then ``proj_out``.
        """
        hidden_states = hidden_states_d[0]
        hidden_states = hidden_states[:, preamble.num_txt_tokens :, :]
        hidden_states = self.norm_out(hidden_states, preamble.temb)
        return self.proj_out(hidden_states)
