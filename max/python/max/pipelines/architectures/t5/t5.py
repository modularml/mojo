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

import math

import numpy as np
from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Embedding, Linear, Module
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import Dim, TensorType

from .model_config import T5Config


class T5LayerNorm(Module[..., Tensor]):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: DType = DType.float32,
    ):
        """Construct a layernorm module in the T5 style. No bias and no subtraction of mean.

        Args:
            hidden_size: Hidden size.
            eps: Epsilon.
            dtype: Data type for the module.
        """
        super().__init__()
        self.weight = Tensor.ones([hidden_size])
        self.variance_epsilon = eps
        self.dtype = dtype

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Process hidden states through the T5 layer norm.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size).

        Returns:
            TensorValue: Output tensor of shape (batch_size, seq_length, hidden_size).
        """
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://huggingface.co/papers/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        hidden_states_f32 = F.cast(hidden_states, DType.float32)
        variance = F.mean(F.pow(hidden_states_f32, 2), axis=-1)
        hidden_states = hidden_states * F.rsqrt(
            variance + self.variance_epsilon
        )

        # convert into half-precision if necessary
        if self.dtype in [DType.float16, DType.bfloat16]:
            hidden_states = F.cast(hidden_states, self.dtype)

        return self.weight * hidden_states


class T5DenseActDense(Module[..., Tensor]):
    def __init__(
        self,
        config: T5Config,
    ):
        """Construct a dense-activation-dense module.

        Args:
            config: T5 configuration for feed-forward dimensions and dtype.
        """
        super().__init__()
        self.wi = Linear(
            config.d_model,
            config.d_ff,
            bias=False,
        )
        self.wo = Linear(
            config.d_ff,
            config.d_model,
            bias=False,
        )
        self.act_fn = lambda x: (
            0.5
            * x
            * (
                1.0
                + F.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * F.pow(x, 3.0))
                )
            )
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Process hidden states through the dense-activation-dense block.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size).

        Returns:
            TensorValue: Output tensor of shape (batch_size, seq_length, hidden_size).
        """
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(Module[..., Tensor]):
    def __init__(
        self,
        config: T5Config,
    ):
        """Construct a dense-gated-activation-dense module.

        Args:
            config: T5 configuration for feed-forward dimensions and dtype.
        """
        super().__init__()
        self.wi_0 = Linear(
            config.d_model,
            config.d_ff,
            bias=False,
        )
        self.wi_1 = Linear(
            config.d_model,
            config.d_ff,
            bias=False,
        )
        self.wo = Linear(
            config.d_ff,
            config.d_model,
            bias=False,
        )
        self.act_fn = lambda x: (
            0.5
            * x
            * (
                1.0
                + F.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * F.pow(x, 3.0))
                )
            )
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Process hidden states through the dense-gated-activation-dense block.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size).

        Returns:
            TensorValue: Output tensor of shape (batch_size, seq_length, hidden_size).
        """
        hidden_gelu = self.act_fn(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(Module[..., Tensor]):
    def __init__(
        self,
        config: T5Config,
    ):
        """Construct a feed-forward layer.

        Args:
            config: T5 configuration for gating, dimensions, and dtype.
        """
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense: T5DenseGatedActDense | T5DenseActDense = (
                T5DenseGatedActDense(config)
            )
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Process hidden states through the feed-forward layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size).

        Returns:
            TensorValue: Output tensor of shape (batch_size, seq_length, hidden_size).
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states


class T5Attention(
    Module[
        ..., tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]
    ]
):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias: bool = False,
        layer_idx: int | None = None,
    ):
        """Construct an attention layer.

        Args:
            config: T5 configuration.
            has_relative_attention_bias: Whether to use relative attention bias.
            layer_idx: Index of the layer.
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = (
            config.relative_attention_num_buckets
        )
        self.relative_attention_max_distance = (
            config.relative_attention_max_distance
        )
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.device = config.device
        self.dtype = config.dtype

        self.q = Linear(
            self.d_model,
            self.inner_dim,
            bias=False,
        )
        self.k = Linear(
            self.d_model,
            self.inner_dim,
            bias=False,
        )
        self.v = Linear(
            self.d_model,
            self.inner_dim,
            bias=False,
        )
        self.o = Linear(
            self.inner_dim,
            self.d_model,
            bias=False,
        )

        if self.has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets,
                dim=self.n_heads,
            )

    def _relative_position_bucket(
        self,
        relative_position: Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> Tensor:
        """Compute relative position bucket.

        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Args:
            relative_position: Tensor with relative positions.
            bidirectional: Whether the attention is bidirectional.
            num_buckets: Number of buckets.
            max_distance: Maximum distance for relative positions.

        Returns:
            TensorValue: Relative position buckets.
        """
        relative_buckets = Tensor(
            0, dtype=DType.int32, device=relative_position.device
        )

        if bidirectional:
            num_buckets = num_buckets // 2
            is_positive = F.greater(relative_position, 0)
            relative_buckets = relative_buckets + (
                F.cast(is_positive, DType.int32) * num_buckets
            )
            relative_position = F.abs(relative_position)
        else:
            relative_position = -F.min(relative_position, 0)

        max_exact = num_buckets // 2
        is_small = F.greater(max_exact, relative_position)

        scale = (num_buckets - max_exact) / math.log(max_distance / max_exact)
        rel_pos_float = F.cast(relative_position, DType.float32)
        val_log = F.log(rel_pos_float / float(max_exact))
        relative_position_if_large = max_exact + F.cast(
            val_log * scale, DType.int32
        )
        relative_position_if_large = F.min(
            relative_position_if_large, num_buckets - 1
        )
        return relative_buckets + F.where(
            is_small, relative_position, relative_position_if_large
        )

    def compute_bias(
        self, query_length: int | Dim, key_length: int | Dim, device: Device
    ) -> Tensor:
        """Compute relative attention bias.

        Args:
            query_length: Length of the query sequence. Can be a dimension or an integer.
            key_length: Length of the key sequence. Can be a dimension or an integer.
            device: Device to compute the bias on.

        Returns:
            TensorValue: Relative attention bias tensor.
        """
        context_position = F.arange(
            0, query_length, step=1, dtype=DType.int32, device=device
        )
        context_position = F.unsqueeze(context_position, 1)

        memory_position = F.arange(
            0, key_length, step=1, dtype=DType.int32, device=device
        )
        memory_position = F.unsqueeze(memory_position, 0)

        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = F.permute(values, (2, 0, 1))
        values = F.unsqueeze(values, 0)
        return values

    def forward(
        self,
        hidden_states: Tensor,
        mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        position_bias: Tensor | None = None,
        past_key_values: Tensor | None = None,
        layer_head_mask: Tensor | None = None,
        query_length: int | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]:
        """Process hidden states through the attention layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size).
            mask: Attention mask.
            key_value_states: Key-value states for cross-attention.
            position_bias: Position bias tensor.
            past_key_values: Past key values for caching (not implemented).
            layer_head_mask: Mask for attention heads.
            query_length: Length of the query sequence.
            use_cache: Whether to use cache (not implemented).
            output_attentions: Whether to return attention weights.
            cache_position: Cache position.

        Returns:
            Tuple[TensorValue, TensorValue] or Tuple[TensorValue, TensorValue, TensorValue]: Output tensor, position bias, and optionally attention weights.
        """
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None
        if is_cross_attention:
            raise NotImplementedError(
                "T5 CrossAttention is not implemented yet."
            )
        if past_key_values is not None:
            raise NotImplementedError(
                "T5 auto regressive model is not implemented yet."
            )

        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        # Reshape to (batch, seq, heads, head_dim)
        query = F.reshape(
            query,
            (batch_size, seq_length, self.n_heads, self.key_value_proj_dim),
        )
        key = F.reshape(
            key, (batch_size, seq_length, self.n_heads, self.key_value_proj_dim)
        )
        value = F.reshape(
            value,
            (batch_size, seq_length, self.n_heads, self.key_value_proj_dim),
        )

        # Transpose to (batch, heads, seq, head_dim)
        query = F.permute(query, (0, 2, 1, 3))
        key = F.permute(key, (0, 2, 1, 3))
        value = F.permute(value, (0, 2, 1, 3))

        scores = F.matmul(query, F.permute(key, (0, 1, 3, 2)))

        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(
                seq_length, seq_length, hidden_states.device
            )

        if position_bias is not None:
            scores = scores + position_bias

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(F.cast(scores, DType.float32), axis=-1)
        attn_weights = F.cast(attn_weights, self.dtype)

        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = F.matmul(attn_weights, value)
        attn_output = F.permute(attn_output, (0, 2, 1, 3))
        attn_output = F.reshape(
            attn_output, (batch_size, seq_length, self.inner_dim)
        )
        attn_output = self.o(attn_output)

        if output_attentions:
            return (attn_output, position_bias, attn_weights)
        return (attn_output, position_bias)


class T5LayerSelfAttention(
    Module[
        ..., tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]
    ]
):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias: bool = False,
        layer_idx: int | None = None,
    ):
        """Construct a self-attention layer.

        Args:
            config: T5 configuration.
            has_relative_attention_bias: Whether to use relative attention bias.
            layer_idx: Index of the layer.
        """
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            layer_idx=layer_idx,
        )
        self.layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_bias: Tensor | None = None,
        layer_head_mask: Tensor | None = None,
        past_key_values: Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]:
        """Process hidden states through the self-attention layer.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size).
            attention_mask: Attention mask.
            position_bias: Position bias tensor.
            layer_head_mask: Mask for attention heads.
            past_key_values: Past key values for caching (not implemented).
            use_cache: Whether to use cache (not implemented).
            output_attentions: Whether to return attention weights.
            cache_position: Cache position.

        Returns:
            Tuple containing output tensor and optionally position bias and attention weights.
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = hidden_states + attention_output[0]
        outputs = (hidden_states,) + attention_output[1:]
        return outputs  # type: ignore[return-value]


class T5Block(
    Module[
        ..., tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]
    ]
):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias: bool = False,
        layer_idx: int | None = None,
    ):
        """Construct a T5 block.

        Args:
            config: T5 configuration.
            has_relative_attention_bias: Whether to use relative attention bias.
            layer_idx: Index of the layer.
        """
        super().__init__()
        layers: list[T5LayerSelfAttention | T5LayerFF] = list()
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            raise NotImplementedError(
                "T5 LayerCrossAttention is not implemented yet."
            )

        layers.append(
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                layer_idx=layer_idx,
            )
        )
        layers.append(T5LayerFF(config))
        self.layer = ModuleList(layers)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_bias: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        encoder_decoder_position_bias: Tensor | None = None,
        cross_attn_layer_head_mask: Tensor | None = None,
        layer_head_mask: Tensor | None = None,
        past_key_values: Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, Tensor]:
        """Process hidden states through the T5 block.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size).
            attention_mask: Attention mask.
            position_bias: Position bias tensor.
            encoder_hidden_states: Encoder hidden states (not implemented).
            encoder_attention_mask: Encoder attention mask (not implemented).
            encoder_decoder_position_bias: Encoder-decoder position bias (not implemented).
            cross_attn_layer_head_mask: Cross attention layer head mask (not implemented).
            layer_head_mask: Mask for attention heads.
            past_key_values: Past key values for caching (not implemented).
            use_cache: Whether to use cache (not implemented).
            output_attentions: Whether to return attention weights.
            cache_position: Cache position.

        Returns:
            Tuple containing output tensor and optionally position bias and attention weights.
        """
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]

        if hidden_states.dtype == DType.float16:
            clamp_value = float(np.finfo(np.float16).max) - 1000
            hidden_states = hidden_states.clip(
                min=-clamp_value, max=clamp_value
            )

        do_cross_attention = (
            self.is_decoder and encoder_hidden_states is not None
        )
        if do_cross_attention:
            raise NotImplementedError(
                "T5 CrossAttention is not implemented yet."
            )

        ff_output = self.layer[-1](hidden_states)
        hidden_states = (
            ff_output if isinstance(ff_output, Tensor) else ff_output[0]
        )
        if hidden_states.dtype == DType.float16:
            clamp_value = float(np.finfo(np.float16).max) - 1000
            hidden_states = hidden_states.clip(
                min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)
        result = outputs + attention_outputs
        return result  # type: ignore[return-value]


class T5Stack(Module[..., Tensor]):
    def __init__(
        self,
        config: T5Config,
        embed_tokens: Embedding | None = None,
    ):
        """Construct a T5 stack.

        Args:
            config: T5 configuration.
            embed_tokens: Embedding module.
        """
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = ModuleList(
            [
                T5Block(
                    config,
                    has_relative_attention_bias=bool(i == 0),
                    layer_idx=i,
                )
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
        )
        self.dropout = config.dropout_rate
        self.device = config.device
        self.dtype = config.dtype

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        encoder_decoder_position_bias: Tensor | None = None,
        cross_attn_layer_head_mask: Tensor | None = None,
        layer_head_mask: Tensor | None = None,
        past_key_values: Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Tensor | None = None,
    ) -> Tensor:
        """Process input through the T5 stack.

        Args:
            input_ids: Input IDs tensor of shape (batch_size, seq_length).
            attention_mask: Attention mask tensor of shape (batch_size, seq_length).
            inputs_embeds: Input embeddings tensor of shape (batch_size, seq_length, hidden_size).
            encoder_hidden_states: Encoder hidden states (not implemented).
            encoder_attention_mask: Encoder attention mask (not implemented).
            encoder_decoder_position_bias: Encoder-decoder position bias (not implemented).
            cross_attn_layer_head_mask: Cross attention layer head mask (not implemented).
            layer_head_mask: Mask for attention heads.
            past_key_values: Past key values for caching (not implemented).
            use_cache: Whether to use cache (not implemented).
            output_attentions: Whether to return attention weights.
            cache_position: Cache position.

        Returns:
            TensorValue: Output tensor of shape (batch_size, seq_length, hidden_size).
        """
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        if input_ids is not None:
            if input_ids.rank == 1:
                input_ids = F.unsqueeze(input_ids, 0)
            if self.embed_tokens is None:
                raise ValueError(
                    "embed_tokens must be provided when input_ids is used"
                )
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )
        elif inputs_embeds.rank == 2:
            inputs_embeds = F.unsqueeze(inputs_embeds, 0)

        if self.is_decoder or use_cache:
            raise NotImplementedError("T5 decoder is not implemented yet.")

        hidden_states = inputs_embeds

        if attention_mask is not None:
            if attention_mask.rank == 1:
                attention_mask = F.unsqueeze(attention_mask, 0)
            dtype_np = (
                hidden_states.dtype.to_numpy()
                if hasattr(hidden_states.dtype, "to_numpy")
                else np.float32
            )
            mask_multiplier = F.constant(
                float(np.finfo(dtype_np).min),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            causal_mask = (
                F.constant(
                    1.0,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                - F.cast(attention_mask, hidden_states.dtype)
            ) * mask_multiplier
            causal_mask = F.unsqueeze(causal_mask, 1)
            causal_mask = F.unsqueeze(causal_mask, 1)
        else:
            causal_mask = None
        encoder_extended_attention_mask = None

        position_bias = None
        all_attentions: tuple[Tensor, ...] = ()
        for layer_module in self.block:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=causal_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                layer_head_mask=layer_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1] if len(layer_outputs) > 1 else None
            if output_attentions:
                if len(layer_outputs) > 2:
                    all_attentions = all_attentions + (layer_outputs[2],)

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class T5EncoderModel(Module[..., Tensor]):
    def __init__(
        self,
        config: T5Config,
    ):
        """Construct a T5 encoder model.

        Args:
            config: T5 configuration for vocabulary size, layer counts, and
                device/dtype settings.
        """
        super().__init__()
        act_info = config.feed_forward_proj.split("-")
        config.dense_act_fn = act_info[-1]
        config.is_gated_act = act_info[0] == "gated"

        self.shared = Embedding(
            config.vocab_size,
            dim=config.d_model,
        )

        encoder_config = config
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = T5Stack(encoder_config, self.shared)
        self.device = config.device
        self.dtype = config.dtype

    def input_types(self) -> tuple[TensorType, ...]:
        """Get input types for the model.

        Returns:
            tuple[TensorType, ...]: Input types.
        """
        return (
            TensorType(
                DType.int64,
                shape=["batch_size", "sequence_length"],
                device=self.device,
            ),
        )

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Process input through the T5 encoder model.

        Args:
            input_ids: Input IDs tensor of shape (batch_size, seq_length).
            attention_mask: Attention mask tensor of shape (batch_size, seq_length).

        Returns:
            TensorValue: Output tensor of shape (batch_size, seq_length, hidden_size).
        """
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)
