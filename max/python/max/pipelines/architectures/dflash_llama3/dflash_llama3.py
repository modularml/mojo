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
"""DFlash draft module — non-causal block-mode transformer with KV materialization."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.attention import AttentionWithRope
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.embedding import Embedding
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, Linear
from max.nn.norm import RMSNorm
from max.nn.transformer import TransformerBlock

from ..llama3.model_config import Llama3Config, create_rope_embedding


class DFlashLlama3(Module):
    """DFlash draft transformer for a Llama3 target."""

    def __init__(
        self,
        config: Llama3Config,
        *,
        num_context_features: int,
        layer_types: Sequence[str] | None = None,
    ) -> None:
        """Builds the draft stack.

        Args:
            config: The draft's own Llama3-shaped config.
            num_context_features: Number of target hidden-state taps the
                ``fc`` projection consumes. Decoupled from the draft's layer
                count: the Gemma4 DFlash drafter fuses six taps into five
                layers.
            layer_types: Per-layer ``"sliding_attention"`` /
                ``"full_attention"`` selection from the draft checkpoint.
                ``None`` applies ``config.sliding_window`` to every layer.
        """
        super().__init__()
        if num_context_features <= 0:
            raise ValueError(
                "num_context_features must be positive, got"
                f" {num_context_features}."
            )
        if config.rms_norm_eps is None:
            raise ValueError(
                "DFlashLlama3 requires rms_norm_eps to be set on its config."
            )
        if len(config.devices) != 1:
            raise ValueError(
                "DFlashLlama3 currently supports a single device only."
            )
        if layer_types is not None:
            if len(layer_types) != config.num_hidden_layers:
                raise ValueError(
                    "DFlash layer_types must have one entry per draft layer."
                    f" Got {len(layer_types)} entries for"
                    f" {config.num_hidden_layers} layers."
                )
            unknown = sorted(
                set(layer_types) - {"sliding_attention", "full_attention"}
            )
            if unknown:
                raise ValueError(
                    f"DFlash draft has unsupported layer_types {unknown};"
                    " expected only 'sliding_attention' or 'full_attention'."
                )
            if (
                "sliding_attention" in layer_types
                and config.sliding_window is None
            ):
                raise ValueError(
                    "DFlash sliding_attention layers require a"
                    " sliding_window on the draft config."
                )

        self.config = config
        self.num_context_features = num_context_features
        device = config.devices[0]
        norm_dtype = config.norm_dtype or config.dtype
        # The None check above ensures rms_norm_eps is set; bind a
        # non-Optional local so the closure below has the right type.
        rms_norm_eps = config.rms_norm_eps

        self.rope = create_rope_embedding(
            # RoPE spans num_heads x head_dim, which is only hidden_size when
            # the draft's head_dim happens to be hidden_size // num_heads.
            # The Gemma4 DFlash drafter breaks that (64 x 128 over a 5376
            # hidden), so derive the span from the KV head_dim.
            hidden_size=(
                config.kv_params.head_dim * config.num_attention_heads
            ),
            num_attention_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved_rope_weights=config.interleaved_rope_weights,
            rope_scaling_params=config.rope_scaling_params,
            longrope_scaling_params=config.longrope_scaling_params,
            device=device,
        )

        def _make_norm() -> RMSNorm:
            return RMSNorm(
                config.hidden_size,
                norm_dtype,
                rms_norm_eps,
                multiply_before_cast=True,
            )

        layers: list[TransformerBlock] = []
        for layer_idx in range(config.num_hidden_layers):
            sliding_window = (
                config.sliding_window
                if layer_types is None
                or layer_types[layer_idx] == "sliding_attention"
                else None
            )
            attention = AttentionWithRope(
                rope=self.rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                kv_params=config.kv_params,
                devices=config.devices,
                dtype=config.dtype,
                linear_cls=Linear,
                stacked_qkv=config.stacked_qkv,
                scale=config.attention_multiplier,
                has_bias=config.attention_bias,
                quant_config=config.quant_config,
                clip_qkv=config.clip_qkv,
                use_qk_norm=True,
                rms_norm_eps=rms_norm_eps,
                mask_variant=(
                    MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK
                    if sliding_window is not None
                    else MHAMaskVariant.NULL_MASK
                ),
                sliding_window=sliding_window,
            )
            mlp = MLP(
                config.dtype,
                config.model_quantization_encoding,
                config.hidden_size,
                config.intermediate_size,
                config.devices,
                Linear,
                quant_config=config.quant_config,
            )
            layers.append(
                TransformerBlock(
                    attention=attention,
                    mlp=mlp,
                    attention_norm=_make_norm(),
                    mlp_norm=_make_norm(),
                    residual_multiplier=config.residual_multiplier,
                )
            )

        self.layers = LayerList(layers)
        self.norm = _make_norm()

        # Target-hidden projection: [N, K_sel * H] -> [N, H].
        self.fc = Linear(
            in_dim=num_context_features * config.hidden_size,
            out_dim=config.hidden_size,
            dtype=config.dtype,
            device=device,
            has_bias=False,
        )
        self.hidden_norm = _make_norm()

        # Aliased to the target's modules by the unified pipeline at load
        # time. ``lm_head`` is typed as a generic callable (rather than
        # ``Linear``) to match the target's inferred type.
        self.embed_tokens: Embedding | None = None
        self.lm_head: Callable[[TensorValue], TensorValue] | None = None

    def project_target_hidden(
        self, target_hs_concat: TensorValue
    ) -> TensorValue:
        return self.hidden_norm(self.fc(target_hs_concat))

    def materialize_kv(
        self,
        ctx_hidden: TensorValue,
        input_row_offsets: TensorValue,
        kv_collection: PagedCacheValues,
    ) -> None:
        freqs_cis = self.rope.freqs_cis
        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, TransformerBlock)
            attn = layer.self_attn
            assert isinstance(attn, AttentionWithRope)
            attn.materialize_kv_from_hidden(
                layer_idx=ops.constant(
                    layer_idx, DType.uint32, device=DeviceRef.CPU()
                ),
                hidden=ctx_hidden,
                kv_collection=kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )

    def __call__(
        self,
        input_embeds: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Alias for forward_block to satisfy the Module ABC.
        return self.forward_block(
            input_embeds, kv_collection, input_row_offsets
        )

    def forward_block(
        self,
        input_embeds: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        h = input_embeds
        freqs_cis = self.rope.freqs_cis
        for idx, layer in enumerate(self.layers):
            assert isinstance(layer, TransformerBlock)
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )
        return self.norm(h)
