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
"""DFlash2 draft module for a Qwen3.5-family target.

DFlash2 is DFlash v1's non-causal block drafter plus two mechanisms: a
two-tap grouped dynamic convolution wrapped around each sublayer, and a
candidate-path selector that keeps ``selector_top_k`` tokens at every block
position and traces one coherent path through them. Everything else — the
``fc``/``hidden_norm`` tap fusion, the KV materialization from target hidden
states, the windowed non-causal block forward — is the v1 body and is shaped
exactly like :class:`~..dflash_llama3.DFlashLlama3`.

The drafter ships no ``embed_tokens`` and no ``lm_head``: both are borrowed
from the target, so both slots stay ``None`` until the unified pipeline wires
them.
"""

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

from ..llama3.model_config import Llama3Config, create_rope_embedding
from .layers import DFlash2CandidateSelector, DFlash2GroupedConv


class DFlash2TransformerBlock(Module):
    """A DFlash draft block with a dynamic convolution around each sublayer.

    ``r = x; h = norm(r); h = conv_a.prepare(h); h = attn(h);
    h = conv_a.finish(h); r += h`` and the same again for the MLP — four
    convolution applications and two ``kernel_projection`` evaluations per
    layer.
    """

    def __init__(
        self,
        *,
        attention: AttentionWithRope,
        mlp: MLP,
        attention_norm: RMSNorm,
        mlp_norm: RMSNorm,
        attention_conv: DFlash2GroupedConv,
        mlp_conv: DFlash2GroupedConv,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm
        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        h, coefficients = self.attention_conv.prepare(self.input_layernorm(x))
        h = self.self_attn(
            layer_idx,
            h,
            kv_collection,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
        )
        residual = x + self.attention_conv.finish(h, coefficients)

        h, coefficients = self.mlp_conv.prepare(
            self.post_attention_layernorm(residual)
        )
        h = self.mlp(h)
        return residual + self.mlp_conv.finish(h, coefficients)


class DFlash2Qwen3_5(Module):
    """DFlash2 draft transformer for a Qwen3.5 target."""

    def __init__(
        self,
        config: Llama3Config,
        *,
        num_context_features: int,
        block_size: int,
        conv_kernel_size: int,
        conv_group_size: int,
        selector_rank: int,
        selector_top_k: int,
        layer_types: Sequence[str] | None = None,
    ) -> None:
        """Builds the draft stack.

        Args:
            config: The draft's own Llama3-shaped config.
            num_context_features: Number of target hidden-state taps the
                ``fc`` projection consumes.
            block_size: Query tokens per request: the anchor plus the mask
                tokens. The convolution's block-local axis.
            conv_kernel_size: Convolution taps.
            conv_group_size: Channels sharing one dynamic coefficient.
            selector_rank: Codebook rank.
            selector_top_k: Candidates kept per mask slot.
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
                "DFlash2Qwen3_5 requires rms_norm_eps to be set on its config."
            )
        if len(config.devices) != 1:
            raise ValueError(
                "DFlash2Qwen3_5 currently supports a single device only."
            )
        if layer_types is not None:
            if len(layer_types) != config.num_hidden_layers:
                raise ValueError(
                    "DFlash2 layer_types must have one entry per draft layer."
                    f" Got {len(layer_types)} entries for"
                    f" {config.num_hidden_layers} layers."
                )
            unknown = sorted(
                set(layer_types) - {"sliding_attention", "full_attention"}
            )
            if unknown:
                raise ValueError(
                    f"DFlash2 draft has unsupported layer_types {unknown};"
                    " expected only 'sliding_attention' or 'full_attention'."
                )
            if (
                "sliding_attention" in layer_types
                and config.sliding_window is None
            ):
                raise ValueError(
                    "DFlash2 sliding_attention layers require a"
                    " sliding_window on the draft config."
                )

        self.config = config
        self.num_context_features = num_context_features
        self.block_size = block_size
        self.selector_top_k = selector_top_k
        device = config.devices[0]
        norm_dtype = config.norm_dtype or config.dtype
        rms_norm_eps = config.rms_norm_eps

        self.rope = create_rope_embedding(
            # RoPE spans num_heads x head_dim, which is only hidden_size when
            # the draft's head_dim happens to be hidden_size // num_heads.
            # This drafter's 32 x 128 over a 5120 hidden does coincide, but
            # the target it drafts for uses head_dim 256.
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

        def _make_conv() -> DFlash2GroupedConv:
            return DFlash2GroupedConv(
                config.hidden_size,
                taps=conv_kernel_size,
                group_size=conv_group_size,
                block_size=block_size,
                dtype=config.dtype,
                device=device,
            )

        layers: list[DFlash2TransformerBlock] = []
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
                # Never causal: DFlash2 attends bidirectionally inside the
                # block, bounded only by the draft's own sliding window.
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
                DFlash2TransformerBlock(
                    attention=attention,
                    mlp=mlp,
                    attention_norm=_make_norm(),
                    mlp_norm=_make_norm(),
                    attention_conv=_make_conv(),
                    mlp_conv=_make_conv(),
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

        self.candidate_selector = DFlash2CandidateSelector(
            config.hidden_size,
            vocab_size=config.vocab_size,
            rank=selector_rank,
            top_k=selector_top_k,
            dtype=config.dtype,
            device=device,
        )

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
        """Writes per-layer context K/V projected from the target's states.

        No ``input_layernorm`` and no convolution on this path. Q is not
        skipped, though: ``materialize_kv_from_hidden`` runs the fused
        ``qkv_proj`` and, because the drafter sets ``use_qk_norm``,
        normalizes Q alongside K before ``rope_split_store_ragged`` keeps
        only K/V. Storing the context K/V therefore costs a full QKV
        projection, not a ``k_proj``/``v_proj`` pair.
        """
        freqs_cis = self.rope.freqs_cis
        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, DFlash2TransformerBlock)
            layer.self_attn.materialize_kv_from_hidden(
                layer_idx=ops.constant(
                    layer_idx, DType.uint32, device=DeviceRef.CPU()
                ),
                hidden=ctx_hidden,
                kv_collection=kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )

    def forward_block(
        self,
        input_embeds: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Runs ``block_size`` query rows per sequence through the stack.

        ``input_embeds`` must be laid out as dense ``block_size``-row blocks,
        one per sequence: the convolution's block axis is a reshape of the
        token axis, not a function of ``input_row_offsets``.
        """
        h = input_embeds
        freqs_cis = self.rope.freqs_cis
        for idx, layer in enumerate(self.layers):
            assert isinstance(layer, DFlash2TransformerBlock)
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )
        return self.norm(h)

    def compute_candidates(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Top-k candidate ids and their logits from the borrowed head.

        Args:
            hidden_states: ``[batch, steps, hidden]`` mask-slot outputs.

        Returns:
            ``(candidate_ids, unary_logits)``, both
            ``[batch, steps, selector_top_k]``.
        """
        if self.lm_head is None:
            raise ValueError(
                "DFlash2Qwen3_5.lm_head is borrowed from the target and must"
                " be wired before compute_candidates is called."
            )
        hidden = hidden_states.shape[-1]
        logits = ops.cast(
            self.lm_head(hidden_states.reshape((-1, hidden))), DType.float32
        )
        unary_logits, candidate_ids = ops.top_k(logits, self.selector_top_k)
        batch, steps = hidden_states.shape[0], hidden_states.shape[1]
        shape = (batch, steps, self.selector_top_k)
        return candidate_ids.reshape(shape), unary_logits.reshape(shape)

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
