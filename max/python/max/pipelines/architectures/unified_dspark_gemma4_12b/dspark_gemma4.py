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
"""DSpark draft module for a Gemma4 target.

Non-causal block-mode draft transformer with KV materialization from
target hidden states, following the DeepSpec reference
(``deepspec/modeling/dspark/gemma4/modeling.py``). Structurally the DFlash
draft pattern (fc + hidden_norm conditioning, materialize ctx K/V, block
forward), but with Gemma4 semantics: k_eq_v attention (V is the scale-free
RMSNorm of the K projection, no RoPE on V), per-head Q/K norms, partial
proportional RoPE, the 4-norm decoder sandwich, and a per-layer scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.embedding import Embedding
from max.nn.kernels import rope_split_store_ragged
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, Linear

from ..gemma3.model_config import _HIDDEN_ACTIVATION_MAP
from ..gemma4.layers.attention import Gemma4Attention
from ..gemma4.layers.rms_norm import Gemma4RMSNorm
from ..gemma4.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
    ProportionalScalingParams,
)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Reads *name* from an HF config object or a plain (nested) dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass(frozen=True)
class DSparkGemma4DraftConfig:
    """DSpark drafter hyperparameters from the draft HF checkpoint config.

    The checkpoint config (e.g. ``deepseek-ai/dspark_gemma4_12b_block7``) is
    a gemma4_text-style config extended with the DSpark fields
    (``block_size``, ``mask_token_id``, ``target_layer_ids``,
    ``markov_rank``, ...). Only full-attention (k_eq_v) draft layers are
    supported, matching the reference drafter.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    """Global (k_eq_v) key/value head count; every draft layer is global."""
    head_dim: int
    """Global head dim (``global_head_dim`` in the HF config)."""
    rms_norm_eps: float
    vocab_size: int
    hidden_activation: str
    final_logit_softcapping: float | None
    rope_theta: float
    partial_rotary_factor: float
    max_seq_len: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    markov_rank: int
    markov_head_type: str

    @property
    def num_context_features(self) -> int:
        return len(self.target_layer_ids)

    @classmethod
    def from_huggingface_config(
        cls,
        huggingface_config: Any,
        *,
        max_seq_len: int | None = None,
    ) -> DSparkGemma4DraftConfig:
        """Parses the DSpark draft checkpoint config (object or dict)."""
        cfg = huggingface_config
        if not bool(_get(cfg, "attention_k_eq_v")):
            raise ValueError(
                "DSpark Gemma4 drafter requires attention_k_eq_v=true."
            )
        if bool(_get(cfg, "enable_moe_block", False)):
            raise ValueError(
                "DSpark Gemma4 drafter does not support MoE blocks."
            )
        if int(_get(cfg, "hidden_size_per_layer_input", 0)) != 0:
            raise ValueError(
                "DSpark Gemma4 drafter does not support per-layer inputs."
            )
        layer_types = list(_get(cfg, "layer_types") or [])
        if any(lt != "full_attention" for lt in layer_types):
            raise ValueError(
                "DSpark Gemma4 drafter expects all layer_types to be"
                f" 'full_attention', got {layer_types}."
            )
        # rope_parameters is a nested plain dict on the HF config object.
        rope_parameters = _get(cfg, "rope_parameters") or {}
        full_rope = rope_parameters.get("full_attention") or {}
        rope_theta = full_rope.get("rope_theta")
        if rope_theta is None:
            raise ValueError(
                "DSpark draft config is missing"
                " rope_parameters.full_attention.rope_theta."
            )
        target_layer_ids = tuple(
            int(x) for x in (_get(cfg, "target_layer_ids") or ())
        )
        if not target_layer_ids:
            raise ValueError("DSpark draft config is missing target_layer_ids.")
        markov_rank = int(_get(cfg, "markov_rank", 0))
        markov_head_type = str(_get(cfg, "markov_head_type", "vanilla"))
        if markov_rank > 0 and markov_head_type != "vanilla":
            raise ValueError(
                "Only the vanilla markov head is supported, got"
                f" markov_head_type={markov_head_type!r}."
            )
        hidden_activation = str(_get(cfg, "hidden_activation"))
        hidden_activation = _HIDDEN_ACTIVATION_MAP.get(
            hidden_activation, hidden_activation
        )
        softcap = _get(cfg, "final_logit_softcapping")

        return cls(
            hidden_size=int(_get(cfg, "hidden_size")),
            intermediate_size=int(_get(cfg, "intermediate_size")),
            num_hidden_layers=int(_get(cfg, "num_hidden_layers")),
            num_attention_heads=int(_get(cfg, "num_attention_heads")),
            num_key_value_heads=int(_get(cfg, "num_global_key_value_heads")),
            head_dim=int(_get(cfg, "global_head_dim")),
            rms_norm_eps=float(_get(cfg, "rms_norm_eps")),
            vocab_size=int(_get(cfg, "vocab_size")),
            hidden_activation=hidden_activation,
            final_logit_softcapping=(
                float(softcap) if softcap is not None else None
            ),
            rope_theta=float(rope_theta),
            partial_rotary_factor=float(
                _get(full_rope, "partial_rotary_factor", 1.0)
            ),
            max_seq_len=(
                int(max_seq_len)
                if max_seq_len is not None
                else int(_get(cfg, "max_position_embeddings"))
            ),
            block_size=int(_get(cfg, "block_size")),
            mask_token_id=int(_get(cfg, "mask_token_id")),
            target_layer_ids=target_layer_ids,
            markov_rank=markov_rank,
            markov_head_type=markov_head_type,
        )


class DSparkMarkovHead(Module):
    """Vanilla markov head: ``bias = markov_w2(markov_w1[prev_token])``.

    Mirrors the reference ``VanillaMarkov`` (``markov_head.py``): a low-rank
    per-token logit bias conditioned only on the previous draft token. The
    hidden states are ignored by the vanilla head.
    """

    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.markov_w1 = Embedding(
            vocab_size=vocab_size,
            hidden_dim=markov_rank,
            dtype=dtype,
            device=device,
        )
        self.markov_w2 = Linear(
            in_dim=markov_rank,
            out_dim=vocab_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def compute_step_bias(self, prev_tokens: TensorValue) -> TensorValue:
        """Returns the ``[batch, vocab]`` bias for ``[batch]`` prev tokens."""
        return self.markov_w2(self.markov_w1(prev_tokens))

    def __call__(
        self,
        base_logits: TensorValue,
        first_prev_tokens: TensorValue,
    ) -> TensorValue:
        """Greedily samples the markov-corrected block tokens.

        Sequentially (unrolled at graph-build time over the static block
        axis): ``tokens[k] = argmax(base_logits[:, k] + bias(prev))`` with
        ``prev`` starting at ``first_prev_tokens`` (the anchor token) and
        following the sampled chain. Matches the reference
        ``sample_block_tokens`` at temperature 0.

        Args:
            base_logits: Soft-capped base draft logits ``[batch, block, vocab]``
                with a static block axis.
            first_prev_tokens: Anchor token ids ``[batch]``.

        Returns:
            The sampled draft tokens ``[batch, block]`` (int64).
        """
        block = int(base_logits.shape[1])
        prev = first_prev_tokens
        sampled: list[TensorValue] = []
        for k in range(block):
            step_logits = base_logits[:, k, :] + self.compute_step_bias(prev)
            prev = ops.squeeze(ops.argmax(step_logits, axis=-1), axis=-1)
            sampled.append(ops.unsqueeze(prev, axis=1))
        return ops.concat(sampled, axis=1)


class DSparkGemma4Attention(Gemma4Attention):
    """Non-causal DSpark draft attention over ``[materialized ctx ; block]``."""

    mask_variant = MHAMaskVariant.NULL_MASK

    def materialize_kv_from_hidden(
        self,
        hidden: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> None:
        """Projects ctx hidden states to K/V and writes the paged KV cache.

        Matches the reference ctx path: K gets ``k_norm`` + RoPE at the ctx
        positions (``cache_length + row``); V is the scale-free ``v_norm``
        of the pre-norm K projection and is not roped.
        """
        layer_idx = ops.constant(
            self.layer_idx_in_cache, DType.uint32, device=DeviceRef.CPU()
        )
        num_kv_heads = self.kv_weight_dim // self.head_dim

        qk = self.qk_proj(hidden)
        x_q, x_k = ops.split(
            qk, [self.q_weight_dim, self.kv_weight_dim], axis=-1
        )
        # The roped Q output is discarded here, so x_q skips q_norm.
        x_v = self.v_norm(
            x_k.reshape((-1, num_kv_heads, self.head_dim))
        ).reshape((-1, self.kv_weight_dim))
        x_k = self.k_norm(
            x_k.reshape((-1, num_kv_heads, self.head_dim))
        ).reshape((-1, self.kv_weight_dim))
        qkv = ops.concat((x_q, x_k, x_v), axis=-1)

        freqs_cis = ops.cast(self.rope_global.freqs_cis, qkv.dtype).to(
            qkv.device
        )
        _ = rope_split_store_ragged(
            self.kv_params,
            qkv,
            input_row_offsets,
            freqs_cis,
            kv_collection,
            layer_idx,
            n_heads=self.n_heads,
            interleaved=self.rope_global.interleaved,
            q_out_dtype=self.kv_params.dtype,
        )


class DSparkGemma4DecoderLayer(Module):
    """Single-device Gemma4 decoder sandwich with a per-layer scalar.

    Mirrors the reference ``Gemma4DSparkDecoderLayer``: input_ln -> attn ->
    post_attn_ln -> +residual; pre_ffw_ln -> mlp -> post_ffw_ln ->
    +residual; then multiply by ``layer_scalar``.
    """

    def __init__(
        self,
        attention: DSparkGemma4Attention,
        mlp: MLP,
        hidden_size: int,
        rms_norm_eps: float,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = Gemma4RMSNorm(
            hidden_size, dtype, eps=rms_norm_eps
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            hidden_size, dtype, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, dtype, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, dtype, eps=rms_norm_eps
        )
        self.layer_scalar = Weight(
            "layer_scalar", dtype, shape=[1], device=DeviceRef.CPU()
        )

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(
            h, kv_collection, input_row_offsets=input_row_offsets
        )
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        h = residual + h
        return h * self.layer_scalar.to(h.device)


class DSparkGemma4(Module):
    """DSpark draft transformer for a Gemma4 target."""

    def __init__(
        self,
        config: DSparkGemma4DraftConfig,
        *,
        kv_params: KVCacheParams,
        devices: list[DeviceRef],
        dtype: DType,
    ) -> None:
        super().__init__()
        if len(devices) != 1:
            raise ValueError(
                "DSparkGemma4 currently supports a single device only."
            )
        self.config = config
        device = devices[0]

        self.rope = ProportionalRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=ProportionalScalingParams(
                partial_rotary_factor=config.partial_rotary_factor
            ),
        )

        layers: list[DSparkGemma4DecoderLayer] = []
        for layer_idx in range(config.num_hidden_layers):
            attention = DSparkGemma4Attention(
                rope_global=self.rope,
                # Every draft layer is full_attention; the local rope slot is
                # never selected.
                rope_local=self.rope,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                num_global_key_value_heads=config.num_key_value_heads,
                attention_k_eq_v=True,
                hidden_size=config.hidden_size,
                kv_params=kv_params,
                global_head_dim=config.head_dim,
                layer_idx=layer_idx,
                layer_idx_in_cache=layer_idx,
                is_sliding=False,
                dtype=dtype,
                devices=devices,
                qk_norm_eps=config.rms_norm_eps,
            )
            mlp = MLP(
                dtype,
                None,
                config.hidden_size,
                config.intermediate_size,
                devices,
                Linear,
                activation_function=config.hidden_activation,
            )
            layers.append(
                DSparkGemma4DecoderLayer(
                    attention=attention,
                    mlp=mlp,
                    hidden_size=config.hidden_size,
                    rms_norm_eps=config.rms_norm_eps,
                    dtype=dtype,
                )
            )
        self.layers = LayerList(layers)
        self.norm = Gemma4RMSNorm(
            config.hidden_size, dtype, eps=config.rms_norm_eps
        )

        # Target-hidden projection: [N, K_sel * H] -> [N, H], shared across
        # all draft layers.
        self.fc = Linear(
            in_dim=config.num_context_features * config.hidden_size,
            out_dim=config.hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.hidden_norm = Gemma4RMSNorm(
            config.hidden_size, dtype, eps=config.rms_norm_eps
        )

        self.markov_head: DSparkMarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = DSparkMarkovHead(
                vocab_size=config.vocab_size,
                markov_rank=config.markov_rank,
                dtype=dtype,
                device=device,
            )

        # Aliased to the target's modules by the unified pipeline at load
        # time (the checkpoint ships frozen copies that are skipped). Held
        # only so the shared weights appear in this module's state dict; the
        # unified graph invokes the target's modules directly.
        self.embed_tokens: Module | None = None
        self.lm_head: Module | None = None

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
        for layer in self.layers:
            assert isinstance(layer, DSparkGemma4DecoderLayer)
            layer.self_attn.materialize_kv_from_hidden(
                hidden=ctx_hidden,
                kv_collection=kv_collection,
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
        for layer in self.layers:
            h = layer(h, kv_collection, input_row_offsets)
        return self.norm(h)
