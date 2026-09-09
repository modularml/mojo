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
"""Speculators-format DSpark draft module (DFlash ``1 + N`` fill-in family).

Follows the vLLM runtime reference (``qwen3_dflash.py`` / ``qwen3_dspark.py``
/ ``dspark/speculator.py``), which is what runs speculators-format DSpark
checkpoints such as ``RedHatAI/gemma-4-31B-it-speculator.dspark``. This is
NOT the dense deepseek-style drafter that ``unified_dspark_gemma4_12b``
mirrors:
the block here is a qwen3/llama-style transformer with separate K/V GQA
projections, per-head-dim q/k RMSNorm, plain (non-gemma) RMSNorm, and
per-layer config-driven causality (causal sliding-window attention iff the
layer type is ``sliding_attention``).

Semantics the caller (the unified graph) must uphold:

- Block inputs are RAW embedding rows (``embed_tokens[token]``) with NO gemma
  ``sqrt(hidden)`` scaling; slot 0 embeds the anchor/bonus token and slots
  ``1..block_size-1`` embed ``mask_token_id``.
- :meth:`forward_block` returns hidden states for ALL block slots including
  the anchor slot 0; when ``sample_from_anchor`` is false the caller drops
  slot 0 (its output is untrained) before computing logits for the
  ``block_size - 1`` drafted positions.
- Context K/V is a pure function of the projected target taps
  (:meth:`project_target_hidden` output); the fc/hidden_norm output feeds
  ONLY the context-KV path and never enters the block token stream.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.attention import AttentionWithRope
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import TransformerBlock

from ..speculators_common.d2t import map_draft_to_target_vocab


def _select_mask_variant(causal: bool, sliding_window: bool) -> MHAMaskVariant:
    """Per-layer mask from the checkpoint config (never hardcode causality).

    ``sliding_window`` selects the mask family only; the window size plumbs
    separately to the attention op.
    """
    if causal:
        if sliding_window:
            return MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
        return MHAMaskVariant.CAUSAL_MASK
    if sliding_window:
        return MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK
    return MHAMaskVariant.NULL_MASK


class DSparkSpeculatorsMarkovHead(Module):
    """Vanilla low-rank markov transition head with asymmetric vocabs.

    ``markov_w1`` embeds the previous token in TARGET vocab space;
    ``markov_w2`` projects the rank-``r`` embedding to a DRAFT-vocab logit
    bias added to the base draft logits.
    """

    def __init__(
        self,
        vocab_size: int,
        draft_vocab_size: int,
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
            out_dim=draft_vocab_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def compute_step_bias(self, prev_tokens: TensorValue) -> TensorValue:
        """Returns the ``[batch, draft_vocab]`` bias for ``[batch]``
        TARGET-vocab prev tokens."""
        return self.markov_w2(self.markov_w1(prev_tokens))

    def __call__(self, prev_tokens: TensorValue) -> TensorValue:
        # Alias for compute_step_bias to satisfy the Module ABC.
        return self.compute_step_bias(prev_tokens)


class DSparkSpeculatorsDraft(Module):
    """Speculators-format DSpark draft transformer.

    Structure per block slot: ``input_layernorm -> attn -> +residual ->
    post_attention_layernorm -> silu MLP -> +residual`` (2-norm sandwich),
    final ``norm`` after the stack. Attention is GQA with separate K/V,
    per-head-dim q/k RMSNorm before full-rotary neox RoPE, and a per-layer
    causal/sliding mask derived from ``layer_causal``/``sliding_window``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        sliding_window: int | None,
        layer_causal: Sequence[bool],
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        block_size: int,
        sample_from_anchor: bool,
        mask_token_id: int,
        num_context_features: int,
        max_seq_len: int,
        kv_params: KVCacheParams,
        devices: list[DeviceRef],
        dtype: DType,
    ) -> None:
        super().__init__()
        if len(devices) != 1:
            raise ValueError(
                "DSparkSpeculatorsDraft currently supports a single device"
                " only."
            )
        if len(layer_causal) != num_hidden_layers:
            raise ValueError(
                f"layer_causal has {len(layer_causal)} entries for"
                f" {num_hidden_layers} layers."
            )
        if not 0 < draft_vocab_size <= vocab_size:
            raise ValueError(
                f"draft_vocab_size {draft_vocab_size} must be in"
                f" (0, {vocab_size}]."
            )
        if markov_rank <= 0:
            raise ValueError(
                "DSparkSpeculatorsDraft requires a vanilla markov head"
                f" (markov_rank > 0), got {markov_rank}."
            )
        if block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {block_size}.")
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError(
                f"mask_token_id {mask_token_id} outside [0, {vocab_size})."
            )
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError(
                f"sliding_window must be positive, got {sliding_window}."
            )
        if num_context_features <= 0:
            raise ValueError(
                "num_context_features must be positive, got"
                f" {num_context_features}."
            )

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.block_size = block_size
        self.sample_from_anchor = sample_from_anchor
        self.mask_token_id = mask_token_id
        self.num_context_features = num_context_features
        device = devices[0]

        self.rope = RotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=False,
        )

        def _make_norm() -> RMSNorm:
            return RMSNorm(hidden_size, dtype, rms_norm_eps)

        layers: list[TransformerBlock] = []
        for layer_idx in range(num_hidden_layers):
            attention = AttentionWithRope(
                rope=self.rope,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                hidden_size=hidden_size,
                kv_params=kv_params,
                devices=devices,
                dtype=dtype,
                linear_cls=Linear,
                use_qk_norm=True,
                rms_norm_eps=rms_norm_eps,
                mask_variant=_select_mask_variant(
                    bool(layer_causal[layer_idx]), sliding_window is not None
                ),
                sliding_window=sliding_window,
            )
            mlp = MLP(
                dtype,
                None,
                hidden_size,
                intermediate_size,
                devices,
                Linear,
                activation_function="silu",
            )
            layers.append(
                TransformerBlock(
                    attention=attention,
                    mlp=mlp,
                    attention_norm=_make_norm(),
                    mlp_norm=_make_norm(),
                )
            )
        self.layers = LayerList(layers)
        self.norm = _make_norm()

        # Target-tap projection: [N, K_sel * H] -> [N, H]; consumed only by
        # the context-KV path (see module docstring).
        self.fc = Linear(
            in_dim=num_context_features * hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.hidden_norm = _make_norm()

        # Draft-owned pruned-vocab head; NO logit softcapping (the qwen3
        # dspark head has none, unlike the gemma-style dense drafter).
        self.lm_head = Linear(
            in_dim=hidden_size,
            out_dim=draft_vocab_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

        self.markov_head = DSparkSpeculatorsMarkovHead(
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
            markov_rank=markov_rank,
            dtype=dtype,
            device=device,
        )

        # Draft-to-target vocab offset table (int64, [draft_vocab_size]).
        self.d2t = Weight(
            "d2t", DType.int64, shape=[draft_vocab_size], device=device
        )

        # Aliased to the target's embedding by the unified pipeline at load
        # time when sharing weights (the checkpoint ships a byte-identical
        # frozen copy). The draft path must use the RAW rows — never the
        # target's sqrt(hidden)-scaled embedding output.
        self.embed_tokens: Embedding | None = None

    def project_target_hidden(
        self, target_hs_concat: TensorValue
    ) -> TensorValue:
        """Projects concatenated target taps to the context-KV input,
        ``hidden_norm(fc(concat(taps)))``."""
        return self.hidden_norm(self.fc(target_hs_concat))

    def materialize_kv(
        self,
        ctx_hidden: TensorValue,
        input_row_offsets: TensorValue,
        kv_collection: PagedCacheValues,
    ) -> None:
        """Projects context hidden states to per-layer K/V and writes the
        paged KV cache.

        K gets ``k_norm`` + RoPE at the context positions; V is the plain
        ``v_proj`` output (un-normed, un-roped), matching the reference
        ``precompute_and_store_context_kv``.
        """
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
        """Runs the draft stack over the block tokens.

        Returns hidden states for ALL block slots (anchor slot 0 included);
        dropping slot 0 when ``sample_from_anchor`` is false is the unified
        graph's job.
        """
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

    def sample_draft_tokens(
        self,
        base_logits: TensorValue,
        anchor_tokens: TensorValue,
    ) -> TensorValue:
        """Greedily samples the markov-corrected draft tokens.

        Unrolled at graph-build time over the static slot axis. Per slot:
        ``bias = markov_w2(markov_w1[prev])`` with ``prev`` a TARGET-vocab id
        (the anchor seeds the first slot), ``draft = argmax(base + bias)``
        over the draft vocab, then the in-chain d2t gather maps ``draft`` to
        the target id that both feeds the next slot's bias and is emitted.

        Args:
            base_logits: Base draft logits ``[batch, num_slots, draft_vocab]``
                with a static slot axis (slot 0 = the first drafted position,
                i.e. after the caller's anchor-slot drop).
            anchor_tokens: Anchor/bonus token ids ``[batch]``, target vocab.

        Returns:
            The drafted TARGET-vocab token ids ``[batch, num_slots]`` (int64).
        """
        num_slots = int(base_logits.shape[1])
        prev = anchor_tokens
        sampled: list[TensorValue] = []
        for k in range(num_slots):
            step_logits = base_logits[:, k, :] + (
                self.markov_head.compute_step_bias(prev)
            )
            draft_ids = ops.squeeze(ops.argmax(step_logits, axis=-1), axis=-1)
            prev = map_draft_to_target_vocab(draft_ids, TensorValue(self.d2t))
            sampled.append(ops.unsqueeze(prev, axis=1))
        return ops.concat(sampled, axis=1)
