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
"""Text graphs for the DiffusionGemmaForBlockDiffusion port.

DiffusionGemma is an encoder/decoder block-diffusion model whose transformer
layers are exactly Gemma4's (parallel dense-MLP + MoE branches, mixed
sliding/global attention with per-type KV geometry, K==V on global layers,
q/k/v norms, attention scale 1.0). This module therefore composes the donor
``gemma4`` layer classes into two graphs sharing one weight set:

- ``DiffusionGemmaEncoderTextModel``: causal forward that populates the paged
  KV cache (prompt prefill and accepted-canvas commits). Mirrors
  ``DiffusionGemmaEncoderTextModel`` in HF ``modular_diffusion_gemma.py``.
- ``DiffusionGemmaDecoderTextModel``: bidirectional forward over a
  ``canvas_length`` block that reads the encoder cache without committing to
  it, applies the self-conditioning block to its input embeddings, and
  computes the per-step sampling statistics (argmax, entropy, categorical
  sample, temperature-processed self-conditioning logits) on device. Mirrors
  HF ``DiffusionGemmaDecoderModel`` plus the logits math from
  ``DiffusionGemmaForBlockDiffusion.forward`` and ``_denoising_step``.

Deltas vs the donor graph (each tagged at its implementation site):
fp32 router softmax (HF ``DiffusionGemmaTextRouter.forward``), final logit
softcapping ``tanh(l/30)*30`` (HF ``DiffusionGemmaForBlockDiffusion.forward``),
noncausal decoder attention masks, and the self-conditioning module.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged, rope_split_store_ragged
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.moe import MoE, MoEQuantized, make_concatenated_gated_activation_fn
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import (
    ScaledWordEmbedding,
)
from max.pipelines.architectures.gemma4.layers.attention import Gemma4Attention
from max.pipelines.architectures.gemma4.layers.decoder_layer import (
    Gemma4TextDecoderLayer,
)
from max.pipelines.architectures.gemma4.layers.moe import Gemma4MoEGate
from max.pipelines.architectures.gemma4.layers.rms_norm import Gemma4RMSNorm
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from .model_config import DiffusionGemmaForBlockDiffusionConfig

_LAYER_TYPE_TO_KV_INDEX = {
    "sliding_attention": 0,
    "full_attention": 1,
}


class DiffusionGemmaMoEGate(Gemma4MoEGate):
    """Gemma4 MoE gate with the router softmax computed in float32.

    HF ``DiffusionGemmaTextRouter.forward`` overrides Gemma4 to run
    ``softmax(..., dtype=torch.float32)``; the donor gate softmaxes in model
    dtype. Top-k selection, renormalization, and the per-expert scale are
    applied in fp32 and the weights are cast back to the score dtype for the
    expert combine, matching HF's effective numerics.
    """

    def __call__(
        self, hidden_state: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        hidden_state = self.norm(hidden_state)
        hidden_state = hidden_state * self.scale * self.scalar_root_size

        expert_scores = self.gate_score(hidden_state)
        router_probs = ops.softmax(ops.cast(expert_scores, DType.float32))

        top_k_weights, top_k_index = ops.top_k(
            router_probs, k=self.num_experts_per_token, axis=-1
        )
        top_k_weights = top_k_weights / ops.sum(top_k_weights, axis=-1)
        top_k_weights = top_k_weights * ops.cast(
            ops.gather(self.per_expert_scale, top_k_index, axis=0),
            DType.float32,
        )
        return top_k_index, ops.cast(top_k_weights, expert_scores.dtype)


class DiffusionGemmaDecoderAttention(Gemma4Attention):
    """Gemma4 attention with noncausal masks for canvas denoising.

    The decoder attends bidirectionally: every canvas token sees all valid
    encoder-cache positions plus the whole canvas (HF
    ``DiffusionGemmaDecoderTextAttention`` with ``is_causal=False``). Canvas
    K/V are written to the cache slots after the committed length via the
    fused rope-store; commitment is controlled by the pipeline, so successive
    denoise steps overwrite the same slots and the cache stays effectively
    read-only.

    ``Gemma4Attention.shard()`` constructs shards via ``type(self)``, so the
    sharded instances retain this subclass's noncausal ``__call__``.
    """

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        layer_idx = ops.constant(
            self.layer_idx_in_cache, DType.uint32, device=DeviceRef.CPU()
        )

        head_dim = self.head_dim
        q_dim = self.q_weight_dim
        kv_dim = self.kv_weight_dim
        num_kv_heads = kv_dim // head_dim

        if self._has_v_proj:
            qkv = self.qkv_proj(x)
            x_q, x_k, x_v = ops.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)
        else:
            qk = self.qk_proj(x)
            x_q, x_k = ops.split(qk, [q_dim, kv_dim], axis=-1)
            x_v = x_k

        x_q = self.q_norm(x_q.reshape((-1, self.n_heads, head_dim))).reshape(
            (-1, q_dim)
        )
        x_k = self.k_norm(x_k.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )
        x_v = self.v_norm(x_v.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )

        qkv = ops.concat((x_q, x_k, x_v), axis=-1)
        rope = self.rope_local if self.use_local else self.rope_global
        freqs_cis = ops.cast(rope.freqs_cis, qkv.dtype).to(qkv.device)
        # Default position ids are cache_length + token index, which is
        # exactly the canvas position rule (positions continue after the
        # encoder sequence; HF DiffusionGemmaDecoderModel.forward).
        xq = rope_split_store_ragged(
            self.kv_params,
            qkv,
            kwargs["input_row_offsets"],
            freqs_cis,
            kv_collection,
            layer_idx,
            n_heads=self.n_heads,
            interleaved=rope.interleaved,
            q_out_dtype=self.kv_params.dtype,
        )
        xq = xq.reshape((-1, self.n_heads, self.head_dim))

        # The decoder attends bidirectionally over the canvas (and to the
        # read-only encoder cache), so both mask variants are noncausal.
        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK
            if self.use_local
            else MHAMaskVariant.NULL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size if self.use_local else -1,
            output_dtype=DType.bfloat16,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)


class DiffusionGemmaSelfConditioning(Module):
    """Gated-MLP self-conditioning block (decoder-only weights).

    Mirrors HF ``DiffusionGemmaSelfConditioning``: the soft-embedding signal
    from the previous denoise step is pre-normed, passed through a
    gelu-tanh-gated MLP, added to the canvas input embeddings, and the sum is
    RMS-normalized without a learned scale.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps: float,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.pre_norm = Gemma4RMSNorm(hidden_size, dtype, eps)
        self.post_norm = Gemma4RMSNorm(
            hidden_size, dtype, eps, with_weight=False
        )
        self.gate_proj = Linear(
            in_dim=hidden_size,
            out_dim=intermediate_size,
            dtype=dtype,
            device=device,
        )
        self.up_proj = Linear(
            in_dim=hidden_size,
            out_dim=intermediate_size,
            dtype=dtype,
            device=device,
        )
        self.down_proj = Linear(
            in_dim=intermediate_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
        )

    def __call__(
        self, inputs_embeds: TensorValue, sc_signal: TensorValue
    ) -> TensorValue:
        normed = self.pre_norm(sc_signal)
        gated = ops.gelu(self.gate_proj(normed), approximate="tanh")
        sc = self.down_proj(gated * self.up_proj(normed))
        return self.post_norm(inputs_embeds + sc)


class _DiffusionGemmaTextModelBase(DistributedLogitsPostprocessMixin, Module):
    """Shared construction for the encoder and decoder text graphs.

    Builds embeddings, final norm, tied lm_head, RoPE tables, and the 30
    Gemma4 decoder layers (always-on parallel MoE branch, fp32 router gate).
    Identical to the donor ``Gemma4TextModel.__init__`` except for the
    injected attention class and the fp32 gate. Both subclasses register the
    same weight FQNs, so loading them from one combined weights registry
    shares every tensor between the two graphs.
    """

    def __init__(
        self,
        config: DiffusionGemmaForBlockDiffusionConfig,
        attention_cls: type[Gemma4Attention],
    ) -> None:
        super().__init__()
        text_config = config.text_config
        self.config = config
        self.devices = config.devices

        rope_sliding, rope_global = self._build_ropes(config)
        unquantized_dtype = config.unquantized_dtype
        quant_config = text_config.quant_config
        embedding_output_dtype = config.dtype
        if quant_config and quant_config.embedding_output_dtype:
            embedding_output_dtype = quant_config.embedding_output_dtype

        self.embed_tokens = ScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            embedding_output_dtype,
            config.devices,
            embed_scale=text_config.hidden_size**0.5,
        )
        self.norm = Gemma4RMSNorm(
            text_config.hidden_size, unquantized_dtype, text_config.rms_norm_eps
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)
        self.lm_head = ColumnParallelLinear(
            text_config.hidden_size,
            text_config.vocab_size,
            dtype=unquantized_dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params_by_layer_type: dict[str, KVCacheParams] = {}
        for _k, _p in config.kv_params.children.items():
            assert isinstance(_p, KVCacheParams)
            kv_params_by_layer_type[_k] = _p
        layer_type_counts = {"sliding_attention": 0, "full_attention": 0}
        layers = []
        for i in range(text_config.num_hidden_layers):
            layer_type = text_config.layer_types[i]
            layers.append(
                self._build_decoder_layer(
                    config,
                    attention_cls,
                    rope_global,
                    rope_sliding,
                    layer_idx=i,
                    layer_idx_in_cache=layer_type_counts[layer_type],
                    is_sliding=layer_type == "sliding_attention",
                    kv_params=kv_params_by_layer_type[layer_type],
                )
            )
            layer_type_counts[layer_type] += 1

        self._layer_kv_index = [
            _LAYER_TYPE_TO_KV_INDEX[text_config.layer_types[i]]
            for i in range(text_config.num_hidden_layers)
        ]
        self.dim = text_config.hidden_size
        self.n_heads = text_config.num_attention_heads
        self.layers = LayerList(layers)
        self.kv_params = config.kv_params
        self.return_logits = text_config.return_logits
        self.final_logit_softcapping = text_config.final_logit_softcapping

    @staticmethod
    def _build_ropes(
        config: DiffusionGemmaForBlockDiffusionConfig,
    ) -> tuple[Llama3RotaryEmbedding, ProportionalRotaryEmbedding]:
        """Builds the (sliding default, global proportional) RoPE pair."""
        text_config = config.text_config
        rope_sliding = Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.sliding_window_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=None,
        )
        rope_global = ProportionalRotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.global_rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            head_dim=text_config.global_head_dim,
            interleaved=False,
            scaling_params=text_config.global_rope_scaling,
        )
        return rope_sliding, rope_global

    def _build_decoder_layer(
        self,
        config: DiffusionGemmaForBlockDiffusionConfig,
        attention_cls: type[Gemma4Attention],
        rope_global: ProportionalRotaryEmbedding,
        rope_sliding: Llama3RotaryEmbedding,
        *,
        layer_idx: int,
        layer_idx_in_cache: int,
        is_sliding: bool,
        kv_params: KVCacheParams,
    ) -> Gemma4TextDecoderLayer:
        """Builds one Gemma4 decoder layer (parallel dense-MLP + MoE branch).

        Mirrors the donor layer construction; the only DiffusionGemma delta is
        the fp32 router gate (``DiffusionGemmaMoEGate``).
        """
        text_config = config.text_config
        unquantized_dtype = config.unquantized_dtype
        quant_config = text_config.quant_config
        is_nvfp4 = quant_config is not None and quant_config.is_nvfp4
        moe_nvfp4 = is_nvfp4 and text_config.enable_moe_block

        # HF's DiffusionGemmaTextRouter runs the router softmax in fp32.
        moe_gate_cls = functools.partial(
            DiffusionGemmaMoEGate, eps=text_config.rms_norm_eps
        )
        moe_norm_cls = functools.partial(
            Gemma4RMSNorm,
            text_config.hidden_size,
            unquantized_dtype,
            eps=text_config.rms_norm_eps,
        )
        moe_act = make_concatenated_gated_activation_fn(
            functools.partial(ops.gelu, approximate="tanh")
        )
        moe_block: MoE
        if is_nvfp4:
            moe_block = MoEQuantized(
                devices=config.devices,
                hidden_dim=text_config.hidden_size,
                num_experts=text_config.num_experts,
                num_experts_per_token=text_config.top_k_experts,
                moe_dim=text_config.moe_intermediate_size,
                gate_cls=moe_gate_cls,
                gated_activation_fn=moe_act,
                pre_expert_norm_cls=moe_norm_cls,
                dtype=config.dtype,
                quant_config=quant_config,
            )
        else:
            moe_block = MoE(
                devices=config.devices,
                hidden_dim=text_config.hidden_size,
                num_experts=text_config.num_experts,
                num_experts_per_token=text_config.top_k_experts,
                moe_dim=text_config.moe_intermediate_size,
                gate_cls=moe_gate_cls,
                gated_activation_fn=moe_act,
                pre_expert_norm_cls=moe_norm_cls,
                dtype=config.dtype,
            )

        return Gemma4TextDecoderLayer(
            attention=attention_cls(
                rope_global=rope_global,
                rope_local=rope_sliding,
                num_attention_heads=text_config.num_attention_heads,
                num_key_value_heads=text_config.num_key_value_heads,
                num_global_key_value_heads=text_config.num_global_key_value_heads,
                attention_k_eq_v=text_config.attention_k_eq_v,
                hidden_size=text_config.hidden_size,
                kv_params=kv_params,
                global_head_dim=text_config.global_head_dim,
                layer_idx=layer_idx,
                layer_idx_in_cache=layer_idx_in_cache,
                is_sliding=is_sliding,
                dtype=unquantized_dtype if is_nvfp4 else config.dtype,
                devices=config.devices,
                qk_norm_eps=text_config.rms_norm_eps,
                local_window_size=text_config.sliding_window,
                quant_config=None if is_nvfp4 else quant_config,
            ),
            mlp=MLP(
                dtype=unquantized_dtype if moe_nvfp4 else config.dtype,
                quantization_encoding=None,
                hidden_dim=text_config.hidden_size,
                feed_forward_length=text_config.intermediate_size,
                devices=config.devices,
                activation_function=text_config.hidden_activation,
                quant_config=None if moe_nvfp4 else quant_config,
            ),
            hidden_size=text_config.hidden_size,
            rms_norm_eps=text_config.rms_norm_eps,
            devices=config.devices,
            unquantized_dtype=unquantized_dtype,
            enable_moe_block=text_config.enable_moe_block,
            moe_block=moe_block,
        )

    def _run_layers(
        self,
        h: list[TensorValue],
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        global_kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
        *,
        taps: list[TensorValue] | None = None,
        **kwargs: object,
    ) -> list[TensorValue]:
        kv_collections_by_type = [
            sliding_kv_collections,
            global_kv_collections,
        ]
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            kv_collections = kv_collections_by_type[self._layer_kv_index[idx]]
            h = layer(
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections,
                input_row_offsets=input_row_offsets,
                **kwargs,
            )
            if taps is not None:
                taps.append(ops.cast(h[0], DType.float32))
        return h

    def _softcap(self, logits: TensorValue) -> TensorValue:
        """fp32 ``tanh(l / cap) * cap`` (HF DiffusionGemmaForBlockDiffusion).

        Returns fp32 logits, matching HF's ``logits.to(torch.float32)``.
        """
        cap = self.final_logit_softcapping
        assert cap is not None, "DiffusionGemma sets final_logit_softcapping"
        l32 = ops.cast(logits, DType.float32)
        return ops.tanh(l32 / cap) * cap


class DiffusionGemmaEncoderTextModel(_DiffusionGemmaTextModelBase):
    """Causal text encoder: populates the paged KV cache.

    Same interface as the donor ``Gemma4TextModel.__call__`` (including the
    multimodal-embedding merge inputs, fed empty for text-only requests).
    Logits pass through the final softcap so encoder-side probes compare
    directly against the HF reference.
    """

    def __init__(self, config: DiffusionGemmaForBlockDiffusionConfig) -> None:
        super().__init__(config, attention_cls=Gemma4Attention)

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        global_kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        image_embeddings: Sequence[TensorValue],
        image_token_indices: Sequence[TensorValue],
        **kwargs: object,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)
        h = [
            merge_multimodal_embeddings(
                inputs_embeds=h_device,
                multimodal_embeddings=img_embed,
                image_token_indices=img_tok_indices,
            )
            for h_device, img_embed, img_tok_indices in zip(
                h, image_embeddings, image_token_indices, strict=True
            )
        ]
        h = self._run_layers(
            h,
            signal_buffers,
            sliding_kv_collections,
            global_kv_collections,
            input_row_offsets,
        )
        outputs = self._postprocess_logits(
            h, input_row_offsets, return_n_logits, signal_buffers
        )
        # Softcap the logit tensors; the optional third output is offsets.
        out = list(outputs)
        for i in range(min(2, len(out))):
            out[i] = self._softcap(out[i])
        return tuple(out)


class DiffusionGemmaDecoderTextModel(_DiffusionGemmaTextModelBase):
    """Bidirectional canvas denoiser with on-device sampling statistics.

    One call = one denoise step (HF ``_denoising_step`` minus the host-side
    acceptance/renoise logic):

    1. Canvas embeddings + self-conditioning signal (soft embeddings of the
       previous step's processed logits; zeroed via ``sc_enabled`` on the
       first step).
    2. 30 layers with noncausal attention over [encoder cache | canvas].
    3. Tied lm_head -> fp32 softcap -> divide by ``temperature`` (the linear
       schedule lives in the host loop).
    4. Outputs: processed logits as bf16 (next step's self-conditioning
       input, kept device-resident), per-token argmax, top-64
       probabilities/indices for host-side categorical sampling, and
       per-token full-vocab entropy.
    """

    def __init__(self, config: DiffusionGemmaForBlockDiffusionConfig) -> None:
        super().__init__(config, attention_cls=DiffusionGemmaDecoderAttention)
        text_config = config.text_config
        self.self_conditioning = DiffusionGemmaSelfConditioning(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            eps=text_config.rms_norm_eps,
            dtype=config.unquantized_dtype,
            device=config.devices[0],
        )

    def __call__(
        self,
        canvas_tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        global_kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
        sc_logits: TensorValue,
        sc_enabled: TensorValue,
        temperature: TensorValue,
        **kwargs: object,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(canvas_tokens, signal_buffers)

        # Soft embeddings: softmax(prev processed logits, fp32) @ embed_table
        # * embed_scale, gated to zero on the first denoise step (HF passes
        # self_conditioning_logits=None there).
        embed_weight = self.embed_tokens.weight.to(self.devices[0])
        soft_probs = ops.softmax(ops.cast(sc_logits, DType.float32))
        soft_embeds = ops.matmul(
            ops.cast(soft_probs, embed_weight.dtype), embed_weight
        )
        soft_embeds = (
            soft_embeds
            * ops.constant(
                self.embed_tokens.embed_scale,
                embed_weight.dtype,
                device=self.devices[0],
            )
            * ops.cast(sc_enabled, embed_weight.dtype)
        )
        h0 = self.self_conditioning(h[0], soft_embeds)

        # DG_DUMP_HIDDEN=1: emit fp32 per-layer hidden states as extra graph
        # outputs (tap 0 = post-self-conditioning layer input, taps 1..30 =
        # layer outputs) for the debug-model layer comparator.
        taps: list[TensorValue] | None = (
            [ops.cast(h0, DType.float32)]
            if os.environ.get("DG_DUMP_HIDDEN")
            else None
        )

        h = self._run_layers(
            [h0],
            signal_buffers,
            sliding_kv_collections,
            global_kv_collections,
            input_row_offsets,
            taps=taps,
            **kwargs,
        )

        normed = [
            shard(h_dev)
            for shard, h_dev in zip(self.norm_shards, h, strict=False)
        ]
        logits = self.lm_head(normed, signal_buffers)[0]
        capped = self._softcap(logits)  # fp32 [N, vocab]
        processed = capped / ops.cast(temperature, DType.float32)

        probs = ops.softmax(processed)
        argmax = ops.argmax(processed, axis=-1)

        # Entropy of Categorical(logits=processed):
        # H = logsumexp(processed) - sum(p * processed).
        m = ops.max(processed, axis=-1)
        lse = ops.log(ops.sum(ops.exp(processed - m), axis=-1)) + m
        entropy = lse - ops.sum(probs * processed, axis=-1)

        # Top-k probabilities for host-side categorical sampling. The exact
        # inverse-CDF needed ops.cumsum over [N, vocab] fp32, which has no
        # GPU kernel (KERN-1095) and cost two ~268 MB host transfers per
        # denoise step. Sampling from the renormalized top-64 instead ships
        # only [N, 64]; the truncated tail mass is negligible at the
        # schedule's temperatures, and the entropy/argmax used for
        # acceptance and stopping remain full-vocab and exact.
        topk_probs, topk_idx = ops.top_k(probs, k=64, axis=-1)

        sc_out = ops.cast(processed, DType.bfloat16)
        return (
            sc_out,
            ops.cast(argmax, DType.int64),
            ops.cast(topk_probs, DType.float32),
            ops.cast(topk_idx, DType.int64),
            ops.cast(entropy, DType.float32),
            *(taps or ()),
        )
