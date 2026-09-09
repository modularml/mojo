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
"""Nemotron-H nn.Module layers (hybrid Mamba-2 + NoPE attention + relu2 MLP).

Math mirrors the HuggingFace ``NemotronHForCausalLM`` reference
(``torch_forward`` for the mixer). Translated to idiomatic MAX:

* Block: pre-norm RMSNorm -> mixer -> residual add (residual_in_fp32=False).
* Mamba-2 mixer: in_proj -> [gate, hidden_states_B_C, dt]; depthwise SiLU conv
  over hidden_states_B_C; SSD chunked scan (prefill+decode);
  gated RMSNorm (norm_before_gate=False); out_proj.
* Attention: GQA, NoPE (no rotary), no bias.
* MLP: relu2 = down(relu(up(x))**2), non-gated, no bias.

NOTE: the SSD + varlen-conv ops are not yet in the builtin kernel library, so a
graph that instantiates the mamba mixer cannot compile until that handoff lands
(see KERNEL_HANDOFF_register_ssd_in_builtin.md). The layer math here is written
against the verified HF reference and is ready to logit-verify once invocable.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Sequence

from max.driver import accelerator_api
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationEncoding
from max.nn.attention import MHAMaskVariant
from max.nn.embedding import Embedding
from max.nn.kernels import (
    flash_attention_ragged,
    grouped_matmul_ragged,
    moe_create_indices,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, Linear
from max.nn.moe import MoE, MoEGate
from max.nn.norm import RMSNorm
from max.nn.quant_config import QuantConfig
from max.nn.state_space import (
    causal_conv1d_varlen_fwd,
    gated_group_rmsnorm,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace,
)
from max.nn.transformer import ReturnLogits, logits_postprocess

from .model_config import NemotronHConfig


def _relu2(x: TensorValue) -> TensorValue:
    """relu(x) ** 2 (Nemotron-H MLP activation, ``relu2``)."""
    r = ops.relu(x)
    return r * r


def _weight_dtype(
    model_dtype: DType, quant_config: QuantConfig | None
) -> DType:
    """Linear weight storage dtype. FP8 (per-tensor static) stores the weight
    as ``float8_e4m3fn``; the ``quantized_matmul`` kernel quantizes the bf16
    activation with ``input_scale`` and dequantizes. Otherwise the weight is the
    model dtype."""
    if quant_config is not None:
        return DType.float8_e4m3fn
    return model_dtype


def _ssm_state_dtype() -> DType:
    """SSM state-pool STORAGE dtype: bf16 on Apple GPUs, fp32 elsewhere.

    The Mamba-2 SSD scan always accumulates its recurrence in fp32 registers;
    this only selects the storage dtype of the persistent slot-indexed state
    pool. On Apple silicon the pool read+write dominates the decode-step
    traffic, and only the Apple GPU kernel implements the bf16 load ->
    fp32-accumulate -> round-on-store path, so bf16 is gated to Apple
    (matching the kernel-side guard in
    ``mamba2_ssd_chunk_scan_varlen_fwd_inplace``).
    """
    try:
        is_apple = accelerator_api() == "metal"
    except Exception:
        is_apple = False
    return DType.bfloat16 if is_apple else DType.float32


class NemotronHMLP(Module):
    """relu2 MLP: ``down(relu(up(x))**2)``. Non-gated, no bias."""

    def __init__(
        self,
        config: NemotronHConfig,
        *,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        dev = config.devices[0]
        wdtype = _weight_dtype(config.dtype, quant_config)
        self.up_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            dtype=wdtype,
            device=dev,
            has_bias=config.mlp_bias,
            quant_config=quant_config,
        )
        self.down_proj = Linear(
            in_dim=config.intermediate_size,
            out_dim=config.hidden_size,
            dtype=wdtype,
            device=dev,
            has_bias=config.mlp_bias,
            quant_config=quant_config,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.down_proj(_relu2(self.up_proj(x)))


class NemotronHExpertMLP(MLP):
    """Non-gated relu2 expert MLP: ``down(relu(up(x))**2)`` (no ``gate_proj``).

    Serves two roles in :class:`NemotronHMoE`:

    * routed experts (``mlp_cls``): the base :class:`~max.nn.moe.MoE` reads only
      ``.up_proj.weight`` / ``.down_proj.weight`` to build the grouped-matmul
      weight stacks and never calls ``__call__``, so the up/down Linears are all
      that is required;
    * the shared expert (``shared_mlp_cls``): ``MoE.__call__`` invokes
      ``shared_experts(x)`` directly, so ``__call__`` runs the relu2 MLP.

    The constructor keywords match what ``MoE`` passes to the expert MLP class.
    """

    def __init__(
        self,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        hidden_dim: int,
        feed_forward_length: int,
        devices: list[DeviceRef],
        quant_config: QuantConfig | None = None,
    ) -> None:
        # Subclass MLP so this is a valid `Callable[..., MLP]` expert factory
        # for the base MoE. Init the MLP base with is_sharding=True: it sets the
        # shared MLP attributes but skips building the gated gate/up/down
        # projections. This expert is non-gated (no gate_proj) and builds only
        # up/down below, so the base MLP projections must not be created.
        super().__init__(
            dtype=dtype,
            quantization_encoding=quantization_encoding,
            hidden_dim=hidden_dim,
            feed_forward_length=feed_forward_length,
            devices=devices,
            quant_config=quant_config,
            is_sharding=True,
        )
        dev = devices[0]
        # FP8 (per-tensor static) stores the expert weight as float8_e4m3fn;
        # the routed-expert grouped matmul widens it on load (weight-only) and
        # the shared expert runs the dense FP8 Linear path. bf16 otherwise.
        wdtype = _weight_dtype(dtype, quant_config)
        self.up_proj = Linear(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            dtype=wdtype,
            device=dev,
            has_bias=False,
            quant_config=quant_config,
        )
        self.down_proj = Linear(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            dtype=wdtype,
            device=dev,
            has_bias=False,
            quant_config=quant_config,
        )

    def __call__(self, x: TensorValueLike) -> TensorValue:
        x = TensorValue(x)
        return self.down_proj(_relu2(self.up_proj(x)))

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the expert MLP sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the sharding strategy for the up/down projections.

        Overrides :attr:`MLP.sharding_strategy`, which also shards
        ``gate_proj`` — this expert is non-gated and never builds one.
        Tensor parallelism shards ``up_proj`` rowwise and ``down_proj``
        columnwise, matching the base MLP.
        """
        self._sharding_strategy = strategy

        if strategy.is_replicate:
            self.up_proj.sharding_strategy = strategy
            self.down_proj.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            self.up_proj.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.down_proj.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
        else:
            raise ValueError(f"Unsupported sharding strategy: {strategy}")

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> Sequence[NemotronHExpertMLP]:
        """Creates sharded views of this expert MLP across multiple devices.

        Overrides :meth:`MLP.shard`, which shards ``gate_proj`` (absent
        here) and constructs base :class:`MLP` shards; this builds
        :class:`NemotronHExpertMLP` shards from the sharded up/down
        projections.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            Sharded :class:`NemotronHExpertMLP` instances, one per device.
        """
        if self.sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        sharded_up_projs = self.up_proj.shard(devices)
        sharded_down_projs = self.down_proj.shard(devices)

        shards: list[NemotronHExpertMLP] = []
        for device, up_proj, down_proj in zip(
            devices, sharded_up_projs, sharded_down_projs, strict=True
        ):
            sharded = NemotronHExpertMLP(
                dtype=self.up_proj.weight.dtype,
                quantization_encoding=self.quantization_encoding,
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.feed_forward_length,
                devices=[device],
                quant_config=self.quant_config,
            )
            sharded.up_proj = up_proj
            sharded.down_proj = down_proj
            # Keep the original weights reachable for stacking checks,
            # mirroring MLP.shard.
            sharded._parent_layer = self
            shards.append(sharded)

        return shards


class NemotronHMoEGate(MoEGate):
    """Sigmoid top-k router with an additive score-correction bias.

    Nemotron-3's router is DeepSeek-style — sigmoid gate scores, an additive
    ``e_score_correction_bias`` applied for *selection* only, and the pre-bias
    scores used for the *weights* — but with ``n_group == topk_group == 1`` the
    group-limited scheme degenerates to plain top-k. It therefore uses only
    ``ops.top_k`` + ``ops.gather`` (both have native Apple/Metal branches) and
    avoids ``moe_router_group_limited`` (``DeepseekV3TopKRouter``), whose
    warp-collective ``WARP_SIZE % group_size`` constraint fails at 128 experts
    with ``n_group == 1``.
    """

    def __init__(
        self,
        devices: list[DeviceRef],
        hidden_dim: int,
        num_experts: int,
        num_experts_per_token: int,
        dtype: DType,
        routed_scaling_factor: float,
        norm_topk_prob: bool,
        correction_bias_dtype: DType = DType.float32,
    ) -> None:
        super().__init__(
            devices=devices,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            dtype=dtype,
        )
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.e_score_correction_bias = Weight(
            "e_score_correction_bias",
            shape=[num_experts],
            device=devices[0],
            dtype=correction_bias_dtype,
        )

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        # Routing math in fp32 (matches the HF reference); the weights are cast
        # back to the activation dtype for the downstream combine matmul in
        # ``MoE.__call__``.
        scores = ops.sigmoid(self.gate_score(hidden_states).cast(DType.float32))
        bias = self.e_score_correction_bias.to(self.devices[0]).cast(
            DType.float32
        )
        # Select experts on the bias-corrected scores.
        sel = scores + bias
        topk_sel, topk_idx = ops.top_k(
            sel, k=self.num_experts_per_token, axis=-1
        )
        # Weight with the *pre-bias* sigmoid scores at the selected experts:
        # scores[t, e] == sel[t, e] - bias[e]. Subtracting the gathered bias
        # recovers those weights without a per-row take-along-axis gather
        # (``ops.gather`` is numpy-take, not torch.gather); gathering the 1-D
        # ``bias`` with the [tokens, k] indices yields ``bias[idx]`` directly.
        topk_weight = topk_sel - ops.gather(bias, topk_idx, axis=0)
        if self.norm_topk_prob:
            topk_weight = topk_weight / ops.sum(topk_weight, axis=-1)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight.cast(hidden_states.dtype)


class NemotronHMoE(MoE):
    """Nemotron-3 MoE mixer: 128 non-gated relu2 experts (top-6) + 1 shared.

    :attr:`gate_up_proj` is overridden: the experts are non-gated (relu2, no
    ``gate_proj``), so the routed-expert weight stack is the up-projection only,
    ``[num_experts, moe_intermediate_size, hidden]``.

    bf16 (no ``quant_config``) reuses the base :class:`~max.nn.moe.MoE`
    grouped-matmul routing verbatim, which on Apple resolves to MAX's own
    ``naive_grouped_matmul`` (the else-branch of ``grouped_matmul_ragged``).

    FP8 (per-tensor static ``quant_config``, the 30B-A3B) keeps the routed
    expert weights as ``float8_e4m3fn`` and feeds them to the SAME grouped
    matmul: the op is dtype-generic, so the naive kernel widens each E4M3 weight
    to fp32 on load (bf16 activation x fp8 weight -> fp32 accumulate -> bf16),
    i.e. weight-only W8A16 — the grouped analog of the dense Apple FP8 Linear.
    The per-tensor scalar ``weight_scale`` (one per expert) factors out of the
    matmul sum, so it is folded EXACTLY as a post-matmul per-row multiply
    (gathered by each permuted row's expert). The shared expert runs the dense
    FP8 Linear path. Activations stay bf16 (``input_scale`` is unused, matching
    the dense weight-only path).
    """

    @property
    def gate_up_proj(self) -> TensorValue:
        # Non-gated experts: the base property interleaves gate+up and would
        # AttributeError on our gate-less experts. Stack the up projections
        # only -> [num_experts, moe_intermediate_size, hidden]. FP8 experts
        # stack as float8_e4m3fn (weight-only); bf16 experts stack as bf16.
        return ops.stack(
            [expert.up_proj.weight for expert in self.experts], axis=0
        )

    def _expert_weight_scales(self, proj_name: str) -> TensorValue:
        """Stack the per-expert scalar ``weight_scale`` -> ``[num_experts]``.

        Each FP8 expert Linear carries one per-tensor ``weight_scale`` (shape
        ``()``); stacking yields the per-expert dequant factor gathered by the
        grouped matmul's permuted rows.
        """
        scales: list[TensorValue] = []
        for expert in self.experts:
            ws = getattr(expert, proj_name).weight_scale
            assert ws is not None, (
                f"FP8 MoE expert {proj_name} is missing weight_scale"
            )
            scales.append(ws.to(self.devices[0]))
        return ops.stack(scales, axis=0)

    def __call__(self, x: TensorValue) -> TensorValue:
        # bf16 MoE (no quant_config): the base grouped-matmul path is exact.
        if self.quant_config is None:
            return super().__call__(x)

        # FP8 weight-only (W8A16): mirrors the base routing, but the routed
        # expert grouped matmuls consume FP8-E4M3 weight stacks (widened to
        # fp32 on load by the dtype-generic naive grouped-matmul kernel) and the
        # per-expert scalar ``weight_scale`` is folded post-matmul. A per-tensor
        # scalar factors out of the sum, so the fold is exact (not merely within
        # tolerance) -- the grouped analog of the dense Apple FP8 Linear.
        seq_len = x.shape[0]
        router_idx, router_weight = self.gate(x)
        router_idx = ops.reshape(router_idx, [-1])

        (
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
        ) = moe_create_indices(
            ops.cast(router_idx, DType.int32), self.num_experts
        )

        permutated_states = ops.gather(
            x,
            ops.cast(
                ops.floor_div(token_expert_order, self.num_experts_per_token),
                DType.int32,
            ),
            axis=0,
        )

        # Per-permuted-row expert id -> per-row dequant scale. The expert of
        # permuted row p is ``router_idx[token_expert_order[p]]`` (the same
        # ordering the grouped matmul groups by), so a gather recovers each
        # row's ``weight_scale`` without a scatter over the CSR offsets.
        expert_per_row = ops.gather(
            ops.cast(router_idx, DType.int32), token_expert_order, axis=0
        )
        # Keep the per-row dequant scales in f32 (the stacked ``weight_scale``
        # is f32); the fold below multiplies in f32 and casts once, matching the
        # dense Apple FP8 path (`quant_ops.py`), where downcasting the scale to
        # bf16 before the multiply would lose mantissa bits.
        up_scale = ops.reshape(
            ops.gather(
                self._expert_weight_scales("up_proj"), expert_per_row, axis=0
            ),
            [-1, 1],
        )
        down_scale = ops.reshape(
            ops.gather(
                self._expert_weight_scales("down_proj"), expert_per_row, axis=0
            ),
            [-1, 1],
        )

        # Up projection (weight-only FP8 grouped matmul) -> dequant fold ->
        # relu2, all in f32. Folding the up scale BEFORE relu2 matches
        # dequantizing the weight then squaring (weight_scale >= 0, so relu
        # commutes with it); the trailing cast returns the model dtype for the
        # down matmul activation.
        up = grouped_matmul_ragged(
            permutated_states,
            self.gate_up_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        up = _relu2(up.cast(DType.float32) * up_scale).cast(x.dtype)

        # Down projection (weight-only FP8 grouped matmul) -> dequant fold in
        # f32, then cast to the model dtype.
        down = grouped_matmul_ragged(
            up,
            self.down_proj,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        )
        down = (down.cast(DType.float32) * down_scale).cast(x.dtype)

        down = ops.gather(down, restore_token_order, axis=0).reshape(
            [seq_len, self.num_experts_per_token, self.hidden_dim]
        )
        routed_expert_out = ops.unsqueeze(router_weight, axis=1) @ down
        routed_expert_out = ops.squeeze(routed_expert_out, axis=1).cast(x.dtype)

        if self.has_shared_experts:
            routed_expert_out += self.shared_experts(x)

        return routed_expert_out


class NemotronHAttention(Module):
    """GQA attention, NoPE (no rotary), no bias.

    Position information flows through the SSM layers, so attention layers add
    no positional encoding (NoPE) — matching the HF reference where
    ``position_embeddings`` is unused.
    """

    def __init__(
        self,
        config: NemotronHConfig,
        kv_layer_idx: int,
    ) -> None:
        super().__init__()
        dev = config.devices[0]
        self.kv_params: KVCacheParams = config.kv_params
        self.kv_layer_idx = kv_layer_idx
        # Preserve the activation dtype at the FP8 attention output boundary.
        self.dtype = config.dtype
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.scale = math.sqrt(1.0 / self.head_dim)

        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        # Attention projections are bf16 here: the 4B FP8 checkpoint excludes
        # them from quantization, and the 8B Reasoning checkpoint's per-tensor
        # FP8 q/k/v/o are dequantized to bf16 at load by the weight adapter. So
        # no quantization config is wired here.
        #
        # Fused QKV: one matmul over the concatenated [q | k | v] out-dim, then
        # ``ops.split``. The HF checkpoint ships q/k/v as separate weights; the
        # adapter concatenates them (q, then k, then v) into ``qkv_proj.weight``.
        # Unlike the mamba ``in_proj`` (whose merged split feeds a group-RMSNorm
        # reduce that misaligns on a strided source — see
        # known-limitations/strided-split-misaligns-gpu-group-reduce), the q/k/v
        # split chunks feed reshape -> flash_attention_ragged, a different kernel
        # path that tolerates the split-view stride. Validated on a minimized
        # smoke before the full serve.
        self.qkv_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=self.q_dim + 2 * self.kv_dim,
            dtype=config.dtype,
            device=dev,
            has_bias=config.attention_bias,
        )
        self.o_proj = Linear(
            in_dim=self.q_dim,
            out_dim=config.hidden_size,
            dtype=config.dtype,
            device=dev,
            has_bias=config.attention_bias,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        # One fused QKV matmul, then split into q | k | v (sizes q_dim, kv_dim,
        # kv_dim along the feature axis).
        qkv = self.qkv_proj(x)
        q, k, v = ops.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], axis=1)
        query = ops.reshape(q, [-1, self.n_heads, self.head_dim])
        key = ops.reshape(k, [-1, self.n_kv_heads, self.head_dim])
        value = ops.reshape(v, [-1, self.n_kv_heads, self.head_dim])

        # Paged stores require K/V to match the cache dtype, and FP8 flash
        # attention requires the query to match it as well. Convert the
        # attention output back to the activation dtype before ``o_proj``.
        if self.kv_params.is_fp8_kv_dtype:
            cache_dtype = self.kv_params.dtype
            query = ops.cast(query, cache_dtype)
            key = ops.cast(key, cache_dtype)
            value = ops.cast(value, cache_dtype)

        # NoPE: write K/V to cache as-is (no rotary), then ragged flash attn.
        store_k_cache_ragged(kv_collection, key, input_row_offsets, layer_idx)
        store_v_cache_ragged(kv_collection, value, input_row_offsets, layer_idx)

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=query,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
            output_dtype=self.dtype,
        )
        attn_out = ops.reshape(attn_out, [total_seq_len, -1])
        return self.o_proj(attn_out)


class NemotronHMamba2Mixer(Module):
    """Mamba-2 mixer (selective state-space), matching ``NemotronHMamba2Mixer``.

    State plumbing (carried by ``model.py``):
    * conv1d state lives in a slot-indexed pool mutated in place by
      ``causal_conv1d_varlen_fwd`` (handles prefill + decode).
    * SSM state lives in a slot-indexed pool (fp32; bf16 on Apple GPUs — see
      ``_ssm_state_dtype``) mutated in place by
      ``mamba2_ssd_chunk_scan_varlen_fwd_inplace``: the kernel reads initial
      state from ``ssm_pool[slot_idx[b]]`` and writes the final state back to
      the same slot without any graph-side gather/scatter_nd/buffer_store.
      The SSD kernel serves both prefill and decode (decode = seqlen-1 seqs).
    """

    def __init__(
        self,
        config: NemotronHConfig,
        *,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        dev = config.devices[0]
        self.config = config
        self.dtype = config.dtype
        self.nheads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.ngroups = config.n_groups
        self.dstate = config.ssm_state_size
        self.conv_kernel = config.conv_kernel
        self.intermediate = config.mamba_intermediate_size
        self.conv_dim = config.conv_dim
        self.group_size = self.intermediate // self.ngroups
        self.eps = config.layer_norm_epsilon

        # in_proj: hidden -> [gate(intermediate), hidden_states_B_C(conv_dim),
        # dt(nheads)]. ONE fused matmul matching the HF reference and vLLM (the
        # checkpoint ships a single ``in_proj.weight`` with ONE per-tensor FP8
        # weight_scale / input_scale, so the fused FP8 matmul is numerically
        # identical to three matmuls that each replicate that shared scale —
        # accuracy-neutral). Output is split into gate / hidden_BC / dt.
        #
        # The earlier 3-separate-Linears workaround existed because a naive
        # fused matmul + ``ops.split`` made a strided ``gate`` view whose row
        # stride (the full fused width 17504) broke the downstream gated
        # group-RMSNorm regroup+reduce on GPU
        # (known-limitations/strided-split-misaligns-gpu-group-reduce). That
        # limitation has since decayed: a minimized GPU repro at the exact real
        # geometry (fused 17504, group_size 960) feeds the strided bf16 ``gate``
        # straight into the group-RMSNorm reduce and runs clean (no
        # CUDA_ERROR_MISALIGNED_ADDRESS). ``_gated_group_rmsnorm`` also casts
        # ``gate`` -> fp32 as its first op, materializing a contiguous buffer.
        # Validated on the FP8 quantized path (mini-smoke + serve), not just the
        # bf16 repro, before claiming servable.
        wdtype = _weight_dtype(config.dtype, quant_config)
        self.in_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=self.intermediate + self.conv_dim + self.nheads,
            dtype=wdtype,
            device=dev,
            has_bias=config.mamba_proj_bias,
            quant_config=quant_config,
        )
        self.out_proj = Linear(
            in_dim=self.intermediate,
            out_dim=config.hidden_size,
            dtype=wdtype,
            device=dev,
            has_bias=config.mamba_proj_bias,
            quant_config=quant_config,
        )

        # Depthwise conv1d weight [conv_dim, 1, K] (reshaped in adapter) + bias.
        self.conv1d_weight = Weight(
            "conv1d.weight",
            config.dtype,
            [self.conv_dim, 1, self.conv_kernel],
            device=DeviceRef.CPU(),
        )
        self.conv1d_bias: Weight | None = None
        if config.use_conv_bias:
            self.conv1d_bias = Weight(
                "conv1d.bias",
                config.dtype,
                [self.conv_dim],
                device=DeviceRef.CPU(),
            )

        # Per-head scalar SSM params (fp32).
        self.A_log = Weight(
            "A_log", DType.float32, [self.nheads], device=DeviceRef.CPU()
        )
        self.D = Weight(
            "D", DType.float32, [self.nheads], device=DeviceRef.CPU()
        )
        self.dt_bias = Weight(
            "dt_bias", DType.float32, [self.nheads], device=DeviceRef.CPU()
        )

        # Gated RMSNorm weight (group-normed). Direct weight (offset 0).
        self.norm_weight = Weight(
            "norm.weight",
            DType.float32,
            [self.intermediate],
            device=DeviceRef.CPU(),
        )

    @property
    def mamba_in_proj_out(self) -> int:
        return self.intermediate + self.conv_dim + self.nheads

    def _gated_group_rmsnorm(
        self, y: TensorValue, gate: TensorValue
    ) -> TensorValue:
        """HF ``Zamba2RMSNormGated`` with ``norm_before_gate=False``.

        ``y`` and ``gate`` are ``[N, intermediate]``. One fused kernel replaces
        the ``cast -> silu(gate)*y -> group rms_norm -> *norm_weight -> cast``
        chain (which otherwise lowers to ~3-4 serial GPU dispatches per layer
        because the group ``rms_norm`` reduction is a hard fusion boundary). The
        kernel reproduces the reference bit-for-bit: silu-gate in fp32, group
        RMSNorm over ``group_size``, cast to the input dtype, multiply by the
        fp32 norm weight, then cast to the model dtype (folding the final
        ``out_proj`` cast — hence it returns ``y.dtype``).
        """
        return gated_group_rmsnorm(
            y,
            gate,
            self.norm_weight.to(y.device),
            self.eps,
            self.group_size,
        )

    def __call__(
        self,
        x: TensorValue,
        conv_pool: BufferValue,
        ssm_pool: BufferValue,
        has_initial_state: TensorValue,
        slot_idx: TensorValue,
        query_start_loc: TensorValue,
    ) -> TensorValue:
        """Returns ``output[N, hidden]``.

        Per-mamba-layer state lives in two mutable graph-input pools:

        * conv: ``causal_conv1d_varlen_fwd`` mutates ``conv_pool`` in place at
          slot ``cache_indices[b] = slot_idx[b]`` (Qwen3.5 conv pattern).
        * SSM: ``mamba2_ssd_chunk_scan_varlen_fwd_inplace`` reads initial state
          from ``ssm_pool[slot_idx[b]]`` and writes the updated final state back
          to the same slot in-place (no graph-side gather/scatter_nd/
          buffer_store whole-pool RMW).

        ``query_start_loc`` is the ragged ``input_row_offsets``; both kernels
        need it as int32.
        """
        device = x.device
        query_start_loc = ops.cast(query_start_loc, DType.int32)
        # ONE fused in_proj matmul, then split into gate / hidden_BC / dt along
        # the feature axis.
        proj = self.in_proj(x)  # [N, intermediate + conv_dim + nheads]
        # The ``gate`` chunk feeds the gated group-RMSNorm regroup+reduce, which
        # misaligns on a strided source (the fused matmul's row stride 17504 —
        # known-limitations/strided-split-misaligns-gpu-group-reduce; confirmed
        # still live on the FP8 path). Make ``gate`` a CONTIGUOUS fp32 buffer by
        # casting the whole fused output to fp32 BEFORE the split: the fp32
        # split chunk's 4-byte stride keeps the reduce aligned (validated by the
        # FP8 mini-smoke). ``_gated_group_rmsnorm`` then consumes an fp32 gate
        # directly (it casts gate->fp32 internally anyway). hidden_BC / dt are
        # split from the original bf16 ``proj`` (they feed conv / SSD kernels
        # that tolerate the split-view stride).
        proj_f32 = ops.cast(proj, DType.float32)
        gate = ops.slice_tensor(
            proj_f32, [slice(None), slice(0, self.intermediate)]
        )
        _, hidden_BC, dt = ops.split(
            proj, [self.intermediate, self.conv_dim, self.nheads], axis=1
        )

        # Depthwise SiLU conv over hidden_BC.
        conv_w = ops.reshape(
            self.conv1d_weight.to(device), [self.conv_dim, self.conv_kernel]
        )
        conv_bias = (
            self.conv1d_bias.to(device)
            if self.conv1d_bias is not None
            else ops.constant(0.0, self.dtype, device=device).broadcast_to(
                [self.conv_dim]
            )
        )
        # Slot-indexed in-place conv: the kernel reads+writes the conv pool at
        # slot ``cache_indices[b] = slot_idx[b]`` (Qwen3.5 GatedDeltaNet conv
        # pattern). No graph-side gather/scatter — the pool is mutated directly.
        # ``channels_last=True``: the op consumes/produces the tokens-major
        # [N, conv_dim] layout the surrounding graph already carries,
        # eliminating the materialized [conv_dim, N] transposes on both sides
        # of the conv (the dominant prefill glue-kernel cost, ~9.5 ms per
        # 4k-token request on B200).
        conv_out = causal_conv1d_varlen_fwd(
            x=hidden_BC,
            weight=conv_w,
            bias=conv_bias,
            conv_states=conv_pool,
            query_start_loc=query_start_loc,
            cache_indices=ops.cast(slot_idx, DType.int32),
            has_initial_state=has_initial_state,
            activation="silu",
            channels_last=True,
        )  # [N, conv_dim]

        # Split conv output: [hidden(intermediate), B(ng*ds), C(ng*ds)].
        gtss = self.ngroups * self.dstate
        hidden, B, C = ops.split(
            conv_out, [self.intermediate, gtss, gtss], axis=1
        )

        # Reshape for the SSD kernel.
        x_ssm = ops.reshape(hidden, [-1, self.nheads, self.head_dim])
        B = ops.reshape(B, [-1, self.ngroups, self.dstate])
        C = ops.reshape(C, [-1, self.ngroups, self.dstate])

        # A = -exp(A_log); the kernel applies dt softplus internally.
        A = ops.cast(ops.negate(ops.exp(self.A_log.to(device))), self.dtype)
        D = ops.cast(self.D.to(device), self.dtype)
        dt_bias = ops.cast(self.dt_bias.to(device), self.dtype)

        # In-place SSM-pool RMW: the kernel reads initial state from
        # ssm_pool[slot_idx[b]] (when has_initial_state[b]) and writes the
        # updated final state back to the same slot directly — no graph-side
        # buffer_load/gather/scatter_nd/buffer_store whole-pool round-trip.
        # This eliminates ~30% of decode GPU wall-clock (B200 profile).
        y = mamba2_ssd_chunk_scan_varlen_fwd_inplace(
            x=x_ssm,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            ssm_pool=ssm_pool,
            query_start_loc=query_start_loc,
            has_initial_state=has_initial_state,
            cache_indices=ops.cast(slot_idx, DType.uint32),
        )
        y = ops.reshape(y, [-1, self.intermediate])

        # Gated group RMSNorm (returns fp32), cast back to model dtype, out_proj
        # — matching the reference ``self.out_proj(scan_output.to(dtype))``.
        y = self._gated_group_rmsnorm(y, gate)
        return self.out_proj(ops.cast(y, self.dtype))


class NemotronHBlock(Module):
    """Pre-norm residual block dispatching to one of the three mixers."""

    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        kv_layer_idx: int,
        *,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.kind = config.layer_kinds[layer_idx]
        self.norm = RMSNorm(
            config.hidden_size,
            dtype=config.dtype,
            eps=config.layer_norm_epsilon,
            multiply_before_cast=False,
        )
        self.mixer: Module
        if self.kind == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, quant_config=quant_config)
        elif self.kind == "attention":
            self.mixer = NemotronHAttention(config, kv_layer_idx)
        elif self.kind == "moe":
            # Non-gated relu2 MoE. ``quant_config`` is set only when this MoE
            # layer is FP8 (the 30B-A3B FP8 checkpoint); it makes the routed +
            # shared expert weights float8_e4m3fn. When None the experts are
            # bf16 and NemotronHMoE.__call__ delegates to the base MoE.
            self.mixer = NemotronHMoE(
                devices=config.devices,
                hidden_dim=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                gate_cls=functools.partial(
                    NemotronHMoEGate,
                    routed_scaling_factor=config.routed_scaling_factor,
                    norm_topk_prob=config.norm_topk_prob,
                ),
                mlp_cls=NemotronHExpertMLP,
                shared_mlp_cls=NemotronHExpertMLP,
                has_shared_experts=True,
                shared_experts_dim=config.moe_shared_expert_intermediate_size,
                dtype=config.dtype,
                apply_router_weight_first=False,
                quant_config=quant_config,
                # Non-gated: relu2 over the whole up-projection (the moe_dim
                # split arg from the base MoE is ignored).
                gated_activation_fn=lambda gate_up, moe_dim: _relu2(gate_up),
            )
        else:  # mlp
            self.mixer = NemotronHMLP(config, quant_config=quant_config)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "NemotronHBlock dispatches per-kind in NemotronH.__call__; do not "
            "call the block directly."
        )


class NemotronH(Module):
    """Full Nemotron-H decoder: embed -> hybrid blocks -> norm -> lm_head."""

    def __init__(
        self,
        config: NemotronHConfig,
        *,
        quant_config: QuantConfig | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__()
        self.config = config
        self.kv_params = config.kv_params
        self.return_logits = return_logits
        dev = config.devices[0]

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            dev,
        )

        # Map each attention layer to a sequential KV cache index. FP8 is wired
        # per-module: a Linear is FP8 only if its layer is in the config's FP8
        # set (driven by the checkpoint's exclude list); attention, conv1d,
        # norms and lm_head always stay bf16.
        blocks: list[NemotronHBlock] = []
        self.block_kinds: list[str] = list(config.layer_kinds)
        self.mamba_layer_indices: list[int] = []
        kv_idx = 0
        for li, kind in enumerate(config.layer_kinds):
            kv_layer = kv_idx if kind == "attention" else -1
            qc: QuantConfig | None = None
            if kind == "mamba" and li in config.fp8_mamba_layers:
                qc = quant_config
            elif kind == "mlp" and li in config.fp8_mlp_layers:
                qc = quant_config
            elif kind == "moe" and li in config.fp8_moe_layers:
                qc = quant_config
            blocks.append(NemotronHBlock(config, li, kv_layer, quant_config=qc))
            if kind == "attention":
                kv_idx += 1
            if kind == "mamba":
                self.mamba_layer_indices.append(li)
        # LayerList so the Module weight-naming traversal prefixes each block's
        # weights as ``blocks.{i}.*`` (a plain Python list is not traversed).
        self.blocks = LayerList(blocks)

        self.norm_f = RMSNorm(
            config.hidden_size,
            dtype=config.dtype,
            eps=config.layer_norm_epsilon,
            multiply_before_cast=False,
        )
        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            dtype=config.dtype,
            device=dev,
            has_bias=False,
        )

        # Dims for state pool allocation (model.py reads these).
        self.num_mamba_layers = len(self.mamba_layer_indices)
        self.conv_dim = config.conv_dim
        self.conv_kernel = config.conv_kernel
        self.mamba_nheads = config.mamba_num_heads
        self.mamba_head_dim = config.mamba_head_dim
        self.dstate = config.ssm_state_size

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        return_n_logits: TensorValue,
        kv_collections: list[PagedCacheValues],
        slot_idx: TensorValue,
        conv_pools: list[BufferValue],
        ssm_pools: list[BufferValue],
        has_initial_state: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Run the hybrid stack.

        Returns the logits tuple from :func:`logits_postprocess`. The conv and
        SSM pools are mutable graph inputs mutated in place at slot
        ``slot_idx[batch_item]`` (the SSD kernel reads/writes the ssm_pool
        directly via ``mamba2_ssd_chunk_scan_varlen_fwd_inplace``), so the
        only graph outputs are the logits.
        """
        h = self.embed_tokens(tokens)

        kv_i = 0
        mamba_i = 0
        for block in self.blocks:
            residual = h
            normed = block.norm(h)
            if block.kind == "mamba":
                out = block.mixer(
                    normed,
                    conv_pools[mamba_i],
                    ssm_pools[mamba_i],
                    has_initial_state,
                    slot_idx,
                    input_row_offsets,
                )
                mamba_i += 1
            elif block.kind == "attention":
                # The KV cache holds all attention layers; ``layer_idx`` selects
                # this layer's slice. There is one PagedCacheValues per device
                # (single-device here), so always index [0] — not [kv_i].
                layer_idx = ops.constant(
                    kv_i, DType.uint32, device=DeviceRef.CPU()
                )
                out = block.mixer(
                    layer_idx,
                    normed,
                    kv_collections[0],
                    input_row_offsets,
                )
                kv_i += 1
            elif block.kind in ("moe", "mlp"):
                # MoE and MLP are both stateless (hidden states only; no
                # SSM/KV state), so they dispatch identically.
                out = block.mixer(normed)
            h = residual + out

        return logits_postprocess(
            h,
            input_row_offsets,
            return_n_logits,
            self.norm_f,
            self.lm_head,
            self.return_logits,
        )

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Graph input types for the Nemotron-H language graph.

        Order: ``tokens, input_row_offsets, return_n_logits, *kv_inputs,
        slot_idx, *conv_pools, *ssm_pools, has_initial_state``. The conv pools
        are model-dtype mutable buffers; the SSM pools are mutable buffers of
        ``_ssm_state_dtype()`` (fp32; bf16 on Apple GPUs, storage only — the
        scan accumulates in fp32) mutated in place by
        ``mamba2_ssd_chunk_scan_varlen_fwd_inplace`` at
        slot ``slot_idx[batch_item]``. ``has_initial_state`` is ``[batch]``
        bool (empty for a fresh prefill, all-True for decode).
        """
        dev = self.config.devices[0]
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=dev
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=dev
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        kv_types = list(kv_params.get_symbolic_inputs().flatten())

        slot_idx_type = TensorType(
            DType.uint32, shape=["batch_size"], device=dev
        )
        conv_pool_types: list[TensorType | BufferType] = [
            BufferType(
                self.config.dtype,
                shape=["max_slots", self.conv_dim, self.conv_kernel - 1],
                device=dev,
            )
            for _ in range(self.num_mamba_layers)
        ]
        ssm_pool_types: list[TensorType | BufferType] = [
            BufferType(
                _ssm_state_dtype(),
                shape=[
                    "max_slots",
                    self.mamba_nheads,
                    self.mamba_head_dim,
                    self.dstate,
                ],
                device=dev,
            )
            for _ in range(self.num_mamba_layers)
        ]
        has_initial_state_type = TensorType(
            DType.bool, shape=["has_initial_state_len"], device=dev
        )

        base: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]
        return tuple(
            base
            + kv_types
            + [slot_idx_type]
            + conv_pool_types
            + ssm_pool_types
            + [has_initial_state_type]
        )
