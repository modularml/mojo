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
"""Eagle3 MHA draft model for DeepseekV3-shaped MLA targets.

This is the Llama-style MHA counterpart to :class:`Eagle3MLADraft`. The target
is a DeepseekV3-shaped MLA model (DeepseekV3 or Kimi K2.5) and produces
per-device hidden states; the draft swaps in a single MHA decoder block whose
KV cache geometry is independent of the target's. The unified graph wires a
separate ``PagedCacheValues`` per device for this draft.

The draft fuses two or three captured target hidden states via ``fc`` (width
detected from the checkpoint) and concatenates the result with the token
embedding before a single MHA attention block. Layout matches
``EagleLlama3``'s 2-way fusion when the draft checkpoint emits a
[hidden*2, hidden] ``fc.weight``, and matches the existing
``Eagle3MLADraft`` 3-way fusion otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.attention.attention_with_rope import (
    AttentionWithRope,
    DataParallelAttentionWithRope,
    TensorParallelAttentionWithRope,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
    RotaryEmbedding,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.distributed_transformer import (
    extract_hs,
    forward_sharded_layers,
)
from max.nn.transformer.transformer import fuse_captured_hidden_states


@dataclass(kw_only=True)
class Eagle3MHADraftConfig:
    """Minimal config for an Eagle3 MHA draft over a DeepseekV3-shaped MLA target.

    Held separate from ``DeepseekV3Config`` so MLA-specific fields
    (``kv_lora_rank``, ``v_head_dim``, etc.) and validators don't apply to
    the MHA draft.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    devices: list[DeviceRef]
    data_parallel_degree: int
    dtype: DType
    norm_dtype: DType
    kv_params: KVCacheParams
    rope_scaling: dict[str, Any] | None = None
    """Yarn rope scaling params (Deepseek-flavored: beta_fast, beta_slow,
    mscale, mscale_all_dim, factor, original_max_position_embeddings).

    Optional: when ``None``, the caller must pass a pre-built ``rope`` to
    :class:`Eagle3MHADraft` via its constructor kwarg. This lets M3-style
    targets whose draft checkpoint has ``rope_scaling=None`` inject a plain
    :class:`RotaryEmbedding` (or the target's partial-RoPE flavor) without
    faking yarn parameters just to pass this validator."""

    fc_input_multiplier: int
    """Number of fused target hidden states (2 or 3). Set from the
    ``fc.weight`` shape in the draft checkpoint at load time."""

    sliding_window: int | None = None
    """If set, the draft attention uses a sliding-window causal mask of
    this size (in tokens). ``None`` keeps the default full causal mask."""

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE

    sampling_logits_dtype: DType = DType.float32
    """Dtype exposed to the sampling consumer."""

    fc_norm: bool = False
    """Apply a separate RMSNorm to each captured target hidden-state chunk
    before the ``fc`` fusion (EAGLE3 ``fc_norm: true``). The chunks come from
    target layers at different depths with very different magnitudes, and ``fc``
    is trained on per-chunk-normalized inputs, so skipping this collapses the
    draft."""


def _sampling_logits_output(
    logits: TensorValue, output_dtype: DType
) -> TensorValue:
    """Returns LM-head logits in the dtype their sampling consumer requires."""
    if output_dtype in (DType.bfloat16, DType.float16):
        assert logits.dtype == output_dtype, (
            f"native {output_dtype} sampling logits require an LM-head output "
            f"of the same dtype, got {logits.dtype}"
        )
        return logits
    if output_dtype != DType.float32:
        raise ValueError(
            "Eagle3 MHA draft sampling logits must be float32, bfloat16 or "
            f"float16, got {output_dtype}"
        )
    return ops.cast(logits, DType.float32)


def project_captured_hidden_states(
    captures_per_device: Sequence[Sequence[TensorValue]],
    *,
    fc_norm_shards: Sequence[Sequence[Callable[[TensorValue], TensorValue]]]
    | None,
    fc_shards: Sequence[Callable[[TensorValue], TensorValue]],
) -> list[TensorValue]:
    """Applies the per-capture ``fc_norm``, then the ``fc`` fusion."""
    num_devices = len(captures_per_device)
    num_captures = len(captures_per_device[0])
    per_capture = [
        [captures_per_device[d][j] for d in range(num_devices)]
        for j in range(num_captures)
    ]
    if fc_norm_shards is not None:
        per_capture = [
            forward_sharded_layers(fc_norm_shards[j], per_capture[j])
            for j in range(num_captures)
        ]
    fused = fuse_captured_hidden_states(
        [
            [per_capture[j][d] for j in range(num_captures)]
            for d in range(num_devices)
        ]
    )
    return forward_sharded_layers(fc_shards, fused)


class Eagle3MHADraft(Module):
    """Eagle3 MHA draft over a DeepseekV3-shaped MLA target.

    Build the draft from an :class:`Eagle3MHADraftConfig`. The pipeline
    constructs this config from the draft checkpoint at load time:

    .. code-block:: python

        from max.driver import Accelerator, accelerator_count
        from max.dtype import DType
        from max.graph import DeviceRef
        from max.nn.kv_cache import MHAKVCacheParams
        from max.pipelines.architectures.eagle_common.eagle_mha_draft import (
            Eagle3MHADraft,
            Eagle3MHADraftConfig,
        )

        # Eagle3MHADraft builds an Allreduce across its devices, so it needs
        # at least one accelerator.
        if accelerator_count() > 0:
            devices = [DeviceRef.from_device(Accelerator())]
            config = Eagle3MHADraftConfig(
                hidden_size=512,
                num_attention_heads=8,
                num_key_value_heads=8,
                head_dim=64,
                intermediate_size=1024,
                vocab_size=32000,
                rms_norm_eps=1e-5,
                rope_theta=10000.0,
                max_position_embeddings=2048,
                devices=devices,
                data_parallel_degree=1,
                dtype=DType.float32,
                norm_dtype=DType.float32,
                kv_params=MHAKVCacheParams(
                    dtype=DType.float32,
                    n_kv_heads=8,
                    head_dim=64,
                    num_layers=1,
                    devices=devices,
                    page_size=128,
                ),
                rope_scaling={
                    "factor": 1.0,
                    "original_max_position_embeddings": 2048,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                },
                fc_input_multiplier=2,
            )
            draft = Eagle3MHADraft(config)

    A forward pass is a symbolic graph build inside a
    :class:`~max.graph.Graph`. Its ``__call__`` contract mirrors
    :class:`Eagle3MLADraft` so the unified graph can swap drafts without
    changing the call site, invoked with the per-device tensors the target
    produces:

    .. code-block:: text

        draft(
            tokens,
            fused_target_hs,        # per-device list[TensorValue] or
                                    # list[list[TensorValue]]
            signal_buffers,
            kv_collections,         # per-device list[PagedCacheValues]
            return_n_logits,
            input_row_offsets,      # per-device list[TensorValue]
            host_input_row_offsets,
            data_parallel_splits,
            batch_context_lengths,
            split_prefix=...,
        )

    See :meth:`__call__` for the full argument types.
    """

    def __init__(
        self,
        config: Eagle3MHADraftConfig,
        *,
        rope: RotaryEmbedding | None = None,
    ) -> None:
        """Build the draft module.

        Args:
            config: Draft configuration. When ``config.rope_scaling`` is set
                and ``rope`` is not, a Deepseek-flavored yarn RoPE is built
                from those params (the original K2.5 behavior). When ``rope``
                is passed, it is used verbatim and ``config.rope_scaling`` is
                ignored — this is the path M3-style targets take, since their
                Llama-Eagle3 draft checkpoint (``LlamaForCausalLMEagle3`` with
                ``rope_scaling: null``) needs a plain / partial RoPE built by
                the caller from the target's head geometry.
            rope: Optional pre-built rotary embedding. If ``None`` and
                ``config.rope_scaling`` is also ``None``, this raises.
        """
        super().__init__()
        self.config = config
        devices = config.devices
        num_devices = len(devices)
        device0 = devices[0]
        dtype = config.dtype
        norm_dtype = config.norm_dtype

        assert config.fc_input_multiplier > 1, (
            f"fc_input_multiplier must be at least 2, got "
            f"{config.fc_input_multiplier}"
        )

        self.dp_degree = config.data_parallel_degree
        assert num_devices % self.dp_degree == 0, (
            f"num_devices={num_devices} not divisible by "
            f"data_parallel_degree={self.dp_degree}"
        )
        self.tp_degree = num_devices // self.dp_degree
        # Pure TP (EP): one replica spanning all devices.
        self.use_tp_ep = self.dp_degree == 1 and num_devices > 1
        # Pure DP: one replica per device.
        self.use_data_parallel_attention = (
            num_devices > 1 and self.dp_degree == num_devices
        )
        # Mixed TP+DP (e.g. TP4DP2): ``dp_degree`` replicas, each spanning
        # ``tp_degree`` devices. Mirrors ``MiniMaxM3TransformerBlock``: attention
        # is sharded per replica group and the post-attention / post-MLP
        # collectives are ``reducescatter.sum`` / ``allgather`` with
        # ``group_size=tp_degree`` on the FULL signal-buffer set (no subset
        # signal-buffer collectives).
        self.use_tp_dp = num_devices > 1 and 1 < self.dp_degree < num_devices

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            devices=devices,
            quantization_encoding=None,
        )

        # fc: fuses the captured target hidden states.
        # fc_input_multiplier=2  → [seq, 2*hidden] -> [seq, hidden]
        # fc_input_multiplier=3  → [seq, 3*hidden] -> [seq, hidden]
        # The Llama-style EAGLE3 checkpoints we target ship a 2-way fc
        # (Kimi's existing MLA draft ships a 3-way fc); pipeline_model
        # detects which one and passes the multiplier.
        self.fc = Linear(
            config.hidden_size * config.fc_input_multiplier,
            config.hidden_size,
            dtype,
            device0,
            quantization_encoding=None,
            has_bias=False,
        )
        self.fc.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.fc_shards = self.fc.shard(devices)

        if rope is not None:
            self.rope: RotaryEmbedding = rope
        else:
            if config.rope_scaling is None:
                raise ValueError(
                    "Eagle3MHADraft requires either an explicit ``rope`` "
                    "kwarg or ``config.rope_scaling`` (Deepseek-yarn) so it "
                    "can build the default yarn RoPE. Got neither."
                )
            # Deepseek-flavored yarn rope (matches the draft HF config's
            # rope_scaling block; the per-head q/k dim is the MHA head_dim).
            scaling_params = DeepseekYarnRopeScalingParams(
                scaling_factor=config.rope_scaling["factor"],
                original_max_position_embeddings=config.rope_scaling[
                    "original_max_position_embeddings"
                ],
                beta_fast=config.rope_scaling["beta_fast"],
                beta_slow=config.rope_scaling["beta_slow"],
                mscale=config.rope_scaling["mscale"],
                mscale_all_dim=config.rope_scaling["mscale_all_dim"],
            )
            self.rope = DeepseekYarnRotaryEmbedding(
                config.head_dim,
                n_heads=config.num_attention_heads,
                theta=config.rope_theta,
                max_seq_len=config.max_position_embeddings,
                scaling_params=scaling_params,
                interleaved=False,
            )

        wide_hidden_size = config.hidden_size * 2
        attn_kwargs: dict[str, Any] = dict(
            rope=self.rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=wide_hidden_size,
            kv_params=config.kv_params,
            devices=devices,
            dtype=dtype,
            has_bias=False,
            sliding_window=config.sliding_window,
        )
        if config.sliding_window is not None:
            # The flash-attention kernel only honors ``local_window_size`` when
            # the mask variant is a windowed one; the default CAUSAL_MASK would
            # silently run full causal and ignore the window.
            attn_kwargs["mask_variant"] = (
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            )
        if self.use_data_parallel_attention:
            self.self_attn: AttentionWithRope = DataParallelAttentionWithRope(
                **attn_kwargs
            )
        elif self.use_tp_dp:
            # Mixed TP+DP: unwrapped base attention sharded per replica group.
            # We do NOT use ``TensorParallelAttentionWithRope`` (its internal
            # allreduce runs over the full device set with no group_size); the
            # post-attention reduce is done externally in ``__call__`` with
            # ``reducescatter.sum(..., group_size=tp_degree)``.
            self.self_attn = AttentionWithRope(**attn_kwargs)
            self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
                self.tp_degree
            )
        elif num_devices > 1:
            self.self_attn = TensorParallelAttentionWithRope(**attn_kwargs)
        else:
            self.self_attn = AttentionWithRope(**attn_kwargs)

        # Replacement o_proj: [n_heads * head_dim] -> [hidden_size]
        # (NOT wide_hidden_size). Same trick as Eagle3MLADraft: the
        # attention reads a wide concat but the residual addition runs on
        # the regular hidden_size, so the output must be narrow.
        q_weight_dim = config.num_attention_heads * config.head_dim
        replacement_o_proj = Linear(
            q_weight_dim,
            config.hidden_size,
            dtype,
            device0,
            quantization_encoding=None,
        )
        if self.use_tp_ep:
            replacement_o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices,
                    config.num_attention_heads,
                    config.head_dim,
                )
            )
        elif self.use_tp_dp:
            replacement_o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    self.tp_degree,
                    config.num_attention_heads,
                    config.head_dim,
                )
            )
        else:
            replacement_o_proj.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        self.self_attn.o_proj = replacement_o_proj

        # ``list_of_attentions`` is the per-device attention shard list used by
        # the multi-device call paths. The TP wrapper builds it at construction;
        # mixed TP+DP builds it here, one shard per device, sharding each
        # replica's ``tp_degree`` group independently (mirrors
        # ``MiniMaxM3TransformerBlock.self_attn_shards``).
        self.list_of_attentions: list[AttentionWithRope]
        if self.use_tp_dp:
            o_proj_shards = []
            self.list_of_attentions = []
            for start in range(0, num_devices, self.tp_degree):
                grp = devices[start : start + self.tp_degree]
                o_proj_shards.extend(replacement_o_proj.shard(grp))
                self.list_of_attentions.extend(self.self_attn.shard(grp))
            for shard_idx, attn_shard in enumerate(self.list_of_attentions):
                attn_shard.o_proj = o_proj_shards[shard_idx]
        else:
            o_proj_shards = replacement_o_proj.shard(devices)
            shards_seq = getattr(
                self.self_attn,
                "replicated_attentions",
                None,
            )
            if shards_seq is None:
                shards_seq = getattr(self.self_attn, "list_of_attentions", None)
            if shards_seq is not None:
                for shard_idx, attn_shard in enumerate(shards_seq):
                    attn_shard.o_proj = o_proj_shards[shard_idx]

        def _replicated_rmsnorm() -> RMSNorm:
            n = RMSNorm(
                config.hidden_size,
                norm_dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,
            )
            n.sharding_strategy = ShardingStrategy.replicate(num_devices)
            return n

        self.input_layernorm = _replicated_rmsnorm()
        self.input_layernorm_shards = self.input_layernorm.shard(devices)

        self.hidden_norm = _replicated_rmsnorm()
        self.hidden_norm_shards = self.hidden_norm.shard(devices)

        # Per-chunk pre-``fc`` norms (EAGLE3 ``fc_norm``). One RMSNorm per fused
        # target hidden state; ``LayerList`` gives ``fc_norm.{i}.weight`` FQNs
        # matching the checkpoint. Each is replicated and sharded like the other
        # norms; applied only at step 0 (when the ``fc`` fusion runs).
        self.fc_norm: LayerList | None = None
        if config.fc_norm:
            self.fc_norm = LayerList(
                [
                    _replicated_rmsnorm()
                    for _ in range(config.fc_input_multiplier)
                ]
            )
            self.fc_norm_shards = [norm.shard(devices) for norm in self.fc_norm]

        self.post_attention_layernorm = _replicated_rmsnorm()
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(devices)
        )

        self.mlp = MLP(
            dtype=dtype,
            quantization_encoding=None,
            hidden_dim=config.hidden_size,
            feed_forward_length=config.intermediate_size,
            devices=devices,
            quant_config=None,
        )
        if self.use_tp_ep:
            self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
        else:
            self.mlp.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.mlp_shards = list(self.mlp.shard(devices))

        self.norm = _replicated_rmsnorm()
        self.norm_shards = self.norm.shard(devices)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype,
            devices=devices,
            quantization_encoding=None,
        )

        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.sampling_logits_dtype = config.sampling_logits_dtype
        self.logits_scaling = 1.0

    def __call__(
        self,
        tokens: TensorValue,
        fused_target_hs: Sequence[TensorValue]
        | Sequence[Sequence[TensorValue]],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: list[TensorValue],
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        split_prefix: str = "eagle3_mha_draft",
    ) -> tuple[TensorValue, ...]:
        """Forward pass.

        Args mirror :meth:`Eagle3MLADraft.__call__` so the unified module can
        switch implementations without changing the call site.
        """
        config = self.config
        devices = config.devices
        num_devices = len(devices)

        # Wrap a bare per-device hidden so capture count identifies the step.
        captures_per_dev: list[list[TensorValue]] = []
        for per_device in fused_target_hs:
            if isinstance(per_device, TensorValue):
                captures_per_dev.append([per_device])
            else:
                captures_per_dev.append(list(per_device))
        assert len(captures_per_dev) == num_devices, (
            f"expected {num_devices} per-device hidden states, got "
            f"{len(captures_per_dev)}"
        )
        counts = {len(c) for c in captures_per_dev}
        assert counts in ({1}, {config.fc_input_multiplier}), (
            f"every device must carry 1 hidden state or all "
            f"{config.fc_input_multiplier} captures, got {sorted(counts)}"
        )
        if counts != {1}:
            fused_hs = project_captured_hidden_states(
                captures_per_dev,
                fc_norm_shards=self.fc_norm_shards
                if self.fc_norm is not None
                else None,
                fc_shards=self.fc_shards,
            )
        else:
            fused_hs = [c[0] for c in captures_per_dev]

        h_embed = self.embed_tokens(tokens, signal_buffers)

        freqs_cis = [self.rope.freqs_cis.to(device) for device in devices]
        input_row_offsets_ = list(input_row_offsets)

        if self.use_data_parallel_attention:
            host_offsets_i64 = host_input_row_offsets.cast(DType.int64)
            h_embed, input_row_offsets_ = split_batch_replicated(
                list(devices),
                h_embed,
                input_row_offsets_,
                host_offsets_i64,
                data_parallel_splits,
                prefix=split_prefix,
            )
            h_embed = [
                ops.rebind(
                    h_embed[i],
                    [f"{split_prefix}_seq_dev_{i}", config.hidden_size],
                )
                for i in range(num_devices)
            ]
            fused_hs = [
                ops.rebind(
                    fused_hs[i],
                    [f"{split_prefix}_seq_dev_{i}", config.hidden_size],
                )
                for i in range(num_devices)
            ]
        elif self.use_tp_dp:
            # Mixed TP+DP: ``tokens`` is the full merged batch broadcast to
            # every device by ``embed_tokens``. Split it per DP replica on the
            # replica leader devices, then broadcast each replica's slice to
            # its TP group (the "broadcast-full-and-slice-locally" workaround,
            # since grouped broadcast is unavailable). Mirrors
            # ``MiniMaxM3.__call__`` (minimax_m3.py). ``fused_hs`` already
            # arrives per-device TP-replicated from the target, so it is only
            # rebound (not re-split) to the per-device split dim so the concat
            # with ``h_embed`` shares a symbolic seq length.
            host_offsets_i64 = host_input_row_offsets.cast(DType.int64)
            replica_leader_indices = [
                r * self.tp_degree for r in range(self.dp_degree)
            ]
            replica_h_embed, replica_offsets = split_batch_replicated(
                [devices[i] for i in replica_leader_indices],
                [h_embed[i] for i in replica_leader_indices],
                [input_row_offsets_[i] for i in replica_leader_indices],
                host_offsets_i64,
                data_parallel_splits,
                prefix=split_prefix,
            )
            # Grouped allgather (group_size=tp) instead of dp_degree*2 full-set
            # broadcasts. Each replica's split slice lives only on its leader
            # rank; the other tp-1 ranks contribute an empty slice, and a
            # tp-scoped allgather concatenates within each TP group so every
            # rank receives its replica's slice (the empties add no rows, so the
            # result is identical to the broadcast-and-slice above). Replaces 4
            # all-device broadcast barriers -- which synchronize BOTH DP
            # replicas and let one replica's skew stall the other -- with 2
            # allgather barriers scoped to a single replica's TP group.
            h_gather_inputs: list[TensorValue] = []
            off_gather_inputs: list[TensorValue] = []
            for i in range(num_devices):
                d, r = divmod(i, self.tp_degree)
                if r == 0:
                    # Replica leader: contribute the real split slice.
                    h_gather_inputs.append(replica_h_embed[d])
                    off_gather_inputs.append(replica_offsets[d])
                else:
                    # Non-leader: contribute an empty (0-row) slice on this
                    # device so the group allgather sees a matching shape.
                    h_gather_inputs.append(
                        ops.slice_tensor(
                            h_embed[i],
                            [(slice(0, 0), f"{split_prefix}_empty_h_{i}"), ...],
                        )
                    )
                    off_gather_inputs.append(
                        ops.slice_tensor(
                            input_row_offsets_[i],
                            [(slice(0, 0), f"{split_prefix}_empty_off_{i}")],
                        )
                    )
            bcast_h_embed = ops.allgather(
                h_gather_inputs,
                signal_buffers,
                axis=0,
                group_size=self.tp_degree,
            )
            bcast_offsets = ops.allgather(
                off_gather_inputs,
                signal_buffers,
                axis=0,
                group_size=self.tp_degree,
            )
            input_row_offsets_ = bcast_offsets
            # Rebind to a per-REPLICA seq dim (not per-device): every device in
            # a TP group holds the same replica sequence, so they must share one
            # symbolic dim or ``reducescatter.sum`` / ``allgather`` (which check
            # shape-equality within each group) reject the mismatched symbols.
            # ``h_embed`` (from the draft's own split) and ``fused_hs`` (from the
            # target) arrive with different symbols but equal runtime lengths;
            # binding both to ``seq_replica_{r}`` also lets them concat.
            h_embed = [
                ops.rebind(
                    bcast_h_embed[i],
                    [
                        f"{split_prefix}_seq_replica_{i // self.tp_degree}",
                        config.hidden_size,
                    ],
                )
                for i in range(num_devices)
            ]
            fused_hs = [
                ops.rebind(
                    fused_hs[i],
                    [
                        f"{split_prefix}_seq_replica_{i // self.tp_degree}",
                        config.hidden_size,
                    ],
                )
                for i in range(num_devices)
            ]
        else:
            common_dim = f"{split_prefix}_seq_len"
            h_embed = [
                ops.rebind(h_embed[i], [common_dim, config.hidden_size])
                for i in range(num_devices)
            ]
            fused_hs = [
                ops.rebind(fused_hs[i], [common_dim, config.hidden_size])
                for i in range(num_devices)
            ]
            # Pure TP (EP): every rank holds the same replicated sequence, so
            # the per-device row offsets (distinct symbolic names from the
            # caller's per-device broadcast/range) are runtime-identical. Bind
            # them to one shared dim so the last-token gather yields a matching
            # row count on every rank; otherwise the vocab-parallel lm_head
            # allgather rejects the mismatched per-device symbols.
            common_offsets_dim = f"{split_prefix}_offsets_len"
            input_row_offsets_ = [
                ops.rebind(input_row_offsets_[i], [common_offsets_dim])
                for i in range(num_devices)
            ]

        norm_embed = forward_sharded_layers(
            self.input_layernorm_shards, h_embed
        )
        norm_fused = forward_sharded_layers(self.hidden_norm_shards, fused_hs)
        concat_inputs = [
            ops.concat([norm_embed[i], norm_fused[i]], axis=-1)
            for i in range(num_devices)
        ]

        layer_idx_cpu = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

        attn_outs: list[TensorValue]
        if self.use_data_parallel_attention:
            assert isinstance(self.self_attn, DataParallelAttentionWithRope)
            attn_outs = self.self_attn(
                layer_idx_cpu,
                concat_inputs,
                kv_collections,
                freqs_cis,
                input_row_offsets_,
            )
        elif self.use_tp_ep:
            assert isinstance(self.self_attn, TensorParallelAttentionWithRope)
            attn_outs = self.self_attn(
                layer_idx_cpu,
                concat_inputs,
                signal_buffers,
                kv_collections,
                freqs_cis,
                input_row_offsets_,
            )
        elif self.use_tp_dp:
            # Per-replica sharded attention: call each device's shard directly
            # (no wrapper allreduce); the post-attention reduce runs externally.
            attn_outs = [
                self.list_of_attentions[i](
                    layer_idx_cpu,
                    concat_inputs[i],
                    kv_collections[i],
                    freqs_cis[i],
                    input_row_offsets_[i],
                )
                for i in range(num_devices)
            ]
        else:
            single_out = self.self_attn(
                layer_idx_cpu,
                concat_inputs[0],
                kv_collections[0],
                freqs_cis[0],
                input_row_offsets_[0],
            )
            attn_outs = [single_out]

        if self.use_tp_dp:
            # M3-target residual pattern: only the leader TP rank adds the
            # residual; ``reducescatter.sum`` over the TP group sums the shards
            # (adding the residual exactly once) and leaves each rank with its
            # own 1/tp token slice. The dense MLP then runs replicated on that
            # slice, and a group ``allgather`` re-materializes the full replica
            # sequence. All collectives use the FULL signal-buffer set with
            # ``group_size=tp_degree`` — no subset-buffer collectives.
            hs_pre = [
                (fused + attn_out).cast(fused.dtype)
                if i % self.tp_degree == 0
                else attn_out
                for i, (fused, attn_out) in enumerate(
                    zip(fused_hs, attn_outs, strict=True)
                )
            ]
            hs = ops.reducescatter.sum(
                hs_pre, signal_buffers, axis=0, group_size=self.tp_degree
            )
            norm_outs = forward_sharded_layers(
                self.post_attention_layernorm_shards, hs
            )
            mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)
            hs = [
                (h + mlp_out).cast(h.dtype)
                for h, mlp_out in zip(hs, mlp_outs, strict=True)
            ]
            hs = ops.allgather(
                hs, signal_buffers, axis=0, group_size=self.tp_degree
            )
        else:
            hs = [
                fused + attn_out
                for fused, attn_out in zip(fused_hs, attn_outs, strict=True)
            ]

            norm_outs = forward_sharded_layers(
                self.post_attention_layernorm_shards, hs
            )
            mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)
            if self.use_tp_ep:
                mlp_outs = ops.allreduce.sum(mlp_outs, signal_buffers)
            hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        if self.use_tp_dp:
            # Within a replica all TP ranks hold the same sequence, so only the
            # first TP rank contributes real last-token rows; the others pass an
            # empty slice into a full-group allgather (mirrors MiniMaxM3's
            # last-token gather). This avoids counting each replica's rows
            # ``tp_degree`` times.
            last_token_per_dev: list[TensorValue] = []
            for dev_idx in range(num_devices):
                lt = ops.gather(
                    hs[dev_idx],
                    input_row_offsets_[dev_idx][1:] - 1,
                    axis=0,
                )
                if dev_idx % self.tp_degree != 0:
                    lt = ops.slice_tensor(
                        lt,
                        [(slice(0, 0), f"empty_last_token_{dev_idx}"), ...],
                    )
                last_token_per_dev.append(lt)
            last_token_distributed = ops.allgather(
                last_token_per_dev, signal_buffers
            )
        elif config.data_parallel_degree > 1:
            last_token_per_dev = []
            for dev_idx in range(num_devices):
                h0 = hs[dev_idx]
                last_token_indices = input_row_offsets_[dev_idx][1:] - 1
                last_token_per_dev.append(
                    ops.gather(h0, last_token_indices, axis=0)
                )
            last_token_distributed = ops.allgather(
                last_token_per_dev, signal_buffers
            )
        else:
            last_token_distributed = [
                ops.gather(h_i, offsets_i[1:] - 1, axis=0)
                for h_i, offsets_i in zip(hs, input_row_offsets_, strict=True)
            ]

        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = _sampling_logits_output(
            self.lm_head(norm_last_token, signal_buffers)[0],
            self.sampling_logits_dtype,
        )

        ret_val: tuple[TensorValue, ...] = (last_logits,)

        if self.return_logits == ReturnLogits.VARIABLE:
            # Compute the range on device 0 and broadcast to all
            # devices. Using distributed_broadcast instead of per-device
            # .to() copies avoids cross-stream D2D event sync that
            # breaks CUDA graph capture. Per-device ops.range with a
            # shared out_dim was also attempted and hit "input device
            # gpu:0 must match result device gpu:1 in rebind()" — the
            # shared symbolic dim triggers a cross-device rebind
            # downstream.
            draft_return_n_logits_range = ops.range(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="draft_mha_return_n_logits_range",
                dtype=DType.int64,
                device=devices[0],
            )
            draft_return_n_logits_range_per_dev = ops.distributed_broadcast(
                draft_return_n_logits_range, signal_buffers
            )
            variable_per_dev: list[TensorValue] = []
            for dev_idx in range(num_devices):
                dev_offsets = (
                    ops.unsqueeze(input_row_offsets_[dev_idx][1:], -1)
                    - draft_return_n_logits_range_per_dev[dev_idx]
                )
                dev_indices = ops.reshape(dev_offsets, shape=(-1,))
                gathered = ops.gather(hs[dev_idx], dev_indices, axis=0)
                # Mixed TP+DP: only the leader TP rank per replica contributes
                # real rows; empty-slice the others before the full-group
                # allgather so no verify row is counted ``tp_degree`` times.
                if self.use_tp_dp and dev_idx % self.tp_degree != 0:
                    gathered = ops.slice_tensor(
                        gathered,
                        [(slice(0, 0), f"empty_variable_{dev_idx}"), ...],
                    )
                variable_per_dev.append(gathered)
            if self.use_data_parallel_attention or self.use_tp_dp:
                variable_per_dev = ops.allgather(
                    variable_per_dev, signal_buffers
                )

            variable_logits = _sampling_logits_output(
                self.lm_head(
                    forward_sharded_layers(self.norm_shards, variable_per_dev),
                    signal_buffers,
                )[0],
                self.sampling_logits_dtype,
            )
            logit_offsets = ops.range(
                0,
                TensorValue(variable_logits.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="draft_mha_logit_offsets",
                dtype=DType.int64,
                device=devices[0],
            )
            ret_val += (variable_logits, logit_offsets)

        ret_val += extract_hs(
            return_hidden_states=self.return_hidden_states,
            last_token_hs_distributed=last_token_distributed,
            all_hs_distributed=hs,
            normalizer=self.norm_shards,
        )

        return ret_val
