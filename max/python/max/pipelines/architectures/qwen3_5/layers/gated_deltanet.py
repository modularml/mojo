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

"""Gated DeltaNet (linear attention) layer for Qwen3.5.

This implements the linear attention mechanism used in 48 out of 64 layers
of Qwen3.5. It uses a gated delta rule recurrence with causal convolution
for sequence modeling without the quadratic cost of full attention.

State is held in two pools, one per layer per direction, that the kernels
mutate in place at slot ``slot_idx[batch_item]``:

- ``conv_pool``: sliding window of recent inputs for the causal conv1d.
  Shape ``[max_slots, conv_dim, kernel_size - 1]``.
- ``recurrent_pool``: the accumulated key-value memory.
  Shape ``[max_slots, num_v_heads, key_head_dim, value_head_dim]``.

Under tensor parallelism the heads are split, so ``conv_dim`` and
``num_v_heads`` above are the per-device widths and each device owns its own
pool pair per layer.

Both prefill (seq_len > 1) and decode (seq_len == 1) are handled by the
same two slot-indexed fused GPU kernels:

- Pass 1 (``gated_delta_conv1d_fwd``): one GPU thread per (batch_item,
  conv_channel). Each thread reads/writes its slot's window in place;
  no gather/scatter, no working buffers.

- Pass 2 (``gated_delta_recurrence_fwd``): one GPU thread per (batch_item,
  value_head, value_dim_element). Each thread owns a KD-element state
  column in registers and iterates over its sequence, applying the
  five-step gated delta rule, then writes the final state back into its
  slot. For decode (seqlen=1) the loop runs once.

This matches vLLM's ``selective_state_update`` design: the kernel does
pointer arithmetic ``state_ptr += slot * stride`` into a long-lived pool,
so there is no per-step pool allocation and no Python-level
gather/scatter loop.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.graph.weight import Segment
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.quant_config import QuantConfig
from max.nn.stacked_linear import StackedLinear
from max.nn.state_space import (
    gated_delta_conv1d_fwd,
    gated_delta_recurrence_fwd,
)


class GatedDeltaReplayInputs(NamedTuple):
    """One layer's per-token inputs to the two state kernels.

    Speculative decoding runs the verify pass over K+1 positions and then has
    to land the pools on the accepted prefix instead. Every op feeding these
    tensors is causal or pointwise, so re-running the two kernels over the
    accepted rows from the pre-verify state reproduces the state the verify
    pass held at that length. Captured only when the caller asks for it; see
    ``mach/docs/qwen38_27b/mtp-serving.md``.
    """

    qkv: TensorValue
    """``[total_seq_len, conv_dim]`` float32 conv input."""
    conv_weight: TensorValue
    """``[conv_dim, kernel_size]`` float32 depthwise weights."""
    decay: TensorValue
    """``[total_seq_len, num_value_heads]`` float32 decays."""
    beta: TensorValue
    """``[total_seq_len, num_value_heads]`` float32 beta gates."""


def _projection(stack: StackedLinear, name: str) -> Linear:
    """Returns one projection of an unfused stack.

    The children are set as dynamically named attributes, so this narrows
    what would otherwise be an untyped attribute read.
    """
    child = getattr(stack, name)
    assert isinstance(child, Linear)
    return child


class GatedDeltaNet(Module, Shardable):
    """Gated DeltaNet linear attention layer.

    This replaces standard attention in linear attention layers. It uses:
    1. Input projections (QKV + gate Z + beta B + decay A)
    2. Causal conv1d for local context
    3. Gated delta rule recurrence for long-range memory
    4. Gated RMSNorm on output

    Args:
        hidden_size: Input/output hidden dimension.
        num_key_heads: Number of key heads.
        num_value_heads: Number of value heads.
        key_head_dim: Dimension per key head.
        value_head_dim: Dimension per value head.
        conv_kernel_size: Kernel size for the causal conv1d.
        dtype: Weight data type for the unquantized weights and activations.
        device: Device for computation.
        rms_norm_eps: Epsilon for the gated RMSNorm.
        ssm_dtype: Dtype of the recurrence arithmetic.
        proj_dtype: Storage dtype of the quantized projections; defaults to
            ``dtype``.
        quant_config: Quantization of ``in_proj_qkv``, ``in_proj_z`` and
            ``out_proj``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        dtype: DType,
        device: DeviceRef,
        rms_norm_eps: float = 1e-6,
        ssm_dtype: DType = DType.float32,
        proj_dtype: DType | None = None,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.conv_kernel_size = conv_kernel_size
        self.dtype = dtype
        self.device = device
        self.ssm_dtype = ssm_dtype
        self.rms_norm_eps = rms_norm_eps
        self.quant_config = quant_config
        self._sharding_strategy: ShardingStrategy | None = None
        # Only in_proj_qkv, in_proj_z and out_proj are quantized; conv1d, the
        # b/a gates, the norm and every activation stay at `dtype`.
        proj_dtype = dtype if proj_dtype is None else proj_dtype
        self.proj_dtype = proj_dtype

        self.key_dim = key_head_dim * num_key_heads
        self.value_dim = value_head_dim * num_value_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Quantized checkpoints put in_proj_qkv and in_proj_z in FP8 but leave
        # the per-head in_proj_b and in_proj_a in bf16, so the four cannot
        # share one stacked matmul. Unquantized models keep the single
        # four-way stack the BF16 logit gate was measured against.
        self.in_proj = StackedLinear(
            in_dim=hidden_size,
            out_dims=(
                [self.conv_dim, self.value_dim]
                if quant_config
                else [
                    self.conv_dim,
                    self.value_dim,
                    num_value_heads,
                    num_value_heads,
                ]
            ),
            names=(
                ["in_proj_qkv", "in_proj_z"]
                if quant_config
                else [
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                ]
            ),
            dtype=proj_dtype,
            device=device,
            stacked=False,
            has_bias=False,
            quant_config=quant_config,
        )
        self.in_proj_ba: StackedLinear | None = (
            StackedLinear(
                in_dim=hidden_size,
                out_dims=[num_value_heads, num_value_heads],
                names=["in_proj_b", "in_proj_a"],
                dtype=dtype,
                device=device,
                stacked=False,
                has_bias=False,
            )
            if quant_config
            else None
        )

        # Causal conv1d weight (depthwise): [conv_dim, 1, kernel_size]
        # Stored as [conv_dim, kernel_size] in the checkpoint
        self.conv1d = Weight(
            "conv1d.weight",
            dtype,
            [self.conv_dim, 1, conv_kernel_size],
            device=DeviceRef.CPU(),
        )

        # Decay parameters
        self.dt_bias = Weight(
            "dt_bias", DType.float32, [num_value_heads], device=DeviceRef.CPU()
        )
        self.A_log = Weight(
            "A_log", DType.float32, [num_value_heads], device=DeviceRef.CPU()
        )

        # Gated RMSNorm: uses DIRECT weight (weight_offset=0.0), not (1+weight).
        # HF Qwen3NextRMSNormGated initializes weight to ones and applies
        # `weight * normalized`, unlike the regular Qwen3.5 RMSNorm which
        # uses (1 + weight) with zero-initialized weights.
        self.norm = RMSNorm(
            value_head_dim,
            dtype=DType.float32,
            eps=rms_norm_eps,
            weight_offset=0.0,
            multiply_before_cast=False,
        )

        # Output projection
        self.out_proj = Linear(
            in_dim=self.value_dim,
            out_dim=hidden_size,
            dtype=proj_dtype,
            device=device,
            has_bias=False,
            quant_config=quant_config,
        )

    @property
    def _ba_owner(self) -> StackedLinear:
        """Where ``in_proj_b`` / ``in_proj_a`` live.

        They join the four-way ``in_proj`` stack when the model is
        unquantized and split into ``in_proj_ba`` when it is not, because a
        quantized checkpoint leaves the two gates at the compute dtype.
        """
        return self.in_proj if self.in_proj_ba is None else self.in_proj_ba

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the layer's sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the layer by head, and propagates that to every weight.

        Both SSM kernels are per-``(batch_item, value_head)`` with no
        cross-head or cross-channel reduction, and the recurrence maps value
        head ``v`` onto key head ``v // (num_value_heads // num_key_heads)``.
        Cutting both head counts by the same factor therefore leaves the
        mapping intact on every device and the same compiled kernel runs on
        the shard.

        Args:
            strategy: Must be tensor-parallel; there is no data-parallel path.

        Raises:
            ValueError: If the strategy is not tensor-parallel, or if the
                device count does not divide either head count.
        """
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "GatedDeltaNet supports only tensor-parallel sharding, got "
                f"{strategy}"
            )
        num_devices = strategy.num_devices
        for count, name in (
            (self.num_key_heads, "num_key_heads"),
            (self.num_value_heads, "num_value_heads"),
        ):
            if count % num_devices:
                raise ValueError(
                    f"GatedDeltaNet {name} ({count}) must be divisible by the "
                    f"device count ({num_devices})"
                )

        self._sharding_strategy = strategy

        # `in_proj_qkv` and the depthwise conv share the conv channel layout
        # [Q | K | V], each block head-major. Splitting each block at its own
        # head boundary keeps Q/K/V aligned per device; a flat split of
        # `conv_dim` would hand a device K channels belonging to another
        # device's value heads.
        key_segment = Segment.head_aware(self.num_key_heads, self.key_head_dim)
        qkv_strategy = ShardingStrategy.segmented(
            num_devices,
            axis=0,
            segments=(
                key_segment,
                key_segment,
                Segment.head_aware(self.num_value_heads, self.value_head_dim),
            ),
        )
        _projection(
            self.in_proj, "in_proj_qkv"
        ).sharding_strategy = qkv_strategy
        self.conv1d.sharding_strategy = qkv_strategy

        # Everything else is indexed by value head alone.
        value_rows = ShardingStrategy.rowwise(num_devices)
        _projection(self.in_proj, "in_proj_z").sharding_strategy = value_rows
        for name in ("in_proj_b", "in_proj_a"):
            _projection(self._ba_owner, name).sharding_strategy = value_rows
        self.dt_bias.sharding_strategy = value_rows
        self.A_log.sharding_strategy = value_rows

        # The gated norm reduces over `value_head_dim` inside one head, so it
        # needs no cross-device statistic.
        self.norm.sharding_strategy = ShardingStrategy.replicate(num_devices)

        # Row-parallel: each device holds the columns of its own value heads
        # and produces a partial sum the caller all-reduces.
        self.out_proj.sharding_strategy = (
            ShardingStrategy.head_aware_columnwise(
                num_devices, self.num_value_heads, self.value_head_dim
            )
        )

    def shard(self, devices: Iterable[DeviceRef]) -> list[GatedDeltaNet]:
        """Creates one per-device view of this layer, split by head.

        Args:
            devices: Devices to place the shards on.

        Returns:
            One :class:`GatedDeltaNet` per device, each dimensioned for its
            own head slice.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if self._sharding_strategy is None:
            raise ValueError(
                "GatedDeltaNet cannot be sharded because no sharding strategy "
                "was provided."
            )
        devices = list(devices)
        num_devices = len(devices)

        in_proj_shards = self.in_proj.shard(devices)
        ba_shards = (
            self.in_proj_ba.shard(devices)
            if self.in_proj_ba is not None
            else None
        )
        conv1d_shards = self.conv1d.shard(devices)
        dt_bias_shards = self.dt_bias.shard(devices)
        a_log_shards = self.A_log.shard(devices)
        norm_shards = self.norm.shard(devices)
        out_proj_shards = self.out_proj.shard(devices)

        shards: list[GatedDeltaNet] = []
        for i, device in enumerate(devices):
            shard = GatedDeltaNet(
                hidden_size=self.hidden_size,
                num_key_heads=self.num_key_heads // num_devices,
                num_value_heads=self.num_value_heads // num_devices,
                key_head_dim=self.key_head_dim,
                value_head_dim=self.value_head_dim,
                conv_kernel_size=self.conv_kernel_size,
                dtype=self.dtype,
                device=device,
                rms_norm_eps=self.rms_norm_eps,
                ssm_dtype=self.ssm_dtype,
                proj_dtype=self.proj_dtype,
                quant_config=self.quant_config,
            )
            shard.in_proj = in_proj_shards[i]
            if ba_shards is not None:
                shard.in_proj_ba = ba_shards[i]
            shard.conv1d = conv1d_shards[i]
            shard.dt_bias = dt_bias_shards[i]
            shard.A_log = a_log_shards[i]
            shard.norm = norm_shards[i]
            shard.out_proj = out_proj_shards[i]
            shards.append(shard)
        return shards

    def __call__(
        self,
        x: TensorValue,
        conv_pool: BufferValue,
        recurrent_pool: BufferValue,
        slot_idx: TensorValue,
        input_row_offsets: TensorValue,
        replay_capture: list[GatedDeltaReplayInputs] | None = None,
    ) -> TensorValue:
        """Forward pass through the Gated DeltaNet layer.

        The conv and recurrent state pools live in graph-input buffers that
        the slot-indexed SSM kernels mutate in place at slot
        ``slot_idx[batch_item]``; there are no graph outputs for the new
        state. This matches vLLM's ``selective_state_update`` design and
        avoids per-decode pool allocation.

        Args:
            x: Input hidden states ``[total_seq_len, hidden_size]``.
            conv_pool: Per-layer conv pool (mutable),
                ``[max_slots, conv_dim, kernel_size - 1]``.
            recurrent_pool: Per-layer recurrent pool (mutable),
                ``[max_slots, num_v_heads, key_head_dim, value_head_dim]``.
            slot_idx: ``[batch_size]`` uint32 slot indices into the pools.
            input_row_offsets: Row offsets ``[batch_size + 1]`` (uint32).
            replay_capture: When given, this call's
                :class:`GatedDeltaReplayInputs` are appended to it so a
                speculative rollback can re-run the two state kernels over a
                shorter prefix.

        Returns:
            Output hidden states ``[total_seq_len, hidden_size]``.
        """
        device = x.device
        nv = self.num_value_heads
        vd = self.value_head_dim
        K = self.conv_kernel_size

        # ---- Projections (all tokens, fully parallel) ----
        if self.in_proj_ba is None:
            qkv, z, b_proj, a_proj = ops.split(
                self.in_proj(x),
                [self.conv_dim, self.value_dim, nv, nv],
                axis=-1,
            )
        else:
            qkv, z = ops.split(
                self.in_proj(x), [self.conv_dim, self.value_dim], axis=-1
            )
            b_proj, a_proj = ops.split(self.in_proj_ba(x), [nv, nv], axis=-1)
        qkv_f32 = ops.cast(qkv, DType.float32)  # [N, conv_dim]

        # ---- Decay / beta params ----
        dt_bias = self.dt_bias.to(device)
        A_log = self.A_log.to(device)
        A = ops.exp(ops.cast(A_log, self.ssm_dtype))
        a_float = ops.cast(a_proj, self.ssm_dtype)  # [N, nv]
        # Stabilised softplus: for x>20 return x directly (avoids float32 overflow)
        x_sp = a_float + ops.cast(dt_bias, self.ssm_dtype)
        softplus_val = ops.where(
            x_sp > ops.constant(20.0, self.ssm_dtype, device=device),
            x_sp,
            ops.log(
                ops.constant(1.0, self.ssm_dtype, device=device) + ops.exp(x_sp)
            ),
        )
        # Cast to float32 for downstream recurrence arithmetic (q/k/v ops always float32).
        decay = ops.exp(ops.cast(-A * softplus_val, DType.float32))  # [N, nv]
        beta = ops.cast(ops.sigmoid(b_proj), DType.float32)  # [N, nv] float32

        # ---- Conv weight (loaded once, shared) ----
        conv_weight_f32 = ops.cast(self.conv1d.to(device), DType.float32)
        conv_weight_flat = ops.reshape(
            conv_weight_f32, [self.conv_dim, K]
        )  # [conv_dim, K]

        # ---- Two-pass fused kernel path (handles both prefill and decode) ----
        # Pass 1: causal conv1d — one GPU thread per (batch_item, conv_channel)
        # Pass 2: gated delta recurrence — one GPU thread per
        #         (batch_item, value_head, vd_element); state column lives in
        #         registers. For decode (seqlen=1) both loops execute once.
        # The pools are mutable graph inputs at the model's native dtype
        # (typically bf16); the kernels cast on read/write so the per-token
        # working tensors stay at fp32.
        offsets_uint32 = ops.cast(input_row_offsets, DType.uint32)
        slot_idx_uint32 = ops.cast(slot_idx, DType.uint32)

        if replay_capture is not None:
            replay_capture.append(
                GatedDeltaReplayInputs(
                    qkv=qkv_f32,
                    conv_weight=conv_weight_flat,
                    decay=decay,
                    beta=beta,
                )
            )

        conv_output_ragged = gated_delta_conv1d_fwd(
            qkv_input_ragged=qkv_f32,
            conv_weight=conv_weight_flat,
            conv_state=conv_pool,
            slot_idx=slot_idx_uint32,
            input_row_offsets=offsets_uint32,
        )
        conv_output_ragged = ops.silu(conv_output_ragged)

        recurrence_output_flat = gated_delta_recurrence_fwd(
            qkv_conv_output=conv_output_ragged,
            decay_per_token=decay,
            beta_per_token=beta,
            recurrent_state=recurrent_pool,
            slot_idx=slot_idx_uint32,
            input_row_offsets=offsets_uint32,
        )

        output_flat = ops.rebind(
            recurrence_output_flat,
            [x.shape[0], self.value_dim],
            "recurrence_output_flat total_seq_len rebind",
        )

        # ---- Post-process: gated RMSNorm + output projection ----
        output_3d = ops.cast(
            ops.reshape(output_flat, [-1, nv, vd]),
            x.dtype,
        )
        output_normed = self.norm(output_3d)  # [N, nv, vd]

        z_reshaped = ops.reshape(z, [-1, nv, vd])
        z_gate = ops.silu(ops.cast(z_reshaped, DType.float32))
        output_gated = ops.cast(output_normed, DType.float32) * z_gate
        output_gated = ops.cast(output_gated, x.dtype)

        result = self.out_proj(ops.reshape(output_gated, [-1, self.value_dim]))
        return result
