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
"""Kimi Delta Attention: GLM-5.3-Flash's 34 linear-attention layers.

Linear attention drops the softmax so the sum over past positions reassociates
into a fixed ``[head_dim, head_dim]`` state, and the delta rule replaces what
that state already predicts for ``k`` rather than adding to it. KDA's one
change from Gated DeltaNet is that the forget gate decays the state **per
channel** -- ``alpha_t`` is a ``head_dim``-wide vector per head, so
``S_t = diag(alpha_t) S_{t-1} + k_t delta_t^T``. MAX's
:mod:`~max.nn.state_space.gated_delta` recurrence takes one decay per head per
token and cannot express that, which is why this is a sibling of Qwen3.5's
``GatedDeltaNet`` rather than a branch inside it.

Per token: ``q/k/v_proj`` 4096 -> 8192 each, concatenated to 24576 and run
through one depthwise causal conv1d of kernel 4 then ``silu``; the forget gate
``f_a_proj`` -> ``f_b_proj`` plus ``dt_bias`` and ``A_log``; the input gate
``b_proj``; the recurrence over a ``[64, 128, 128]`` state; the output gate
``g_a_proj`` -> ``g_b_proj``; a gated RMSNorm over ``head_dim``; ``o_proj``
8192 -> 4096.

Three things here are silently wrong if reordered, and each is pinned by a test
under ``max/tests/integration/architectures/glm5_next/kda/`` rather than by a
comment:

* **The conv channel order is q, k, v.** The weight adapter concatenates the
  three ``[8192, 1, 4]`` checkpoint tensors in that order and this layer
  concatenates its three projections the same way. Permuting either scrambles
  channels without changing a shape.
* **The gated norm normalizes first.** ``o_norm`` is
  ``rms_norm(y) * weight * sigmoid(gate)`` in float32, not
  ``rms_norm(gate_activation(gate) * y)``, which is what MAX's fused
  ``gated_group_rmsnorm`` computes.
* **Tensor parallelism splits heads inside each of q, k and v.** A contiguous
  slice of the 24576-wide concatenated conv would cut across the projection
  boundary: at TP8 the per-rank width is 3072 either way, but ``8192 / 3072``
  is not integral, so ranks 2 and 5 would get a mix of two projections.
"""

from __future__ import annotations

from collections.abc import Iterable

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
from max.nn.comm import Allreduce
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.state_space import gated_delta_conv1d_fwd, kda_decode
from max.nn.state_space.kimi_delta import (
    KDA_GATE_LOWER_BOUND,
    KdaGateMode,
    KdaStateLayout,
)

from ..model_config import Glm5NextConfig
from .kda import KdaReplayInputs, KdaSublayerInputs

__all__ = ["Glm5NextKdaSublayer", "KimiDeltaAttention"]

STATE_LAYOUT: KdaStateLayout = "K_FIRST"
"""Pool axis order, ``[max_slots, heads, head_dim, head_dim]``.

Pinned here so both recurrence ops, the pool allocator and the memory planner
cannot disagree; it also matches the ``[max_slots, num_v_heads, key_dim,
val_dim]`` layout Qwen3.5's state cache already allocates, which core clones.
"""


def _gate_mode(lower_bound: float | None) -> KdaGateMode:
    """Selects the kernel's forget-gate form from the config's lower bound.

    Raises:
        ValueError: If the bound is neither absent nor the one value the
            ``"safe"`` kernel path hard-codes. The constant lives in the
            kernel, so a checkpoint with a different bound would run and be
            quietly wrong.
    """
    if lower_bound is None:
        return "original"
    if lower_bound != KDA_GATE_LOWER_BOUND:
        raise ValueError(
            f"KDA's bounded forget gate is compiled with a lower bound of "
            f"{KDA_GATE_LOWER_BOUND}, but this checkpoint asks for "
            f"{lower_bound}. Thread a new compile-time axis through the kernel "
            "rather than accepting the mismatch."
        )
    return "safe"


class KimiDeltaAttention(Module, Shardable):
    """One KDA layer on one device.

    Args:
        hidden_size: Model width.
        num_heads: KDA heads, shared by q, k and v -- there is no GQA here.
        head_dim: Per-head width of q, k, v and both state axes.
        conv_kernel_size: Depthwise causal conv1d kernel size.
        dtype: Dtype of the projections and of this layer's output.
        device: Device this shard computes on.
        rms_norm_eps: Epsilon of the output gated RMSNorm.
        lower_bound: The checkpoint's bounded-gate lower bound, or ``None``
            for the softplus gate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        dtype: DType,
        device: DeviceRef,
        rms_norm_eps: float,
        lower_bound: float | None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size
        self.dtype = dtype
        self.device = device
        self.rms_norm_eps = rms_norm_eps
        self.lower_bound = lower_bound
        self.gate_mode = _gate_mode(lower_bound)
        self._sharding_strategy: ShardingStrategy | None = None

        self.qkv_dim = num_heads * head_dim
        self.conv_dim = 3 * self.qkv_dim

        # Three separate projections, matching the checkpoint. They are
        # concatenated into one conv input at call time; fusing them into a
        # stacked weight would have to undo the adapter's per-tensor layout.
        def head_major_projection() -> Linear:
            return Linear(
                in_dim=hidden_size,
                out_dim=self.qkv_dim,
                dtype=dtype,
                device=device,
                has_bias=False,
            )

        self.q_proj = head_major_projection()
        self.k_proj = head_major_projection()
        self.v_proj = head_major_projection()

        self.conv1d = Weight(
            "conv1d.weight",
            dtype,
            [self.conv_dim, 1, conv_kernel_size],
            device=DeviceRef.CPU(),
        )

        self.f_a_proj = Linear(
            in_dim=hidden_size,
            out_dim=head_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.f_b_proj = Linear(
            in_dim=head_dim,
            out_dim=self.qkv_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        # Both are added to the gate pre-activation inside the kernel, so they
        # are float32 here and stay float32 all the way in.
        self.dt_bias = Weight(
            "dt_bias", DType.float32, [self.qkv_dim], device=DeviceRef.CPU()
        )
        self.A_log = Weight(
            "A_log", DType.float32, [num_heads], device=DeviceRef.CPU()
        )

        self.b_proj = Linear(
            in_dim=hidden_size,
            out_dim=num_heads,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.g_a_proj = Linear(
            in_dim=hidden_size,
            out_dim=head_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.g_b_proj = Linear(
            in_dim=head_dim,
            out_dim=self.qkv_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

        # The reference's RMSNormGated multiplies by the weight directly rather
        # than by `1 + weight`, and holds the whole computation in float32.
        self.o_norm = RMSNorm(
            head_dim,
            dtype=DType.float32,
            eps=rms_norm_eps,
            weight_offset=0.0,
            multiply_before_cast=False,
        )
        self.o_proj = Linear(
            in_dim=self.qkv_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    @property
    def _omit_module_attr_name(self) -> bool:
        """Keeps the weight FQNs at ``self_attn.q_proj.weight``.

        This layer is always held by :class:`Glm5NextKdaSublayer`, which is what
        the decoder layer binds as ``self_attn``. The checkpoint puts the KDA
        weights directly under ``self_attn``, so the intermediate attribute
        name must not appear in between.
        """
        return True

    def conv_pool_shape(self, max_slots: int) -> list[int]:
        """Returns this shard's conv state pool shape."""
        return [max_slots, self.conv_dim, self.conv_kernel_size - 1]

    def recurrent_pool_shape(self, max_slots: int) -> list[int]:
        """Returns this shard's recurrent state pool shape.

        Laid out for :data:`STATE_LAYOUT`.
        """
        return [max_slots, self.num_heads, self.head_dim, self.head_dim]

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the layer's sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the layer by head and propagates that to every weight.

        Args:
            strategy: Must be tensor-parallel; there is no data-parallel path.

        Raises:
            ValueError: If the strategy is not tensor-parallel, or the device
                count does not divide the head count.
        """
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "KimiDeltaAttention supports only tensor-parallel sharding, "
                f"got {strategy}"
            )
        num_devices = strategy.num_devices
        if self.num_heads % num_devices:
            raise ValueError(
                f"KimiDeltaAttention num_heads ({self.num_heads}) must be "
                f"divisible by the device count ({num_devices})"
            )
        self._sharding_strategy = strategy

        # `conv1d` is the only weight whose rows span all three projections:
        # the adapter concatenated q, k and v along the channel axis. Splitting
        # each block at its own head boundary keeps the three aligned with the
        # three per-device projections that feed them. A flat split of
        # `conv_dim` has the same per-device width and the wrong channels.
        head_rows = ShardingStrategy.rowwise(num_devices)
        segment = Segment.head_aware(self.num_heads, self.head_dim)
        self.conv1d.sharding_strategy = ShardingStrategy.segmented(
            num_devices, axis=0, segments=(segment, segment, segment)
        )

        # Head-major along their output axis, so a row split is a head split.
        self.q_proj.sharding_strategy = head_rows
        self.k_proj.sharding_strategy = head_rows
        self.v_proj.sharding_strategy = head_rows
        self.f_b_proj.sharding_strategy = head_rows
        self.g_b_proj.sharding_strategy = head_rows
        self.b_proj.sharding_strategy = head_rows
        self.dt_bias.sharding_strategy = head_rows
        self.A_log.sharding_strategy = head_rows

        # `f_a_proj` and `g_a_proj` project down to `head_dim`, which is not a
        # head axis, and the gated norm reduces inside one head. Neither has
        # anything to split.
        replicate = ShardingStrategy.replicate(num_devices)
        self.f_a_proj.sharding_strategy = replicate
        self.g_a_proj.sharding_strategy = replicate
        self.o_norm.sharding_strategy = replicate

        # Row-parallel: each device holds the columns of its own heads and
        # emits a partial sum the caller all-reduces.
        self.o_proj.sharding_strategy = ShardingStrategy.head_aware_columnwise(
            num_devices, self.num_heads, self.head_dim
        )

    def shard(self, devices: Iterable[DeviceRef]) -> list[KimiDeltaAttention]:
        """Creates one per-device view of this layer, split by head.

        Args:
            devices: Devices to place the shards on.

        Returns:
            One :class:`KimiDeltaAttention` per device, dimensioned for its own
            head slice.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if self._sharding_strategy is None:
            raise ValueError(
                "KimiDeltaAttention cannot be sharded because no sharding "
                "strategy was provided."
            )
        devices = list(devices)
        num_devices = len(devices)
        q_proj = self.q_proj.shard(devices)
        k_proj = self.k_proj.shard(devices)
        v_proj = self.v_proj.shard(devices)
        conv1d = self.conv1d.shard(devices)
        f_a_proj = self.f_a_proj.shard(devices)
        f_b_proj = self.f_b_proj.shard(devices)
        dt_bias = self.dt_bias.shard(devices)
        a_log = self.A_log.shard(devices)
        b_proj = self.b_proj.shard(devices)
        g_a_proj = self.g_a_proj.shard(devices)
        g_b_proj = self.g_b_proj.shard(devices)
        o_norm = self.o_norm.shard(devices)
        o_proj = self.o_proj.shard(devices)

        shards: list[KimiDeltaAttention] = []
        for i, device in enumerate(devices):
            shard = KimiDeltaAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads // num_devices,
                head_dim=self.head_dim,
                conv_kernel_size=self.conv_kernel_size,
                dtype=self.dtype,
                device=device,
                rms_norm_eps=self.rms_norm_eps,
                lower_bound=self.lower_bound,
            )
            shard.q_proj = q_proj[i]
            shard.k_proj = k_proj[i]
            shard.v_proj = v_proj[i]
            shard.conv1d = conv1d[i]
            shard.f_a_proj = f_a_proj[i]
            shard.f_b_proj = f_b_proj[i]
            shard.dt_bias = dt_bias[i]
            shard.A_log = a_log[i]
            shard.b_proj = b_proj[i]
            shard.g_a_proj = g_a_proj[i]
            shard.g_b_proj = g_b_proj[i]
            shard.o_norm = o_norm[i]
            shard.o_proj = o_proj[i]
            shards.append(shard)
        return shards

    def __call__(
        self,
        x: TensorValue,
        conv_pool: BufferValue,
        recurrent_pool: BufferValue,
        slot_idx: TensorValue,
        input_row_offsets: TensorValue,
        replay_capture: list[KdaReplayInputs] | None = None,
    ) -> TensorValue:
        """Runs one KDA layer, mutating both state pools in place.

        Both pools are mutable graph inputs the kernels address at slot
        ``slot_idx[batch_item]``, so there is no state graph output and no
        Python-side gather or scatter.

        Args:
            x: ``[total_tokens, hidden_size]``, already normalized by the
                decoder layer.
            conv_pool: This layer's conv pool, mutated in place.
            recurrent_pool: This layer's recurrent pool, mutated in place.
            slot_idx: ``[batch_size]`` pool slot per sequence.
            input_row_offsets: ``[batch_size + 1]`` exclusive prefix offsets.
            replay_capture: When given, this call's :class:`KdaReplayInputs`
                are appended to it so a speculative rollback can re-run the
                state kernels over a shorter prefix.

        Returns:
            ``[total_tokens, hidden_size]``, a partial sum over the head axis
            when this layer is sharded.
        """
        device = x.device
        heads, head_dim = self.num_heads, self.head_dim
        head_shape = [-1, heads, head_dim]

        # Concatenated q, k, v in that order: the adapter's conv channel
        # layout, and the order the shard's conv weight was sliced for.
        qkv = ops.concat(
            [self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1
        ).cast(DType.float32)

        conv_weight = ops.reshape(
            self.conv1d.to(device).cast(DType.float32),
            [self.conv_dim, self.conv_kernel_size],
        )
        row_offsets_uint32 = input_row_offsets.cast(DType.uint32)
        conv_out = ops.silu(
            gated_delta_conv1d_fwd(
                qkv_input_ragged=qkv,
                conv_weight=conv_weight,
                conv_state=conv_pool,
                slot_idx=slot_idx.cast(DType.uint32),
                input_row_offsets=row_offsets_uint32,
            )
        )
        q, k, v = (
            ops.reshape(part, head_shape)
            for part in ops.split(conv_out, [self.qkv_dim] * 3, axis=-1)
        )

        # The kernel adds `dt_bias`, applies `exp(A_log)` and the bounded
        # sigmoid, and sigmoids `beta_logits`; all of that is folded in by
        # `gate_mode` and `beta_mode`, so what goes in is the raw
        # pre-activation. Unlike Gated DeltaNet's, it is per channel.
        raw_gate = ops.reshape(
            self.f_b_proj(self.f_a_proj(x)).cast(DType.float32), head_shape
        )
        beta_logits = self.b_proj(x).cast(DType.float32)

        if replay_capture is not None:
            replay_capture.append(
                KdaReplayInputs(
                    qkv=qkv,
                    conv_weight=conv_weight,
                    raw_gate=raw_gate,
                    beta_logits=beta_logits,
                )
            )

        a_log = self.A_log.to(device)
        dt_bias = ops.reshape(self.dt_bias.to(device), [heads, head_dim])
        state_indices = slot_idx.cast(DType.int32)
        # float32 out: the next stage is a float32 gated norm, so rounding to
        # the compute dtype in between would cost a round trip for nothing.
        core_out = kda_decode(
            q,
            k,
            v,
            raw_gate,
            beta_logits,
            a_log,
            dt_bias,
            input_row_offsets.cast(DType.int32),
            recurrent_pool,
            state_indices,
            output_dtype=DType.float32,
            gate_mode=self.gate_mode,
            beta_mode="logits",
            state_layout=STATE_LAYOUT,
        )

        # Normalize, scale by the learned weight, and only then gate. Gating
        # first is a different function of the same tensors.
        gate = ops.reshape(
            self.g_b_proj(self.g_a_proj(x)).cast(DType.float32), head_shape
        )
        normed = self.o_norm(core_out) * ops.sigmoid(gate)
        return self.o_proj(
            ops.reshape(normed.cast(self.dtype), [-1, self.qkv_dim])
        )


class Glm5NextKdaSublayer(Module):
    """The KDA sublayer as the decoder layer calls it, across all devices."""

    replay_capture: list[list[KdaReplayInputs]] | None = None
    """Per-device sink for this layer's state-kernel inputs, or ``None``.

    Set by the speculative graph, which re-runs those kernels over the accepted
    prefix after the verify pass. ``None`` in the base graph.
    """

    def __init__(self, config: Glm5NextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        devices = config.devices
        self.kda = KimiDeltaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.linear_num_heads,
            head_dim=config.linear_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            dtype=config.compute_dtype,
            device=devices[0],
            rms_norm_eps=config.rms_norm_eps,
            lower_bound=config.linear_lower_bound,
        )
        self.kda.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.shards = self.kda.shard(devices)
        self.allreduce = Allreduce(num_accelerators=len(devices))

    def __call__(
        self, xs: list[TensorValue], inputs: KdaSublayerInputs
    ) -> list[TensorValue]:
        """Runs the layer on every device and all-reduces the result.

        Args:
            xs: ``[total_tokens, hidden_size]`` per device, normalized.
            inputs: This layer's per-step pools and offsets.

        Returns:
            ``[total_tokens, hidden_size]`` per device.
        """
        return self.allreduce(
            [
                shard(
                    xs[i],
                    conv_pool=inputs.conv_pools[i],
                    recurrent_pool=inputs.recurrent_pools[i],
                    slot_idx=inputs.slot_idx[i],
                    input_row_offsets=inputs.input_row_offsets[i],
                    replay_capture=(
                        None
                        if self.replay_capture is None
                        else self.replay_capture[i]
                    ),
                )
                for i, shard in enumerate(self.shards)
            ],
            inputs.signal_buffers,
        )
