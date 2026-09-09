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
"""Inkling short convolution: depthwise causal conv1d with a residual.

Semantics ported from the vLLM reference's ``fused_sconv(activation=None,
use_residual=True)``: depthwise, causal, width 4, no bias, ``x + conv(x)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.layer import Module, Shardable
from max.nn.state_space import causal_conv1d_varlen_fwd

_COMPUTE_DTYPE = DType.float32


class ShortConvolution(Module, Shardable):
    """Depthwise causal conv1d with a residual, stateful across decode
    steps via a caller-owned conv state pool."""

    def __init__(
        self,
        *,
        channels: int,
        kernel_size: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self._sharding_strategy: ShardingStrategy | None = None
        self.weight = Weight(
            "weight", dtype, [channels, 1, kernel_size], device=device
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the convolution channels evenly across devices."""
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "ShortConvolution only supports the tensor parallel sharding "
                "strategy."
            )
        if self.channels % strategy.num_devices:
            raise ValueError(
                f"{self.channels} convolution channels do not divide over "
                f"{strategy.num_devices} devices"
            )
        self.weight.sharding_strategy = ShardingStrategy.rowwise(
            strategy.num_devices
        )
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[ShortConvolution]:
        """Creates one per-device view of this convolution."""
        if self._sharding_strategy is None:
            raise ValueError(
                "ShortConvolution cannot be sharded: no sharding strategy."
            )
        devices = list(devices)
        shards = []
        for device, weight in zip(
            devices, self.weight.shard(devices), strict=True
        ):
            sharded = ShortConvolution(
                channels=self.channels // len(devices),
                kernel_size=self.kernel_size,
                dtype=self.dtype,
                device=device,
            )
            sharded.weight = weight
            shards.append(sharded)
        return shards

    def __call__(
        self,
        x: TensorValue,
        conv_state_pool: BufferValue,
        slot_idx: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Returns ``x + conv(x)``; updates ``conv_state_pool`` in place."""
        device = x.device
        channels, _, kernel_size = self.weight.shape
        x_f32 = ops.cast(x, _COMPUTE_DTYPE)

        conv = causal_conv1d_varlen_fwd(
            x_f32,
            ops.cast(
                self.weight.reshape([channels, kernel_size]), _COMPUTE_DTYPE
            ),
            # No bias tensor at any site; the kernel wants one anyway.
            ops.broadcast_to(
                ops.constant(0.0, _COMPUTE_DTYPE, device=device), [channels]
            ),
            conv_state_pool,
            ops.cast(input_row_offsets, DType.int32),
            ops.cast(slot_idx, DType.int32),
            # A freshly claimed slot is zeroed, so reading the pool window is
            # right at a sequence start too.
            ops.broadcast_to(
                ops.constant(True, DType.bool, device=device),
                [slot_idx.shape[0]],
            ),
            activation="none",
            channels_last=True,
        )

        return ops.cast(conv + x_f32, x.dtype)
