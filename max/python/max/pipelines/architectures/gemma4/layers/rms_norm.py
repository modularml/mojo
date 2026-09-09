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
from __future__ import annotations

from collections.abc import Iterable, Sequence

from max._core.dialects import builtin, kgen, mo
from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Weight,
    ops,
)
from max.graph.graph import Graph
from max.nn.layer import Module, Shardable


class Gemma4RMSNorm(Module, Shardable):
    """RMSNorm for Gemma4 with an optional learned weight.

    When ``with_weight`` is True (default), owns a ``weight`` parameter
    directly and computes ``weight * rms_norm(x)``.

    When ``with_weight`` is False, applies bare normalization with no
    learned weight: ``rms_norm(x)``.
    """

    def __init__(
        self,
        dim: int,
        dtype: DType,
        eps: float = 1e-6,
        with_weight: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.eps = eps
        self.with_weight = with_weight
        self._sharding_strategy: ShardingStrategy | None = None

        if with_weight:
            self.weight = Weight("weight", dtype, [dim], device=DeviceRef.CPU())

    def __call__(self, x: TensorValue) -> TensorValue:
        if self.with_weight:
            w = self.weight.cast(x.dtype).to(x.device)
        else:
            # Gemma-style offset still applies: (0 + 1) * norm == norm.
            w = ops.broadcast_to(
                ops.constant(
                    1.0,
                    dtype=x.dtype,
                    device=DeviceRef.CPU(),
                ),
                (self.dim,),
            ).to(x.device)
        return Graph.current._add_op_generated(
            mo.ReduceRmsNormOp,
            result=TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
            input=x,
            weight=w,
            # epsilon is a float32 host-scalar operand of the generated
            # `mo.reduce.rms_norm` op: the kernel (and `ops.rms_norm`) carry it
            # in float32. Constructing it at `x.dtype` (e.g. bf16) makes KGEN
            # reject the operand against the Float32 `ReduceRMSNorm.execute`
            # signature at graph-compile time.
            epsilon=ops.constant(
                self.eps, dtype=DType.float32, device=DeviceRef.CPU()
            ),
            weight_offset=ops.constant(
                0.0,
                dtype=x.dtype,
                device=DeviceRef.CPU(),
            ),
            multiply_before_cast=builtin.BoolAttr(True),
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if self.with_weight:
            self.weight.sharding_strategy = strategy
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[Gemma4RMSNorm]:
        """Creates sharded views of this Gemma4RMSNorm across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Gemma4RMSNorm instances, one for each device.
        """
        if self.sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        if not self.with_weight:
            return [
                Gemma4RMSNorm(
                    dim=self.dim,
                    dtype=self.dtype,
                    eps=self.eps,
                    with_weight=False,
                )
                for _ in devices
            ]

        weight_shards = self.weight.shard(devices)
        shards = []
        for weight_shard in weight_shards:
            sharded = Gemma4RMSNorm(
                dim=self.dim,
                dtype=self.dtype,
                eps=self.eps,
                with_weight=True,
            )
            sharded.weight = weight_shard
            shards.append(sharded)

        return shards
