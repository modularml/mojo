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

"""Quantized tensor wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeGuard

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental.nn import Module
from max.experimental.nn.module import module_dataclass
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Sharded,
)
from max.experimental.tensor import Tensor
from typing_extensions import Self


def _ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


class QTensor(Module[[], None]):
    """Base class for quantized tensor wrappers.

    QTensors are :class:`~max.experimental.nn.Module` subclasses so that
    parameter discovery finds the inner tensors. They are data wrappers, not
    callable layers — pass them to a quantized kernel op
    (e.g. ``quant_ops.matmul``).
    """

    def forward(self) -> None:
        raise NotImplementedError("QTensor is not a callable layer")

    @property
    def local_shards(self) -> tuple[Self, ...]:
        raise NotImplementedError("QTensor does not support local_shards")

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        raise NotImplementedError("to() is not implemented for QTensor")


class FP8BlockTensor(QTensor):
    """FP8 block-scaled quantized tensor.
    Args:
        data: Packed ``float8_e4m3fn`` data of shape ``[rows, cols]``.
        weight_scale_inv: Per-block inverse ``float32`` scales of shape
            ``[ceil(rows / block_m), ceil(cols / block_k)]``.
        block_size: Per-block size as ``(block_m, block_k)``.
    """

    data: Tensor
    weight_scale_inv: Tensor

    def __init__(
        self,
        *,
        data: Tensor,
        weight_scale_inv: Tensor,
        block_size: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.data = data
        self.weight_scale_inv = weight_scale_inv
        self._block_size = block_size

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        *,
        block_size: tuple[int, int] = (128, 128),
    ) -> FP8BlockTensor:
        """Builds a :class:`FP8BlockTensor` of the given shape, with all zeros."""
        rows, cols = int(shape[0]), int(shape[1])
        block_m, block_k = block_size
        return cls(
            data=Tensor.zeros((rows, cols), dtype=DType.float8_e4m3fn),
            weight_scale_inv=Tensor.zeros(
                (_ceildiv(rows, block_m), _ceildiv(cols, block_k)),
                dtype=DType.float32,
            ),
            block_size=block_size,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        return self._block_size

    def shard(self, axis: int, mesh: DeviceMesh) -> FP8BlockTensor:
        """Co-shard both leaves along ``axis`` onto ``mesh``.

        Args:
            axis: Tensor axis to shard along (same index into both
                ``data.shape`` and ``weight_scale_inv.shape``).
            mesh: 1-D :class:`DeviceMesh` to scatter onto.

        Returns:
            A new :class:`FP8BlockTensor` whose ``data`` and ``weight_scale_inv``
            leaves are each distributed ``Sharded(axis=axis)`` across
            ``mesh``.
        """
        mapping = PlacementMapping(mesh, (Sharded(axis=axis),))
        return FP8BlockTensor(
            data=self.data.to(mapping),
            weight_scale_inv=self.weight_scale_inv.to(mapping),
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[FP8BlockTensor, ...]:
        return tuple(
            FP8BlockTensor(
                data=self.data.local_shards[i],
                weight_scale_inv=self.weight_scale_inv.local_shards[i],
                block_size=self.block_size,
            )
            for i in range(self.data.num_shards)
        )

    @property
    def mesh(self) -> DeviceMesh:
        return self.data.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.data.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.data._mapping = mapping
        self.weight_scale_inv._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> FP8BlockTensor:
        return FP8BlockTensor(
            data=self.data.to(target),
            weight_scale_inv=self.weight_scale_inv.to(target),
            block_size=self.block_size,
        )


class NVFP4Tensor(QTensor):
    """NVFP4 (modelopt) block-scaled quantized tensor.

    NVFP4 packs two ``float4_e2m1fn`` values per ``uint8`` and applies two
    scale levels: a per-16-element block scale (``float8_e4m3fn``) and a
    per-tensor global scale (``weight_scale_2``, ``float32``). The activation
    scale (``input_scale``, ``float32``) is *static* — loaded from the
    checkpoint — and used to normalize activations before dynamic block
    quantization (see :func:`~.quant_ops._matmul_nvfp4`).

    Args:
        data: Packed ``uint8`` weight data of shape ``[rows, cols // 2]``.
        weight_scale: Per-block ``float8_e4m3fn`` scales of shape
            ``[ceil(rows / block_m), ceil(cols / block_k)]``.
        weight_scale_2: Per-tensor ``float32`` global scale, scalar ``()``
            (or ``[num_experts]`` when stacked for grouped MoE).
        input_scale: Static per-tensor ``float32`` activation scale, scalar
            ``()`` (or ``[num_experts]`` when stacked).
        block_size: Per-block size as ``(block_m, block_k)``; ``(1, 16)`` for
            NVFP4.
    """

    data: Tensor
    weight_scale: Tensor
    weight_scale_2: Tensor
    input_scale: Tensor

    def __init__(
        self,
        *,
        data: Tensor,
        weight_scale: Tensor,
        weight_scale_2: Tensor,
        input_scale: Tensor,
        block_size: tuple[int, int] = (1, 16),
    ) -> None:
        super().__init__()
        self.data = data
        self.weight_scale = weight_scale
        self.weight_scale_2 = weight_scale_2
        self.input_scale = input_scale
        self._block_size = block_size

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        *,
        block_size: tuple[int, int] = (1, 16),
    ) -> NVFP4Tensor:
        """Builds a zero-filled :class:`NVFP4Tensor` of logical ``shape``.

        ``shape`` is the *unpacked* Linear-convention ``[out_dim, in_dim]``.
        The packed ``data`` leaf has half the trailing dimension.
        """
        rows, cols = int(shape[0]), int(shape[1])
        block_m, block_k = block_size
        return cls(
            data=Tensor.zeros((rows, cols // 2), dtype=DType.uint8),
            weight_scale=Tensor.zeros(
                (_ceildiv(rows, block_m), _ceildiv(cols, block_k)),
                dtype=DType.float8_e4m3fn,
            ),
            weight_scale_2=Tensor.zeros((), dtype=DType.float32, device=CPU()),
            input_scale=Tensor.zeros((), dtype=DType.float32, device=CPU()),
            block_size=block_size,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        return self._block_size

    def shard(self, axis: int, mesh: DeviceMesh) -> NVFP4Tensor:
        """Co-shard ``data``/``weight_scale`` along ``axis`` onto ``mesh``."""
        mapping = PlacementMapping(mesh, (Sharded(axis=axis),))
        return NVFP4Tensor(
            data=self.data.to(mapping),
            weight_scale=self.weight_scale.to(mapping),
            weight_scale_2=self.weight_scale_2,
            input_scale=self.input_scale,
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[NVFP4Tensor, ...]:
        n = self.data.num_shards
        return tuple(
            NVFP4Tensor(
                data=self.data.local_shards[i],
                weight_scale=self.weight_scale.local_shards[i],
                weight_scale_2=self.weight_scale_2,
                input_scale=self.input_scale,
                block_size=self.block_size,
            )
            for i in range(n)
        )

    @property
    def mesh(self) -> DeviceMesh:
        return self.data.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.data.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.data._mapping = mapping
        self.weight_scale._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> NVFP4Tensor:
        return NVFP4Tensor(
            data=self.data.to(target),
            weight_scale=self.weight_scale.to(target),
            # weight_scale_2 and input_scale stay on CPU
            weight_scale_2=self.weight_scale_2,
            input_scale=self.input_scale,
            block_size=self.block_size,
        )


@module_dataclass(frozen=True)
class NVFP4Activation(QTensor):
    """NVFP4-quantized activation tensor.

    Args:
        data: Packed ``uint8`` FP4 tokens of shape ``[total_tokens, K / 2]``.
        scales: Block scales in the padded SF-atom layout.
        input_scale: The ``float32`` scale these tokens were quantized with.
        scales_offset: Per-expert offset into the padded ``scales``.
    """

    data: Tensor
    scales: Tensor
    input_scale: Tensor
    scales_offset: Tensor


QuantAwareTensor: TypeAlias = (
    Tensor | FP8BlockTensor | NVFP4Tensor | NVFP4Activation
)


def all_fp8_block(
    weights: Sequence[QuantAwareTensor],
) -> TypeGuard[Sequence[FP8BlockTensor]]:
    """Narrow a sequence of mixed weights to a homogeneous FP8 sequence."""
    return all(isinstance(w, FP8BlockTensor) for w in weights)


def all_nvfp4(
    weights: Sequence[QuantAwareTensor],
) -> TypeGuard[Sequence[NVFP4Tensor]]:
    """Narrow a sequence of mixed weights to a homogeneous NVFP4 sequence."""
    return all(isinstance(w, NVFP4Tensor) for w in weights)
