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

"""Multi-layer Perceptron."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weight import Segment
from max.nn.quant_config import QuantConfig, ScaleGranularity, fp4_packed_k
from max.nn.quant_ops import quantized_matmul
from max.support.math import ceildiv

from .activation import activation_function_from_name
from .clamp import clamp
from .layer import Module, Shardable


class Linear(Module, Shardable):
    """Applies a linear transformation to incoming data: :math:`y = xW^T + b`.

    This layer implements a fully connected layer where inputs are multiplied
    by a weight matrix and optionally added with a bias vector. When called,
    ``Linear`` accepts a :class:`~max.graph.TensorValue` of shape ``(...,
    in_dim)`` and returns a :class:`~max.graph.TensorValue` of shape ``(...,
    out_dim)``.

    Both weights and bias initially reside on CPU and are moved to the
    specified device during model initialization.

    .. code-block:: python

        from max.driver import Accelerator, CPU, accelerator_count
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType
        from max.nn import Linear

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)

        linear_layer = Linear(
            in_dim=256,
            out_dim=128,
            dtype=DType.float32,
            device=device_ref,
            name="linear",
            has_bias=True
        )

        input_type = TensorType(DType.float32, [1, 256], device=device_ref)
        with Graph("linear", input_types=[input_type]) as graph:
            output = linear_layer(graph.inputs[0])
            graph.output(output)
    """

    weight: Weight
    """The weight matrix stored on CPU with shape (out_dim, in_dim).
    Model init transposes the weight and moves it to the target device."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_dim,).
    Model init moves the bias to the target device if present."""

    input_scale: Weight | None = None
    """The optional input scale stored on CPU with shape ().
    Model init moves the input_scale to the target device if present."""

    weight_scale: Weight | None = None
    """The optional weight scale stored on CPU with shape () or (N,).
    Model init moves the weight_scale to the target device if present."""

    weight_scale_2: Weight | None = None
    """The optional weight scale 2 used for fp4 quantization."""

    device: DeviceRef
    """The device where matrix operations are performed."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        quant_config: QuantConfig | None = None,
        name: str | None = None,
        clip_weight: float | None = None,
        is_sharding: bool = False,
    ) -> None:
        """Initializes the linear layer with weights and optional bias.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The :class:`~max.dtype.DType` for both weights and bias.
            device: The target :class:`~max.graph.DeviceRef` for computation.
                Weights remain on CPU until moved during computation.
            name: Base name for weights (appended with ``.weight`` and
                ``.bias`` if applicable).
            has_bias: When ``True``, adds a bias vector to the layer.
                Defaults to ``False``.
            quantization_encoding: :class:`~max.graph.quantization.QuantizationEncoding` for the weights.
            quant_config: :class:`~max.nn.quant_config.QuantConfig` for scaled quantization.
            clip_weight: Optional weight clipping threshold.
            is_sharding: Disable child layer creation during sharding.
        """
        super().__init__()

        self.device = device
        self.clip_weight = clip_weight
        self.quant_config = quant_config

        # Packed FP4 weights are stored as uint8 (two values per byte); int8
        # W8A8 weights are stored as int8 (RTN-quantized at load).
        weight_dtype = dtype
        if quant_config is not None:
            if quant_config.is_fp4:
                weight_dtype = DType.uint8
            elif quant_config.is_int8_w8a8:
                weight_dtype = DType.int8

        if not is_sharding:
            self.weight = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=weight_dtype,
                shape=(out_dim, fp4_packed_k(in_dim, quant_config)),
                device=device,
                quantization_encoding=quantization_encoding,
            )

        if has_bias:
            bias_dtype = dtype
            if quant_config and quant_config.bias_dtype:
                bias_dtype = quant_config.bias_dtype

            if not is_sharding:
                self.bias = Weight(
                    name=f"{name}.bias" if name else "bias",
                    dtype=bias_dtype,
                    shape=(out_dim,),
                    device=device,
                    quantization_encoding=quantization_encoding,
                )

                if self.bias.device != self.weight.device:
                    raise ValueError(
                        f"Bias is on device {self.bias.device} while weight is on {self.weight.device}."
                    )

        if quant_config and not is_sharding:
            if quant_config.is_static:
                self.input_scale = Weight(
                    name=f"{name}.input_scale" if name else "input_scale",
                    dtype=quant_config.input_scale.dtype,
                    shape=(),
                    device=DeviceRef.CPU(),
                    quantization_encoding=quantization_encoding,
                )

            if quant_config.input_scale.granularity not in (
                ScaleGranularity.TENSOR,
                ScaleGranularity.COLWISE,
                ScaleGranularity.BLOCK,
            ):
                raise ValueError(
                    f"unsupported input scale granularity {quant_config.input_scale.granularity}. "
                    "Only TENSOR, COLWISE and BLOCK granularities are supported, currently"
                )

            weight_scale = quant_config.weight_scale
            weight_scale_shape = self._infer_weight_scale_shape(quant_config)

            self.weight_scale = Weight(
                name=f"{name}.weight_scale" if name else "weight_scale",
                dtype=weight_scale.dtype,
                # TODO: Pass a per-layer quantization type.
                # For now since we only support row-wise
                shape=weight_scale_shape,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            )
            if quant_config.is_nvfp4:
                self.weight_scale_2 = Weight(
                    name=f"{name}.weight_scale_2" if name else "weight_scale_2",
                    dtype=quant_config.input_scale.dtype,
                    shape=(),
                    device=DeviceRef.CPU(),
                    quantization_encoding=quantization_encoding,
                )

    def _infer_weight_scale_shape(
        self, quant_config: QuantConfig
    ) -> tuple[int, ...]:
        weight_scale_shape: tuple[int, ...]
        weight_scale = quant_config.weight_scale
        if weight_scale.is_rowwise:
            weight_scale_shape = (int(self.weight.shape[0]), 1)
        elif weight_scale.is_tensor:
            weight_scale_shape = ()
        elif weight_scale.is_block:
            assert quant_config.weight_scale.block_size is not None
            k_dim = int(self.weight.shape[1])
            if quant_config.is_fp4:
                k_dim *= 2  # FP4 weights are packed 2x as uint8
            elif quant_config.is_mxfp6:
                # Four 6-bit codes per three bytes. Exact: K is a multiple of
                # 32, so the packed width is always a multiple of 3.
                k_dim = k_dim * 4 // 3
            weight_scale_shape = (
                ceildiv(
                    int(self.weight.shape[0]),
                    quant_config.weight_scale.block_size[0],
                ),
                ceildiv(
                    k_dim,
                    quant_config.weight_scale.block_size[1],
                ),
            )
        else:
            raise ValueError(
                "only row-wise and tensor scaling are "
                f"supported currently, but got {weight_scale.granularity}"
            )
        return weight_scale_shape

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the weight sharding strategy."""
        return self.weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the weight sharding strategy.

        Args:
            strategy: The :class:`~max.graph.ShardingStrategy` describing the weight sharding.
        """
        self.weight.sharding_strategy = strategy

        if self.weight_scale:
            # Weight scale should only be added when a quant config is passed.
            assert self.quant_config

            # Determine weight scale sharding strategy based on weight scale type
            # and weight sharding strategy.
            if self.quant_config.weight_scale.is_tensor:
                # Tensor scaling: always replicate
                self.weight_scale.sharding_strategy = (
                    ShardingStrategy.replicate(strategy.num_devices)
                )
            elif self.quant_config.weight_scale.is_rowwise:
                if strategy.is_colwise or strategy.is_head_aware_colwise:
                    # Rowwise scale + columnwise weight: replicate to avoid shape mismatch
                    self.weight_scale.sharding_strategy = (
                        ShardingStrategy.replicate(strategy.num_devices)
                    )
                else:
                    # Rowwise scale + rowwise weight: shard along same dimension
                    self.weight_scale.sharding_strategy = strategy
            elif self.quant_config.weight_scale.is_block:
                # Block scaling: blocks correspond to regions in the weight matrix.
                # For rowwise weight sharding, shard scale's first dim (N blocks).
                # For columnwise weight sharding, shard scale's second dim (K blocks).
                if strategy.is_rowwise:
                    if (
                        int(self.weight_scale.shape[0]) % strategy.num_devices
                        != 0
                    ):
                        raise ValueError(
                            f"Weight scale dim 0 ({self.weight_scale.shape[0]}) is "
                            f"not divisible by the number of devices ({strategy.num_devices}) for block-wise scaling."
                        )
                    self.weight_scale.sharding_strategy = strategy
                elif strategy.is_colwise or strategy.is_head_aware_colwise:
                    # For columnwise weight sharding, we need to shard the scale's
                    # second dimension (K blocks) to match the sharded input.
                    if (
                        int(self.weight_scale.shape[1]) % strategy.num_devices
                        != 0
                    ):
                        raise ValueError(
                            f"Weight scale dim 1 ({self.weight_scale.shape[1]}) is "
                            f"not divisible by the number of devices ({strategy.num_devices}) for block-wise scaling with columnwise weight sharding."
                        )

                    if strategy.is_head_aware_colwise:
                        # Extract num_heads and head_dim from the partial function
                        # and compute corresponding values for the scale tensor.
                        assert isinstance(strategy.shard, partial)
                        num_heads = strategy.shard.keywords["num_heads"]
                        head_dim = strategy.shard.keywords["head_dim"]
                        # block_size is guaranteed non-None when is_block is True
                        assert (
                            self.quant_config.weight_scale.block_size
                            is not None
                        )
                        block_size_k = (
                            self.quant_config.weight_scale.block_size[1]
                        )

                        # Check if head boundaries align with block boundaries
                        if head_dim % block_size_k == 0:
                            # Each head maps to (head_dim / block_size_k) blocks
                            scale_head_dim = head_dim // block_size_k
                            self.weight_scale.sharding_strategy = (
                                ShardingStrategy.head_aware_columnwise(
                                    strategy.num_devices,
                                    num_heads,
                                    scale_head_dim,
                                )
                            )
                        else:
                            # Head boundaries don't align with blocks, fall back
                            # to even columnwise sharding.
                            self.weight_scale.sharding_strategy = (
                                ShardingStrategy.columnwise(
                                    strategy.num_devices
                                )
                            )
                    else:
                        self.weight_scale.sharding_strategy = (
                            ShardingStrategy.columnwise(strategy.num_devices)
                        )
                elif strategy.is_segmented and strategy.sharded_axis == 1:
                    # Segmented sharding along K: weight_scale's K is reduced
                    # by block_size_k, so each segment's size shrinks by the
                    # same factor. Head-aware segments shrink their head_dim;
                    # even segments shrink their total size.
                    assert self.quant_config.weight_scale.block_size is not None
                    block_size_k = self.quant_config.weight_scale.block_size[1]
                    assert isinstance(strategy.shard, partial)
                    segments = strategy.shard.keywords["segments"]
                    scale_segments: list[Segment] = []
                    for seg in segments:
                        if seg.size % block_size_k != 0:
                            raise ValueError(
                                f"Segmented sharding: segment size {seg.size} "
                                f"is not divisible by block_size_k "
                                f"({block_size_k}) for block-wise scaling."
                            )
                        if seg.num_heads is not None:
                            head_dim = seg.size // seg.num_heads
                            if head_dim % block_size_k != 0:
                                raise ValueError(
                                    f"Segmented sharding: head_dim {head_dim} "
                                    f"is not divisible by block_size_k "
                                    f"({block_size_k}) for block-wise scaling."
                                )
                            scale_segments.append(
                                Segment.head_aware(
                                    seg.num_heads, head_dim // block_size_k
                                )
                            )
                        else:
                            scale_segments.append(
                                Segment.even(seg.size // block_size_k)
                            )
                    self.weight_scale.sharding_strategy = (
                        ShardingStrategy.segmented(
                            strategy.num_devices,
                            axis=1,
                            segments=scale_segments,
                        )
                    )
                else:
                    self.weight_scale.sharding_strategy = strategy
            else:
                # Colwise scaling (if supported in future)
                self.weight_scale.sharding_strategy = strategy

        if self.bias:
            # When the weight is sharded along axis 0 the output dim is split
            # per device, so the bias (1D, indexed by output dim) is sharded
            # by the same strategy. Otherwise the output dim is unchanged, so
            # replicate the bias. Linear.shard handles setting bias to None
            # on devices >= 1 to prevent bias duplication, which would be
            # incorrect.
            if strategy.sharded_axis == 0:
                self.bias.sharding_strategy = strategy
            else:
                self.bias.sharding_strategy = ShardingStrategy.replicate(
                    strategy.num_devices
                )

    def shard(self, devices: Iterable[DeviceRef]) -> list[Linear]:
        """Creates sharded views of this Linear layer across multiple devices.

        Args:
            devices: Iterable of :class:`~max.graph.DeviceRef` devices to place the shards on.

        Returns:
            List of sharded :class:`~max.nn.Linear` instances, one for each device.
        """
        if not self.weight.sharding_strategy:
            raise ValueError(
                "Linear layer cannot be sharded because no sharding strategy was provided."
            )

        # Calculate sharded dimensions. The placeholder Linear constructed
        # below has its weight overwritten with the true sharded weight, so
        # this only needs to be a reasonable approximation of the per-device
        # output dim — for uneven distributions the actual shape comes from
        # ``weight_shard``.
        strategy = self.weight.sharding_strategy
        if strategy.sharded_axis == 0:
            out_dim = int(self.weight.shape[0]) // strategy.num_devices
        else:
            out_dim = int(self.weight.shape[0])

        # Get sharded weights
        sharded_weights = self.weight.shard(devices)
        sharded_biases = []
        sharded_weight_scales = []

        if self.bias is not None:
            sharded_biases = self.bias.shard(devices)

        if (
            self.quant_config
            and self.weight_scale is not None
            and len(self.weight_scale.shape) > 0
        ):
            sharded_weight_scales = self.weight_scale.shard(devices)

        shards = []
        for shard_idx, (device, weight_shard) in enumerate(
            zip(devices, sharded_weights, strict=True)
        ):
            # Create new Linear with same configuration.
            sharded = Linear(
                in_dim=int(self.weight.shape[1]),
                out_dim=out_dim,
                dtype=self.weight.dtype,
                device=device,
                has_bias=self.bias is not None,
                quant_config=self.quant_config,
                clip_weight=self.clip_weight,
                is_sharding=True,
            )

            # Replace the weights with sharded versions.
            sharded.weight = weight_shard

            # Handle bias sharding
            if self.bias is not None:
                # When the K axis is sharded (axis=1 of [N, K]) the per-device
                # outputs are partial sums summed by allreduce, so the bias
                # must only be added once — on device 0 — to avoid being
                # multiplied by num_devices.
                k_sharded = (
                    self.weight.sharding_strategy.is_colwise
                    or self.weight.sharding_strategy.is_head_aware_colwise
                    or (
                        self.weight.sharding_strategy.is_segmented
                        and self.weight.sharding_strategy.sharded_axis == 1
                    )
                )
                if k_sharded and (shard_idx > 0):
                    sharded.bias = None
                else:
                    sharded.bias = sharded_biases[shard_idx]

            # Handle float8 scales.
            if self.quant_config:
                if self.input_scale is not None:
                    # Input scale is always shared (scalar), which should be
                    # checked upstream.
                    assert len(self.input_scale.shape) == 0
                    sharded.input_scale = self.input_scale

                if self.weight_scale is not None:
                    # Share a reference to the original weight scale if scalar, and
                    # shard if on device.
                    # This is because scalars are always on CPU by convention.
                    sharded.weight_scale = (
                        self.weight_scale
                        if len(self.weight_scale.shape) == 0
                        else sharded_weight_scales[shard_idx]
                    )
                if (
                    self.quant_config is not None
                    and self.quant_config.is_nvfp4
                    and hasattr(self, "weight_scale_2")
                ):
                    sharded.weight_scale_2 = self.weight_scale_2

            shards.append(sharded)

        return shards

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies a linear transformation to the input data.

        Args:
            x: Input :class:`~max.graph.TensorValue` of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensor must reside on the target device.

        Returns:
            Output :class:`~max.graph.TensorValue` of shape ``(..., out_dim)``.
            The result resides on the target device.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        weight: TensorValue = self.weight.to(x.device)
        if self.clip_weight:
            weight = clamp(weight, -self.clip_weight, self.clip_weight)

        # The int8 W8A8 (Apple M5) matmul fuses the bias into its dequant
        # epilogue; hand it in so the `.bias` op fires instead of a separate
        # add. Every other path still adds the bias below.
        fuse_bias = (
            self.bias is not None
            and self.quant_config is not None
            and self.quant_config.is_int8_w8a8
        )
        fused_bias = (
            self.bias.to(x.device)
            if fuse_bias and self.bias is not None
            else None
        )

        res = linear(
            x,
            weight,
            self.weight.quantization_encoding,
            self.quant_config,
            self.input_scale,
            self.weight_scale,
            self.weight_scale_2,
            bias=fused_bias,
        )

        if self.bias is not None and not fuse_bias:
            res += self.bias.to(res.device)
        return res


def linear(
    x: TensorValue,
    weight: TensorValue,
    quantization_encoding: QuantizationEncoding | None = None,
    quant_config: QuantConfig | None = None,
    input_scale: TensorValue | None = None,
    weight_scale: TensorValue | None = None,
    weight_scale_2: TensorValue | None = None,
    bias: TensorValue | None = None,
) -> TensorValue:
    """Computes x @ weight.T (+bias) with quantization support.

    ``bias`` is only consumed by the int8 W8A8 (Apple M5) path, which fuses it
    into the matmul dequant epilogue; for every other path it is ignored and
    the caller is responsible for adding it.
    """
    if quantization_encoding is not None:
        return ops.qmatmul(quantization_encoding, None, x, weight)
    elif quant_config:
        assert weight_scale is not None

        # The FP4 and static-scaled FP8 matmul kernels require rank-2
        # inputs. Flatten leading dims before the call and restore them
        # afterward. (LLM callers already pass rank-2 ragged activations;
        # this only engages for batched rank-3+ inputs such as the Wan DiT.)
        leading_dims: list[Dim] | None = None
        if x.rank > 2:
            leading_dims = list(x.shape[:-1])
            m_dim: Dim = Dim(1)
            for d in leading_dims:
                m_dim = m_dim * d
            x = ops.reshape(x, [m_dim, x.shape[-1]])

        res = quantized_matmul(
            x,
            weight,
            weight_scale,
            input_scale,
            quant_config,
            weight_scale_2,
            bias=bias,
        )

        if leading_dims is not None:
            res = ops.reshape(res, [*leading_dims, res.shape[-1]])

        return res
    else:
        return x @ weight.T


class ColumnParallelLinear(Linear):
    """A :class:`~max.nn.Linear` layer where the weight and bias are sharded onto multiple devices.

    This layer first computes :math:`y = xW_i^T + b_i` for each device `i` in
    `[0,..., num_devices]`:

    .. code-block::

        +-----+       +-----+ T     +-----+       +-----+
        |     |       | W_0 |       | b_0 |       | y_0 | GPU0
        |     |       +-----+       +-----+       +-----+
        |     |       | W_1 |       | b_1 |       | y_1 | GPU1
        |  x  |   @   +-----+   +   +-----+   =   +-----+
        |     |       | W_2 |       | b_2 |       | y_2 | GPU2
        |     |       +-----+       +-----+       +-----+
        |     |       | W_3 |       | b_3 |       | y_3 | GPU3
        +-----+       +-----+       +-----+       +-----+

    The values are then collected using an Allgather op, producing the same
    output tensor :math:`y = xW^T + b` on each device:

    .. code-block::

        GPU0  GPU1  GPU2  GPU3                      GPU0  GPU1  GPU2  GPU3
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+
        | y_0 |  -  |  -  |  -  |                   | y_0 | y_0 | y_0 | y_0 |
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+
        |  -  | y_1 |  -  |  -  |                   | y_1 | y_1 | y_1 | y_1 |
        +-----+-----+-----+-----+  -- Allgather --> +-----+-----+-----+-----+
        |  -  |  -  | y_2 |  -  |                   | y_2 | y_2 | y_2 | y_2 |
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+
        |  -  |  -  |  -  | y_3 |                   | y_3 | y_3 | y_3 | y_3 |
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+

    Example usage:

    .. code-block:: python

        from max.driver import Accelerator, CPU, accelerator_count
        from max.dtype import DType
        from max.graph import DeviceRef
        from max.nn import ColumnParallelLinear

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)

        distributed_linear = ColumnParallelLinear(
            256,
            128,
            DType.float32,
            devices=[device_ref],
        )
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        devices: Sequence[DeviceRef],
        tied_weight: Weight | None = None,
        **kwargs,
    ) -> None:
        """Initializes the column-parallel linear layer.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The :class:`~max.dtype.DType` for both weights and bias.
            devices: The target :class:`~max.graph.DeviceRef` devices for computation.
                Weights remain on CPU until sharded and moved to device during
                computation.
            tied_weight: Optional :class:`~max.graph.Weight` to tie with this layer.
            **kwargs: Additional keyword arguments passed to the Linear initializer.
        """
        if len(devices) == 0:
            raise ValueError(
                "ColumnParallelLinear requires a non-empty devices argument"
            )

        if tied_weight and (
            kwargs.get("quant_config") is not None
            or kwargs.get("has_bias", False)
        ):
            raise ValueError(
                "float8 and bias are both unsupported by "
                "ColumnParallelLinear currently"
            )

        super().__init__(in_dim, out_dim, dtype, devices[0], **kwargs)

        if tied_weight:
            # Overwrite the weight we just constructed with the tied weight.
            # In contrast with overriding outside the constructor, this ensures
            # that the sharding strategy captures the tied weight correctly.
            self.weight = tied_weight
            self.set_shared_weight("weight", tied_weight)

        self.devices = devices
        self.num_devices = len(self.devices)

        self.sharding_strategy = ShardingStrategy.rowwise(self.num_devices)

        self.distributed_linear_layers = super().shard(self.devices)

    def __call__(  # type: ignore[override]
        self, x: Sequence[TensorValue], signal_buffers: Iterable[BufferValue]
    ) -> list[TensorValue]:
        """Applies a linear transformation to the input data.

        Args:
            x: Input sequence of :class:`~max.graph.TensorValue` tensors of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensors must reside on their respective devices.
            signal_buffers: :class:`~max.graph.BufferValue` buffers for peer-to-peer communication in allgather.

        Returns:
            List of output :class:`~max.graph.TensorValue` tensors of shape ``(..., out_dim)``.
            The results reside on their respective devices.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        linear_outs = [
            self.distributed_linear_layers[i](x[i])
            for i in range(self.num_devices)
        ]
        return ops.allgather(linear_outs, signal_buffers, axis=-1)


@dataclass
class GPTQLinear(Linear):
    """A :class:`~max.nn.Linear` layer for GPTQ encoding."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        quantization_config: QuantizationConfig | None = None,
        quant_config: QuantConfig | None = None,
    ) -> None:
        """Initializes the linear layer with weights and optional bias with GPTQ quantization.

        Initializes the layer for GPTQ quantized linear transformations.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The :class:`~max.dtype.DType` for both weights and bias.
            device: The target :class:`~max.graph.DeviceRef` for computation.
                Weights remain on CPU until moved during computation.
            has_bias: When ``True``, adds a bias vector to the layer.
                Defaults to ``False``.
            quantization_encoding: The :class:`~max.graph.quantization.QuantizationEncoding` of the weights.
            quantization_config: Extra :class:`~max.graph.quantization.QuantizationConfig` for the weight quantization.
            quant_config: :class:`~max.nn.quant_config.QuantConfig` for scaled quantization (not supported).
        """
        del out_dim, dtype  # Unused.
        if has_bias:
            raise ValueError("has_bias=True is not supported in GPTQLinear.")
        if quant_config:
            raise ValueError(
                "Scaled quantization is not supported in GPTQLinear."
            )

        # Skip Linear initialization.
        Module.__init__(self)
        self.device = device
        self.qweight = Weight(
            name="qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            device=device,
            quantization_encoding=quantization_encoding,
        )
        self.scales = Weight(
            name="scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            device=device,
            quantization_encoding=quantization_encoding,
        )

        assert quantization_config, (
            "QuantizationConfig must be provided for GPTQLinear"
        )
        assert quantization_config.sym, "GPTQ with sym=False is not supported."

        self.quantization_config = quantization_config

        desc_act = self.quantization_config.desc_act
        self.perm_idx = None
        if desc_act:
            self.perm_idx = Weight(
                "perm_idx", DType.int32, [in_dim], device=device
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies the GPTQLinear transformation to the input.

        Args:
            x: Input tensor to transform.

        Returns:
            The transformed tensor after applying the GPTQ linear layer.
        """
        assert self.qweight.quantization_encoding is not None
        qweight_dtype, qweight_shape = self.qweight.original_dtype_and_shape
        qweight = ops.reshape(
            self.qweight,
            (qweight_shape[0] * qweight_dtype.size_in_bytes, qweight_shape[1]),
        ).transpose(0, 1)

        scales_dtype, scales_shape = self.scales.original_dtype_and_shape
        scales = ops.reshape(
            self.scales,
            (scales_shape[0] * scales_dtype.size_in_bytes, scales_shape[1]),
        ).transpose(0, 1)
        weight = ops.concat((qweight, scales), axis=1).transpose(0, 1)
        if self.device:
            weight = weight.to(self.device)
        if self.perm_idx is not None:
            perm_idx: TensorValue = self.perm_idx
            if self.device:
                perm_idx = perm_idx.to(self.device)
            res = ops.qmatmul(
                self.qweight.quantization_encoding,
                self.quantization_config,
                ops.gather(x, perm_idx, axis=(x.rank - 1)),
                weight,
                perm_idx,
            )
        else:
            res = ops.qmatmul(
                self.qweight.quantization_encoding,
                self.quantization_config,
                x,
                weight,
            )
        if self.bias is not None:
            res += TensorValue(self.bias)
        return res


class MLP(Module, Shardable):
    """Simple multi-layer perceptron composed of three :class:`~max.nn.Linear` layers.

    When called, ``MLP`` accepts a :class:`~max.graph.TensorValueLike` of shape
    ``(..., hidden_dim)`` and returns a :class:`~max.graph.TensorValue` of
    the same shape ``(..., hidden_dim)``.
    """

    def __init__(
        self,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        hidden_dim: int,
        feed_forward_length: int,
        devices: Sequence[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        has_bias: bool = False,
        activation_function: str = "silu",
        quant_config: QuantConfig | None = None,
        swiglu_limit: float = 0.0,
        is_sharding: bool = False,
    ) -> None:
        """Initializes the MLP layer.

        Args:
            dtype: :class:`~max.dtype.DType` to use for the layer weights, which should match the
                input dtype.
            quantization_encoding: :class:`~max.graph.quantization.QuantizationEncoding` of the layer weights.
            hidden_dim: The last dimension of the layer input.
            feed_forward_length: Size of dimension used to project the inputs.
            linear_cls: :class:`~max.nn.Linear` class to use to create the projection layers.
            devices: :class:`~max.graph.DeviceRef` devices to run the ``MLP`` layer.
            has_bias: Whether to include bias terms in the linear layers.
            activation_function: Activation function to use. Options are:

                - ``silu``
                - ``gelu``
                - ``gelu_tanh``
                - ``relu``
                - ``tanh``
                - ``sigmoid``

            quant_config: :class:`~max.nn.quant_config.QuantConfig` for scaled quantization.
            is_sharding: Disable child layer creation during sharding.
        """
        super().__init__()
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_dim = hidden_dim
        self.feed_forward_length = feed_forward_length

        if not is_sharding:
            self.gate_proj = linear_cls(  # [ffl, hidden]
                in_dim=hidden_dim,
                out_dim=feed_forward_length,
                dtype=dtype,
                device=devices[0],
                quantization_encoding=quantization_encoding,
                has_bias=has_bias,
                quant_config=quant_config,
            )
            self.down_proj = linear_cls(
                in_dim=feed_forward_length,
                out_dim=hidden_dim,
                dtype=dtype,
                device=devices[0],
                quantization_encoding=quantization_encoding,
                has_bias=has_bias,
                quant_config=quant_config,
            )
            self.up_proj = linear_cls(
                in_dim=hidden_dim,
                out_dim=feed_forward_length,
                dtype=dtype,
                device=devices[0],
                quantization_encoding=quantization_encoding,
                has_bias=has_bias,
                quant_config=quant_config,
            )

        self.quantization_encoding = quantization_encoding
        self.quant_config = quant_config
        self.swiglu_limit = swiglu_limit
        self._activation_function_name = activation_function
        self.activation_function = activation_function_from_name(
            activation_function
        )
        self._sharding_strategy: ShardingStrategy | None = None

    def _concat_or_max_gate_up_tensors(
        self,
        gate_tensor: TensorValue | None,
        up_tensor: TensorValue | None,
    ) -> TensorValue | None:
        """Concatenates the gate and up projection tensors for fused gate/up matmul."""
        if gate_tensor is None or up_tensor is None:
            return None

        # If the tensors are scalars, get the max of the two values.
        if len(gate_tensor.shape) == 0:
            assert len(up_tensor.shape) == 0
            return ops.max(
                ops.concat((gate_tensor.reshape((1,)), up_tensor.reshape((1,))))
            ).reshape([])

        if self.gate_proj.device:
            gate_tensor = gate_tensor.to(self.gate_proj.device)
        if self.up_proj.device:
            up_tensor = up_tensor.to(self.up_proj.device)

        return ops.concat((gate_tensor, up_tensor))

    def _concat_or_max_gate_up_weights(self) -> TensorValue:
        """Concatenates the gate and up projection weights."""
        result = self._concat_or_max_gate_up_tensors(
            self.gate_proj.weight, self.up_proj.weight
        )
        assert result is not None
        return result

    def _concat_or_max_gate_up_scales(self) -> TensorValue | None:
        """Builds the fused gate/up weight scale.

        For tensor-wise FP8, broadcasts each scalar to [ffl, 1] and
        concatenates into a [2*ffl, 1] rowwise scale so each row keeps
        its exact original scale with no accuracy loss.
        """
        gate_scale = self.gate_proj.weight_scale
        up_scale = self.up_proj.weight_scale
        if gate_scale is None or up_scale is None:
            return None

        if (
            self.quant_config is not None
            and self.quant_config.weight_scale.is_tensor
            and not self.quant_config.is_static
        ):
            ffl = self.gate_proj.weight.shape[0]
            gate_row_scale = ops.broadcast_to(
                gate_scale.reshape([1, 1]), [ffl, 1]
            )
            up_row_scale = ops.broadcast_to(up_scale.reshape([1, 1]), [ffl, 1])
            return ops.concat((gate_row_scale, up_row_scale))

        return self._concat_or_max_gate_up_tensors(gate_scale, up_scale)

    def _concat_or_max_gate_up_bias(self) -> TensorValue | None:
        """Concatenates the gate and up projection biases."""
        return self._concat_or_max_gate_up_tensors(
            self.gate_proj.bias, self.up_proj.bias
        )

    def _concat_or_max_gate_up_input_scale(self) -> TensorValue | None:
        """Gets the max input scale of the gate and up projection."""
        return self._concat_or_max_gate_up_tensors(
            self.gate_proj.input_scale, self.up_proj.input_scale
        )

    def _concat_or_max_gate_up_weight_scale_2(self) -> TensorValue | None:
        """Gets the max weight scale 2 of the gate and up projection."""
        return self._concat_or_max_gate_up_tensors(
            self.gate_proj.weight_scale_2, self.up_proj.weight_scale_2
        )

    def _can_used_fused_mlp(self) -> bool:
        """Checks if the gate/up matmuls can be fused."""
        if self.quantization_encoding:
            return False
        if self.quant_config is None:
            return True
        return self.quant_config.can_use_fused_mlp

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """Applies the MLP transformation to the input.

        Args:
            x: Input tensor to transform.

        Returns:
            The transformed tensor after applying the MLP layers.
        """
        if not self._can_used_fused_mlp():
            gate_out = self.activation_function(self.gate_proj(TensorValue(x)))
            up_out = self.up_proj(TensorValue(x))
        else:
            # Optimization to compute a single matmul by merging the
            # gate and up projection weights.
            output = linear(
                TensorValue(x),
                self._concat_or_max_gate_up_weights(),
                self.quantization_encoding,
                self.quant_config,
                input_scale=self._concat_or_max_gate_up_input_scale(),
                weight_scale=self._concat_or_max_gate_up_scales(),
                weight_scale_2=self._concat_or_max_gate_up_weight_scale_2(),
            )

            bias = self._concat_or_max_gate_up_bias()
            if bias is not None:
                output += bias

            feed_forward_length = self.gate_proj.weight.shape[0]

            gate_out, up_out = ops.split(
                output, [feed_forward_length, feed_forward_length], axis=1
            )

            gate_out = self.activation_function(gate_out)

        if self.swiglu_limit > 0:
            lim = ops.constant(
                self.swiglu_limit, gate_out.dtype, device=gate_out.device
            )
            neg_lim = ops.constant(
                -self.swiglu_limit, up_out.dtype, device=up_out.device
            )
            gate_out = ops.min(gate_out, lim)
            up_out = ops.min(ops.max(up_out, neg_lim), lim)

        return self.down_proj(gate_out * up_out)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the MLP sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the MLP layers.

        Args:
            strategy: The sharding strategy to apply.
        """
        self._sharding_strategy = strategy

        if strategy.is_replicate:
            # For replicate strategy, both layers use the same strategy
            self.gate_proj.sharding_strategy = strategy
            self.down_proj.sharding_strategy = strategy
            self.up_proj.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            self.gate_proj.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.down_proj.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
            self.up_proj.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
        else:
            raise ValueError(f"Unsupported sharding strategy: {strategy}")

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[MLP]:
        """Creates sharded views of this MLP across multiple devices.

        The return type is the covariant ``Sequence`` so subclasses (for
        example non-gated experts) can override with their own shard type
        while still satisfying ``Shardable``'s ``Sequence[Self]`` contract.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            Sharded MLP instances, one for each device.
        """
        if self.sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        # Get sharded layers
        sharded_gate_projs = self.gate_proj.shard(devices)
        sharded_down_projs = self.down_proj.shard(devices)
        sharded_up_projs = self.up_proj.shard(devices)

        shards = []
        for device, gate_proj, down_proj, up_proj in zip(
            devices,
            sharded_gate_projs,
            sharded_down_projs,
            sharded_up_projs,
            strict=True,
        ):
            # Create new MLP instance with the sharded layers
            sharded = MLP(
                dtype=self.gate_proj.weight.dtype,
                quantization_encoding=self.quantization_encoding,
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.feed_forward_length,
                devices=[device],
                has_bias=self.gate_proj.bias is not None,
                activation_function=self._activation_function_name,
                quant_config=self.quant_config,
                swiglu_limit=self.swiglu_limit,
                is_sharding=True,
            )

            # Assign the sharded linear layers
            sharded.gate_proj = gate_proj
            sharded.down_proj = down_proj
            sharded.up_proj = up_proj

            # Store parent layer to access the original weights to check
            # if the weights can be stacked.
            sharded._parent_layer = self

            shards.append(sharded)

        return shards


class FusedMLP(Module, Shardable):
    """Stores the gate and up projections as one pre-fused weight.

    The checkpoint provides ``gate_up_proj_fused`` with shape
    ``[2 * feed_forward_length, hidden_dim]``, with gate rows followed by up
    rows. Only bias-free, unquantized projections are supported.
    """

    def __init__(
        self,
        dtype: DType,
        hidden_dim: int,
        feed_forward_length: int,
        devices: Sequence[DeviceRef],
        activation_function: str = "silu",
        swiglu_limit: float = 0.0,
        is_sharding: bool = False,
    ) -> None:
        """Initializes the fused MLP layer.

        Args:
            dtype: :class:`~max.dtype.DType` for the layer weights.
            hidden_dim: The last dimension of the layer input.
            feed_forward_length: Size of the intermediate projection.
            devices: Devices to place the weights on. Single-device only.
            activation_function: Activation applied to the gate output.
            swiglu_limit: Optional SwiGLU clamp limit (0 disables clamping).
            is_sharding: Disable weight creation during sharding.
        """
        super().__init__()
        if len(devices) != 1:
            raise ValueError(
                f"FusedMLP requires exactly one device, got {len(devices)}"
            )
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_dim = hidden_dim
        self.feed_forward_length = feed_forward_length
        self.swiglu_limit = swiglu_limit
        self._activation_function_name = activation_function
        self.activation_function = activation_function_from_name(
            activation_function
        )
        self._sharding_strategy: ShardingStrategy | None = None

        if not is_sharding:
            # Byte-layout contract with the engine / weight adapter (row-major):
            # [gate rows; up rows] -> [2 * feed_forward_length, hidden_dim].
            self.gate_up_proj_fused = Weight(
                "gate_up_proj_fused",
                dtype,
                [2 * feed_forward_length, hidden_dim],
                devices[0],
            )
            self.down_proj = Linear(
                in_dim=feed_forward_length,
                out_dim=hidden_dim,
                dtype=dtype,
                device=devices[0],
            )

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """Applies the fused MLP transformation to the input."""
        output = linear(TensorValue(x), self.gate_up_proj_fused)
        gate_out, up_out = ops.split(
            output,
            [self.feed_forward_length, self.feed_forward_length],
            axis=-1,
        )
        gate_out = self.activation_function(gate_out)

        if self.swiglu_limit > 0:
            lim = ops.constant(
                self.swiglu_limit, gate_out.dtype, device=gate_out.device
            )
            neg_lim = ops.constant(
                -self.swiglu_limit, up_out.dtype, device=up_out.device
            )
            gate_out = ops.min(gate_out, lim)
            up_out = ops.min(ops.max(up_out, neg_lim), lim)

        return self.down_proj(gate_out * up_out)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the fused MLP sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the sharding strategy.

        Replication across devices and single-device tensor parallelism are
        supported. Tensor parallelism across multiple devices is not.
        """
        if strategy.is_tensor_parallel and strategy.num_devices != 1:
            raise ValueError(
                "FusedMLP does not support tensor parallelism across "
                f"{strategy.num_devices} devices"
            )
        self._sharding_strategy = strategy
        if strategy.is_replicate:
            self.gate_up_proj_fused.sharding_strategy = strategy
            self.down_proj.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            self.gate_up_proj_fused.sharding_strategy = (
                ShardingStrategy.rowwise(strategy.num_devices)
            )
            self.down_proj.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
        else:
            raise ValueError(f"Unsupported sharding strategy: {strategy}")

    def shard(self, devices: Iterable[DeviceRef]) -> list[FusedMLP]:
        """Creates sharded views of this fused MLP (single-device only).

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded ``FusedMLP`` instances, one per device.
        """
        if self.sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        sharded_gate_up = self.gate_up_proj_fused.shard(devices)
        sharded_down_projs = self.down_proj.shard(devices)

        shards = []
        for device, gate_up, down_proj in zip(
            devices, sharded_gate_up, sharded_down_projs, strict=True
        ):
            sharded = FusedMLP(
                dtype=self.gate_up_proj_fused.dtype,
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.feed_forward_length,
                devices=[device],
                activation_function=self._activation_function_name,
                swiglu_limit=self.swiglu_limit,
                is_sharding=True,
            )
            sharded.gate_up_proj_fused = gate_up
            sharded.down_proj = down_proj
            shards.append(sharded)

        return shards
