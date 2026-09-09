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
"""Op implementation for fused reduce-scatter + RMSNorm."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo
from max._core.dialects.builtin import BoolAttr, IntegerAttr, IntegerType
from max.dtype import DType

from ..graph import Graph
from ..type import DeviceRef, _ChainType
from ..value import BufferValueLike, TensorType, TensorValue, TensorValueLike
from .constant import constant
from .utils import _buffer_values, _tensor_values


def reduce_scatter_rms_norm(
    inputs: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
    gammas: Iterable[TensorValueLike],
    epsilon: float,
    residuals: Iterable[TensorValueLike] | None = None,
    weight_offset: float = 0.0,
    group_size: int | None = None,
) -> tuple[list[TensorValue], list[TensorValue]]:
    """Fused reduce-scatter sum + optional residual add + RMSNorm (bf16 in/out).

    Reduce-scatters ``inputs`` (one ``[rows, cols]`` tensor per device) along
    axis 0 across all devices, adds ``residuals`` when given, then
    RMSNorm-normalizes each device's owned row shard in the same collective
    launch, keeping the reduced sum in float32 registers so there is no
    global-memory round-trip between the reduce-scatter and the norm. The norm
    is ``multiply_before_cast=True`` (gamma folded in float32, single cast to
    the input dtype last).

    Rows are partitioned with the same ragged binning as
    :func:`reducescatter.sum` (remainder rows go to low ranks), so the sum
    output is a drop-in for ``reducescatter.sum`` along axis 0.

    Args:
        inputs: The input tensors to reduce and scatter, one per device.
        signal_buffers: Device buffer values used for synchronization.
        gammas: RMSNorm gamma weights, one per device (input dtype, length
            ``cols``).
        epsilon: RMSNorm epsilon for numerical stability.
        residuals: Optional residual stream, one per device, same shape as
            ``inputs``. **Must be replicated**: bit-identical on every device of
            a group. Each device adds only its own row shard, so do NOT pre-add
            the residual on the group leader as well -- the reduce-scatter sums
            across ranks, so a leader-side add already lands once for the whole
            group and this reproduces it without a separate full-width
            elementwise launch on that one device. Omit it to get a plain
            reduce-scatter + norm, byte-for-byte what this op did before the
            fold existed.
        weight_offset: Constant offset added to gamma at runtime (folded in
            float32). ``1.0`` for Gemma-style norms, ``0.0`` otherwise.
        group_size: Optional number of contiguous devices per independent
            reduce-scatter group. Defaults to all devices (a full-world
            collective). Under TP-within-DP this is the TP degree, so each
            replica reduces only within its own group.

    Returns:
        A tuple ``(normed, residual)`` of two lists, each with one tensor per
        device: ``normed[i]`` is the RMSNorm of device ``i``'s reduce-scatter
        shard and ``residual[i]`` is the reduce-scatter sum shard itself (the
        residual stream). Both have the input shape with axis 0 divided across
        the device's group.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    gammas = _tensor_values(gammas)
    has_residual = residuals is not None
    residuals = _tensor_values(residuals) if residuals is not None else []

    num_devices = len(inputs)
    if num_devices < 2:
        raise ValueError(
            "reduce_scatter_rms_norm requires at least two inputs (one per "
            f"device); the reduce-scatter is a no-op otherwise. Got: {num_devices}"
        )
    if len(signal_buffers) != num_devices:
        raise ValueError(
            f"expected number of inputs ({num_devices}) and number of signal "
            f"buffers ({len(signal_buffers)}) to match"
        )
    if len(gammas) != num_devices:
        raise ValueError(
            f"expected number of inputs ({num_devices}) and number of gammas "
            f"({len(gammas)}) to match"
        )
    if has_residual and len(residuals) != num_devices:
        raise ValueError(
            f"expected number of inputs ({num_devices}) and number of "
            f"residuals ({len(residuals)}) to match"
        )

    input_dtype = inputs[0].dtype
    if input_dtype != DType.bfloat16:
        raise ValueError(
            "reduce_scatter_rms_norm is bfloat16-only (the kernel and fuse "
            f"threshold assume it). Got: {input_dtype}"
        )
    if not all(t.dtype == input_dtype for t in inputs[1:]):
        raise ValueError(
            "reduce_scatter_rms_norm requires the same dtype across all input "
            f"tensors. Got: {inputs=}"
        )
    if not all(t.shape.rank == inputs[0].shape.rank for t in inputs[1:]):
        raise ValueError(
            "reduce_scatter_rms_norm requires the same rank across all input "
            f"tensors. Got: {inputs=}"
        )
    group_size = group_size or num_devices
    if group_size < 2:
        raise ValueError(
            "reduce_scatter_rms_norm requires group_size to be at least 2 (the "
            f"reduce-scatter is a no-op otherwise). Got: {group_size=}"
        )
    if num_devices % group_size != 0:
        raise ValueError(
            "reduce_scatter_rms_norm requires group_size to evenly divide the "
            f"number of input tensors. Got: {group_size=} and {num_devices=}"
        )
    # Shapes need only match within a group: DP replicas are independent
    # collectives and may carry different symbolic dims.
    for group_start in range(0, num_devices, group_size):
        group_inputs = inputs[group_start : group_start + group_size]
        if not all(t.shape == group_inputs[0].shape for t in group_inputs[1:]):
            raise ValueError(
                "reduce_scatter_rms_norm requires the same shape across all "
                f"input tensors in each group. Got: {inputs=}"
            )
    # The residual is indexed by GLOBAL row, so a shard-shaped one runs past its
    # own storage on every rank whose shard does not start at row 0.
    if has_residual:
        for dev_idx, (inp, res) in enumerate(
            zip(inputs, residuals, strict=True)
        ):
            if res.shape != inp.shape or res.dtype != inp.dtype:
                raise ValueError(
                    "reduce_scatter_rms_norm requires each residual to match "
                    f"its input's shape and dtype on device {dev_idx}. Got "
                    f"residual {res.shape}/{res.dtype} vs input "
                    f"{inp.shape}/{inp.dtype}"
                )
            if res.device != inp.device:
                raise ValueError(
                    "reduce_scatter_rms_norm requires each residual to live on "
                    f"its input's device. Got {res.device} vs {inp.device}"
                )

    devices = [t.device for t in inputs]
    if len(set(devices)) < num_devices:
        raise ValueError(
            "reduce_scatter_rms_norm requires unique devices across its input "
            f"tensors. Got: {devices=}"
        )

    graph = Graph.current

    # Per-device output types: axis-0 ragged binning across the device's own
    # group (matches reducescatter.sum so the residual is a drop-in). Normed and
    # residual share the shard shape.
    normed_types: list[TensorType] = []
    residual_types: list[TensorType] = []
    for dev_idx, device in enumerate(devices):
        shard_shape = list(inputs[dev_idx].shape)
        local_rank = dev_idx % group_size
        shard_shape[0] = (
            shard_shape[0] + (group_size - local_rank - 1)
        ) // group_size
        normed_types.append(
            TensorType(dtype=input_dtype, shape=shard_shape, device=device)
        )
        residual_types.append(
            TensorType(dtype=input_dtype, shape=shard_shape, device=device)
        )

    # epsilon/weight_offset as CPU scalars: the kernel reads them host-side for
    # launch params. One operand slot per device (SameVariadicOperandSize),
    # reusing the same host constant.
    cpu = DeviceRef.CPU()
    eps_const = constant(epsilon, DType.float32, cpu)
    weight_offset_const = constant(weight_offset, input_dtype, cpu)
    epsilons = [eps_const] * num_devices
    weight_offsets = [weight_offset_const] * num_devices

    in_chain = graph.device_chains.merge_for(devices)

    # The op's variadic groups must all match in size, so there is no empty
    # form: the slots carry the inputs and `has_residual` says not to read them.
    residual_operands = residuals if has_residual else inputs

    *results, out_chain = graph._add_op_generated(
        mo.CompositeDistributedReduceScatterRmsNormOp,
        normed_types,
        residual_types,
        _ChainType(),
        inputs,
        signal_buffers,
        gammas,
        epsilons,
        weight_offsets,
        residual_operands,
        in_chain,
        IntegerAttr(IntegerType(64), group_size),
        BoolAttr(has_residual),
    )

    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    normed = [res.tensor for res in results[:num_devices]]
    residual = [res.tensor for res in results[num_devices:]]
    return normed, residual
