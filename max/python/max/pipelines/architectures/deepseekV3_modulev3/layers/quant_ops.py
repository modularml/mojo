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

"""Quantization-aware kernel dispatch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.functional_kernels import (
    fused_silu,
    fused_silu_quantized,
    grouped_matmul_ragged,
)
from max.experimental.sharding import DeviceMesh, PlacementMapping
from max.experimental.sharding.action import Action, ActionSet, AxisAssignment
from max.experimental.sharding.cost import (
    P,
    R,
    build_action_set,
)
from max.experimental.sharding.placements import Placement, Sharded
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn.comm.ep import EPConfig
from max.nn.kernels import (
    block_scales_interleave as _block_scales_interleave,
)
from max.nn.kernels import (
    dynamic_block_scaled_matmul as _dynamic_block_scaled_matmul,
)
from max.nn.kernels import (
    dynamic_scaled_matmul as _dynamic_scaled_matmul,
)
from max.nn.kernels import (
    grouped_dynamic_scaled_fp8_matmul as _grouped_dynamic_scaled_fp8_matmul,
)
from max.nn.kernels import (
    grouped_matmul_block_scaled as _grouped_matmul_block_scaled,
)
from max.nn.kernels import (
    grouped_matmul_blocked_swiglu as _grouped_matmul_blocked_swiglu,
)
from max.nn.kernels import (
    grouped_quantize_dynamic_block_scaled as _grouped_quantize_dynamic_block_scaled,
)
from max.nn.kernels import (
    quantize_dynamic_block_scaled as _quantize_dynamic_block_scaled,
)
from max.nn.kernels import (
    quantize_dynamic_scaled_float8 as _quantize_dynamic_scaled_float8,
)
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from .quant_tensor import (
    FP8BlockTensor,
    NVFP4Activation,
    NVFP4Tensor,
    QuantAwareTensor,
    all_fp8_block,
    all_nvfp4,
)

# NVFP4 packs two float4-e2m1 values per byte and uses one float8_e4m3fn scale
# per 16-element block along K.
_NVFP4_SF_VECTOR_SIZE = 16


def _transpose2d_placement(p: Placement) -> Placement:
    """Map a rank-2 tensor placement to its transpose (swap axes 0 and 1).

    The activation block scales produced by
    :func:`quantize_dynamic_scaled_float8` are laid out transposed relative to
    the activations they describe (``[K / block_k, M]`` vs ``[M, K]``), so a
    shard of the activations on tensor axis ``a`` corresponds to a shard of the
    scales on axis ``1 - a``. :class:`~max.experimental.sharding.Replicated`
    and :class:`~max.experimental.sharding.Partial` placements are unchanged.
    """
    if isinstance(p, Sharded):
        return Sharded(axis=1 - p.axis)
    return p


def _quantize_finalize(action: Action) -> Action:
    """Expand the picked output into ``(data, scales)`` mappings.

    The picker derives one output placement, which we use for the FP8 data
    output (same layout as the input). This restores the second, transposed
    mapping for the block-scale output (see :func:`_transpose2d_placement`).
    """
    (data_mapping,) = action.outputs
    scales_mapping = PlacementMapping(
        data_mapping.mesh,
        tuple(_transpose2d_placement(p) for p in data_mapping.placements),
    )
    return Action(inputs=action.inputs, outputs=(data_mapping, scales_mapping))


def _quantize_rule(x: TensorLayout, *extras: Any) -> ActionSet:
    """Sharding rule for dynamic FP8 activation quantization.

    The FP8 data output follows the input placement; the block-scale output is
    its transpose (handled by :func:`_quantize_finalize`). Covers a replicated
    input (the common case), a contraction-sharded input (``o_proj`` under
    tensor parallelism), and a row-sharded input (sequence / data parallel).
    """
    rows = [
        AxisAssignment((R,), R),
        AxisAssignment((Sharded(0),), Sharded(0)),
        AxisAssignment((Sharded(1),), Sharded(1)),
    ]
    return build_action_set(rows, layouts=(x,), finalize=_quantize_finalize)


def _scaled_matmul_rule(
    a: TensorLayout,
    b: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    *extras: Any,
) -> ActionSet:
    """Sharding rule for the block-scaled FP8 matmul ``a @ b.T``.

    ``a`` is ``[M, K]`` and ``b`` (Linear-convention weight) is ``[N, K]``, so
    the output is ``[M, N]``. The activation scales ``a_scales`` are
    ``[K / block_k, M]`` (transposed) while the weight scales ``b_scales`` are
    ``[N / block_m, K / block_k]``. The rows mirror the bf16 matmul strategies:
    column-parallel weights (shard ``N``), row-parallel weights (shard the
    ``K`` contraction, producing a partial sum), and row-sharded activations.
    """
    layouts = (a, b, a_scales, b_scales)
    rows = [
        AxisAssignment((R, R, R, R), R),
        # Column-parallel: weight rows (N) sharded -> output columns (N).
        AxisAssignment((R, Sharded(0), R, Sharded(0)), Sharded(1)),
        # Row-parallel: contraction (K) sharded on every operand -> partial.
        AxisAssignment((Sharded(1), Sharded(1), Sharded(0), Sharded(1)), P),
        # Row-sharded activations (sequence / data parallel) -> output rows.
        AxisAssignment((Sharded(0), R, Sharded(1), R), Sharded(0)),
    ]
    return build_action_set(rows, layouts=layouts)


def _grouped_scaled_matmul_rule(
    hidden_states: TensorLayout,
    weight: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    expert_start_indices: TensorLayout,
    expert_ids: TensorLayout,
    *unused_kwargs: Any,
) -> ActionSet:
    """Sharding rule for the FP8 block-scaled grouped (MoE) matmul."""
    layouts = (
        hidden_states,
        weight,
        a_scales,
        b_scales,
        expert_start_indices,
        expert_ids,
    )
    rows = [
        AxisAssignment((R, R, R, R, R, R), R),
        # Column-parallel: weight's N (out) axis sharded -> output's N axis.
        AxisAssignment((R, Sharded(1), R, Sharded(1), R, R), Sharded(1)),
        # Row-parallel: weight's K (contraction) axis sharded, matched by
        # hidden_states' K axis; a_scales is transposed relative to
        # hidden_states, so its matching axis is 0 -> partial sum.
        AxisAssignment(
            (Sharded(1), Sharded(2), Sharded(0), Sharded(2), R, R), P
        ),
    ]
    return build_action_set(rows, layouts=layouts)


def _nvfp4_quantize_finalize(action: Action) -> Action:
    """Give the NVFP4 block-scale output the same placement as the data."""
    (data_mapping,) = action.outputs
    return Action(inputs=action.inputs, outputs=(data_mapping, data_mapping))


def _nvfp4_quantize_rule(x: TensorLayout, *extras: Any) -> ActionSet:
    """Sharding rule for dynamic NVFP4 activation block quantization."""
    rows = [
        AxisAssignment((R,), R),
        AxisAssignment((Sharded(0),), Sharded(0)),
        AxisAssignment((Sharded(1),), Sharded(1)),
    ]
    return build_action_set(
        rows, layouts=(x,), finalize=_nvfp4_quantize_finalize
    )


def _nvfp4_interleave_rule(scales: TensorLayout, *extras: Any) -> ActionSet:
    """Sharding rule for ``block_scales_interleave`` (rank-2 -> rank-5)."""
    rows = [
        AxisAssignment((R,), R),
        AxisAssignment((Sharded(0),), Sharded(0)),
        AxisAssignment((Sharded(1),), Sharded(1)),
    ]
    return build_action_set(rows, layouts=(scales,))


def _nvfp4_matmul_rule(
    a: TensorLayout,
    b: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    *extras: Any,
) -> ActionSet:
    """Sharding rule for the block-scaled NVFP4 matmul ``a @ b.T``."""
    layouts = (a, b, a_scales, b_scales)
    rows = [
        AxisAssignment((R, R, R, R), R),
        # Column-parallel: weight rows (N) sharded -> output columns (N).
        AxisAssignment((R, Sharded(0), R, Sharded(0)), Sharded(1)),
        # Row-parallel: contraction (K) sharded on every operand -> partial.
        AxisAssignment((Sharded(1), Sharded(1), Sharded(1), Sharded(1)), P),
        # Row-sharded activations (sequence / data parallel) -> output rows.
        AxisAssignment((Sharded(0), R, Sharded(0), R), Sharded(0)),
    ]
    return build_action_set(rows, layouts=layouts)


def _nvfp4_grouped_quantize_rule(
    input: TensorLayout,
    row_offsets: TensorLayout,
    scales_offsets: TensorLayout,
    expert_ids: TensorLayout,
    sf_tensor: TensorLayout,
    *extras: Any,
) -> ActionSet:
    """Sharding rule for grouped NVFP4 activation quantization."""
    rows = [
        AxisAssignment((R, R, R, R, R), R),
        AxisAssignment((Sharded(1), R, R, R, R), Sharded(1)),
    ]
    return build_action_set(
        rows,
        layouts=(input, row_offsets, scales_offsets, expert_ids, sf_tensor),
        finalize=_nvfp4_quantize_finalize,
    )


def _nvfp4_grouped_matmul_rule(
    hidden_states: TensorLayout,
    weight: TensorLayout,
    a_scales: TensorLayout,
    b_scales: TensorLayout,
    expert_start_indices: TensorLayout,
    a_scale_offsets: TensorLayout,
    expert_ids: TensorLayout,
    expert_scales: TensorLayout,
    *unused: Any,
) -> ActionSet:
    """Sharding rule for the NVFP4 block-scaled grouped (MoE) matmul."""
    layouts = (
        hidden_states,
        weight,
        a_scales,
        b_scales,
        expert_start_indices,
        a_scale_offsets,
        expert_ids,
        expert_scales,
    )
    rows = [
        AxisAssignment((R, R, R, R, R, R, R, R), R),
        # Column-parallel: weight's N axis (1) -> output's N axis.
        AxisAssignment((R, Sharded(1), R, Sharded(1), R, R, R, R), Sharded(1)),
        # Row-parallel: weight's K axis (2) matched by hidden/a_scales K -> P.
        AxisAssignment(
            (Sharded(1), Sharded(2), Sharded(1), Sharded(2), R, R, R, R),
            P,
        ),
    ]
    return build_action_set(rows, layouts=layouts)


# Wrap raw graph ops so they accept ``Tensor`` and run inside an
# ``ensure_context()``.
quantize_dynamic_scaled_float8 = F.functional(
    _quantize_dynamic_scaled_float8, rule=_quantize_rule
)
dynamic_scaled_matmul = F.functional(
    _dynamic_scaled_matmul, rule=_scaled_matmul_rule
)
grouped_dynamic_scaled_fp8_matmul = F.functional(
    _grouped_dynamic_scaled_fp8_matmul, rule=_grouped_scaled_matmul_rule
)
quantize_dynamic_block_scaled = F.functional(
    _quantize_dynamic_block_scaled, rule=_nvfp4_quantize_rule
)
block_scales_interleave = F.functional(
    _block_scales_interleave, rule=_nvfp4_interleave_rule
)
dynamic_block_scaled_matmul = F.functional(
    _dynamic_block_scaled_matmul, rule=_nvfp4_matmul_rule
)
grouped_quantize_dynamic_block_scaled = F.functional(
    _grouped_quantize_dynamic_block_scaled, rule=_nvfp4_grouped_quantize_rule
)
grouped_matmul_block_scaled = F.functional(
    _grouped_matmul_block_scaled, rule=_nvfp4_grouped_matmul_rule
)


def is_fp8_block_quantized(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if ``quant_config`` selects FP8 block-scaled weights."""
    return (
        quant_config is not None
        and quant_config.format == QuantFormat.BLOCKSCALED_FP8
    )


def is_nvfp4_quantized(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if ``quant_config`` selects NVFP4 block-scaled weights."""
    return quant_config is not None and quant_config.is_nvfp4


@dataclass(frozen=True)
class EPDispatchPayload:
    """Named, per-device view of the fused EP dispatch outputs.

    - bf16 dispatch: ``tokens``, ``expert_start``, ``expert_ids``,
      ``usage_stats``.
    - FP8 dispatch: adds ``scales``.
    - NVFP4 dispatch: adds ``scales`` and ``scales_offset``; ``usage_stats`` is
      absent (the NVFP4 grouped matmul synthesizes its own host stats).
    """

    tokens: list[Tensor]
    expert_start: list[Tensor]
    expert_ids: list[Tensor]
    usage_stats: list[Tensor] | None = None
    scales: list[Tensor] | None = None
    scales_offset: list[Tensor] | None = None

    @classmethod
    def from_dispatch(
        cls,
        dispatch_results: list[tuple[TensorValue, ...]],
        quant_config: QuantConfig | None,
        ep_config: EPConfig,
    ) -> EPDispatchPayload:
        """Parse the per-device dispatch tuples into named per-column fields.

        Args:
            dispatch_results: Per-device dispatch output tuples (the return of
                :meth:`~max.nn.comm.ep.EPBatchManager.ep_dispatch`).
            quant_config: Routed-expert quant config selecting the format.
            ep_config: EP batch-manager config; its ``dispatch_dtype`` decides
                whether a ``scales`` column is present (a separate axis from the
                weight quant format).
        """
        columns = [
            [Tensor.from_graph_value(v) for v in column]
            for column in zip(*dispatch_results, strict=True)
        ]

        if (
            quant_config is not None
            and quant_config.format == QuantFormat.NVFP4
        ):
            tokens, scales, expert_start, scales_offset, expert_ids, _ = columns
            return cls(
                tokens=tokens,
                expert_start=expert_start,
                expert_ids=expert_ids,
                scales=scales,
                scales_offset=scales_offset,
            )
        else:
            fp8_dispatch = (
                ep_config.dispatch_dtype.is_float8()
                and ep_config.dispatch_quant_config is not None
            )
            expert_start, expert_ids, usage_stats = columns[-3:]
            for stat in usage_stats:
                assert stat.device.is_host
            return cls(
                tokens=columns[0],
                expert_start=expert_start,
                expert_ids=expert_ids,
                usage_stats=usage_stats,
                scales=columns[1] if fp8_dispatch else None,
            )

    def local_map_tokens(
        self,
        quant_config: QuantConfig | None,
        *,
        nvfp4_global_scale: Tensor | None = None,
    ) -> Sequence[QuantAwareTensor]:
        """Wrap the raw per-device token columns into activations.

        Args:
            quant_config: Routed-expert quant config (supplies the FP8 block
                size).
            nvfp4_global_scale: Uniform NVFP4 activation scale the dispatch
                quantized with (the max static ``input_scale`` across experts);
                required when the payload carries NVFP4 tokens.
        """
        if self.scales_offset is not None:
            assert self.scales is not None and nvfp4_global_scale is not None, (
                "NVFP4 tokens require dispatch scales and the uniform scale"
            )
            return [
                NVFP4Activation(
                    data=d,
                    scales=s,
                    input_scale=nvfp4_global_scale,
                    scales_offset=o,
                )
                for d, s, o in zip(
                    self.tokens, self.scales, self.scales_offset, strict=True
                )
            ]
        if self.scales is not None:
            assert quant_config is not None
            weight_block = quant_config.weight_scale.block_size
            assert weight_block is not None
            return [
                FP8BlockTensor(
                    data=d,
                    weight_scale_inv=s,
                    block_size=(1, weight_block[1]),
                )
                for d, s in zip(self.tokens, self.scales, strict=True)
            ]
        return self.tokens


def ep_requires_dispatch_scales(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if the EP dispatch kernel needs input scales."""
    return quant_config is not None and quant_config.format == QuantFormat.NVFP4


def moe_requires_scales_offsets(quant_config: QuantConfig | None) -> bool:
    """Return ``True`` if the MoE dispatch kernel needs scales offsets.

    Note, currently matches `ep_requires_dispatch_scales` but may diverge
    in the future.
    """
    return quant_config is not None and quant_config.format == QuantFormat.NVFP4


_SUPPORTED_FORMATS = (QuantFormat.BLOCKSCALED_FP8, QuantFormat.NVFP4)


def routed_weight_dtype(quant_config: QuantConfig | None) -> DType:
    """The storage dtype :func:`quantized_weight` picks for ``quant_config``."""
    if is_nvfp4_quantized(quant_config):
        return DType.uint8
    if is_fp8_block_quantized(quant_config):
        return DType.float8_e4m3fn
    return DType.bfloat16


def quantized_weight(
    out_dim: int,
    in_dim: int,
    quant_config: QuantConfig | None,
) -> QuantAwareTensor:
    """Build a Linear-shaped ``[out_dim, in_dim]`` weight parameter.

    Returns an :class:`FP8BlockTensor` for FP8 block scaling, an
    :class:`NVFP4Tensor` for NVFP4, otherwise a plain bf16 :class:`Tensor`.
    """
    if quant_config and quant_config.format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Quant type {quant_config.format} is not yet supported."
        )
    if is_nvfp4_quantized(quant_config):
        assert quant_config is not None
        block_size = quant_config.weight_scale.block_size
        assert block_size is not None
        return NVFP4Tensor.zeros(
            (int(out_dim), int(in_dim)), block_size=block_size
        )
    if is_fp8_block_quantized(quant_config):
        assert quant_config is not None
        block_size = quant_config.weight_scale.block_size
        assert block_size is not None
        return FP8BlockTensor.zeros(
            (int(out_dim), int(in_dim)), block_size=block_size
        )
    return Tensor.zeros((int(out_dim), int(in_dim)))


def stack(items: list[QuantAwareTensor], axis: int = 0) -> QuantAwareTensor:
    """Stack a homogeneous bundle along ``axis``, dispatching on quant type.

    For FP8 items, both leaves (``data`` and ``weight_scale_inv``) are stacked and
    rewrapped in an :class:`FP8BlockTensor`; for plain tensors the list is
    stacked directly. Companion to :func:`concat_weights`.

    Args:
        items: Homogeneous list of :class:`QuantAwareTensor`s (all plain
            tensors, or all :class:`FP8BlockTensor`s).
        axis: Axis to stack along (a new dimension is inserted here).

    Returns:
        A single stacked :class:`QuantAwareTensor` of the same kind as
        ``items``.
    """
    first = items[0]
    if isinstance(first, FP8BlockTensor):
        assert all_fp8_block(items)
        return FP8BlockTensor(
            data=F.stack([w.data for w in items], axis=axis),
            weight_scale_inv=F.stack(
                [w.weight_scale_inv for w in items], axis=axis
            ),
            block_size=first.block_size,
        )
    if isinstance(first, NVFP4Tensor):
        assert all_nvfp4(items)
        return NVFP4Tensor(
            data=F.stack([w.data for w in items], axis=axis),
            weight_scale=F.stack([w.weight_scale for w in items], axis=axis),
            weight_scale_2=F.stack(
                [w.weight_scale_2 for w in items], axis=axis
            ),
            input_scale=F.stack([w.input_scale for w in items], axis=axis),
            block_size=first.block_size,
        )
    assert all(isinstance(w, Tensor) for w in items)
    return F.stack(list(items), axis=axis)


def combine_quant_per_device(
    items: list[QuantAwareTensor],
    combine: Callable[[list[Tensor]], list[Tensor]],
    *,
    global_scale_items: list[QuantAwareTensor] | None = None,
) -> list[QuantAwareTensor]:
    """Map a per-device leaf transform over a homogeneous bundle.

    ``combine`` merges all items' tensors for one leaf into a per-device list
    (one tensor per mesh device) — e.g. a TP shard-and-stack.  For FP8 input,
    ``combine`` is applied to the ``data`` and ``weight_scale_inv`` leaves
    independently and the per-device leaves are zipped back into one
    :class:`FP8BlockTensor` per device, so the FP8 invariant (``data`` and
    ``weight_scale_inv`` are each a single :class:`~max.experimental.tensor.Tensor`)
    is preserved without the caller transposing a struct-of-lists into a
    list-of-structs.  For plain tensors the per-device list is returned as-is.

    ``combine`` must be leaf-agnostic — read any per-leaf difference (e.g. the
    block-scale leaf's smaller trailing dim) off the leaf tensors' own shapes
    rather than branching on which leaf it is.

    Args:
        items: Homogeneous list of :class:`QuantAwareTensor`s (all plain
            tensors, or all :class:`FP8BlockTensor`s).
        combine: Callable that merges a list of leaf tensors into a per-device
            list of tensors.
        global_scale_items: Items to read the NVFP4 per-tensor scales from, when
            ``items`` holds more than one entry per expert. The TP gate/up
            bundle interleaves each expert's gate and up weights, which share
            one global scale (see :func:`concat_nvfp4`), so stacking over
            ``items`` would give two entries per expert. Defaults to ``items``.

    Returns:
        One :class:`QuantAwareTensor` per device. For FP8 input, each is an
        :class:`FP8BlockTensor` whose ``data``/``weight_scale_inv`` are that device's
        leaves; for plain tensors, the per-device list is returned directly.
    """
    first = items[0]
    if isinstance(first, FP8BlockTensor):
        assert all_fp8_block(items)
        data = combine([w.data for w in items])
        weight_scale_inv = combine([w.weight_scale_inv for w in items])
        return [
            FP8BlockTensor(
                data=d, weight_scale_inv=s, block_size=first.block_size
            )
            for d, s in zip(data, weight_scale_inv, strict=True)
        ]
    if isinstance(first, NVFP4Tensor):
        assert all_nvfp4(items)
        scale_items = (
            items if global_scale_items is None else global_scale_items
        )
        assert all_nvfp4(scale_items)
        data = combine([w.data for w in items])
        weight_scale = combine([w.weight_scale for w in items])
        weight_scale_2 = F.stack(
            [w.weight_scale_2 for w in scale_items], axis=0
        )
        input_scale = F.stack([w.input_scale for w in scale_items], axis=0)
        return [
            NVFP4Tensor(
                data=d,
                weight_scale=s,
                weight_scale_2=weight_scale_2,
                input_scale=input_scale,
                block_size=first.block_size,
            )
            for d, s in zip(data, weight_scale, strict=True)
        ]
    plain = [w for w in items if isinstance(w, Tensor)]
    result: list[QuantAwareTensor] = [*combine(plain)]
    return result


def stack_device_shards(
    shards: Sequence[QuantAwareTensor], axis: int, mesh: DeviceMesh
) -> QuantAwareTensor:
    """Reassembles a per-device weight-shard bundle into one ``Sharded`` tensor."""
    if len(shards) == 1:
        return shards[0]
    mapping = PlacementMapping(mesh, (Sharded(axis=axis),))
    first = shards[0]
    if isinstance(first, FP8BlockTensor):
        assert all_fp8_block(shards)
        return FP8BlockTensor(
            data=Tensor.from_shard_values(
                [TensorValue(s.data) for s in shards], mapping
            ),
            weight_scale_inv=Tensor.from_shard_values(
                [TensorValue(s.weight_scale_inv) for s in shards], mapping
            ),
            block_size=first.block_size,
        )
    if isinstance(first, NVFP4Tensor):
        assert all_nvfp4(shards)
        return NVFP4Tensor(
            data=Tensor.from_shard_values(
                [TensorValue(s.data) for s in shards], mapping
            ),
            weight_scale=Tensor.from_shard_values(
                [TensorValue(s.weight_scale) for s in shards], mapping
            ),
            weight_scale_2=first.weight_scale_2,
            input_scale=first.input_scale,
            block_size=first.block_size,
        )
    return Tensor.from_shard_values(
        [TensorValue(s) for s in shards if isinstance(s, Tensor)], mapping
    )


def concat_weights(
    *weights: QuantAwareTensor, axis: int = 0
) -> QuantAwareTensor:
    """Concatenate weights along ``axis``, dispatching on the weight type."""
    if not weights:
        raise ValueError("concat_weights requires at least one tensor")
    if isinstance(weights[0], FP8BlockTensor):
        assert all_fp8_block(weights), (
            "concat_weights requires all weights to be FP8BlockTensor when "
            "the first is"
        )
        return concat_fp8_block(*weights, axis=axis)
    if isinstance(weights[0], NVFP4Tensor):
        assert all_nvfp4(weights), (
            "concat_weights requires all weights to be NVFP4Tensor when the "
            "first is"
        )
        return concat_nvfp4(*weights, axis=axis)
    assert all(isinstance(w, Tensor) for w in weights)
    return F.concat(list(weights), axis=axis)


def _fp8_block_specs(
    weight_block: tuple[int, int],
    *,
    input_block: tuple[int, int] = (1, 128),
) -> tuple[InputScaleSpec, WeightScaleSpec]:
    """Standard FP8 block-scale specs for matmul/grouped-matmul kernels."""
    return (
        InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=input_block,
        ),
        WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=weight_block,
        ),
    )


def matmul(x: Tensor, weight: QuantAwareTensor) -> Tensor:
    """Matmul ``x @ weight.T`` dispatching on the weight type.

    ``weight`` follows the Linear convention: shape ``[out_dim, in_dim]``.

    - ``Tensor`` weight: regular bf16/float matmul.
    - :class:`FP8BlockTensor` weight: quantizes ``x`` to FP8 with
      ``(1, block_k)`` activation blocks, then runs the block-scaled FP8
      matmul kernel and returns bf16.
    - :class:`NVFP4Tensor` weight: normalizes ``x`` by the static
      ``input_scale``, dynamically block-quantizes it to FP4, then runs the
      SM100 block-scaled FP4 matmul and returns bf16.
    """
    if isinstance(weight, FP8BlockTensor):
        return _matmul_fp8_block(x, weight)
    if isinstance(weight, NVFP4Tensor):
        return _matmul_nvfp4(x, weight)
    assert isinstance(weight, Tensor)
    return x @ weight.T


def _matmul_nvfp4(x: Tensor, weight: NVFP4Tensor) -> Tensor:
    """Block-scaled NVFP4 matmul ``x @ weight.T`` with dynamic activation quant."""
    x_fp4, x_scales = quantize_dynamic_block_scaled(
        x,
        tensor_sf=1.0 / weight.input_scale,
        sf_vector_size=_NVFP4_SF_VECTOR_SIZE,
        scales_type=DType.float8_e4m3fn,
        out_type=DType.uint8,
    )
    weight_scale = block_scales_interleave(
        weight.weight_scale, sf_vector_size=_NVFP4_SF_VECTOR_SIZE
    )
    return dynamic_block_scaled_matmul(
        x_fp4,
        weight.data,
        x_scales,
        weight_scale,
        tensor_sf=weight.weight_scale_2 * weight.input_scale,
        sf_vector_size=_NVFP4_SF_VECTOR_SIZE,
        out_type=DType.bfloat16,
    )


def _matmul_fp8_block(x: Tensor, weight: FP8BlockTensor) -> Tensor:
    """Block-scaled FP8 matmul ``x @ weight.data.T`` with dynamic activation
    quantization.

    The activation block is ``(1, block_k)`` and the weight block is
    ``(block_m, block_k) = weight.block_size``. The kernel returns bf16.
    """
    block_m, block_k = weight.block_size
    input_spec, weight_spec = _fp8_block_specs(
        (block_m, block_k), input_block=(1, block_k)
    )

    x_fp8, x_scales = quantize_dynamic_scaled_float8(
        x,
        input_spec,
        weight_spec,
        group_size_or_per_token=block_k,
        scales_type=DType.float32,
        out_type=DType.float8_e4m3fn,
    )

    return dynamic_scaled_matmul(
        x_fp8,
        weight.data,
        x_scales,
        weight.weight_scale_inv,
        input_spec,
        weight_spec,
        out_type=DType.bfloat16,
    )


def grouped_matmul(
    x: QuantAwareTensor,
    weight: QuantAwareTensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_usage_stats: Tensor | None,
    *,
    scales_offset: Tensor | None = None,
    out_type: DType = DType.bfloat16,
) -> Tensor:
    """Grouped (MoE) matmul dispatching on the stacked-weight type.

    For a plain ``Tensor`` weight of shape ``[num_experts, N, K]``, this
    falls back to the standard ragged grouped matmul. For an
    :class:`FP8BlockTensor` weight, it runs the block-scaled FP8 grouped
    matmul kernel. For an :class:`NVFP4Tensor` weight, it runs the SM100
    block-scaled FP4 grouped matmul (``scales_offset`` required).

    Args:
        x: Ragged activations of shape ``[total_tokens, K]``.
        weight: Stacked expert weights, ``[num_experts, N, K]``.
        expert_start_indices: Ragged group offsets, ``uint32``.
        expert_ids: Per-group expert id, ``int32``.
        expert_usage_stats: ``[max_tokens_per_expert, num_active_experts]``
            device tensor. The bf16 fallback requires it on-device (the SM100
            kernel reads ``num_active_experts`` there); the FP8 branch copies it
            to CPU itself. The NVFP4 branch synthesizes its own host stats and
            ignores this (so it may be ``None``).
        scales_offset: Per-expert block-scale offsets from
            :func:`moe_create_indices` with ``needs_scales_offset=True``;
            required for the NVFP4 branch, ignored otherwise.
        out_type: Output dtype for the FP8/NVFP4 branch (bf16 by default).
    """
    if isinstance(weight, NVFP4Tensor):
        if isinstance(x, NVFP4Activation):
            x_fp4, x_scales = x.data, x.scales
            offsets, act_scale = x.scales_offset, x.input_scale
        else:
            assert isinstance(x, Tensor), (
                "NVFP4 grouped matmul expects bf16 activations (quantized here)"
            )
            assert scales_offset is not None, (
                "NVFP4 grouped matmul requires scales_offset from "
                "moe_create_indices(needs_scales_offset=True)"
            )
            offsets = scales_offset
            x_fp4, x_scales = grouped_quantize_dynamic_block_scaled(
                x,
                row_offsets=expert_start_indices,
                scales_offsets=scales_offset,
                expert_ids=expert_ids,
                sf_tensor=1.0 / weight.input_scale,
                sf_vector_size=_NVFP4_SF_VECTOR_SIZE,
                scales_type=DType.float8_e4m3fn,
                out_type=DType.uint8,
            )
            act_scale = weight.input_scale
        expert_scales = (weight.weight_scale_2 * act_scale).to(x_fp4.mesh)
        return _nvfp4_grouped_matmul(
            x_fp4,
            weight,
            x_scales,
            offsets,
            expert_start_indices,
            expert_ids,
            expert_scales,
            out_type=out_type,
        )
    if isinstance(weight, FP8BlockTensor):
        if isinstance(x, FP8BlockTensor):
            # Activations already block-quantized (e.g. by the EP FP8 dispatch
            # or a quantized SiLU); pass them through directly.
            x_fp8, x_scales = x.data, x.weight_scale_inv
        else:
            assert isinstance(x, Tensor)
            x_fp8, x_scales = _quantize_activation_fp8(x, weight.block_size)
        assert expert_usage_stats is not None
        return _grouped_fp8_matmul(
            x_fp8,
            x_scales,
            weight,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
            out_type=out_type,
        )
    assert isinstance(x, Tensor) and isinstance(weight, Tensor)
    assert scales_offset is None
    assert expert_usage_stats is not None
    return grouped_matmul_ragged(
        x, weight, expert_start_indices, expert_ids, expert_usage_stats
    )


def _interleave_grouped_scales(scales: Tensor) -> Tensor:
    """Interleave per-expert block scales into the rank-6 TCGEN layout."""
    num_experts = int(scales.shape[0])
    m = scales.shape[1]
    k = scales.shape[2]
    per_expert = F.split(scales, [1] * num_experts, axis=0)
    return F.stack(
        [
            block_scales_interleave(
                s.reshape([m, k]), sf_vector_size=_NVFP4_SF_VECTOR_SIZE
            )
            for s in per_expert
        ],
        axis=0,
    )


def _nvfp4_grouped_matmul(
    x_fp4: Tensor,
    weight: NVFP4Tensor,
    x_scales: Tensor,
    scales_offset: Tensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_scales: Tensor,
    *,
    out_type: DType = DType.bfloat16,
) -> Tensor:
    """Block-scaled NVFP4 grouped matmul on already-quantized activations."""
    b_scales = _interleave_grouped_scales(weight.weight_scale)
    num_active = int(expert_ids.shape[0])
    usage_stats_host = F.constant(
        [8192, num_active], dtype=DType.uint32, device=CPU()
    )
    return grouped_matmul_block_scaled(
        x_fp4,
        weight.data,
        x_scales,
        b_scales,
        expert_start_indices,
        scales_offset,
        expert_ids,
        expert_scales,
        usage_stats_host,
        out_type=out_type,
    )


def sigma_permute_gate_up_nvfp4(weight: NVFP4Tensor) -> NVFP4Tensor:
    """Sigma-permutation when fused SwiGLU+NVFP4 is enabled.

    The stacked [2E, scale_m, scale_k] tensor splits to
    [E, 2, scale_m, scale_k], then permute axes 1,2 → collapse to
    [E, 2*scale_m, scale_k] with rows row-interleaved (g_0, u_0, ...).
    This sits before NvMxf4f8Strategy.prepare_weight_scales lifts to the
    5D tcgen05 layout the kernel expects.
    """

    def _perm(t: Tensor) -> Tensor:
        e = int(t.shape[0])
        two_d = int(t.shape[1])
        c = int(t.shape[-1])
        return (
            t.reshape([e, 2, two_d // 2, c])
            .permute([0, 2, 1, 3])
            .reshape([e, two_d, c])
        )

    return NVFP4Tensor(
        data=_perm(weight.data),
        weight_scale=_perm(weight.weight_scale),
        weight_scale_2=weight.weight_scale_2,
        input_scale=weight.input_scale,
        block_size=weight.block_size,
    )


def grouped_matmul_swiglu_nvfp4_ep(
    tokens: Tensor,
    token_scales: Tensor,
    weight: NVFP4Tensor,
    expert_start_indices: Tensor,
    scales_offset: Tensor,
    expert_ids: Tensor,
    expert_scales: Tensor,
    down_input_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fused NVFP4 gate/up grouped matmul + SwiGLU + re-quantize (EP path)."""
    b_scales = _interleave_grouped_scales(weight.weight_scale)
    num_active = int(expert_ids.shape[0])
    usage_stats_host = F.constant(
        [8192, num_active], dtype=DType.uint32, device=CPU()
    )
    # The fused kernel consumes the inverted down input scale internally;
    # invert here to match the fused_silu_quantized path, which passes the raw
    # scale (and inverts it itself).
    c_input_scales = 1.0 / down_input_scale
    c_packed, c_scales = _grouped_matmul_blocked_swiglu(
        TensorValue(tokens),
        TensorValue(weight.data),
        TensorValue(token_scales),
        TensorValue(b_scales),
        TensorValue(expert_start_indices),
        TensorValue(scales_offset),
        TensorValue(expert_ids),
        TensorValue(usage_stats_host),
        expert_scales=TensorValue(expert_scales.to(tokens.device)),
        c_input_scales=TensorValue(c_input_scales.to(tokens.device)),
    )
    return Tensor.from_graph_value(c_packed), Tensor.from_graph_value(c_scales)


def grouped_matmul_silu(
    tokens: QuantAwareTensor,
    gate_up: QuantAwareTensor,
    down: QuantAwareTensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_usage_stats: Tensor | None,
    quant_config: QuantConfig | None,
    scales_offset: Tensor | None = None,
) -> QuantAwareTensor:
    """Gate/up grouped matmul + SwiGLU, returning the down-projection input."""
    # Pre-quantized EP activations carry their own per-expert scale offset;
    # single-device tokens rely on the caller-supplied one.
    offset = (
        tokens.scales_offset
        if isinstance(tokens, NVFP4Activation)
        else scales_offset
    )
    if (
        isinstance(tokens, NVFP4Activation)
        and quant_config is not None
        and quant_config.can_use_fused_swiglu
    ):
        assert isinstance(gate_up, NVFP4Tensor) and isinstance(
            down, NVFP4Tensor
        )
        assert offset is not None
        # Gate/up epilogue uses the uniform scale the dispatch quantized with;
        # the down input is re-quantized with the per-expert down.input_scale.
        gate_up_scale = gate_up.weight_scale_2 * tokens.input_scale
        c_packed, c_scales = grouped_matmul_swiglu_nvfp4_ep(
            tokens.data,
            tokens.scales,
            gate_up,
            expert_start_indices,
            offset,
            expert_ids,
            gate_up_scale,
            down.input_scale,
        )
        return NVFP4Activation(
            data=c_packed,
            scales=c_scales,
            input_scale=down.input_scale,
            scales_offset=offset,
        )

    gate_up_out = grouped_matmul(
        tokens,
        gate_up,
        expert_start_indices,
        expert_ids,
        expert_usage_stats,
        scales_offset=scales_offset,
    )
    # Fuse the down-proj re-quant into the SiLU only when the activation was
    # pre-quantized upstream (EP dispatch); otherwise emit bf16 and let the
    # down matmul quantize.
    prequantized = isinstance(tokens, (FP8BlockTensor, NVFP4Activation))
    requant_weight = down if prequantized else None
    return grouped_silu(
        gate_up_out,
        expert_start_indices,
        requant_weight,
        quant_config,
        scales_offset=offset,
    )


def _quantize_activation_fp8(
    x: Tensor, weight_block: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Per-token FP8 block quantization of ``x`` matching ``weight_block``."""
    block_m, block_k = weight_block
    input_spec, weight_spec = _fp8_block_specs(
        (block_m, block_k), input_block=(1, block_k)
    )
    return quantize_dynamic_scaled_float8(
        x,
        input_spec,
        weight_spec,
        scales_type=DType.float32,
        out_type=DType.float8_e4m3fn,
    )


def _grouped_fp8_matmul(
    x_fp8: Tensor,
    x_scales: Tensor,
    weight: FP8BlockTensor,
    expert_start_indices: Tensor,
    expert_ids: Tensor,
    expert_usage_stats: Tensor,
    *,
    out_type: DType = DType.bfloat16,
) -> Tensor:
    """Block-scaled FP8 grouped matmul on already-quantized activations."""
    block_m, block_k = weight.block_size
    input_spec, weight_spec = _fp8_block_specs(
        (block_m, block_k), input_block=(1, block_k)
    )
    # This kernel reads the usage stats host-side, so copy to CPU here rather
    # than at the call site (the bf16 path needs them on-device).
    return grouped_dynamic_scaled_fp8_matmul(
        x_fp8,
        weight.data,
        x_scales,
        weight.weight_scale_inv,
        expert_start_indices,
        expert_ids,
        expert_usage_stats.to(CPU()),
        input_spec,
        weight_spec,
        out_type=out_type,
    )


def grouped_silu(
    x: Tensor,
    expert_start_indices: Tensor,
    out_weight: QuantAwareTensor | None = None,
    quant_config: QuantConfig | None = None,
    *,
    scales_offset: Tensor | None = None,
) -> QuantAwareTensor:
    """SiLU-gate a grouped gate/up output."""
    if isinstance(out_weight, NVFP4Tensor):
        assert quant_config is not None and scales_offset is not None
        data, scales = fused_silu_quantized(
            x,
            expert_start_indices,
            quant_config,
            DType.uint8,
            input_scales=out_weight.input_scale,
            scales_offsets=scales_offset,
        )
        return NVFP4Activation(
            data=data,
            scales=scales,
            input_scale=out_weight.input_scale,
            scales_offset=scales_offset,
        )
    if isinstance(out_weight, FP8BlockTensor):
        assert quant_config is not None
        _, block_k = out_weight.block_size
        data, weight_scale_inv = fused_silu_quantized(
            x, expert_start_indices, quant_config, DType.float8_e4m3fn
        )
        return FP8BlockTensor(
            data=data,
            weight_scale_inv=weight_scale_inv,
            block_size=(1, block_k),
        )
    # Exhaustive: a new quantized down-weight must not silently skip its
    # re-quant by taking the plain bf16 SiLU.
    assert out_weight is None or isinstance(out_weight, Tensor)
    return fused_silu(x, expert_start_indices)


def concat_fp8_block(*tensors: FP8BlockTensor, axis: int = 0) -> FP8BlockTensor:
    """Concatenate two or more :class:`FP8BlockTensor`s along ``axis``."""
    if not tensors:
        raise ValueError("concat_fp8_block requires at least one tensor")
    if axis != 0:
        raise ValueError(
            "FP8BlockTensor concat currently only supports axis=0 (row axis)"
        )
    block_size = tensors[0].block_size
    for q in tensors[1:]:
        if q.block_size != block_size:
            raise ValueError(
                "All FP8BlockTensors must have the same block_size to "
                f"concat; got {block_size} and {q.block_size}"
            )

    data = F.concat([q.data for q in tensors], axis=0)
    weight_scale_inv = F.concat([q.weight_scale_inv for q in tensors], axis=0)
    return FP8BlockTensor(
        data=data, weight_scale_inv=weight_scale_inv, block_size=block_size
    )


def concat_nvfp4(*tensors: NVFP4Tensor, axis: int = 0) -> NVFP4Tensor:
    """Concatenate two or more :class:`NVFP4Tensor`s along the row axis.

    Concatenates the packed ``data`` and per-block ``weight_scale`` along
    ``axis`` (the ``N``/row axis). The per-tensor ``weight_scale_2`` and static
    ``input_scale`` are taken from the first tensor: modelopt emits a shared
    global scale for the fused ``gate``/``up`` projections.
    """
    if not tensors:
        raise ValueError("concat_nvfp4 requires at least one tensor")
    if axis != 0:
        raise ValueError(
            "NVFP4Tensor concat currently only supports axis=0 (row axis)"
        )
    block_size = tensors[0].block_size
    for q in tensors[1:]:
        if q.block_size != block_size:
            raise ValueError(
                "All NVFP4Tensors must have the same block_size to concat; "
                f"got {block_size} and {q.block_size}"
            )
    return NVFP4Tensor(
        data=F.concat([q.data for q in tensors], axis=0),
        weight_scale=F.concat([q.weight_scale for q in tensors], axis=0),
        weight_scale_2=tensors[0].weight_scale_2,
        input_scale=tensors[0].input_scale,
        block_size=block_size,
    )
