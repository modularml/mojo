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
"""Quantization strategies for MoE layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from ..comm.ep.ep_kernels import fused_silu_quantized
from ..kernels import (
    block_scales_interleave,
    grouped_dynamic_block_scaled_matmul_amd,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_dynamic_scaled_mxfp6_matmul,
    grouped_matmul_block_scaled,
    grouped_matmul_blocked_swiglu,
    grouped_quantize_dynamic_block_scaled,
    quantize_dynamic_block_scaled_mxfp4,
    quantize_dynamic_block_scaled_mxfp6,
    quantize_dynamic_scaled_float8,
)
from ..quant_config import QuantConfig


@dataclass
class Nvfp4Scales:
    """Bundled scales for NVFP4 quantization.

    Note: ``gate_up_input`` is broadcast-uniform (all experts share the
    global max), whereas ``down_input`` contains genuinely per-expert
    scales.  The non-EP path must account for this asymmetry when
    converting per-expert scales to a single global activation scale.
    """

    gate_up_input: TensorValue
    down_input: TensorValue
    gate_up_expert: TensorValue
    down_expert: TensorValue


class QuantStrategy(Protocol):
    """Quantization strategy for MoE layers."""

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations and returns (quantized, scales)."""
        ...

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
        a_scales_preshuffled: bool = False,
        a_scales_max_padded_m: int = 0,
    ) -> TensorValue:
        """Runs grouped matmul for routed experts."""
        ...

    def prepare_weight_scales(
        self,
        gate_up: TensorValue,
        down: TensorValue,
        device: DeviceRef,
    ) -> tuple[TensorValue, TensorValue]:
        """Prepares weight scales for kernel consumption."""
        ...

    def grouped_quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None,
        expert_start: TensorValue,
        scales_offset: TensorValue,
        expert_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations with per-expert scales and padding."""
        ...

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        max_padded_M: int = 0,
        clamp_activation: bool = False,
        swiglu_alpha: float = 0.0,
        swiglu_limit: float = 0.0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies gating and quantizes activations for the down proj."""
        ...


class Fp8Strategy:
    """FP8 quantization for MoE."""

    def __init__(self, config: QuantConfig, dtype: DType):
        self.config = config
        self.dtype = dtype

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations to FP8 and returns (quantized, scales)."""
        return quantize_dynamic_scaled_float8(
            tensor,
            self.config.input_scale,
            self.config.weight_scale,
            group_size_or_per_token=group_size,
            out_type=self.dtype,
            scales_type=self.config.weight_scale.dtype,
        )

    def grouped_quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None,
        expert_start: TensorValue,
        scales_offset: TensorValue,
        expert_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Falls back to ungrouped FP8 quantization."""
        return self.quantize(tensor, group_size)

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
        a_scales_preshuffled: bool = False,
        a_scales_max_padded_m: int = 0,
    ) -> TensorValue:
        """Runs grouped FP8 matmul for the routed experts.

        ``a_scales_preshuffled``/``a_scales_max_padded_m`` are accepted for
        ``QuantStrategy`` conformance; they only apply to the MXFP4 EP scale
        fusion and are ignored here.
        """
        hidden, input_scales, expert_start, expert_ids, usage_stats = (
            expert_inputs
        )

        return grouped_dynamic_scaled_fp8_matmul(
            hidden,
            weight,
            input_scales,
            weight_scales,
            expert_start,
            expert_ids,
            usage_stats.to(DeviceRef.CPU()),
            self.config.input_scale,
            self.config.weight_scale,
        )

    def prepare_weight_scales(
        self,
        gate_up: TensorValue,
        down: TensorValue,
        device: DeviceRef,
    ) -> tuple[TensorValue, TensorValue]:
        """Passes FP8 weight scales through without reformatting."""
        return gate_up, down

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        max_padded_M: int = 0,
        clamp_activation: bool = False,
        swiglu_alpha: float = 0.0,
        swiglu_limit: float = 0.0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies fused SiLU gate and returns quantized activations.

        ``max_padded_M`` and the ``clamp_activation``/``swiglu_*`` args are
        accepted for ``QuantStrategy`` conformance; they only apply to the
        MXFP4 EP path and are ignored here.
        """
        assert not clamp_activation, (
            "clamped SwiGLU-OAI activation is only supported on the MXFP4 EP"
            " path"
        )
        _, _, expert_start_indices, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
        )


class NvMxf4f8Strategy:
    """NVIDIA NVFP4/MXFP4/MXFP8 quantization for MoE."""

    def __init__(self, config: QuantConfig, dtype: DType):
        self.config = config
        self.dtype = dtype

    @property
    def is_nvfp4(self) -> bool:
        """Whether this strategy is handling NVFP4 rather than MXFP4 or MXFP8."""
        return self.config.is_nvfp4

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
    ) -> tuple[TensorValue, TensorValue]:
        raise NotImplementedError(
            "To quantize to NVFP4/MXFP4/MXFP8, use grouped_quantize instead"
        )

    def grouped_quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None,
        expert_start: TensorValue,
        scales_offset: TensorValue,
        expert_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations per-expert with padded scale alignment."""
        if self.is_nvfp4 and input_scale is None:
            raise ValueError("NVFP4 requires input_scale")
        sf_tensor = (
            (1.0 / input_scale).to(tensor.device)
            if input_scale is not None
            else ops.broadcast_to(
                ops.constant(1.0, DType.float32, device=tensor.device),
                expert_ids.shape,
            )
        )
        return grouped_quantize_dynamic_block_scaled(
            tensor,
            row_offsets=expert_start,
            scales_offsets=scales_offset,
            expert_ids=expert_ids,
            sf_tensor=sf_tensor,
            sf_vector_size=16 if self.is_nvfp4 else 32,
            scales_type=(
                DType.float8_e4m3fn if self.is_nvfp4 else DType.float8_e8m0fnu
            ),
            out_type=self.dtype,
        )

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
        a_scales_preshuffled: bool = False,
        a_scales_max_padded_m: int = 0,
    ) -> TensorValue:
        """Runs grouped NVIDIA block-scaled matmul with per-expert scales.

        ``a_scales_preshuffled``/``a_scales_max_padded_m`` are accepted for
        ``QuantStrategy`` conformance; they only apply to the MXFP4 EP scale
        fusion and are ignored here.
        """
        if self.is_nvfp4 and expert_scales is None:
            raise ValueError("NVFP4 requires expert_scales")

        (
            hidden,
            hidden_scales,
            expert_start,
            scales_offsets,
            expert_ids,
            usage_stats,
        ) = expert_inputs

        # Replace the gpu usage stats with a dummy cpu usage stats
        if usage_stats.device.is_gpu():
            usage_stats = ops.constant(
                [8192, int(expert_ids.shape[0])],
                dtype=DType.uint32,
                device=DeviceRef.CPU(),
            )

        if expert_scales is None:
            # Create a dummy expert scales tensor with shape (num_experts,) for
            # MXFP4 and MXFP8.
            expert_scales = ops.broadcast_to(
                ops.constant(1.0, DType.float32, device=hidden.device),
                expert_ids.shape,
            )

        return grouped_matmul_block_scaled(
            hidden,
            weight,
            hidden_scales,
            weight_scales,
            expert_start,
            scales_offsets,
            expert_ids,
            expert_scales.to(hidden.device),
            usage_stats,
            estimated_total_m=estimated_total_m,
        )

    def prepare_weight_scales(
        self,
        gate_up: TensorValue,
        down: TensorValue,
        device: DeviceRef,
    ) -> tuple[TensorValue, TensorValue]:
        """Interleaves NVIDIA block scales for kernel layout."""
        return (
            _nv_interleave_block_scales(gate_up, device),
            _nv_interleave_block_scales(down, device),
        )

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        max_padded_M: int = 0,
        clamp_activation: bool = False,
        swiglu_alpha: float = 0.0,
        swiglu_limit: float = 0.0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies SiLU gate then quantizes the result.

        ``max_padded_M`` and the ``clamp_activation``/``swiglu_*`` args are
        accepted for ``QuantStrategy`` conformance; they only apply to the
        MXFP4 EP path and are ignored here.
        """
        assert not clamp_activation, (
            "clamped SwiGLU-OAI activation is only supported on the MXFP4 EP"
            " path"
        )
        _, _, expert_start_indices, scales_offsets, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
            input_scales,
            scales_offsets,
        )

    def grouped_matmul_swiglu(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        *,
        expert_scales: TensorValue | None = None,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...],
        estimated_total_m: TensorValue | None = None,
        use_swigluoai: bool = False,
        swiglu_alpha: float = 0.0,
        swiglu_limit: float = 0.0,
    ) -> tuple[TensorValue, TensorValue]:
        """Runs the fused quantized grouped matmul + SwiGLU + quant kernel.

        Equivalent to ``grouped_matmul`` followed by ``fused_silu_quantize``,
        but folds both into a single SM100 kernel. The caller must pass a
        sigma-permuted ``weight`` and ``weight_scales`` (see
        ``max.nn.kernels.grouped_matmul_blocked_swiglu``); the layout is
        produced by :meth:`MoE.gate_up_proj` and
        :meth:`MoEQuantized.gate_up_proj_scales` when
        ``quant_config.can_use_fused_swiglu`` is set, or ships natively
        (gate, up)-interleaved in the checkpoint (the TP path gated by
        :meth:`MoEQuantized._can_fuse_swiglu_interleaved`).

        Args:
            weight: Sigma-permuted gate/up projection weights.
            weight_scales: Sigma-permuted gate/up projection scales (6D).
            expert_scales: Per-expert matmul-epilogue scaling factors.
                Typically ``nvfp4.gate_up_expert``.
            input_scales: Raw per-expert SiLU-output scale (``nvfp4.down_input``).
                The fused kernel consumes the inverted scale internally; this
                method inverts here to match the chained
                :meth:`fused_silu_quantize` convention.
            expert_inputs: Same tuple shape as :meth:`grouped_matmul`:
                ``(hidden, hidden_scales, expert_start, scales_offsets,
                expert_ids, usage_stats)``.
            estimated_total_m: Estimated total non-padded token count.
            use_swigluoai: Whether to use the OAI-style clamped SwiGLU activation
                function.
            swiglu_alpha: The alpha value for the clamped SwiGLU activation function.
            swiglu_limit: The limit value for the clamped SwiGLU activation function.

        Returns:
            Tuple ``(c_packed, c_swiglu_scales)`` matching the chained
            reference path byte-for-byte under the kernel's default
            ``match_bf16=True`` setting.
        """
        if self.is_nvfp4 and input_scales is None:
            raise ValueError("NVFP4 requires input_scales")
        if self.is_nvfp4 and expert_scales is None:
            raise ValueError("NVFP4 requires expert_scales")

        (
            hidden,
            hidden_scales,
            expert_start,
            scales_offsets,
            expert_ids,
            usage_stats,
        ) = expert_inputs

        # Replace gpu usage stats with a dummy cpu usage stats (same as
        # grouped_matmul above).
        if usage_stats.device.is_gpu():
            usage_stats = ops.constant(
                [8192, int(expert_ids.shape[0])],
                dtype=DType.uint32,
                device=DeviceRef.CPU(),
            )

        c_input_scales = (
            (1.0 / input_scales).to(hidden.device)
            if input_scales is not None
            else None
        )
        expert_scales = (
            expert_scales.to(hidden.device)
            if expert_scales is not None
            else None
        )

        return grouped_matmul_blocked_swiglu(
            hidden,
            weight,
            hidden_scales,
            weight_scales,
            expert_start,
            scales_offsets,
            expert_ids,
            usage_stats,
            expert_scales=expert_scales,
            c_input_scales=c_input_scales,
            estimated_total_m=estimated_total_m,
            clamp_activation=use_swigluoai,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )


class BlockScaledStrategy:
    """MXFP4 quantization for MoE.

    When `preshuffled_b=True`, the MOGG MXFP4 grouped-matmul op dispatches
    to the preshuffled-B kernel variant (`block_scaled_grouped_matmul_amd_preb`),
    which expects B in the 5D layout from `Shuffler.preshuffle_b_5d`. The
    caller is responsible for applying that preshuffle at weight load
    time (e.g. Kimi K2.5's `weight_adapters.py:_batch_preshuffle_experts`).
    Models without a preshuffle weight adapter must leave
    `preshuffled_b=False` so the dense row-major kernel is used.
    """

    def __init__(
        self,
        config: QuantConfig,
        dtype: DType,
        preshuffled_b: bool = False,
    ):
        self.config = config
        self.dtype = dtype
        self.preshuffled_b = preshuffled_b

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations to MXFP4 and returns (quantized, scales)."""
        if self.config.is_mxfp8:
            # Would silently emit MXFP4 otherwise. The MXFP8 path gets its
            # activations from EP dispatch or `fused_silu_quantize`; a
            # standalone MXFP8 activation quantize kernel does not exist yet.
            raise NotImplementedError(
                "standalone MXFP8 activation quantize is not implemented; "
                "MXFP8 activations come from EP dispatch or fused_silu_quantize"
            )
        return quantize_dynamic_block_scaled_mxfp4(
            tensor,
            scales_type=self.config.weight_scale.dtype,
            out_type=DType.uint8,
        )

    def grouped_quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None,
        expert_start: TensorValue,
        scales_offset: TensorValue,
        expert_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Falls back to ungrouped MXFP4 quantization."""
        return self.quantize(tensor, group_size)

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
        a_scales_preshuffled: bool = False,
        a_scales_max_padded_m: int = 0,
        decode_grid_m_cap: int = 0,
        decode_grid_m_rows: int = 0,
    ) -> TensorValue:
        """Runs grouped MXFP4 matmul with per-expert scales."""
        (
            hidden,
            hidden_scales,
            expert_start,
            expert_ids,
            usage_stats,
        ) = expert_inputs

        return grouped_dynamic_block_scaled_matmul_amd(
            hidden,
            weight,
            hidden_scales,
            weight_scales,
            expert_start,
            expert_ids,
            usage_stats.to(DeviceRef.CPU()),
            estimated_total_m=estimated_total_m,
            preshuffled_b=self.preshuffled_b,
            a_scales_preshuffled=a_scales_preshuffled,
            a_scales_max_padded_m=a_scales_max_padded_m,
            decode_grid_m_cap=decode_grid_m_cap,
            decode_grid_m_rows=decode_grid_m_rows,
        )

    def prepare_weight_scales(
        self,
        gate_up: TensorValue,
        down: TensorValue,
        device: DeviceRef,
    ) -> tuple[TensorValue, TensorValue]:
        return gate_up, down

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        max_padded_M: int = 0,
        clamp_activation: bool = False,
        swiglu_alpha: float = 0.0,
        swiglu_limit: float = 0.0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies SiLU (or clamped SwiGLU-OAI) gate then MXFP4 quantizes."""
        _, _, expert_start_indices, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
            input_scales,
            max_padded_M=max_padded_M,
            clamp_activation=clamp_activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )


class Mxfp6Strategy:
    """MXFP6 quantization for MoE (A6W6).

    Preshuffled-B only: an FP6 lane fragment is 24 bytes, which the kernel
    reads plane-split, and the dense row-major grouped kernel has no path for
    that layout. The weight loader must apply
    ``preshuffle_mxfp4_b_experts(..., lane_bytes=MXFP6_LANE_BYTES)``.

    Unlike :class:`Mxfp4Strategy` there is no fused activation kernel, so the
    down-projection input is produced as bf16 SwiGLU followed by a standalone
    MXFP6 quantize. That is a real extra pass over the activations; it is
    correct, and the fusion is a performance item, not a correctness one.
    """

    def __init__(self, config: QuantConfig, dtype: DType) -> None:
        self.config = config
        self.dtype = dtype
        self.fp6_format = config.mxfp6_format

    def quantize(
        self,
        tensor: TensorValue,
        group_size: int,
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes activations to MXFP6 and returns (quantized, scales)."""
        return quantize_dynamic_block_scaled_mxfp6(
            tensor,
            fp6_format=self.fp6_format,
            scales_type=self.config.weight_scale.dtype,
            out_type=DType.uint8,
        )

    def grouped_quantize(
        self,
        tensor: TensorValue,
        group_size: int,
        input_scale: TensorValue | None,
        expert_start: TensorValue,
        scales_offset: TensorValue,
        expert_ids: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Falls back to ungrouped MXFP6 quantization."""
        return self.quantize(tensor, group_size)

    def grouped_matmul(
        self,
        weight: TensorValue,
        weight_scales: TensorValue,
        expert_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        estimated_total_m: TensorValue | None = None,
        a_scales_preshuffled: bool = False,
        a_scales_max_padded_m: int = 0,
        decode_grid_m_cap: int = 0,
        decode_grid_m_rows: int = 0,
    ) -> TensorValue:
        """Runs the grouped MXFP6 matmul with per-expert scales."""
        if a_scales_preshuffled and a_scales_max_padded_m <= 0:
            raise ValueError(
                "a_scales_max_padded_m must be > 0 when a_scales_preshuffled"
                " is set for MXFP6"
            )

        (
            hidden,
            hidden_scales,
            expert_start,
            expert_ids,
            usage_stats,
        ) = expert_inputs

        return grouped_dynamic_scaled_mxfp6_matmul(
            hidden,
            weight,
            hidden_scales,
            weight_scales,
            expert_start,
            expert_ids,
            usage_stats.to(DeviceRef.CPU()),
            fp6_format=self.fp6_format,
            estimated_total_m=estimated_total_m,
            decode_grid_m_cap=decode_grid_m_cap,
            decode_grid_m_rows=decode_grid_m_rows,
            a_scales_preshuffled=a_scales_preshuffled,
            a_scales_max_padded_m=a_scales_max_padded_m,
        )

    def prepare_weight_scales(
        self,
        gate_up: TensorValue,
        down: TensorValue,
        device: DeviceRef,
    ) -> tuple[TensorValue, TensorValue]:
        return gate_up, down

    def fused_silu_quantize(
        self,
        gate_up_projs: TensorValue,
        input_scales: TensorValue | None = None,
        expert_inputs: tuple[TensorValue, ...] = (),
        max_padded_M: int = 0,
        clamp_activation: bool = False,
        swiglu_alpha: float = 0.0,
        swiglu_limit: float = 0.0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies SiLU gating and MXFP6-quantizes, in one fused kernel.

        Emits ``ep.fused_silu.mxfp6`` (``fused_silu_mxfp6_kernel``), matching
        how MXFP4 and MXFP8 drive this step. It replaces a bf16 SwiGLU built
        from graph primitives plus a standalone quantize -- an IR diff against
        MXFP8 showed that costing an extra ``split``/``sigmoid``/``mul``/``max``
        per MoE layer per device (456 of each at dp=2/ep=8) and an extra
        ``mo.quantize.dynamic.block.scaled.mxfp6``.

        ``max_padded_M`` > 0 enables the down-proj A-scale fold: the kernel
        writes the E8M0 scale straight into the grouped matmul's per-expert
        slot, dropping the standalone preshuffle. The slot layout is shared
        across MX formats (``scale_4d_slot_byte_off`` keys on ``K_SCALES`` and
        ``MXFP6_SF_VECTOR_SIZE`` is 32 like FP4/FP8), so the reader is
        unchanged. The up-proj fold stays MXFP4-only -- its producer is the EP
        dispatch path, which has no FP6 slot writer.
        """
        _, _, expert_start_indices, _, _ = expert_inputs
        return fused_silu_quantized(
            gate_up_projs,
            expert_start_indices,
            self.config,
            self.dtype,
            input_scales,
            max_padded_M=max_padded_M,
            clamp_activation=clamp_activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
        )


def _nv_interleave_block_scales(
    scales: TensorValue, device: DeviceRef
) -> TensorValue:
    """Interleaves rank-3 block scales for SM100 kernel consumption."""
    if scales.rank != 3:
        raise ValueError(
            f"expected block scales of rank 3 but got {scales.rank}"
        )
    # Only NVFP4 uses one FP8_E4M3FN scale per 16 elements,
    group_size = 16 if scales.dtype == DType.float8_e4m3fn else 32
    num_experts = int(scales.shape[0])
    scales = scales.to(device)
    scale_m = scales.shape[1]
    scale_k = scales.shape[2]
    expert_scales = ops.split(scales, [1] * num_experts, axis=0)
    return ops.stack(
        [
            block_scales_interleave(s.reshape([scale_m, scale_k]), group_size)
            for s in expert_scales
        ],
        axis=0,
    )
