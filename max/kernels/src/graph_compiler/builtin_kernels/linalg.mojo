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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

"""Registers matmul-family graph ops (matmul, grouped, batched, and quantized variants) and dispatches them to the `linalg` kernels."""

from std.collections import OptionalReg
from std.sys import align_of
from std.sys.info import simd_width_of, _accelerator_arch
from std.sys.info import (
    simd_width_of,
    _accelerator_arch,
    has_apple_gpu_accelerator,
)
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from max.gpu.host import DeviceContext, get_gpu_target
from layout.tile_tensor import row_major
from max.gpu.host.info import is_gpu, _is_sm10x_gpu
from layout import (
    Coord,
    Idx,
    IntTuple,
    DefaultEngine,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from linalg.bmm import batched_matmul, batched_matmul_shape
from linalg.bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)
from linalg.fp8_quantization import matmul_dynamic_scaled_fp8
from linalg.block_scaled_quantization import block_scaled_matmul
from linalg.arch.amd.block_scaled_mma import CDNA4F8F6F4MatrixFormat
from linalg.matmul.gpu.amd.block_scaled_matmul_amd import (
    mxfp6_block_scaled_matmul_amd,
)
from linalg.matmul.gpu.amd import (
    block_scaled_matmul_amd,
    block_scaled_grouped_matmul_amd,
    block_scaled_grouped_matmul_amd_preb,
)
from linalg.gemv import router_gate_mixed_gemv, router_gate_use_mixed_gemv
from linalg.matmul.gpu.amd.smallm_streaming_matmul import (
    smallm_streaming_matmul,
)
from linalg.mxfp4_matmul_sm90 import mxfp4_matmul_sm90
from linalg.matmul.gpu.apple.fp4_matmul import enqueue_apple_fp4_matmul
from linalg.matmul.gpu.apple.fp8_gemv import enqueue_apple_fp8_matmul
from linalg.matmul.gpu.apple.int8_matmul import (
    enqueue_apple_int8_matmul,
    enqueue_apple_int8_quantize_activation,
)
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_dynamic_scaled_fp8,
)
from linalg.grouped_matmul_block_scaled_dispatch import (
    grouped_matmul_block_scaled_dispatch,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled_swiglu_sm100_dispatch,
)
from linalg.matmul.gpu.sm100_structured.default.dispatch_fused_bias_residual import (
    fused_bias_residual_matmul_dispatch_sm100,
)
from linalg.matmul.gpu.sm100_structured.fused_swiglu import (
    matmul_swiglu_dispatch_sm100_bf16,
)
from linalg.bmm import batched_matmul_dynamic_scaled_fp8
from linalg.grouped_matmul import (
    grouped_matmul,
    grouped_matmul_rowwise_dynamic_scaled_fp8,
)
from linalg.lora import expand_qkv_sm100, shrink_qkv_permute_3mn_sm100
from linalg.matmul import matmul
from linalg.matmul.gpu import _matmul_gpu
from linalg.matrix_band_part import matrix_band_part
from linalg.packing import _pack_b_ndbuffer_impl, pack_matmul_b_shape_func
from linalg.utils import (
    elementwise_compute_lambda_type as matmul_elementwise_compute_lambda_type,
)
from linalg.utils import (
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)
from nn._ragged_utils import merge_ragged_tensors
from nn.gemv_partial_norm import gemv_and_partial_norm
from extensibility import InputTensor, OutputTensor
from extensibility import Tensor, TileTensorable
from extensibility import _FusedComputeOutputTensor
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from std.logger import Logger

comptime logger = Logger()

from max.algorithm.functional import elementwise
from std.utils import IndexList

# ===-----------------------------------------------------------------------===#
from .kernels import *


@extensibility.register("mo.composite.matmul_fused_partial_rms_norm")
struct MatmulFusedPartialRMSNorm:
    """Fuses GEMV (M=1 matmul) with partial RMS normalization.

    Computes y = x @ W.T, then applies RMS normalization to the first N_normed
    columns while passing the remaining columns through unchanged.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        transpose_b: Bool = True,
    ](
        normed_output: OutputTensor[dtype=dtype, rank=rank, ...],
        unnormed_output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Float32,
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        """Execute fused GEMV + partial RMS norm.

        Calls `gemv_and_partial_norm` from `nn.gemv_partial_norm` which
        computes y = x @ W.T, then partitions y into normed and
        unnormed outputs.

        Parameters:
            dtype: Element type of all input and output tensors.
            rank: Tensor rank of the input and output tensors.
            target: The target GPU device.
            transpose_b: Whether to transpose `weight` before the matmul
                (defaults to `True`).

        Args:
            normed_output: Output tensor holding the RMS-normalized
                leading columns of the matmul result.
            unnormed_output: Output tensor holding the trailing columns
                of the matmul result, passed through unchanged.
            input: Input activation tensor `x` of the GEMV.
            weight: Weight matrix `W` of shape `(N, K)` (rank 2).
            gamma: RMS normalization scale vector (rank 1).
            epsilon: Small constant added to the squared mean before
                the reciprocal square root for numerical stability.
            weight_offset: Reserved for API consistency with other RMS
                norm ops; not consumed by this kernel.
            ctx: The device context used to enqueue the kernel.
        """
        # weight_offset is passed but not used in this kernel - it's kept
        # for API consistency with other RMS norm ops.
        _ = weight_offset

        gemv_and_partial_norm[
            c_type=dtype,
            a_type=dtype,
            transpose_b=transpose_b,
            fused=True,
        ](
            normed_output.to_tile_tensor[.int64](),
            unnormed_output.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
            weight.to_tile_tensor[.int64](),
            gamma.to_tile_tensor[.int64](),
            epsilon,
            ctx,
        )


@extensibility.register_shape_function(
    "mo.composite.matmul_fused_partial_rms_norm"
)
def composite_matmul_fused_partial_rms_norm_shape[
    dtype: DType,
](
    input: Some[Tensor],
    weight: Some[Tensor],
    gamma: Some[Tensor],
    epsilon: Float32,
    weight_offset: Scalar[dtype=dtype],
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.composite.matmul_fused_partial_rms_norm` graph op.

    Args:
        input: Input activation tensor `x` of the GEMV.
        weight: Weight matrix `W` of shape `(N, K)` (rank 2).
        gamma: RMS normalization scale vector (rank 1).
        epsilon: Small constant added to the squared mean before the
            reciprocal square root for numerical stability.
        weight_offset: Reserved for API consistency with other RMS norm
            ops; not consumed by the shape function.

    Returns:
        The output shape, which matches the `input` shape.
    """
    comptime assert (
        type_of(weight).dtype == type_of(input).dtype
    ), "weight and input must share a dtype"
    comptime assert (
        type_of(gamma).dtype == type_of(input).dtype
    ), "gamma and input must share a dtype"
    comptime assert type_of(weight).rank == 2, "weight must be rank 2"
    comptime assert type_of(gamma).rank == 1, "gamma must be rank 1"
    comptime assert (
        dtype == type_of(input).dtype
    ), "weight_offset dtype must match input dtype"
    # Return the input shape for normed output
    # The actual shape split is handled by the op semantics
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape().tuple())
    )


@extensibility.register("mo.matmul")
struct Matmul:
    """Registers the `mo.matmul` graph op with the graph compiler."""

    @staticmethod
    def execute[
        transpose_b: Bool,
        packed_b: Bool,
        has_epilogue_fusion: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: _FusedComputeOutputTensor[rank=2, ...],
        a: InputTensor[rank=2, ...],
        b: InputTensor[rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert not (packed_b and transpose_b), (
            "transpose_b and b_packed cannot both be true because"
            " pre-packing transposes B"
        )

        comptime transposed_a = False

        @__parameter
        @always_inline
        def epilogue_fn[
            _dtype: DType, _width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]):
            c._lambda_store[width=_width, element_alignment=alignment](
                coords,
                rebind[SIMD[c.dtype, _width]](val),
            )

        @__parameter
        @always_inline
        def output_compute_fn[
            _dtype: DType, _width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]) -> SIMD[
            _dtype, _width
        ]:
            return rebind[SIMD[_dtype, _width]](
                c._fused_compute_output_lambda[element_alignment=alignment](
                    coords, rebind[SIMD[c.dtype, _width]](val)
                )
            )

        comptime has_compute_lambda = type_of(c)._has_compute_fusion

        comptime elementwise_lambda = Optional[
            matmul_elementwise_epilogue_type
        ](
            epilogue_fn
        ) if has_epilogue_fusion and not has_compute_lambda else None

        comptime compute_lambda = Optional[
            matmul_elementwise_compute_lambda_type
        ](
            output_compute_fn
        ) if has_epilogue_fusion and has_compute_lambda else None

        matmul[
            transposed_a,
            transpose_b,
            packed_b,
            elementwise_lambda,
            compute_lambda,
            target=target,
            _trace_description=_trace_name,
        ](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("mo.batch_matmul")
struct BatchMatmul:
    """Registers the `mo.batch_matmul` graph op with the graph compiler."""

    @staticmethod
    def execute[
        has_epilogue_fusion: Bool,
        rank: Int,
        transpose_b: Bool,
        target: StaticString,
    ](
        c: _FusedComputeOutputTensor[rank=rank, ...],
        a: InputTensor[rank=rank, ...],
        b: InputTensor[rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime transpose_a = False

        var a_tile = a.to_tile_tensor[.int64]()
        var b_tile = b.to_tile_tensor[.int64]()
        var c_tile = c.to_tile_tensor[.int64]()

        @__parameter
        @always_inline
        def output_fn[
            _type: DType, _width: SIMDLength, _rank: Int, *, alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_type, _width]):
            comptime has_compute_lambda = type_of(c)._has_compute_fusion

            comptime if has_compute_lambda:
                var output = c._fused_compute_output_lambda[
                    element_alignment=alignment
                ](
                    rebind[IndexList[c.rank]](coords),
                    rebind[SIMD[c.dtype, _width]](val),
                )
                c.store[element_alignment=alignment](
                    rebind[IndexList[c.rank]](coords), output
                )
            else:
                c._lambda_store[width=_width, element_alignment=alignment](
                    rebind[IndexList[c.rank]](coords),
                    rebind[SIMD[c.dtype, _width]](val),
                )

        batched_matmul[
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            elementwise_epilogue_fn=Optional[
                batched_matmul_elementwise_epilogue_type
            ](output_fn) if has_epilogue_fusion else None,
            target=target,
        ](c_tile, a_tile, b_tile, context=ctx)


@extensibility.register_shape_function("mo.batch_matmul")
def batch_matmul_shape(
    a: Some[TileTensorable], b: Some[TileTensorable]
) raises -> IndexList[type_of(a).rank]:
    """Computes the output shape for the `mo.batch_matmul` graph op.

    Args:
        a: Left-hand batched input tensor.
        b: Right-hand batched input tensor.

    Returns:
        The output shape of the batched matmul.
    """
    comptime assert (
        type_of(a).rank == type_of(b).rank
    ), "a and b must share a rank"
    return batched_matmul_shape[type_of(a).rank](
        a.to_tile_tensor(),
        b.to_tile_tensor(),
    )


@extensibility.register("mo.composite.matmul_add")
struct FusedMatmulAdd:
    """Registers the `mo.composite.matmul_add` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        transpose_b: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[rank=2, ...],
        a: InputTensor[rank=2, ...],
        b: InputTensor[rank=2, ...],
        residual: InputTensor[dtype=c.dtype, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            residual.rank == 1 or residual.rank == 2
        ), "residual must be rank 1 (bias) or rank 2"
        comptime epilogue_is_1d = residual.rank == 1
        var epi_m: Int64
        var epi_n: Int64
        comptime if epilogue_is_1d:
            epi_m = 1
            epi_n = Int64(residual.dim_size(0))
        else:
            epi_m = Int64(residual.dim_size(0))
            epi_n = Int64(residual.dim_size(1))
        var epilogue = TileTensor(
            residual.unsafe_ptr(), row_major(Coord(epi_m, epi_n))
        ).as_immut()

        fused_bias_residual_matmul_dispatch_sm100[
            transpose_b=transpose_b,
            has_epilogue_tensor=True,
            epilogue_is_1d=epilogue_is_1d,
        ](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            epilogue,
            ctx,
        )


@extensibility.register("mo.linalg.band_part")
struct LinalgBandPart:
    """Registers the `mo.linalg.band_part` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        int_type: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        num_lower: InputTensor[dtype=int_type, rank=1, ...],
        num_upper: InputTensor[dtype=int_type, rank=1, ...],
        exclude: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) {var input} -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        matrix_band_part[
            simd_width=simd_width_of[dtype](),
            target=target,
        ](
            input_fn,
            input.shape(),
            num_lower.to_tile_tensor[int_type](),
            num_upper.to_tile_tensor[int_type](),
            exclude.to_tile_tensor[.int64](),
            output.to_tile_tensor[dtype](),
            ctx,
        )


@extensibility.register("mo.grouped.matmul.ragged")
struct Struct_grouped_matmul_ragged:
    """Registers the `mo.grouped.matmul.ragged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        expert_usage_stats: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "grouped matmul only support GPUs"
        grouped_matmul(
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            expert_start_indices.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            expert_usage_stats.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.composite.grouped_matmul_block_scaled")
struct Struct_grouped_matmul_block_scaled:
    """MOGG wrapper for grouped block-scaled matrix multiplication.

    Provides graph compiler integration for block-scaled grouped matmul
    operations used in Mixture of Experts (MoE) layers on SM100 GPUs.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        scales_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=scales_type, rank=5, ...],
        b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        a_scale_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        expert_scales: InputTensor[dtype=.float32, rank=1, ...],
        estimated_total_m: UInt32,
        num_active_experts: UInt32,
        context: DeviceContext,
    ) raises:
        """Executes grouped block-scaled matrix multiplication.

        Computes C = A @ B^T for multiple expert groups where A and B are
        block-scaled. `uint8` operands are nibble-packed 4-bit E2M1, so their
        rows are `K // 2` wide; `float8_e4m3fn` operands are unpacked. The
        operand and scale dtypes together select the UMMA kind (see
        `block_scaled_umma_kind`): NVFP4, MXFP4, MXFP8, or the mixed W4A8 pair.

        Parameters:
            c_type: The output tensor data type.
            a_type: The input A data type.
                Constraints: Must be `uint8` (NVFP4/MXFP4) or `float8_e4m3fn`
                (MXFP8/W4A8).
            b_type: The input B data type.
                Constraints: Must equal `a_type`, except for W4A8, which pairs
                a `float8_e4m3fn` A with a `uint8` B.
            scales_type: The scale factor data type.
                Constraints: Must be `float8_e4m3fn` (NVFP4) or
                `float8_e8m0fnu` (MXFP4/MXFP8/W4A8).
            target: The target GPU device.

        Args:
            c: The output tensor of shape (total_tokens, N).
            a: The input tensor of shape (total_tokens, K) for `float8_e4m3fn`
                or (total_tokens, K // 2) for packed `uint8`.
            b: The weight tensor of shape (num_experts, N, K) for
                `float8_e4m3fn` or (num_experts, N, K // 2) for packed `uint8`.
            a_scales: The A scale factors in tcgen05 5D layout.
            b_scales: The B scale factors in tcgen05 6D layout.
            expert_start_indices: The starting token index for each expert.
            expert_ids: The expert ID for each group.
            a_scale_offsets: The starting scale index for each expert.
            expert_scales: The per-expert scaling factors for the epilogue.
            estimated_total_m: The estimated total number of tokens.
            num_active_experts: The number of active experts.
            context: The device context pointer.
        """
        comptime assert is_gpu[
            target
        ](), "grouped block-scaled matmul only supports GPUs"
        if num_active_experts == 0:
            return
        grouped_matmul_block_scaled_dispatch[transpose_b=True, target=target](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            expert_start_indices.to_tile_tensor[.int64](),
            a_scale_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            expert_scales.to_tile_tensor[.int64](),
            Int(num_active_experts),
            Int(estimated_total_m),
            context,
        )


@extensibility.register("mo.composite.grouped_matmul_swiglu_nvfp4")
struct Struct_grouped_matmul_swiglu_nvfp4:
    """MOGG wrapper for fused grouped NVFP4 matmul + SwiGLU + NVFP4 quant.

    Fuses the MoE gate/up grouped matmul, SwiGLU activation, and per-block
    NVFP4 quantization into a single SM100 kernel. The caller must pre-permute
    the weight `b` and its scale tile `b_scales` on the N axis with
    `sigma(2i)=i, sigma(2i+1)=D+i` (where `D = moe_dim`, `N = 2D`).
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        scales_type: DType,
        //,
        clamp_activation: Bool,
        target: StaticString,
    ](
        c_packed: OutputTensor[dtype=c_type, rank=2, ...],
        c_swiglu_scales: OutputTensor[dtype=scales_type, rank=5, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=scales_type, rank=5, ...],
        b_scales: InputTensor[dtype=scales_type, rank=6, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        a_scale_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        expert_scales: InputTensor[dtype=.float32, rank=1, ...],
        c_input_scales: InputTensor[dtype=.float32, rank=1, ...],
        estimated_total_m: UInt32,
        num_active_experts: UInt32,
        swiglu_alpha: Float32,
        swiglu_limit: Float32,
        context: DeviceContext,
    ) raises:
        """Executes fused grouped NVFP4 matmul + SwiGLU + NVFP4 quant.

        Computes `(c_packed, c_swiglu_scales) =
        quantize_nvfp4(silu(C[..., even]) * C[..., odd], c_input_scales)`
        where `C = A @ B^T` for multiple expert groups. Because `B` is
        sigma-permuted on N, adjacent matmul-output columns carry
        `(gate, up)` pairs that the epilogue consumes in-place.

        Parameters:
            c_type: The output tensor data type.
            a_type: The input A data type. Constraints: Must be `uint8`.
            b_type: The input B data type. Constraints: Must be `uint8`.
            scales_type: The scale factor data type.
                Constraints: Must be `float8_e4m3fn`.
            clamp_activation: Whether to clamp the activation (swigluoai).
            target: The target GPU device.

        Args:
            c_packed: Packed NVFP4 output of shape (total_tokens, D // 2).
            c_swiglu_scales: 5D FP8 SF tile in tcgen05 layout for the output.
            a: The input tensor of shape (total_tokens, K // 2).
            b: The sigma-permuted weight of shape (num_experts, 2D, K // 2).
            a_scales: The A scale factors in tcgen05 5D layout.
            b_scales: The sigma-permuted B scale factors in tcgen05 6D layout.
            expert_start_indices: The starting token index for each expert.
            expert_ids: The expert ID for each group.
            a_scale_offsets: The starting scale index for each expert.
            expert_scales: The per-expert scaling factors for the epilogue.
            c_input_scales: Per-expert SiLU input scale (= 1/output_inv_scale).
            estimated_total_m: The estimated total number of tokens.
            num_active_experts: The number of active experts.
            swiglu_alpha: The alpha value for the clamped activation.
            swiglu_limit: The limit value for the clamped activation.
            context: The device context pointer.
        """
        comptime assert is_gpu[
            target
        ](), "fused SwiGLU+NVFP4 grouped matmul only supports GPUs"
        if num_active_experts == 0:
            return
        grouped_matmul_block_scaled_swiglu_sm100_dispatch[
            transpose_b=True, target=target, clamp_activation=clamp_activation
        ](
            c_packed.to_tile_tensor[.int64](),
            c_swiglu_scales.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            expert_start_indices.to_tile_tensor[.int64](),
            a_scale_offsets.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            expert_scales.to_tile_tensor[.int64](),
            c_input_scales.to_tile_tensor[.int64](),
            Int(num_active_experts),
            Int(estimated_total_m),
            context,
            swiglu_alpha,
            swiglu_limit,
        )


@extensibility.register("mo.grouped.matmul.dynamic.scaled.fp8")
struct Struct_grouped_matmul_dynamic_scaled_fp8:
    """Registers the `mo.grouped.matmul.dynamic.scaled.fp8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        a_scales_type: DType,
        b_scales_type: DType,
        //,
        input_scale_granularity: StaticString,
        weight_scale_granularity: StaticString,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=a_scales_type, rank=2, ...],
        b_scales: InputTensor[dtype=b_scales_type, rank=3, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "grouped dynamic scaled matmul only support GPUs with native"
            " FP8 support"
        )
        grouped_matmul_dynamic_scaled_fp8[
            input_scale_granularity,
            weight_scale_granularity,
            m_scale_granularity,
            n_scale_granularity,
            k_scale_granularity,
            transpose_b=True,
            target=target,
        ](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            expert_start_indices.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            Int(max_num_tokens_per_expert),
            Int(num_active_experts),
            context,
        )


@extensibility.register("mo.grouped.matmul.rowwise.dynamic.scaled.fp8")
struct Struct_grouped_matmul_rowwise_dynamic_scaled_fp8:
    """MOGG wrapper for grouped (ragged MoE) rowwise/per-token scaled FP8 matmul.

    Serves rowwise (per-output-channel) weight scales + per-token (colwise)
    dynamic activation scales - the compressed-tensors FP8 layout used by
    e.g. ``RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic``. Targets
    NVIDIA SM100 (B200) with a correctness-first naive grouped kernel.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        a_scales_type: DType,
        b_scales_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=a_scales_type, rank=2, ...],
        b_scales: InputTensor[dtype=b_scales_type, rank=3, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "grouped rowwise dynamic scaled matmul only supports GPUs with"
            " native FP8 support"
        )
        var cuda_ctx = context
        grouped_matmul_rowwise_dynamic_scaled_fp8[
            transpose_b=True,
            target=target,
        ](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            expert_start_indices.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            Int(max_num_tokens_per_expert),
            Int(num_active_experts),
            cuda_ctx,
        )


@extensibility.register("mo.grouped.matmul.block.scaled.mxfp6")
struct Struct_grouped_matmul_block_scaled_mxfp6[FP6_FORMAT: Int = 0]:
    """MOGG wrapper for the grouped MXFP6 block-scaled matmul.

    The FP6 sibling of `mo.grouped.matmul.block.scaled.mxfp4`, and a separate
    op for the same reason the dense one is: both FP6 encodings put 24 bytes in
    a lane, so `lane_bytes` cannot choose between them.

    Preshuffled-B only. FP6's 24-byte lane fragment is plane-split (see
    `Shuffler.b_plane_byte_off`), which the dense row-major
    `block_scaled_grouped_matmul_amd` kernel has no path for.

    Parameters:
        FP6_FORMAT: 0 selects E2M3, 1 selects E3M2, matching `FP6Format`.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        b_scales: InputTensor[dtype=.float8_e8m0fnu, rank=3, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        estimated_total_m: UInt32,
        decode_grid_m_cap: UInt32,
        decode_grid_m_rows: UInt32,
        context: DeviceContext,
    ) raises:
        """Computes C = A @ B^T over expert groups with MXFP6 operands.

        Parameters:
            c_type: The output tensor data type.
            a_type: The packed activation data type; must be one byte wide.
            b_type: The packed weight data type; must be one byte wide.
            target: The target GPU device.

        Args:
            c: The output tensor of shape (total_tokens, N).
            a: The packed activations, (total_tokens, K * 3 // 4).
            b: The plane-split preshuffled weights, (num_experts, N, K * 3//4).
            a_scales: The A scale factors, (total_tokens, K // 32).
            b_scales: The B scale factors, (num_experts, N, K // 32).
            expert_start_indices: The starting token index for each expert.
            expert_ids: The expert ID for each group.
            max_num_tokens_per_expert: The maximum token count for any expert.
            num_active_experts: The number of active experts.
            estimated_total_m: Estimated total received tokens for this GPU,
                used to pick the persistent vs direct kernel path.
            decode_grid_m_cap: Decode-band gate selecting the direct kernel
                over the persistent one; 0 disables.
            decode_grid_m_rows: Rows grid.y must cover per expert on the
                decode bands.
            context: The device context pointer.
        """
        comptime assert is_gpu[
            target
        ](), "grouped block-scaled matmul only supports GPUs"
        comptime assert (
            size_of[a_type]() == 1 and size_of[b_type]() == 1
        ), "MXFP6 operands are packed into uint8; four codes per three bytes"
        comptime assert Self.FP6_FORMAT in (
            0,
            1,
        ), "FP6_FORMAT must be 0 (E2M3) or 1 (E3M2)"
        if num_active_experts == 0:
            return

        block_scaled_grouped_matmul_amd_preb[
            lane_bytes=24, fp6_format=Self.FP6_FORMAT
        ](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64]().bitcast[.uint8](),
            b.to_tile_tensor[.int64]().bitcast[.uint8](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            expert_start_indices.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            Int(max_num_tokens_per_expert),
            Int(num_active_experts),
            context,
            Int(estimated_total_m),
            -1 if decode_grid_m_cap == 0 else Int(decode_grid_m_cap),
            Int(decode_grid_m_rows),
        )


@extensibility.register("mo.grouped.matmul.block.scaled.amd")
struct Struct_grouped_matmul_block_scaled_amd[
    preshuffled_b: Bool = False, lane_bytes: Int = 16
]:
    """MOGG wrapper for grouped block-scaled matrix multiplication.

    Provides graph compiler integration for block-scaled grouped matmul
    operations used in Mixture of Experts (MoE) layers on AMD GPUs.

    Parameters:
        preshuffled_b: When True, dispatches to `block_scaled_grouped_matmul_amd_preb`
            which expects B in the 5D preshuffled layout from
            `Shuffler.preshuffle_b_5d` (typically produced by the model's
            weight adapter at load time, e.g. Kimi K2.5). When False
            (default), dispatches to the dense `block_scaled_grouped_matmul_amd`
            kernel that reads B row-major. The caller is responsible for
            preparing B in the matching layout.
        lane_bytes: Element packing of A and B — 16 for MXFP4 (default) or 32
            for MXFP8. The kernel reads `a`/`b` as raw bytes, so this rather
            than the operand dtype selects the format, and with it the K extent
            (`K` at MXFP8, `K // 2` at MXFP4). Preshuffled-B path only.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        b_scales: InputTensor[dtype=.float8_e8m0fnu, rank=3, ...],
        expert_start_indices: InputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: InputTensor[dtype=.int32, rank=1, ...],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        estimated_total_m: UInt32,
        decode_grid_m_cap: UInt32,
        decode_grid_m_rows: UInt32,
        context: DeviceContext,
    ) raises:
        """Executes grouped block-scaled matrix multiplication.

        Computes C = A @ B^T for multiple expert groups where A and B are
        block-scaled (e.g. MXFP4: 4-bit floating point packed as uint8).

        Parameters:
            c_type: The output tensor data type.
            target: The target GPU device.

        Args:
            c: The output tensor of shape (total_tokens, N).
            a: The input tensor of shape (total_tokens, K // 2).
            b: The weight tensor of shape (num_experts, N, K // 2).
            a_scales: The A scale factors in 2D layout.
            b_scales: The B scale factors in 3D layout.
            expert_start_indices: The starting token index for each expert.
            expert_ids: The expert ID for each group.
            max_num_tokens_per_expert: The maximum token count for any expert.
            num_active_experts: The number of active experts.
            estimated_total_m: Estimated total received tokens for this GPU,
                used by the preb dispatcher to pick the persistent vs direct
                kernel path. Pass 0 to default to persistent. Ignored when
                `preshuffled_b == False`.
            decode_grid_m_cap: Decode-band gate selecting the direct kernel over
                the persistent one; 0 disables. Ignored unless `preshuffled_b`.
            decode_grid_m_rows: Rows grid.y must cover per expert on the
                decode bands. Ignored unless `preshuffled_b`.
            context: The device context pointer.
        """
        comptime assert is_gpu[
            target
        ](), "grouped block-scaled matmul only supports GPUs"
        # The kernel takes raw byte operands and derives the format from
        # `lane_bytes`, so view either dtype as `uint8`: both are one byte
        # wide, leaving the layout untouched.
        comptime assert size_of[a_type]() == 1 and size_of[b_type]() == 1, (
            "grouped block-scaled matmul operands must be one byte wide"
            " (uint8 for MXFP4, float8_e4m3fn for MXFP8)"
        )
        if num_active_experts == 0:
            return
        comptime if Self.preshuffled_b:
            # Preshuffled-B kernel path (block_scaled_grouped_matmul_amd_preb).
            # Requires B in the 5D layout from `Shuffler.preshuffle_b_5d`,
            # typically produced by the model's weight adapter at load
            # time (e.g. kimik2_5/weight_adapters.py). Correctness
            # requires EP-MoE sharding (axis-0); TP-MoE is unsupported.
            block_scaled_grouped_matmul_amd_preb[lane_bytes=Self.lane_bytes](
                c.to_tile_tensor[.int64](),
                a.to_tile_tensor[.int64]().bitcast[.uint8](),
                b.to_tile_tensor[.int64]().bitcast[.uint8](),
                a_scales.to_tile_tensor[.int64](),
                b_scales.to_tile_tensor[.int64](),
                expert_start_indices.to_tile_tensor[.int64](),
                expert_ids.to_tile_tensor[.int64](),
                Int(max_num_tokens_per_expert),
                Int(num_active_experts),
                context,
                Int(estimated_total_m),
                -1 if decode_grid_m_cap == 0 else Int(decode_grid_m_cap),
                Int(decode_grid_m_rows),
            )
        else:
            # Dense row-major B path. Safe default for arbitrary callers.
            # MXFP8 is wired on the preshuffled-B path only; reject rather
            # than silently reinterpret the K extent as FP4-packed.
            comptime assert Self.lane_bytes == 16, (
                "lane_bytes=32 (MXFP8) requires preshuffled_b=True; the dense"
                " row-major B path is MXFP4-only"
            )
            block_scaled_grouped_matmul_amd(
                c.to_tile_tensor[.int64](),
                a.to_tile_tensor[.int64]().bitcast[.uint8](),
                b.to_tile_tensor[.int64]().bitcast[.uint8](),
                a_scales.to_tile_tensor[.int64](),
                b_scales.to_tile_tensor[.int64](),
                expert_start_indices.to_tile_tensor[.int64](),
                expert_ids.to_tile_tensor[.int64](),
                Int(max_num_tokens_per_expert),
                Int(num_active_experts),
                context,
            )


@extensibility.register("mo.batched.matmul.dynamic.scaled.fp8")
struct Struct_batched_matmul_dynamic_scaled_fp8:
    """Registers the `mo.batched.matmul.dynamic.scaled.fp8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        a_scales_type: DType,
        b_scales_type: DType,
        //,
        input_scale_granularity: StaticString,
        weight_scale_granularity: StaticString,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=3, ...],
        a: InputTensor[dtype=a_type, rank=3, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        a_scales: InputTensor[dtype=a_scales_type, rank=3, ...],
        b_scales: InputTensor[dtype=b_scales_type, rank=3, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "batched dynamic scaled matmul only support GPUs with native"
            " FP8 support"
        )

        if a.dim_size(1) == 0:
            return
        batched_matmul_dynamic_scaled_fp8[
            input_scale_granularity,
            weight_scale_granularity,
            m_scale_granularity,
            n_scale_granularity,
            k_scale_granularity,
            transpose_b=True,
            target=target,
        ](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.matmul.dynamic.block.scaled")
struct Struct_matmul_dynamic_block_scaled:
    """Registers the `mo.matmul.dynamic.block.scaled` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        scales_type: DType,
        //,
        has_epilogue_fusion: Bool,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        c: _FusedComputeOutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=2, ...],
        a_scales: InputTensor[dtype=scales_type, rank=5, ...],
        b_scales: InputTensor[dtype=scales_type, rank=5, ...],
        tensor_sf: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "dynamic block scaled matmul only support GPUs with native"
            " block scaled support"
        )

        # The SM100 block-scaled matmul applies the epilogue on the f32
        # accumulator (`_dtype` may be f32), whereas the fusion lambdas operate
        # on the logical output dtype `c.dtype`. Use `cast` rather than `rebind`
        # so the conversion is correct on both the structured Mojo path (f32
        # accumulator) and the vendor path (`_dtype == c.dtype`).
        @__parameter
        @always_inline
        def epilogue_fn[
            _dtype: DType, _width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]):
            c._lambda_store[width=_width, element_alignment=alignment](
                coords,
                val.cast[c.dtype](),
            )

        @__parameter
        @always_inline
        def output_compute_fn[
            _dtype: DType, _width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]) -> SIMD[
            _dtype, _width
        ]:
            return c._fused_compute_output_lambda[element_alignment=alignment](
                coords, val.cast[c.dtype]()
            ).cast[_dtype]()

        comptime has_compute_lambda = type_of(c)._has_compute_fusion

        comptime elementwise_lambda = Optional[
            matmul_elementwise_epilogue_type
        ](
            epilogue_fn
        ) if has_epilogue_fusion and not has_compute_lambda else None

        comptime compute_lambda = Optional[
            matmul_elementwise_compute_lambda_type
        ](
            output_compute_fn
        ) if has_epilogue_fusion and has_compute_lambda else None

        block_scaled_matmul[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=True,
            elementwise_lambda_fn=elementwise_lambda,
            elementwise_compute_lambda_fn=compute_lambda,
            target=target,
        ](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            tensor_sf,
            context,
        )


@extensibility.register("mo.matmul.dynamic.block.scaled.amd")
struct Struct_matmul_dynamic_block_scaled_amd[lane_bytes: Int = 16]:
    """Registers the `mo.matmul.dynamic.block.scaled.amd` graph op with the graph compiler.

    Parameters:
        lane_bytes: Operand bytes per lane per MFMA — 16 for MXFP4 (default) or
            32 for MXFP8. The kernel reads `a`/`b` as raw bytes, so this rather
            than the operand dtype selects the format, and with it the K extent
            (`K` at MXFP8, `K // 2` at MXFP4).
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=2, ...],
        a_scales: InputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        b_scales: InputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "dynamic block scaled matmul only support GPUs with native"
            " block scaled support"
        )
        # Both dtypes are one byte wide, so the raw-byte view is layout-neutral.
        comptime assert size_of[a_type]() == 1 and size_of[b_type]() == 1, (
            "dynamic block-scaled matmul operands must be one byte wide"
            " (uint8 for MXFP4, float8_e4m3fn for MXFP8)"
        )

        block_scaled_matmul_amd[lane_bytes=Self.lane_bytes](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64]().bitcast[.uint8](),
            b.to_tile_tensor[.int64]().bitcast[.uint8](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.matmul.dynamic.block.scaled.mxfp6")
struct Struct_matmul_dynamic_block_scaled_mxfp6[
    FP6_FORMAT: Int = 0, preshuffled_b: Bool = False
]:
    """Registers the `mo.matmul.dynamic.block.scaled.mxfp6` graph op.

    Separate from the MXFP4/MXFP8 op rather than another `lane_bytes` value:
    both FP6 encodings put 24 bytes in a lane, so the byte count cannot choose
    between them.

    Parameters:
        FP6_FORMAT: 0 selects E2M3, 1 selects E3M2, matching `FP6Format`.
        preshuffled_b: When True, `b` and `b_scales` must already be in the
            plane-split / packed-scale layouts from
            `Shuffler.preshuffle_b_planes` / `preshuffle_scale_4d` (a
            one-time, load-time cost for the static weight; see
            `preshuffle_block_scaled_b_dense`). Ignored (falls back to the
            row-major dispatch) when `M` is within the decode range -- see
            `mxfp6_block_scaled_matmul_amd`.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=2, ...],
        a_scales: InputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        b_scales: InputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), (
            "MXFP6 block-scaled matmul requires a GPU with native block-scaled"
            " support"
        )
        comptime assert (
            size_of[a_type]() == 1 and size_of[b_type]() == 1
        ), "MXFP6 operands are packed into uint8; four elements per three bytes"
        comptime assert Self.FP6_FORMAT in (
            0,
            1,
        ), "FP6_FORMAT must be 0 (E2M3) or 1 (E3M2)"
        comptime fmt = (
            CDNA4F8F6F4MatrixFormat.FLOAT6_E2M3 if Self.FP6_FORMAT
            == 0 else CDNA4F8F6F4MatrixFormat.FLOAT6_E3M2
        )

        mxfp6_block_scaled_matmul_amd[fmt, preshuffled_b=Self.preshuffled_b](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64]().bitcast[.uint8](),
            b.to_tile_tensor[.int64]().bitcast[.uint8](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.matmul.mxfp4.dequant.fp8")
struct Struct_matmul_mxfp4_dequant_fp8:
    """Registers the `mo.matmul.mxfp4.dequant.fp8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        b_scales_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=2, ...],
        b_scales: InputTensor[dtype=b_scales_type, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "MXFP4 dequant-to-FP8 matmul only supports GPUs"
        comptime assert (
            "sm_90" in _accelerator_arch()
        ), "MXFP4 dequant-to-FP8 matmul requires SM90"
        comptime assert (
            c_type == .bfloat16
        ), "MXFP4 matmul output must be bfloat16"
        comptime assert (
            a_type == .bfloat16
        ), "MXFP4 matmul activations must be bfloat16"
        comptime assert (
            b_type == .uint8
        ), "MXFP4 matmul weights must be uint8 (packed FP4)"
        comptime assert (
            b_scales_type == .float8_e8m0fnu
        ), "MXFP4 matmul scales must be float8_e8m0fnu"

        mxfp4_matmul_sm90(
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.matmul.weight.only.block.scaled.apple")
struct Struct_matmul_weight_only_block_scaled_apple:
    """Apple M5 weight-only NVFP4 (W4A16) matmul: `out = a @ dequant(b)^T`.

    Unlike `mo.matmul.dynamic.block.scaled` (the NVIDIA SM100 path), the
    activation `a` stays in bf16 (NOT dynamically quantized to FP4) and the
    weight block scales are PLAIN rank-2 `[N, K // 16]` (NOT the SM100 rank-5
    TCGEN05 interleave). The FP4 weight is dequantized to bf16 in-register at
    the MMA loader seam (`enqueue_apple_fp4_matmul`); weights stay packed in
    DRAM. The NVFP4 per-tensor `weight_scale_2` scalar is applied at the graph
    level by the caller (a post-matmul multiply), so it is NOT an input here.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.uint8, rank=2, ...],
        b_scales: InputTensor[dtype=.float8_e4m3fn, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "Apple weight-only block-scaled matmul only supports GPUs"
        comptime assert has_apple_gpu_accelerator(), (
            "mo.matmul.weight.only.block.scaled.apple requires an Apple"
            " (Metal) GPU accelerator"
        )

        enqueue_apple_fp4_matmul[c_type=c_type](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.matmul.weight.only.scaled.float8.apple")
struct Struct_matmul_weight_only_scaled_float8_apple:
    """Apple M5 weight-only FP8 (W8A16) matmul: `out = a @ dequant(b)^T`.

    The FP8 sibling of `mo.matmul.weight.only.block.scaled.apple`. The activation
    `a` stays in bf16 (NOT dynamically quantized to FP8), and the FP8-E4M3 weight
    `b` is widened to f32/bf16 at the point of consumption (register-resident in
    the `M == 1` GEMV, or a transient bf16 buffer for the `M > 1` interim). Unlike
    the NVFP4 sibling there is NO per-block weight scale (modelopt static FP8
    carries one per-tensor scalar `weight_scale`); that scalar is applied at the
    graph level by the caller as a post-matmul multiply (the FP8 analog of NVFP4's
    `weight_scale_2`), so it is NOT an input here. `input_scale` cancels for a
    bf16 activation, so it is not an input either.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.float8_e4m3fn, rank=2, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "Apple weight-only scaled FP8 matmul only supports GPUs"
        comptime assert has_apple_gpu_accelerator(), (
            "mo.matmul.weight.only.scaled.float8.apple requires an Apple"
            " (Metal) GPU accelerator"
        )

        enqueue_apple_fp8_matmul[c_type=c_type](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            context,
        )


@always_inline
def _apple_int8_w8a8_dispatch[
    c_type: DType, has_bias: Bool, target: StaticString
](
    c_tt: TileTensor[mut=True, c_type, ...],
    a_tt: TileTensor[.bfloat16, ...],
    b_tt: TileTensor[.int8, Engine=DefaultEngine[], ...],
    bs_tt: TileTensor[.float32, ...],
    bias_tt: TileTensor[c_type, ...],
    context: DeviceContext,
) raises:
    """Fused int8 W8A8: online-quant A -> int8 widening-MMA GEMM -> dequant.

    Shared body for the bias / no-bias op registrations. The bf16 activation is
    dynamically quantized to int8 (symmetric per-token absmax/127) into KERNEL
    SCRATCH (not graph values -- fusing avoids the per-Linear dispatch overhead
    that made the FP4 path slower than bf16 on FLUX.2-Klein), matmul'd against
    the pre-quantized int8 weight on the M5 int8 widening-MMA datapath (int32
    accumulate), then dequantized by the per-token activation scale times the
    per-output-channel weight scale (+ bias iff `has_bias`). Both enqueue fns are
    tested at the Klein shapes in `test_apple_int8_matmul.mojo`.
    """
    comptime assert is_gpu[
        target
    ](), "Apple int8 W8A8 matmul only supports GPUs"
    comptime assert (
        has_apple_gpu_accelerator()
    ), "mo.matmul.int8.w8a8.apple requires an Apple (Metal) GPU accelerator"

    var M = Int(a_tt.dim[0]())
    var K = Int(a_tt.dim[1]())

    # Kernel scratch: int8 quantized activation `[M, K]` + per-row fp32 scale
    # `[M]`, both dynamic-M/K. Freed at scope exit via the stream-ordered
    # `DeviceBuffer.__deinit__`; the `_ = ...^` pins them past the async enqueues
    # (same lifetime idiom as the FP4 materialize path).
    var aq_buf = context.enqueue_create_buffer[.int8](M * K)
    var asc_buf = context.enqueue_create_buffer[.float32](M)
    var aq_tt = TileTensor(
        aq_buf.unsafe_ptr(), row_major(Coord(Int64(M), Int64(K)))
    )
    var asc_tt = TileTensor(asc_buf.unsafe_ptr(), row_major(Coord(Int64(M))))

    enqueue_apple_int8_quantize_activation[.bfloat16](
        aq_tt, a_tt.as_immut(), asc_tt, context
    )

    enqueue_apple_int8_matmul[c_type=c_type, has_bias=has_bias](
        c_tt,
        aq_tt.as_immut(),
        b_tt,
        asc_tt.as_immut(),
        bs_tt,
        bias_tt,
        context,
    )

    _ = aq_buf^
    _ = asc_buf^


@extensibility.register("mo.matmul.int8.w8a8.apple")
struct Struct_matmul_int8_w8a8_apple:
    """Apple M5 int8 W8A8 matmul (no bias): `out = dequant(quant(a) @ b^T)`.

    A single FUSED graph op wrapping `int8_matmul.mojo` (online activation quant
    + int8 widening-MMA GEMM + dequant, all internal). Inputs (matching
    `nn/kernels.py::_apple_int8_w8a8_matmul` with `bias=None`): `a` bf16
    `[M, K]`; `b` int8 `[N, K]` (`transpose_b`); `b_scale` fp32 per-channel `[N]`
    (the Python squeezes the rowwise `[N, 1]`). Output `c_type` `[M, N]`. The
    bias variant is the sibling `mo.matmul.int8.w8a8.apple.bias` op (custom-op
    input arity is fixed per registration, so the optional bias is a separate op
    name -- the same idiom as `mo.fused_qkv_matmul.ragged.paged{,.bias}`).
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.int8, rank=2, ...],
        b_scale: InputTensor[dtype=.float32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var c_tt = c.to_tile_tensor[.int64]()
        # No bias: a length-1 dummy bias TileTensor (the `has_bias=False` GEMM
        # path ignores it). Reuse `b_scale` as the dummy source (same dtype is
        # not required -- it is never read -- but a valid 1-elem view is).
        var dummy_bias = TileTensor(
            c_tt._storage, row_major(Coord(Int64(1)))
        ).as_immut()
        _apple_int8_w8a8_dispatch[c_type, has_bias=False, target=target](
            c_tt,
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            b_scale.to_tile_tensor[.int64](),
            dummy_bias,
            context,
        )


@extensibility.register("mo.matmul.int8.w8a8.apple.bias")
struct Struct_matmul_int8_w8a8_apple_bias:
    """Apple M5 int8 W8A8 matmul WITH per-output-channel bias.

    The bias sibling of `mo.matmul.int8.w8a8.apple`: identical fused body plus a
    `bias` input in `c_type` `[N]` added after dequant. Selected by
    `nn/kernels.py::_apple_int8_w8a8_matmul` (which appends `.bias` to the op
    name when a bias is provided, mirroring the FP8 fused-QKV op).
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.int8, rank=2, ...],
        b_scale: InputTensor[dtype=.float32, rank=1, ...],
        bias: InputTensor[dtype=c_type, rank=1, ...],
        context: DeviceContext,
    ) raises:
        _apple_int8_w8a8_dispatch[c_type, has_bias=True, target=target](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            b_scale.to_tile_tensor[.int64](),
            bias.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("layout_transform_KN_to_KNkni")
struct LayoutTransformMatmulKN2KNkni:
    """Registers the `layout_transform_KN_to_KNkni` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        a_type: DType,
        a_shape: IntTuple,
        b_type: DType,
        b_shape: IntTuple,
        c_type: DType,
        c_shape: IntTuple,
    ](
        output_buffer: OutputTensor[dtype=b_type, rank=2, ...],
        b_input: InputTensor[dtype=b_type, rank=2, ...],
    ) raises:
        # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
        var kernel_type_m = 0

        comptime if a_shape[0] != UNKNOWN_VALUE:
            kernel_type_m = Int(a_shape[0])
        _pack_b_ndbuffer_impl[
            a_type=a_type,
            c_type=c_type,
            transposed=False,
        ](
            b_input.to_tile_tensor[.int64](),
            output_buffer.to_tile_tensor[.int64](),
            kernel_type_m,
        )


@extensibility.register("layout_transform_NK_to_KNkni")
struct LayoutTransformMatmulNK2KNkni:
    """Registers the `layout_transform_NK_to_KNkni` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        a_type: DType,
        a_shape: IntTuple,
        b_type: DType,
        b_shape: IntTuple,
        c_type: DType,
        c_shape: IntTuple,
    ](
        output_buffer: OutputTensor[dtype=b_type, rank=2, ...],
        b_input: InputTensor[dtype=b_type, rank=2, ...],
    ) raises:
        # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
        var kernel_type_m = 0

        comptime if a_shape[0] != UNKNOWN_VALUE:
            kernel_type_m = Int(a_shape[0])
        _pack_b_ndbuffer_impl[
            a_type=a_type,
            c_type=c_type,
            transposed=True,
        ](
            b_input.to_tile_tensor[.int64](),
            output_buffer.to_tile_tensor[.int64](),
            kernel_type_m,
        )


@extensibility.register("pack_matmul_b_shape_func")
struct PackMatmulBShapeFunc:
    """Registers the `pack_matmul_b_shape_func` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute(b_input: InputTensor) raises:
        raise Error("Only meant to be used for shape function!")


@extensibility.register_shape_function("pack_matmul_b_shape_func")
def pack_matmul_b_shape_func_shape[
    a_type: DType,
    a_shape: IntTuple,
    b_shape: IntTuple,
    c_type: DType,
    c_shape: IntTuple,
    transpose_in_0: Bool,
](b_input: Some[TileTensorable]) -> IndexList[2]:
    """Computes the output shape for the `pack_matmul_b_shape_func` graph op.

    Parameters:
        a_type: Element type of the A (activation) operand of the matmul
            the packed B will be used in.
        a_shape: Static shape of the A operand; `a_shape[0]` is the M
            dimension used to select the matmul kernel variant
            (`UNKNOWN_VALUE` for dynamic M).
        b_shape: Static shape of the B operand.
        c_type: Element type of the C (output) operand of the matmul
            the packed B will be used in.
        c_shape: Static shape of the C (output) operand.
        transpose_in_0: True if the B operand is transposed, stored as
            `[N, K]` instead of `[K, N]`.

    Args:
        b_input: Rank-2 B input tensor whose packed output shape is
            computed.

    Returns:
        The packed output shape for the B operand.
    """
    comptime assert type_of(b_input).rank == 2, "b_input must be rank 2"
    var kernel_type_m = 0
    comptime if a_shape[0] != UNKNOWN_VALUE:
        kernel_type_m = Int(a_shape[0])
    return pack_matmul_b_shape_func[
        a_type,
        c_type,
        transpose_in_0,
    ](b_input.to_tile_tensor().as_immut(), kernel_type_m)


@extensibility.register("mo.matmul_dynamic_scaled_fp8")
struct MatmulDynamicScaledFloat8:
    """Registers the `mo.matmul_dynamic_scaled_fp8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        input_type: DType,
        scales_type: DType,
        output_type: DType,
        //,
        input_scale_granularity: StaticString,
        weight_scale_granularity: StaticString,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        a: InputTensor[dtype=input_type, rank=2, ...],
        b: InputTensor[dtype=input_type, rank=2, ...],
        a_scales: InputTensor[dtype=scales_type, rank=2, ...],
        b_scales: InputTensor[dtype=scales_type, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        matmul_dynamic_scaled_fp8[
            input_scale_granularity,
            weight_scale_granularity,
            m_scale_granularity,
            n_scale_granularity,
            k_scale_granularity,
            transpose_b=True,
            target=target,
        ](
            output.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            a_scales.to_tile_tensor[.int64](),
            b_scales.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("mo.matmul_static_scaled_float8")
struct MatmulStaticScaledFloat8:
    """Registers the `mo.matmul_static_scaled_float8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        output_type: DType,
        input_dtype: DType,
        scale_type: DType,
        target: StaticString,
    ](
        output_tensor: OutputTensor[dtype=output_type, rank=2, ...],
        input_tensor: InputTensor[dtype=input_dtype, rank=2, ...],
        weight_tensor: InputTensor[dtype=input_dtype, rank=2, ...],
        input_scale: Scalar[scale_type],
        weight_scale: Scalar[scale_type],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        var output_tt = output_tensor.to_tile_tensor[.int64]()
        var input_tt = input_tensor.to_tile_tensor[.int64]()
        var weight_tt = weight_tensor.to_tile_tensor[.int64]()

        comptime if _is_sm10x_gpu(ctx.default_device_info):
            # Fold the per-tensor scales into the SM100 compute epilogue and
            # write `output_type` (bf16) directly into the real output. This
            # presents c_type==bf16 + a_type==float8_e4m3fn to
            # `matmul_dispatch_sm100`, which routes the static-scaled FP8 GEMM
            # through the tuned tcgen05 Mojo SM100 FP8 pipeline
            # (`matmul_dispatch_sm100_fp8`) instead of DISPATCH_MISSing on the
            # fp32 accumulator dtype and falling back to vendor cuBLASLt for
            # all m>1. The compute lambda returns the value already cast to the
            # output dtype, as required by `_matmul_gpu`'s compute-lambda
            # wrapper (output.dtype must equal c_type).
            @__parameter
            @__copy_capture(input_scale, weight_scale)
            @always_inline
            def scaled_compute_fn[
                dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[dtype, width]](),
            ](idx: IndexList[2], val: SIMD[dtype, width]) capturing -> SIMD[
                dtype, width
            ]:
                var scale = (
                    input_scale.cast[.float32]() * weight_scale.cast[.float32]()
                )
                var scaled_val = val.cast[.float32]() * scale
                return scaled_val.cast[dtype]()

            matmul[
                target=target,
                transpose_b=True,
                elementwise_compute_lambda_fn=scaled_compute_fn,
            ](
                output_tt,
                input_tt,
                weight_tt,
                Optional(ctx),
            )
        else:

            @__parameter
            @__copy_capture(output_tt, input_scale, weight_scale)
            @always_inline
            def scaled_output_fn[
                dtype: DType, width: SIMDLength, *, alignment: Int = 1
            ](idx: IndexList[2], val: SIMD[dtype, width]):
                var scale = (
                    input_scale.cast[dtype]() * weight_scale.cast[dtype]()
                )
                var scaled_val = val * scale

                output_tt.store_linear[width=width, alignment=alignment](
                    idx, scaled_val.cast[output_type]()
                )

            # Allocate an fp32 scratch buffer for the matmul accumulator;
            # the epilogue lambda reads from it, applies scaling, and writes
            # the quantized result into the real output.
            comptime N = type_of(weight_tt).static_shape[0]
            var M = Int(input_tt.dim[0]())
            var device_ctx = ctx
            var scratch_buffer = device_ctx.enqueue_create_buffer[.float32](
                M * N
            )
            var output_scratch = TileTensor(
                scratch_buffer.unsafe_ptr(),
                row_major(Coord(Int64(M), Idx[N])),
            )

            matmul[
                target=target,
                transpose_b=True,
                elementwise_lambda_fn=scaled_output_fn,
            ](
                output_scratch,
                input_tt,
                weight_tt,
                Optional(device_ctx),
            )


@extensibility.register("mo.merge_ragged_tensors")
struct MergeRaggedTensors:
    """Registers the `mo.merge_ragged_tensors` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        output_row_offsets: OutputTensor[dtype=.uint32, rank=1, ...],
        a: InputTensor[dtype=dtype, rank=rank, ...],
        a_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        b: InputTensor[dtype=dtype, rank=rank, ...],
        b_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        merge_ragged_tensors[rank=rank, target=target](
            output.to_tile_tensor[.int64](),
            output_row_offsets.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            a_row_offsets.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            b_row_offsets.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("mo.lora_sgmv.ragged")
struct Struct_lora_sgmv_ragged:
    """Registers the `mo.lora_sgmv.ragged` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        lora_ids: InputTensor[dtype=.int32, rank=1, ...],
        max_seq_length: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "SGMV only supported on GPUs"

        if a.dim_size[0]() == 0:
            return

        grouped_matmul(
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            lora_ids.to_tile_tensor[.int64](),
            min(Int(max_seq_length), a.dim_size[0]()),
            lora_ids.dim_size[0](),
            context,
        )


@extensibility.register("mo.lora_sgmv.qkv_shrink.ragged")
struct Struct_lora_sgmv_qkv_shrink_ragged:
    """Registers the `mo.lora_sgmv.qkv_shrink.ragged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=3, ...],
        a: InputTensor[dtype=a_type, rank=2, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        lora_ids: InputTensor[dtype=.int32, rank=1, ...],
        max_seq_length: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "SGMV only supported on GPUs"

        if a.dim_size[0]() == 0:
            return

        shrink_qkv_permute_3mn_sm100(
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            lora_ids.to_tile_tensor[.int64](),
            min(Int(max_seq_length), a.dim_size[0]()),
            lora_ids.dim_size[0](),
            context,
        )


@extensibility.register("mo.matmul_swiglu", type="gpu")
struct MatmulSwiGLU:
    """Fused GEMM+SwiGLU on SM100 for BF16 inputs.

    Computes ``output[m, h] = silu(x @ W_gate[h, :]) * (x @ W_up[h, :])``
    in a single SM100 kernel. The weight ``b`` must be pre-permuted on its N
    axis so that gate/up column pairs are adjacent (sigma permutation:
    ``sigma(2i)=i, sigma(2i+1)=H+i`` where ``H=N/2``).

    Output shape is ``[M, H]`` where ``H = N/2``, saving the slice+silu+mul
    elementwise kernel entirely.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.bfloat16, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        matmul_swiglu_dispatch_sm100_bf16(
            output.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("mo.matmul_swiglu_bias", type="gpu")
struct MatmulSwiGLUBias:
    """Fused GEMM+SwiGLU+bias on SM100 for BF16 inputs.

    Like ``mo.matmul_swiglu`` but adds a 1D bias vector before the activation.
    The bias must be sigma-permuted to match the weight layout: even element
    ``bias[2h]`` is added to the gate column and odd element ``bias[2h+1]`` to
    the up column before ``silu(gate) * up`` is computed.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.bfloat16, rank=2, ...],
        bias: InputTensor[dtype=.bfloat16, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        matmul_swiglu_dispatch_sm100_bf16[has_bias=True](
            output.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            ctx,
            OptionalReg(
                UnsafePointer[BFloat16, ImmutAnyOrigin](
                    unsafe_from_address=Int(bias.unsafe_ptr())
                )
            ),
        )


@extensibility.register("mo.lora_sgmv.qkv_expand.ragged")
struct Struct_lora_sgmv_qkv_expand_ragged:
    @always_inline
    @staticmethod
    def execute[
        q_type: DType,
        kv_type: DType,
        p_type: DType,
        b_type: DType,
        //,
        target: StaticString,
    ](
        q_out: OutputTensor[dtype=q_type, rank=2, ...],
        kv_out: OutputTensor[dtype=kv_type, rank=2, ...],
        p: InputTensor[dtype=p_type, rank=3, ...],
        b: InputTensor[dtype=b_type, rank=3, ...],
        lora_grouped_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        lora_ids: InputTensor[dtype=.int32, rank=1, ...],
        max_seq_length: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "SGMV only supported on GPUs"

        if p.dim_size[1]() == 0:
            return

        expand_qkv_sm100(
            q_out.to_tile_tensor[.int64](),
            kv_out.to_tile_tensor[.int64](),
            p.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            lora_grouped_offsets.to_tile_tensor[.int64](),
            lora_ids.to_tile_tensor[.int64](),
            min(Int(max_seq_length), p.dim_size[1]()),
            lora_ids.dim_size[0](),
            context,
        )


@extensibility.register("mo.router.gate.mixed.gemv")
struct Struct_router_gate_mixed_gemv:
    """MOGG wrapper for the mixed-input router-gate GEMV.

    Computes `c = a @ b^T` for a bf16 activation `a` and an fp32 router weight
    `b`, returning fp32 `c`. `M` is runtime-dynamic (one graph serves both
    decode and prefill), so the op branches on runtime `M`:

    - `M == 0` (graph-capture warmup): no launch.
    - tiny `M` (`M <= 64`, decode): the fused mixed GEMV — `a` is loaded as bf16
      and widened to fp32 in registers, then dotted against the unchanged fp32
      `b` in a SINGLE launch, fusing away the standalone bf16->fp32 activation
      cast that otherwise precedes the fp32 router GEMV.
    - large `M` (prefill): the fused GEMV is catastrophically slow, so the op
      falls back to the baseline chain — cast `a` to fp32, then an ordinary
      fp32 matmul against the fp32 `b` (two launches, matching baseline).

    Because bf16->fp32 widening is lossless, every route is numerically
    equivalent to casting `a` to fp32 first and running the fp32 GEMM.

    This is not part of generic matmul/GEMV dispatch. The MiniMax-M3 router emits
    this explicit graph op on MI355X; other architectures retain the standard
    router path.

    `N` (expert count) and `K` (hidden size) are read from the weight `b`'s
    static `[N, K]` layout at compile time — the op carries no redundant `N`/`K`
    custom-op parameters. `N` still selects the kernel's `check_bounds_n` guard.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        c: OutputTensor[dtype=.float32, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.float32, rank=2, ...],
        context: DeviceContext,
    ) raises:
        """Executes the mixed bf16-A × fp32-B router-gate GEMV with an M-based
        route to the baseline cast + fp32 matmul for large (prefill) M.

        Constraints:
            The weight `b` must have a static `[N, K]` shape (both dims known at
            compile time).

        Args:
            c: Output `[M, N]` fp32 tensor.
            a: Activation `[M, K]` bf16 tensor.
            b: Weight `[N, K]` fp32 tensor (transpose_b layout).
            context: The device context.
        """
        comptime assert is_gpu[target](), "router-gate mixed GEMV is GPU-only"

        var M = c.dim_size[0]()
        # Empty-launch guard: a graph-capture warmup can call with M == 0.
        if M == 0:
            return

        # `b` carries the static `[N, K]` router-gate shape from the graph, so
        # the standard projection drives the compile-time N/K specialization the
        # kernels rely on (`N` picks the `check_bounds_n` guard). `c`/`a` stay
        # manual because their M dimension is runtime-dynamic.
        var b_tt = b.to_tile_tensor()
        comptime N = type_of(b_tt).static_shape[0]
        comptime K = type_of(b_tt).static_shape[1]
        comptime assert (
            N != UNKNOWN_VALUE and K != UNKNOWN_VALUE
        ), "router-gate mixed GEMV requires a static [N, K] weight shape"

        var c_tt = TileTensor(c.unsafe_ptr(), row_major(Coord(M, Idx[N])))

        if router_gate_use_mixed_gemv(M):
            # Tiny-M decode: fused mixed bf16-A × fp32-B GEMV, one launch.
            var a_tt = TileTensor(a.unsafe_ptr(), row_major(Coord(M, Idx[K])))
            router_gate_mixed_gemv[N](
                c_tt,
                a_tt.as_immut(),
                b_tt.as_immut(),
                M,
                N,
                K,
                context,
            )
            return

        # Large-M prefill: preserve the baseline's semantics and performance —
        # widen the bf16 activation to fp32 (the standalone cast the fused path
        # avoids), then run the ordinary fp32 matmul against the fp32 weight.
        # Routing large M through the tiny-M GEMV is catastrophically slow.
        var a_f32 = context.enqueue_create_buffer[.float32](M * K)
        var a_bf16_tt = TileTensor(a.unsafe_ptr(), row_major(Coord(M, Idx[K])))
        var a_f32_tt = TileTensor(a_f32, row_major(Coord(M, Idx[K])))

        @__parameter
        @always_inline
        @__copy_capture(a_bf16_tt, a_f32_tt)
        def _cast_bf16_to_fp32[width: Int, alignment: Int = 1](idx: Coord):
            var il = coord_to_index_list(idx)
            a_f32_tt.store_linear(
                il, a_bf16_tt.load_linear[width](il).cast[.float32]()
            )

        elementwise[
            _cast_bf16_to_fp32,
            simd_width_of[DType.bfloat16, target=get_gpu_target()](),
            target=target,
        ](Coord(M, Idx[K]), context)

        matmul[transpose_b=True, target=target](
            c_tt, a_f32_tt.as_immut(), b_tt.as_immut(), context
        )
        _ = a_f32^


@extensibility.register("mo.smallm.streaming.matmul")
struct Struct_smallm_streaming_matmul:
    """MOGG wrapper for the MI355X small-M streaming matmul.

    Computes ``c = a @ b^T`` where ``b`` is a bf16 weight ALREADY permuted
    into the fragment-major layout of ``smallm_preshuffle_b`` (done on the
    CPU at weight-load time). The layout is private to this op: reading a
    row-major weight here is silently wrong, so nothing routes here through
    generic dispatch — MiniMax-M3's MTP draft emits this op explicitly on
    MI355X for its decode-band vocab projections.

    ``N``/``K`` come from the weight's static ``[N, K]`` layout. The
    streaming kernel wins up to ``M == 32`` (measured on the MiniMax-M3
    vocab-head shapes); larger runtime ``M`` falls back to generic matmul
    dispatch over ``b``, the row-major twin of the same weight, so a
    batch-config change can never turn into a regression or a failure.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        c: OutputTensor[dtype=.bfloat16, rank=2, ...],
        a_scratch: OutputTensor[dtype=.bfloat16, rank=2, ...],
        a: InputTensor[dtype=.bfloat16, rank=2, ...],
        b_shuffled: InputTensor[dtype=.bfloat16, rank=2, ...],
        b: InputTensor[dtype=.bfloat16, rank=2, ...],
        context: DeviceContext,
    ) raises:
        """Executes the streaming matmul over a preshuffled weight.

        Constraints:
            ``b_shuffled`` must have a static ``[N, K]`` shape with
            ``N % 16 == 0`` and ``K % 256 == 0``; ``b`` is the same weight
            in row-major ``[N, K]``.

        Args:
            c: Output ``[M, N]`` bf16 tensor.
            a_scratch: Graph-managed ``[32, K]`` bf16 workspace for the
                activation shuffle. Graph memory keeps the captured launches'
                pointers valid across device-graph replays; a transient
                buffer here is silently wrong under capture.
            a: Activation ``[M, K]`` bf16 tensor (row-major).
            b_shuffled: Weight in ``smallm_preshuffle_b`` layout.
            b: The same weight, row-major (the above-band fallback operand).
            context: The device context.
        """
        comptime assert is_gpu[target](), "smallm streaming matmul is GPU-only"

        var M = c.dim_size[0]()
        if M == 0:
            return

        var b_tt = b_shuffled.to_tile_tensor()
        comptime N = type_of(b_tt).static_shape[0]
        comptime K = type_of(b_tt).static_shape[1]
        comptime assert (
            N != UNKNOWN_VALUE and K != UNKNOWN_VALUE
        ), "smallm streaming matmul requires a static [N, K] weight shape"

        if M <= 32:
            var b_ptr = UnsafePointer[BFloat16, ImmutAnyOrigin](
                unsafe_from_address=Int(b_shuffled.unsafe_ptr())
            )
            var c_ptr = UnsafePointer[BFloat16, MutAnyOrigin](
                unsafe_from_address=Int(c.unsafe_ptr())
            )
            var c_tt = TileTensor[
                .bfloat16, type_of(row_major(Coord(1, Idx[N]))), MutAnyOrigin
            ](c_ptr, row_major(Coord(M, Idx[N])))
            var a_ptr = UnsafePointer[BFloat16, ImmutAnyOrigin](
                unsafe_from_address=Int(a.unsafe_ptr())
            )
            var a_tt = TileTensor[
                .bfloat16, type_of(row_major(Coord(1, Idx[K]))), ImmutAnyOrigin
            ](a_ptr, row_major(Coord(M, Idx[K])))
            var scratch_ptr = UnsafePointer[BFloat16, MutAnyOrigin](
                unsafe_from_address=Int(a_scratch.unsafe_ptr())
            )
            smallm_streaming_matmul[k_static=K](
                c_tt, a_tt, b_ptr, scratch_ptr, M, N, context
            )
            return

        matmul[
            False,
            True,
            False,
            None,
            None,
            target=target,
        ](
            c.to_tile_tensor[.int64](),
            a.to_tile_tensor[.int64](),
            b.to_tile_tensor[.int64](),
            context,
        )
