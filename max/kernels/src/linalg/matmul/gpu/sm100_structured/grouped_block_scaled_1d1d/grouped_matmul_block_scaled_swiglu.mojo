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
"""Unified NVFP4/MXFP8 grouped block-scaled matmul + SwiGLU dispatch."""

from max.gpu.host import DeviceContext
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from max.gpu.primitives.grid_controls import PDLLevel
from layout import TileTensor

from linalg.fp4_utils import block_scaled_umma_kind, is_w4a8_operand_pair
from .grouped_matmul_swiglu_nvfp4 import grouped_matmul_swiglu_nvfp4_dispatch
from .grouped_matmul_swiglu_mxfp8 import grouped_matmul_swiglu_mxfp8_dispatch


def grouped_matmul_block_scaled_swiglu_sm100_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel.ON,
    clamp_activation: Bool = False,
](
    c: TileTensor,
    c_swiglu_scales: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    c_input_scales: TileTensor,
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
    alpha: Float32 = Float32(0.0),
    limit: Float32 = Float32(0.0),
) raises:
    """Dispatches grouped block-scaled matmul with fused SwiGLU by dtype.

    Routes NVFP4 inputs to `grouped_matmul_swiglu_nvfp4_dispatch` and
    MXFP8 inputs to `grouped_matmul_swiglu_mxfp8_dispatch` based on the
    `a`/`a_scales` dtypes. See those entry points for the per-format
    contract.

    Parameters:
        transpose_b: Whether B is transposed (must be True).
        target: Target device (unused, for MOGG interface compatibility).
        pdl_level: Programmatic dependent launch level.
        clamp_activation: Activation flavor. `False` for plain SwiGLU
            (`silu(g)·u`), `True` for the clamped `swigluoai` form. When
            `True`, pass the `alpha`/`limit` runtime args.

    Args:
        c: Packed SwiGLU output tensor. NVFP4: packed `uint8`, shape
            `(M_total, D/2)`. MXFP8: `float8_e4m3fn`, shape
            `(M_total, D)`. `D = moe_dim` and `N = 2D` is the matmul's
            N dim.
        c_swiglu_scales: Output 5D scale tile for the packed result.
            NVFP4: `FP8-E4M3`, shape `(c_scale_dim0, ceildiv(D, 64), 32,
            4, 4)`. MXFP8: `E8M0`, shape `(c_scale_dim0, ceildiv(D,
            MXFP8_SF_VECTOR_SIZE * SF_ATOM_K), 32, 4, 4)`.
        a: Input A. NVFP4: packed `uint8`, shape `(M_total, K/2)`. MXFP8:
            `float8_e4m3fn`, shape `(M_total, K)`.
        b: Pre-permuted weight with the σ permutation already applied
            on its N axis. NVFP4: `uint8`, shape `(num_experts, 2D,
            K/2)`. MXFP8: `float8_e4m3fn`, shape `(num_experts, 2D, K)`.
        a_scales: A's 5D scale tile. NVFP4: `FP8-E4M3`. MXFP8: `E8M0`.
        b_scales: B's 6D scale tile, with the matching σ permutation
            already applied on its N axis.
        a_offsets: Per-expert prefix-sum token offsets, shape
            `(num_active_experts + 1,)`.
        a_scale_offsets: Per-expert offsets into `a_scales`'s first
            dim, shape `(num_active_experts,)`. Re-used as
            `c_swiglu_scales`'s per-expert offsets.
        expert_ids: Active expert IDs (`-1` for skipped slots), shape
            `(num_active_experts,)`.
        expert_scales: Per-expert output scaling, shape
            `(num_experts,)`. Applied inside the matmul.
        c_input_scales: Per-expert input scales for the SwiGLU+quant
            kernel, shape `(num_active_experts,)`. Used only on the
            NVFP4 path; ignored for MXFP8.
        num_active_experts: Number of active experts.
        estimated_total_m: Estimated total non-padded token count,
            used to size the BF16 scratch buffer.
        ctx: Device context.
        alpha: Runtime α for the clamped activation. Ignored when
            `clamp_activation=False`. For `swigluoai` models pass the
            HF config `swiglu_alpha` value.
        limit: Runtime L for the clamped activation. Ignored when
            `clamp_activation=False`. For `swigluoai` models pass the
            HF config `swiglu_limit` value.
    """

    # Neither fused epilogue builds the padded FP4 TMA copy that a packed B
    # needs, so the mixed pair has to be rejected here rather than inferred
    # into `kind::mxf8f6f4` and silently fed unpadded weights.
    comptime assert not is_w4a8_operand_pair[a.dtype, b.dtype](), (
        "the fused SwiGLU epilogue does not support W4A8; run the plain"
        " grouped block-scaled matmul and fuse separately"
    )
    comptime scaling_kind = block_scaled_umma_kind[
        a.dtype, b.dtype, a_scales.dtype
    ]()

    comptime if scaling_kind == UMMAKind.KIND_MXF4NVF4:
        grouped_matmul_swiglu_nvfp4_dispatch[
            transpose_b=transpose_b,
            target=target,
            pdl_level=pdl_level,
            clamp_activation=clamp_activation,
        ](
            c,
            c_swiglu_scales,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            c_input_scales,
            num_active_experts,
            estimated_total_m,
            ctx,
            alpha,
            limit,
        )
    elif scaling_kind == UMMAKind.KIND_MXF8F6F4:
        grouped_matmul_swiglu_mxfp8_dispatch[
            transpose_b=transpose_b,
            target=target,
            pdl_level=pdl_level,
            clamp_activation=clamp_activation,
        ](
            c,
            c_swiglu_scales,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            estimated_total_m,
            ctx,
            alpha,
            limit,
        )
    else:
        raise Error(t"Unsupported scaling kind: {scaling_kind}")
