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
"""Unified dispatch for SwiGLU + MXFP8 grouped matmul.

MXFP8 counterpart of `grouped_matmul_swiglu_nvfp4_dispatch`. The caller
pre-permutes the weight `W` on the N axis with `sigma(2i)=i,
sigma(2i+1)=D+i` (`D = moe_dim`, `N = 2D`), the same permutation as
the NVFP4 fused path. This dispatch produces packed MXFP8 (one
`float8_e4m3fn` per byte) and a 5D `float8_e8m0fnu` (E8M0) scale tile
in one entry point.

The implementation routes through `grouped_matmul_mxfp8_dispatch` with
`fuse_swiglu=True`, which selects the in-tile-fused epilogue (SwiGLU
+ per-block MXFP8 quant fused into the matmul's epilogue, avoiding the
BF16 GMEM round trip).

"""

from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from std.memory import UnsafePointer
from layout import Coord, Idx, TileTensor, row_major

from linalg.fp4_utils import MXFP8_SF_DTYPE, MXFP8_SF_VECTOR_SIZE
from .dispatch import grouped_matmul_mxfp8_dispatch
from .grouped_1d1d_matmul_kernel import RealSwiGLUOutput


def grouped_matmul_swiglu_mxfp8_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel.ON,
    # When True (default), the in-tile fused epilogue rounds through
    # bf16 in the SMEM scatter so its output matches the chained
    # reference (matmul -> bf16 GMEM -> SwiGLU+quant). When False, fp32
    # is preserved end-to-end (numerically slightly more accurate, but
    # a tiny fraction of values may quantize to a different fp8 bucket).
    match_bf16: Bool = True,
    # When True (default), the kernel takes the register-only in-place
    # epilogue on the decode regime (avg tokens/expert <= 8), skipping
    # the bf16 SMEM scratchpad. The dispatch-level gate forces the
    # cooperative path on prefill regimes; flip to False to benchmark
    # the cooperative path on decode.
    use_inplace: Bool = True,
    # Activation flavor. False = plain SwiGLU; True = clamped
    # (`swigluoai`). Pass the HF config `swiglu_alpha`/`swiglu_limit`
    # as the runtime `alpha`/`limit` args when set to True.
    clamp_activation: Bool = False,
](
    c_packed: TileTensor,
    c_swiglu_scales: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
    # Runtime α and L for clamped activation. Ignored when
    # `clamp_activation=False`.
    alpha: Float32 = Float32(0.0),
    limit: Float32 = Float32(0.0),
) raises:
    """SwiGLU + MXFP8 fused MoE up-projection dispatch.

    Caller pre-permutes `W` on its N axis so that adjacent output
    columns `(2i, 2i+1)` carry `(gate, up)` pairs (σ). The dispatch
    then computes:

        bf16_scratch = grouped_matmul(A, W_perm, A_scales, B_scales_perm)
        c_packed, c_swiglu_scales =
            fused_silu_mxfp8_interleaved(bf16_scratch)

    in a single entry point. MXFP8 does NOT carry a separate per-expert
    `tensor_sf` (E8M0 cannot represent non-power-of-2 multipliers
    losslessly); per-block scales are the only quantization parameter.

    Parameters:
        transpose_b: Transpose the B weight in the matmul (defaults to
            True).
        target: Target backend selector forwarded to the inner
            dispatch as a `StaticString` (defaults to `"cpu"`).
        pdl_level: Program-Dependent Launch level forwarded to the
            grid controls (defaults to `PDLLevel.ON`).
        match_bf16: Round the in-tile fused epilogue through `bf16`
            in the SMEM scatter so its output matches the chained
            reference (matmul -> bf16 GMEM -> SwiGLU+quant) (defaults
            to True). When False, `fp32` is preserved end-to-end,
            which is numerically slightly more accurate but may
            quantize a tiny fraction of values to a different fp8
            bucket.
        use_inplace: Take the register-only in-place epilogue on the
            decode regime (avg tokens/expert <= 8), skipping the
            `bf16` SMEM scratchpad (defaults to True). The
            dispatch-level gate forces the cooperative path on
            prefill regimes; flip to False to benchmark the
            cooperative path on decode.
        clamp_activation: Activation flavor (defaults to False).
            False is plain SwiGLU; True is clamped (`swigluoai`).
            Pass the HF config `swiglu_alpha`/`swiglu_limit` as the
            runtime `alpha`/`limit` args when set to True.

    Args:
        c_packed: Output `float8_e4m3fn`, shape `(M_total, D)` (one
            byte per element). `D = moe_dim` and `N = 2D` is the
            matmul's N dim.
        c_swiglu_scales: Output 5D `float8_e8m0fnu` scale tile, shape
            `(c_scale_dim0, ceildiv(D, MXFP8_SF_VECTOR_SIZE *
            SF_ATOM_K), 32, 4, 4)`. Indexed via `set_scale_factor[
            SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE]` against (token,
            k_block) coords.
        a: Input A (MXFP8 `float8_e4m3fn`, shape `(M_total, K)`).
        b: Pre-permuted weight (MXFP8, shape `(num_experts, 2D, K)`).
        a_scales: A's 5D E8M0 scale tile.
        b_scales: B's 6D E8M0 scale tile, **with the matching σ
            permutation already applied on its N axis**.
        a_offsets: Per-expert prefix-sum token offsets,
            shape `(num_active_experts + 1,)`.
        a_scale_offsets: Per-expert offsets into `a_scales`'s first
            dim, shape `(num_active_experts,)`. Re-used as
            `c_swiglu_scales`'s per-expert offsets.
        expert_ids: Active expert IDs (`-1` for skipped slots).
        expert_scales: Per-expert output scaling.
        num_active_experts: Number of active experts.
        estimated_total_m: Estimated total non-padded token count.
        ctx: Device context.
        alpha: Runtime α for the clamped activation. Ignored when
            `clamp_activation=False`. Pass the HF config
            `swiglu_alpha` value (canonically 1.702 for the
            `swigluoai` family).
        limit: Runtime L for the clamped activation. Ignored when
            `clamp_activation=False`. Pass the HF config
            `swiglu_limit` value (canonically 7.0 for the
            `swigluoai` family).
    """
    comptime c_type = DType.bfloat16
    comptime N = type_of(b).static_shape[1]

    # C is unused on the fused path: the epilogue writes results through
    # `swiglu_out` into `c_packed`, and the launcher + kernel comptime-gate out
    # the C TMA encode, prefetch, and store when `fuse_swiglu`. We still pass a
    # real BF16 tensor so `grouped_matmul_block_scaled` can infer
    # `c_type`/`N`/layout and satisfy the kernel ABI, but it is a fixed 1-row
    # placeholder decoupled from `estimated_total_m` (which floors to 0 in
    # low-concurrency EP decode and previously produced a zero-dim C TMA
    # descriptor -> CUDA_ERROR_INVALID_VALUE). The buffer is never read or
    # written. (Mojo's `UnsafePointer` is non-nullable, so this is a minimal
    # 1xN allocation rather than a null view.)
    var dummy_c_buffer = ctx.enqueue_create_buffer[c_type](N)
    var dummy_c_shape = row_major(Coord(Idx[1], Idx[N]))
    var dummy_c_tensor = TileTensor(dummy_c_buffer, dummy_c_shape)

    comptime c_packed_row_stride = type_of(c_packed).static_shape[1]
    comptime sf_dim1 = type_of(c_swiglu_scales).static_shape[1]
    var c_packed_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
        c_packed._storage
    )
    var c_swiglu_scales_ptr = rebind[
        UnsafePointer[Scalar[MXFP8_SF_DTYPE], MutAnyOrigin]
    ](c_swiglu_scales._storage)
    # MXFP8 doesn't use a per-expert tensor_sf; the trait method is
    # gated out for MXFP8 so this pointer is never dereferenced.
    var c_input_scales_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
        expert_scales._storage
    )
    var swiglu_out = RealSwiGLUOutput[
        c_packed_row_stride,
        sf_dim1,
        MXFP8_SF_DTYPE,
        MXFP8_SF_VECTOR_SIZE,
        clamp_activation,
    ](
        c_packed_ptr,
        c_swiglu_scales_ptr,
        c_input_scales_ptr,
        alpha,
        limit,
    )

    grouped_matmul_mxfp8_dispatch[
        transpose_b=transpose_b,
        target=target,
        pdl_level=pdl_level,
        fuse_swiglu=True,
        SwiGLUOutputT=type_of(swiglu_out),
        swiglu_match_bf16=match_bf16,
        swiglu_use_inplace=use_inplace,
    ](
        dummy_c_tensor,
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
        swiglu_out,
    )

    _ = dummy_c_buffer^
