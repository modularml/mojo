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
"""Unified dispatch for SwiGLU + NVFP4 grouped matmul.

Caller pre-permutes the weight `W` on the N axis with `σ(2i)=i, σ(2i+1)=D+i`
(`D = moe_dim`, `N = 2D`). This dispatch produces packed NVFP4 + a 5D
FP8-E4M3 scale tile in one entry point.

The implementation routes through `grouped_matmul_nvfp4_dispatch` with
`fuse_swiglu=True`, which selects the in-tile-fused epilogue
(SwiGLU + per-block NVFP4 quant fused into the matmul's epilogue,
avoiding the BF16 GMEM round trip).

Lives next to its callees: `grouped_matmul_nvfp4_dispatch` (in this
package's `dispatch.mojo`) and the `RealSwiGLUOutput` carrier (in
`grouped_1d1d_matmul_kernel.mojo`).
"""

from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, pdl_launch_attributes
from std.memory import UnsafePointer
from layout import Coord, Idx, TileTensor, row_major

from linalg.fp4_utils import NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
from .dispatch import grouped_matmul_nvfp4_dispatch
from .grouped_1d1d_matmul_kernel import RealSwiGLUOutput


def grouped_matmul_swiglu_nvfp4_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel.ON,
    # When True (default), the in-tile fused epilogue casts fp32 -> bf16
    # -> fp32 in the SMEM scatter so its output is byte-identical to the
    # chained reference (matmul -> bf16 GMEM -> SwiGLU+quant). When False,
    # fp32 is preserved end-to-end through the SwiGLU computation:
    # numerically slightly more accurate, but a tiny fraction of values
    # may quantize to a different fp4 bucket.
    match_bf16: Bool = True,
    # When True, the kernel takes a register-only in-place epilogue path
    # that skips the bf16 SMEM scratchpad entirely on the small-BN
    # decode regime (mma_bn <= 8). The dispatch-level gate in
    # `_launch_grouped_block_scaled` forces `False` for prefill
    # (mma_bn >= 64) where the cooperative loop's SMEM-amortized work
    # pattern outperforms the per-tile shuffle cost. Default True so
    # the decode regime auto-uses the faster path; flip to False to
    # benchmark the legacy cooperative path on decode.
    use_inplace: Bool = True,
    # Activation flavor. False = plain SwiGLU (silu(g)·u), True =
    # clamped (`swigluoai`: g'=min(g,L); u'=clamp(u,-L,L); (u'+1)·g'·
    # σ(g'·α)). For `hidden_act = "silu"` models leave False; for
    # `hidden_act = "swigluoai"` set True and pass `alpha`/`limit`
    # runtime args.
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
    c_input_scales: TileTensor,
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
    # Runtime α and L for clamped activation. Ignored when
    # `clamp_activation=False`. Default 0.0 keeps the kernel argument
    # ABI stable for standard-SwiGLU callers that omit them.
    alpha: Float32 = Float32(0.0),
    limit: Float32 = Float32(0.0),
) raises:
    """SwiGLU + NVFP4 fused MoE up-projection dispatch.

    Caller pre-permutes `W` on its N axis so that adjacent output columns
    `(2i, 2i+1)` carry `(gate, up)` pairs (Phase 1 σ). The dispatch then
    computes:

        bf16_scratch = grouped_matmul(A, W_perm, A_scales, B_scales_perm)
        c_packed, c_swiglu_scales =
            fused_silu_nvfp4_interleaved(bf16_scratch, c_input_scales)

    in a single entry point.

    Parameters:
        transpose_b: Whether to transpose the `B` weight in the matmul
            (defaults to True).
        target: Compilation target backend string (defaults to `"cpu"`).
        pdl_level: Programmatic dependent launch level controlling the
            kernel's PDL launch attributes (defaults to `PDLLevel.ON`).
        match_bf16: When True (default), the in-tile fused epilogue casts
            `fp32` -> `bf16` -> `fp32` in the SMEM scatter so its output is
            byte-identical to the chained reference (matmul -> `bf16` GMEM
            -> SwiGLU+quant). When False, `fp32` is preserved end-to-end
            through the SwiGLU computation: numerically slightly more
            accurate, but a tiny fraction of values may quantize to a
            different fp4 bucket.
        use_inplace: When True (default), the kernel takes a register-only
            in-place epilogue path that skips the `bf16` SMEM scratchpad
            entirely on the small-BN decode regime (`mma_bn <= 8`). The
            dispatch-level gate in `_launch_grouped_block_scaled` forces
            `False` for prefill (`mma_bn >= 64`) where the cooperative
            loop's SMEM-amortized work pattern outperforms the per-tile
            shuffle cost. Flip to False to benchmark the legacy cooperative
            path on decode.
        clamp_activation: Activation flavor. `False` (default) is plain
            SwiGLU (`silu(g)·u`); `True` is clamped (`swigluoai`:
            `g'=min(g,L)`, `u'=clamp(u,-L,L)`, `(u'+1)·g'·σ(g'·α)`). For
            `hidden_act = "silu"` models leave False; for `hidden_act =
            "swigluoai"` set True and pass the `alpha`/`limit` runtime
            args.

    Args:
        c_packed: Output, packed NVFP4 (uint8). Shape `(M_total, D/2)` where
            `D = moe_dim` and `N = 2D` is the matmul's N dim.
        c_swiglu_scales: Output 5D FP8-E4M3 scale tile, shape
            `(c_scale_dim0, ceildiv(D, 64), 32, 4, 4)`. Indexed via
            `set_scale_factor[SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE]` against
            (token, k_block) coords.
        a: Input A (NVFP4-packed uint8, shape `(M_total, K/2)`).
        b: Pre-permuted weight (NVFP4-packed uint8, shape
            `(num_experts, 2D, K/2)`).
        a_scales: A's 5D FP8-E4M3 scale tile.
        b_scales: B's 6D FP8-E4M3 scale tile, **with the matching σ
            permutation already applied on its N axis**.
        a_offsets: Per-expert prefix-sum token offsets,
            shape `(num_active_experts + 1,)`.
        a_scale_offsets: Per-expert offsets into `a_scales`'s first dim,
            shape `(num_active_experts,)`. Re-used as
            `c_swiglu_scales`'s per-expert offsets (the SF tile
            geometry is identical).
        expert_ids: Active expert IDs (`-1` for skipped slots),
            shape `(num_active_experts,)`.
        expert_scales: Per-expert output scaling, shape `(num_experts,)`.
            Applied inside the matmul.
        c_input_scales: Per-expert input scales for the SwiGLU+quant kernel,
            shape `(num_active_experts,)`. The `tensor_sf` of `ep_comm.mojo`.
        num_active_experts: Number of active experts.
        estimated_total_m: Estimated total non-padded token count, used to
            size the BF16 scratch buffer.
        ctx: Device context.
        alpha: Runtime α for the clamped activation. Ignored when
            `clamp_activation=False`. For `swigluoai` models pass the
            HF config `swiglu_alpha` value.
        limit: Runtime L for the clamped activation. Ignored when
            `clamp_activation=False`. For `swigluoai` models pass the
            HF config `swiglu_limit` value.
    """
    comptime c_type = DType.bfloat16
    comptime N = type_of(b).static_shape[1]

    # C is unused on the fused path: the epilogue writes results through
    # `swiglu_out`, and the launcher + kernel comptime-gate out the C TMA
    # encode, prefetch, and store when `fuse_swiglu`. We still pass a real BF16
    # tensor so `grouped_matmul_block_scaled` can infer `c_type`/`N`/layout and
    # satisfy the kernel ABI, but it is a fixed 1-row placeholder decoupled from
    # `estimated_total_m` (which floors to 0 in low-concurrency EP decode and
    # previously produced a zero-dim C TMA descriptor ->
    # CUDA_ERROR_INVALID_VALUE). The buffer is never read or written. (Mojo's
    # `UnsafePointer` is non-nullable, so this is a minimal 1xN allocation
    # rather than a null view.)
    var dummy_c_buffer = ctx.enqueue_create_buffer[c_type](N)
    var dummy_c_shape = row_major(Coord(Idx[1], Idx[N]))
    var dummy_c_tensor = TileTensor(dummy_c_buffer, dummy_c_shape)

    # Wrap the three real destinations in a RealSwiGLUOutput carrier.
    # c_packed shape: (M_total, D/2). Row stride = D/2 = N/4 (since N is
    # the matmul's N dim and D = N/2, so D/2 = N/4 bytes per row).
    comptime c_packed_row_stride = type_of(c_packed).static_shape[1]
    # SF tile shape (n_blocks, sf_dim1, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K).
    # The static dim1 is ceildiv(D, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K).
    comptime sf_dim1 = type_of(c_swiglu_scales).static_shape[1]
    var c_packed_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
        c_packed._storage
    )
    var c_swiglu_scales_ptr = rebind[
        UnsafePointer[Scalar[NVFP4_SF_DTYPE], MutAnyOrigin]
    ](c_swiglu_scales._storage)
    var c_input_scales_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
        c_input_scales._storage
    )
    var swiglu_out = RealSwiGLUOutput[
        c_packed_row_stride,
        sf_dim1,
        NVFP4_SF_DTYPE,
        NVFP4_SF_VECTOR_SIZE,
        clamp_activation,
    ](
        c_packed_ptr,
        c_swiglu_scales_ptr,
        c_input_scales_ptr,
        alpha,
        limit,
    )

    # The fused matmul writes SF for both live tokens and the per-expert
    # tail-pad rows in their last 128-row block (zero-filled by the
    # epilogue). No host-side memset of the SF buffer is needed.

    grouped_matmul_nvfp4_dispatch[
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
