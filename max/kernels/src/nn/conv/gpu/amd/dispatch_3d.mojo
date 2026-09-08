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
"""AMD MI355X (CDNA4, gfx950) conv3d dispatch to `amd_4wave_conv`.

Sibling of `nn.conv.gpu.amd.dispatch.dispatch_amd_4wave_conv2d`, extended
to 3D NDHWC inputs via the loader's `Q > 1` mode. Single-kernel
implicit-GEMM: no `M*K` im2col scratch, fp32 accumulator inside the
MMA, no per-q Q-slice round-trips.

Returns True when the input/filter/output shapes + runtime stride /
pad / dilation are handled by the 4-wave conv kernel; False to fall
back to the caller's im2col / cuDNN path. Acceptance rules:

  - Hardware: MI355X (gfx950) only.
  - Input dtype: float8_e4m3fn, bfloat16, or float16. Output dtype:
    bfloat16 for FP8, otherwise tracks input.
  - All input/filter/output spatial shapes must be **static**
    (TileTensor `static_shape[i] >= 0`). Dynamic shapes fall through.
  - `num_groups == 1`.
  - `dilation == (1, 1, 1)`.
  - `stride[0] == stride[1] == stride[2]` and stride ∈ {1, 2}.
  - `symmetric_padding[i] ∈ {0, 1, 2}` for each axis, with the
    further constraint that `pad_h == pad_w` (the kernel takes a
    single (stride, pad) tuple) and `pad_d` may differ, but for now
    we require all three pads equal so the static-launch enumeration
    stays bounded.

When accepted, the dispatcher:

  1. Allocates a K-padded [F, K_padded] filter buffer
     (K_padded = round_up(Q*R*S*C_in, 2*BK = 256), zero-filling the
     trailing K columns).
  2. Transposes the caller's filter (QRSCF or FCQRS) into [F, K_padded]
     row-major.
  3. Comptime-materializes (stride, pad) and calls `amd_4wave_conv`
     with Q > 1 (3D mode).

Mirrors the structure of `nn.conv.gpu.amd.dispatch`.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.math.uutils import udivmod
from std.sys import simd_width_of
from std.sys.info import _accelerator_arch
from std.utils import IndexList

from layout import Coord, TileTensor, row_major

from linalg.utils import elementwise_epilogue_type

from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv


# ---------------------------------------------------------------------------
# Filter transposes to [F, K_padded] with K-padding (zero-fill).
# ---------------------------------------------------------------------------


@__name(t"amd_4wave_3d_transpose_qrscf_to_fk_kpad_{dtype}")
def _transpose_qrscf_to_fk_kpad[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    Q: Int32,
    R: Int32,
    S: Int32,
    C: Int32,
    F: Int32,
    K_padded: Int32,
):
    """GPU kernel: filter QRSCF `[Q*R*S*C, F]` -> `[F, K_padded]`.

    `K_padded >= Q*R*S*C`; padded trailing columns are zero-filled.
    """
    var _Q = Int(Q)
    var _R = Int(R)
    var _S = Int(S)
    var _C = Int(C)
    var _F = Int(F)
    var _K_padded = Int(K_padded)
    var K_real = _Q * _R * _S * _C
    var total = _F * _K_padded
    var tid = global_idx.x
    if tid >= total:
        return
    var f = tid // _K_padded
    var k = tid - f * _K_padded
    if k >= K_real:
        dst_ptr.store(tid, Scalar[dtype](0))
        return
    var q = k // (_R * _S * _C)
    var rsc = k - q * (_R * _S * _C)
    var r = rsc // (_S * _C)
    var sc = rsc - r * (_S * _C)
    var s = sc // _C
    var c = sc - s * _C
    # QRSCF source index: (((q*_R + r)*_S + s)*_C + c)*_F + f.
    var qrscf_idx = (((q * _R + r) * _S + s) * _C + c) * _F + f
    dst_ptr.store(tid, src_ptr.load(qrscf_idx))


@__name(t"amd_4wave_3d_transpose_fcqrs_to_fk_kpad_{dtype}")
def _transpose_fcqrs_to_fk_kpad[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    F: Int32,
    C: Int32,
    Q: Int32,
    R: Int32,
    S: Int32,
    K_padded: Int32,
):
    """GPU kernel: filter FCQRS `[F, C, Q, R, S]` -> `[F, K_padded]`.

    `K_padded >= Q*R*S*C`; padded trailing columns are zero-filled.
    """
    var _F = Int(F)
    var _C = Int(C)
    var _Q = Int(Q)
    var _R = Int(R)
    var _S = Int(S)
    var _K_padded = Int(K_padded)
    var K_real = _Q * _R * _S * _C
    var total = _F * _K_padded
    var tid = global_idx.x
    if tid >= total:
        return
    var f = tid // _K_padded
    var k = tid - f * _K_padded
    if k >= K_real:
        dst_ptr.store(tid, Scalar[dtype](0))
        return
    var q = k // (_R * _S * _C)
    var rsc = k - q * (_R * _S * _C)
    var r = rsc // (_S * _C)
    var sc = rsc - r * (_S * _C)
    var s = sc // _C
    var c = sc - s * _C
    # FCQRS source index: f*_C*_Q*_R*_S + c*_Q*_R*_S + q*_R*_S + r*_S + s.
    var fcqrs_idx = (
        f * _C * _Q * _R * _S + c * _Q * _R * _S + q * _R * _S + r * _S + s
    )
    dst_ptr.store(tid, src_ptr.load(fcqrs_idx))


# ---------------------------------------------------------------------------
# Dispatch entry point.
# ---------------------------------------------------------------------------


def dispatch_amd_4wave_conv3d[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcqrs: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    # Autotune-override knobs. When non-zero, override the heuristic
    # pick of (BM, BN, BK). Used by `bench_conv3d.mojo`'s kbench sweep
    # to populate a per-shape dispatch table. Default 0 = use heuristic.
    block_m_override: Int = 0,
    block_n_override: Int = 0,
    block_k_override: Int = 0,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    symmetric_padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Try to dispatch a Conv3D to `amd_4wave_conv` on MI355X. Returns
    True if handled; False if the caller should fall through (typically
    to `dispatch_im2col_matmul_conv3d`).

    Parameters:
        input_type: `DType` of the input tensor; must be
            `float8_e4m3fn`, `bfloat16`, or `float16`.
        filter_type: `DType` of the filter tensor; must equal
            `input_type`.
        output_type: `DType` of the output tensor; `bfloat16` for
            FP8 input, otherwise tracks `input_type`.
        filter_is_fcqrs: `True` if the filter is laid out as FCQRS
            `[F, C, Q, R, S]`; `False` if QRSCF `[Q*R*S*C, F]`.
        elementwise_lambda_fn: Optional epilogue applied to the conv
            output (defaults to `None`).
        block_m_override: Override for the BM tile size, 0 uses the
            heuristic (defaults to 0).
        block_n_override: Override for the BN tile size, 0 uses the
            heuristic (defaults to 0).
        block_k_override: Override for the BK tile size, 0 uses the
            heuristic (defaults to 0).

    Args:
        input: Input NDHWC tensor of shape `[N, D, H, W, C_in]`;
            all spatial dims must be static.
        filter: Filter tensor; FCQRS `[F, C, Q, R, S]` when
            `filter_is_fcqrs` else QRSCF `[Q, R, S, C, F]`.
        output: Output NDHWC tensor of shape
            `[N, D_out, H_out, W_out, C_out]`.
        stride: Per-axis stride `[stride_d, stride_h, stride_w]`;
            all three must be equal and in `{1, 2}`.
        dilation: Per-axis dilation; must be `(1, 1, 1)`.
        symmetric_padding: Per-axis symmetric padding
            `[pad_d, pad_h, pad_w]`; each in `{0, 1, 2}` and
            `pad_h` must equal `pad_w`.
        num_groups: Convolution group count; must be 1.
        ctx: `DeviceContext` used to enqueue the transpose and
            conv kernels.
    """
    comptime assert input.flat_rank == 5, "input must be rank 5 (NDHWC)"
    comptime assert filter.flat_rank == 5, "filter must be rank 5"
    comptime assert output.flat_rank == 5, "output must be rank 5 (NDHWC)"

    # -------- Hardware gate --------
    comptime if _accelerator_arch() != "amdgpu:gfx950":
        return False

    # -------- Dtype gate --------
    comptime if input_type not in (
        DType.float8_e4m3fn,
        DType.bfloat16,
        DType.float16,
    ):
        return False
    comptime assert (
        input_type == filter_type
    ), "amd_4wave conv requires input_type == filter_type"

    # -------- Static-shape gate --------
    comptime _N_static = input.static_shape[0]
    comptime _D_static = input.static_shape[1]
    comptime _H_static = input.static_shape[2]
    comptime _W_static = input.static_shape[3]
    comptime _C_in = input.static_shape[4]
    comptime _C_out = output.static_shape[4]
    comptime _all_dhw_static = (
        _N_static >= 0 and _D_static >= 0 and _H_static >= 0 and _W_static >= 0
    )
    # Filter Q/R/S come from different dims based on layout.
    comptime _Q = (
        filter.static_shape[2] if filter_is_fcqrs else filter.static_shape[0]
    )
    comptime _R = (
        filter.static_shape[3] if filter_is_fcqrs else filter.static_shape[1]
    )
    comptime _S = (
        filter.static_shape[4] if filter_is_fcqrs else filter.static_shape[2]
    )

    # AMD 4-wave conv2d gates: C_in >= simd_width (so the per-lane loader's
    # contiguous-NHWC assumption holds), C_out >= 64 (BN minimum), R/S/Q > 0.
    # Q == 1 would be more efficient through the 2D dispatcher (flatten
    # N*D → N'); decline so callers route via 2D.
    comptime _simd_w = simd_width_of[input_type]()
    comptime _shapes_ok = (
        _C_in >= _simd_w and _C_out >= 64 and _Q > 1 and _R > 0 and _S > 0
    )
    # Gate the entire dispatch body — including the `@__parameter` closure
    # definitions below — on `_shapes_ok`. Mojo type-checks closure
    # bodies even past a `comptime if not _shapes_ok: return False`
    # early-return, so a shape that we *intend* to decline (e.g.
    # `Q=R=S=1, stride=2`) would still try to instantiate
    # `amd_4wave_conv[Q=1, ...]` inside the closure body, which the 2D
    # loader's rank-4 assert rejects. Matches the pattern in
    # `dispatch_amd_4wave_conv2d`.
    comptime if _shapes_ok:
        # -------- Runtime constraints --------
        if num_groups != 1:
            return False
        if dilation[0] != 1 or dilation[1] != 1 or dilation[2] != 1:
            return False
        if stride[0] != stride[1] or stride[1] != stride[2]:
            return False
        if stride[0] != 1 and stride[0] != 2:
            return False
        # Per-axis pads in {0, 1, 2} (matches the static-launch enumeration
        # used on the 2D path). Each axis is independent.
        var pad_d_v = symmetric_padding[0]
        var pad_h_v = symmetric_padding[1]
        var pad_w_v = symmetric_padding[2]
        if pad_d_v not in (0, 1, 2):
            return False
        if pad_h_v not in (0, 1, 2):
            return False
        if pad_w_v not in (0, 1, 2):
            return False
        # The static-launch enumeration below is bounded; require
        # pad_h == pad_w so we only enumerate 3 spatial pads and pad_d
        # may differ (3 temporal pads). 3*3 = 9 (stride, pad_spatial,
        # pad_d) static launches per stride choice, 18 total. Keeps
        # compile time bounded.
        if pad_h_v != pad_w_v:
            return False

        # -------- Pick BK --------
        # Empirical (kbench sweep on WAN VAE bf16 shapes): BK is the
        # single most impactful tile knob. BK=128 is uniformly 30-50%
        # slower than {32, 64}. Between BK=32 and BK=64 the winner
        # tracks `C_in % 64`:
        #
        #   - C_in % 64 != 0 (e.g. 16 or 96) → BK=32 wins by 10-18%.
        #     K = Q*R*S*C_in isn't 64-aligned, so BK=64 pads to the
        #     next 128-multiple, wasting MFMAs. BK=32 pads to a 64-
        #     multiple instead; padding waste vanishes and the extra
        #     K iterations are cheap enough to net out positive.
        #   - C_in % 64 == 0 (192, 384) → BK=64 wins by 5-25%. K is
        #     already 128-aligned so BK=64 has no padding penalty,
        #     and the lower K-iter count amortizes LDS-load cost.
        #
        # Measured winners (TFLOPS, native_3d vs MIOpen):
        #   WAN_conv_in (C_in=16):  BK=32 → 341 vs 340 = 100%
        #   WAN_level3_res (C_in=96): BK=32 → 544 vs 438 = 124%
        #   WAN_mid_res (C_in=384): BK=64 → 722 vs 596 = 121%
        #   WAN_time_conv (C_in=192): BK=64 → 522 vs 449 = 116%
        #   WAN_upsampled_res (C_in=192): BK=64 → 559 vs 613 = 91%
        comptime _K_real = _Q * _R * _S * _C_in
        comptime _K_padded_bk128 = ((_K_real + 255) // 256) * 256
        comptime _K_padded_bk64 = ((_K_real + 127) // 128) * 128
        comptime _K_padded_bk32 = ((_K_real + 63) // 64) * 64
        comptime _BK_heuristic = 32 if (_C_in % 64) != 0 else 64
        # Honor the kbench-sweep override when non-zero.
        comptime _BK_chosen = (
            block_k_override if block_k_override > 0 else _BK_heuristic
        )
        comptime _K_padded = (
            _K_padded_bk128 if _BK_chosen
            == 128 else (_K_padded_bk64 if _BK_chosen == 64 else _K_padded_bk32)
        )
        var filter_fk_buf = ctx.enqueue_create_buffer[filter_type](
            _C_out * _K_padded
        )
        var filter_fk_ptr = filter_fk_buf.unsafe_ptr()
        comptime _transpose_block = 256
        var transpose_total = _C_out * _K_padded
        var transpose_grid = ceildiv(transpose_total, _transpose_block)
        comptime if filter_is_fcqrs:
            ctx.enqueue_function[_transpose_fcqrs_to_fk_kpad[filter_type]](
                filter.ptr,
                filter_fk_ptr,
                Int32(_C_out),
                Int32(_C_in),
                Int32(_Q),
                Int32(_R),
                Int32(_S),
                Int32(_K_padded),
                grid_dim=transpose_grid,
                block_dim=_transpose_block,
            )
        else:
            ctx.enqueue_function[_transpose_qrscf_to_fk_kpad[filter_type]](
                filter.ptr,
                filter_fk_ptr,
                Int32(_Q),
                Int32(_R),
                Int32(_S),
                Int32(_C_in),
                Int32(_C_out),
                Int32(_K_padded),
                grid_dim=transpose_grid,
                block_dim=_transpose_block,
            )

        # -------- Filter view: [C_out, K_padded] ---------
        comptime _filter_fk_layout = row_major[_C_out, _K_padded]()
        var filter_fk_tt = TileTensor(filter_fk_ptr, _filter_fk_layout)

        # -------- Static vs runtime-HW dispatch --------------------
        comptime if _all_dhw_static:

            @__parameter
            @always_inline
            def _launch_static[
                stride_v: Int, pad_d_c: Int, pad_hw_c: Int
            ]() raises -> Bool:
                comptime _eff_Q = _Q
                comptime _eff_R = _R
                comptime _eff_S = _S
                comptime _D_out_v = (
                    _D_static + 2 * pad_d_c - _eff_Q
                ) // stride_v + 1
                comptime _H_out_v = (
                    _H_static + 2 * pad_hw_c - _eff_R
                ) // stride_v + 1
                comptime _W_out_v = (
                    _W_static + 2 * pad_hw_c - _eff_S
                ) // stride_v + 1
                comptime _M_total_v = (
                    _N_static * _D_out_v * _H_out_v * _W_out_v
                )
                # Per-shape (BM, BN) dispatch table. Extend as the
                # sweep finds wins. Currently empty: the BK sweep on
                # the WAN VAE shape set showed BK dominates and the
                # (BM, BN) variance at the chosen BK is <1% on every
                # shape, so the M-based BM/BN heuristic below suffices.
                comptime _BM_table = 0
                comptime _BN_table = 0

                # M-based BM/BN pick. Small M (early-decoder latent
                # stages, ≤ 1024 rows) → BM=BN=64 keeps occupancy up
                # by reducing VGPR pressure per block. Larger M (mid
                # / late decoder) → BM=BN=128 amortizes the LDS-load
                # cost across more output rows. Precedence:
                # explicit `block_*_override` > per-shape table >
                # M-based heuristic.
                comptime _BM_heuristic = (64 if _M_total_v <= 1024 else 128)
                comptime _BM_static = (
                    block_m_override if block_m_override
                    > 0 else (_BM_table if _BM_table > 0 else _BM_heuristic)
                )
                comptime _BN_static = (
                    block_n_override if block_n_override
                    > 0 else (_BN_table if _BN_table > 0 else _BM_static)
                )
                # `rows_per_iteration = BM/2` in the loader. The 3D
                # m-decomposition needs `rows_per_iter // (H_out*W_out)
                # < D_out` so the depth carry stays within range
                # (`amd_tile_io_conv.mojo:1085`). Small-spatial /
                # heavily-strided shapes (e.g. `test_conv_gpu`'s D=5,
                # H=W=6, stride=2 → D_out=2, H_out=W_out=4) trip it.
                comptime _loader_3d_ok = (
                    (_BM_static // 2) // (_H_out_v * _W_out_v) < _D_out_v
                )
                # The kernel's load-side `num_iterations` and the
                # schedule's `vm_per_load_*` are both `ceildiv`-based,
                # so BM=64+BK=32 (was: silent corruption from floor-div
                # to 0) is now mathematically correct on the kernel
                # side. The 4-wave path's MFMA-quadrant accumulation
                # order still differs from im2col's row-major K
                # reduction enough to drift on shapes the
                # `test_conv_gpu` CPU-naive reference comparisons
                # were calibrated against, so this routing gate stays
                # as a *perf/test-stability* choice (not a
                # correctness workaround). Drop it once the test set
                # uses an MFMA-equivalent reference (e.g. MIOpen) or
                # the bf16 tolerance is loosened.
                comptime _cout_partial_ok = (
                    (_C_out % 64) == 0 or _BM_static >= 128
                )
                # Wrap the body in a positive `comptime if`. Negative
                # branch returns False — wrapping (rather than an
                # early-return guard) short-circuits type-checking of
                # the body, so shapes the dispatcher intends to
                # decline don't try to instantiate the kernel and
                # trip downstream comptime asserts.
                comptime if (
                    _D_out_v >= 1
                    and _H_out_v >= 1
                    and _W_out_v >= 1
                    and _M_total_v >= 64
                    and (_M_total_v % 64) == 0
                    and _loader_3d_ok
                    and _cout_partial_ok
                ):
                    comptime _ndhwc_in_layout = row_major[
                        _N_static, _D_static, _H_static, _W_static, _C_in
                    ]()
                    comptime _output_2d_layout = row_major[_M_total_v, _C_out]()
                    var input_ndhwc_tt = input.reshape(_ndhwc_in_layout)
                    var output_2d_tt = output.reshape(_output_2d_layout)
                    amd_4wave_conv[
                        elementwise_lambda_fn=elementwise_lambda_fn,
                        block_m_override=_BM_static,
                        block_n_override=_BN_static,
                        block_k_override=_BK_chosen,
                        H=_H_static,
                        W=_W_static,
                        H_out=_H_out_v,
                        W_out=_W_out_v,
                        R=_R,
                        S=_S,
                        stride_h=stride_v,
                        stride_w=stride_v,
                        pad_h=pad_hw_c,
                        pad_w=pad_hw_c,
                        C_in=_C_in,
                        Q=_Q,
                        D=_D_static,
                        D_out=_D_out_v,
                        stride_d=stride_v,
                        pad_d=pad_d_c,
                    ](
                        input_ndhwc_tt,
                        filter_fk_tt,
                        output_2d_tt,
                        ctx,
                    )
                    return True
                return False

            # Enumerate (stride, pad_d, pad_hw) ∈ {1,2} × {0,1,2} × {0,1,2}.
            if stride[0] == 1:
                if pad_d_v == 0:
                    if pad_h_v == 0:
                        if not _launch_static[1, 0, 0]():
                            return False
                    elif pad_h_v == 1:
                        if not _launch_static[1, 0, 1]():
                            return False
                    else:
                        if not _launch_static[1, 0, 2]():
                            return False
                elif pad_d_v == 1:
                    if pad_h_v == 0:
                        if not _launch_static[1, 1, 0]():
                            return False
                    elif pad_h_v == 1:
                        if not _launch_static[1, 1, 1]():
                            return False
                    else:
                        if not _launch_static[1, 1, 2]():
                            return False
                else:
                    if pad_h_v == 0:
                        if not _launch_static[1, 2, 0]():
                            return False
                    elif pad_h_v == 1:
                        if not _launch_static[1, 2, 1]():
                            return False
                    else:
                        if not _launch_static[1, 2, 2]():
                            return False
            else:
                if pad_d_v == 0:
                    if pad_h_v == 0:
                        if not _launch_static[2, 0, 0]():
                            return False
                    elif pad_h_v == 1:
                        if not _launch_static[2, 0, 1]():
                            return False
                    else:
                        if not _launch_static[2, 0, 2]():
                            return False
                elif pad_d_v == 1:
                    if pad_h_v == 0:
                        if not _launch_static[2, 1, 0]():
                            return False
                    elif pad_h_v == 1:
                        if not _launch_static[2, 1, 1]():
                            return False
                    else:
                        if not _launch_static[2, 1, 2]():
                            return False
                else:
                    if pad_h_v == 0:
                        if not _launch_static[2, 2, 0]():
                            return False
                    elif pad_h_v == 1:
                        if not _launch_static[2, 2, 1]():
                            return False
                    else:
                        if not _launch_static[2, 2, 2]():
                            return False
        else:
            # Dynamic-DHW path. Materialize the runtime input dims and
            # output dims, then call the kernel via use_runtime_hw=True.

            @__parameter
            @always_inline
            def _launch_runtime[
                stride_v: Int, pad_d_c: Int, pad_hw_c: Int
            ]() raises:
                var _rt_N = Int(input.dim[0]())
                var _rt_D = Int(input.dim[1]())
                var _rt_H = Int(input.dim[2]())
                var _rt_W = Int(input.dim[3]())
                comptime _eff_Q = _Q
                comptime _eff_R = _R
                comptime _eff_S = _S
                var _rt_D_out = (_rt_D + 2 * pad_d_c - _eff_Q) // stride_v + 1
                var _rt_H_out = (_rt_H + 2 * pad_hw_c - _eff_R) // stride_v + 1
                var _rt_W_out = (_rt_W + 2 * pad_hw_c - _eff_S) // stride_v + 1
                var _rt_M_total = _rt_N * _rt_D_out * _rt_H_out * _rt_W_out
                var _output_dims = IndexList[2](_rt_M_total, _C_out)
                var _dyn_out_layout = row_major(Coord(_output_dims))
                var output_2d_tt = output.reshape(_dyn_out_layout)
                # Runtime-HW shape isn't known at comptime; leave
                # BM/BN at the launcher default (128/128 for bf16) and
                # only set BK from the static K-padding heuristic.
                amd_4wave_conv[
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    block_k_override=_BK_chosen,
                    R=_R,
                    S=_S,
                    stride_h=stride_v,
                    stride_w=stride_v,
                    pad_h=pad_hw_c,
                    pad_w=pad_hw_c,
                    C_in=_C_in,
                    use_runtime_hw=True,
                    Q=_Q,
                    stride_d=stride_v,
                    pad_d=pad_d_c,
                ](
                    input,
                    filter_fk_tt,
                    output_2d_tt,
                    ctx,
                )

            if stride[0] == 1:
                if pad_d_v == 0:
                    if pad_h_v == 0:
                        _launch_runtime[1, 0, 0]()
                    elif pad_h_v == 1:
                        _launch_runtime[1, 0, 1]()
                    else:
                        _launch_runtime[1, 0, 2]()
                elif pad_d_v == 1:
                    if pad_h_v == 0:
                        _launch_runtime[1, 1, 0]()
                    elif pad_h_v == 1:
                        _launch_runtime[1, 1, 1]()
                    else:
                        _launch_runtime[1, 1, 2]()
                else:
                    if pad_h_v == 0:
                        _launch_runtime[1, 2, 0]()
                    elif pad_h_v == 1:
                        _launch_runtime[1, 2, 1]()
                    else:
                        _launch_runtime[1, 2, 2]()
            else:
                if pad_d_v == 0:
                    if pad_h_v == 0:
                        _launch_runtime[2, 0, 0]()
                    elif pad_h_v == 1:
                        _launch_runtime[2, 0, 1]()
                    else:
                        _launch_runtime[2, 0, 2]()
                elif pad_d_v == 1:
                    if pad_h_v == 0:
                        _launch_runtime[2, 1, 0]()
                    elif pad_h_v == 1:
                        _launch_runtime[2, 1, 1]()
                    else:
                        _launch_runtime[2, 1, 2]()
                else:
                    if pad_h_v == 0:
                        _launch_runtime[2, 2, 0]()
                    elif pad_h_v == 1:
                        _launch_runtime[2, 2, 1]()
                    else:
                        _launch_runtime[2, 2, 2]()

        return True
    else:
        return False
