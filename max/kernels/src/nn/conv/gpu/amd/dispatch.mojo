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
"""AMD MI355X (CDNA4, gfx950) conv2d dispatch to `amd_4wave_conv`.

Returns True when the input/filter/output shapes + runtime stride / pad
/ dilation are handled by the 4-wave conv kernel; False to fall back to
the caller's MIOpen path. Acceptance rules:

  - Hardware: MI355X (gfx950) only (the 4-wave kernel inherits the
    chiplet/L2 swizzle and MFMA shapes specific to CDNA4).
  - Input dtype: float8_e4m3fn, bfloat16, or float16. Output dtype:
    bfloat16 for FP8, otherwise tracks input.
  - All input / filter / output spatial shapes must be **static**
    (TileTensor `static_shape[i] >= 0`). Dynamic-shape conv shapes
    fall through to MIOpen.
  - `num_groups == 1` (4-wave conv is single-group).
  - `dilation == (1, 1)`.
  - `stride ∈ {(1, 1), (2, 2)}`, with `stride[0] == stride[1]`.
  - `symmetric_padding ∈ {(0, 0), (1, 1), (2, 2)}`, square pad only.

When accepted, the dispatcher:

  1. Allocates a K-padded FRSC filter buffer (zero-filled trailing
     K columns when `R*S*C_in` isn't a multiple of `2*BK = 256`).
  2. Transposes the caller's filter (FCRS or RSCF) into FRSC.
  3. Comptime-materializes (stride, pad) and calls `amd_4wave_conv`
     with the appropriate kernel template parameters.

Mirrors the structure of `nn.conv.gpu.amd.rdna.dispatch` and
`nn.conv.gpu.nvidia.sm100.dispatch`.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import simd_width_of
from std.sys.info import _accelerator_arch
from std.utils import IndexList

from layout import Coord, TileTensor, row_major

from linalg.utils import elementwise_epilogue_type

from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv


# ---------------------------------------------------------------------------
# Filter transposes to FRSC, with K-padding zero-fill.
# ---------------------------------------------------------------------------


@__name(t"amd_4wave_transpose_rscf_to_frsc_kpad_{dtype}")
def _transpose_rscf_to_frsc_kpad[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    R: Int32,
    S: Int32,
    C: Int32,
    F: Int32,
    K_padded: Int32,
):
    """GPU kernel: filter RSCF `[R*S*C, F]` -> FRSC `[F, K_padded]`.

    `K_padded >= R*S*C`; padded trailing columns are zero-filled.
    """
    var _R = Int(R)
    var _S = Int(S)
    var _C = Int(C)
    var _F = Int(F)
    var _K_padded = Int(K_padded)
    var K_real = _R * _S * _C
    var total = _F * _K_padded
    var tid = global_idx.x
    if tid >= total:
        return
    var f = tid // _K_padded
    var k = tid - f * _K_padded
    if k >= K_real:
        dst_ptr.store(tid, Scalar[dtype](0))
        return
    var r = k // (_S * _C)
    var sc = k - r * (_S * _C)
    var s = sc // _C
    var c = sc - s * _C
    # RSCF source index: ((r * _S + s) * _C + c) * _F + f.
    var rscf_idx = ((r * _S + s) * _C + c) * _F + f
    dst_ptr.store(tid, src_ptr.load(rscf_idx))


@__name(t"amd_4wave_transpose_fcrs_to_frsc_kpad_{dtype}")
def _transpose_fcrs_to_frsc_kpad[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    F: Int32,
    C: Int32,
    R: Int32,
    S: Int32,
    K_padded: Int32,
):
    """GPU kernel: filter FCRS `[F, C, R, S]` -> FRSC `[F, K_padded]`.

    `K_padded >= R*S*C`; padded trailing columns are zero-filled.
    """
    var _F = Int(F)
    var _C = Int(C)
    var _R = Int(R)
    var _S = Int(S)
    var _K_padded = Int(K_padded)
    var K_real = _R * _S * _C
    var total = _F * _K_padded
    var tid = global_idx.x
    if tid >= total:
        return
    var f = tid // _K_padded
    var k = tid - f * _K_padded
    if k >= K_real:
        dst_ptr.store(tid, Scalar[dtype](0))
        return
    var r = k // (_S * _C)
    var sc = k - r * (_S * _C)
    var s = sc // _C
    var c = sc - s * _C
    # FCRS source index: f*_C*_R*_S + c*_R*_S + r*_S + s.
    var fcrs_idx = f * _C * _R * _S + c * _R * _S + r * _S + s
    dst_ptr.store(tid, src_ptr.load(fcrs_idx))


# ---------------------------------------------------------------------------
# Runtime-HW launcher (separated so the dispatcher's static path
# doesn't pay compile-time for runtime-HW kernel specializations).
# ---------------------------------------------------------------------------


@always_inline
def _launch_amd_4wave_conv2d_runtime[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    //,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    K_padded: Int,
    has_residual: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    input: TileTensor[input_type, ...],
    filter_frsc_tt: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: Int,
    pad: Int,
    ctx: DeviceContext,
    source_ptr: UnsafePointer[
        Scalar[output_type], ImmutAnyOrigin
    ] = UnsafePointer[Scalar[output_type], ImmutAnyOrigin].unsafe_dangling(),
    beta: Float32 = 0.0,
) raises -> Bool:
    """Runtime-HW launch for the 4-wave conv dispatcher.

    Materializes `(stride, pad)` to a comptime pair and invokes the
    runtime-HW path of `amd_4wave_conv`. Returns False only for
    unreachable (defensive) (stride, pad) combinations; the outer
    dispatcher's runtime gate handles validity.

    When `has_residual=True`, plumbs `source_ptr` (NHWC-contiguous,
    same shape as output) + `beta` through to `amd_4wave_conv` for
    fused in-kernel `D = Conv + beta * source`. The 2D NHWC view's
    row stride is `C_out` (NHWC contiguous).
    """

    @__parameter
    @always_inline
    def _launch[stride_v: Int, pad_v: Int]() raises:
        var _rt_N = Int(input.dim[0]())
        var _rt_H = Int(input.dim[1]())
        var _rt_W = Int(input.dim[2]())
        comptime _eff_R = R
        comptime _eff_S = S
        var _rt_H_out = (_rt_H + 2 * pad_v - _eff_R) // stride_v + 1
        var _rt_W_out = (_rt_W + 2 * pad_v - _eff_S) // stride_v + 1
        var _rt_M_total = _rt_N * _rt_H_out * _rt_W_out
        # Dynamic 2D `[M_total, C_out]` output view.
        var _output_dims = IndexList[2](_rt_M_total, C_out)
        var _dyn_out_layout = row_major(Coord(_output_dims))
        var output_2d_tt = output.reshape(_dyn_out_layout)
        amd_4wave_conv[
            elementwise_lambda_fn=elementwise_lambda_fn,
            R=R,
            S=S,
            stride_h=stride_v,
            stride_w=stride_v,
            pad_h=pad_v,
            pad_w=pad_v,
            C_in=C_in,
            use_runtime_hw=True,
            has_residual=has_residual,
        ](
            input,
            filter_frsc_tt,
            output_2d_tt,
            ctx,
            source_ptr=source_ptr,
            source_row_stride=C_out,
            beta=beta,
        )

    if stride == 1 and pad == 0:
        _launch[1, 0]()
    elif stride == 1 and pad == 1:
        _launch[1, 1]()
    elif stride == 1 and pad == 2:
        _launch[1, 2]()
    elif stride == 2 and pad == 0:
        _launch[2, 0]()
    elif stride == 2 and pad == 1:
        _launch[2, 1]()
    elif stride == 2 and pad == 2:
        _launch[2, 2]()
    else:
        return False
    return True


# ---------------------------------------------------------------------------
# Dispatch entry point.
# ---------------------------------------------------------------------------


def dispatch_amd_4wave_conv2d[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool,
    has_residual: Bool = False,
    # Optional fused 2D-coord epilogue lambda. The caller is expected
    # to have wrapped any 4D NHWC `elementwise_simd_epilogue_type` into
    # this 2D-coord form (m=b*H_out*W_out+h*W_out+w, n=channel). See
    # the SM100 dispatch site in `nn/conv/conv.mojo` for the wrapper
    # template. When None, no epilogue fuses; pure conv (+ residual).
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    symmetric_padding: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
    source_ptr: Optional[
        UnsafePointer[Scalar[output_type], MutAnyOrigin]
    ] = None,
    beta: Float32 = 0.0,
) raises -> Bool:
    """Try to dispatch a Conv2D to `amd_4wave_conv` on MI355X.

    Returns True if the convolution was handled; False if the caller
    should fall back (typically to MIOpen). See module docstring for the
    full acceptance criteria.

    When `has_residual=True` and `source_ptr` is set, computes
    `D = Conv(A, B) + beta * source` via the in-kernel fused residual
    path (`amd_4wave_conv[has_residual=True]`). The source pointer is
    expected to point to an NHWC-contiguous buffer with the same shape
    as `output`. When `has_residual=False` (default), the call is
    identical to the no-residual variant: no extra ABI overhead beyond
    the launch packet's 16 bytes (DCE'd source_ptr / stride / beta).

    Parameters:
        input_type: `DType` of the input activation tensor; must be
            `float8_e4m3fn`, `bfloat16`, or `float16`.
        filter_type: `DType` of the filter tensor; must equal
            `input_type`.
        output_type: `DType` of the output tensor; `bfloat16` for FP8
            inputs, otherwise tracks `input_type`.
        filter_is_fcrs: `True` when `filter` is in FCRS layout; `False`
            for RSCF. Selects the transpose path and the R/S dim indices.
        has_residual: `True` to fuse `D = Conv + beta * source` in-kernel
            via `source_ptr` (defaults to `False`).
        elementwise_lambda_fn: Optional fused 2D-coord epilogue lambda
            wrapping a 4D NHWC `elementwise_simd_epilogue_type`; `None`
            fuses no epilogue (defaults to `None`).

    Args:
        input: Input activation `TileTensor`, rank-4 NHWC with static
            `C_in`; `N`, `H`, `W` may be dynamic.
        filter: Filter `TileTensor`, rank-4 FCRS or RSCF per
            `filter_is_fcrs`, with static `R`, `S`, `C_in`, `C_out`.
        output: Output activation `TileTensor`, rank-4 NHWC, mutable,
            with static `C_out`.
        stride: Convolution stride `[stride_h, stride_w]`; must be
            `(1, 1)` or `(2, 2)` with `stride_h == stride_w`.
        dilation: Dilation `[dh, dw]`; must be `(1, 1)`.
        symmetric_padding: Symmetric padding `[ph, pw]`; must be
            `(0, 0)`, `(1, 1)`, or `(2, 2)` with `ph == pw`.
        num_groups: Number of convolution groups; must be `1`.
        ctx: `DeviceContext` for kernel launches and buffer allocation.
        source_ptr: Optional pointer to an NHWC-contiguous residual
            source buffer with the same shape as `output`; read only
            when `has_residual=True` (defaults to `None`).
        beta: Scale factor for the residual term in
            `D = Conv + beta * source` (defaults to `0.0`).
    """
    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"

    # -------- Hardware gate --------
    # Host-side check on the build target's accelerator string. The
    # `_is_amd_mi355x()` helper is GPU-only (false in host code); the
    # accelerator-arch string is "amdgpu:gfx950" on MI355X.
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
    # Required comptime: C_in, C_out, R, S, stride, dilation, pad — the
    # 4-wave conv kernel uses these in MMA shape selection and the
    # `(kh, kw, c)` substrip decomposition. N_batch, H, W may be
    # dynamic (e.g. FLUX VAE with symbolic image resolution); the
    # kernel then routes through the `use_runtime_hw=True` path which
    # reads them from input.dim() at runtime. C_in / C_out / R / S
    # being dynamic falls back to MIOpen.
    comptime _C_in = input.static_shape[3]
    comptime _C_out = output.static_shape[3]
    comptime _N_static = input.static_shape[0]
    comptime _H_static = input.static_shape[1]
    comptime _W_static = input.static_shape[2]
    comptime _all_hw_static = (
        _N_static >= 0 and _H_static >= 0 and _W_static >= 0
    )
    # Filter R, S come from different filter dims based on layout.
    comptime _R = (
        filter.static_shape[2] if filter_is_fcrs else filter.static_shape[0]
    )
    comptime _S = (
        filter.static_shape[3] if filter_is_fcrs else filter.static_shape[1]
    )

    # `C_in` lower bound (per-lane substrip soundness): the per-lane
    # substrip loader code path emits one `buffer_load_*_lds` per lane
    # of `load_width = simd_width_of[dtype]` contiguous NHWC elements,
    # under the assumption that all `load_width` elements live in the
    # same `(kh, kw)` substrip. When `C_in < load_width` (e.g. FLUX.2
    # VAE stem: C_in=3 with BF16 load_width=8), a lane's 8-element
    # load crosses pixel boundaries — so the input data landing at
    # LDS positions `(k_lane+1 .. k_lane+load_width-1)` does NOT
    # correspond to the `(kh, kw, c)` the schedule assigns those
    # slots, and the MMA reduces against mis-mapped activations.
    # The filter zero-padding only protects positions `k >= K_real`,
    # not in-range positions whose loaded data is from the wrong
    # pixel. Until the per-lane loader is rewritten to align loads
    # to substrip boundaries, gate this case off so the caller falls
    # back to MIOpen.
    comptime _simd_w = simd_width_of[input_type]()
    comptime _shapes_ok = (
        _C_in >= _simd_w and _C_out >= 64 and _R > 0 and _S > 0
    )
    comptime if _shapes_ok:
        # -------- Runtime constraints --------
        if num_groups != 1:
            return False
        if dilation[0] != 1 or dilation[1] != 1:
            return False
        if stride[0] != stride[1]:
            return False
        if symmetric_padding[0] != symmetric_padding[1]:
            return False
        if stride[0] != 1 and stride[0] != 2:
            return False
        var pad = symmetric_padding[0]
        if pad != 0 and pad != 1 and pad != 2:
            return False

        # -------- Filter repack: caller layout -> FRSC, K-padded -----
        comptime _K_real = _R * _S * _C_in
        comptime _K_padded = ((_K_real + 255) // 256) * 256
        var filter_frsc_buf = ctx.enqueue_create_buffer[filter_type](
            _C_out * _K_padded
        )
        var filter_frsc_ptr = filter_frsc_buf.unsafe_ptr()
        comptime _transpose_block = 256
        var transpose_total = _C_out * _K_padded
        var transpose_grid = ceildiv(transpose_total, _transpose_block)
        comptime if filter_is_fcrs:
            ctx.enqueue_function[_transpose_fcrs_to_frsc_kpad[filter_type]](
                filter.ptr,
                filter_frsc_ptr,
                Int32(_C_out),
                Int32(_C_in),
                Int32(_R),
                Int32(_S),
                Int32(_K_padded),
                grid_dim=transpose_grid,
                block_dim=_transpose_block,
            )
        else:
            ctx.enqueue_function[_transpose_rscf_to_frsc_kpad[filter_type]](
                filter.ptr,
                filter_frsc_ptr,
                Int32(_R),
                Int32(_S),
                Int32(_C_in),
                Int32(_C_out),
                Int32(_K_padded),
                grid_dim=transpose_grid,
                block_dim=_transpose_block,
            )

        # -------- Filter view: FRSC, K-padded (built above) ---------
        comptime _filter_frsc_layout = row_major[_C_out, _K_padded]()
        var filter_frsc_tt = TileTensor(filter_frsc_ptr, _filter_frsc_layout)

        # Residual placeholder: when has_residual=False, source_ptr may
        # be None or unused. We materialize a dangling placeholder for
        # the kernel arg (which only reads it when has_residual=True at
        # comptime, so the value is never dereferenced in the
        # has_residual=False path).
        var _src_ptr_immut = UnsafePointer[
            Scalar[output_type], ImmutAnyOrigin
        ].unsafe_dangling()

        @__parameter
        @always_inline
        def _src_immut() -> UnsafePointer[Scalar[output_type], ImmutAnyOrigin]:
            comptime if has_residual:
                # When called with has_residual=True, the caller must
                # supply source_ptr. Materialize an immutable view of it
                # (the kernel only reads).
                if source_ptr:
                    return UnsafePointer[Scalar[output_type], ImmutAnyOrigin](
                        unsafe_from_address=Int(source_ptr.value())
                    )
            return _src_ptr_immut

        var _src_ptr_for_kernel = _src_immut()

        # -------- Static vs runtime-HW dispatch --------------------
        comptime if _all_hw_static:

            @__parameter
            @always_inline
            def _launch_static[stride_v: Int, pad_v: Int]() raises -> Bool:
                comptime _eff_R = _R
                comptime _eff_S = _S
                comptime _H_out_v = (
                    _H_static + 2 * pad_v - _eff_R
                ) // stride_v + 1
                comptime _W_out_v = (
                    _W_static + 2 * pad_v - _eff_S
                ) // stride_v + 1
                comptime _M_total_v = (_N_static * _H_out_v * _W_out_v)
                # Comptime gate: M must be a positive multiple of 64
                # (block size) for the 4-wave tiling to launch. When the
                # gate fails, return False so the outer dispatcher
                # surfaces the rejection — the caller then falls back to
                # MIOpen. (Previous behavior was a silent `return` while
                # the outer kept saying True, leaving `output`
                # uninitialized.)
                comptime if (
                    _H_out_v < 1
                    or _W_out_v < 1
                    or _M_total_v < 64
                    or (_M_total_v % 64) != 0
                ):
                    return False
                comptime _nhwc_in_layout = row_major[
                    _N_static, _H_static, _W_static, _C_in
                ]()
                comptime _output_2d_layout = row_major[_M_total_v, _C_out]()
                var input_nhwc_tt = input.reshape(_nhwc_in_layout)
                var output_2d_tt = output.reshape(_output_2d_layout)
                amd_4wave_conv[
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    H=_H_static,
                    W=_W_static,
                    H_out=_H_out_v,
                    W_out=_W_out_v,
                    R=_R,
                    S=_S,
                    stride_h=stride_v,
                    stride_w=stride_v,
                    pad_h=pad_v,
                    pad_w=pad_v,
                    C_in=_C_in,
                    has_residual=has_residual,
                ](
                    input_nhwc_tt,
                    filter_frsc_tt,
                    output_2d_tt,
                    ctx,
                    source_ptr=_src_ptr_for_kernel,
                    source_row_stride=_C_out,
                    beta=beta,
                )
                return True

            if stride[0] == 1 and pad == 0:
                if not _launch_static[1, 0]():
                    return False
            elif stride[0] == 1 and pad == 1:
                if not _launch_static[1, 1]():
                    return False
            elif stride[0] == 1 and pad == 2:
                if not _launch_static[1, 2]():
                    return False
            elif stride[0] == 2 and pad == 0:
                if not _launch_static[2, 0]():
                    return False
            elif stride[0] == 2 and pad == 1:
                if not _launch_static[2, 1]():
                    return False
            elif stride[0] == 2 and pad == 2:
                if not _launch_static[2, 2]():
                    return False
            else:
                return False
        else:
            if not _launch_amd_4wave_conv2d_runtime[
                C_in=_C_in,
                C_out=_C_out,
                R=_R,
                S=_S,
                K_padded=_K_padded,
                has_residual=has_residual,
                elementwise_lambda_fn=elementwise_lambda_fn,
            ](
                input,
                filter_frsc_tt,
                output,
                stride[0],
                pad,
                ctx,
                source_ptr=_src_ptr_for_kernel,
                beta=beta,
            ):
                return False
        return True
    else:
        return False
