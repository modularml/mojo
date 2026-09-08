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
"""Q-slice 3-D conv → Q sequential SM100 2-D conv calls with fp32
accumulator.

A 3D conv with stride=1, dilation=1 decomposes exactly as:

    output[n, d_o, h_o, w_o, f]
      = Σ_q  [ Σ_{r,s,c}  input[n, d_o+q, h_o+r-pad_h, w_o+s-pad_w, c]
                          * filter[q, r, s, c, f] ]

The inner Σ_{r,s,c} is a 2-D conv on a single depth slice, so by
invoking the existing `dispatch_sm100_conv2d` Q times we inherit SM100
UMMA performance without forking any TMA code.

ACCUMULATION STRATEGY
---------------------
Each SM100 2-D conv produces a bf16 result (the kernel uses an fp32
accumulator internally but stores bf16). A direct attempt to chain
Q convs via `has_residual=True` / `beta=1.0` round-trips the running
sum through bf16 every step, which compounds per-element rounding
far beyond what a single 3-D conv produces (empirically ~100-1000
diff vs the reference on WAN mid_res shapes).

This dispatcher instead maintains a **dedicated fp32 accumulator
buffer** outside the conv calls:

  1. Allocate `accum_fp32` of output size and zero-fill it.
  2. Allocate one reusable `temp_bf16` buffer of output size.
  3. For q in [0, Q):
        - Call `dispatch_sm100_conv2d(has_residual=False)` to write
          `conv(input[q], filter[q])` into `temp_bf16`.
        - Launch an elementwise kernel that reads `temp_bf16`, casts
          to fp32, and adds into `accum_fp32`.
  4. Launch a final kernel that casts `accum_fp32` → bf16 and writes
     to user `output` (fused with the caller's 5-D epilogue when one
     is provided).

Only the final cast is lossy; all intermediate accumulation is fp32,
matching what a single all-at-once 3-D conv would produce internally.

Gate:
- bf16 input/filter/output dtype.
- SM100 device (`_is_sm10x_gpu`).
- `filter_is_fcrs=False` (QRSCF only: the per-q slab is a
  contiguous RSCF view at offset `q*R*S*C*F`; FCQRS would need a
  separate extraction kernel because a fixed-q FCQRS slice is
  non-contiguous).
- stride=1, dilation=1, groups=1, Q>1.
- Zero temporal padding (WAN's causal 3D convs pre-pad temporal
  externally).
- `C_in % 64 == 0` and `C_out % 64 == 0` (SM100 alignment).

Declined shapes fall through to `dispatch_im2col_matmul_conv3d`.
"""

from std.collections import OptionalReg
from std.math import ceildiv, gcd
from std.math.uutils import udivmod
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from max.gpu.host.info import _is_sm10x_gpu
from layout import Coord, Idx, TileTensor, row_major
from std.sys import align_of, simd_width_of
from std.utils import IndexList
from linalg.utils import elementwise_epilogue_type
from nn.conv.conv_utils import elementwise_simd_epilogue_type

from .dispatch import dispatch_sm100_conv2d, test_alignment_sm100_conv2d


# =========================================================================
# Elementwise helper kernels
# =========================================================================


@__name(t"qslice_accum_bf16_to_fp32_{dtype}")
def _accum_bf16_to_fp32_kernel[
    dtype: DType,
    output_simd_width: Int,
](
    accum_fp32_ptr: UnsafePointer[Float32, MutAnyOrigin],
    src_bf16_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    per_batch_elems: Int32,
):
    """Elementwise `accum_fp32[i] += src_bf16[i].cast[fp32]()`.

    One thread per element; no atomics. Each thread owns its slot.
    """
    var _per_batch_elems = Int(per_batch_elems)
    comptime bf16_alignment = align_of[SIMD[dtype, output_simd_width]]()
    comptime fp32_alignment = align_of[SIMD[.float32, output_simd_width]]()

    var accum_idx = global_idx.x * output_simd_width
    if accum_idx >= _per_batch_elems:
        return
    var src_val = src_bf16_ptr.load[
        width=output_simd_width, alignment=bf16_alignment
    ](accum_idx).cast[.float32]()
    var accum_val = accum_fp32_ptr.load[
        width=output_simd_width, alignment=fp32_alignment
    ](accum_idx)
    accum_fp32_ptr.store[alignment=fp32_alignment](
        accum_idx, accum_val + src_val
    )


@__name(t"qslice_fp32_to_dtype_plain_{dtype}")
def _fp32_to_dtype_plain_kernel[
    dtype: DType,
    output_simd_width: Int,
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src_fp32_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    output_elems: Int32,
):
    """Elementwise cast fp32 → `dtype` with no epilogue."""
    var _output_elems = Int(output_elems)
    var output_idx = global_idx.x * output_simd_width
    if output_idx >= _output_elems:
        return
    dst_ptr.store[alignment=align_of[SIMD[dtype, output_simd_width]]()](
        output_idx,
        src_fp32_ptr.load[
            width=output_simd_width,
            alignment=align_of[SIMD[.float32, output_simd_width]](),
        ](output_idx).cast[dtype](),
    )


@__name(t"qslice_fp32_to_dtype_epilogue_{dtype}")
def _fp32_to_dtype_epilogue_kernel[
    dtype: DType,
    C_out: Int,
    epilogue: elementwise_simd_epilogue_type,
    output_simd_width: Int,
](
    src_fp32_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    batch: Int32,
    D_out: Int32,
    H_out: Int32,
    W_out: Int32,
    output_elems: Int32,
):
    """Elementwise cast fp32 → `dtype`, then call the caller's 5-D
    epilogue. The epilogue is expected to perform the write
    (matching the MOGG `output._lambda_store` contract).
    """
    var _batch = Int(batch)
    var _D_out = Int(D_out)
    var _H_out = Int(H_out)
    var _W_out = Int(W_out)
    var _output_elems = Int(output_elems)
    var output_idx = global_idx.x * output_simd_width
    if output_idx >= _output_elems:
        return
    var DHW_out = _D_out * _H_out * _W_out
    var HW_out = _H_out * _W_out
    var b, rem = udivmod(output_idx, DHW_out * C_out)
    var d: Int
    d, rem = udivmod(rem, HW_out * C_out)
    var h: Int
    h, rem = udivmod(rem, _W_out * C_out)
    var w: Int
    var c: Int
    w, c = udivmod(rem, C_out)
    var val = src_fp32_ptr.load[
        width=output_simd_width,
        alignment=align_of[SIMD[.float32, output_simd_width]](),
    ](output_idx).cast[dtype]()
    epilogue[alignment=output_simd_width](IndexList[5](b, d, h, w, c), val)


# =========================================================================
# Public dispatch entry point
# =========================================================================


def dispatch_qslice_conv3d_sm100[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    //,
    filter_is_fcrs: Bool = False,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
](
    input: TileTensor[mut=True, input_type, address_space=.GENERIC, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    symmetric_padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Try to dispatch a 3-D conv as Q × SM100 2-D conv calls with a
    dedicated fp32 accumulator.

    Parameters:
        input_type: The element type of `input` (inferred). The gate
            requires `DType.bfloat16`.
        filter_type: The element type of `filter` (inferred).
        output_type: The element type of `output` (inferred). The
            gate requires `DType.bfloat16`.
        filter_is_fcrs: Whether `filter` is laid out as FCQRS rather
            than QRSCF (defaults to `False`). The gate requires
            `False` because a fixed-`q` FCQRS slab is non-contiguous.
        maybe_epilogue_func: Optional 5-D elementwise epilogue fused
            into the final fp32-to-`output_type` write (defaults to
            `None`).

    Args:
        input: Rank-5 input tensor in NDHWC layout.
        filter: Rank-5 filter tensor in QRSCF layout, shaped
            `[Q, R, S, C_in, C_out]`.
        output: Rank-5 mutable output tensor in NDHWC layout.
        stride: Per-axis stride for the `(D, H, W)` dimensions.
            The gate requires `(1, 1, 1)`.
        dilation: Per-axis dilation for the `(D, H, W)` dimensions.
            The gate requires `(1, 1, 1)`.
        symmetric_padding: Symmetric padding for the `(D, H, W)`
            dimensions. Temporal (`D`) padding must be zero.
        num_groups: Number of convolution groups. The gate
            requires `1`.
        ctx: Device context used for buffer allocation and kernel
            launches.
    """
    comptime assert input.flat_rank == 5, "input must be rank 5 (NDHWC)"
    comptime assert filter.flat_rank == 5, "filter must be rank 5"
    comptime assert output.flat_rank == 5, "output must be rank 5 (NDHWC)"

    comptime if not filter.shape_known:
        return False

    comptime if input_type != .bfloat16:
        return False
    comptime if output_type != .bfloat16:
        return False

    # FCQRS slab extraction would need a dedicated kernel (a fixed-q
    # slice through FCQRS is non-contiguous). QRSCF slab q is just
    # RSCF at offset q*R*S*C*F so we can lean on dispatch_sm100_conv2d.
    comptime if filter_is_fcrs:
        return False

    if num_groups != 1:
        return False
    if stride[0] != 1 or stride[1] != 1 or stride[2] != 1:
        return False
    if dilation[0] != 1 or dilation[1] != 1 or dilation[2] != 1:
        return False

    comptime _is_sm100 = _is_sm10x_gpu(ctx.default_device_info)
    comptime if not _is_sm100:
        return False

    if symmetric_padding[0] != 0:
        return False

    comptime Q = filter.static_shape[0]
    comptime R = filter.static_shape[1]
    comptime S = filter.static_shape[2]

    comptime if Q <= 1:
        return False

    # R=S=1 degenerates the per-q 2-D conv into a matmul. Running Q
    # sequential SM100 conv2d calls (each with its own filter
    # transpose + internal synchronize) is structurally slower than
    # im2col+matmul's single fused matmul on these shapes — im2col
    # collapses R*S*Q into the K dimension and runs one GEMM. Decline
    # so we fall through to the im2col path.
    comptime if R == 1 and S == 1:
        return False

    var batch = Int(input.dim[0]())
    var D = Int(input.dim[1]())
    var H = Int(input.dim[2]())
    var W = Int(input.dim[3]())

    var D_out = Int(output.dim[1]())
    var H_out = Int(output.dim[2]())
    var W_out = Int(output.dim[3]())

    comptime C_in = filter.static_shape[3]
    comptime C_out = filter.static_shape[4]

    # Check the alignment of the channel parameters for `dispatch_sm100_conv2d`.
    comptime if not test_alignment_sm100_conv2d[input_type, output_type](
        C_in, C_out
    ):
        return False

    if D_out != D - Q + 1:
        return False

    var pad_h = symmetric_padding[1]
    var pad_w = symmetric_padding[2]
    var symmetric_padding_2d = IndexList[2](pad_h, pad_w)

    comptime output_simd_width = gcd(
        simd_width_of[output_type, target=get_gpu_target()](), C_out
    )

    var per_batch_elems = D_out * H_out * W_out * C_out
    var output_elems = batch * per_batch_elems

    # --- 1. Allocate fp32 accumulator (zeroed) + reusable bf16 temp. ---
    var accum_fp32_buf = ctx.enqueue_create_buffer[.float32](output_elems)
    accum_fp32_buf.enqueue_fill(Float32(0.0))
    var accum_fp32_ptr = accum_fp32_buf.unsafe_ptr()

    var temp_bf16_buf = ctx.enqueue_create_buffer[output_type](per_batch_elems)
    # Redundant for correctness (conv2d fully overwrites this via TMA store);
    # works around initcheck not tracking TMA bulk stores as initializing writes.
    temp_bf16_buf.enqueue_fill(Scalar[output_type](0))

    # --- 2. Per-(n, q) conv into temp_bf16, then accumulate into fp32. ---
    comptime accum_block = 256

    for n_batch in range(batch):
        var accum_n_offset = n_batch * per_batch_elems
        var accum_n_ptr = accum_fp32_ptr + accum_n_offset

        for q in range(Q):
            var input_offset = input.layout[
                linear_idx_type=input.linear_idx_type
            ](Coord(n_batch, q, 0, 0, 0))
            var filter_offset = filter.layout[
                linear_idx_type=filter.linear_idx_type
            ](Coord(q, 0, 0, 0, 0))
            var act_tt = TileTensor(
                input._offset_storage(input_offset),
                row_major(D_out, H, W, C_in),
            )
            var filter_rscf_tt = TileTensor(
                filter._offset_storage(filter_offset),
                row_major(R, S, C_in, C_out),
            )
            var temp_tt = TileTensor(
                temp_bf16_buf,
                row_major(D_out, H_out, W_out, C_out),
            )

            dispatch_sm100_conv2d(
                act_tt,
                filter_rscf_tt,
                temp_tt,
                symmetric_padding_2d,
                ctx,
            )

            # Accumulate: accum_fp32_n += temp_bf16.cast[fp32]().
            var accum_grid = ceildiv(
                per_batch_elems // output_simd_width, accum_block
            )
            comptime accum_kernel = _accum_bf16_to_fp32_kernel[
                output_type, output_simd_width
            ]
            ctx.enqueue_function[accum_kernel](
                accum_n_ptr,
                temp_bf16_buf,
                Int32(per_batch_elems),
                grid_dim=accum_grid,
                block_dim=accum_block,
            )

    # --- 3. Final fp32 → bf16 write to user output (fuse epilogue). ---
    comptime final_block = 256
    var final_grid = ceildiv(output_elems // output_simd_width, final_block)

    comptime if maybe_epilogue_func:
        comptime epilogue_5d = maybe_epilogue_func.value()
        comptime output_kernel = _fp32_to_dtype_epilogue_kernel[
            output_type, C_out, epilogue_5d, output_simd_width
        ]
        ctx.enqueue_function[output_kernel](
            accum_fp32_ptr,
            Int32(batch),
            Int32(D_out),
            Int32(H_out),
            Int32(W_out),
            Int32(output_elems),
            grid_dim=final_grid,
            block_dim=final_block,
        )
    else:
        comptime output_kernel = _fp32_to_dtype_plain_kernel[
            output_type, output_simd_width
        ]
        ctx.enqueue_function[output_kernel](
            output.ptr,
            accum_fp32_ptr,
            Int32(output_elems),
            grid_dim=final_grid,
            block_dim=final_block,
        )

    return True
