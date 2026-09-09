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
"""SM100 Conv2D dispatch for the nn conv op.

This module provides a dtype-safe dispatch function that gates SM100
conv2d kernel instantiation to supported dtypes (bf16/fp16) only.
Importing this module does NOT trigger kernel compilation -- the kernel
is only compiled when `dispatch_sm100_conv2d` is called with a supported
dtype inside a comptime if guard.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.sys import size_of
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import Idx, TileTensor, row_major
from std.utils.index import IndexList
from linalg.utils import elementwise_epilogue_type


# =========================================================================
# Filter transpose kernels (no SM100 deps, safe for any dtype)
# =========================================================================


@__name(t"transpose_rscf_to_krsc_{dtype}")
def _transpose_rscf_to_krsc[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    R: Int32,
    S: Int32,
    C: Int32,
    F: Int32,
):
    """GPU kernel: transpose filter RSCF [R,S,C,F] -> KRSC [K,R,S,C]."""
    var _R = Int(R)
    var _S = Int(S)
    var _C = Int(C)
    var _F = Int(F)
    var tid = global_idx.x
    if tid >= _R * _S * _C * _F:
        return
    var k, rem = divmod(tid, _R * _S * _C)
    var r: Int
    r, rem = divmod(rem, _S * _C)
    var s, c = divmod(rem, _C)
    var src_idx = r * _S * _C * _F + s * _C * _F + c * _F + k
    dst_ptr.store(tid, src_ptr.load(src_idx))


@__name(t"transpose_fcrs_to_krsc_{dtype}")
def _transpose_fcrs_to_krsc[
    dtype: DType,
](
    src_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    F: Int32,
    C: Int32,
    R: Int32,
    S: Int32,
):
    """GPU kernel: transpose filter FCRS [F,C,R,S] -> KRSC [K,R,S,C]."""
    var _F = Int(F)
    var _C = Int(C)
    var _R = Int(R)
    var _S = Int(S)
    var tid = global_idx.x
    if tid >= _F * _C * _R * _S:
        return
    var k, rem = divmod(tid, _R * _S * _C)
    var r: Int
    r, rem = divmod(rem, _S * _C)
    var s, c = divmod(rem, _C)
    var src_idx = k * _C * _R * _S + c * _R * _S + r * _S + s
    dst_ptr.store(tid, src_ptr.load(src_idx))


# =========================================================================
# SM100 Conv2D dispatch
# =========================================================================


@always_inline
def test_alignment_sm100_conv2d[
    input_type: DType, output_type: DType
](in_channels: Int, out_channels: Int) -> Bool:
    """Checks whether the input and output channel counts meet SM100 conv2d alignment requirements.

    Returns True when the input activation row is 64-byte aligned and the
    output row is 4-byte aligned, otherwise False.

    Parameters:
        input_type: Element type of the input activation tensor, used to
            compute the per-row byte width for the 64-byte alignment check.
        output_type: Element type of the output tensor, used to compute the
            per-row byte width for the 4-byte alignment check.

    Args:
        in_channels: Number of input channels per activation row.
        out_channels: Number of output channels per output row.

    Returns:
        True if both alignment constraints are satisfied, False otherwise.
    """
    return (in_channels * size_of[input_type]()) % 64 == 0 and (
        out_channels * size_of[output_type]()
    ) % 4 == 0


def dispatch_sm100_conv2d[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    //,
    filter_is_fcrs: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    has_residual: Bool = False,
](
    input: TileTensor[mut=True, input_type, address_space=.GENERIC, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[mut=True, output_type, ...],
    symmetric_padding: IndexList[2],
    ctx: DeviceContext,
    source_ptr: OptionalReg[
        UnsafePointer[Scalar[output_type], MutAnyOrigin]
    ] = None,
    beta: Float32 = 0.0,
) raises:
    """Dispatch to SM100 structured conv2d with filter transpose.

    This function gates the SM100 kernel import behind comptime if
    on dtype, so the kernel is never compiled for unsupported dtypes.

    Parameters:
        input_type: Data type of the input activation tensor.
        filter_type: Data type of the filter weights tensor.
        output_type: Data type of the output tensor.
        filter_is_fcrs: If True, filter is FCRS layout; otherwise RSCF.
        elementwise_lambda_fn: Optional void epilogue lambda applied after
            output write. Signature: `def(IndexList[2], SIMD) -> None`.
        has_residual: If True, fuse residual add D = Conv(A,B) + beta*C.

    Args:
        input: Input activation tensor in NHWC layout.
        filter: Filter weights tensor.
        output: Output tensor in NHWC layout.
        symmetric_padding: Symmetric padding (pad_h, pad_w).
        ctx: Device context for kernel launch.
        source_ptr: Pointer to residual source tensor C (NHWC, same shape
            as output). Only used when has_residual is True.
        beta: Residual scale factor. D = Conv(A,B) + beta*C.

    Raises:
        Error if kernel launch fails.
    """
    comptime assert input.flat_rank == 4, "input must be rank 4 (NHWC)"
    comptime assert filter.flat_rank == 4, "filter must be rank 4"
    comptime assert output.flat_rank == 4, "output must be rank 4 (NHWC)"

    comptime if input_type == .bfloat16:
        from .conv2d import conv2d_fprop, conv2d_fprop_with_residual
        from .conv_config import Conv2dConfig, Conv2dProblemShape

        # Extract dimensions
        var batch = Int(input.dim[0]())
        var in_h = Int(input.dim[1]())
        var in_w = Int(input.dim[2]())
        var in_c = Int(input.dim[3]())
        var out_h = Int(output.dim[1]())
        var out_w = Int(output.dim[2]())
        var out_c = Int(output.dim[3]())

        var fh: Int
        var fw: Int

        comptime if filter_is_fcrs:
            fh = Int(filter.dim[2]())
            fw = Int(filter.dim[3]())
        else:
            fh = Int(filter.dim[0]())
            fw = Int(filter.dim[1]())

        # Transpose filter to KRSC layout
        var filter_size = filter.num_elements()
        var filter_buf = ctx.enqueue_create_buffer[filter_type](filter_size)
        var filter_krsc_ptr = filter_buf.unsafe_ptr()

        comptime transpose_block = 256
        var grid = ceildiv(filter_size, transpose_block)

        comptime if filter_is_fcrs:
            var F = Int(filter.dim[0]())
            var C = Int(filter.dim[1]())
            var R = Int(filter.dim[2]())
            var S = Int(filter.dim[3]())
            ctx.enqueue_function[_transpose_fcrs_to_krsc[filter_type]](
                filter.ptr,
                filter_krsc_ptr,
                Int32(F),
                Int32(C),
                Int32(R),
                Int32(S),
                grid_dim=grid,
                block_dim=transpose_block,
            )
        else:
            var R = Int(filter.dim[0]())
            var S = Int(filter.dim[1]())
            var C = Int(filter.dim[2]())
            var F = Int(filter.dim[3]())
            ctx.enqueue_function[_transpose_rscf_to_krsc[filter_type]](
                filter.ptr,
                filter_krsc_ptr,
                Int32(R),
                Int32(S),
                Int32(C),
                Int32(F),
                grid_dim=grid,
                block_dim=transpose_block,
            )

        # Construct problem shape and TileTensors
        var problem = Conv2dProblemShape(
            batch=batch,
            in_height=in_h,
            in_width=in_w,
            in_channels=in_c,
            out_channels=out_c,
            filter_h=fh,
            filter_w=fw,
            pad_h=symmetric_padding[0],
            pad_w=symmetric_padding[1],
        )

        var act_tt = input.reshape(row_major(batch, in_h, in_w, in_c))
        var filter_tt = TileTensor(
            filter_krsc_ptr,
            row_major(out_c, fh, fw, in_c),
        )
        var out_tt = output.reshape(row_major(batch, out_h, out_w, out_c))

        # Pick activation/filter swizzle based on C_in alignment. SWIZZLE_128B
        # requires the inner C-row to be 128B-aligned; SWIZZLE_64B relaxes that
        # to 64B (e.g. covers bf16 C_in=96 → 192 B per row). Each path
        # compiles a separately-instantiated kernel.
        var in_c_bytes = in_c * size_of[input_type]()

        @__parameter
        @always_inline
        def _launch[
            swizzle: TensorMapSwizzle,
            num_pipeline_stages_override: Int = 0,
        ]() raises:
            comptime config = Conv2dConfig[
                input_type, filter_type, output_type
            ].default_bf16_1sm[
                swizzle=swizzle,
                num_pipeline_stages_override=num_pipeline_stages_override,
            ]()

            comptime if has_residual:
                var src_tt = TileTensor(
                    # SAFETY: set when has_residual == True
                    source_ptr.unsafe_value(),
                    row_major(batch, out_h, out_w, out_c),
                )
                conv2d_fprop_with_residual[
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    has_residual=True,
                ](out_tt, act_tt, filter_tt, src_tt, beta, problem, ctx)
            else:
                conv2d_fprop[
                    config=config,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](out_tt, act_tt, filter_tt, problem, ctx)

        if in_c_bytes % 128 == 0:
            _launch[TensorMapSwizzle.SWIZZLE_128B]()
        else:
            # Dispatch gate guarantees in_c_bytes % 64 == 0 when we reach here.
            # BK=32 (half of BK=64) makes the auto-sizer over-estimate stages
            # (e.g. bf16: predicts 13 where <=6 fit). Pin to 6 as a safe
            # override matching what the SMEM budget actually allows.
            _launch[
                TensorMapSwizzle.SWIZZLE_64B,
                num_pipeline_stages_override=6,
            ]()
