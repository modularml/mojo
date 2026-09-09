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
"""Direct `_matmul_gpu` dispatch for 1x1x1 3D convolutions.

A 1x1x1 conv (Q=R=S=1) with stride=1, dilation=1, groups=1, and zero
padding is algebraically identical to a single matmul:

    output[b, d, h, w, f] = Σ_c input[b, d, h, w, c] * filter[0, 0, 0, c, f]

NDHWC input is already C-innermost contiguous, so we can view it as
[M, C_in] with M = B*D*H*W on the same pointer. Filter FCQRS/QRSCF with
Q=R=S=1 reduces to [F, C] or [C, F] respectively; no transpose kernel
needed. Output NDHWC collapses to [M, F]. No scratch allocation is
required, and the epilogue unflattens (m, f) -> (b, d, h, w, f) in one
call to the 5D lambda.

Covers every 1x1x1 case in the WAN VAE (`post_quant_conv`, per-block
`conv_shortcut`). Used in `conv_gpu`'s 5D arm as the first branch in
the QRSCF dispatch chain, before `dispatch_im2col_matmul_conv3d`.
"""

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from std.utils.index import IndexList
from linalg.utils import elementwise_epilogue_type
from nn.conv.conv_utils import elementwise_simd_epilogue_type


def dispatch_1x1x1_matmul_conv3d[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    //,
    filter_is_fcrs: Bool = False,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
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
    """Try to dispatch a 1x1x1 3D conv directly as a single `_matmul_gpu`.

    Returns True if the conv was handled; False if the caller should
    fall back to another implementation.

    Skips on: non-bf16 dtype, grouped conv, dilation != 1, stride != 1,
    non-zero padding, kernel size other than 1x1x1, and K (= C_in)
    below the matmul's minimum useful size.

    Parameters:
        input_type: Element type of the input tensor (inferred).
        filter_type: Element type of the filter tensor (inferred).
        output_type: Element type of the output tensor (inferred).
        filter_is_fcrs: Whether the filter is in FCQRS layout (`True`) or
            QRSCF layout (`False`) (defaults to `False`).
        maybe_epilogue_func: Optional SIMD elementwise epilogue invoked per
            output element in 5D coordinates (defaults to `None`).
    Args:
        input: 5D input tensor in NDHWC layout.
        filter: 5D convolution filter in FCQRS or QRSCF layout.
        output: 5D output tensor in NDHWC layout, written in place.
        stride: Per-axis convolution stride for (D, H, W).
        dilation: Per-axis dilation factor for (D, H, W).
        symmetric_padding: Per-axis symmetric zero padding for (D, H, W).
        num_groups: Number of convolution groups.
        ctx: GPU device context used to launch the matmul.
    """
    comptime assert input.flat_rank == 5, "input must be rank 5 (NDHWC)"
    comptime assert filter.flat_rank == 5, "filter must be rank 5"
    comptime assert output.flat_rank == 5, "output must be rank 5 (NDHWC)"

    comptime if input_type != .bfloat16:
        return False

    if num_groups != 1:
        return False
    if dilation[0] != 1 or dilation[1] != 1 or dilation[2] != 1:
        return False
    if stride[0] != 1 or stride[1] != 1 or stride[2] != 1:
        return False
    if (
        symmetric_padding[0] != 0
        or symmetric_padding[1] != 0
        or symmetric_padding[2] != 0
    ):
        return False

    var Q: Int
    var R: Int
    var S: Int
    comptime if filter_is_fcrs:
        Q = Int(filter.dim[2]())
        R = Int(filter.dim[3]())
        S = Int(filter.dim[4]())
    else:
        Q = Int(filter.dim[0]())
        R = Int(filter.dim[1]())
        S = Int(filter.dim[2]())

    if Q != 1 or R != 1 or S != 1:
        return False

    var batch = Int(input.dim[0]())
    var D = Int(input.dim[1]())
    var H = Int(input.dim[2]())
    var W = Int(input.dim[3]())
    var C_in = Int(input.dim[4]())

    var D_out = Int(output.dim[1]())
    var H_out = Int(output.dim[2]())
    var W_out = Int(output.dim[3]())
    var C_out = Int(output.dim[4]())

    # K below the MMA_K threshold isn't worth the launch overhead; keep
    # the same floor as the im2col path.
    if C_in < 16:
        return False

    # For 1x1x1 with stride=1 and zero padding the output spatial dims
    # must equal the input spatial dims. Bail if anything upstream
    # violated this invariant.
    if D_out != D or H_out != H or W_out != W:
        return False

    var full_M = batch * D_out * H_out * W_out
    var K = C_in
    var N = C_out
    var DHW_out = D_out * H_out * W_out
    var HW_out = H_out * W_out

    # --- Zero-copy TileTensor views of input, filter, output. ---
    # Input NDHWC is already C-innermost contiguous, so [M, C_in] with
    # M = batch * D_out * H_out * W_out is a pure pointer reinterpret.
    var a_tt = TileTensor(input.ptr, row_major(Coord(full_M, K)))
    var c_tt = TileTensor(output.ptr, row_major(Coord(full_M, N)))

    comptime if maybe_epilogue_func:
        comptime epilogue_5d = maybe_epilogue_func.value()

        @__parameter
        @always_inline
        @__copy_capture(DHW_out, HW_out, H_out, W_out)
        def _gemm_epilogue[
            _dtype: DType,
            _width: SIMDLength,
            *,
            alignment: Int = 1,
        ](coords_2d: IndexList[2], val: SIMD[_dtype, _width]):
            var m_idx = coords_2d[0]
            var n_idx = coords_2d[1]
            var batch_idx = m_idx // DHW_out
            var sp = m_idx - batch_idx * DHW_out
            var d_idx = sp // HW_out
            var rem = sp - d_idx * HW_out
            var h_idx = rem // W_out
            var w_idx = rem - h_idx * W_out
            epilogue_5d(
                IndexList[5](batch_idx, d_idx, h_idx, w_idx, n_idx),
                rebind[SIMD[output_type, _width]](val),
            )

        comptime if filter_is_fcrs:
            # FCQRS [F, C, 1, 1, 1] -> [F, C] view. _matmul_gpu wants
            # B as [N, K] row-major when transpose_b=True.
            var b_tt = TileTensor(filter.ptr, row_major(Coord(N, K)))
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=True,
                elementwise_lambda_fn=Optional[elementwise_epilogue_type](
                    _gemm_epilogue
                ),
            ](c_tt, a_tt.as_immut(), b_tt.as_immut(), ctx)
        else:
            # QRSCF [1, 1, 1, C, F] -> [C, F] view. _matmul_gpu wants
            # B as [K, N] row-major when transpose_b=False.
            var b_tt = TileTensor(filter.ptr, row_major(Coord(K, N)))
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=False,
                elementwise_lambda_fn=Optional[elementwise_epilogue_type](
                    _gemm_epilogue
                ),
            ](c_tt, a_tt.as_immut(), b_tt.as_immut(), ctx)
    else:
        comptime if filter_is_fcrs:
            var b_tt = TileTensor(filter.ptr, row_major(Coord(N, K)))
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=True,
            ](c_tt, a_tt.as_immut(), b_tt.as_immut(), ctx)
        else:
            var b_tt = TileTensor(filter.ptr, row_major(Coord(K, N)))
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=False,
            ](c_tt, a_tt.as_immut(), b_tt.as_immut(), ctx)

    return True
