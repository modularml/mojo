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
"""MXFP4 matmul on H100 (SM90) via dequant-to-FP8 + FP8 GEMM.

Dequantizes MXFP4 weights to FP8, then uses the SM90 warp-specialized FP8 GEMM.
Activations (BF16) are cast to FP8 on-the-fly.
"""

from max.gpu.host import DeviceContext
from std.sys.info import _accelerator_arch
from layout import Coord, Idx, TileTensor, row_major

from .mxfp4_dequant import dequant_mxfp4, _cast_bf16_to_fp8
from .matmul.gpu import _matmul_gpu


def mxfp4_matmul_sm90(
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b_packed: TileTensor,
    b_scales: TileTensor,
    ctx: DeviceContext,
) raises:
    """MXFP4 matmul: dequant B weights to FP8, cast A to FP8, SM90 FP8 GEMM.

    Args:
        c: Output [M, N] in bfloat16.
        a: Activations [M, K] in bfloat16.
        b_packed: Weights [N, K//2] in uint8 (packed MXFP4).
        b_scales: Weight scales [N, K//32] in float8_e8m0fnu.
        ctx: Device context.
    """
    comptime assert (
        "sm_90" in _accelerator_arch()
    ), "mxfp4_matmul_sm90 requires SM90"

    comptime c_type = c.dtype
    comptime a_type = a.dtype
    comptime b_type = b_packed.dtype
    comptime b_scales_type = b_scales.dtype

    comptime assert c_type == .bfloat16, "output must be bfloat16"
    comptime assert a_type == .bfloat16, "activations must be bfloat16"
    comptime assert b_type == .uint8, "weights must be uint8 (packed FP4)"
    comptime assert (
        b_scales_type == .float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"

    var M = Int(c.dim[0]())
    comptime static_N = type_of(c).static_shape[1]
    comptime static_K = type_of(a).static_shape[1]
    comptime fp8_type = DType.float8_e4m3fn

    # TODO: This implementation materializes the full FP8 weights and casted
    # activations into global memory before dispatching the GEMM, which negates
    # the memory bandwidth benefits of MXFP4. Replace with a fused SM90
    # prologue that unpacks MXFP4 directly in shared memory or registers.

    # Step 1: Dequantize MXFP4 weights to FP8
    var b_fp8_buf = ctx.enqueue_create_buffer[fp8_type](static_N * static_K)
    var b_fp8_tt = TileTensor(
        b_fp8_buf, row_major(Idx[static_N], Idx[static_K])
    )

    dequant_mxfp4(
        ctx,
        b_fp8_tt,
        b_packed,
        b_scales,
        num_rows=static_N,
        num_cols=static_K,
    )

    # Step 2: Cast BF16 activations to FP8
    var a_fp8_buf = ctx.enqueue_create_buffer[fp8_type](M * static_K)
    var a_fp8_tt = TileTensor(a_fp8_buf, row_major(M, Idx[static_K]))

    _cast_bf16_to_fp8(ctx, a_fp8_tt, a, M, static_K)

    # Step 3: FP8 GEMM via _matmul_gpu (handles dispatch + fallback)
    _matmul_gpu[transpose_b=True](c, a_fp8_tt, b_fp8_tt, ctx)

    # Keep temp buffers alive through async GEMM enqueue.
    _ = b_fp8_buf^
    _ = a_fp8_buf^
