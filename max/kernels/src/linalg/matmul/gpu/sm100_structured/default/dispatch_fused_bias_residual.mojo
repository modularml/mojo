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
"""Dispatcher for fused matmul + bias/residual (`mo.composite.matmul_add`).

Kept separate from `matmul_dispatch_sm100` so the generic matmul dispatch does
not have to thread an epilogue tensor through every kernel. The bias/residual is
honored one of two mutually exclusive ways:

  * Native: the SM100 blackwell GEMM applies it via its TMA epilogue load (the
    tensor is passed, no lambda). Used for GEMM-shaped problems (M>1, N>1, bf16,
    16B-aligned N/K) -- the fast prefill path.
  * Fallback: the bias/residual is wrapped as a normal elementwise (store)
    epilogue and the generic `matmul_dispatch_sm100` is reused. GEMV (M=1/N=1),
    small-MN, and vendor (cuBLAS) all already apply an elementwise epilogue, so
    correctness is universal. No tensor is passed, so there is no double-add.
"""

from std.collections import OptionalReg
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel
from std.sys import size_of

from internal_utils import Table
from layout import Coord, RowMajorLayout, TileTensor
from std.logger import Logger
from std.utils.index import Index, IndexList

from .dispatch import (
    matmul_dispatch_sm100,
    sm100_heuristic_and_outliers_dispatch,
    small_MN_gemms,
    _vendor_blas_matmul_sm100,
)
from .tuning_configs import (
    TuningConfigSmallMNGemms,
    _get_tuning_list_small_MN_gemms_bf16,
)

comptime logger = Logger()


@always_inline
def fused_bias_residual_matmul_dispatch_sm100[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    //,
    transpose_b: Bool = False,
    pdl_level: PDLLevel = PDLLevel(),
    epilogue_is_1d: Bool = False,
    has_epilogue_tensor: Bool = True,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    epilogue_tensor: TileTensor[
        c_type, RowMajorLayout[Int64, Int64], ImmutAnyOrigin
    ],
    ctx: DeviceContext,
) raises:
    """Dispatches a fused matmul plus bias/residual on SM100 (Blackwell) GPUs.

    Selects between a native TMA-epilogue GEMM path for aligned GEMM-shaped
    problems and a fallback that applies the bias/residual as a normal
    elementwise store epilogue through the generic `matmul_dispatch_sm100`
    dispatcher (covering GEMV, small-MN, and vendor paths).

    Parameters:
        c_type: Output element dtype of `c`.
        a_type: Element dtype of `a`; must equal `b_type` and be bfloat16.
        b_type: Element dtype of `b`; must equal `a_type` and be bfloat16.
        transpose_b: Whether `b` is transposed.
        pdl_level: Persistent kernel launch grid control level.
        epilogue_is_1d: Treats `epilogue_tensor` as a 1D bias broadcast across rows.
        has_epilogue_tensor: Indicates an epilogue tensor is supplied for the TMA path.

    Args:
        c: Rank-2 output tile tensor accumulating `a @ b` plus the residual.
        a: Rank-2 left-hand side input tile tensor.
        b: Rank-2 right-hand side input tile tensor.
        epilogue_tensor: Row-major bias/residual tensor added into `c`.
        ctx: Device context used to launch the selected kernel.

    Raises:
        On comptime assertion failures for rank, dtype, or dtype-pair mismatches.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"
    comptime assert a_type == b_type, "a_type and b_type must be the same"

    var m = Int(c.dim[0]())
    comptime static_N = c.static_shape[1]
    comptime static_K = a.static_shape[1]

    comptime assert (
        a_type == b_type == .bfloat16
    ), "a_type and b_type must be bfloat16 and must be the same"
    comptime assert c_type in (DType.bfloat16,), "c_type must be bfloat16"

    # Zero-sized output (e.g. an ``(0, N)`` GEMM from the fused Linear layers
    # of a diffusion VAE encoder running on the text-to-image ``(0, 0, 3)``
    # placeholder image): nothing to compute.  The generic ``matmul`` entry
    # guards this identically. The output buffer is pre-allocated zero-element
    # by the caller, so an early return produces the correct empty output.
    if m == 0 or Int(c.dim[1]()) == 0:
        return

    # The bias/residual as a store epilogue: `c = (a @ b) + epilogue[coords]`,
    # broadcasting row 0 for a 1D bias. Every non-TMA kernel (GEMV, small-MN,
    # cuBLAS) applies this exactly once.
    @__parameter
    @always_inline
    @__copy_capture(c, epilogue_tensor)
    def bias_residual_elementwise_lambda[
        _dtype: DType, _width: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[_dtype, _width]):
        var row = 0 if epilogue_is_1d else coords[0]
        var resid = rebind[SIMD[_dtype, _width]](
            epilogue_tensor.load[
                width=_width, alignment=alignment * size_of[c_type]()
            ](Coord(row, coords[1]))
        )

        c.store_linear[alignment=alignment * size_of[c.dtype]()](
            coords, rebind[SIMD[c.dtype, _width]](val + resid)
        )

    comptime small_MN_gemms_table = Table(
        _get_tuning_list_small_MN_gemms_bf16(), "small_MN_gemms_configs"
    )

    comptime small_MN_gemms_configs = small_MN_gemms_table.find(
        rule=lambda (x: TuningConfigSmallMNGemms) -> Bool: x.K == static_K
        and x.N == static_N
    )

    comptime if small_MN_gemms_configs:
        comptime for config in small_MN_gemms_configs:
            if m >= config.M and m < config.M_end:
                logger.info("Dispatching to small_MN_gemms: ", config)
                small_MN_gemms[
                    config=config,
                    elementwise_lambda_fn=bias_residual_elementwise_lambda,
                    pdl_level=pdl_level,
                ](c, a, b, ctx)
                return

    comptime low_perf_shapes = [
        Index(2112, 14336),
    ]

    # fallback to vendor matmul for shapes that Mojo kernel is lagging behind
    comptime if (static_N, static_K) in low_perf_shapes:
        _vendor_blas_matmul_sm100[
            c_type,
            a_type,
            b_type,
            transpose_b,
            elementwise_lambda_wrapper=bias_residual_elementwise_lambda,
        ](c, a, b, ctx)
        return

    comptime has_static_NK = static_N > 0 and static_K > 0

    # Native fast path: the pure SM100 blackwell GEMM applies the bias/residual
    comptime if (
        has_static_NK
        and transpose_b
        and static_N * size_of[c_type]() % 16 == 0
        and static_K * size_of[a_type]() % 16 == 0
    ):
        if m != 1:
            var status = sm100_heuristic_and_outliers_dispatch[
                transpose_b=transpose_b,
                pdl_level=pdl_level,
                has_epilogue_tensor=True,
                epilogue_is_1d=epilogue_is_1d,
            ](c, a, b, ctx, epilogue_tensor=OptionalReg(epilogue_tensor))
            if status:
                logger.info(
                    "------ Fused matmul+bias/residual: TMA epilogue ----"
                )
                return

    # Fallback: residual is a normal elementwise epilogue applied by whichever
    # kernel the generic dispatcher selects (GEMV / small-MN / cuBLAS). No
    # tensor is passed, so there is no double-add with a TMA epilogue.
    logger.info("------ Fused matmul+bias/residual: elementwise epilogue ----")
    matmul_dispatch_sm100[
        transpose_b=transpose_b,
        elementwise_lambda_fn=bias_residual_elementwise_lambda,
        elementwise_lambda_wrapper=bias_residual_elementwise_lambda,
        pdl_level=pdl_level,
    ](c, a, b, ctx)
