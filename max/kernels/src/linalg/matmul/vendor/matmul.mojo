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

"""Vendor-backed matrix multiplication dispatch for TileTensor operands."""

from std.sys import simd_width_of, size_of, has_nvidia_gpu_accelerator

from max.algorithm import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import B200
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)

from std.utils import Index, IndexList

from ...utils import elementwise_epilogue_type
from .blas import matmul as vendor_matmul


def matmul[
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    ctx: DeviceContext,
) raises:
    """Vendor matmul dispatch for TileTensor operands. Callers are
    responsible for allocating `c`; for fp32-accumulate-and-quantize
    patterns, allocate an fp32 scratch buffer at the call site and supply
    an `elementwise_lambda_fn` that reads from it and writes the final
    quantized output.

    Parameters:
        transpose_b: Whether to treat `b` as transposed, computing
            `a @ b.T` instead of `a @ b` (defaults to `False`).
        elementwise_lambda_fn: Optional elementwise epilogue applied to
            each output tile after the matmul. When `None`, the raw
            matmul result is written to `c` unchanged (defaults to
            `None`).

    Args:
        c: Output matrix of shape `(m, n)` and rank 2 that receives the
            matmul result. Caller-allocated and mutable.
        a: Left-hand input matrix of rank 2.
        b: Right-hand input matrix of rank 2, transposed when
            `transpose_b` is `True`.
        ctx: Device context used to select the vendor dispatch path and
            query device capabilities.
    """
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime assert a.flat_rank == 2, "a must be of rank 2"
    comptime assert b.flat_rank == 2, "b must be of rank 2"

    comptime c_type = c.dtype

    comptime if not elementwise_lambda_fn:
        vendor_matmul[use_tf32=True](
            ctx,
            c,
            a,
            b,
            c_row_major=True,
            transpose_b=transpose_b,
        )
        return
    else:
        comptime epilogue = elementwise_lambda_fn.value()
        # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
        # arch support 32B load/store to global memory, see KERN-2037.
        comptime use_32b_simd = (
            has_nvidia_gpu_accelerator()
            and ctx.default_device_info.compute >= B200.compute
        )
        comptime simd_size = 32 // size_of[c_type]() if use_32b_simd else (
            simd_width_of[c_type, target=get_gpu_target()]()
        )

        var c_tt = TileTensor(
            rebind[UnsafePointer[Scalar[c_type], MutAnyOrigin]](c.ptr),
            row_major(Coord(Int(c.dim[0]()), Int(c.dim[1]()))),
        )

        def epilogue_wrapper[
            simd_width: Int, alignment: Int = 1
        ](idx: Coord) {var}:
            var c_val = c_tt.load[
                width=simd_width,
                # Load takes alignment in bytes, lambda takes number of elements
                alignment=alignment * size_of[c_type](),
            ](idx)
            epilogue[c_type, simd_width, alignment=alignment](
                Index(idx[0].value(), idx[1].value()), c_val
            )

        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())

        # For D = alpha * A * B + beta * C, vendor matmul currently sets
        # C to null, i.e don't fuse linear operations into gemm, KERN-1774.
        vendor_matmul[use_tf32=True](
            ctx,
            c,
            a,
            b,
            c_row_major=True,
            transpose_b=transpose_b,
        )
        elementwise[simd_size, target="gpu"](epilogue_wrapper, (m, n), ctx)
