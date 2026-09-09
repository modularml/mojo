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
"""8x8 `simdgroup_matrix` GEMM kernel for Apple Silicon GPUs (M1-M4)."""

from max.gpu import (
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.compute.arch.mma_apple import _mma_apple_8x8
from layout import TensorLayout, TensorEngine, TileTensor
from std.utils import Index
from std.utils.numerics import get_accum_type

from ....utils import elementwise_epilogue_type


# ===----------------------------------------------------------------------=== #
# M1-M4: 8x8 simdgroup_matrix MMA (uses stdlib `_mma_apple_8x8`)
# ===----------------------------------------------------------------------=== #

comptime MMA8_DIM = 8
comptime FRAG8 = 2  # 8x8 = 64 elems / 32 lanes = 2 per lane


@always_inline
def _frag8_layout(lane: Int) -> Tuple[Int, Int]:
    """Apple 8x8 simdgroup-matrix per-lane layout (ground-truthed via Metal
    `thread_elements()`). Lane owns (row, col_base) and (row, col_base+1)."""
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@always_inline
def _simdgroup8x8_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    transpose_b: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    s_type: DType,
    BLOCK_M: Int,
    BLOCK_N: Int,
    BLOCK_K: Int,
    NUM_SIMDGROUPS: Int,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=_],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=_],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=_],
    m: Int,
    n: Int,
    k: Int,
):
    """8x8 simdgroup-matrix GEMM for M1-M4 (no neural accelerator).

    64x64 block, 4 simdgroups, each owning a 32x32 subtile = 4x4 grid of 8x8
    simdgroup-matrix tiles (16 f32 accumulators / lane-pair). K-loop steps by 8.
    Fully-interior simdgroup subtiles take an unguarded fast path; edge subtiles
    zero-fill OOB rows/cols so ragged M/N is correct (in-bounds outputs only use
    in-bounds A rows / B cols; K is always full since dispatch gates k%16==0).
    f32 accum, cast to `c_type` + optional epilogue.
    """
    comptime assert c.flat_rank == 2, "c must have flat_rank == 2"
    comptime assert a.flat_rank == 2, "a must have flat_rank == 2"
    comptime assert b.flat_rank == 2, "b must have flat_rank == 2"
    comptime assert NUM_SIMDGROUPS == 4, "8x8 path assumes 4 simdgroups"

    comptime SG_M = BLOCK_M // 2  # 32
    comptime SG_N = BLOCK_N // 2  # 32
    comptime NT_M = SG_M // MMA8_DIM  # 4
    comptime NT_N = SG_N // MMA8_DIM  # 4

    var lane = Int(lane_id())
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var row_base = Int(block_idx.y) * BLOCK_M + (sg // 2) * SG_M
    var col_base = Int(block_idx.x) * BLOCK_N + (sg % 2) * SG_N
    # Fully-interior simdgroup subtile -> unguarded loads (the common case).
    var interior = (row_base + SG_M <= m) and (col_base + SG_N <= n)

    var accum = Array[SIMD[.float32, FRAG8], NT_M * NT_N](
        fill=SIMD[.float32, FRAG8](0)
    )

    var a_ptr = a.ptr
    var b_ptr = b.ptr

    for ks in range(k // MMA8_DIM):
        var kk = ks * MMA8_DIM
        # A (M,K) row-major: lane's 2 frag elems are consecutive K cols (K is
        # always in-bounds); only the row needs a bound for ragged M.
        var afrag = Array[SIMD[a_type, FRAG8], NT_M](uninitialized=True)
        comptime for mi in range(NT_M):
            var grow = row_base + mi * MMA8_DIM + frow
            if interior or grow < m:
                afrag[mi] = (a_ptr + grow * k + kk + fcol).load[width=FRAG8]()
            else:
                afrag[mi] = SIMD[a_type, FRAG8](0)
        # B holds B[k_idx, j]: row=k_idx (always in-bounds), col=j (bound for n).
        var bfrag = Array[SIMD[b_type, FRAG8], NT_N](uninitialized=True)
        comptime for ni in range(NT_N):
            comptime if transpose_b:
                # B stored (N,K): B[k,j]=b_ptr[j*k+k_idx]; slots differ in j.
                # We gather the transposed fragment manually. The hardware
                # `simdgroup_load`-with-transpose intrinsic
                # (`air.simdgroup_matrix_8x8_load`) could maybe do this in one
                # op, but that's skipped for now and left for future
                # investigation.
                var bf = SIMD[b_type, FRAG8](0)
                comptime for s in range(FRAG8):
                    var gj = col_base + ni * MMA8_DIM + fcol + s
                    if interior or gj < n:
                        bf[s] = b_ptr[gj * k + kk + frow]
                bfrag[ni] = bf
            else:
                # B stored (K,N): B[k,j]=b_ptr[k_idx*n+j]; slots consecutive j.
                var krow = kk + frow
                var gj = col_base + ni * MMA8_DIM + fcol
                if interior or gj + 1 < n:
                    bfrag[ni] = (b_ptr + krow * n + gj).load[width=FRAG8]()
                else:
                    var bf = SIMD[b_type, FRAG8](0)
                    if gj < n:
                        bf[0] = b_ptr[krow * n + gj]
                    bfrag[ni] = bf
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                var c_frag = accum[mi * NT_N + ni]
                _mma_apple_8x8(
                    accum[mi * NT_N + ni], afrag[mi], bfrag[ni], c_frag
                )

    comptime for mi in range(NT_M):
        comptime for ni in range(NT_N):
            var frag = accum[mi * NT_N + ni]
            comptime for s in range(FRAG8):
                var grow = row_base + mi * MMA8_DIM + frow
                var gcol = col_base + ni * MMA8_DIM + fcol + s
                if grow < m and gcol < n:
                    comptime if elementwise_lambda_fn:
                        comptime ep = elementwise_lambda_fn.value()
                        ep[c_type, 1](Index(grow, gcol), frag[s].cast[c_type]())
                    else:
                        c[grow, gcol] = frag[s].cast[c_type]()


@__name(t"gemm_kernel_apple_8x8_{c_type}_{a_type}_{b_type}_{transpose_b}")
def gemm_kernel_apple_8x8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
    BLOCK_M: Int = 64,
    BLOCK_N: Int = 64,
    BLOCK_K: Int = 16,
    NUM_SIMDGROUPS: Int = 4,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    m: Int32,
    n: Int32,
    k: Int32,
):
    """Launchable wrapper for the 8x8 simdgroup-matrix GEMM (bench/test)."""
    var _m = Int(m)
    var _n = Int(n)
    var _k = Int(k)
    _simdgroup8x8_matmul_kernel[
        c_type,
        a_type,
        b_type,
        c_layout,
        a_layout,
        b_layout,
        transpose_b,
        elementwise_lambda_fn,
        s_type,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        NUM_SIMDGROUPS,
    ](c, a, b, _m, _n, _k)
