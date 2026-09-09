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

"""Provides general matrix-vector (GEMV) and general vector-matrix (GEVM) kernels for CPU and GPU."""

from std.collections import Optional
from std.math import align_down, align_up, ceildiv
from std.math.uutils import umod, ufloordiv
from std.sys import (
    has_amd_gpu_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
    simd_width_of,
)
from std.sys.info import _is_amd_mi250x, _is_sm_100x, size_of
from max.gpu.compute.mma import ld_matrix, mma
from max.gpu.sync import async_copy_arrive, named_barrier
from layout.tma_async import SharedMemBarrier
from structured_kernels.kernel_common import _to_batched_3d
from structured_kernels.smem_types import SMemArray
from structured_kernels.tile_types import SMemTileArray2D
from layout.swizzle import Swizzle, make_swizzle

import max.gpu.primitives.warp as warp
from max.algorithm.reduction import _reduce_generator
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    global_idx,
    thread_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import (
    DeviceAttribute,
    DeviceContext,
    FuncAttribute,
    get_gpu_target,
)
from max.gpu.host.info import B200
from max.gpu.memory import async_copy, external_memory
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    pdl_launch_attributes,
    launch_dependent_grids,
    wait_on_dependent_grids,
)

# layout imports
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from std.logger import Logger
from std.memory import bitcast, unsafe_stack_allocation
from std.utils import IndexList
from std.utils.coord import Coord
from std.utils.index import Index
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from .matmul.gpu import matmul_kernel_naive
from .utils import GemmShape, elementwise_epilogue_type

comptime logger = Logger()


@fieldwise_init
struct GEMVAlgorithm(Equatable, Hashable, TrivialRegisterPassable, Writable):
    """Enumerates the GEMV kernel algorithm variants used by the GPU dispatcher.

    Each variant targets a distinct operand shape, memory access pattern, or
    hardware capability. The dispatcher in `gemv_gpu` selects among these based
    on runtime shape information and target architecture.
    """

    var _value: Int

    comptime GEMV_KERNEL = Self(0)
    comptime GEMV_KERNEL_VECTOR = Self(1)
    comptime GEMV_SPLIT_K = Self(2)
    comptime GEVM_KERNEL_VECTOR = Self(3)
    comptime GEVM_KERNEL = Self(4)
    comptime MATMUL_NAIVE = Self(5)
    comptime GEMM_MMA_CPASYNC = Self(6)

    @always_inline("nodebug")
    def __int__(self) -> Int:
        return self._value

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    @always_inline
    def __is__(self, other: Self) -> Bool:
        return self == other

    @always_inline
    def __isnot__(self, other: Self) -> Bool:
        return self != other

    @always_inline
    def __hash__(self) -> Int:
        return self._value

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.GEMV_KERNEL:
            writer.write("GEMV")
        elif self == Self.GEMV_KERNEL_VECTOR:
            writer.write("GEMV_KERNEL_VECTOR")
        elif self == Self.GEMV_SPLIT_K:
            writer.write("GEMV_SPLIT_K")
        elif self == Self.GEVM_KERNEL_VECTOR:
            writer.write("GEVM_KERNEL_VECTOR")
        elif self == Self.GEVM_KERNEL:
            writer.write("GEVM_KERNEL")
        elif self == Self.MATMUL_NAIVE:
            writer.write("MATMUL_NAIVE")
        elif self == Self.GEMM_MMA_CPASYNC:
            writer.write("GEMM_MMA_CPASYNC")
        else:
            writer.write("UNKNOWN")


@always_inline
def reverse_idx[transpose: Bool](x: Int, y: Int) -> IndexList[2]:
    """Returns an index pair (x, y) or (y, x) depending on the transpose parameter.
    """
    return Index(y, x) if transpose else Index(x, y)


# Matrix-Column Vector Multiplication using scalar arithmetic
@__name(t"gemv_kernel_{c_type}_{a_type}_{b_type}_{transpose_b}")
def gemv_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    pdl_level: PDLLevel = PDLLevel(),
](
    c: UnsafePointer[Scalar[c_type], AnyOrigin[mut=True]],
    a: UnsafePointer[Scalar[a_type], ImmUnsafeAnyOrigin],
    b: UnsafePointer[Scalar[b_type], ImmUnsafeAnyOrigin],
    m: Int32,
    n: Int32,
    k: Int32,
):
    var _m = Int(m)
    var _n = Int(n)
    var _k = Int(k)
    var tid = global_idx.x
    var global_warp_id = warp.broadcast(ufloordiv(tid, WARP_SIZE))
    var lane_id = lane_id()

    if global_warp_id >= _m:
        return

    var accum = Scalar[accum_type](0)

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Every warp processes a single row of the resultant vector
    for i in range(ceildiv(_k, WARP_SIZE)):
        var idx = i * WARP_SIZE + lane_id
        if idx < _k:
            accum += (
                a.load(global_warp_id * _k + idx).cast[accum_type]()
                * b.load(idx).cast[accum_type]()
            )

    accum = warp.sum(accum)

    if lane_id == 0:
        comptime if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda[c_type, 1](
                reverse_idx[transpose_b](global_warp_id, 0),
                accum.cast[c_type](),
            )
        else:
            c[global_warp_id] = accum.cast[c_type]()

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


# Matrix-Column Vector Multiplication using vectorized instructions
@__name(
    t"gemv_kernel_vector_{c_type}_{a_type}_{b_type}_{transpose_b}_{simd_width}",
)
def gemv_kernel_vector[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    *,
    simd_width: Int,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    check_bounds: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],  # m
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],  # m * k
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],  # 1 * k
    m: Int32,
    n: Int32,
    k: Int32,
):
    var _m = Int(m)
    var _k = Int(k)
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime assert a.flat_rank == 2, "a must be of rank 2"
    comptime assert b.flat_rank == 2, "b must be of rank 2"

    var tid = global_idx.x
    var global_warp_id: Int = warp.broadcast(ufloordiv(tid, WARP_SIZE))
    var lane_id = lane_id()
    if global_warp_id >= _m:
        return

    # Every warp processes a single row of the resultant vector
    var local_accum = SIMD[accum_type, simd_width](0)

    comptime local_accum_type = type_of(local_accum)

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    var num_iters = (
        ceildiv(_k // simd_width, WARP_SIZE) if comptime (
            check_bounds
        ) else ufloordiv(_k, WARP_SIZE * simd_width)
        + 1
    )

    # Main loop: all lanes are in bounds, no check needed.
    for i in range(num_iters - 1):
        var a_tile = a.tile[1, WARP_SIZE * simd_width](global_warp_id, i)
        var b_tile = b.tile[1, WARP_SIZE * simd_width](0, i)
        var a_vec = a_tile.vectorize[1, simd_width]()[0, lane_id]
        var b_vec = b_tile.vectorize[1, simd_width]()[0, lane_id]
        local_accum += a_vec.cast[accum_type]() * b_vec.cast[accum_type]()

    # Last iteration: only lanes with valid K indices participate and
    # only if check_bounds is True.
    comptime if check_bounds:
        if num_iters > 0:
            var last = num_iters - 1
            var a_tile = a.tile[1, WARP_SIZE * simd_width](global_warp_id, last)
            var b_tile = b.tile[1, WARP_SIZE * simd_width](0, last)
            if (lane_id + last * WARP_SIZE) * simd_width < _k:
                var a_vec = a_tile.vectorize[1, simd_width]()[0, lane_id]
                var b_vec = b_tile.vectorize[1, simd_width]()[0, lane_id]
                local_accum += (
                    a_vec.cast[accum_type]() * b_vec.cast[accum_type]()
                )

    var accum = warp.sum(local_accum)

    if lane_id == 0:
        comptime if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda(
                reverse_idx[transpose_b](global_warp_id, 0),
                accum.cast[c_type](),
            )
        else:
            comptime if transpose_b:
                c[0, global_warp_id] = accum.cast[c_type]()
            else:
                c[global_warp_id, 0] = accum.cast[c_type]()

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


# Same per-row dot product as `gemv_kernel_vector` (identical lane/vector
# layout and reduction order, so results are bit-identical), but each warp
# reduces `rows_per_warp` consecutive rows. One row per warp streams only
# `k * size_of[a_type]()` bytes before the warp retires, so a wide-N shallow-K
# GEMV becomes bound by the CTA launch/retire pipeline (measured ~1 CTA/ns
# chip-wide on B200, invariant to SM clock) instead of by HBM. Widening the
# tile restores the memory-bound regime: at (n=262144, k=256) bf16 on B200,
# 257 us / 0.52 TB/s -> 41.6 us / 3.2 TB/s.
@__name(
    t"gemv_kernel_vector_multirow_{c_type}_{a_type}_{b_type}_{transpose_b}_{simd_width}_{rows_per_warp}",
)
def gemv_kernel_vector_multirow[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    *,
    simd_width: Int,
    rows_per_warp: Int,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    check_bounds: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],  # m
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],  # m * k
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],  # 1 * k
    m: Int32,
    n: Int32,
    k: Int32,
):
    var _m = Int(m)
    var _k = Int(k)
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime assert a.flat_rank == 2, "a must be of rank 2"
    comptime assert b.flat_rank == 2, "b must be of rank 2"

    var tid = global_idx.x
    var global_warp_id: Int = warp.broadcast(ufloordiv(tid, WARP_SIZE))
    var lane_id = lane_id()
    var row_base = global_warp_id * rows_per_warp
    if row_base >= _m:
        return

    # Rows are contiguous per warp, so only the last live warp can run past
    # `_m`. Clamping the row index keeps those loads in bounds; the stores
    # below drop the out-of-range results.
    var last_row = _m - 1
    var local_accum = Array[SIMD[accum_type, simd_width], rows_per_warp](fill=0)

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    var num_iters = (
        ceildiv(_k // simd_width, WARP_SIZE) if comptime (
            check_bounds
        ) else ufloordiv(_k, WARP_SIZE * simd_width)
        + 1
    )

    # Main loop: all lanes are in bounds, no check needed.
    for i in range(num_iters - 1):
        var b_tile = b.tile[1, WARP_SIZE * simd_width](0, i)
        var b_vec = b_tile.vectorize[1, simd_width]()[0, lane_id]

        comptime for r in range(rows_per_warp):
            var a_tile = a.tile[1, WARP_SIZE * simd_width](
                min(row_base + r, last_row), i
            )
            var a_vec = a_tile.vectorize[1, simd_width]()[0, lane_id]
            local_accum[r] += (
                a_vec.cast[accum_type]() * b_vec.cast[accum_type]()
            )

    # Last iteration: only lanes with valid K indices participate and
    # only if check_bounds is True.
    comptime if check_bounds:
        if num_iters > 0:
            var last = num_iters - 1
            var b_tile = b.tile[1, WARP_SIZE * simd_width](0, last)
            if (lane_id + last * WARP_SIZE) * simd_width < _k:
                var b_vec = b_tile.vectorize[1, simd_width]()[0, lane_id]

                comptime for r in range(rows_per_warp):
                    var a_tile = a.tile[1, WARP_SIZE * simd_width](
                        min(row_base + r, last_row), last
                    )
                    var a_vec = a_tile.vectorize[1, simd_width]()[0, lane_id]
                    local_accum[r] += (
                        a_vec.cast[accum_type]() * b_vec.cast[accum_type]()
                    )

    comptime for r in range(rows_per_warp):
        var accum = warp.sum(local_accum[r])
        var row = row_base + r
        if lane_id == 0 and row < _m:
            comptime if elementwise_lambda_fn:
                comptime elementwise_lambda = elementwise_lambda_fn.value()
                elementwise_lambda(
                    reverse_idx[transpose_b](row, 0),
                    accum.cast[c_type](),
                )
            else:
                comptime if transpose_b:
                    c[0, row] = accum.cast[c_type]()
                else:
                    c[row, 0] = accum.cast[c_type]()

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


@always_inline
def _dot_accum[
    a_type: DType,
    b_type: DType,
    accum_type: DType,
    width: SIMDLength,
](
    a: SIMD[a_type, width], b: SIMD[b_type, width], acc: Scalar[accum_type]
) -> Scalar[accum_type]:
    """Compute dot(a, b) + acc with fused bf16→f32 dot product on AMD.

    `a` and `b` may have different element types. On AMD GPUs except gfx90a,
    when BOTH operands are bf16 with an f32 accumulator we use v_dot2_f32_bf16
    to avoid explicit bf16→f32 conversion (120 v_perm/v_bfi instructions). When
    the operands differ (e.g. bf16 activation × fp32 router weight) or on other
    targets/types, each operand is widened to `accum_type` and multiplied — the
    widening is exact for bf16→f32, so a mixed bf16×fp32 dot is numerically
    identical to first casting the bf16 activation to f32 then dotting.
    """
    var result = acc

    comptime if (
        is_amd_gpu()
        and not _is_amd_mi250x()
        and a_type == .bfloat16
        and b_type == .bfloat16
        and accum_type == .float32
    ):
        # v_dot2_f32_bf16: D.f32 = S0.bf16[0]*S1.bf16[0] + S0.bf16[1]*S1.bf16[1] + S2.f32
        comptime for p in range(width // 2):
            var a_pair = rebind[SIMD[.bfloat16, 2]](a.slice[2, offset=p * 2]())
            var b_pair = rebind[SIMD[.bfloat16, 2]](b.slice[2, offset=p * 2]())
            result = rebind[Scalar[accum_type]](
                llvm_intrinsic[
                    "llvm.amdgcn.fdot2.f32.bf16",
                    Float32,
                ](
                    a_pair,
                    b_pair,
                    rebind[Float32](result),
                    False,
                )
            )

        comptime if width % 2 != 0:
            result += (
                a[width - 1].cast[accum_type]()
                * b[width - 1].cast[accum_type]()
            )
    elif is_amd_gpu():
        # AMD non-BF16 (e.g. FP8): vector multiply + horizontal reduce.
        result += (a.cast[accum_type]() * b.cast[accum_type]()).reduce_add()
    else:
        # NVIDIA/generic: scalar element-wise loop. reduce_add() generates
        # wider intermediates that increase NVIDIA register pressure vs
        # sequential FMA chains (13% regression on small-K shapes).
        var ac = a.cast[accum_type]()
        var bc = b.cast[accum_type]()
        comptime for l in range(width):
            result += ac[l] * bc[l]

    return result


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"gemv_split_k_{c_type}_{a_type}_{b_type}_{num_threads}")
def gemv_split_k[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    simd_width: Int,
    tile_m: Int,
    tile_n: Int,
    num_threads: Int,
    unroll_factor: Int = 2,
    weight_non_temporal: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    check_bounds_m: Bool = True,
    check_bounds_n: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    output: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    act: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    weight: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    m: Int32,
    n: Int32,
    k: Int32,
):
    """GEMV with tiling in K dimension.
    Assuming the B (weight) matrix is transposed i.e. row major N x K, this kernel
    implements a vector (1 x K) times a matrix (N x K).
    The impl can actually handle M > 1 but it's only optimal for tiny M. We use
    it for M = 1 only.

    The launch grid covers ceildiv(m, tile_m) * tile_m rows and
    ceildiv(n, tile_n) * tile_n columns, so the final blocks read and write
    past the buffers unless the bounds guards are on: `check_bounds_m=False`
    is only safe when the launcher guarantees m % tile_m == 0 (m is a runtime
    value, so tile_m == 1 is the usual way to guarantee it), and
    `check_bounds_n=False` is only safe when n % tile_n == 0.

    Parameters:
        c_type: Output element type.
        a_type: Activation matrix element type.
        b_type: Weight matrix element type.
        c_layout: Layout descriptor for the output tensor.
        a_layout: Layout descriptor for the activation matrix.
        b_layout: Layout descriptor for the weight matrix.
        c_engine: Engine of the output tensor.
        a_engine: Engine of the activation matrix.
        b_engine: Engine of the weight matrix.
        simd_width: Number of elements per vectorized load; sets
            `tile_k` with `num_threads`.
        tile_m: Number of output rows each thread accumulates.
        tile_n: Number of weight rows each thread accumulates.
        num_threads: Threads per block; sets K-tile width and
            cross-warp reduction count.
        unroll_factor: K-loop unroll factor for instruction-level
            parallelism (defaults to 2).
        weight_non_temporal: When True, load the weight matrix with
            non-temporal (streaming) hints to bypass the cache (defaults
            to True).
        elementwise_lambda_fn: Optional epilogue applied to each
            output element.
        accum_type: Accumulation precision type.
        check_bounds_m: When True, guards M-tail rows when `m` is
            not a multiple of `tile_m`.
        check_bounds_n: When True, guards N-tail columns when `n` is
            not a multiple of `tile_n`.
        pdl_level: Programmatic dependent launch level for PDL
            barriers.

    Args:
        output: Output tensor, shape (m, n), row-major.
        act: Activation matrix, shape (m, k).
        weight: Weight matrix, shape (n, k), row-major transposed B.
        m: Number of activation rows, output rows.
        n: Number of weight rows, output columns.
        k: Reduction dimension shared by activation and weight.
    """
    var _m = Int(m)
    var _n = Int(n)
    var _k = Int(k)
    comptime assert output.flat_rank == 2, "output must be of rank 2"
    comptime assert act.flat_rank == 2, "act must be of rank 2"
    comptime assert weight.flat_rank == 2, "weight must be of rank 2"

    # tile_m represents how many rows each thread will process of the output activation matrix
    # tile_n represents how many rows each thread will process of the weight matrix.
    # Nvidia vectorized load is 16B.
    comptime tile_k = simd_width * num_threads
    # which rows of the activation matrix each thread will process
    var tile_id_m = block_idx.x * tile_m
    # which rows of the weight matrix each thread will process
    var tile_id_n = block_idx.y * tile_n
    var tid = thread_idx.x
    var tile_w = tt_stack_allocation[
        dtype=b_type,
        address_space=.LOCAL,
        alignment=simd_width * size_of[b_type](),
    ](row_major[tile_n, simd_width]())
    # these are the partial accumlations for each thread this a matrix of values
    # since each thread will process a tile_m x tile_n partials of the output vector
    var acc = tt_stack_allocation[dtype=accum_type, address_space=.LOCAL](
        row_major[tile_m, tile_n]()
    ).fill(0)
    var iteration = 0
    comptime WeightVecType = SIMD[b_type, simd_width]

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Each thread sums local data in K.
    @__parameter
    @always_inline
    def _k_iter_body():
        """Single K-iteration: load weights, load activations, accumulate."""
        var weight_tile = weight.tile[tile_n, tile_k](block_idx.y, iteration)
        var act_tile = act.tile[tile_m, tile_k](block_idx.x, iteration)

        # Load weights into tile_w.
        # Streaming is best when each weight is read once. Tiny-M router GEMV
        # launches one row block per M value, so those blocks should retain and
        # reuse the same weight rows instead.
        comptime for i in range(tile_n):
            comptime if check_bounds_n:
                if i + tile_id_n >= _n:
                    continue
            comptime if is_amd_gpu():
                var b_vec = weight_tile.load[
                    simd_width, non_temporal=weight_non_temporal
                ](Coord(i, thread_idx.x * simd_width))
                tile_w.store(Coord(i, Idx[0]), b_vec)
            else:
                var vec_weight_tile = weight_tile.vectorize[1, simd_width]()
                var b_vec = vec_weight_tile[i, thread_idx.x]
                tile_w.store(Coord(i, Idx[0]), b_vec)

        # Load activations and accumulate dot products.
        comptime for i in range(tile_m):
            comptime if check_bounds_m:
                if i + tile_id_m >= _m:
                    continue
            var act_vec = act_tile.vectorize[1, simd_width]()[i, thread_idx.x]

            # `act` is a_type, `tile_w` is b_type; keep each fragment in its
            # native element type so mixed A/B dtypes (e.g. bf16 activation ×
            # fp32 router weight) need no unsafe reinterpretation. `_dot_accum`
            # widens both to accum_type unless both are bf16 (fused fdot2).
            var act_native = rebind[SIMD[a_type, simd_width]](act_vec)
            comptime for j in range(tile_n):
                var weight_native = rebind[SIMD[b_type, simd_width]](
                    tile_w.vectorize[1, simd_width]()[j, 0]
                )
                var local_accum = rebind[Scalar[accum_type]](acc[i, j])
                local_accum = _dot_accum(act_native, weight_native, local_accum)
                acc[i, j] = local_accum

        iteration += 1

    comptime if unroll_factor == 1:
        # Simple loop — no ceildiv, no main_iters/remainder split.
        # Produces minimal PTX with fewest registers on NVIDIA.
        for _ in range(tid * simd_width, _k, tile_k):
            _k_iter_body()
    else:
        # Unrolled loop for ILP — comptime for duplicates the body.
        var k_start = tid * simd_width
        var num_k_iters = ceildiv(_k - k_start, tile_k) if _k > k_start else 0
        var main_iters = align_down(num_k_iters, unroll_factor)

        # Main unrolled loop.
        for _outer in range(0, main_iters, unroll_factor):
            comptime for _u in range(unroll_factor):
                _k_iter_body()

        # Remainder iterations (at most unroll_factor - 1).
        for _rem in range(main_iters, num_k_iters):
            _k_iter_body()

    # Warps are arranged along K.
    comptime k_warp_num = num_threads // WARP_SIZE
    var warp_id = warp_id()
    var lane_id = lane_id()
    var shmem = tt_stack_allocation[dtype=accum_type, address_space=.SHARED](
        row_major[1, tile_m * tile_n * k_warp_num]()
    )

    # Each warp sums across its threads and stages results in shared memory.
    # Shared memory data is row mojor (num_warps, tile_m, tile_n) stored in 1D.
    comptime for mi in range(tile_m):
        comptime for ni in range(tile_n):
            var val = warp.sum(acc[mi, ni])
            if lane_id == 0:
                shmem[0, mi * tile_n + ni + warp_id * tile_m * tile_n] = val
    barrier()
    # Sum across warps' results in shared memory then output (vectorized in N).
    for mid in range(tid, tile_m, num_threads):
        var vals = SIMD[accum_type, tile_n]()

        comptime for jj in range(k_warp_num):
            comptime for ni in range(tile_n):
                vals[ni] += shmem[0, jj * tile_m * tile_n + mid * tile_n + ni]

        var row = tile_id_m + mid
        var col = tile_id_n

        # The grid covers ceildiv(m, tile_m) * tile_m rows, so the last
        # block's tail rows fall outside the output when m % tile_m != 0.
        comptime if check_bounds_m:
            if row >= _m:
                continue

        comptime if check_bounds_n:
            comptime for ni in range(tile_n):
                if col + ni < _n:
                    comptime if elementwise_lambda_fn:
                        comptime elementwise_lambda = (
                            elementwise_lambda_fn.value()
                        )
                        elementwise_lambda(
                            Index(row, col + ni),
                            vals[ni].cast[c_type](),
                        )
                    else:
                        output[row, col + ni] = vals[ni].cast[c_type]()
        else:
            comptime if elementwise_lambda_fn:
                comptime elementwise_lambda = elementwise_lambda_fn.value()
                elementwise_lambda(Index(row, col), vals.cast[c_type]())
            else:
                comptime for ni in range(tile_n):
                    output[row, col + ni] = vals[ni].cast[c_type]()

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


def router_gate_use_mixed_gemv(m: Int) -> Bool:
    """Returns whether runtime `m` should take the fused mixed router GEMV.

    Only tiny-M (decode) rows use the single-launch mixed GEMV. `m == 0`
    (graph-capture warmup) takes neither path, and large `m` (prefill) must fall
    back to cast + fp32 matmul to preserve the baseline's performance.

    Args:
        m: The runtime row count of the router-gate activation.

    Returns:
        True if the fused mixed GEMV should be launched for this `m`.
    """
    comptime max_m = 64
    return m > 0 and m <= max_m


def router_gate_mixed_gemv[
    static_N: Int,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
](
    c: TileTensor[.float32, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[.bfloat16, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[.float32, b_layout, ImmutAnyOrigin, Engine=b_engine],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Launches the mixed bf16-activation × fp32-weight router-gate GEMV.

    Fuses the standalone bf16→fp32 activation cast into the router GEMV: `a` is
    loaded as bf16 and widened to fp32 in registers, then dotted against the
    unchanged fp32 weight `b` (`c = a @ b^T`) in a single launch. MiniMax-M3
    uses this path on MI355X; other architectures retain the standard router.

    Because bf16→fp32 widening is lossless and the reduction structure matches
    `gemv_split_k`, the result is numerically identical to casting `a` to fp32
    first and running the fp32 GEMV.

    The launch config is the MI355X (gfx950 / CDNA4) cache-busting sweep winner
    for the `N=128, K=6144` router-gate shape: `simd_width=4, tile_m=1,
    128 threads, unroll=2`, with `tile_n=2` at `M<=16` and `tile_n=4` above,
    and the weight kept cache-resident (`weight_non_temporal=False`) so the
    per-row-block weight rereads hit L2 — the same cache policy the merged
    KERN-3219 fp32 router path selects.

    Parameters:
        static_N: Static output width (weight rows / expert count). Selects the
            `check_bounds_n` guard at compile time.
        c_layout: Layout of the fp32 output tensor.
        a_layout: Layout of the bf16 activation tensor.
        b_layout: Layout of the fp32 weight tensor.
        c_engine: Engine of the fp32 output tensor.
        a_engine: Engine of the bf16 activation tensor.
        b_engine: Engine of the fp32 weight tensor.

    Args:
        c: Output `[M, N]` fp32 tensor.
        a: Activation `[M, K]` bf16 tensor.
        b: Weight `[N, K]` fp32 tensor (transpose_b layout).
        m: Runtime row count (`M`). Optimal for tiny `M` (router: `M<=64`).
        n: Output width (`N`).
        k: Contraction dim (`K`).
        ctx: The device context.
    """
    # gfx950 fp32 vectorized load = 16 B = 4 fp32 elements; the same element
    # count vectorizes the bf16 activation load (8 B), matching the K tiling.
    comptime simd_width = 16 // size_of[DType.float32]()
    comptime tile_m = 1
    comptime num_threads = 128
    comptime unroll_factor = 2
    # Empty-launch guard: a graph-capture warmup can call the router with M==0
    # (and N is never 0 for a real gate); skip the launch so no block reads or
    # writes past the zero-length buffers.
    if m == 0 or n == 0:
        return

    @__parameter
    def _launch[tile_n: Int]() raises:
        comptime kernel = gemv_split_k[
            DType.float32,
            DType.bfloat16,
            DType.float32,
            c_layout,
            a_layout,
            b_layout,
            c_engine,
            a_engine,
            b_engine,
            simd_width=simd_width,
            tile_m=tile_m,
            tile_n=tile_n,
            num_threads=num_threads,
            unroll_factor=unroll_factor,
            weight_non_temporal=False,
            check_bounds_m=tile_m > 1,
            check_bounds_n=static_N % tile_n != 0,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(m, tile_m), ceildiv(n, tile_n)),
            block_dim=num_threads,
        )

    if m <= 16:
        _launch[2]()
    else:
        _launch[4]()


# Row Vector-Matrix multiplication
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(tile_size))
)
@__name(t"gevm_kernel_{c_type}_{a_type}_{b_type}_{tile_size}")
def gevm_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    tile_size: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    pdl_level: PDLLevel = PDLLevel(),
](
    c: UnsafePointer[Scalar[c_type], AnyOrigin[mut=True]],
    a: UnsafePointer[Scalar[a_type], ImmUnsafeAnyOrigin],
    b: UnsafePointer[Scalar[b_type], ImmUnsafeAnyOrigin],
    m: Int32,
    n: Int32,
    k: Int32,
):
    var _k = Int(k)
    var _n = Int(n)
    comptime warps_per_block = tile_size // WARP_SIZE

    var warp_id = warp_id()
    var lane_id = lane_id()
    var col = block_idx.x * WARP_SIZE + lane_id
    var global_warp_id = global_idx.x // warps_per_block

    var x_shared = unsafe_stack_allocation[
        tile_size,
        accum_type,
        address_space=.SHARED,
    ]()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    var accum = Scalar[accum_type]()

    # Every block computes warp size length of output values
    for i in range(ceildiv(_k, warps_per_block)):
        var row = i * warps_per_block + warp_id
        var lhs = a[row]
        var rhs = b[row * _n + col]
        accum += lhs.cast[accum_type]() * rhs.cast[accum_type]()

    x_shared[lane_id * warps_per_block + warp_id] = accum
    barrier()

    var total = warp.lane_group_sum[num_lanes=warps_per_block](
        x_shared[thread_idx.x]
    )

    if lane_id % warps_per_block == 0:
        comptime if elementwise_lambda_fn:
            comptime elementwise_lambda = elementwise_lambda_fn.value()
            elementwise_lambda(Index(0, global_warp_id), total.cast[c_type]())
        else:
            c[global_warp_id] = total.cast[c_type]()

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


def _amd_gemv_config[
    simd_width: Int,
    max_thread_block_size: Int,
    static_K: Int,
    has_N: Bool,
    static_N: Int,
]() -> IndexList[3]:
    """Compute GEMV split-K dispatch config for AMD GPUs.

    Returns (num_threads, tile_n, unroll_factor).

    Works for both FP8 (simd_width=16) and BF16 (simd_width=8). All
    thresholds derive from simd_width, WARP_SIZE, and
    max_thread_block_size.

    Thread count: pick from {64, 128, 256} to balance wave parallelism
    vs K-iteration count (tile_k = num_threads × simd_width). Single warp
    (64T) for K≤2048 (both BF16 and FP8) to avoid LDS sync. 256T when K
    provides ≥2 clean iterations or exactly 1 clean iteration. 128T for
    mid-K with bad fractional iterations at 256T.

    tile_n: when there are enough K-iterations (≥3 for BF16, ≥4 for FP8),
    pick the largest tile_n from {4,2,1} that gives ≥ min_waves_per_simd
    waves/SIMD. Otherwise default to tile_n=2 (grid parallelism >
    loads-per-iter).
    """
    comptime tile_k_256 = 256 * simd_width
    # BF16 (sw=8) has 2× more K-iterations than FP8 (sw=16) for the same K,
    # so each wave keeps the SIMD busy longer and fewer waves/SIMD suffice.
    # FP8 needs ≥10 waves/SIMD (Exp Q showed tile_n=1 optimal for small N).
    # BF16 needs ≥5 waves/SIMD to hide L2 latency on K=16384 shapes.
    comptime min_waves_per_simd = 5 if simd_width <= 8 else 10

    # --- Thread count ---
    # Single warp (64T) avoids LDS cross-warp reduction overhead.
    # BF16: K≤1024 (tile_k=512, 2 iters), FP8: K≤2048 (tile_k=1024, 2 iters).
    var num_threads: Int
    if static_K <= 2 * WARP_SIZE * simd_width:
        num_threads = 64
    elif static_K >= 2 * tile_k_256 or static_K % tile_k_256 == 0:
        # ≥2 clean iterations, or exactly 1 clean iteration at 256T.
        num_threads = 256
    else:
        # Mid-K with fractional iters at 256T. 128T halves tile_k,
        # giving more iterations with better pipelining.
        num_threads = 128

    # --- tile_n ---
    # With ≥4 K-iterations per wave, there's enough work to tolerate
    # fewer grid blocks — pick largest tile_n with sufficient waves/SIMD.
    # With <4 iterations, grid parallelism matters more — keep tile_n=2.
    var tile_n = 2
    var k_iters = static_K // (num_threads * simd_width)
    # BF16 has NT loads + fdot2 doing more work per iteration, so tile_n=4
    # is profitable at fewer K-iterations (≥3 vs ≥4 for FP8).
    comptime min_k_iters_for_tile_n = 3 if simd_width <= 8 else 4
    if k_iters >= min_k_iters_for_tile_n and has_N:
        var wavefront_capacity = static_N * (num_threads // WARP_SIZE)
        if wavefront_capacity >= min_waves_per_simd * max_thread_block_size * 4:
            tile_n = 4
        elif (
            wavefront_capacity >= min_waves_per_simd * max_thread_block_size * 2
        ):
            tile_n = 2
        else:
            # tile_n=1 only benefits FP8 (more grid parallelism needed).
            # BF16 has more work per iteration, so tile_n=2 is the floor.
            tile_n = 1 if simd_width > 8 else 2

    # unroll=4 when there are enough K-iterations and tile_n is small enough
    # to avoid register pressure (tile_n=4 + unroll=4 hurts large-N shapes).
    var unroll = 4 if k_iters >= 8 and tile_n <= 2 else 2
    return IndexList[3](num_threads, tile_n, unroll)


def _nvidia_gemv_config[
    a_type: DType,
    simd_width: Int,
    static_K: Int,
    has_N: Bool,
    static_N: Int,
]() -> IndexList[3]:
    """Compute GEMV split-K dispatch config for NVIDIA B200 GPUs.

    Returns (num_threads, tile_n, unroll_factor).
    B200 has 160 SMs, warp size 32.
    """
    comptime tile_k_256 = 256 * simd_width
    comptime tile_k_128 = 128 * simd_width

    # FP32 (16B vectorized load -> simd_width 4 on B200). The tiny-M / small-N
    # router GEMM (e.g. M<=16, N=128, K=6144) is HBM-bandwidth-bound on the N*K
    # weight, so tile_n=1 maximizes the launched CTA count (one output column
    # per block); 256 threads/block and unroll=2 are the swept winner, while
    # tile_n>=2 and 128T regress. (KERN-3076.)
    comptime if a_type == .float32:
        return IndexList[3](256, 1, 2)

    var num_threads: Int
    comptime if simd_width <= 8:
        # BF16: 128T default. 256T only for large N with ~4 k_iters
        # at 128T, where halving iterations improves BW utilization.
        if (
            has_N
            and static_N >= 16384
            and static_K >= 4 * tile_k_128
            and static_K < 5 * tile_k_128
        ):
            num_threads = 256
        else:
            num_threads = 128
    else:
        # FP8: scale threads with K.
        if static_K < 3 * tile_k_128:
            num_threads = 64
        elif static_K >= 4 * tile_k_256:
            num_threads = 256
        else:
            num_threads = 128

    # tile_n=4 halves grid but doubles weight loads per block.
    var tile_n = 2
    # k_iters is per-thread K work (tile_n affects N, not K).
    var k_iters = static_K // (num_threads * simd_width)
    # Only use tile_n=4 at 128T; 256T + tile_n=4 regresses BF16.
    if num_threads <= 128 and k_iters >= 3 and has_N:
        var blocks_tn4 = static_N // 4
        if k_iters <= 3:
            tile_n = 4
        elif k_iters <= 6 and blocks_tn4 >= 960:
            tile_n = 4
        elif blocks_tn4 >= 960 and blocks_tn4 < 1600:
            tile_n = 4
        else:
            tile_n = 2
    elif has_N:
        var blocks_tn2 = static_N // 2
        if blocks_tn2 < 160:
            tile_n = 1
        else:
            tile_n = 2

    # BF16: always unroll=1 (I-cache sensitive due to scalar FMA chain).
    # FP8: unroll benefits from fewer instructions per iteration.
    var unroll: Int
    comptime if simd_width <= 8:
        unroll = 1
    else:
        if k_iters == 4:
            unroll = 4
        elif k_iters >= 3:
            unroll = 2
        else:
            unroll = 1
    return IndexList[3](num_threads, tile_n, unroll)


# Row tile of `gemv_kernel_vector_multirow`, and the K depth up to which it
# pays. The one-row launch gives a CTA `k_iters = ceildiv(K, WARP_SIZE *
# simd_width)` warps and `k_iters * K * size_of[dtype]()` bytes of work; the
# CTA launch/retire pipeline runs at roughly 1 CTA/ns chip-wide, so a CTA must
# carry several KB for the grid to keep HBM busy. At `k_iters <= 2` it carries
# at most 2 KB and the kernel runs far below peak no matter how wide N is,
# and four rows per warp is the measured optimum there (B200, N=262144 bf16,
# met us):
#
# | k_iters   | one row | 2 rows | 4 rows | 8 rows | 16 rows | 32 rows |
# |-----------|---------|--------|--------|--------|---------|---------|
# | 1 (K=256) | 257.1   | 34.7   | 25.8   | 48.5   | 36.5    | 117.1   |
# | 2 (K=512) | 130.2   | 56.1   | 42.7   |        |         |         |
#
# Deeper K already fills the pipe from the K loop alone (a K=3840 CTA carries
# 115 KB and measures 5.4 TB/s in the serve today), so it keeps one row per
# warp and its launch is unchanged.
comptime _GEMV_MULTIROW_ROWS = 4
comptime _GEMV_MULTIROW_MAX_K_ITERS = 2


@always_inline
def is_minimax_router_gemm[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    static_N: Int,
    static_K: Int,
]() -> Bool:
    """Returns whether a GEMM has the MiniMax-M3 fp32 router signature."""
    return (
        a_type == .float32
        and b_type == .float32
        and c_type == .float32
        and static_N == 128
        and static_K == 6144
    )


@always_inline
def gemv_gpu_dispatch[
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel.ON,
    tile_m: Int = 1,
](
    kernel_func: GEMVAlgorithm,
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...],
    ctx: DeviceContext,
) raises:
    """Launches the GPU GEMV kernel indicated by kernel_func with appropriate grid and block dims.

    Translates a `GEMVAlgorithm` variant into a concrete kernel call with shape-derived
    launch parameters, handling input/output layout transformation where needed.

    Parameters:
        transpose_b: When True, B is treated as transposed (N, K) row-major.
        elementwise_lambda_fn: Optional epilogue applied element-wise to each output.
        pdl_level: Programmatic dependent launch level.
        tile_m: Number of output rows processed per CTA (used by GEMV_SPLIT_K).

    Args:
        kernel_func: The GEMV algorithm variant to launch.
        c: Rank-2 output TileTensor.
        a: Rank-2 input matrix TileTensor.
        b: Rank-2 input vector or matrix TileTensor.
        ctx: Device context for kernel launch.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"

    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K

    comptime WARPS_PER_BLOCK = 1024 // WARP_SIZE
    comptime c_type = c.dtype
    comptime a_type = a.dtype
    comptime b_type = b.dtype
    comptime simd_width = simd_width_of[a_type, target=get_gpu_target()]()

    comptime has_N = c.static_shape[1] > -1
    comptime static_N = c.static_shape[1] if has_N else UNKNOWN_VALUE
    comptime static_K = a.static_shape[1]

    if kernel_func is GEMVAlgorithm.GEMV_SPLIT_K:
        logger.info("Executing: GEMV_SPLIT_K kernel")

        @__parameter
        def _gemv_split_k_dispatch[
            num_threads: Int,
            tile_n: Int,
            unroll_factor: Int = 2,
            weight_non_temporal: Bool = True,
        ]() raises:
            comptime kernel = gemv_split_k[
                c_type,
                a_type,
                b_type,
                type_of(c).LayoutType,
                type_of(a).LayoutType,
                type_of(b).LayoutType,
                type_of(c).Engine,
                type_of(a).Engine,
                type_of(b).Engine,
                simd_width=simd_width,
                tile_m=tile_m,
                tile_n=tile_n,
                num_threads=num_threads,
                unroll_factor=unroll_factor,
                weight_non_temporal=weight_non_temporal,
                elementwise_lambda_fn=elementwise_lambda_fn,
                check_bounds_m=tile_m > 1,
                check_bounds_n=static_N % tile_n != 0,
                pdl_level=pdl_level,
            ]
            ctx.enqueue_function[kernel](
                c,
                a,
                b,
                Int32(m),
                Int32(n),
                Int32(k),
                grid_dim=(ceildiv(m, tile_m), ceildiv(n, tile_n)),
                block_dim=num_threads,
                attributes=pdl_launch_attributes(pdl_level),
            )

        comptime if has_amd_gpu_accelerator():
            comptime if is_minimax_router_gemm[
                c_type, a_type, b_type, static_N, static_K
            ]():
                # MiniMax-M3 router gate: M row blocks reread the same 3 MB
                # weight. Cached loads turn those rereads into L2/MALL hits.
                _gemv_split_k_dispatch[128, 2, 2, False]()
            else:
                comptime config = _amd_gemv_config[
                    simd_width,
                    ctx.default_device_info.max_thread_block_size,
                    static_K,
                    has_N,
                    static_N,
                ]()
                _gemv_split_k_dispatch[
                    config[0],
                    config[1],
                    config[2],
                ]()
        else:
            # NVIDIA B200: shape-dependent dispatch for FP8 and BF16.
            comptime config = _nvidia_gemv_config[
                a_type,
                simd_width,
                static_K,
                has_N,
                static_N,
            ]()
            _gemv_split_k_dispatch[
                config[0],
                config[1],
                config[2],
            ]()

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL_VECTOR:
        logger.info("Executing: GEMV_KERNEL_VECTOR kernel")

        comptime check_bounds_k = static_K % (WARP_SIZE * simd_width) != 0
        var block_dim = min(
            align_up(k // simd_width, WARP_SIZE),
            WARP_SIZE * WARPS_PER_BLOCK,
        )
        if n == 1:
            comptime if transpose_b:
                comptime kernel = gemv_kernel_vector[
                    c_type,
                    a_type,
                    b_type,
                    type_of(c).LayoutType,
                    type_of(a).LayoutType,
                    type_of(b).LayoutType,
                    type_of(c).Engine,
                    type_of(a).Engine,
                    type_of(b).Engine,
                    simd_width=simd_width,
                    transpose_b=False,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    check_bounds=check_bounds_k,
                    pdl_level=pdl_level,
                ]
                ctx.enqueue_function[kernel](
                    c,
                    a,
                    b,
                    Int32(m),
                    Int32(n),
                    Int32(k),
                    grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(pdl_level),
                )
            else:
                # runtime transpose since TileTensor.transpose requires static shape
                var b_n_major_layout = row_major(Coord(n, k))
                var b_ptr = UnsafePointer[Scalar[b_type], b.origin](
                    unsafe_from_address=Int(b.ptr)
                )
                var b_tile_n_major = TileTensor[
                    b_type, type_of(b_n_major_layout), b.origin
                ](b_ptr, b_n_major_layout)

                comptime kernel = gemv_kernel_vector[
                    c_type,
                    a_type,
                    b_type,
                    type_of(c).LayoutType,
                    type_of(a).LayoutType,
                    type_of(b_tile_n_major).LayoutType,
                    type_of(c).Engine,
                    type_of(a).Engine,
                    type_of(b_tile_n_major).Engine,
                    simd_width=simd_width,
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    check_bounds=check_bounds_k,
                    pdl_level=pdl_level,
                ]
                ctx.enqueue_function[kernel](
                    c,
                    a,
                    b_tile_n_major,
                    Int32(m),
                    Int32(n),
                    Int32(k),
                    grid_dim=ceildiv(m, block_dim // WARP_SIZE),
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(pdl_level),
                )
        elif m == 1:

            @__parameter
            def _one_row_per_warp() raises:
                comptime kernel = gemv_kernel_vector[
                    c_type,
                    b_type,
                    a_type,
                    type_of(c).LayoutType,
                    type_of(b).LayoutType,
                    type_of(a).LayoutType,
                    type_of(c).Engine,
                    type_of(b).Engine,
                    type_of(a).Engine,
                    simd_width=simd_width,
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    check_bounds=check_bounds_k,
                    pdl_level=pdl_level,
                ]
                ctx.enqueue_function[kernel](
                    c,
                    b,
                    a,
                    Int32(n),
                    Int32(m),
                    Int32(k),
                    grid_dim=ceildiv(n, block_dim // WARP_SIZE),
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(pdl_level),
                )

            @__parameter
            def _rows_per_warp[rows: Int]() raises:
                logger.info("Rows per warp: ", rows)
                # 128-thread blocks measured slightly ahead of 256 and stay
                # clear of the 64K-register-per-block ceiling that a wide row
                # tile runs into.
                comptime warps_per_block = 4
                comptime kernel = gemv_kernel_vector_multirow[
                    c_type,
                    b_type,
                    a_type,
                    type_of(c).LayoutType,
                    type_of(b).LayoutType,
                    type_of(a).LayoutType,
                    type_of(c).Engine,
                    type_of(b).Engine,
                    type_of(a).Engine,
                    simd_width=simd_width,
                    rows_per_warp=rows,
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                    check_bounds=check_bounds_k,
                    pdl_level=pdl_level,
                ]
                ctx.enqueue_function[kernel](
                    c,
                    b,
                    a,
                    Int32(n),
                    Int32(m),
                    Int32(k),
                    grid_dim=ceildiv(n, rows * warps_per_block),
                    block_dim=WARP_SIZE * warps_per_block,
                    attributes=pdl_launch_attributes(pdl_level),
                )

            @__parameter
            def _grid_outruns_chip() -> Bool:
                # One warp per output row needs `n` resident warps. Once that
                # is several times what the chip can hold, the grid drains at
                # the CTA launch/retire rate and never reaches HBM bandwidth,
                # so the tile is worth widening. The 4x margin also keeps the
                # widened grid above one block per SM.
                var resident_warps = (
                    ctx.default_device_info.sm_count
                    * ctx.default_device_info.threads_per_multiprocessor
                    // WARP_SIZE
                )
                return n > 4 * resident_warps

            comptime multirow_max_k = (
                _GEMV_MULTIROW_MAX_K_ITERS * WARP_SIZE * simd_width
            )

            comptime if static_K <= 0:
                # A graph that leaves K symbolic still gets the fix; testing
                # `k` here costs one extra instantiation, missing it costs 10x.
                if k <= multirow_max_k and _grid_outruns_chip():
                    _rows_per_warp[_GEMV_MULTIROW_ROWS]()
                else:
                    _one_row_per_warp()
            elif static_K <= multirow_max_k:
                if _grid_outruns_chip():
                    _rows_per_warp[_GEMV_MULTIROW_ROWS]()
                else:
                    _one_row_per_warp()
            else:
                _one_row_per_warp()

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL and transpose_b == False:
        logger.info("Executing: GEMV_KERNEL (no transpose)")

        comptime kernel = gemv_kernel[
            c_type,
            a_type,
            b_type,
            elementwise_lambda_fn=elementwise_lambda_fn,
            pdl_level=pdl_level,
        ]

        ctx.enqueue_function[kernel](
            c.to_device_buffer(ctx),
            a.to_device_buffer(ctx),
            b.to_device_buffer(ctx),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=ceildiv(m, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            attributes=pdl_launch_attributes(pdl_level),
        )

    elif kernel_func is GEMVAlgorithm.GEMV_KERNEL and transpose_b == True:
        logger.info("Executing: GEMV_KERNEL (with transpose)")

        comptime kernel = gemv_kernel[
            c_type,
            b_type,
            a_type,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            pdl_level=pdl_level,
        ]
        ctx.enqueue_function[kernel](
            c.to_device_buffer(ctx),
            b.to_device_buffer(ctx),
            a.to_device_buffer(ctx),
            Int32(n),
            Int32(m),
            Int32(k),
            grid_dim=ceildiv(n, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            attributes=pdl_launch_attributes(pdl_level),
        )
    elif kernel_func is GEMVAlgorithm.GEVM_KERNEL:
        logger.info("Executing: GEVM_KERNEL")
        comptime kernel = gevm_kernel[
            c_type,
            a_type,
            b_type,
            tile_size=WARP_SIZE * WARPS_PER_BLOCK,
            elementwise_lambda_fn=elementwise_lambda_fn,
            pdl_level=pdl_level,
        ]
        ctx.enqueue_function[kernel](
            c.to_device_buffer(ctx),
            a.to_device_buffer(ctx),
            b.to_device_buffer(ctx),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=ceildiv(n, WARP_SIZE),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
            attributes=pdl_launch_attributes(pdl_level),
        )

    else:
        logger.info("Executing: MATMUL_NAIVE kernel")
        comptime BLOCK_DIM = 16

        comptime kernel = matmul_kernel_naive[
            c_type,
            a_type,
            b_type,
            type_of(c).LayoutType,
            type_of(a).LayoutType,
            type_of(b).LayoutType,
            BLOCK_DIM,
            transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            c_engine=type_of(c).Engine,
            a_engine=type_of(a).Engine,
            b_engine=type_of(b).Engine,
        ]
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )


def log_shape[
    has_mode_1: Bool, has_mode_2: Bool, name: String
](mode_1: Int, mode_2: Int,) -> None:
    """Logs the shape of a named tensor dimension pair to the info logger.

    Parameters:
        has_mode_1: When True, prefixes the first dimension value with an underscore
            to indicate a dynamic dimension.
        has_mode_2: When True, prefixes the second dimension value with an underscore.
        name: The label to print before the shape tuple.

    Args:
        mode_1: Value of the first dimension.
        mode_2: Value of the second dimension.
    """
    logger.info(
        name,
        ": (",
        "_" if has_mode_1 else "",
        mode_1,
        ", ",
        "_" if has_mode_2 else "",
        mode_2,
        ")",
        sep="",
    )


@always_inline
def gemv_gpu[
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...],
    ctx: DeviceContext,
) raises:
    """Selects and dispatches the appropriate GPU GEMV kernel based on shape and hardware.

    Examines runtime dimensions M, N, K and static shape information to choose among
    GEMV_KERNEL, GEMV_KERNEL_VECTOR, GEMV_SPLIT_K, GEVM_KERNEL, and MATMUL_NAIVE variants.

    Parameters:
        transpose_b: When True, B is treated as transposed (N, K) row-major.
        elementwise_lambda_fn: Optional epilogue applied element-wise to each output.
        pdl_level: Programmatic dependent launch level for PDL barriers.

    Args:
        c: Rank-2 output TileTensor.
        a: Rank-2 input matrix TileTensor.
        b: Rank-2 input vector or matrix TileTensor.
        ctx: Device context for kernel launch.
    """
    comptime assert c.rank == 2, "c must be of rank 2"
    comptime assert a.rank == 2, "a must be of rank 2"
    comptime assert b.rank == 2, "b must be of rank 2"

    comptime a_type = a.dtype
    comptime b_type = b.dtype
    comptime c_type = c.dtype

    var shape = GemmShape.get[transpose_b=False](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K
    comptime simd_width = simd_width_of[a_type, target=get_gpu_target()]()

    comptime has_M = c.static_shape[0] > -1
    comptime has_N = c.static_shape[1] > -1
    comptime has_K = a.static_shape[1] > -1
    comptime static_N = c.static_shape[1] if has_N else UNKNOWN_VALUE
    comptime static_K = a.static_shape[1] if has_K else UNKNOWN_VALUE

    logger.info("------ Dispatching to GEMV ------")

    # Log dimension static/dynamic status
    log_shape[has_M, has_K, "A"](m, k)
    log_shape[has_K, has_N, "B"](k, n)
    log_shape[has_M, has_N, "C"](m, n)

    # Kernel selection
    var kernel_func: GEMVAlgorithm

    if n == 1:
        comptime if a_type == .bfloat16:
            if k % simd_width == 0:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL_VECTOR
            else:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL
        else:
            kernel_func = GEMVAlgorithm.GEMV_KERNEL

    elif (
        m == 1
        or (
            has_N
            and has_K
            and is_minimax_router_gemm[
                c_type, a_type, b_type, static_N, static_K
            ]()
            and m <= 16
        )
    ) and transpose_b == True:
        comptime if a_type in (
            DType.float32,
            DType.bfloat16,
            DType.float16,
            DType.float8_e4m3fn,
        ):
            if k % simd_width == 0:
                if ceildiv(n, 2) <= ctx.get_attribute(
                    DeviceAttribute.MAX_GRID_DIM_Y
                ):
                    kernel_func = GEMVAlgorithm.GEMV_SPLIT_K
                else:
                    kernel_func = GEMVAlgorithm.GEMV_KERNEL_VECTOR
            else:
                kernel_func = GEMVAlgorithm.GEMV_KERNEL
        else:
            kernel_func = GEMVAlgorithm.GEMV_KERNEL

    elif m == 1 and n % WARP_SIZE == 0 and k % WARP_SIZE == 0:
        kernel_func = GEMVAlgorithm.GEVM_KERNEL

    else:
        kernel_func = GEMVAlgorithm.MATMUL_NAIVE

    gemv_gpu_dispatch[
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
        pdl_level=pdl_level,
    ](kernel_func, c, a, b, ctx)


# Parallelized version of Gemv


@always_inline
def gemv[
    parallelize: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_buf: TileTensor[mut=True, ...],
    a_buf: TileTensor[mut=False, ...],
    b_buf: TileTensor[mut=False, ...],
) raises:
    """Computes a CPU matrix-vector product C = A * b using vectorized element-wise reduction.

    Optionally parallelizes across rows. Accumulates the product of each row of A
    with the vector b into the corresponding element of c.

    Parameters:
        parallelize: When True, the reduction runs in parallel across output rows.
        elementwise_lambda_fn: Optional epilogue applied to each output element.

    Args:
        c_buf: Output TileTensor of length M.
        a_buf: Input matrix TileTensor of shape (M, K).
        b_buf: Input vector TileTensor of length K.
    """
    comptime c_type = c_buf.dtype
    comptime simd_width = simd_width_of[c_type]()

    var M = Int(a_buf.dim[0]())
    var K = Int(a_buf.dim[1]())

    @always_inline
    @__parameter
    def input_fn[
        dtype: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[dtype, width]:
        return (
            a_buf.load_linear[width=width](Index(idx[0], idx[1])).cast[dtype]()
            * b_buf.load_linear[width=width](IndexList[1](idx[1])).cast[dtype]()
        ).cast[dtype]()

    @always_inline
    @__parameter
    def output_fn[
        out_type: DType, width: SIMDLength, rank: Int
    ](idx: IndexList[rank], value: SIMD[out_type, width]):
        comptime if elementwise_lambda_fn:
            comptime func = elementwise_lambda_fn.value()

            comptime for i in range(width):
                func[out_type, 1]((idx[0] + i, 0), value[i])
        else:
            c_buf.store_linear[width=width](
                IndexList[1](idx[0]), value.cast[c_type]()
            )

    @always_inline
    @__parameter
    def reduce_impl[
        ty: DType, width: SIMDLength
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    _reduce_generator[
        input_fn,
        output_fn,
        reduce_impl,
        reduce_dim=1,
    ](
        Coord((M, K)),
        init=Scalar[c_type](0),
    )


def naive_gemv(
    c_buf: TileTensor[mut=True, ...],
    a_buf: TileTensor[mut=False, ...],
    b_buf: TileTensor[mut=False, ...],
):
    """Computes a reference matrix-vector product C = A * b using a scalar nested loop.

    Iterates over K then M, accumulating each A[m, k] * b[k] into c[m]. Intended for
    correctness testing rather than performance-critical paths.

    Args:
        c_buf: Output TileTensor of length M, zero-filled on entry.
        a_buf: Input matrix TileTensor of shape (M, K).
        b_buf: Input vector TileTensor of length K.
    """
    comptime c_type = c_buf.dtype
    var M = Int(a_buf.dim[0]())
    var K = Int(a_buf.dim[1]())
    var c_ptr = c_buf.ptr
    var a_ptr = a_buf.ptr
    var b_ptr = b_buf.ptr

    _ = c_buf.fill(0)
    for k in range(K):
        var b_val = b_ptr[k].cast[c_type]()
        for m in range(M):
            var a_val = a_ptr[m * K + k].cast[c_type]()
            c_ptr[m] += a_val * b_val


struct _MmaCpAsyncGmemLoaderA[
    origin: Origin,
    //,
    a_type: DType,
    a_layout: TensorLayout,
    tile_m: Int,
    tile_k: Int,
    stage_cnt: Int,
]:
    """Producer warp pair (warps 0-1) for activation tiles."""

    comptime LOAD_THREADS = 64
    comptime VEC_ELEMS = simd_width_of[Self.a_type, target=get_gpu_target()]()
    comptime VEC_BYTES = Self.VEC_ELEMS * size_of[Self.a_type]()
    comptime vec_per_iter = (Self.tile_m * Self.tile_k) // (
        Self.VEC_ELEMS * Self.LOAD_THREADS
    )
    comptime per_warp_k = Self.tile_k // 4
    comptime SmemTiles = SMemTileArray2D[
        Self.a_type, Self.tile_m, Self.tile_k, Self.stage_cnt
    ]
    comptime Barriers = SMemArray[SharedMemBarrier, Self.stage_cnt * 2]
    comptime ActTensor = TileTensor[Self.a_type, Self.a_layout, Self.origin]
    comptime swizzle = make_swizzle[8, Self.tile_k, 8]()

    @always_inline
    def _k_project(
        self,
        tile_k_idx: Int,
    ) -> Int:
        return (tile_k_idx // Self.per_warp_k) * self.k_each_chunk + (
            tile_k_idx % Self.per_warp_k
        )

    var act: Self.ActTensor
    var smem_a: Self.SmemTiles
    var smem_barrier: Self.Barriers
    var local_tid: Int
    var batch_idx: Int
    var cta_m: Int
    var gemm_m: Int
    var k_each_chunk: Int
    var stage: Int
    var phase: UInt32
    var need_wait: Bool
    var smem_offsets: Array[Int, Self.vec_per_iter]
    var preds: Array[Bool, Self.vec_per_iter]

    def __init__(
        out self,
        act: Self.ActTensor,
        smem_a: Self.SmemTiles,
        smem_barrier: Self.Barriers,
        local_tid: Int,
        batch_idx: Int,
        cta_m: Int,
        gemm_m: Int,
        k_each_chunk: Int,
    ):
        self.act = act
        self.smem_a = smem_a
        self.smem_barrier = smem_barrier
        self.local_tid = local_tid
        self.batch_idx = batch_idx
        self.cta_m = cta_m
        self.gemm_m = gemm_m
        self.k_each_chunk = k_each_chunk
        self.stage = 0
        self.phase = UInt32(1)
        self.need_wait = True
        self.smem_offsets = Array[Int, Self.vec_per_iter](uninitialized=True)
        self.preds = Array[Bool, Self.vec_per_iter](fill=False)

    def prepare(mut self):
        comptime for v in range(Self.vec_per_iter):
            var linear = (
                self.local_tid * Self.VEC_ELEMS
                + v * Self.LOAD_THREADS * Self.VEC_ELEMS
            )
            var m_idx = linear // Self.tile_k
            self.smem_offsets[v] = Self.swizzle(linear)
            self.preds[v] = self.cta_m + m_idx < self.gemm_m

    def issue_mainloop(mut self, k_iters: Int):
        var gmem_a_off = 0
        for loop_idx in range(k_iters):
            if self.need_wait:
                self.smem_barrier[1 + self.stage * 2][].wait(self.phase)

            var raw_next = self.stage + 1
            var next_phase = self.phase ^ (
                UInt32(1) if raw_next == Self.stage_cnt else UInt32(0)
            )
            var next_stage = 0 if raw_next == Self.stage_cnt else raw_next
            if loop_idx != k_iters - 1:
                self.need_wait = not self.smem_barrier[
                    1 + next_stage * 2
                ][].try_wait(next_phase)

            var gmem_base = self.act.ptr.address_space_cast[.GLOBAL]()
            var smem_tile_ptr = self.smem_a[self.stage].ptr
            comptime for v in range(Self.vec_per_iter):
                var linear = (
                    self.local_tid * Self.VEC_ELEMS
                    + v * Self.LOAD_THREADS * Self.VEC_ELEMS
                )
                var m_idx = linear // Self.tile_k
                var k_idx = linear % Self.tile_k
                var gmem_k = self._k_project(k_idx) + gmem_a_off
                if self.preds[v]:
                    var offset = self.act._linear_offset(
                        Index(self.batch_idx, self.cta_m + m_idx, gmem_k)
                    )
                    async_copy[16, bypass_L1_16B=True, l2_prefetch=128](
                        gmem_base + Int(offset),
                        smem_tile_ptr + self.smem_offsets[v],
                    )
                else:
                    # The grid covers ceildiv(gemm_m, tile_m) * tile_m rows;
                    # zero-fill rows past gemm_m instead of reading OOB. The
                    # epilogue's row guard discards their results.
                    async_copy[
                        Self.VEC_BYTES,
                        bypass_L1_16B=True,
                        fill=Scalar[Self.a_type](0),
                    ](
                        gmem_base,
                        smem_tile_ptr + self.smem_offsets[v],
                        src_size=0,
                    )

            async_copy_arrive[noinc=True](self.smem_barrier[self.stage * 2])

            gmem_a_off += Self.per_warp_k
            self.stage = next_stage
            self.phase = next_phase


struct _MmaCpAsyncGmemLoaderB[
    weight_origin: ImmOrigin,
    //,
    b_type: DType,
    b_layout: TensorLayout,
    tile_n: Int,
    tile_k: Int,
    stage_cnt: Int,
]:
    """Producer warp pair (warps 2-3) for weight tiles."""

    comptime LOAD_THREADS = 64
    comptime VEC_ELEMS = simd_width_of[Self.b_type, target=get_gpu_target()]()
    comptime VEC_BYTES = Self.VEC_ELEMS * size_of[Self.b_type]()
    comptime vec_per_iter = (Self.tile_n * Self.tile_k) // (
        Self.VEC_ELEMS * Self.LOAD_THREADS
    )
    comptime per_warp_k = Self.tile_k // 4
    comptime SmemTiles = SMemTileArray2D[
        Self.b_type, Self.tile_n, Self.tile_k, Self.stage_cnt
    ]
    comptime Barriers = SMemArray[SharedMemBarrier, Self.stage_cnt * 2]
    comptime WeightTensor = TileTensor[
        Self.b_type, Self.b_layout, Self.weight_origin
    ]
    comptime swizzle = make_swizzle[8, Self.tile_k, 8]()

    @always_inline
    def _k_project(
        self,
        tile_k_idx: Int,
    ) -> Int:
        return (tile_k_idx // Self.per_warp_k) * self.k_each_chunk + (
            tile_k_idx % Self.per_warp_k
        )

    var weight: Self.WeightTensor
    var smem_b: Self.SmemTiles
    var smem_barrier: Self.Barriers
    var local_tid: Int
    var batch_idx: Int
    var cta_n: Int
    var gemm_n: Int
    var k_each_chunk: Int
    var stage: Int
    var phase: UInt32
    var need_wait: Bool
    var smem_offsets: Array[Int, Self.vec_per_iter]
    var preds: Array[Bool, Self.vec_per_iter]

    def __init__(
        out self,
        weight: Self.WeightTensor,
        smem_b: Self.SmemTiles,
        smem_barrier: Self.Barriers,
        local_tid: Int,
        batch_idx: Int,
        cta_n: Int,
        gemm_n: Int,
        k_each_chunk: Int,
    ):
        self.weight = weight
        self.smem_b = smem_b
        self.smem_barrier = smem_barrier
        self.local_tid = local_tid
        self.batch_idx = batch_idx
        self.cta_n = cta_n
        self.gemm_n = gemm_n
        self.k_each_chunk = k_each_chunk
        self.stage = 0
        self.phase = UInt32(1)
        self.need_wait = True
        self.smem_offsets = Array[Int, Self.vec_per_iter](uninitialized=True)
        self.preds = Array[Bool, Self.vec_per_iter](fill=False)

    def prepare(mut self):
        comptime for v in range(Self.vec_per_iter):
            var linear = (
                self.local_tid * Self.VEC_ELEMS
                + v * Self.LOAD_THREADS * Self.VEC_ELEMS
            )
            var n_idx = linear // Self.tile_k
            self.smem_offsets[v] = Self.swizzle(linear)
            self.preds[v] = self.cta_n + n_idx < self.gemm_n

    def issue_mainloop(mut self, k_iters: Int):
        var gmem_b_off = 0
        for loop_idx in range(k_iters):
            if self.need_wait:
                self.smem_barrier[1 + self.stage * 2][].wait(self.phase)

            var raw_next = self.stage + 1
            var next_phase = self.phase ^ (
                UInt32(1) if raw_next == Self.stage_cnt else UInt32(0)
            )
            var next_stage = 0 if raw_next == Self.stage_cnt else raw_next
            if loop_idx != k_iters - 1:
                self.need_wait = not self.smem_barrier[
                    1 + next_stage * 2
                ][].try_wait(next_phase)

            var gmem_base = self.weight.ptr.address_space_cast[.GLOBAL]()
            var smem_tile_ptr = self.smem_b[self.stage].ptr
            comptime for v in range(Self.vec_per_iter):
                var linear = (
                    self.local_tid * Self.VEC_ELEMS
                    + v * Self.LOAD_THREADS * Self.VEC_ELEMS
                )
                var n_idx = linear // Self.tile_k
                var k_idx = linear % Self.tile_k
                var gmem_k = self._k_project(k_idx) + gmem_b_off
                if self.preds[v]:
                    var offset = self.weight._linear_offset(
                        Index(self.batch_idx, self.cta_n + n_idx, gmem_k)
                    )
                    async_copy[16, bypass_L1_16B=True, l2_prefetch=128](
                        gmem_base + Int(offset),
                        smem_tile_ptr + self.smem_offsets[v],
                    )
                else:
                    async_copy[
                        Self.VEC_BYTES,
                        bypass_L1_16B=True,
                        fill=Scalar[Self.b_type](0),
                    ](
                        gmem_base,
                        smem_tile_ptr + self.smem_offsets[v],
                        src_size=0,
                    )

            async_copy_arrive[noinc=True](self.smem_barrier[self.stage * 2])

            gmem_b_off += Self.per_warp_k
            self.stage = next_stage
            self.phase = next_phase


struct _MmaCpAsyncMmaComputer[
    out_origin: MutOrigin,
    //,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    accum_type: DType,
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    stage_cnt: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    swapAB: Bool = False,
]:
    """Consumer warp group (warps 4-7) for tensor-core MMA with 4-way split-K.
    """

    comptime COMPUTE_THREADS = 128
    comptime per_warp_k = Self.tile_k // 4
    comptime k_phases = Self.per_warp_k // 16
    comptime SmemTilesA = SMemTileArray2D[
        Self.a_type, Self.tile_m, Self.tile_k, Self.stage_cnt
    ]
    comptime SmemTilesB = SMemTileArray2D[
        Self.b_type, Self.tile_n, Self.tile_k, Self.stage_cnt
    ]
    comptime Barriers = SMemArray[SharedMemBarrier, Self.stage_cnt * 2]
    comptime swizzle_a = make_swizzle[8, Self.tile_k, 8]()
    comptime swizzle_b = make_swizzle[8, Self.tile_k, 8]()

    var smem_a: Self.SmemTilesA
    var smem_b: Self.SmemTilesB
    var smem_barrier: Self.Barriers
    var out_ptr: UnsafePointer[Scalar[Self.c_type], Self.out_origin]
    var compute_warp: Int
    var lane_idx: Int
    var warp_k_off: Int
    var cta_m: Int
    var cta_n: Int
    var gemm_m: Int
    var gemm_n: Int
    var stage: Int
    var phase: UInt32
    var acc: SIMD[Self.accum_type, 4]

    def __init__(
        out self,
        smem_a: Self.SmemTilesA,
        smem_b: Self.SmemTilesB,
        smem_barrier: Self.Barriers,
        out_ptr: UnsafePointer[Scalar[Self.c_type], Self.out_origin],
        compute_warp: Int,
        lane_idx: Int,
        warp_k_off: Int,
        cta_m: Int,
        cta_n: Int,
        gemm_m: Int,
        gemm_n: Int,
    ):
        self.smem_a = smem_a
        self.smem_b = smem_b
        self.smem_barrier = smem_barrier
        self.out_ptr = out_ptr
        self.compute_warp = compute_warp
        self.lane_idx = lane_idx
        self.warp_k_off = warp_k_off
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.gemm_m = gemm_m
        self.gemm_n = gemm_n
        self.stage = 0
        self.phase = UInt32(0)
        self.acc = SIMD[Self.accum_type, 4](0)

    def issue_mainloop(mut self, k_iters: Int):
        for loop_idx in range(k_iters):
            self.smem_barrier[self.stage * 2][].wait(self.phase)

            var a_tile_ptr = self.smem_a[self.stage].ptr
            var b_tile_ptr = self.smem_b[self.stage].ptr
            comptime for phase in range(Self.k_phases):
                var k_in_tile = self.warp_k_off + phase * 16

                # ld_matrix for A (16x16 bf16 block).
                var a_m_idx = self.lane_idx % 16
                var a_k_base = k_in_tile + (self.lane_idx // 16) * 8
                var a_reg = ld_matrix[8](
                    a_tile_ptr
                    + Self.swizzle_a(a_m_idx * Self.tile_k + a_k_base)
                )

                # ld_matrix for B (8x16 bf16 block).
                var b_n_idx = self.lane_idx % 8
                var b_k_base = k_in_tile + (self.lane_idx // 8) * 8
                if b_k_base >= Self.tile_k:
                    b_k_base = k_in_tile
                var b_reg_full = ld_matrix[8](
                    b_tile_ptr
                    + Self.swizzle_b(b_n_idx * Self.tile_k + b_k_base)
                )
                var b_reg = SIMD[Self.b_type, 4](
                    b_reg_full[0], b_reg_full[1], b_reg_full[2], b_reg_full[3]
                )

                mma(self.acc, a_reg, b_reg, self.acc)

            _ = self.smem_barrier[self.stage * 2 + 1][].arrive()

            var raw_next = self.stage + 1
            self.phase = self.phase ^ (
                UInt32(1) if raw_next == Self.stage_cnt else UInt32(0)
            )
            self.stage = 0 if raw_next == Self.stage_cnt else raw_next

    def epi(mut self):
        """Epilogue: reduce acc across 4 compute-warp partials, write the C tile.

        The output buffer is always row-major `[M, N]`. When `swapAB`, the
        launcher fed the kernel the transposed problem (A=weight, B=act, with
        `gemm_m`/`gemm_n` = N/M), so this CTA's tile covers N in the m-direction
        and M in the n-direction; the store transposes index order back into the
        row-major `[M, N]` buffer (row stride = N = `gemm_m`).
        """
        var smem_epi = self.smem_a.ptr.bitcast[Scalar[Self.accum_type]]()
        var base_off = self.compute_warp * Self.tile_m * Self.tile_n
        var m0 = self.lane_idx // 4
        var n0 = (self.lane_idx % 4) * 2

        smem_epi[base_off + m0 * Self.tile_n + n0] = self.acc[0]
        smem_epi[base_off + m0 * Self.tile_n + n0 + 1] = self.acc[1]
        smem_epi[base_off + (m0 + 8) * Self.tile_n + n0] = self.acc[2]
        smem_epi[base_off + (m0 + 8) * Self.tile_n + n0 + 1] = self.acc[3]

        named_barrier[Int32(Self.COMPUTE_THREADS)](Int32(1))

        if self.compute_warp == 0:
            comptime out_per_thread = (Self.tile_m * Self.tile_n + 31) // 32

            comptime for r in range(out_per_thread):
                var lin = r * 32 + self.lane_idx
                var m_idx = lin % Self.tile_m
                var n_idx = lin // Self.tile_m

                var total = Scalar[Self.accum_type](0)
                comptime for w in range(4):
                    total += smem_epi[
                        w * Self.tile_m * Self.tile_n
                        + m_idx * Self.tile_n
                        + n_idx
                    ]

                if (
                    self.cta_m + m_idx < self.gemm_m
                    and self.cta_n + n_idx < self.gemm_n
                ):
                    # True output coordinate in the row-major [M, N] buffer.
                    # Non-swap: (row=M, col=N) = (cta_m+m, cta_n+n), stride gemm_n.
                    # Swap:     (row=M, col=N) = (cta_n+n, cta_m+m), stride gemm_m
                    #           (the kernel's gemm_m == true N under swapAB).
                    var out_row = (self.cta_n + n_idx) if Self.swapAB else (
                        self.cta_m + m_idx
                    )
                    var out_col = (self.cta_m + m_idx) if Self.swapAB else (
                        self.cta_n + n_idx
                    )
                    var out_stride = self.gemm_m if Self.swapAB else self.gemm_n
                    var out_off = out_row * out_stride + out_col

                    comptime if Self.elementwise_lambda_fn:
                        comptime elementwise_lambda = Self.elementwise_lambda_fn.value()
                        elementwise_lambda[Self.c_type, 1](
                            Index(out_row, out_col),
                            total.cast[Self.c_type](),
                        )
                    else:
                        self.out_ptr[out_off] = total.cast[Self.c_type]()


struct _MmaCpAsyncSmem[
    a_type: DType,
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    stage_cnt: Int,
]:
    comptime SmemA = SMemTileArray2D[
        Self.a_type, Self.tile_m, Self.tile_k, Self.stage_cnt
    ]
    comptime SmemB = SMemTileArray2D[
        Self.a_type, Self.tile_n, Self.tile_k, Self.stage_cnt
    ]
    comptime Barriers = SMemArray[SharedMemBarrier, Self.stage_cnt * 2]

    var a_engine: Array[Scalar[Self.a_type], Self.SmemA.num_elements]
    var b_engine: Array[Scalar[Self.a_type], Self.SmemB.num_elements]
    var barrier_storage: Self.Barriers.Storage

    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.SmemA:
        return Self.SmemA(self.a_engine.unsafe_ptr())

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.SmemB:
        return Self.SmemB(self.b_engine.unsafe_ptr())

    @always_inline
    def barriers(ref[AddressSpace.SHARED] self) -> Self.Barriers:
        return Self.Barriers(self.barrier_storage)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
@__name(
    t"gemm_mma_cpasync_{c_type}_{a_type}_{b_type}_{tile_k}_{stage_cnt}",
)
def gemm_mma_cpasync_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    *,
    tile_m: Int = 16,
    tile_n: Int = 8,
    tile_k: Int = 128,
    stage_cnt: Int = 2,
    accum_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    swapAB: Bool = False,
](
    output: TileTensor[c_type, c_layout, MutAnyOrigin],
    act: TileTensor[a_type, a_layout, ImmutAnyOrigin],
    weight: TileTensor[b_type, b_layout, ImmutAnyOrigin],
    gemm_m: Int32,
    gemm_k: Int32,
    gemm_n: Int32,
    batch_size: Int32,
):
    var _gemm_m = Int(gemm_m)
    var _gemm_k = Int(gemm_k)
    var _gemm_n = Int(gemm_n)
    var _batch_size = Int(batch_size)
    comptime assert _is_sm_100x(), "gemm_mma_cpasync requires B200 (sm_100x)"
    comptime assert tile_m == 16, "tile_m must be 16 for m16n8k16 MMA"
    comptime assert tile_n == 8, "tile_n must be 8 for m16n8k16 MMA"
    comptime assert tile_k % 64 == 0, "tile_k must be a multiple of 64"
    comptime assert stage_cnt >= 1, "stage_cnt must be at least 1"

    comptime LOAD_THREADS = 64
    comptime COMPUTE_THREADS = 128
    comptime VEC_ELEMS = simd_width_of[a_type, target=get_gpu_target()]()
    comptime VEC_BYTES = VEC_ELEMS * size_of[a_type]()

    var tid = thread_idx.x
    var warp_idx_ = warp_id()
    var lane_idx_ = lane_id()

    # CTA tile origin.
    var cta_m = tile_m * Int(block_idx.x)
    var cta_n = tile_n * Int(block_idx.y)
    var batch_idx = Int(block_idx.z)

    var out_ptr = output.ptr + batch_idx * _gemm_m * _gemm_n

    # K-loop parameters.
    var k_iters = _gemm_k // tile_k
    comptime mma_warp_cnt = 4
    comptime per_warp_k = tile_k // mma_warp_cnt
    var k_each_chunk = _gemm_k // mma_warp_cnt

    var a_local_tid = tid
    var b_local_tid = tid - LOAD_THREADS
    var compute_warp = warp_idx_ - 4

    comptime SmemType = _MmaCpAsyncSmem[
        a_type, tile_m, tile_n, tile_k, stage_cnt
    ]
    ref smem = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=128,
    ]().bitcast[SmemType]()[]
    var smem_a = smem.a_tiles()
    var smem_b = smem.b_tiles()
    var smem_barrier = smem.barriers()

    if warp_idx_ == 4:
        for stage in range(stage_cnt):
            smem_barrier[stage * 2 + 0][].init(Int32(LOAD_THREADS * 2))
            smem_barrier[stage * 2 + 1][].init(Int32(COMPUTE_THREADS))
    barrier()

    # PDL selective gating: only the activation loader waits on the producer so
    # the weight load streams during it. The launcher
    # routes the activation to the A loader normally, or to the B loader under
    # `swapAB` (where the weight becomes the free-streaming A operand).
    if warp_idx_ < 2:
        comptime if pdl_level > PDLLevel.OFF and not swapAB:
            wait_on_dependent_grids()
        var loader = _MmaCpAsyncGmemLoaderA[
            a_type, type_of(act).LayoutType, tile_m, tile_k, stage_cnt
        ](
            act,
            smem_a,
            smem_barrier,
            Int(a_local_tid),
            batch_idx,
            cta_m,
            _gemm_m,
            k_each_chunk,
        )
        loader.prepare()
        loader.issue_mainloop(k_iters)

    elif warp_idx_ < 4:
        comptime if pdl_level > PDLLevel.OFF and swapAB:
            wait_on_dependent_grids()
        comptime LoaderB = _MmaCpAsyncGmemLoaderB[
            weight_origin=weight.origin,
            a_type,
            type_of(weight).LayoutType,
            tile_n,
            tile_k,
            stage_cnt,
        ]
        var loader = LoaderB(
            rebind[LoaderB.WeightTensor](weight),
            smem_b,
            smem_barrier,
            Int(b_local_tid),
            batch_idx,
            cta_n,
            _gemm_n,
            k_each_chunk,
        )
        loader.prepare()
        loader.issue_mainloop(k_iters)

    else:
        var computer = _MmaCpAsyncMmaComputer[
            c_type,
            a_type,
            a_type,
            accum_type,
            tile_m,
            tile_n,
            tile_k,
            stage_cnt,
            elementwise_lambda_fn=elementwise_lambda_fn,
            swapAB=swapAB,
        ](
            smem_a,
            smem_b,
            smem_barrier,
            out_ptr,
            Int(compute_warp),
            Int(lane_idx_),
            Int(compute_warp) * per_warp_k,
            cta_m,
            cta_n,
            _gemm_m,
            _gemm_n,
        )
        computer.issue_mainloop(k_iters)
        computer.epi()

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


def gemm_mma_cpasync[
    pdl_level: PDLLevel = PDLLevel(),
    tile_k: Int = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    swapAB: Bool = False,
](
    c: TileTensor[mut=True, ...],
    act: TileTensor[mut=False, ...],
    weight: TileTensor[mut=False, ...],
    gemm_m: Int,
    gemm_k: Int,
    gemm_n: Int,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    """Launch the batched GEMM tensor-core kernel.

    C[gemm_m, gemm_n] = act[gemm_m, K] x weight[gemm_n, K]^T.

    The caller always passes `act`/`weight`/`c` with the same shapes regardless
    of `swapAB`, and `c` is always written as a row-major `[gemm_m, gemm_n]`
    (`[M, N]`) buffer. `swapAB` only changes the internal tiling: when True the
    weight is fed to the A (free-streaming) operand slot and the activation to
    the B slot (so the grid tiles N by `tile_m` and M by `tile_n`), and the
    epilogue transposes the store back into the row-major `[M, N]` buffer. This
    makes the large weight the producer-independent operand for PDL overlap at
    small M (decode), while keeping the output layout identical.

    Parameters:
        pdl_level: Programmatic dependent launch level for PDL barriers.
        tile_k: K-dimension tile size for the MMA kernel (defaults to 128).
        elementwise_lambda_fn: Optional epilogue applied to each output
            element.
        swapAB: When True, feeds the weight to the A operand slot and the
            activation to the B slot for PDL overlap at small M.

    Args:
        c:          Output, shape (gemm_m, gemm_n) or (batch, gemm_m, gemm_n),
                    always row-major.
        act:        Activation, shape (gemm_m, gemm_k) or (batch, gemm_m, gemm_k).
        weight:     Weight, shape (gemm_n, gemm_k) or (batch, gemm_n, gemm_k).
        gemm_m:     Activation rows (output rows, M).
        gemm_k:     Reduction dimension.
        gemm_n:     Weight rows (output cols, N).
        batch_size: Batch size; ignored for 2D inputs (treated as 1).
        ctx:        GPU device context.
    """
    comptime assert (
        act.rank in (2, 3) and act.rank == weight.rank == c.rank
    ), "act, weight, and c must have the same rank and be 2D or 3D"

    comptime is_batched = act.rank == 3

    comptime c_type = c.dtype
    comptime a_type = act.dtype
    comptime b_type = weight.dtype

    comptime assert a_type == b_type, "a_type and b_type must be the same"
    comptime assert a_type == .bfloat16, "a_type/b_type must be bfloat16"
    # Output may be bfloat16 (production) or float32 (accuracy verification): the
    # kernel always accumulates in f32 and only casts to c_type on store, so an
    # f32 output simply skips the final bf16 rounding.
    comptime assert c_type in (
        DType.bfloat16,
        DType.float32,
    ), "c_type must be bfloat16 or float32"
    comptime assert (
        ctx.default_device_info.compute == B200.compute
    ), "This kernel is only supported on SM100"

    comptime tile_m = 16
    comptime tile_n = 8
    comptime TOTAL_THREADS = 256

    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
    comptime per_stage = (
        (tile_m + tile_n) * tile_k * size_of[a_type]()
        + 2 * size_of[SharedMemBarrier]()
    )
    comptime stage_cnt = b200_smem // per_stage
    comptime SmemType = _MmaCpAsyncSmem[
        a_type, tile_m, tile_n, tile_k, stage_cnt
    ]
    comptime smem_size = size_of[SmemType]()

    logger.info("------ Dispatching gemm_mma_cpasync ------")
    logger.info(
        "batch=",
        Int32(batch_size),
        " gemm_m=",
        gemm_m,
        " gemm_k=",
        Int32(gemm_k),
        " gemm_n=",
        gemm_n,
        " stage_cnt=",
        stage_cnt,
    )

    # Kernel-facing dims: under swapAB the A operand is the weight (rows = N) and
    # the B operand is the activation (rows = M), so the kernel sees gemm_m=N,
    # gemm_n=M. The output buffer `c` stays row-major [M, N]; the epilogue
    # transposes the store. Grid tiles the A-operand rows by tile_m, B by tile_n.
    var k_gemm_m = gemm_n if swapAB else gemm_m
    var k_gemm_n = gemm_m if swapAB else gemm_n
    var grid_x = ceildiv(k_gemm_m, tile_m)
    var grid_y = ceildiv(k_gemm_n, tile_n)

    comptime if is_batched:
        comptime if swapAB:
            comptime kernel = gemm_mma_cpasync_kernel[
                c_type,
                a_type,
                b_type,
                type_of(c).LayoutType,
                type_of(weight).LayoutType,
                type_of(act).LayoutType,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                stage_cnt=stage_cnt,
                elementwise_lambda_fn=elementwise_lambda_fn,
                pdl_level=pdl_level,
                swapAB=True,
            ]
            ctx.enqueue_function[kernel, dump_asm=False](
                c,
                weight,
                act,
                Int32(k_gemm_m),
                Int32(gemm_k),
                Int32(k_gemm_n),
                Int32(batch_size),
                grid_dim=(grid_x, grid_y, batch_size),
                block_dim=TOTAL_THREADS,
                shared_mem_bytes=smem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(b200_smem)
                ),
                attributes=pdl_launch_attributes(pdl_level),
            )
        else:
            comptime kernel = gemm_mma_cpasync_kernel[
                c_type,
                a_type,
                b_type,
                type_of(c).LayoutType,
                type_of(act).LayoutType,
                type_of(weight).LayoutType,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                stage_cnt=stage_cnt,
                elementwise_lambda_fn=elementwise_lambda_fn,
                pdl_level=pdl_level,
                swapAB=False,
            ]
            ctx.enqueue_function[kernel, dump_asm=False](
                c,
                act,
                weight,
                Int32(k_gemm_m),
                Int32(gemm_k),
                Int32(k_gemm_n),
                Int32(batch_size),
                grid_dim=(grid_x, grid_y, batch_size),
                block_dim=TOTAL_THREADS,
                shared_mem_bytes=smem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(b200_smem)
                ),
                attributes=pdl_launch_attributes(pdl_level),
            )
    else:
        var c3d = _to_batched_3d(c)
        var a3d = _to_batched_3d(act)
        var w3d = _to_batched_3d(weight)
        comptime if swapAB:
            comptime kernel = gemm_mma_cpasync_kernel[
                c_type,
                a_type,
                b_type,
                type_of(c3d).LayoutType,
                type_of(w3d).LayoutType,
                type_of(a3d).LayoutType,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                stage_cnt=stage_cnt,
                elementwise_lambda_fn=elementwise_lambda_fn,
                pdl_level=pdl_level,
                swapAB=True,
            ]
            ctx.enqueue_function[kernel, dump_asm=False](
                c3d,
                w3d,
                a3d,
                Int32(k_gemm_m),
                Int32(gemm_k),
                Int32(k_gemm_n),
                Int32(1),
                grid_dim=(grid_x, grid_y, 1),
                block_dim=TOTAL_THREADS,
                shared_mem_bytes=smem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(b200_smem)
                ),
                attributes=pdl_launch_attributes(pdl_level),
            )
        else:
            comptime kernel = gemm_mma_cpasync_kernel[
                c_type,
                a_type,
                b_type,
                type_of(c3d).LayoutType,
                type_of(a3d).LayoutType,
                type_of(w3d).LayoutType,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                stage_cnt=stage_cnt,
                elementwise_lambda_fn=elementwise_lambda_fn,
                pdl_level=pdl_level,
                swapAB=False,
            ]
            ctx.enqueue_function[kernel, dump_asm=False](
                c3d,
                a3d,
                w3d,
                Int32(k_gemm_m),
                Int32(gemm_k),
                Int32(k_gemm_n),
                Int32(1),
                grid_dim=(grid_x, grid_y, 1),
                block_dim=TOTAL_THREADS,
                shared_mem_bytes=smem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(b200_smem)
                ),
                attributes=pdl_launch_attributes(pdl_level),
            )
