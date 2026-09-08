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

"""Fused matvec (M=1) + partial RMS norm on B200.

Given `x [1, K]`, `W [N, K]` (with `transpose_b=True`), `gamma
[N_normed]`, and `eps`, this module computes:

    y        = x @ W.T                             # [1, N]
    normed   = rms_norm(y[:, :N_normed], gamma, eps)
    unnormed = y[:, N_normed:]

The fused kernel does this in a single launch. Every normed block
does a single device-scope acq_rel `fetch_add` on `finish_counter`:
the release half orders each block's prior `normed_output` writes
before the counter increment, and the acquire half in the
global-last arriver (`prev == num_normed_blocks - 1`) makes every
peer's writes visible so it can read `normed_output` back and do a
single-pass intra-block RMS reduction + apply with gamma.

The unfused path is a 2-launch baseline using existing primitives
(matmul followed by `rms_norm_gpu`). The matmul writes the full
`[M, N]` output to a caller-provided `y_scratch` buffer; the RMS
norm then reads `y_scratch[:, :N_normed]` and writes the normed
values to `normed_output`. The unnormed tail lives as a view into
`y_scratch[:, N_normed:]`, matching how model code naturally
expresses this.
"""

from std.math import ceildiv, rsqrt
from std.builtin.device_passable import DevicePassable
from std.memory import AddressSpace
from std.atomic import Atomic, Ordering, fence
from std.sys.info import _is_sm_100x_or_newer, simd_width_of, size_of
from std.time import global_perf_counter_ns

import max.gpu.primitives.block as block
import max.gpu.primitives.warp as warp
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.memory import cp_async_bulk_prefetch
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)

from layout import (
    Coord,
    Idx,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from std.utils import IndexList
from std.utils.index import Index
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from linalg.matmul.gpu import _matmul_gpu
from shmem.ep_comm import DEVICE_SCOPE
from structured_kernels.trace_buf import GmemTrace, NullTrace, TraceBuf

from .normalization import rms_norm_gpu


# ===----------------------------------------------------------------------=== #
# Trace instrumentation: zero-overhead pipeline event recording
# ===----------------------------------------------------------------------=== #
#
# Records per-block timestamps (`global_perf_counter_ns` → PTX
# `globaltimer`, SM-synchronized) at key phase boundaries. Uses the
# shared `TraceBuf` trait + `NullTrace` / `GmemTrace` impls from
# `structured_kernels/trace_buf.mojo` so the no-trace path has ZERO
# extra kernel arguments and the record() calls fully inline away:
#
#   comptime if enable_trace:  -> strips the store body at compile time
#   TraceBufT: TraceBuf        -> NullTrace (zero-sized) when disabled
#
# Event roles per block:
#   0 = T_ENTER          (kernel entry, tid==0)
#   1 = T_K_START        (after PDL wait, before K-loop)
#   2 = T_K_END          (after cross-warp reduce; shmem populated)
#   3 = T_ATOMIC_IN      (tid==0, after output write, before fetch_add)
#   4 = T_ATOMIC_OUT     (tid==0, after fetch_add; normed only)
#   5 = T_APPLY_START    (global-last only, at start of apply body)
#   6 = T_LOADS_DONE     (global-last, after input + gamma LDGs)
#   7 = T_REDUCE_DONE    (global-last, after block.sum)
#   8 = T_APPLY_END      (global-last, after apply-norm store)
#   9 = T_EXIT           (kernel end, tid==0)
#  10..15 = reserved
# ===----------------------------------------------------------------------=== #

comptime GEMV_TRACE_EVENTS_PER_BLOCK = 16
"""Number of `UInt64` timestamp slots reserved per block in a trace
buffer. Only 10 slots are used today (roles 0 through 9); slots 10
through 15 are reserved for future per-iteration instrumentation."""


# ===----------------------------------------------------------------------=== #
# Fused GEMV + partial RMS norm kernel (M=1)
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"gemv_partial_norm_kernel_{c_type}_{a_type}_{b_type}_{num_threads}_{tile_n}",
)
def gemv_partial_norm_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    normed_layout: TensorLayout,
    unnormed_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    gamma_layout: TensorLayout,
    normed_engine: TensorEngine,
    unnormed_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    gamma_engine: TensorEngine,
    TraceBufT: TraceBuf,
    //,
    simd_width: Int,
    tile_n: Int,
    num_threads: Int,
    enable_trace: Bool = False,
    pdl_level: PDLLevel = PDLLevel(),
](
    normed_output: TileTensor[
        c_type, normed_layout, MutAnyOrigin, Engine=normed_engine
    ],
    unnormed_output: TileTensor[
        c_type, unnormed_layout, MutAnyOrigin, Engine=unnormed_engine
    ],
    act: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    weight: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    gamma: TileTensor[
        a_type, gamma_layout, ImmutAnyOrigin, Engine=gamma_engine
    ],
    finish_counter: MutPointer[Int32, MutAnyOrigin],
    trace_buf: TraceBufT,
    eps: Float32,
    n: Int32,
    k: Int32,
    n_normed: Int32,
    num_normed_blocks: Int32,
):
    """Fused GEMV (M=1) + partial RMS norm, single launch.

    Grid layout: `(1, ceildiv(n, tile_n))`. Each block computes one
    `tile_n`-wide column tile of `y = act @ weight.T`, writes to
    either `normed_output` or `unnormed_output`, and every normed
    block does a single `fetch_add` on `finish_counter`. The
    global-last arriver reads `normed_output` back, does a single-
    pass intra-block RMS reduction, applies gamma in place, and
    resets the counter.

    Parameters:
        c_type: Output dtype (of `normed_output` and `unnormed_output`).
        a_type: Activation / gamma dtype.
        b_type: Weight dtype.
        normed_layout: Layout of `normed_output`.
        unnormed_layout: Layout of `unnormed_output`.
        a_layout: Layout of `act`.
        b_layout: Layout of `weight`.
        gamma_layout: Layout of `gamma`.
        normed_engine: Engine policy of `normed_output`.
        unnormed_engine: Engine policy of `unnormed_output`.
        a_engine: Engine policy of `act`.
        b_engine: Engine policy of `weight`.
        gamma_engine: Engine policy of `gamma`.
        TraceBufT: Trace-buffer implementation (`NullTrace` or
            `GmemTrace`). Pass `NullTrace` for zero-overhead untraced
            runs.
        simd_width: Vectorization width for K-loop loads and apply-norm.
        tile_n: Columns of `y` computed per block.
        num_threads: Block size (threads per block).
        enable_trace: When `True`, record per-phase timestamps into
            `trace_buf`. When `False` (default), all record sites
            compile away to zero PTX.
        pdl_level: Programmatic Dependent Launch level for chaining
            with upstream/downstream kernels.

    Args:
        normed_output: `[M, N_normed]` output buffer. The global-last
            arriver rewrites this in place after the RMS reduction
            has been applied.
        unnormed_output: `[M, N - N_normed]` output buffer. Written
            exactly once by each unnormed block.
        act: `[M, K]` activations.
        weight: `[N, K]` weights (used as `weight.T`).
        gamma: `[N_normed]` RMS norm scale.
        finish_counter: Single `Int32` counter used for the flat grid
            sync. Must be zero-initialized on first use. The kernel
            resets it to zero before returning.
        trace_buf: Instance of `TraceBufT` carrying the device-side
            trace-buffer pointer (or a zero-sized no-op).
        eps: RMS norm epsilon.
        n: Full output width `N` (normed + unnormed).
        k: Activation / weight inner dimension.
        n_normed: Length of the normed region.
        num_normed_blocks: `_n_normed / tile_n`, used by the global-
            last election (`prev == num_normed_blocks - 1`).

    Constraints:
        - `_n_normed` must be divisible by `tile_n`.
        - `_n_normed` must be divisible by `simd_width` (apply-norm
          uses vectorized loads).
    """
    var _n = Int(n)
    var _k = Int(k)
    var _n_normed = Int(n_normed)
    comptime assert normed_output.flat_rank == 2
    comptime assert unnormed_output.flat_rank == 2
    comptime assert act.flat_rank == 2
    comptime assert weight.flat_rank == 2
    comptime assert gamma.flat_rank == 1
    comptime accum_type = get_accum_type[c_type]()
    comptime tile_k = simd_width * num_threads
    var tile_id_n = block_idx.y * tile_n
    var tid = thread_idx.x

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 0,
                UInt64(global_perf_counter_ns()),
            )

    var tile_w = tt_stack_allocation[
        dtype=b_type,
        address_space=.LOCAL,
        alignment=simd_width * size_of[b_type](),
    ](row_major[tile_n, simd_width]())
    var acc = tt_stack_allocation[dtype=accum_type, address_space=.LOCAL](
        row_major[1, tile_n]()
    ).fill(0)

    comptime WeightVecType = SIMD[b_type, simd_width]
    comptime NativeVecType = SIMD[a_type, simd_width]

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 1,
                UInt64(global_perf_counter_ns()),
            )

    # K-reduction.
    var iteration = 0
    for _ in range(tid * simd_width, _k, tile_k):
        var weight_tile = weight.tile[tile_n, tile_k](block_idx.y, iteration)
        var act_tile = act.tile[1, tile_k](0, iteration)

        comptime for i in range(tile_n):
            var vec_weight_tile = weight_tile.vectorize[1, simd_width]()
            var b_vec = vec_weight_tile[i, thread_idx.x]
            tile_w.store(Coord(i, Idx[0]), b_vec)

        var act_vec = act_tile.vectorize[1, simd_width]()[0, thread_idx.x]
        var act_native = act_vec
        comptime for j in range(tile_n):
            var weight_native = rebind[NativeVecType](
                tile_w.vectorize[1, simd_width]()[j, 0]
            )
            var local_accum = acc[0, j]
            var ac = act_native.cast[accum_type]()
            var bc = weight_native.cast[accum_type]()
            comptime for l in range(simd_width):
                local_accum += ac[l] * bc[l]
            acc[0, j] = local_accum

        iteration += 1

    # Cross-warp reduce into shmem.
    comptime k_warp_num = num_threads // WARP_SIZE
    var wid = warp_id()
    var lid = lane_id()
    var shmem = tt_stack_allocation[dtype=accum_type, address_space=.SHARED](
        row_major[1, tile_n * k_warp_num]()
    )

    comptime for ni in range(tile_n):
        var val = warp.sum(acc[0, ni])
        if lid == 0:
            shmem[0, wid * tile_n + ni] = val
    barrier()

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 2,
                UInt64(global_perf_counter_ns()),
            )

    var is_normed_block = tile_id_n + tile_n <= _n_normed
    var is_last_block: Bool = False

    # Gamma prefetch to L2: any single normed block issues an async bulk
    # prefetch of the entire gamma vector to L2 right after K_END, in
    # parallel with the 1-us fetch_add wait below. By the time the
    # global-last block reaches the apply phase, gamma is L2-resident and
    # the input+gamma LDG bundle inside the apply body is pure L2-hit.
    # Block 0 is always in the first wave (SM scheduling is in grid
    # order), so this fires early enough to amortize the HBM round-trip.
    # cp_async_bulk_prefetch lowers to a 1D TMA load, which the stdlib
    # constrains to SM100+ only. On H100 we elide the prefetch at
    # comptime and fall through to the regular LDG of gamma inside the
    # apply body: no L2 warm-up, but no correctness impact either.
    comptime if _is_sm_100x_or_newer():
        # Gate to a single warp of a single block to avoid 384x
        # redundant prefetches (L2 coalesces but the issues are wasted).
        if block_idx.y == 0 and tid == 0:
            cp_async_bulk_prefetch(
                gamma.ptr, Int32(_n_normed * size_of[a_type]())
            )

    if tid == 0:
        var vals = SIMD[accum_type, tile_n](0)
        comptime for jj in range(k_warp_num):
            comptime for ni in range(tile_n):
                vals[ni] += shmem[0, jj * tile_n + ni]

        if is_normed_block:
            comptime for ni in range(tile_n):
                var col = tile_id_n + ni
                normed_output[0, col] = vals[ni].cast[c_type]()

            comptime if enable_trace:
                trace_buf.store(
                    Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 3,
                    UInt64(global_perf_counter_ns()),
                )

            # Flat grid sync: single device-scope acq_rel fetch_add.
            # The release half orders the normed_output stores above
            # before the counter increment; the acquire half in the
            # global-last arriver's fetch_add makes every peer's
            # writes visible when it reads normed_output back below.
            var prev_global = Atomic[Int32, scope=DEVICE_SCOPE].fetch_add[
                ordering=Ordering.ACQUIRE_RELEASE
            ](finish_counter, Int32(1))
            comptime if enable_trace:
                trace_buf.store(
                    Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 4,
                    UInt64(global_perf_counter_ns()),
                )
            is_last_block = prev_global == num_normed_blocks - Int32(1)
        else:
            comptime for ni in range(tile_n):
                var col = tile_id_n + ni
                if col < _n:
                    unnormed_output[0, col - _n_normed] = vals[ni].cast[
                        c_type
                    ]()

    # Post-atomic broadcast of `is_last_block` from tid=0 to the whole
    # block via shmem + barrier. Only tid=0 saw the fetch_add result.
    # If this block is the global-last arriver, tid=0 issues a device-
    # scope acquire fence so the subsequent apply body's reads of
    # `normed_output` see every peer CTA's releasing writes. The
    # `barrier()` below extends tid=0's acquire to all threads via
    # happens-before transitivity: tid=0's fence synchronizes-with peer
    # writes, and barrier synchronizes all threads, so each thread's
    # later `normed_output` load happens-after peer writes.
    if tid == 0:
        if is_last_block:
            fence[ordering=Ordering.ACQUIRE, scope=DEVICE_SCOPE]()
        shmem[0, 0] = Scalar[accum_type](1) if is_last_block else Scalar[
            accum_type
        ](0)

    if is_normed_block:
        barrier()
        var flag = shmem[0, 0]
        if flag != 0:
            comptime if enable_trace:
                if tid == 0:
                    trace_buf.store(
                        Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 5,
                        UInt64(global_perf_counter_ns()),
                    )

            # Single-pass intra-block RMS norm on `normed_output`
            # (pattern from `rms_norm_gpu_warp_tiling` in
            # `nn/normalization.mojo`). Keep vec_data in registers
            # across the reduce; overlap gamma LDGs with block.sum.
            comptime CVecType = SIMD[c_type, simd_width]
            comptime AVecType = SIMD[a_type, simd_width]
            var idx = Int(tid) * simd_width
            var vec_data = SIMD[accum_type, simd_width](0)
            var gamma_val = SIMD[a_type, simd_width](0)
            if idx < _n_normed:
                vec_data = normed_output.load[simd_width](
                    Coord(Idx[0], idx)
                ).cast[accum_type]()
                gamma_val = gamma.load[simd_width](Coord(idx))

            comptime if enable_trace:
                if tid == 0:
                    trace_buf.store(
                        Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 6,
                        UInt64(global_perf_counter_ns()),
                    )

            var thread_m2 = (vec_data * vec_data).reduce_add()
            var row_m2 = block.sum[block_size=num_threads](thread_m2)

            comptime if enable_trace:
                if tid == 0:
                    trace_buf.store(
                        Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 7,
                        UInt64(global_perf_counter_ns()),
                    )

            var norm_factor = rsqrt(
                row_m2 / Scalar[accum_type](_n_normed) + eps.cast[accum_type]()
            )

            if idx < _n_normed:
                var gamma_accum = gamma_val.cast[accum_type]()
                var out = (vec_data * norm_factor * gamma_accum).cast[c_type]()
                normed_output.store[simd_width](Coord(Idx[0], idx), out)

            comptime if enable_trace:
                if tid == 0:
                    trace_buf.store(
                        Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 8,
                        UInt64(global_perf_counter_ns()),
                    )

            # Reset finish_counter for the next call. Stream-ordering
            # between successive kernel launches provides visibility
            # for the next launch's readers.
            if tid == 0:
                finish_counter.unsafe_store(Int32(0))

    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                Int(block_idx.y) * GEMV_TRACE_EVENTS_PER_BLOCK + 9,
                UInt64(global_perf_counter_ns()),
            )


# ===----------------------------------------------------------------------=== #
# Host wrappers
# ===----------------------------------------------------------------------=== #


def _gemv_partial_norm_fused[
    c_type: DType,
    a_type: DType,
    TraceBufT: TraceBuf,
    //,
    transpose_b: Bool,
    pdl_level: PDLLevel,
    tile_n: Int = 4,
    num_threads: Int = 256,
    enable_trace: Bool = False,
](
    normed_output: TileTensor[mut=True, c_type, ...],
    unnormed_output: TileTensor[mut=True, c_type, ...],
    act: TileTensor[mut=False, a_type, ...],
    weight: TileTensor[mut=False, a_type, ...],
    gamma: TileTensor[mut=False, a_type, ...],
    eps: Float32,
    finish_counter: MutPointer[Int32, _],
    trace_buf: TraceBufT,
    ctx: DeviceContext,
) raises:
    """Fused single-kernel path. Requires M=1 and `transpose_b=True`.

    `finish_counter` (1 i32) must be zero-initialized on first use.
    The kernel resets it to zero before returning so the same buffer
    can be reused across calls without an external memset.
    """
    comptime assert (
        transpose_b
    ), "gemv_and_partial_norm fused path requires transpose_b=True"
    comptime assert act.rank == 2 and weight.rank == 2
    comptime assert normed_output.rank == 2 and unnormed_output.rank == 2
    comptime assert gamma.flat_rank == 1
    comptime assert normed_output.element_size == 1
    comptime assert unnormed_output.element_size == 1
    comptime assert act.element_size == 1
    comptime assert weight.element_size == 1
    comptime assert gamma.element_size == 1

    var m = Int(act.dim[0]())
    assert m == 1, (
        "gemv_and_partial_norm fused path is GEMV-only (M=1); got M="
        + String(m)
        + ". Use matmul + rms_norm_gpu for M>1."
    )

    var k = Int(act.dim[1]())
    var n_normed = Int(gamma.dim[0]())
    var n_unnormed = Int(unnormed_output.dim[1]())
    var n = n_normed + n_unnormed

    comptime simd_width = simd_width_of[a_type, target=get_gpu_target()]()
    assert (
        n_normed % tile_n == 0
    ), "n_normed must be divisible by tile_n in the fused kernel"
    assert n_normed % simd_width == 0, (
        "n_normed must be divisible by simd_width for the vectorized"
        " apply-norm loop"
    )

    var num_blocks = ceildiv(n, tile_n)
    var num_normed_blocks = n_normed // tile_n

    comptime kernel = gemv_partial_norm_kernel[
        c_type=c_type,
        a_type=a_type,
        b_type=a_type,
        normed_layout=type_of(normed_output).LayoutType,
        unnormed_layout=type_of(unnormed_output).LayoutType,
        a_layout=type_of(act).LayoutType,
        b_layout=type_of(weight).LayoutType,
        gamma_layout=type_of(gamma).LayoutType,
        normed_engine=type_of(normed_output).Engine,
        unnormed_engine=type_of(unnormed_output).Engine,
        a_engine=type_of(act).Engine,
        b_engine=type_of(weight).Engine,
        gamma_engine=type_of(gamma).Engine,
        TraceBufT=TraceBufT,
        simd_width=simd_width,
        tile_n=tile_n,
        num_threads=num_threads,
        enable_trace=enable_trace,
        pdl_level=pdl_level,
    ]
    ctx.enqueue_function[kernel](
        normed_output,
        unnormed_output,
        act,
        weight,
        gamma,
        finish_counter,
        trace_buf,
        eps.cast[.float32](),
        Int32(n),
        Int32(k),
        Int32(n_normed),
        Int32(num_normed_blocks),
        grid_dim=(1, num_blocks),
        block_dim=num_threads,
        attributes=pdl_launch_attributes(pdl_level),
    )


def _gemv_partial_norm_unfused_with_scratch[
    c_type: DType,
    a_type: DType,
    //,
    transpose_b: Bool,
    pdl_level: PDLLevel,
](
    normed_output: TileTensor[mut=True, c_type, ...],
    act: TileTensor[mut=False, a_type, ...],
    weight: TileTensor[mut=False, a_type, ...],
    gamma: TileTensor[mut=False, a_type, ...],
    eps: Float32,
    y_scratch: MutPointer[Scalar[c_type], _],
    ctx: DeviceContext,
) raises:
    """Unfused 2-launch path using caller-provided y scratch.

    Launches exactly two kernels:

    1. A matmul into `y_scratch`. After this kernel returns,
       `y_scratch[:, :M * N]` holds the FULL `y = act @ weight.T`.
    2. `rms_norm_gpu` reads `y_scratch[:, :n_normed]` and writes the
       normalized values into `normed_output` via callbacks that
       translate from `[M, N_normed]` coords into the `[M, N]`
       storage layout of `y_scratch`. After this kernel returns,
       `normed_output` holds `rms_norm(y[:, :n_normed], gamma, eps)`.

    The unnormed tail lives as `y_scratch[:, n_normed:]`. The caller
    takes a view, no slice copy.
    """
    comptime assert c_type == a_type, (
        "unfused baseline requires c_type == a_type; rms_norm_gpu takes a"
        " single dtype for both input and gamma"
    )
    comptime assert act.flat_rank == 2 and weight.flat_rank == 2
    comptime assert normed_output.flat_rank == 2

    var m = Int(act.dim[0]())
    var n_normed = Int(gamma.dim[0]())
    var n = Int(weight.dim[0]())

    var y_layout = row_major(Coord(m, n))
    var y = TileTensor[c_type, type_of(y_layout)](y_scratch, y_layout)

    _matmul_gpu[transpose_b=transpose_b](y, act, weight, ctx)

    @always_inline
    @__copy_capture(y)
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[c_type, width]:
        var idx = y.layout(coords)
        return y.ptr.unsafe_load[width=width](idx)

    @always_inline
    @__copy_capture(normed_output)
    @__parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[c_type, width]) -> None:
        var idx = normed_output.layout(coords)
        normed_output.ptr.unsafe_store[width=width, alignment=alignment](
            idx, val
        )

    var gamma_c = rebind[TileTensor[c_type, gamma.LayoutType, gamma.origin]](
        gamma
    )
    # `rms_norm_gpu` migrated to a `Coord` shape boundary (softmax PR #88203);
    # `[M, N_normed]` is built at runtime so an all-dynamic `Coord` is passed.
    rms_norm_gpu[
        2, input_fn, output_fn, multiply_before_cast=True, pdl_level=pdl_level
    ](
        Coord(m, n_normed),
        gamma_c,
        eps,
        Scalar[c_type](0.0),
        ctx,
    )


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def gemv_and_partial_norm[
    c_type: DType,
    a_type: DType,
    //,
    *,
    transpose_b: Bool = True,
    fused: Bool = True,
    tile_n: Int = 4,
    num_threads: Int = 256,
    pdl_level: PDLLevel = PDLLevel(),
](
    normed_output: TileTensor[mut=True, c_type, ...],
    unnormed_output: TileTensor[mut=True, c_type, ...],
    act: TileTensor[mut=False, a_type, ...],
    weight: TileTensor[mut=False, a_type, ...],
    gamma: TileTensor[mut=False, a_type, ...],
    eps: Float32,
    ctx: DeviceContext,
) raises:
    """Computes `y = act @ weight.T`, then partitions `y` into a normed
    front and an unnormed tail.

    Parameters:
        c_type: Output dtype.
        a_type: Activation / weight / gamma dtype.
        transpose_b: If `True`, `weight` is row-major `[N, K]` used as
            `weight.T`.
        fused: Compile-time flag. `True` (default) selects the single-
            kernel fused path (M=1 only). `False` selects the
            2-launch baseline (matmul + `rms_norm_gpu`; the unnormed
            tail is a view into the matmul output, so
            `unnormed_output` is left untouched).
        tile_n: Comptime tile width in columns (fused only).
        num_threads: Comptime threads per block (fused only).
        pdl_level: Programmatic Dependent Launch level.

    Args:
        normed_output: `[M, N_normed]` output buffer. Receives
            `rms_norm(y[:, :N_normed], gamma, eps)` in both paths.
        unnormed_output: `[M, N - N_normed]` output buffer. The fused
            path writes `y[:, N_normed:]` here; the unfused path
            leaves this untouched (the unnormed tail is a view into
            the internally-allocated matmul scratch).
        act: `[M, K]` activations.
        weight: `[N, K]` weights (when `transpose_b=True`).
        gamma: `[N_normed]` RMS norm scale.
        eps: RMS norm epsilon.
        ctx: Device context.

    Raises:
        Error: If `_matmul_gpu` or `rms_norm_gpu` fail to launch, or
            if internal scratch allocation fails.
    """
    var m = Int(act.dim[0]())
    var n_normed = Int(gamma.dim[0]())
    var n_unnormed = Int(unnormed_output.dim[1]())
    var n = n_normed + n_unnormed

    comptime if fused:
        var counter_buf = ctx.enqueue_create_buffer[.int32](1)
        ctx.enqueue_memset(counter_buf, Int32(0))
        _gemv_partial_norm_fused[
            transpose_b=transpose_b,
            pdl_level=pdl_level,
            tile_n=tile_n,
            num_threads=num_threads,
        ](
            normed_output,
            unnormed_output,
            act,
            weight,
            gamma,
            eps,
            counter_buf.unsafe_ptr(),
            NullTrace(),
            ctx,
        )
        _ = counter_buf^
    else:
        var y_buf = ctx.enqueue_create_buffer[c_type](m * n)
        _gemv_partial_norm_unfused_with_scratch[
            transpose_b=transpose_b,
            pdl_level=pdl_level,
        ](
            normed_output,
            act,
            weight,
            gamma,
            eps,
            y_buf.unsafe_ptr(),
            ctx,
        )
        _ = y_buf^


def gemv_and_partial_norm_unfused_with_scratch[
    c_type: DType,
    a_type: DType,
    //,
    *,
    transpose_b: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    normed_output: TileTensor[mut=True, c_type, ...],
    act: TileTensor[mut=False, a_type, ...],
    weight: TileTensor[mut=False, a_type, ...],
    gamma: TileTensor[mut=False, a_type, ...],
    eps: Float32,
    y_scratch: MutPointer[Scalar[c_type], MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Unfused 2-launch path with caller-provided y scratch.

    Launches exactly two kernels:

    1. A matmul into `y_scratch`. After the matmul,
       `y_scratch[:, :M * N]` holds the FULL output `y = act @ weight.T`.
    2. `rms_norm_gpu` reads `y_scratch[:, :N_normed]` and writes the
       normalized values into `normed_output`. After this call,
       `normed_output` holds `rms_norm(y[:, :N_normed], gamma, eps)`.

    The unnormed tail lives as `y_scratch[:, N_normed:]`; the caller
    takes a view, no slice copy.

    Parameters:
        c_type: Output dtype (of `y_scratch` and `normed_output`).
        a_type: Activation / weight / gamma dtype. Must equal
            `c_type`; `rms_norm_gpu` uses a single dtype for input
            and gamma.
        transpose_b: If `True`, `weight` is row-major `[N, K]` used
            as `weight.T`.
        pdl_level: Programmatic Dependent Launch level, forwarded to
            `rms_norm_gpu` to chain with downstream kernels.

    Args:
        normed_output: `[M, N_normed]` output buffer. Holds
            `rms_norm(y[:, :N_normed], gamma, eps)` on return.
        act: `[M, K]` activations.
        weight: `[N, K]` weights (when `transpose_b=True`).
        gamma: `[N_normed]` RMS norm scale.
        eps: RMS norm epsilon.
        y_scratch: Caller-provided `[M, N]` device buffer (at least
            `M * N` elements of `c_type`). Holds the full matmul
            output `y = act @ weight.T` on return.
        ctx: Device context.

    Raises:
        Error: If either kernel launch fails.
    """
    _gemv_partial_norm_unfused_with_scratch[
        transpose_b=transpose_b, pdl_level=pdl_level
    ](
        normed_output,
        act,
        weight,
        gamma,
        eps,
        y_scratch,
        ctx,
    )


def gemv_and_partial_norm_with_scratch[
    c_type: DType,
    a_type: DType,
    TraceBufT: TraceBuf = NullTrace,
    //,
    *,
    transpose_b: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    tile_n: Int = 4,
    num_threads: Int = 256,
    enable_trace: Bool = False,
](
    normed_output: TileTensor[mut=True, c_type, ...],
    unnormed_output: TileTensor[mut=True, c_type, ...],
    act: TileTensor[mut=False, a_type, ...],
    weight: TileTensor[mut=False, a_type, ...],
    gamma: TileTensor[mut=False, a_type, ...],
    eps: Float32,
    finish_counter: MutPointer[Int32, MutAnyOrigin],
    ctx: DeviceContext,
    trace_buf: TraceBufT = NullTrace(),
) raises:
    """Fused path with caller-provided scratch.

    `finish_counter` must be zero-initialized on first use. The kernel
    resets it to zero before returning so the same buffer can be
    reused across calls without an external memset.

    Set `enable_trace=True` and pass a `GmemTrace(ptr)` to record per-
    block timestamps into `ptr` (sized `num_normed_blocks *
    GEMV_TRACE_EVENTS_PER_BLOCK` u64s). When disabled (default), the
    trace path dead-code-eliminates, yielding byte-identical PTX to
    the untraced kernel.

    Parameters:
        c_type: Output dtype.
        a_type: Activation / weight / gamma dtype.
        TraceBufT: Trace-buffer implementation. Defaults to `NullTrace`
            for zero-overhead untraced runs.
        transpose_b: If `True`, `weight` is row-major `[N, K]` used
            as `weight.T`.
        pdl_level: Programmatic Dependent Launch level.
        tile_n: Comptime tile width in columns.
        num_threads: Comptime threads per block.
        enable_trace: When `True`, record per-phase timestamps into
            `trace_buf`. When `False` (default), all record sites
            compile away.

    Args:
        normed_output: `[M, N_normed]` output buffer. Holds
            `rms_norm(y[:, :N_normed], gamma, eps)` on return.
        unnormed_output: `[M, N - N_normed]` output buffer. Holds
            `y[:, N_normed:]` on return.
        act: `[M, K]` activations.
        weight: `[N, K]` weights (when `transpose_b=True`).
        gamma: `[N_normed]` RMS norm scale.
        eps: RMS norm epsilon.
        finish_counter: Single-`Int32` device counter. Must be
            zero-initialized on first use; the kernel resets it to
            zero before returning.
        ctx: Device context.
        trace_buf: Trace-buffer instance. Defaults to `NullTrace()`.

    Raises:
        Error: If the kernel launch fails.
    """
    _gemv_partial_norm_fused[
        transpose_b=transpose_b,
        pdl_level=pdl_level,
        tile_n=tile_n,
        num_threads=num_threads,
        enable_trace=enable_trace,
    ](
        normed_output,
        unnormed_output,
        act,
        weight,
        gamma,
        eps,
        finish_counter,
        trace_buf,
        ctx,
    )
