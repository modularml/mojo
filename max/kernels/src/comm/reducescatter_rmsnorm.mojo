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
"""Fused reduce-scatter + RMSNorm (bf16 in/out, no FP8).

Reduce-scatters a `[rows, cols]` stream across `ngpus` GPUs and RMSNorms each
owned row, keeping the sum in registers so there is no HBM round-trip between
the two. The sum is rounded to `in_dtype` before the norm so the result is
bit-for-bit the standalone `reduce-scatter -> bf16 shard -> rms_norm` path
(norming the wider f32 sum silently shifts model behavior). Emits two
`[rank_units, cols]` shards: `sum_out` (the reduce-scatter sum / residual
stream) and `normed_out` (its RMSNorm). Inherently `multiply_before_cast=True`.

Factored from the 2-stage AR kernel (`allreduce_residual_rmsnorm.mojo`); keep
the reduce/norm math in sync with it -- but NOT the barrier structure: this
kernel closes with an end barrier, which the AR+norm kernels do not (their
`is_start=False` barrier is an intermediate stage fence).
"""

from std.collections import Array
from std.math import ceildiv, rsqrt
from std.sys import (
    align_of,
    simd_width_of,
    size_of,
)

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.primitives import block
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from layout.tile_tensor import _ComptimeConditionalTileTensor
from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from .allreduce import allreduce_tuning_table
from .device_query import dispatch_select_comm_config, get_sm_version
from .reducescatter import ReduceScatterConfig, _target_address_space
from .sync import MAX_GPUS, Signal, _multi_gpu_barrier, is_p2p_enabled


# Per-rank shard bytes at/below which `_dispatch_rs_norm` fuses. Fused
# in-register `block.sum` matches `rms_norm_gpu`'s row-count-dependent reduction
# bit-for-bit only up to a crossover: measured M <= 512 at H=6144 (4xMI355, TP4,
# bf16, wo=1.0, mbc=True), diverging 1 bf16 ULP at M=1024. Value = M=512 shard
# (128 rows * 6144 * 2 B). H-specific; recalibrate before fusing another H.
comptime RS_NORM_FUSE_THRESHOLD = 128 * 6144 * size_of[DType.bfloat16]()

# Stand-in layout for a disengaged optional residual (mirrors
# `allreduce_residual_rmsnorm.mojo`): never indexed, so it carries no storage.
comptime _ZeroSizedLayout = type_of(row_major(Coord(Idx[0], Idx[0])))


# --- GPU Kernel ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
@__name(t"reducescatter_rmsnorm_{in_dtype}_{ngpus}_{has_residual}")
def _reducescatter_rmsnorm_kernel[
    mut: Bool,
    origin: Origin[mut=mut],
    GammaLayoutType: TensorLayout,
    normed_origin: MutOrigin,
    NormedLayoutType: TensorLayout,
    sum_origin: MutOrigin,
    SumLayoutType: TensorLayout,
    residual_origin: Origin,
    ResidualLayoutType: TensorLayout,
    in_dtype: DType,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
    has_residual: Bool,
    domain_id: Int = 0,
    pdl_level: PDLLevel = PDLLevel(),
](
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    gamma: TileTensor[in_dtype, GammaLayoutType, origin],
    normed_out: TileTensor[mut=True, in_dtype, NormedLayoutType, normed_origin],
    sum_out: TileTensor[mut=True, in_dtype, SumLayoutType, sum_origin],
    residual: _ComptimeConditionalTileTensor[
        in_dtype,
        ResidualLayoutType,
        residual_origin,
        engaged=has_residual,
    ],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rows_dev: Int32,
    cols_dev: Int32,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank_dev: Int32,
):
    """Reduce-scatter each owned row in f32, then RMSNorm it in registers.

    Blocks stride over the ragged partition's local rows; `normed_out`/`sum_out`
    are this rank's `[rank_units, cols]` shards, indexed by local row.

    When `has_residual`, `residual` is the FULL `[rows, cols]` pre-scatter
    tensor and each rank adds only its own shard of it, indexed by the global
    row. Callers fold a TP-replicated residual this way instead of adding it on
    the group leader before the collective: the reduce-scatter sums across
    ranks, so a leader-side add would otherwise be counted once for the whole
    group. Both spellings give the same value only because the residual is
    bit-identical on every rank of the group -- see `reducescatter_rmsnorm`.
    """
    var rows = Int(rows_dev)
    var cols = Int(cols_dev)
    var my_rank = Int(my_rank_dev)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    # 2D stores below use Coord(local_row, col_idx).
    comptime assert normed_out.flat_rank >= 2
    comptime assert sum_out.flat_rank >= 2
    # Residual load below uses Coord(global_row, col_idx). Unconditional:
    # `_ZeroSizedLayout` is (0, 0), so the disengaged case is rank 2 too.
    comptime assert residual.T.flat_rank >= 2
    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()

    var tid = thread_idx.x
    var col_idx = tid * simd_width
    var is_valid = col_idx < cols
    var num_blocks = grid_dim.x

    # Ragged partition (matches `ReduceScatterConfig`, NOT ceildiv): remainder
    # rows to low ranks -> drop-in for standalone RS, no OOB at rows%ngpus!=0.
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=rows, unit_numel=cols, threads_per_gpu=0
    )
    var my_start = config.rank_unit_start(my_rank)
    var my_count = config.rank_units(my_rank)

    # Round-robin peer order (RS's `circular_add`): peer 0 is self, so accum
    # from 0 over all peers is bit-for-bit RS's `accum = peer[0]` init (AMD
    # non-multimem).
    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var ptrs = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> PtrType: src_ptrs[
            (my_rank + i) % ngpus
        ]
    )

    # Gamma is a model weight, not predecessor output, so it can be loaded ahead
    # of the wait below (local data, latency-hidden).
    var gamma_vec = SIMD[accum_type, simd_width](0)
    if is_valid:
        gamma_vec = (
            gamma.load[width=simd_width, alignment=align](Coord(col_idx)).cast[
                accum_type
            ]()
            + weight_offset.cast[accum_type]()
        )

    # We P2P-read buffers the previous kernel writes, so its stores must land
    # before the reduction below.
    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # Start barrier: we P2P-read peers' inputs, so all ranks must be ready.
    _multi_gpu_barrier[ngpus, is_start=True, domain_id=domain_id](
        rank_sigs, rank_sigs[my_rank], my_rank
    )

    # Grid-strided over local owned rows; the ragged partition bounds them, so
    # no `row < rows` guard is needed.
    for local_row in range(block_idx.x, my_count, num_blocks):
        var row = my_start + local_row

        # P2P load from all GPUs, accumulate in f32.
        var accum = SIMD[accum_type, simd_width](0)
        if is_valid:
            var global_elem = row * cols + col_idx
            comptime for gpu_idx in range(ngpus):
                accum += (
                    ptrs[gpu_idx]
                    .address_space_cast[_target_address_space]()
                    .load[
                        width=simd_width,
                        alignment=align,
                        invariant=True,
                    ](global_elem)
                    .cast[accum_type]()
                )

        # Round to bf16 BEFORE norming: matches the two-launch path (RS emits a
        # bf16 shard); norming the wider f32 sum silently shifts MXFP4 accuracy.
        # Invalid lanes hold 0.
        var reduced = accum.cast[in_dtype]()

        # Folded AFTER that round, not into `accum`: the two-launch arm only
        # ever sees a rounded shard, so f32 here would split the arms by 1 ULP
        # and let the fuse threshold change model output.
        comptime if has_residual:
            if is_valid:
                reduced = (
                    reduced.cast[accum_type]()
                    + residual[]
                    .load[width=simd_width, alignment=align](
                        Coord(row, col_idx)
                    )
                    .cast[accum_type]()
                ).cast[in_dtype]()
        var reduced_f = reduced.cast[accum_type]()
        if is_valid:
            # `reduced` == the standalone reduce-scatter shard (residual stream).
            sum_out.store[width=simd_width](Coord(local_row, col_idx), reduced)

        # Mean-square over the full row. Divide by `cols` (full H) FIRST, add
        # epsilon SECOND (not the shard count; epsilon outside the division).
        var thread_m2 = (reduced_f**2).reduce_add()
        var row_m2 = block.sum[block_size=threads_per_block, broadcast=True](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](cols)) + epsilon.cast[accum_type]()
        )

        if is_valid:
            # (x * norm) * gamma in f32, cast to in_dtype last: mbc=True.
            var normalized = (reduced_f * norm_factor) * gamma_vec
            normed_out.store[width=simd_width](
                Coord(local_row, col_idx), normalized.cast[in_dtype]()
            )

    # End barrier: peers P2P-read this rank's `src_ptrs`, so they must not be
    # reused until every peer is done. Deferring to the next collective's start
    # barrier breaks once a second grouped op shares the domain (`domain_id`).
    _multi_gpu_barrier[ngpus, is_start=False, domain_id=domain_id](
        rank_sigs, rank_sigs[my_rank], my_rank
    )

    # Must stay after the end barrier: released earlier, the next kernel could
    # reuse `src_ptrs` memory while peers are still reading it. The graph
    # compiler sees this op as the last reader and cannot know about the peers.
    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()


# --- Launcher ---


def _reducescatter_rmsnorm_launch[
    simd_width: Int,
    in_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
    has_residual: Bool = False,
    domain_id: Int = 0,
    pdl_level: PDLLevel = PDLLevel(),
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
    ctx: DeviceContext,
    residual: _ComptimeConditionalTileTensor[
        in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        ImmutAnyOrigin,
        engaged=False,
    ](),
) raises:
    """Launch the fused reduce-scatter + RMSNorm kernel."""
    # A DP replica can legitimately get an empty batch, and a zero grid is
    # rejected by `enqueue_function`. Skipping the barrier pair is group-uniform
    # because `rows` derives from the group's shared input set, so every rank
    # skips together and no peer is stranded at the start barrier.
    if rows == 0:
        return

    comptime sm_version = get_sm_version()
    var payload_bytes = rows * cols * size_of[in_dtype]()
    var max_blocks = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](payload_bytes).get_num_blocks()
    # Grid capped at rank 0's max shard (ceildiv(rows, ngpus)) so every block
    # has work. Signal scratch only.
    var grid_size = min(ceildiv(rows, ngpus), max_blocks)
    var block_dim = threads_per_block

    comptime assert normed_out.flat_rank >= 2
    comptime assert sum_out.flat_rank >= 2

    comptime kernel = _reducescatter_rmsnorm_kernel[
        mut=gamma.mut,
        origin=gamma.origin,
        GammaLayoutType=gamma.LayoutType,
        normed_origin=normed_out.origin,
        NormedLayoutType=normed_out.LayoutType,
        sum_origin=sum_out.origin,
        SumLayoutType=sum_out.LayoutType,
        residual_origin=residual.T.origin,
        ResidualLayoutType=residual.T.LayoutType,
        in_dtype=in_dtype,
        ngpus=ngpus,
        simd_width=simd_width,
        threads_per_block=threads_per_block,
        has_residual=has_residual,
        domain_id=domain_id,
        pdl_level=pdl_level,
    ]
    ctx.enqueue_function[kernel](
        src_ptrs,
        gamma,
        normed_out,
        sum_out,
        residual,
        epsilon,
        weight_offset,
        Int32(rows),
        Int32(cols),
        rank_sigs,
        Int32(my_rank),
        grid_dim=grid_size,
        block_dim=block_dim,
        attributes=pdl_launch_attributes(pdl_level),
    )


# --- Public API ---


@always_inline
def _check_residual_extent[
    in_dtype: DType,
    //,
    has_residual: Bool,
](
    residual: _ComptimeConditionalTileTensor[
        in_dtype, engaged=has_residual, ...
    ],
    rows: Int,
    cols: Int,
) raises:
    """Reject a residual that is not the FULL pre-scatter `[rows, cols]` tensor.

    Indexed by GLOBAL row, so a shard-shaped one runs past its own storage on
    every rank whose shard does not start at row 0.
    """
    comptime if has_residual:
        if residual[].num_elements() != rows * cols:
            raise Error(
                String(
                    "reducescatter_rmsnorm: residual holds ",
                    residual[].num_elements(),
                    " elements, expected the FULL pre-scatter ",
                    rows * cols,
                    " (",
                    rows,
                    " x ",
                    cols,
                    "); it is indexed by global row, not by shard",
                )
            )


def reducescatter_rmsnorm[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
    has_residual: Bool = False,
    domain_id: Int = 0,
    pdl_level: PDLLevel = PDLLevel(),
](
    input_buffers: Array[TileTensor[in_dtype, in_layout, in_origin], ngpus],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    local_rank: Optional[Int] = None,
    residual: _ComptimeConditionalTileTensor[
        in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        ImmutAnyOrigin,
        engaged=False,
    ](),
) raises:
    """Fused reduce-scatter + RMSNorm across `ngpus` GPUs (bf16 in/out).

    Reduce-scatters `input_buffers` (each `[rows, cols]`, one per GPU) along
    rows and RMSNorm-normalizes each owned row. Writes this GPU's
    `[rank_units, cols]` shards: the reduce-scatter sum to `sum_out` and its
    RMSNorm to `normed_out`. `weight_offset` (1.0 for M3, Gemma-style) is folded
    into gamma in f32.

    Parameters:
        in_dtype: Input/output data type (bf16).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        has_residual: Fold `residual` into the reduce-scatter sum.
        domain_id: Barrier counter bank (0 for full-world, nonzero for a
            grouped collective so its counters never poison the full-world
            bank). Ops of the same width deliberately share a bank, which
            requires every rank in the domain to issue the same barrier
            sequence -- see `NUM_BARRIER_DOMAINS` in `sync.mojo` for the full
            invariant. Enforced here by the `rows == 0` guard and
            `_dispatch_rs_norm`'s group-invariant fuse gate.
        pdl_level: Enables PDL, so this kernel can start before the previous one
            retires and the next one is released once the end barrier clears.

    Args:
        input_buffers: Per-GPU input buffers as TileTensors (peer access
            required). Grouped collectives pass only their own group's buffers.
        normed_out: This GPU's normed output shard `[rank_units, cols]`.
        sum_out: This GPU's reduce-scatter sum shard `[rank_units, cols]` (the
            residual stream).
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        local_rank: Optional rank of THIS GPU within the collective's group.
            Defaults to the physical device id for full-world collectives.
            Grouped collectives MUST pass it: `input_buffers`/`rank_sigs` are
            group-local, so a global device id indexes them out of range.
        residual: Optional FULL `[rows, cols]` tensor added to the sum, in f32,
            before the pre-norm round. Each rank adds only its own shard of it.
            PRECONDITION: it must be bit-identical on every rank of the group.
            Callers fold a TP-replicated residual stream here rather than adding
            it on the group leader before the collective -- the reduce-scatter
            sums across ranks, so a leader-side add lands once for the whole
            group and a per-rank add of a NON-replicated residual would not
            reproduce it.

    Note:
        An end barrier is issued, so the P2P-read input buffers are free to be
        reused once this op retires. The outputs are still safe to read only on
        the local GPU; a remote-GPU consumer must insert its own barrier.
    """
    comptime assert ngpus >= 2, "reducescatter_rmsnorm requires at least 2 GPUs"
    comptime assert (
        in_dtype.is_floating_point()
    ), "in_dtype must be floating point"

    if not is_p2p_enabled():
        raise Error("reducescatter_rmsnorm requires P2P access between GPUs")

    # Compute rows/cols from the full (pre-scatter) input.
    var in_num_elems = input_buffers[0].num_elements()
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    var rows = in_num_elems // cols

    # Raw peer pointers, origin erased to ImmutAnyOrigin (matches standalone RS).
    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var src_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> PtrType: input_buffers[i]
        ._storage.as_imm()
        .as_unsafe_any_origin()
    )

    # Each thread owns `simd_width` cols; H=6144 fits the base width
    # (64*8*16=8192) on all targets (no AR two-width dispatch). Assert fit +
    # divisibility below.
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime threads_per_block = max_warps_per_block * WARP_SIZE
    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()
    comptime max_supported_cols = WARP_SIZE * simd_width * max_warps_per_block
    if cols > max_supported_cols:
        raise Error(
            String(
                "reducescatter_rmsnorm: cols (",
                cols,
                ") exceeds max supported (",
                max_supported_cols,
                ") for the warp-tiling kernel",
            )
        )
    if cols % simd_width != 0:
        raise Error(
            String(
                "reducescatter_rmsnorm: cols (",
                cols,
                ") must be a multiple of simd_width (",
                simd_width,
                ")",
            )
        )

    # The rank is caller-supplied and indexes the group-local `src_ptrs` and
    # `rank_sigs` arrays, so a global device id is out of range here.
    var my_rank = local_rank.value() if local_rank else Int(ctx.id())
    if not 0 <= my_rank < ngpus:
        raise Error(
            String(
                "reducescatter_rmsnorm: local_rank (",
                my_rank,
                ") must be the GROUP-local rank in [0, ",
                ngpus,
                (
                    "); a global device id indexes the group-local peer and"
                    " signal arrays out of range"
                ),
            )
        )

    # Nothing ties the rank-derived shard height to the caller's allocation.
    # Under a ragged partition a mismatched rank overruns both outputs by one
    # row with CORRECT values, which a byte-compare oracle cannot see. Mirrors
    # `reducescatter`'s output validation.
    var config_check = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=rows, unit_numel=cols, threads_per_gpu=0
    )
    var expected_numel = config_check.rank_num_elements(my_rank)
    if normed_out.num_elements() != expected_numel:
        raise Error(
            String(
                "reducescatter_rmsnorm: normed_out holds ",
                normed_out.num_elements(),
                " elements, expected ",
                expected_numel,
                " for rank ",
                my_rank,
                " of ",
                ngpus,
                " over ",
                rows,
                " rows",
            )
        )
    if sum_out.num_elements() != expected_numel:
        raise Error(
            String(
                "reducescatter_rmsnorm: sum_out holds ",
                sum_out.num_elements(),
                " elements, expected ",
                expected_numel,
                " for rank ",
                my_rank,
                " of ",
                ngpus,
                " over ",
                rows,
                " rows",
            )
        )

    _check_residual_extent[has_residual](residual, rows, cols)

    _reducescatter_rmsnorm_launch[
        simd_width,
        in_dtype,
        ngpus,
        threads_per_block,
        has_residual=has_residual,
        domain_id=domain_id,
        pdl_level=pdl_level,
    ](
        rows,
        cols,
        src_ptrs,
        normed_out,
        sum_out,
        gamma,
        epsilon,
        weight_offset,
        rank_sigs,
        my_rank,
        ctx,
        residual=residual,
    )


# --- Dispatch (A/B selector) ---


def _dispatch_rs_norm[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
    two_launch: def() raises capturing -> None,
    has_residual: Bool = False,
    domain_id: Int = 0,
    pdl_level: PDLLevel = PDLLevel(),
](
    input_buffers: Array[TileTensor[in_dtype, in_layout, in_origin], ngpus],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    threshold: Int = RS_NORM_FUSE_THRESHOLD,
    local_rank: Optional[Int] = None,
    residual: _ComptimeConditionalTileTensor[
        in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        ImmutAnyOrigin,
        engaged=False,
    ](),
) raises:
    """Runtime-select the fused kernel vs a caller-supplied two-launch path.

    `two_launch` (standalone reduce-scatter + `rms_norm`) is caller-supplied so
    `comm` stays free of `nn` (nn -> comm already exists). The graph op and the
    bench share this one selector.

    Parameters:
        in_dtype: Input/output data type (bf16).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        two_launch: Caller-supplied standalone reduce-scatter + RMSNorm closure.
            It must fold `residual` itself when `has_residual` -- this selector
            only threads the residual into the FUSED arm.
        has_residual: Fold `residual` into the fused arm's reduce-scatter sum.
        domain_id: Barrier counter bank for the fused kernel (0 for full-world;
            nonzero for grouped collectives, which deliberately share a bank per
            width). See `reducescatter_rmsnorm` for the invariant a shared bank
            requires, and `_multi_gpu_barrier`.
        pdl_level: PDL setting for the fused kernel; see `reducescatter_rmsnorm`.
            Comptime, so it cannot make the fuse gate below group-variant.

    Args:
        input_buffers: Per-GPU input buffers as TileTensors.
        normed_out: This GPU's normed output shard `[rank_units, cols]`.
        sum_out: This GPU's reduce-scatter sum shard `[rank_units, cols]`.
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        threshold: Per-rank-bytes fuse threshold; fuse at/below, else
            `two_launch`. Defaults to `RS_NORM_FUSE_THRESHOLD`.
        local_rank: Optional rank of THIS GPU within the collective's group;
            defaults to the physical device id. Required when grouped.
        residual: Optional FULL `[rows, cols]` residual for the fused arm; see
            `reducescatter_rmsnorm` for the group-replication precondition.
    """
    # Threshold is a bf16 row-count crossover in bytes; another element size
    # maps to the wrong row count and could fuse a diverging shape. Fail loud.
    comptime assert (
        in_dtype == .bfloat16
    ), "_dispatch_rs_norm fuse threshold is bf16-calibrated (bf16 in/out only)"

    # Fuse-vs-two-launch MUST be group-invariant: the paths issue different
    # barrier sequences on shared `rank_sigs`, so disagreement deadlocks. Gate
    # on group-rank 0's shard, NEVER `rank_units(local_rank)` -- under a ragged
    # partition low ranks own an extra row and could straddle the threshold.
    # Invariance is per-GROUP: sibling groups may legitimately diverge, their
    # `rank_sigs` being disjoint, so do not "fix" this to a world-wide gate.
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    var rows = input_buffers[0].num_elements() // cols
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=rows, unit_numel=cols, threads_per_gpu=0
    )
    var per_rank_bytes = config.rank_units(0) * cols * size_of[in_dtype]()
    var use_fused = per_rank_bytes <= threshold

    # Before branching: `two_launch` folds against the same global-row
    # partition, so a bad residual is wrong on both arms, not just the fused.
    _check_residual_extent[has_residual](residual, rows, cols)

    if use_fused:
        reducescatter_rmsnorm[
            has_residual=has_residual,
            domain_id=domain_id,
            pdl_level=pdl_level,
        ](
            input_buffers,
            normed_out,
            sum_out,
            gamma,
            epsilon,
            weight_offset,
            rank_sigs,
            ctx,
            local_rank,
            residual=residual,
        )
    else:
        two_launch()
