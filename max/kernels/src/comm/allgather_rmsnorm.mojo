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
"""Fused all-gather + RMSNorm (bf16 in/out, no FP8).

All-gathers `ngpus` row-shards into the full replicated `[rows, cols]` stream and
RMSNorms every gathered row in one launch (no separate-norm HBM round-trip).
Emits `sum_out` (gathered residual) and `normed_out` (its RMSNorm), both full and
replicated; `mbc=True`. A gathered row is a verbatim copy, so `sum_out` is
bit-for-bit a standalone `allgather` (no f32 peer-sum, unlike reduce-scatter).

Factored from `allreduce_residual_rmsnorm.mojo`'s Stage-2 gather + the norm math
of `reducescatter_rmsnorm.mojo` -- keep in sync. One block per GLOBAL row so each
row's `block.sum` runs parallel across blocks, not `ngpus` serial within one.
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
from layout import Coord, TensorLayout, TileTensor
from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from .allgather import allgather_tuning_table
from .device_query import dispatch_select_comm_config, get_sm_version
from .reducescatter import _target_address_space
from .sync import MAX_GPUS, Signal, _multi_gpu_barrier, is_p2p_enabled


# H the fuse threshold was calibrated at; a different H must recalibrate (both the
# perf crossover and the bit-identity vs `rms_norm_gpu` were measured here).
comptime AG_NORM_CALIBRATED_COLS = 6144

# `[rows, cols]` bytes at/below which `_dispatch_ag_norm` fuses (~128 rows at
# `AG_NORM_CALIBRATED_COLS`). A perf cut, not correctness: fused is
# bit-identical at every M; below it the gather absorbs the norm free, above it
# the fabric-saturated standalone gather wins.
comptime AG_NORM_FUSE_THRESHOLD = (
    128 * AG_NORM_CALIBRATED_COLS * size_of[DType.bfloat16]()
)


# --- GPU Kernel ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
@__name(t"allgather_rmsnorm_{in_dtype}_{ngpus}")
def _allgather_rmsnorm_kernel[
    mut: Bool,
    origin: Origin[mut=mut],
    GammaLayoutType: TensorLayout,
    normed_origin: MutOrigin,
    NormedLayoutType: TensorLayout,
    sum_origin: MutOrigin,
    SumLayoutType: TensorLayout,
    in_dtype: DType,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
    quant_epilogue: Optional[
        def[
            width: Int
        ](row: Int, col: Int, val: SIMD[in_dtype, width]) capturing -> None
    ] = None,
    domain_id: Int = 0,
](
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    in_lengths: StaticTuple[Int32, ngpus],
    gamma: TileTensor[in_dtype, GammaLayoutType, origin],
    normed_out: TileTensor[mut=True, in_dtype, NormedLayoutType, normed_origin],
    sum_out: TileTensor[mut=True, in_dtype, SumLayoutType, sum_origin],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    cols_dev: Int32,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank_dev: Int32,
):
    """Gather every source-GPU row into `[rows, cols]` and RMSNorm it in registers.

    `src_ptrs[g]` is peer `g`'s `[in_lengths[g], cols]` shard at global rows
    `[prefix(g), prefix(g)+in_lengths[g])`; `normed_out`/`sum_out` are the full
    replicated outputs, indexed by GLOBAL row.
    """
    var cols = Int(cols_dev)
    var my_rank = Int(my_rank_dev)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    # 2D stores below use Coord(grow, col_idx).
    comptime assert normed_out.flat_rank >= 2
    comptime assert sum_out.flat_rank >= 2
    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()

    var tid = Int(thread_idx.x)
    var col_idx = tid * simd_width
    var is_valid = col_idx < cols
    var num_blocks = Int(grid_dim.x)

    # Preload gamma in f32 before the barrier (local data, latency-hidden).
    var gamma_vec = SIMD[accum_type, simd_width](0)
    if is_valid:
        gamma_vec = (
            gamma.load[width=simd_width, alignment=align](Coord(col_idx)).cast[
                accum_type
            ]()
            + weight_offset.cast[accum_type]()
        )

    # Shard row counts -> global row starts (natural peer/concat order).
    var row_starts = StaticTuple[Int, ngpus](0)
    var rows = 0
    comptime for g in range(ngpus):
        row_starts[g] = rows
        rows += Int(in_lengths[g])

    # Start barrier: we P2P-read peers' shards, so all ranks must be ready.
    _multi_gpu_barrier[ngpus, is_start=True, domain_id=domain_id](
        rank_sigs, rank_sigs[my_rank], my_rank
    )

    for grow in range(Int(block_idx.x), rows, num_blocks):
        # Owning peer = last shard with start <= grow; grow < rows always lands
        # in a non-empty shard (trailing empties skipped).
        var g = 0
        comptime for gg in range(ngpus):
            if grow >= row_starts[gg]:
                g = gg
        var src_elem = (grow - row_starts[g]) * cols + col_idx

        # Gathered bf16 IS the residual (verbatim copy). Cast to f32 only for
        # the mean-square.
        var gathered = SIMD[in_dtype, simd_width](0)
        if is_valid:
            gathered = (
                src_ptrs[g]
                .address_space_cast[_target_address_space]()
                .load[width=simd_width, alignment=align, invariant=True](
                    src_elem
                )
            )
            sum_out.store[width=simd_width](Coord(grow, col_idx), gathered)

        var reduced_f = gathered.cast[accum_type]()
        # Divide by `cols` FIRST, add epsilon after (eps outside the division).
        # Invalid lanes hold 0.
        var thread_m2 = (reduced_f**2).reduce_add()
        var row_m2 = block.sum[block_size=threads_per_block, broadcast=True](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](cols)) + epsilon.cast[accum_type]()
        )
        # (x * norm) * gamma in f32, cast to in_dtype last: mbc=True.
        # Outside `is_valid` because `quant_epilogue` may reduce across lanes.
        var normalized = (reduced_f * norm_factor) * gamma_vec
        var normed = normalized.cast[in_dtype]()
        if is_valid:
            normed_out.store[width=simd_width](Coord(grow, col_idx), normed)
        # Same bf16 a standalone quantize would read back from `normed_out`, so
        # folding one in is byte-identical.
        comptime if quant_epilogue:
            comptime epilogue = quant_epilogue.value()
            epilogue[width=simd_width](grow, col_idx, normed)

    # End barrier: peers P2P-read this rank's shard, so it must not be reused
    # until every peer is done. Deferring to the next collective's start barrier
    # breaks once a second grouped op shares the domain (`domain_id`).
    _multi_gpu_barrier[ngpus, is_start=False, domain_id=domain_id](
        rank_sigs, rank_sigs[my_rank], my_rank
    )


# --- Launcher ---


def _allgather_rmsnorm_launch[
    simd_width: Int,
    in_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
    quant_epilogue: Optional[
        def[
            width: Int
        ](row: Int, col: Int, val: SIMD[in_dtype, width]) capturing -> None
    ] = None,
    domain_id: Int = 0,
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    in_lengths: StaticTuple[Int, ngpus],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
    ctx: DeviceContext,
) raises:
    """Launch the fused all-gather + RMSNorm kernel."""
    # A DP replica can legitimately get an empty batch, and a zero grid is
    # rejected by `enqueue_function`. Skipping the barrier pair is group-uniform
    # because `rows` sums the group's whole shard list, identical on every rank,
    # so no peer is stranded at the start barrier.
    if rows == 0:
        return

    comptime sm_version = get_sm_version()
    var payload_bytes = rows * cols * size_of[in_dtype]()
    # All-gather table (CDNA4 = 128), not allreduce (64): the norm is free only at
    # one block per row, and 64 would pack 2 rows/block (serial reductions) at
    # M=128. 128 covers the fuse band.
    var max_blocks = dispatch_select_comm_config[
        ngpus, sm_version, allgather_tuning_table
    ](payload_bytes).get_num_blocks()
    # One block per GLOBAL row, grid-strided beyond the cap.
    var grid_size = min(rows, max_blocks)
    var block_dim = threads_per_block

    comptime assert normed_out.flat_rank >= 2
    comptime assert sum_out.flat_rank >= 2

    comptime kernel = _allgather_rmsnorm_kernel[
        mut=gamma.mut,
        origin=gamma.origin,
        GammaLayoutType=gamma.LayoutType,
        normed_origin=normed_out.origin,
        NormedLayoutType=normed_out.LayoutType,
        sum_origin=sum_out.origin,
        SumLayoutType=sum_out.LayoutType,
        in_dtype=in_dtype,
        ngpus=ngpus,
        simd_width=simd_width,
        threads_per_block=threads_per_block,
        quant_epilogue=quant_epilogue,
        domain_id=domain_id,
    ]
    var in_lengths_dev = StaticTuple[Int32, ngpus](0)
    comptime for i in range(ngpus):
        in_lengths_dev[i] = Int32(in_lengths[i])
    ctx.enqueue_function[kernel](
        src_ptrs,
        in_lengths_dev,
        gamma,
        normed_out,
        sum_out,
        epsilon,
        weight_offset,
        Int32(cols),
        rank_sigs,
        Int32(my_rank),
        grid_dim=grid_size,
        block_dim=block_dim,
    )


# --- Public API ---


def allgather_rmsnorm[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
    domain_id: Int = 0,
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
) raises:
    """Fused all-gather + RMSNorm across `ngpus` GPUs (bf16 in/out).

    The bf16-only entry point: no quantized copy of the normed stream. See
    `_allgather_rmsnorm_impl`.

    Parameters:
        in_dtype: Input/output data type (bf16).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input shard TileTensors.
        in_origin: Origin of the input shard TileTensors.
        domain_id: Barrier counter bank; see `_allgather_rmsnorm_impl`.

    Args:
        input_buffers: Per-GPU input row-shards as TileTensors.
        normed_out: This GPU's full normed output `[rows, cols]`.
        sum_out: This GPU's full gathered residual `[rows, cols]`.
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        local_rank: Optional group-local rank of THIS GPU.
    """

    _allgather_rmsnorm_impl[domain_id=domain_id](
        input_buffers,
        normed_out,
        sum_out,
        gamma,
        epsilon,
        weight_offset,
        rank_sigs,
        ctx,
        local_rank,
    )


def _allgather_rmsnorm_impl[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
    quant_epilogue: Optional[
        def[
            width: Int
        ](row: Int, col: Int, val: SIMD[in_dtype, width]) capturing -> None
    ] = None,
    domain_id: Int = 0,
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
) raises:
    """Fused all-gather + RMSNorm across `ngpus` GPUs (bf16 in/out).

    All-gathers `input_buffers` (each `[shard_i, cols]`, one shard per GPU) along
    rows into the full replicated `[rows, cols]` stream and RMSNorm-normalizes
    every gathered row. Writes the full (replicated) outputs on this GPU: the
    gathered residual to `sum_out` and its RMSNorm to `normed_out`.
    `weight_offset` (1.0 for M3, Gemma-style) is folded into gamma in f32.

    Parameters:
        in_dtype: Input/output data type (bf16).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input shard TileTensors.
        in_origin: Origin of the input shard TileTensors.
        quant_epilogue: Optional. Receives each normed value already cast
            to `in_dtype`, so a folded-in quantize is byte-identical. Called on
            ALL lanes (invalid ones carry 0), so it must guard its own stores.
            `None` elides it entirely.
        domain_id: Barrier counter bank (0 for full-world, nonzero for a
            grouped collective so its counters never poison the full-world
            bank). Ops of the same width deliberately share a bank, which
            requires every rank in the domain to issue the same barrier
            sequence -- see `NUM_BARRIER_DOMAINS` in `sync.mojo` for the full
            invariant. Enforced here by the `rows == 0` guard and
            `_dispatch_ag_norm`'s group-invariant fuse gate.

    Args:
        input_buffers: Per-GPU input row-shards as TileTensors (peer access
            required); shard `i` becomes global rows
            `[prefix(i), prefix(i)+shard_i)`. Grouped collectives pass only
            their own group's shards, so `rows` is the group's gathered total.
        normed_out: This GPU's full normed output `[rows, cols]`.
        sum_out: This GPU's full gathered residual `[rows, cols]`.
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        local_rank: Optional rank of THIS GPU within the collective's group.
            Defaults to the physical device id for full-world collectives.
            Grouped collectives MUST pass it: `input_buffers`/`rank_sigs` are
            group-local, so a global device id indexes them out of range.

    Note:
        An end barrier is issued, so the P2P-read input shards are free to be
        reused once this op retires. Outputs are still safe to read only on the
        local GPU; a remote consumer must add its own barrier.
    """
    comptime assert ngpus >= 2, "allgather_rmsnorm requires at least 2 GPUs"
    comptime assert (
        in_dtype.is_floating_point()
    ), "in_dtype must be floating point"

    if not is_p2p_enabled():
        raise Error("allgather_rmsnorm requires P2P access between GPUs")

    # cols = shard last dim; per-peer row counts (may be ragged/0); `rows` is
    # the gathered total.
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())

    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var src_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> PtrType: input_buffers[i]
        ._storage.as_imm()
        .as_unsafe_any_origin()
    )
    var in_lengths = StaticTuple[Int, ngpus](0)
    var rows = 0
    comptime for i in range(ngpus):
        var len_i = input_buffers[i].num_elements() // cols
        in_lengths[i] = len_i
        rows += len_i

    # Each thread owns `simd_width` cols; H=6144 fits the base width
    # (64*8*16=8192). Assert fit + divisibility below.
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime threads_per_block = max_warps_per_block * WARP_SIZE
    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()
    comptime max_supported_cols = WARP_SIZE * simd_width * max_warps_per_block
    if cols > max_supported_cols:
        raise Error(
            String(
                "allgather_rmsnorm: cols (",
                cols,
                ") exceeds max supported (",
                max_supported_cols,
                ") for the warp-tiling kernel",
            )
        )
    if cols % simd_width != 0:
        raise Error(
            String(
                "allgather_rmsnorm: cols (",
                cols,
                ") must be a multiple of simd_width (",
                simd_width,
                ")",
            )
        )

    # The caller-supplied rank indexes group-local arrays, so a global device
    # id is out of range. Only the first `ngpus` `rank_sigs` slots are
    # initialized: reading past them faults on-device, far from the call site.
    var my_rank = local_rank.value() if local_rank else Int(ctx.id())
    if not 0 <= my_rank < ngpus:
        raise Error(
            String(
                "allgather_rmsnorm: local_rank (",
                my_rank,
                ") must be the GROUP-local rank in [0, ",
                ngpus,
                (
                    "); a global device id indexes the group-local peer and"
                    " signal arrays out of range"
                ),
            )
        )

    # Both outputs hold the GROUP's gathered tensor, replicated. Sizing them
    # from the whole world instead is the natural TP-within-DP mistake, and the
    # kernel writes by global row, so it would overrun with correct values.
    var expected_numel = rows * cols
    if normed_out.num_elements() != expected_numel:
        raise Error(
            String(
                "allgather_rmsnorm: normed_out holds ",
                normed_out.num_elements(),
                " elements, expected ",
                expected_numel,
                " (",
                rows,
                " gathered rows over ",
                ngpus,
                " shards x ",
                cols,
                " cols)",
            )
        )
    if sum_out.num_elements() != expected_numel:
        raise Error(
            String(
                "allgather_rmsnorm: sum_out holds ",
                sum_out.num_elements(),
                " elements, expected ",
                expected_numel,
                " (",
                rows,
                " gathered rows over ",
                ngpus,
                " shards x ",
                cols,
                " cols)",
            )
        )

    _allgather_rmsnorm_launch[
        simd_width,
        in_dtype,
        ngpus,
        threads_per_block,
        quant_epilogue=quant_epilogue,
        domain_id=domain_id,
    ](
        rows,
        cols,
        src_ptrs,
        in_lengths,
        normed_out,
        sum_out,
        gamma,
        epsilon,
        weight_offset,
        rank_sigs,
        my_rank,
        ctx,
    )


def allgather_rmsnorm_quant[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
    quant_epilogue: def[width: Int](
        row: Int, col: Int, val: SIMD[in_dtype, width]
    ) capturing -> None,
    domain_id: Int = 0,
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
) raises:
    """`allgather_rmsnorm` that also hands each normed value to an epilogue.

    The epilogue sees the bf16 that lands in `normed_out`, so a folded-in
    quantizer emits the same bytes one launch fewer. Caller-supplied because
    `comm` cannot depend on `linalg` (which already depends on `comm`).

    Parameters:
        in_dtype: Input/output data type (bf16).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input shard TileTensors.
        in_origin: Origin of the input shard TileTensors.
        quant_epilogue: Normed-value epilogue; see `_allgather_rmsnorm_impl`.
        domain_id: Barrier counter bank; see `_allgather_rmsnorm_impl`.

    Args:
        input_buffers: Per-GPU input row-shards as TileTensors.
        normed_out: This GPU's full normed output `[rows, cols]`.
        sum_out: This GPU's full gathered residual `[rows, cols]`.
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        local_rank: Optional group-local rank of THIS GPU.
    """
    _allgather_rmsnorm_impl[quant_epilogue=quant_epilogue, domain_id=domain_id](
        input_buffers,
        normed_out,
        sum_out,
        gamma,
        epsilon,
        weight_offset,
        rank_sigs,
        ctx,
        local_rank,
    )


# --- Dispatch (A/B selector) ---


def _dispatch_ag_norm[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
    two_launch: def() raises capturing -> None,
    domain_id: Int = 0,
](
    input_buffers: Array[TileTensor[in_dtype, in_layout, in_origin], ngpus],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    threshold: Int = AG_NORM_FUSE_THRESHOLD,
    local_rank: Optional[Int] = None,
) raises:
    """Runtime-select the fused kernel vs a caller-supplied two-launch path.

    `two_launch` (standalone all-gather + `rms_norm`) is caller-supplied so `comm`
    stays free of `nn`. Shared by the graph op and the bench.

    Parameters:
        in_dtype: Input/output data type (bf16).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input shard TileTensors.
        in_origin: Origin of the input shard TileTensors.
        two_launch: Caller-supplied standalone all-gather + RMSNorm closure.
        domain_id: Barrier counter bank for the fused kernel (0 for full-world;
            nonzero for grouped collectives, which deliberately share a bank per
            width). See `allgather_rmsnorm` for the invariant a shared bank
            requires, and `_multi_gpu_barrier`.

    Args:
        input_buffers: Per-GPU input row-shards as TileTensors.
        normed_out: This GPU's full normed output `[rows, cols]`.
        sum_out: This GPU's full gathered residual `[rows, cols]`.
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        threshold: Full-`[rows, cols]`-bytes fuse threshold; fuse at/below, else
            `two_launch`. Defaults to `AG_NORM_FUSE_THRESHOLD`.
        local_rank: Optional rank of THIS GPU within the collective's group;
            defaults to the physical device id. Required when grouped.
    """
    # Threshold is bf16-row-count in bytes; another element size fuses a diverging
    # shape. Fail loud.
    comptime assert (
        in_dtype == .bfloat16
    ), "_dispatch_ag_norm fuse threshold is bf16-calibrated (bf16 in/out only)"

    # Gates on the full replicated row count, identical on every rank: the two
    # paths issue different barriers on shared `rank_sigs`, so disagreement
    # deadlocks. Invariance is per-GROUP -- sibling groups may legitimately
    # diverge, their `rank_sigs` being disjoint, so do not widen this gate.
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    # Fuse only at the calibrated H (else the byte threshold maps to the wrong
    # row count).
    debug_assert(
        cols == AG_NORM_CALIBRATED_COLS,
        (
            "allgather_rmsnorm auto-fuse threshold is calibrated for"
            " H=6144; recalibrate AG_NORM_FUSE_THRESHOLD before fusing"
            " another H"
        ),
    )
    var rows = 0
    comptime for i in range(ngpus):
        rows += input_buffers[i].num_elements() // cols
    var full_bytes = rows * cols * size_of[in_dtype]()

    if full_bytes <= threshold:
        allgather_rmsnorm[domain_id=domain_id](
            input_buffers,
            normed_out,
            sum_out,
            gamma,
            epsilon,
            weight_offset,
            rank_sigs,
            ctx,
            local_rank,
        )
    else:
        two_launch()


def _dispatch_ag_norm_quant[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
    two_launch_with_quant: def() raises capturing -> None,
    quant_epilogue: def[width: Int](
        row: Int, col: Int, val: SIMD[in_dtype, width]
    ) capturing -> None,
    domain_id: Int = 0,
](
    input_buffers: Array[TileTensor[in_dtype, in_layout, in_origin], ngpus],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    threshold: Int = AG_NORM_FUSE_THRESHOLD,
    local_rank: Optional[Int] = None,
) raises:
    """`_dispatch_ag_norm` whose fused branch also runs `quant_epilogue`.

    `two_launch_with_quant` MUST also quantize: above the threshold it is the
    whole computation. The name is deliberately not `two_launch` -- that
    bf16-only closure has an identical type and would fit here silently.

    Parameters:
        in_dtype: Input/output data type (bf16).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input shard TileTensors.
        in_origin: Origin of the input shard TileTensors.
        two_launch_with_quant: Standalone all-gather + RMSNorm + quantize
            closure.
        quant_epilogue: Fused-path normed-value epilogue; see
            `allgather_rmsnorm_quant`.
        domain_id: Barrier counter bank for the fused kernel; see
            `_dispatch_ag_norm`.

    Args:
        input_buffers: Per-GPU input row-shards as TileTensors.
        normed_out: This GPU's full normed output `[rows, cols]`.
        sum_out: This GPU's full gathered residual `[rows, cols]`.
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        threshold: Full-`[rows, cols]`-bytes fuse threshold; fuse at/below, else
            `two_launch_with_quant`. Defaults to `AG_NORM_FUSE_THRESHOLD`. MUST
            be group-uniform -- see the deadlock note below.
        local_rank: Optional group-local rank of THIS GPU; defaults to the
            physical device id. Required when grouped.
    """
    comptime assert in_dtype == .bfloat16, (
        "_dispatch_ag_norm_quant fuse threshold is bf16-calibrated (bf16 in/out"
        " only)"
    )

    # Gates on the full replicated row count, identical on every rank: the two
    # paths issue different barriers on shared `rank_sigs`, so disagreement
    # deadlocks. Invariance is per-GROUP -- sibling groups may legitimately
    # diverge, their `rank_sigs` being disjoint, so do not widen this gate.
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    debug_assert(
        cols == AG_NORM_CALIBRATED_COLS,
        (
            "allgather_rmsnorm auto-fuse threshold is calibrated for"
            " H=6144; recalibrate AG_NORM_FUSE_THRESHOLD before fusing"
            " another H"
        ),
    )
    var rows = 0
    comptime for i in range(ngpus):
        rows += input_buffers[i].num_elements() // cols
    var full_bytes = rows * cols * size_of[in_dtype]()

    if full_bytes <= threshold:
        allgather_rmsnorm_quant[
            quant_epilogue=quant_epilogue, domain_id=domain_id
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
        )
    else:
        two_launch_with_quant()
