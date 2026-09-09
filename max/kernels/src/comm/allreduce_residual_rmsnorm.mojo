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
"""Fused allreduce + RMSNorm (+ optional FP8 quantization) kernel.

Combines P2P allreduce, RMSNorm normalization, and optional FP8 dynamic
quantization into a single kernel launch. Data stays in registers
throughout, with no global memory intermediate between allreduce and RMSNorm.

Quantization is fused in only when the output dtype differs from the input
dtype (`out_dtype != in_dtype`). When `out_dtype == in_dtype` (e.g. a bf16
input with a bf16 output) the FP8 phases (per-row max, dynamic scale, and
quantize) are skipped at compile time, the normalized value is written
directly in the input dtype, and no per-row scale is produced.

Three dispatch paths:

1-stage (small payloads): Each thread loads from all GPUs, accumulates
  in float32 registers, then applies RMSNorm + FP8 quantization. Simple
  but O(N x ngpus) P2P traffic.

2-stage (large payloads, non-residual or medium residual): Single kernel
  with two stages separated by a per-block barrier:
    Stage 1 (RS + RMSNorm + FP8): each block reduces rows in its partition
      from all GPUs in f32 registers, then normalizes and quantizes to FP8
      immediately, with no f32 scratch. Writes compact fp8 data + per-row scale
      (+ optional bf16 residual) to local scratch.
    Stage 2 (AG copy): each block reads compact fp8/scale/bf16 from the
      owning GPU's scratch (P2P) and writes to local output buffers. No
      compute, just copies of data that is 4x smaller than f32.
  Both stages use the same row-to-block mapping, so the per-block barrier
  correctly synchronizes all data dependencies. Total P2P traffic is
  N * sizeof(in_dtype) for Stage 1 (same as 1-stage) plus
  N * sizeof(fp8) [+ N * sizeof(in_dtype) if residual] for Stage 2.
  The AG phase copies fp8 (1 byte/elem) instead of f32 (4 bytes/elem),
  reducing Stage 2 bandwidth by ~4x for the non-residual case.

Split (large residual payloads): Two separate kernel launches:
  allreduce with add epilogue followed by fused rmsnorm+fp8. Avoids
  carrying bf16 residual data through scratch buffers in both stages,
  which nearly doubles Stage 2 copy bandwidth in the fused path. The
  caller-provided residual_output buffer serves as the intermediate,
  so no extra allocation is needed.

Each block processes multiple rows via a grid-strided loop, allowing
row counts beyond the hardware-tuned block limit. Gamma weights are
preloaded once and reused across all rows in the loop.
"""

from std.collections import Array, Optional
from std.collections._conditional import _ComptimeConditional
from std.math import align_up, ceildiv, rsqrt
from std.sys import (
    align_of,
    get_defined_int,
    has_amd_gpu_accelerator,
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
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.primitives import block
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.tile_tensor import _ComptimeConditionalTileTensor
from std.utils import IndexList, StaticTuple
from std.utils.numerics import get_accum_type


from internal_utils.fp8_utils import compute_dynamic_fp8_scale, fp8_quantize

from .rms_norm_fp8 import rms_norm_fused_fp8

from .allreduce import (
    allreduce,
    elementwise_epilogue_type,
    allreduce_tuning_table,
)
from .device_query import get_sm_version, dispatch_select_comm_config
from .reducescatter import _target_address_space
from .sync import (
    MAX_GPUS,
    Signal,
    _multi_gpu_barrier,
    is_p2p_enabled,
)


# --- GPU Kernel ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
@__name(
    t"allreduce_rmsnorm_fp8_warp_tiling_{in_dtype}_{out_dtype}_{ngpus}_{has_residual}",
)
def _allreduce_rmsnorm_fp8_kernel_warp_tiling[
    mut: Bool,
    origin: Origin[mut=mut],
    LayoutType: TensorLayout,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    scale_origin: MutOrigin,
    ScaleLayoutType: TensorLayout,
    ResidualLayoutType: TensorLayout,
    residual_origin: Origin,
    ResidualOutLayoutType: TensorLayout,
    residual_out_origin: MutOrigin,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
    has_residual: Bool,
    output_fn: def[width: SIMDLength](
        row: Int, col: Int, val: SIMD[out_dtype, width]
    ) capturing -> None,
](
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    gamma: TileTensor[in_dtype, LayoutType, origin],
    scale_buffer: TileTensor[
        mut=True, scales_dtype, ScaleLayoutType, scale_origin
    ],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rows: Int32,
    cols: Int32,
    scale_ub: Float32,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int32,
    residual: _ComptimeConditionalTileTensor[
        in_dtype,
        ResidualLayoutType,
        residual_origin,
        engaged=has_residual,
    ],
    residual_output: _ComptimeConditionalTileTensor[
        in_dtype,
        ResidualOutLayoutType,
        residual_out_origin,
        engaged=has_residual,
    ],
):
    """Fused allreduce + RMSNorm + FP8 kernel using warp-tiling.

    Each thread block processes one or more rows via a persistent row loop
    (stride = grid_dim). Data stays in registers across all four phases:
    P2P load+reduce, mean-square, normalize+max, quantize+write.
    Gamma weights are preloaded once and reused across all rows.
    """
    var _rows = Int(rows)
    var _cols = Int(cols)
    var _my_rank = Int(my_rank)
    var _epsilon = Scalar[in_dtype](epsilon)
    var _weight_offset = Scalar[in_dtype](weight_offset)
    var _scale_ub = Scalar[scales_dtype](scale_ub)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert scale_buffer.flat_rank == 1, "scale_buffer must have rank 1"
    # Provide evidence that flat_rank >= 1 for the Coord(...) loads below.
    comptime assert gamma.flat_rank >= 1
    # Provide evidence that flat_rank >= 2 for the Coord(..., ...)
    # loads/stores on residual and residual_output below.
    comptime assert residual.T.flat_rank >= 2
    comptime assert residual_output.T.flat_rank >= 2
    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()
    # Fuse FP8 quantization only when the output dtype differs from the input.
    # When out_dtype == in_dtype the max/scale/quantize phases are skipped.
    comptime quantize = out_dtype != in_dtype

    var tid = thread_idx.x
    var idx = tid * simd_width
    var is_valid = idx < _cols

    # Barrier: wait for all GPUs to have their input data ready.
    _multi_gpu_barrier[ngpus, is_start=True](
        rank_sigs, rank_sigs[_my_rank], _my_rank
    )

    # Preload gamma weights into registers. The global memory latency is
    # hidden behind the P2P loads below (~200+ cycles per GPU).
    # Gamma depends only on column index — preload once, reuse across all _rows.
    var gamma_vec = SIMD[accum_type, simd_width](0)
    if is_valid:
        gamma_vec = (
            gamma.load[width=simd_width, alignment=align](Coord(idx)).cast[
                accum_type
            ]()
            + _weight_offset.cast[accum_type]()
        )

    # Round-robin access pattern for NVLink load-balancing.
    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var ptrs = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> PtrType: src_ptrs[
            (_my_rank + i) % ngpus
        ]
    )

    # Row loop: each block processes _rows with stride = grid_dim.
    # For _rows <= grid_dim, the loop body runs exactly once per block.
    var num_blocks = grid_dim.x
    for row in range(block_idx.x, _rows, num_blocks):
        # Phase 0: P2P load from all GPUs + accumulate in float32 regs.
        # Gamma load latency from above is hidden during P2P stalls.
        var vec_data = SIMD[accum_type, simd_width](0)
        if is_valid:
            var elem_idx = row * _cols + idx

            comptime for gpu_idx in range(ngpus):
                vec_data += (
                    ptrs[gpu_idx]
                    .address_space_cast[_target_address_space]()
                    .load[
                        width=simd_width,
                        alignment=simd_width,
                        invariant=True,
                    ](elem_idx)
                    .cast[accum_type]()
                )

            # Add residual and write pre-norm sum (compile-time gated).
            comptime if has_residual:
                vec_data += (
                    residual[]
                    .load[width=simd_width](Coord(row, idx))
                    .cast[accum_type]()
                )
                # Write bf16 pre-normalization sum for the residual stream.
                residual_output[].store[width=simd_width](
                    Coord(row, idx), vec_data.cast[in_dtype]()
                )

        # Phase 1: Compute mean-square.
        var thread_m2 = (vec_data**2).reduce_add()
        var row_m2 = block.sum[block_size=threads_per_block, broadcast=True](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](_cols)) + _epsilon.cast[accum_type]()
        )

        # Phase 2: Normalize (preloaded gamma, no global load).
        var normalized = SIMD[accum_type, simd_width](0)
        if is_valid:
            normalized = (vec_data * norm_factor) * gamma_vec

        comptime if quantize:
            # Phase 2b (FP8 only): find row max and compute the dynamic scale.
            var thread_max = Scalar[accum_type](0)
            if is_valid:
                thread_max = abs(normalized).reduce_max()
            var row_max = block.max[
                block_size=threads_per_block, broadcast=True
            ](thread_max)
            var scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
                out_dtype
            ](row_max, _scale_ub)
            if tid == 0:
                scale_buffer[row] = scale_factor

            # Phase 3: Quantize and write (normalized values in registers).
            if is_valid:
                var output_fp8 = fp8_quantize[out_dtype](
                    normalized, scale_factor_recip
                )
                output_fn[simd_width](row, idx, output_fp8)
        else:
            # Phase 3 (no quant): write the normalized value directly in
            # out_dtype (== in_dtype). No per-row scale is produced.
            if is_valid:
                output_fn[simd_width](row, idx, normalized.cast[out_dtype]())

    # NOTE: No end barrier needed. The FP8 output is consumed only by the
    # local GPU, so stream ordering guarantees writes complete before the
    # next kernel reads them. The start barrier of the NEXT allreduce call
    # protects the input buffers (which ARE read by remote GPUs).


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
@__name(
    t"allreduce_rmsnorm_fp8_2stage_{in_dtype}_{out_dtype}_{ngpus}_{has_residual}",
)
def _allreduce_rmsnorm_fp8_kernel_2stage[
    mut: Bool,
    origin: Origin[mut=mut],
    LayoutType: TensorLayout,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    scale_origin: MutOrigin,
    ScaleLayoutType: TensorLayout,
    ResidualLayoutType: TensorLayout,
    residual_origin: Origin,
    ResidualOutLayoutType: TensorLayout,
    residual_out_origin: MutOrigin,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
    has_residual: Bool,
    output_fn: def[width: SIMDLength](
        row: Int, col: Int, val: SIMD[out_dtype, width]
    ) capturing -> None,
](
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    gamma: TileTensor[in_dtype, LayoutType, origin],
    scale_buffer: TileTensor[
        mut=True, scales_dtype, ScaleLayoutType, scale_origin
    ],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rows: Int32,
    cols: Int32,
    scale_ub: Float32,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int32,
    residual: _ComptimeConditionalTileTensor[
        in_dtype,
        ResidualLayoutType,
        residual_origin,
        engaged=has_residual,
    ],
    residual_output: _ComptimeConditionalTileTensor[
        in_dtype,
        ResidualOutLayoutType,
        residual_out_origin,
        engaged=has_residual,
    ],
):
    """Single-kernel 2-stage fused RS + RMSNorm + FP8 + AG.

    Stage 1 (RS + RMSNorm + FP8): Each block reduces rows in its partition
    from all GPUs in f32 registers, normalizes, quantizes to FP8, and writes
    compact results (fp8 + scale + optional bf16 residual) to local scratch.
    No f32 scratch is written: data stays in registers through RS → Norm → FP8.

    Inter-stage: Per-block barrier with release/acquire fence ensures all
    scratch writes from block B on every GPU are visible before block B
    on any GPU starts Stage 2.

    Stage 2 (AG copy): Each block iterates over each GPU's partition and
    reads fp8/scale/bf16 from that GPU's scratch via P2P, then writes to
    local output buffers. Consecutive reads from the same peer improve
    NVLink pipelining. No compute, just copies of compact data.

    Scratch layout per GPU (after Signal header):
      [fp8:      rows_per_rank * cols bytes]
      [scales:   align_up(rows_per_rank * sizeof(scales_dtype), simd_width) bytes]
      [residual: rows_per_rank * cols * sizeof(in_dtype) bytes] (if has_residual)

    The scale section is padded to a multiple of simd_width bytes so the
    residual section starts at a simd_width-byte aligned offset (required
    for SIMD residual loads/stores with alignment=simd_width).

    Minimum signal buffer size per GPU:
        sizeof(Signal)
        + rows_per_rank * cols                                          (fp8 data)
        + align_up(rows_per_rank * sizeof(scales_dtype), simd_width)   (scales + pad)
        + rows_per_rank * cols * sizeof(in_dtype)                       (if has_residual)

    Both stages iterate over local rows within each partition using
    `for local_row in range(block_idx.x, rows_per_rank, num_blocks)`.
    Block B processes local_rows {B, B+num_blocks, B+2*num_blocks, ...}
    in both stages, so the per-block barrier correctly synchronizes:
    block B on GPU C writes those local rows in Stage 1, and block B
    on any GPU reads them from GPU C's scratch in Stage 2.
    """
    var _rows = Int(rows)
    var _cols = Int(cols)
    var _my_rank = Int(my_rank)
    var _epsilon = Scalar[in_dtype](epsilon)
    var _weight_offset = Scalar[in_dtype](weight_offset)
    var _scale_ub = Scalar[scales_dtype](scale_ub)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert scale_buffer.flat_rank == 1, "scale_buffer must have rank 1"
    # Provide evidence that flat_rank >= 1 for the Coord(...) loads below.
    comptime assert gamma.flat_rank >= 1
    # Provide evidence that flat_rank >= 2 for the Coord(..., ...)
    # loads/stores on residual and residual_output below.
    comptime assert residual.T.flat_rank >= 2
    comptime assert residual_output.T.flat_rank >= 2
    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()
    # Fuse FP8 quantization only when the output dtype differs from the input.
    # When out_dtype == in_dtype the scratch carries the normalized value in
    # out_dtype (== in_dtype) and no scale section is reserved or written.
    comptime quantize = out_dtype != in_dtype

    var tid = thread_idx.x
    var col_idx = tid * simd_width
    var is_valid = col_idx < _cols
    var num_blocks = grid_dim.x

    # Row-aligned partitioning: each GPU owns ceildiv(_rows, ngpus) _rows.
    var rows_per_rank = ceildiv(_rows, ngpus)
    var my_row_start = _my_rank * rows_per_rank

    # --- Scratch pointers (after Signal header) ---
    # Layout: [fp8_data | scales (padded) | bf16_residual (optional)]
    # See kernel docstring for the full size formula.
    var fp8_per_rank = rows_per_rank * _cols
    comptime if quantize:
        assert (
            fp8_per_rank % 4 == 0
        ), "fp8 scratch must be 4-byte aligned for scale stores"

    # Round up the scale element count to the next multiple of
    # (simd_width / sizeof(scales_dtype)) so that the residual scratch
    # pointer (scale_ptr + scale_pad_elements) is simd_width-byte aligned.
    # Without this, rows_per_rank * sizeof(scales_dtype) may not be
    # divisible by simd_width, causing a misaligned SIMD residual store
    # when ceildiv(_rows, ngpus) % (simd_width / sizeof(scales_dtype)) != 0.
    # When not quantizing there is no scale section, so the residual scratch
    # follows the output scratch directly (scale_pad_elements = 0).
    comptime scales_per_simd_chunk = simd_width // size_of[scales_dtype]()
    var scale_pad_elements: Int
    comptime if quantize:
        scale_pad_elements = (
            ceildiv(rows_per_rank, scales_per_simd_chunk)
            * scales_per_simd_chunk
        )
    else:
        scale_pad_elements = 0

    # Local scratch pointers (for writes in Stage 1).
    # +1 advances the Signal pointer by sizeof(Signal) bytes, stepping
    # past the Signal header to reach the scratch region of the buffer.
    var scratch_fp8 = (
        rank_sigs[_my_rank].address_space_cast[.GENERIC]() + 1
    ).bitcast[Scalar[out_dtype]]()
    var scratch_scale = (scratch_fp8 + fp8_per_rank).bitcast[
        Scalar[scales_dtype]
    ]()
    var scratch_residual = (scratch_scale + scale_pad_elements).bitcast[
        Scalar[in_dtype]
    ]()

    # P2P scratch pointers for all GPUs (for reads in Stage 2).
    # +1 advances by sizeof(Signal) bytes (see local scratch note above).
    comptime Fp8PtrType = MutPointer[Scalar[out_dtype], MutAnyOrigin]
    var fp8_ptrs = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> Fp8PtrType: (
            rank_sigs[i].address_space_cast[.GENERIC]() + 1
        ).bitcast[Scalar[out_dtype]]()
    )

    comptime ScalePtrType = MutPointer[Scalar[scales_dtype], MutAnyOrigin]
    var scale_ptrs = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> ScalePtrType: (
            (rank_sigs[i].address_space_cast[.GENERIC]() + 1).bitcast[
                Scalar[out_dtype]
            ]()
            + fp8_per_rank
        ).bitcast[Scalar[scales_dtype]]()
    )
    var residual_ptrs = Array[
        MutPointer[Scalar[in_dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)

    comptime if has_residual:
        comptime for i in range(ngpus):
            residual_ptrs[i] = (scale_ptrs[i] + scale_pad_elements).bitcast[
                Scalar[in_dtype]
            ]()

    # Round-robin P2P input pointers for NVLink load-balancing.
    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var ptrs = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> PtrType: src_ptrs[
            (_my_rank + i) % ngpus
        ]
    )

    # Preload gamma weights into registers BEFORE start barrier.
    # Gamma is local data — safe to load early, hides latency behind barrier.
    var gamma_vec = SIMD[accum_type, simd_width](0)
    if is_valid:
        gamma_vec = (
            gamma.load[width=simd_width, alignment=align](Coord(col_idx)).cast[
                accum_type
            ]()
            + _weight_offset.cast[accum_type]()
        )

    # Start barrier: wait for all GPUs to have their input data ready.
    _multi_gpu_barrier[ngpus, is_start=True](
        rank_sigs, rank_sigs[_my_rank], _my_rank
    )

    # === Stage 1: RS + RMSNorm + FP8 (partition _rows only) ===
    # Each block iterates over its partition _rows directly. With
    # grid_dim <= rows_per_rank, every block has at least one row.
    for local_row in range(block_idx.x, rows_per_rank, num_blocks):
        var row = my_row_start + local_row
        if row < _rows:
            # P2P load from all GPUs + accumulate in f32 registers.
            var accum = SIMD[accum_type, simd_width](0)
            if is_valid:
                var global_elem = row * _cols + col_idx

                comptime for gpu_idx in range(ngpus):
                    accum += (
                        ptrs[gpu_idx]
                        .address_space_cast[_target_address_space]()
                        .load[
                            width=simd_width,
                            alignment=simd_width,
                            invariant=True,
                        ](global_elem)
                        .cast[accum_type]()
                    )

                # Add residual and write residual to scratch (compile-time gated).
                comptime if has_residual:
                    accum += (
                        residual[]
                        .load[width=simd_width](Coord(row, col_idx))
                        .cast[accum_type]()
                    )
                    var local_elem = local_row * _cols + col_idx
                    scratch_residual.address_space_cast[
                        _target_address_space
                    ]().store[width=simd_width, alignment=simd_width](
                        local_elem, accum.cast[in_dtype]()
                    )

            # RMSNorm: compute mean-square.
            var thread_m2 = (accum**2).reduce_add()
            var row_m2 = block.sum[
                block_size=threads_per_block, broadcast=True
            ](thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](_cols))
                + _epsilon.cast[accum_type]()
            )

            # Normalize.
            var normalized = SIMD[accum_type, simd_width](0)
            if is_valid:
                normalized = (accum * norm_factor) * gamma_vec

            comptime if quantize:
                # Find max for FP8 scale, write scale + fp8 to local scratch.
                var thread_max = Scalar[accum_type](0)
                if is_valid:
                    thread_max = abs(normalized).reduce_max()

                var row_max = block.max[
                    block_size=threads_per_block, broadcast=True
                ](thread_max)
                var scale_factor, scale_factor_recip = (
                    compute_dynamic_fp8_scale[out_dtype](row_max, _scale_ub)
                )

                # Write scale to local scratch.
                if tid == 0:
                    scratch_scale.address_space_cast[
                        _target_address_space
                    ]().store(local_row, scale_factor)

                # Quantize and write fp8 to local scratch.
                if is_valid:
                    var output_fp8 = fp8_quantize[out_dtype](
                        normalized, scale_factor_recip
                    )
                    var local_elem = local_row * _cols + col_idx
                    scratch_fp8.address_space_cast[
                        _target_address_space
                    ]().store[width=simd_width, alignment=simd_width](
                        local_elem, output_fp8
                    )
            else:
                # No quant: write normalized value (out_dtype == in_dtype) to
                # local scratch. No scale is computed or written.
                if is_valid:
                    var local_elem = local_row * _cols + col_idx
                    scratch_fp8.address_space_cast[
                        _target_address_space
                    ]().store[width=simd_width, alignment=simd_width](
                        local_elem, normalized.cast[out_dtype]()
                    )

    # Per-block barrier with fence: block B waits for block B on all GPUs.
    # `syncthreads` plus the release/acquire atomics inside `_multi_gpu_barrier`
    # ensure all of block B's Stage 1 writes on every GPU are visible before
    # Stage 2 reads them.
    _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
        rank_sigs, rank_sigs[_my_rank], _my_rank
    )

    # === Stage 2: All-Gather (lightweight P2P copy, no compute) ===
    # Batch reads by GPU: consecutive reads from the same peer improve
    # NVLink pipelining. No runtime owner_rank division per row.
    comptime for gpu in range(ngpus):
        var gpu_row_start = gpu * rows_per_rank
        var gpu_row_end = min(gpu_row_start + rows_per_rank, _rows)
        var gpu_rows = gpu_row_end - gpu_row_start
        for local_row in range(block_idx.x, gpu_rows, num_blocks):
            var row = gpu_row_start + local_row
            var local_elem = local_row * _cols + col_idx

            # Read output value (fp8 when quantizing, else out_dtype) from
            # gpu's scratch and write to output.
            if is_valid:
                var out_val = (
                    fp8_ptrs[gpu]
                    .address_space_cast[_target_address_space]()
                    .load[
                        width=simd_width,
                        alignment=simd_width,
                        invariant=True,
                    ](local_elem)
                )
                output_fn[simd_width](row, col_idx, out_val)

            # Thread 0: read scale from gpu's scratch → scale_buffer (FP8 only).
            comptime if quantize:
                if tid == 0:
                    scale_buffer[row] = (
                        scale_ptrs[gpu]
                        .address_space_cast[_target_address_space]()
                        .load[invariant=True](local_row)
                    )

            # Read bf16 residual from gpu's scratch (compile-time gated).
            comptime if has_residual:
                if is_valid:
                    var bf16_val = (
                        residual_ptrs[gpu]
                        .address_space_cast[_target_address_space]()
                        .load[
                            width=simd_width,
                            alignment=simd_width,
                            invariant=True,
                        ](local_elem)
                    )
                    residual_output[].store[width=simd_width](
                        Coord(row, col_idx), bf16_val
                    )

    # NOTE: No end barrier needed (same reasoning as 1-stage kernel).


# --- Launcher ---

comptime _ZeroSizedLayout = type_of(row_major(Coord(Idx[0], Idx[0])))


def _allreduce_rmsnorm_fp8_launch[
    simd_width: Int,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
    has_residual: Bool = False,
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    output: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
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
    residual_output: _ComptimeConditionalTileTensor[
        mut=True, in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        MutAnyOrigin,
        engaged=False,
    ](),
) raises:
    """Launch the fused allreduce + RMSNorm + FP8 kernel."""
    comptime sm_version = get_sm_version()
    var payload_bytes = rows * cols * size_of[in_dtype]()
    var max_blocks = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](payload_bytes).get_num_blocks()
    var grid_dim = min(rows, max_blocks)
    var block_dim = threads_per_block

    # Provide evidence that flat_rank >= 2 for output.store(Coord(..., ...)).
    comptime assert output.flat_rank >= 2

    @always_inline
    @__parameter
    @__copy_capture(output)
    def output_fn[
        width: SIMDLength
    ](row: Int, col: Int, val: SIMD[out_dtype, width]):
        output.store[width=width](Coord(row, col), val)

    comptime kernel = _allreduce_rmsnorm_fp8_kernel_warp_tiling[
        mut=gamma.mut,
        origin=gamma.origin,
        LayoutType=gamma.LayoutType,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scales_dtype=scales_dtype,
        scale_origin=scale_output.origin,
        ScaleLayoutType=scale_output.LayoutType,
        ResidualLayoutType=residual.T.LayoutType,
        residual_origin=residual.T.origin,
        ResidualOutLayoutType=residual_output.T.LayoutType,
        residual_out_origin=residual_output.T.origin,
        ngpus=ngpus,
        simd_width=simd_width,
        threads_per_block=threads_per_block,
        has_residual=has_residual,
        output_fn=output_fn,
    ]
    ctx.enqueue_function[kernel](
        src_ptrs,
        gamma,
        scale_output,
        epsilon.cast[.float32](),
        weight_offset,
        Int32(rows),
        Int32(cols),
        Float32(scale_ub),
        rank_sigs,
        Int32(my_rank),
        residual,
        residual_output,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )


def _allreduce_rmsnorm_fp8_launch_2stage[
    simd_width: Int,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
    has_residual: Bool = False,
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    output: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
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
    residual_output: _ComptimeConditionalTileTensor[
        mut=True, in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        MutAnyOrigin,
        engaged=False,
    ](),
) raises:
    """Launch the single-kernel 2-stage fused RS + RMSNorm + FP8 + AG.

    A single kernel with two stages separated by a per-block barrier:
      Stage 1 (RS + RMSNorm + FP8): each block reduces rows in its partition
        from all GPUs in f32 registers, normalizes, quantizes to FP8, and
        writes compact results (fp8 + scale + optional bf16) to local scratch.
      Stage 2 (AG copy): each block iterates over each GPU's partition and
        reads compact fp8/scale/bf16 via P2P, then writes to local output.

    Grid is sized to rows_per_rank (not total rows) so every block has
    work in Stage 1. Both stages use the same local_row-to-block mapping,
    so the per-block barrier correctly synchronizes all data dependencies.
    """
    # Cap grid to rows_per_rank so every block has work in Stage 1.
    # With `rows` blocks, 1-1/ngpus of them would be idle during the
    # RS+Norm+FP8 stage (e.g. 75% idle at 4 GPUs). Using rows_per_rank
    # ensures 100% block utilization in both stages.
    comptime sm_version = get_sm_version()
    var payload_bytes = rows * cols * size_of[in_dtype]()
    var max_blocks = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](payload_bytes).get_num_blocks()
    var grid_dim = min(ceildiv(rows, ngpus), max_blocks)
    var block_dim = threads_per_block

    # Fuse FP8 quantization only when the output dtype differs from the input.
    comptime quantize = out_dtype != in_dtype

    # Validate that the signal buffer scratch layout is correctly aligned.
    # The 2-stage kernel places the output data immediately after the Signal
    # header (fp8 when quantizing, else out_dtype == in_dtype). When
    # quantizing, the scale values (padded to simd_width bytes) follow; with no
    # quantization there is no scale section. The residual (if has_residual)
    # comes last. For scale values to be properly aligned, the output block
    # size must be a multiple of sizeof(scales_dtype); the scale section is
    # padded to simd_width bytes so the residual section is simd_width-byte
    # aligned.
    # Callers must ensure each rank_sigs[i] buffer has at least:
    #   sizeof(Signal) + output_scratch [+ scale_scratch_padded if quantizing]
    # (plus residual_scratch = rows_per_rank * cols * sizeof(in_dtype) if
    # has_residual). See the kernel docstring for the full formula.
    var _rows_per_rank = ceildiv(rows, ngpus)
    var _out_scratch = _rows_per_rank * cols * size_of[out_dtype]()
    var _scale_scratch = 0
    comptime if quantize:
        var _scale_scratch_raw = _rows_per_rank * size_of[scales_dtype]()
        # Pad scale section to simd_width bytes (matches scale_pad_elements).
        _scale_scratch = align_up(_scale_scratch_raw, simd_width)
    var _min_buf = size_of[Signal]() + _out_scratch + _scale_scratch
    comptime if has_residual:
        _min_buf += _rows_per_rank * cols * size_of[in_dtype]()
    comptime if quantize:
        assert _out_scratch % size_of[scales_dtype]() == 0, (
            String("2-stage: output scratch (")
            + String(_out_scratch)
            + " B) must be a multiple of sizeof(scales_dtype) for scale pointer"
            + " alignment; rank_sigs[i] must be >= "
            + String(_min_buf)
            + " bytes"
        )
    comptime if has_residual:
        assert (_out_scratch + _scale_scratch) % simd_width == 0, (
            String("2-stage: residual scratch offset (output=")
            + String(_out_scratch)
            + " B + scales_padded="
            + String(_scale_scratch)
            + " B = "
            + String(_out_scratch + _scale_scratch)
            + " B) must be a multiple of simd_width ("
            + String(simd_width)
            + " B) for SIMD residual stores"
        )

    # Provide evidence that flat_rank >= 2 for output.store(Coord(..., ...)).
    comptime assert output.flat_rank >= 2

    @always_inline
    @__parameter
    @__copy_capture(output)
    def output_fn[
        width: SIMDLength
    ](row: Int, col: Int, val: SIMD[out_dtype, width]):
        output.store[width=width](Coord(row, col), val)

    comptime kernel = _allreduce_rmsnorm_fp8_kernel_2stage[
        mut=gamma.mut,
        origin=gamma.origin,
        LayoutType=gamma.LayoutType,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scales_dtype=scales_dtype,
        scale_origin=scale_output.origin,
        ScaleLayoutType=scale_output.LayoutType,
        ResidualLayoutType=residual.T.LayoutType,
        residual_origin=residual.T.origin,
        ResidualOutLayoutType=residual_output.T.LayoutType,
        residual_out_origin=residual_output.T.origin,
        ngpus=ngpus,
        simd_width=simd_width,
        threads_per_block=threads_per_block,
        has_residual=has_residual,
        output_fn=output_fn,
    ]
    ctx.enqueue_function[kernel](
        src_ptrs,
        gamma,
        scale_output,
        epsilon.cast[.float32](),
        weight_offset,
        Int32(rows),
        Int32(cols),
        Float32(scale_ub),
        rank_sigs,
        Int32(my_rank),
        residual,
        residual_output,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )


# --- Split (2-kernel) path ---


def _launch_split_allreduce_rmsnorm_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    output_2d: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output_1d: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    residual: TileTensor[mut=False, in_dtype, ...],
    residual_output: TileTensor[mut=True, in_dtype, ...],
) raises:
    """Two-kernel fallback: allreduce+add epilogue, then rmsnorm+fp8.

    Faster than fused 2-stage for large with-residual payloads because
    it avoids carrying bf16 residual data through scratch buffers.
    """
    # Construct TileTensor inputs for allreduce.
    comptime _TT = type_of(
        TileTensor(src_ptrs[0], row_major(Coord(rows, cols)))
    )
    var input_buffers = Array[_, ngpus](
        fill_with=lambda (i: Int) -> _TT: TileTensor(
            src_ptrs[i], row_major(Coord(rows, cols))
        )
    )

    var _cols = cols

    # Define input_fn for RMSNorm (reads from residual_output after allreduce).
    @__copy_capture(residual_output, _cols)
    @always_inline
    @__parameter
    def input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var li = idx[0] * _cols + idx[1]
        return residual_output.raw_load[width=width, alignment=width](li)

    var shape = IndexList[2](rows, cols)
    var scale_output_2d = TileTensor(
        scale_output_1d._storage, row_major(Coord(rows, Idx[1]))
    )

    # Pre-compile the RMSNorm+FP8 kernel before launching allreduce.
    # This avoids a deadlock where cuModuleLoadDataEx (JIT compilation)
    # blocks on the CUDA context while the allreduce kernel is waiting
    # at a multi-GPU barrier for other ranks to be enqueued.
    rms_norm_fused_fp8[
        in_dtype, out_dtype, scales_dtype, 2, input_fn, compile_only=True
    ](
        shape,
        output_2d,
        gamma,
        epsilon,
        weight_offset,
        ctx,
        scale_ub,
        scale_output_2d,
    )

    # Step 1: Allreduce with add epilogue → residual_output.
    @__copy_capture(residual, residual_output, _cols)
    @always_inline
    @__parameter
    def add_epilogue[
        _dtype: DType,
        _width: SIMDLength,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, length=_width]) -> None:
        var il = coord_to_index_list(coords)
        var flat_idx = il[0] * _cols + il[1]
        var res = residual.raw_load[width=_width, alignment=_alignment](
            flat_idx
        )
        # Add in f32 for precision parity with the fused kernel,
        # which accumulates allreduce + residual in f32 before casting.
        var sum_f32 = (
            val.cast[.float32]()
            + rebind[SIMD[_dtype, _width]](res).cast[.float32]()
        )
        residual_output.raw_store[width=_width, alignment=_alignment](
            flat_idx,
            rebind[SIMD[in_dtype, _width]](sum_f32.cast[_dtype]()),
        )

    allreduce[
        ngpus=ngpus,
        output_lambda=Optional[elementwise_epilogue_type](add_epilogue),
    ](input_buffers, residual_output, rank_sigs, ctx)

    # Step 2: Fused RMSNorm + FP8 on residual_output (kernel already compiled).
    rms_norm_fused_fp8[in_dtype, out_dtype, scales_dtype, 2, input_fn](
        shape,
        output_2d,
        gamma,
        epsilon,
        weight_offset,
        ctx,
        scale_ub,
        scale_output_2d,
    )


# --- Dispatch ---


def _dispatch_fused_kernel[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    has_residual: Bool = False,
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    output_2d: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output_1d: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    residual: _ComptimeConditionalTileTensor[
        in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        ImmutAnyOrigin,
        engaged=False,
    ](),
    residual_output: _ComptimeConditionalTileTensor[
        mut=True, in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        MutAnyOrigin,
        engaged=False,
    ](),
) raises:
    """Dispatch the fused kernel with appropriate simd width and stage count.

    Centralizes the dispatch logic shared by allreduce_rmsnorm and
    allreduce_residual_rmsnorm. Selects simd width based on column count
    and 1-stage vs 2-stage based on payload size.
    """
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime threads_per_block = max_warps_per_block * WARP_SIZE
    comptime base_simd_width = simd_width_of[
        in_dtype, target=get_gpu_target()
    ]()
    comptime sw1 = base_simd_width
    comptime sw2 = base_simd_width * 2

    # Fuse FP8 quantization only when the output dtype differs from the input.
    # The split (2-kernel) path is FP8-only: it relies on the compact FP8
    # all-gather and on `rms_norm_fused_fp8`'s `compile_only` JIT-deadlock
    # guard, neither of which has a non-quant analog. So when not quantizing
    # the dispatch only ever picks the 1-stage or 2-stage fused kernels.
    comptime quantize = out_dtype != in_dtype

    # Use 2-stage (reduce-scatter + fused all-gather) for large payloads.
    # Threshold is per-rank payload: each GPU processes ceildiv(rows, ngpus)
    # full rows, so per-rank bytes stays roughly constant across column widths.
    var per_rank_bytes = ceildiv(rows, ngpus) * cols * size_of[in_dtype]()

    # Override for threshold tuning experiments:
    #   0 = use default threshold (production)
    #   1 = force 1-stage for all sizes
    #   2 = force 2-stage for all sizes
    #   3 = force split (2-kernel) path for all sizes (residual only)
    comptime force_stage = get_defined_int["FORCE_ALLREDUCE_STAGE", 0]()

    # Per-rank byte thresholds for switching to 2-stage allreduce.
    # Derived from empirical sweep data (bf16, per-rank crossover points):
    #   MI355 4GPU: 128 KB non-res,  96 KB residual
    #   MI355 8GPU:  80 KB non-res,  96 KB residual
    #   B200  4GPU: 512 KB non-res, 256 KB residual
    #   B200  8GPU:  80 KB non-res, 80/100 KB residual (column-aware, see below)
    @__parameter
    def _rank_4_per_rank_thresh() -> Int:
        comptime if has_amd_gpu_accelerator():
            return 128 * 1024 if not has_residual else 96 * 1024
        else:
            return 512 * 1024 if not has_residual else 256 * 1024

    @__parameter
    def _rank_8_per_rank_thresh() -> Int:
        comptime if has_amd_gpu_accelerator():
            return 80 * 1024 if not has_residual else 96 * 1024
        else:
            return 80 * 1024

    # B200 8-GPU residual: the 1-stage/2-stage crossover is NOT a constant
    # per-rank byte count -- it rises with column width because the 1-stage
    # cost is dominated by a fixed per-launch floor plus a row-count term, so
    # for the same per-rank bytes a wider row (fewer rows) keeps 1-stage ahead
    # longer. Measured crossovers (B200 8xGPU, bf16, residual):
    #   cols 4096 -> ~72 KB ; cols 7168 -> ~104 KB ; cols 8192 -> ~104 KB.
    # A flat 80 KB mis-routes the wide-column (Kimi hidden=7168) M=40..56 band
    # onto the slower 2-stage path (e.g. M=48: 2-stage 41.2us vs 1-stage
    # 36.3us, a 12% loss). Use a column-aware threshold: 100 KB for wide
    # columns (>= 6144, the large-hidden regime), 80 KB otherwise (matches the
    # validated narrow-column crossover; cols=4096 crosses near 72 KB).
    @__parameter
    def _rank_8_residual_thresh_for_cols(c: Int) -> Int:
        comptime if has_amd_gpu_accelerator():
            return _rank_8_per_rank_thresh()
        else:
            return 100 * 1024 if c >= 6144 else 80 * 1024

    var threshold: Int
    comptime if ngpus <= 4:
        threshold = _rank_4_per_rank_thresh()
    elif has_residual and not has_amd_gpu_accelerator():
        threshold = _rank_8_residual_thresh_for_cols(cols)
    else:
        threshold = _rank_8_per_rank_thresh()

    # Per-rank byte thresholds for switching to split (2-kernel) path
    # when has_residual=True. Above this, allreduce+add + rmsnorm_fp8 beats
    # fused 2-stage because it avoids carrying bf16 residual through scratch.
    #   MI355: column-dependent; 2-stage beats split for cols <= 8192,
    #          split beats 2-stage for cols > 8192. See dispatch below.
    #   B200 4GPU: 1536 KB per-rank crossover
    #   B200 8GPU: conservative, same as 2-stage threshold
    @__parameter
    def _rank_4_split_thresh() -> Int:
        comptime if has_amd_gpu_accelerator():
            return _rank_4_per_rank_thresh()
        else:
            return 1536 * 1024

    @__parameter
    def _rank_8_split_thresh() -> Int:
        # For 8 GPUs the split threshold equals the 2-stage threshold on
        # both AMD and NVIDIA.  This intentionally means the 2-stage
        # *residual* path is never selected at 8 GPUs: once per-rank
        # payload crosses threshold the dispatch jumps straight to split,
        # skipping the `elif per_rank_bytes >= threshold` arm below.
        # Benchmarks show the split (2-kernel) path is always faster than
        # fused 2-stage with residual at 8-GPU scale because the extra
        # bf16 scratch traffic in Stage 2 outweighs the launch overhead.
        comptime if has_amd_gpu_accelerator():
            return _rank_8_per_rank_thresh()
        else:
            return _rank_8_per_rank_thresh()

    comptime split_threshold = _rank_4_split_thresh() if ngpus <= 4 else _rank_8_split_thresh()

    var use_2stage: Bool
    var use_split: Bool = False
    comptime if force_stage == 1:
        use_2stage = False
    elif force_stage == 2:
        use_2stage = True
    elif force_stage == 3:
        # Force split (2-kernel) path (FP8/quantizing residual only).
        use_2stage = False
        comptime if has_residual and quantize:
            use_split = True
    else:
        comptime if has_residual and quantize:
            comptime if has_amd_gpu_accelerator():
                # MI355: 2-stage fused beats split for cols <= 8192,
                # split beats 2-stage for wider columns (16384+).
                if per_rank_bytes >= threshold:
                    if cols > 8192:
                        use_split = True
                        use_2stage = False
                    else:
                        use_2stage = True
                else:
                    use_2stage = False
            else:
                # NVIDIA residual dispatch.  For 8 GPUs split_threshold
                # == threshold, so the elif is unreachable and the
                # 2-stage residual path is intentionally skipped (see
                # _rank_8_split_thresh comment).  For 4 GPUs the two
                # thresholds differ and all three paths are reachable.
                if per_rank_bytes >= split_threshold:
                    use_split = True
                    use_2stage = False
                elif per_rank_bytes >= threshold:
                    use_2stage = True
                else:
                    use_2stage = False
        elif has_residual:
            # Non-quant residual: the split path is FP8-only, so pick between
            # the 1-stage and 2-stage fused kernels by per-rank payload only.
            use_2stage = ngpus <= 8 and per_rank_bytes >= threshold
        else:
            use_2stage = ngpus <= 8 and per_rank_bytes >= threshold

    @__parameter
    def launch_1stage[sw: Int]() raises:
        _allreduce_rmsnorm_fp8_launch[
            sw,
            in_dtype,
            out_dtype,
            scales_dtype,
            ngpus,
            threads_per_block,
            has_residual=has_residual,
        ](
            rows,
            cols,
            src_ptrs,
            output_2d,
            gamma,
            epsilon,
            weight_offset,
            scale_ub,
            scale_output_1d,
            rank_sigs,
            Int(ctx.id()),
            ctx,
            residual,
            residual_output,
        )

    @__parameter
    def launch_2stage[sw: Int]() raises:
        _allreduce_rmsnorm_fp8_launch_2stage[
            sw,
            in_dtype,
            out_dtype,
            scales_dtype,
            ngpus,
            threads_per_block,
            has_residual=has_residual,
        ](
            rows,
            cols,
            src_ptrs,
            output_2d,
            gamma,
            epsilon,
            weight_offset,
            scale_ub,
            scale_output_1d,
            rank_sigs,
            Int(ctx.id()),
            ctx,
            residual,
            residual_output,
        )

    if use_split:
        # use_split is only ever set when has_residual and quantize, and the
        # comptime guard keeps the FP8-only split launcher from being
        # instantiated for the non-quant case.
        comptime if has_residual and quantize:
            _launch_split_allreduce_rmsnorm_fp8[
                in_dtype, out_dtype, scales_dtype, ngpus
            ](
                rows,
                cols,
                src_ptrs,
                output_2d,
                gamma,
                epsilon,
                weight_offset,
                scale_ub,
                scale_output_1d,
                rank_sigs,
                ctx,
                residual[],
                residual_output[],
            )
        return

    # Warp-tiling: each thread handles simd_width elements. Wider simd for
    # larger column counts.
    if cols <= (WARP_SIZE * sw1 * max_warps_per_block):
        if use_2stage:
            launch_2stage[sw1]()
        else:
            launch_1stage[sw1]()
    elif cols <= (WARP_SIZE * sw2 * max_warps_per_block) and cols % sw2 == 0:
        if use_2stage:
            launch_2stage[sw2]()
        else:
            launch_1stage[sw2]()
    else:
        comptime max_cols = WARP_SIZE * sw2 * max_warps_per_block
        raise Error(
            "allreduce_rmsnorm: cols ("
            + String(cols)
            + ") exceeds max supported ("
            + String(max_cols)
            + ") for warp-tiling kernel"
        )


# --- Public API ---


def allreduce_rmsnorm[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
](
    input_buffers: Array[TileTensor[in_dtype, in_layout, in_origin], ngpus],
    output: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
) raises:
    """Fused allreduce + RMSNorm with optional FP8 quantization.

    Combines a P2P allreduce across GPUs, RMSNorm, and (when the output dtype
    differs from the input dtype) FP8 dynamic quantization into a single
    kernel launch, eliminating the global memory round-trip between allreduce
    output and RMSNorm input. When `out_dtype == in_dtype` the quantization is
    skipped: the normalized value is written directly in the input dtype and
    `scale_ub` and `scale_output` are ignored.

    Parameters:
        in_dtype: Input data type (e.g. bfloat16).
        out_dtype: Output data type. Either a float8 type (fuses
            quantization) or equal to `in_dtype` (no quantization).
        scales_dtype: Scale factor data type (e.g. float32). Ignored when
            `out_dtype == in_dtype`.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.

    Args:
        input_buffers: Per-GPU input buffers as TileTensors.
        output: Output buffer (FP8 values when quantizing, else `in_dtype`).
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        scale_ub: Upper bound for FP8 scale clamping (ignored when not
            quantizing).
        scale_output: Output buffer for per-row FP8 scales (ignored, and not
            written, when not quantizing).
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.

    Note:
        This kernel does not issue an end barrier. The output and scale
        buffers are safe to read only on the local GPU (stream ordering
        guarantees visibility). If a remote GPU needs to read these outputs,
        the caller must insert an explicit barrier. The start barrier of the
        NEXT allreduce call protects the input buffers that are read by remote
        GPUs.

        Signal buffer sizing:
          1-stage path (payload < threshold): size_of[Signal]() only.
          2-stage path (payload > threshold):
            size_of[Signal]()
            + ceildiv(rows, ngpus) * cols * sizeof(out_dtype)          (output)
            + align_up(ceildiv(rows, ngpus) * sizeof(scales_dtype),
                       simd_width)                          (scales + pad, if
                                                             quantizing)
    """
    comptime assert ngpus >= 2, "allreduce_rmsnorm requires at least 2 GPUs"
    comptime assert (
        in_dtype.is_floating_point()
    ), "in_dtype must be floating point"
    comptime assert (out_dtype == in_dtype) or out_dtype.is_float8(), (
        "out_dtype must be a float8 type (fuse quantization) or equal to"
        " in_dtype (skip quantization)"
    )

    if not is_p2p_enabled():
        raise Error("allreduce_rmsnorm requires P2P access between GPUs")

    # Compute rows/cols from TileTensor dims.
    var in_num_elems = input_buffers[0].num_elements()
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    var rows = in_num_elems // cols

    # Extract raw pointers from TileTensors.
    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var src_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> PtrType: rebind[
            ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
        ](input_buffers[i]._storage)
    )

    # Create internal 2D/1D TileTensor views for _dispatch_fused_kernel.
    var output_2d = TileTensor(
        rebind[MutPointer[Scalar[out_dtype], MutAnyOrigin]](output._storage),
        row_major(Coord(rows, cols)),
    )
    var scale_output_1d = TileTensor(
        rebind[MutPointer[Scalar[scales_dtype], MutAnyOrigin]](
            scale_output._storage
        ),
        row_major(
            Coord(
                rows,
            )
        ),
    )

    _dispatch_fused_kernel[in_dtype, out_dtype, scales_dtype, ngpus](
        rows,
        cols,
        src_ptrs,
        output_2d,
        gamma,
        epsilon,
        weight_offset,
        scale_ub,
        scale_output_1d,
        rank_sigs,
        ctx,
    )


def allreduce_residual_rmsnorm[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
](
    input_buffers: Array[TileTensor[in_dtype, in_layout, in_origin], ngpus],
    residual: TileTensor[mut=False, in_dtype, ...],
    output: TileTensor[mut=True, out_dtype, ...],
    residual_output: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
) raises:
    """Fused allreduce + residual add + RMSNorm with optional FP8 quantization.

    Combines a P2P allreduce, a residual add, RMSNorm, and (when the output
    dtype differs from the input dtype) FP8 dynamic quantization into a fused
    kernel. When `out_dtype == in_dtype` the quantization is skipped: the
    normalized value is written directly in the input dtype and `scale_ub` and
    `scale_output` are ignored (the per-row scale is neither computed nor
    written).

    Parameters:
        in_dtype: Input data type (e.g. bfloat16).
        out_dtype: Output data type. Either a float8 type (fuses
            quantization) or equal to `in_dtype` (no quantization).
        scales_dtype: Scale factor data type (e.g. float32). Ignored when
            `out_dtype == in_dtype`.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.

    Args:
        input_buffers: Per-GPU input buffers as TileTensors.
        residual: Residual buffer as a TileTensor.
        output: Output buffer (FP8 values when quantizing, else `in_dtype`).
        residual_output: Output buffer for pre-norm sum as a TileTensor.
        gamma: RMSNorm gamma weights (1D TileTensor).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        scale_ub: Upper bound for FP8 scale clamping (ignored when not
            quantizing).
        scale_output: Output buffer for per-row FP8 scales (ignored, and not
            written, when not quantizing).
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
    """
    comptime assert (
        in_dtype.is_floating_point()
    ), "in_dtype must be floating point"
    comptime assert (out_dtype == in_dtype) or out_dtype.is_float8(), (
        "out_dtype must be a float8 type (fuse quantization) or equal to"
        " in_dtype (skip quantization)"
    )

    if not is_p2p_enabled():
        raise Error(
            "allreduce_residual_rmsnorm requires P2P access between GPUs"
        )

    # Compute rows/cols from TileTensor dims. The internal dispatch flattens
    # to 2D, so we only need the last dim for cols and total/cols for rows.
    var in_num_elems = input_buffers[0].num_elements()
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    var rows = in_num_elems // cols

    # Extract raw pointers from TileTensors.
    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var src_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> PtrType: rebind[
            ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
        ](input_buffers[i]._storage)
    )

    # Create internal 2D/1D TileTensor views for _dispatch_fused_kernel.
    var output_2d = TileTensor(
        rebind[MutPointer[Scalar[out_dtype], MutAnyOrigin]](output._storage),
        row_major(Coord(rows, cols)),
    )
    var residual_2d = TileTensor(
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](residual._storage),
        row_major(Coord(rows, cols)),
    )
    var residual_output_2d = TileTensor(
        rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](
            residual_output._storage
        ),
        row_major(Coord(rows, cols)),
    )
    var scale_output_1d = TileTensor(
        rebind[MutPointer[Scalar[scales_dtype], MutAnyOrigin]](
            scale_output._storage
        ),
        row_major(
            Coord(
                rows,
            )
        ),
    )

    _dispatch_fused_kernel[
        in_dtype, out_dtype, scales_dtype, ngpus, has_residual=True
    ](
        rows,
        cols,
        src_ptrs,
        output_2d,
        gamma,
        epsilon,
        weight_offset,
        scale_ub,
        scale_output_1d,
        rank_sigs,
        ctx,
        residual_2d,
        residual_output_2d,
    )
