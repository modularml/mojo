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
"""
MLA Decode Split-K Combine Kernel for SM100 (B200).

This kernel combines partial outputs from split-K attention computation.
Each split computes attention over a portion of the KV cache. The combine
kernel merges these partial results using LSE (Log-Sum-Exp) for numerical
stability.

Algorithm:
1. Load partial LSE values for all splits
2. Compute global LSE: log2(sum(exp2(lse_i - max_lse))) + max_lse
3. Compute per-split scale factors: scale_i = exp2(lse_i - global_lse)
4. Weighted sum: output = sum(scale_i * partial_output_i)

"""

from std.collections import OptionalReg
from std.math import ceildiv, exp2, log2, max, min
from std.math.constants import log2e

import max.gpu.primitives.warp as warp
from max.gpu import WARP_SIZE, block_idx, lane_id, warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    wait_on_dependent_grids,
    pdl_launch_attributes,
)
from layout import TileTensor
from std.memory import unsafe_stack_allocation
from std.sys import get_defined_bool
from std.utils.numerics import min_or_neg_inf
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder


# The kernels below already wait on dependent grids and the decode producer
# already triggers them, but neither has an effect unless the consumer launch
# opts in. Opting in makes that whole contract live for the first time, and a
# rare illegal-address abort at context teardown tracks with it, so the level
# stays off until the mechanism is settled. Enabling it is a source edit: the
# define cannot be set through the build, because the package that reads it is
# precompiled and `mojo precompile` takes no `-D`.
comptime MLA_DECODE_COMBINE_PDL_LEVEL = PDLLevel.OVERLAP_AT_END if get_defined_bool[
    "MLA_DECODE_COMBINE_PDL", False
]() else PDLLevel.OFF


# ===----------------------------------------------------------------------=== #
# Combine Kernel Parameters
# ===----------------------------------------------------------------------=== #
struct CombineParams[
    output_type: DType,
    accum_type: DType,
    num_splits: Int,
    ragged: Bool = False,
    warps_per_head: Int = 2,
    has_attn_sink: Bool = False,
](Copyable, DevicePassable, TrivialRegisterPassable):
    """Holds the pointers, strides, and shape metadata for the main split-K combine kernel.

    Carries the per-split partial output and LSE accumulators, the final output
    tensor, optional ragged row offsets and attention-sink pointer, and the
    derived strides used by `mla_combine_kernel` to index the split, batch,
    sequence, and head dimensions.

    Parameters:
        output_type: The element type of the per-split partial output
            accumulator and the final combined output tensor.
        accum_type: The element type of the per-split LSE accumulator values.
        num_splits: The compile-time number of KV cache splits to combine.
            Used to unroll the per-split accumulation loop.
        ragged: Whether batches have variable-length sequences in a packed
            layout (defaults to `False`). When true,
            `input_row_offsets_ptr` selects each batch's token range.
        warps_per_head: Number of warps assigned to each attention head
            (defaults to 2). Must divide 8; controls vector load width and
            threads per block.
        has_attn_sink: Whether to apply attention-sink correction to the
            global LSE (defaults to `False`). When true, `attn_sink_ptr`
            must be provided.
    """

    # Invariant: warps_per_head must divide 8 (checked in mla_combine_kernel).
    comptime heads_per_block = 8 // Self.warps_per_head
    comptime num_threads = Self.heads_per_block * Self.warps_per_head * WARP_SIZE

    @__allow_legacy_any_origin_fields
    var out_accum_split_ptr: UnsafePointer[
        Scalar[Self.output_type], origin=MutAnyOrigin
    ]

    @__allow_legacy_any_origin_fields
    var lse_accum_split_ptr: UnsafePointer[
        Scalar[Self.accum_type], origin=MutAnyOrigin
    ]

    # Final output tensor
    @__allow_legacy_any_origin_fields
    var output_ptr: UnsafePointer[Scalar[Self.output_type], origin=MutAnyOrigin]

    # Input row offsets for ragged mode (cumulative token counts per batch)
    # In ragged mode: input_row_offsets[i] = start token index for batch i
    @__allow_legacy_any_origin_fields
    var input_row_offsets_ptr: UnsafePointer[UInt32, origin=MutAnyOrigin]

    # Per-head attn_sink values: shape [num_heads_q], float32, nullable.
    # Contains log-sum-exp of non-selected tokens' attention scores (natural log).
    # Only used when has_attn_sink is True at compile time.
    @__allow_legacy_any_origin_fields
    var attn_sink_ptr: OptionalReg[UnsafePointer[Float32, origin=MutAnyOrigin]]
    var batch_size: Int
    var seq_len: Int
    var num_heads: Int
    var head_dim: Int

    var lse_stride_split: Int
    var lse_stride_batch: Int
    var lse_stride_seq: Int

    var out_accum_stride_split: Int
    var out_accum_stride_head: Int

    var out_stride_row: Int

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "CombineParams"

    @staticmethod
    def get_device_type_name() -> String:
        return "CombineParams"

    def __init__(
        out self,
        out_accum_split_ptr: UnsafePointer[
            Scalar[Self.output_type], origin=MutAnyOrigin
        ],
        lse_accum_split_ptr: UnsafePointer[
            Scalar[Self.accum_type], origin=MutAnyOrigin
        ],
        output_ptr: UnsafePointer[
            Scalar[Self.output_type], origin=MutAnyOrigin
        ],
        input_row_offsets_ptr: UnsafePointer[UInt32, origin=MutAnyOrigin],
        attn_sink_ptr: OptionalReg[UnsafePointer[Float32, origin=MutAnyOrigin]],
        batch_size: Int,
        seq_len: Int,
        num_heads: Int,
        head_dim: Int,
    ):
        self.out_accum_split_ptr = out_accum_split_ptr
        self.lse_accum_split_ptr = lse_accum_split_ptr
        self.output_ptr = output_ptr
        self.input_row_offsets_ptr = input_row_offsets_ptr
        self.attn_sink_ptr = attn_sink_ptr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.lse_stride_split = batch_size * seq_len * num_heads
        self.lse_stride_batch = seq_len * num_heads
        self.lse_stride_seq = num_heads

        self.out_accum_stride_split = (
            batch_size * seq_len * num_heads * head_dim
        )
        self.out_accum_stride_head = head_dim

        self.out_stride_row = head_dim


# ===----------------------------------------------------------------------=== #
# Main Combine Kernel - Optimized version matching FlashMLA pattern
# ===----------------------------------------------------------------------=== #
@__name(
    t"sm100_mla_decode_combine_{output_type}_{accum_type}_{num_splits}_{ragged}",
)
def mla_combine_kernel[
    output_type: DType,
    accum_type: DType,
    head_dim: Int,
    num_splits: Int,
    ragged: Bool = False,
    warps_per_head: Int = 2,
    has_attn_sink: Bool = False,
](
    params: CombineParams[
        output_type,
        accum_type,
        num_splits,
        ragged,
        warps_per_head,
        has_attn_sink,
    ]
):
    """Combines partial split-K attention outputs into a single decoded result using LSE-based weighting.

    Loads each split's partial output and LSE value, computes a global LSE via
    warp reduction, derives per-split scale factors as `exp2(lse_i - global_lse)`,
    and accumulates the weighted sum of partial outputs into the final output
    tensor. Supports ragged batch layouts and optional attention-sink correction.

    Parameters:
        output_type: The element type of the per-split partial output
            accumulator and the final combined output tensor.
        accum_type: The element type of the per-split LSE accumulator values.
        head_dim: The dimension of each attention head. Must be divisible by
            `WARP_SIZE * warps_per_head`.
        num_splits: The compile-time number of KV cache splits to combine.
            Used to unroll the per-split accumulation loop.
        ragged: Whether batches have variable-length sequences in a packed
            layout (defaults to `False`). When true,
            `input_row_offsets_ptr` selects each batch's token range.
        warps_per_head: Number of warps assigned to each attention head
            (defaults to 2). Must divide 8; controls vector load width and
            threads per block.
        has_attn_sink: Whether to apply attention-sink correction to the
            global LSE (defaults to `False`). When true, `attn_sink_ptr`
            must be provided.

    Args:
        params: The `CombineParams` instance carrying the per-split partial
            output and LSE accumulators, the final output tensor, optional
            ragged row offsets and attention-sink pointer, and the derived
            strides used to index the split, batch, sequence, and head
            dimensions.
    """
    # PDL: Wait for the MLA decode kernel (dependent kernel) to complete.
    wait_on_dependent_grids()

    comptime ParamsType = CombineParams[
        output_type,
        accum_type,
        num_splits,
        ragged,
        warps_per_head,
        has_attn_sink,
    ]
    comptime heads_per_block = ParamsType.heads_per_block
    comptime assert 8 % warps_per_head == 0, "warps_per_head must divide 8"

    var batch_idx = block_idx.x
    var seq_idx = block_idx.y
    var head_block_idx = block_idx.z
    var warp_idx = warp_id()
    var lane_idx = lane_id()

    # In ragged mode, each batch can have a different number of Q tokens.
    # The grid launches with seq_len = q_max_seq_len, so CTAs with
    # seq_idx >= this batch's actual seq_len must exit early to avoid
    # writing garbage to output locations belonging to other batches.
    comptime if ragged:
        var batch_seq_len = Int(
            params.input_row_offsets_ptr[batch_idx + 1]
        ) - Int(params.input_row_offsets_ptr[batch_idx])
        if seq_idx >= batch_seq_len:
            return

    var warp_idx_q, sub_warp_idx = divmod(warp_idx, warps_per_head)
    var head_idx = head_block_idx * heads_per_block + warp_idx_q

    if head_idx >= params.num_heads:
        return

    # =========================================================================
    # Step 1: Prefetch first split's data (SIMD vector loads)
    # =========================================================================
    # vec_size and elems_per_thread are derived from warps_per_head so the
    # user only needs to change warps_per_head (1, 2, 4, or 8) and the rest
    # auto-adjusts.
    #
    # Constraint: vec_size * elems_per_thread * WARP_SIZE * warps_per_head
    #             == head_dim
    # vec_size is capped at 8 (128-bit max load width for bf16).
    #
    # warps_per_head=1: max_vec=16, vec_size=8, elems_per_thread=2
    # warps_per_head=2: max_vec=8,  vec_size=8, elems_per_thread=1
    # warps_per_head=4: max_vec=4,  vec_size=4, elems_per_thread=1
    # warps_per_head=8: max_vec=2,  vec_size=2, elems_per_thread=1
    comptime assert (
        head_dim % (WARP_SIZE * warps_per_head) == 0
    ), "head_dim must be divisible by WARP_SIZE * warps_per_head"
    comptime max_vec = head_dim // (WARP_SIZE * warps_per_head)
    comptime vec_size = min(max_vec, 8)
    comptime elems_per_thread = head_dim // (
        WARP_SIZE * vec_size * warps_per_head
    )

    # Offset for this sub-warp's portion of head_dim.
    var head_dim_offset = sub_warp_idx * (head_dim // warps_per_head)

    # Base pointer for this head's partial output accumulator
    var out_row = (
        batch_idx * params.seq_len * params.num_heads
        + seq_idx * params.num_heads
        + head_idx
    )
    var oaccum_base = (
        params.out_accum_split_ptr + out_row * params.out_accum_stride_head
    ).as_imm()

    # Prefetch first split's data into registers
    var datas = Array[_, elems_per_thread](
        fill_with_unrolled=lambda [i: Int]() -> SIMD[
            output_type, vec_size
        ]: oaccum_base.load[width=vec_size](
            head_dim_offset + lane_idx * vec_size + i * (WARP_SIZE * vec_size)
        )
    )

    # =========================================================================
    # Step 2: Load LSE values and compute global LSE
    # =========================================================================
    # For >32 splits, each thread loads multiple LSE values (FlashMLA pattern).
    var lse_base = (
        batch_idx * params.lse_stride_batch
        + seq_idx * params.lse_stride_seq
        + head_idx
    )

    comptime num_lse_per_thread = ceildiv(num_splits, WARP_SIZE)

    # Load LSE values into registers (multiple per lane for >32 splits)
    # and track whether any split is empty (LSE=-inf) for the fast-path
    # check.
    var local_lse = Array[Float32, num_lse_per_thread](
        fill=min_or_neg_inf[.float32]()
    )

    comptime for k in range(num_lse_per_thread):
        comptime split_idx_base = k * WARP_SIZE
        var split_idx = split_idx_base + lane_idx
        if split_idx < num_splits:
            var lse_offset = split_idx * params.lse_stride_split + lse_base
            local_lse[k] = params.lse_accum_split_ptr[lse_offset].cast[
                DType.float32
            ]()

    # Thread-local max reduction first, then warp-level reduction
    var thread_max: Float32 = local_lse[0]

    comptime for k in range(1, num_lse_per_thread):
        thread_max = max(thread_max, local_lse[k])

    var max_lse = warp.max(thread_max)

    # set max_lse to 0 if all LSEs are -inf
    if max_lse == min_or_neg_inf[.float32]():
        max_lse = 0.0

    # Compute sum of exp2(lse - max_lse) with thread-local accumulation
    var thread_sum: Float32 = 0.0

    comptime for k in range(num_lse_per_thread):
        comptime split_idx_base = k * WARP_SIZE
        var split_idx = split_idx_base + lane_idx
        if split_idx < num_splits:
            thread_sum += exp2(local_lse[k] - max_lse)

    var sum_exp = warp.sum(thread_sum)

    # Compute global LSE
    var global_lse: Float32
    if sum_exp == 0.0:
        global_lse = Float32.MAX  # +inf placeholder
    else:
        global_lse = log2(sum_exp) + max_lse

    # Attn sink correction (split-K path): adjust global_lse to account for
    # non-selected tokens. This is deferred from the decode kernel to avoid
    # double-counting across splits.
    # FlashMLA reference: combine.cu:101-112
    comptime if has_attn_sink:
        var attn_sink_val = params.attn_sink_ptr.unsafe_value()[head_idx]
        var attn_sink_log2_val = attn_sink_val * Float32(log2e)
        if global_lse != Float32.MAX:
            # Normal case: merge attn_sink into the global LSE.
            # global_lse += log2(1 + exp2(attn_sink_log2 - global_lse))
            global_lse += log2(
                Float32(1.0) + exp2(attn_sink_log2_val - global_lse)
            )
        else:
            # No tokens attended (all splits empty): output depends on
            # attn_sink alone. If attn_sink is -inf, output is zero
            # (global_lse = +inf makes all scales 0).
            if attn_sink_val == min_or_neg_inf[.float32]():
                global_lse = Float32.MAX  # +inf => output = 0
            else:
                global_lse = attn_sink_log2_val

    # Compute scale factors in-place in local_lse registers (no shared memory).
    # Each lane already holds its split's LSE value; we overwrite with the
    # scale factor and broadcast via shuffle_idx in the accumulation loop.
    # No branch needed: lanes beyond num_splits have local_lse[k] == -inf,
    # and exp2(-inf - global_lse) = 0.0 naturally.
    comptime for k in range(num_lse_per_thread):
        local_lse[k] = exp2(local_lse[k] - global_lse)

    # =========================================================================
    # Step 3: Weighted accumulation with prefetching (compile-time unrolled)
    # =========================================================================
    var result = Array[SIMD[.float32, vec_size], elems_per_thread](
        fill=SIMD[.float32, vec_size](0.0)
    )

    comptime for split_idx in range(num_splits):
        # Broadcast scale from the owning lane via register shuffle (no smem).
        comptime k, src_lane = divmod(split_idx, WARP_SIZE)
        var lse_scale = warp.shuffle_idx(local_lse[k], UInt32(src_lane))
        var is_valid = SIMD[.bool, vec_size](fill=lse_scale != Float32(0))

        comptime for i in range(elems_per_thread):
            var data_f32 = datas[i].cast[.float32]()
            var clean_data = is_valid.select(
                data_f32,
                SIMD[.float32, vec_size](0),
            )
            result[i] = result[i] + lse_scale * clean_data

            comptime if split_idx < num_splits - 1:
                var next_offset = (
                    (split_idx + 1) * params.out_accum_stride_split
                    + head_dim_offset
                    + lane_idx * vec_size
                    + i * (WARP_SIZE * vec_size)
                )
                datas[i] = oaccum_base.load[width=vec_size](next_offset)

    # =========================================================================
    # Step 4: Write final result to output (convert to output_type)
    # =========================================================================
    # For ragged mode, the final output uses packed/ragged layout where each
    # batch's tokens are contiguous. Use input_row_offsets to compute the
    # correct output position. For non-ragged, use the same padded layout.
    var final_out_row: Int

    comptime if ragged:
        # Ragged output: start_of_seq * num_heads + seq_idx * num_heads + head_idx
        var start_of_seq = Int(params.input_row_offsets_ptr[batch_idx])
        final_out_row = (
            start_of_seq * params.num_heads
            + seq_idx * params.num_heads
            + head_idx
        )
    else:
        # Non-ragged: same padded layout as o_accum_split
        final_out_row = out_row

    var out_ptr = params.output_ptr + final_out_row * params.out_stride_row

    comptime for i in range(elems_per_thread):
        var offset = (
            head_dim_offset + lane_idx * vec_size + i * (WARP_SIZE * vec_size)
        )
        # Convert float32 result back to output_type (bf16) and store
        var out_data = result[i].cast[output_type]()
        out_ptr.store(offset, out_data)


# ===----------------------------------------------------------------------=== #
# Split-Parallel Combine Kernel
# ===----------------------------------------------------------------------=== #
# Each of the 8 warps in the CTA independently accumulates a RANGE of splits
# in parallel, giving 8 concurrent memory streams instead of 1. After per-warp
# accumulation, a 3-step tree reduction in shared memory merges the partial
# results. Uses DEFERRED normalization: tracks (max, sum-of-exps) separately
# throughout all merge steps, dividing only once at the very end.
#
# SMEM layout (per head):
#   smem_result: [8 warps][head_dim] float32  = 8 * 512 * 4 = 16 KB
#   smem_m:      [8 warps]           float32  = 8 * 4       = 32 B
#   smem_l:      [8 warps]           float32  = 8 * 4       = 32 B
# ===----------------------------------------------------------------------=== #


struct SplitParallelCombineParams[
    output_type: DType,
    accum_type: DType,
    num_splits: Int,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
](Copyable, DevicePassable, TrivialRegisterPassable):
    """Holds the pointers, strides, and shape metadata for the split-parallel combine kernel.

    Carries the per-split partial output and LSE accumulators, the final output
    tensor, optional ragged row offsets and attention-sink pointer, and the
    derived strides used by `mla_combine_kernel_split_parallel`. All 8 warps
    cooperate on a single head, so `heads_per_block` is fixed at 1.

    Parameters:
        output_type: The element type of the per-split partial output
            accumulator and the final combined output tensor.
        accum_type: The element type of the per-split LSE accumulator values.
        num_splits: The compile-time number of KV cache splits to combine.
            Used to partition splits across the 8 warps.
        ragged: Whether batches have variable-length sequences in a packed
            layout (defaults to `False`). When true,
            `input_row_offsets_ptr` selects each batch's token range.
        has_attn_sink: Whether to apply attention-sink correction to the
            global LSE (defaults to `False`). When true, `attn_sink_ptr`
            must be provided.
    """

    # All 8 warps work on one head, so heads_per_block = 1.
    comptime num_warps = 8
    comptime heads_per_block = 1
    comptime num_threads = Self.num_warps * WARP_SIZE

    @__allow_legacy_any_origin_fields
    var out_accum_split_ptr: UnsafePointer[
        Scalar[Self.output_type], origin=MutAnyOrigin
    ]

    @__allow_legacy_any_origin_fields
    var lse_accum_split_ptr: UnsafePointer[
        Scalar[Self.accum_type], origin=MutAnyOrigin
    ]

    # Final output tensor
    @__allow_legacy_any_origin_fields
    var output_ptr: UnsafePointer[Scalar[Self.output_type], origin=MutAnyOrigin]

    # Input row offsets for ragged mode (cumulative token counts per batch)
    @__allow_legacy_any_origin_fields
    var input_row_offsets_ptr: UnsafePointer[UInt32, origin=MutAnyOrigin]

    # Per-head attn_sink values: shape [num_heads_q], float32, nullable.
    @__allow_legacy_any_origin_fields
    var attn_sink_ptr: OptionalReg[UnsafePointer[Float32, origin=MutAnyOrigin]]
    var batch_size: Int
    var seq_len: Int
    var num_heads: Int
    var head_dim: Int

    var lse_stride_split: Int
    var lse_stride_batch: Int
    var lse_stride_seq: Int

    var out_accum_stride_split: Int
    var out_accum_stride_head: Int

    var out_stride_row: Int

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SplitParallelCombineParams"

    @staticmethod
    def get_device_type_name() -> String:
        return "SplitParallelCombineParams"

    def __init__(
        out self,
        out_accum_split_ptr: UnsafePointer[
            Scalar[Self.output_type], origin=MutAnyOrigin
        ],
        lse_accum_split_ptr: UnsafePointer[
            Scalar[Self.accum_type], origin=MutAnyOrigin
        ],
        output_ptr: UnsafePointer[
            Scalar[Self.output_type], origin=MutAnyOrigin
        ],
        input_row_offsets_ptr: UnsafePointer[UInt32, origin=MutAnyOrigin],
        attn_sink_ptr: OptionalReg[UnsafePointer[Float32, origin=MutAnyOrigin]],
        batch_size: Int,
        seq_len: Int,
        num_heads: Int,
        head_dim: Int,
    ):
        self.out_accum_split_ptr = out_accum_split_ptr
        self.lse_accum_split_ptr = lse_accum_split_ptr
        self.output_ptr = output_ptr
        self.input_row_offsets_ptr = input_row_offsets_ptr
        self.attn_sink_ptr = attn_sink_ptr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.lse_stride_split = batch_size * seq_len * num_heads
        self.lse_stride_batch = seq_len * num_heads
        self.lse_stride_seq = num_heads

        self.out_accum_stride_split = (
            batch_size * seq_len * num_heads * head_dim
        )
        self.out_accum_stride_head = head_dim

        self.out_stride_row = head_dim


@__name(
    t"sm100_mla_decode_combine_split_parallel_{output_type}_{accum_type}_{num_splits}_{ragged}",
)
def mla_combine_kernel_split_parallel[
    output_type: DType,
    accum_type: DType,
    head_dim: Int,
    num_splits: Int,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
](
    params: SplitParallelCombineParams[
        output_type,
        accum_type,
        num_splits,
        ragged,
        has_attn_sink,
    ]
):
    """Split-parallel combine: 8 warps process different splits in parallel.

    Each warp independently accumulates its assigned range of splits using
    online log-sum-exp. After the per-warp loop, partial results are written
    to shared memory and tree-reduced in log2(8)=3 steps.

    Parameters:
        output_type: The element type of the per-split partial output
            accumulator and the final combined output tensor.
        accum_type: The element type of the per-split LSE accumulator values.
        head_dim: The dimension of each attention head. Must be divisible by
            `WARP_SIZE`.
        num_splits: The compile-time number of KV cache splits to combine.
            Used to partition splits across the 8 warps.
        ragged: Whether batches have variable-length sequences in a packed
            layout (defaults to `False`). When true,
            `input_row_offsets_ptr` selects each batch's token range.
        has_attn_sink: Whether to apply attention-sink correction to the
            global LSE (defaults to `False`). When true, `attn_sink_ptr`
            must be provided.

    Args:
        params: The `SplitParallelCombineParams` instance carrying the
            per-split partial output and LSE accumulators, the final output
            tensor, optional ragged row offsets and attention-sink pointer,
            and the derived strides used to index the split, batch,
            sequence, and head dimensions.
    """
    # PDL: Wait for the MLA decode kernel (dependent kernel) to complete.
    wait_on_dependent_grids()

    comptime NUM_WARPS = 8
    comptime NEG_INF = min_or_neg_inf[.float32]()

    # Each warp covers the full head_dim.
    # 32 lanes * vec_size * elems_per_thread = head_dim
    comptime assert (
        head_dim % WARP_SIZE == 0
    ), "head_dim must be divisible by WARP_SIZE"
    comptime max_vec = head_dim // WARP_SIZE
    comptime vec_size = min(max_vec, 8)
    comptime elems_per_thread = head_dim // (WARP_SIZE * vec_size)

    var batch_idx = block_idx.x
    var seq_idx = block_idx.y
    var head_idx = block_idx.z
    var warp_idx = warp_id()
    var lane_idx = lane_id()

    # Early exit for out-of-range heads.
    if head_idx >= params.num_heads:
        return

    # In ragged mode, each batch can have a different number of Q tokens.
    comptime if ragged:
        var batch_seq_len = Int(
            params.input_row_offsets_ptr[batch_idx + 1]
        ) - Int(params.input_row_offsets_ptr[batch_idx])
        if seq_idx >= batch_seq_len:
            return

    # =========================================================================
    # Shared memory for tree reduction.
    # Layout: smem_result[warp][elem], smem_m[warp], smem_l[warp]
    # =========================================================================
    var smem_result = unsafe_stack_allocation[
        NUM_WARPS * head_dim,
        DType.float32,
        address_space=.SHARED,
    ]()
    var smem_m = unsafe_stack_allocation[
        NUM_WARPS,
        DType.float32,
        address_space=.SHARED,
    ]()
    var smem_l = unsafe_stack_allocation[
        NUM_WARPS,
        DType.float32,
        address_space=.SHARED,
    ]()

    # =========================================================================
    # Step 1: Determine this warp's split range (runtime).
    # Splits are divided evenly across 8 warps: warp w processes splits
    # [w * splits_per_warp, min((w+1) * splits_per_warp, num_splits)).
    # =========================================================================
    comptime splits_per_warp = ceildiv(num_splits, NUM_WARPS)
    var split_start = warp_idx * splits_per_warp
    var split_end = min(split_start + splits_per_warp, num_splits)

    # Base pointers for this head.
    var out_row = (
        batch_idx * params.seq_len * params.num_heads
        + seq_idx * params.num_heads
        + head_idx
    )
    var oaccum_base = (
        params.out_accum_split_ptr + out_row * params.out_accum_stride_head
    ).as_imm()

    var lse_base = (
        batch_idx * params.lse_stride_batch
        + seq_idx * params.lse_stride_seq
        + head_idx
    )

    # =========================================================================
    # Step 2: Per-warp online log-sum-exp accumulation over assigned splits.
    # Uses DEFERRED normalization: track running max (warp_m) and running
    # sum-of-exps (warp_l) separately, deferring the final division to the
    # very end. This eliminates 2 FP32 divisions per merge step.
    #
    # Invariant maintained:
    #   result = sum_over_seen_splits(exp2(lse_i - warp_m) * O_i)
    #   warp_l = sum_over_seen_splits(exp2(lse_i - warp_m))
    # Final output = result / warp_l
    # =========================================================================
    var result = Array[SIMD[.float32, vec_size], elems_per_thread](
        fill=SIMD[.float32, vec_size](0.0)
    )
    var warp_m = Float32(NEG_INF)
    var warp_l = Float32(0.0)

    for s in range(split_start, split_end):
        # Load this split's LSE.
        var lse_offset = s * params.lse_stride_split + lse_base
        var split_lse = params.lse_accum_split_ptr[lse_offset].cast[
            DType.float32
        ]()

        # Skip empty splits (lse = -inf).
        if split_lse == NEG_INF:
            continue

        # Update running max and compute rescale factors.
        var new_m = max(warp_m, split_lse)
        var rescale = exp2(warp_m - new_m)  # always <= 1
        var split_l = exp2(split_lse - new_m)  # this split's weight

        # Load split data and accumulate (unnormalized).
        comptime for i in range(elems_per_thread):
            var offset = (
                s * params.out_accum_stride_split
                + lane_idx * vec_size
                + i * (WARP_SIZE * vec_size)
            )
            var data_f32 = oaccum_base.load[width=vec_size](offset).cast[
                DType.float32
            ]()
            result[i] = rescale * result[i] + split_l * data_f32

        # Update running sum-of-exps and max.
        warp_l = warp_l * rescale + split_l
        warp_m = new_m

    # =========================================================================
    # Step 3: Write per-warp partial results to shared memory.
    # =========================================================================
    var smem_warp_base = warp_idx * head_dim

    comptime for i in range(elems_per_thread):
        var elem_offset = lane_idx * vec_size + i * (WARP_SIZE * vec_size)
        smem_result.store[width=vec_size](
            smem_warp_base + elem_offset, result[i]
        )

    if lane_idx == 0:
        smem_m[warp_idx] = warp_m
        smem_l[warp_idx] = warp_l

    barrier()

    # =========================================================================
    # Step 4: Tree reduction across warps (3 steps for 8 warps).
    # At each step, the lower-indexed warp merges with the higher-indexed warp.
    # Step 0: warps 0-3 merge with warps 4-7  (stride=4)
    # Step 1: warps 0-1 merge with warps 2-3  (stride=2)
    # Step 2: warp 0 merges with warp 1       (stride=1)
    #
    # Uses deferred normalization: each partial result is stored as
    # (result, m, l) where result = sum(exp2(lse_i - m) * O_i) and
    # l = sum(exp2(lse_i - m)). Division by l is deferred to the end.
    # =========================================================================
    comptime for step in range(3):
        comptime stride = NUM_WARPS >> (step + 1)  # 4, 2, 1
        if warp_idx < stride:
            var partner = warp_idx + stride

            # Load partner's (m, l).
            var partner_m = smem_m[partner]
            var my_m = smem_m[warp_idx]
            var partner_l = smem_l[partner]
            var my_l = smem_l[warp_idx]

            # Compute new max and rescale factors.
            var new_m = max(my_m, partner_m)
            var my_rescale: Float32
            var partner_rescale: Float32

            if new_m == NEG_INF:
                # Both are empty.
                my_rescale = Float32(0.0)
                partner_rescale = Float32(0.0)
            else:
                my_rescale = exp2(my_m - new_m)
                partner_rescale = exp2(partner_m - new_m)

            var smem_partner_base = partner * head_dim
            var smem_my_base = warp_idx * head_dim

            comptime for i in range(elems_per_thread):
                var elem_offset = lane_idx * vec_size + i * (
                    WARP_SIZE * vec_size
                )
                var my_data = smem_result.load[width=vec_size](
                    smem_my_base + elem_offset
                )
                var partner_data = smem_result.load[width=vec_size](
                    smem_partner_base + elem_offset
                )
                var merged = (
                    my_rescale * my_data + partner_rescale * partner_data
                )
                smem_result.store[width=vec_size](
                    smem_my_base + elem_offset, merged
                )

            # Update merged (m, l).
            if lane_idx == 0:
                smem_m[warp_idx] = new_m
                smem_l[warp_idx] = (
                    my_l * my_rescale + partner_l * partner_rescale
                )

        barrier()

    # =========================================================================
    # Step 5: Final normalization and attn sink correction.
    # Only warp 0 needs to do this. The accumulated result in smem_result[0]
    # is unnormalized: result = sum(exp2(lse_i - m) * O_i). We need to
    # divide by total_l = sum(exp2(lse_i - m)) to get the final output.
    # =========================================================================
    if warp_idx == 0:
        var total_m = smem_m[0]
        var total_l = smem_l[0]

        # Attn sink correction: add attn_sink's contribution to total_l.
        # The sink contributes exp2(attn_sink_log2 - total_m) to the
        # denominator but nothing to the numerator (no output vector).
        comptime if has_attn_sink:
            var attn_sink_val = params.attn_sink_ptr.unsafe_value()[head_idx]
            var attn_sink_log2_val = attn_sink_val * Float32(log2e)

            if total_m != NEG_INF:
                # Normal case: add attn_sink's weight to total_l.
                total_l = total_l + exp2(attn_sink_log2_val - total_m)
            else:
                # All splits empty. If attn_sink is also -inf, output is
                # zero. Otherwise, attn_sink contributes only to the
                # denominator making output zero anyway (no numerator).
                # Set total_l to 0 to produce zero output.
                total_l = Float32(0.0)

        # Compute the final normalization factor: 1 / total_l.
        # If total_l is 0 (all splits empty), output zero.
        var inv_l: Float32
        if total_l == Float32(0.0):
            inv_l = Float32(0.0)
        else:
            inv_l = Float32(1.0) / total_l

        var final_out_row: Int

        comptime if ragged:
            var start_of_seq = Int(params.input_row_offsets_ptr[batch_idx])
            final_out_row = (
                start_of_seq * params.num_heads
                + seq_idx * params.num_heads
                + head_idx
            )
        else:
            final_out_row = out_row

        var out_ptr = params.output_ptr + final_out_row * params.out_stride_row

        comptime for i in range(elems_per_thread):
            var elem_offset = lane_idx * vec_size + i * (WARP_SIZE * vec_size)
            var data = smem_result.load[width=vec_size](elem_offset)
            var out_data = (inv_l * data).cast[output_type]()
            out_ptr.store(elem_offset, out_data)


# ===----------------------------------------------------------------------=== #
# Kernel Dispatch Function
# ===----------------------------------------------------------------------=== #
def launch_mla_combine_kernel_split_parallel[
    output_type: DType,
    accum_type: DType,
    head_dim: Int,
    num_splits: Int,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
](
    out_accum_split: TileTensor[output_type, address_space=.GENERIC, ...],
    lse_accum_split: TileTensor[accum_type, address_space=.GENERIC, ...],
    output: TileTensor[output_type, address_space=.GENERIC, ...],
    input_row_offsets_ptr: UnsafePointer[UInt32, origin=MutAnyOrigin],
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    ctx: DeviceContext,
) raises:
    """Launches the split-parallel combine kernel on the device with one CTA per (batch, seq, head).

    Extracts raw device pointers from the input `TileTensor` arguments,
    constructs a `SplitParallelCombineParams` instance, and enqueues
    `mla_combine_kernel_split_parallel` with a grid of `(batch_size, seq_len,
    num_heads)` blocks and PDL launch attributes.

    Parameters:
        output_type: The element type of the per-split partial output
            accumulator and the final combined output tensor.
        accum_type: The element type of the per-split LSE accumulator values.
        head_dim: The dimension of each attention head. Must be divisible by
            `WARP_SIZE`.
        num_splits: The compile-time number of KV cache splits to combine.
            Used to partition splits across the 8 warps.
        ragged: Whether batches have variable-length sequences in a packed
            layout (defaults to `False`). When true,
            `input_row_offsets_ptr` selects each batch's token range.
        has_attn_sink: Whether to apply attention-sink correction to the
            global LSE (defaults to `False`). When true, `attn_sink_ptr`
            must be provided.

    Args:
        out_accum_split: The per-split partial output accumulator tensor of
            shape `[num_splits, batch_size, seq_len, num_heads, head_dim]`.
        lse_accum_split: The per-split LSE accumulator tensor of shape
            `[num_splits, batch_size, seq_len, num_heads]`.
        output: The final combined output tensor of shape `[batch_size,
            seq_len, num_heads, head_dim]` (or the ragged-packed equivalent).
        input_row_offsets_ptr: Pointer to cumulative token counts per batch;
            `input_row_offsets_ptr[i]` is the start token index for batch
            `i`. Used only in ragged mode.
        attn_sink_ptr: Optional pointer to per-head attention-sink LSE values
            of shape `[num_heads]` in natural log. Used only when
            `has_attn_sink` is true.
        batch_size: Number of batches in the request.
        seq_len: Maximum number of query tokens per batch (the padded grid
            dimension along the sequence axis).
        num_heads: Number of query attention heads.
        ctx: The device context used to enqueue the kernel.
    """
    comptime ParamsType = SplitParallelCombineParams[
        output_type,
        accum_type,
        num_splits,
        ragged,
        has_attn_sink,
    ]
    comptime num_threads = ParamsType.num_threads

    var out_accum_ptr = rebind[
        UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]
    ](out_accum_split.to_device_buffer(ctx).unsafe_ptr())

    var lse_ptr = rebind[
        UnsafePointer[Scalar[accum_type], origin=MutAnyOrigin]
    ](lse_accum_split.to_device_buffer(ctx).unsafe_ptr())

    var out_ptr = rebind[
        UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]
    ](output.to_device_buffer(ctx).unsafe_ptr())

    var params = ParamsType(
        out_accum_ptr,
        lse_ptr,
        out_ptr,
        input_row_offsets_ptr,
        attn_sink_ptr,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    )

    # Grid: one CTA per (batch, seq, head).
    var grid_dim = (batch_size, seq_len, num_heads)
    var block_dim = (num_threads, 1, 1)

    ctx.enqueue_function[
        mla_combine_kernel_split_parallel[
            output_type,
            accum_type,
            head_dim,
            num_splits,
            ragged,
            has_attn_sink,
        ]
    ](
        params,
        grid_dim=grid_dim,
        block_dim=block_dim,
        attributes=pdl_launch_attributes(MLA_DECODE_COMBINE_PDL_LEVEL),
    )


# ===----------------------------------------------------------------------=== #
# Kernel Dispatch Function (original)
# ===----------------------------------------------------------------------=== #
def launch_mla_combine_kernel[
    output_type: DType,
    accum_type: DType,
    head_dim: Int,
    num_splits: Int,  # Compile-time number of splits for loop unrolling
    ragged: Bool = False,
    warps_per_head: Int = 2,
    has_attn_sink: Bool = False,
](
    out_accum_split: TileTensor[output_type, address_space=.GENERIC, ...],
    lse_accum_split: TileTensor[accum_type, address_space=.GENERIC, ...],
    output: TileTensor[output_type, address_space=.GENERIC, ...],
    input_row_offsets_ptr: UnsafePointer[UInt32, origin=MutAnyOrigin],
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    ctx: DeviceContext,
) raises:
    """Launches the main split-K combine kernel on the device with one CTA per (batch, seq, head-block).

    Extracts raw device pointers from the input `TileTensor` arguments,
    constructs a `CombineParams` instance, and enqueues `mla_combine_kernel`
    with a grid of `(batch_size, seq_len, ceildiv(num_heads, heads_per_block))`
    blocks and PDL launch attributes.

    Parameters:
        output_type: The element type of the per-split partial output
            accumulator and the final combined output tensor.
        accum_type: The element type of the per-split LSE accumulator values.
        head_dim: The dimension of each attention head. Must be divisible by
            `WARP_SIZE * warps_per_head`.
        num_splits: The compile-time number of KV cache splits to combine.
            Used to unroll the per-split accumulation loop.
        ragged: Whether batches have variable-length sequences in a packed
            layout (defaults to `False`). When true,
            `input_row_offsets_ptr` selects each batch's token range.
        warps_per_head: Number of warps assigned to each attention head
            (defaults to 2). Must divide 8; controls vector load width and
            threads per block.
        has_attn_sink: Whether to apply attention-sink correction to the
            global LSE (defaults to `False`). When true, `attn_sink_ptr`
            must be provided.

    Args:
        out_accum_split: The per-split partial output accumulator tensor of
            shape `[num_splits, batch_size, seq_len, num_heads, head_dim]`.
        lse_accum_split: The per-split LSE accumulator tensor of shape
            `[num_splits, batch_size, seq_len, num_heads]`.
        output: The final combined output tensor of shape `[batch_size,
            seq_len, num_heads, head_dim]` (or the ragged-packed equivalent).
        input_row_offsets_ptr: Pointer to cumulative token counts per batch;
            `input_row_offsets_ptr[i]` is the start token index for batch
            `i`. Used only in ragged mode.
        attn_sink_ptr: Optional pointer to per-head attention-sink LSE values
            of shape `[num_heads]` in natural log. Used only when
            `has_attn_sink` is true.
        batch_size: Number of batches in the request.
        seq_len: Maximum number of query tokens per batch (the padded grid
            dimension along the sequence axis).
        num_heads: Number of query attention heads.
        ctx: The device context used to enqueue the kernel.
    """
    comptime ParamsType = CombineParams[
        output_type,
        accum_type,
        num_splits,
        ragged,
        warps_per_head,
        has_attn_sink,
    ]
    comptime heads_per_block = ParamsType.heads_per_block
    comptime num_threads = ParamsType.num_threads

    var out_accum_ptr = rebind[
        UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]
    ](out_accum_split.to_device_buffer(ctx).unsafe_ptr())

    var lse_ptr = rebind[
        UnsafePointer[Scalar[accum_type], origin=MutAnyOrigin]
    ](lse_accum_split.to_device_buffer(ctx).unsafe_ptr())

    var out_ptr = rebind[
        UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]
    ](output.to_device_buffer(ctx).unsafe_ptr())

    var params = ParamsType(
        out_accum_ptr,
        lse_ptr,
        out_ptr,
        input_row_offsets_ptr,
        attn_sink_ptr,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    )

    var grid_dim = (batch_size, seq_len, ceildiv(num_heads, heads_per_block))
    var block_dim = (num_threads, 1, 1)

    ctx.enqueue_function[
        mla_combine_kernel[
            output_type,
            accum_type,
            head_dim,
            num_splits,
            ragged,
            warps_per_head,
            has_attn_sink,
        ]
    ](
        params,
        grid_dim=grid_dim,
        block_dim=block_dim,
        attributes=pdl_launch_attributes(MLA_DECODE_COMBINE_PDL_LEVEL),
    )


# ===----------------------------------------------------------------------=== #
# High-level dispatcher to be called from mla_decode_sm100_dispatch.mojo
# ===----------------------------------------------------------------------=== #
def mla_decode_combine_partial_outputs[
    output_type: DType,
    accum_type: DType,
    head_dim: Int,
    num_splits: Int,
    ragged: Bool = False,
    warps_per_head: Int = 2,
    has_attn_sink: Bool = False,
    split_parallel: Bool = False,
](
    out_accum_split: TileTensor[output_type, address_space=.GENERIC, ...],
    lse_accum_split: TileTensor[accum_type, address_space=.GENERIC, ...],
    output: TileTensor[output_type, address_space=.GENERIC, ...],
    input_row_offsets_ptr: UnsafePointer[UInt32, origin=MutAnyOrigin],
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    ctx: DeviceContext,
) raises:
    """Dispatches split-K partial output combination to either the split-parallel or the main combine kernel.

    Selects `launch_mla_combine_kernel_split_parallel` when `split_parallel` is
    true, otherwise falls back to `launch_mla_combine_kernel`, forwarding all
    tensor pointers, shape, and attention-sink parameters unchanged.

    Parameters:
        output_type: The element type of the per-split partial output
            accumulator and the final combined output tensor.
        accum_type: The element type of the per-split LSE accumulator values.
        head_dim: The dimension of each attention head. Must be divisible by
            `WARP_SIZE` (and by `WARP_SIZE * warps_per_head` when
            `split_parallel` is false).
        num_splits: The compile-time number of KV cache splits to combine.
            Used to unroll the per-split accumulation loop in the main kernel
            and to partition splits across warps in the split-parallel kernel.
        ragged: Whether batches have variable-length sequences in a packed
            layout (defaults to `False`). When true,
            `input_row_offsets_ptr` selects each batch's token range.
        warps_per_head: Number of warps assigned to each attention head
            (defaults to 2). Must divide 8; controls vector load width and
            threads per block. Used only when `split_parallel` is false.
        has_attn_sink: Whether to apply attention-sink correction to the
            global LSE (defaults to `False`). When true, `attn_sink_ptr`
            must be provided.
        split_parallel: Selects the split-parallel combine kernel when true,
            otherwise the main combine kernel (defaults to `False`).

    Args:
        out_accum_split: The per-split partial output accumulator tensor of
            shape `[num_splits, batch_size, seq_len, num_heads, head_dim]`.
        lse_accum_split: The per-split LSE accumulator tensor of shape
            `[num_splits, batch_size, seq_len, num_heads]`.
        output: The final combined output tensor of shape `[batch_size,
            seq_len, num_heads, head_dim]` (or the ragged-packed equivalent).
        input_row_offsets_ptr: Pointer to cumulative token counts per batch;
            `input_row_offsets_ptr[i]` is the start token index for batch
            `i`. Used only in ragged mode.
        attn_sink_ptr: Optional pointer to per-head attention-sink LSE values
            of shape `[num_heads]` in natural log. Used only when
            `has_attn_sink` is true.
        batch_size: Number of batches in the request.
        seq_len: Maximum number of query tokens per batch (the padded grid
            dimension along the sequence axis).
        num_heads: Number of query attention heads.
        ctx: The device context used to enqueue the kernel.
    """
    comptime if split_parallel:
        launch_mla_combine_kernel_split_parallel[
            output_type,
            accum_type,
            head_dim,
            num_splits,
            ragged,
            has_attn_sink,
        ](
            out_accum_split,
            lse_accum_split,
            output,
            input_row_offsets_ptr,
            attn_sink_ptr,
            batch_size,
            seq_len,
            num_heads,
            ctx,
        )
    else:
        launch_mla_combine_kernel[
            output_type,
            accum_type,
            head_dim,
            num_splits,
            ragged,
            warps_per_head,
            has_attn_sink,
        ](
            out_accum_split,
            lse_accum_split,
            output,
            input_row_offsets_ptr,
            attn_sink_ptr,
            batch_size,
            seq_len,
            num_heads,
            ctx,
        )
