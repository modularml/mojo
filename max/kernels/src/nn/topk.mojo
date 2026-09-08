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
"""Provides top-K selection kernels using warp- and block-level reductions for CPU and GPU."""

from std.builtin.debug_assert import ASSERT_MODE
from std.atomic import Atomic, Ordering, fence
from std.math import align_up, ceildiv, exp, iota
from std.math.uutils import ufloordiv, udivmod
from std.memory import ThinAllocation, alloc, dealloc
from std.memory.alloc import Layout as AllocLayout
from std.sys import align_of, simd_width_of, size_of

import max.gpu.primitives.warp as warp
from max.algorithm.functional import parallelize_over_rows
from max.algorithm.reduction import _get_nd_indices_from_flat_index
from std.bit import log2_floor
from max.gpu import (
    WARP_SIZE,
    thread_idx,
    block_dim,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host.info import is_cpu
from max.gpu.memory import external_memory
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    is_apple_gpu,
)
from std.random import Random
from layout import (
    Coord,
    CoordLike,
    Idx,
    DefaultEngine,
    RowMajorLayout,
    TensorLayout,
    TensorEngine,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.coord import DynamicCoord
from layout.tile_layout import Layout
from std.math import log2
from std.memory import unsafe_stack_allocation
from nn.gather_scatter import normalize_neg_index
from nn.reshape import reshape
from nn.sampling import topk_topp_sampling_from_prob
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import IndexList, product
from std.utils.numerics import max_or_inf, min_or_neg_inf
from max.gpu.primitives.grid_controls import PDLLevel

from .normalization import (
    _APPLE_STATIC_SHMEM_MAX_COUNT,
    _APPLE_STATIC_SHMEM_MAX_BYTES,
)


# `_APPLE_STATIC_SHMEM_MAX_COUNT` fills the whole 32K of Apple's static
# threadgroup memory; using it for the main buffer leaves no room for the small
# auxiliary allocations (the per-warp `s_sum`/counters) the compiler sums into
# the same kernel -- which pushed `fused_token_sampling` to 32800 B, 32 over
# Metal's 32768 limit. Reserve headroom so main buffer + auxiliaries stay under
# 32K. The main buffers here are vastly over-allocated (sized to the 32K bound
# but indexed only up to `block_size`), so shrinking them is free.
comptime _APPLE_STATIC_SHMEM_RESERVE_BYTES = 2 * 1024
comptime _APPLE_STATIC_SHMEM_USABLE_COUNT[T: AnyType] = (
    _APPLE_STATIC_SHMEM_MAX_BYTES - _APPLE_STATIC_SHMEM_RESERVE_BYTES
) // size_of[T]()


@always_inline
def top_k_shape_impl[
    dtype: DType
](
    input: TileTensor[mut=False, dtype, ...], max_k: Int, axis: Int
) raises -> IndexList[input.rank]:
    """
    Compute the output shape of a top/bottom k operation.

    Parameters:
        dtype: Data type of the input buffer.

    Args:
        input: The input tensor.
        max_k: The maximum K value.
        axis: The axis value in a tensor.

    Returns:
        The output shape.
    """

    # Normalize a negative axis up front, matching what `top_k()` itself does;
    # `TileTensor.dim()` has no negative-index case and would abort otherwise.
    var normalized_axis = normalize_neg_index(axis, input.rank)

    # Clamp max_k
    var bound_max_k = Int(input.dim(normalized_axis)) if max_k == -1 else max_k

    if bound_max_k < 0 or bound_max_k > Int(input.dim(normalized_axis)):
        raise Error("[top/bottom-k] k must be within [0, input_shape[axis]]")

    var shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )
    shape[normalized_axis] = bound_max_k

    return shape


@always_inline
def _adjust_top_p[
    T: DType,
    address_space: AddressSpace = .GENERIC,
](
    top_p: Scalar[T],
    values: UnsafePointer[Scalar[T], _, address_space=address_space],
    k: Int,
    total_sum: Scalar[T],
) -> Scalar[T]:
    # Align the given top_p to the cumulative probability of the tokens.
    # For example, if after top_k we have three tokens with probabilities
    # [0.7, 0.2, 0.1] and top_p = 0.8, then we should sample from the first
    # two tokens with probabilities [0.7, 0.2], so we set _top_p = 0.9.
    var _top_p = Scalar[T](1)
    if top_p < 1:
        var cum_prob = Scalar[T](0)
        for ki in range(k):
            cum_prob += values[ki]
            if cum_prob >= top_p * total_sum:
                break
        _top_p = cum_prob / total_sum
    return _top_p


def top_k[
    dtype: DType,
    out_idx_type: DType,
    //,
    largest: Bool = True,
    target: StaticString = "cpu",
](
    input: TileTensor[mut=False, dtype, ...],
    max_k: Int,
    axis: Int,
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    sorted: Bool,
    ctx: DeviceContext,
    k: Optional[
        TileTensor[.int64, RowMajorLayout[Int64], ImmutAnyOrigin],
    ] = None,
) raises:
    """
    Implementation of the Top K algorithm. Returns the top or bottom K elements
    and their index along a specified axis.

    Parameters:
        dtype: Data type of the input buffer.
        out_idx_type: The data dtype of the output indices (default == .int64).
        largest: Whether to find the maximum (top k) or minimum value (bottom k).
        target: The target to run on.

    Args:
        input: The input tensor.
        max_k: The largest number of top elements.
        axis: The axis along which to operate.
        out_vals: Output values.
        out_idxs: Output indices.
        sorted: Indicates if the top/bottom K elements are in (stable) sorted order.
        ctx: The device call context.
        k: Per batch element k value.
    """
    comptime assert (
        input.rank == out_vals.rank
    ), "input.rank must match out_vals.rank"
    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"

    var input_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("input", input_shape, dtype),
                    "max_k=" + String(max_k),
                    "axis=" + String(axis),
                    "largest=" + String(largest),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=target](
        "top_k",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var normalized_axis = normalize_neg_index(Int64(axis), input.rank)

        # Clamp max_k
        var bound_max_k = 255 if max_k == -1 else max_k

        comptime if is_cpu[target]():
            comptime assert (
                out_idx_type == .int64
            ), "out_idx_type must be int64 for cpu"

            comptime grain_size = 1000
            _top_k_cpu[largest=largest](
                input,
                bound_max_k,
                Int(normalized_axis),
                out_vals,
                out_idxs,
                grain_size,
                sorted=sorted,
                ctx=Optional[DeviceContext](ctx),
                k=k,
            )
        else:
            if normalized_axis != Int(input.rank - 1):
                raise Error("axis other than -1 not supported on GPU")
            if not sorted:
                print(
                    "Warning: Unsorted top-k is not supported on GPU. Falling"
                    " back to sorted top-k."
                )
            topk_gpu[sampling=False, largest=largest](
                ctx,
                bound_max_k,
                input,
                out_vals,
                out_idxs,
                k=k,
            )


def _top_k_cpu[
    dtype: DType,
    out_idx_type: DType,
    largest: Bool,
    KLayoutType: TensorLayout = RowMajorLayout[Int64],
    KEngine: TensorEngine = DefaultEngine[element_width=1],
](
    input: TileTensor[mut=False, dtype, ...],
    max_k: Int,
    axis: Int,
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    parallelism_grain_size: Int,  # impl detail, exposed for testing
    sorted: Bool,
    ctx: Optional[DeviceContext] = None,
    k: Optional[
        TileTensor[.int64, KLayoutType, ImmutAnyOrigin, Engine=KEngine]
    ] = None,
):
    comptime assert (
        input.rank == out_vals.rank
    ), "input.rank must match out_vals.rank"
    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"
    comptime assert k.T.flat_rank == 1
    var shape = coord_to_index_list(input.layout.shape_coord())

    def process_rows(start_row: Int, end_row: Int) {var shape, imm}:
        # Allocate the index list without initializing its elements.
        var idxs = List[Int64](unsafe_uninit_length=shape[axis])

        for row_idx in range(start_row, end_row):
            var indices = _get_nd_indices_from_flat_index(row_idx, shape, axis)
            iota(idxs)

            var batch_idx = indices[0] if axis != 0 else 0
            var k_val = max_k
            if k:
                var k_raw = Int(k.value()[batch_idx])
                k_val = max_k if k_raw == -1 else k_raw

            # Clamp k to the size of the axis to avoid out-of-bounds access
            if k_val > shape[axis]:
                k_val = shape[axis]

            comptime if largest:

                @always_inline
                def _val_greater_than(
                    lhs: Int64, rhs: Int64
                ) {mut indices, input, axis} -> Bool:
                    indices[axis] = Int(lhs)
                    var lhs_val = input.raw_load(input.layout(Coord(indices)))
                    indices[axis] = Int(rhs)
                    var rhs_val = input.raw_load(input.layout(Coord(indices)))
                    return lhs_val > rhs_val

                if sorted:
                    sort(idxs, _val_greater_than)
                else:
                    _ = partition(idxs, k_val, _val_greater_than)
            else:

                @always_inline
                def _val_less_than(
                    lhs: Int64, rhs: Int64
                ) {mut indices, input, axis} -> Bool:
                    indices[axis] = Int(lhs)
                    var lhs_val = input.raw_load(input.layout(Coord(indices)))
                    indices[axis] = Int(rhs)
                    var rhs_val = input.raw_load(input.layout(Coord(indices)))
                    return lhs_val < rhs_val

                if sorted:
                    sort(idxs, _val_less_than)
                else:
                    _ = partition(idxs, k_val, _val_less_than)

            if sorted:
                # for duplicate vals, the smaller index needs to appear first
                # _quicksort is not stable, so do another pass to enforce this
                # could use a stable sorting algorithm but the complexity is O(n*log(n)*log(n))
                # this is also what tensorflow and PT do:
                # https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/kernels/topk_op.cc#L171-L172
                var i = 0
                while i < shape[axis] - 1:
                    indices[axis] = Int(idxs[i])
                    var input_idx = input.layout(Coord(indices))
                    var curr = input.raw_load(input_idx)
                    var num_equal = 1
                    for j in range(i + 1, shape[axis]):
                        indices[axis] = Int(idxs[j])
                        var input_idx = input.layout(Coord(indices))
                        var next = input.raw_load(input_idx)
                        if curr != next:
                            break
                        num_equal += 1
                    if num_equal > 1:
                        var idxs_ptr: UnsafePointer[
                            idxs.T, origin_of(idxs)
                        ] = idxs.unsafe_ptr()
                        var ptr = idxs_ptr + i
                        sort(
                            Span[idxs.T, origin_of(idxs)](
                                unsafe_ptr=ptr, length=num_equal
                            )
                        )
                    i += num_equal

            for i in range(k_val):
                indices[axis] = Int(idxs[i])
                var input_idx = input.layout(Coord(indices))
                var val = input.raw_load(input_idx)
                indices[axis] = i
                var out_vals_idx = out_vals.layout(Coord(indices))
                var out_idxs_idx = out_idxs.layout(Coord(indices))
                out_vals.raw_store(out_vals_idx, val)
                out_idxs.ptr[out_idxs_idx] = rebind[Scalar[out_idx_type]](
                    idxs[i]
                )

    parallelize_over_rows(
        process_rows, shape, axis, parallelism_grain_size, ctx
    )


@always_inline
def fused_token_sampling_cpu[
    dtype: DType,
    out_idx_type: DType,
    KLayoutType: TensorLayout = RowMajorLayout[Int64],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[Int64],
    TopPLayoutType: TensorLayout = RowMajorLayout[Int64],
    SeedLayoutType: TensorLayout = RowMajorLayout[Int64],
    KEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
](
    max_k: Int,
    input: TileTensor[mut=False, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    k: Optional[
        TileTensor[
            .int64,
            KLayoutType,
            ImmutAnyOrigin,
            Engine=KEngine,
        ]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    top_p: Optional[
        TileTensor[
            .float32,
            TopPLayoutType,
            ImmutAnyOrigin,
            Engine=TopPEngine,
        ]
    ] = None,
    seed: Optional[
        TileTensor[
            .uint64,
            SeedLayoutType,
            ImmutAnyOrigin,
            Engine=SeedEngine,
        ]
    ] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume.

    Parameters:
        dtype: Data type of the input buffer.
        out_idx_type: Data type of the output indices.
        KLayoutType: Layout type of the k buffer.
        TemperatureLayoutType: Layout type of the temperature buffer.
        TopPLayoutType: Layout type of the top_p buffer.
        SeedLayoutType: Layout type of the seed buffer.
        KEngine: Engine policy of the k buffer.
        TemperatureEngine: Engine policy of the temperature buffer.
        TopPEngine: Engine policy of the top_p buffer.
        SeedEngine: Engine policy of the seed buffer.

    Args:
        max_k: Largest number of top elements.
        input: TileTensor[dtype] (Any shape)- The input tensor.
        out_idxs: TileTensor[out_idx_type] (shape of [input_shape[:-1]] + [1]) - The output indices.
        k: Optional device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.
    """
    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"
    comptime assert out_idx_type == .int64, "out_idx_type must be int64"

    var input_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("input", input_shape, dtype),
                    "max_k=" + String(max_k),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("cpu")](
        "fused_token_sampling",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
    ):
        var bound_max_k = 255 if max_k == -1 else max_k

        # materialize the out_vals which is of shape [input[:-1]] + [k]
        var out_vals_shape = coord_to_index_list(input.layout.shape_coord())
        out_vals_shape[input.rank - 1] = bound_max_k
        var out_vals_alloc = alloc(
            AllocLayout[Scalar[dtype]](count=out_vals_shape.flattened_length())
        ).into_managed()
        var out_vals_ptr: UnsafePointer[
            Scalar[dtype], origin_of(out_vals_alloc)
        ] = out_vals_alloc.unsafe_ptr()
        var out_vals = TileTensor(
            out_vals_ptr,
            row_major(Coord(out_vals_shape)),
        )

        _top_k_sampling(
            bound_max_k,
            input,
            out_vals,
            out_idxs.bitcast[.int64](),
            k,
            temperature,
            top_p,
            seed,
        )

        dealloc(out_vals_alloc^)


def _top_k_sampling[
    dtype: DType,
    KLayoutType: TensorLayout = RowMajorLayout[Int64],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[Int64],
    TopPLayoutType: TensorLayout = RowMajorLayout[Int64],
    SeedLayoutType: TensorLayout = RowMajorLayout[Int64],
    KEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
](
    max_k: Int,
    input: TileTensor[mut=False, dtype, ...],
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, .int64, ...],
    k: Optional[
        TileTensor[
            .int64,
            KLayoutType,
            ImmutAnyOrigin,
            Engine=KEngine,
        ]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    top_p: Optional[
        TileTensor[
            .float32,
            TopPLayoutType,
            ImmutAnyOrigin,
            Engine=TopPEngine,
        ]
    ] = None,
    seed: Optional[
        TileTensor[
            .uint64,
            SeedLayoutType,
            ImmutAnyOrigin,
            Engine=SeedEngine,
        ]
    ] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume.

    Parameters:
        dtype: Data type of the input buffer.
        KLayoutType: Layout type of the k buffer.
        TemperatureLayoutType: Layout type of the temperature buffer.
        TopPLayoutType: Layout type of the top_p buffer.
        SeedLayoutType: Layout type of the seed buffer.
        KEngine: Engine policy of the k buffer.
        TemperatureEngine: Engine policy of the temperature buffer.
        TopPEngine: Engine policy of the top_p buffer.
        SeedEngine: Engine policy of the seed buffer.

    Args:
        max_k: Largest number of top elements.
        input: TileTensor[dtype] (Any shape)- The input tensor.
        out_vals: TileTensor[dtype] (shape of [input[:-1]] + [k]) - The output values.
        out_idxs: TileTensor[.int64] (shape of [input[:-1]] + [1]) - The output indices.
        k: Optional buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        seed: The seed to use for the random number generator.
    """
    comptime assert dtype.is_floating_point(), "dtype must be floating point"
    comptime assert (
        input.rank == out_vals.rank
    ), "input.rank must match out_vals.rank"
    comptime assert (
        input.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"
    comptime assert temperature.T.flat_rank == 1
    comptime assert k.T.flat_rank == 1
    comptime assert top_p.T.flat_rank == 1
    comptime assert seed.T.flat_rank == 1

    # Now reshape for sampling
    var orig_in_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )
    var last_dim = orig_in_shape[input.rank - 1]

    comptime internal_rank = 2
    var internal_bs: Int
    var internal_in_shape: IndexList[internal_rank]

    comptime if input.rank == 1:
        internal_bs = 1
        internal_in_shape = IndexList[internal_rank](1, input.num_elements())
    elif input.rank == internal_rank:
        internal_bs = orig_in_shape[0]
        internal_in_shape = rebind[IndexList[internal_rank]](orig_in_shape)
    elif input.rank > internal_rank:
        internal_bs = Int(
            Float64(orig_in_shape.flattened_length()) / Float64(last_dim)
        )
        internal_in_shape = IndexList[internal_rank](internal_bs, last_dim)
    else:
        raise Error("Unsupported input rank. Must be >= 1.")

    var internal_out_shape = IndexList[internal_rank](internal_bs, max_k)
    var internal_out_idxs_shape = IndexList[internal_rank](internal_bs, 1)

    var reshaped_out_idxs = reshape(out_idxs, internal_out_idxs_shape)
    var reshaped_out_vals = reshape(out_vals, internal_out_shape)

    var out_idxs_tmp_alloc = alloc(
        AllocLayout[Int64](count=out_vals.num_elements())
    )
    var out_idxs_tmp_ptr: UnsafePointer[
        Int64, origin_of(out_idxs_tmp_alloc._alloc)
    ] = out_idxs_tmp_alloc.unsafe_ptr()
    var out_idxs_tmp = TileTensor(
        out_idxs_tmp_ptr,
        row_major(Coord(internal_out_shape)),  # topk returns K as last dim
    )
    var reshaped_input = reshape(input, internal_in_shape)
    _top_k_cpu[dtype=dtype, largest=True](
        reshaped_input,
        max_k,
        axis=internal_rank - 1,  # Always operate on the last axis
        out_vals=reshaped_out_vals,
        out_idxs=out_idxs_tmp,
        sorted=True,
        parallelism_grain_size=1,
        k=k,
    )

    # Sample from the top K elements
    for batch in range(internal_bs):
        var temperature_val = Float32(1.0)
        if temperature:
            temperature_val = temperature.value()[batch][0]

        var k_val = max_k
        if k:
            var k_raw = Int(k.value()[batch])
            k_val = max_k if k_raw == -1 else k_raw

        # Clamp k_val to the number of valid top-k entries available in internal_out_vals
        var avail_k = Int(reshaped_out_vals.dim[1]())
        if k_val > avail_k:
            k_val = avail_k

        # Calculate softmax normalization
        var max_val = reshaped_out_vals[batch, 0][0]
        var sum_exp = Scalar[dtype](0)
        var exp_vals = alloc(AllocLayout[Scalar[dtype]](count=k_val))
        var exp_vals_ptr: UnsafePointer[
            Scalar[dtype], origin_of(exp_vals._alloc)
        ] = exp_vals.unsafe_ptr()
        var temp_val = temperature_val.cast[dtype]()
        for i in range(k_val):
            var val = reshaped_out_vals[batch, i][0]
            var exp_val = exp((val - max_val) / max(temp_val, 1e-6))
            exp_vals_ptr[i] = exp_val
            sum_exp += exp_val

        # Handle top_p parameter - extract scalar value from buffer
        var top_p_val = Scalar[dtype](1.0)
        if top_p:
            top_p_val = top_p.value()[batch][0].cast[dtype]()
        var _top_p = _adjust_top_p[dtype](
            top_p_val, exp_vals_ptr, k_val, sum_exp
        )

        # Handle seed parameter - extract scalar value from buffer
        var seed_val = UInt64(0)
        if seed:
            seed_val = seed.value()[batch][0]

        # Use the same RNG as the GPU sampling implementation
        var rng_state = Random(seed=seed_val)
        var rng = rng_state.step_uniform()

        # Sample using the normalized probabilities
        var r = sum_exp * _top_p * rng[0].cast[dtype]()
        for i in range(k_val):
            r -= exp_vals_ptr[i]
            if r <= 0 or i == k_val - 1:
                # Store the sampled index and value
                reshaped_out_idxs[batch, 0] = out_idxs_tmp[batch, i]
                break
        dealloc(exp_vals^)

        # Fill remaining positions with sentinel values for unused elements
        for remaining_k in range(k_val, max_k):
            if remaining_k < Int(reshaped_out_vals.dim[1]()):
                reshaped_out_vals[batch, remaining_k] = _topk_dead_val[
                    dtype, True
                ]()
            # Note: out_idxs for sampling only has 1 element in last dim, so no need to fill indices
    dealloc(out_idxs_tmp_alloc^)


@always_inline("nodebug")
def _topk_dead_val[T: DType, largest: Bool = True]() -> Scalar[T]:
    comptime if largest:
        return min_or_neg_inf[T]()
    else:
        return max_or_inf[T]()


# Define the TopK_2 structure to keep track of the top element per thread
@fieldwise_init
struct TopK_2[T: DType, largest: Bool = True](
    Defaultable, TrivialRegisterPassable
):
    """Tracks the single best (value, index) pair per thread during top-K reductions.

    Parameters:
        T: Data type of the tracked values.
        largest: Whether the best value is the maximum (top k) or minimum (bottom k).

    Fields:
        p: Flattened index of the tracked element.
        u: Value of the tracked element.
    """

    var p: Int  # flattened index of the element
    var u: Scalar[Self.T]  # value of the element

    def __init__(out self):
        """Initializes the tracker with a dead value and a zero index."""
        self.p = 0  # 0 to solve OOB
        self.u = _topk_dead_val[Self.T, Self.largest]()

    def insert(mut self, elem: Scalar[Self.T], elem_id: Int):
        """Replaces the tracked element when the candidate beats the current best.
        """
        comptime if Self.largest:
            if elem > self.u:
                self.u = elem
                self.p = elem_id
        else:
            if elem < self.u:
                self.u = elem
                self.p = elem_id


struct TopKHeap[T: DType, largest: Bool, M: Int]:
    """Fixed-capacity register heap for per-thread top-M tracking.

    Stores up to M (value, index) pairs in registers. During the scan
    phase, a cached threshold provides O(1) rejection of non-competitive
    elements. All internal loops are compile-time unrolled to keep data
    in registers on GPU. Indices are stored as Int32 to reduce register
    pressure for large block sizes.
    """

    var vals: Array[Scalar[Self.T], Self.M]
    var idxs: Array[Int32, Self.M]
    var threshold: Scalar[Self.T]

    @always_inline
    def __init__(out self):
        self.vals = Array[Scalar[Self.T], Self.M](
            fill=_topk_dead_val[Self.T, Self.largest]()
        )
        self.idxs = Array[Int32, Self.M](fill=Int32(-1))
        self.threshold = _topk_dead_val[Self.T, Self.largest]()

    @always_inline
    def insert(mut self, val: Scalar[Self.T], idx: Int):
        """Insert an element, evicting the worst if full."""
        # Fast reject against threshold. When the heap has empty slots
        # the threshold equals dead_val, so the check naturally fails
        # for all real values and we fall through to the empty-slot path.
        # Phrased as the negation of a strict compare so a NaN candidate is
        # rejected here too, matching `TopK_2.insert`. Accepting one makes it
        # the threshold, and `vals[i] == threshold` is false for NaN, so every
        # later candidate finds no slot to evict and is dropped.
        comptime if Self.largest:
            if not (val > self.threshold):
                return
        else:
            if not (val < self.threshold):
                return

        var idx32 = Int32(idx)

        # Try to place in an empty slot.
        var inserted = False
        comptime for i in range(Self.M):
            if not inserted and self.idxs[i] == -1:
                self.vals[i] = val
                self.idxs[i] = idx32
                inserted = True

        if not inserted:
            # All slots full — replace the first element at threshold.
            comptime for i in range(Self.M):
                if not inserted and self.vals[i] == self.threshold:
                    self.vals[i] = val
                    self.idxs[i] = idx32
                    inserted = True
        self._update_threshold()

    @always_inline
    def _update_threshold(mut self):
        """Recompute eviction threshold (worst value in the heap)."""
        self.threshold = self.vals[0]
        comptime for i in range(1, Self.M):
            comptime if Self.largest:
                if self.vals[i] < self.threshold:
                    self.threshold = self.vals[i]
            else:
                if self.vals[i] > self.threshold:
                    self.threshold = self.vals[i]

    @always_inline
    def best(self) -> TopK_2[Self.T, Self.largest]:
        """Return the best element, ties broken by smallest index.

        Returns a dead TopK_2 (p=-1) when all entries are exhausted.
        """
        var best_u = self.vals[0]
        var best_p = self.idxs[0]
        comptime for i in range(1, Self.M):
            comptime if Self.largest:
                if self.vals[i] > best_u or (
                    self.vals[i] == best_u and self.idxs[i] < best_p
                ):
                    best_u = self.vals[i]
                    best_p = self.idxs[i]
            else:
                if self.vals[i] < best_u or (
                    self.vals[i] == best_u and self.idxs[i] < best_p
                ):
                    best_u = self.vals[i]
                    best_p = self.idxs[i]
        return TopK_2[Self.T, Self.largest](p=Int(best_p), u=best_u)

    @always_inline
    def remove(mut self, idx: Int):
        """Remove element by global index, replacing with dead value."""
        var idx32 = Int32(idx)
        comptime for i in range(Self.M):
            if self.idxs[i] == idx32:
                self.vals[i] = _topk_dead_val[Self.T, Self.largest]()
                self.idxs[i] = Int32(-1)


# Function to perform warp-level reduction to find the maximum TopK_2
@always_inline
@__parameter
def _warp_reduce_topk[
    T: DType,
    largest: Bool,
    num_lanes: Int = WARP_SIZE,
    broadcast: Bool = False,
](val: TopK_2[T, largest]) -> TopK_2[T, largest]:
    """
    Performs warp-level reduction to find the maximum TopK_2 element.
    Uses shuffle down operations to efficiently compute the warp-wide
    maximum of TopK_2 values across all threads in a warp.

    Parameters:
        T: DType - Data type of the values being compared.
        largest: Bool - Whether to find the maximum or minimum value.
        num_lanes: Int - Number of lanes that participate in the reduction.
        broadcast: Bool - Whether to broadcast the result to all lanes.

    Arguments:
        val: TopK_2[T, largest] - TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T, largest] - Maximum TopK_2 value across the warp.
    """
    comptime assert (
        num_lanes.is_power_of_two()
    ), "num_lanes must be a power of two"

    var res = val

    # Shuffle function for TopK_2 structure
    @__parameter
    def shuffle_topk2(v: TopK_2[T, largest], offset: Int) -> TopK_2[T, largest]:
        comptime fn_type = def[dtype: DType, simd_width: SIMDLength](
            val: SIMD[dtype, simd_width], offset: UInt32
        ) thin -> SIMD[dtype, simd_width]
        comptime xor_fn: fn_type = warp.shuffle_xor
        comptime down_fn: fn_type = warp.shuffle_down

        comptime shuffle_fn = xor_fn if broadcast else down_fn

        return TopK_2[T, largest](
            u=shuffle_fn(v.u, UInt32(offset)),  # u is the value
            p=Int(shuffle_fn(Int32(v.p), UInt32(offset))),  # p is the index
        )

    @__parameter
    def reduce_fn(
        a: TopK_2[T, largest], b: TopK_2[T, largest]
    ) -> TopK_2[T, largest]:
        comptime if largest:
            if a.u > b.u:
                return a
            elif a.u < b.u:
                return b
            return a if a.p < b.p else b
        else:
            if a.u < b.u:
                return a
            elif a.u > b.u:
                return b
            return a if a.p < b.p else b

    # Reimplement `warp_reduce` for TopK_2 reduce and shuffle function
    comptime limit = log2_floor(num_lanes)

    comptime for i in reversed(range(limit)):
        comptime mask = 1 << i
        res = reduce_fn(res, shuffle_topk2(res, mask))

    return res


# Function to perform block-level reduction to find the maximum TopK_2
@always_inline
def _block_reduce_topk[
    T: DType,
    //,
    ascending: Bool,
    MAX_BLOCK_SIZE: Int = WARP_SIZE if is_apple_gpu() else 1024,
](val: TopK_2[T, ascending]) -> TopK_2[T, ascending]:
    """
    Performs a block-level reduction to find the maximum TopK_2 element.

    This function takes a TopK_2 value from each thread in a block and performs
    a reduction to find the maximum across all threads. It uses shared memory
    and warp-level reductions to efficiently compute the block-wide maximum.

    Parameters:
        T: DType - The data dtype of the values being compared.
        ascending: Bool - Whether to find the maximum or minimum value.
        MAX_BLOCK_SIZE: Int - The maximum number of threads in a block.

    Arguments:
        val: TopK_2[T, ascending] - The TopK_2 value from each thread to be reduced.

    Returns:
        TopK_2[T, ascending] - The maximum TopK_2 value across all threads in the block.

    Note:
    This function assumes that BLOCK_SIZE is a multiple of WARP_SIZE.
    It uses shared memory to store intermediate results and performs
    a final warp-level reduction to compute the block-wide maximum.
    """
    comptime assert (
        MAX_BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must be a multiple of the warp size"

    # Calculate sizes for shared memory allocation
    comptime p_width = simd_width_of[DType.int]()
    comptime u_width = simd_width_of[Scalar[T]]()

    # Allocate shared memory for indices and values
    var p_sram = unsafe_stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * p_width,
        Int,
        address_space=.SHARED,
    ]()
    var u_sram = unsafe_stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * u_width,
        Scalar[T],
        address_space=.SHARED,
    ]()

    # Calculate warp id and thread information
    var warp = warp_id()
    comptime num_warps_needed = MAX_BLOCK_SIZE // WARP_SIZE

    # Each warp reduces its own TopK_2 value
    var warp_accum: TopK_2[T, ascending] = _warp_reduce_topk[T, ascending](val)

    # Store warp-level results in shared memory
    if lane_id() == 0 and warp < num_warps_needed:
        # Note: Potential bank conflict for sub 4 byte data elements
        p_sram[warp * p_width] = Int(warp_accum.p)
        u_sram[warp * u_width] = warp_accum.u
    barrier()

    # Load warp results into final warp for block-level reduction
    var block_accum = TopK_2[T, ascending]()
    var thread_in_final_warp = thread_idx.x < ufloordiv(block_dim.x, WARP_SIZE)
    if thread_in_final_warp:
        var p_idx = p_sram[lane_id() * p_width]  # loaded value is a scalar
        block_accum = TopK_2[T, ascending](
            p=Int(p_idx),  # Convert back to int
            u=u_sram[lane_id() * u_width],
        )
    else:
        # Initialize unused threads with dummy values
        block_accum.p = 0
        block_accum.u = _topk_dead_val[T, ascending]()

    # Perform final warp-level reduction for block result
    return _warp_reduce_topk[T, ascending](block_accum)


@__name(t"topk_stage1_{T}_{out_idx_type}_{largest}")
def _topk_stage1[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    K: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
    max_k: Int32,
    num_elements: Int32,
    num_blocks_per_input: Int32,
    in_buffer_tmp: UnsafePointer[Scalar[T], MutAnyOrigin],
    local_topk_vals: UnsafePointer[
        Scalar[T], MutAnyOrigin
    ],  # Output buffer of size num_blocks_per_input * max_k
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type], MutAnyOrigin
    ],  # Output buffer of size num_blocks_per_input * max_k
):
    """
    Computes the Top-K elements within each block.

    This kernel function is the first stage of a two-stage Top-K algorithm.
    Each thread block processes a portion of the input data and finds its local top-K elements.
    The local top-K results are stored in global memory for further processing in stage 2.

    The input data must be pre-copied into in_buffer_tmp before launching this kernel
    (via device-to-device DMA copy), allowing the copy engine to operate in parallel.

    Parameters:
        T: Data type of the elements.
        out_idx_type: DType - The data dtype of the output indices.
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        K: Number of top elements to select per block. Varies for each batch element.
        max_k: Largest number of top elements to keep for each batch element.
        num_elements: Size of last dimension of input buffer (vocab size).
        num_blocks_per_input: Number of blocks used to process the input data.
        in_buffer_tmp: Pre-copied input buffer to read and modify during top-K.
        local_topk_vals: Output buffer to store the local top-K values.
        local_topk_idxs: Output buffer to store the indices of local top-K elements.

    Note:
        The output buffers (local_topk_vals and local_topk_idxs) should be of size num_blocks_per_input * max_k.
    """
    var _max_k = Int(max_k)
    var _num_elements = Int(num_elements)
    var _num_blocks_per_input = Int(num_blocks_per_input)

    var tid = thread_idx.x
    var bid = block_idx.x
    var block_size = block_dim.x

    var batch_id, block_lane = udivmod(bid, _num_blocks_per_input)

    var block_offset = block_lane * block_size
    var stride = block_size * _num_blocks_per_input

    var _in_buffer_tmp = in_buffer_tmp + batch_id * _num_elements

    # Hoist per-block output base pointers out of the k loop.
    var out_vals = local_topk_vals + bid * _max_k
    var out_idxs = local_topk_idxs + bid * _max_k

    var k_batch = _max_k
    if K:
        var k_raw = Int(K.unsafe_value()[batch_id])
        k_batch = _max_k if k_raw == -1 else k_raw

    # Clamp k_batch to the number of elements we can actually draw from
    if k_batch > _num_elements:
        k_batch = _num_elements

    # Shared memory to broadcast the winner index so the owning thread
    # can write the dead value (better L1 locality than thread 0).
    var winner_sram = unsafe_stack_allocation[1, Int, address_space=.SHARED]()

    comptime HEAP_SIZE = 8

    with PDL():
        # Phase 1: Single scan to build per-thread register heap.
        var heap = TopKHeap[T, largest, HEAP_SIZE]()
        for i in range(tid + block_offset, _num_elements, stride):
            heap.insert(_in_buffer_tmp[i], i)

        # Phase 2: Extract winners from heaps without re-scanning.
        # Threads whose heap is exhausted fall back to a global-memory
        # re-scan so that non-top-M elements are still discoverable.
        var heap_iters = min(k_batch, HEAP_SIZE)
        for k in range(heap_iters):
            # Use heap if it has valid entries, else fall back to re-scan.
            var partial = heap.best()
            if partial.p < 0:
                partial = TopK_2[T, largest]()
                for i in range(tid + block_offset, _num_elements, stride):
                    partial.insert(_in_buffer_tmp[i], i)

            var total = _block_reduce_topk[ascending=largest](partial)

            if tid == 0:
                out_vals[k] = total.u
                out_idxs[k] = Int(total.p).cast[out_idx_type]()
                winner_sram[0] = total.p
            barrier()

            var winner_p = winner_sram[0]
            if (
                partial.p == winner_p
                and partial.u != _topk_dead_val[T, largest]()
            ):
                heap.remove(winner_p)
                _in_buffer_tmp[winner_p] = _topk_dead_val[T, largest]()

        # Phase 3: Fallback to global-memory re-scan for remaining k.
        for k in range(heap_iters, k_batch):
            var partial = TopK_2[T, largest]()

            for i in range(tid + block_offset, _num_elements, stride):
                var val = _in_buffer_tmp[i]
                partial.insert(val, i)

            var total = _block_reduce_topk[ascending=largest](partial)

            if tid == 0:
                out_vals[k] = total.u
                out_idxs[k] = Int(total.p).cast[out_idx_type]()
                winner_sram[0] = total.p
            barrier()

            var winner_p = winner_sram[0]
            if (
                partial.p == winner_p
                and partial.u != _topk_dead_val[T, largest]()
            ):
                _in_buffer_tmp[winner_p] = _topk_dead_val[T, largest]()

        # Parallel sentinel fill using all threads.
        for remaining_k in range(k_batch + tid, _max_k, block_size):
            out_vals[remaining_k] = _topk_dead_val[T, largest]()
            out_idxs[remaining_k] = Scalar[out_idx_type](-1)


@__name(t"topk_stage2_{T}_{out_idx_type}_{sampling}_{largest}")
def _topk_stage2[
    T: DType,
    out_idx_type: DType,
    sampling: Bool = True,
    largest: Bool = True,
](
    K: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
    max_k: Int32,
    num_blocks_per_input: Int32,
    local_topk_vals: UnsafePointer[
        Scalar[T], ImmutAnyOrigin
    ],  # Input array of size n_batch * num_blocks_per_input * K
    local_topk_idxs: UnsafePointer[
        Scalar[out_idx_type], ImmutAnyOrigin
    ],  # Input array of size n_batch * num_blocks_per_input * K
    global_topk_vals: UnsafePointer[
        Scalar[T], MutAnyOrigin
    ],  # sampling ? undefined : output array of size K
    global_topk_idxs: UnsafePointer[
        Scalar[out_idx_type], MutAnyOrigin
    ],  # sampling ? sampled token : Output array of size K
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    top_p: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    min_p: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    seed: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]],
    valid: Optional[UnsafePointer[Int8, MutAnyOrigin]] = None,
):
    """
    Computes the global Top-K elements from the local Top-K results produced by stage 1.

    This kernel is designed to be executed with a single block, performing the final
    reduction step to obtain the global Top-K elements.

    Parameters:
        T: Data type of the elements.
        out_idx_type: DType - The data dtype of the output indices.
        sampling: Bool - Whether to sample a token from the top-K distribution.
        largest: Bool - Whether to find the maximum or minimum value.

    Args:
        K: Number of top elements to select per batch element.
        max_k: Largest number of top elements to keep for each batch element.
        num_blocks_per_input: Number of blocks used in stage 1.
        local_topk_vals: Pointer to local Top-K values from stage 1 (size: batch_size * num_blocks_per_input * K).
        local_topk_idxs: Pointer to local Top-K indices from stage 1 (size: batch_size * num_blocks_per_input * K).
        global_topk_vals: Pointer to store the final global Top-K values (size: batch_size * K).
        global_topk_idxs: Pointer to store the final global Top-K indices (size: batch_size * (1 if sampling else K)).
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        min_p: Per-row min-p threshold. Tokens with probability below
            ``min_p * max_prob`` are excluded from sampling.
        seed: The seed to use for the random number generator.
        valid: Optional per-row validity flags (1 = valid, 0 = invalid).
            The sampling kernel writes 0 for rows where no finite logit
            was found (e.g. all-NaN rows); rows default to 1 via memset.

    The function uses shared memory to store and process the local Top-K results,
    and performs a block-level reduction to find the global Top-K elements.
    """
    var _max_k = Int(max_k)
    var _num_blocks_per_input = Int(num_blocks_per_input)
    # compute the total number of elements reduced from stage 1
    var num_elem_reduced = _num_blocks_per_input * _max_k

    var tid = thread_idx.x
    var batch_id = block_idx.x
    # assert (block_idx.x == 0)
    # assert (grid_dim.x == 1)
    var batch_i_topk_vals = global_topk_vals + batch_id * _max_k
    var batch_i_topk_idxs = global_topk_idxs + batch_id * (
        1 if sampling else _max_k
    )
    var _local_topk_vals = local_topk_vals + batch_id * num_elem_reduced
    var _local_topk_idxs = local_topk_idxs + batch_id * num_elem_reduced

    # Allocate shared memory for values and indices
    var num_e_rounded = align_up(num_elem_reduced, WARP_SIZE)
    var vals_smem_size = num_e_rounded
    var vals_sram = unsafe_stack_allocation[
        _APPLE_STATIC_SHMEM_USABLE_COUNT[TopK_2[T]],
        Scalar[T],
        address_space=.SHARED,
    ]() if comptime (is_apple_gpu()) else external_memory[
        Scalar[T],
        address_space=.SHARED,
        alignment=align_of[Scalar[T]](),
    ]()

    var idxs_sram = (vals_sram + vals_smem_size).bitcast[Int]()

    # SAFETY: Only dereferenced inside `comptime if sampling` blocks;
    # overwritten with real pointers before use when sampling is enabled.
    var s_val2 = type_of(vals_sram).unsafe_dangling()
    var s_id = type_of(idxs_sram).unsafe_dangling()

    with PDL():
        # Handle the case where stage 1 is executed with a single block
        var k_batch = _max_k
        if K:
            var k_raw = Int(K.unsafe_value()[batch_id])
            k_batch = _max_k if k_raw == -1 else k_raw

        # Clamp k_batch to not exceed the reduced elements per batch and _max_k
        if k_batch > num_elem_reduced:
            k_batch = num_elem_reduced

        if _num_blocks_per_input == 1 and not sampling:
            if tid < k_batch:
                batch_i_topk_vals[tid] = _local_topk_vals[tid]
                # cast to out_idx_type
                batch_i_topk_idxs[tid] = _local_topk_idxs[tid]
            elif tid >= k_batch and tid < _max_k:
                # Fill unused positions with sentinel values
                batch_i_topk_vals[tid] = _topk_dead_val[T, largest]()
                batch_i_topk_idxs[tid] = Scalar[out_idx_type](-1)
            return

        comptime if sampling:
            # Storing the top-K logits in shmem for sampling
            s_id = (idxs_sram + vals_smem_size).bitcast[Int]()
            # The 2* below is for warp align safety
            s_val2 = (s_id + 2 * k_batch).bitcast[Scalar[T]]()

        var s_sum = unsafe_stack_allocation[
            1, Scalar[T], address_space=.SHARED
        ]()
        s_sum[0] = Scalar[T](0)
        var max_logit = Scalar[T](0)

        # Cache local top-K results from stage 1 into shared memory
        for i in range(tid, num_elem_reduced, block_dim.x):
            vals_sram[i] = _local_topk_vals[i]
            idxs_sram[i] = i
        barrier()

        for k in range(_max_k):
            if k >= k_batch:
                # Fill remaining positions with sentinel values for unused elements
                comptime if not sampling:
                    if tid == 0:
                        for remaining_k in range(k, _max_k):
                            batch_i_topk_vals[remaining_k] = _topk_dead_val[
                                T, largest
                            ]()
                            batch_i_topk_idxs[remaining_k] = Scalar[
                                out_idx_type
                            ](-1)
                else:
                    if tid == 0:
                        for remaining_k in range(k, _max_k):
                            batch_i_topk_vals[remaining_k] = _topk_dead_val[
                                T, largest
                            ]()
                        # Skip-token sentinel: use 0, not -1. This index is the
                        # sampled token returned downstream and is used as an
                        # array index (gather_nd / embedding lookup), where a
                        # negative index would read out of bounds.
                        batch_i_topk_idxs[0] = Scalar[out_idx_type](0)
                break

            # Re-initialize partial for each thread
            var partial = TopK_2[T, largest]()
            # TODO: unroll this
            for i in range(tid, num_elem_reduced, block_dim.x):
                partial.insert(vals_sram[i], i)

            barrier()
            # Perform block-level reduction to find the maximum TopK_2
            var total = _block_reduce_topk[ascending=largest](partial)

            if tid == 0:
                comptime if sampling:
                    if k == 0:
                        max_logit = total.u

                # Remove the found maximum from consideration in the next iteration
                idxs_sram[total.p] = -1
                vals_sram[total.p] = _topk_dead_val[T, largest]()

                comptime if sampling:
                    comptime assert (
                        T.is_floating_point()
                    ), "T must be floating point for sampling"
                    batch_i_topk_vals[k] = total.u
                    s_id[k] = total.p
                    var temp_val = Float32(1.0)
                    if temperature:
                        temp_val = temperature.unsafe_value()[batch_id]
                    total.u = exp(
                        (total.u - max_logit) / max(temp_val.cast[T](), 1e-6)
                    )
                    s_val2[k] = total.u
                    s_sum[0] += total.u
                else:
                    # Store the global top-K values and indices
                    batch_i_topk_vals[k] = total.u
                    batch_i_topk_idxs[k] = _local_topk_idxs[total.p]

                # Early exit if no valid index
                if total.u == _topk_dead_val[T, largest]():
                    break
            barrier()

        # do sampling
        comptime if sampling:
            if tid == 0:
                # Apply min_p mask: zero out probs below min_p * max_prob.
                # Since s_val2[0] = exp(0/temp) = 1.0 is the max, the
                # threshold in this unnormalized softmax domain is simply
                # min_p_val.
                if min_p:
                    var min_p_val = Scalar[T](
                        min_p.unsafe_value()[batch_id].cast[T]()
                    )
                    if min_p_val > 0:
                        for ki in range(k_batch):
                            if s_val2[ki] < min_p_val:
                                s_sum[0] -= s_val2[ki]
                                s_val2[ki] = Scalar[T](0)

                var top_p_val = Scalar[T](1.0)
                if top_p:
                    top_p_val = top_p.unsafe_value()[batch_id].cast[T]()
                var _top_p = _adjust_top_p[T](
                    top_p_val, s_val2, k_batch, s_sum[0]
                )

                # Use the largest logit's id as the offset for the random number
                # generator, so that we don't use the same random number for every
                # token in the sequence.
                var seed_val = UInt64(0)
                if seed:
                    seed_val = seed.unsafe_value()[batch_id]
                var rng_state = Random(seed=seed_val)
                var rng = rng_state.step_uniform()
                var softmax_norm = s_sum[0]
                var r = softmax_norm * _top_p * rng[0].cast[T]()
                for ki in range(k_batch):
                    var exp_logit = s_val2[ki]

                    r -= exp_logit
                    if r <= 0.0 or ki == k_batch - 1:
                        # uncomment below to return prob of largest logit
                        # batch_i_topk_vals[0] = exp_logit / softmax_norm
                        var idx: Int = s_id[ki]
                        var token_id = _local_topk_idxs[idx]
                        batch_i_topk_idxs[0] = token_id

                        # post-validation
                        if valid:
                            if max_logit == _topk_dead_val[T, largest]():
                                valid.unsafe_value()[batch_id] = 0
                        break


def _topk_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    sampling: Bool = True,
    largest: Bool = True,
    KLayoutType: TensorLayout = RowMajorLayout[Int64],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[Int64],
    TopPLayoutType: TensorLayout = RowMajorLayout[Int64],
    MinPLayoutType: TensorLayout = RowMajorLayout[Int64],
    SeedLayoutType: TensorLayout = RowMajorLayout[Int64],
    KEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPEngine: TensorEngine = DefaultEngine[element_width=1],
    MinPEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    max_k: Int,
    input_buf: TileTensor[mut=False, dtype, ...],
    device_local_topk_vals: TileTensor[dtype, ...],
    device_local_topk_idxs: TileTensor[out_idx_type, ...],
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    k: Optional[
        TileTensor[.int64, KLayoutType, ImmutAnyOrigin, Engine=KEngine]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    block_size: Int = 256,
    num_blocks_per_input: Optional[Int] = None,
    top_p: Optional[
        TileTensor[.float32, TopPLayoutType, ImmutAnyOrigin, Engine=TopPEngine]
    ] = None,
    min_p: Optional[
        TileTensor[.float32, MinPLayoutType, ImmutAnyOrigin, Engine=MinPEngine]
    ] = None,
    seed: Optional[
        TileTensor[.uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine]
    ] = None,
    valid: Optional[UnsafePointer[Int8, MutAnyOrigin]] = None,
) raises:
    """Computes the Top-K elements from the input tensor using a GPU-accelerated two-stage algorithm.

    This function implements a two-stage Top-K algorithm:
    1. Stage 1 (_topk_stage1): Divides the input into blocks and computes local Top-K for each block.
    2. Stage 2 (_topk_stage2): Merges the local Top-K results to obtain the global Top-K.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        out_idx_type: DType - The data dtype of the output indices (default == .int).
        sampling: Bool - Whether to return token samples from topK dist (default is True).
        largest: Bool - Whether to find the maximum or minimum value.
        KLayoutType: Layout type of the k buffer.
        TemperatureLayoutType: Layout type of the temperature buffer.
        TopPLayoutType: Layout type of the top_p buffer.
        MinPLayoutType: Layout type of the min_p buffer.
        SeedLayoutType: Layout type of the seed buffer.
        KEngine: Engine of the k buffer.
        TemperatureEngine: Engine of the temperature buffer.
        TopPEngine: Engine of the top_p buffer.
        MinPEngine: Engine of the min_p buffer.
        SeedEngine: Engine of the seed buffer.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        max_k: Int
            Largest number of top elements to keep for each batch element.
        input_buf: TileTensor[dtype, [batch_size, N]]
            Input tensor as a device TileTensor.
        device_local_topk_vals: TileTensor[dtype, [batch_size, num_blocks_per_input * max(K)]]
            Temporary buffer for locally reduced top-K values from stage 1.
        device_local_topk_idxs: TileTensor[.int, [batch_size, num_blocks_per_input * max(K)]]
            Temporary buffer for locally reduced top-K indices from stage 1.
        out_vals: TileTensor[dtype, [batch_size, max(K)]]
            Output buffer on device for the K largest values.
        out_idxs: TileTensor[.int, [batch_size, 1 if sampling else max(K)]]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
        k: Optional TileTensor[.int64]
            Device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling for each batch element.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: Optional[Int]
            Number of blocks per input (default computed from input size and block size).
            This is the equivalent of "BLOCKS_PER_BEAM" in TRT-LLM kernel allowing for much larger
            batch sizes through packing several elements per thread in the first stage.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        min_p: Per-row min-p threshold. Tokens with probability below
            ``min_p * max_prob`` are excluded from sampling.
        seed: The seed to use for the random number generator.
        valid: Optional per-row validity flags (1 = valid, 0 = invalid).
            The sampling kernel writes 0 for rows where no finite logit
            was found (e.g. all-NaN rows); rows default to 1 via memset.

    The implementation uses shared memory and warp-level primitives for efficient GPU execution.
    It's modeled from the following similar algos in [InternLM]
    (https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/sampling_topk_kernels.cu)
    and [TRT-LLM]
    (https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/samplingTopKKernels.cu).

    """
    comptime assert input_buf.rank == 2, "rank must be 2"
    comptime assert not (
        sampling and not largest
    ), "sampling not supported for largest=False"
    comptime assert (
        input_buf.rank == out_vals.rank
    ), "input.rank must match out_vals.rank"
    comptime assert (
        input_buf.rank == out_idxs.rank
    ), "input.rank must match out_idx.rank"

    # Use largest number of threads per block
    var batch_size = Int(input_buf.dim[0]()) if input_buf.rank == 2 else 1
    var N = Int(input_buf.dim[1]())

    # Do not launch gpu kernels with grid_dim = 0
    if batch_size == 0:
        return

    # On Apple GPUs, clamp block_size to a single warp so the shared-memory
    # top-k kernels stay within Apple's static shared-memory budget, then
    # recompute blocks.
    var effective_block_size = block_size
    comptime if has_apple_gpu_accelerator():
        effective_block_size = WARP_SIZE

    # Define the number of blocks per grid
    var num_blocks_per_input_: Int = ceildiv(
        N, effective_block_size
    ) if not num_blocks_per_input else num_blocks_per_input.value()
    # Calculate largest num bytes of shmem for each stage
    if effective_block_size % WARP_SIZE != 0:
        # TODO: Need to pad in this case
        raise Error("block_size must be a multiple of WARP_SIZE")

    # Define grid and block dimensions for stage 1
    var grid_dim_stage1 = num_blocks_per_input_ * batch_size
    var block_dim_stage1 = effective_block_size

    # Handle optional k parameter
    var k_ptr: Optional[UnsafePointer[Int64, ImmutAnyOrigin]] = None
    if k:
        k_ptr = rebind[UnsafePointer[Int64, ImmutAnyOrigin]](k.value().ptr)

    # Enqueue the first kernel (stage 1)
    var input_buf_tmp = ctx.enqueue_create_buffer[dtype](batch_size * N)
    # Use DMA copy engine instead of kernel-based copy
    ctx.enqueue_copy(input_buf_tmp, input_buf.to_device_buffer(ctx))
    comptime kernel_1 = _topk_stage1[dtype, out_idx_type, largest]
    ctx.enqueue_function[kernel_1](
        k_ptr,
        Int32(max_k),
        Int32(N),
        Int32(num_blocks_per_input_),
        input_buf_tmp,
        device_local_topk_vals.to_device_buffer(ctx),
        device_local_topk_idxs.to_device_buffer(ctx),
        grid_dim=grid_dim_stage1,
        block_dim=block_dim_stage1,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
    _ = input_buf_tmp^

    var num_elem_reduced = align_up(num_blocks_per_input_ * max_k, WARP_SIZE)
    var num_bytes_sample_cache = max_k * (
        size_of[Scalar[dtype]]() + 2 * size_of[DType.int]()
    )
    var shared_mem_bytes_2 = (
        num_elem_reduced * (size_of[Scalar[dtype]]() + size_of[DType.int]())
        + num_bytes_sample_cache
    )
    # align to warp size
    shared_mem_bytes_2 = align_up(shared_mem_bytes_2, WARP_SIZE)
    comptime if has_apple_gpu_accelerator():
        if shared_mem_bytes_2 > _APPLE_STATIC_SHMEM_MAX_BYTES:
            raise Error(
                t"shared memory of {shared_mem_bytes_2} exceeds static"
                t" allocation capacity of {_APPLE_STATIC_SHMEM_MAX_BYTES} for"
                t" the second stage top-k kernel, consider reducing the"
                t" block_size or num_blocks_per_input"
            )

    # Define grid and block dimensions for stage 2
    var grid_dim_stage2 = (
        batch_size  # Single block since num_elements_stage2 is small
    )
    var block_dim_stage2 = block_size

    # Handle optional temperature parameter
    var temp_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
    if temperature:
        temp_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
            temperature.value().ptr
        )

    # Handle optional top_p parameter
    var top_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
    if top_p:
        top_p_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
            top_p.value().ptr
        )

    # Handle optional min_p parameter
    var min_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
    if min_p:
        min_p_ptr = rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
            min_p.value().ptr
        )

    # Handle optional seed parameter
    var seed_ptr: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]] = None
    if seed:
        seed_ptr = seed.value().ptr

    # Enqueue the second kernel (stage 2)
    comptime kernel_2 = _topk_stage2[dtype, out_idx_type, sampling, largest]
    ctx.enqueue_function[kernel_2](
        k_ptr,
        Int32(max_k),
        Int32(num_blocks_per_input_),
        device_local_topk_vals.to_device_buffer(ctx),
        device_local_topk_idxs.to_device_buffer(ctx),
        out_vals.to_device_buffer(ctx),
        out_idxs.to_device_buffer(ctx),
        temp_ptr,
        top_p_ptr,
        min_p_ptr,
        seed_ptr,
        valid,
        grid_dim=grid_dim_stage2,
        block_dim=block_dim_stage2,
        shared_mem_bytes=shared_mem_bytes_2,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@always_inline
def topk_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    sampling: Bool = True,
    largest: Bool = True,
    KLayoutType: TensorLayout = RowMajorLayout[Int64],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[Int64],
    TopPLayoutType: TensorLayout = RowMajorLayout[Int64],
    MinPLayoutType: TensorLayout = RowMajorLayout[Int64],
    SeedLayoutType: TensorLayout = RowMajorLayout[Int64],
    KEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPEngine: TensorEngine = DefaultEngine[element_width=1],
    MinPEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    max_k: Int,
    input: TileTensor[mut=False, dtype, ...],
    out_vals: TileTensor[mut=True, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    block_size: Optional[Int] = None,
    num_blocks_per_input: Optional[Int] = None,
    k: Optional[
        TileTensor[.int64, KLayoutType, ImmutAnyOrigin, Engine=KEngine]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    top_p: Optional[
        TileTensor[.float32, TopPLayoutType, ImmutAnyOrigin, Engine=TopPEngine]
    ] = None,
    min_p: Optional[
        TileTensor[.float32, MinPLayoutType, ImmutAnyOrigin, Engine=MinPEngine]
    ] = None,
    seed: Optional[
        TileTensor[.uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine]
    ] = None,
    valid: Optional[UnsafePointer[Int8, MutAnyOrigin]] = None,
) raises:
    """
    Generalized implementation of the Top K algorithm with/without sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume or the top K values and indices across the tensor.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        out_idx_type: DType - The data dtype of the output indices (default == .int).
        sampling: Bool - Whether to return token samples from topK dist (default is True).
        largest: Bool - Whether to find the maximum or minimum value.
        KLayoutType: Layout type of the k buffer.
        TemperatureLayoutType: Layout type of the temperature buffer.
        TopPLayoutType: Layout type of the top_p buffer.
        MinPLayoutType: Layout type of the min_p buffer.
        SeedLayoutType: Layout type of the seed buffer.
        KEngine: Engine of the k buffer.
        TemperatureEngine: Engine of the temperature buffer.
        TopPEngine: Engine of the top_p buffer.
        MinPEngine: Engine of the min_p buffer.
        SeedEngine: Engine of the seed buffer.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        max_k: Int
            Largest number of top elements to keep for each batch element.
        input: TileTensor[dtype]
            Input tensor as a device TileTensor.
        out_vals: TileTensor[dtype]
            Output buffer on device for the K largest values.
        out_idxs: TileTensor[.int]
            Output buffer on device for the indices of the K largest values, or sampled token indices.
            Last dimension is 1 if sampling is True, otherwise K.
        block_size: Int
            The number of threads per block (default is 256 from TRT and empirical testing).
        num_blocks_per_input: Optional[Int]
            Number of blocks per input (default computed from input size and block size).
            This is the equivalent of "BLOCKS_PER_BEAM" in TRT-LLM kernel allowing for much larger
            batch sizes through packing several elements per thread in the first stage.
        k: Optional TileTensor[.int64]
            Device buffer of top elements to keep for each batch element.
        temperature: The temperature based scaling.
        top_p: Only use the tokens whose cumulative probability exceeds this threshold.
        min_p: Per-row min-p threshold. Tokens with probability below
            ``min_p * max_prob`` are excluded from sampling.
        seed: The seed to use for the random number generator.
        valid: Optional per-row validity flags (1 = valid, 0 = invalid).
            The sampling kernel writes 0 for rows where no finite logit
            was found (e.g. all-NaN rows); rows default to 1 via memset.
    """
    comptime assert input.rank > 0, "Input rank must be positive"
    var orig_in_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("input", orig_in_shape, dtype),
                    "max_k=" + String(max_k),
                    "sampling=" + String(sampling),
                    "largest=" + String(largest),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_gpu",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var N = orig_in_shape[input.rank - 1]
        var last_idx_dim = 1 if sampling else max_k

        # Clamp max_k
        var bound_max_k = 255 if max_k == -1 else max_k

        # heuristic to set block size
        var block_size_: Int
        if input.num_elements() <= 1024 * 64 * 3:
            block_size_ = 256
        elif input.num_elements() <= 32000 * 256:
            block_size_ = 512
        else:
            block_size_ = 1024
        block_size_ = block_size.value() if block_size else block_size_

        # On Apple GPUs, clamp block_size to a single warp so the shared-memory
        # top-k kernels stay within Apple's static shared-memory budget.
        comptime if has_apple_gpu_accelerator():
            block_size_ = min(block_size_, WARP_SIZE)

        # This section handles different input ranks by reshaping to a 2D tensor
        var internal_bs: Int  # Internal batch size
        comptime internal_rank = 2  # We always reshape to 2D for internal processing
        var internal_input: TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, 2].element_types,
                stride_types=DynamicCoord[.int64, 2].element_types,
            ],
            input.origin,
            address_space=input.address_space,
            Engine=input.Engine,
        ]
        var internal_out_idxs: TileTensor[
            out_idx_type,
            Layout[
                shape_types=DynamicCoord[.int64, 2].element_types,
                stride_types=DynamicCoord[.int64, 2].element_types,
            ],
            out_idxs.origin,
            address_space=out_idxs.address_space,
            Engine=out_idxs.Engine,
        ]
        var internal_out_vals: TileTensor[
            dtype,
            Layout[
                shape_types=DynamicCoord[.int64, 2].element_types,
                stride_types=DynamicCoord[.int64, 2].element_types,
            ],
            out_vals.origin,
            address_space=out_vals.address_space,
            Engine=out_vals.Engine,
        ]

        comptime if input.rank == 1:
            # Handle 1D input: treat it as a single batch with one element
            internal_bs = 1
            var internal_in_shape = IndexList[internal_rank](
                1, input.num_elements()
            )
            var internal_out_vals_shape = IndexList[internal_rank](
                1, bound_max_k
            )
            var internal_out_idxs_shape = IndexList[internal_rank](
                1, last_idx_dim
            )
            # Reshape 1D inputs to 2D
            internal_input = reshape(input, internal_in_shape)
            internal_out_idxs = reshape(out_idxs, internal_out_idxs_shape)
            internal_out_vals = reshape(out_vals, internal_out_vals_shape)
        elif input.rank == internal_rank:
            # Input is already 2D, no reshaping needed
            internal_bs = orig_in_shape[0]
            internal_input = rebind[type_of(internal_input)](
                input.make_dynamic[.int64]()
            )
            internal_out_idxs = rebind[type_of(internal_out_idxs)](
                out_idxs.make_dynamic[.int64]()
            )
            internal_out_vals = rebind[type_of(internal_out_vals)](
                out_vals.make_dynamic[.int64]()
            )
        else:  # rank > 2
            # Handle higher dimensional inputs by flattening all but the last dimension
            var _last_dim = orig_in_shape[input.rank - 1]
            internal_bs = Int(
                Float64(orig_in_shape.flattened_length()) / Float64(_last_dim)
            )

            var internal_in_shape = IndexList[internal_rank](
                internal_bs, _last_dim
            )
            var internal_out_idxs_shape = IndexList[internal_rank](
                internal_bs, last_idx_dim
            )
            var internal_out_vals_shape = IndexList[internal_rank](
                internal_bs, bound_max_k
            )

            # Reshape higher dimensional inputs to 2D
            internal_input = reshape(input, internal_in_shape)
            internal_out_idxs = reshape(out_idxs, internal_out_idxs_shape)
            internal_out_vals = reshape(out_vals, internal_out_vals_shape)

        # Calculate the number of blocks per input
        var num_blocks_per_input_ = min(
            ceildiv(N, block_size_), 8
        ) if not num_blocks_per_input else num_blocks_per_input.value()

        # Bound stage-2 shared memory: `_topk_stage2` reduces
        # `num_blocks_per_input * max_k` candidates inside a SINGLE block's
        # dynamic shared memory. For large `max_k` (e.g. the MLA indexer's
        # top_k=2048) the default of 8 blocks makes that request exceed the
        # device per-block shared-memory limit, so the launch fails with
        # CUDA_ERROR_INVALID_VALUE. Reduce `num_blocks_per_input` until the
        # stage-2 request fits a conservative device budget. This runs BEFORE
        # the cache buffers are sized below, so the buffers, the stage-1 grid
        # and the stage-2 shared memory all stay consistent. It is a no-op for
        # the common small-`max_k` case (8 blocks already fit).
        var _topk_val_idx_bytes = (
            size_of[Scalar[dtype]]() + size_of[DType.int]()
        )
        var _topk_cache_bytes = (
            size_of[Scalar[dtype]]() + 2 * size_of[DType.int]()
        )
        var _topk_smem_budget = (
            Int(ctx.default_device_info.shared_memory_per_multiprocessor) - 8192
        )
        if bound_max_k > 0 and num_blocks_per_input_ > 1:
            var _topk_max_nb = (
                _topk_smem_budget - bound_max_k * _topk_cache_bytes
            ) // (bound_max_k * _topk_val_idx_bytes)
            if _topk_max_nb < 1:
                _topk_max_nb = 1
            if num_blocks_per_input_ > _topk_max_nb:
                num_blocks_per_input_ = _topk_max_nb

        # Define shape for the kernel's internal cache buffers
        var internal_cache_shape = IndexList[2](
            internal_bs, num_blocks_per_input_ * bound_max_k
        )

        # Create temporary buffer for local top-K values
        var internal_vals_buf = ctx.enqueue_create_buffer[dtype](
            product(internal_cache_shape)
        )
        var device_local_topk_vals = TileTensor(
            internal_vals_buf,
            row_major(Coord(internal_cache_shape)),
        )

        # Create temporary buffer for local top-K indices
        var internal_idxs_buf = ctx.enqueue_create_buffer[out_idx_type](
            product(internal_cache_shape)
        )
        var device_local_topk_idxs = TileTensor(
            internal_idxs_buf,
            row_major(Coord(internal_cache_shape)),
        )

        _topk_gpu[
            dtype=dtype,
            out_idx_type=out_idx_type,
            sampling=sampling,
            largest=largest,
        ](
            ctx,
            bound_max_k,
            internal_input,
            device_local_topk_vals,
            device_local_topk_idxs,
            internal_out_vals,
            internal_out_idxs,
            k=k,
            temperature=temperature,
            block_size=block_size_,
            num_blocks_per_input=num_blocks_per_input_,
            top_p=top_p,
            min_p=min_p,
            seed=seed,
            valid=valid,
        )

        # Clean up buffers
        _ = internal_vals_buf^
        _ = internal_idxs_buf^


def _topk_topp_sampling_fi[
    dtype: DType,
    out_idx_type: DType,
    KLayoutType: TensorLayout = RowMajorLayout[Int64],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[Int64],
    TopPLayoutType: TensorLayout = RowMajorLayout[Int64],
    MinPLayoutType: TensorLayout = RowMajorLayout[Int64],
    SeedLayoutType: TensorLayout = RowMajorLayout[Int64],
    KEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPEngine: TensorEngine = DefaultEngine[element_width=1],
    MinPEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    max_k: Int,
    min_top_p: Float32,
    input: TileTensor[mut=False, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    k: Optional[
        TileTensor[out_idx_type, KLayoutType, ImmutAnyOrigin, Engine=KEngine]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    top_p: Optional[
        TileTensor[.float32, TopPLayoutType, ImmutAnyOrigin, Engine=TopPEngine]
    ] = None,
    min_p: Optional[
        TileTensor[.float32, MinPLayoutType, ImmutAnyOrigin, Engine=MinPEngine]
    ] = None,
    rng_seed: Optional[
        TileTensor[.uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine]
    ] = None,
) raises:
    """Top-K + top-P + min-P sampling.

    Performs top-k+top-p rejection sampling via the dual-pivot algorithm.
    Softmax with per-row temperature scaling and the optional min-p mask are
    fused into the sampling kernel (`from_logits=True`), so no intermediate
    [batch_size, d] probability buffer is materialized.
    """
    # Reshape out_idxs from [batch, 1] (rank 2) to [batch] (rank 1).
    var out_shape = coord_to_index_list(out_idxs.layout.shape_coord())
    var out_1d = TileTensor(
        out_idxs._storage,
        row_major(out_shape[0]),
    )
    topk_topp_sampling_from_prob[dtype, out_idx_type, from_logits=True](
        ctx,
        input,
        out_1d,
        max_k,
        top_p_val=min_top_p,
        top_k_arr=k,
        top_p_arr=top_p,
        rng_seed=rng_seed,
        temperature=temperature,
        min_p=min_p,
    )


@always_inline
def fused_token_sampling_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    KLayoutType: TensorLayout = RowMajorLayout[Int64],
    TemperatureLayoutType: TensorLayout = RowMajorLayout[Int64],
    TopPLayoutType: TensorLayout = RowMajorLayout[Int64],
    MinPLayoutType: TensorLayout = RowMajorLayout[Int64],
    SeedLayoutType: TensorLayout = RowMajorLayout[Int64],
    KEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPEngine: TensorEngine = DefaultEngine[element_width=1],
    MinPEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    max_k: Int,
    min_top_p: Float32,
    input: TileTensor[mut=False, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    block_size: Optional[Int] = None,
    num_blocks_per_input: Optional[Int] = None,
    k: Optional[
        TileTensor[.int64, KLayoutType, ImmutAnyOrigin, Engine=KEngine]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    top_p: Optional[
        TileTensor[.float32, TopPLayoutType, ImmutAnyOrigin, Engine=TopPEngine]
    ] = None,
    min_p: Optional[
        TileTensor[.float32, MinPLayoutType, ImmutAnyOrigin, Engine=MinPEngine]
    ] = None,
    seed: Optional[
        TileTensor[.uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine]
    ] = None,
) raises:
    """
    Top K algorithm with fused sampling.
    Returns the sampled indices from the Top-K of the innermost
    dimension of the input tensor for each row/subvolume.
    """

    var input_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("input", input_shape, dtype),
                    "max_k=" + String(max_k),
                    "min_top_p=" + String(min_top_p),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "fused_token_sampling_gpu",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        # If all items in the batch, want to sample all tokens (top_k==-1, top_p=1)
        # Use the fused kernel: generates Gumbel noise inline and argmax-reduces
        # in a single pass with no intermediate HBM buffer.
        if max_k == -1 and min_top_p == 1.0:
            gumbel_sampling_fused_gpu(
                ctx,
                input,
                out_idxs,
                temperature,
                seed,
            )
            return

        comptime assert (
            input.rank == out_idxs.rank
        ), "input.rank must match out_idx.rank"

        comptime assert input.flat_rank == 2

        var vocab_size = Int(input.layout.shape[1]().value())
        var adjusted_max_k = vocab_size if max_k == -1 else max_k

        if adjusted_max_k >= 10:
            _topk_topp_sampling_fi[dtype, out_idx_type](
                ctx,
                adjusted_max_k,
                min_top_p,
                input,
                out_idxs,
                k=rebind[
                    Optional[
                        TileTensor[
                            out_idx_type,
                            KLayoutType,
                            ImmutAnyOrigin,
                            Engine=KEngine,
                        ]
                    ]
                ](k),
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                rng_seed=seed,
            )
            return

        var out_vals_shape = coord_to_index_list(input.layout.shape_coord())
        out_vals_shape[input.rank - 1] = adjusted_max_k
        var out_vals_buf = ctx.enqueue_create_buffer[dtype](
            out_vals_shape.flattened_length()
        )
        var out_vals = TileTensor(
            out_vals_buf,
            row_major(Coord(out_vals_shape)),
        )

        var batch_size = input_shape[0]
        var valid_buf = Optional[DeviceBuffer[.int8]](None)
        var valid = Optional[UnsafePointer[Int8, MutAnyOrigin]](None)
        comptime if ASSERT_MODE == "all":
            valid_buf = ctx.enqueue_create_buffer[.int8](batch_size)
            ctx.enqueue_memset(valid_buf.value(), 1)
            valid = valid_buf.value().unsafe_ptr().as_unsafe_any_origin()

        topk_gpu[sampling=True, largest=True](
            ctx,
            adjusted_max_k,
            input,
            out_vals,
            out_idxs,
            k=k,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            block_size=block_size,
            num_blocks_per_input=num_blocks_per_input,
            seed=seed,
            valid=valid,
        )

        comptime if ASSERT_MODE == "all":
            var valid_host = ctx.enqueue_create_host_buffer[.int8](batch_size)
            ctx.enqueue_copy(valid_host, valid_buf.value())
            ctx.synchronize()

            for i in range(batch_size):
                if not valid_host[i]:
                    raise Error("NaN logits detected in batch row " + String(i))

        _ = valid_buf^
        _ = out_vals_buf^


# ===-----------------------------------------------------------------------===#
# Sampling Kernel with the Gumbel-max trick
# ===-----------------------------------------------------------------------===#


@__name(t"apply_gumbel_noise_{dtype}")
def apply_gumbel_noise_kernel[
    dtype: DType,
    OutputLayoutType: TensorLayout,
    InputLayoutType: TensorLayout,
    num_sms: Int,
    num_threads: Int,
](
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InputLayoutType, ImmutAnyOrigin],
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    seed: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]],
):
    """Adds Gumbel(0,1) noise to logits for sampling via the Gumbel-max trick.

    Parameters:
        dtype: Data type of the input and output logit buffers.
        OutputLayoutType: Layout of the output tensor.
        InputLayoutType: Layout of the input tensor.
        num_sms: Number of streaming multiprocessors to launch with.
        num_threads: Number of threads per block.

    Args:
        output: Output tensor of noised logits.
        input: Input tensor of logits.
        temperature: Optional per-token temperature scaling.
        seed: Optional per-token random seed.
    """
    comptime EPS = Float32(1e-20)
    comptime LOG2 = Float32(0.6931471806)
    comptime MIN_TEMP = Float32(1e-6)

    comptime simd_width = simd_width_of[dtype]()
    var N = Int(input.dim(1))
    comptime num_blocks_per_token = 8
    comptime group_size = num_blocks_per_token * num_threads
    comptime num_groups = num_sms // num_blocks_per_token

    var tid = thread_idx.x
    var sm_id = block_idx.x
    var group_id, sm_id_rem = divmod(sm_id, num_blocks_per_token)
    var tid_in_group = tid + sm_id_rem * num_threads

    var num_tokens = input.dim[0]()

    comptime assert (
        simd_width % 4 == 0
    ), "SIMD width must be divisible by 4 to match RNG output size."

    # split workload across blocks
    with PDL():
        if sm_id >= num_groups * num_blocks_per_token:
            return

        for tok_idx in range(group_id, Int(num_tokens), num_groups):
            var temp_val = Float32(1.0)
            if temperature:
                temp_val = temperature.unsafe_value()[tok_idx]
                temp_val = max(temp_val, MIN_TEMP)

            var seed_val = UInt64(0)
            if seed:
                seed_val = seed.unsafe_value()[tok_idx]

            var ld_ptr = input.ptr + tok_idx * N
            var st_ptr = output.ptr + tok_idx * N
            comptime align = align_of[SIMD[dtype, simd_width]]()

            for i in range(tid_in_group, N // simd_width, group_size):
                var rng_state = Random(
                    seed=seed_val * UInt64(N) + UInt64(i),
                )
                var input_val: SIMD[dtype, simd_width]
                if N % simd_width == 0:
                    input_val = ld_ptr.load[width=simd_width, alignment=align](
                        i * simd_width
                    )
                else:
                    input_val = ld_ptr.load[width=simd_width](i * simd_width)
                var noised_logits = input_val.cast[.float32]() / temp_val

                comptime for loop_i in range(simd_width // 4):
                    var rnd_val = rng_state.step_uniform()
                    rnd_val = -LOG2 * log2(-log2(rnd_val + EPS) + EPS)

                    comptime for vec_i in range(4):
                        noised_logits[4 * loop_i + vec_i] += rnd_val[vec_i]

                if N % simd_width == 0:
                    st_ptr.store[width=simd_width, alignment=align](
                        i * simd_width, noised_logits.cast[dtype]()
                    )
                else:
                    st_ptr.store[width=simd_width](
                        i * simd_width, noised_logits.cast[dtype]()
                    )

            # If N is not divisible by simd_width, handle remaining elements
            if N % simd_width != 0:
                var N_res = N % simd_width
                var rng_state = Random(
                    seed=seed_val * UInt64(N)
                    + UInt64(N - N_res)
                    + UInt64(tid_in_group),
                )
                if tid_in_group < N_res:
                    var input_val = ld_ptr.load(
                        (N - N_res) + tid_in_group
                    ).cast[.float32]()
                    var noised_logit = input_val / temp_val
                    var rnd_val = rng_state.step_uniform()[0]
                    rnd_val = -LOG2 * log2(-log2(rnd_val + EPS) + EPS)
                    noised_logit += rnd_val
                    st_ptr.store(
                        (N - N_res) + tid_in_group,
                        noised_logit.cast[dtype](),
                    )


# ===-----------------------------------------------------------------------===#
# Fused Gumbel-noise + argmax (single kernel, no intermediate HBM buffer)
# ===-----------------------------------------------------------------------===#
#
# Target families: NVIDIA (SM90/SM100) and AMD CDNA. One block per batch row,
# grid-strides over the vocab dimension, generates Gumbel(0,1) noise inline in
# registers (identical RNG/indexing to `apply_gumbel_noise_kernel`), feeds each
# noised logit into a per-thread `TopK_2` running max, then a block-level argmax
# reduction (`_block_reduce_topk`) writes the winning token to `out_idxs`.
#
# This fuses `apply_gumbel_noise_kernel` + the stage-1/2 argmax of
# `topk_gpu[..., max_k=1]` into a single launch, saving one full
# `[batch, vocab]` HBM round-trip (the noised-logits temp buffer the two-kernel
# path writes then reads back).
#
# Bit-identity contract: for vocab element `j` of token `b`, the noise added
# here is the SAME value the two-kernel path adds. Bit-identity holds because
# the noise for element `j` depends only on `(seed_val, N, j)` via the exact
# same SIMD-chunked RNG sequence (`Random(seed=seed_val*N + chunk_idx)` then
# `step_uniform()` per 4-lane group), independent of grid layout. The argmax
# winner is therefore identical (argmax is permutation-invariant; ties resolve
# to the lowest index in both paths via `TopK_2.insert`'s strict `>` and the
# block reduce keeping the first maximum).

comptime _GUMBEL_PARTIAL_BYTES = 16


@__name(
    t"gumbel_argmax_fused_{dtype}_{out_idx_type}_{from_probs}_{multi_block}"
)
def _gumbel_argmax_fused_kernel[
    dtype: DType,
    out_idx_type: DType,
    InputLayoutType: TensorLayout,
    OutIdxLayoutType: TensorLayout,
    from_probs: Bool = False,
    multi_block: Bool = False,
    InputEngine: TensorEngine = DefaultEngine[element_width=1],
    OutIdxEngine: TensorEngine = DefaultEngine[element_width=1],
](
    input: TileTensor[
        dtype, InputLayoutType, ImmutAnyOrigin, Engine=InputEngine
    ],
    out_idxs: TileTensor[
        mut=True,
        out_idx_type,
        OutIdxLayoutType,
        MutAnyOrigin,
        Engine=OutIdxEngine,
    ],
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    seed: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]],
    blocks_per_row: Int32 = Int32(1),
    partials: Optional[UnsafePointer[UInt8, MutAnyOrigin]] = None,
    counters: Optional[UnsafePointer[Int32, MutAnyOrigin]] = None,
):
    """Fused Gumbel-noise + argmax.

    Each thread grid-strides over the vocab dimension, applies Gumbel(0,1)
    noise to its logits using the same RNG sequence as
    `apply_gumbel_noise_kernel` (bit-identical), tracks a per-thread argmax in a
    `TopK_2`, and the block reduces to the global argmax which is written to
    `out_idxs[batch_id]`.

    With `from_probs` the input rows are unnormalized probabilities instead of
    logits: each value enters the race as `ln(p) + g`, so the winner is a
    categorical draw proportional to `p`. A zero probability maps to `-inf`
    and cannot win while the row has any positive mass. Row normalization does
    not matter, because a per-row constant shifts every `ln(p)` equally.
    `temperature` is ignored in this mode.

    Parameters:
        dtype: Element type of the input logits.
        out_idx_type: Output index dtype.
        InputLayoutType: Layout of the `[batch, vocab]` input.
        OutIdxLayoutType: Layout of the `[batch, 1]` output indices.
        from_probs: Treat the input as unnormalized probabilities.
        multi_block: Split each input row across multiple blocks.
        InputEngine: Engine of the `[batch, vocab]` input.
        OutIdxEngine: Engine of the `[batch, 1]` output indices.

    Args:
        input: Input logits `[batch, vocab]`.
        out_idxs: Output sampled indices `[batch, 1]`.
        temperature: Optional per-row temperature scaling `[batch]`.
        seed: Optional per-row random seeds `[batch]`.
        blocks_per_row: Number of blocks assigned to each row.
        partials: Cross-block partial score and index storage.
        counters: Zero-initialized per-row arrival counters.
    """
    # `_block_reduce_topk` caps its shared storage at WARP_SIZE on Apple while
    # this kernel launches `max_thread_block_size` threads, so the reduction
    # would not cover the block. The logits path predates that mismatch; the
    # from-probs path is new surface and refuses to build on it.
    comptime assert (
        not from_probs or not is_apple_gpu()
    ), "from_probs is not supported on Apple GPUs"
    comptime assert not multi_block or (
        has_amd_gpu_accelerator() and from_probs and dtype == DType.float32
    ), "multi-block Gumbel requires AMD FP32 from-probs"

    comptime EPS = Float32(1e-20)
    comptime LOG2 = Float32(0.6931471806)
    comptime MIN_TEMP = Float32(1e-6)

    comptime simd_width = simd_width_of[dtype]()
    comptime assert (
        simd_width % 4 == 0
    ), "SIMD width must be divisible by 4 to match RNG output size."

    var N = Int(input.dim(1))
    var tid = thread_idx.x
    var block_size = block_dim.x
    var blocks_per_row_int = Int(blocks_per_row)
    var batch_id = Int(block_idx.x)
    var block_in_row = 0
    comptime if multi_block:
        batch_id = Int(block_idx.x) // blocks_per_row_int
        block_in_row = Int(block_idx.x) % blocks_per_row_int

    var temp_val = Float32(1.0)
    if temperature:
        temp_val = temperature.unsafe_value()[batch_id]
        temp_val = max(temp_val, MIN_TEMP)

    var seed_val = UInt64(0)
    if seed:
        seed_val = seed.unsafe_value()[batch_id]

    var ld_ptr = input.ptr + batch_id * N
    comptime align = align_of[SIMD[dtype, simd_width]]()

    var chunk_begin = 0
    var chunk_end = N // simd_width
    comptime if multi_block:
        var chunks_per_block = ceildiv(chunk_end, blocks_per_row_int)
        chunk_begin = block_in_row * chunks_per_block
        chunk_end = min(chunk_begin + chunks_per_block, chunk_end)

    # Per-thread running argmax over the (noised) logits.
    var partial = TopK_2[dtype, True]()

    with PDL():
        # Main region: process vocab in `simd_width`-sized chunks. The RNG is
        # seeded per chunk index `i` exactly as `apply_gumbel_noise_kernel`, so
        # the noise added to each element is bit-identical.
        for i in range(chunk_begin + tid, chunk_end, block_size):
            var rng_state = Random(
                seed=seed_val * UInt64(N) + UInt64(i),
            )
            var input_val: SIMD[dtype, simd_width]
            if N % simd_width == 0:
                input_val = ld_ptr.load[width=simd_width, alignment=align](
                    i * simd_width
                )
            else:
                input_val = ld_ptr.load[width=simd_width](i * simd_width)
            var noised_logits: SIMD[.float32, simd_width]
            comptime if from_probs:
                noised_logits = LOG2 * log2(input_val.cast[.float32]())
            else:
                noised_logits = input_val.cast[.float32]() / temp_val

            comptime for loop_i in range(simd_width // 4):
                var rnd_val = rng_state.step_uniform()
                rnd_val = -LOG2 * log2(-log2(rnd_val + EPS) + EPS)

                comptime for vec_i in range(4):
                    noised_logits[4 * loop_i + vec_i] += rnd_val[vec_i]

            # Feed each noised element into the per-thread argmax.
            comptime for lane in range(simd_width):
                partial.insert(
                    noised_logits[lane].cast[dtype](),
                    i * simd_width + lane,
                )

        # Tail region: elements not covered by a full `simd_width` chunk. Uses
        # the same per-thread seed as `apply_gumbel_noise_kernel`.
        if N % simd_width != 0 and block_in_row == blocks_per_row_int - 1:
            var N_res = N % simd_width
            var rng_state = Random(
                seed=seed_val * UInt64(N) + UInt64(N - N_res) + UInt64(tid),
            )
            if tid < N_res:
                var input_val = ld_ptr.load((N - N_res) + tid).cast[.float32]()
                var noised_logit: Float32
                comptime if from_probs:
                    noised_logit = LOG2 * log2(input_val)
                else:
                    noised_logit = input_val / temp_val
                var rnd_val = rng_state.step_uniform()[0]
                rnd_val = -LOG2 * log2(-log2(rnd_val + EPS) + EPS)
                noised_logit += rnd_val
                partial.insert(
                    noised_logit.cast[dtype](),
                    (N - N_res) + tid,
                )

        # Block-level argmax reduction over per-thread winners.
        var total = _block_reduce_topk[ascending=True](partial)

        comptime if multi_block:
            var is_last = unsafe_stack_allocation[
                1, Bool, address_space=.SHARED
            ]()
            if tid == 0:
                var slot = batch_id * blocks_per_row_int + block_in_row
                var slot_ptr = (
                    partials.unsafe_value() + slot * _GUMBEL_PARTIAL_BYTES
                )
                slot_ptr.bitcast[Float32]()[0] = total.u.cast[.float32]()
                (slot_ptr + 8).bitcast[Int64]()[0] = Int64(total.p)
                var previous = Atomic[Int32].fetch_add[
                    ordering=Ordering.RELEASE
                ](counters.unsafe_value() + batch_id, Int32(1))
                is_last[0] = Int(previous) + 1 == blocks_per_row_int
            barrier()

            if tid == 0 and is_last[0]:
                # This acquire pairs with the release increments that publish
                # each partial.
                fence[Ordering.ACQUIRE]()
                var row_base = batch_id * blocks_per_row_int
                var first = (
                    partials.unsafe_value() + row_base * _GUMBEL_PARTIAL_BYTES
                )
                var best_value = first.bitcast[Float32]()[0]
                var best_index = (first + 8).bitcast[Int64]()[0]
                for block in range(1, blocks_per_row_int):
                    var slot_ptr = (
                        partials.unsafe_value()
                        + (row_base + block) * _GUMBEL_PARTIAL_BYTES
                    )
                    var value = slot_ptr.bitcast[Float32]()[0]
                    var index = (slot_ptr + 8).bitcast[Int64]()[0]
                    if value > best_value or (
                        value == best_value and index < best_index
                    ):
                        best_value = value
                        best_index = index
                out_idxs.ptr[batch_id] = best_index.cast[out_idx_type]()
        else:
            if tid == 0:
                out_idxs.ptr[batch_id] = Int(total.p).cast[out_idx_type]()


@always_inline
def gumbel_sampling_fused_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    TemperatureLayoutType: TensorLayout = RowMajorLayout[Int64],
    SeedLayoutType: TensorLayout = RowMajorLayout[Int64],
    from_probs: Bool = False,
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    input: TileTensor[mut=False, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    seed: Optional[
        TileTensor[.uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine]
    ] = None,
) raises:
    """
    Fused Gumbel sampling: applies Gumbel(0,1) noise and selects the argmax in a
    single GPU kernel launch (no intermediate noised-logits HBM buffer).

    Mathematically equivalent to `gumbel_sampling_gpu` and produces bit-identical
    results for the same seed, but saves one full `[batch, vocab]` HBM
    round-trip by fusing noise generation and argmax.

    With `from_probs` the input rows are unnormalized probabilities and the
    draw is proportional to them; see `_gumbel_argmax_fused_kernel`.

    Args:
        ctx: Device context for GPU operations.
        input: Input logits tensor [batch, vocab_size].
        out_idxs: Output tensor for sampled indices [batch, 1].
        temperature: Optional per-token temperature scaling [batch].
        seed: Optional per-token random seeds [batch] for reproducibility.
    """
    # Mirrors the kernel's guard so a host caller fails at its own call site.
    comptime assert (
        not from_probs or not is_apple_gpu()
    ), "from_probs is not supported on Apple GPUs"

    var input_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    def trace_information() {imm} -> String:
        return trace_arg("input", input_shape, dtype)

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "gumbel_sampling_fused_gpu",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var batch_size = Int(input.dim(0))
        # Speculative decoding can ask for zero rows, and a grid of 0 is not
        # a legal launch.
        if batch_size == 0:
            return

        comptime hw_info = ctx.default_device_info
        comptime block_size = hw_info.max_thread_block_size

        var temperature_ptr: Optional[
            UnsafePointer[Float32, ImmutAnyOrigin]
        ] = None
        if temperature:
            temperature_ptr = temperature.value().ptr
        var seed_ptr: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]] = None
        if seed:
            seed_ptr = seed.value().ptr

        comptime split_capable = (
            has_amd_gpu_accelerator() and from_probs and dtype == DType.float32
        )
        comptime if split_capable:
            var vocab = Int(input.dim(1))
            var blocks_per_row = hw_info.sm_count // batch_size
            if vocab >= 32768 and blocks_per_row > 1:
                var partials_buf = ctx.enqueue_create_buffer[.uint8](
                    batch_size * blocks_per_row * _GUMBEL_PARTIAL_BYTES
                )
                var counters_buf = ctx.enqueue_create_buffer[.int32](batch_size)
                ctx.enqueue_memset(counters_buf, Int32(0))
                var partials_ptr = Optional[UnsafePointer[UInt8, MutAnyOrigin]](
                    partials_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
                )
                var counters_ptr = Optional[UnsafePointer[Int32, MutAnyOrigin]](
                    counters_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
                )
                comptime split_kernel = _gumbel_argmax_fused_kernel[
                    dtype,
                    out_idx_type,
                    input.LayoutType,
                    out_idxs.LayoutType,
                    from_probs=from_probs,
                    multi_block=True,
                    InputEngine=input.Engine,
                    OutIdxEngine=out_idxs.Engine,
                ]
                ctx.enqueue_function[split_kernel](
                    input.as_immut(),
                    out_idxs,
                    temperature_ptr,
                    seed_ptr,
                    Int32(blocks_per_row),
                    partials_ptr,
                    counters_ptr,
                    grid_dim=batch_size * blocks_per_row,
                    block_dim=block_size,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )
                _ = partials_buf^
                _ = counters_buf^
                return

        comptime kernel = _gumbel_argmax_fused_kernel[
            dtype,
            out_idx_type,
            input.LayoutType,
            out_idxs.LayoutType,
            from_probs=from_probs,
            multi_block=False,
            InputEngine=input.Engine,
            OutIdxEngine=out_idxs.Engine,
        ]
        ctx.enqueue_function[kernel](
            input.as_immut(),
            out_idxs,
            temperature_ptr,
            seed_ptr,
            Int32(1),
            Optional[UnsafePointer[UInt8, MutAnyOrigin]](None),
            Optional[UnsafePointer[Int32, MutAnyOrigin]](None),
            grid_dim=batch_size,
            block_dim=block_size,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


@always_inline
def gumbel_sampling_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    TemperatureLayoutType: TensorLayout = RowMajorLayout[Int64],
    SeedLayoutType: TensorLayout = RowMajorLayout[Int64],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    input: TileTensor[mut=False, dtype, ...],
    out_idxs: TileTensor[mut=True, out_idx_type, ...],
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    seed: Optional[
        TileTensor[.uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine]
    ] = None,
) raises:
    """
    Gumbel sampling using the Gumbel-max trick for categorical distributions.

    Applies Gumbel(0,1) noise to input logits, then selects the argmax.
    This is mathematically equivalent to sampling from softmax(logits/temperature)
    but avoids expensive softmax computation.

    Args:
        ctx: Device context for GPU operations.
        input: Input logits tensor [batch, vocab_size].
        out_idxs: Output tensor for sampled indices [batch, 1].
        temperature: Optional per-token temperature scaling [batch].
        seed: Optional per-token random seeds [batch] for reproducibility.
    """

    var input_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    def trace_information() {imm} -> String:
        return trace_arg("input", input_shape, dtype)

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "gumbel_sampling_gpu",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        # create a buffer to hold the Gumbel noise applied input
        var noised_input_buf = ctx.enqueue_create_buffer[dtype](
            input.num_elements()
        )
        var noised_input = TileTensor(noised_input_buf, input.layout)

        comptime hw_info = ctx.default_device_info
        comptime gumbel_kernel = apply_gumbel_noise_kernel[
            dtype,
            noised_input.LayoutType,
            input.LayoutType,
            hw_info.sm_count,
            hw_info.max_thread_block_size,
        ]

        var temperature_ptr: Optional[
            UnsafePointer[Float32, ImmutAnyOrigin]
        ] = None
        if temperature:
            temperature_ptr = temperature.value().ptr
        var seed_ptr: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]] = None
        if seed:
            seed_ptr = seed.value().ptr

        ctx.enqueue_function[gumbel_kernel](
            noised_input,
            input.as_immut(),
            temperature_ptr,
            seed_ptr,
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

        # Extract argmax after Gumbel noise application.
        var out_vals_shape = coord_to_index_list(input.layout.shape_coord())
        out_vals_shape[input.rank - 1] = 1
        var out_vals_buf = ctx.enqueue_create_buffer[dtype](
            out_vals_shape.flattened_length()
        )
        var out_vals = TileTensor(
            out_vals_buf,
            row_major(Coord(out_vals_shape)),
        )

        topk_gpu[sampling=False](
            ctx,
            1,
            noised_input,
            out_vals,
            out_idxs,
        )

        _ = noised_input_buf^
        _ = out_vals_buf^
