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

# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

"""Registers core tensor graph ops (range, copy, reshape, and related utilities)."""

from std.collections import OptionalReg
from std.math import align_up, ceildiv, iota
from std.random import seed
from std.sys.info import size_of
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from max.algorithm import mean
from comm.allreduce import allreduce
from internal_utils.fp8_utils import cast_saturating

from comm.allreduce_residual_rmsnorm import allreduce_residual_rmsnorm
from comm.device_collective import _launch_device_collective
from comm import MAX_GPUS, Signal
from extensibility import StaticTensorSpec
from max.gpu.host import CompletionFlag, DeviceContext, DeviceContextArray
from layout.tile_tensor import row_major
from max.gpu.host.info import B200, is_cpu, is_gpu, is_valid_target
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from layout.coord import DynamicCoord
from nn._ragged_utils import eagle_prefill_shift_tokens
from nn.arange import arange_shape
from nn.argmaxmin import argmax
from nn.conv.conv import pack_filter as _pack_conv_filter
from nn.conv.conv import pack_filter_from_fcrs as _pack_conv_filter_from_fcrs
from nn.conv.conv_transpose import pack_filter as _pack_conv_transpose_filter
from nn.conv.conv_transpose import (
    pack_filter_shape as pack_filter_shape_conv_transpose,
)
from nn.fold import fold, fold_shape
from nn.gather_scatter import normalize_neg_index
from nn.irfft import irfft
from nn.kv_cache import (
    generic_fused_qkv_matmul_kv_cache_bshd_paged,
    generic_get_paged_cache,
    print_kv_cache_paged_generic_cpu,
    print_kv_cache_paged_generic_gpu,
)
from nn.rope_split_store import (
    rope_split_store_paged_ragged,
    rope_split_store_paged_ragged_with_position_ids,
)
from nn.kv_cache_ragged import (
    generic_flash_attention_kv_cache_ragged,
    generic_flash_attention_kv_cache_ragged_rel_logits,
    generic_flash_attention_kv_cache_ragged_sink,
    generic_fused_qk_rope_bshd_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias,
)
from nn.attention.gpu.mha import MHADecodeDispatchMetadata
from nn.attention.mha_utils import as_dynamic_row_major_1d
from nn.moe import (
    eplb_remap,
    moe_create_indices,
    router_group_limited,
    sink_gate_router,
    single_group_router,
    single_group_router_eplb,
)
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.pool import max_pool, pool_shape, pool_shape_ceil
from nn.rand_normal import random_normal
from nn.rand_uniform import random_uniform
from nn.repeat_interleave import repeat_interleave, repeat_interleave_shape
from nn.roi_align import roi_align_nhwc
from nn.rope import rope_ragged
from nn.sampling import apply_penalties_to_logits, update_frequency_data
from nn.split import split
from nn.topk import fused_token_sampling_cpu as _fused_token_sampling_cpu
from nn.topk import fused_token_sampling_gpu as _fused_token_sampling_gpu
from nn.topk import gumbel_sampling_fused_gpu
from nn.sampling import topk_topp_masked_probs, topk_topp_sampling_from_prob
from nn.toppminp import min_p_sampling as min_p_sampling_cpu
from nn.toppminp_gpu import min_p_sampling_gpu
from state_space.gated_delta_conv1d import gated_delta_conv1d_fwd_gpu
from state_space.gated_delta import gated_delta_recurrence_fwd_gpu
from state_space.gated_group_rmsnorm import (
    gated_group_rmsnorm_cpu,
    gated_group_rmsnorm_gpu,
)
from state_space.mamba2_ssd_scan import (
    mamba2_ssd_chunk_scan_varlen_fwd_cpu,
    mamba2_ssd_chunk_scan_varlen_fwd_gpu,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split,
)
from state_space.varlen_causal_conv1d import (
    causal_conv1d_varlen_fwd_cpu,
    causal_conv1d_varlen_fwd_gpu,
    causal_conv1d_varlen_fwd_seqparallel_gpu,
)
from max.runtime.tracing import trace_arg
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    IOSpec,
    ManagedTensorSlice,
    OutputTensor,
    Tensor,
    TileTensorable,
    VariadicTensors,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
)
from builtin_primitives.primitives import foreach
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from extensibility import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from std.memory import UnsafePointer, unsafe_memcpy
from std.time import sleep
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList
from std.utils.index import Index
from nn.learnable_2d_interp_pos_emb import learnable_2d_interp_pos_emb
from nn.spatial_merge import spatial_merge
from nn.tpool_patch_merger import (
    tpool_patch_merger as nn_tpool_patch_merger,
)

# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def reduce_shape(
    input_buf: Some[Tensor], axis: Int
) raises -> IndexList[type_of(input_buf).rank]:
    """
    Compute the output shape of a `reduce` operation, and assert the inputs are
    compatible.

    Args:
        input_buf: The input tensor.
        axis: The axis tensor.

    Returns:
        The output shape.
    """

    # compute and return the output shape
    var output_shape = rebind[IndexList[type_of(input_buf).rank]](
        coord_to_index_list(input_buf.shape().tuple())
    )
    output_shape[normalize_neg_index(axis, type_of(input_buf).rank)] = 1
    return output_shape


@always_inline
def _unsafe_str_to_coord[
    str_slice: StaticString
]() -> DynamicCoord[.int64, len(str_slice.split("_"))]:
    """
    Convert a string of integers separated by "_" to an IntTuple.

    Parameters:
        str_slice: The string of integers separated by "_".

    Returns:
        The IntTuple.
    """
    comptime size = len(str_slice.split("_"))
    var coord = DynamicCoord[.int64, size]()

    comptime for i in range(size):
        comptime sub_string = str_slice.split("_")[i]
        comptime str_len = sub_string.byte_length()
        var result = 0

        comptime for pos in range(str_len):
            result = result * 10 + (ord(sub_string[byte=pos]) - ord("0"))
        coord[i] = rebind[coord.element_types[i]](Int64(result))

    return coord


# TODO(MOCO-1413): remove this need to keep imported exported funcs alive.
@export
def export() abi("Mojo"):
    """Keeps the managed-tensor-slice load/store entry points alive for export when this package is linked.
    """
    comptime _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    comptime _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice


# ===-----------------------------------------------------------------------===#
# Elementwise Kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.range")
struct Range:
    """Registers the `mo.range` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=1, ...],
        start: Scalar[dtype],
        stop: Scalar[dtype],
        step: Scalar[dtype],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        def func[
            width: Int, element_alignment: Int
        ](idx: IndexList[1]) {var start, var step} -> SIMD[dtype, width]:
            return start + step * (iota[dtype, width](Scalar[dtype](idx[0])))

        foreach[
            target=target,
            _trace_name=_trace_name,
        ](func, output, ctx)


@extensibility.register_shape_function("mo.range")
def range_shape[
    dtype: DType
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    step: Scalar[dtype],
) raises -> IndexList[1]:
    """Computes the output shape for the `mo.range` graph op.

    Parameters:
        dtype: Element type of the range values.

    Args:
        start: First value of the range.
        stop: Exclusive upper or lower bound of the range.
        step: Spacing between consecutive values; must be non-zero.

    Returns:
        The output shape `[num_elements]` for the range tensor.
    """
    return arange_shape(start, stop, step)


# ===-----------------------------------------------------------------------===#
# Binary Elementwise Kernels
# ===-----------------------------------------------------------------------===#


# useful for testing --> identity op that simply copies input into output
@extensibility.register("copy")
struct Copy:
    """Registers the `copy` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @__parameter
        @always_inline
        def func[
            width: Int, element_alignment: Int
        ](idx: IndexList[rank]) -> SIMD[dtype, width]:
            return input._fused_load[
                width, element_alignment=element_alignment
            ](idx)

        foreach[func](output, ctx)


@extensibility.register("nan_check_count")
struct NanCheckCountOp:
    """Counts NaN/Inf values in a floating-point tensor.

    See nn.nan_check for implementation details.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString = "",
    ](
        nan_count_out: OutputTensor[dtype=.int32, rank=1, ...],
        inf_count_out: OutputTensor[dtype=.int32, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        from .nan_check import nan_check_count

        nan_check_count[dtype, rank, target](
            nan_count_out, inf_count_out, input, ctx
        )


@extensibility.register("nan_check_raise")
struct NanCheckRaiseOp:
    """Raises an error if NaN or Inf counts are non-zero.

    See nn.nan_check for implementation details.
    """

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString = "",
        label: StaticString = "",
        type_str: StaticString = "",
    ](
        nan_count: InputTensor[dtype=.int32, rank=1, ...],
        inf_count: InputTensor[dtype=.int32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        from .nan_check import nan_check_raise

        nan_check_raise[label, type_str](nan_count, inf_count)


# ===-----------------------------------------------------------------------===#
# View kernels
# ===-----------------------------------------------------------------------===#


# Type-level transpose stride computation. Permute input stride CoordLike types
# according to a permutation IntTuple. This avoids the interpreter heap limit
# that prevents direct IntTuple element access in comptime-for loops.
comptime _TransposeStrideTypesTabulator[
    permutations: IntTuple,
    input_stride_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = Int if Int(
    permutations[idx]
) == UNKNOWN_VALUE else input_stride_types[
    Int(permutations[idx])
]


comptime _TransposeStrideTypes[
    permutations: IntTuple,
    rank: Int,
    input_stride_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    rank, _TransposeStrideTypesTabulator[permutations, input_stride_types, _]
]()


# Type-level slice stride computation: multiplies input stride types by step
# types element-wise.
comptime _SliceStrideTypesTabulator[
    input_stride_types: TypeList[Trait=CoordLike, ...],
    step_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = ComptimeInt[
    input_stride_types[idx].static_value * step_types[idx].static_value
] if input_stride_types[
    idx
].is_static_value and step_types[
    idx
].is_static_value else Scalar[
    DType.int
]

comptime _SliceStrideTypes[
    rank: Int,
    input_stride_types: TypeList[Trait=CoordLike, ...],
    step_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    rank, _SliceStrideTypesTabulator[input_stride_types, step_types, _]
]()

# No shape function as we just directly embed the logic to check the shape
# of the 'slice' operand of the MO op directly in the kernel.


# ===-----------------------------------------------------------------------===#
# Pooling kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.max_pool")
struct MaxPool:
    """Registers the `mo.max_pool` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=4, ...],
        filter: InputTensor[dtype=int_type, rank=1, ...],
        strides: InputTensor[dtype=int_type, rank=1, ...],
        dilations: InputTensor[dtype=int_type, rank=1, ...],
        paddings: InputTensor[dtype=int_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        max_pool[target=target](
            input.to_tile_tensor[.int64](),
            filter.to_tile_tensor[.int64](),
            strides.to_tile_tensor[.int64](),
            dilations.to_tile_tensor[.int64](),
            paddings.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            False,
            ctx,
        )


@extensibility.register_shape_function("mo.max_pool")
def max_pool_shape(
    input: Some[TileTensorable],
    filter: Some[TileTensorable],
    strides: Some[TileTensorable],
    dilations: Some[TileTensorable],
    paddings: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.max_pool` graph op.

    Args:
        input: Rank-4 batched image input to the pooling operator.
        filter: One-dimensional tensor of filter sizes on the height and
            width dimensions, `(filter_h, filter_w)`.
        strides: One-dimensional tensor of strides on the height and width
            dimensions, `(stride_h, stride_w)`.
        dilations: One-dimensional tensor of dilations on the height and
            width dimensions, `(dilation_h, dilation_w)`.
        paddings: One-dimensional tensor of paddings on the height and
            width dimensions, ``(pad_h_before, pad_h_after, pad_w_before,
            pad_w_after)``.

    Returns:
        The output shape of the max pooling operation.
    """
    comptime assert type_of(input).rank == 4, "input must be rank 4"
    comptime assert type_of(filter).rank == 1, "filter must be rank 1"
    comptime assert type_of(strides).rank == 1, "strides must be rank 1"
    comptime assert type_of(dilations).rank == 1, "dilations must be rank 1"
    comptime assert type_of(paddings).rank == 1, "paddings must be rank 1"
    comptime assert (
        type_of(strides).dtype == type_of(filter).dtype
        and type_of(dilations).dtype == type_of(filter).dtype
        and type_of(paddings).dtype == type_of(filter).dtype
    ), "filter, strides, dilations, and paddings must share a dtype"
    return rebind[IndexList[type_of(input).rank]](
        pool_shape(
            input.to_tile_tensor(),
            filter.to_tile_tensor(),
            strides.to_tile_tensor(),
            dilations.to_tile_tensor(),
            paddings.to_tile_tensor(),
        )
    )


@extensibility.register("mo.max_pool_ceil_mode_true")
struct MaxPoolCeilModeTrue:
    """Registers the `mo.max_pool_ceil_mode_true` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=4, ...],
        filter: InputTensor[dtype=int_type, rank=1, ...],
        strides: InputTensor[dtype=int_type, rank=1, ...],
        dilations: InputTensor[dtype=int_type, rank=1, ...],
        paddings: InputTensor[dtype=int_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        max_pool[target=target](
            input.to_tile_tensor[.int64](),
            filter.to_tile_tensor[.int64](),
            strides.to_tile_tensor[.int64](),
            dilations.to_tile_tensor[.int64](),
            paddings.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            True,
            ctx,
        )


@extensibility.register_shape_function("mo.max_pool_ceil_mode_true")
def max_pool_ceil_mode_true_shape(
    input: Some[TileTensorable],
    filter: Some[TileTensorable],
    strides: Some[TileTensorable],
    dilations: Some[TileTensorable],
    paddings: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.max_pool_ceil_mode_true` graph op.

    Args:
        input: Rank-4 batched image input to the pooling operator.
        filter: One-dimensional tensor of filter sizes on the height and
            width dimensions, `(filter_h, filter_w)`.
        strides: One-dimensional tensor of strides on the height and width
            dimensions, `(stride_h, stride_w)`.
        dilations: One-dimensional tensor of dilations on the height and
            width dimensions, `(dilation_h, dilation_w)`.
        paddings: One-dimensional tensor of paddings on the height and
            width dimensions, ``(pad_h_before, pad_h_after, pad_w_before,
            pad_w_after)``.

    Returns:
        The output shape of the max pooling operation with ceil mode enabled.
    """
    comptime assert type_of(input).rank == 4, "input must be rank 4"
    comptime assert type_of(filter).rank == 1, "filter must be rank 1"
    comptime assert type_of(strides).rank == 1, "strides must be rank 1"
    comptime assert type_of(dilations).rank == 1, "dilations must be rank 1"
    comptime assert type_of(paddings).rank == 1, "paddings must be rank 1"
    comptime assert (
        type_of(strides).dtype == type_of(filter).dtype
        and type_of(dilations).dtype == type_of(filter).dtype
        and type_of(paddings).dtype == type_of(filter).dtype
    ), "filter, strides, dilations, and paddings must share a dtype"
    return rebind[IndexList[type_of(input).rank]](
        pool_shape_ceil(
            input.to_tile_tensor(),
            filter.to_tile_tensor(),
            strides.to_tile_tensor(),
            dilations.to_tile_tensor(),
            paddings.to_tile_tensor(),
        )
    )


# ===-----------------------------------------------------------------------===#
# Non maximum suppression kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.non_maximum_suppression")
struct NonMaximumSuppression:
    """Registers the `mo.non_maximum_suppression` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType
    ](
        output: OutputTensor[dtype=.int64, rank=2, ...],
        boxes: InputTensor[dtype=dtype, rank=3, ...],
        scores: InputTensor[dtype=dtype, rank=3, ...],
        max_output_boxes_per_class: Int64,
        iou_threshold: Float32,
        score_threshold: Float32,
    ):
        var max_output_boxes_int = Int(max_output_boxes_per_class)
        var iou_threshold_float = iou_threshold
        var score_threshold_float = score_threshold

        non_max_suppression(
            boxes.to_tile_tensor[.int64](),
            scores.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )


@extensibility.register_shape_function("mo.non_maximum_suppression")
def non_maximum_suppression_shape(
    boxes: Some[TileTensorable],
    scores: Some[TileTensorable],
    max_output_boxes_per_class: Int64,
    iou_threshold: Float32,
    score_threshold: Float32,
) -> IndexList[2]:
    """Computes the output shape for the `mo.non_maximum_suppression` graph op.

    Args:
        boxes: Rank-3 tensor of bounding boxes with shape
            `(batch, num_boxes, 4)` where each box is
            `[y1, x1, y2, x2]`.
        scores: Rank-3 tensor of scores with shape
            `(batch, num_classes, num_boxes)`.
        max_output_boxes_per_class: Maximum number of boxes to select per
            class.
        iou_threshold: Intersection-over-union threshold for suppression;
            boxes with IoU above this value are suppressed.
        score_threshold: Minimum score for a box to be considered; boxes
            with score below this value are filtered out.

    Returns:
        Two-element `IndexList` of shape `(num_selected_boxes, 3)`.
    """
    comptime assert type_of(boxes).rank == 3, "boxes must be rank 3"
    comptime assert type_of(scores).rank == 3, "scores must be rank 3"
    comptime assert (
        type_of(scores).dtype == type_of(boxes).dtype
    ), "scores dtype must match boxes dtype"
    var max_output_boxes_int = Int(max_output_boxes_per_class)
    var iou_threshold_float = iou_threshold
    var score_threshold_float = score_threshold

    return non_max_suppression_shape_func(
        boxes.to_tile_tensor(),
        scores.to_tile_tensor().bitcast[type_of(boxes).dtype](),
        max_output_boxes_int,
        iou_threshold_float,
        score_threshold_float,
    )


# ===-----------------------------------------------------------------------===#
# ROI align kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.roi_align")
struct ROIAlign:
    """Registers the `mo.roi_align` graph op with the graph compiler."""

    @staticmethod
    def execute[
        aligned: Bool,
        mode: StaticString,
        dtype: DType,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=4, ...],
        rois: InputTensor[dtype=dtype, rank=2, ...],
        output_height: Int64,
        output_width: Int64,
        spatial_scale: Scalar,
        sampling_ratio: Scalar,
    ):
        roi_align_nhwc[aligned, mode](
            output.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
            rois.to_tile_tensor[.int64](),
            Int(output_height),
            Int(output_width),
            spatial_scale,
            sampling_ratio,
        )


@extensibility.register_shape_function("mo.roi_align")
def roi_align_shape(
    input: Some[Tensor],
    rois: Some[Tensor],
    output_height: Int64,
    output_width: Int64,
    spatial_scale: Scalar,
    sampling_ratio: Scalar,
) -> IndexList[4]:
    """Computes the output shape for the `mo.roi_align` graph op.

    Args:
        input: Rank-4 batched image input in NHWC format with shape
            `(N, H, W, C)`.
        rois: Rank-2 tensor of ROI box coordinates with shape `(M, 5)`
            where each row is `(batch_idx, y0, x0, y1, x1)`.
        output_height: Pooled output height, in elements.
        output_width: Pooled output width, in elements.
        spatial_scale: Scale factor remapping ROI coordinates to input
            coordinates.
        sampling_ratio: Number of sampling points in the interpolation
            grid used to compute each pooled bin.

    Returns:
        The output shape `(num_rois, output_height, output_width, channels)`.
    """
    comptime assert type_of(input).rank == 4, "input must be rank 4"
    comptime assert type_of(rois).rank == 2, "rois must be rank 2"
    var input_shape = coord_to_index_list(input.shape().tuple())
    var rois_shape = coord_to_index_list(rois.shape().tuple())

    var shape = IndexList[4]()
    # input shape is [N, H, W, C]
    # rois shape is [M, 5]
    # output shape is [M, output_height, output_width, C]
    shape[0] = rois_shape[0]
    shape[1] = Int(output_height)
    shape[2] = Int(output_width)
    shape[3] = input_shape[3]

    return shape


# ===-----------------------------------------------------------------------===#
# Repeat Interleave kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("repeat_interleave")
struct RepeatInterleave:
    """Registers the `repeat_interleave` graph op with the graph compiler."""

    @staticmethod
    def execute(
        output: OutputTensor,
        input: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        repeats: InputTensor[rank=1, ...],
        axis: Scalar,
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            axis.dtype.is_integral()
        ), "axis value must be integer type"

        repeat_interleave(
            input.to_tile_tensor[.int64](),
            repeats.to_tile_tensor[.int64](),
            Int(normalize_neg_index(axis, input.rank)),
            output.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register_shape_function("repeat_interleave")
def repeat_interleave_kernel_shape(
    input: Some[TileTensorable], repeats: Some[TileTensorable], axis: Scalar
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `repeat_interleave` graph op.

    Args:
        input: Input tensor whose `axis` dimension is expanded by the
            repeat counts.
        repeats: One-dimensional integral tensor of per-element repeat
            counts, either size 1 or equal to `input.dim(axis)`.
        axis: Axis along which elements are repeated; negative values
            index from the last dimension.

    Returns:
        The output shape with `axis` expanded by the repeat counts.
    """
    comptime assert axis.dtype.is_integral(), "axis value must be integer type"
    comptime assert type_of(repeats).rank == 1, "repeats must be rank 1"

    var interleave_shape = repeat_interleave_shape(
        input.to_tile_tensor(),
        repeats.to_tile_tensor(),
        Int(normalize_neg_index(axis, type_of(input).rank)),
    )

    return rebind[IndexList[type_of(input).rank]](interleave_shape)


# ===-----------------------------------------------------------------------===#
# Random kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.random.normal")
struct RandomNormal:
    """Registers the `mo.random.normal` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, ...],
        shape: InputTensor[rank=1, ...],
        mean: Float32,
        variance: Float32,
        seed_value: InputTensor[dtype=.uint64, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        def output_fn[
            _width: SIMDLength,
            _rank: Int,
        ](coords: IndexList[_rank], val: SIMD[dtype, _width]) {imm output}:
            output._lambda_store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        random_normal[target=target](
            output.shape(),
            mean,
            variance,
            seed_value.unsafe_ptr[.uint64](),
            ctx,
            output_fn,
        )


@extensibility.register_shape_function("mo.random.normal")
def random_normal_shape[
    output_rank: Int,
](
    shape: Some[TileTensorable],
    mean: Scalar,
    variance: Scalar,
    seed_value: Some[Tensor],
) -> IndexList[output_rank]:
    """Computes the output shape for the `mo.random.normal` graph op.

    Parameters:
        output_rank: Number of dimensions of the output tensor; must equal
            the length of `shape`.

    Args:
        shape: One-dimensional tensor of output dimensions, length
            `output_rank`.
        mean: Mean of the normal distribution; does not affect the output
            shape.
        variance: Variance of the normal distribution; does not affect the
            output shape.
        seed_value: One-dimensional tensor holding the random seed; does not
            affect the output shape.

    Returns:
        The output shape specified by `shape`.
    """
    comptime assert type_of(shape).rank == 1, "shape must be rank 1"
    comptime assert type_of(seed_value).rank == 1, "seed_value must be rank 1"
    comptime assert (
        type_of(seed_value).dtype == .uint64
    ), "seed_value dtype must be uint64"
    var shape_tt = shape.to_tile_tensor()
    var unrolled_shape = IndexList[output_rank]()
    for i in range(output_rank):
        unrolled_shape[i] = Int(shape_tt.load[1](Coord(i)))

    return unrolled_shape


@extensibility.register("mo.random.uniform")
struct RandomUniform:
    """Registers the `mo.random.uniform` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, ...],
        shape: InputTensor[rank=1, ...],
        lower_bound: Scalar[dtype],
        upper_bound: Scalar[dtype],
        seed_value: InputTensor[dtype=.uint64, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        def output_fn[
            _width: SIMDLength,
            _rank: Int,
        ](coords: IndexList[_rank], val: SIMD[dtype, _width]) {imm output}:
            output._lambda_store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        random_uniform[target=target](
            output.shape(),
            lower_bound,
            upper_bound,
            seed_value.unsafe_ptr[.uint64](),
            ctx,
            output_fn,
        )


@extensibility.register_shape_function("mo.random.uniform")
def random_uniform_shape[
    output_rank: Int,
](
    shape: Some[TileTensorable],
    mean: Scalar,
    variance: Scalar,
    seed_value: Some[Tensor],
) -> IndexList[output_rank]:
    """Computes the output shape for the `mo.random.uniform` graph op.

    Parameters:
        output_rank: Number of dimensions of the output tensor; must equal
            the length of `shape`.

    Args:
        shape: One-dimensional tensor of output dimensions, length
            `output_rank`.
        mean: Scalar parameter from the uniform op; does not affect the
            output shape.
        variance: Scalar parameter from the uniform op; does not affect the
            output shape.
        seed_value: One-dimensional tensor holding the random seed; does not
            affect the output shape.

    Returns:
        The output shape specified by `shape`.
    """
    comptime assert type_of(shape).rank == 1, "shape must be rank 1"
    comptime assert type_of(seed_value).rank == 1, "seed_value must be rank 1"
    comptime assert (
        type_of(seed_value).dtype == .uint64
    ), "seed_value dtype must be uint64"
    var shape_tt = shape.to_tile_tensor()
    assert Int(shape_tt.dim(0)) == output_rank

    var unrolled_shape = IndexList[output_rank]()
    for i in range(output_rank):
        unrolled_shape[i] = Int(shape_tt.load[1](Coord(i)))

    return unrolled_shape


# ===-----------------------------------------------------------------------===#
# Concat kernels
# ===-----------------------------------------------------------------------===#


def concat_shape_impl[
    dtype: DType, rank: Int, size: Int, io_spec: IOSpec
](
    axis0: Int,
    inputs: VariadicTensors[dtype=dtype, rank=rank, size, io_spec=io_spec, ...],
) raises -> IndexList[rank]:
    """Computes the concatenated output shape from the input tensor shapes.

    Parameters:
        dtype: Element type of the input tensors.
        rank: Number of dimensions in each input tensor.
        size: Number of variadic input tensors.
        io_spec: Input/output specification for the variadic tensors.

    Args:
        axis0: Axis along which to concatenate; negative values index
            from the last dimension.
        inputs: Variadic input tensors to concatenate, all matching in
            shape except along the concat axis.

    Returns:
        The output shape with the concat axis dimension summed across
        all inputs.
    """
    var axis = normalize_neg_index(axis0, rank)

    @__parameter
    @always_inline
    def shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) -> Bool:
        comptime for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0

    comptime for i in range(inputs.size):
        concat_axis_dim_sum += inputs[i].dim_size(axis)
        if not shape_equal_ignore_axis(
            inputs[0].shape(),
            inputs[i].shape(),
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = inputs[0].shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


# NOTE: there are a lot of similarities between this and the shape func
# for mo.concat.
def concat_from_list_shape_impl[
    dtype: DType, rank: Int
](
    axis0: Int,
    inputs: List[
        InputTensor[
            static_spec=StaticTensorSpec[dtype, rank, ...].get_unknown(),
        ]
    ],
) raises -> IndexList[rank]:
    """Computes the concatenated output shape for a variadic list of input tensors.

    Parameters:
        dtype: Element type of the input tensors.
        rank: Number of dimensions in each input tensor.

    Args:
        axis0: Axis along which to concatenate; negative values index
            from the last dimension.
        inputs: List of input tensors to concatenate, all matching in
            shape except along the concat axis.

    Returns:
        The output shape with the concat axis dimension summed across
        all inputs.
    """
    var axis = normalize_neg_index(axis0, rank)

    @__parameter
    @always_inline
    def shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) -> Bool:
        for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(len(inputs)):
        concat_axis_dim_sum += inputs[i].dim_size(axis)
        if not shape_equal_ignore_axis(
            inputs[0].shape(),
            inputs[i].shape(),
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = inputs[0].shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


# ===-----------------------------------------------------------------------===#
# Split kernels
# ===-----------------------------------------------------------------------===#


# The shape function for split is special and there is special
# handling in the graph compiler to make things work.


# In practice this is how it's done. The graph compiler has additional logic
# to properly dispatch this function.


# ===-----------------------------------------------------------------------===#
# Convolution kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("fold")
struct Fold:
    """Registers the `fold` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType,
        stride_h: Int,
        stride_w: Int,
        dilation_h: Int,
        dilation_w: Int,
        padding_h: Int,
        padding_w: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=3, ...],
        output_size: InputTensor,
        kernel_size: InputTensor,
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            kernel_size.dtype.is_integral() and output_size.dtype.is_integral()
        ), "kernel_size and output_size must have integral type"
        var output_size_tuple = Index(output_size._ptr[0], output_size._ptr[1])
        var kernel_size_tuple = Index(kernel_size._ptr[0], kernel_size._ptr[1])
        var input_tensor = input.to_tile_tensor[.int64]()
        var output_tensor = output.to_tile_tensor[.int64]()

        fold[
            stride=(stride_h, stride_w),
            dilation=(dilation_h, dilation_w),
            padding=(padding_h, padding_w),
            target=target,
        ](
            input_tensor,
            output_tensor,
            output_size_tuple,
            kernel_size_tuple,
            ctx,
        )


@extensibility.register_shape_function("fold")
def fold_kernel_shape[
    stride_h: Int,
    stride_w: Int,
    dilation_h: Int,
    dilation_w: Int,
    padding_h: Int,
    padding_w: Int,
](
    input: Some[TileTensorable],
    output_size: Some[TileTensorable],
    kernel_size: Some[TileTensorable],
) raises -> IndexList[4]:
    """Computes the output shape for the `fold` graph op.

    Parameters:
        stride_h: Vertical stride of the sliding window, in elements.
        stride_w: Horizontal stride of the sliding window, in elements.
        dilation_h: Vertical dilation factor applied to the kernel.
        dilation_w: Horizontal dilation factor applied to the kernel.
        padding_h: Vertical padding applied to the output, in elements.
        padding_w: Horizontal padding applied to the output, in elements.

    Args:
        input: Three-dimensional input tensor of shape
            `(C, H * W, ...)` representing the unfolded feature map.
        output_size: One-dimensional tensor holding the output
            `(height, width)` pair.
        kernel_size: One-dimensional tensor holding the kernel
            `(height, width)` pair.

    Returns:
        The four-dimensional output shape of the folded tensor.
    """
    comptime assert type_of(input).rank == 3, "input must be rank 3"
    comptime assert type_of(output_size).rank == 1, "output_size must be rank 1"
    comptime assert type_of(kernel_size).rank == 1, "kernel_size must be rank 1"
    comptime assert (
        type_of(kernel_size).dtype.is_integral()
        and type_of(output_size).dtype.is_integral()
    ), "kernel_size and output_size must have integral type"
    var output_size_tt = output_size.to_tile_tensor()
    var kernel_size_tt = kernel_size.to_tile_tensor()
    var output_size_tuple = Index(
        Int(output_size_tt.load[1](Coord(0))),
        Int(output_size_tt.load[1](Coord(1))),
    )
    var kernel_size_tuple = Index(
        Int(kernel_size_tt.load[1](Coord(0))),
        Int(kernel_size_tt.load[1](Coord(1))),
    )
    return fold_shape(
        input.to_tile_tensor(),
        output_size_tuple,
        kernel_size_tuple,
    )


# ===-----------------------------------------------------------------------===#
# FFT kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("irfft")
struct IRFFT:
    """Registers the `irfft` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        rank: Int,
        n: Int,
        buffer_size_mb: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        irfft(
            input.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            n,
            buffer_size_mb,
            ctx,
        )


# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Quantization for CPU
# ===-----------------------------------------------------------------------===#

######
# Q4_0
######


######
# Q4_K
######


######
# Q6_K
######


######
# 4-bit quant GPU implementation
######


# ===----------------------------------------------------------------------===#
# KV Cache
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Fused QKV matmul
#
# Expected kernel name format:
# mo.fused_qkv_matmul.<padded/ragged>.paged
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
    dtype: DType,
    weight_type: DType,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    input_row_offsets: ManagedTensorSlice[dtype=DType.uint32, rank=1, ...],
    weight: ManagedTensorSlice[dtype=weight_type, rank=2, ...],
    kv_collection: PagedKVCacheCollection[dtype, ...],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    ctx: DeviceContext,
) raises:
    """Implements the fused QKV matmul for ragged inputs, writing the K and V projections directly into a paged KV cache.

    Parameters:
        dtype: Element type of the `hidden_state` input and `output`
            tensors.
        weight_type: Element type of the `weight` tensor.
        target: Target device identifier for kernel dispatch.
        group_size: Block size for GPTQ-style quantization of `weight`;
            when set, `weight` must be `uint8` (defaults to `None` for no
            quantization).
        has_zp: Whether the weight quantization uses a zero point;
            currently must be falsy when `group_size` is set (defaults to
            `None`).

    Args:
        hidden_state: Input tensor of shape
            `(sum(seq_lens), num_heads * head_size)`.
        input_row_offsets: One-dimensional tensor of shape
            `(batch_size + 1)` whose value at each index is the start
            offset of the corresponding batch in `hidden_state`.
        weight: Weight matrix of shape
            `(num_heads * head_size, num_kv_heads * head_size)`.
        kv_collection: Paged KV cache collection holding the keys and
            values; the cache for this layer is retrieved via
            `layer_idx`.
        layer_idx: Index of the layer whose K and V projections are
            written into the cache.
        output: Pre-allocated output buffer of shape
            `(sum(seq_lens), num_heads * head_size)` for the Q
            projections; K and V projections are written in place to the
            cache.
        ctx: Device context used for kernel dispatch.
    """
    generic_fused_qkv_matmul_kv_cache_paged_ragged[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        weight.to_layout_tensor(),
        kv_collection,
        layer_idx,
        output.to_layout_tensor(),
        ctx,
    )


@always_inline
def generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
    dtype: DType,
    weight_type: DType,
    target: StaticString,
    group_size: Optional[Int] = None,
    has_zp: Optional[Bool] = None,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    input_row_offsets: ManagedTensorSlice[dtype=DType.uint32, rank=1, ...],
    weight: ManagedTensorSlice[dtype=weight_type, rank=2, ...],
    kv_collection: PagedKVCacheCollection[dtype, ...],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    bias: ManagedTensorSlice[dtype=dtype, rank=1, ...],
    ctx: DeviceContext,
) raises:
    """Implements the fused QKV matmul with bias for ragged inputs, writing the K and V projections directly into a paged KV cache.

    Parameters:
        dtype: Element type of the `hidden_state`, `output`, and `bias`
            tensors.
        weight_type: Element type of the `weight` tensor.
        target: Target device identifier for kernel dispatch.
        group_size: Block size for GPTQ-style quantization of `weight`;
            when set, `weight` must be `uint8` (defaults to `None` for no
            quantization).
        has_zp: Whether the weight quantization uses a zero point;
            currently must be falsy when `group_size` is set (defaults to
            `None`).

    Args:
        hidden_state: Input tensor of shape
            `(sum(seq_lens), num_heads * head_size)`.
        input_row_offsets: One-dimensional tensor of shape
            `(batch_size + 1)` whose value at each index is the start
            offset of the corresponding batch in `hidden_state`.
        weight: Weight matrix of shape
            `(num_heads * head_size, num_kv_heads * head_size)`.
        kv_collection: Paged KV cache collection holding the keys and
            values; the cache for this layer is retrieved via
            `layer_idx`.
        layer_idx: Index of the layer whose K and V projections are
            written into the cache.
        output: Pre-allocated output buffer of shape
            `(sum(seq_lens), num_heads * head_size)` for the Q
            projections; K and V projections are written in place to the
            cache.
        bias: One-dimensional bias tensor of length
            `num_kv_heads * head_size` added to the K and V projections
            before they are written to the cache.
        ctx: Device context used for kernel dispatch.
    """
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        hidden_state.to_layout_tensor(),
        input_row_offsets.to_layout_tensor(),
        weight.to_layout_tensor(),
        kv_collection,
        layer_idx,
        output.to_layout_tensor(),
        bias.to_layout_tensor(),
        ctx,
    )


@always_inline
def generic_fused_qkv_matmul_kv_cache_bshd_paged_kernel_api[
    dtype: DType,
    target: StaticString,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=3, ...],
    weight: ManagedTensorSlice[dtype=dtype, rank=2, ...],
    kv_collection: PagedKVCacheCollection[dtype, ...],
    layer_idx: UInt32,
    valid_lengths: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    output: ManagedTensorSlice[dtype=dtype, rank=3, ...],
    ctx: DeviceContext,
) raises:
    """Implements the fused QKV matmul for BSHD inputs, writing the K and V projections directly into a paged KV cache.

    Parameters:
        dtype: Element type of the `hidden_state`, `weight`, and `output`
            tensors.
        target: Target device identifier for kernel dispatch.

    Args:
        hidden_state: Input tensor of shape
            `(batch_size, seq_len, num_heads * head_size)`.
        weight: Weight matrix of shape
            `(num_heads * head_size, num_kv_heads * head_size)`.
        kv_collection: Paged KV cache collection holding the keys and
            values; the cache for this layer is retrieved via
            `layer_idx`.
        layer_idx: Index of the layer whose K and V projections are
            written into the cache.
        valid_lengths: One-dimensional tensor of shape `[batch]` giving
            the valid length of each sequence; K and V are only written
            to the cache for positions within these lengths.
        output: Pre-allocated output buffer of shape
            `(batch_size, seq_len, num_heads * head_size)` for the Q
            projections; K and V projections are written in place to the
            cache.
        ctx: Device context used for kernel dispatch.
    """
    generic_fused_qkv_matmul_kv_cache_bshd_paged[target=target,](
        hidden_state.to_layout_tensor(),
        weight.to_layout_tensor(),
        kv_collection,
        layer_idx,
        valid_lengths,
        output.to_layout_tensor(),
        ctx,
    )


@extensibility.register("mo.rope_split_store.ragged.paged")
struct Struct_rope_split_store_ragged_paged[interleaved: Bool]:
    """Registers the `mo.rope_split_store.ragged.paged` graph op with the graph compiler.

    Parameters:
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components (real, imag, real, imag, ...) as in GGUF, rather
            than splitting them into halves (real, ..., real, imag, ...,
            imag) as in safetensors.
    """

    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        qkv_dtype: DType,
        freq_dtype: DType,
        cache_dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_dtype, rank=2, ...],
        qkv: InputTensor[dtype=qkv_dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return rope_split_store_paged_ragged[
            q_out_dtype=out_dtype,
            target=target,
            interleaved=Self.interleaved,
        ](
            qkv.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            freqs_cis.to_tile_tensor[.int64](),
            kv_collection,
            layer_idx,
            output.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("mo.rope_split_store.ragged.paged.with_position_id")
struct Struct_rope_split_store_ragged_paged_with_position_id[interleaved: Bool]:
    """Registers the `mo.rope_split_store.ragged.paged.with_position_id` graph op with the graph compiler.

    Parameters:
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components (real, imag, real, imag, ...) as in GGUF, rather
            than splitting them into halves (real, ..., real, imag, ...,
            imag) as in safetensors.
    """

    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        qkv_dtype: DType,
        freq_dtype: DType,
        cache_dtype: DType,
        //,
        mrope_section: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=out_dtype, rank=2, ...],
        qkv: InputTensor[dtype=qkv_dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        position_ids: InputTensor[dtype=.uint32, rank=2, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime if mrope_section == "":
            return rope_split_store_paged_ragged_with_position_ids[
                target=target,
                interleaved=Self.interleaved,
            ](
                qkv.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_collection,
                position_ids.to_tile_tensor[.int64](),
                layer_idx,
                output.to_tile_tensor[.int64](),
                ctx,
            )
        else:
            comptime mrope = _unsafe_str_to_coord[mrope_section]()
            return rope_split_store_paged_ragged_with_position_ids[
                target=target,
                interleaved=Self.interleaved,
                mrope_types=mrope.element_types,
                mrope_section=mrope,
            ](
                qkv.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_collection,
                position_ids.to_tile_tensor[.int64](),
                layer_idx,
                output.to_tile_tensor[.int64](),
                ctx,
            )


# ===-----------------------------------------------------------------------===#
# Fused QK Rope Ragged
# ===-----------------------------------------------------------------------===#


@always_inline
def generic_fused_qk_rope_bshd_paged_ragged_kernel_api[
    dtype: DType,
    freq_dtype: DType,
    cache_dtype: DType,
    //,
    *,
    interleaved: Bool,
    has_position_ids: Bool,
    target: StaticString,
    mrope_types: TypeList[Trait=CoordLike, ...] = TypeList.of[
        Trait=CoordLike
    ](),
    mrope_section: Optional[Coord[*mrope_types]] = None,
](
    q_proj: ManagedTensorSlice[dtype=dtype, rank=3, ...],
    input_row_offsets: ManagedTensorSlice[dtype=DType.uint32, rank=1, ...],
    kv_collection: PagedKVCacheCollection[cache_dtype, ...],
    freqs_cis: ManagedTensorSlice[dtype=freq_dtype, rank=2, ...],
    position_ids: ManagedTensorSlice[dtype=DType.uint32, rank=2, ...],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=3, ...],
    context: DeviceContext,
) raises:
    """Applies fused rotary position embedding to Q and K, updating keys in the paged KV cache.

    Parameters:
        dtype: Element type of the query projection and output tensors
            (inferred).
        freq_dtype: Element type of the `freqs_cis` RoPE frequency table
            (inferred).
        cache_dtype: Element type of the paged KV cache entries (inferred).
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components rather than splitting them into halves.
        has_position_ids: Whether per-token `position_ids` are supplied and
            used for the rotation.
        target: Target backend identifier for kernel dispatch.
        mrope_types: Coordinate element types for the M-RoPE section split
            (defaults to an empty `TypeList`).
        mrope_section: Optional M-RoPE section coordinate describing how
            the head dimension is partitioned across modalities (defaults
            to `None`).

    Args:
        q_proj: Query projection tensor of shape
            `[batch, seq, head_dim]`.
        input_row_offsets: Cumulative row offsets per batch, shape
            `[batch_size + 1]`.
        kv_collection: Paged KV cache collection whose keys are read and
            updated in place.
        freqs_cis: RoPE frequency table applied to the query and key
            projections.
        position_ids: Per-token position ids used to index `freqs_cis`.
        layer_idx: Index of the layer whose keys are updated in the cache.
        output: Output tensor for the rotation-applied query projection,
            shape `[batch, seq, head_dim]`.
        context: Device context for kernel dispatch.
    """
    generic_fused_qk_rope_bshd_paged_ragged[
        interleaved=interleaved,
        has_position_ids=has_position_ids,
        target=target,
        mrope_types=mrope_types,
        mrope_section=mrope_section,
    ](
        q_proj.to_tile_tensor[.int64](),
        input_row_offsets.to_tile_tensor[.int64](),
        kv_collection,
        freqs_cis.to_tile_tensor[.int64](),
        position_ids.to_tile_tensor[.int64](),
        layer_idx,
        output.to_tile_tensor[.int64](),
        context,
    )


# ===-----------------------------------------------------------------------===#
# RoPE Ragged
#
# Expected kernel name format:
# mo.composite.rope.ragged
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.composite.rope.ragged")
struct Struct_rope_ragged_paged[interleaved: Bool, rope_first: Bool]:
    """Registers the `mo.composite.rope.ragged` graph op with the graph compiler.

    Parameters:
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components (real, imag, real, imag, ...) as in GGUF, rather
            than splitting them into halves (real, ..., real, imag, ...,
            imag) as in safetensors.
        rope_first: Whether a partial RoPE rotates the leading columns of
            each head, leaving the trailing ones to pass through, rather
            than the other way around.
    """

    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        dtype: DType,
        freq_dtype: DType,
        //,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=out_dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        start_pos: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        @__parameter
        def description_fn() -> String:
            return String(";").join(
                Span(
                    [
                        trace_arg("output", output.shape()),
                        trace_arg("x", x.shape()),
                        trace_arg(
                            "input_row_offsets", input_row_offsets.shape()
                        ),
                        trace_arg("start_pos", start_pos.shape()),
                        trace_arg("freqs_cis", freqs_cis.shape()),
                        "interleaved=" + String(Self.interleaved),
                        "rope_first=" + String(Self.rope_first),
                        "target=" + String(target),
                    ]
                )
            )

        @always_inline
        def output_fn[
            width: SIMDLength, alignment: Int
        ](idx: IndexList[3], val: SIMD[dtype, width]) {var output} -> None:
            output._lambda_store[width=width, element_alignment=alignment](
                idx,
                cast_saturating[out_dtype](val),
            )

        var x_tensor = x.to_tile_tensor[.int64]()
        var row_offsets_tensor = input_row_offsets.to_tile_tensor[.int64]()
        var start_tensor = start_pos.to_tile_tensor[.int64]()
        var freqs_cis_tensor = freqs_cis.to_tile_tensor[.int64]()
        comptime assert row_offsets_tensor.flat_rank == 1
        comptime assert start_tensor.flat_rank == 1
        comptime assert freqs_cis_tensor.flat_rank == 2

        rope_ragged[
            interleaved=Self.interleaved,
            target=target,
            rope_first=Self.rope_first,
        ](
            x_tensor,
            row_offsets_tensor,
            start_tensor,
            freqs_cis_tensor,
            ctx,
            output_fn,
        )


@extensibility.register("mo.rope.ragged.with_position_id")
struct Struct_rope_ragged_paged_with_position_id[interleaved: Bool]:
    """Registers the `mo.rope.ragged.with_position_id` graph op with the graph compiler.

    Parameters:
        interleaved: Whether RoPE pairs adjacent real and imaginary
            components (real, imag, real, imag, ...) as in GGUF, rather
            than splitting them into halves (real, ..., real, imag, ...,
            imag) as in safetensors.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        //,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        start_pos: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        position_ids: InputTensor[dtype=.uint32, rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @always_inline
        @__parameter
        def description_fn() -> String:
            return String(";").join(
                Span(
                    [
                        trace_arg("output", output.shape()),
                        trace_arg("x", x.shape()),
                        trace_arg(
                            "input_row_offsets", input_row_offsets.shape()
                        ),
                        trace_arg("start_pos", start_pos.shape()),
                        trace_arg("freqs_cis", freqs_cis.shape()),
                        trace_arg("position_ids", position_ids.shape()),
                        "interleaved=" + String(Self.interleaved),
                        "target=" + String(target),
                    ]
                )
            )

        @always_inline
        def output_fn[
            width: SIMDLength, alignment: Int
        ](idx: IndexList[3], val: SIMD[dtype, width]) {var output} -> None:
            output._lambda_store[width=width, element_alignment=alignment](
                idx,
                rebind[SIMD[dtype, width]](val),
            )

        var x_tensor = x.to_tile_tensor[.int64]()
        var row_offsets_tensor = input_row_offsets.to_tile_tensor[.int64]()
        var start_tensor = start_pos.to_tile_tensor[.int64]()
        var freqs_cis_tensor = freqs_cis.to_tile_tensor[.int64]()
        var position_ids_tensor = position_ids.to_tile_tensor[.int64]()
        comptime assert row_offsets_tensor.flat_rank == 1
        comptime assert start_tensor.flat_rank == 1
        comptime assert freqs_cis_tensor.flat_rank == 2
        comptime assert position_ids_tensor.flat_rank == 2

        rope_ragged[
            interleaved=Self.interleaved,
            target=target,
        ](
            x_tensor,
            row_offsets_tensor,
            start_tensor,
            freqs_cis_tensor,
            ctx,
            output_fn,
            position_ids=position_ids_tensor.as_unsafe_any_origin().as_immut(),
        )


# ===-----------------------------------------------------------------------===#
# MHA
#
# Expected kernel name format:
# mo.mha.<padded/ragged>.paged
# ===-----------------------------------------------------------------------===#


@always_inline
def _unmarshal_mha_decode_dispatch_metadata(
    mha_decode_dispatch_metadata: InputTensor[dtype=.int64, rank=1, ...],
) -> MHADecodeDispatchMetadata:
    return MHADecodeDispatchMetadata(
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[0]),
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[1]),
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[2]),
        Int(mha_decode_dispatch_metadata.unsafe_ptr()[3]),
    )


@always_inline
def _execute_mha_ragged_paged_scalar_args[
    q_dtype: DType,
    //,
    target: StaticString,
    mask_str: StaticString,
    sink: Bool = False,
    local_window_size: Int = -1,
    output_dtype: DType = q_dtype,
    cache_dtype: DType = q_dtype,
](
    output: OutputTensor[dtype=output_dtype, rank=3, ...],
    q: InputTensor[dtype=q_dtype, rank=3, ...],
    input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
    kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
    cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
    kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
    max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
    max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
    layer_idx: UInt32,
    scale: Float32,
    mha_decode_dispatch_metadata: InputTensor[dtype=.int64, rank=1, ...],
    context: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q_dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises:
    var decode_dispatch_metadata = _unmarshal_mha_decode_dispatch_metadata(
        mha_decode_dispatch_metadata
    )
    var kv_collection = generic_get_paged_cache(
        kv_blocks,
        cache_lengths,
        kv_lookup_table,
        max_prompt_length,
        max_cache_length,
    )
    var input_row_offsets_lt = as_dynamic_row_major_1d(
        input_row_offsets.to_layout_tensor().as_imm()
    )

    comptime if sink:
        generic_flash_attention_kv_cache_ragged_sink[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q.to_layout_tensor(),
            input_row_offsets_lt,
            kv_collection,
            layer_idx,
            scale,
            output.to_layout_tensor(),
            context,
            sink_weights.value(),
            decode_dispatch_metadata,
        )
    else:
        generic_flash_attention_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=output_dtype,
        ](
            q.to_layout_tensor(),
            input_row_offsets_lt,
            kv_collection,
            layer_idx,
            scale,
            output.to_layout_tensor(),
            context,
            decode_dispatch_metadata,
        )


@always_inline
def _execute_mha_ragged_paged_rel_logits[
    q_dtype: DType,
    //,
    target: StaticString,
    local_window_size: Int = -1,
    output_dtype: DType = q_dtype,
    cache_dtype: DType = q_dtype,
](
    output: OutputTensor[dtype=output_dtype, rank=3, ...],
    q: InputTensor[dtype=q_dtype, rank=3, ...],
    input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
    kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
    cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
    kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
    max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
    max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
    layer_idx: UInt32,
    scale: Float32,
    bias: InputTensor[dtype=q_dtype, rank=3, ...],
    mha_decode_dispatch_metadata: InputTensor[dtype=.int64, rank=1, ...],
    context: DeviceContext,
) raises:
    var decode_dispatch_metadata = _unmarshal_mha_decode_dispatch_metadata(
        mha_decode_dispatch_metadata
    )
    var kv_collection = generic_get_paged_cache(
        kv_blocks,
        cache_lengths,
        kv_lookup_table,
        max_prompt_length,
        max_cache_length,
    )
    var input_row_offsets_lt = as_dynamic_row_major_1d(
        input_row_offsets.to_layout_tensor().as_imm()
    )
    var cache_lengths_lt = as_dynamic_row_major_1d(
        cache_lengths.to_layout_tensor().as_imm()
    )

    generic_flash_attention_kv_cache_ragged_rel_logits[
        target=target,
        local_window_size=local_window_size,
        output_dtype=output_dtype,
    ](
        q.to_layout_tensor(),
        input_row_offsets_lt,
        kv_collection,
        layer_idx,
        scale,
        bias.to_layout_tensor(),
        cache_lengths_lt,
        output.to_layout_tensor(),
        context,
        decode_dispatch_metadata,
    )


# ===-----------------------------------------------------------------------===#
# MLA
#
# Expected kernel name format:
# mo.mla.<prefill/decode>.ragged.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Sparse MLA prefill (DSv3.2 absorbed shape, BF16, SM100)
#
# Wraps `mla_prefill_sparse` (the SM100 sparse prefill attention kernel). The
# kernel hardcodes the DSv3.2 absorbed/latent dims:
#   qk_depth = kv_lora_rank(512) + qk_rope_head_dim(64) = 576
#   v_depth  = kv_lora_rank(512)
#   num_q_heads = 128, num_kv_heads = 1
#
# Inputs follow the existing sparse MLA MOGG convention: the indexer emits
# logical token positions in `[0, cache_length)`; this entry point remaps them
# to physical paged-cache rows via `paged_sparse_kv_index_remap` before
# invoking the kernel.
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Cross attention
#
# Expected kernel name format:
# mo.cross_attention.<padded/ragged>.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Mixture of Experts
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.moe.create.indices")
struct Struct_moe_create_indices:
    """Registers the `mo.moe.create.indices` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        token_expert_order: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_start_indices: OutputTensor[dtype=.uint32, rank=1, ...],
        restore_token_order: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        expert_usage_stats: OutputTensor[dtype=.uint32, rank=1, ...],
        topk_ids: InputTensor[dtype=.int32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        moe_create_indices[target=target](
            token_expert_order.to_tile_tensor[.int64](),
            expert_start_indices.to_tile_tensor[.int64](),
            restore_token_order.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            expert_usage_stats.to_tile_tensor[.int64](),
            topk_ids.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.moe.router.group.limited")
struct Struct_moe_router_group_limited:
    """Registers the `mo.moe.router.group.limited` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        scores_type: DType,
        bias_type: DType,
        //,
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        n_groups: Int,
        topk_group: Int,
        norm_weights: Bool,
        target: StaticString,
    ](
        expert_indices: OutputTensor[dtype=.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        expert_scores: FusedInputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def scores_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[scores_type, width]:
            return expert_scores._lambda_load[width=width](coords)

        router_group_limited[
            n_routed_experts,
            n_experts_per_tok,
            n_groups,
            topk_group,
            norm_weights,
            target=target,
            scores_input_fn=OptionalReg[
                def[
                    width: Int
                ](IndexList[2]) capturing -> SIMD[scores_type, width]
            ](scores_input_fn),
        ](
            expert_indices.to_tile_tensor[.int64](),
            expert_weights.to_tile_tensor[.int64](),
            expert_scores.to_tile_tensor[.int64]().as_immut(),
            expert_bias.to_tile_tensor[.int64]().as_immut(),
            routed_scaling_factor,
            context,
        )


@extensibility.register("mo.moe.create.indices.with.scales.offset")
struct Struct_moe_create_indices_with_scales_offset:
    """Registers the `mo.moe.create.indices.with.scales.offset` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
    ](
        token_expert_order: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_start_indices: OutputTensor[dtype=.uint32, rank=1, ...],
        restore_token_order: OutputTensor[dtype=.uint32, rank=1, ...],
        expert_ids: OutputTensor[dtype=.int32, rank=1, ...],
        expert_usage_stats: OutputTensor[dtype=.uint32, rank=1, ...],
        scales_offset: OutputTensor[dtype=.uint32, rank=1, ...],
        topk_ids: InputTensor[dtype=.int32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        moe_create_indices[target=target](
            token_expert_order.to_tile_tensor[.int64](),
            expert_start_indices.to_tile_tensor[.int64](),
            restore_token_order.to_tile_tensor[.int64](),
            expert_ids.to_tile_tensor[.int64](),
            expert_usage_stats.to_tile_tensor[.int64](),
            topk_ids.to_tile_tensor[.int64](),
            context,
            scales_offset_p=scales_offset._ptr.as_unsafe_any_origin(),
        )


@extensibility.register("mo.moe.single.group.router.eplb")
struct Struct_moe_single_group_router_eplb:
    """Registers the `mo.moe.single.group.router.eplb` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        scores_type: DType,
        bias_type: DType,
        //,
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        norm_weights: Bool,
        num_log: Int,
        max_replicas: Int,
        hash_decorrelate: Bool,
        target: StaticString,
    ](
        expert_indices: OutputTensor[dtype=.int32, rank=2, ...],
        expert_indices_log: OutputTensor[dtype=.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        expert_scores: FusedInputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        logcnt: InputTensor[dtype=.int32, rank=2, ...],
        log2phy: InputTensor[dtype=.int32, rank=3, ...],
        layer_idx: InputTensor[dtype=.int32, rank=1, ...],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def scores_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[scores_type, width]:
            return expert_scores._lambda_load[width=width](coords)

        single_group_router_eplb[
            n_routed_experts,
            n_experts_per_tok,
            norm_weights=norm_weights,
            num_log=num_log,
            max_replicas=max_replicas,
            hash_decorrelate=hash_decorrelate,
            target=target,
            scores_input_fn=OptionalReg[
                def[
                    width: Int
                ](IndexList[2]) capturing -> SIMD[scores_type, width]
            ](scores_input_fn),
        ](
            expert_indices.to_tile_tensor[.int64](),
            expert_indices_log.to_tile_tensor[.int64](),
            expert_weights.to_tile_tensor[.int64](),
            expert_scores.to_tile_tensor[.int64]().as_immut(),
            expert_bias.to_tile_tensor[.int64]().as_immut(),
            logcnt.to_tile_tensor[.int64]().as_immut(),
            log2phy.to_tile_tensor[.int64]().as_immut(),
            layer_idx.to_tile_tensor[.int64]().as_immut(),
            routed_scaling_factor,
            context,
        )


@extensibility.register("mo.moe.single.group.router")
struct Struct_moe_single_group_router:
    """Registers the `mo.moe.single.group.router` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        scores_type: DType,
        bias_type: DType,
        //,
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        norm_weights: Bool,
        target: StaticString,
    ](
        expert_indices: OutputTensor[dtype=.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        expert_scores: FusedInputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        routed_scaling_factor: Float32,
        context: DeviceContext,
    ) raises:
        @__parameter
        @always_inline
        def scores_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[scores_type, width]:
            return expert_scores._lambda_load[width=width](coords)

        single_group_router[
            n_routed_experts,
            n_experts_per_tok,
            norm_weights=norm_weights,
            target=target,
            scores_input_fn=OptionalReg[
                def[
                    width: Int
                ](IndexList[2]) capturing -> SIMD[scores_type, width]
            ](scores_input_fn),
        ](
            expert_indices.to_tile_tensor[.int64](),
            expert_weights.to_tile_tensor[.int64](),
            expert_scores.to_tile_tensor[.int64]().as_immut(),
            expert_bias.to_tile_tensor[.int64]().as_immut(),
            routed_scaling_factor,
            context,
        )


@extensibility.register("mo.moe.sink.gate.router")
struct Struct_moe_sink_gate_router:
    """Registers the `mo.moe.sink.gate.router` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        scores_type: DType,
        bias_type: DType,
        //,
        n_routed_experts: Int,
        n_experts_per_tok: Int,
        n_shared_experts: Int,
        target: StaticString,
    ](
        expert_indices: OutputTensor[dtype=.int32, rank=2, ...],
        expert_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        sink_weights: OutputTensor[dtype=scores_type, rank=2, ...],
        logits: InputTensor[dtype=scores_type, rank=2, ...],
        expert_bias: InputTensor[dtype=bias_type, rank=1, ...],
        global_scale: InputTensor[dtype=scores_type, rank=1, ...],
        route_scale: Float32,
        context: DeviceContext,
    ) raises:
        sink_gate_router[
            n_routed_experts,
            n_experts_per_tok,
            n_shared_experts,
            target=target,
        ](
            expert_indices.to_tile_tensor[.int64](),
            expert_weights.to_tile_tensor[.int64](),
            sink_weights.to_tile_tensor[.int64](),
            logits.to_tile_tensor[.int64]().as_immut(),
            expert_bias.to_tile_tensor[.int64]().as_immut(),
            global_scale.to_tile_tensor[.int64]().as_immut(),
            route_scale,
            context,
        )


@extensibility.register("mo.moe.eplb.remap")
struct Struct_moe_eplb_remap:
    """Registers the `mo.moe.eplb.remap` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        num_log: Int,
        max_replicas: Int,
        K: Int,
        hash_decorrelate: Bool,
        target: StaticString,
    ](
        phy_idx: OutputTensor[dtype=.int32, rank=2, ...],
        router_idx: InputTensor[dtype=.int32, rank=2, ...],
        logcnt: InputTensor[dtype=.int32, rank=2, ...],
        log2phy: InputTensor[dtype=.int32, rank=3, ...],
        layer_idx: InputTensor[dtype=.int32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        eplb_remap[
            num_log=num_log,
            max_replicas=max_replicas,
            K=K,
            hash_decorrelate=hash_decorrelate,
            target=target,
        ](
            phy_idx.to_tile_tensor[.int64](),
            router_idx.to_tile_tensor[.int64]().as_immut(),
            logcnt.to_tile_tensor[.int64]().as_immut(),
            log2phy.to_tile_tensor[.int64]().as_immut(),
            layer_idx.to_tile_tensor[.int64]().as_immut(),
            context,
        )


# ===-----------------------------------------------------------------------===#
# KV Cache Store
#
# Expected kernel name format:
# mo.kv_cache.store.paged.<ragged/padded>
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# LayoutTransforms
# ===-----------------------------------------------------------------------===#


# TODO(GEX-1492): use filter_rank+1 instead of packed_filter_rank
def layout_transform_conv_transpose_filter_common[
    dtype: DType,
    filter_rank: Int,
    packed_filter_rank: Int,
](
    packed_filter: ManagedTensorSlice[
        dtype=dtype, rank=packed_filter_rank, ...
    ],
    filter: ManagedTensorSlice[dtype=dtype, rank=filter_rank, ...],
):
    """Packs a transposed-convolution filter into the layout expected by the conv_transpose kernels.

    Parameters:
        dtype: Element type of the `filter` and `packed_filter` tensors.
        filter_rank: Number of dimensions of the input `filter` tensor.
        packed_filter_rank: Number of dimensions of the `packed_filter`
            output tensor; must equal `filter_rank + 1`.

    Args:
        packed_filter: Output tensor holding the packed filter in the
            layout expected by the conv_transpose kernels.
        filter: Input transposed-convolution filter tensor to pack.
    """
    comptime assert filter_rank + 1 == packed_filter_rank
    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_transpose_filter(
        filter.to_tile_tensor[.int64](),
        packed_filter.to_tile_tensor[.int64](),
        1,
    )


@extensibility.register("pack_conv_transpose_filter_shape")
struct PackConvTransposeFilterShape:
    """Registers the `pack_conv_transpose_filter_shape` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        rank: Int,
        filter_type: DType,
    ](filter_buf: InputTensor[dtype=filter_type, rank=rank, ...]) raises:
        raise Error("Only meant to be used for shape function!")


@extensibility.register_shape_function("pack_conv_transpose_filter_shape")
def pack_conv_transpose_filter_shape_shape(
    filter_buf: Some[TileTensorable],
) -> IndexList[type_of(filter_buf).rank + 1]:
    """Computes the output shape for the `pack_conv_transpose_filter_shape` graph op.

    Args:
        filter_buf: Input transposed-convolution filter tensor whose packed
            output shape is computed.
    """
    return rebind[IndexList[type_of(filter_buf).rank + 1]](
        pack_filter_shape_conv_transpose(filter_buf.to_tile_tensor(), 1)
    )


# Wrapper that take `num_groups` as a parameter.
# This is required unti `mo.layout.transform` passes `num_groups` as a runtime
# value.
def layout_transform_conv_filter_common[
    dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
](
    packed_filter: ManagedTensorSlice[dtype=dtype, rank=packed_rank, ...],
    filter: ManagedTensorSlice[dtype=dtype, rank=filter_rank, ...],
):
    """Packs a convolution filter into the layout expected by the conv kernels.

    Parameters:
        dtype: Element type of the `filter` and `packed_filter` tensors.
        filter_rank: Number of dimensions of the input `filter` tensor.
        packed_rank: Number of dimensions of the `packed_filter` output
            tensor; must equal `filter_rank + 1`.
        num_groups: Number of convolution groups for groupwise
            convolution.

    Args:
        packed_filter: Output tensor holding the packed filter in the
            layout expected by the conv kernels.
        filter: Input convolution filter tensor to pack.
    """
    comptime assert packed_rank == filter_rank + 1

    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_filter(
        filter.to_tile_tensor[.int64](),
        packed_filter.to_tile_tensor[.int64](),
        num_groups,
    )


def _layout_transform_conv_filter_from_fcrs[
    dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
](
    packed_filter: ManagedTensorSlice[dtype=dtype, rank=packed_rank, ...],
    filter: ManagedTensorSlice[dtype=dtype, rank=filter_rank, ...],
):
    comptime assert packed_rank == filter_rank + 1

    # With the compiler-level FCRS→RSCF transpose in PatternFusion,
    # this kernel should no longer be called. But keep it as a fallback
    # using int64 convention (same as the RSCF path).
    _pack_conv_filter_from_fcrs(
        filter.to_tile_tensor[.int64](),
        packed_filter.to_tile_tensor[.int64](),
        num_groups,
    )


# Note: These FCRS/FCQRS kernels are currently unused — the compiler
# transposes FCRS to RSCF in PatternFusion before packing, so only the
# RSCF kernels above are invoked. Kept as fallback; can be removed in cleanup.


# ===-----------------------------------------------------------------------===#
# RMSNorm
#
# Expected kernel name format:
# mo.rms_norm_kv_cache.<padded/ragged>.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Print KV Cache
#
# Expected kernel name format:
# mo.print_kv_cache.paged
# ===-----------------------------------------------------------------------===#


def print_kv_cache_paged_generic_kernel_api[
    dtype: DType,
    //,
    target: StaticString,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    valid_lengths: InputTensor[dtype=.uint32, rank=1, ...],
    kv_collection: PagedKVCacheCollection[dtype, kv_params, page_size, ...],
    layer_idx: UInt32,
    is_print_compact: InputTensor[dtype=.bool, rank=1, ...],
    context: DeviceContext,
) raises:
    """Prints the contents of a paged KV cache for debugging.

    Parameters:
        dtype: Element type of the KV cache `blocks` tensor (inferred).
        target: Target device identifier for kernel dispatch.
        kv_params: Static KV cache parameters carrying `num_heads` and
            `head_size`.
        page_size: Number of tokens stored per page in the paged KV cache.

    Args:
        valid_lengths: One-dimensional tensor of shape `[batch]` giving the
            valid length of each sequence; only positions within these
            lengths are printed.
        kv_collection: Paged KV cache collection holding the keys and values
            to print; the cache for this layer is retrieved via `layer_idx`.
        layer_idx: Index of the layer whose key and value caches are printed.
        is_print_compact: One-element boolean tensor; when element zero is
            true, abbreviates the output with ellipses (CPU only).
        context: Device context used to copy device buffers to host on GPU
            targets.
    """
    comptime if is_gpu[target]():
        print_kv_cache_paged_generic_gpu[target](
            valid_lengths.to_layout_tensor(),
            kv_collection,
            layer_idx,
            True,
            context,
        )
    elif is_cpu[target]():
        print_kv_cache_paged_generic_cpu[target](
            valid_lengths.to_layout_tensor(),
            kv_collection,
            layer_idx,
            is_print_compact[0],
            context,
        )


# ===-----------------------------------------------------------------------===#
# Matmul KV cache
#
# Expected kernel name format:
# mo.kv_matmul.ragged.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Matmul K cache
#
# Expected kernel name format:
# mo.k_matmul.ragged.paged
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Sampling Operations
# ===-----------------------------------------------------------------------===#


@extensibility.register("sampler.fused_token_sampling")
struct Struct_fused_token_sampling:
    """Registers the `sampler.fused_token_sampling` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        out_idx_type: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_idxs: OutputTensor[dtype=out_idx_type, rank=rank, ...],
        K: InputTensor[dtype=.int64, rank=1, ...],
        max_k: Scalar,
        temperature: InputTensor[dtype=.float32, rank=1, ...],
        top_p: InputTensor[dtype=.float32, rank=1, ...],
        min_top_p: Float32,
        min_p: InputTensor[dtype=.float32, rank=1, ...],
        seed: InputTensor[dtype=.uint64, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_valid_target[target](), "not a valid target"

        comptime if is_cpu[target]():
            # When top_k == 1, argmax is equivalent to our topk_fused_sampling with k == 1
            # However, switching to just using our topk_fused_sampling leads to a -37% perf
            # drop in q4_k benchmarking for llama 3.
            if max_k == 1:
                argmax(
                    input.to_tile_tensor[.int64](),
                    rank - 1,
                    out_idxs.to_tile_tensor[.int64](),
                    Optional[DeviceContext](ctx),
                )
                return
            _fused_token_sampling_cpu(
                Int(max_k),
                input.to_tile_tensor[.int64](),
                out_idxs.to_tile_tensor[.int64](),
                k=K.to_tile_tensor[.int64]().as_unsafe_any_origin().as_immut(),
                temperature=temperature.to_tile_tensor[.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                top_p=top_p.to_tile_tensor[.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                seed=seed.to_tile_tensor[.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
            )
        else:
            _fused_token_sampling_gpu(
                ctx,
                Int(max_k),
                min_top_p,
                input.to_tile_tensor[.int64](),
                out_idxs.to_tile_tensor[.int64](),
                k=K.to_tile_tensor[.int64]().as_unsafe_any_origin().as_immut(),
                temperature=temperature.to_tile_tensor[.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                top_p=top_p.to_tile_tensor[.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                min_p=min_p.to_tile_tensor[.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
                seed=seed.to_tile_tensor[.int64]()
                .as_unsafe_any_origin()
                .as_immut(),
            )


@extensibility.register("sampler.fused_token_sampling_with_dist")
struct Struct_fused_token_sampling_with_dist:
    """Registers the `sampler.fused_token_sampling_with_dist` graph op.

    Samples one token per row under joint top-k/top-p with temperature, and
    also returns the masked, renormalized distribution it drew from.
    Speculative decoding subtracts that distribution to build its rejection
    residual, and reads the sampled token's probability out of it. Op arity
    is fixed, so this is a second registration over the same kernel as
    `sampler.fused_token_sampling` rather than an optional output on it.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        dist_dtype: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_tokens: OutputTensor[dtype=.int64, rank=1, ...],
        out_dist: OutputTensor[dtype=dist_dtype, rank=2, ...],
        K: InputTensor[dtype=.int64, rank=1, ...],
        max_k: Scalar,
        temperature: InputTensor[dtype=.float32, rank=1, ...],
        top_p: InputTensor[dtype=.float32, rank=1, ...],
        seed: InputTensor[dtype=.uint64, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "sampler.fused_token_sampling_with_dist is GPU-only"
        topk_topp_sampling_from_prob[
            from_logits=True, emit_dist=True, dist_dtype=dist_dtype
        ](
            ctx,
            input.to_tile_tensor[.int64](),
            out_tokens.to_tile_tensor[.int64](),
            Int(max_k),
            top_k_arr=K.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_immut(),
            top_p_arr=top_p.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_immut(),
            temperature=temperature.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_immut(),
            rng_seed=seed.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_immut(),
            out_dist=out_dist.to_tile_tensor[
                DType.int64
            ]().as_unsafe_any_origin(),
        )


@extensibility.register("sampler.topk_topp_masked_probs")
struct Struct_topk_topp_masked_probs:
    """Registers the `sampler.topk_topp_masked_probs` graph op.

    Writes each row's top-k/top-p masked renormalized softmax, without
    sampling. Speculative decoding verification reads the target's masked
    probabilities and builds its rejection residual straight from this
    tensor, so nothing is sorted and no distribution is rebuilt in-graph.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        probs: OutputTensor[dtype=.float32, rank=2, ...],
        K: InputTensor[dtype=.int64, rank=1, ...],
        temperature: InputTensor[dtype=.float32, rank=1, ...],
        top_p: InputTensor[dtype=.float32, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "sampler.topk_topp_masked_probs is GPU-only"
        topk_topp_masked_probs(
            ctx,
            input.to_tile_tensor[.int64](),
            probs.to_tile_tensor[.int64]().as_unsafe_any_origin(),
            top_k_val=-1,
            top_k_arr=K.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_immut(),
            top_p_arr=top_p.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_immut(),
            temperature=temperature.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_immut(),
        )


@extensibility.register("sampler.gumbel_argmax_from_probs")
struct Struct_gumbel_argmax_from_probs:
    """Registers the `sampler.gumbel_argmax_from_probs` graph op.

    Draws one token per row proportionally to a row of unnormalized
    probabilities, by Gumbel-max over `ln(p)`. The noise comes from the
    per-row seed inside the kernel, so the caller passes no noise tensor.
    Rows with equal seeds draw with equal noise, which is how a request's
    draft positions share one noise row.
    """

    @always_inline
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_tokens: OutputTensor[dtype=.int64, rank=1, ...],
        seed: InputTensor[dtype=.uint64, rank=1, ...],
        probs: InputTensor[dtype=.float32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "sampler.gumbel_argmax_from_probs is GPU-only"
        gumbel_sampling_fused_gpu[from_probs=True](
            ctx,
            probs.to_tile_tensor[.int64](),
            out_tokens.to_tile_tensor[.int64](),
            seed=seed.to_tile_tensor[.int64]()
            .as_unsafe_any_origin()
            .as_immut(),
        )


@extensibility.register("min_p_sampling")
struct Struct_min_p_sampling:
    """Registers the `min_p_sampling` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        out_idx_type: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_token_ids: OutputTensor[dtype=out_idx_type, rank=rank, ...],
        min_ps: InputTensor[dtype=dtype, rank=1, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        temperature: Scalar[dtype],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_valid_target[target](), "not a valid target"

        comptime if is_cpu[target]():
            min_p_sampling_cpu(
                min_ps.to_tile_tensor[.int64](),
                input.to_tile_tensor[.int64](),
                out_token_ids.to_tile_tensor[.int64](),
                temperature,
            )
        else:
            min_p_sampling_gpu(
                ctx,
                min_ps.to_tile_tensor[.int64](),
                input.to_tile_tensor[.int64](),
                out_token_ids.to_tile_tensor[.int64](),
                temperature,
            )


@extensibility.register("sampler.apply_penalties")
struct Struct_sampler_apply_penalties:
    """Registers the `sampler.apply_penalties` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        logit_type: DType,
        penalty_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        logits: MutableInputTensor[dtype=logit_type, rank=rank, ...],
        compressed_frequency_data: InputTensor[dtype=.int32, rank=2, ...],
        frequency_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        frequency_penalty: InputTensor[dtype=penalty_type, rank=1, ...],
        presence_penalty: InputTensor[dtype=penalty_type, rank=1, ...],
        repetition_penalty: InputTensor[dtype=penalty_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_valid_target[target](), "not a valid target"

        apply_penalties_to_logits[target=target](
            logits.to_tile_tensor[.int64](),
            compressed_frequency_data.to_tile_tensor[.int64](),
            frequency_offsets.to_tile_tensor[.int64](),
            frequency_penalty.to_tile_tensor[.int64](),
            presence_penalty.to_tile_tensor[.int64](),
            repetition_penalty.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("sampler.update_frequency_data")
struct Struct_sampler_update_frequency_data:
    """Registers the `sampler.update_frequency_data` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        token_type: DType,
        //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        compressed_frequency_data: MutableInputTensor[
            dtype=DType.int32, rank=2, ...
        ],
        frequency_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        new_tokens: InputTensor[dtype=token_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_valid_target[target](), "not a valid target"

        update_frequency_data[target=target](
            compressed_frequency_data.to_tile_tensor[.int64](),
            frequency_offsets.to_tile_tensor[.int64](),
            new_tokens.to_tile_tensor[.int64](),
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Misc Operations
# ===-----------------------------------------------------------------------===#


@always_inline
def _check_signal_buffer_size(
    signal_buffer_size: Int, input_size_bytes: Int
) raises:
    # The signal buffer has to be large enough to hold the entire input buffer.
    var min_signal_buffer_size = size_of[Signal]() + input_size_bytes
    if signal_buffer_size < min_signal_buffer_size:
        raise Error(
            "Expected signal buffer to be at least ",
            min_signal_buffer_size,
            " bytes, but got ",
            signal_buffer_size,
            (
                ". This error can appear when running large requests through"
                " MAX serve without chunked prefill. If so, try enabling"
                " chunked prefill with --enable-chunked-prefill. Otherwise,"
                " consider increasing the signal buffer size."
            ),
        )


def _partitioned_scratch_requirement[
    num_devices: Int, dtype: DType
](input_elems: Int) -> Int:
    """Calculate a trivial scratch memory requirement for comm kernels.

    This applies for comm kernels which simply partition the input tensor between devices.
    """
    comptime pessemistic_simd_width = 32
    var num_vecs = ceildiv(input_elems, pessemistic_simd_width)
    var vecs_per_device = ceildiv(num_vecs, num_devices)

    return vecs_per_device * pessemistic_simd_width * size_of[dtype]()


@extensibility.register("mo.bundled.allreduce.sum")
struct BundledAllReduceSum:
    """Registers the `mo.bundled.allreduce.sum` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        ctx: DeviceContext,
    ) capturing raises:
        """Per-device allreduce sum, for use with mo.parallel dispatch.

        Unlike DistributedAllReduceSum which dispatches to all GPUs internally,
        this kernel handles a single GPU. The mo.parallel framework is
        responsible for launching one instance per device and passing all N
        input buffers to each launch.

        Parameters:
            dtype: Element type of the input and output tensors.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            output: Output tensor for THIS GPU.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # allreduce 2-stage uses size/ngpus scratch space
        var scratch_buffer_size_bytes = _partitioned_scratch_requirement[
            num_devices, dtype
        ](inputs[0].size())
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[.int64]().as_immut()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        var out_buf = output.to_tile_tensor[.int64]()
        var rank_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )

        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[.int64]().as_immut()
            )
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        @always_inline
        @__parameter
        def output_lambda[
            _dtype: DType,
            _width: SIMDLength,
            *,
            _alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            output._lambda_store[width=_width, element_alignment=_alignment](
                rebind[IndexList[rank]](coord_to_index_list(coords)),
                rebind[SIMD[dtype, _width]](val),
            )

        allreduce[
            ngpus=num_devices,
            output_lambda=output_lambda,
        ](
            in_tensors,
            out_buf,
            rank_sigs,
            ctx,
        )


@extensibility.register("mo.composite.bundled.allreduce_add_rms_norm_quant_fp8")
struct BundledAllReduceAddRMSNormQuantFP8:
    """Registers the `mo.composite.bundled.allreduce_add_rms_norm_quant_fp8` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        output_type: DType,
        scales_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=rank, ...],
        out_scale: OutputTensor[dtype=scales_type, rank=rank, ...],
        out_residual: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: InputVariadicTensors[dtype=dtype, rank=rank, ...],
        signal_buffers: MutableInputVariadicTensors[
            dtype=DType.uint8, rank=1, ...
        ],
        residual: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: InputTensor[dtype=DType.float32, ...],
        weight_offset: InputTensor[dtype=dtype, ...],
        scale_ub: InputTensor[dtype=DType.float32, ...],
        ctx: DeviceContext,
    ) capturing raises:
        """Per-device fused allreduce.sum + add + rms_norm + fp8 quantize.

        Single-device analog of `DistributedAllReduceAddRMSNormQuantFP8`, for
        use inside `mo.parallel`. The parallel framework launches one
        instance per GPU; this kernel invokes the same underlying primitive
        (`allreduce_residual_rmsnorm`) that the distributed variant calls
        from within `_launch_device_collective`, but for a single device.

        Parameters:
            dtype: Element type of the input and residual tensors.
            output_type: Element type of the quantized output tensor.
            scales_type: Element type of the per-token scale tensor.
            rank: Number of dimensions in the input and output tensors.
            target: Target device identifier for code generation.
            _trace_name: Trace name used for profiling and debugging.

        Args:
            output: FP8 quantized output tensor for THIS GPU.
            out_scale: Per-token scale tensor for THIS GPU.
            out_residual: Post-add residual tensor for THIS GPU.
            inputs: Input tensors from ALL participating GPUs.
            signal_buffers: Signal buffers for ALL participating GPUs.
            residual: Residual tensor for THIS GPU.
            gamma: RMSNorm weight for THIS GPU.
            epsilon: RMSNorm epsilon scalar (host).
            weight_offset: RMSNorm weight offset scalar (host).
            scale_ub: Quantization scale upper bound scalar (host).
            ctx: Device context for THIS GPU.
        """
        comptime num_devices = inputs.size
        comptime assert signal_buffers.size == num_devices, (
            "expected allreduce inputs and signal buffers to have"
            " the same number of elements"
        )

        # Logic copied from kernel host code
        var in_num_elems = inputs[0].size()
        comptime last_dim_idx = type_of(inputs[0]).rank - 1
        var cols = inputs[0].dim_size[last_dim_idx]()
        var rows = in_num_elems // cols
        var rows_per_rank = ceildiv(rows, num_devices)

        # Output scratch holds fp8 (1 byte) when quantizing; this op is
        # FP8-only, but size by output_type so the math stays correct if the
        # output ever matches the input dtype (no-quant path).
        var output_size_bytes = cols * rows_per_rank * size_of[output_type]()
        var pessimistic_simd_width = 32  # just to be safe...
        var scales_size_bytes = (
            align_up(
                rows_per_rank * size_of[scales_type](), pessimistic_simd_width
            ) if output_type
            != dtype else 0
        )
        var residual_size_bytes = cols * rows_per_rank * size_of[dtype]()

        var scratch_buffer_size_bytes = (
            output_size_bytes + scales_size_bytes + residual_size_bytes
        )
        _check_signal_buffer_size(
            signal_buffers[0].size(), scratch_buffer_size_bytes
        )

        comptime InputTensorType = type_of(
            inputs[0].to_tile_tensor[.int64]().as_immut()
        )
        var in_tensors = Array[InputTensorType, num_devices](uninitialized=True)
        var rank_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )

        comptime for i in range(num_devices):
            in_tensors[i] = rebind[InputTensorType](
                inputs[i].to_tile_tensor[.int64]().as_immut()
            )
            rank_sigs[i] = (
                signal_buffers[i]._ptr.bitcast[Signal]().as_unsafe_any_origin()
            )

        allreduce_residual_rmsnorm(
            in_tensors,
            residual.to_tile_tensor[.int64]().as_immut(),
            output.to_tile_tensor[.int64](),
            out_residual.to_tile_tensor[.int64](),
            gamma.to_tile_tensor[.int64](),
            epsilon.unsafe_ptr()[],
            weight_offset.unsafe_ptr()[],
            scale_ub.unsafe_ptr()[],
            out_scale.to_tile_tensor[.int64](),
            rank_sigs,
            ctx,
        )


# Note: this is not a "real" index_tensor op that covers all cases, but rather
# a stopgap measure for some important models (DLRM, CLIP-ViT, LLaMa2)


# ===-----------------------------------------------------------------------===#
# Eagle Prefill Shift Tokens
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.eagle_prefill_shift_tokens")
struct EaglePrefillShiftTokens:
    """Registers the `mo.eagle_prefill_shift_tokens` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        tokens: InputTensor[dtype=dtype, rank=rank, ...],
        offsets: InputTensor[dtype=.uint32, rank=1, ...],
        shift_next_tokens: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        eagle_prefill_shift_tokens[target=target](
            output.to_tile_tensor[.int64](),
            tokens.to_tile_tensor[.int64](),
            offsets.to_tile_tensor[.uint32](),
            shift_next_tokens.to_tile_tensor[.int64](),
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# KV Cache Ragged RAdd Kernel
# ===-----------------------------------------------------------------------===#


@extensibility.register("learnable_2d_interp_pos_emb")
struct Learnable2DInterpPosEmb:
    """Registers the `learnable_2d_interp_pos_emb` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        weight: InputTensor[dtype=dtype, rank=3, ...],
        grid_thws: InputTensor[dtype=.int64, rank=2, ...],
        time_weight: InputTensor[dtype=.float32, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "learnable_2d_interp_pos_emb only supported on GPUs"

        learnable_2d_interp_pos_emb[dtype](
            output.to_tile_tensor[.int64](),
            x.to_tile_tensor[.int64](),
            weight.to_tile_tensor[.int64](),
            grid_thws.to_tile_tensor[.int64](),
            time_weight.to_tile_tensor[.int64](),
            ctx,
        )


@extensibility.register("mo.spatial_merge")
struct SpatialMerge:
    """Registers the `mo.spatial_merge` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
        grid_thw: InputTensor[dtype=.int64, rank=2, ...],
        hidden_size: Int32,
        merge_size: Int32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "spatial_merge only supported on GPUs"

        spatial_merge[dtype](
            output.to_tile_tensor[.int64](),
            input.to_tile_tensor[.int64](),
            grid_thw.to_tile_tensor[.int64](),
            Int(hidden_size),
            Int(merge_size),
            ctx,
        )


@extensibility.register("tpool_patch_merger")
struct TPoolPatchMerger:
    """Registers the `tpool_patch_merger` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        input: InputTensor[dtype=dtype, rank=2, ...],
        grid_thws: InputTensor[dtype=.int64, rank=2, ...],
        kH: Int32,
        kW: Int32,
        max_h: Int32,
        max_w: Int32,
        _total_output_patches: Int32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "tpool_patch_merger only supported on GPUs"

        var out_tt = output.to_tile_tensor[.int64]()
        var in_tt = input.to_tile_tensor[.int64]()
        var grid_tt = grid_thws.to_tile_tensor[.int64]()

        nn_tpool_patch_merger[dtype](
            TileTensor(
                out_tt._storage.unsafe_origin_cast[MutAnyOrigin](),
                out_tt.layout,
            ),
            TileTensor(
                in_tt._storage.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
                in_tt.layout,
            ),
            TileTensor(
                grid_tt._storage.as_imm().unsafe_origin_cast[ImmutAnyOrigin](),
                grid_tt.layout,
            ),
            Int(kH),
            Int(kW),
            Int(max_h),
            Int(max_w),
            ctx,
        )


@extensibility.register_shape_function("tpool_patch_merger")
def tpool_patch_merger_shape(
    input: Some[Tensor],
    _grid_thws: Some[Tensor],
    _kH: Int32,
    _kW: Int32,
    _max_h: Int32,
    _max_w: Int32,
    total_output_patches: Int32,
) -> IndexList[2]:
    """Computes the output shape for the `tpool_patch_merger` graph op.

    Args:
        input: Rank-2 input tensor of shape `(n_tokens, D)`; the second
            dimension is preserved as the output's second dimension.
        _grid_thws: Rank-2 grid dimensions tensor of shape
            `(n_vids, 3)` holding the `(T, H, W)` triple per video.
        _kH: Merge kernel height, in elements.
        _kW: Merge kernel width, in elements.
        _max_h: Maximum `H` across all videos, used for launch grid
            sizing.
        _max_w: Maximum `W` across all videos, used for launch grid
            sizing.
        total_output_patches: Total number of output patches across all
            videos; becomes the first dimension of the output shape.
    """
    comptime assert type_of(input).rank == 2, "input must be rank 2"
    comptime assert type_of(_grid_thws).rank == 2, "_grid_thws must be rank 2"
    comptime assert (
        type_of(_grid_thws).dtype == .int64
    ), "_grid_thws dtype must be int64"
    var input_shape = coord_to_index_list(input.shape().tuple())
    return IndexList[2](Int(total_output_patches), Int(input_shape[1]))


# ===-----------------------------------------------------------------------===#
# State-space kernels
# ===-----------------------------------------------------------------------===#


@extensibility.register("gated_delta_conv1d_fwd")
struct GatedDeltaConv1dFwd:
    """Gated DeltaNet causal conv1d forward pass (Pass 1 of two-pass prefill).

    Reads/writes a single mutable conv-state pool of shape
    `[max_slots, conv_dim, kernel_size-1]` in place at slot
    `slot_idx[batch_item]`. No state-out tensor; the only output is the
    per-token conv output. The pool's dtype is independent of the working
    dtype, so the caller can keep the per-token tensors at fp32 while
    storing the pool at the model's native dtype (bf16).

    Tensor Shapes:
        - conv_output_ragged : [total_seq_len, conv_dim]                  (OUT)
        - qkv_input_ragged   : [total_seq_len, conv_dim]
        - conv_weight        : [conv_dim, kernel_size]
        - conv_state         : [max_slots, conv_dim, kernel_size-1]       (MUT)
        - slot_idx           : [batch_size]                          uint32
        - input_row_offsets  : [batch_size + 1]                      uint32
    """

    @staticmethod
    def execute[
        work_dtype: DType,
        state_dtype: DType,
        target: StaticString,
    ](
        conv_output_ragged: OutputTensor[dtype=work_dtype, rank=2, ...],
        qkv_input_ragged: InputTensor[dtype=work_dtype, rank=2, ...],
        conv_weight: InputTensor[dtype=work_dtype, rank=2, ...],
        conv_state: MutableInputTensor[dtype=state_dtype, rank=3, ...],
        slot_idx: InputTensor[dtype=.uint32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # Number of threads per block along the conv_dim axis.
        comptime CONV1D_BLOCK_DIM: Int = 128

        var total_seq_len = qkv_input_ragged.dim_size(0)
        var conv_dim = qkv_input_ragged.dim_size(1)
        var kernel_size = conv_weight.dim_size(1)
        var batch_size = slot_idx.dim_size(0)

        # Host-side shape sanity checks. The kernel indexes the conv_state pool
        # via `slot = slot_idx[batch_item]`; slot bounds are guaranteed by the
        # Python-side `GatedDeltaNetStateCache.claim`, so we only validate that
        # tensor shapes are mutually consistent here.
        debug_assert(
            input_row_offsets.dim_size(0) == batch_size + 1,
            (
                "gated_delta_conv1d_fwd: input_row_offsets must"
                " have batch_size + 1 entries"
            ),
        )
        debug_assert(
            conv_state.dim_size(0) > 0,
            (
                "gated_delta_conv1d_fwd: conv_state pool must"
                " have at least one slot"
            ),
        )
        debug_assert(
            conv_state.dim_size(1) == conv_dim,
            (
                "gated_delta_conv1d_fwd: conv_state pool channel"
                " dim must equal conv_dim"
            ),
        )
        debug_assert(
            conv_state.dim_size(2) == kernel_size - 1,
            (
                "gated_delta_conv1d_fwd: conv_state pool window"
                " dim must equal kernel_size - 1"
            ),
        )

        var conv_output_ragged_tt = conv_output_ragged.to_tile_tensor[
            DType.int64
        ]()
        var qkv_input_ragged_tt = qkv_input_ragged.to_tile_tensor[.int64]()
        var conv_weight_tt = conv_weight.to_tile_tensor[.int64]()
        var conv_state_tt = conv_state.to_tile_tensor[.int64]()
        var slot_idx_tt = slot_idx.to_tile_tensor[.int64]()
        var input_row_offsets_tt = input_row_offsets.to_tile_tensor[
            DType.int64
        ]()

        var qkv_input_strides = qkv_input_ragged.strides()
        var conv_weight_strides = conv_weight.strides()
        var conv_state_strides = conv_state.strides()
        var conv_output_strides = conv_output_ragged.strides()

        var qkv_input_seqlen_stride = UInt32(qkv_input_strides[0])
        var qkv_input_channel_stride = UInt32(qkv_input_strides[1])
        var conv_weight_channel_stride = UInt32(conv_weight_strides[0])
        var conv_weight_offset_stride = UInt32(conv_weight_strides[1])
        var conv_state_pool_stride = UInt32(conv_state_strides[0])
        var conv_state_channel_stride = UInt32(conv_state_strides[1])
        var conv_state_window_stride = UInt32(conv_state_strides[2])
        var conv_output_seqlen_stride = UInt32(conv_output_strides[0])
        var conv_output_channel_stride = UInt32(conv_output_strides[1])

        comptime assert is_gpu[
            target
        ](), "gated_delta_conv1d_fwd is only supported on GPU."

        var gpu_ctx = ctx
        var grid_dim_batch = batch_size
        var grid_dim_channels = ceildiv(conv_dim, CONV1D_BLOCK_DIM)

        # NOTE: Only kernel_size=4 is currently compiled (Qwen3.5 default).
        # To support a new model with a different kernel size, add a further
        # elif branch here following the same pattern.
        if kernel_size == 4:
            comptime kKernelSize = 4
            gpu_ctx.enqueue_function[
                gated_delta_conv1d_fwd_gpu[
                    work_dtype,
                    state_dtype,
                    kKernelSize,
                    CONV1D_BLOCK_DIM,
                    qkv_input_ragged_tt.LayoutType,
                    conv_weight_tt.LayoutType,
                    conv_state_tt.LayoutType,
                    slot_idx_tt.LayoutType,
                    input_row_offsets_tt.LayoutType,
                    conv_output_ragged_tt.LayoutType,
                ]
            ](
                Int32(batch_size),
                Int32(total_seq_len),
                Int32(conv_dim),
                qkv_input_ragged_tt,
                conv_weight_tt,
                conv_state_tt,
                slot_idx_tt,
                input_row_offsets_tt,
                conv_output_ragged_tt,
                qkv_input_seqlen_stride,
                qkv_input_channel_stride,
                conv_weight_channel_stride,
                conv_weight_offset_stride,
                conv_state_pool_stride,
                conv_state_channel_stride,
                conv_state_window_stride,
                conv_output_seqlen_stride,
                conv_output_channel_stride,
                grid_dim=(grid_dim_batch, grid_dim_channels),
                block_dim=(CONV1D_BLOCK_DIM,),
            )
        else:
            raise Error(
                "gated_delta_conv1d_fwd: unsupported kernel_size "
                + String(kernel_size)
                + ". Only kernel_size=4 is currently compiled; add a new elif"
                + " branch in kernels to support other sizes."
            )


@extensibility.register_shape_function("gated_delta_conv1d_fwd")
def gated_delta_conv1d_fwd_shape(
    qkv_input_ragged: Some[Tensor],
    conv_weight: Some[Tensor],
    conv_state: Some[Tensor],
    slot_idx: Some[Tensor],
    input_row_offsets: Some[Tensor],
) -> IndexList[2]:
    """Computes the output shape for the `gated_delta_conv1d_fwd` graph op.

    Args:
        qkv_input_ragged: Ragged QKV input tensor of shape
            `[total_seq_len, conv_dim]`.
        conv_weight: Convolution filter of shape
            `[conv_dim, kernel_size]`.
        conv_state: Mutable conv-state pool of shape
            `[max_slots, conv_dim, kernel_size-1]`.
        slot_idx: Per-batch slot indices into the conv-state pool, shape
            `[batch_size]`.
        input_row_offsets: Cumulative row offsets per batch, shape
            `[batch_size + 1]`.
    """
    comptime assert (
        type_of(qkv_input_ragged).rank == 2
    ), "qkv_input_ragged must be rank 2"
    comptime assert type_of(conv_weight).rank == 2, "conv_weight must be rank 2"
    comptime assert (
        type_of(conv_weight).dtype == type_of(qkv_input_ragged).dtype
    ), "qkv_input_ragged and conv_weight must share a dtype"
    comptime assert type_of(conv_state).rank == 3, "conv_state must be rank 3"
    comptime assert type_of(slot_idx).rank == 1, "slot_idx must be rank 1"
    comptime assert (
        type_of(slot_idx).dtype == .uint32
    ), "slot_idx dtype must be uint32"
    comptime assert (
        type_of(input_row_offsets).rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert (
        type_of(input_row_offsets).dtype == .uint32
    ), "input_row_offsets dtype must be uint32"
    # conv_output_ragged has same shape as qkv_input_ragged
    return rebind[IndexList[2]](
        coord_to_index_list(qkv_input_ragged.shape().tuple())
    )


@extensibility.register("gated_delta_recurrence_fwd")
struct GatedDeltaRecurrenceFwd:
    """Gated DeltaNet recurrence forward pass (Pass 2 of two-pass prefill).

    Reads/writes a single mutable recurrent-state pool of shape
    `[max_slots, num_value_heads, key_head_dim, value_head_dim]` in place
    at slot `slot_idx[batch_item]`. No state-out tensor; the only output
    is the per-token recurrence output.

    Tensor Shapes:
        - recurrence_output : [total_seq_len, value_dim]                  (OUT)
        - qkv_conv_output   : [total_seq_len, conv_dim]
        - decay_per_token   : [total_seq_len, num_value_heads]
        - beta_per_token    : [total_seq_len, num_value_heads]
        - recurrent_state   : [max_slots, num_value_heads, KD, VD]        (MUT)
        - slot_idx          : [batch_size]                          uint32
        - input_row_offsets : [batch_size + 1]                      uint32
    """

    @staticmethod
    def execute[
        work_dtype: DType,
        state_dtype: DType,
        target: StaticString,
    ](
        recurrence_output: OutputTensor[dtype=work_dtype, rank=2, ...],
        qkv_conv_output: InputTensor[dtype=work_dtype, rank=2, ...],
        decay_per_token: InputTensor[dtype=work_dtype, rank=2, ...],
        beta_per_token: InputTensor[dtype=work_dtype, rank=2, ...],
        recurrent_state: MutableInputTensor[dtype=state_dtype, rank=4, ...],
        slot_idx: InputTensor[dtype=.uint32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var total_seq_len = qkv_conv_output.dim_size(0)
        var conv_dim = qkv_conv_output.dim_size(1)
        var num_value_heads = decay_per_token.dim_size(1)
        var batch_size = slot_idx.dim_size(0)
        var key_head_dim = recurrent_state.dim_size(2)
        var value_head_dim = recurrent_state.dim_size(3)
        var value_dim = num_value_heads * value_head_dim
        # key_dim = (conv_dim - value_dim) / 2  where conv_dim = 2*key_dim + value_dim
        var key_dim = (conv_dim - value_dim) // 2
        # Validate that the config is well-formed (no corruption in input shapes).
        debug_assert(
            (conv_dim - value_dim) % 2 == 0,
            "gated_delta_recurrence_fwd: (conv_dim - value_dim) must be even",
        )
        # num_key_heads derived from recurrence_output vs decay shapes:
        # key_dim = num_key_heads * key_head_dim
        var num_key_heads = key_dim // key_head_dim
        debug_assert(
            key_dim % key_head_dim == 0,
            (
                "gated_delta_recurrence_fwd: key_dim must be"
                " divisible by key_head_dim"
            ),
        )

        # Host-side shape sanity checks. The kernel indexes the recurrent_state
        # pool via `slot = slot_idx[batch_item]`; slot bounds are guaranteed by
        # the Python-side `GatedDeltaNetStateCache.claim`, so we only validate
        # tensor shapes here.
        debug_assert(
            input_row_offsets.dim_size(0) == batch_size + 1,
            (
                "gated_delta_recurrence_fwd: input_row_offsets"
                " must have batch_size + 1 entries"
            ),
        )
        debug_assert(
            recurrent_state.dim_size(0) > 0,
            (
                "gated_delta_recurrence_fwd: recurrent_state pool"
                " must have at least one slot"
            ),
        )
        debug_assert(
            recurrent_state.dim_size(1) == num_value_heads,
            (
                "gated_delta_recurrence_fwd: recurrent_state pool"
                " value-head dim must equal num_value_heads"
            ),
        )
        debug_assert(
            decay_per_token.dim_size(0) == total_seq_len,
            (
                "gated_delta_recurrence_fwd: decay_per_token"
                " seqlen must equal qkv_conv_output seqlen"
            ),
        )
        debug_assert(
            beta_per_token.dim_size(0) == total_seq_len,
            (
                "gated_delta_recurrence_fwd: beta_per_token"
                " seqlen must equal qkv_conv_output seqlen"
            ),
        )

        var recurrence_output_tt = recurrence_output.to_tile_tensor[
            DType.int64
        ]()
        var qkv_conv_output_tt = qkv_conv_output.to_tile_tensor[.int64]()
        var decay_per_token_tt = decay_per_token.to_tile_tensor[.int64]()
        var beta_per_token_tt = beta_per_token.to_tile_tensor[.int64]()
        var recurrent_state_tt = recurrent_state.to_tile_tensor[.int64]()
        var slot_idx_tt = slot_idx.to_tile_tensor[.int64]()
        var input_row_offsets_tt = input_row_offsets.to_tile_tensor[
            DType.int64
        ]()

        var qkv_strides = qkv_conv_output.strides()
        var per_token_decay_strides = decay_per_token.strides()
        var recurrent_state_strides = recurrent_state.strides()
        var recurrence_output_strides = recurrence_output.strides()

        debug_assert(
            beta_per_token.strides() == per_token_decay_strides,
            (
                "gated_delta_recurrence_fwd: beta_per_token"
                " strides must match decay_per_token strides"
            ),
        )

        var qkv_conv_output_seqlen_stride = UInt32(qkv_strides[0])
        var qkv_conv_output_channel_stride = UInt32(qkv_strides[1])
        var per_token_seqlen_stride = UInt32(per_token_decay_strides[0])
        var per_token_head_stride = UInt32(per_token_decay_strides[1])
        var recurrent_state_slot_stride = UInt32(recurrent_state_strides[0])
        var recurrent_state_value_head_stride = UInt32(
            recurrent_state_strides[1]
        )
        var recurrent_state_key_dim_stride = UInt32(recurrent_state_strides[2])
        var recurrent_state_value_dim_stride = UInt32(
            recurrent_state_strides[3]
        )
        var recurrence_output_seqlen_stride = UInt32(
            recurrence_output_strides[0]
        )
        var recurrence_output_valuedim_stride = UInt32(
            recurrence_output_strides[1]
        )

        comptime assert is_gpu[
            target
        ](), "gated_delta_recurrence_fwd is only supported on GPU."

        var gpu_ctx = ctx
        # One CTA per (batch_item, value_head); block has value_head_dim threads.
        var num_blocks = batch_size * num_value_heads

        # NOTE: Only (key_head_dim=128, value_head_dim=128) is currently
        # compiled (Qwen3.5 default). To support a new model with different
        # head dims, add a further elif branch here following the same pattern.
        if key_head_dim == 128 and value_head_dim == 128:
            comptime kKD = 128
            comptime kVD = 128
            gpu_ctx.enqueue_function[
                gated_delta_recurrence_fwd_gpu[
                    work_dtype,
                    state_dtype,
                    kKD,
                    kVD,
                    recurrence_output_tt.LayoutType,
                    qkv_conv_output_tt.LayoutType,
                    decay_per_token_tt.LayoutType,
                    beta_per_token_tt.LayoutType,
                    recurrent_state_tt.LayoutType,
                    slot_idx_tt.LayoutType,
                    input_row_offsets_tt.LayoutType,
                ]
            ](
                Int32(batch_size),
                Int32(num_value_heads),
                Int32(num_key_heads),
                Int32(key_dim),
                recurrence_output_tt,
                recurrent_state_tt,
                slot_idx_tt,
                qkv_conv_output_tt,
                decay_per_token_tt,
                beta_per_token_tt,
                input_row_offsets_tt,
                qkv_conv_output_seqlen_stride,
                qkv_conv_output_channel_stride,
                per_token_seqlen_stride,
                per_token_head_stride,
                recurrent_state_slot_stride,
                recurrent_state_value_head_stride,
                recurrent_state_key_dim_stride,
                recurrent_state_value_dim_stride,
                recurrence_output_seqlen_stride,
                recurrence_output_valuedim_stride,
                grid_dim=(num_blocks,),
                block_dim=(kVD,),
            )
        else:
            raise Error(
                "gated_delta_recurrence_fwd: unsupported"
                + " (key_head_dim, value_head_dim) = ("
                + String(key_head_dim)
                + ", "
                + String(value_head_dim)
                + "). Only (128, 128) is currently compiled; add a new elif"
                + " branch in kernels to support other sizes."
            )


@extensibility.register_shape_function("gated_delta_recurrence_fwd")
def gated_delta_recurrence_fwd_shape(
    qkv_conv_output: Some[Tensor],
    decay_per_token: Some[Tensor],
    beta_per_token: Some[Tensor],
    recurrent_state: Some[Tensor],
    slot_idx: Some[Tensor],
    input_row_offsets: Some[Tensor],
) -> IndexList[2]:
    """Computes the output shape for the `gated_delta_recurrence_fwd` graph op.

    Args:
        qkv_conv_output: Ragged conv output of shape
            `[total_seq_len, conv_dim]`.
        decay_per_token: Per-token decay factors of shape
            `[total_seq_len, num_value_heads]`.
        beta_per_token: Per-token beta gates of shape
            `[total_seq_len, num_value_heads]`.
        recurrent_state: Mutable recurrent-state pool of shape
            `[max_slots, num_value_heads, key_head_dim, value_head_dim]`.
        slot_idx: Per-batch slot indices into the recurrent-state pool,
            shape `[batch_size]`.
        input_row_offsets: Cumulative row offsets per batch, shape
            `[batch_size + 1]`.
    """
    comptime assert (
        type_of(qkv_conv_output).rank == 2
    ), "qkv_conv_output must be rank 2"
    comptime assert (
        type_of(decay_per_token).rank == 2
    ), "decay_per_token must be rank 2"
    comptime assert (
        type_of(beta_per_token).rank == 2
    ), "beta_per_token must be rank 2"
    comptime assert (
        type_of(recurrent_state).rank == 4
    ), "recurrent_state must be rank 4"
    comptime assert (
        type_of(decay_per_token).dtype == type_of(qkv_conv_output).dtype
        and type_of(beta_per_token).dtype == type_of(qkv_conv_output).dtype
    ), "qkv_conv_output, decay_per_token, and beta_per_token must share a dtype"
    comptime assert type_of(slot_idx).rank == 1, "slot_idx must be rank 1"
    comptime assert (
        type_of(slot_idx).dtype == .uint32
    ), "slot_idx dtype must be uint32"
    comptime assert (
        type_of(input_row_offsets).rank == 1
    ), "input_row_offsets must be rank 1"
    comptime assert (
        type_of(input_row_offsets).dtype == .uint32
    ), "input_row_offsets dtype must be uint32"
    # recurrence_output: [total_seq_len, value_dim]
    var qkv_conv_output_shape = coord_to_index_list(
        qkv_conv_output.shape().tuple()
    )
    var decay_per_token_shape = coord_to_index_list(
        decay_per_token.shape().tuple()
    )
    var recurrent_state_shape = coord_to_index_list(
        recurrent_state.shape().tuple()
    )
    var total_seq_len = qkv_conv_output_shape[0]
    var num_value_heads = decay_per_token_shape[1]
    var value_head_dim = recurrent_state_shape[3]
    var value_dim = num_value_heads * value_head_dim
    return IndexList[2](total_seq_len, value_dim)


@extensibility.register("mamba2_ssd_chunk_scan_varlen_fwd")
struct Mamba2SSDChunkScanVarlenFwd[dt_softplus: Bool = True]:
    """Varlen Mamba-2 SSD chunked-scan prefill forward.

    Matches `mamba_chunk_scan_combined` semantics for the Nemotron-H
    `NemotronHMamba2Mixer`. Per-head scalar `A`, grouped `B`/`C`, per-head `dt`
    + `dt_bias` softplus. State resets at each `query_start_loc` boundary (no
    cross-sequence bleed). Gating `z` + `MambaRMSNormGated` are applied OUTSIDE
    this op (`norm_before_gate=False`).

    The registration lives here in the built-in kernel library (mirroring the
    `gated_delta_conv1d_fwd` / `gated_delta_recurrence_fwd` precedent) so the
    graph compiler / serve path can resolve the op with no out-of-tree
    `custom_extensions`. The kernel math lives in
    `state_space.mamba2_ssd_scan` (B200 / sm_100, bf16 in/out, fp32 states).

    Parameters:
        dt_softplus: If True (default), apply softplus to `dt + dt_bias`.

    Tensor shapes (varlen / ragged; time dim is the packed `total_len`):
        - y: (total_len, nheads, head_dim) - output (dtype)
        - final_states: (batch, nheads, head_dim, dstate) - out, fp32
        - x: (total_len, nheads, head_dim) - input (dtype)
        - dt: (total_len, nheads) - per-head time deltas (dtype)
        - A: (nheads,) - per-head scalar (dtype)
        - B: (total_len, ngroups, dstate) - grouped input proj (dtype)
        - C: (total_len, ngroups, dstate) - grouped output proj (dtype)
        - D: (nheads,) - skip connection (dtype, optional / empty)
        - dt_bias: (nheads,) - dt bias (dtype, optional / empty)
        - initial_states: (batch, nheads, head_dim, dstate) - in, fp32
          (optional / empty)
        - query_start_loc: (batch + 1,) - cumulative sequence lengths (int32)
        - has_initial_state: (batch,) - whether to load initial_states (bool,
          optional / empty)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        y: OutputTensor[dtype=dtype, rank=3, ...],
        final_states: OutputTensor[dtype=.float32, rank=4, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        dt: InputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=1, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        dt_bias: InputTensor[dtype=dtype, rank=1, ...],
        initial_states: InputTensor[dtype=.float32, rank=4, ...],
        query_start_loc: InputTensor[dtype=.int32, rank=1, ...],
        has_initial_state: InputTensor[dtype=.bool, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var nheads = x.dim_size(1)
        var head_dim = x.dim_size(2)
        var ngroups = B.dim_size(1)
        var dstate = B.dim_size(2)
        var batch = query_start_loc.dim_size(0) - 1
        var nheads_ngroups_ratio = nheads // ngroups

        # TileTensors: the layout-type placeholder dtype int32 follows the
        # existing varlen_selective_scan_ops idiom; kernel_dtype is supplied
        # separately. fp32 state tensors use a fp32 placeholder.
        var y_tt = y.to_tile_tensor[.int32]()
        var final_states_tt = final_states.to_tile_tensor[.float32]()
        var x_tt = x.to_tile_tensor[.int32]()
        var dt_tt = dt.to_tile_tensor[.int32]()
        var A_tt = A.to_tile_tensor[.int32]()
        var B_tt = B.to_tile_tensor[.int32]()
        var C_tt = C.to_tile_tensor[.int32]()
        var D_tt = D.to_tile_tensor[.int32]()
        var dt_bias_tt = dt_bias.to_tile_tensor[.int32]()
        var initial_states_tt = initial_states.to_tile_tensor[.float32]()
        var query_start_loc_tt = query_start_loc.to_tile_tensor[.int32]()
        var has_initial_state_tt = has_initial_state.to_tile_tensor[
            DType.int32
        ]()

        var x_strides = IndexList[3](
            x.strides()[0], x.strides()[1], x.strides()[2]
        )
        var dt_strides = IndexList[2](dt.strides()[0], dt.strides()[1])
        var A_strides = IndexList[1](A.strides()[0])
        var B_strides = IndexList[3](
            B.strides()[0], B.strides()[1], B.strides()[2]
        )
        var C_strides = IndexList[3](
            C.strides()[0], C.strides()[1], C.strides()[2]
        )
        var D_strides = IndexList[1](D.strides()[0] if D.dim_size(0) > 0 else 1)
        var dt_bias_strides = IndexList[1](
            dt_bias.strides()[0] if dt_bias.dim_size(0) > 0 else 1
        )
        var initial_states_strides = IndexList[4](
            initial_states.strides()[0],
            initial_states.strides()[1],
            initial_states.strides()[2],
            initial_states.strides()[3],
        )
        var y_strides = IndexList[3](
            y.strides()[0], y.strides()[1], y.strides()[2]
        )
        var final_states_strides = IndexList[4](
            final_states.strides()[0],
            final_states.strides()[1],
            final_states.strides()[2],
            final_states.strides()[3],
        )

        comptime dt_softplus_int8: Int8 = Int8(1) if Self.dt_softplus else Int8(
            0
        )

        if dstate != 16 and dstate != 64 and dstate != 128 and dstate != 256:
            raise Error(
                "Unsupported dstate: "
                + String(dstate)
                + ". Expected 16, 64, 128, or 256."
            )

        @__parameter
        @always_inline
        def launch_cpu[DSTATE_VAL: Int]() raises:
            mamba2_ssd_chunk_scan_varlen_fwd_cpu[dtype, DSTATE_VAL](
                nheads,
                head_dim,
                ngroups,
                nheads_ngroups_ratio,
                batch,
                dt_softplus_int8,
                x_tt,
                dt_tt,
                A_tt,
                B_tt,
                C_tt,
                D_tt,
                dt_bias_tt,
                initial_states_tt,
                y_tt,
                final_states_tt,
                query_start_loc_tt,
                has_initial_state_tt,
                x_strides,
                dt_strides,
                A_strides,
                B_strides,
                C_strides,
                D_strides,
                dt_bias_strides,
                initial_states_strides,
                y_strides,
                final_states_strides,
                Optional[DeviceContext](ctx),
            )

        @__parameter
        @always_inline
        def launch_gpu[DSTATE_VAL: Int]() raises:
            comptime BLOCK_SIZE = 64
            var num_p_blocks = ceildiv(head_dim, BLOCK_SIZE)
            comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_gpu[
                dtype,
                DSTATE_VAL,
                x_tt.LayoutType,
                dt_tt.LayoutType,
                A_tt.LayoutType,
                B_tt.LayoutType,
                C_tt.LayoutType,
                D_tt.LayoutType,
                dt_bias_tt.LayoutType,
                initial_states_tt.LayoutType,
                y_tt.LayoutType,
                final_states_tt.LayoutType,
                query_start_loc_tt.LayoutType,
                has_initial_state_tt.LayoutType,
            ]
            var compiled = ctx.compile_function[kernel]()
            ctx.enqueue_function(
                compiled,
                Int32(nheads),
                Int32(head_dim),
                Int32(ngroups),
                Int32(nheads_ngroups_ratio),
                Int32(batch),
                dt_softplus_int8,
                x_tt,
                dt_tt,
                A_tt,
                B_tt,
                C_tt,
                D_tt,
                dt_bias_tt,
                initial_states_tt,
                y_tt,
                final_states_tt,
                query_start_loc_tt,
                has_initial_state_tt,
                x_strides,
                dt_strides,
                A_strides,
                B_strides,
                C_strides,
                D_strides,
                dt_bias_strides,
                initial_states_strides,
                y_strides,
                final_states_strides,
                grid_dim=(num_p_blocks, nheads, batch),
                block_dim=(BLOCK_SIZE, 1, 1),
            )

        comptime if is_cpu[target]():
            if dstate == 256:
                launch_cpu[256]()
            elif dstate == 128:
                launch_cpu[128]()
            elif dstate == 64:
                launch_cpu[64]()
            else:
                launch_cpu[16]()
        elif is_gpu[target]():
            if dstate == 256:
                launch_gpu[256]()
            elif dstate == 128:
                launch_gpu[128]()
            elif dstate == 64:
                launch_gpu[64]()
            else:
                launch_gpu[16]()
        else:
            raise Error("Unsupported target device")


@extensibility.register_shape_function("mamba2_ssd_chunk_scan_varlen_fwd")
def mamba2_ssd_chunk_scan_varlen_fwd_shape(
    x: Some[Tensor],
    dt: Some[Tensor],
    A: Some[Tensor],
    B: Some[Tensor],
    C: Some[Tensor],
    D: Some[Tensor],
    dt_bias: Some[Tensor],
    initial_states: Some[Tensor],
    query_start_loc: Some[Tensor],
    has_initial_state: Some[Tensor],
) -> IndexList[3]:
    """Computes the output shape for the `mamba2_ssd_chunk_scan_varlen_fwd` graph op.

    Args:
        x: Packed input tensor of shape
            `(total_len, nheads, head_dim)`.
        dt: Per-head time deltas of shape `(total_len, nheads)`.
        A: Per-head scalar decay of shape `(nheads,)`.
        B: Grouped input projection of shape
            `(total_len, ngroups, dstate)`.
        C: Grouped output projection of shape
            `(total_len, ngroups, dstate)`.
        D: Per-head skip connection of shape `(nheads,)`; may
            be empty when unused.
        dt_bias: Per-head bias added to `dt` of shape `(nheads,)`;
            may be empty when unused.
        initial_states: Optional initial SSM states of shape
            `(batch, nheads, head_dim, dstate)` in `float32`; may
            be empty when `has_initial_state` is all false.
        query_start_loc: Cumulative sequence lengths of shape
            `(batch + 1,)` in `int32`.
        has_initial_state: Per-sequence flag of shape `(batch,)`
            in `bool` indicating whether to load `initial_states`;
            may be empty when no initial states are used.
    """
    comptime assert type_of(x).rank == 3, "x must be rank 3"
    comptime assert type_of(dt).rank == 2, "dt must be rank 2"
    comptime assert type_of(A).rank == 1, "A must be rank 1"
    comptime assert type_of(B).rank == 3, "B must be rank 3"
    comptime assert type_of(C).rank == 3, "C must be rank 3"
    comptime assert type_of(D).rank == 1, "D must be rank 1"
    comptime assert type_of(dt_bias).rank == 1, "dt_bias must be rank 1"
    comptime assert (
        type_of(initial_states).rank == 4
    ), "initial_states must be rank 4"
    comptime assert (
        type_of(initial_states).dtype == .float32
    ), "initial_states dtype must be float32"
    comptime assert (
        type_of(query_start_loc).rank == 1
    ), "query_start_loc must be rank 1"
    comptime assert (
        type_of(query_start_loc).dtype == .int32
    ), "query_start_loc dtype must be int32"
    comptime assert (
        type_of(has_initial_state).rank == 1
    ), "has_initial_state must be rank 1"
    comptime assert (
        type_of(has_initial_state).dtype == .bool
    ), "has_initial_state dtype must be bool"
    comptime assert (
        type_of(dt).dtype == type_of(x).dtype
        and type_of(A).dtype == type_of(x).dtype
        and type_of(B).dtype == type_of(x).dtype
        and type_of(C).dtype == type_of(x).dtype
        and type_of(D).dtype == type_of(x).dtype
        and type_of(dt_bias).dtype == type_of(x).dtype
    ), "x, dt, A, B, C, D, and dt_bias must share a dtype"
    # y has the same shape as x: (total_len, nheads, head_dim).
    return rebind[IndexList[3]](coord_to_index_list(x.shape().tuple()))


@extensibility.register("mamba2_ssd_chunk_scan_varlen_fwd_inplace")
struct Mamba2SSDChunkScanVarlenFwdInplace[dt_softplus: Bool = True]:
    """Varlen Mamba-2 SSD chunked-scan: in-place SSM-pool write-back.

    Identical to `Mamba2SSDChunkScanVarlenFwd` except final states are
    written **directly into** the `ssm_pool` buffer at
    `ssm_pool[cache_indices[b], ...]` instead of producing a separate
    `final_states` output tensor. This eliminates the graph-side
    `buffer_load → gather → scatter_nd → buffer_store` whole-pool RMW that
    otherwise dominates decode GPU time (~30 % wall-clock on B200).

    The `ssm_pool` is declared as a `MutableInputTensor` (slot-indexed
    in/out), matching the `causal_conv1d_varlen_fwd` / `gated_delta_recurrence_fwd`
    precedent. `initial_states` is also read from `ssm_pool` when
    `has_initial_state[b]` is true (no separate initial-states input needed).

    Parameters:
        dt_softplus: If True (default), apply softplus to `dt + dt_bias`.

    Tensor shapes:
        - y: (total_len, nheads, head_dim), output (dtype)  [OUT]
        - x: (total_len, nheads, head_dim), input (dtype)
        - dt: (total_len, nheads)
        - A: (nheads,)
        - B: (total_len, ngroups, dstate)
        - C: (total_len, ngroups, dstate)
        - D: (nheads,) optional/empty
        - dt_bias: (nheads,) optional/empty
        - ssm_pool: (max_slots, nheads, head_dim, dstate) fp32, or bf16 on
          Apple GPUs (storage dtype only; the scan accumulates in fp32)  [MUT]
        - query_start_loc: (batch + 1,) int32
        - has_initial_state: (batch,) bool optional/empty
        - cache_indices: (batch,) uint32, slot indices into ssm_pool
    """

    @staticmethod
    def execute[
        dtype: DType,
        state_dtype: DType,
        target: StaticString,
    ](
        y: OutputTensor[dtype=dtype, rank=3, ...],
        x: InputTensor[dtype=dtype, rank=3, ...],
        dt: InputTensor[dtype=dtype, rank=2, ...],
        A: InputTensor[dtype=dtype, rank=1, ...],
        B: InputTensor[dtype=dtype, rank=3, ...],
        C: InputTensor[dtype=dtype, rank=3, ...],
        D: InputTensor[dtype=dtype, rank=1, ...],
        dt_bias: InputTensor[dtype=dtype, rank=1, ...],
        # ssm_pool is declared MutableInputTensor so the graph binds the
        # caller's persistent pool buffer and routes it through the chain. Its
        # storage dtype is independent of the working dtype (fp32 everywhere;
        # bf16 on Apple GPUs — see the Apple kernel's numerics contract).
        ssm_pool: MutableInputTensor[dtype=state_dtype, rank=4, ...],
        query_start_loc: InputTensor[dtype=.int32, rank=1, ...],
        has_initial_state: InputTensor[dtype=.bool, rank=1, ...],
        cache_indices: InputTensor[dtype=.uint32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        var nheads = x.dim_size(1)
        var head_dim = x.dim_size(2)
        var ngroups = B.dim_size(1)
        var batch = query_start_loc.dim_size(0) - 1
        var nheads_ngroups_ratio = nheads // ngroups

        var y_tt = y.to_tile_tensor[.int32]()
        var x_tt = x.to_tile_tensor[.int32]()
        var dt_tt = dt.to_tile_tensor[.int32]()
        var A_tt = A.to_tile_tensor[.int32]()
        var B_tt = B.to_tile_tensor[.int32]()
        var C_tt = C.to_tile_tensor[.int32]()
        var D_tt = D.to_tile_tensor[.int32]()
        var dt_bias_tt = dt_bias.to_tile_tensor[.int32]()
        var ssm_pool_tt = ssm_pool.to_tile_tensor[.float32]()
        var query_start_loc_tt = query_start_loc.to_tile_tensor[.int32]()
        var has_initial_state_tt = has_initial_state.to_tile_tensor[
            DType.int32
        ]()
        var cache_indices_tt = cache_indices.to_tile_tensor[.uint32]()

        var x_strides = IndexList[3](
            x.strides()[0], x.strides()[1], x.strides()[2]
        )
        var dt_strides = IndexList[2](dt.strides()[0], dt.strides()[1])
        var A_strides = IndexList[1](A.strides()[0])
        var B_strides = IndexList[3](
            B.strides()[0], B.strides()[1], B.strides()[2]
        )
        var C_strides = IndexList[3](
            C.strides()[0], C.strides()[1], C.strides()[2]
        )
        var D_strides = IndexList[1](D.strides()[0] if D.dim_size(0) > 0 else 1)
        var dt_bias_strides = IndexList[1](
            dt_bias.strides()[0] if dt_bias.dim_size(0) > 0 else 1
        )
        var ssm_pool_strides = IndexList[4](
            ssm_pool.strides()[0],
            ssm_pool.strides()[1],
            ssm_pool.strides()[2],
            ssm_pool.strides()[3],
        )
        var y_strides = IndexList[3](
            y.strides()[0], y.strides()[1], y.strides()[2]
        )

        comptime dt_softplus_int8: Int8 = Int8(1) if Self.dt_softplus else Int8(
            0
        )

        var dstate = B.dim_size(2)
        if dstate != 16 and dstate != 64 and dstate != 128 and dstate != 256:
            raise Error(
                "Unsupported dstate: "
                + String(dstate)
                + ". Expected 16, 64, 128, or 256."
            )

        @__parameter
        @always_inline
        def launch_cpu[DSTATE_VAL: Int]() raises:
            # The CPU kernel stores fp32 state only; bf16 state is wired on
            # the Apple GPU kernel alone. The `comptime if` keeps the bf16
            # instantiation from elaborating this branch, but the direct call
            # below is still type-checked with `state_dtype` symbolic
            # (`comptime if` does not narrow parameter types), so the pool is
            # `rebind`-ed to its fp32 spelling — a compile-time promise the
            # compiler verifies at instantiation, where this branch only
            # exists with `state_dtype == float32`.
            comptime if state_dtype == .float32:
                mamba2_ssd_chunk_scan_varlen_fwd_inplace_cpu[dtype, DSTATE_VAL](
                    nheads,
                    head_dim,
                    ngroups,
                    nheads_ngroups_ratio,
                    batch,
                    dt_softplus_int8,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    dt_bias_tt,
                    y_tt,
                    rebind[
                        TileTensor[
                            .float32, ssm_pool_tt.LayoutType, ssm_pool_tt.origin
                        ]
                    ](ssm_pool_tt),
                    query_start_loc_tt,
                    has_initial_state_tt,
                    cache_indices_tt,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    dt_bias_strides,
                    y_strides,
                    ssm_pool_strides,
                )
            else:
                raise Error(
                    "non-fp32 SSM state is only supported on the Apple GPU"
                    " kernel"
                )

        @__parameter
        @always_inline
        def launch_gpu[DSTATE_VAL: Int]() raises:
            # NVIDIA B200 (sm_100) gets the cooperative DSTATE-split
            # decode-occupancy variant (r7); every other device (AMD MI355
            # gfx950, Hopper, Apple, ...) runs the portable v1
            # one-thread-per-channel kernel. The split uses a `lane_group_sum`
            # full-warp shuffle + 2D block that assume warp width 32, which is
            # invalid on AMD's wavefront-64 (it failed 2/9 MI355 tests, why
            # round-2 was reverted in 07c5e0b7533). The gate keeps AMD/non-B200
            # byte-identical to main. Predicate is `== B200` (not `version ==
            # "sm_100"`: B200's version string is "sm_100a"; the split was
            # measured and validated on B200 only, so B100/B300 stay on v1 too).
            # Gate on `ctx.default_device_info == B200` (the comptime device
            # `GPUInfo` for the accelerator arch): the wrapper `target` is a
            # `StaticString`, so `GPUInfo.from_target[target]()` is ill-typed
            # (`from_target` wants a `!kgen.target`) and hard-errors the
            # `builtin_kernels` build on every arch.
            comptime use_dstate_split = ctx.default_device_info == B200
            # Apple silicon GPU (Metal, cc==5) gets the vectorized-contiguous
            # dstate I/O variant: same one-thread-per-channel mapping/launch as
            # v1, but the scalar dstate load/store loops (mem-pipe-bound on M5)
            # become VEC-wide SIMD chunk loads/stores. Gate on the comptime
            # device API, which identifies the vendor (matching the `== B200`
            # gate rationale above).
            comptime use_apple_vec = ctx.default_device_info.api == "metal"
            # bf16 SSM state is only wired on the Apple vectorized kernel; the
            # B200 dstate-split and portable v1 kernels are fp32-state. The
            # Python side only allocates a bf16 pool on Apple (nemotron_h
            # `_ssm_state_dtype`), so this guard is defensive.
            comptime assert (
                state_dtype == .float32 or use_apple_vec
            ), "non-fp32 SSM state is only supported on the Apple GPU kernel"

            comptime if use_dstate_split:
                # Cooperative DSTATE-split: DSTATE_SPLIT threads cooperate on
                # each head_dim channel's DSTATE recurrence (lifts decode bs=1
                # occupancy; v1 one-thread-per-channel was ~4% achieved occupancy
                # on B200). The block holds CH_PER_BLOCK channels x DSTATE_SPLIT
                # threads = 128 threads (4 warps). DSTATE_SPLIT must divide both
                # DSTATE and 32 (warp) so each channel's lane group stays
                # warp-aligned for the lane_group_sum reduction; it divides every
                # dispatched DSTATE (16/64/128/256) cleanly.
                # Sweep (decode-shape microbench, B200, bf16, dstate=128):
                #   per-launch us @ bs=1: split1=46.8, split4=22.5, split8=16.6
                #   (-64.6% vs split1). split8 wins at bs=1/16/32.
                comptime DSTATE_SPLIT = 8
                comptime BLOCK_THREADS = 128
                comptime CH_PER_BLOCK = BLOCK_THREADS // DSTATE_SPLIT
                var num_p_blocks = ceildiv(head_dim, CH_PER_BLOCK)
                comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split[
                    dtype,
                    DSTATE_VAL,
                    x_tt.LayoutType,
                    dt_tt.LayoutType,
                    A_tt.LayoutType,
                    B_tt.LayoutType,
                    C_tt.LayoutType,
                    D_tt.LayoutType,
                    dt_bias_tt.LayoutType,
                    y_tt.LayoutType,
                    ssm_pool_tt.LayoutType,
                    query_start_loc_tt.LayoutType,
                    has_initial_state_tt.LayoutType,
                    cache_indices_tt.LayoutType,
                    x_tt.Engine,
                    DSTATE_SPLIT,
                ]
                var compiled = ctx.compile_function[kernel]()
                ctx.enqueue_function(
                    compiled,
                    Int32(nheads),
                    Int32(head_dim),
                    Int32(ngroups),
                    Int32(nheads_ngroups_ratio),
                    Int32(batch),
                    dt_softplus_int8,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    dt_bias_tt,
                    y_tt,
                    ssm_pool_tt,
                    query_start_loc_tt,
                    has_initial_state_tt,
                    cache_indices_tt,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    dt_bias_strides,
                    y_strides,
                    ssm_pool_strides,
                    grid_dim=(num_p_blocks, nheads, batch),
                    block_dim=(DSTATE_SPLIT, CH_PER_BLOCK, 1),
                )
            elif use_apple_vec:
                comptime BLOCK_SIZE = 64
                var num_p_blocks = ceildiv(head_dim, BLOCK_SIZE)
                comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_apple[
                    dtype,
                    DSTATE_VAL,
                    x_tt.LayoutType,
                    dt_tt.LayoutType,
                    A_tt.LayoutType,
                    B_tt.LayoutType,
                    C_tt.LayoutType,
                    D_tt.LayoutType,
                    dt_bias_tt.LayoutType,
                    y_tt.LayoutType,
                    ssm_pool_tt.LayoutType,
                    query_start_loc_tt.LayoutType,
                    has_initial_state_tt.LayoutType,
                    cache_indices_tt.LayoutType,
                    state_dtype,
                ]
                var compiled = ctx.compile_function[kernel]()
                ctx.enqueue_function(
                    compiled,
                    Int32(nheads),
                    Int32(head_dim),
                    Int32(ngroups),
                    Int32(nheads_ngroups_ratio),
                    Int32(batch),
                    dt_softplus_int8,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    dt_bias_tt,
                    y_tt,
                    ssm_pool_tt,
                    query_start_loc_tt,
                    has_initial_state_tt,
                    cache_indices_tt,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    dt_bias_strides,
                    y_strides,
                    ssm_pool_strides,
                    grid_dim=(num_p_blocks, nheads, batch),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )
            else:
                comptime BLOCK_SIZE = 64
                var num_p_blocks = ceildiv(head_dim, BLOCK_SIZE)
                comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu[
                    dtype,
                    DSTATE_VAL,
                    x_tt.LayoutType,
                    dt_tt.LayoutType,
                    A_tt.LayoutType,
                    B_tt.LayoutType,
                    C_tt.LayoutType,
                    D_tt.LayoutType,
                    dt_bias_tt.LayoutType,
                    y_tt.LayoutType,
                    ssm_pool_tt.LayoutType,
                    query_start_loc_tt.LayoutType,
                    has_initial_state_tt.LayoutType,
                    cache_indices_tt.LayoutType,
                ]
                var compiled = ctx.compile_function[kernel]()
                ctx.enqueue_function(
                    compiled,
                    Int32(nheads),
                    Int32(head_dim),
                    Int32(ngroups),
                    Int32(nheads_ngroups_ratio),
                    Int32(batch),
                    dt_softplus_int8,
                    x_tt,
                    dt_tt,
                    A_tt,
                    B_tt,
                    C_tt,
                    D_tt,
                    dt_bias_tt,
                    y_tt,
                    ssm_pool_tt,
                    query_start_loc_tt,
                    has_initial_state_tt,
                    cache_indices_tt,
                    x_strides,
                    dt_strides,
                    A_strides,
                    B_strides,
                    C_strides,
                    D_strides,
                    dt_bias_strides,
                    y_strides,
                    ssm_pool_strides,
                    grid_dim=(num_p_blocks, nheads, batch),
                    block_dim=(BLOCK_SIZE, 1, 1),
                )

        comptime if is_cpu[target]():
            if dstate == 256:
                launch_cpu[256]()
            elif dstate == 128:
                launch_cpu[128]()
            elif dstate == 64:
                launch_cpu[64]()
            else:
                launch_cpu[16]()
        elif is_gpu[target]():
            if dstate == 256:
                launch_gpu[256]()
            elif dstate == 128:
                launch_gpu[128]()
            elif dstate == 64:
                launch_gpu[64]()
            else:
                launch_gpu[16]()
        else:
            raise Error("Unsupported target device")


@extensibility.register_shape_function(
    "mamba2_ssd_chunk_scan_varlen_fwd_inplace"
)
def mamba2_ssd_chunk_scan_varlen_fwd_inplace_shape(
    x: Some[Tensor],
    dt: Some[Tensor],
    A: Some[Tensor],
    B: Some[Tensor],
    C: Some[Tensor],
    D: Some[Tensor],
    dt_bias: Some[Tensor],
    ssm_pool: Some[Tensor],
    query_start_loc: Some[Tensor],
    has_initial_state: Some[Tensor],
    cache_indices: Some[Tensor],
) -> IndexList[3]:
    """Computes the output shape for the `mamba2_ssd_chunk_scan_varlen_fwd_inplace` graph op.

    Args:
        x: Packed input tensor of shape
            `(total_len, nheads, head_dim)`.
        dt: Per-head time deltas of shape `(total_len, nheads)`.
        A: Per-head scalar decay of shape `(nheads,)`.
        B: Grouped input projection of shape
            `(total_len, ngroups, dstate)`.
        C: Grouped output projection of shape
            `(total_len, ngroups, dstate)`.
        D: Per-head skip connection of shape `(nheads,)`; may
            be empty when unused.
        dt_bias: Per-head bias added to `dt` of shape `(nheads,)`;
            may be empty when unused.
        ssm_pool: Mutable SSM state pool of shape
            `(max_slots, nheads, head_dim, dstate)` in `float32`;
            final states are written in place at the slots indexed
            by `cache_indices`.
        query_start_loc: Cumulative sequence lengths of shape
            `(batch + 1,)` in `int32`.
        has_initial_state: Per-sequence flag of shape `(batch,)`
            in `bool` indicating whether to load the initial state
            from `ssm_pool`; may be empty when no initial states are
            used.
        cache_indices: Per-sequence slot indices of shape
            `(batch,)` in `uint32` selecting where in `ssm_pool`
            the final states are written.
    """
    comptime assert type_of(x).rank == 3, "x must be rank 3"
    comptime assert type_of(dt).rank == 2, "dt must be rank 2"
    comptime assert type_of(A).rank == 1, "A must be rank 1"
    comptime assert type_of(B).rank == 3, "B must be rank 3"
    comptime assert type_of(C).rank == 3, "C must be rank 3"
    comptime assert type_of(D).rank == 1, "D must be rank 1"
    comptime assert type_of(dt_bias).rank == 1, "dt_bias must be rank 1"
    comptime assert type_of(ssm_pool).rank == 4, "ssm_pool must be rank 4"
    comptime assert (
        type_of(query_start_loc).rank == 1
    ), "query_start_loc must be rank 1"
    comptime assert (
        type_of(query_start_loc).dtype == .int32
    ), "query_start_loc dtype must be int32"
    comptime assert (
        type_of(has_initial_state).rank == 1
    ), "has_initial_state must be rank 1"
    comptime assert (
        type_of(has_initial_state).dtype == .bool
    ), "has_initial_state dtype must be bool"
    comptime assert (
        type_of(cache_indices).rank == 1
    ), "cache_indices must be rank 1"
    comptime assert (
        type_of(cache_indices).dtype == .uint32
    ), "cache_indices dtype must be uint32"
    comptime assert (
        type_of(dt).dtype == type_of(x).dtype
        and type_of(A).dtype == type_of(x).dtype
        and type_of(B).dtype == type_of(x).dtype
        and type_of(C).dtype == type_of(x).dtype
        and type_of(D).dtype == type_of(x).dtype
        and type_of(dt_bias).dtype == type_of(x).dtype
    ), "x, dt, A, B, C, D, and dt_bias must share a dtype"
    # y has the same shape as x: (total_len, nheads, head_dim).
    return rebind[IndexList[3]](coord_to_index_list(x.shape().tuple()))


@extensibility.register("causal_conv1d_varlen_fwd")
struct CausalConv1DVarlenFwd[
    activation: StaticString, channels_last: Bool = False
]:
    """Varlen causal 1D convolution forward pass.

    Performs causal 1D convolution on variable-length sequences that are
    concatenated together. Uses cumulative sequence lengths to identify
    sequence boundaries.

    The registration lives here in the built-in kernel library (mirroring the
    `gated_delta_conv1d_fwd` precedent) so the graph compiler / serve path can
    resolve the op with no out-of-tree `custom_extensions`. The kernel math
    lives in `state_space.varlen_causal_conv1d`.

    The underlying kernels index `x`/`output` purely through runtime
    dim/seqlen strides, so the token-axis memory layout is a free parameter.
    With `channels_last=True` the op consumes and produces tokens-major
    `(total_seqlen, dim)` tensors — the layout the surrounding graph
    naturally carries — eliminating the materialized `(dim, total_seqlen)`
    transposes on both sides of the op. Only the stride/extent bookkeeping
    below changes; the per-element compute is identical in both layouts.

    Parameters:
        activation: Activation function - "none" or "silu".
        channels_last: If True, `x` and `output` are tokens-major
            (total_seqlen, dim) instead of (dim, total_seqlen).

    Tensor Shapes:
        - output: (dim, total_seqlen) - Output tensor
          ((total_seqlen, dim) when `channels_last`)
        - x: (dim, total_seqlen) - Input tensor (concatenated sequences)
          ((total_seqlen, dim) when `channels_last`)
        - weight: (dim, width) - Convolution weights per channel
        - bias: (dim,) - Per-channel bias
        - query_start_loc: (batch + 1,) - Cumulative sequence lengths
        - cache_indices: (batch,) - Indices into conv_states (optional)
        - has_initial_state: (batch,) - Whether to use initial state (optional)
        - conv_states: (batch, dim, width - 1) - Conv states (optional, in/out)
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        x: InputTensor[dtype=dtype, rank=2, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        bias: InputTensor[dtype=dtype, rank=1, ...],
        # `conv_states` is a slot-indexed in/out pool of shape
        # [max_slots, dim, width - 1], read+written in place at slot
        # `cache_indices[b]`. It must be a `MutableInputTensor` (not an
        # `OutputTensor`) so the graph binds the caller's persistent pool
        # buffer rather than treating it as a freshly-produced output --
        # mirroring the `gated_delta_conv1d_fwd` precedent above.
        conv_states: MutableInputTensor[dtype=dtype, rank=3, ...],
        query_start_loc: InputTensor[dtype=.int32, rank=1, ...],
        cache_indices: InputTensor[dtype=.int32, rank=1, ...],
        has_initial_state: InputTensor[dtype=.bool, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # Axis of `x`/`output` holding channels vs. tokens (see
        # `channels_last`). The GPU/CPU kernels take dim/seqlen strides as
        # runtime arguments, so both layouts run the same code.
        comptime dim_axis = 1 if Self.channels_last else 0
        comptime seq_axis = 0 if Self.channels_last else 1
        var dim = x.dim_size(dim_axis)
        var total_seqlen = x.dim_size(seq_axis)
        var width = weight.dim_size(1)
        var batch = query_start_loc.dim_size(0) - 1

        var output_tt = output.to_tile_tensor[.int32]()
        var x_tt = x.to_tile_tensor[.int32]()
        var weight_tt = weight.to_tile_tensor[.int32]()
        var bias_tt = bias.to_tile_tensor[.int32]()
        var query_start_loc_tt = query_start_loc.to_tile_tensor[.int32]()
        var cache_indices_tt = cache_indices.to_tile_tensor[.int32]()
        var has_initial_state_tt = has_initial_state.to_tile_tensor[
            DType.int32
        ]()
        var conv_states_tt = conv_states.to_tile_tensor[.int32]()

        # Get strides as UInt32
        var x_strides = x.strides()
        var weight_strides = weight.strides()
        var output_strides = output.strides()
        var conv_states_strides = conv_states.strides()

        var x_dim_stride = UInt32(x_strides[dim_axis])
        var x_seqlen_stride = UInt32(x_strides[seq_axis])
        var weight_dim_stride = UInt32(weight_strides[0])
        var weight_width_stride = UInt32(weight_strides[1])
        var out_dim_stride = UInt32(output_strides[dim_axis])
        var out_seqlen_stride = UInt32(output_strides[seq_axis])

        var has_conv_states = conv_states.dim_size(0) > 0
        var conv_states_batch_stride = UInt32(
            conv_states_strides[0] if has_conv_states else 0
        )
        var conv_states_dim_stride = UInt32(
            conv_states_strides[1] if has_conv_states else 0
        )
        var conv_states_width_stride = UInt32(
            conv_states_strides[2] if has_conv_states else 0
        )

        var has_cache_indices = cache_indices.dim_size(0) > 0
        var has_initial_state_flag = has_initial_state.dim_size(0) > 0
        var has_bias = bias.dim_size(0) > 0

        var silu_activation = Self.activation == "silu"
        comptime PAD_SLOT_ID: Int32 = -1

        comptime if is_cpu[target]():
            causal_conv1d_varlen_fwd_cpu[
                x_tt.dtype,
                weight_tt.dtype,
                bias_tt.dtype,
                output_tt.dtype,
                query_start_loc_tt.dtype,
                cache_indices_tt.dtype,
                has_initial_state_tt.dtype,
                conv_states_tt.dtype,
            ](
                dim,
                total_seqlen,
                width,
                batch,
                x_tt,
                weight_tt,
                bias_tt,
                query_start_loc_tt,
                cache_indices_tt,
                has_initial_state_tt,
                conv_states_tt,
                output_tt,
                x_dim_stride,
                x_seqlen_stride,
                weight_dim_stride,
                weight_width_stride,
                out_dim_stride,
                out_seqlen_stride,
                conv_states_batch_stride,
                conv_states_dim_stride,
                conv_states_width_stride,
                silu_activation,
                PAD_SLOT_ID,
                has_cache_indices,
                has_initial_state_flag,
                has_conv_states,
                has_bias,
            )
        elif is_gpu[target]():
            var gpu_ctx = ctx
            comptime BLOCK_DIM = 128
            comptime BLOCK_SEQ = 1
            # Sequence-tile size for the seq-parallel prefill kernel (slice 1
            # of run7/designs/state-space-prefill-conv-seqparallel.md). Only
            # used on the `total_seqlen > batch` (prefill/mixed) branch below;
            # pure decode (`total_seqlen == batch`) keeps the untouched serial
            # kernel + BLOCK_SEQ path so decode stays byte-identical.
            comptime TILE_SEQ = 128
            var silu_activation_int8 = Int8(silu_activation)

            @__parameter
            @always_inline
            def launch_gpu[kWidth: Int]() raises:
                # Prefill/mixed segments (at least one sequence has >1
                # token) route to the grid-z sequence-tiled kernel; pure
                # decode (every sequence has exactly 1 token, so
                # total_seqlen == batch) keeps the serial per-thread kernel
                # unchanged below. This mirrors the shape-only heuristic
                # already used for the Mamba-2 SSD chunked-prefill gate.
                if total_seqlen > batch:
                    var compiled_func = gpu_ctx.compile_function[
                        causal_conv1d_varlen_fwd_seqparallel_gpu[
                            x_tt.dtype,
                            weight_tt.dtype,
                            bias_tt.dtype,
                            output_tt.dtype,
                            query_start_loc_tt.dtype,
                            cache_indices_tt.dtype,
                            has_initial_state_tt.dtype,
                            conv_states_tt.dtype,
                            kWidth,
                            BLOCK_DIM,
                            TILE_SEQ,
                            x_tt.LayoutType,
                            weight_tt.LayoutType,
                            bias_tt.LayoutType,
                            query_start_loc_tt.LayoutType,
                            cache_indices_tt.LayoutType,
                            has_initial_state_tt.LayoutType,
                            conv_states_tt.LayoutType,
                            output_tt.LayoutType,
                            x_tt.Engine,
                            weight_tt.Engine,
                            bias_tt.Engine,
                            query_start_loc_tt.Engine,
                            cache_indices_tt.Engine,
                            has_initial_state_tt.Engine,
                            conv_states_tt.Engine,
                            output_tt.Engine,
                        ]
                    ]()
                    # Host-side safe upper bound on the per-sequence tile
                    # count, avoiding a host max-reduction over ragged
                    # seqlens: grid-z indexes each sequence's LOCAL tile, and
                    # no sequence is longer than total_seqlen, so
                    # `ceildiv(total_seqlen, TILE_SEQ)` bounds every
                    # sequence's tile count (tile 0 stays alive for every
                    # sequence, including empty ones, since the dispatcher
                    # only reaches this kernel when total_seqlen > batch > 0).
                    # Blocks whose z-index exceeds a given sequence's actual
                    # tile count early-return inside the kernel.
                    gpu_ctx.enqueue_function(
                        compiled_func,
                        Int32(dim),
                        Int32(total_seqlen),
                        Int32(batch),
                        x_tt,
                        weight_tt,
                        bias_tt,
                        query_start_loc_tt,
                        cache_indices_tt,
                        has_initial_state_tt,
                        conv_states_tt,
                        output_tt,
                        UInt32(x_dim_stride),
                        UInt32(x_seqlen_stride),
                        UInt32(weight_dim_stride),
                        UInt32(weight_width_stride),
                        UInt32(out_dim_stride),
                        UInt32(out_seqlen_stride),
                        UInt32(conv_states_batch_stride),
                        UInt32(conv_states_dim_stride),
                        UInt32(conv_states_width_stride),
                        silu_activation_int8,
                        Int32(PAD_SLOT_ID),
                        Int8(has_cache_indices),
                        Int8(has_initial_state_flag),
                        Int8(has_conv_states),
                        Int8(has_bias),
                        grid_dim=(
                            batch,
                            ceildiv(dim, BLOCK_DIM),
                            ceildiv(total_seqlen, TILE_SEQ),
                        ),
                        block_dim=(BLOCK_DIM, 1),
                    )
                    return
                var compiled_func = gpu_ctx.compile_function[
                    causal_conv1d_varlen_fwd_gpu[
                        x_tt.dtype,
                        weight_tt.dtype,
                        bias_tt.dtype,
                        output_tt.dtype,
                        query_start_loc_tt.dtype,
                        cache_indices_tt.dtype,
                        has_initial_state_tt.dtype,
                        conv_states_tt.dtype,
                        kWidth,
                        BLOCK_DIM,
                        BLOCK_SEQ,
                        x_tt.LayoutType,
                        weight_tt.LayoutType,
                        bias_tt.LayoutType,
                        query_start_loc_tt.LayoutType,
                        cache_indices_tt.LayoutType,
                        has_initial_state_tt.LayoutType,
                        conv_states_tt.LayoutType,
                        output_tt.LayoutType,
                        x_tt.Engine,
                        weight_tt.Engine,
                        bias_tt.Engine,
                        query_start_loc_tt.Engine,
                        cache_indices_tt.Engine,
                        has_initial_state_tt.Engine,
                        conv_states_tt.Engine,
                        output_tt.Engine,
                    ]
                ]()
                gpu_ctx.enqueue_function(
                    compiled_func,
                    Int32(dim),
                    Int32(total_seqlen),
                    Int32(batch),
                    x_tt,
                    weight_tt,
                    bias_tt,
                    query_start_loc_tt,
                    cache_indices_tt,
                    has_initial_state_tt,
                    conv_states_tt,
                    output_tt,
                    UInt32(x_dim_stride),
                    UInt32(x_seqlen_stride),
                    UInt32(weight_dim_stride),
                    UInt32(weight_width_stride),
                    UInt32(out_dim_stride),
                    UInt32(out_seqlen_stride),
                    UInt32(conv_states_batch_stride),
                    UInt32(conv_states_dim_stride),
                    UInt32(conv_states_width_stride),
                    silu_activation_int8,
                    Int32(PAD_SLOT_ID),
                    Int8(has_cache_indices),
                    Int8(has_initial_state_flag),
                    Int8(has_conv_states),
                    Int8(has_bias),
                    grid_dim=(batch, ceildiv(dim, BLOCK_DIM)),
                    block_dim=(BLOCK_DIM, BLOCK_SEQ),
                )

            if width == 1:
                launch_gpu[1]()
            elif width == 2:
                launch_gpu[2]()
            elif width == 3:
                launch_gpu[3]()
            elif width == 4:
                launch_gpu[4]()
            else:
                raise Error(
                    "Unsupported kernel width: only widths 1, 2, 3, 4 are"
                    " supported"
                )
        else:
            raise Error("Unsupported target device")


@extensibility.register_shape_function("causal_conv1d_varlen_fwd")
def causal_conv1d_varlen_fwd_shape(
    x: Some[Tensor],
    weight: Some[Tensor],
    bias: Some[Tensor],
    # Must mirror the execute function's input-tensor list (incl. the in/out
    # `conv_states` pool) or the MOGG kernel-library validator rejects the op
    # ("Execute and shape functions do not have the same input tensors").
    # Bound as a role-less tensor trait here (the shape fn does not mutate),
    # matching the SSD-inplace `ssm_pool` shape-fn convention.
    conv_states: Some[Tensor],
    query_start_loc: Some[Tensor],
    cache_indices: Some[Tensor],
    has_initial_state: Some[Tensor],
) -> IndexList[2]:
    """Computes the output shape for the `causal_conv1d_varlen_fwd` graph op.

    Args:
        x: Input tensor of concatenated sequence elements with shape
            (dim, total_seqlen).
        weight: Convolution weight tensor with shape (dim, width).
        bias: Per-channel bias tensor with shape (dim,).
        conv_states: Slot-indexed in/out pool of conv states with shape
            (batch, dim, width - 1).
        query_start_loc: Cumulative sequence lengths with shape
            (batch + 1,).
        cache_indices: Indices into the conv_states pool with shape (batch,).
        has_initial_state: Whether each sequence has an initial state with
            shape (batch,).
    """
    comptime assert type_of(x).rank == 2, "x must be rank 2"
    comptime assert type_of(weight).rank == 2, "weight must be rank 2"
    comptime assert type_of(bias).rank == 1, "bias must be rank 1"
    comptime assert type_of(conv_states).rank == 3, "conv_states must be rank 3"
    comptime assert (
        type_of(weight).dtype == type_of(x).dtype
        and type_of(bias).dtype == type_of(x).dtype
        and type_of(conv_states).dtype == type_of(x).dtype
    ), "x, weight, bias, and conv_states must share a dtype"
    comptime assert (
        type_of(query_start_loc).rank == 1
    ), "query_start_loc must be rank 1"
    comptime assert (
        type_of(query_start_loc).dtype == .int32
    ), "query_start_loc dtype must be int32"
    comptime assert (
        type_of(cache_indices).rank == 1
    ), "cache_indices must be rank 1"
    comptime assert (
        type_of(cache_indices).dtype == .int32
    ), "cache_indices dtype must be int32"
    comptime assert (
        type_of(has_initial_state).rank == 1
    ), "has_initial_state must be rank 1"
    comptime assert (
        type_of(has_initial_state).dtype == .bool
    ), "has_initial_state dtype must be bool"
    return rebind[IndexList[2]](coord_to_index_list(x.shape().tuple()))


# ===-----------------------------------------------------------------------===#
# Gated group-RMSNorm (Mamba-2 mixer, `norm_before_gate=False`)
# ===-----------------------------------------------------------------------===#


@extensibility.register("gated_group_rmsnorm")
struct GatedGroupRMSNorm[group_size: Int]:
    """Fused silu-gate + group RMSNorm + weight-scale for the Mamba-2 mixer.

    Collapses `cast -> silu(gate) * y -> group rms_norm -> * norm_weight -> cast`
    into one dispatch, matching HF `Zamba2RMSNormGated` with
    `norm_before_gate=False`. The registration lives here in the built-in kernel
    library (mirroring the `causal_conv1d_varlen_fwd` /
    `mamba2_ssd_chunk_scan_varlen_fwd_inplace` precedents) so the graph compiler
    / serve path resolves the op with no out-of-tree `custom_extensions`. The
    kernel math lives in `state_space.gated_group_rmsnorm`.

    Parameters:
        group_size: Width of each independently normalized group along the
            intermediate axis (`intermediate // n_groups`).

    Tensor shapes:
        - output: `(n_rows, intermediate)` - model dtype.
        - y: `(n_rows, intermediate)` - SSD scan output, model dtype.
        - gate: `(n_rows, intermediate)` - gate projection (any float dtype;
          may be a strided split view of the fused in-proj).
        - weight: `(intermediate,)` - fp32 RMSNorm weight.
        - eps: Scalar epsilon (fp32) inside `rsqrt(mean_sq + eps)`.
    """

    @staticmethod
    def execute[
        dtype: DType,
        gate_dtype: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        y: InputTensor[dtype=dtype, rank=2, ...],
        gate: InputTensor[dtype=gate_dtype, rank=2, ...],
        weight: InputTensor[dtype=.float32, rank=1, ...],
        eps: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        var n_rows = y.dim_size(0)
        var intermediate = y.dim_size(1)
        comptime gs = Self.group_size
        # The kernel floor-divides `intermediate // gs`; a non-multiple would
        # silently drop the tail columns of the last (partial) group. The
        # unfused `ops.reshape(yf, [-1, group_size])` this replaced errored on
        # a non-multiple, so guard the invariant here (production 7680/960=8 is
        # exact -- this is a guard, not a behavior change).
        debug_assert(
            intermediate % gs == 0,
            (
                "gated_group_rmsnorm: intermediate must be a multiple of"
                " group_size"
            ),
        )
        var num_groups = intermediate // gs

        var output_tt = output.to_tile_tensor[.int32]()
        var y_tt = y.to_tile_tensor[.int32]()
        var gate_tt = gate.to_tile_tensor[.int32]()
        var weight_tt = weight.to_tile_tensor[.int32]()

        comptime if is_cpu[target]():
            gated_group_rmsnorm_cpu[dtype, gate_dtype](
                output_tt,
                y_tt,
                gate_tt,
                weight_tt,
                n_rows,
                num_groups,
                gs,
                eps,
            )
        elif is_gpu[target]():
            gated_group_rmsnorm_gpu[dtype, gate_dtype](
                output_tt,
                y_tt,
                gate_tt,
                weight_tt,
                n_rows,
                num_groups,
                gs,
                eps,
                ctx,
            )
        else:
            raise Error("gated_group_rmsnorm: unsupported target device")


@extensibility.register_shape_function("gated_group_rmsnorm")
def gated_group_rmsnorm_shape(
    y: Some[Tensor],
    gate: Some[Tensor],
    weight: Some[Tensor],
    eps: Float32,
) -> IndexList[2]:
    comptime assert type_of(y).rank == 2, "y must be rank 2"
    comptime assert type_of(gate).rank == 2, "gate must be rank 2"
    comptime assert type_of(weight).rank == 1, "weight must be rank 1"
    comptime assert (
        type_of(weight).dtype == .float32
    ), "weight dtype must be float32"
    return rebind[IndexList[2]](coord_to_index_list(y.shape().tuple()))


# ===-----------------------------------------------------------------------===#
# Sleep kernel
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.sleep")
struct Sleep:
    """Registers the `mo.sleep` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        # In order to prevent this kernel from being DCE'd, we pass in a mutable
        # input buffer. A fix is tracked in GEX-3080.
        duration_sec_buffer: MutableInputTensor[
            dtype=DType.float64, rank=1, ...
        ],
        ctx: DeviceContext,
    ) raises:
        var duration_sec = duration_sec_buffer[0]
        if duration_sec < 0:
            raise Error(
                "Sleep duration must be non-negative. Found: ", duration_sec
            )

        if is_gpu[target]():

            @__name("sleep")
            def sleep_kernel(duration_sec: Float64):
                sleep(duration_sec)

            var device_ctx = ctx
            device_ctx.enqueue_function[sleep_kernel](
                duration_sec, grid_dim=(1,), block_dim=(1,)
            )
        else:
            sleep(duration_sec)


# ===-----------------------------------------------------------------------===#
# In-place memcpy kernel
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.inplace_memcpy")
struct InplaceMemcpy[DstDevice: StaticString, SrcDevice: StaticString]:
    """Copies the contents of `src` into `dst` in place.

    Semantically equivalent to `Buffer.inplace_copy_from`, but exposed
    as a graph op so the copy can be scheduled as part of a compiled MAX
    graph. Both operands must have the same dtype, rank, and total
    element count.

    Supports the four direction combinations expressible with a single
    `DeviceContext`: GPU-to-GPU on the same device, GPU-to-CPU,
    CPU-to-GPU, and CPU-to-CPU. Cross-GPU memcpy (different GPU ids) is
    rejected by the Python wrapper at graph build time.

    Parameters:
        DstDevice: Device type of the destination buffer, for example, `\"gpu\"` or
            `\"cpu\"`.
        SrcDevice: Device type of the source buffer, for example, `\"gpu\"` or
            `\"cpu\"`.
    """

    @staticmethod
    def execute[
        target: StaticString,
        dtype: DType,
        rank: Int,
    ](
        dst: MutableInputTensor[dtype=dtype, rank=rank, ...],
        src: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        var count = dst.size()
        comptime if is_gpu[Self.DstDevice]() and is_gpu[Self.SrcDevice]():
            # Same-GPU async memcpy.
            ctx.enqueue_copy[dtype](dst.unsafe_ptr(), src.unsafe_ptr(), count)
        elif is_gpu[Self.DstDevice]() and is_cpu[Self.SrcDevice]():
            # Host-to-device async memcpy. Wrap the GPU dst pointer as a
            # non-owning `DeviceBuffer` so the typed overload is selected.
            ctx.enqueue_copy[dtype](
                dst.to_device_buffer(ctx),
                src.unsafe_ptr(),
            )
        elif is_cpu[Self.DstDevice]() and is_gpu[Self.SrcDevice]():
            # Device-to-host async memcpy.
            ctx.enqueue_copy[dtype](
                dst.unsafe_ptr(),
                src.to_device_buffer(ctx),
            )
        elif is_cpu[Self.DstDevice]() and is_cpu[Self.SrcDevice]():
            # Host-to-host. Plain synchronous memcpy.
            unsafe_memcpy(
                dest=dst.unsafe_ptr(),
                src=src.unsafe_ptr(),
                count=count,
            )
        else:
            # Cross-device memcpy are unsupported since stream is ambiguous.
            raise Error("InplaceMemcpy does not support cross-gpu memcpy")


# ===-----------------------------------------------------------------------===#
# Host function launch kernel
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.launch_host_func")
struct LaunchHostFunc:
    """Enqueues a pre-packed host callback on the device's default stream.

    Corresponds to CUDA's `cuLaunchHostFunc`. Accepts a 1-D int64 buffer of
    shape `[2]` whose elements are raw pointer-sized integers:

    - `payload[0]`: address of a `void (*)(void *)` trampoline function.
    - `payload[1]`: address of an opaque user-data block owned by the
      trampoline (freed after the callback runs).

    Both values are produced by `max._core.driver._pack_host_func(fn)` on
    the Python side. Currently only CUDA streams support host callbacks;
    non-CUDA backends raise at runtime.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        # A mutable input buffer prevents the op from being DCE'd (see
        # `mo.sleep` above; tracked in GEX-3080).
        payload: MutableInputTensor[dtype=.int64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime _HostFuncTy = def(OpaquePointer[MutAnyOrigin]) thin -> None
        var tr_addr = Int(payload[0])
        var ud_addr = Int(payload[1])
        var tr_ptr = OpaquePointer[MutAnyOrigin](unsafe_from_address=tr_addr)
        var ud_ptr = OpaquePointer[MutAnyOrigin](unsafe_from_address=ud_addr)
        # Reinterpret the raw trampoline address as a thin function value.
        var tr_fn = Pointer(to=tr_ptr).unsafe_bitcast[_HostFuncTy]()[]
        ctx.stream().enqueue_host_func(tr_fn, ud_ptr)


@extensibility.register("mo.wait_host_value")
struct WaitHostValue:
    """Stalls the stream until a host-visible flag reaches a given value.

    Lowers to CUDA's `cuStreamWaitValue64` via
    `DeviceStream.wait_for_host_value`. Accepts a 1-D int64 buffer of
    shape `[2]`, mirroring `mo.launch_host_func`'s payload shape:

    - `payload[0]`: raw address of a `M::Driver::CompletionFlag` (as
      u64). Typically obtained from
      `max.driver.CompletionFlag._unsafe_ptr` and packed into the
      buffer by the Python caller; the C++ object must outlive any
      graph execution that references it.
    - `payload[1]`: the 64-bit value to wait for (the int64 element is
      reinterpreted as a u64).

    Records into a captured device graph as a wait-value / batch-mem-op
    node, so this op can sit inside a captured forward graph just before
    the sampling kernel to gate the sampler on the bitmask compute
    finishing while the rest of the forward body runs concurrently. On
    HIP that node is recorded explicitly by the driver layer, because
    ROCm's stream capture does not record `hipStreamWaitValue64` itself.
    Supported on CUDA and HIP streams; other backends raise at runtime.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        # MutableInputTensor mirrors `mo.launch_host_func` so this op is
        # not DCE'd. Both the CompletionFlag pointer and the expected
        # value encode into 64-bit elements.
        payload: MutableInputTensor[dtype=.int64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var flag = CompletionFlag(unsafe_from_address=Int(payload[0]))
        var value = UInt64(Int(payload[1]))
        ctx.stream().wait_for_host_value(flag, value)


@extensibility.register("mo.wait_host_value_with_dep")
struct WaitHostValueWithDep:
    """Variant of `mo.wait_host_value` that takes a fake mutable
    dependency operand.

    Behaves identically to `mo.wait_host_value` at runtime -- the `dep`
    tensor is never read or written by the kernel body -- but the
    graph compiler sees `dep` as mutated by this op, which forces any
    downstream op that consumes `dep` (e.g. `mo.inplace_memcpy(scratch,
    dep)`) to chain after this wait.

    Use this when you need the wait to gate a `cuStreamWaitValue64`
    followed by an `inplace_memcpy` of the buffer the host callback
    fills: without a shared mutable operand the two custom ops carry no
    data dependency and the graph compiler / cuGraph capture is free to
    parallelise them, so the memcpy can read stale pinned data before
    the worker signals the flag.
    """

    @staticmethod
    def execute[
        target: StaticString,
        dep_dtype: DType,
        dep_rank: Int,
    ](
        payload: MutableInputTensor[dtype=.int64, rank=1, ...],
        # `dep` is intentionally unused: it exists only to register a
        # mutation on the buffer so downstream consumers of the same
        # buffer chain after this op.
        dep: MutableInputTensor[dtype=dep_dtype, rank=dep_rank, ...],
        ctx: DeviceContext,
    ) raises:
        var flag = CompletionFlag(unsafe_from_address=Int(payload[0]))
        var value = UInt64(Int(payload[1]))
        ctx.stream().wait_for_host_value(flag, value)


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#
