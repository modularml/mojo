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

"""Implements transposed convolution kernels for CPU and GPU backends.

Includes direct register-tiled computation, filter packing, and a
cuDNN-backed path for NVIDIA GPUs.
"""

from std.math import align_down, ceildiv

from std.memory import ThinAllocation, alloc, dealloc
from std.memory.alloc import Layout as AllocLayout
from std.sys import align_of, simd_width_of

from _cudnn.cnn_infer import (
    cudnnConvolutionBackwardData,
    cudnnConvolutionMode_t,
    cudnnGetConvolutionBackwardDataWorkspaceSize,
    cudnnSetConvolution2dDescriptor,
)
from _cudnn.infer import (
    cudnnConvolutionBwdDataAlgo_t,
    cudnnDataType_t,
    cudnnSetFilter4dDescriptor,
    cudnnSetTensor4dDescriptor,
    cudnnTensorFormat_t,
)
from std.algorithm import (
    tile,
    tile_middle_unswitch_boundaries,
    vectorize,
)

from max.algorithm import (
    elementwise,
    sync_parallelize,
)
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from layout.coord import DynamicCoord
from layout.tensor_engine import TensorEngine
from linalg.accumulate import _Accumulator
from linalg.utils import partition_work
from nn.conv.conv import _get_cudnn_meta, check_cudnn_error
from max.runtime.asyncrt import parallelism_level
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import Index, IndexList

from .conv_utils import (
    ConvInfoStatic,
    ConvPartition,
    ConvShape,
    align_down_residual,
    elementwise_epilogue_type,
    elementwise_simd_epilogue_type,
    get_conv_num_tasks,
    get_conv_shape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
)

# TODO: All attributes, except for groups and auto_pad, are supported.
#       - Kernel assumes groups = 1.
#       - For auto_pad, need to set `AutoPadMode.NOTSET` (default).
#       Only remaining issue is handling of optional attributes & setting defaults,
#       and the associated logic (e.g., if one attribute is specified another is
#       ignored).
#       Specifically, in the ONNX spec for ConvTranspose
#       (https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose)
#       - dilations (optional): if not provided, create (1,1) as default.
#       - kernel_shape (optional): if not provided, obtain from argument-provided kernel.
#       - strides (optional): if not provided, create (1,1) as default.
#       - output_shape (optional): if specified, pads values are ignored.
#       modular/Kernels/test/test_convtranspose.mojo provides examples of calls.
#       StarGAN, CycleGAN-and-pix2pix, Mask-RCNN are covered by this version.


@always_inline
def conv_transpose_naive[
    dtype: DType,
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[mut=False, dtype, ...],
    filter: TileTensor[mut=False, dtype, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
):
    """
    Implements the ConvTranspose operator from the MO spec.

    Parameters:
        dtype: Type of the input, output, and kernel tensors.

    Args:
        output: Output data tensor that contains the result of the convolution.
        input: Input data tensor from previous layer, with size of (N x H x W x C),
               where N is the batch size, C is the number of channels, and H and
               W are the height and width.
        filter: The weight (kernel) tensor, with size of (kH x kW x M/groups x C),
                where C is the number of channels, kH and kW are the height and
                width of the kernel, and M is the number of feature maps.
        stride: Stride along each spatial axis.
        dilation: Dilation value along each spatial axis of the filter.
        pad_d: Padding in depth dimension.
        pad_h: Padding in height dimension.
        pad_w: Padding in width dimension.
    """
    comptime assert output.flat_rank == 5
    comptime assert input.flat_rank == 5
    comptime assert filter.flat_rank == 5
    comptime assert output.element_size == 1
    comptime assert input.element_size == 1
    comptime assert filter.element_size == 1

    var N = Int(input.dim[0]())  # Number of images (num. batches)

    var D = Int(input.dim[1]())
    var H = Int(input.dim[2]())
    var W = Int(input.dim[3]())

    var C = Int(input.dim[4]())  # Number of input channels

    var Q = Int(filter.dim[0]())
    var R = Int(filter.dim[1]())
    var S = Int(filter.dim[2]())

    var DO = Int(output.dim[1]())
    var HO = Int(output.dim[2]())
    var WO = Int(output.dim[3]())

    var F = Int(output.dim[4]())

    # Initialize output to zero
    _ = output.fill(0)

    for n in range(N):
        for c in range(C):
            for f in range(F):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            var do = d * stride[0] - pad_d[0]
                            var ho = h * stride[1] - pad_h[0]
                            var wo = w * stride[2] - pad_w[0]

                            for q in range(Q):
                                var do_nbr = do + q * dilation[0]
                                if do_nbr < 0 or do_nbr >= DO:
                                    continue

                                for r in range(R):
                                    var ho_nbr = ho + r * dilation[1]
                                    if ho_nbr < 0 or ho_nbr >= HO:
                                        continue

                                    for s in range(S):
                                        var wo_nbr = wo + s * dilation[2]
                                        if wo_nbr < 0 or wo_nbr >= WO:
                                            continue

                                        var output_val = output[
                                            n, do_nbr, ho_nbr, wo_nbr, f
                                        ]
                                        var input_val = input[n, d, h, w, c][0]
                                        var filter_val = filter[q, r, s, f, c][
                                            0
                                        ]

                                        output[n, do_nbr, ho_nbr, wo_nbr, f] = (
                                            output_val + input_val * filter_val
                                        )


@always_inline
def conv_transpose_shape[
    dtype: DType,
    strides_type: DType,
    dilations_type: DType,
    pads_type: DType,
    output_pads_type: DType,
](
    input: TileTensor[mut=False, dtype, ...],
    kernel: TileTensor[mut=False, dtype, ...],
    strides: TileTensor[mut=False, strides_type, ...],
    dilations: TileTensor[mut=False, dilations_type, ...],
    pads: TileTensor[mut=False, pads_type, ...],
    output_pads: TileTensor[mut=False, output_pads_type, ...],
) raises -> IndexList[input.rank]:
    """
    Compute the output shape of a `conv-transpose` operation, and assert the
    inputs are compatible.

    Parameters:
        dtype: Element type of the input and kernel tensor.
        strides_type: Element type of the strides tensor.
        dilations_type: Element type of the dilations tensor.
        pads_type: Element type of the pads tensor.
        output_pads_type: Element type of the output_pads tensor.

    Args:
        input: The input tensor.
        kernel: The kernel tensor.
        strides: The strides tensor.
        dilations: The dilations tensor.
        pads: The paddings tensor.
        output_pads: The output paddings tensor.

    Returns:
        The output shape.
    """
    comptime assert strides.flat_rank == 1
    comptime assert pads.flat_rank == 1
    comptime assert dilations.flat_rank == 1
    comptime assert output_pads.flat_rank == 1

    if input.rank != 4 and input.rank != 5:
        raise Error("[conv_transpose] requires (input_rank == 4 or 5)")
    if input.rank != kernel.rank:
        raise Error("[conv_transpose] requires (input_rank == kernel_rank)")
    if strides.dim(0) != Scalar[strides.linear_idx_type](
        input.rank - 2
    ) or dilations.dim(0) != Scalar[dilations.linear_idx_type](input.rank - 2):
        raise Error(
            "[conv_transpose] requires (len(strides) == len(dilations) =="
            " input_rank - 2)"
        )
    if pads.dim(0) != Scalar[pads.linear_idx_type](2 * (input.rank - 2)):
        raise Error(
            "[conv_transpose] requires (len(paddings) == 2 * (input rank - 2))"
        )

    # Assume input has channel last layout, NHWC or NDHWC.
    var batch_size = Int(input.dim(0))
    var output_shape = IndexList[input.rank]()
    # Assume kernel has layout RSFC or QRSFC. The output channel is F because
    # this is a convolution transpose shape function (inverse of regular
    # convolution).
    var output_channels = Int(kernel.dim(input.rank - 2))

    # Check input and kernel channels
    if Int(input.dim(input.rank - 1)) != Int(kernel.dim(input.rank - 1)):
        raise Error(
            "[conv_transpose] requires input channel to match output channel in"
            " kernel"
        )

    # Compute output shape
    output_shape[0] = batch_size
    output_shape[input.rank - 1] = output_channels

    @__parameter
    @always_inline
    def compute_output_spatial_dim(
        input_spatial_dim: Int,
        kernel_spatial_dim: Int,
        stride: Int,
        dilation: Int,
        pre_pad: Int,
        post_pad: Int,
        out_pad: Int,
    ) -> Int:
        return (
            stride * (input_spatial_dim - 1)
            + out_pad
            + ((kernel_spatial_dim - 1) * dilation + 1)
            - pre_pad
            - post_pad
        )

    # Compute the spatial dims
    comptime for i in range(input.rank - 2):
        var input_spatial_dim = Int(input.dim(i + 1))
        var kernel_spatial_dim = Int(kernel.dim(i))
        var stride = Int(strides[i])
        var dilation = Int(dilations[i])
        var pre_pad = Int(pads[2 * i])
        var post_pad = Int(pads[2 * i + 1])
        var out_pad = Int(output_pads[i])
        var output_spatial_dim = compute_output_spatial_dim(
            input_spatial_dim,
            kernel_spatial_dim,
            stride,
            dilation,
            pre_pad,
            post_pad,
            out_pad,
        )
        if output_spatial_dim <= 0:
            raise Error("[conv_transpose] output spatial dim must be positive")
        output_shape[i + 1] = output_spatial_dim

    return output_shape


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution Helpers                                        #
# ===----------------------------------------------------------------------=== #


def get_num_partitions[
    micro_kernel_height: Int, micro_kernel_f_size: Int
](num_threads: Int, conv_shape: ConvShape) -> IndexList[4]:
    """Partition the workload in (batch&group, C, F, H) dimensions.
    HOWO is the combination of HO and WO dimensions.
    The actual number of tasks are the product of return num_partitions.

    Parameters:
        micro_kernel_height: Height of the micro kernel in the output spatial
            dimension.
        micro_kernel_f_size: Size of the micro kernel along the output channel
            dimension.

    Args:
        num_threads: Number of parallel threads available for partitioning.
        conv_shape: Shape descriptor of the transposed convolution.
    """

    var max_num_tasks = get_conv_num_tasks(num_threads, conv_shape)

    var num_batches_and_groups = conv_shape.n * conv_shape.num_groups

    var max_f_tasks = ceildiv(conv_shape.f, micro_kernel_f_size)

    var num_partitions = IndexList[4](1)

    num_partitions[0] = min(max_num_tasks, num_batches_and_groups)
    num_partitions[2] = min(max_f_tasks, max_num_tasks // num_partitions[0])

    return num_partitions


def get_partition(
    task_id: Int,
    num_partitions: IndexList[4],
    conv_shape: ConvShape,
    micro_kernel_height: Int,
    micro_kernel_f_size: Int,
) -> ConvPartition:
    """Computes the partition of the transposed convolution workload assigned to a single task.

    Args:
        task_id: Identifier of the task to compute the partition for.
        num_partitions: Number of partitions along each dimension.
        conv_shape: Shape descriptor of the transposed convolution.
        micro_kernel_height: Height of the micro kernel in the output spatial dimension.
        micro_kernel_f_size: Size of the micro kernel along the output channel dimension.

    Returns:
        The partition describing the offset and size of the work assigned to the task.
    """
    var quotient, task_id_f = divmod(task_id, num_partitions[2])
    var task_id_ng, task_id_howo = divmod(quotient, num_partitions[3])

    var ng_range = partition_work(
        task_id_ng, num_partitions[0], conv_shape.n * conv_shape.num_groups, 1
    )

    var c_range = Index(0, conv_shape.c)

    var f_range = partition_work(
        task_id_f,
        num_partitions[2],
        conv_shape.f // conv_shape.num_groups,
        micro_kernel_f_size,
    )

    var howo_range = partition_work(
        task_id_howo, num_partitions[3], conv_shape.h(), 1
    )

    return ConvPartition(
        ng_offset=ng_range[0],
        ng_size=ng_range[1],
        f_offset=f_range[0],
        f_size=f_range[1],
        ho_or_howo_offset=howo_range[0],
        ho_or_howo_size=howo_range[1],
        c_offset=c_range[0],
        c_size=c_range[1],
    )


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution Entry Point                                    #
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ConvTransposedPacked[
    input_linear_idx_type: DType,
    input_engine: TensorEngine,
    filter_linear_idx_type: DType,
    filter_engine: TensorEngine,
    output_linear_idx_type: DType,
    output_engine: TensorEngine,
    InputLayoutType: TensorLayout,
    FilterLayoutType: TensorLayout,
    OutputLayoutType: TensorLayout,
    conv_attr_rank: Int,
    //,
    input_origin: ImmOrigin,
    filter_origin: ImmOrigin,
    output_origin: MutOrigin,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    conv_attr: ConvInfoStatic[conv_attr_rank],
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](ImplicitlyCopyable):
    """Holds the packed tensors and partition state for one task of a direct transposed convolution.

    Encapsulates the output, input, and packed filter tile tensors along with the
    convolution shape and the partition assigned to a single parallel task, and
    drives the tiled batch, channel, and feature loops that accumulate the result.

    Parameters:
        input_linear_idx_type: Linear index dtype of the input tensor (inferred).
        input_engine: Engine backing the input tile tensor (inferred).
        filter_linear_idx_type: Linear index dtype of the filter tensor (inferred).
        filter_engine: Engine backing the filter tile tensor (inferred).
        output_linear_idx_type: Linear index dtype of the output tensor (inferred).
        output_engine: Engine backing the output tile tensor (inferred).
        InputLayoutType: Compile-time layout of the input tensor (inferred).
        FilterLayoutType: Compile-time layout of the filter tensor (inferred).
        OutputLayoutType: Compile-time layout of the output tensor (inferred).
        conv_attr_rank: Number of spatial dimensions in the convolution (inferred).
        input_origin: Immutable memory origin of the input tile tensor.
        filter_origin: Immutable memory origin of the filter tile tensor.
        output_origin: Mutable memory origin of the output tile tensor.
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.
        conv_attr: Static convolution attributes carrying strides, pads,
            dilations, and group count.
        elementwise_epilogue: Optional elementwise epilogue applied to the output
            after accumulation (defaults to `None`).
    """

    var output: TileTensor[
        Self.output_type,
        Self.OutputLayoutType,
        Self.output_origin,
        Engine=Self.output_engine,
        linear_idx_type=Self.output_linear_idx_type,
    ]
    var input: TileTensor[
        Self.input_type,
        Self.InputLayoutType,
        Self.input_origin,
        Engine=Self.input_engine,
        linear_idx_type=Self.input_linear_idx_type,
    ]
    var filter: TileTensor[
        Self.filter_type,
        Self.FilterLayoutType,
        Self.filter_origin,
        Engine=Self.filter_engine,
        linear_idx_type=Self.filter_linear_idx_type,
    ]

    var conv_shape: ConvShape[Self.conv_attr_rank]

    # Support partition in 4 dims: (n, c, f, ho_or_howo). If the input is
    # padded, the output spatial dims are merged into one as howo. If not
    # padded, only ho is partitioned for now.
    var partition: ConvPartition

    var cf_tile_size: DynamicCoord[.int64, 2]

    @staticmethod
    def run(
        output: TileTensor[
            mut=True,
            Self.output_type,
            Self.OutputLayoutType,
            Self.output_origin,
            Engine=Self.output_engine,
            linear_idx_type=Self.output_linear_idx_type,
            address_space=.GENERIC,
            ...,
        ],
        input: TileTensor[
            Self.input_type,
            Self.InputLayoutType,
            Self.input_origin,
            Engine=Self.input_engine,
            linear_idx_type=Self.input_linear_idx_type,
            address_space=.GENERIC,
            ...,
        ],
        filter: TileTensor[
            Self.filter_type,
            Self.FilterLayoutType,
            Self.filter_origin,
            Engine=Self.filter_engine,
            linear_idx_type=Self.filter_linear_idx_type,
            address_space=.GENERIC,
            ...,
        ],
        conv_shape: ConvShape[Self.conv_attr_rank],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime assert input.rank - 2 == Self.conv_attr_rank
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_shape = get_micro_kernel_shape[
            Self.conv_attr_rank,
            Int(Self.OutputLayoutType.static_shape[output.rank - 2]),  # WO
            Int(Self.OutputLayoutType.static_shape[output.rank - 1]),  # F
            Self.conv_attr,
            simd_size,
        ]()
        comptime micro_kernel_height = micro_kernel_shape[0]
        comptime micro_kernel_width = micro_kernel_shape[1]
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        var cf_tile_size = get_conv_tile_shape[Self.filter_type](
            conv_shape.c,
            conv_shape.filter_window_flat_size(),
            micro_kernel_width,
        )

        comptime if Self.conv_attr.num_groups != UNKNOWN_VALUE:
            comptime assert (
                Self.conv_attr.num_groups == 1
            ), "Don't support grouped transposed conv for now."

        # Number of partitions in n, ho_wo, c, f dimensions.
        var num_threads = parallelism_level(ctx)
        var num_partitions = get_num_partitions[
            micro_kernel_height, micro_kernel_f_size
        ](num_threads, conv_shape)
        var num_tasks = num_partitions.flattened_length()

        def task_func(task_id: Int) {var num_partitions, var cf_tile_size, imm}:
            var partition = get_partition(
                task_id,
                num_partitions,
                conv_shape,
                micro_kernel_height,
                micro_kernel_f_size,
            )

            if partition.empty():
                return

            var task_tile_size = Index(
                min(cf_tile_size[0], partition.c_size), cf_tile_size[1]
            )

            var instance = ConvTransposedPacked[
                Self.input_origin,
                Self.filter_origin,
                Self.output_origin,
                Self.input_type,
                Self.filter_type,
                Self.output_type,
                Self.conv_attr,
                elementwise_epilogue=Self.elementwise_epilogue,
            ](
                output,
                input,
                filter,
                conv_shape,
                partition,
                Coord(task_tile_size),
            )
            instance._batch_group_loop()

        sync_parallelize(task_func, num_tasks, ctx)

    @always_inline
    def _zero_output(self, n: Int, g: Int):
        """Zero the output buffer."""
        comptime simd_size = simd_width_of[Self.output_type]()

        var f_offset = (
            g * self.conv_shape.f_per_group() + self.partition.f_offset
        )
        var num_rows = self.conv_shape.output_image_flat_size()
        var output_ptr = (
            self.output.ptr + n * num_rows * self.conv_shape.f + f_offset
        )

        for _ in range(num_rows):

            @always_inline
            def zero[width: Int](offset: Int) {output_ptr, mut}:
                output_ptr.store(offset, SIMD[Self.output_type, width](0))

            vectorize[simd_size](self.partition.f_size, zero)

            output_ptr += self.conv_shape.f

    def _batch_group_loop(self):
        """Loop over the batch and group dimensions. Only support groups = 1 for now.
        """

        for ng in range(
            self.partition.ng_offset,
            self.partition.ng_offset + self.partition.ng_size,
        ):
            var n, g = divmod(ng, self.conv_shape.num_groups)

            # Initialize the output buffer for current batch and group.
            self._zero_output(n, g)

            # ConvTransposed computation
            self._c_tile_loop(n, g, Int(self.cf_tile_size[0].value()))

            # Epilogue. Avoid putting it after register tiling for now because
            # input row i may update output row i-1. It's hard to tell if a row
            # has been updated lastly especially with stride and dilation.
            self.apply_epilogue(n, g)

    def _c_tile_loop(self, n: Int, g: Int, c_tile_size: Int):
        """Loop over C tiles."""

        var c_offset = g * self.conv_shape.c_per_group()
        var c_round_by_tile = align_down(
            (self.conv_shape.c_per_group() - 1), c_tile_size
        )

        # Update c tiles before the last one.
        for c_tile_offset in range(c_offset, c_round_by_tile, c_tile_size):
            self._f_tile_loop[False](n, g, c_tile_offset, c_tile_size)

        # Update the last c tile with fusion
        var c_round_by_tile_residual = (
            self.conv_shape.c_per_group() - c_round_by_tile
        )
        self._f_tile_loop[True](
            n,
            g,
            c_offset + c_round_by_tile,
            c_round_by_tile_residual,
        )

    def _f_tile_loop[
        last_c_tile: Bool
    ](self, n: Int, g: Int, c_tile_offset: Int, c_tile_size: Int):
        """Loop over F tiles."""
        comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
        comptime micro_kernel_height = get_direct_conv_micro_kernel_height()
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        @always_inline
        def f_tile_iteration[
            size: Int
        ](f_tile_offset: Int, f_tile_size: Int) {imm}:
            self.input_space_loop[
                micro_kernel_height, size // simd_size, False, last_c_tile
            ](n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size)

        var f_per_group = self.conv_shape.f_per_group()

        # The partition heuristic sees F_per_group and may partition it.
        # The partition's F_offset should be added to the group's F offset to
        # get the actually offset in output's F dim.
        var group_f_offset = g * f_per_group + self.partition.f_offset

        var group_f_end_align_simd = group_f_offset + align_down(
            self.partition.f_size, simd_size
        )

        # The first tile size is based on cache size. Within the tile
        # it's stepped by the micro kernel size in F. The rest is stepped
        # by simd_size. If F is not multiple of simd_size, the residual
        # is padded with 0 to fit a simd vector in the packed filter.
        tile[
            [micro_kernel_f_size, simd_size],
            simd_size,
        ](
            group_f_offset,
            group_f_end_align_simd,
            micro_kernel_f_size,
            simd_size,
            primary_cleanup_tile=simd_size,
            workgroup_function=f_tile_iteration,
        )

        # If this is the last partition in F and it's not a multiple of simd_size.
        # The partition is aligned by micro_kernel_f_size, so only the last
        # partition is possible to have residual.
        var residual = align_down_residual(f_per_group, simd_size)
        if (
            self.partition.f_offset + self.partition.f_size == f_per_group
            and residual > 0
        ):
            self.input_space_loop[micro_kernel_height, 1, True, last_c_tile](
                n,
                group_f_end_align_simd,
                simd_size,
                c_tile_offset,
                c_tile_size,
            )

    @always_inline
    def input_space_loop[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        n: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        # Current group index.
        var g = self.conv_shape.f_to_group(f_tile_offset)

        # Filter pointer to the current cf tile offset location.
        var filter_ptr: UnsafePointer[
            Scalar[Self.filter_type], ImmUntrackedOrigin
        ]

        # Move the pointer to the current group's start.
        filter_ptr = _get_group_filter_base(
            self.filter.as_immut(), g, self.conv_shape.f_per_group()
        ).unsafe_origin_cast[ImmUntrackedOrigin]()
        # Move the pointer to (c_tile_offset, f_tile_offset) mapped in
        # current group.
        filter_ptr = filter_ptr + (
            # Jump over f_tile_offset in current group.
            self.conv_shape.f_in_group(f_tile_offset)
            * self.conv_shape.c_per_group()
            * self.conv_shape.filter_window_flat_size()
            # Jump over c_tile_offset in current group.
            + self.conv_shape.c_in_group(c_tile_offset) * micro_kernel_f_size
        )

        # Pointer to input and output of the current sample (batch dim).
        # fmt: off
        var input_ptr  = self.input.ptr + c_tile_offset \
                       + self.conv_shape.input_image_flat_size() \
                       * self.conv_shape.c * n

        var output_ptr = self.output.ptr + f_tile_offset \
                       + self.conv_shape.output_image_flat_size() \
                       * self.conv_shape.f * n

        # fmt: on

        # Divide each input row into three parts:
        # [0, left_pad_impact_end)
        # [left_pad_impact_end, right_pad_impact_start)
        # [right_pad_impact_start, WO)
        comptime w_axis = Self.InputLayoutType.rank - 3
        var left_pad_impact_end = ceildiv(
            self.conv_shape.pad_w_lower(),
            self.conv_shape.stride_at[w_axis](),
        )
        var right_pad_impact_start = (
            self.conv_shape.wo()
            + self.conv_shape.pad_w_lower()
            - self.conv_shape.s() * self.conv_shape.dilation_at[w_axis]()
        ) // self.conv_shape.stride_at[w_axis]() + 1
        # print("pad effect", left_pad_impact_end, right_pad_impact_start)

        comptime if Self.InputLayoutType.rank == 4:
            self.input_space_loop_2d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                output_ptr.as_unsafe_any_origin(),  # FIXME: Why is this needed?
                input_ptr,
                filter_ptr,
                n,
                # self.is_new_c_accum(c_tile_offset),
                False,
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )
        elif Self.InputLayoutType.rank == 5:
            self.input_space_loop_3d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                output_ptr.as_unsafe_any_origin(),  # FIXME: Why is this needed?
                input_ptr,
                filter_ptr,
                n,
                # self.is_new_c_accum(c_tile_offset),
                False,
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )

    @always_inline
    def input_space_loop_2d[
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
        //,
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        output: UnsafePointer[mut=True, Scalar[output_dt], _],
        input: UnsafePointer[Scalar[input_dt], _],
        filter: UnsafePointer[Scalar[filter_dt], _],
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()

        for h in range(
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        ):
            # Compute the row index as if there is no padding. This index may be
            # < 0 while the actual output ho index is within [0, ho). In the
            # inner loops, `ho_nbr = ho + r * dilation` where r within [0, R)
            # can tell if a row is in padding i.e. ho_nbr < 0  or ho_nbr > wo-1.
            var ho = (
                h * self.conv_shape.stride_at[0]()
                - self.conv_shape.pad_h_lower()
            )

            var input_base = input + self.conv_shape.c * self.conv_shape.w() * h

            # Points output to the start of the row
            var output_base = output + self.conv_shape.f * (
                -self.conv_shape.pad_w_lower() + self.conv_shape.wo() * ho
            )

            # TODO(MOCO-4664): `var h` copy-captures the loop variable to work
            # around wrong debug-info scopes on implicit nested-scope captures.
            @always_inline
            def work_fn[
                height: Int, effected_by_padding: Bool
            ](w: Int) {var h, mut input_base, mut output_base, imm}:
                update_w_tile_2d[
                    height,
                    micro_kernel_width,
                    simd_size,
                    effected_by_padding,
                    has_residual,
                    last_c_tile,
                ](
                    output_base,
                    input_base,
                    filter,
                    first_c_tile_in_group,
                    c_tile_size,
                    f_tile_offset,
                    f_tile_size,
                    rebind[ConvShape[2]](self.conv_shape),
                    n,
                    Index(h, w),
                )

                input_base = input_base + height * self.conv_shape.c
                output_base = output_base + (
                    height * self.conv_shape.stride_at[1]() * self.conv_shape.f
                )

            tile_middle_unswitch_boundaries[
                [micro_kernel_height, 5, 4, 3, 2, 1]
            ](
                0,
                left_pad_impact_end,
                right_pad_impact_start,
                self.conv_shape.w(),
                work_fn,
            )
            # TODO(MOCO-2074): Suppress false positive unused var warning.
            _ = input_base
            _ = output_base

    @always_inline
    def input_space_loop_3d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[mut=True, Scalar[output_dt], _],
        input: UnsafePointer[Scalar[input_dt], _],
        filter: UnsafePointer[Scalar[filter_dt], _],
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()

        for d in range(self.conv_shape.d()):
            var do = (
                d * self.conv_shape.stride_at[0]()
                - self.conv_shape.pad_d_lower()
            )

            for h in range(
                self.partition.ho_or_howo_offset,
                self.partition.ho_or_howo_offset
                + self.partition.ho_or_howo_size,
            ):
                # fmt: off
                var ho = h * self.conv_shape.stride_at[1]() - self.conv_shape.pad_h_lower()
                # fmt: on

                var input_base = (
                    input
                    + self.conv_shape.c
                    * self.conv_shape.w()
                    * (h + d * self.conv_shape.h())
                )

                var output_base = output + self.conv_shape.f * (
                    -self.conv_shape.pad_w_lower()
                    + self.conv_shape.wo() * (ho + self.conv_shape.ho() * do)
                )

                # TODO(MOCO-4664): `var d`/`var h` copy-capture the loop
                # variables to work around wrong debug-info scopes on
                # implicit nested-scope captures.
                @always_inline
                def work_fn[
                    height: Int, effected_by_padding: Bool
                ](w: Int) {var d, var h, mut input_base, mut output_base, imm}:
                    update_w_tile_3d[
                        height,
                        micro_kernel_width,
                        simd_size,
                        effected_by_padding,
                        has_residual,
                        last_c_tile,
                    ](
                        output_base,
                        input_base,
                        filter,
                        first_c_tile_in_group,
                        c_tile_size,
                        f_tile_offset,
                        f_tile_size,
                        rebind[ConvShape[3]](self.conv_shape),
                        n,
                        Index(d, h, w),
                    )

                    input_base = input_base + height * self.conv_shape.c
                    output_base = output_base + (
                        height
                        * self.conv_shape.stride_at[2]()
                        * self.conv_shape.f
                    )

                tile_middle_unswitch_boundaries[
                    [micro_kernel_height, 5, 4, 3, 2, 1],
                ](
                    0,
                    left_pad_impact_end,
                    right_pad_impact_start,
                    self.conv_shape.w(),
                    work_fn,
                )
                # TODO(MOCO-2074): Suppress false positive unused var warning.
                _ = input_base
                _ = output_base

    @always_inline
    def apply_epilogue(self, n: Int, g: Int):
        comptime simd_size = simd_width_of[Self.output_type]()

        var f_offset = (
            g * self.conv_shape.f_per_group() + self.partition.f_offset
        )
        var num_rows = self.conv_shape.output_image_flat_size()
        var output_base = (
            self.output.ptr + n * num_rows * self.conv_shape.f + f_offset
        )

        comptime if Self.elementwise_epilogue:
            comptime epilogue = Self.elementwise_epilogue.value()

            var output_ptr = output_base

            comptime if Self.InputLayoutType.rank == 4:  # 2D ConvTransposed.
                for ho in range(self.conv_shape.ho()):
                    for wo in range(self.conv_shape.wo()):
                        epilogue(
                            Index(n, ho, wo, f_offset), self.partition.f_size
                        )
                        output_ptr += self.conv_shape.f

            elif Self.InputLayoutType.rank == 5:  # 3D ConvTransposed.
                for do in range(self.conv_shape.do()):
                    for ho in range(self.conv_shape.ho()):
                        for wo in range(self.conv_shape.wo()):
                            epilogue(
                                Index(n, do, ho, wo, f_offset),
                                self.partition.f_size,
                            )
                            output_ptr += self.conv_shape.f


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution 2D Resigter Tiling
# ===----------------------------------------------------------------------=== #


@always_inline
def update_w_tile_2d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
](
    output: UnsafePointer[mut=True, Scalar[output_dt], _],
    input: UnsafePointer[mut=False, Scalar[input_dt], _],
    filter: UnsafePointer[mut=False, Scalar[filter_dt], _],
    _init_output: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[2],
    n: Int,
    hw: IndexList[2],
):
    """Updates one output tile of a 2D transposed convolution using register tiling.

    Iterates over the filter window for a single output point and accumulates the
    input and packed filter contributions into the output tile, skipping neighbors
    that fall inside the padding region.

    Parameters:
        micro_kernel_height: Height of the micro kernel in the output spatial dimension.
        micro_kernel_width: Width of the micro kernel along the output channel dimension.
        simd_size: SIMD vector width for the output element type.
        effected_by_padding: Whether the current tile is affected by spatial padding.
        has_residual: Whether the tile contains a residual not aligned to the micro kernel size.
        last_c_tile: Whether this is the last channel tile for epilogue fusion.
        output_dt: Element type of the output tensor.
        input_dt: Element type of the input tensor.
        filter_dt: Element type of the filter tensor.

    Args:
        output: Pointer to the output tile to accumulate into.
        input: Pointer to the input tile to read from.
        filter: Pointer to the packed filter tile to read from.
        _init_output: Unused flag retained for API compatibility.
        c_tile_size: Size of the channel tile.
        f_tile_offset: Offset of the feature tile within the output channel dimension.
        f_tile_size: Size of the feature tile.
        conv_shape: Shape descriptor of the 2D transposed convolution.
        n: Batch index of the current sample.
        hw: Spatial coordinates of the current output point.
    """
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Output stride to neighbor point in the filter window (R, S).
    # fmt: off
    var output_stride_by_s = conv_shape.dilation_at[1]() * conv_shape.f
    var output_stride_by_r = conv_shape.dilation_at[0]() * conv_shape.wo() * conv_shape.f
    # fmt: on

    # Filter stride when s increments by 1.
    var filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size

    var filter_stride_by_r = conv_shape.s() * filter_stride_by_s

    # Filter stride in F dimension in FRSCf or RSFC
    var filter_stride = micro_kernel_f_size

    # Output coordinates
    var howo = Index(
        hw[0] * conv_shape.stride_at[0]() - conv_shape.pad_h_lower(),
        hw[1] * conv_shape.stride_at[1]() - conv_shape.pad_w_lower(),
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.

    for r in range(conv_shape.r()):
        # Skip the row if it falls into padding.
        var ho_nbr = howo[0] + r * conv_shape.dilation_at[0]()
        if ho_nbr < 0 or ho_nbr >= conv_shape.ho():
            continue

        for s in range(conv_shape.s()):
            var output_ptr = (
                output + r * output_stride_by_r + s * output_stride_by_s
            )
            var filter_ptr = (
                filter + r * filter_stride_by_r + s * filter_stride_by_s
            )

            comptime if effected_by_padding:
                comptime assert (
                    micro_kernel_height == 1
                ), "The tile must only have 1 point when effected bypadding."
                var wo_nbr = howo[1] + s * conv_shape.dilation_at[1]()
                if wo_nbr < 0 or wo_nbr >= conv_shape.wo():
                    continue

            accumulate_wo_tile[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                has_residual,
            ](
                c_tile_size,
                output_ptr,
                conv_shape.f * conv_shape.stride_at[1](),
                input,
                conv_shape.c,
                filter_ptr,
                filter_stride,
                conv_shape.f_per_group() % simd_size,
            )


@always_inline
def update_w_tile_3d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
](
    output: UnsafePointer[mut=True, Scalar[output_dt], _],
    input: UnsafePointer[Scalar[input_dt], _],
    filter: UnsafePointer[Scalar[filter_dt], _],
    _init_output: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[3],
    n: Int,
    hw: IndexList[3],
):
    """Updates one output tile of a 3D transposed convolution using register tiling.

    Iterates over the depth, height, and width of the filter window for a single
    output point and accumulates the input and packed filter contributions into
    the output tile, skipping neighbors that fall inside the padding region.

    Parameters:
        micro_kernel_height: Height of the micro kernel in the output spatial dimension.
        micro_kernel_width: Width of the micro kernel along the output channel dimension.
        simd_size: SIMD vector width for the output element type.
        effected_by_padding: Whether the current tile is affected by spatial padding.
        has_residual: Whether the tile contains a residual not aligned to the micro kernel size.
        last_c_tile: Whether this is the last channel tile for epilogue fusion.
        output_dt: Element type of the output tensor.
        input_dt: Element type of the input tensor.
        filter_dt: Element type of the filter tensor.

    Args:
        output: Pointer to the output tile to accumulate into.
        input: Pointer to the input tile to read from.
        filter: Pointer to the packed filter tile to read from.
        _init_output: Unused flag retained for API compatibility.
        c_tile_size: Size of the channel tile.
        f_tile_offset: Offset of the feature tile within the output channel dimension.
        f_tile_size: Size of the feature tile.
        conv_shape: Shape descriptor of the 3D transposed convolution.
        n: Batch index of the current sample.
        hw: Spatial coordinates of the current output point.
    """
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Output stride to neighbor point in the filter window (R, S).
    # fmt: off
    var output_stride_by_s = conv_shape.dilation_at[2]() * conv_shape.f
    var output_stride_by_r = conv_shape.dilation_at[1]() * conv_shape.wo() * conv_shape.f
    var output_stride_by_q = conv_shape.dilation_at[0]() * conv_shape.wo() * conv_shape.ho() * conv_shape.f
    # fmt: on

    # Filter stride when s increments by 1.
    var filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    var filter_stride_by_r = conv_shape.s() * filter_stride_by_s
    var filter_stride_by_q = conv_shape.r() * filter_stride_by_r

    # Filter stride in F dimension in FRSCf or RSFC
    var filter_stride = micro_kernel_f_size

    # Output coordinates
    var howo = Index(
        hw[0] * conv_shape.stride_at[0]() - conv_shape.pad_d_lower(),
        hw[1] * conv_shape.stride_at[1]() - conv_shape.pad_h_lower(),
        hw[2] * conv_shape.stride_at[2]() - conv_shape.pad_w_lower(),
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.

    for q in range(conv_shape.q()):
        var do_nbr = howo[0] + q * conv_shape.dilation_at[0]()
        if do_nbr < 0 or do_nbr >= conv_shape.do():
            continue

        for r in range(conv_shape.r()):
            var ho_nbr = howo[1] + r * conv_shape.dilation_at[1]()
            if ho_nbr < 0 or ho_nbr >= conv_shape.ho():
                continue

            for s in range(conv_shape.s()):
                var output_ptr = output + (
                    q * output_stride_by_q
                    + r * output_stride_by_r
                    + s * output_stride_by_s
                )
                var filter_ptr = filter + (
                    q * filter_stride_by_q
                    + r * filter_stride_by_r
                    + s * filter_stride_by_s
                )

                comptime if effected_by_padding:
                    comptime assert micro_kernel_height == 1
                    var wo_nbr = howo[2] + s * conv_shape.dilation_at[2]()
                    if wo_nbr < 0 or wo_nbr >= conv_shape.wo():
                        continue

                accumulate_wo_tile[
                    micro_kernel_height,
                    micro_kernel_width,
                    simd_size,
                    has_residual,
                ](
                    c_tile_size,
                    output_ptr,
                    conv_shape.f * conv_shape.stride_at[2](),
                    input,
                    conv_shape.c,
                    filter_ptr,
                    filter_stride,
                    conv_shape.f_per_group() % simd_size,
                )


@always_inline
def accumulate_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    partial_load: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
](
    c_tile_size: Int,
    output: UnsafePointer[mut=True, Scalar[output_dt], _],
    output_stride: Int,
    input: UnsafePointer[mut=False, Scalar[input_dt], _],
    input_stride: Int,
    filter: UnsafePointer[mut=False, Scalar[filter_dt], _],
    filter_stride: Int,
    partial_load_size: Int,
):
    """Accumulates one width tile of the transposed convolution into the output buffer.

    Loads the current output tile into an accumulator, multiplies and accumulates
    the input and packed filter contributions across the channel tile, then stores
    the result back to the output buffer.

    Parameters:
        micro_kernel_height: Height of the micro kernel in the output spatial dimension.
        micro_kernel_width: Width of the micro kernel along the output channel dimension.
        simd_size: SIMD vector width for the output element type.
        partial_load: Whether the tile contains a residual not aligned to the SIMD width.
        output_dt: Element type of the output tensor.
        input_dt: Element type of the input tensor.
        filter_dt: Element type of the filter tensor.

    Args:
        c_tile_size: Size of the channel tile to accumulate.
        output: Pointer to the output tile to accumulate into.
        output_stride: Stride between consecutive output rows.
        input: Pointer to the input tile to read from.
        input_stride: Stride between consecutive input rows.
        filter: Pointer to the packed filter tile to read from.
        filter_stride: Stride of the filter in the feature dimension.
        partial_load_size: Number of valid elements when the tile is a partial load.
    """
    var acc = _Accumulator[
        output_dt, micro_kernel_height, micro_kernel_width, simd_size
    ]()

    acc.load[partial_load=partial_load](
        output,
        output_stride,
        partial_load_size,
    )

    acc.accumulate[prefetch_offset=4, partial_load_b=partial_load](
        c_tile_size,
        input,
        input_stride,
        filter,
        filter_stride,
        partial_load_size,
    )

    acc.store[partial_store=partial_load](
        output,
        output_stride,
        partial_load_size,
    )


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution Filter Packing                                 #
# ===----------------------------------------------------------------------=== #


@always_inline
def _get_group_filter_base(
    packed_filter: TileTensor, group_idx: Int, f_per_group: Int
) -> UnsafePointer[
    Scalar[packed_filter.dtype],
    packed_filter.origin,
    address_space=packed_filter.address_space,
]:
    # TODO: support groups > 1.
    return packed_filter.ptr


@always_inline
def pack_filter_shape(
    filter: TileTensor[mut=False, ...], num_groups: Int
) -> IndexList[filter.rank + 1]:
    """
    Compute the output shape of transposed convolution filter packing.

    Args:
        filter: The filter to be packed.
        num_groups: The number of groups in the convolution.

    Returns:
        The output shape.
    """

    comptime simd_size = simd_width_of[filter.dtype]()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Filter is in RSFC layout. The 2nd last dim is F.
    var F = Int(filter.dim[filter.rank - 2]())

    assert (
        F % num_groups == 0
    ), "number of filters F must be divisible by number of groups"
    var F_per_group = F // num_groups

    # FRSCf layout.
    var packed_shape = IndexList[filter.rank + 1]()
    packed_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    packed_shape[filter.rank] = micro_kernel_f_size
    # Input channel
    packed_shape[filter.rank - 1] = Int(filter.dim[filter.rank - 1]())

    comptime for i in range(filter.rank - 2):
        packed_shape[i + 1] = Int(filter.dim[i]())

    return packed_shape


@always_inline
def pack_filter(
    filter: TileTensor[mut=False, ...],
    packed_filter: TileTensor[mut=True, ...],
    num_groups: Int,
):
    """Packs the filter from RSFC layout into the FRSCf layout expected by the direct transposed convolution kernel.

    Each group is zero-padded so the feature dimension is a multiple of the micro kernel feature size, and any residual not aligned to the SIMD width is handled separately.

    Args:
        filter: Unpacked filter tensor in RSFC layout to read from.
        packed_filter: Output tensor in packed FRSCf layout that receives the packed filter.
        num_groups: Number of convolution groups; the filter feature dimension must be divisible by this value.
    """

    comptime simd_size = simd_width_of[filter.dtype]()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Product of filter window dims.
    var window_dims_prod = 1

    comptime for i in range(filter.rank - 2):
        window_dims_prod *= Int(filter.dim[i]())

    var C = Int(filter.dim[filter.rank - 1]())
    var F = Int(filter.dim[filter.rank - 2]())
    var F_per_group = F // num_groups

    _ = packed_filter.fill(0)

    # Each group is zero padded to
    #
    #                   ceildiv(F_per_group, micro_kernel_f_size)
    #                 * outer_dims_prod
    #                 * micro_kernel_f_size.
    #
    # There can be a remainder: F_per_group % micro_kernel_f_size. That's further
    # tiled by simd_size. The elements beyond the remainder is set to 0. E.x.
    # micro_kernel_f_size = 8, simd_size = 2, 21 values in total, follows
    #
    #                       |--------|--------|--|--|-0|00|

    for g in range(num_groups):
        var group_start = _get_group_filter_base(packed_filter, g, F_per_group)

        # TODO(MOCO-4664): `var g` copy-captures the loop variable to work
        # around wrong debug-info scopes on implicit nested-scope captures.
        @always_inline
        def pack[
            f_tile_size: Int
        ](f_tile_start: Int) {
            var g, var group_start, var C, var F_per_group, var F, imm
        }:
            var packed_filter_ptr = (
                group_start + f_tile_start * window_dims_prod * C
            )

            # Consider a point in filter window as a neighbor to input point.
            for nbr in range(window_dims_prod):
                var filter_ptr = (
                    filter.ptr
                    + nbr * F * C
                    + (g * F_per_group + f_tile_start) * C
                )

                for _ in range(C):
                    for f in range(f_tile_size):
                        packed_filter_ptr.store(
                            f,
                            filter_ptr.load(f * C).cast[
                                packed_filter_ptr.T.dtype
                            ](),
                        )

                    # FRSCf layout. Increment by `f` in c loop.
                    packed_filter_ptr += f_tile_size
                    # RSFC layout. Increment by 1 in c loop.
                    filter_ptr += 1

        # If F % simd_size != 0, the following won't touch the remainder.
        tile[[micro_kernel_f_size, simd_size]](0, F_per_group, pack)

    # Check the remainder if any
    var F_round_by_simd = align_down(F_per_group, simd_size)
    var residual = F_per_group - F_round_by_simd

    # Handle the remainder if any
    if residual > 0:
        for g in range(num_groups):
            var group_start = _get_group_filter_base(
                packed_filter, g, F_per_group
            )
            var packed_filter_ptr = (
                group_start + F_round_by_simd * window_dims_prod * C
            )

            for nbr in range(window_dims_prod):
                var filter_ptr = (
                    filter.ptr
                    + nbr * F * C
                    + (g * F_per_group + F_round_by_simd) * C
                )

                for _ in range(C):
                    for f in range(residual):
                        packed_filter_ptr.store(
                            f,
                            filter_ptr.load(f * C).cast[
                                packed_filter_ptr.T.dtype
                            ](),
                        )

                    filter_ptr += 1
                    packed_filter_ptr = packed_filter_ptr + simd_size


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution API to MOGG                                    #
# ===----------------------------------------------------------------------=== #


def conv_transposed_cpu[
    filter_packed: Bool,
    filter_is_cfrs: Bool,
    has_epilogue_fusion: Bool,
    elementwise_lambda: elementwise_simd_epilogue_type,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, address_space=.GENERIC, ...],
    filter: TileTensor[mut=True, address_space=.GENERIC, ...],
    stride: IndexList[input.rank - 2],
    dilation: IndexList[input.rank - 2],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Runs a transposed convolution on the CPU using the direct register-tiled kernel.

    Packs the filter when needed, builds the convolution shape, and dispatches the
    tiled computation across the available CPU parallelism, optionally fusing an
    elementwise epilogue.

    Parameters:
        filter_packed: Whether the filter is already in packed FRSCf layout.
        filter_is_cfrs: Whether the filter is in CFRS layout, which is unsupported.
        has_epilogue_fusion: Whether to fuse an elementwise epilogue into the kernel.
        elementwise_lambda: Elementwise epilogue closure applied to each output row segment.

    Args:
        output: Output tensor that receives the transposed convolution result.
        input: Input tensor in NHWC or NDHWC layout.
        filter: Filter tensor in RSFC layout, packed or unpacked.
        stride: Stride along each spatial axis.
        dilation: Dilation value along each spatial axis of the filter.
        pad_d: Padding in the depth dimension.
        pad_h: Padding in the height dimension.
        pad_w: Padding in the width dimension.
        ctx: Optional device context used to query the parallelism level.
    """

    @always_inline
    def description_fn() {imm} -> String:
        # fmt: off
        return String(
            trace_arg("input", coord_to_index_list(input.layout.shape_coord())),
            ";", trace_arg("filter", coord_to_index_list(filter.layout.shape_coord())),
            ";", trace_arg("output", coord_to_index_list(output.layout.shape_coord())),
            ";group=1",
            ";stride=", stride,
            ";padding_d=", Index(0, 0),
            ";padding_h=", pad_h,
            ";padding_w=", pad_w,
        )
        # fmt: on

    comptime assert not filter_is_cfrs, "Filter layout CFRS is not supported"

    with Trace[TraceLevel.OP, target=StaticString("cpu")](
        "conv_transposed",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
    ):
        comptime packed_filter_rank = filter.rank if filter_packed else filter.rank + 1

        var packed_filter_ptr = filter.ptr.unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var packed_filter_shape: IndexList[packed_filter_rank]

        # If filter is not packed, we have to pack it before the kernel.
        comptime if not filter_packed:
            # Only support single group.
            packed_filter_shape = rebind[IndexList[packed_filter_rank]](
                pack_filter_shape(filter, 1)
            )
        else:
            packed_filter_shape = IndexList[packed_filter_rank]()

            comptime for i in range(packed_filter_rank):
                packed_filter_shape[i] = Int(filter.layout.shape[i]().value())

        var packed_filter_alloc_layout = AllocLayout[Scalar[filter.dtype]](
            count=packed_filter_shape.flattened_length()
        )
        comptime if not filter_packed:
            packed_filter_ptr = alloc(packed_filter_alloc_layout).unsafe_leak()

        var packed_filter = TileTensor(
            packed_filter_ptr,
            row_major(Coord(packed_filter_shape)),
        )

        comptime if not filter_packed:
            pack_filter(filter, packed_filter, 1)

        comptime conv_attr = ConvInfoStatic[input.rank - 2]()

        var conv_shape = get_conv_shape[input.rank - 2, True](
            output,
            input,
            packed_filter,
            stride,
            dilation,
            pad_d,
            pad_h,
            pad_w,
            1,
        )

        # The closure updates a row segment of the output.
        @always_inline
        @__parameter
        def elementwise_epilogue[
            rank: Int
        ](coords: IndexList[rank], f_size: Int):
            comptime simd_size = simd_width_of[output.dtype]()
            comptime input_rank = input.rank

            @always_inline
            def body[width: Int](idx: Int) {coords, output, mut}:
                # Coordinates of the current index.
                var curr_coords = rebind[IndexList[input_rank]](coords)
                curr_coords[input_rank - 1] += idx

                var output_idx = output.layout(Coord(curr_coords))

                var vec = output.raw_load[width=width](output_idx)
                elementwise_lambda(curr_coords, vec)

            vectorize[simd_size](f_size, body)

        ConvTransposedPacked[
            input.origin,
            packed_filter.origin,
            output.origin,
            input.dtype,
            filter.dtype,
            output.dtype,
            conv_attr,
            Optional[elementwise_epilogue_type](
                elementwise_epilogue
            ) if has_epilogue_fusion else None,
        ].run(output, input, packed_filter, conv_shape, ctx)

        comptime if not filter_packed:
            dealloc(
                ThinAllocation(
                    unsafe_owned_ptr=packed_filter_ptr
                ).unsafe_with_layout(packed_filter_alloc_layout)
            )


# ===----------------------------------------------------------------------=== #
# cuDNN Convolution Backward Data (i.e., Transposed Convolution) Helper        #
# ===----------------------------------------------------------------------=== #


def conv_transposed_gpu[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    elementwise_epilogue: Optional[elementwise_simd_epilogue_type] = None,
](
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, input_type, address_space=.GENERIC, ...],
    filter: TileTensor[mut=False, filter_type, address_space=.GENERIC, ...],
    stride: IndexList[input.rank - 2],
    dilation: IndexList[input.rank - 2],
    padding: IndexList[input.rank - 2],
    ctx: DeviceContext,
) raises:
    """Runs a transposed convolution on the GPU via the cuDNN backward-data path.

    Delegates the transposed convolution to cuDNN and, when an elementwise epilogue
    is provided, applies it as a separate elementwise pass over a temporary buffer.

    Parameters:
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.
        elementwise_epilogue: Optional elementwise epilogue closure applied after the convolution.

    Args:
        output: Output tensor that receives the transposed convolution result.
        input: Input tensor in NHWC layout.
        filter: Filter tensor in RSFC layout.
        stride: Stride along each spatial axis.
        dilation: Dilation value along each spatial axis of the filter.
        padding: Padding along each spatial axis.
        ctx: Device context used to access the cuDNN handle and device buffers.
    """
    comptime if elementwise_epilogue:
        comptime epilogue = elementwise_epilogue.value()

        var output_tmp_data = ctx.enqueue_create_buffer[output_type](
            output.num_elements()
        )

        var output_tmp = TileTensor(output_tmp_data, output.layout)

        conv_transposed_cudnn[input_type, filter_type, output_type](
            input,
            filter,
            output_tmp,
            rebind[IndexList[2]](stride),
            rebind[IndexList[2]](dilation),
            rebind[IndexList[2]](padding),
            ctx,
        )

        @always_inline
        def epilogue_wrapper[
            _width: Int, alignment: Int = 1
        ](coords: Coord) {var}:
            comptime align = align_of[SIMD[output_type, _width]]()
            var idx = output_tmp.layout(coords)
            var vec = output_tmp.raw_load[width=_width, alignment=align](idx)
            epilogue(coord_to_index_list(coords), vec)

        elementwise[simd_width_of[output_type](), target="gpu"](
            epilogue_wrapper, output.layout.shape_coord(), ctx
        )

        _ = output_tmp_data^

    else:
        conv_transposed_cudnn[input_type, filter_type, output_type](
            input,
            filter,
            output,
            rebind[IndexList[2]](stride),
            rebind[IndexList[2]](dilation),
            rebind[IndexList[2]](padding),
            ctx,
        )


def _conv_transposed_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[mut=False, input_type, address_space=.GENERIC, ...],
    filter: TileTensor[mut=False, filter_type, address_space=.GENERIC, ...],
    output: TileTensor[output_type, address_space=.GENERIC, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
    ctx: DeviceContext,
) raises:
    var cudnn_handle = _get_cudnn_meta(ctx)

    # basically, vibes are that a cuda handle is the gateway to using cudnn
    # we want all the work from that handle to be done on a separate stream
    # than the main stream, otherwise, everything goes on main stream and
    # slows down the whole thing. binding handle to stream unclocks parallelism, and now
    # 2 handles , with 2 separate functions, can work at same time.

    # ---------------- Tensor / filter descriptors -------------------------
    check_cudnn_error(
        cudnnSetFilter4dDescriptor(
            cudnn_handle[].ptr_filter_desc,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,  # cudnn documentation correction: cudnnSetFilter4dDescriptor() takes CKRS, not KCRS
            Int16(filter.dim[0]()),  # C (out channels)
            Int16(filter.dim[1]()),  # K (in channels)
            Int16(filter.dim[2]()),  # R (kernel height)
            Int16(filter.dim[3]()),  # S (kernel width)
        )
    )

    check_cudnn_error(
        cudnnSetTensor4dDescriptor(
            cudnn_handle[].ptr_input_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
            Int16(input.dim[0]()),  # N
            Int16(input.dim[1]()),  # C_in
            Int16(input.dim[2]()),  # H_in
            Int16(input.dim[3]()),  # W_in
        )
    )

    check_cudnn_error(
        cudnnSetTensor4dDescriptor(
            cudnn_handle[].ptr_output_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
            Int16(output.dim[0]()),  # N
            Int16(output.dim[1]()),  # C_out
            Int16(output.dim[2]()),  # H_out
            Int16(output.dim[3]()),  # W_out
        )
    )

    check_cudnn_error(
        cudnnSetConvolution2dDescriptor(
            cudnn_handle[].ptr_conv_desc,
            Int16(padding[0]),
            Int16(padding[1]),
            Int16(stride[0]),
            Int16(stride[1]),
            Int16(dilation[0]),
            Int16(dilation[1]),
            cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
        )
    )

    # ---------------- Algorithm & workspace -------------------------------
    var algo = cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0

    # ALGO_0 is the implicit deconv formulation; cuDNN may still require a
    # workspace for larger shapes. Query the exact size and allocate a
    # per-call device buffer so we don't trip CUDNN_STATUS_NOT_SUPPORTED.
    var workspace_size: Int = 0
    check_cudnn_error(
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnn_handle[].ptr_handle,
            cudnn_handle[].ptr_filter_desc,
            cudnn_handle[].ptr_input_desc,
            cudnn_handle[].ptr_conv_desc,
            cudnn_handle[].ptr_output_desc,
            algo,
            UnsafePointer(to=workspace_size),
        )
    )

    var workspace_buffer = ctx.enqueue_create_buffer[.uint8](workspace_size)

    var alpha = Float32(1.0)
    var beta = Float32(0.0)

    check_cudnn_error(
        cudnnConvolutionBackwardData(
            cudnn_handle[].ptr_handle,
            UnsafePointer(to=alpha).bitcast[NoneType](),
            cudnn_handle[].ptr_filter_desc,
            filter.ptr.bitcast[NoneType](),
            cudnn_handle[].ptr_input_desc,
            input.ptr.bitcast[NoneType](),
            cudnn_handle[].ptr_conv_desc,
            algo,
            workspace_buffer.unsafe_ptr()
            .bitcast[NoneType]()
            .as_unsafe_any_origin(),
            workspace_size,
            UnsafePointer(to=beta).bitcast[NoneType](),
            cudnn_handle[].ptr_output_desc,
            output.ptr.bitcast[NoneType](),
        )
    )
    _ = workspace_buffer^


def conv_transposed_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[mut=False, input_type, address_space=.GENERIC, ...],
    filter: TileTensor[mut=False, filter_type, address_space=.GENERIC, ...],
    output: TileTensor[output_type, address_space=.GENERIC, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
    ctx: DeviceContext,
) raises:
    """Runs a 2D transposed convolution on the GPU using cuDNN backward-data.

    Pushes the device context and delegates to the internal cuDNN backward-data
    helper that configures the tensor, filter, and convolution descriptors and
    launches the cuDNN kernel.

    Parameters:
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.

    Args:
        input: Input tensor in NCHW layout.
        filter: Filter tensor in CKRS layout.
        output: Output tensor in NCHW layout.
        stride: Stride along the height and width axes.
        dilation: Dilation along the height and width axes of the filter.
        padding: Padding along the height and width axes.
        ctx: Device context used to bind the cuDNN handle to the current stream.
    """
    # Set the CUcontext as current to satisfy stateful cuDNN APIs.
    with ctx.push_context():
        _conv_transposed_cudnn(
            input, filter, output, stride, dilation, padding, ctx
        )
