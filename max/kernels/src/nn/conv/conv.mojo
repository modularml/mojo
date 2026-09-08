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

"""Convolution kernels for CPU and GPU targets.

Provides direct (register-tiled) convolution, cuDNN-backed convolution
(NVIDIA), MIOpen-backed convolution (AMD), and naive GPU reference kernels
for 1D, 2D, and 3D convolutions in NHWC/NDHWC layouts.
"""

from std.collections import Optional
from std.math import align_down, ceildiv
from std.math.uutils import udivmod
from std.memory import ThinAllocation, dealloc


from std.os import abort, getenv
from std.ffi import _get_global_or_null, external_call
from std.sys.info import align_of, simd_width_of, size_of

from _cudnn.cnn_infer import (
    cudnnConvolutionForward,
    cudnnConvolutionFwdAlgoPerfStruct,
    cudnnConvolutionMode_t,
    cudnnConvolutionStruct,
    cudnnCreateConvolutionDescriptor,
    cudnnDestroyConvolutionDescriptor,
    cudnnFindConvolutionForwardAlgorithmEx,
    cudnnGetConvolutionForwardWorkspaceSize,
    cudnnSetConvolution2dDescriptor,
    cudnnSetConvolutionGroupCount,
    cudnnSetConvolutionMathType,
    cudnnSetConvolutionNdDescriptor,
    cudnnGetConvolutionForwardAlgorithm_v7,
    cudnnConvolutionFwdAlgoPerf_t,
)
from _cudnn.infer import (
    cudnnContext,
    cudnnConvolutionFwdAlgo_t,
    cudnnCreate,
    cudnnCreateFilterDescriptor,
    cudnnCreateTensorDescriptor,
    cudnnDataType_t,
    cudnnDestroy,
    cudnnDestroyFilterDescriptor,
    cudnnDestroyTensorDescriptor,
    cudnnFilterStruct,
    cudnnMathType_t,
    cudnnSetFilter4dDescriptor,
    cudnnSetFilterNdDescriptor,
    cudnnSetStream,
    cudnnSetTensor4dDescriptor,
    cudnnSetTensorNdDescriptorEx,
    cudnnStatus_t,
    cudnnTensorFormat_t,
    cudnnTensorStruct,
)
from _miopen.miopen import (
    miopenCreate,
    miopenSetStream,
    miopenCreateTensorDescriptor,
    miopenSetTensorDescriptorV2,
    miopenCreateConvolutionDescriptor,
    miopenInitConvolutionNdDescriptor,
    miopenSetConvolutionGroupCount,
    miopenConvolutionForwardGetWorkSpaceSize,
    miopenFindConvolutionForwardAlgorithm,
    miopenConvolutionForward,
)
from _miopen.types import (
    Handle as MIOpenHandle,
    TensorDescriptor as MIOpenTensorDescriptor,
    ConvolutionDescriptor as MIOpenConvolutionDescriptor,
    DataType as MIOpenDataType,
    ConvolutionMode,
    ConvFwdAlgorithm,
    ConvAlgoPerf,
)
from _miopen.utils import check_error as check_miopen_error
from std.algorithm import (
    tile,
    tile_middle_unswitch_boundaries,
    unswitch,
    vectorize,
)

from max.algorithm import (
    elementwise,
    sync_parallelize,
)
from linalg.utils import (
    partial_simd_load,
    partial_simd_store,
)
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import get_gpu_target, DeviceBuffer, DeviceContext
from max.gpu.host._amdgpu_hip import HIP
from max.gpu.host._nvidia_cuda import CUDA
from max.gpu.host.info import _is_sm10x_gpu
from layout import (
    Coord,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from layout.coord import DynamicCoord
from linalg.accumulate import _Accumulator
from linalg.utils import partition_work
from max.runtime.asyncrt import parallelism_level
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.sys import (
    has_amd_gpu_accelerator,
    has_amd_rdna_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type


from .conv_utils import (
    ConvInfoStatic,
    ConvPartition,
    ConvShape,
    align_down_residual,
    elementwise_epilogue_type,
    elementwise_simd_epilogue_type,
    get_conv_num_partitions,
    get_conv_shape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
    get_partition,
    reorder_padding,
)
from .gpu.amd.dispatch_3d import dispatch_amd_4wave_conv3d
from .gpu.im2col_matmul_3d import dispatch_im2col_matmul_conv3d
from .gpu.matmul_1x1x1_conv3d import dispatch_1x1x1_matmul_conv3d
from .gpu.nvidia.sm100.qslice_conv3d import dispatch_qslice_conv3d_sm100
from nn.shapes import get_sliding_window_out_dim
from nn.pad_gpu import pad_constant as pad_constant_gpu
from layout import lt_to_tt


struct Naive2dConvolution[
    output_origin: Origin[mut=True],
    input_origin: ImmOrigin,
    filter_origin: ImmOrigin,
    //,
    output_type: DType,
    input_type: DType,
    filter_type: DType,
](ImplicitlyCopyable):
    """Struct wrapper for naive 2d convolution implementation.

    Parameters:
        output_origin: Mutable memory origin of the output tensor (inferred).
        input_origin: Immutable memory origin of the input tensor (inferred).
        filter_origin: Immutable memory origin of the filter tensor (inferred).
        output_type: Element type of the output tensor.
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
    """

    # Input params.
    var output: UnsafePointer[Scalar[Self.output_type], Self.output_origin]
    var input: UnsafePointer[Scalar[Self.input_type], Self.input_origin]
    var filter: UnsafePointer[Scalar[Self.filter_type], Self.filter_origin]
    var pad_d: DynamicCoord[.int64, 2]
    var pad_h: DynamicCoord[.int64, 2]
    var pad_w: DynamicCoord[.int64, 2]
    var stride: DynamicCoord[.int64, 3]
    var dilation: DynamicCoord[.int64, 3]
    var num_groups: Int

    # Derived params.
    var output_shape: DynamicCoord[.int64, 5]  # NDHWC layout.
    var input_shape: DynamicCoord[.int64, 5]  # NDHWC layout.
    var filter_shape: DynamicCoord[.int64, 5]  # QRSCF layout.

    @staticmethod
    def run(
        output: UnsafePointer[Scalar[Self.output_type], Self.output_origin],
        input: UnsafePointer[Scalar[Self.input_type], Self.input_origin],
        filter: UnsafePointer[Scalar[Self.filter_type], Self.filter_origin],
        output_shape: IndexList[5],
        input_shape: IndexList[5],
        filter_shape: IndexList[5],
        pad_d: IndexList[2],
        pad_h: IndexList[2],
        pad_w: IndexList[2],
        stride: IndexList[3],
        dilation: IndexList[3],
        num_groups: Int,
    ):
        # Create an instance of the convolution op.
        var naive2d_convolution = Naive2dConvolution[
            Self.output_type, Self.input_type, Self.filter_type
        ](
            output,
            input,
            filter,
            output_shape,
            input_shape,
            filter_shape,
            pad_d,
            pad_h,
            pad_w,
            stride,
            dilation,
            num_groups,
        )

        # Run the actual loops and computations.
        naive2d_convolution._outer_loop()

    def __init__(
        out self,
        output: UnsafePointer[Scalar[Self.output_type], Self.output_origin],
        input: UnsafePointer[Scalar[Self.input_type], Self.input_origin],
        filter: UnsafePointer[Scalar[Self.filter_type], Self.filter_origin],
        output_shape: IndexList[5],
        input_shape: IndexList[5],
        filter_shape: IndexList[5],
        pad_d: IndexList[2],
        pad_h: IndexList[2],
        pad_w: IndexList[2],
        stride: IndexList[3],
        dilation: IndexList[3],
        num_groups: Int,
    ):
        self.output = output
        self.input = input
        self.filter = filter
        self.output_shape = Coord(output_shape)
        self.input_shape = Coord(input_shape)
        self.filter_shape = Coord(filter_shape)
        self.pad_d = Coord(pad_d)
        self.pad_h = Coord(pad_h)
        self.pad_w = Coord(pad_w)
        self.stride = Coord(stride)
        self.dilation = Coord(dilation)
        self.num_groups = num_groups

    def _outer_loop(self):
        """Implementation of the outermost loop of a convolution operator with
        loops covering the iteration space of batch, filter count, height and wi-
        dth dimensions.
        """
        # Iterate on output batch dimension.
        for n in range(Int(self.output_shape[0].value())):
            # Iterate on filter dimension.
            for f in range(Int(self.output_shape[4].value())):
                # Iterate on output H dimension.
                for do in range(Int(self.output_shape[1].value())):
                    # Iterate on output H dimension.
                    for ho in range(Int(self.output_shape[2].value())):
                        # Iterate on output W dimension.
                        for wo in range(Int(self.output_shape[3].value())):
                            # Compute the result value at this specific output posit-
                            #  ion.
                            self._compute_point(n, do, ho, wo, f)

    def _compute_point(self, n: Int, do: Int, ho: Int, wo: Int, f: Int):
        """Implementation of the inner loop computation of a conv2d operator
        producing a single scalar value at the given output tensor index.
        """
        # Initialize the result of this point.
        var value: Scalar[Self.output_type] = 0

        # Input dims.
        var D = Int(self.input_shape[1].value())
        var H = Int(self.input_shape[2].value())
        var W = Int(self.input_shape[3].value())
        var C = Int(self.input_shape[4].value())
        var image_bound = Index(D, H, W)
        var C_per_group = C // self.num_groups

        # Filter dims.
        var Q = Int(self.filter_shape[0].value())
        var R = Int(self.filter_shape[1].value())
        var S = Int(self.filter_shape[2].value())

        # Output dims.
        var DO = Int(self.output_shape[1].value())
        var HO = Int(self.output_shape[2].value())
        var WO = Int(self.output_shape[3].value())
        var F = Int(self.output_shape[4].value())

        var g = f // (F // self.num_groups)

        var stride = coord_to_index_list(self.stride)
        var dilation = coord_to_index_list(self.dilation)
        # Padding offset, using the left padding only here.
        var pad_lower = Index(
            Int(self.pad_d[0].value()),
            Int(self.pad_h[0].value()),
            Int(self.pad_w[0].value()),
        )

        for q in range(Q):
            for r in range(R):
                for s in range(S):
                    # Compute input access index, on the H and W dimension.
                    var dhw = (
                        # Output HxW with striding.
                        Index(do, ho, wo) * stride
                        +
                        # Filter RxS with dilation.
                        (Index(q, r, s) * dilation)
                        - pad_lower
                    )

                    # Check that the current image index is within valid range
                    #  on the input image data tensor.
                    if Index(0, 0, 0) <= dhw < image_bound:
                        # Iterate on channels dimension.
                        for c in range(C_per_group * g, C_per_group * (g + 1)):
                            # Accumulate product of input data filter data.
                            var input_val = self.input[
                                c
                                + C
                                * (dhw[2] + W * (dhw[1] + H * (dhw[0] + D * n)))
                            ]
                            var c_in_group = c % C_per_group
                            var filter_val = self.filter[
                                f
                                + F
                                * (
                                    c_in_group
                                    + C_per_group * (s + S * (r + R * q))
                                )
                            ]
                            value += (
                                input_val.cast[Self.output_type]()
                                * filter_val.cast[Self.output_type]()
                            )

        # Store the computed output at the given output position..
        self.output.store(f + F * (wo + WO * (ho + HO * (do + DO * n))), value)


# ===----------------------------------------------------------------------=== #
# Direct convolution helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _m_to_n_ho_wo_nhwc(m: Int, HO: Int, WO: Int) -> IndexList[3]:
    """Converts post-im2col m dimension index to pre-im2col coordinates on
    (N, Hout, Wout) dimensions.
        Args:
            m (Int): Index on M dimension.
            conv_shape (ConvShape): convolution dimension description.

        Returns (IndexList):
            The translated 3d indices in (N, Hout, Wout) format.
    TODO: This utility should be generalized into a im2col util
    class with some additional layout agnostic logic.
    """
    var n, rem = divmod(m, HO * WO)
    var ho, wo = divmod(rem, WO)
    return Index(n, ho, wo)


# Reduce helper when the input channel dimension is partitioned.
@always_inline
def _reduce_output[
    dtype: DType,
    //,
    simd_size: Int,
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](
    scratch: UnsafePointer[mut=False, Scalar[dtype], _],
    output: UnsafePointer[mut=True, Scalar[dtype], _],
    N: Int,
    output_space_dims: IndexList,
    F: Int,
    num_partitions: Int,
    num_threads: Int,
    ctx: Optional[DeviceContext] = None,
):
    var num_rows = N * output_space_dims.flattened_length()
    var buf_size = num_rows * F

    # Reduce from the output scratch buffer to the actual output.
    @always_inline
    def reduce_task(tid: Int) {imm}:
        # Use all threads in reduction.
        var reduce_range = partition_work(tid, num_threads, num_rows, 1)

        @always_inline
        def sum[
            width: Int
        ](offset: Int) {F, scratch, num_partitions, output, imm buf_size, mut}:
            var tid_output_offset = reduce_range[0] * F + offset
            var vec = scratch.load[width=width](tid_output_offset)
            # The number of partitions here is typically small.
            # There may not be much benefit from unrolling the reduction axis.
            # Only unroll the last dimension.
            for i in range(1, num_partitions):
                vec += scratch.load[width=width](
                    tid_output_offset + i * buf_size
                )
            output.store(tid_output_offset, vec)

        vectorize[simd_size, unroll_factor=4](reduce_range[1] * F, sum)

        comptime if elementwise_epilogue:
            comptime epilogue = elementwise_epilogue.value()
            for m in range(reduce_range[0], reduce_range[0] + reduce_range[1]):
                var nhowo = _m_to_n_ho_wo_nhwc(
                    m, output_space_dims[0], output_space_dims[1]
                )
                epilogue(Index(nhowo[0], nhowo[1], nhowo[2], 0), F)

    # NOTE: _synchronous, so use of locally allocated output_ptr is safe.
    sync_parallelize(reduce_task, num_threads, ctx)


# ===----------------------------------------------------------------------=== #
# Direct Convolution Entry Point                                               #
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ConvDirectNHWC[
    conv_attr_rank: Int,
    input_origin: ImmOrigin,
    filter_origin: ImmOrigin,
    output_origin: Origin[mut=True],
    //,
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_packed: Bool,
    conv_attr: ConvInfoStatic[conv_attr_rank],
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](ImplicitlyCopyable):
    """Implement the outer loops for direct convolution.
    Collapse N, HO, WO into one dimension n_ho_wo. Tile n_ho_wo, C, and F.
    The tile factor for C and F are chosen by a heuristic prioritizing C.
    n_ho_wo is tiled by micro kernel's height.

    If n_ho_wo is large enough to spill LLC, we may need to tile n_ho_wo as the
    outer most loop with a factor fit in LLC.

    Assume F is divisible at least by simd_size.

    Parameters:
        conv_attr_rank: Number of spatial dimensions in the convolution
            (1, 2, or 3) (inferred).
        input_origin: Immutable memory origin of the input tensor (inferred).
        filter_origin: Immutable memory origin of the filter tensor (inferred).
        output_origin: Mutable memory origin of the output tensor (inferred).
        input_layout: Memory layout of the input tensor.
        filter_layout: Memory layout of the filter tensor.
        output_layout: Memory layout of the output tensor.
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.
        filter_packed: True when the filter is prepacked for grouped
            convolution.
        conv_attr: Statically known convolution attributes including
            padding, stride, dilation, and group count.
        elementwise_epilogue: Optional elementwise function applied to
            the output after the last channel tile (defaults to `None`).
    """

    var output: LayoutTensor[
        Self.output_type, Self.output_layout, Self.output_origin
    ]
    var input: LayoutTensor[
        Self.input_type, Self.input_layout, Self.input_origin
    ]
    var filter: LayoutTensor[
        Self.filter_type, Self.filter_layout, Self.filter_origin
    ]

    var conv_shape: ConvShape[Self.conv_attr_rank]

    # Support partition in 4 dims: (n, c, f, ho_or_howo). If the input is
    # padded, the output spatial dims are merged into one as howo. If not
    # padded, only ho is partitioned for now.
    var partition: ConvPartition

    var cf_tile_size: DynamicCoord[.int64, 2]

    # If shapes and attributes are known at compile time
    comptime packed_and_fully_static = Self.conv_attr.all_known() and Self.input_layout.shape.all_known[
        1, Self.input_layout.rank()
    ]() and Self.output_layout.shape.all_known[
        1, Self.output_layout.rank()
    ]() and Self.filter_layout.shape.all_known() and Self.filter_packed

    @staticmethod
    def run(
        output: LayoutTensor[
            Self.output_type, Self.output_layout, Self.output_origin
        ],
        input: LayoutTensor[
            Self.input_type, Self.input_layout, Self.input_origin
        ],
        filter: LayoutTensor[
            Self.filter_type, Self.filter_layout, Self.filter_origin
        ],
        conv_shape: ConvShape[Self.conv_attr_rank],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime assert Self.conv_attr_rank == Self.input_layout.rank() - 2
        comptime simd_size = simd_width_of[Self.output_type]()
        # TODO: extend to 1d/3d.
        comptime WO = Int(
            Self.output_layout.shape[output.rank - 2]
        ) if input.rank == 4 else UNKNOWN_VALUE
        comptime F = Int(Self.output_layout.shape[output.rank - 1])
        comptime micro_kernel_shape = get_micro_kernel_shape[
            Self.conv_attr_rank,
            WO,
            F,
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
                Self.filter_packed or Self.conv_attr.num_groups == 1
            ), (
                "if number of conv groups is statically known, conv filter"
                " must be prepacked when num_groups > 1"
            )

        if conv_shape.num_groups > 1 and not Self.filter_packed:
            raise Error("grouped conv requires packed filter")
        if conv_shape.c % conv_shape.num_groups != 0:
            raise Error("channel count must be divisible by group count")
        if conv_shape.f % conv_shape.num_groups != 0:
            raise Error("filter count must be divisible by group count")

        # Number of partitions in n, ho_wo, c, f dimensions.
        var num_threads = parallelism_level(ctx)
        var num_partitions = get_conv_num_partitions[
            micro_kernel_height, micro_kernel_f_size
        ](num_threads, conv_shape)
        var num_tasks = num_partitions.flattened_length()

        # Safety: the scratch pointer below will alias the output_ptr, so cast to MutAnyOrigin
        # here to turn off the check.
        var output_ptr = output.ptr.unsafe_origin_cast[MutUntrackedOrigin]()
        var output_size = output.size()
        var scratch_size = num_partitions[1] * output_size
        if num_partitions[1] > 1:
            output_ptr = alloc[Scalar[Self.output_type]](
                {count = scratch_size}
            ).unsafe_leak()
        # Wrap the pointer inside LayoutTensor so it can be properly captured by async closure.
        var output_scratch = LayoutTensor[
            Self.output_type, Layout.row_major(UNKNOWN_VALUE)
        ](
            output_ptr,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                Index(scratch_size)
            ),
        )

        @always_inline
        def task_func(
            task_id: Int,
        ) {
            var num_partitions,
            var cf_tile_size,
            var output_scratch,
            var output_size,
            imm,
        }:
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

            # TODO: Need to have a more robust way to compute task_id_c
            var task_id_c = (task_id // num_partitions[2]) % num_partitions[1]
            var task_output = LayoutTensor[
                Self.output_type, Self.output_layout
            ](
                output_scratch.ptr + task_id_c * output_size,
                RuntimeLayout[Self.output_layout].row_major(
                    output.runtime_layout.shape.value.canonicalize()
                ),
            )

            var instance = ConvDirectNHWC[
                Self.input_layout,
                Self.filter_layout,
                Self.output_layout,
                Self.input_type,
                Self.filter_type,
                Self.output_type,
                Self.filter_packed,
                Self.conv_attr,
                Self.elementwise_epilogue,
            ](
                task_output,
                input,
                filter,
                conv_shape,
                partition,
                Coord(task_tile_size),
            )
            instance._batch_group_loop()

        if num_partitions[1] > 1:
            sync_parallelize(task_func, num_tasks, ctx)

            # Reduce from the output scratch buffer to the actual output.
            _reduce_output[
                simd_size,
                # Only support channel partition for 2D shapes (ResNet).
                elementwise_epilogue=Self.elementwise_epilogue if input.rank
                == 4 else None,
            ](
                output_scratch.ptr,
                output.ptr,
                conv_shape.n,
                conv_shape.output_space_dims(),
                conv_shape.f,
                num_partitions[1],
                num_threads,
                ctx,
            )
            dealloc(
                ThinAllocation(unsafe_owned_ptr=output_ptr).unsafe_with_layout(
                    {count = scratch_size}
                )
            )
        else:
            # Use sync to work around #12624
            sync_parallelize(task_func, num_tasks, ctx)

    def _batch_group_loop(self):
        """Loop over the batch and group dimensions. The two dimension are
        merged and partitioned for parallelism."""

        @always_inline
        def body[padded: Bool]() {imm}:
            for ng in range(
                self.partition.ng_offset,
                self.partition.ng_offset + self.partition.ng_size,
            ):
                var n, g = divmod(ng, self.conv_shape.num_groups)
                self._c_tile_loop[padded](
                    n, g, Int(self.cf_tile_size[0].value())
                )

        unswitch(self.conv_shape.padded(), body)

    def _c_tile_loop[padded: Bool](self, n: Int, g: Int, tile_size: Int):
        """Loop over C tiles."""

        # TODO: Extend to 1D/3D.
        # fmt: off
        comptime apply_static_shape_optimization = \
            self.packed_and_fully_static \
            and padded \
            and Self.conv_attr.num_groups == 1 \
            and Self.input_layout.rank() == 4
        # fmt: on

        @always_inline
        def c_tile_iteration(c_tile_offset: Int, c_tile_size: Int) {imm}:
            # Only apply static shape optimizations to shapes with padding since
            # there is a fast path for pointwise (no padding) conv with strides.
            # Grouped conv logic has not been plumbed into static specialized funcs yet.
            comptime if apply_static_shape_optimization:
                self._f_tile_loop_static[False](n, c_tile_offset, c_tile_size)
            else:
                self._f_tile_loop[padded, False](
                    n, g, c_tile_offset, c_tile_size
                )

        # Can't fuse epilogue inside conv if C is partitioned
        if self.partition.c_size < self.conv_shape.c:
            tile(
                self.partition.c_offset,
                self.partition.c_offset + self.partition.c_size,
                tile_size,
                workgroup_function=c_tile_iteration,
            )
        # C is not partitioned, fuse epilogue in the last C tile.
        else:
            # for g in range(self.conv_shape.num_groups):
            var c_start = g * self.conv_shape.c_per_group()
            var c_round_by_tile = align_down(
                (self.conv_shape.c_per_group() - 1), tile_size
            )
            var c_round_by_tile_residual = (
                self.conv_shape.c_per_group() - c_round_by_tile
            )
            tile(
                c_start,
                c_start + c_round_by_tile,
                tile_size,
                workgroup_function=c_tile_iteration,
            )

            # Update the last c tile with fusion
            comptime if apply_static_shape_optimization:
                self._f_tile_loop_static[True](
                    n,
                    c_start + c_round_by_tile,
                    c_round_by_tile_residual,
                )
            else:
                self._f_tile_loop[padded, True](
                    n,
                    g,
                    c_start + c_round_by_tile,
                    c_round_by_tile_residual,
                )

    def _f_tile_loop[
        padded: Bool, last_c_tile: Bool
    ](self, n: Int, g: Int, c_tile_offset: Int, c_tile_size: Int):
        """Loop over F tiles."""
        comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
        comptime micro_kernel_height = get_direct_conv_micro_kernel_height()
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        # TODO: Extend the merged loop to support 1d and 3d.
        # For now, only merge HO and WO dims for 2D conv w/o padding.
        comptime merge_output_space_loops = (
            not padded
        ) and Self.input_layout.rank() == 4

        @always_inline
        def f_tile_iteration[
            size: Int
        ](f_tile_offset: Int, f_tile_size: Int) {imm}:
            comptime if not merge_output_space_loops:
                self.output_space_loop[
                    micro_kernel_height, size // simd_size, False, last_c_tile
                ](n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size)
            else:
                self.output_space_flat_loop[size, False, last_c_tile](
                    n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size
                )

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
            comptime if not merge_output_space_loops:
                self.output_space_loop[
                    micro_kernel_height, 1, True, last_c_tile
                ](
                    n,
                    group_f_end_align_simd,
                    simd_size,
                    c_tile_offset,
                    c_tile_size,
                )
            else:
                self.output_space_flat_loop[simd_size, True, last_c_tile](
                    n,
                    group_f_end_align_simd,
                    simd_size,
                    c_tile_offset,
                    c_tile_size,
                )

    @always_inline
    def is_new_c_accum(self, c_idx: Int) -> Bool:
        # returns true when processing first C in a group or first C in a C partition
        if self.conv_shape.num_groups > 1:
            return self.conv_shape.c_in_group(c_idx) == 0
        return c_idx == self.partition.c_offset

    def update_output_tile_no_padding[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        c_fully_cached: Bool,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        n: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
        output_flat_coord: Int,
    ):
        comptime assert not has_residual or (
            has_residual and micro_kernel_width == 1
        ), "Use Height x 1 kernel for residual in F."

        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        # Base input offsets.
        var input_base_stack = Array[_, micro_kernel_height](
            fill_with=lambda (i: Int) {
                imm self, imm n, imm output_flat_coord, imm c_tile_offset
            } -> Int32: Int32(
                self.conv_shape.output_flat_coord_to_input_offset(
                    n, output_flat_coord + i
                )
                + c_tile_offset
            )
        )
        var input_base_offsets = TileTensor(
            input_base_stack, row_major[micro_kernel_height]()
        )

        comptime alignment = align_of[SIMD[Self.output_type, simd_size]]()

        var acc = _Accumulator[
            Self.output_type,
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
        ]()

        var output_offset = (
            self.conv_shape.f
            * (n * self.conv_shape.output_image_flat_size() + output_flat_coord)
            + f_tile_offset
        )

        if self.is_new_c_accum(c_tile_offset):
            acc.init(0)
        else:
            acc.load[partial_load=has_residual](
                self.output.ptr + output_offset,
                self.conv_shape.f,
                self.conv_shape.f_per_group() % simd_size,
            )
        var filter_ptr: UnsafePointer[
            Scalar[Self.filter_type], Self.filter_origin
        ] = self.filter.ptr

        comptime if Self.filter_packed:
            # Move the pointer to the current group's start.
            filter_ptr = _get_group_filter_base(
                self.filter,
                self.conv_shape.c_to_group(c_tile_offset),  # group index
                self.conv_shape.f_per_group(),
            )
            # Move the pointer to (c_tile_offset, f_tile_offset) mapped in
            # current group.
            filter_ptr = filter_ptr + (
                # Jump over f_tile_offset in current group.
                self.conv_shape.f_in_group(f_tile_offset)
                * self.conv_shape.r()
                * self.conv_shape.s()
                * self.conv_shape.c_per_group()
                # Jump over c_tile_offset in current group.
                + self.conv_shape.c_in_group(c_tile_offset)
                * micro_kernel_f_size
            )

        for r in range(self.conv_shape.r()):
            for s in range(self.conv_shape.s()):
                var input_offset = self.conv_shape.c * (
                    s * self.conv_shape.dilation_at[1]()
                    + self.conv_shape.w() * r * self.conv_shape.dilation_at[0]()
                )

                # Unpacked version. For each (r, s), we first offset the
                # filter pointer by (r, s) plus c_tile_offset. Later for
                # each c, we access micro_kernel_f_size contiguous elements.
                # These contiguous segments are strided by F.
                comptime if not Self.filter_packed:
                    filter_ptr = self.filter.ptr + (
                        (s + r * self.conv_shape.s())
                        * self.conv_shape.c
                        * self.conv_shape.f
                        + c_tile_offset * self.conv_shape.f
                        + f_tile_offset
                    )

                self._accumulate[
                    micro_kernel_height,
                    micro_kernel_width,
                    simd_size,
                    has_residual and not Self.filter_packed,
                    prefetch_offset=4,
                ](
                    input_base_offsets,
                    input_offset,
                    c_tile_size,
                    self.input.ptr,
                    filter_ptr,
                    acc,
                )

                # Shift C*f to get the next point in stencil (s+1) for FRSCf layout.
                if Self.filter_packed:
                    filter_ptr = filter_ptr + (
                        self.conv_shape.c_per_group() * micro_kernel_f_size
                    )

        acc.store[partial_store=has_residual](
            self.output.ptr + output_offset,
            self.conv_shape.f,
            self.conv_shape.f_per_group() % simd_size,
        )

        comptime if Self.elementwise_epilogue.__bool__() and last_c_tile.__bool__():
            comptime epilogue = Self.elementwise_epilogue.value()

            # If has residual, the tile size has been extended to a simd_size.
            # Here needs to use the real bound F.
            var f_tile_size_bounded: Int

            comptime if has_residual:
                f_tile_size_bounded = (
                    self.conv_shape.f_per_group()
                    - self.conv_shape.f_in_group(f_tile_offset)
                )
            else:
                f_tile_size_bounded = f_tile_size

            for m in range(
                output_flat_coord, output_flat_coord + micro_kernel_height
            ):
                # The micro tile may cover points in different rows/images.
                # Convert the 1D index back to (n, ho, wo).
                var ho, wo = divmod(m, self.conv_shape.wo())
                epilogue(
                    Index(
                        n,
                        ho,
                        wo,
                        f_tile_offset,
                    ),
                    f_tile_size_bounded,
                )

    @always_inline
    def _init_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
    ](
        self,
        output_micro_tile: LayoutTensor[
            mut=True,
            Self.output_type,
            Layout.row_major(
                micro_kernel_height, micro_kernel_width * simd_size
            ),
            _,
        ],
    ):
        """Initialize a micro tile to zero.
        Arguments:
            n_ho_wo: offset of micro tile in fused (n, ho, wo) dimension.
            f: offset of micro tile in F dimension.
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
        """

        comptime for idx0 in range(micro_kernel_height):
            comptime for idx1 in range(micro_kernel_width):
                output_micro_tile.store[width=simd_size](
                    Index(idx0, idx1 * simd_size),
                    SIMD[Self.output_type, simd_size](0.0),
                )

    @always_inline
    def _load_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        output_base: UnsafePointer[Scalar[Self.output_type], ...],
        output_micro_tile: LayoutTensor[
            mut=True,
            Self.output_type,
            Layout.row_major(
                micro_kernel_height, micro_kernel_width * simd_size
            ),
            _,
        ],
    ):
        """Load a micro tile from the output buffer.
        Parameters:
            has_residual: True when F is not multiple of simd_size. The residual
              is loaded and padded with zero to fit a simd vector.

        Arguments:
            output_base: Point to micro tile start, (n, ho, wo, f).
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
        """
        var output_ptr = output_base

        comptime for i in range(micro_kernel_height):
            comptime for j in range(micro_kernel_width):
                comptime if has_residual:
                    var residual = align_down_residual(
                        self.conv_shape.f_per_group(), simd_size
                    )
                    output_micro_tile.store[width=simd_size](
                        Index(i, j * simd_size),
                        partial_simd_load[simd_size](
                            output_ptr + j * simd_size, 0, residual, 0.0
                        ),
                    )
                else:
                    output_micro_tile.store[width=simd_size](
                        Index(i, j * simd_size),
                        (output_ptr + j * simd_size).load[width=simd_size](),
                    )

            comptime if (
                Self.output_layout.shape[Self.output_layout.rank() - 1]
                != UNKNOWN_VALUE
            ):
                comptime F = Int(
                    Self.output_layout.shape[Self.output_layout.rank() - 1]
                )
                output_ptr = output_ptr + F
            else:
                output_ptr = output_ptr + self.conv_shape.f

    @always_inline
    def _store_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        output_micro_tile: LayoutTensor[
            mut=True,
            Self.output_type,
            Layout.row_major(
                micro_kernel_height, micro_kernel_width * simd_size
            ),
            _,
        ],
        output_base: UnsafePointer[mut=True, Scalar[Self.output_type], ...],
    ):
        """Store a micro tile from the output buffer.
        Parameters:
            has_residual: True when F is not multiple of simd_size. Only the
              residual elements within the simd vector are stored to output.

        Arguments:
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
            output_base: Point to micro tile start, (n, ho, wo, f).
        """
        var output_ptr = output_base

        comptime for i in range(micro_kernel_height):
            comptime for j in range(micro_kernel_width):
                var output_vec = output_micro_tile.load[width=simd_size](
                    Index(i, j * simd_size)
                )

                comptime if has_residual:
                    var residual = align_down_residual(
                        self.conv_shape.f_per_group(), simd_size
                    )
                    partial_simd_store[simd_size](
                        output_ptr + j * simd_size,
                        0,
                        residual,
                        output_vec,
                    )
                else:
                    output_ptr.store(j * simd_size, output_vec)

            comptime if (
                Self.output_layout.shape[Self.output_layout.rank() - 1]
                != UNKNOWN_VALUE
            ):
                comptime F = Int(
                    Self.output_layout.shape[Self.output_layout.rank() - 1]
                )
                output_ptr = output_ptr + F
            else:
                output_ptr = output_ptr + self.conv_shape.f

    @always_inline
    def _accumulate[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
        prefetch_offset: Int,
    ](
        self,
        input_base_offsets: TileTensor[mut=False, .int32, ...],
        input_offset: Int,
        c_tile_size: Int,
        input: UnsafePointer[mut=False, Scalar[Self.input_type], ...],
        filter: UnsafePointer[mut=False, Scalar[Self.filter_type], ...],
        mut acc: _Accumulator[
            Self.output_type,
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
        ],
    ):
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        var F = self.output.dim[3]()
        var filter_stride = micro_kernel_f_size if Self.filter_packed else F

        acc.accumulate[
            prefetch_offset=prefetch_offset,
            partial_load_b=has_residual and not Self.filter_packed,
        ](
            c_tile_size,
            input,
            input_base_offsets,
            input_offset,
            filter,
            filter_stride,
            F % simd_size,
        )

    @always_inline
    def _accumulate[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
        prefetch_offset: Int,
        row_start: Int,
        row_stop: Int,
    ](
        self,
        c_tile_size: Int,
        input_stride: Int,
        input_base: UnsafePointer[Scalar[Self.input_type], ...],
        filter_base: UnsafePointer[Scalar[Self.filter_type], ...],
        mut acc_in: _Accumulator[
            Self.output_type, micro_kernel_height, micro_kernel_width, simd_size
        ],
    ):
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        var F = self.output.dim[3]()
        var filter_stride = micro_kernel_f_size if Self.filter_packed else F

        # NOTE: To avoid initial load and final store after accumulation, this
        # function is rewritten to use a subset of storage in acc_in for rows
        # in range [row_start, row_stop].
        var acc = _Accumulator[
            Self.output_type,
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            row_start,
            row_stop,
        ](acc_in._storage.as_unsafe_any_origin())

        acc.accumulate[
            prefetch_offset=prefetch_offset,
            partial_load_b=has_residual and not Self.filter_packed,
        ](
            c_tile_size,
            input_base,
            input_stride,
            filter_base,
            filter_stride,
            F % simd_size,
        )

    def output_space_flat_loop[
        micro_kernel_f_size: Int, has_residual: Bool, last_c_tile: Bool
    ](
        self,
        n: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_height = get_direct_conv_micro_kernel_height()
        comptime micro_kernel_width = micro_kernel_f_size // simd_size

        @always_inline
        def iteration[tile_size: Int](output_flat_coord: Int) {imm}:
            @always_inline
            def body[c_fully_cached: Bool]() {imm}:
                self.update_output_tile_no_padding[
                    tile_size,  # micro kernel height
                    micro_kernel_width,
                    c_fully_cached,
                    has_residual,
                    last_c_tile,
                ](
                    n,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    output_flat_coord,
                )

            # c_fully_cached means the C dimension is fully covered in the
            # cache tile.
            unswitch(self.conv_shape.c == c_tile_size, body)

        # After the loop can't be stepped with micro_kernel_height,
        # it will step by 5, 4, 3, 2, 1.
        tile[[micro_kernel_height, 5, 4, 3, 2, 1]](
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
            iteration,
        )

    def output_space_loop[
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
        # Use ImmutAnyOrigin to detach from self's filter_origin for aliasing.
        var filter_ptr: UnsafePointer[
            Scalar[Self.filter_type], type_of(self.filter.ptr).origin
        ]

        comptime if Self.filter_packed:
            # Move the pointer to the current group's start.
            filter_ptr = _get_group_filter_base(
                self.filter, g, self.conv_shape.f_per_group()
            )
            # Move the pointer to (c_tile_offset, f_tile_offset) mapped in
            # current group.
            filter_ptr = filter_ptr + (
                # Jump over f_tile_offset in current group.
                self.conv_shape.f_in_group(f_tile_offset)
                * self.conv_shape.c_per_group()
                * self.conv_shape.filter_window_flat_size()
                # Jump over c_tile_offset in current group.
                + self.conv_shape.c_in_group(c_tile_offset)
                * micro_kernel_f_size
            )
        else:
            filter_ptr = self.filter.ptr + (
                c_tile_offset * self.conv_shape.f + f_tile_offset
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

        # Divide each row into three part:
        # [0, left_pad_impact_end)
        # [left_pad_impact_end, right_pad_impact_start)
        # [right_pad_impact_start, WO)
        comptime w_axis = Self.input_layout.rank() - 3
        var left_pad_impact_end = ceildiv(
            self.conv_shape.pad_w_lower(),
            self.conv_shape.stride_at[w_axis](),
        )
        var right_pad_impact_start = (
            self.conv_shape.w()
            + self.conv_shape.pad_w_lower()
            - self.conv_shape.s() * self.conv_shape.dilation_at[w_axis]()
        ) // self.conv_shape.stride_at[w_axis]() + 1

        comptime if Self.input_layout.rank() == 3:
            self.output_space_loop_1d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                # Safety: turn off mutable aliasing pointer check
                output_ptr.unsafe_origin_cast[AnyOrigin[mut=True]](),
                input_ptr,
                filter_ptr,
                n,
                self.is_new_c_accum(c_tile_offset),
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )
        elif Self.input_layout.rank() == 4:
            self.output_space_loop_2d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                # Safety: turn off mutable aliasing pointer check
                output_ptr.unsafe_origin_cast[AnyOrigin[mut=True]](),
                input_ptr,
                filter_ptr,
                n,
                self.is_new_c_accum(c_tile_offset),
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )
        elif Self.input_layout.rank() == 5:
            self.output_space_loop_3d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                # Safety: turn off mutable aliasing pointer check
                output_ptr.unsafe_origin_cast[AnyOrigin[mut=True]](),
                input_ptr,
                filter_ptr,
                n,
                self.is_new_c_accum(c_tile_offset),
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )

    def output_space_loop_1d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[mut=True, Scalar[output_dt], ...],
        input: UnsafePointer[mut=False, Scalar[input_dt], ...],
        filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()

        # Offset by -pad_w because s loop starts from the leftmost neighbor
        # in padding. The kernel skip the padding point and increment the
        # pointer.
        var input_base = (
            input - self.conv_shape.c * self.conv_shape.pad_w_lower()
        )

        # Points output to the start of the row
        var output_base = output

        # The bases can't be captured mutably, so derive the per-tile
        # pointers from wo instead of incrementing across calls.
        @always_inline
        def work_fn[height: Int, effected_by_padding: Bool](wo: Int) {imm}:
            conv1d_update_wo_tile[
                height,
                micro_kernel_width,
                simd_size,
                Self.filter_packed,
                effected_by_padding,
                has_residual,
                last_c_tile,
                elementwise_epilogue=Self.elementwise_epilogue,
            ](
                output_base + wo * self.conv_shape.f,
                input_base
                + wo * self.conv_shape.stride_at[0]() * self.conv_shape.c,
                filter,
                first_c_tile_in_group,
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                rebind[ConvShape[1]](self.conv_shape),
                n,
                wo,
            )

        tile_middle_unswitch_boundaries[[micro_kernel_height, 5, 4, 3, 2, 1]](
            0,
            left_pad_impact_end,
            right_pad_impact_start,
            self.conv_shape.wo(),
            work_fn,
        )

    def output_space_loop_2d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[mut=True, Scalar[output_dt], ...],
        input: UnsafePointer[mut=False, Scalar[input_dt], ...],
        filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()

        for ho in range(
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        ):
            var h = (
                ho * self.conv_shape.stride_at[0]()
                - self.conv_shape.pad_h_lower()
            )

            # Points input to the start of the row.
            # Offset by -pad_w because s loop starts from the leftmost neighbor
            # in padding. The kernel skip the padding point and increment the
            # pointer.
            var input_base = input + self.conv_shape.c * (
                -self.conv_shape.pad_w_lower() + self.conv_shape.w() * h
            )

            # Points output to the start of the row
            var output_base = (
                output + self.conv_shape.f * self.conv_shape.wo() * ho
            )

            # The bases can't be captured mutably, so derive the per-tile
            # pointers from wo instead of incrementing across calls.
            # TODO(MOCO-4664): `var ho` copy-captures the loop variable to work
            # around wrong debug-info scopes on implicit nested-scope captures.
            @always_inline
            def work_fn[
                height: Int, effected_by_padding: Bool
            ](wo: Int) {var ho, imm}:
                conv2d_update_wo_tile[
                    height,
                    micro_kernel_width,
                    simd_size,
                    Self.filter_packed,
                    effected_by_padding,
                    has_residual,
                    last_c_tile,
                    elementwise_epilogue=Self.elementwise_epilogue,
                ](
                    output_base + wo * self.conv_shape.f,
                    input_base
                    + wo * self.conv_shape.stride_at[1]() * self.conv_shape.c,
                    filter,
                    first_c_tile_in_group,
                    c_tile_size,
                    f_tile_offset,
                    f_tile_size,
                    rebind[ConvShape[2]](self.conv_shape),
                    n,
                    Index(ho, wo),
                )

            tile_middle_unswitch_boundaries[
                [micro_kernel_height, 5, 4, 3, 2, 1]
            ](
                0,
                left_pad_impact_end,
                right_pad_impact_start,
                self.conv_shape.wo(),
                work_fn,
            )

    def output_space_loop_3d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[mut=True, Scalar[output_dt], ...],
        input: UnsafePointer[mut=False, Scalar[input_dt], ...],
        filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()

        for do in range(0, self.conv_shape.do()):
            var d = (
                do * self.conv_shape.stride_at[0]()
                - self.conv_shape.pad_d_lower()
            )

            for ho in range(
                self.partition.ho_or_howo_offset,
                self.partition.ho_or_howo_offset
                + self.partition.ho_or_howo_size,
            ):
                # fmt: off
                var h = ho * self.conv_shape.stride_at[1]() - self.conv_shape.pad_h_lower()
                # fmt: on

                # Points input to the start of the row.
                # Offset by -pad_w because s loop starts from the leftmost neighbor
                # in padding. The kernel skip the padding point and increment the
                # pointer.
                var input_base = input + self.conv_shape.c * (
                    -self.conv_shape.pad_w_lower()
                    + self.conv_shape.w() * (h + self.conv_shape.h() * d)
                )

                # Points output to the start of the row
                var output_base = (
                    output
                    + self.conv_shape.f
                    * self.conv_shape.wo()
                    * (ho + self.conv_shape.ho() * do)
                )
                var conv_shape = self.conv_shape

                # The bases can't be captured mutably, so derive the per-tile
                # pointers from wo instead of incrementing across calls.
                @always_inline
                def work_fn[
                    height: Int, effected_by_padding: Bool
                ](wo: Int) {
                    var input_base,
                    var output_base,
                    var conv_shape,
                    var filter,
                    var first_c_tile_in_group,
                    var c_tile_size,
                    var f_tile_offset,
                    var f_tile_size,
                    var n,
                    var do,
                    var ho,
                }:
                    conv3d_update_wo_tile[
                        height,
                        micro_kernel_width,
                        simd_size,
                        Self.filter_packed,
                        effected_by_padding,
                        has_residual,
                        last_c_tile,
                        elementwise_epilogue=Self.elementwise_epilogue,
                    ](
                        output_base + wo * conv_shape.f,
                        input_base
                        + wo * conv_shape.stride_at[2]() * conv_shape.c,
                        filter,
                        first_c_tile_in_group,
                        c_tile_size,
                        f_tile_offset,
                        f_tile_size,
                        rebind[ConvShape[3]](conv_shape),
                        n,
                        Index(do, ho, wo),
                    )

                tile_middle_unswitch_boundaries[
                    [micro_kernel_height, 5, 4, 3, 2, 1],
                ](
                    0,
                    left_pad_impact_end,
                    right_pad_impact_start,
                    self.conv_shape.wo(),
                    work_fn,
                )

    def _f_tile_loop_static[
        last_c_tile: Bool
    ](self, n: Int, c_tile_offset: Int, c_tile_size: Int):
        comptime assert Self.conv_attr_rank == Self.input_layout.rank() - 2
        comptime WO = Int(Self.output_layout.shape[2])  # NHWC
        comptime F = Int(Self.output_layout.shape[3])  # NHWC
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_shape = get_micro_kernel_shape[
            Self.conv_attr_rank, WO, F, Self.conv_attr, simd_size
        ]()
        comptime micro_kernel_f_size = micro_kernel_shape[1] * simd_size

        var f_round_by_simd = (
            (self.partition.f_offset + self.partition.f_size) // simd_size
        ) * simd_size

        @always_inline
        def f_tile_iteration[
            size: Int
        ](f_tile_offset: Int, f_tile_size: Int) {imm}:
            self._h_loop_static[
                micro_kernel_shape[0],
                size // simd_size,
                False,
                last_c_tile,
            ](n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size)

        tile[
            [micro_kernel_f_size, simd_size],
            simd_size,
        ](
            self.partition.f_offset,
            f_round_by_simd,
            micro_kernel_f_size,
            simd_size,
            primary_cleanup_tile=simd_size,
            workgroup_function=f_tile_iteration,
        )

        var residual = F - f_round_by_simd
        if (
            self.partition.f_offset + self.partition.f_size == F
            and residual > 0
        ):
            self._h_loop_static[
                micro_kernel_shape[0],
                1,
                True,
                last_c_tile,
            ](n, f_round_by_simd, simd_size, c_tile_offset, c_tile_size)

    @always_inline
    def _h_loop_static[
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
        """Loop over H dimension
        Each row is divied into three parts: (1) effected by left padding, (2)
        not effected by padding, (3) effected by right padding. Use pointwise
        micro kernel 1 x micro_kernel_width for (1) and (3) and exploits the
        default micro kernel for (2).
        """
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        comptime H = Int(Self.input_layout.shape[1])  # NHWC
        comptime W = Int(Self.input_layout.shape[2])  # NHWC
        comptime C = Int(Self.input_layout.shape[3])  # NHWC
        comptime R = Int(Self.filter_layout.shape[1])  # FRSCf
        comptime S = Int(Self.filter_layout.shape[2])  # FRSCf
        comptime HO = Int(Self.output_layout.shape[1])  # NHWC
        comptime WO = Int(Self.output_layout.shape[2])  # NHWC
        comptime F = Int(Self.output_layout.shape[3])  # NHWC

        var filter_base: UnsafePointer[
            Scalar[Self.filter_type], type_of(self.filter.ptr).origin
        ]

        comptime if Self.filter_packed:
            filter_base = self.filter.ptr + (
                f_tile_offset * C * R * S + c_tile_offset * micro_kernel_f_size
            )
        else:
            filter_base = self.filter.ptr + (c_tile_offset * F + f_tile_offset)

        var input_curr_image = self.input.ptr + n * W * H * C
        var output_curr_image = self.output.ptr + n * WO * HO * F
        var conv_attr_dyn = materialize[Self.conv_attr]()

        for ho in range(
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        ):
            var h = ho * conv_attr_dyn.strides()[0] - conv_attr_dyn.pad_bottom()
            # Point to (n, 0, ho, c_tile_offset) mapped in input
            var input_base = input_curr_image + (
                c_tile_offset + C * (-conv_attr_dyn.pad_left() + W * h)
            )
            # Point to (n, 0, ho, f_tile_offset) mapped in input.
            # Safety: erase the origin to turn off the mutable aliasing
            # pointer check; `update_middle` captures both this pointer and
            # `self`, which embeds the same output origin.
            var output_base = (
                output_curr_image + (f_tile_offset + F * WO * ho)
            ).unsafe_origin_cast[AnyOrigin[mut=True]]()

            # The entire row fits in one micro kernel.
            comptime if WO <= micro_kernel_height:
                self._inner_loops_static[
                    WO,
                    micro_kernel_width,
                    True,
                    True,
                    has_residual,
                    last_c_tile,
                ](
                    input_base,
                    filter_base,
                    output_base,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    n,
                    ho,
                    0,  # wo
                )
            # The row is split into multiple micro kernels.
            else:
                # micro kernel height for left and right boundaries.
                # IF WO is just 1-2 points more than micro kernel height, the
                # following would divide the row evely by two micro kernels.
                comptime micro_kernel_height_lbound = min(
                    micro_kernel_height, WO // 2
                )
                comptime micro_kernel_height_rbound = min(
                    micro_kernel_height, WO - WO // 2
                )
                # Left boundary
                self._inner_loops_static[
                    micro_kernel_height_lbound,
                    micro_kernel_width,
                    True,
                    False,
                    has_residual,
                    last_c_tile,
                ](
                    input_base,
                    filter_base,
                    output_base,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    n,
                    ho,
                    0,  # beginning of wo dimension
                )

                # Update middle points if any. They aren't effected by padding.
                # The bases can't be captured mutably, so derive the per-tile
                # pointers from wo instead of incrementing across calls.
                # `var self` in a capture list crashes the parser (MRValue
                # assert in IRValues.cpp), so capture a copy under another
                # name. ConvInfoStatic is not implicitly copyable, so capture
                # the stride value instead of the attrs.
                var conv = self
                var stride_w = conv_attr_dyn.strides()[1]

                @always_inline
                def update_middle[
                    height: Int
                ](wo: Int) {
                    var conv,
                    var input_base,
                    var filter_base,
                    var output_base,
                    var stride_w,
                    var f_tile_offset,
                    var f_tile_size,
                    var c_tile_offset,
                    var c_tile_size,
                    var n,
                    var ho,
                }:
                    conv._inner_loops_static[
                        height,
                        micro_kernel_width,
                        False,
                        False,
                        has_residual,
                        last_c_tile,
                    ](
                        input_base + wo * stride_w * C,
                        filter_base,
                        output_base + wo * F,
                        f_tile_offset,
                        f_tile_size,
                        c_tile_offset,
                        c_tile_size,
                        n,
                        ho,
                        wo,
                    )

                # Middle points are the points not updated by micro kernels
                # on left or right boundary
                comptime num_middle_points = WO - micro_kernel_height_lbound - micro_kernel_height_rbound
                # `tile` can't handle zero tile size.
                comptime micro_kernel_height_middle = num_middle_points % micro_kernel_height if num_middle_points % micro_kernel_height > 0 else 1
                tile[[micro_kernel_height, micro_kernel_height_middle],](
                    micro_kernel_height_lbound,
                    WO - micro_kernel_height_rbound,
                    update_middle,
                )

                # Right boundary.
                self._inner_loops_static[
                    micro_kernel_height_rbound,
                    micro_kernel_width,
                    False,
                    True,
                    has_residual,
                    last_c_tile,
                ](
                    input_base
                    + (WO - micro_kernel_height_rbound)
                    * conv_attr_dyn.strides()[1]
                    * C,
                    filter_base,
                    output_base + (WO - micro_kernel_height_rbound) * F,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    n,
                    ho,
                    WO - micro_kernel_height_rbound,  # offset in wo dimension
                )

    @always_inline
    def _inner_loops_static[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        padded_left: Bool,
        padded_right: Bool,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        input_base: UnsafePointer[
            mut=False, Scalar[Self.input_type], ...
        ],  # points to (ho, wo) mapped in input
        filter_base: UnsafePointer[
            mut=False, Scalar[Self.filter_type], ...
        ],  # point to filter in cf tile
        output_base: UnsafePointer[
            mut=True, Scalar[Self.output_type], ...
        ],  # point to (ho, wo) in output
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
        n: Int,  # batch Index
        ho: Int,  # index in output height
        wo: Int,  # index in output width
    ):
        comptime if micro_kernel_height == 0:
            return

        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        comptime R = Int(Self.filter_layout.shape[1])  # FRSCf
        comptime S = Int(Self.filter_layout.shape[2])  # FRSCf
        comptime C = Int(Self.input_layout.shape[3])  # NHWC
        comptime s_stride_in_input = Self.conv_attr.dilations()[1] * C
        comptime wo_stride_in_input = Self.conv_attr.strides()[1] * C
        comptime filter_S_stride = C * micro_kernel_f_size
        comptime filter_F_stride = R * S * filter_S_stride

        comptime output_tile_layout = Layout.row_major(
            micro_kernel_height, micro_kernel_width * simd_size
        )
        var output_tile_stack = Array[
            Scalar[Self.output_type], output_tile_layout.size()
        ](uninitialized=True)
        var output_micro_tile = LayoutTensor[
            Self.output_type,
            output_tile_layout,
        ](output_tile_stack)

        # Initialize micro tile with 0 for its first use
        if self.is_new_c_accum(c_tile_offset):
            self._init_output_micro_tile[
                micro_kernel_height, micro_kernel_width, simd_size
            ](output_micro_tile)
        # Load micro tile from output buffer.
        else:
            self._load_output_micro_tile[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                has_residual,
            ](output_base, output_micro_tile)

        var acc = _Accumulator[
            Self.output_type, micro_kernel_height, micro_kernel_width, simd_size
        ]()
        acc.load(output_micro_tile.ptr, micro_kernel_width * simd_size)

        comptime W = Int(Self.input_layout.shape[2])  # NHWC
        comptime H = Int(Self.input_layout.shape[1])  # NHWC
        comptime WO = Int(Self.output_layout.shape[2])  # NHWC
        # Shift in input H when shifting 1 in filter stencil' R dimension.
        var h_shift = 0
        var conv_attr_dyn = materialize[Self.conv_attr]()
        # h index in input image
        var h = ho * conv_attr_dyn.strides()[0] - conv_attr_dyn.pad_bottom()
        for r in range(R):
            # Skip if row falls in padding.
            if h + h_shift < 0 or h + h_shift >= H:
                h_shift += conv_attr_dyn.dilations()[0]
                continue

            var input_ptr = input_base + h_shift * C * W
            var filter_ptr = filter_base + r * S * filter_S_stride

            comptime for s in range(S):
                # Adjustment of micro kernel height for left padding
                # The first left_adjust x micro_kernel_width registers are
                # ignored because they fall in padding.
                comptime left_adjust = max(
                    ceildiv(
                        Self.conv_attr.pad_left()
                        - s * Self.conv_attr.dilations()[1],
                        Self.conv_attr.strides()[1],
                    ),
                    0,
                ) if padded_left else 0
                # Adjustment of micro kernel height for right padding
                # The last left_adjust x micro_kernel_width registers are ignored.
                # fmt: off
                comptime right_adjust = max(
                    WO - 1 - (W - 1 + Self.conv_attr.pad_left() - s * Self.conv_attr.dilations()[1])
                             // Self.conv_attr.strides()[1],
                    0,
                ) if padded_right else 0
                # fmt: on

                # Revised calculation of tile_height to avoid cases of tile_height<=0.
                comptime tile_height = micro_kernel_height - left_adjust - right_adjust

                comptime if tile_height > 0:
                    self._accumulate[
                        micro_kernel_height,
                        micro_kernel_width,
                        simd_size,
                        has_residual,
                        # prefetch offset, default to 4 for now
                        4,
                        left_adjust,
                        left_adjust + tile_height,
                    ](
                        c_tile_size,
                        wo_stride_in_input,
                        input_ptr,
                        filter_ptr,
                        acc,
                    )

                filter_ptr = filter_ptr + filter_S_stride
                input_ptr = input_ptr + s_stride_in_input

            h_shift += conv_attr_dyn.dilations()[0]

        acc.store(output_micro_tile.ptr, micro_kernel_width * simd_size)
        # Store the micro tile
        self._store_output_micro_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            has_residual,
        ](output_micro_tile, output_base)

        # Apply elmentwise epilogue to the
        comptime F = Int(Self.output_layout.shape[3])  # NHWC

        comptime if Self.elementwise_epilogue.__bool__() and last_c_tile.__bool__():
            comptime epilogue = Self.elementwise_epilogue.value()
            # If has residual, the tile size has been extended to a simd_size.
            # Here needs to use the real bound F.
            var f_tile_size_bounded = (
                F - f_tile_offset if has_residual else f_tile_size
            )
            for wo_idx in range(wo, wo + micro_kernel_height):
                epilogue(
                    Index(n, ho, wo_idx, f_tile_offset), f_tile_size_bounded
                )

        return


# ===----------------------------------------------------------------------=== #
# Direct Convolution 1D Resigter Tiling
# ===----------------------------------------------------------------------=== #


@always_inline
def accumulate_wo_tile_1d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    partial_load_filter: Bool,
    effected_by_padding: Bool,
    input_dt: DType,
    filter_dt: DType,
](
    c_tile_size: Int,
    S: Int,
    mut acc: _Accumulator,
    input: UnsafePointer[mut=False, Scalar[input_dt], ...],
    input_stride: Int,
    input_stride_to_nbr: Int,
    filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
    filter_stride: Int,
    filter_stride_to_nbr: Int,
    partial_load_filter_size: Int,
    w: Int,
    W: Int,
    dilation: Int,
):
    """Update one row in the output for a given (c, f) tile.

    Parameters:
        micro_kernel_height: Number of input points in register tiling.
        micro_kernel_width: Number of SIMD resgiters assigned to F.
        simd_size: Number of elements in a SIMD register.
        partial_load_filter: Whether using partial load for filter.
        effected_by_padding: Whether the tile is effected by padding.
        input_dt: DType of input.
        filter_dt: DType of filter.

    Args:
        c_tile_size: Tile size in input channel.
        S: Filter window width.
        acc: Pointer to register tile accumulator.
        input: Pointer to the first input point in WO tile.
        input_stride: Stride between two input points, i.e., C w/ NHWC layout.
        input_stride_to_nbr: Stride between an input point and its neighbor.
        filter: Pointer to the first coef in the filter window.
        filter_stride: Stride between two segments of size `micro_kernel_width * simd_size`.
        filter_stride_to_nbr: Stride between between two neighbor coefs, i.e.,
            CF w/ RSCF layout.
        partial_load_filter_size: Size of partial load for filter.
        w: Coordinate in an input row.
        W: Input width.
        dilation: Convolution dilation.
    """

    for s in range(S):
        # Offset in the input row.

        var input_ptr = input + s * input_stride_to_nbr
        var filter_ptr = filter + s * filter_stride_to_nbr

        # When effected by padding, we update 1 output point a time.
        # Skip this point's neighbor if it's in padding.
        comptime if effected_by_padding:
            comptime assert (
                micro_kernel_height == 1
            ), "The tile must only have 1 point when effected bypadding."
            var w_nbr = w + s * dilation
            if w_nbr < 0 or w_nbr >= W:
                continue

        # Accumulat in output registers.
        acc.accumulate[prefetch_offset=4, partial_load_b=partial_load_filter](
            c_tile_size,
            input_ptr,
            input_stride,
            filter_ptr,
            filter_stride,
            partial_load_filter_size,
        )


def conv1d_update_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    filter_packed: Bool,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](
    output: UnsafePointer[mut=True, Scalar[output_dt], ...],
    input: UnsafePointer[mut=False, Scalar[input_dt], ...],
    filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
    first_c_tile: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape,
    n: Int,
    wo: Int,
):
    """Updates one micro tile of the 1D convolution output for a given
    (c, f) tile, accumulating over the S filter window and optionally
    applying an elementwise epilogue on the last C tile.

    Parameters:
        micro_kernel_height: Number of output points along WO covered by
            the micro tile in register tiling.
        micro_kernel_width: Number of SIMD registers assigned to the F
            dimension per output point.
        simd_size: Number of elements in a SIMD register.
        filter_packed: True when the filter is in packed `FSCf` layout,
            False for `SCF` layout.
        effected_by_padding: True when the tile may touch padded input
            regions, requiring per-point bounds checks.
        has_residual: True when F is not a multiple of `simd_size`,
            requiring partial load and store of the trailing SIMD vector.
        last_c_tile: True when this is the last C tile in the group,
            triggering the elementwise epilogue.
        output_dt: Element type of the output tensor.
        input_dt: Element type of the input tensor.
        filter_dt: Element type of the filter tensor.
        elementwise_epilogue: Optional elementwise function applied to
            the output on the last C tile (defaults to `None`).

    Args:
        output: Pointer to the first element of the WO micro tile in the
            output tensor.
        input: Pointer to the first input element of the WO tile.
        filter: Pointer to the first filter coefficient in the filter
            window.
        first_c_tile: True when this is the first C tile in the group,
            initializing the accumulator to zero instead of loading.
        c_tile_size: Number of input channels in the current C tile.
        f_tile_offset: Offset of the F tile within the current group.
        f_tile_size: Number of output channels in the F tile.
        conv_shape: Convolution shape descriptor carrying spatial
            extents, padding, stride, dilation, and channel counts.
        n: Batch index of the input image being convolved.
        wo: Starting output width index of the micro tile.
    """
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Input stride when s increments by 1
    var input_stride_by_s = conv_shape.dilation_at[0]() * conv_shape.c

    # Filter stride when s increments by 1.
    var filter_stride_by_s: Int

    comptime if filter_packed:  # FSCf layout
        filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    else:  # SCF layout
        filter_stride_by_s = conv_shape.c * conv_shape.f

    # Filter stride in F dimension in FRSCf
    var filter_stride = micro_kernel_f_size if filter_packed else conv_shape.f

    # Input coordinates
    var w = wo * conv_shape.stride_at[0]() - conv_shape.pad_w_lower()

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.
    var acc = _Accumulator[
        output_dt, micro_kernel_height, micro_kernel_width, simd_size
    ]()

    if first_c_tile:
        acc.init(0)
    else:
        acc.load[partial_load=has_residual](
            output,
            conv_shape.f,
            conv_shape.f_per_group() % simd_size,
        )

    accumulate_wo_tile_1d[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        has_residual and not filter_packed,
        effected_by_padding,
    ](
        c_tile_size,
        conv_shape.s(),
        acc,
        input,
        conv_shape.c * conv_shape.stride_at[0](),
        input_stride_by_s,
        filter,
        filter_stride,
        filter_stride_by_s,
        conv_shape.f % simd_size,
        w,
        conv_shape.w(),
        conv_shape.dilation_at[0](),
    )

    # Store the micro tile
    acc.store[partial_store=has_residual](
        output,
        conv_shape.f,
        conv_shape.f_per_group() % simd_size,
    )

    # Apply elementwise epilogue if necessary
    comptime if elementwise_epilogue.__bool__() and last_c_tile.__bool__():
        comptime epilogue = elementwise_epilogue.value()
        # If has residual, the tile size has been extended to a simd_size.
        # Here needs to use the real bound F.
        var f_tile_size_bounded: Int

        comptime if has_residual:
            f_tile_size_bounded = (
                conv_shape.f_per_group() - conv_shape.f_in_group(f_tile_offset)
            )
        else:
            f_tile_size_bounded = f_tile_size

        for wo_idx in range(wo, wo + micro_kernel_height):
            epilogue(Index(n, wo_idx, f_tile_offset), f_tile_size_bounded)


# ===----------------------------------------------------------------------=== #
# Direct Convolution 2D Register Tiling
# ===----------------------------------------------------------------------=== #


@always_inline
def accumulate_wo_tile_2d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    partial_load_filter: Bool,
    effected_by_padding: Bool,
    input_dt: DType,
    filter_dt: DType,
](
    c_tile_size: Int,
    RS: IndexList[2],
    mut acc: _Accumulator,
    input: UnsafePointer[mut=False, Scalar[input_dt], ...],
    input_stride: Int,
    input_stride_to_nbr: IndexList[2],
    filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
    filter_stride: Int,
    filter_stride_to_nbr: IndexList[2],
    partial_load_filter_size: Int,
    hw: IndexList[2],
    HW: IndexList[2],
    dilation: IndexList[2],
):
    """Accumulates one output row tile for a 2D convolution by iterating over the
    R and S filter-window dimensions and delegating each row to
    `accumulate_wo_tile_1d`.

    Parameters:
        micro_kernel_height: Number of input rows covered by the micro tile
            in register tiling.
        micro_kernel_width: Number of SIMD registers assigned to the F
            dimension per row.
        simd_size: Number of elements in a SIMD register.
        partial_load_filter: True when the final filter segment is smaller
            than a full SIMD vector and must be partially loaded.
        effected_by_padding: True when the tile may touch padded input
            regions, requiring per-point bounds checks.
        input_dt: Element type of the input tensor.
        filter_dt: Element type of the filter tensor.

    Args:
        c_tile_size: Number of input channels in the current C tile.
        RS: Filter window extents as `(R, S)` with R the height and S the
            width.
        acc: Register-tile accumulator updated in place with the
            convolution products.
        input: Pointer to the first input element of the WO tile.
        input_stride: Stride between consecutive output points along WO in
            the input, equal to `C * stride_w` in NHWC layout.
        input_stride_to_nbr: Strides to the input neighbor for each spatial
            axis `(R, S)`, i.e. `(stride_to_R_neighbor, stride_to_S_neighbor)`.
        filter: Pointer to the first filter coefficient in the filter
            window.
        filter_stride: Stride between consecutive filter segments of size
            `micro_kernel_width * simd_size` along the F dimension.
        filter_stride_to_nbr: Strides to the filter neighbor for each
            spatial axis `(R, S)`.
        partial_load_filter_size: Number of valid elements in the final
            partial filter SIMD vector when F is not a multiple of `simd_size`.
        hw: Input spatial coordinate `(h, w)` of the tile's first output
            point before padding adjustment.
        HW: Input spatial extents `(H, W)` used for padding bounds checks.
        dilation: Dilation factors `(dilation_h, dilation_w)` applied to
            the filter window.
    """
    for r in range(RS[0]):
        # Skip the row if it falls into padding.
        var h_nbr = hw[0] + r * dilation[0]
        if h_nbr < 0 or h_nbr >= HW[0]:
            continue

        var input_ptr = input + r * input_stride_to_nbr[0]
        var filter_ptr = filter + r * filter_stride_to_nbr[0]

        accumulate_wo_tile_1d[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            partial_load_filter,
            effected_by_padding,
        ](
            c_tile_size,
            RS[1],
            acc,
            input_ptr,
            input_stride,
            input_stride_to_nbr[1],
            filter_ptr,
            filter_stride,
            filter_stride_to_nbr[1],
            partial_load_filter_size,
            hw[1],
            HW[1],
            dilation[1],
        )


def conv2d_update_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    filter_packed: Bool,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](
    output: UnsafePointer[mut=True, Scalar[output_dt], ...],
    input: UnsafePointer[mut=False, Scalar[input_dt], ...],
    filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
    first_c_tile: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[2],
    n: Int,
    howo: IndexList[2],
):
    """Updates one micro tile of the 2D convolution output for a given
    (c, f) tile, accumulating over the R x S filter window and optionally
    applying an elementwise epilogue on the last C tile.

    Parameters:
        micro_kernel_height: Number of output points along the WO
            dimension covered by the micro tile in register tiling.
        micro_kernel_width: Number of SIMD registers assigned to the F
            dimension per output point.
        simd_size: Number of elements in a SIMD register.
        filter_packed: True when the filter is prepacked in `FRSCf`
            layout for grouped convolution.
        effected_by_padding: True when the tile may touch padded input
            regions, requiring per-point bounds checks.
        has_residual: True when F is not a multiple of `simd_size`. The
            residual elements are loaded and padded with zero to fit
            a simd vector.
        last_c_tile: True when this is the last C tile, enabling the
            elementwise epilogue after accumulation.
        output_dt: Element type of the output tensor.
        input_dt: Element type of the input tensor.
        filter_dt: Element type of the filter tensor.
        elementwise_epilogue: Optional elementwise function applied to
            the output after the last channel tile (defaults to
            `None`).

    Args:
        output: Pointer to the start of the output micro tile at
            `(n, howo[0], howo[1], f_tile_offset)`.
        input: Pointer to the first input element of the micro tile
            before padding adjustment.
        filter: Pointer to the first filter coefficient in the filter
            window for the current `(c, f)` tile.
        first_c_tile: True when this is the first C tile,
            zero-initializing the accumulator instead of loading from
            the output.
        c_tile_size: Number of input channels in the current C tile.
        f_tile_offset: Offset of the current tile along the F (output
            channel) dimension.
        f_tile_size: Number of output channels in the current F tile.
        conv_shape: Convolution dimension description for the 2D
            convolution.
        n: Batch index of the current input image.
        howo: Output spatial coordinates `(ho, wo)` of the tile's
            first output point, with the micro tile spanning
            `micro_kernel_height` consecutive `wo` values starting at
            `howo[1]`.
    """
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Input stride to neighbor point in the filter window (R, S).
    var input_stride_by_s = conv_shape.dilation_at[1]() * conv_shape.c
    var input_stride_by_r = (
        conv_shape.dilation_at[0]() * conv_shape.w() * conv_shape.c
    )

    # Filter stride when s increments by 1.
    var filter_stride_by_s: Int

    comptime if filter_packed:  # FRSCf layout
        filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    else:  # RSCF layout
        filter_stride_by_s = conv_shape.c * conv_shape.f

    var filter_stride_by_r = conv_shape.s() * filter_stride_by_s

    # Filter stride in F dimension in FRSCf
    var filter_stride = micro_kernel_f_size if filter_packed else conv_shape.f

    # Input coordinates
    var hw = Index(
        howo[0] * conv_shape.stride_at[0]() - conv_shape.pad_h_lower(),
        howo[1] * conv_shape.stride_at[1]() - conv_shape.pad_w_lower(),
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.
    var acc = _Accumulator[
        output_dt, micro_kernel_height, micro_kernel_width, simd_size
    ]()

    if first_c_tile:
        acc.init(0)
    else:
        acc.load[partial_load=has_residual](
            output,
            conv_shape.f,
            conv_shape.f_per_group() % simd_size,
        )

    accumulate_wo_tile_2d[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        has_residual and not filter_packed,
        effected_by_padding,
    ](
        c_tile_size,
        Index(conv_shape.r(), conv_shape.s()),
        acc,
        input,
        conv_shape.c * conv_shape.stride_at[1](),
        Index(input_stride_by_r, input_stride_by_s),
        filter,
        filter_stride,
        Index(filter_stride_by_r, filter_stride_by_s),
        conv_shape.f % simd_size,
        hw,
        Index(conv_shape.h(), conv_shape.w()),
        coord_to_index_list(conv_shape.dilation),
    )

    # Store the micro tile
    acc.store[partial_store=has_residual](
        output,
        conv_shape.f,
        conv_shape.f_per_group() % simd_size,
    )

    # Apply elmentwise epilogue to the
    # if elementwise_epilogue_enabled and last_c_tile:
    comptime if elementwise_epilogue.__bool__() and last_c_tile.__bool__():
        comptime epilogue = elementwise_epilogue.value()

        # If has residual, the tile size has been extended to a simd_size.
        # Here needs to use the real bound F.
        var f_tile_size_bounded: Int

        comptime if has_residual:
            f_tile_size_bounded = (
                conv_shape.f_per_group() - conv_shape.f_in_group(f_tile_offset)
            )
        else:
            f_tile_size_bounded = f_tile_size

        for wo_idx in range(howo[1], howo[1] + micro_kernel_height):
            # elementwise_epilogue_fn[4](
            epilogue(
                Index(n, howo[0], wo_idx, f_tile_offset), f_tile_size_bounded
            )


# ===----------------------------------------------------------------------=== #
# Direct Convolution 3D Resigter Tiling
# ===----------------------------------------------------------------------=== #


# TODO: Simplify this with a rank parameter + recursion.
@always_inline
def accumulate_wo_tile_3d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    partial_load_filter: Bool,
    effected_by_padding: Bool,
    input_dt: DType,
    filter_dt: DType,
](
    c_tile_size: Int,
    QRS: IndexList[3],
    mut acc: _Accumulator,
    input: UnsafePointer[mut=False, Scalar[input_dt], ...],
    input_stride: Int,
    input_stride_to_nbr: IndexList[3],
    filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
    filter_stride: Int,
    filter_stride_to_nbr: IndexList[3],
    partial_load_filter_size: Int,
    dhw: IndexList[3],
    DHW: IndexList[3],
    dilation: IndexList[3],
):
    """Accumulates one output row tile for a 3D convolution by iterating over
    the Q filter-window depth dimension and delegating each depth slice to
    `accumulate_wo_tile_2d`.

    Parameters:
        micro_kernel_height: Number of input rows covered by the micro tile
            in register tiling.
        micro_kernel_width: Number of SIMD registers assigned to the F
            dimension per row.
        simd_size: Number of elements in a SIMD register.
        partial_load_filter: True when the final filter segment is smaller
            than a full SIMD vector and must be partially loaded.
        effected_by_padding: True when the tile may touch padded input
            regions, requiring per-point bounds checks.
        input_dt: Element type of the input tensor.
        filter_dt: Element type of the filter tensor.

    Args:
        c_tile_size: Number of input channels in the current C tile.
        QRS: Filter window extents as `(Q, R, S)` with Q the depth, R the
            height, and S the width.
        acc: Register-tile accumulator updated in place with the
            convolution products.
        input: Pointer to the first input element of the WO tile.
        input_stride: Stride between consecutive output points along WO in
            the input, equal to `C * stride_w` in NDHWC layout.
        input_stride_to_nbr: Strides to the input neighbor for each spatial
            axis `(Q, R, S)`.
        filter: Pointer to the first filter coefficient in the filter
            window.
        filter_stride: Stride between consecutive filter segments of size
            `micro_kernel_width * simd_size` along the F dimension.
        filter_stride_to_nbr: Strides to the filter neighbor for each
            spatial axis `(Q, R, S)`.
        partial_load_filter_size: Number of valid elements in the final
            partial filter SIMD vector when F is not a multiple of
            `simd_size`.
        dhw: Input spatial coordinate `(d, h, w)` of the tile's first
            output point before padding adjustment.
        DHW: Input spatial extents `(D, H, W)` used for padding bounds
            checks.
        dilation: Dilation factors `(dilation_d, dilation_h, dilation_w)`
            applied to the filter window.
    """
    for q in range(QRS[0]):
        var d_nbr = dhw[0] + q * dilation[0]
        if d_nbr < 0 or d_nbr >= DHW[0]:
            continue

        var input_ptr = input + q * input_stride_to_nbr[0]
        var filter_ptr = filter + q * filter_stride_to_nbr[0]

        accumulate_wo_tile_2d[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            partial_load_filter,
            effected_by_padding,
        ](
            c_tile_size,
            Index(QRS[1], QRS[2]),
            acc,
            input_ptr,
            input_stride,
            Index(input_stride_to_nbr[1], input_stride_to_nbr[2]),
            filter_ptr,
            filter_stride,
            Index(filter_stride_to_nbr[1], filter_stride_to_nbr[2]),
            partial_load_filter_size,
            Index(dhw[1], dhw[2]),
            Index(DHW[1], DHW[2]),
            Index(dilation[1], dilation[2]),
        )


def conv3d_update_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    filter_packed: Bool,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](
    output: UnsafePointer[mut=True, Scalar[output_dt], ...],
    input: UnsafePointer[mut=False, Scalar[input_dt], ...],
    filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
    first_c_tile: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[3],
    n: Int,
    dohowo: IndexList[3],
):
    """Updates one micro tile of the 3D convolution output for a given
    (c, f) tile, accumulating over the Q x R x S filter window and
    optionally applying an elementwise epilogue on the last C tile.

    Parameters:
        micro_kernel_height: Number of output WO positions processed
            per micro tile along the WO dimension.
        micro_kernel_width: Number of SIMD vectors along the F
            dimension per micro tile.
        simd_size: Width of a SIMD vector in elements.
        filter_packed: True when the filter uses the packed `FRSCf`
            layout.
        effected_by_padding: True when the WO positions in this tile
            fall within the padding-impacted boundary region.
        has_residual: True when F per group is not a multiple of
            `simd_size`, requiring partial SIMD load and store.
        last_c_tile: True when this is the last tile along the C
            dimension, triggering the elementwise epilogue.
        output_dt: Element type of the output tensor (inferred).
        input_dt: Element type of the input tensor (inferred).
        filter_dt: Element type of the filter tensor (inferred).
        elementwise_epilogue: Optional elementwise function applied to
            the output on the last C tile (defaults to `None`).

    Args:
        output: Pointer to the start of the output micro tile.
        input: Pointer to the input data for the current sample at
            the current C tile offset.
        filter: Pointer to the filter data at the current (c, f) tile
            offset.
        first_c_tile: True when this is the first C tile in the
            group, zero-initializing the accumulator.
        c_tile_size: Number of input channels accumulated in this C
            tile.
        f_tile_offset: Offset of this tile along the F dimension.
        f_tile_size: Size of this tile along the F dimension.
        conv_shape: Statically known 3D convolution shape
            descriptor.
        n: Batch index of the current sample.
        dohowo: Output spatial coordinates `(do, ho, wo)` of the
            micro tile origin.
    """
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Input stride to neighbor point in the filter window (Q, R, S).
    # fmt: off
    var input_stride_by_s = conv_shape.dilation_at[2]() * conv_shape.c
    var input_stride_by_r = conv_shape.dilation_at[1]() * conv_shape.w() * conv_shape.c
    var input_stride_by_q = conv_shape.dilation_at[0]() * conv_shape.w() * conv_shape.h() * conv_shape.c
    # fmt: on

    # Filter stride when s increments by 1.
    var filter_stride_by_s: Int

    comptime if filter_packed:  # FRSCf layout
        filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    else:  # RSCF layout
        filter_stride_by_s = conv_shape.c * conv_shape.f

    var filter_stride_by_r = conv_shape.s() * filter_stride_by_s
    var filter_stride_by_q = conv_shape.r() * filter_stride_by_r

    # Filter stride in F dimension in FRSCf
    var filter_stride = micro_kernel_f_size if filter_packed else conv_shape.f

    # Input coordinates
    var dhw = Index(
        dohowo[0] * conv_shape.stride_at[0]() - conv_shape.pad_d_lower(),
        dohowo[1] * conv_shape.stride_at[1]() - conv_shape.pad_h_lower(),
        dohowo[2] * conv_shape.stride_at[2]() - conv_shape.pad_w_lower(),
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.
    var acc = _Accumulator[
        output_dt, micro_kernel_height, micro_kernel_width, simd_size
    ]()

    if first_c_tile:
        acc.init(0)
    else:
        acc.load[partial_load=has_residual](
            output,
            conv_shape.f,
            conv_shape.f_per_group() % simd_size,
        )

    accumulate_wo_tile_3d[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        has_residual and not filter_packed,
        effected_by_padding,
    ](
        c_tile_size,
        coord_to_index_list(conv_shape.filter_dims),
        acc,
        input,
        conv_shape.c * conv_shape.stride_at[2](),
        Index(input_stride_by_q, input_stride_by_r, input_stride_by_s),
        filter,
        filter_stride,
        Index(filter_stride_by_q, filter_stride_by_r, filter_stride_by_s),
        conv_shape.f % simd_size,
        dhw,
        coord_to_index_list(conv_shape.input_dims),
        coord_to_index_list(conv_shape.dilation),
    )

    # Store the micro tile
    acc.store[partial_store=has_residual](
        output,
        conv_shape.f,
        conv_shape.f_per_group() % simd_size,
    )

    # Apply elmentwise epilogue to the
    comptime if elementwise_epilogue.__bool__() and last_c_tile.__bool__():
        comptime epilogue = elementwise_epilogue.value()

        # If has residual, the tile size has been extended to a simd_size.
        # Here needs to use the real bound F.
        var f_tile_size_bounded: Int

        comptime if has_residual:
            f_tile_size_bounded = (
                conv_shape.f_per_group() - conv_shape.f_in_group(f_tile_offset)
            )
        else:
            f_tile_size_bounded = f_tile_size

        for wo_idx in range(dohowo[2], dohowo[2] + micro_kernel_height):
            epilogue(
                Index(n, dohowo[0], dohowo[1], wo_idx, f_tile_offset),
                f_tile_size_bounded,
            )


# ===----------------------------------------------------------------------=== #
# Direct Convolution Filter Packing                                            #
# ===----------------------------------------------------------------------=== #


@always_inline
def pack_filter_shape_impl[
    filter_type: DType
](Q: Int, R: Int, S: Int, C: Int, F: Int, num_groups: Int) -> IndexList[6]:
    """
    Compute the shape of packed filter. The packed layout is FRSCf.
    shape_ref should be allocated with size 5 outside this kernel.

    Parameters:
        filter_type: Element type of the filter, used to determine the
            SIMD width for packing.

    Args:
        Q: Original Q filter dimension.
        R: Original R filter dimension.
        S: Original S filter dimension.
        C: Original C filter dimension.
        F: Original F filter dimension.
        num_groups: Number of groups in the convolution.

    Returns:
        The output shape.
    """
    comptime simd_size = simd_width_of[filter_type]()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    assert (
        F % num_groups == 0
    ), "number of filters F must be divisible by number of groups"
    var F_per_group = F // num_groups

    var output_shape = IndexList[6]()
    output_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    output_shape[1] = Q
    output_shape[2] = R
    output_shape[3] = S
    output_shape[4] = C
    output_shape[5] = micro_kernel_f_size

    return output_shape


@always_inline
def pack_conv_filter_shape(
    filter: TileTensor, num_groups: Int
) -> IndexList[filter.flat_rank + 1]:
    """
    Compute the output shape of convolution filter packing.

    Args:
        filter: The filter to be packed.
        num_groups: The number of groups in the convolution.

    Returns:
        The output shape.
    """

    comptime simd_size = simd_width_of[filter.dtype]()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Filter is in RSCF layout. The last dim is F no matter it's 1d, 2d, or 3d.
    var F = Int(filter.dim[filter.flat_rank - 1]())

    assert (
        F % num_groups == 0
    ), "number of filters F must be divisible by number of groups"
    var F_per_group = F // num_groups

    # FRSCf layout.
    var packed_shape = IndexList[filter.flat_rank + 1]()
    packed_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    packed_shape[filter.flat_rank] = micro_kernel_f_size

    comptime for i in range(filter.flat_rank - 1):
        packed_shape[i + 1] = Int(filter.dim[i]())

    return packed_shape


@always_inline
def pack_filter_shape[
    filter_type: DType,
    input_shape: IntTuple,
    filter_shape: IntTuple,
    output_shape: IntTuple,
    strides: IntTuple,
    dilations: IntTuple,
    paddings: IntTuple,
    num_groups: Int,
](filter: TileTensor) -> IndexList[filter.flat_rank + 1]:
    """
    Compute the shape of packed filter. The packed layout is FRSCf.
    shape_ref should be allocated with size 5 outside this kernel.

    Parameters:
        filter_type: Element type of the filter, used to determine the
            SIMD width for packing.
        input_shape: Shape of the convolution input tensor in NHWC
            layout.
        filter_shape: Shape of the filter tensor in RSCF or QRSCF layout.
        output_shape: Shape of the convolution output tensor in NHWC
            layout.
        strides: Stride along each spatial dimension of the convolution.
        dilations: Dilation factor along each spatial dimension of the
            convolution.
        paddings: Padding applied before and after each spatial dimension
            of the input.
        num_groups: Number of convolution groups for grouped convolution.

    Args:
        filter: The unpacked filter tensor in RSCF layout whose packed
            shape is computed.

    Returns:
        The output shape.
    """
    comptime simd_size = simd_width_of[filter_type]()

    var F = Int(filter.dim[filter.flat_rank - 1]())  # RSCF layout

    assert (
        F % num_groups == 0
    ), "number of filters F must be divisible by number of groups"
    var F_per_group = F // num_groups

    comptime conv_attr = ConvInfoStatic[filter.flat_rank - 2](
        pad=reorder_padding[filter.flat_rank - 2](IntTuple(paddings)),
        stride=IntTuple(strides),
        dilation=IntTuple(dilations),
        num_groups=num_groups,
    )

    # TODO: extend to 1D/3D.
    comptime WO = output_shape[
        2
    ].value() if filter.flat_rank == 4 and output_shape[
        2
    ].value() != UNKNOWN_VALUE else UNKNOWN_VALUE
    comptime F_NHWC = output_shape[
        filter.flat_rank - 1
    ].value() if output_shape[
        filter.flat_rank - 1
    ].value() != UNKNOWN_VALUE else UNKNOWN_VALUE
    comptime micro_kernel_shape = get_micro_kernel_shape[
        filter.flat_rank - 2,
        WO,
        F_NHWC,
        conv_attr,
        simd_size,
    ]()

    comptime micro_kernel_width = micro_kernel_shape[1]
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # FSCf/FRSCf/FQRSCf layout.
    var packed_shape = IndexList[filter.flat_rank + 1]()
    packed_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    packed_shape[filter.flat_rank] = micro_kernel_f_size

    comptime for i in range(filter.flat_rank - 1):
        packed_shape[i + 1] = Int(filter.dim[i]())

    return packed_shape


@always_inline
def _get_group_filter_base(
    packed_filter: LayoutTensor, group_idx: Int, f_per_group: Int
) -> UnsafePointer[
    Scalar[packed_filter.dtype],
    packed_filter.origin,
    address_space=packed_filter.address_space,
]:
    """Returns the pointer of the input group's start in the packed filter."""
    # Each group is zero padded to
    #     ceildiv(F_per_group, micro_kernel_width)
    #   * filter_window_size
    #   * C
    #   * micro_kernel_f_width
    # Output pointer points to the start of the current group.

    var micro_kernel_f_size = packed_filter.dim[packed_filter.rank - 1]()
    comptime rank = packed_filter.rank

    var filter_window_size = 1

    # The packed filter has layout e.x. FRSCf. The [1, rank-2) dims are filter
    # window sizes.
    comptime for i in range(rank - 3):
        filter_window_size *= packed_filter.dim[i + 1]()

    # Size of one group's packed filter.
    # fmt: off
    var group_size = ceildiv(f_per_group , micro_kernel_f_size) \
                   * filter_window_size * packed_filter.dim[rank-2]() \
                   * micro_kernel_f_size
    # fmt: on

    return packed_filter.ptr + group_idx * group_size


@always_inline
def pack_filter(
    filter: TileTensor,
    packed_filter: TileTensor[mut=True, ...],
    num_groups: Int,
):
    """This packs the filter form RSCF to FRSCf.
    Use the default micro kernel size for dynamic shapes.

    Args:
        filter: The unpacked filter tensor in RSCF layout.
        packed_filter: The destination tensor for the packed filter in
            FRSCf layout.
        num_groups: Number of convolution groups for grouped convolution.
    """

    comptime assert (
        filter.dtype == packed_filter.dtype
    ), "Type mismatch between the filter and the packed filter."

    # Bridge to LayoutTensor for legacy Layout shape access and fill().
    var filter_lt = filter.to_layout_tensor()
    var packed_filter_lt = packed_filter.to_layout_tensor()

    comptime simd_size = simd_width_of[filter.dtype]()
    comptime f_size_default = get_direct_conv_micro_kernel_width() * simd_size

    comptime if packed_filter_lt.layout.shape[
        packed_filter_lt.rank - 1
    ] != UNKNOWN_VALUE:
        comptime f_size = Int(
            packed_filter_lt.layout.shape[packed_filter_lt.rank - 1]
        )
        pack_filter_lt[simd_size, f_size](
            filter_lt, packed_filter_lt, num_groups
        )
    else:
        pack_filter_lt[simd_size, f_size_default](
            filter_lt, packed_filter_lt, num_groups
        )


@always_inline
def pack_filter_lt[
    simd_size: Int,
    micro_kernel_f_size: Int,  # 64
](
    filter: LayoutTensor,
    packed_filter: LayoutTensor[mut=True, ...],
    num_groups: Int,
):
    """This packs the filter form RSCF to FRSCf.

    Parameters:
        simd_size: Can differ from the simd size of the input type.
        micro_kernel_f_size: The size of the last dimension in FRSCf, which is
            equals the size of the micro kernel's F dimension.

    Args:
        filter: Filter in RSCF layout (if 2D).
        packed_filter: Packed filter in FRSCf layout (if 2D).
            F       - the index of continuous segments in micro kernel.
            R, S, C - original R, S, C.
            f       - the index within a continuous segments.
        num_groups: The number of groups in the convolution.

    F is first broken down to segments of size micro_kernel_f_size, then the
    remainder is further divided by simd_size. The last residual elements if
    any is padded with zero to fill simd_size.
    """

    # The micro kernel should be multiple of simd_size in F dimension.
    comptime assert micro_kernel_f_size % simd_size == 0

    # The input simd size should not exceed filter type's simd size.
    # E.x. we can pack int8 filter based on int32 simd size.
    comptime assert simd_size <= simd_width_of[filter.dtype]()

    # Product of filter dims upto (rank - 1).
    var outer_dims_prod = 1

    comptime for i in range(filter.rank - 1):
        outer_dims_prod *= filter.dim[i]()

    var F = filter.dim[filter.rank - 1]()
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

    # Referring to `packed_filter.dtype` inside `pack` would capture
    # `packed_filter` itself, which aliases the captured `group_start`.
    comptime packed_filter_dtype = packed_filter.dtype

    for g in range(num_groups):
        var group_start = _get_group_filter_base(packed_filter, g, F_per_group)

        # TODO(MOCO-4664): `var g` copy-captures the loop variable to work
        # around wrong debug-info scopes on implicit nested-scope captures.
        @always_inline
        def pack[
            f_tile_size: Int
        ](f_tile_start: Int) {
            var g, var group_start, var F_per_group, var F, imm
        }:
            var packed_filter_ptr = group_start + f_tile_start * outer_dims_prod

            for row in range(outer_dims_prod):
                var filter_ptr = (
                    filter.ptr + row * F + g * F_per_group + f_tile_start
                )

                comptime for i in range(f_tile_size // simd_size):
                    packed_filter_ptr.store(
                        i * simd_size,
                        filter_ptr.load[width=simd_size](i * simd_size).cast[
                            packed_filter_dtype
                        ](),
                    )

                packed_filter_ptr += f_tile_size

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
                group_start + F_round_by_simd * outer_dims_prod
            )

            for row in range(outer_dims_prod):
                var filter_ptr = (
                    filter.ptr + row * F + g * F_per_group + F_round_by_simd
                )

                # Load remainder elements and pad with zero to
                # to fill a simd vector.
                var filter_vec = partial_simd_load[simd_size](
                    filter_ptr, 0, residual, 0
                ).cast[packed_filter.dtype]()
                packed_filter_ptr.store(filter_vec)

                # Hence, packed filter is incremented by simd_size
                packed_filter_ptr = packed_filter_ptr + simd_size


@always_inline
def pack_filter_from_fcrs(
    filter: TileTensor,
    packed_filter: TileTensor[mut=True, ...],
    num_groups: Int,
):
    """This packs the filter from FCRS to FRSCf (2D) or FCQRS to FQRSCf (3D).

    The filter arrives with its actual dtype (e.g., float32). We transpose
    FCRS→RSCF in a temp buffer using the actual dtype, then create an int64-
    reinterpreted TileTensor for pack_filter (which expects int64).

    Args:
        filter: Filter TileTensor with actual element dtype in FCRS layout.
        packed_filter: Output TileTensor with int64 dtype (for pack_filter).
        num_groups: Number of groups in the convolution.
    """

    var filter_lt = filter.to_layout_tensor()
    var total_elems = filter_lt.size()

    # Allocate temporary buffer for RSCF-ordered data (actual dtype).
    var rscf_buf_alloc = alloc[Scalar[filter.dtype]](
        {count = total_elems}
    ).into_managed()
    var rscf_buf = rscf_buf_alloc.unsafe_ptr()

    # Transpose FCRS→RSCF or FCQRS→QRSCF and create a TileTensor for packing.
    comptime if filter_lt.rank == 4:
        # FCRS [F, C, R, S] → RSCF [R, S, C, F]
        var dim_F = filter_lt.dim[0]()
        var dim_C = filter_lt.dim[1]()
        var dim_R = filter_lt.dim[2]()
        var dim_S = filter_lt.dim[3]()
        for f in range(dim_F):
            for c in range(dim_C):
                for r in range(dim_R):
                    for s in range(dim_S):
                        var src = (
                            f * dim_C * dim_R * dim_S
                            + c * dim_R * dim_S
                            + r * dim_S
                            + s
                        )
                        var dst = (
                            r * dim_S * dim_C * dim_F
                            + s * dim_C * dim_F
                            + c * dim_F
                            + f
                        )
                        rscf_buf.store(dst, filter_lt.ptr.load(src))
        # Reinterpret as int64 for pack_filter (matches existing convention).
        var rscf_tile = TileTensor(
            rscf_buf.bitcast[Int64](),
            row_major((dim_R, dim_S, dim_C, dim_F)),
        )
        pack_filter(rscf_tile, packed_filter, num_groups)
    else:
        # FCQRS [F, C, Q, R, S] → QRSCF [Q, R, S, C, F]
        var dim_F = filter_lt.dim[0]()
        var dim_C = filter_lt.dim[1]()
        var dim_Q = filter_lt.dim[2]()
        var dim_R = filter_lt.dim[3]()
        var dim_S = filter_lt.dim[4]()
        for f in range(dim_F):
            for c in range(dim_C):
                for q in range(dim_Q):
                    for r in range(dim_R):
                        for s in range(dim_S):
                            var src = (
                                f * dim_C * dim_Q * dim_R * dim_S
                                + c * dim_Q * dim_R * dim_S
                                + q * dim_R * dim_S
                                + r * dim_S
                                + s
                            )
                            var dst = (
                                q * dim_R * dim_S * dim_C * dim_F
                                + r * dim_S * dim_C * dim_F
                                + s * dim_C * dim_F
                                + c * dim_F
                                + f
                            )
                            rscf_buf.store(dst, filter_lt.ptr.load(src))
        var rscf_tile = TileTensor(
            rscf_buf.bitcast[Int64](),
            row_major(
                (
                    dim_Q,
                    dim_R,
                    dim_S,
                    dim_C,
                    dim_F,
                )
            ),
        )
        pack_filter(rscf_tile, packed_filter, num_groups)

    dealloc(rscf_buf_alloc^)


@always_inline
def conv_shape[
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
](
    input_buf: TileTensor[mut=False, input_type, address_space=.GENERIC, ...],
    filter_buf: TileTensor[mut=False, filter_type, address_space=.GENERIC, ...],
    strides_buf: TileTensor[
        mut=False, strides_type, address_space=.GENERIC, ...
    ],
    dilations_buf: TileTensor[
        mut=False, dilations_type, address_space=.GENERIC, ...
    ],
    paddings_buf: TileTensor[
        mut=False, paddings_type, address_space=.GENERIC, ...
    ],
    num_groups_scalar: Scalar,
) raises -> IndexList[input_buf.flat_rank]:
    """
    Compute the output shape of a `conv` operation, and assert the inputs are
    compatible.

    Parameters:
        input_type: Type of the input tensor.
        filter_type: Type of the filter tensor.
        strides_type: Type of the strides tensor.
        dilations_type: Type of the dilations tensor.
        paddings_type: Type of the paddings tensor.

    Args:
        input_buf: The input tensor.
        filter_buf: The filter tensor.
        strides_buf: The strides tensor.
        dilations_buf: The dilations tensor.
        paddings_buf: The paddings tensor.
        num_groups_scalar: The num_groups scalar.

    Returns:
        The output shape.
    """
    # Bridge to LayoutTensor for runtime dim access.
    var input_lt = input_buf.to_layout_tensor()
    var filter_lt = filter_buf.to_layout_tensor()
    var strides_lt = strides_buf.to_layout_tensor()
    var dilations_lt = dilations_buf.to_layout_tensor()
    var paddings_lt = paddings_buf.to_layout_tensor()

    comptime assert strides_buf.flat_rank == 1
    comptime assert dilations_buf.flat_rank == 1
    comptime assert paddings_buf.flat_rank == 1

    if input_lt.rank < 3:
        raise Error("[convolution] requires (input_rank >= 3)")
    if input_lt.rank != filter_lt.rank:
        raise Error("[convolution] requires (input_rank == filter_rank)")
    if (
        strides_lt.dim(0) != input_lt.rank - 2
        or dilations_lt.dim(0) != input_lt.rank - 2
    ):
        raise Error(
            "[convolution] requires (len(strides) == len(dilations) =="
            " input_rank - 2)"
        )
    if paddings_lt.dim(0) != 2 * (input_lt.rank - 2):
        raise Error(
            "[convolution] requires (len(paddings) == 2 * (input rank - 2))"
        )

    # Assume
    # - input and output have layout [batch_size, ...spatial_dims..., input_channels]
    # - filter has layout [...spatial_dims..., filter_channels, output_channels]
    var batch_size = input_lt.dim(0)
    var input_channels = input_lt.dim(input_lt.rank - 1)
    var filter_channels = filter_lt.dim(input_lt.rank - 2)
    var output_channels = filter_lt.dim(input_lt.rank - 1)
    var num_groups = Int(num_groups_scalar)

    if input_channels != (num_groups * filter_channels):
        raise Error(
            "[convolution] requires (input_channels == num_groups *"
            " filter_channels)"
        )
    if (output_channels % num_groups) != 0:
        raise Error(
            "[convolution] output_channels must be divisible by num_groups"
        )

    var output_shape = IndexList[input_lt.rank]()
    output_shape[0] = batch_size
    output_shape[input_lt.rank - 1] = output_channels

    comptime for i in range(1, input_lt.rank - 1):
        var input_spatial_dim = input_lt.dim(i)
        var filter_spatial_dim = filter_lt.dim(i - 1)

        # Zero input spatial -> zero output spatial. Strided convs over a
        # zero-spatial input would otherwise compute a negative
        # ``output_spatial_dim`` (e.g. ``1 + (0 + 0 - 3) // 2 = -1`` for a
        # 3x3 stride=2 pad=0 downsample) and trip the positivity check
        # below; short-circuit here so the encoder can run unconditionally
        # on an empty placeholder image.
        if input_spatial_dim == 0:
            output_shape[i] = 0
            continue

        var output_spatial_dim = get_sliding_window_out_dim(
            input_spatial_dim,
            filter_spatial_dim,
            Int(dilations_lt[i - 1]),
            Int(strides_lt[i - 1]),
            Int(paddings_lt[2 * i - 2] + paddings_lt[2 * i - 1]),
        )

        if output_spatial_dim <= 0:
            raise Error("[convolution] output spatial dim must be positive")

        output_shape[i] = output_spatial_dim

    comptime assert (
        input_buf.flat_rank == input_lt.rank
    ), "TileTensor flat_rank must match LayoutTensor rank for rebind safety"
    return rebind[IndexList[input_buf.flat_rank]](output_shape)


def conv_nhwc_direct[
    conv_info_rank: Int,
    //,
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_packed: Bool,
    conv_info_static: ConvInfoStatic[conv_info_rank],
    has_epilogue_fusion: Bool,
    elementwise_lambda: elementwise_simd_epilogue_type,
](
    input: TileTensor[mut=False, input_type, address_space=.GENERIC, ...],
    filter: TileTensor[mut=False, filter_type, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    stride: IndexList[conv_info_rank],
    dilation: IndexList[conv_info_rank],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    num_groups: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Runs a direct (register-tiled) NHWC convolution on CPU, bridging
    TileTensor inputs to LayoutTensors and dispatching to
    `ConvDirectNHWC.run` with optional elementwise epilogue fusion.

    Parameters:
        conv_info_rank: Number of spatial dimensions in the convolution (1,
            2, or 3) (inferred).
        input_layout: Memory layout of the input tensor.
        filter_layout: Memory layout of the filter tensor.
        output_layout: Memory layout of the output tensor.
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.
        filter_packed: True when the filter is prepacked for grouped
            convolution.
        conv_info_static: Statically known convolution attributes including
            padding, stride, dilation, and group count.
        has_epilogue_fusion: True when an elementwise epilogue is fused into
            the convolution.
        elementwise_lambda: Elementwise SIMD function applied to each output
            vector after the convolution.

    Args:
        input: Input activation TileTensor in NHWC or NDHWC layout.
        filter: Filter weights TileTensor.
        output: Output TileTensor in NHWC or NDHWC layout.
        stride: Stride along each spatial dimension.
        dilation: Dilation factor along each spatial dimension.
        pad_d: Padding before and after the depth dimension, stored as
            `(before, after)`.
        pad_h: Padding before and after the height dimension, stored as
            `(before, after)`.
        pad_w: Padding before and after the width dimension, stored as
            `(before, after)`.
        num_groups: Number of convolution groups for grouped convolution.
        ctx: Optional device context for parallel kernel launch (defaults to
            `None`).
    """
    # Construct LayoutTensors with explicit Layouts passed by the caller,
    # using the TileTensor's pointer and runtime shape. The Layouts must come
    # from ManagedTensorSlice.to_layout_tensor() (via the caller) so that
    # ConvDirectNHWC gets the same compile-time shape/stride info as it did
    # before the TileTensor migration.
    comptime ILT = LayoutTensor[input_type, input_layout, MutAnyOrigin]
    comptime FLT = LayoutTensor[filter_type, filter_layout, MutAnyOrigin]
    comptime OLT = LayoutTensor[output_type, output_layout, AnyOrigin[mut=True]]
    var input_lt = ILT(
        UnsafePointer[Scalar[input_type], MutAnyOrigin](
            unsafe_from_address=Int(input.ptr)
        ),
        ILT.RuntimeLayoutType.row_major(
            coord_to_index_list(input.layout.shape_coord()).cast[
                ILT.layout_int_type
            ]()
        ),
    )
    var filter_lt = FLT(
        UnsafePointer[Scalar[filter_type], MutAnyOrigin](
            unsafe_from_address=Int(filter.ptr)
        ),
        FLT.RuntimeLayoutType.row_major(
            coord_to_index_list(filter.layout.shape_coord()).cast[
                FLT.layout_int_type
            ]()
        ),
    )
    var output_lt = OLT(
        UnsafePointer[Scalar[output_type], AnyOrigin[mut=True]](
            unsafe_from_address=Int(output.ptr)
        ),
        OLT.RuntimeLayoutType.row_major(
            coord_to_index_list(output.layout.shape_coord()).cast[
                OLT.layout_int_type
            ]()
        ),
    )

    comptime assert conv_info_rank == input_layout.rank() - 2
    comptime assert (
        input_type == filter_type and input_type == output_type
    ), "conv input/output/filter types must be the same."
    comptime assert (filter_packed and filter_lt.rank == input_lt.rank + 1) or (
        not filter_packed and filter_lt.rank == input_lt.rank
    ), "Filter and input ranks mismatch."

    @always_inline
    def description_fn() {imm} -> String:
        return ";".join(
            [
                trace_arg("input", input_lt.runtime_layout.shape.value),
                trace_arg("filter", filter_lt.runtime_layout.shape.value),
                trace_arg("output", output_lt.runtime_layout.shape.value),
                "group=" + String(num_groups),
                "stride=" + "x".join([stride]),
                "padding_h=" + "x".join([pad_h]),
                "padding_w=" + "x".join([pad_w]),
            ]
        )

    with Trace[TraceLevel.OP, target=StaticString("cpu")](
        "conv",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
    ):
        var conv_shape = get_conv_shape[conv_info_rank, filter_packed](
            output,
            input,
            filter,
            stride,
            dilation,
            pad_d,
            pad_h,
            pad_w,
            num_groups,
        )

        # The closure updates a row segment of the output.
        @always_inline
        @__parameter
        def elementwise_epilogue[
            rank: Int
        ](coords: IndexList[rank], f_size: Int):
            comptime simd_size = simd_width_of[output_type]()

            @always_inline
            def body[width: Int](idx: Int) {coords, output_lt, mut}:
                # Coordinates of the current index.
                var curr_coords = rebind[IndexList[input_lt.rank]](coords)
                curr_coords[input_lt.rank - 1] += idx

                var vec = output_lt.load[width=width](curr_coords)
                elementwise_lambda(curr_coords, vec)

            vectorize[simd_size](f_size, body)

        ConvDirectNHWC[
            input_layout,
            filter_layout,
            output_layout,
            input_type,
            filter_type,
            output_type,
            filter_packed,
            conv_info_static,
            Optional[elementwise_epilogue_type](
                elementwise_epilogue
            ) if has_epilogue_fusion else None,
        ].run(
            output_lt,
            input_lt,
            filter_lt,
            conv_shape,
            ctx,
        )


# ===----------------------------------------------------------------------=== #
# GPU Convolution using cuDNN                                                  #
# ===----------------------------------------------------------------------=== #


@__name(
    t"conv2d_gpu_naive_nhwc_rscf_{input_type}_{filter_type}_{output_type}",
)
def conv2d_gpu_naive_nhwc_rscf[
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    block_size: Int,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type],
](
    input: LayoutTensor[input_type, input_layout, MutAnyOrigin],
    filter: LayoutTensor[filter_type, filter_layout, MutAnyOrigin],
    output: LayoutTensor[output_type, output_layout, MutAnyOrigin],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
    num_groups: Int32,
):
    """Naive GPU kernel for 2D NHWC convolution with RSCF filter layout.

    Each thread computes one output pixel across all output channels,
    iterating over the R x S filter window and the per-group input
    channels with scalar accumulation.

    Parameters:
        input_layout: Memory layout of the input tensor.
        filter_layout: Memory layout of the filter tensor.
        output_layout: Memory layout of the output tensor.
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.
        block_size: Square thread block extent used for both the x and
            y block dimensions in the launch grid.
        maybe_epilogue_func: Optional SIMD elementwise epilogue applied
            to each output value before storing.

    Args:
        input: Input tensor in NHWC layout.
        filter: Filter tensor in RSCF layout.
        output: Output tensor in NHWC layout.
        stride: Convolution stride `(stride_h, stride_w)`.
        dilation: Filter dilation factors
            `(dilation_h, dilation_w)`.
        padding: Symmetric zero padding `(pad_h, pad_w)` applied to
            the input H and W dimensions.
        num_groups: Number of convolution groups for grouped
            convolution.
    """
    var _num_groups = Int(num_groups)
    var N = input.dim[0]()
    var H = input.dim[1]()
    var W = input.dim[2]()
    var C_in = input.dim[3]()  # channel_in
    var R = filter.dim[0]()
    var S = filter.dim[1]()
    var C_per_group = filter.dim[2]()  # C_in / num_groups
    var H_out = output.dim[1]()
    var W_out = output.dim[2]()
    var C_out = output.dim[3]()  # channel_out or #F
    var F_per_group = C_out // _num_groups
    var pad_h = padding[0]
    var pad_w = padding[1]
    var stride_h = stride[0]
    var stride_w = stride[1]
    var dil_h = dilation[0]
    var dil_w = dilation[1]

    var n = block_idx.z
    var h = block_idx.y * block_dim.y + thread_idx.y
    var w = block_idx.x * block_dim.x + thread_idx.x

    if h >= H_out or w >= W_out:
        return

    for co in range(C_out):
        comptime accum_type = get_accum_type[output_type]()
        var value = Scalar[accum_type](0)
        var g = co // F_per_group
        var ci_base = g * C_per_group
        for r in range(R):
            for s in range(S):
                var h_in = h * stride_h - pad_h + r * dil_h
                var w_in = w * stride_w - pad_w + s * dil_w
                if 0 <= h_in < H and 0 <= w_in < W:
                    for ci in range(C_per_group):
                        value += (
                            input.load[width=1](
                                IndexList[4](n, h_in, w_in, ci_base + ci)
                            ).cast[accum_type]()
                            * filter.load[width=1](
                                IndexList[4](r, s, ci, co)
                            ).cast[accum_type]()
                        )

        comptime if maybe_epilogue_func:
            comptime epilogue_func = maybe_epilogue_func.value()
            epilogue_func(
                IndexList[4](n, h, w, co),
                value.cast[output_type](),
            )
        else:
            output.store(
                IndexList[4](n, h, w, co),
                value.cast[output_type](),
            )


# ===----------------------------------------------------------------------=== #
# GPU Convolution using cuDNN                                                  #
# ===----------------------------------------------------------------------=== #


@always_inline
def check_cudnn_error(stat: cudnnStatus_t) raises:
    """Raises an error if a cuDNN call returns a non-success status.

    Args:
        stat: Status code returned by a cuDNN API call.
    """
    if stat != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
        raise Error(t"cuDNN call failed with status {stat}")


struct CuDNNConvMeta(ImplicitlyCopyable, RegisterPassable):
    """Holds a cuDNN handle and the associated input, filter, convolution, and
    output descriptors for a single device.
    """

    @__allow_legacy_any_origin_fields
    var ptr_handle: UnsafePointer[cudnnContext, AnyOrigin[mut=True]]

    @__allow_legacy_any_origin_fields
    var ptr_input_desc: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]

    @__allow_legacy_any_origin_fields
    var ptr_filter_desc: UnsafePointer[cudnnFilterStruct, AnyOrigin[mut=True]]

    @__allow_legacy_any_origin_fields
    var ptr_conv_desc: UnsafePointer[
        cudnnConvolutionStruct, AnyOrigin[mut=True]
    ]

    @__allow_legacy_any_origin_fields
    var ptr_output_desc: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]

    def __init__(out self) raises:
        var ptr_handle: Optional[type_of(self.ptr_handle)] = None
        check_cudnn_error(cudnnCreate(UnsafePointer(to=ptr_handle)))

        var ptr_input_desc: Optional[type_of(self.ptr_input_desc)] = None
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=ptr_input_desc))
        )

        var ptr_filter_desc: Optional[type_of(self.ptr_filter_desc)] = None
        check_cudnn_error(
            cudnnCreateFilterDescriptor(UnsafePointer(to=ptr_filter_desc))
        )

        var ptr_conv_desc: Optional[type_of(self.ptr_conv_desc)] = None
        check_cudnn_error(
            cudnnCreateConvolutionDescriptor(UnsafePointer(to=ptr_conv_desc))
        )

        var ptr_output_desc: Optional[type_of(self.ptr_output_desc)] = None
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=ptr_output_desc))
        )

        self.ptr_handle = ptr_handle.value()
        self.ptr_input_desc = ptr_input_desc.value()
        self.ptr_filter_desc = ptr_filter_desc.value()
        self.ptr_conv_desc = ptr_conv_desc.value()
        self.ptr_output_desc = ptr_output_desc.value()

    def __deinit__(deinit self):
        try:
            check_cudnn_error(
                cudnnDestroyTensorDescriptor(self.ptr_output_desc)
            )
            check_cudnn_error(
                cudnnDestroyConvolutionDescriptor(self.ptr_conv_desc)
            )
            check_cudnn_error(
                cudnnDestroyFilterDescriptor(self.ptr_filter_desc)
            )
            check_cudnn_error(cudnnDestroyTensorDescriptor(self.ptr_input_desc))
            check_cudnn_error(cudnnDestroy(self.ptr_handle))
        except e:
            abort(String(e))


def _get_cudnn_meta(
    ctx: DeviceContext,
) raises -> UnsafePointer[CuDNNConvMeta, AnyOrigin[mut=True]]:
    """Get the cuDNN metadata with proper device context management.

    If the metadata is not found for this device, create a new one and insert
    it into the global cache keyed by device ID.

    IMPORTANT: this function _must_ be called with `ctx`'s CUcontext active via:

    ```mojo
    from max.gpu.host import DeviceContext
    var ctx = DeviceContext()
    with ctx.push_context():
        ptr_meta = _get_cudnn_meta(ctx)
    ```

    This is to satisfy the stateful `cudnn*` API calls.

    Args:
        ctx: The device context.

    Returns:
        The cuDNN metadata.
    """
    # Key the cuDNN metadata cache on the device ID.
    var cache_key = "CUDA_CUDNN_META_CACHE" + String(ctx.id())

    # Get or create the per-device cache dictionary.
    var ptr_meta = _get_global_or_null(cache_key)
    if ptr_meta:
        var ptr = ptr_meta.unsafe_value().unsafe_bitcast[CuDNNConvMeta]()
        check_cudnn_error(cudnnSetStream(ptr[].ptr_handle, CUDA(ctx.stream())))
        return ptr.as_unsafe_any_origin()

    var new_ptr_meta = alloc[CuDNNConvMeta]({count = 1}).unsafe_leak()
    new_ptr_meta.unsafe_write(CuDNNConvMeta())

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_key),
        new_ptr_meta.bitcast[NoneType](),
    )

    return new_ptr_meta.as_unsafe_any_origin()


def get_cudnn_dtype[dtype: DType]() raises -> cudnnDataType_t:
    """Map Mojo DType to cuDNN data type.

    Support only floating point dtypes for now.

    Parameters:
        dtype: The Mojo element type to map to a cuDNN data type.

    Raises:
        If the dtype is not supported by cuDNN.
    """

    comptime if dtype == .float32:
        return cudnnDataType_t.CUDNN_DATA_FLOAT
    elif dtype == .float16:
        return cudnnDataType_t.CUDNN_DATA_HALF
    elif dtype == .bfloat16:
        return cudnnDataType_t.CUDNN_DATA_BFLOAT16
    else:
        raise Error("unsupported dtype", dtype, "for cuDNN")


struct CachedCuDNNMetaNHWCFull(ImplicitlyCopyable):
    """Caches cuDNN descriptors, selected forward algorithm, and workspace
    size for a full NHWC 2D convolution, keyed by input/filter/output shapes
    and convolution parameters.
    """

    @__allow_legacy_any_origin_fields
    var ptr_handle: UnsafePointer[cudnnContext, AnyOrigin[mut=True]]

    @__allow_legacy_any_origin_fields
    var ptr_input_desc: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]

    @__allow_legacy_any_origin_fields
    var ptr_filter_desc: UnsafePointer[cudnnFilterStruct, AnyOrigin[mut=True]]

    @__allow_legacy_any_origin_fields
    var ptr_conv_desc: UnsafePointer[
        cudnnConvolutionStruct, AnyOrigin[mut=True]
    ]

    @__allow_legacy_any_origin_fields
    var ptr_output_desc: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]

    # Workspace size cache (actual buffer is allocated per-call via ctx)
    var workspace_size: Int

    # Algo Cache
    var best_algo: cudnnConvolutionFwdAlgo_t

    # Cache key fields
    var is_set: Bool
    var in_dtype: Optional[DType]
    var in_: Tuple[Int, Int, Int, Int]
    var filt: Tuple[Int, Int, Int, Int]
    var out: Tuple[Int, Int, Int, Int]

    var pad: Tuple[Int, Int]
    var stride: Tuple[Int, Int]
    var dil: Tuple[Int, Int]

    def __init__(out self) raises:
        var ptr_handle: Optional[type_of(self.ptr_handle)] = None
        check_cudnn_error(cudnnCreate(UnsafePointer(to=ptr_handle)))

        var ptr_input_desc: Optional[type_of(self.ptr_input_desc)] = None
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=ptr_input_desc))
        )

        var ptr_filter_desc: Optional[type_of(self.ptr_filter_desc)] = None
        check_cudnn_error(
            cudnnCreateFilterDescriptor(UnsafePointer(to=ptr_filter_desc))
        )

        var ptr_conv_desc: Optional[type_of(self.ptr_conv_desc)] = None
        check_cudnn_error(
            cudnnCreateConvolutionDescriptor(UnsafePointer(to=ptr_conv_desc))
        )

        var ptr_output_desc: Optional[type_of(self.ptr_output_desc)] = None
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=ptr_output_desc))
        )

        self.ptr_handle = ptr_handle.value()
        self.ptr_input_desc = ptr_input_desc.value()
        self.ptr_filter_desc = ptr_filter_desc.value()
        self.ptr_conv_desc = ptr_conv_desc.value()
        self.ptr_output_desc = ptr_output_desc.value()

        self.workspace_size = 0
        self.best_algo = (
            cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        )

        self.is_set = False
        self.in_dtype = None
        self.in_ = (0, 0, 0, 0)
        self.filt = (0, 0, 0, 0)
        self.out = (0, 0, 0, 0)
        self.pad = (0, 0)
        self.stride = (0, 0)
        self.dil = (0, 0)


def _get_cached_cudnn_meta_nhwc_full(
    ctx: DeviceContext,
) raises -> UnsafePointer[CachedCuDNNMetaNHWCFull, AnyOrigin[mut=True]]:
    var cache_key = "CUDA_CUDNN_CACHED_META_NHWC_FULL_" + String(ctx.id())

    var ptr_meta = _get_global_or_null(cache_key)
    if ptr_meta:
        var ptr = ptr_meta.unsafe_value().unsafe_bitcast[
            CachedCuDNNMetaNHWCFull
        ]()
        check_cudnn_error(cudnnSetStream(ptr[].ptr_handle, CUDA(ctx.stream())))
        return ptr.as_unsafe_any_origin()

    var new_ptr_meta = alloc[CachedCuDNNMetaNHWCFull]({count = 1}).unsafe_leak()
    new_ptr_meta.unsafe_write(CachedCuDNNMetaNHWCFull())

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_key),
        new_ptr_meta.bitcast[NoneType](),
    )

    check_cudnn_error(
        cudnnSetStream(new_ptr_meta[].ptr_handle, CUDA(ctx.stream()))
    )

    return new_ptr_meta.as_unsafe_any_origin()


def _conv_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[mut=False, input_type, ...],
    filter: TileTensor[mut=False, filter_type, ...],
    output: TileTensor[output_type, ...],
    stride_list: IndexList[2],
    dilation_list: IndexList[2],
    padding_list: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    # Use the optimized cached metadata implementation
    var ptr_meta = _get_cached_cudnn_meta_nhwc_full(ctx)

    # Input shape: NHWC
    var in_: Tuple[Int, Int, Int, Int] = (
        Int(input.dim[0]()),
        Int(input.dim[1]()),
        Int(input.dim[2]()),
        Int(input.dim[3]()),
    )

    # Filter shape: FCRS (K, C, R, S)
    var filt: Tuple[Int, Int, Int, Int] = (
        Int(filter.dim[0]()),
        Int(filter.dim[1]()),
        Int(filter.dim[2]()),
        Int(filter.dim[3]()),
    )

    # Output shape: NHWC
    var out: Tuple[Int, Int, Int, Int] = (
        Int(output.dim[0]()),
        Int(output.dim[1]()),
        Int(output.dim[2]()),
        Int(output.dim[3]()),
    )

    var pad: Tuple[Int, Int] = (padding_list[0], padding_list[1])
    var stride: Tuple[Int, Int] = (stride_list[0], stride_list[1])
    var dil: Tuple[Int, Int] = (dilation_list[0], dilation_list[1])

    var params_match = ptr_meta[].is_set

    if params_match:
        if ptr_meta[].in_dtype != input_type:
            params_match = False
        elif ptr_meta[].in_ != in_:
            params_match = False
        elif ptr_meta[].filt != filt:
            params_match = False
        elif ptr_meta[].out != out:
            params_match = False
        elif ptr_meta[].pad != pad:
            params_match = False
        elif ptr_meta[].stride != stride:
            params_match = False
        elif ptr_meta[].dil != dil:
            params_match = False

    if not params_match:
        # Update Input Descriptor (NHWC)
        check_cudnn_error(
            cudnnSetTensor4dDescriptor(
                ptr_meta[].ptr_input_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                get_cudnn_dtype[input_type](),
                Int16(in_[0]),
                Int16(in_[3]),
                Int16(in_[1]),
                Int16(in_[2]),
            )
        )

        # Update Filter Descriptor (NCHW for filter)
        check_cudnn_error(
            cudnnSetFilter4dDescriptor(
                ptr_meta[].ptr_filter_desc,
                get_cudnn_dtype[filter_type](),
                cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
                Int16(filt[0]),
                Int16(filt[1]),
                Int16(filt[2]),
                Int16(filt[3]),
            )
        )

        # Update Conv Descriptor
        check_cudnn_error(
            cudnnSetConvolution2dDescriptor(
                ptr_meta[].ptr_conv_desc,
                Int16(pad[0]),
                Int16(pad[1]),
                Int16(stride[0]),
                Int16(stride[1]),
                Int16(dil[0]),
                Int16(dil[1]),
                cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
            )
        )

        check_cudnn_error(
            cudnnSetConvolutionGroupCount(
                ptr_meta[].ptr_conv_desc, Int16(num_groups)
            )
        )

        # Update Output Descriptor (NHWC)
        check_cudnn_error(
            cudnnSetTensor4dDescriptor(
                ptr_meta[].ptr_output_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                get_cudnn_dtype[output_type](),
                Int16(out[0]),
                Int16(out[3]),
                Int16(out[1]),
                Int16(out[2]),
            )
        )

        # Use ALLOW_CONVERSION only for half-precision types to enable tensor
        # core acceleration. For float32, use DEFAULT_MATH to avoid incorrect
        # results on some GPU architectures (e.g., B200).
        comptime if input_type == .float16 or input_type == .bfloat16:
            check_cudnn_error(
                cudnnSetConvolutionMathType(
                    ptr_meta[].ptr_conv_desc,
                    cudnnMathType_t.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
                )
            )

        # Algorithm Autotuning.
        # The Mojo binding cudnnConvolutionFwdAlgoPerfStruct has incorrect
        # layout (Int8 enums vs C's 4-byte int enums). We bypass it by
        # allocating a raw 48-byte buffer matching the C ABI layout and
        # reading the algo Int32 at offset 0.
        var perf_buf_alloc = alloc[UInt8]({count = 48}).into_managed()
        var perf_buf = UnsafePointer(perf_buf_alloc.unsafe_ptr())
        var requested_count: Int16 = 1
        var returned_count: Int16 = 0

        check_cudnn_error(
            cudnnGetConvolutionForwardAlgorithm_v7(
                ptr_meta[].ptr_handle,
                ptr_meta[].ptr_input_desc,
                ptr_meta[].ptr_filter_desc,
                ptr_meta[].ptr_conv_desc,
                ptr_meta[].ptr_output_desc,
                requested_count,
                UnsafePointer(to=returned_count),
                perf_buf.bitcast[cudnnConvolutionFwdAlgoPerf_t](),
            )
        )

        if returned_count > 0:
            # Read algo enum (C int32) at byte offset 0
            var algo_val = perf_buf.bitcast[Int32]()[]
            ptr_meta[].best_algo = cudnnConvolutionFwdAlgo_t(Int(algo_val))
        else:
            ptr_meta[].best_algo = (
                cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            )

        # Query workspace size
        var ws_size: Int = 0
        check_cudnn_error(
            cudnnGetConvolutionForwardWorkspaceSize(
                ptr_meta[].ptr_handle,
                ptr_meta[].ptr_input_desc,
                ptr_meta[].ptr_filter_desc,
                ptr_meta[].ptr_conv_desc,
                ptr_meta[].ptr_output_desc,
                ptr_meta[].best_algo,
                UnsafePointer(to=ws_size),
            )
        )
        ptr_meta[].workspace_size = ws_size

        # Update Cache State
        ptr_meta[].is_set = True
        ptr_meta[].in_dtype = input_type
        ptr_meta[].in_ = in_
        ptr_meta[].filt = filt
        ptr_meta[].out = out
        ptr_meta[].pad = pad
        ptr_meta[].stride = stride
        ptr_meta[].dil = dil

    # Allocate workspace per-call using ctx (runtime-managed buffer)
    var workspace_buffer = ctx.enqueue_create_buffer[.uint8](
        ptr_meta[].workspace_size
    )

    var alpha: Float32 = 1.0
    var beta: Float32 = 0.0

    check_cudnn_error(
        cudnnConvolutionForward(
            ptr_meta[].ptr_handle,
            UnsafePointer(to=alpha).bitcast[NoneType](),
            ptr_meta[].ptr_input_desc,
            input.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_filter_desc,
            filter.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_conv_desc,
            ptr_meta[].best_algo,
            workspace_buffer.unsafe_ptr().bitcast[NoneType](),
            ptr_meta[].workspace_size,
            UnsafePointer(to=beta).bitcast[NoneType](),
            ptr_meta[].ptr_output_desc,
            output.ptr.bitcast[NoneType](),
        )
    )
    _ = workspace_buffer^


def conv_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[mut=False, input_type, ...],
    filter: TileTensor[mut=False, filter_type, ...],
    output: TileTensor[output_type, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    """Runs a 2D convolution via cuDNN with NHWC input/output and FCRS filter
    layout, activating the device context before dispatching.

    Parameters:
        input_type: Element type of the input tensor (inferred).
        filter_type: Element type of the filter tensor (inferred).
        output_type: Element type of the output tensor (inferred).

    Args:
        input: Input activation tensor in NHWC layout.
        filter: Filter weights tensor in FCRS layout.
        output: Output tensor in NHWC layout.
        stride: Stride along the height and width dimensions.
        dilation: Dilation factor along the height and width dimensions.
        padding: Symmetric padding along the height and width dimensions.
        num_groups: Number of convolution groups for grouped convolution.
        ctx: Device context for the cuDNN stream.
    """
    # Set `ctx`'s CUcontext as current to satisfy cudnn's stateful API.
    with ctx.push_context() as ctx:
        _conv_cudnn(
            input, filter, output, stride, dilation, padding, num_groups, ctx
        )


# ===----------------------------------------------------------------------=== #
# GPU Convolution using MIOpen (AMD)                                           #
# ===----------------------------------------------------------------------=== #


struct CachedMIOpenMeta[conv_rank: Int](Movable):
    """Caches MIOpen handle, tensor/filter/convolution descriptors, selected
    forward algorithm, and workspace size for a convolution of the given rank,
    keyed by input/filter/output shapes and convolution parameters.

    Parameters:
        conv_rank: Number of spatial dimensions in the convolution (1, 2,
            or 3).
    """

    comptime tensor_rank = Self.conv_rank + 2

    var handle: MIOpenHandle
    var input_desc: MIOpenTensorDescriptor
    var filter_desc: MIOpenTensorDescriptor
    var output_desc: MIOpenTensorDescriptor
    var conv_desc: MIOpenConvolutionDescriptor

    # Legacy API: cached algorithm and workspace size from FindAlgorithm
    var algo: ConvFwdAlgorithm
    var workspace_size: UInt64

    # Cache key fields
    var is_set: Bool
    var input_dtype: Optional[DType]
    var input_shape: Array[UInt64, Self.tensor_rank]
    var filter_shape: Array[UInt64, Self.tensor_rank]
    var output_shape: Array[UInt64, Self.tensor_rank]
    var padding: Array[Int32, Self.conv_rank]
    var stride: Array[Int32, Self.conv_rank]
    var dilation: Array[Int32, Self.conv_rank]

    def __init__(out self) raises:
        self.handle = MIOpenHandle()
        check_miopen_error(
            miopenCreate(UnsafePointer(to=self.handle).bitcast[NoneType]())
        )

        self.input_desc = MIOpenTensorDescriptor()
        check_miopen_error(
            miopenCreateTensorDescriptor(
                UnsafePointer(to=self.input_desc).bitcast[NoneType]()
            )
        )

        self.filter_desc = MIOpenTensorDescriptor()
        check_miopen_error(
            miopenCreateTensorDescriptor(
                UnsafePointer(to=self.filter_desc).bitcast[NoneType]()
            )
        )

        self.output_desc = MIOpenTensorDescriptor()
        check_miopen_error(
            miopenCreateTensorDescriptor(
                UnsafePointer(to=self.output_desc).bitcast[NoneType]()
            )
        )

        self.conv_desc = MIOpenConvolutionDescriptor()
        check_miopen_error(
            miopenCreateConvolutionDescriptor(
                UnsafePointer(to=self.conv_desc).bitcast[NoneType]()
            )
        )

        self.algo = ConvFwdAlgorithm(0)
        self.workspace_size = 0

        self.is_set = False
        self.input_dtype = None
        self.input_shape = Array[UInt64, Self.tensor_rank](fill=0)
        self.filter_shape = Array[UInt64, Self.tensor_rank](fill=0)
        self.output_shape = Array[UInt64, Self.tensor_rank](fill=0)
        self.padding = Array[Int32, Self.conv_rank](fill=0)
        self.stride = Array[Int32, Self.conv_rank](fill=0)
        self.dilation = Array[Int32, Self.conv_rank](fill=0)


def _get_cached_miopen_meta[
    conv_rank: Int
](
    ctx: DeviceContext,
) raises -> UnsafePointer[
    CachedMIOpenMeta[conv_rank], MutAnyOrigin
]:
    var cache_key = String(
        "MIOPEN_CACHED_META_", String(conv_rank), "D_", String(ctx.id())
    )

    var ptr_meta = _get_global_or_null(cache_key)
    if ptr_meta:
        var ptr = ptr_meta.unsafe_value().unsafe_bitcast[
            CachedMIOpenMeta[conv_rank]
        ]()
        check_miopen_error(miopenSetStream(ptr[].handle, HIP(ctx.stream())))
        return ptr.as_unsafe_any_origin()

    var new_ptr_meta = alloc[CachedMIOpenMeta[conv_rank]](
        {count = 1}
    ).unsafe_leak()
    new_ptr_meta.unsafe_write(CachedMIOpenMeta[conv_rank]())

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_key),
        new_ptr_meta.bitcast[NoneType](),
    )

    check_miopen_error(
        miopenSetStream(new_ptr_meta[].handle, HIP(ctx.stream()))
    )

    return new_ptr_meta.as_unsafe_any_origin()


def _conv_miopen[
    conv_rank: Int,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    //,
    filter_is_fcrs: Bool = False,
](
    input: TileTensor[mut=False, input_type, ...],
    filter: TileTensor[mut=False, filter_type, ...],
    output: TileTensor[output_type, ...],
    stride_list: IndexList[conv_rank],
    dilation_list: IndexList[conv_rank],
    padding_list: IndexList[conv_rank],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    comptime tensor_rank = conv_rank + 2

    comptime assert input.rank == tensor_rank, "incorrect tensor rank"
    comptime assert filter.rank == tensor_rank, "incorrect tensor rank"
    comptime assert output.rank == tensor_rank, "incorrect tensor rank"

    # MIOpen needs all tensors in the same layout. Since input/output use
    # NHWC strides, the filter must also be NHWC (FRSC physical layout).
    # Transpose RSCF→FRSC or FCRS→FRSC on GPU. This is a small weight
    # tensor — the cost is negligible compared to the conv itself.
    var filter_size = filter.num_elements()
    var filter_frsc_buf = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_frsc_ptr = filter_frsc_buf.unsafe_ptr()
    var filter_shape = Array[UInt64, tensor_rank](fill=0)

    comptime if filter_is_fcrs:
        comptime assert conv_rank == 2, "FCRS requires 2D convolution"

        # FCRS [F,C,R,S] -> FRSC [F,R,S,C]
        var F_dim = Int(filter.dim[0]())
        var C_dim = Int(filter.dim[1]())
        var R_dim = Int(filter.dim[2]())
        var S_dim = Int(filter.dim[3]())

        @always_inline
        def transpose_fcrs_to_frsc[
            _width: Int, alignment: Int = 1
        ](coords: Coord) {
            var filter,
            var filter_frsc_ptr,
            var R_dim,
            var S_dim,
            var C_dim,
        }:
            var f = Int(coords[0].value())
            var r = Int(coords[1].value())
            var s = Int(coords[2].value())
            var c = Int(coords[3].value())
            var val = filter.load[width=_width]((f, c, r, s))
            var out_idx = (
                f * R_dim * S_dim * C_dim + r * S_dim * C_dim + s * C_dim + c
            )
            filter_frsc_ptr.store(out_idx, val)

        elementwise[1, target="gpu"](
            transpose_fcrs_to_frsc, (F_dim, R_dim, S_dim, C_dim), ctx
        )
        filter_shape[0] = UInt64(F_dim)
        filter_shape[1] = UInt64(C_dim)
        filter_shape[2] = UInt64(R_dim)
        filter_shape[3] = UInt64(S_dim)

    elif conv_rank == 2:
        # RSCF [R,S,C,F] -> FRSC [F,R,S,C]
        var R_dim = Int(filter.dim[0]())
        var S_dim = Int(filter.dim[1]())
        var C_dim = Int(filter.dim[2]())
        var F_dim = Int(filter.dim[3]())

        @always_inline
        def transpose_rscf_to_frsc[
            _width: Int, alignment: Int = 1
        ](coords: Coord) {
            var filter,
            var filter_frsc_ptr,
            var R_dim,
            var S_dim,
            var C_dim,
        }:
            var f = Int(coords[0].value())
            var r = Int(coords[1].value())
            var s = Int(coords[2].value())
            var c = Int(coords[3].value())
            var val = filter.load[width=_width]((r, s, c, f))
            var out_idx = (
                f * R_dim * S_dim * C_dim + r * S_dim * C_dim + s * C_dim + c
            )
            filter_frsc_ptr.store(out_idx, val)

        elementwise[1, target="gpu"](
            transpose_rscf_to_frsc, (F_dim, R_dim, S_dim, C_dim), ctx
        )

        filter_shape[0] = UInt64(F_dim)
        filter_shape[1] = UInt64(C_dim)
        filter_shape[2] = UInt64(R_dim)
        filter_shape[3] = UInt64(S_dim)

    else:
        comptime assert conv_rank == 3, "Only support 2D/3D convolution"

        # QRSCF [Q,R,S,C,F] -> FQRSC [F,Q,R,S,C]
        var Q_dim = Int(filter.dim[0]())
        var R_dim = Int(filter.dim[1]())
        var S_dim = Int(filter.dim[2]())
        var C_dim = Int(filter.dim[3]())
        var F_dim = Int(filter.dim[4]())

        @always_inline
        def transpose_qrscf_to_fqrsc[
            _width: Int, alignment: Int = 1
        ](coords: Coord) {
            var filter,
            var filter_frsc_ptr,
            var Q_dim,
            var R_dim,
            var S_dim,
            var C_dim,
        }:
            var f = Int(coords[0].value())
            var q = Int(coords[1].value())
            var r = Int(coords[2].value())
            var s = Int(coords[3].value())
            var c = Int(coords[4].value())
            var val = filter.load[width=_width]((q, r, s, c, f))
            var out_idx = (
                f * Q_dim * R_dim * S_dim * C_dim
                + q * R_dim * S_dim * C_dim
                + r * S_dim * C_dim
                + s * C_dim
                + c
            )
            filter_frsc_ptr.store(out_idx, val)

        elementwise[1, target="gpu"](
            transpose_qrscf_to_fqrsc, (F_dim, Q_dim, R_dim, S_dim, C_dim), ctx
        )

        filter_shape[0] = UInt64(F_dim)
        filter_shape[1] = UInt64(C_dim)
        filter_shape[2] = UInt64(Q_dim)
        filter_shape[3] = UInt64(R_dim)
        filter_shape[4] = UInt64(S_dim)

    @always_inline
    def image_shape_from_tensor(
        tensor: TileTensor,
    ) -> Array[UInt64, tensor_rank]:
        # Convert to channels first format.
        var shape = Array[UInt64, tensor_rank](fill=0)
        shape[0] = UInt64(tensor.dim[0]())
        shape[1] = UInt64(tensor.dim[tensor_rank - 1]())
        comptime for i in range(conv_rank):
            shape[2 + i] = UInt64(tensor.dim[1 + i]())
        return shape^

    var input_shape = image_shape_from_tensor(input)
    var output_shape = image_shape_from_tensor(output)

    @always_inline
    def int32_array_from_list[
        name: StaticString
    ](list: IndexList[conv_rank],) raises -> Array[Int32, conv_rank]:
        var array = Array[Int32, conv_rank](fill=0)
        for i in range(conv_rank):
            array[i] = Int32(list[i])
            if Int(array[i]) != list[i]:
                raise Error(t"{name} value is too large: ", list[i])
        return array^

    var padding = int32_array_from_list["padding"](padding_list)
    var stride = int32_array_from_list["stride"](stride_list)
    var dilation = int32_array_from_list["dilation"](dilation_list)

    var ptr_meta = _get_cached_miopen_meta[conv_rank](ctx)

    if (
        not ptr_meta[].is_set
        or ptr_meta[].input_dtype != input_type
        or ptr_meta[].input_shape != input_shape
        or ptr_meta[].filter_shape != filter_shape
        or ptr_meta[].output_shape != output_shape
        or ptr_meta[].padding != padding
        or ptr_meta[].stride != stride
        or ptr_meta[].dilation != dilation
    ):

        @always_inline
        def strides_from_shape(
            shape: Array[UInt64, tensor_rank]
        ) -> Array[UInt64, tensor_rank]:
            # For logical image (NCHW) or filter (FCRS) shapes, the innermost physical
            # stride is the channels dimension (C) with all other channels expanding
            # out from that.
            var strides = Array[UInt64, tensor_rank](fill=0)
            var product = shape[1]
            comptime for i in reversed(range(2, tensor_rank)):
                strides[i] = product
                product *= shape[i]
            strides[1] = 1
            strides[0] = product
            return strides^

        var input_strides = strides_from_shape(input_shape)
        var filter_strides = strides_from_shape(filter_shape)
        var output_strides = strides_from_shape(output_shape)

        # Input descriptor: NCHW logical dims with NHWC physical strides
        check_miopen_error(
            miopenSetTensorDescriptorV2(
                ptr_meta[].input_desc,
                MIOpenDataType(input_type),
                Int32(tensor_rank),
                input_shape.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                input_strides.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            )
        )

        # Filter descriptor: NHWC strides (matching input/output layout).
        # Filter data must be in FRSC physical layout for NHWC strides.
        check_miopen_error(
            miopenSetTensorDescriptorV2(
                ptr_meta[].filter_desc,
                MIOpenDataType(filter_type),
                Int32(tensor_rank),
                filter_shape.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                filter_strides.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            )
        )

        # Output descriptor: NCHW logical dims with NHWC physical strides
        check_miopen_error(
            miopenSetTensorDescriptorV2(
                ptr_meta[].output_desc,
                MIOpenDataType(output_type),
                Int32(tensor_rank),
                output_shape.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                output_strides.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            )
        )

        # Convolution descriptor
        check_miopen_error(
            miopenInitConvolutionNdDescriptor(
                ptr_meta[].conv_desc,
                Int32(conv_rank),
                padding.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                stride.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                dilation.unsafe_ptr().as_imm().as_unsafe_any_origin(),
                ConvolutionMode.CONVOLUTION,
            )
        )

        if num_groups > 1:
            check_miopen_error(
                miopenSetConvolutionGroupCount(
                    ptr_meta[].conv_desc, Int32(num_groups)
                )
            )

        # Get workspace size
        var workspace_size: UInt64 = 0
        check_miopen_error(
            miopenConvolutionForwardGetWorkSpaceSize(
                ptr_meta[].handle,
                ptr_meta[].filter_desc,
                ptr_meta[].input_desc,
                ptr_meta[].conv_desc,
                ptr_meta[].output_desc,
                UnsafePointer(to=workspace_size).as_unsafe_any_origin(),
            )
        )

        var find_workspace = ctx.enqueue_create_buffer[.uint8](
            Int(workspace_size)
        )

        var perf = ConvAlgoPerf()
        var returned_count: Int32 = 0
        check_miopen_error(
            miopenFindConvolutionForwardAlgorithm(
                ptr_meta[].handle,
                ptr_meta[].input_desc,
                input.ptr.bitcast[NoneType](),
                ptr_meta[].filter_desc,
                filter_frsc_ptr.bitcast[NoneType](),
                ptr_meta[].conv_desc,
                ptr_meta[].output_desc,
                output.ptr.bitcast[NoneType](),
                Int32(1),
                UnsafePointer(to=returned_count).as_unsafe_any_origin(),
                UnsafePointer(to=perf).as_unsafe_any_origin(),
                find_workspace.unsafe_ptr().bitcast[NoneType](),
                workspace_size,
                False,  # non-exhaustive search
            )
        )

        if returned_count == 0:
            raise Error("MIOpen: no algorithm found for convolution")

        ptr_meta[].algo = ConvFwdAlgorithm(perf.fwd_algo)
        ptr_meta[].workspace_size = perf.memory

        # Update cache state
        ptr_meta[].is_set = True
        ptr_meta[].input_dtype = input_type
        ptr_meta[].input_shape = input_shape.copy()
        ptr_meta[].filter_shape = filter_shape.copy()
        ptr_meta[].output_shape = output_shape.copy()
        ptr_meta[].padding = padding.copy()
        ptr_meta[].stride = stride.copy()
        ptr_meta[].dilation = dilation.copy()

    # Run forward convolution
    var forward_workspace = ctx.enqueue_create_buffer[.uint8](
        Int(ptr_meta[].workspace_size)
    )

    var alpha = Float32(1.0)
    var beta = Float32(0.0)

    check_miopen_error(
        miopenConvolutionForward(
            ptr_meta[].handle,
            UnsafePointer(to=alpha).as_imm().as_unsafe_any_origin(),
            ptr_meta[].input_desc,
            input.ptr.bitcast[NoneType](),
            ptr_meta[].filter_desc,
            filter_frsc_ptr.bitcast[NoneType](),
            ptr_meta[].conv_desc,
            ptr_meta[].algo,
            UnsafePointer(to=beta).as_imm().as_unsafe_any_origin(),
            ptr_meta[].output_desc,
            output.ptr.bitcast[NoneType](),
            forward_workspace.unsafe_ptr().bitcast[NoneType](),
            ptr_meta[].workspace_size,
        )
    )


def _conv_miopen[
    conv_rank: Int,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    //,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
    filter_is_fcrs: Bool = False,
](
    input: TileTensor[mut=False, input_type, ...],
    filter: TileTensor[mut=False, filter_type, ...],
    output: TileTensor[output_type, ...],
    stride: IndexList[conv_rank],
    dilation: IndexList[conv_rank],
    padding: IndexList[conv_rank],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    comptime if maybe_epilogue_func:
        # MIOpen doesn't support epilogues. Compute to temp buffer,
        # then apply epilogue.
        comptime epilogue = maybe_epilogue_func.value()
        var output_tmp_data = ctx.enqueue_create_buffer[output_type](
            output.num_elements()
        )
        var output_tmp = TileTensor[
            output_type, output.LayoutType, MutAnyOrigin, ...
        ](output_tmp_data, output.layout)
        _conv_miopen[filter_is_fcrs=filter_is_fcrs](
            input,
            filter,
            output_tmp,
            stride,
            dilation,
            padding,
            num_groups,
            ctx,
        )

        @always_inline
        def miopen_epilogue[
            _width: Int, alignment: Int = 1
        ](coords: Coord) {var output_tmp}:
            epilogue(
                coord_to_index_list(coords),
                output_tmp.load[width=_width](coords),
            )

        elementwise[
            simd_width_of[output_type, target=get_gpu_target()](),
            target="gpu",
        ](
            miopen_epilogue,
            output.layout.shape_coord(),
            ctx,
        )
        _ = output_tmp_data^
    else:
        _conv_miopen[filter_is_fcrs=filter_is_fcrs](
            input,
            filter,
            output,
            stride,
            dilation,
            padding,
            num_groups,
            ctx,
        )


def conv_miopen[
    conv_rank: Int,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    //,
    filter_is_fcrs: Bool = False,
](
    input: TileTensor[mut=False, input_type, ...],
    filter: TileTensor[mut=False, filter_type, ...],
    output: TileTensor[output_type, ...],
    stride: IndexList[conv_rank],
    dilation: IndexList[conv_rank],
    padding: IndexList[conv_rank],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    """Runs a convolution via MIOpen on AMD GPUs, transposing the filter to
    FRSC physical layout and dispatching to the cached MIOpen forward
    convolution path.

    Parameters:
        conv_rank: Number of spatial dimensions in the convolution (1, 2,
            or 3) (inferred).
        input_type: Element type of the input tensor (inferred).
        filter_type: Element type of the filter tensor (inferred).
        output_type: Element type of the output tensor (inferred).
        filter_is_fcrs: True when the filter uses FCRS layout, otherwise RSCF
            (defaults to `False`).

    Args:
        input: Input activation tensor in NHWC or NDHWC layout.
        filter: Filter weights tensor in RSCF, FCRS, or QRSCF layout.
        output: Output tensor in NHWC or NDHWC layout.
        stride: Stride along each spatial dimension.
        dilation: Dilation factor along each spatial dimension.
        padding: Symmetric padding applied to each spatial dimension.
        num_groups: Number of convolution groups for grouped convolution.
        ctx: Device context for kernel launch.
    """
    _conv_miopen[filter_is_fcrs=filter_is_fcrs](
        input, filter, output, stride, dilation, padding, num_groups, ctx
    )


def conv_gpu[
    conv_rank: Int,
    //,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
    filter_is_fcrs: Bool = False,
    has_residual: Bool = False,
](
    input: TileTensor[mut=True, input_type, address_space=.GENERIC, ...],
    filter: TileTensor[filter_type, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    stride: IndexList[conv_rank],
    dilation: IndexList[conv_rank],
    padding: IndexList[2 * conv_rank],
    num_groups: Int,
    ctx: DeviceContext,
    source_ptr: Optional[
        UnsafePointer[Scalar[output_type], MutAnyOrigin]
    ] = None,
    beta: Float32 = 0.0,
) raises:
    """Dispatches a GPU convolution to the best available backend for the
    current device and shape, including SM100 structured conv, im2col+matmul,
    AMD 4-wave, Apple M5 fused, cuDNN, MIOpen, and naive reference kernels,
    with optional asymmetric padding pre-processing and elementwise epilogue
    fusion.

    Parameters:
        conv_rank: Number of spatial dimensions in the convolution (1, 2,
            or 3) (inferred).
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.
        maybe_epilogue_func: Optional elementwise SIMD epilogue applied to
            the output (defaults to `None`).
        filter_is_fcrs: True when the filter uses FCRS layout, otherwise
            RSCF (defaults to `False`).
        has_residual: True when fusing a residual add of the form
            `D = Conv(A, B) + beta * C` (defaults to `False`).

    Args:
        input: Input activation tensor in NHWC or NDHWC layout.
        filter: Filter weights tensor.
        output: Output tensor in NHWC or NDHWC layout.
        stride: Stride along each spatial dimension.
        dilation: Dilation factor along each spatial dimension.
        padding: Padding before and after each spatial dimension, stored
            as interleaved `(before, after)` pairs.
        num_groups: Number of convolution groups for grouped convolution.
        ctx: Device context for kernel launch.
        source_ptr: Pointer to the residual source tensor `C`, used only
            when `has_residual` is true (defaults to `None`).
        beta: Residual scale factor in `D = Conv(A, B) + beta * C` (defaults
            to `0.0`).
    """
    # Bridge to LayoutTensor for internal GPU kernel dispatch and cuDNN/MIOpen
    # which require Layout type parameters.
    var input_lt = input.to_layout_tensor()
    var filter_lt = filter.to_layout_tensor()
    var output_lt = output.to_layout_tensor()

    comptime input_layout = input_lt.layout
    comptime filter_layout = filter_lt.layout
    comptime output_layout = output_lt.layout

    comptime assert conv_rank == input_lt.rank - 2

    # Zero-sized output (e.g. a ``(B, 0, 0, C)`` input flowing through a
    # diffusion VAE encoder for the text-to-image placeholder): nothing
    # to compute. The output buffer is pre-allocated zero-element by
    # the caller -- an early return produces the correct empty output
    # and skips downstream dispatch paths that would otherwise build
    # zero-extent TMA descriptors or launch zero-grid kernels.
    if output_lt.size() == 0:
        return

    var has_asymmetric_padding = False
    var pad_before = IndexList[conv_rank](0)

    comptime for i in range(conv_rank):
        pad_before[i] = padding[2 * i]
        var after = padding[2 * i + 1]
        if pad_before[i] != after:
            has_asymmetric_padding = True

    if has_asymmetric_padding:
        # Pre-pad on GPU so downstream kernels (including cuDNN) can assume symmetric padding.
        comptime full_rank = input_layout.rank()
        var paddings_tensor = tt_stack_allocation[dtype=DType.int](
            row_major[2 * full_rank]()
        )

        comptime for axis in range(full_rank):
            paddings_tensor[2 * axis] = 0
            paddings_tensor[2 * axis + 1] = 0

        comptime for i in range(conv_rank):
            comptime SIMDInt = Int

            var axis = i + 1  # skip batch axis
            paddings_tensor[2 * axis] = SIMDInt(padding[2 * i])  # before
            paddings_tensor[2 * axis + 1] = SIMDInt(padding[2 * i + 1])  # after

        var input_shape = rebind[IndexList[full_rank]](
            input_lt.runtime_layout.shape.value.canonicalize()
        )
        var padded_shape = IndexList[full_rank]()

        comptime for axis in range(full_rank):
            var before = 0
            var after = 0
            if axis > 0 and axis < full_rank - 1:
                var spatial_idx = axis - 1
                before = padding[2 * spatial_idx]
                after = padding[2 * spatial_idx + 1]
            padded_shape[axis] = input_shape[axis] + before + after

        var padded_elements = padded_shape.flattened_length()
        var tmp_buffer = ctx.enqueue_create_buffer[input_type](padded_elements)
        var padded_device_buffer = tmp_buffer.unsafe_ptr()
        var zero_scalar = Scalar[input_type](0)

        pad_constant_gpu[full_rank, input_type, DType.int](
            padded_device_buffer,
            padded_shape,
            input.ptr,
            input_shape,
            paddings_tensor.ptr,
            zero_scalar,
            ctx,
        )

        # Construct padded input as LayoutTensor, then bridge to TileTensor
        # for the recursive call. Using LayoutTensor here because full_rank
        # is variable and row_major(Coord) requires a fixed-rank tuple.
        var padded_input_lt = LayoutTensor[
            input_type,
            Layout.row_major[full_rank](),
            MutAnyOrigin,
        ](
            padded_device_buffer.as_unsafe_any_origin(),
            RuntimeLayout[Layout.row_major[full_rank]()].row_major(
                padded_shape
            ),
        )
        var padded_input_tt = lt_to_tt(padded_input_lt)

        var zero_padding = IndexList[2 * conv_rank](0)

        conv_gpu[
            input_type,
            filter_type,
            output_type,
            maybe_epilogue_func,
            filter_is_fcrs,
            has_residual,
        ](
            padded_input_tt,
            filter,
            output,
            stride,
            dilation,
            zero_padding,
            num_groups,
            ctx,
            source_ptr,
            beta,
        )

        return

    # We can now use pad_before (which is now confirmed equal to pad_after) as
    # the symmetric padding.
    var symmetric_padding = pad_before

    comptime block_size = 16

    comptime conv_gpu_n = conv2d_gpu_naive_nhwc_rscf[
        input_layout,
        filter_layout,
        output_layout,
        input_type,
        filter_type,
        output_type,
        block_size,
        maybe_epilogue_func,
    ]

    comptime conv_gpu_3d = conv3d_gpu_naive_ndhwc_qrscf[
        input_layout,
        filter_layout,
        output_layout,
        input_type,
        filter_type,
        output_type,
        block_size,
        maybe_epilogue_func,
    ]
    var grid_dim_y = ceildiv(
        output_lt.dim[1](), block_size
    )  # height for 2d and depth for 3d
    var grid_dim_z = input_lt.dim[0]()  # n for both

    comptime if input_lt.rank == 4:
        # Try SM100 structured conv2d on Blackwell GPUs (4-7x faster than cuDNN)
        comptime _is_sm100 = _is_sm10x_gpu(ctx.default_device_info)
        comptime _is_supported_dtype = input_type == DType.bfloat16

        comptime if _is_sm100 and _is_supported_dtype:
            from nn.conv.gpu.nvidia.sm100.dispatch import (
                dispatch_sm100_conv2d,
                test_alignment_sm100_conv2d,
            )
            from linalg.utils import elementwise_epilogue_type

            # SM100 dispatch: stride=1, dilation=1, groups=1, and inner
            # C-row is 64B-aligned (TMA swizzle alignment). The dispatch
            # picks SWIZZLE_128B when C*sizeof is 128B-aligned, otherwise
            # SWIZZLE_64B (e.g. bf16 C_in=96 → 192 B per row).
            #
            # out_c (GEMM N) does NOT need MMA_N alignment: the output TMA
            # descriptor is created with the actual out_c as its N bound, so
            # the hardware drops OOB stores for the tail, and the filter TMA
            # zero-fills OOB rows on load. The only remaining constraint is
            # SIMD-pair alignment in the epilogue lambda path (the
            # `top_col >= self.N` guard in epilogue_components.mojo fires at
            # pair granularity, not per-element), hence
            # `out_c * sizeof(output) % 4 == 0` (bf16: out_c % 2 == 0).
            var s = rebind[IndexList[2]](stride)
            var d = rebind[IndexList[2]](dilation)
            var in_c = input_lt.dim[input_lt.rank - 1]()
            var out_c = output_lt.dim[output_lt.rank - 1]()
            if (
                s[0] == 1
                and s[1] == 1
                and d[0] == 1
                and d[1] == 1
                and num_groups == 1
                and test_alignment_sm100_conv2d[input_type, output_type](
                    in_c, out_c
                )
            ):

                @__parameter
                @always_inline
                def _sm100_dispatch[
                    _epilogue: Optional[elementwise_epilogue_type] = None,
                ]() raises:
                    dispatch_sm100_conv2d[
                        filter_is_fcrs,
                        elementwise_lambda_fn=_epilogue,
                        has_residual=has_residual,
                    ](
                        input,
                        filter,
                        output,
                        rebind[IndexList[2]](symmetric_padding),
                        ctx,
                        source_ptr,
                        beta,
                    )

                comptime if maybe_epilogue_func:
                    # Wrap the 4D NHWC epilogue into a 2D GEMM-space
                    # void epilogue for the SM100 kernel. The kernel
                    # calls this with (m, n) coords where
                    # m = batch*H_out*W_out + h*W_out + w, n = channel.
                    comptime epilogue = maybe_epilogue_func.value()
                    var out_h = output_lt.dim[1]()
                    var out_w = output_lt.dim[2]()
                    var hw = out_h * out_w

                    @__parameter
                    @always_inline
                    @__copy_capture(hw, out_w)
                    def sm100_void_epilogue[
                        _dtype: DType,
                        _width: SIMDLength,
                        *,
                        alignment: Int = 1,
                    ](coords_2d: IndexList[2], val: SIMD[_dtype, _width],):
                        var m = coords_2d[0]
                        var n = coords_2d[1]
                        var batch_idx: Int
                        var rem: Int
                        var h_idx: Int
                        var w_idx: Int
                        batch_idx, rem = divmod(m, hw)
                        h_idx, w_idx = divmod(rem, out_w)
                        epilogue(
                            IndexList[4](batch_idx, h_idx, w_idx, n),
                            rebind[SIMD[output_type, _width]](val),
                        )

                    _sm100_dispatch[
                        Optional[elementwise_epilogue_type](sm100_void_epilogue)
                    ]()
                else:
                    _sm100_dispatch[]()
                return

        # SM100 im2col+matmul: route non-128-aligned channels through
        # `_matmul_gpu` (UMMA-on-Blackwell for bf16) instead of the naive
        # thread-per-pixel fallback.
        comptime if _is_sm100:
            from nn.conv.gpu.im2col_matmul_2d import (
                dispatch_im2col_matmul_conv2d,
            )

            if dispatch_im2col_matmul_conv2d[
                filter_is_fcrs,
                maybe_epilogue_func,
            ](
                input,
                filter,
                output,
                rebind[IndexList[2]](stride),
                rebind[IndexList[2]](dilation),
                rebind[IndexList[2]](symmetric_padding),
                num_groups,
                ctx,
            ):
                return

        # Apple M5 (compute_capability == 5): fused online-im2col conv2d.
        # `dispatch_fused_im2col_conv2d_apple` runs `AppleM5MatMul.run_conv`
        # (the structured simdgroup-tiled GEMM, 16x16x16 hardware MMA) with the
        # A operand gathered directly from the NHWC input per MMA-fragment -- the
        # `[M, K]` im2col matrix is never materialised to global memory. This
        # eliminates the materialised path's DRAM round-trip, so conv wins across
        # both compute- and memory-bound regimes (no memory-bound naive guard
        # needed). The dispatcher self-gates (bf16, groups=1, dilation=1,
        # kernel > 1x1, K >= 16, N >= 16, compute_capability == 5); on decline
        # (incl. non-M5) it falls through to the materialised matmul below.
        # Hardware-agnostic path -- no SM100 TMA / swizzle machinery involved.
        comptime if has_apple_gpu_accelerator():
            from nn.conv.gpu.im2col_matmul_2d import (
                dispatch_fused_im2col_conv2d_apple,
                dispatch_im2col_matmul_conv2d,
            )

            if dispatch_fused_im2col_conv2d_apple[
                filter_is_fcrs,
                maybe_epilogue_func,
            ](
                input,
                filter,
                output,
                rebind[IndexList[2]](stride),
                rebind[IndexList[2]](dilation),
                rebind[IndexList[2]](symmetric_padding),
                num_groups,
                ctx,
            ):
                return

            # M3/M4 fallback: materialised im2col + `_matmul_gpu` (the Apple
            # FMA / 8x8 GEMM, no M5-only fragment MMA). Handles FCRS filters.
            if dispatch_im2col_matmul_conv2d[
                filter_is_fcrs,
                maybe_epilogue_func,
            ](
                input,
                filter,
                output,
                rebind[IndexList[2]](stride),
                rebind[IndexList[2]](dilation),
                rebind[IndexList[2]](symmetric_padding),
                num_groups,
                ctx,
            ):
                return

        # AMD RDNA 3+ dispatch: im2col + WMMA matmul for supported shapes.
        comptime if has_amd_rdna_gpu_accelerator() and input_type in (
            DType.bfloat16,
            DType.float16,
        ):
            from nn.conv.gpu.amd.rdna.dispatch import dispatch_rdna_conv2d

            if dispatch_rdna_conv2d[
                input_type,
                filter_type,
                output_type,
                filter_is_fcrs,
                maybe_epilogue_func=maybe_epilogue_func,
                has_residual=has_residual,
            ](
                input,
                filter,
                output,
                rebind[IndexList[2]](stride),
                rebind[IndexList[2]](dilation),
                rebind[IndexList[2]](symmetric_padding),
                num_groups,
                ctx,
                source_ptr.value() if has_residual else UnsafePointer[
                    Scalar[output_type], MutAnyOrigin
                ].unsafe_dangling(),
                beta,
            ):
                return

        # AMD MI355X (CDNA4): try the 4-wave implicit-GEMM conv first;
        # falls back to MIOpen for shapes / configs the kernel can't
        # cover. Beats MIOpen by ~1.2-2.6x on FLUX VAE / ResNet shapes
        # on MI355X (see `bench_amd_4wave_conv_vs_miopen.mojo`). When
        # `has_residual` is set, the dispatcher routes to the in-kernel
        # fused residual path (`amd_4wave_conv[has_residual=True]`).
        comptime if has_amd_gpu_accelerator():
            from nn.conv.gpu.amd.dispatch import dispatch_amd_4wave_conv2d
            from linalg.utils import elementwise_epilogue_type as _ew_2d_t

            @__parameter
            @always_inline
            def _amd_4wave_dispatch[
                _epilogue_2d: Optional[_ew_2d_t] = None,
            ]() raises -> Bool:
                return dispatch_amd_4wave_conv2d[
                    input_type,
                    filter_type,
                    output_type,
                    filter_is_fcrs,
                    has_residual=has_residual,
                    elementwise_lambda_fn=_epilogue_2d,
                ](
                    input,
                    filter,
                    output,
                    rebind[IndexList[2]](stride),
                    rebind[IndexList[2]](dilation),
                    rebind[IndexList[2]](symmetric_padding),
                    num_groups,
                    ctx,
                    source_ptr=source_ptr,
                    beta=beta,
                )

            # MIOpen-oracle audit. Gated on `MODULAR_CONV_AUDIT_MIOPEN=1`;
            # re-runs the same conv via MIOpen and emits a single line
            # with `max_abs / L1_rel / mean_abs` vs the just-produced
            # `output`.
            #
            # Subtlety: the user-supplied `maybe_epilogue_func` is a
            # `@__copy_capture(output, ...)` closure that writes its
            # result into the *real* `output` tensor (see e.g.
            # `Conv2dResidualAdd.output_fn` in kernels) — its
            # write destination is captured, not derived from the
            # `output` arg of `_conv_miopen`. So we bracket the MIOpen
            # call with snapshot/restore D2D copies: snapshot 4-wave
            # output → run MIOpen (overwrites `output` via the user's
            # epilogue) → diff snapshot vs `output` → restore `output`
            # from snapshot so downstream layers still see the 4-wave
            # result.
            #
            # Residual convs (`has_residual=True`) extend the host-side
            # diff with `+ beta * source[i]`, since MIOpen has no
            # residual path and the 4-wave kernel does the residual add
            # in-kernel. We pull `source_ptr`'s contents to host along
            # with the other two buffers and combine them in the loop.
            @__parameter
            @always_inline
            def _audit_amd_4wave_vs_miopen() raises:
                if getenv("MODULAR_CONV_AUDIT_MIOPEN", "0") != "1":
                    return

                var n_elements = output_lt.size()

                # Snapshot our 4-wave result before MIOpen overwrites
                # `output` via the user epilogue.
                var our_buf = ctx.enqueue_create_buffer[output_type](n_elements)
                var output_view = DeviceBuffer[output_type](
                    ctx, output.ptr, n_elements, owning=False
                )
                ctx.enqueue_copy(our_buf, output_view)

                # Run MIOpen + user's epilogue. The epilogue writes to
                # its captured `output`, so `output` now contains
                # MIOpen+bias.
                _conv_miopen[
                    maybe_epilogue_func=maybe_epilogue_func,
                    filter_is_fcrs=filter_is_fcrs,
                ](
                    input,
                    filter,
                    output,
                    stride,
                    dilation,
                    symmetric_padding,
                    num_groups,
                    ctx,
                )

                # Pull both to host for elementwise diff.
                var host_our = ctx.enqueue_create_host_buffer[output_type](
                    n_elements
                )
                var host_mio = ctx.enqueue_create_host_buffer[output_type](
                    n_elements
                )
                var host_src = ctx.enqueue_create_host_buffer[output_type](
                    n_elements if (
                        has_residual and source_ptr.__bool__()
                    ) else 1
                )
                ctx.enqueue_copy(host_our, our_buf)
                ctx.enqueue_copy(host_mio, output_view)

                comptime if has_residual:
                    if source_ptr:
                        var source_view = DeviceBuffer[output_type](
                            ctx, source_ptr.value(), n_elements, owning=False
                        )
                        ctx.enqueue_copy(host_src, source_view)
                ctx.synchronize()

                var max_abs: Float32 = 0
                var sum_abs_diff: Float32 = 0
                var sum_abs_ref: Float32 = 0
                # The 4-wave residual path does
                # `result = (conv + beta*source) + bias` in kernel +
                # epilogue. Our oracle here mirrors that as
                # `mio + beta*source` (MIOpen already includes +bias
                # from the user epilogue). Order of `+bias` and
                # `+beta*source` differs but they commute modulo BF16
                # rounding; the noise floor is unaffected.
                var beta_f32: Float32 = beta
                for i in range(n_elements):
                    var a = Float32(host_our[i])
                    var b = Float32(host_mio[i])

                    comptime if has_residual:
                        if source_ptr:
                            b += beta_f32 * Float32(host_src[i])

                    var d = abs(a - b)
                    if d > max_abs:
                        max_abs = d
                    sum_abs_diff += d
                    sum_abs_ref += abs(b)
                var l1_rel = (
                    sum_abs_diff / sum_abs_ref if sum_abs_ref
                    > 0 else Float32(0)
                )
                var mean_abs = sum_abs_diff / Float32(n_elements)

                var in_n = input_lt.dim[0]()
                var in_h = input_lt.dim[1]()
                var in_w = input_lt.dim[2]()
                var in_c = input_lt.dim[3]()
                var out_c = output_lt.dim[3]()
                var r_dim: Int
                var s_dim: Int

                comptime if filter_is_fcrs:
                    r_dim = filter_lt.dim[2]()
                    s_dim = filter_lt.dim[3]()
                else:
                    r_dim = filter_lt.dim[0]()
                    s_dim = filter_lt.dim[1]()

                var resid_flag = 1 if has_residual else 0
                print(
                    "[CONV_AUDIT]",
                    " N=",
                    in_n,
                    " H=",
                    in_h,
                    " W=",
                    in_w,
                    " C_in=",
                    in_c,
                    " C_out=",
                    out_c,
                    " R=",
                    r_dim,
                    " S=",
                    s_dim,
                    " stride=",
                    stride[0],
                    " has_resid=",
                    resid_flag,
                    " max_abs=",
                    max_abs,
                    " L1_rel=",
                    l1_rel,
                    " mean_abs=",
                    mean_abs,
                )

                # Restore our 4-wave output so downstream layers don't
                # see MIOpen results.
                ctx.enqueue_copy(output_view, our_buf)

                _ = our_buf^
                _ = host_our^
                _ = host_mio^
                _ = host_src^

            comptime if maybe_epilogue_func:
                # Wrap the 4D NHWC epilogue into a 2D GEMM-space
                # epilogue for the AMD 4-wave kernel. The kernel calls
                # this with (m, n) coords where
                # `m = batch*H_out*W_out + h*W_out + w` and `n = channel`.
                # Mirrors the SM100 wrapper just above.
                comptime _amd_4wave_epi = maybe_epilogue_func.value()
                var _amd_4wave_out_h = output_lt.dim[1]()
                var _amd_4wave_out_w = output_lt.dim[2]()
                var _amd_4wave_hw = _amd_4wave_out_h * _amd_4wave_out_w

                @__parameter
                @always_inline
                @__copy_capture(_amd_4wave_hw, _amd_4wave_out_w)
                def _amd_4wave_void_epilogue[
                    _dtype: DType,
                    _width: SIMDLength,
                    *,
                    alignment: Int = 1,
                ](coords_2d: IndexList[2], val: SIMD[_dtype, _width]):
                    var m = coords_2d[0]
                    var n = coords_2d[1]
                    var batch_idx: Int
                    var rem: Int
                    var h_idx: Int
                    var w_idx: Int
                    batch_idx, rem = divmod(m, _amd_4wave_hw)
                    h_idx, w_idx = divmod(rem, _amd_4wave_out_w)
                    _amd_4wave_epi(
                        IndexList[4](batch_idx, h_idx, w_idx, n),
                        rebind[SIMD[output_type, _width]](val),
                    )

                if _amd_4wave_dispatch[
                    Optional[_ew_2d_t](_amd_4wave_void_epilogue)
                ]():
                    _audit_amd_4wave_vs_miopen()
                    return
            else:
                if _amd_4wave_dispatch[]():
                    _audit_amd_4wave_vs_miopen()
                    return

        # AMD GPU path: fall back to MIOpen for conv2d.
        comptime if has_amd_gpu_accelerator():
            _conv_miopen[
                maybe_epilogue_func=maybe_epilogue_func,
                filter_is_fcrs=filter_is_fcrs,
            ](
                input,
                filter,
                output,
                stride,
                dilation,
                symmetric_padding,
                num_groups,
                ctx,
            )
            return

        # Fallback paths for non-SM100, unsupported dtypes, or constraints
        comptime if filter_is_fcrs:
            # The FCRS-filter fallback runs only on cuDNN (NVIDIA). On any
            # other GPU, guard here rather than silently entering cuDNN and
            # failing later with a confusing driver-level error. See MOCO-4172.
            comptime if not has_nvidia_gpu_accelerator():
                raise Error(
                    "conv2d: no GPU kernel for this convolution on this"
                    " device; the FCRS-filter fallback path is implemented"
                    " only via cuDNN (NVIDIA)."
                )

            # Construct row-major TileTensors for cuDNN (shared by both
            # epilogue and non-epilogue paths).
            var _in_s = input_lt.runtime_layout.shape.value.canonicalize()
            var input_rm = TileTensor(
                input.ptr,
                row_major(
                    (
                        _in_s[0],
                        _in_s[1],
                        _in_s[2],
                        _in_s[3],
                    )
                ),
            )
            var _filt_s = filter_lt.runtime_layout.shape.value.canonicalize()
            var filter_rm = TileTensor(
                filter.ptr,
                row_major(
                    (
                        _filt_s[0],
                        _filt_s[1],
                        _filt_s[2],
                        _filt_s[3],
                    )
                ),
            )

            comptime if maybe_epilogue_func:
                comptime epilogue = maybe_epilogue_func.value()
                var output_tmp_data = ctx.enqueue_create_buffer[output_type](
                    output_lt.size()
                )

                var output_tmp_lt = LayoutTensor[
                    output_type, output_layout, MutAnyOrigin
                ](
                    output_tmp_data.unsafe_ptr().as_unsafe_any_origin(),
                    output_lt.runtime_layout,
                )

                var _out_tmp_s = (
                    output_tmp_lt.runtime_layout.shape.value.canonicalize()
                )
                var output_tmp_rm = TileTensor(
                    output_tmp_lt.ptr.unsafe_origin_cast[MutAnyOrigin](),
                    row_major(
                        (
                            _out_tmp_s[0],
                            _out_tmp_s[1],
                            _out_tmp_s[2],
                            _out_tmp_s[3],
                        )
                    ),
                )

                conv_cudnn[input_type, filter_type, output_type](
                    input_rm,
                    filter_rm,
                    output_tmp_rm,
                    rebind[IndexList[2]](stride),
                    rebind[IndexList[2]](dilation),
                    rebind[IndexList[2]](symmetric_padding),
                    num_groups,
                    ctx,
                )

                @always_inline
                def epilogue_wrapper[
                    _width: Int, alignment: Int = 1
                ](coords: Coord) {var output_tmp_lt}:
                    var idx = rebind[IndexList[4]](coord_to_index_list(coords))
                    var vec = output_tmp_lt.load[width=_width](idx)
                    epilogue(idx, vec)

                elementwise[simd_width_of[output_type](), target="gpu"](
                    epilogue_wrapper,
                    Coord(output_lt.runtime_layout.shape.value),
                    ctx,
                )

                _ = output_tmp_data^

            else:
                var _out_s = output_lt.runtime_layout.shape.value.canonicalize()
                var output_rm = TileTensor(
                    output.ptr,
                    row_major(
                        (
                            _out_s[0],
                            _out_s[1],
                            _out_s[2],
                            _out_s[3],
                        )
                    ),
                )

                conv_cudnn[input_type, filter_type, output_type](
                    input_rm,
                    filter_rm,
                    output_rm,
                    rebind[IndexList[2]](stride),
                    rebind[IndexList[2]](dilation),
                    rebind[IndexList[2]](symmetric_padding),
                    num_groups,
                    ctx,
                )

        else:
            var grid_dim_x = ceildiv(
                output_lt.dim[2](), block_size
            )  # w / block size for 2d
            ctx.enqueue_function[conv_gpu_n](
                input_lt,
                filter_lt,
                output_lt,
                stride,
                dilation,
                symmetric_padding,
                Int32(num_groups),
                grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
                block_dim=(block_size, block_size),
            )

    elif input_lt.rank == 5:
        comptime if filter_is_fcrs:
            conv3d_cudnn[input_type, filter_type, output_type](
                input,
                filter,
                output,
                rebind[IndexList[3]](stride),
                rebind[IndexList[3]](dilation),
                rebind[IndexList[3]](symmetric_padding),
                num_groups,
                ctx,
            )
        else:
            # Phase A (1x1x1 fast path): direct _matmul_gpu for bf16 1x1x1
            # convs with stride=1, dilation=1, zero padding, groups=1.
            # Covers WAN post_quant_conv and every conv_shortcut.
            if dispatch_1x1x1_matmul_conv3d[
                filter_is_fcrs,
                maybe_epilogue_func,
            ](
                input,
                filter,
                output,
                rebind[IndexList[3]](stride),
                rebind[IndexList[3]](dilation),
                rebind[IndexList[3]](symmetric_padding),
                num_groups,
                ctx,
            ):
                return

            # Phase B (SM100 Q-slice): decompose the Q filter dimension
            # into Q sequential 2-D SM100 conv calls accumulated into an
            # fp32 buffer. Qualifies shapes with bf16, stride=1,
            # dilation=1, groups=1, C_in%64==0, C_out%128==0 on SM100
            # hardware. Covers WAN mid_res and time_conv; declines
            # conv_in (C_in=16) and upsampled_res (C_out=192).
            # Comptime-gated on SM100 + bf16 so the SM100 conv2d kernel
            # (which uses tcgen05 / Blackwell-only intrinsics) is not
            # instantiated when compiling for non-SM100 targets.
            comptime _is_sm100 = _is_sm10x_gpu(ctx.default_device_info)
            comptime _is_supported_dtype = input_type == DType.bfloat16
            comptime if _is_sm100 and _is_supported_dtype:
                if dispatch_qslice_conv3d_sm100[
                    filter_is_fcrs,
                    maybe_epilogue_func,
                ](
                    input,
                    filter,
                    output,
                    rebind[IndexList[3]](stride),
                    rebind[IndexList[3]](dilation),
                    rebind[IndexList[3]](symmetric_padding),
                    num_groups,
                    ctx,
                ):
                    return

            # AMD MI355X (CDNA4) native 3D implicit-GEMM: extends the
            # 4-wave conv2d loader to NDHWC inputs and Q×R×S filters.
            # Beats the im2col path by ~1.3–2.4× on WAN VAE shapes.
            # Returns False on shapes it can't cover (Q==1, C_in below
            # simd_width, C_out<64, non-square stride, FCQRS filter,
            # grouped, dilated); caller then falls through to the
            # im2col path below.
            comptime if has_amd_gpu_accelerator():
                from linalg.utils import (
                    elementwise_epilogue_type as _ew_3d_t,
                )

                # Wrap the 5D NDHWC epilogue into a 2D GEMM-space
                # void epilogue for the 4-wave kernel. The kernel
                # calls this with (m, n) coords where m flattens
                # batch*d*h*w and n is the channel.
                comptime if maybe_epilogue_func:
                    comptime _amd_3d_epi_5d = maybe_epilogue_func.value()
                    var _amd_3d_D_out = output_lt.dim[1]()
                    var _amd_3d_H_out = output_lt.dim[2]()
                    var _amd_3d_W_out = output_lt.dim[3]()
                    var _amd_3d_HW = _amd_3d_H_out * _amd_3d_W_out
                    var _amd_3d_DHW = _amd_3d_D_out * _amd_3d_HW

                    @__parameter
                    @always_inline
                    @__copy_capture(_amd_3d_DHW, _amd_3d_HW, _amd_3d_W_out)
                    def amd_3d_void_epilogue[
                        _dtype: DType,
                        _width: SIMDLength,
                        *,
                        alignment: Int = 1,
                    ](coords_2d: IndexList[2], val: SIMD[_dtype, _width],):
                        var m = coords_2d[0]
                        var n = coords_2d[1]
                        var batch_idx: Int
                        var rem: Int
                        var d_idx: Int
                        var rem2: Int
                        var h_idx: Int
                        var w_idx: Int
                        batch_idx, rem = divmod(m, _amd_3d_DHW)
                        d_idx, rem2 = divmod(rem, _amd_3d_HW)
                        h_idx, w_idx = divmod(rem2, _amd_3d_W_out)
                        _amd_3d_epi_5d(
                            IndexList[5](batch_idx, d_idx, h_idx, w_idx, n),
                            rebind[SIMD[output_type, _width]](val),
                        )

                    if dispatch_amd_4wave_conv3d[
                        input_type,
                        filter_type,
                        output_type,
                        filter_is_fcqrs=filter_is_fcrs,
                        elementwise_lambda_fn=Optional[_ew_3d_t](
                            amd_3d_void_epilogue
                        ),
                    ](
                        input,
                        filter,
                        output,
                        rebind[IndexList[3]](stride),
                        rebind[IndexList[3]](dilation),
                        rebind[IndexList[3]](symmetric_padding),
                        num_groups,
                        ctx,
                    ):
                        return
                else:
                    if dispatch_amd_4wave_conv3d[
                        input_type,
                        filter_type,
                        output_type,
                        filter_is_fcqrs=filter_is_fcrs,
                    ](
                        input,
                        filter,
                        output,
                        rebind[IndexList[3]](stride),
                        rebind[IndexList[3]](dilation),
                        rebind[IndexList[3]](symmetric_padding),
                        num_groups,
                        ctx,
                    ):
                        return

            # Phase 2 path: explicit im2col + _matmul_gpu for bf16 3D convs.
            # Covers 3x3x3, 3x1x1, etc. and falls back to the naive kernel on
            # shapes it can't handle (grouped, dilated, non-bf16, etc.).
            if dispatch_im2col_matmul_conv3d[
                filter_is_fcrs,
                maybe_epilogue_func,
            ](
                input,
                filter,
                output,
                rebind[IndexList[3]](stride),
                rebind[IndexList[3]](dilation),
                rebind[IndexList[3]](symmetric_padding),
                num_groups,
                ctx,
            ):
                return

            var grid_dim_x = ceildiv(
                output_lt.dim[2]() * output_lt.dim[3](), block_size
            )  # h * w / block size for 3d
            ctx.enqueue_function[conv_gpu_3d](
                input_lt,
                filter_lt,
                output_lt,
                stride,
                dilation,
                symmetric_padding,
                Int32(num_groups),
                grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
                block_dim=(block_size, block_size),
            )


@__name(
    t"conv3d_gpu_naive_ndhwc_qrscf_{input_type}_{filter_type}_{output_type}",
)
def conv3d_gpu_naive_ndhwc_qrscf[
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    block_size: Int,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type],
](
    input: LayoutTensor[input_type, input_layout, MutAnyOrigin],
    filter: LayoutTensor[filter_type, filter_layout, MutAnyOrigin],
    output: LayoutTensor[output_type, output_layout, MutAnyOrigin],
    stride: IndexList[3],
    dilation: IndexList[3],
    padding: IndexList[3],
    num_groups: Int32,
):
    """Naive GPU kernel for 3D NDHWC convolution with QRSCF filter layout.

    Each thread computes one output voxel across all output channels,
    iterating over the Q x R x S filter window with vectorized input
    loads and scalar filter accumulation.

    Parameters:
        input_layout: Memory layout of the input tensor (`NDHWC`).
        filter_layout: Memory layout of the filter tensor (`QRSCF`).
        output_layout: Memory layout of the output tensor (`NDHWC`).
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.
        block_size: Thread block size used for both `x` and `y` block
            dimensions.
        maybe_epilogue_func: Optional elementwise SIMD epilogue applied to
            each computed output value in place of a direct store.

    Arguments:
        input: Input activation tensor in `NDHWC` layout.
        filter: Convolution weight tensor in `QRSCF` layout.
        output: Output activation tensor in `NDHWC` layout.
        stride: Per-spatial-axis convolution stride as `(depth, height,
            width)`.
        dilation: Per-spatial-axis filter dilation as `(depth, height,
            width)`.
        padding: Per-spatial-axis symmetric padding as `(depth, height,
            width)`.
        num_groups: Number of convolution groups for grouped convolution.
    """
    var _num_groups = Int(num_groups)
    var N = input.dim[0]()
    var D = input.dim[1]()  # depth
    var H = input.dim[2]()
    var W = input.dim[3]()
    var C_in = input.dim[4]()  # channel_input

    var Q = filter.dim[0]()
    var R = filter.dim[1]()
    var S = filter.dim[2]()
    var C_per_group = filter.dim[3]()  # C_in / _num_groups

    var D_out = output.dim[1]()  # depth
    var H_out = output.dim[2]()
    var W_out = output.dim[3]()
    var C_out = output.dim[4]()  # channel_output
    var F_per_group = C_out // _num_groups

    var pad_d = padding[0]
    var pad_h = padding[1]
    var pad_w = padding[2]

    var stride_d = stride[0]
    var stride_h = stride[1]
    var stride_w = stride[2]

    var dil_d = dilation[0]
    var dil_h = dilation[1]
    var dil_w = dilation[2]

    var n = block_idx.z  # batch dimension (unchanged)
    # calculate the linear thread id in x-dimension (width*height)
    var x_thread_id = block_idx.x * block_dim.x + thread_idx.x

    # map back to separate height and width
    var h_out_idx, w_out_idx = udivmod(x_thread_id, W_out)

    # calculate depth from y-dimension
    var d_out_idx = block_idx.y * block_dim.y + thread_idx.y

    # bounds check
    if n >= N or d_out_idx >= D_out or h_out_idx >= H_out or w_out_idx >= W_out:
        return

    # ============= convolution =============
    # Input is NDHWC (C innermost, contiguous). Vectorize C_in loads as 128-bit
    # wide SIMD reads. Filter is QRSCF (F innermost, so the C axis is strided
    # by F); we keep scalar filter loads since they're cache-hot across
    # repeated use but assemble them into a SIMD register so the dot product
    # uses a single fused multiply-add chain per chunk.
    comptime accum_type = get_accum_type[output_type]()
    # 128-bit load is the widest single-thread GPU transaction; pick the
    # element count that fills it for this dtype.
    comptime vec_w = 16 // size_of[input_type]()

    for co in range(C_out):
        var g = co // F_per_group
        var ci_base = g * C_per_group
        var simd_value = SIMD[accum_type, vec_w](0)
        var scalar_value = Scalar[accum_type](0)

        for q in range(Q):
            for r in range(R):
                for s in range(S):
                    var d_in = d_out_idx * stride_d + q * dil_d - pad_d
                    var h_in = h_out_idx * stride_h + r * dil_h - pad_h
                    var w_in = w_out_idx * stride_w + s * dil_w - pad_w

                    if 0 <= d_in < D and 0 <= h_in < H and 0 <= w_in < W:
                        var ci = 0
                        while ci + vec_w <= C_per_group:
                            var in_vec = input.load[width=vec_w](
                                IndexList[5](n, d_in, h_in, w_in, ci_base + ci)
                            ).cast[accum_type]()
                            var flt_vec = SIMD[accum_type, vec_w](0)

                            comptime for k in range(vec_w):
                                flt_vec[k] = filter.load[width=1](
                                    IndexList[5](q, r, s, ci + k, co)
                                )[0].cast[accum_type]()
                            simd_value = simd_value + in_vec * flt_vec
                            ci += vec_w
                        while ci < C_per_group:
                            scalar_value += (
                                input.load[width=1](
                                    IndexList[5](
                                        n, d_in, h_in, w_in, ci_base + ci
                                    )
                                )[0].cast[accum_type]()
                                * filter.load[width=1](
                                    IndexList[5](q, r, s, ci, co)
                                )[0].cast[accum_type]()
                            )
                            ci += 1

        var value = simd_value.reduce_add() + scalar_value

        comptime if maybe_epilogue_func:
            comptime epilogue_func = maybe_epilogue_func.value()
            epilogue_func(
                IndexList[5](n, d_out_idx, h_out_idx, w_out_idx, co),
                value.cast[output_type](),
            )
        else:
            output.store(
                IndexList[5](n, d_out_idx, h_out_idx, w_out_idx, co),
                value.cast[output_type](),
            )


# ===----------------------------------------------------------------------=== #
# GPU 3D Convolution using cuDNN (Nd APIs)                                     #
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _Conv3dAlgoCacheEntry(Copyable, Movable):
    """Cached cuDNN algorithm selection result for a conv3d shape."""

    var algo_value: Int8
    var workspace_size: Int

    def algo(self) -> cudnnConvolutionFwdAlgo_t:
        return rebind[cudnnConvolutionFwdAlgo_t](self.algo_value)


def _conv3d_cudnn_depth_tiled[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[mut=False, input_type, ...],
    filter: TileTensor[mut=False, filter_type, ...],
    output: TileTensor[output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    """Depth-tiled cuDNN 3D convolution for tensors exceeding INT32_MAX elements.

    Splits the computation along the depth dimension (dim[1] in NDHWC) into
    tiles small enough for cuDNN's internal Int32 stride calculations.
    Each tile uses a separate set of cuDNN descriptors.
    """
    comptime INT32_MAX_VAL = 2147483647
    comptime FIND_WS_CAP = 256 * 1024 * 1024

    # TileTensor.dim[N]() returns Scalar[tensor.linear_idx_type]; wrap in Int()
    # so the index arithmetic below unifies across input/filter/output (each
    # carries a distinct linear_idx_type that does not auto-unify).
    var N = Int(input.dim[0]())
    var D_in = Int(input.dim[1]())
    var H = Int(input.dim[2]())
    var W = Int(input.dim[3]())
    var C = Int(input.dim[4]())

    var K_d = Int(filter.dim[2]())  # kernel depth (Q in FCQRS)
    var F_out = Int(filter.dim[0]())  # output channels
    var D_out = Int(output.dim[1]())
    var H_out = Int(output.dim[2]())
    var W_out = Int(output.dim[3]())

    var eff_k = (K_d - 1) * dilation[0] + 1  # effective kernel depth

    # Calculate max input depth per tile.
    var per_frame_in = N * H * W * C
    var max_d_in = INT32_MAX_VAL // per_frame_in

    # Also ensure output elements per tile fit in INT32.
    var per_frame_out = N * H_out * W_out * F_out
    var max_d_out = INT32_MAX_VAL // per_frame_out
    # Output frames from max_d_in input frames:
    var tile_d_out_from_in = (max_d_in + 2 * padding[0] - eff_k) // stride[
        0
    ] + 1
    var tile_d_out = min(tile_d_out_from_in, max_d_out)
    if tile_d_out < 1:
        raise "conv3d: tensor too large even for single-frame tiling"

    # Input depth needed for tile_d_out output frames.
    var tile_d_in = (tile_d_out - 1) * stride[0] + eff_k - 2 * padding[0]

    # Strides (in elements) along the depth dimension.
    var in_d_stride = H * W * C  # elements per depth frame
    var out_d_stride = H_out * W_out * F_out

    var ptr_meta = _get_cudnn_meta(ctx)

    # Descriptor arrays (reused across tiles).
    var input_dims_alloc = alloc[Int32]({count = 5}).into_managed()
    var input_dims = input_dims_alloc.unsafe_ptr()
    var output_dims_alloc = alloc[Int32]({count = 5}).into_managed()
    var output_dims = output_dims_alloc.unsafe_ptr()
    var filter_dims_alloc = alloc[Int32]({count = 5}).into_managed()
    var filter_dims = filter_dims_alloc.unsafe_ptr()
    var pad_a_alloc = alloc[Int32]({count = 3}).into_managed()
    var pad_a = pad_a_alloc.unsafe_ptr()
    var stride_a_alloc = alloc[Int32]({count = 3}).into_managed()
    var stride_a = stride_a_alloc.unsafe_ptr()
    var dilation_a_alloc = alloc[Int32]({count = 3}).into_managed()
    var dilation_a = dilation_a_alloc.unsafe_ptr()

    # Filter dims (constant across tiles).
    filter_dims[0] = Int32(filter.dim[0]())
    filter_dims[1] = Int32(filter.dim[1]())
    filter_dims[2] = Int32(filter.dim[2]())
    filter_dims[3] = Int32(filter.dim[3]())
    filter_dims[4] = Int32(filter.dim[4]())

    check_cudnn_error(
        cudnnSetFilterNdDescriptor(
            ptr_meta[].ptr_filter_desc,
            get_cudnn_dtype[filter_type](),
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            Int16(5),
            filter_dims.bitcast[NoneType](),
        )
    )

    # Convolution params (constant except padding for first tile).
    stride_a[0] = Int32(stride[0])
    stride_a[1] = Int32(stride[1])
    stride_a[2] = Int32(stride[2])
    dilation_a[0] = Int32(dilation[0])
    dilation_a[1] = Int32(dilation[1])
    dilation_a[2] = Int32(dilation[2])

    var alpha = Float32(1.0)
    var beta = Float32(0.0)

    var d_out_start = 0
    while d_out_start < D_out:
        var this_d_out = min(tile_d_out, D_out - d_out_start)

        # Determine input range for this output tile.
        # First tile gets front padding, last tile gets back padding.
        var d_in_start: Int
        var this_d_in: Int
        var tile_pad_front: Int
        var tile_pad_back: Int

        if d_out_start == 0:
            # First tile: include front padding.
            tile_pad_front = padding[0]
            d_in_start = 0
            this_d_in = (
                (this_d_out - 1) * stride[0] + eff_k - 2 * tile_pad_front
            )
            # Adjust: no need for more input than available
            if this_d_in > D_in:
                this_d_in = D_in
            tile_pad_back = 0
        else:
            tile_pad_front = 0
            # For stride=1: input frame for output d is at d (with padding=0)
            d_in_start = d_out_start * stride[0] - padding[0]
            if d_in_start < 0:
                tile_pad_front = -d_in_start
                d_in_start = 0
            this_d_in = (this_d_out - 1) * stride[0] + eff_k - tile_pad_front
            # Check if we need back padding
            if d_in_start + this_d_in > D_in:
                tile_pad_back = d_in_start + this_d_in - D_in
                this_d_in = D_in - d_in_start
            else:
                tile_pad_back = 0

        # --- Set up tile descriptors ---
        # Input tile: [N, this_d_in, H, W, C]
        input_dims[0] = Int32(N)
        input_dims[1] = Int32(C)
        input_dims[2] = Int32(this_d_in)
        input_dims[3] = Int32(H)
        input_dims[4] = Int32(W)

        check_cudnn_error(
            cudnnSetTensorNdDescriptorEx(
                ptr_meta[].ptr_input_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                get_cudnn_dtype[input_type](),
                Int16(5),
                input_dims.bitcast[NoneType](),
            )
        )

        # Output tile: [N, this_d_out, H_out, W_out, F]
        output_dims[0] = Int32(N)
        output_dims[1] = Int32(F_out)
        output_dims[2] = Int32(this_d_out)
        output_dims[3] = Int32(H_out)
        output_dims[4] = Int32(W_out)

        check_cudnn_error(
            cudnnSetTensorNdDescriptorEx(
                ptr_meta[].ptr_output_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                get_cudnn_dtype[output_type](),
                Int16(5),
                output_dims.bitcast[NoneType](),
            )
        )

        # Convolution with tile-specific depth padding.
        pad_a[0] = Int32(tile_pad_front)
        pad_a[1] = Int32(padding[1])
        pad_a[2] = Int32(padding[2])

        check_cudnn_error(
            cudnnSetConvolutionNdDescriptor(
                ptr_meta[].ptr_conv_desc,
                Int16(3),
                pad_a.bitcast[NoneType](),
                stride_a.bitcast[NoneType](),
                dilation_a.bitcast[NoneType](),
                cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
            )
        )
        check_cudnn_error(
            cudnnSetConvolutionGroupCount(
                ptr_meta[].ptr_conv_desc, Int16(num_groups)
            )
        )
        check_cudnn_error(
            cudnnSetConvolutionMathType(
                ptr_meta[].ptr_conv_desc,
                cudnnMathType_t.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
            )
        )

        # --- Algorithm selection (use GetWorkspaceSize for PRECOMP_GEMM) ---
        var algo = (
            cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        )
        var ws_size: Int = 0
        var ws_st = cudnnGetConvolutionForwardWorkspaceSize(
            ptr_meta[].ptr_handle,
            ptr_meta[].ptr_input_desc,
            ptr_meta[].ptr_filter_desc,
            ptr_meta[].ptr_conv_desc,
            ptr_meta[].ptr_output_desc,
            algo,
            UnsafePointer(to=ws_size),
        )
        if ws_st != cudnnStatus_t.CUDNN_STATUS_SUCCESS or ws_size > FIND_WS_CAP:
            # Fall back to IMPLICIT_GEMM (no workspace needed).
            algo = (
                cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            )
            ws_size = 0

        # --- Execute tile ---
        var workspace_buffer = ctx.enqueue_create_buffer[.uint8](ws_size)

        # Compute pointer offsets for input and output tiles.
        var in_offset = d_in_start * in_d_stride
        var out_offset = d_out_start * out_d_stride
        var in_ptr = input.ptr + in_offset
        var out_ptr = output.ptr + out_offset

        var fwd_status = cudnnConvolutionForward(
            ptr_meta[].ptr_handle,
            UnsafePointer(to=alpha).bitcast[NoneType](),
            ptr_meta[].ptr_input_desc,
            in_ptr.bitcast[NoneType](),
            ptr_meta[].ptr_filter_desc,
            filter.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_conv_desc,
            algo,
            workspace_buffer.unsafe_ptr().bitcast[NoneType](),
            ws_size,
            UnsafePointer(to=beta).bitcast[NoneType](),
            ptr_meta[].ptr_output_desc,
            out_ptr.bitcast[NoneType](),
        )
        _ = workspace_buffer^

        if fwd_status != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
            dealloc(input_dims_alloc^)
            dealloc(output_dims_alloc^)
            dealloc(filter_dims_alloc^)
            dealloc(pad_a_alloc^)
            dealloc(stride_a_alloc^)
            dealloc(dilation_a_alloc^)
            ctx.synchronize()
            raise String("conv3d tiled forward failed: ", fwd_status)

        d_out_start += this_d_out

    # Clean up.
    dealloc(input_dims_alloc^)
    dealloc(output_dims_alloc^)
    dealloc(filter_dims_alloc^)
    dealloc(pad_a_alloc^)
    dealloc(stride_a_alloc^)
    dealloc(dilation_a_alloc^)


def _conv3d_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[mut=False, input_type, ...],
    filter: TileTensor[mut=False, filter_type, ...],
    output: TileTensor[output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    """cuDNN 3D convolution using Nd descriptor APIs.

    Expects:
      - input:  NDHWC layout [N, D, H, W, C]
      - filter: FCQRS layout [F, C/groups, Q, R, S]
      - output: NDHWC layout [N, D_out, H_out, W_out, F]

    Algorithm selection is cached per unique shape+params combination so that
    the expensive FindEx search only runs once per shape.

    When the total number of elements exceeds INT32_MAX (~2.1B), cuDNN's
    internal stride calculations overflow. In this case we tile along the
    depth (D) dimension, processing each tile with a separate cuDNN call.
    """
    comptime FIND_WS_CAP = 256 * 1024 * 1024
    comptime INT32_MAX_VAL = 2147483647

    # --- Check if depth tiling is needed (INT32 stride overflow) ---
    var total_in = (
        input.dim[0]()
        * input.dim[1]()
        * input.dim[2]()
        * input.dim[3]()
        * input.dim[4]()
    )
    if total_in > INT32_MAX_VAL:
        _conv3d_cudnn_depth_tiled(
            input,
            filter,
            output,
            stride,
            dilation,
            padding,
            num_groups,
            ctx,
        )
        return

    var ptr_meta = _get_cudnn_meta(ctx)

    # --- Set up cuDNN descriptors (required every call — shared state) ---
    # Input: NDHWC in memory, described as NHWC format with dims [N,C,D,H,W].
    var input_dims_alloc = alloc[Int32]({count = 5}).into_managed()
    var input_dims = input_dims_alloc.unsafe_ptr()
    input_dims[0] = Int32(input.dim[0]())  # N
    input_dims[1] = Int32(input.dim[4]())  # C
    input_dims[2] = Int32(input.dim[1]())  # D
    input_dims[3] = Int32(input.dim[2]())  # H
    input_dims[4] = Int32(input.dim[3]())  # W

    check_cudnn_error(
        cudnnSetTensorNdDescriptorEx(
            ptr_meta[].ptr_input_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
            get_cudnn_dtype[input_type](),
            Int16(5),
            input_dims.bitcast[NoneType](),
        )
    )

    # Filter: FCQRS layout [F, C/groups, Q, R, S], described as NCHW format.
    var filter_dims_alloc = alloc[Int32]({count = 5}).into_managed()
    var filter_dims = filter_dims_alloc.unsafe_ptr()
    filter_dims[0] = Int32(filter.dim[0]())  # F (out_channels)
    filter_dims[1] = Int32(filter.dim[1]())  # C (in_channels / groups)
    filter_dims[2] = Int32(filter.dim[2]())  # Q (depth)
    filter_dims[3] = Int32(filter.dim[3]())  # R (height)
    filter_dims[4] = Int32(filter.dim[4]())  # S (width)

    check_cudnn_error(
        cudnnSetFilterNdDescriptor(
            ptr_meta[].ptr_filter_desc,
            get_cudnn_dtype[filter_type](),
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            Int16(5),
            filter_dims.bitcast[NoneType](),
        )
    )

    # Convolution: 3 spatial dimensions.
    var pad_a_alloc = alloc[Int32]({count = 3}).into_managed()
    var pad_a = pad_a_alloc.unsafe_ptr()
    pad_a[0] = Int32(padding[0])
    pad_a[1] = Int32(padding[1])
    pad_a[2] = Int32(padding[2])

    var stride_a_alloc = alloc[Int32]({count = 3}).into_managed()
    var stride_a = stride_a_alloc.unsafe_ptr()
    stride_a[0] = Int32(stride[0])
    stride_a[1] = Int32(stride[1])
    stride_a[2] = Int32(stride[2])

    var dilation_a_alloc = alloc[Int32]({count = 3}).into_managed()
    var dilation_a = dilation_a_alloc.unsafe_ptr()
    dilation_a[0] = Int32(dilation[0])
    dilation_a[1] = Int32(dilation[1])
    dilation_a[2] = Int32(dilation[2])

    check_cudnn_error(
        cudnnSetConvolutionNdDescriptor(
            ptr_meta[].ptr_conv_desc,
            Int16(3),
            pad_a.bitcast[NoneType](),
            stride_a.bitcast[NoneType](),
            dilation_a.bitcast[NoneType](),
            cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
        )
    )

    check_cudnn_error(
        cudnnSetConvolutionGroupCount(
            ptr_meta[].ptr_conv_desc, Int16(num_groups)
        )
    )

    # Output: NDHWC in memory, described as NHWC format with dims [N,C,D,H,W].
    var output_dims_alloc = alloc[Int32]({count = 5}).into_managed()
    var output_dims = output_dims_alloc.unsafe_ptr()
    output_dims[0] = Int32(output.dim[0]())  # N
    output_dims[1] = Int32(output.dim[4]())  # C (out_channels)
    output_dims[2] = Int32(output.dim[1]())  # D_out
    output_dims[3] = Int32(output.dim[2]())  # H_out
    output_dims[4] = Int32(output.dim[3]())  # W_out

    check_cudnn_error(
        cudnnSetTensorNdDescriptorEx(
            ptr_meta[].ptr_output_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
            get_cudnn_dtype[output_type](),
            Int16(5),
            output_dims.bitcast[NoneType](),
        )
    )

    # Allow tensor-op math with automatic type conversion — required for
    # bfloat16 3D convolutions on modern cuDNN (matches PR #5988 approach).
    check_cudnn_error(
        cudnnSetConvolutionMathType(
            ptr_meta[].ptr_conv_desc,
            cudnnMathType_t.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
        )
    )

    # --- Algorithm selection (cached per shape) ---
    var cache_key = String(
        "CONV3D_ALGO_",
        ctx.id(),
        "_",
        input.dim[0](),
        "_",
        input.dim[4](),
        "_",
        input.dim[1](),
        "_",
        input.dim[2](),
        "_",
        input.dim[3](),
        "_F",
        filter.dim[0](),
        "_",
        filter.dim[1](),
        "_",
        filter.dim[2](),
        "_",
        filter.dim[3](),
        "_",
        filter.dim[4](),
        "_p",
        padding[0],
        "_",
        padding[1],
        "_",
        padding[2],
        "_s",
        stride[0],
        "_",
        stride[1],
        "_",
        stride[2],
        "_d",
        dilation[0],
        "_",
        dilation[1],
        "_",
        dilation[2],
        "_g",
        num_groups,
    )

    var algo: cudnnConvolutionFwdAlgo_t
    var workspace_size_var: Int

    var ptr_cached = _get_global_or_null(cache_key)
    if ptr_cached:
        # Cache hit — reuse previously selected algorithm.
        var entry = ptr_cached.unsafe_value().unsafe_bitcast[
            _Conv3dAlgoCacheEntry
        ]()
        algo = entry[].algo()
        workspace_size_var = entry[].workspace_size
    else:
        # Cache miss — run FindEx to find the fastest algorithm.
        var find_ws = ctx.enqueue_create_buffer[.uint8](FIND_WS_CAP)

        # CRITICAL: The Mojo cudnnConvolutionFwdAlgoPerfStruct uses Int8 for
        # enum fields, but the C struct uses int (4 bytes). This causes a
        # size mismatch: Mojo struct = ~32 bytes, C struct = 48 bytes.
        # Allocating with the Mojo struct size would cause a buffer overflow
        # when cuDNN writes 8 * 48 = 384 bytes. We allocate raw bytes with
        # the correct C struct size and read fields at proper offsets.
        comptime C_PERF_STRUCT_SIZE = 48  # sizeof(cudnnConvolutionFwdAlgoPerf_t)
        comptime MAX_ALGOS = 8
        var perf_bytes_alloc = alloc[UInt8](
            {count = MAX_ALGOS * C_PERF_STRUCT_SIZE}
        ).into_managed()
        var perf_bytes = perf_bytes_alloc.unsafe_ptr()

        # returned_algo_count is int* in C (4 bytes), not Int16*.
        # Use Int32 and bitcast the pointer.
        var returned_count_i32 = Int32(0)

        var find_status = cudnnFindConvolutionForwardAlgorithmEx(
            ptr_meta[].ptr_handle,
            ptr_meta[].ptr_input_desc,
            input.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_filter_desc,
            filter.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_conv_desc,
            ptr_meta[].ptr_output_desc,
            output.ptr.bitcast[NoneType](),
            Int16(MAX_ALGOS),
            UnsafePointer(to=returned_count_i32).bitcast[Int16](),
            perf_bytes.unsafe_bitcast[cudnnConvolutionFwdAlgoPerfStruct](),
            find_ws.unsafe_ptr().bitcast[NoneType](),
            FIND_WS_CAP,
        )
        _ = find_ws^

        # Read the returned count (C int at offset 0 of returned_count_i32).
        var returned_count = Int(returned_count_i32)

        # Pick the fastest successful algorithm within workspace cap.
        # Read fields from raw bytes at correct C struct offsets:
        #   offset  0: algo (int32)
        #   offset  4: status (int32)
        #   offset  8: time (float32)
        #   offset 16: memory (size_t / int64)
        algo = (
            cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        )
        workspace_size_var = 0

        var find_status_val = rebind[Int32](find_status)
        if find_status_val == 0:  # CUDNN_STATUS_SUCCESS
            for i in range(returned_count):
                var base = perf_bytes.unsafe_offset(i * C_PERF_STRUCT_SIZE)
                var algo_val = base.unsafe_bitcast[Int32]()[]  # offset 0
                var status_val = base.unsafe_offset(4).unsafe_bitcast[
                    Int32
                ]()[]  # offset 4
                var memory_val = base.unsafe_offset(16).unsafe_bitcast[
                    Int
                ]()[]  # offset 16
                if status_val == 0 and memory_val <= FIND_WS_CAP:
                    algo = rebind[cudnnConvolutionFwdAlgo_t](Int8(algo_val))
                    workspace_size_var = memory_val
                    break
        else:
            print(
                "conv3d FindEx FAILED: status=",
                Int(find_status_val),
                " input=[N=",
                input.dim[0](),
                " C=",
                input.dim[4](),
                " D=",
                input.dim[1](),
                " H=",
                input.dim[2](),
                " W=",
                input.dim[3](),
                "]",
            )
        # Fallback: if FindEx found nothing useful, try PRECOMP_GEMM via
        # workspace size query (cheaper than FindEx).
        if (
            algo
            == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            and workspace_size_var == 0
        ):
            var precomp = (
                cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            )
            var ws_size: Int = 0
            var ws_st = cudnnGetConvolutionForwardWorkspaceSize(
                ptr_meta[].ptr_handle,
                ptr_meta[].ptr_input_desc,
                ptr_meta[].ptr_filter_desc,
                ptr_meta[].ptr_conv_desc,
                ptr_meta[].ptr_output_desc,
                precomp,
                UnsafePointer(to=ws_size),
            )
            if (
                ws_st == cudnnStatus_t.CUDNN_STATUS_SUCCESS
                and ws_size <= FIND_WS_CAP
            ):
                algo = precomp
                workspace_size_var = ws_size

        # Store result in global cache.
        var ptr_entry = alloc[_Conv3dAlgoCacheEntry]({count = 1}).unsafe_leak()
        ptr_entry.unsafe_write(
            _Conv3dAlgoCacheEntry(
                algo_value=rebind[Int8](algo),
                workspace_size=workspace_size_var,
            )
        )
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(cache_key),
            ptr_entry.unsafe_bitcast[NoneType](),
        )

    # --- Execute convolution with cached/selected algorithm ---
    var alpha = Float32(1.0)
    var beta = Float32(0.0)

    var workspace_buffer = ctx.enqueue_create_buffer[.uint8](workspace_size_var)
    var fwd_status = cudnnConvolutionForward(
        ptr_meta[].ptr_handle,
        UnsafePointer(to=alpha).bitcast[NoneType](),
        ptr_meta[].ptr_input_desc,
        input.ptr.bitcast[NoneType](),
        ptr_meta[].ptr_filter_desc,
        filter.ptr.bitcast[NoneType](),
        ptr_meta[].ptr_conv_desc,
        algo,
        workspace_buffer.unsafe_ptr().bitcast[NoneType](),
        workspace_size_var,
        UnsafePointer(to=beta).bitcast[NoneType](),
        ptr_meta[].ptr_output_desc,
        output.ptr.bitcast[NoneType](),
    )
    # Free workspace BEFORE sync to release the buffer back to the pool.
    _ = workspace_buffer^

    # The temporary descriptor arrays free themselves (ManagedAllocation).

    # Retry with IMPLICIT_GEMM + zero workspace on allocation failures.
    # cuDNN may allocate internal scratch beyond what GetWorkspaceSize
    # reports (filter reorder, NHWC/tensor-op padding). When that trips at
    # execute time, fall back to the zero-workspace algorithm and update
    # the shape cache so we don't hit the same OOM on future calls.
    comptime IMPLICIT_GEMM = (
        cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
    )
    if (
        fwd_status == cudnnStatus_t.CUDNN_STATUS_ALLOC_FAILED
        or fwd_status == cudnnStatus_t.CUDNN_STATUS_ALLOC_FAILED_V9
        or fwd_status == cudnnStatus_t.CUDNN_STATUS_ALLOC_FAILED_DEVICE_MEMORY
        or fwd_status == cudnnStatus_t.CUDNN_STATUS_ALLOC_FAILED_HOST_MEMORY
    ) and algo != IMPLICIT_GEMM:
        ctx.synchronize()  # Flush pending work and reclaim held memory.
        algo = IMPLICIT_GEMM
        workspace_size_var = 0

        var retry_workspace = ctx.enqueue_create_buffer[.uint8](0)
        fwd_status = cudnnConvolutionForward(
            ptr_meta[].ptr_handle,
            UnsafePointer(to=alpha).bitcast[NoneType](),
            ptr_meta[].ptr_input_desc,
            input.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_filter_desc,
            filter.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_conv_desc,
            algo,
            retry_workspace.unsafe_ptr().bitcast[NoneType](),
            0,
            UnsafePointer(to=beta).bitcast[NoneType](),
            ptr_meta[].ptr_output_desc,
            output.ptr.bitcast[NoneType](),
        )
        _ = retry_workspace^

        if fwd_status == cudnnStatus_t.CUDNN_STATUS_SUCCESS:
            # Persist the safer algorithm for this shape so subsequent
            # calls skip the OOM-prone pick. InsertGlobal overwrites the
            # existing entry keyed by cache_key.
            var retry_entry = alloc[_Conv3dAlgoCacheEntry](
                {count = 1}
            ).unsafe_leak()
            retry_entry.unsafe_write(
                _Conv3dAlgoCacheEntry(
                    algo_value=rebind[Int8](algo),
                    workspace_size=0,
                )
            )
            external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
                StringSlice(cache_key),
                retry_entry.unsafe_bitcast[NoneType](),
            )

    if fwd_status != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
        # Synchronize device to flush any pending GPU operations and free
        # temporary cuDNN allocations, preventing VRAM accumulation.
        print("conv3d FORWARD FAILED: ", fwd_status, " algo=", algo)
        ctx.synchronize()
        raise String("cudnnConvolutionForward failed: ", fwd_status)


def conv3d_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[mut=False, input_type, ...],
    filter: TileTensor[mut=False, filter_type, ...],
    output: TileTensor[output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    """Runs a 3D convolution via cuDNN using Nd descriptor APIs, activating
    the device context before dispatching.

    Parameters:
        input_type: Element type of the input tensor.
        filter_type: Element type of the filter tensor.
        output_type: Element type of the output tensor.

    Arguments:
        input: Input activation tensor in `NDHWC` layout.
        filter: Convolution weight tensor in `FCQRS` layout.
        output: Output activation tensor in `NDHWC` layout.
        stride: Per-spatial-axis convolution stride as `(depth, height,
            width)`.
        dilation: Per-spatial-axis filter dilation as `(depth, height,
            width)`.
        padding: Per-spatial-axis symmetric padding as `(depth, height,
            width)`.
        num_groups: Number of convolution groups for grouped convolution.
        ctx: Device context activated and used to dispatch the cuDNN
            call.
    """
    # Set `ctx`'s CUcontext as current to satisfy cudnn's stateful API.
    with ctx.push_context() as ctx:
        _conv3d_cudnn(
            input, filter, output, stride, dilation, padding, num_groups, ctx
        )
