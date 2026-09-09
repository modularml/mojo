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

"""Provides batched matrix multiplication (BMM) for CPU and GPU targets."""

from std.math import align_up, ceildiv, gcd
from std.sys import align_of, size_of
from std.sys.info import (
    _has_blackwell_tcgen05,
    _is_amd_rdna,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
    simd_width_of,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from max.algorithm import elementwise, sync_parallelize
from max.algorithm.functional import _get_start_indices_of_nth_subvolume
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import A100, is_cpu, is_valid_target
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TensorLayout,
    TileTensor,
    lt_to_tt,
    coord_to_index_list,
    row_major,
)
from layout.tma_async import TMATensorTile, create_tensor_tile
from layout.tile_layout import Layout as TileLayout
from std.logger import Logger
from std.memory import dealloc
from std.memory.alloc import Layout as AllocLayout
from max.runtime.asyncrt import parallelism_level
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg
from max.gpu.host.info import H100, _is_sm10x_gpu
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple
from .matmul.cpu.apple_accelerate import (
    apple_batched_matmul,
    use_apple_accelerate_lib,
)
from .matmul.cpu.impl import _submatmul_sequential_sync
from .matmul.gpu import _matmul_gpu, _amdgpu_get_mma_shape
from .matmul.gpu._multistage_gemm_gpu import multistage_gemm_kernel
from .matmul.gpu.amd import AMDMatmul
from .matmul.gpu.sm100.blockwise_fp8 import (
    matmul_sm100_blockwise_scaled_fp8_1d2d_kernel,
)
from .matmul.gpu.sm100_structured.default.dispatch import (
    dispatch_sm100_batched_matmul,
)
from .utils import GemmShape
from .utils import elementwise_epilogue_type as matmul_elementwise_epilogue_type
from .utils import (
    get_kernel_config,
    get_kernel_type,
    get_matmul_num_tasks,
    get_min_task_size,
    get_partitioned_matmul,
    packA_i8mm,
    partition_work,
    use_i8mm_fn,
)
from .utils_gpu import MatmulConfig, MatmulKernels

comptime logger = Logger()

comptime elementwise_epilogue_type = def[
    c_type: DType,
    width: SIMDLength,
    rank: Int,
    *,
    alignment: Int = 1,
](
    IndexList[rank],
    SIMD[c_type, width],
) capturing -> None


# Similar to _get_start_indices_of_nth_subvolume but returns only the batch
# dimensions for matmul, skipping the last 2 dimsnions.
@always_inline
def _get_batch_dims[
    rank: Int
](flat_index: Int, shape: IndexList[rank, ...], out res: type_of(shape)):
    res = {}
    var curr_index = flat_index

    comptime for idx in range(rank - 2):
        # Count from the back, skipping last two dims.
        comptime i = rank - idx - 3
        res[i] = curr_index % shape[i]
        curr_index //= shape[i]


comptime _slice_types[
    stride_types: TypeList[Trait=CoordLike, ...], n_dims: Int
] = stride_types.slice[stride_types.length - n_dims]


comptime _shape_types_to_3d_get_first_dim[
    dtype: DType, *coords: CoordLike
]: CoordLike = ComptimeInt[Coord[*coords].static_product] if Coord[
    *coords
].all_dims_known else Scalar[
    dtype
]

comptime _shape_types_to_3d[
    shape_types: TypeList[Trait=CoordLike, ...]
] = TypeList._concat[
    TypeList.of[
        _shape_types_to_3d_get_first_dim[
            DType.int64,
            *_slice_types[shape_types.reverse(), shape_types.length - 2](),
        ]
    ].values,
    _slice_types[shape_types, 2]().values,
]
"""
Reshape the shape types to 3D. The last two dimensions stay the same. The
first dimension will be the product of the batch dimensions if all the batch
dimensions are static, otherwise it's a runtime dimension.
"""


@always_inline
def _reshape_tile_tensor_with_batch_to_3d(
    tensor: TileTensor,
    out result: TileTensor[
        mut=tensor.mut,
        LayoutType=TileLayout[
            _shape_types_to_3d[tensor.LayoutType._shape_types](),
            _slice_types[tensor.LayoutType._stride_types, 3](),
        ],
        tensor.dtype,
        origin=tensor.origin,
        address_space=tensor.address_space,
        linear_idx_type=tensor.linear_idx_type,
    ],
):
    """
    Reshape the TileTensor with batch dimensions to 3D.
    """

    comptime out_shape_types = type_of(result).LayoutType._shape_types
    comptime out_stride_types = type_of(result).LayoutType._stride_types
    comptime rank = tensor.rank
    comptime assert rank >= 3, "expecting at least rank-3 TileTensor"
    var shape = Tuple[*out_shape_types]()
    var strides = Tuple[*out_stride_types]()

    comptime for i in range(3):
        comptime idx = rank - 3 + i

        # copy the stride
        var stride_ptr = UnsafePointer(to=strides[i])
        comptime StrideType = out_stride_types[i]

        comptime if StrideType.is_static_value:
            stride_ptr.write(rebind[StrideType](Idx[StrideType.static_value]))
        else:
            var stride_val = tensor.layout.stride[idx]().value()
            stride_ptr.write(
                rebind[StrideType](Scalar[StrideType.DTYPE](stride_val))
            )

        # copy the shape
        var shape_ptr = UnsafePointer(to=shape[i])
        comptime ShapeType = out_shape_types[i]

        comptime if ShapeType.is_static_value:
            shape_ptr.write(rebind[ShapeType](Idx[ShapeType.static_value]))
        else:
            var shape_val = Int(tensor.layout.shape[idx]().value())

            comptime if i == 0:
                comptime for batch_idx in range(rank - 3):
                    shape_val *= Int(tensor.layout.shape[batch_idx]().value())

            comptime if ShapeType == Int:
                shape_ptr.write(rebind[ShapeType](shape_val))
            else:
                shape_ptr.write(
                    rebind[ShapeType](Scalar[ShapeType.DTYPE](shape_val))
                )

    return type_of(result)(
        tensor.ptr,
        TileLayout[out_shape_types, out_stride_types](
            Coord[*out_shape_types](shape^), Coord[*out_stride_types](strides^)
        ),
    )


@always_inline
def _batched_matmul_cpu[
    rank: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    *,
    transpose_b: Bool,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
](
    c_tile: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a_tile: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b_tile: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime assert rank < 5, "max rank for batched matmul is currently 4"

    # Extract shape as IndexList for downstream functions.
    var c_shape = rebind[IndexList[rank]](
        coord_to_index_list(c_tile.layout.shape_coord())
    )

    # Batched matmul calls for MacOS >= 13.0.0 and a, b, c of type Float32 are
    # directed to the special Apple-specific implementation.
    comptime if use_apple_accelerate_lib[c_type, a_type, b_type]():
        apple_batched_matmul[
            rank,
            transpose_b=transpose_b,
            elementwise_epilogue_fn=elementwise_epilogue_fn,
        ](c_tile, a_tile, b_tile, c_shape)
        return

    # Flatten to 3D TileTensor by collapsing batch dimensions.
    var collapsed_batches = 1
    comptime for i in range(rank - 2):
        collapsed_batches *= c_shape[i]
    var mat_rows = c_shape[rank - 2]
    var mat_cols = c_shape[rank - 1]

    var a_shape = rebind[IndexList[rank]](
        coord_to_index_list(a_tile.layout.shape_coord())
    )
    var b_shape_idx = rebind[IndexList[rank]](
        coord_to_index_list(b_tile.layout.shape_coord())
    )

    var c = TileTensor(
        c_tile.ptr,
        row_major(Coord(collapsed_batches, mat_rows, mat_cols)),
    )
    var a = TileTensor(
        a_tile.ptr,
        row_major(
            Coord(
                collapsed_batches,
                a_shape[rank - 2],
                a_shape[rank - 1],
            )
        ),
    )
    var b = TileTensor(
        b_tile.ptr,
        row_major(
            Coord(
                collapsed_batches,
                b_shape_idx[rank - 2],
                b_shape_idx[rank - 1],
            )
        ),
    )
    var batch_size: Int = Int(c.dim[0]())

    var m = Int(c.dim[1]())
    var n = Int(c.dim[2]())
    var k = Int(a.dim[2]())
    var num_threads = parallelism_level(ctx)
    # Prevent parallelizing tiny matrices, e.x. 1024x4x4x4.
    var max_num_tasks_batch = min(
        ceildiv(m * n * k * batch_size, get_min_task_size()), batch_size
    )
    # Prevent parallelizing matmul with too many threads.
    var max_num_tasks_matmul = get_matmul_num_tasks[
        a_type, b_type, c_type, simd_width_of[c_type](), True
    ](m, n, k, num_threads) if get_kernel_type(
        m, n, k
    ) else get_matmul_num_tasks[
        a_type, b_type, c_type, simd_width_of[c_type](), False
    ](
        m, n, k, num_threads
    )

    # Define temporary variables to hold num_tasks under testing.
    # This is because the closure can't always capture `var` correctly, issue #12167
    var num_tasks_batch_tmp = min(max_num_tasks_batch, num_threads)
    var num_tasks_matmul_tmp = min(
        max_num_tasks_matmul, num_threads // num_tasks_batch_tmp
    )

    # Prioritize partitioning the batch dimension but if there is more than
    # 20% imbalance, we partition more on the matmul.
    # Imbalance ratio is 1 / min_balance_batch_size
    comptime min_balance_batch_size = 5
    var batch_size_per_task = batch_size // num_tasks_batch_tmp
    if (
        batch_size % num_tasks_batch_tmp != 0
        and batch_size_per_task < min_balance_batch_size
    ):
        # In this case, batches are evenly distributed among tasks, and
        # all threads are used unless the matmul is very small.
        num_tasks_batch_tmp = gcd(batch_size, num_threads)
        num_tasks_matmul_tmp = min(
            max_num_tasks_matmul, num_threads // num_tasks_batch_tmp
        )

    var num_tasks_batch = num_tasks_batch_tmp
    var num_tasks_matmul = num_tasks_matmul_tmp
    var num_tasks = num_tasks_batch * num_tasks_matmul

    @always_inline
    def task_func(
        task_id: Int,
    ) {
        var a,
        var b,
        var c,
        var num_tasks_batch,
        var num_tasks_matmul,
        var m,
        var n,
        var k,
        imm,
    }:
        var a_stride_between_batches = a.num_elements() // Int(a.dim[0]())
        var b_stride_between_batches = b.num_elements() // Int(b.dim[0]())
        var c_stride_between_batches = c.num_elements() // Int(c.dim[0]())

        var batch_task_id, matmul_task_id = divmod(task_id, num_tasks_matmul)

        var num_batches = Int(c.dim[0]())
        # Set the granularity to 1 to divide the batches among tasks
        # as even as possible.
        var batch_range = partition_work(
            batch_task_id, num_tasks_batch, num_batches, 1
        )
        var batch_start = batch_range[0]
        var batches_per_task = batch_range[1]

        # Partition the matmul

        for batch in range(batch_start, batch_start + batches_per_task):
            # Get a 2D view of the 3D Tensor.
            var c_view = TileTensor(
                c.ptr + batch * c_stride_between_batches,
                row_major(Coord(Int(c.dim[1]()), Int(c.dim[2]()))),
            )
            var a_view = TileTensor(
                a.ptr + batch * a_stride_between_batches,
                row_major(Coord(Int(a.dim[1]()), Int(a.dim[2]()))),
            )

            comptime config = get_kernel_config[a_type, b_type, c_type]()
            comptime use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
            comptime simd_size = config.simd_size
            comptime alignment = align_of[SIMD[c_type, simd_size]]()
            var kh = align_up(k, 8)
            var mh = align_up(m, 2)

            var b_view = TileTensor(
                b.ptr + batch * b_stride_between_batches,
                row_major(Coord(Int(b.dim[1]()), Int(b.dim[2]()))),
            )

            var batch_coords = _get_start_indices_of_nth_subvolume[2](
                batch, c_shape
            )

            @__parameter
            def elementwise_lambda_2d[
                c_type: DType, width: SIMDLength, *, alignment: Int = 1
            ](out_coords: IndexList[2], out_val: SIMD[c_type, width]):
                # the caller provided the elementwise epilogue def over the original
                # buffer rank, not the collapsed buffer rank
                # so un-collapse the batch dims here
                comptime if elementwise_epilogue_fn:
                    batch_coords[rank - 1] = out_coords[1]
                    batch_coords[rank - 2] = out_coords[0]

                    comptime func = elementwise_epilogue_fn.value()
                    func[c_type, width, rank](batch_coords, out_val)

            var sub_matmul_config = get_partitioned_matmul[
                a_type, b_type, c_type, config.kernel_rows, config.kernel_cols
            ](m, n, k, matmul_task_id, num_tasks_matmul)
            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            comptime if use_i8mm:
                var a_packed_alloc = alloc(
                    AllocLayout[Scalar[a_type]](
                        count=mh * kh, alignment=alignment
                    )
                )
                var a_packed = TileTensor(
                    a_packed_alloc.unsafe_ptr(),
                    row_major(Coord(mh, kh)),
                )
                packA_i8mm[a_type](
                    0, m, k, a_view.ptr, a_packed_alloc.unsafe_ptr()
                )

                _submatmul_sequential_sync[
                    config,
                    transpose_b,
                    b_packed=False,
                    elementwise_lambda_fn=Optional[
                        matmul_elementwise_epilogue_type
                    ](
                        elementwise_lambda_2d
                    ) if elementwise_epilogue_fn else None,
                    saturated_vnni=saturated_vnni,
                ](
                    c_view,
                    a_packed,
                    b_view,
                    GemmShape(sub_matmul_config.shape),
                    GemmShape(sub_matmul_config.offset),
                )
                dealloc(a_packed_alloc^)
            else:
                _submatmul_sequential_sync[
                    config,
                    transpose_b,
                    b_packed=False,
                    elementwise_lambda_fn=Optional[
                        matmul_elementwise_epilogue_type
                    ](
                        elementwise_lambda_2d
                    ) if elementwise_epilogue_fn else None,
                    saturated_vnni=saturated_vnni,
                ](
                    c_view,
                    a_view,
                    b_view,
                    GemmShape(sub_matmul_config.shape),
                    GemmShape(sub_matmul_config.offset),
                )
            _ = batch_coords

    sync_parallelize(task_func, num_tasks, ctx)


@__name(
    t"naive_batched_matmul_kernel_{c_type}_{a_type}_{b_type}_{transpose_b}",
)
def naive_batched_matmul_kernel[
    rank: Int,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    CTensorType: TensorLayout,
    ATensorType: TensorLayout,
    BTensorType: TensorLayout,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
](
    c_tensor: TileTensor[c_type, CTensorType, MutAnyOrigin],  # m
    a_tensor: TileTensor[a_type, ATensorType, ImmutAnyOrigin],  # m * k
    b_tensor: TileTensor[b_type, BTensorType, ImmutAnyOrigin],  # 1 * k
    c_buff_nd_shape: IndexList[rank],
) -> None:
    """
    Computes one element per thread of a batched matrix multiplication using a
    naive scalar accumulation loop over the contraction dimension.

    Parameters:
        rank: Rank of the original (un-collapsed) output tensor.
        c_type: Output tensor element dtype.
        a_type: LHS input tensor element dtype.
        b_type: RHS input tensor element dtype.
        CTensorType: Layout type of the output tensor.
        ATensorType: Layout type of the LHS input tensor.
        BTensorType: Layout type of the RHS input tensor.
        transpose_b: Whether the RHS input is transposed.
        elementwise_lambda_fn: Optional epilogue applied to each output element.
        accum_type: Accumulator dtype used during the contraction.

    Args:
        c_tensor: Rank-3 output tensor of shape `(batch, m, n)`.
        a_tensor: Rank-3 LHS input tensor of shape `(batch, m, k)`.
        b_tensor: Rank-3 RHS input tensor of shape `(batch, k, n)`.
        c_buff_nd_shape: Shape of the original output tensor before collapsing
            to 3D, used to un-collapse batch coordinates for the epilogue.
    """
    comptime assert (
        c_tensor.rank == 3 and a_tensor.rank == 3 and b_tensor.rank == 3
    ), "expecting rank-3 TileTensor"
    # Provide evidence for flat indexing constraint (for non-nested layouts)
    comptime assert (
        c_tensor.flat_rank == 3
        and a_tensor.flat_rank == 3
        and b_tensor.flat_rank == 3
    )
    var batch_size = Int(c_tensor.dim(0))
    var m = Int(c_tensor.dim(1))
    var n = Int(c_tensor.dim(2))
    var k = Int(a_tensor.dim(2))

    var x = global_idx.x
    var y = global_idx.y
    var z = block_idx.z

    if z >= batch_size or x >= n or y >= m:
        return
    var val = Scalar[accum_type](0)

    comptime acc_type = Scalar[accum_type]
    for ki in range(k):
        var b_val = b_tensor[z, x, ki] if transpose_b else b_tensor[z, ki, x]
        val += a_tensor[z, y, ki].cast[accum_type]() * b_val.cast[accum_type]()

    comptime if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        var nd_corrds = _get_start_indices_of_nth_subvolume[2](
            z, c_buff_nd_shape
        )
        nd_corrds[rank - 1] = x
        nd_corrds[rank - 2] = y
        elementwise_lambda[c_type, 1, rank](nd_corrds, val.cast[c_type]())
    else:
        c_tensor[z, y, x] = val.cast[c_type]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
@__name(
    t"batched_matmul_kernel_gpu_{c_type}_{a_type}_{b_type}_{transpose_b}",
)
def batched_matmul_kernel_gpu[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    CTensorType: TensorLayout,
    ATensorType: TensorLayout,
    BTensorType: TensorLayout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_tensor: TileTensor[c_type, CTensorType, MutAnyOrigin],  # m
    a_tensor: TileTensor[a_type, ATensorType, ImmutAnyOrigin],  # m * k
    b_tensor: TileTensor[b_type, BTensorType, ImmutAnyOrigin],  # 1 * k
):
    """
    Computes a single batch slice of a batched matrix multiplication on the
    GPU by dispatching to the multistage GEMM kernel on NVIDIA or the
    `AMDMatmul` kernel on AMD hardware.

    Parameters:
        c_type: Output tensor element dtype.
        a_type: LHS input tensor element dtype.
        b_type: RHS input tensor element dtype.
        CTensorType: Layout type of the output tensor.
        ATensorType: Layout type of the LHS input tensor.
        BTensorType: Layout type of the RHS input tensor.
        transpose_b: Whether the RHS input is transposed.
        config: Matmul kernel configuration for the target hardware.
        elementwise_lambda_fn: Optional epilogue applied to each output element.

    Args:
        c_tensor: Rank-3 output tensor of shape `(batch, m, n)`.
        a_tensor: Rank-3 LHS input tensor of shape `(batch, m, k)`.
        b_tensor: Rank-3 RHS input tensor of shape `(batch, k, n)`.
    """
    var batch_idx = block_idx.z
    var a_ptr = a_tensor.ptr + batch_idx * Int(
        a_tensor.layout.stride[0]().value()
    )
    var b_ptr = b_tensor.ptr + batch_idx * Int(
        b_tensor.layout.stride[0]().value()
    )
    var c_ptr = c_tensor.ptr + batch_idx * Int(
        c_tensor.layout.stride[0]().value()
    )

    var m = Int(c_tensor.dim[1]())

    comptime k_static = a_tensor.static_shape[2]
    comptime n_static = b_tensor.static_shape[1]

    var a = TileTensor(
        a_ptr,
        TileLayout(
            (m, Idx[a_tensor.static_shape[2]]),
            Coord[*_slice_types[ATensorType._stride_types, 2]()](),
        ),
    )
    var b = TileTensor(
        b_ptr,
        TileLayout(
            Coord[*_slice_types[BTensorType._shape_types, 2]()](),
            Coord[*_slice_types[BTensorType._stride_types, 2]()](),
        ),
    )
    var c = TileTensor(
        c_ptr,
        TileLayout(
            (m, Idx[c_tensor.static_shape[2]]),
            Coord[*_slice_types[CTensorType._stride_types, 2]()](),
        ),
    )

    @__parameter
    def elementwise_epilogue_fn_wrapper[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](out_coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        comptime if elementwise_lambda_fn:
            comptime elementwise_epilogue = elementwise_lambda_fn.value()
            var batch_coords = IndexList[3](block_idx.z)
            batch_coords[2] = out_coords[1]
            batch_coords[1] = out_coords[0]
            elementwise_epilogue(batch_coords, val)

    comptime if is_nvidia_gpu():
        multistage_gemm_kernel[
            config=config,
            elementwise_lambda_fn=Optional[matmul_elementwise_epilogue_type](
                elementwise_epilogue_fn_wrapper
            ) if elementwise_lambda_fn else None,
        ](c, a, b)
    elif is_amd_gpu() and not _is_amd_rdna():
        AMDMatmul[
            a_type,
            b_type,
            c_type,
            transpose_b,
            config,
            Optional[matmul_elementwise_epilogue_type](
                elementwise_epilogue_fn_wrapper
            ) if elementwise_lambda_fn else None,
        ].run[
            type_of(c).LayoutType,
            type_of(a).LayoutType,
            type_of(b).LayoutType,
            type_of(c).Engine,
            type_of(a).Engine,
            type_of(b).Engine,
        ](
            c, a, b
        )


@always_inline
def _batched_matmul_gpu[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    *,
    transpose_b: Bool = False,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
](
    c_buf: TileTensor[mut=True, c_type, ...],
    a_buf: TileTensor[mut=False, a_type, ...],
    b_buf: TileTensor[mut=False, b_type, ...],
    ctx: DeviceContext,
) raises:
    comptime rank = c_buf.rank
    comptime assert rank >= 3, "expecting at least rank-3 TileTensor"
    comptime assert (
        rank == a_buf.rank == b_buf.rank
    ), "all tensors must have the same rank"
    var c_tensor_reshaped = _reshape_tile_tensor_with_batch_to_3d(c_buf)
    var a_tensor_reshaped = _reshape_tile_tensor_with_batch_to_3d(a_buf)
    var b_tensor_reshaped = _reshape_tile_tensor_with_batch_to_3d(b_buf)

    var batch_size = c_tensor_reshaped.dim[0]()
    var m = Int(c_tensor_reshaped.dim[1]())
    var n = Int(c_tensor_reshaped.dim[2]())
    var k = Int(a_tensor_reshaped.dim[2]())

    if batch_size == 0 or m == 0 or n == 0 or k == 0:
        return

    comptime has_static_NK = b_tensor_reshaped.LayoutType._shape_types[
        1
    ].is_static_value and b_tensor_reshaped.LayoutType._shape_types[
        2
    ].is_static_value and a_tensor_reshaped.LayoutType._shape_types[
        2
    ].is_static_value and c_tensor_reshaped.LayoutType._shape_types[
        2
    ].is_static_value

    if batch_size == 1:
        logger.info("Dispatching Batched Matmul via Normal Matmul Kernels")
        with Trace[TraceLevel.OP]("batched_matmul_via_matmul"):
            # If the batch size is 1, then this is just a matmul and we can use the
            # matmul kernel directly.

            # batch_size==1, so flatten (1, X, Y) → (X, Y)
            # by constructing rank-2 TileTensors directly.
            var c_2d = TileTensor(
                c_tensor_reshaped.ptr,
                row_major(Coord(m, n)),
            )
            var a_2d = TileTensor(
                a_tensor_reshaped.ptr,
                row_major(Coord(m, k)),
            )
            # Use b's actual dims since their order depends on transpose_b.
            var b_2d = TileTensor(
                b_tensor_reshaped.ptr,
                row_major(
                    Coord(
                        Int(b_tensor_reshaped.dim(1)),
                        Int(b_tensor_reshaped.dim(2)),
                    )
                ),
            )

            comptime if elementwise_epilogue_fn:
                comptime elementwise_epilogue = elementwise_epilogue_fn.value()

                @__parameter
                @__copy_capture(c_buf)
                def elementwise_epilogue_fn_wrapper[
                    dtype: DType, width: SIMDLength, *, alignment: Int = 1
                ](
                    out_coords: IndexList[2], val: SIMD[dtype, width]
                ) capturing -> None:
                    var batch_coords = IndexList[rank](0)

                    batch_coords[rank - 1] = out_coords[1]
                    batch_coords[rank - 2] = out_coords[0]

                    elementwise_epilogue(batch_coords, val)

                _matmul_gpu[
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_epilogue_fn_wrapper,
                ](c_2d, a_2d, b_2d, ctx=ctx)
            else:
                _matmul_gpu[transpose_b=transpose_b](c_2d, a_2d, b_2d, ctx=ctx)

            return

    comptime a_k = a_tensor_reshaped.LayoutType._shape_types[2].static_value
    comptime c_n = c_tensor_reshaped.LayoutType._shape_types[2].static_value

    # SM100 (B200+) batched BF16 matmul dispatch
    comptime use_SM100_kernels = (
        has_nvidia_gpu_accelerator() and _has_blackwell_tcgen05()
    )
    comptime if use_SM100_kernels and has_static_NK and transpose_b:
        logger.info(
            "Dispatching Batched Matmul via SM100",
            a_type,
            b_type,
            c_type,
            a_k,
            c_n,
        )
        comptime if (
            c_type in (DType.bfloat16, DType.float8_e4m3fn)
            and c_n * size_of[c_type]() % 16 == 0
            and a_k * size_of[a_type]() % 16 == 0
            and transpose_b
        ):
            dispatch_sm100_batched_matmul[c_type, a_type, b_type, transpose_b](
                c_tensor_reshaped,
                a_tensor_reshaped,
                b_tensor_reshaped,
                ctx,
            )

            comptime if elementwise_epilogue_fn:
                comptime epilogue = elementwise_epilogue_fn.value()
                # SM100+ supports 32B load/store to global memory.
                comptime simd_size = 32 // size_of[c_type]()

                def epilogue_wrapper[
                    simd_width: Int, alignment: Int = 1
                ](idx: Coord) {var}:
                    var c_val = c_tensor_reshaped.load[
                        width=simd_width,
                        alignment=alignment * size_of[c_type](),
                    ](idx)
                    epilogue[c_type, simd_width, alignment=alignment](
                        coord_to_index_list(idx), c_val
                    )

                elementwise[simd_size, target="gpu"](
                    epilogue_wrapper, Coord(batch_size, m, n), ctx
                )

            return

    comptime multistage_gemm_cond = (
        c_n % 128 == 0 and a_k % 32 == 0 and a_k >= 128
    )

    comptime use_A100_kernels = (
        has_nvidia_gpu_accelerator()
        and ctx.default_device_info.compute >= A100.compute
    )

    comptime if has_static_NK and use_A100_kernels and multistage_gemm_cond:
        logger.info("Dispatching Batched Matmul via A100 Kernels")
        comptime kernels = MatmulKernels[a_type, b_type, c_type, transpose_b]()

        comptime batched_matmul_type = batched_matmul_kernel_gpu[
            c_tensor_reshaped.dtype,
            a_tensor_reshaped.dtype,
            b_tensor_reshaped.dtype,
            c_tensor_reshaped.LayoutType,
            a_tensor_reshaped.LayoutType,
            b_tensor_reshaped.LayoutType,
            transpose_b,
            kernels.ampere_128x128_4,
            elementwise_epilogue_fn,
        ]

        var grid_dim = kernels.ampere_128x128_4.grid_dim(m, n)

        ctx.enqueue_function[batched_matmul_type](
            c_tensor_reshaped,
            a_tensor_reshaped.as_immut(),
            b_tensor_reshaped.as_immut(),
            grid_dim=(grid_dim[0], grid_dim[1], batch_size),
            block_dim=kernels.ampere_128x128_4.block_dim(),
            shared_mem_bytes=kernels.ampere_128x128_4.shared_mem_usage(),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(kernels.ampere_128x128_4.shared_mem_usage())
            ),
        )
    elif has_static_NK and has_amd_gpu_accelerator() and transpose_b:

        @always_inline
        @__parameter
        def kernel_helper[block_m: Int, block_n: Int]() raises:
            comptime block_k = 64
            comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
                block_tile_shape=Index(block_m, block_n, block_k),
                warp_tile_shape=Index(block_m // 2, block_n // 2, block_k),
                mma_shape=_amdgpu_get_mma_shape[a_type, transpose_b](),
                num_pipeline_stages=1,
                num_k_partitions=1,
            )

            comptime batched_matmul_type = batched_matmul_kernel_gpu[
                c_tensor_reshaped.dtype,
                a_tensor_reshaped.dtype,
                b_tensor_reshaped.dtype,
                c_tensor_reshaped.LayoutType,
                a_tensor_reshaped.LayoutType,
                b_tensor_reshaped.LayoutType,
                transpose_b,
                config,
                elementwise_epilogue_fn,
            ]

            ctx.enqueue_function[batched_matmul_type](
                c_tensor_reshaped,
                a_tensor_reshaped.as_immut(),
                b_tensor_reshaped.as_immut(),
                grid_dim=(
                    ceildiv(n, block_n),
                    ceildiv(m, block_m),
                    batch_size,
                ),
                block_dim=config.block_dim(),
            )

        if m <= 32:
            kernel_helper[32, 32]()
        elif m <= 256:
            kernel_helper[64, 64]()
        else:
            kernel_helper[128, 128]()

    else:
        logger.info("Dispatching Batched Matmul via Naive Kernels")
        var c_shape = coord_to_index_list(c_buf.layout.shape_coord())

        comptime BLOCK_DIM = 16
        comptime bmm = naive_batched_matmul_kernel[
            rank,
            c_type,
            a_type,
            b_type,
            c_tensor_reshaped.LayoutType,
            a_tensor_reshaped.LayoutType,
            b_tensor_reshaped.LayoutType,
            transpose_b,
            elementwise_epilogue_fn,
        ]
        ctx.enqueue_function[bmm](
            c_tensor_reshaped,
            a_tensor_reshaped.as_immut(),
            b_tensor_reshaped.as_immut(),
            c_shape,
            grid_dim=(
                ceildiv(n, BLOCK_DIM),
                ceildiv(m, BLOCK_DIM),
                batch_size,
            ),
            block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
        )


@always_inline
def batched_matmul[
    *,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
    elementwise_epilogue_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
    target: StaticString = "cpu",
](
    c_buf: TileTensor[mut=True, address_space=.GENERIC, ...],
    a_buf: TileTensor[mut=False, address_space=.GENERIC, ...],
    b_buf: TileTensor[mut=False, address_space=.GENERIC, ...],
    *,
    context: Optional[DeviceContext] = None,
) raises:
    """TileTensor primary implementation of `batched_matmul`.

    Parameters:
        transpose_a: Whether the LHS input is transposed (defaults to
            `False`; not yet supported).
        transpose_b: Whether the RHS input is transposed (defaults to
            `False`).
        elementwise_epilogue_fn: Optional epilogue applied to each output
            element (defaults to `None`).
        saturated_vnni: Whether to use saturated VNNI accumulation on CPU
            (defaults to `False`; not applicable on GPU).
        target: Target hardware for the operation (defaults to `"cpu"`).

    Args:
        c_buf: Output tensor of shape `(..., m, n)`; rank must be at least
            2 and match the inputs.
        a_buf: LHS input tensor of shape `(..., m, k)`.
        b_buf: RHS input tensor of shape `(..., k, n)`, or `(..., n, k)`
            when `transpose_b` is set.
        context: Optional device context used for dispatch and parallelism
            (defaults to `None`).
    """
    comptime assert c_buf.rank >= 2, "c must be at least rank 2"
    comptime assert (
        c_buf.rank == a_buf.rank == b_buf.rank
    ), "all tensors must have the same rank"
    comptime assert (
        c_buf.flat_rank == c_buf.rank
    ), "c must have a non-nested layout"
    comptime assert (
        a_buf.flat_rank == a_buf.rank
    ), "a must have a non-nested layout"
    comptime assert (
        b_buf.flat_rank == b_buf.rank
    ), "b must have a non-nested layout"
    comptime assert not transpose_a, "transpose_a not yet supported"

    comptime rank = c_buf.rank

    # Build shape IndexLists from TileTensor for tracing.
    var a_shape = rebind[IndexList[rank]](
        coord_to_index_list(a_buf.layout.shape_coord())
    )
    var b_shape = rebind[IndexList[rank]](
        coord_to_index_list(b_buf.layout.shape_coord())
    )
    var c_shape = rebind[IndexList[rank]](
        coord_to_index_list(c_buf.layout.shape_coord())
    )

    @always_inline
    def description_fn() {var a_shape, var b_shape, var c_shape, imm} -> String:
        # fmt: off
        return String(
            trace_arg("A", a_shape, a_buf.dtype),
            ";", trace_arg("B", b_shape, b_buf.dtype),
            ";", trace_arg("C", c_shape, c_buf.dtype),
            ";transpose_a=", transpose_a,
            ";transpose_b=", transpose_b,
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=target](
        "batched_matmul",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(context),
    ):
        comptime assert is_valid_target[target](), "unsupported target"

        comptime if is_cpu[target]():
            _batched_matmul_cpu[
                rank,
                a_buf.dtype,
                b_buf.dtype,
                c_buf.dtype,
                transpose_b=transpose_b,
                elementwise_epilogue_fn=elementwise_epilogue_fn,
                saturated_vnni=saturated_vnni,
            ](c_buf, a_buf, b_buf, ctx=context)
        else:
            comptime assert (
                saturated_vnni == False
            ), "saturated_vnni is not applicable on the gpu"
            _batched_matmul_gpu[
                transpose_b=transpose_b,
                elementwise_epilogue_fn=elementwise_epilogue_fn,
            ](
                c_buf,
                a_buf,
                b_buf,
                context.value(),
            )


@always_inline
def batched_matmul_shape[
    rank: Int
](
    a_buff: TileTensor[mut=False, ...], b_buff: TileTensor[mut=False, ...]
) raises -> IndexList[rank]:
    """
    Compute the output shape of a `batch_matmul` operation, and assert the
    inputs are compatible.

    Parameters:
        rank: Rank of the input and output tensors.

    Args:
        a_buff: The lhs input tensor.
        b_buff: The rhs input tensor.

    Returns:
        The output shape.
    """
    comptime assert a_buff.rank == rank, "a must have the specified rank"
    comptime assert b_buff.rank == rank, "b must have the specified rank"

    if rank <= 2:
        raise Error("[batch_matmul] requires rank > 2")

    if Int(a_buff.dim[rank - 1]()) != Int(b_buff.dim[rank - 2]()):
        raise Error("[batch_matmul] inputs inner dimensions must match")

    # Check batch dimensions
    var foundMismatch = False

    comptime for i in range(rank - 2):
        if Int(a_buff.dim[i]()) != Int(b_buff.dim[i]()):
            foundMismatch = True

    if foundMismatch:
        raise Error("[batch_matmul] inputs batch dimensions must match")

    var output_shape = rebind[IndexList[rank]](
        coord_to_index_list(a_buff.layout.shape_coord())
    )
    output_shape[rank - 1] = Int(b_buff.dim[rank - 1]())

    return output_shape


comptime _2D_layout[layout: Layout] = Layout(
    IntTuple(layout.shape[1], layout.shape[2]),
    IntTuple(layout.stride[1], layout.stride[2]),
)


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
@__name(t"bmm_sm100_blockwise_scaled_fp8_{a_type}_{b_type}_{c_type}")
def _bmm_sm100_blockwise_scaled_fp8_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_layout: TensorLayout,
    c_layout: Layout,
    a_scales_layout: TensorLayout,
    b_scales_layout: Layout,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    a_scales_tile_rank: Int,
    a_scales_tile_shape: IndexList[a_scales_tile_rank],
    a_scales_desc_shape: IndexList[a_scales_tile_rank],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: Int = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    b_scaling_block_n: Int = 128,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c_tensor: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a_scales_tma_op: TMATensorTile[
        a_scales_type,
        a_scales_tile_rank,
        a_scales_tile_shape,
        a_scales_desc_shape,
    ],
    b_scales_tensor: LayoutTensor[
        b_scales_type, b_scales_layout, ImmutAnyOrigin
    ],
    num_iters: Int32,
):
    var _num_iters = Int(num_iters)
    comptime c_2d_layout: Layout = _2D_layout[c_layout]
    comptime b_scales_2d_layout: Layout = _2D_layout[b_scales_layout]

    var M = c_tensor.dim(1)
    var N = c_tensor.dim(2)

    var b_scales_ptr = b_scales_tensor.ptr + (
        block_idx.z * b_scales_tensor.dim(1) * b_scales_tensor.dim(2)
    )

    var c = LayoutTensor[c_type, c_2d_layout](
        c_tensor.ptr_at_offset(Index(block_idx.z, 0, 0)),
        RuntimeLayout[c_2d_layout](
            Index(c_tensor.dim(1), c_tensor.dim(2)),
            Index(c_tensor.stride(1), c_tensor.stride(2)),
        ),
    )

    var b_scales = LayoutTensor[b_scales_type, b_scales_2d_layout](
        b_scales_ptr,
        RuntimeLayout[b_scales_2d_layout].row_major(
            IndexList[2](b_scales_tensor.dim(1), b_scales_tensor.dim(2)),
        ),
    )

    @__parameter
    def elementwise_epilogue_fn_wrapper[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](out_coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        comptime if elementwise_lambda_fn:
            comptime elementwise_epilogue = elementwise_lambda_fn.value()
            var batch_coords = IndexList[3](block_idx.z)
            batch_coords[2] = out_coords[1]
            batch_coords[1] = out_coords[0]
            elementwise_epilogue(batch_coords, val)

    # Compatibility boundary: the SM100 blockwise FP8 kernel is TileTensor-
    # native. This BMM entry point still slices legacy LayoutTensor views, so
    # adapt exactly once at the call boundary instead of reintroducing
    # LayoutTensor inside matmul/gpu/sm100.
    var c_tt = lt_to_tt(c)
    var b_scales_tt = lt_to_tt(b_scales)

    matmul_sm100_blockwise_scaled_fp8_1d2d_kernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        a_layout,
        type_of(c_tt).LayoutType,
        a_scales_layout,
        type_of(b_scales_tt).LayoutType,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(a_scales_tma_op).rank,
        type_of(a_scales_tma_op).tile_shape,
        type_of(a_scales_tma_op).desc_shape,
        block_tile_shape,
        mma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=num_threads,
        elementwise_lambda_fn=Optional[matmul_elementwise_epilogue_type](
            elementwise_epilogue_fn_wrapper
        ) if elementwise_lambda_fn else None,
        b_scaling_block_n=b_scaling_block_n,
    ](
        a_tma_op,
        b_tma_op,
        c_tt,
        a_scales_tma_op,
        b_scales_tt,
        Int32(_num_iters),
    )


def bmm_sm100_blockwise_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    b_scaling_block_n: Int = 128,
](
    c_: TileTensor[mut=True, c_type, ...],
    a_: TileTensor[mut=False, a_type, ...],
    b_: TileTensor[mut=False, b_type, ...],
    a_scales_: TileTensor[mut=False, a_scales_type, ...],
    b_scales_: TileTensor[mut=False, b_scales_type, ...],
    ctx: DeviceContext,
) raises:
    """
    Computes a batched blockwise scaled FP8 matrix multiplication on SM100
    (Blackwell) hardware by constructing TMA descriptors for the inputs and
    scales and enqueuing the blockwise FP8 kernel per batch slice.

    Parameters:
        c_type: Output tensor element dtype.
        a_type: LHS input tensor element dtype.
        b_type: RHS input tensor element dtype.
        a_scales_type: LHS scales tensor element dtype.
        b_scales_type: RHS scales tensor element dtype.
        transpose_b: Whether the RHS input is transposed (must be `True`).
        umma_shape: UMMA instruction shape `(m, n, k)` used by the kernel.
        block_tile_shape: CTA tile shape `(BM, BN, BK)`.
        a_swizzle: TMA swizzle mode for the LHS input tensor.
        b_swizzle: TMA swizzle mode for the RHS input tensor.
        elementwise_lambda_fn: Optional epilogue applied to each output element.
        b_scaling_block_n: N-direction scale block size for the RHS scales.

    Args:
        c_: Rank-3 output tensor of shape `(batch, m, n)`.
        a_: Rank-3 LHS input tensor of shape `(batch, m, k)`.
        b_: Rank-3 RHS input tensor of shape `(batch, k, n)`.
        a_scales_: Rank-3 LHS scales tensor.
        b_scales_: Rank-3 RHS scales tensor.
        ctx: Device context used to enqueue the kernel.
    """
    # Convert to LayoutTensor for internal operations.
    var c = c_.to_layout_tensor()
    var a = a_.to_layout_tensor()
    var b = b_.to_layout_tensor()
    var a_scales = a_scales_.to_layout_tensor()
    var b_scales = b_scales_.to_layout_tensor()

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        a_type == b_type == .float8_e4m3fn
    ), "Only support float8_e4m3fn"

    comptime assert (
        b_scales_type == a_scales_type == .float32
    ), "Only support float32 for a_scales and b_scales"

    comptime assert c.rank == 3, "Only support rank 3 tensors"

    comptime assert (
        c.rank == b.rank and c.rank == a.rank
    ), "all tensors must have the same rank"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime assert BK in (
        64,
        128,
    ), "blockwise scaled fp8 only supports BK in (64, 128)"

    var batch_size = c.dim(0)
    var M = c.dim(1)
    var N = c.dim(2)
    var K = a.dim(2)

    if batch_size == 0 or M == 0 or N == 0 or K == 0:
        return

    var a_scales_dim0 = a_scales.dim(1)
    var a_scales_dim1 = a_scales.dim(2)
    var b_scales_dim0 = b_scales.dim(1)
    var b_scales_dim1 = b_scales.dim(2)

    # The K-direction scale granularity is fixed at BK
    # (k_scale_granularity == BK). The N-direction granularity may be
    # finer and is independent of BK_kernel — encoded in b_scales.dim(1)
    # = N // n_scale_granularity. K need not be a multiple of BK: the
    # last K-tile is covered by TMA OOB zero-padding and the matching
    # ceildiv scale row, contributing zero to the accumulator.
    if a_scales_dim0 != b_scales_dim1 or a_scales_dim0 != ceildiv(K, BK):
        raise Error(
            "a_scales.dim(1) must equal b_scales.dim(2) and equal"
            " ceildiv(K, BK)."
        )

    if N % b_scales_dim0 != 0 or (N // b_scales_dim0) not in (64, 128):
        raise Error(
            "N must be divisible by b_scales.dim(1) and (N // b_scales.dim(1))"
            " must be in (64, 128)."
        )

    var padding_size = 16 // size_of[a_scales_type]()
    if a_scales_dim1 % padding_size != 0:
        raise Error(
            "a_scales_3D.dim(2) must be divisible by 16 bytes. This is required"
            " by NVIDIA SM90+ TMA instructions!"
        )

    logger.info(
        "Executing SM100 Basic Batched 1D2D Blockwise Scaled FP8 GEMM"
        " (BLOCK_SCALE_SIZE = 128)"
    )
    logger.info(
        "Problem Shape: MNK=[", batch_size, ", ", M, ", ", N, ", ", K, "]"
    )
    logger.info(
        "A Scales Shape: [",
        a_scales.dim(1),
        ", ",
        a_scales.dim(2),
        "]",
    )
    logger.info(
        "B Scales Shape: [",
        b_scales.dim(1),
        ", ",
        b_scales.dim(2),
        "]",
    )

    var a_tma_op = create_tensor_tile[
        Index(1, BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a)

    comptime b_tile_shape = Index(1, BN, BK) if transpose_b else Index(
        1, BK, BN
    )

    var b_tma_op = create_tensor_tile[
        b_tile_shape,
        swizzle_mode=b_swizzle,
    ](ctx, b)

    var a_scales_tma_op = create_tensor_tile[
        Index(1, 1, BM),
        __desc_shape=Index(1, 1, BM),
    ](ctx, a_scales)
    # NOTE: desc shape must be specified otherwise a constraint fails

    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24 + size_of[a_scales_type]() * BM

    comptime block_dim = 128

    comptime kernel = _bmm_sm100_blockwise_scaled_fp8_kernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        type_of(a_).LayoutType,
        type_of(c).layout,
        type_of(a_scales_).LayoutType,
        type_of(b_scales).layout,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(a_scales_tma_op).rank,
        type_of(a_scales_tma_op).tile_shape,
        type_of(a_scales_tma_op).desc_shape,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
        b_scaling_block_n=b_scaling_block_n,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c,
        a_scales_tma_op,
        b_scales,
        Int32(ceildiv(K, BK)),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), batch_size),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


def batched_matmul_dynamic_scaled_fp8_naive[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    scales_granularity_mnk: IndexList[3],
    transpose_b: Bool = False,
](
    c_: TileTensor[mut=True, c_type, ...],
    a_: TileTensor[mut=False, a_type, ...],
    b_: TileTensor[mut=False, b_type, ...],
    a_scales_: TileTensor[mut=False, a_scales_type, ...],
    b_scales_: TileTensor[mut=False, b_scales_type, ...],
    ctx: DeviceContext,
) raises:
    """
    Computes a batched blockwise scaled FP8 matrix multiplication using a
    naive per-batch loop that calls the 2D blockwise scaled FP8 kernel for
    each batch slice.

    Parameters:
        c_type: Output tensor element dtype.
        a_type: LHS input tensor element dtype.
        b_type: RHS input tensor element dtype.
        a_scales_type: LHS scales tensor element dtype.
        b_scales_type: RHS scales tensor element dtype.
        scales_granularity_mnk: Scale granularity `(m, n, k)`; only
            `(1, 128, 128)` is currently supported.
        transpose_b: Whether the RHS input is transposed.

    Args:
        c_: Rank-3 output tensor of shape `(batch, m, n)`.
        a_: Rank-3 LHS input tensor of shape `(batch, m, k)`.
        b_: Rank-3 RHS input tensor of shape `(batch, k, n)`.
        a_scales_: Rank-3 LHS scales tensor.
        b_scales_: Rank-3 RHS scales tensor.
        ctx: Device context used to dispatch the per-batch kernels.
    """
    comptime assert (
        scales_granularity_mnk[0] == 1
        and scales_granularity_mnk[1] == scales_granularity_mnk[2] == 128
    ), "Only support (1,128,128) scale granularity. Extend it for other cases."

    comptime BLOCK_SCALE_K = 128

    # Convert to LayoutTensor for internal operations.
    var c_lt = c_.to_layout_tensor()
    var a_lt = a_.to_layout_tensor()
    var b_lt = b_.to_layout_tensor()
    var a_scales_lt = a_scales_.to_layout_tensor()
    var b_scales_lt = b_scales_.to_layout_tensor()

    # naive implementation requires all tensor have AddressSpace.GENERIC
    var c = c_lt.address_space_cast[.GENERIC]()
    var a = a_lt.address_space_cast[.GENERIC]()
    var b = b_lt.address_space_cast[.GENERIC]()
    var a_scales = a_scales_lt.address_space_cast[.GENERIC]()
    var b_scales = b_scales_lt.address_space_cast[.GENERIC]()

    var B = c.dim(0)
    var M = c.dim(1)
    var N = c.dim(2)
    var K = a.dim(2)
    var M_a_scales = a_scales.dim(2)

    # Create 2D layouts by extracting last 2 dims from 3D layouts
    # This preserves the original shape and stride (not assuming row-major)
    comptime c_layout_2d = _2D_layout[c.layout]
    comptime a_layout_2d = _2D_layout[a.layout]
    comptime b_layout_2d = _2D_layout[b.layout]
    comptime a_scales_layout_2d = _2D_layout[a_scales.layout]
    comptime b_scales_layout_2d = _2D_layout[b_scales.layout]

    for batch in range(B):
        # Create 2D LayoutTensor views
        var c_view = LayoutTensor[c_type, c_layout_2d, c.origin](
            c.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[c_layout_2d](
                Index(M, N), Index(c.stride(1), c.stride(2))
            ),
        )
        var a_view = LayoutTensor[a_type, a_layout_2d, a.origin](
            a.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[a_layout_2d](
                Index(M, K), Index(a.stride(1), a.stride(2))
            ),
        )
        var b_view = LayoutTensor[b_type, b_layout_2d, b.origin](
            b.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[b_layout_2d](
                Index(N, K), Index(b.stride(1), b.stride(2))
            ),
        )
        var a_scales_view = LayoutTensor[
            a_scales_type, a_scales_layout_2d, a_scales.origin
        ](
            a_scales.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[a_scales_layout_2d](
                Index(ceildiv(K, BLOCK_SCALE_K), M_a_scales),
                Index(a_scales.stride(1), a_scales.stride(2)),
            ),
        )
        var b_scales_view = LayoutTensor[
            b_scales_type,
            b_scales_layout_2d,
            b_scales.origin,
        ](
            b_scales.ptr_at_offset(Index(batch, 0, 0)),
            RuntimeLayout[b_scales_layout_2d](
                Index(ceildiv(N, BLOCK_SCALE_K), ceildiv(K, BLOCK_SCALE_K)),
                Index(b_scales.stride(1), b_scales.stride(2)),
            ),
        )

        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
            scales_granularity_mnk=Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
        ](
            c_view,
            a_view,
            b_view,
            a_scales_view,
            b_scales_view,
            ctx,
        )


def batched_matmul_dynamic_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    input_scale_granularity: StaticString,
    weight_scale_granularity: StaticString,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    transpose_b: Bool = False,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    a_scales: TileTensor[mut=False, a_scales_type, ...],
    b_scales: TileTensor[mut=False, b_scales_type, ...],
    ctx: DeviceContext,
) raises:
    """
    Dispatches a batched blockwise scaled FP8 matrix multiplication to the
    SM100 blockwise kernel on Blackwell hardware or falls back to the naive
    per-batch implementation on H100.

    Parameters:
        c_type: Output tensor element dtype.
        a_type: LHS input tensor element dtype.
        b_type: RHS input tensor element dtype.
        a_scales_type: LHS scales tensor element dtype.
        b_scales_type: RHS scales tensor element dtype.
        input_scale_granularity: Scale granularity mode for the LHS input
            (only `"block"` is supported).
        weight_scale_granularity: Scale granularity mode for the RHS input
            (only `"block"` is supported).
        m_scale_granularity: M-direction scale granularity (must be `1`).
        n_scale_granularity: N-direction scale granularity (`64` or `128`).
        k_scale_granularity: K-direction scale granularity (`64` or `128`).
        transpose_b: Whether the RHS input is transposed.
        target: Target platform string.

    Args:
        c: Rank-3 output tensor of shape `(batch, m, n)`.
        a: Rank-3 LHS input tensor of shape `(batch, m, k)`.
        b: Rank-3 RHS input tensor of shape `(batch, k, n)`.
        a_scales: Rank-3 LHS scales tensor.
        b_scales: Rank-3 RHS scales tensor.
        ctx: Device context used to dispatch the kernel.
    """
    comptime assert (
        _is_sm10x_gpu(ctx.default_device_info)
        or ctx.default_device_info == H100
    ), "Only support SM100 or SM90"
    comptime assert (
        m_scale_granularity == 1
        and k_scale_granularity in (64, 128)
        and n_scale_granularity in (64, 128)
    ), (
        "Only support m_scale_granularity == 1 and k/n_scale_granularity"
        " in (64, 128)."
    )
    comptime assert (
        a_type == b_type == .float8_e4m3fn
    ), "input A and B dtype should be float8_e4m3fn"
    comptime assert (
        a_scales_type == b_scales_type == .float32
    ), "input A and B scales dtype should be float32"

    comptime assert (
        input_scale_granularity == "block"
        and weight_scale_granularity == "block"
    ), "Only support block-wise scale granularity"

    comptime if _is_sm10x_gpu(ctx.default_device_info):
        # BN per CTA tracks n_scale_granularity so each N-tile fits in
        # exactly one B-scale block; BK_kernel tracks k_scale_granularity.
        # K-tile bytes = BK * sizeof(fp8) = BK. SWIZZLE_128B requires a
        # 128-byte K-tile, so BK=64 needs SWIZZLE_64B instead.
        comptime umma_shape = Index(64, n_scale_granularity, 32)
        comptime block_tile_shape = Index(
            umma_shape[0], umma_shape[1], k_scale_granularity
        )
        comptime swizzle = (
            TensorMapSwizzle.SWIZZLE_128B if k_scale_granularity
            == 128 else TensorMapSwizzle.SWIZZLE_64B
        )

        bmm_sm100_blockwise_scaled_fp8[
            transpose_b=transpose_b,
            umma_shape=umma_shape,
            block_tile_shape=block_tile_shape,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            b_scaling_block_n=n_scale_granularity,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            ctx,
        )

    else:
        batched_matmul_dynamic_scaled_fp8_naive[
            scales_granularity_mnk=Index(
                m_scale_granularity, n_scale_granularity, k_scale_granularity
            ),
            transpose_b=transpose_b,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            ctx,
        )
