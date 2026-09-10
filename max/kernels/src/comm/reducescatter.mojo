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
"""Multi-GPU reducescatter implementation for distributed tensor reduction across GPUs.
"""

from std.collections import Array
from std.collections.optional import Optional

from layout import Coord, Idx, TensorLayout, TileTensor, row_major
from layout.tile_layout import Layout
from layout.coord import _CoordToDynamic
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    grid_dim,
)
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.memory import Consistency, ReduceOp, multimem_ld_reduce
from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from max.gpu.intrinsics import (
    Scope,
)
from std.math.uutils import ualign_down
from std.math import ceildiv
from std.sys import (
    simd_width_of,
    align_of,
    has_amd_gpu_accelerator,
    is_amd_gpu,
)

from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    circular_add,
    is_p2p_enabled,
)

# On AMD Systems, the loads from GLOBAL addressspace gives an improvement
# to the performance.
comptime _target_address_space = AddressSpace.GLOBAL if is_amd_gpu() else AddressSpace.GENERIC

comptime elementwise_epilogue_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int
](Coord, SIMD[dtype, length=width]) capturing -> None


@always_inline
def _load_reduce[
    dtype: DType,
    in_tile_layout: TensorLayout,
    //,
    ngpus: Int,
    simd_width: Int,
    alignment: Int,
    accum_type: DType,
    *,
    use_multimem: Bool = False,
](
    elem_idx: Int,
    in_tiles: Array[
        TileTensor[dtype, in_tile_layout, ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
) -> SIMD[dtype, simd_width]:
    comptime if use_multimem:
        # Multimem mode: use optimized reduction
        return multimem_ld_reduce[
            dtype,
            simd_width=simd_width,
            reduction=ReduceOp.ADD,
            scope=Scope.GPU,
            consistency=Consistency.RELAXED,
            accum_type=accum_type,
        ](
            (in_tiles[0].ptr_at_offset(Coord(elem_idx))).address_space_cast[
                .GLOBAL
            ]()
        )
    else:
        # Regular mode: manual accumulation
        # Initialize with first load to avoid extra zero-add operation
        var accum = (
            in_tiles[0]
            .address_space_cast[_target_address_space]()
            .load[width=simd_width, alignment=alignment, invariant=True](
                Coord(elem_idx)
            )
            .cast[accum_type]()
        )

        comptime for gpu_idx in range(1, ngpus):
            accum += (
                in_tiles[gpu_idx]
                .address_space_cast[_target_address_space]()
                .load[width=simd_width, alignment=alignment, invariant=True](
                    Coord(elem_idx)
                )
                .cast[accum_type]()
            )
        return accum.cast[dtype]()


struct ReduceScatterConfig[
    dtype: DType,
    ngpus: Int,
    simd_width: Int = simd_width_of[dtype, target=get_gpu_target()](),
    alignment: Int = align_of[SIMD[dtype, simd_width]](),
    accum_type: DType = get_accum_type[dtype](),
](TrivialRegisterPassable):
    """Configuration for axis-aware reduce-scatter partitioning.

    Divides `axis_size` units evenly across GPUs. Lower ranks get one extra
    unit when there's a remainder. The 1D case is a special case where
    `axis_size = num_elements // simd_width` and `unit_numel = simd_width`.
    """

    var stride: Int
    var axis_part: Int
    var axis_remainder: Int
    var unit_numel: Int

    @always_inline
    def __init__(
        out self,
        axis_size: Int,
        unit_numel: Int,
        threads_per_gpu: Int,
    ):
        """General constructor for axis-aware partitioning.

        Args:
            axis_size: Number of units along the scatter axis.
            unit_numel: Number of elements per unit.
            threads_per_gpu: Total threads per GPU.
        """
        comptime assert Self.ngpus > 1, "ngpus must be greater than 1"
        self.stride = threads_per_gpu * Self.simd_width
        self.axis_part, self.axis_remainder = divmod(axis_size, Self.ngpus)
        self.unit_numel = unit_numel

    @always_inline
    def __init__(
        out self,
        num_elements: Int,
        threads_per_gpu: Int,
    ):
        """1D convenience constructor. Partitions by SIMD vectors."""
        comptime assert Self.ngpus > 1, "ngpus must be greater than 1"
        self.stride = threads_per_gpu * Self.simd_width
        var num_simd_vectors = num_elements // Self.simd_width
        self.axis_part, self.axis_remainder = divmod(
            num_simd_vectors, Self.ngpus
        )
        self.unit_numel = Self.simd_width

    @always_inline
    def rank_unit_start(self, rank: Int) -> Int:
        """Start unit index along scatter axis for this rank."""
        return rank * self.axis_part + min(rank, self.axis_remainder)

    @always_inline
    def rank_units(self, rank: Int) -> Int:
        """Number of units for this rank."""
        return self.axis_part + Int(rank < self.axis_remainder)

    @always_inline
    def rank_num_elements(self, rank: Int) -> Int:
        """Total elements for this rank."""
        return self.rank_units(rank) * self.unit_numel

    @always_inline
    def rank_start(self, rank: Int) -> Int:
        """Flat element start offset for this rank."""
        return self.rank_unit_start(rank) * self.unit_numel

    @always_inline
    def rank_end(self, rank: Int) -> Int:
        """Flat element end offset for this rank."""
        return self.rank_start(rank + 1)

    @always_inline
    def rank_part(self, rank: Int) -> Int:
        """Number of elements for this rank (alias for rank_num_elements)."""
        return self.rank_num_elements(rank)

    @always_inline
    def thr_local_start(self, thread_idx: Int) -> Int:
        return thread_idx * Self.simd_width


@always_inline
def _reduce_scatter_impl[
    dtype: DType,
    in_tile_layout: TensorLayout,
    //,
    ngpus: Int,
    *,
    output_lambda: elementwise_epilogue_type,
    simd_width: Int = simd_width_of[dtype, target=get_gpu_target()](),
    alignment: Int = align_of[SIMD[dtype, simd_width]](),
    accum_type: DType = get_accum_type[dtype](),
    use_multimem: Bool = False,
](
    in_tiles: Array[
        TileTensor[dtype, in_tile_layout, ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    out_buf: TileTensor[mut=True, dtype, ...],
    # TODO(KERN-2526): pass config here
    num_elements: Int,
    thread_stride: Int,
):
    """TileTensor-based reduce-scatter implementation.

    Iterates flat over sliced+reversed input tiles with coalesced access.
    For 1D inputs: degenerates to contiguous pointer loads (same as flat).
    For 2D axis-0: also contiguous (reversed row-major = flat).
    For 2D axis-1: coalesced within stride-1 dimension strips.
    """
    for c in range(
        global_idx.x * simd_width,
        num_elements,
        thread_stride,
    ):
        # wider accumulator (accum_type) for numerical stability.
        var reduced_result = _load_reduce[
            ngpus,
            simd_width=simd_width,  # TODO(KERN-2526): config. on these 3
            alignment=alignment,
            accum_type=accum_type,
            use_multimem=use_multimem,
        ](c, in_tiles)

        # Note idx2crd only works here because out_buf is compact
        output_lambda[width=simd_width, alignment=alignment](
            out_buf.layout.idx2crd(c),
            reduced_result,
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"reducescatter_{dtype}_{use_multimem}")
def _reducescatter_kernel[
    dtype: DType,
    in_layout: TensorLayout,
    out_layout: TensorLayout,
    ngpus: Int,
    *,
    axis: Int = 0,
    BLOCK_SIZE: Int,
    output_lambda: elementwise_epilogue_type,
    use_multimem: Bool = False,
    domain_id: Int = 0,
](
    in_bufs: Array[
        TileTensor[dtype, in_layout, ImmutAnyOrigin],
        1 if use_multimem else ngpus,
    ],
    out_buf: TileTensor[dtype, out_layout, MutAnyOrigin],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    axis_size: Int32,
    unit_numel: Int32,
    my_rank: Int32,
):
    """Reduce-scatter kernel with axis-aware slicing.

    Each GPU slices its partition from all input buffers, reverses the layout
    for coalesced access, reduces, and writes to its output.
    When use_multimem is True, uses hardware-accelerated multimem reduction.
    """
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime num_buffers = 1 if use_multimem else ngpus

    var _my_rank = Int(my_rank)
    var _axis_size = Int(axis_size)
    var _unit_numel = Int(unit_numel)
    var my_sig = rank_sigs[_my_rank]
    var threads_per_gpu = grid_dim.x * BLOCK_SIZE

    var config = ReduceScatterConfig[dtype, ngpus](
        _axis_size, _unit_numel, threads_per_gpu
    )

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )

        # Round-robin access pattern to balance NVLink traffic across GPUs.
        comptime TileType = TileTensor[dtype, in_layout, ImmutAnyOrigin]
        var reordered = Array[_, num_buffers](
            fill_with_unrolled=lambda [i: Int]() -> TileType: in_bufs[
                circular_add[num_buffers](_my_rank, i)
            ]
        )

        var u_start = config.rank_unit_start(_my_rank)
        var n_units = config.rank_units(_my_rank)
        var n_elements = config.rank_num_elements(_my_rank)

        comptime if in_layout.rank == 1:
            # Flat: construct sliced 1D tiles from input TileTensors (any rank).
            comptime FlatLayout = type_of(row_major(n_elements))
            comptime FlatTile = TileTensor[dtype, FlatLayout, ImmutAnyOrigin]
            var elem_start = u_start * config.unit_numel
            var flat_tiles = Array[_, num_buffers](
                fill_with_unrolled=lambda [i: Int]() -> FlatTile: FlatTile(
                    reordered[i]._storage + elem_start,
                    row_major(n_elements),
                )
            )

            _reduce_scatter_impl[
                ngpus, output_lambda=output_lambda, use_multimem=use_multimem
            ](flat_tiles, out_buf, n_elements, config.stride)
        else:
            # 2D axis-aware: slice + reverse for coalesced access.
            comptime InputTile = TileTensor[dtype, in_layout, ImmutAnyOrigin]
            comptime DynShapeTypes = _CoordToDynamic[
                InputTile.linear_idx_type, in_layout._shape_types
            ]
            comptime RevLayout = Layout[
                DynShapeTypes.reverse(),
                in_layout._stride_types.reverse(),
            ]
            comptime SlicedRevTile = TileTensor[
                dtype, RevLayout, ImmutAnyOrigin
            ]

            def sliced_tiles_at[i: Int]() {imm} -> SlicedRevTile:
                comptime if axis == 0:
                    # Scatter along rows.
                    var sliced = reordered[i].slice(
                        (u_start, u_start + n_units),
                        (0, Int(reordered[0].dim[1]())),
                    )
                    return SlicedRevTile(
                        sliced._storage, sliced.layout.reverse()
                    )
                else:
                    # axis == 1: scatter along columns.
                    var col_start = u_start * simd_width
                    var sliced = reordered[i].slice(
                        (0, Int(reordered[0].dim[0]())),
                        (col_start, col_start + n_units * simd_width),
                    )
                    return SlicedRevTile(
                        sliced._storage, sliced.layout.reverse()
                    )

            var sliced_tiles = Array[_, num_buffers](
                fill_with_unrolled=sliced_tiles_at
            )

            _reduce_scatter_impl[
                ngpus, output_lambda=output_lambda, use_multimem=use_multimem
            ](sliced_tiles, out_buf, n_elements, config.stride)

        _multi_gpu_barrier[ngpus, is_start=False, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )


@always_inline
def _reducescatter_p2p[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    *,
    axis: Int = 0,
    output_lambda: elementwise_epilogue_type,
    pdl_level: PDLLevel = PDLLevel(),
    use_multimem: Bool = False,
    group_size: Int = ngpus,
](
    list_of_in_bufs: Array[
        TileTensor[dtype, in_layout, in_origin],
        1 if use_multimem else ngpus,
    ],
    output_buffers: Array[
        TileTensor[mut=True, dtype, out_layout, out_origin], ngpus
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    max_num_blocks: Int,
    ctx: DeviceContext,
    my_rank: Int,
    axis_size: Int,
    unit_numel: Int,
) raises:
    """Performs reducescatter using peer-to-peer access for a single GPU.

    World view vs. group: `ngpus` is the total number of devices in the
    world; `list_of_in_bufs`, `output_buffers`, and `rank_sigs` carry every
    device's data, indexed by GLOBAL device rank (unless `use_multimem`,
    which is full-world only). `group_size` (defaults to `ngpus`) is the
    number of devices that actually cooperate on this reduce-scatter; it
    must evenly divide `ngpus`. `my_rank` is this device's GLOBAL rank in
    `[0, ngpus)` -- the group-local rank and this device's own group's
    slice of the world arrays are derived here, so the whole world stays
    addressable from this function.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Total number of devices in the world.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensors.
        out_origin: Origin of the output TileTensors.
        axis: Scatter axis.
        output_lambda: Elementwise epilogue function to apply to reduced values.
        pdl_level: Control PDL behavior for the kernel.
        use_multimem: Whether multimem optimization is enabled. Only valid
            for a full-world collective (`group_size == ngpus`).
        group_size: Number of devices per independent reduce-scatter group.
            Must evenly divide `ngpus`. Defaults to `ngpus`.

    Args:
        list_of_in_bufs: Input buffers from ALL `ngpus` devices (peer access
            required), indexed by GLOBAL device rank.
        output_buffers: Output buffers for ALL `ngpus` devices' partitions of
            reduced data, indexed by GLOBAL device rank; only
            `output_buffers[my_rank]` is written by this call.
        rank_sigs: All `ngpus` devices' Signal pointers, indexed by GLOBAL
            device rank.
        max_num_blocks: Maximum number of thread blocks to launch.
        ctx: Device context for THIS GPU.
        my_rank: GLOBAL rank of THIS GPU in `[0, ngpus)`.
        axis_size: Number of units along the scatter axis (this device's
            GROUP).
        unit_numel: Number of elements per unit.
    """
    comptime assert (
        ngpus % group_size == 0
    ), "group_size must evenly divide ngpus"
    comptime domain_id = 0 if group_size == ngpus else group_size
    comptime if use_multimem:
        comptime assert group_size == ngpus, (
            "grouped reducescatter (group_size != ngpus) does not support"
            " multimem"
        )

    # This device's group. `group_start` is 0 for a full-world collective, so
    # every group_start-relative read below is byte-identical to the
    # pre-grouping code path in that case.
    var group_start = ualign_down(my_rank, group_size)
    var loc_rank = my_rank - group_start
    var output_buffer = output_buffers[my_rank]

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime BLOCK_SIZE = 256
    comptime num_buffers = 1 if use_multimem else group_size

    # Grid size based on max per-GPU elements (rank 0 has most).
    var config_for_grid = ReduceScatterConfig[dtype, group_size](
        axis_size, unit_numel, 0
    )
    var max_rank_elements = config_for_grid.rank_num_elements(0)

    # We guard against max_rank_elements % simd_width != 0 in reducescatter
    var grid_size = min(
        max_num_blocks,
        ceildiv(max_rank_elements // simd_width, BLOCK_SIZE),
    )

    # Erase origin to ImmutAnyOrigin for the kernel, slicing down to this
    # device's GROUP.
    # TODO(KERN-2526): is this necessary?
    comptime KernelInputType = TileTensor[dtype, in_layout, ImmutAnyOrigin]
    var kernel_in_bufs = Array[_, num_buffers](
        fill_with=lambda (i: Int) -> KernelInputType: KernelInputType(
            list_of_in_bufs[0 if use_multimem else group_start + i]
            ._storage.as_imm()
            .as_unsafe_any_origin(),
            list_of_in_bufs[0 if use_multimem else group_start + i].layout,
        )
    )

    # This device's GROUP's signal pointers, re-indexed to [0, group_size).
    # Byte-identical to `rank_sigs` for a full-world collective.
    var group_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    comptime for i in range(group_size):
        group_sigs[i] = rank_sigs[group_start + i]

    comptime kernel = _reducescatter_kernel[
        dtype,
        in_layout,
        out_layout,
        group_size,
        axis=axis,
        BLOCK_SIZE=BLOCK_SIZE,
        output_lambda=output_lambda,
        use_multimem=use_multimem,
        domain_id=domain_id,
    ]

    # Launch the kernel
    ctx.enqueue_function[kernel](
        kernel_in_bufs,
        output_buffer,
        group_sigs,
        Int32(axis_size),
        Int32(unit_numel),
        Int32(loc_rank),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(pdl_level),
    )


@__parameter
def reducescatter[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    output_lambda: Optional[elementwise_epilogue_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
    *,
    axis: Int = 0,
    use_multimem: Bool = False,
    group_size: Int = ngpus,
](
    input_buffers: Array[
        TileTensor[dtype, in_layout, in_origin],
        1 if use_multimem else ngpus,
    ],
    output_buffers: Array[
        TileTensor[mut=True, dtype, out_layout, out_origin], ngpus
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    _max_num_blocks: Optional[Int] = None,
    my_rank: Optional[Int] = None,
) raises:
    """Per-device reducescatter operation with axis-aware scatter.

    Performs a reduce-scatter across multiple GPUs: each GPU reduces its assigned
    partition from all input buffers and writes the result to its output buffer.

    World view vs. group
    - `ngpus` is the TOTAL number of devices in the world; `input_buffers`,
      `output_buffers`, and `rank_sigs` carry every device's data, indexed by
      GLOBAL device rank.
    - `group_size` (defaults to `ngpus`) is the number of devices that actually
      cooperate on one reduce-scatter. It must evenly divide `ngpus`. Devices
      `[g*group_size, (g+1)*group_size)` form group `g`; this call's group is
      derived from `my_rank`.
    - `my_rank` is this device's GLOBAL rank in `[0, ngpus)`, not its rank
      within the group -- the group-local rank is derived internally.

    Parameters:
        dtype: Data dtype of tensor elements.
        ngpus: Total number of devices in the world.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensors.
        out_origin: Origin of the output TileTensors.
        output_lambda: Optional elementwise epilogue function. If not provided,
            reduced values are stored directly to this device's output buffer.
        pdl_level: Control PDL behavior for the kernel.
        axis: Scatter axis. 0 to scatter along rows (default), 1 to scatter along columns.
            Requires 2D row-major inputs when axis >= 0.
        use_multimem: If True, use hardware-accelerated multimem reduction.
            Currently only valid with 1D input and a full-world collective
            (`group_size == ngpus`).
        group_size: Number of devices per independent reduce-scatter group.
            Must evenly divide `ngpus`. Defaults to `ngpus` (one full-world
            group, byte-identical to the pre-grouping behavior).

    Args:
        input_buffers: Input TileTensors from ALL `ngpus` devices (peer access
            required), indexed by GLOBAL device rank. When use_multimem is
            True, a single multimem-mapped TileTensor.
        output_buffers: Output TileTensors for ALL `ngpus` devices' partitions
            of reduced data, indexed by GLOBAL device rank; only
            `output_buffers[my_rank]` is written by this call.
        rank_sigs: All `ngpus` devices' Signal pointers, indexed by GLOBAL
            device rank.
        ctx: Device context for THIS GPU.
        _max_num_blocks: Optional maximum number of thread blocks to launch.
            If not specified, uses an arch-specific default (128 on AMD,
            else MAX_NUM_BLOCKS_UPPER_BOUND).
        my_rank: Optional GLOBAL rank of THIS GPU in `[0, ngpus)`. Defaults to
            the physical device id, which is only correct when devices
            `0..ngpus-1` map 1:1 onto physical device ids.

    Raises:
        Error: If P2P access is not available between GPUs (always required;
            a grouped reduce-scatter has no non-P2P fallback).
        Error: If input buffer size is not a multiple of SIMD width.
    """
    comptime assert (
        group_size >= 2
    ), "reducescatter requires at least 2 GPUs per group"
    comptime assert (
        ngpus % group_size == 0
    ), "group_size must evenly divide ngpus"
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime tensor_rank = in_layout.rank

    # Validate axis and rank combination.
    # TODO(KERN-2526): generalize to higher dims
    comptime assert tensor_rank <= 2, "Currently only 1D and 2D input supported"
    comptime assert axis < tensor_rank, "Invalid scatter axis for given rank"
    comptime assert axis >= 0, "Scatter axis must be positive"

    comptime if group_size != ngpus:
        comptime assert not use_multimem, (
            "grouped reducescatter (group_size != ngpus) does not support"
            " multimem"
        )

    # This device's group. `group_start` is 0 for a full-world collective
    # (group_size == ngpus), so every group_start-relative read below is
    # byte-identical to the pre-grouping `input_buffers[0]` in that case.
    var global_rank = my_rank.value() if my_rank else Int(ctx.id())
    var group_start = ualign_down(global_rank, group_size)
    var output_buffer = output_buffers[global_rank]

    # Return early if the input buffer is empty. Read from THIS DEVICE'S
    # GROUP, not world index 0 -- sibling groups may carry different
    # (symbolic) shapes.
    var num_elements = input_buffers[group_start].num_elements()
    if num_elements == 0:
        return

    if not is_p2p_enabled():
        raise Error("Reducescatter currently requires P2P access between GPUs")

    # Compute axis_size and unit_numel based on axis.
    var axis_size: Int
    var unit_numel: Int
    comptime if tensor_rank == 1:
        # 1D: partition by SIMD vectors
        if num_elements % simd_width != 0:
            raise Error(
                "non SIMD-width multiple number of elements unsupported by"
                " reducescatter"
            )
        axis_size = num_elements // simd_width
        unit_numel = simd_width
    elif axis == 0:
        # 2D axis-0: partition rows, unit = one row
        var dim_0 = Int(input_buffers[group_start].layout.shape[0]().value())
        var dim_1 = Int(input_buffers[group_start].layout.shape[1]().value())
        if dim_1 % simd_width != 0:
            raise Error(
                "inner dimension (axis 1) must be a multiple of SIMD width"
                " for axis-0 reduce-scatter"
            )
        axis_size = dim_0
        unit_numel = dim_1
    else:
        # axis == 1: partition column groups, unit = simd_width columns
        var dim_0 = Int(input_buffers[group_start].layout.shape[0]().value())
        var dim_1 = Int(input_buffers[group_start].layout.shape[1]().value())
        if dim_1 % simd_width != 0:
            raise Error(
                "scatter dimension (axis 1) must be a multiple of SIMD width"
                " for axis-1 reduce-scatter"
            )
        axis_size = dim_1 // simd_width
        unit_numel = dim_0 * simd_width

    # Validate output buffer shape for this rank's partition. Validated
    # against the GROUP (not the world): each device's output only ever holds
    # its own group's shard.
    var loc_rank = global_rank - group_start
    var config_check = ReduceScatterConfig[dtype, group_size](
        axis_size, unit_numel, 0
    )
    var expected_numel = config_check.rank_num_elements(loc_rank)
    comptime if tensor_rank == 1:
        if output_buffer.num_elements() != expected_numel:
            raise Error(
                "output buffer has "
                + String(output_buffer.num_elements())
                + " elements, expected "
                + String(expected_numel)
            )
    else:
        comptime assert (
            output_buffer.rank == 2
        ), "axis >= 0 requires 2D output buffer"
        var n_units = config_check.rank_units(loc_rank)
        var expected_rows = n_units if axis == 0 else Int(
            input_buffers[group_start].layout.shape[0]().value()
        )
        var expected_cols = (
            Int(input_buffers[group_start].layout.shape[1]().value()) if axis
            == 0 else n_units * simd_width
        )
        var out_rows = Int(output_buffer.dim[0]())
        var out_cols = Int(output_buffer.dim[1]())
        if out_rows != expected_rows or out_cols != expected_cols:
            raise Error(
                "output buffer shape ("
                + String(out_rows)
                + ", "
                + String(out_cols)
                + "), expected ("
                + String(expected_rows)
                + ", "
                + String(expected_cols)
                + ")"
            )

    # AMD P2P reduce-scatter is PCIe-fabric-bound: ~128 blocks saturate it
    # (CDNA4: 166.5 vs 148.8 GB/s at 1024); more only adds barrier overhead.
    comptime _default_num_blocks = (
        128 if has_amd_gpu_accelerator() else MAX_NUM_BLOCKS_UPPER_BOUND
    )
    var max_num_blocks = (
        _max_num_blocks.value() if _max_num_blocks else _default_num_blocks
    )

    # Default epilogue: store directly to this device's output buffer
    @always_inline
    @__parameter
    @__copy_capture(output_buffer)
    def default_output_lambda[
        _dtype: DType,
        _width: SIMDLength,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
        output_buffer.store[width=_width, alignment=_alignment](
            coords, val.cast[dtype]()
        )

    comptime actual_output_lambda = default_output_lambda if not output_lambda else output_lambda.value()

    # Hand the collective the whole world plus the group width, and let it
    # derive the group-local slice, rank, and barrier domain itself.
    _reducescatter_p2p[
        dtype,
        ngpus,
        axis=axis,
        output_lambda=actual_output_lambda,
        pdl_level=pdl_level,
        use_multimem=use_multimem,
        group_size=group_size,
    ](
        input_buffers,
        output_buffers,
        rank_sigs,
        max_num_blocks,
        ctx,
        global_rank,
        axis_size,
        unit_numel,
    )
