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
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
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
    size_of,
    align_of,
    has_amd_gpu_accelerator,
    is_amd_gpu,
)

from internal_utils import Table

from .device_query import GB, KB, dispatch_select_comm_config
from .relay import (
    RELAY_ARCH,
    RelayTuningConfig,
    _relay_pairs,
    _relay_slice_vectors,
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


@inline(.always)
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
            .load[width=simd_width, alignment=alignment](Coord(elem_idx))
            .cast[accum_type]()
        )

        comptime for gpu_idx in range(1, ngpus):
            accum += (
                in_tiles[gpu_idx]
                .address_space_cast[_target_address_space]()
                .load[width=simd_width, alignment=alignment](Coord(elem_idx))
                .cast[accum_type]()
            )
        return accum.cast[dtype]()


# Tuning table for the relay-assisted grouped reduce-scatter. `num_bytes`
# buckets are per-GPU output partitions -- the quantity that sets per-link
# traffic -- and must stay in ascending order within an (arch, ngpus) group.
# `relay_percent` is the share of each destination's partition that its relays
# reduce.
comptime reducescatter_relay_tuning_table = Table(
    [
        # Default for group widths with no rows of their own.
        RelayTuningConfig(
            group_size=-1,
            num_bytes=-1,
            num_blocks=20,
            num_relay_blocks=12,
            relay_percent=44,
        ),
        # Small partitions are barrier-bound rather than link-bound, so the
        # extra links buy less than the wider barrier costs: measured on
        # MI355X the relay path loses about 10% at and below these sizes and
        # wins from twice them upwards. A zero share declines it outright.
        # Larger partitions fall through to the default above.
        RelayTuningConfig(
            group_size=4,
            num_bytes=(256 * KB),
            num_blocks=0,
            num_relay_blocks=0,
            relay_percent=0,
        ),
        RelayTuningConfig(
            group_size=2,
            num_bytes=(128 * KB),
            num_blocks=0,
            num_relay_blocks=0,
            relay_percent=0,
        ),
    ],
    "reducescatter_relay_table",
)

# Tuning table for the relay-assisted grouped reduce-scatter WITH a residual
# fold, selected by `has_residual`. Same shape and conventions as the table
# above; the fold puts one more read on a relay's first leg, which moves where
# the links balance.
comptime reducescatter_relay_residual_tuning_table = Table(
    [
        RelayTuningConfig(
            group_size=-1,
            num_bytes=-1,
            num_blocks=20,
            num_relay_blocks=12,
            relay_percent=36,
        ),
        # Declined below the same barrier-bound cut as the plain table, which
        # measurement puts one bucket lower here: the fold slows the baseline
        # too, so relaying already pays at 256 KB.
        RelayTuningConfig(
            group_size=4,
            num_bytes=(128 * KB),
            num_blocks=0,
            num_relay_blocks=0,
            relay_percent=0,
        ),
        RelayTuningConfig(
            group_size=2,
            num_bytes=(128 * KB),
            num_blocks=0,
            num_relay_blocks=0,
            relay_percent=0,
        ),
        # Every partition above the cut; `num_bytes` is a ceiling, not a
        # bucket, so this stands in for the per-width default the lookup has
        # no way to express.
        RelayTuningConfig(
            group_size=2,
            num_bytes=(2 * GB),
            num_blocks=20,
            num_relay_blocks=12,
            relay_percent=32,
        ),
    ],
    "reducescatter_relay_residual_table",
)


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

    @inline(.always)
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

    @inline(.always)
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

    @inline(.always)
    def rank_unit_start(self, rank: Int) -> Int:
        """Start unit index along scatter axis for this rank."""
        return rank * self.axis_part + min(rank, self.axis_remainder)

    @inline(.always)
    def rank_units(self, rank: Int) -> Int:
        """Number of units for this rank."""
        return self.axis_part + Int(rank < self.axis_remainder)

    @inline(.always)
    def rank_num_elements(self, rank: Int) -> Int:
        """Total elements for this rank."""
        return self.rank_units(rank) * self.unit_numel

    @inline(.always)
    def rank_start(self, rank: Int) -> Int:
        """Flat element start offset for this rank."""
        return self.rank_unit_start(rank) * self.unit_numel

    @inline(.always)
    def rank_end(self, rank: Int) -> Int:
        """Flat element end offset for this rank."""
        return self.rank_start(rank + 1)

    @inline(.always)
    def rank_part(self, rank: Int) -> Int:
        """Number of elements for this rank (alias for rank_num_elements)."""
        return self.rank_num_elements(rank)

    @inline(.always)
    def thr_local_start(self, thread_idx: Int) -> Int:
        return thread_idx * Self.simd_width


@inline(.always)
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


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"reducescatter_relay_{dtype}")
def _reducescatter_relay_kernel[
    dtype: DType,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    has_residual: Bool = False,
    domain_id: Int = 0,
](
    out_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    in_ptrs: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    peer_out_ptrs: StaticTuple[MutPointer[Scalar[dtype], MutAnyOrigin], ngpus],
    peer_in_ptrs: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_start: Int32,
    my_numel: Int32,
    peer_starts: StaticTuple[Int32, ngpus],
    peer_numels: StaticTuple[Int32, ngpus],
    residual_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    peer_residual_ptrs: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ],
    relay_percent: Int32,
    num_direct_blocks: Int32,
    my_rank: Int32,
):
    """Relay-assisted P2P kernel for a grouped reduce-scatter.

    Blocks below `num_direct_blocks` take the direct role and the rest take the
    relay role; every block joins both barriers, since the barrier pairs blocks
    by id across GPUs and an early return would hang the node.

    With `has_residual`, each reduced value is folded with the residual at the
    same flat offset before it is stored. The residual is per-device data, so
    each role reads the copy belonging to the GPU whose output it is writing:
    the direct role reads `residual_ptr`, and a relay reads the destination's
    over the same inter-group link it already reads that group's inputs on.
    """
    comptime world_size = 2 * ngpus
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var _my_rank = Int(my_rank)
    var my_sig = rank_sigs[_my_rank]
    # Groups occupy contiguous rank ranges within the pair, so the group-local
    # rank is the pair-local rank folded by the group width.
    var group_rank = _my_rank % ngpus

    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var _relay_percent = Int(relay_percent)

    # Synchronize before reading. The domain spans both groups because a
    # relay reads and writes GPUs outside its own group.
    _multi_gpu_barrier[world_size, is_start=True, domain_id=domain_id](
        rank_sigs, my_sig, _my_rank
    )

    if bid < Int(num_direct_blocks):
        # Direct role: reduce the leading region of MY partition out of
        # every group input, exactly as the plain kernel does. The trailing
        # slices of that partition arrive from the other group's relays.
        var num_simd_vectors = Int(my_numel) // simd_width
        var span = (
            num_simd_vectors
            - _relay_slice_vectors[ngpus](num_simd_vectors, _relay_percent)
            * ngpus
        )
        var base = Int(my_start)
        var stride = Int(num_direct_blocks) * BLOCK_SIZE

        for idx in range(bid * BLOCK_SIZE + tid, span, stride):
            var elem_idx = idx * simd_width
            # Rotate the source order by rank, as the plain kernel does, so
            # the group's reads do not all queue on the same peer first.
            var accum = SIMD[accum_type, simd_width](0)
            comptime for i in range(ngpus):
                accum += (
                    in_ptrs[circular_add[ngpus](group_rank, i)]
                    .address_space_cast[_target_address_space]()
                    .load[
                        width=simd_width,
                        alignment=alignment,
                    ](base + elem_idx)
                    .cast[accum_type]()
                )
            # Round the reduction to `dtype` first and fold the residual in
            # `accum_type`, which is the order the plain kernel's epilogue
            # uses; matching it keeps the two bit-identical.
            var reduced = accum.cast[dtype]()
            comptime if has_residual:
                reduced = (
                    reduced.cast[accum_type]()
                    + residual_ptr.address_space_cast[_target_address_space]()
                    .load[width=simd_width, alignment=alignment](
                        base + elem_idx
                    )
                    .cast[accum_type]()
                ).cast[dtype]()
            out_ptr.address_space_cast[_target_address_space]().store[
                width=simd_width, alignment=alignment
            ](elem_idx, reduced)
    else:
        # Reduce-relay role: take a slice of one peer-group destination's
        # partition, pull it from ALL `ngpus` of that group's inputs --
        # including the destination's own contribution, over a link a
        # grouped collective leaves idle -- sum it, and remote-write the
        # finished value into that destination's output.
        #
        # Summing the whole partition here rather than forwarding partial
        # sums is what keeps the second leg cheap: an accumulator would
        # have to travel in `accum_type` and cost twice the bytes. Every
        # add is in `accum_type` with a single final rounding, the same
        # numerics contract as the plain kernel, though the relayed region
        # is summed in a different order than the direct region -- as it
        # already is between ranks, since the plain kernel rotates too.
        var relay_bid = bid - Int(num_direct_blocks)
        var dst = relay_bid % ngpus
        var blocks_per_dst = (Int(grid_dim.x) - Int(num_direct_blocks)) // ngpus
        var dst_block = relay_bid // ngpus

        # Each destination's partition is split on its own length, so a
        # peer group with uneven partitions simply gives its relays uneven
        # work, and one too short to split gives an empty range.
        var dst_vectors = Int(peer_numels[dst]) // simd_width
        var slice_vectors = _relay_slice_vectors[ngpus](
            dst_vectors, _relay_percent
        )
        var slice_start = (
            dst_vectors - slice_vectors * ngpus
        ) + group_rank * slice_vectors
        var start = slice_start + dst_block * slice_vectors // blocks_per_dst
        var end = (
            slice_start + (dst_block + 1) * slice_vectors // blocks_per_dst
        )

        # Hoist the destination's pointers: indexing by a runtime
        # destination inside the loop makes the compiler stage the whole
        # table through scratch memory.
        var base = Int(peer_starts[dst])
        var dst_ptr = peer_out_ptrs[dst].address_space_cast[
            _target_address_space
        ]()
        var dst_res_ptr = peer_residual_ptrs[dst].address_space_cast[
            _target_address_space
        ]()

        # Rotate the source order by the DESTINATION's rank, which is the
        # order that destination's own direct blocks use. Relayed and
        # directly-reduced values then come out of the same sequence of
        # adds, so the result is bit-identical to the plain kernel's --
        # which the fused reduce-scatter + norm op is tested against.
        # Hoisted for the same reason as `dst_ptr`: a runtime index into
        # the table inside the loop would stage it through scratch.
        comptime SrcPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
        var src_ptrs = Array[_, ngpus](
            fill_with_unrolled=lambda [g: Int]() -> SrcPtrType: peer_in_ptrs[
                circular_add[ngpus](dst, g)
            ]
        )

        # Operands a thread batches before reducing: the read links and the
        # write link only overlap while several remote loads -- which are
        # latency-bound per thread -- are in flight.
        comptime UNROLL = 4

        for idx in range(start + tid, end, BLOCK_SIZE * UNROLL):
            var data = Array[SIMD[dtype, simd_width], ngpus * UNROLL](
                uninitialized=True
            )
            comptime for u in range(UNROLL):
                if idx + u * BLOCK_SIZE < end:
                    var elem_idx = (idx + u * BLOCK_SIZE) * simd_width
                    comptime for g in range(ngpus):
                        data[u * ngpus + g] = (
                            src_ptrs[g]
                            .address_space_cast[_target_address_space]()
                            .load[
                                width=simd_width,
                                alignment=alignment,
                            ](base + elem_idx)
                        )

            comptime for u in range(UNROLL):
                if idx + u * BLOCK_SIZE < end:
                    var accum = SIMD[accum_type, simd_width](0)
                    comptime for g in range(ngpus):
                        accum += data[u * ngpus + g].cast[accum_type]()
                    var elem_idx = (idx + u * BLOCK_SIZE) * simd_width
                    var reduced = accum.cast[dtype]()
                    comptime if has_residual:
                        # The DESTINATION's residual, not this relay's: the
                        # tensor is per-device, so only that copy carries the
                        # values the destination would have folded itself.
                        reduced = (
                            reduced.cast[accum_type]()
                            + dst_res_ptr.load[
                                width=simd_width, alignment=alignment
                            ](base + elem_idx).cast[accum_type]()
                        ).cast[dtype]()
                    dst_ptr.store[width=simd_width, alignment=alignment](
                        elem_idx, reduced
                    )

    # Synchronize after writing. This is also what publishes the relay
    # stores to their destination GPUs.
    _multi_gpu_barrier[world_size, is_start=False, domain_id=domain_id](
        rank_sigs, my_sig, _my_rank
    )


@inline(.always)
def _reducescatter_p2p_relay[
    dtype: DType,
    ngpus: Int,
    has_residual: Bool = False,
](
    out_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    in_ptrs: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    peer_out_ptrs: StaticTuple[MutPointer[Scalar[dtype], MutAnyOrigin], ngpus],
    peer_in_ptrs: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_start: Int,
    my_numel: Int,
    peer_starts: StaticTuple[Int32, ngpus],
    peer_numels: StaticTuple[Int32, ngpus],
    residual_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    peer_residual_ptrs: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ],
    recipe: RelayTuningConfig,
    ctx: DeviceContext,
    my_rank: Int,
) raises:
    """Per-device reduce-scatter over one of two groups, assisted by the other.

    Two groups of `ngpus` GPUs running a grouped reduce-scatter concurrently
    use only their intra-group links; every link between the groups sits idle.
    This path hands a trailing fraction of each destination's partition to the
    other group, whose GPUs act as reduce nodes: one pulls that slice from all
    `ngpus` inputs of the destination's group, sums it, and writes the finished
    value straight into the destination's output. Every directed link then
    carries a fraction of a partition instead of a whole one, which lowers the
    topology floor.

    Partitions are split on their own lengths, so a group's ranks may hold
    uneven ones; what all `2 * ngpus` GPUs must agree on is the recipe and the
    block counts, since they run one kernel and one barrier domain and the
    barrier pairs blocks by id. A node running more than two groups pairs them
    up, and each pair is its own relay world.

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Number of GPUs in one group; the relay world holds `2 * ngpus`.
        has_residual: Fold the residual into each reduced value.

    Args:
        out_ptr: This GPU's output partition.
        in_ptrs: This group's inputs, by group-local rank.
        peer_out_ptrs: The other group's output partitions, by that group's
            local rank. This GPU writes them when it acts as a relay.
        peer_in_ptrs: The other group's inputs, by that group's local rank.
            This GPU reads all of them when it acts as a relay.
        rank_sigs: Signals for the `2 * ngpus` GPUs of this relay world, packed
            in relay-world rank order.
        my_start: Element offset of this GPU's partition inside an input.
        my_numel: Elements in this GPU's partition.
        peer_starts: Element offset of each peer-group destination's partition
            inside a peer input.
        peer_numels: Elements in each of those partitions.
        residual_ptr: This device's residual tensor, folded into the values
            this device reduces for itself when `has_residual`.
        peer_residual_ptrs: The other group's residual tensors, by that group's
            local rank. A relay folds the destination's when `has_residual`,
            since the residual is per-device and only that copy holds what the
            destination would have folded itself.
        recipe: Block counts and relayed fraction, from
            `reducescatter_relay_tuning_table`.
        ctx: Device context for THIS GPU.
        my_rank: This GPU's rank within the relay world.
    """
    # The barrier bank follows the file's convention of keying domains by
    # collective width, and this barrier is `2 * ngpus` wide -- NOT the width
    # the plain path's domain describes. Reusing that one would let an
    # `ngpus`-wide grouped barrier and this one advance the same counters at
    # different rates and hang the node.
    comptime relay_domain_id = 0 if 2 * ngpus == MAX_GPUS else 2 * ngpus
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()

    # A relay block only runs if some partition in the relay world is long
    # enough to split. Both groups see both length sets, so they agree on the
    # grid -- deciding from one group's partitions alone would let the two
    # launch different grids and hang the barrier.
    var any_relayed = (
        _relay_slice_vectors[ngpus](
            my_numel // simd_width, recipe.relay_percent
        )
        > 0
    )
    comptime for i in range(ngpus):
        if (
            _relay_slice_vectors[ngpus](
                Int(peer_numels[i]) // simd_width, recipe.relay_percent
            )
            > 0
        ):
            any_relayed = True

    # The direct role splits its blocks over one contiguous range while the
    # relay role fans them over `ngpus` destinations, so only the latter is
    # rounded to a multiple of the group width.
    var num_direct_blocks = max(1, recipe.num_blocks)
    var num_relay_blocks = 0
    if any_relayed:
        num_relay_blocks = ngpus * max(1, recipe.num_relay_blocks // ngpus)

    comptime BLOCK_SIZE = 256
    comptime relay_kernel = _reducescatter_relay_kernel[
        dtype,
        ngpus,
        BLOCK_SIZE=BLOCK_SIZE,
        has_residual=has_residual,
        domain_id=relay_domain_id,
    ]
    ctx.enqueue_function[relay_kernel](
        out_ptr,
        in_ptrs,
        peer_out_ptrs,
        peer_in_ptrs,
        rank_sigs,
        Int32(my_start),
        Int32(my_numel),
        peer_starts,
        peer_numels,
        residual_ptr,
        peer_residual_ptrs,
        Int32(recipe.relay_percent),
        Int32(num_direct_blocks),
        Int32(my_rank),
        grid_dim=num_direct_blocks + num_relay_blocks,
        block_dim=BLOCK_SIZE,
    )


@inline(.always)
def _reducescatter_p2p[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    *,
    axis: Int = 0,
    output_lambda: Optional[elementwise_epilogue_type] = None,
    has_residual: Bool = False,
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
    residuals: Optional[
        Array[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus]
    ] = None,
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
        has_residual: Fold `residuals` into every reduced value.
        pdl_level: Control PDL behavior for the kernel.
        use_multimem: Whether multimem optimization is enabled. Only valid
            for a full-world collective (`group_size == ngpus`).
        group_size: Number of devices per independent reduce-scatter group.
            Must evenly divide `ngpus`. Defaults to `ngpus`.

    Args:
        list_of_in_bufs: Input buffers from ALL `ngpus` devices (peer access
            required), indexed by GLOBAL device rank.
        output_buffers: Output buffers for ALL `ngpus` devices' partitions of
            reduced data, indexed by GLOBAL device rank. Every slot must name
            a real buffer: a grouped reduce-scatter may hand part of a
            partition to the paired group, whose GPUs finish the sum and write
            that partition directly, so peer slots are not spare.
        rank_sigs: All `ngpus` devices' Signal pointers, indexed by GLOBAL
            device rank.
        max_num_blocks: Maximum number of thread blocks to launch.
        ctx: Device context for THIS GPU.
        my_rank: GLOBAL rank of THIS GPU in `[0, ngpus)`.
        axis_size: Number of units along the scatter axis (this device's
            GROUP).
        unit_numel: Number of elements per unit.
        residuals: Every device's residual tensor, indexed by GLOBAL device
            rank; required when `has_residual`. Each is the size of that
            device's whole input, and a device folds only its own partition's
            slice. Peer slots are not spare: a relay folds the destination's
            residual, since the tensors are per-device and only that copy
            holds what the destination would have folded itself.
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

    # Relay path: with an even number of groups, adjacent groups pair off and
    # act as reduce nodes for each other over the inter-group links a grouped
    # collective leaves idle. Gated on an arch it has been measured on, and off
    # for a caller epilogue -- a relay finishes a value into a peer's output
    # and cannot run that peer's epilogue -- and off for multimem, whose
    # single fused input leaves a relay nothing per-source to reduce.
    #
    # Also off for a nonzero `axis`. Nothing below is axis-specific: a relay
    # locates a peer's partition by the running sum of that group's output
    # lengths, which equals `ReduceScatterConfig.rank_start` for any axis. But
    # that is an argument, not a test, and the test harness only builds axis-0
    # partitions -- untested cross-device index arithmetic in a collective is
    # how silent corruption happens.
    # TODO(sliu): cover axis 1 and drop this clause.
    comptime _use_relay = (
        _relay_pairs[ngpus, group_size](ctx.default_device_info.version)
        and axis == 0
        and not use_multimem
        and not output_lambda
    )
    comptime if _use_relay:
        var pair_base = ualign_down(my_rank, 2 * group_size)
        var peer_start = (
            pair_base + group_size if group_start == pair_base else pair_base
        )

        comptime SrcPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
        comptime OutPtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
        var in_ptrs = StaticTuple[SrcPtrType, group_size]()
        var peer_in_ptrs = StaticTuple[SrcPtrType, group_size]()
        var peer_out_ptrs = StaticTuple[OutPtrType, group_size]()
        var peer_starts = StaticTuple[Int32, group_size]()
        var peer_numels = StaticTuple[Int32, group_size]()

        # Partitions tile an input contiguously in flat element order, so the
        # peer group's layout is the running sum of its output lengths -- no
        # need to re-derive its axis split, which this rank cannot see anyway.
        var peer_offset = 0
        # Every rank in the pair must reach the same verdict, so the lookup is
        # keyed on the largest partition anywhere in it.
        var pair_max_numel = config_for_grid.rank_num_elements(loc_rank)

        comptime for i in range(group_size):
            in_ptrs[i] = rebind[SrcPtrType](
                list_of_in_bufs[group_start + i].ptr
            )
            peer_in_ptrs[i] = rebind[SrcPtrType](
                list_of_in_bufs[peer_start + i].ptr
            )
            peer_out_ptrs[i] = rebind[OutPtrType](
                output_buffers[peer_start + i].ptr
            )

            var peer_numel = output_buffers[peer_start + i].num_elements()
            peer_starts[i] = Int32(peer_offset)
            peer_numels[i] = Int32(peer_numel)
            peer_offset += peer_numel
            pair_max_numel = max(pair_max_numel, peer_numel)

        comptime relay_sm_version = ctx.default_device_info.version
        # The fold changes what a relay's first leg costs, so it gets its own
        # recipe rather than a correction on the plain one.
        comptime relay_table = reducescatter_relay_residual_tuning_table if has_residual else reducescatter_relay_tuning_table
        var recipe = dispatch_select_comm_config[
            group_size, relay_sm_version, relay_table
        ](pair_max_numel * size_of[dtype]())

        if pair_max_numel > 0 and recipe.relay_percent > 0:
            var relay_sigs = Array[
                UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS
            ](uninitialized=True)
            for i in range(2 * group_size):
                relay_sigs[i] = rank_sigs[pair_base + i]

            # A relay folds the DESTINATION's residual, so the peer group's
            # copies travel with its inputs. The kernel reads none of this
            # unless `has_residual`; pass the inputs rather than nulls so the
            # arguments stay well-formed.
            var res_ptr = in_ptrs[0]
            var peer_res_ptrs = peer_in_ptrs
            comptime if has_residual:
                res_ptr = residuals.value()[my_rank]
                comptime for i in range(group_size):
                    peer_res_ptrs[i] = residuals.value()[peer_start + i]

            return _reducescatter_p2p_relay[has_residual=has_residual](
                rebind[OutPtrType](output_buffer.ptr),
                in_ptrs,
                peer_out_ptrs,
                peer_in_ptrs,
                relay_sigs,
                config_for_grid.rank_start(loc_rank),
                config_for_grid.rank_num_elements(loc_rank),
                peer_starts,
                peer_numels,
                res_ptr,
                peer_res_ptrs,
                recipe,
                ctx,
                my_rank - pair_base,
            )

    # Only reachable for a relay pair whose partner carried the work, and
    # whose size the table declined to relay: this group has nothing of its
    # own to reduce, and a zero-width grid is not a legal launch.
    if max_rank_elements == 0:
        return

    # Storing straight to the output is the epilogue when the caller supplied
    # none. Keeping the `Optional` this far down is what lets the relay path
    # above see that there is no caller epilogue for a relay to honour.
    # Storing to the output is the epilogue when the caller supplied none.
    # With `has_residual` it also folds the residual, in the same order the
    # relay kernel uses: `val` arrives already rounded to `dtype`, so the add
    # happens in the accumulate type with one further rounding. Expressing the
    # fold here rather than in a caller lambda is what lets the relay path
    # reproduce it exactly.
    comptime _accum_type = get_accum_type[dtype]()
    var res_base = config_for_grid.rank_start(loc_rank)
    var res_ptr_local = residuals.value()[my_rank] if has_residual else rebind[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin]
    ](list_of_in_bufs[0].ptr)

    @__parameter
    @__copy_capture(output_buffer, res_ptr_local, res_base)
    def default_output_lambda[
        _dtype: DType,
        _width: SIMDLength,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
        comptime if has_residual:
            var local_flat = Int(output_buffer.layout(coords))
            var res = res_ptr_local.load[width=_width](local_flat + res_base)
            output_buffer.store[width=_width, alignment=_alignment](
                coords,
                (val.cast[_accum_type]() + res.cast[_accum_type]()).cast[
                    dtype
                ](),
            )
        else:
            output_buffer.store[width=_width, alignment=_alignment](
                coords, val.cast[dtype]()
            )

    comptime actual_output_lambda = default_output_lambda if not output_lambda else output_lambda.value()

    comptime kernel = _reducescatter_kernel[
        dtype,
        in_layout,
        out_layout,
        group_size,
        axis=axis,
        BLOCK_SIZE=BLOCK_SIZE,
        output_lambda=actual_output_lambda,
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
    has_residual: Bool = False,
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
    residuals: Optional[
        Array[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus]
    ] = None,
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
        has_residual: Fold `residuals` into every reduced value, after the
            reduction is rounded to `dtype` and with the add itself in the
            accumulate type -- the order an unfused residual add would give.
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
            of reduced data, indexed by GLOBAL device rank. Every slot must
            name a real buffer: a grouped reduce-scatter may hand part of a
            partition to the paired group, whose GPUs finish the sum and write
            that partition directly, so peer slots are not spare.
        rank_sigs: All `ngpus` devices' Signal pointers, indexed by GLOBAL
            device rank.
        ctx: Device context for THIS GPU.
        _max_num_blocks: Optional maximum number of thread blocks to launch.
            If not specified, uses an arch-specific default (128 on AMD,
            else MAX_NUM_BLOCKS_UPPER_BOUND).
        my_rank: Optional GLOBAL rank of THIS GPU in `[0, ngpus)`. Defaults to
            the physical device id, which is only correct when devices
            `0..ngpus-1` map 1:1 onto physical device ids.
        residuals: Every device's residual tensor, indexed by GLOBAL device
            rank; required when `has_residual` and ignored otherwise. Each has
            the same shape as that device's input, and a device folds only the
            slice covering its own partition.

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

    # Return early only if the whole relay pair is empty. Reads come from THIS
    # DEVICE'S GROUP, not world index 0 -- sibling groups may carry different
    # (symbolic) shapes. The scan widens to the pair when relaying is possible:
    # a group with no input of its own still has to launch, because its GPUs
    # act as reduce relays for the partner and the pair shares one barrier, so
    # returning here would hang the partner.
    comptime _empty_scan = (
        2
        * group_size if _relay_pairs[ngpus, group_size](
            ctx.default_device_info.version
        ) else group_size
    )
    var scan_start = ualign_down(global_rank, _empty_scan)
    var pair_empty = True
    comptime for i in range(_empty_scan):
        if input_buffers[scan_start + i].num_elements() > 0:
            pair_empty = False
            break
    if pair_empty:
        return

    var num_elements = input_buffers[group_start].num_elements()

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

    # Hand the collective the whole world plus the group width, and let it
    # derive the group-local slice, rank, and barrier domain itself.
    _reducescatter_p2p[
        dtype,
        ngpus,
        axis=axis,
        output_lambda=output_lambda,
        has_residual=has_residual,
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
        residuals,
    )
