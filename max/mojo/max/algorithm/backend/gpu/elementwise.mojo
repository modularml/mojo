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
"""GPU implementation of elementwise functions."""

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
)
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    cluster_sync_acquire,
    cluster_sync_release,
    clusterlaunchcontrol_try_cancel,
    clusterlaunchcontrol_query_cancel_is_canceled,
    clusterlaunchcontrol_query_cancel_get_first_ctaid,
    elect_one_sync_with_mask,
)
from max.gpu.primitives.grid_controls import (
    pdl_launch_attributes,
)  # @doc_hidden
from max.gpu.host import DeviceContext
from max.gpu.host.info import B200
from max.gpu.sync import mbarrier_init, mbarrier_arrive_expect_tx_relaxed
from std.math import ceildiv, clamp
from std.math.uutils import ufloordiv, uceildiv, udivmod
from std.memory import unsafe_stack_allocation
from std.sys._assembly import inlined_assembly
from std.sys.defines import get_defined_bool, get_defined_int
from std.sys.info import _has_sm_100x_or_newer, has_nvidia_gpu_accelerator
from std.utils.coord import Coord, CoordLike, coord_to_index_list
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from std.algorithm.backend.unswitch import unswitch
from max.algorithm.functional import _get_start_indices_of_nth_subvolume


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#

comptime _USE_CLC_WORK_STEALING = get_defined_bool[
    "USE_CLC_WORK_STEALING", False
]()

comptime _PDL_LEVEL = PDLLevel.ON


@always_inline("nodebug")
def _mbarrier_wait_acquire_cta(
    mbar: Pointer[mut=True, Int64, _, address_space=.SHARED],
    phase: UInt32,
):
    """Spin-waits on an mbarrier until the given phase completes, with acquire
    semantics at CTA scope."""
    inlined_assembly[
        """{
            .reg .pred P1;
            LAB_WAIT:
            mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [$0], $1;
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
        }""",
        NoneType,
        constraints="r,r",
        has_side_effect=True,
    ](Int32(Int(mbar)), phase)


@always_inline("nodebug")
def _advance_indices[
    rank: Int
](
    mut idx: IndexList[rank, element_type=_],
    shape: IndexList[rank, element_type=_],
):
    """Advances a multi-dimensional index by one element in row-major order."""
    idx[rank - 1] += 1
    comptime for d in reversed(range(1, rank)):
        if idx[d] >= shape[d]:
            idx[d] = 0
            idx[d - 1] += 1


@always_inline("nodebug")
def _start_indices_of_nth_element[
    rank: Int, //, shape_types: TypeList[Trait=CoordLike, ...]
](n: Int, shape: IndexList[rank, element_type=_], out res: type_of(shape)):
    """Converts a flat index into row-major ND start indices.

    Divides by a compile-time literal for every dimension whose extent the
    `Coord` shape knows statically, so those divisions strength-reduce instead
    of lowering to a runtime `div`.

    Matches `_get_start_indices_of_nth_subvolume[0]` exactly, including
    leftover-into-dim-0: the outermost coordinate is the undivided remainder,
    never reduced modulo `shape[0]`. Callers rely on that for indices past the
    end of the packed region.
    """
    comptime IntType = type_of(shape)._int_type

    res = {}
    var curr_index = IntType(n)

    comptime for i in reversed(range(1, rank)):
        comptime if shape_types[i].is_static_value:
            comptime static_extent = shape_types[i].static_value
            curr_index, res.data[i] = divmod(curr_index, IntType(static_extent))
        else:
            curr_index, res.data[i] = divmod(
                curr_index, IntType(shape.get[i]())
            )

    res.data[0] = curr_index


# ===-----------------------------------------------------------------------===#
# Work-stealing elementwise (SM100+ CLC)
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _ClcKernel[
    rank: Int,
    shape_dtype: DType,
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    shape_types: TypeList[Trait=CoordLike, ...],
    handle_uneven_simd: Bool,
    simd_width: Int,
    block_size: Int,
    elems_per_thread: Int,
    trace_description: StaticString = "",
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Work-stealing CLC kernel as a callable struct.

    Parameterizing on `handle_uneven_simd` lets one struct serve both the
    even and uneven dispatch paths — the dead branch is eliminated under
    `comptime if Self.handle_uneven_simd:` per monomorphization, restoring
    the deduplication the pre-`register_passable` parametric closure form
    provided.
    """

    var func: Self.FuncType
    var shape: IndexList[Self.rank, element_type=Self.shape_dtype]
    var num_packed_elems: Int
    var unpacked_tail_length: Int
    var packed_region_length: Int

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.block_size)
        )
    )
    @__name(
        t"{Self.trace_description}_r{Self.rank}_w{Self.simd_width}_b{Self.block_size}.clc.{Self.handle_uneven_simd}"
    )
    def __call__(self) capturing:
        var result = unsafe_stack_allocation[
            1,
            UInt128,
            address_space=.SHARED,
            alignment=16,
        ]()
        var mbar = unsafe_stack_allocation[
            1,
            Int64,
            address_space=.SHARED,
            alignment=8,
        ]()
        # Shared variables for single-barrier broadcast of cancel results.
        var canceled = unsafe_stack_allocation[
            1,
            UInt32,
            address_space=.SHARED,
        ]()
        var next_tile = unsafe_stack_allocation[
            1,
            UInt32,
            address_space=.SHARED,
        ]()

        var tile_id = block_idx.x
        var phase: UInt32 = 0

        # Initialize mbarrier and kick-start CLC pipeline.
        if thread_idx.x == 0:
            mbarrier_init(mbar, Int32(1))
            if elect_one_sync_with_mask(mask=1):
                clusterlaunchcontrol_try_cancel(result, mbar)
            _ = mbarrier_arrive_expect_tx_relaxed(mbar, Int32(16))

        with PDL():
            # Work-stealing loop.
            while True:
                # Process current tile — each thread handles multiple packed
                # elements at stride block_size for coalesced access.
                var base = (
                    tile_id * Self.block_size * Self.elems_per_thread
                    + thread_idx.x
                )

                comptime for e in range(Self.elems_per_thread):
                    var global_packed_idx = base + e * Self.block_size
                    if global_packed_idx < self.num_packed_elems:
                        var start_indices = _start_indices_of_nth_element[
                            Self.shape_types
                        ](global_packed_idx * Self.simd_width, self.shape)
                        comptime if Self.handle_uneven_simd:
                            # Even when simd_width doesn't evenly divide the shape, we can still guarantee alignment when rank == 1.
                            # TODO(GEX-3960): for N-D tensors, `alignment = gcd(simd_width, shape[-1])`.
                            # This needs the static shape propagated via the `Coord` type.
                            comptime alignment = (
                                Self.simd_width if Self.rank == 1 else 1
                            )
                            if (
                                start_indices[Self.rank - 1] + Self.simd_width
                                > self.shape[Self.rank - 1]
                            ):
                                self.func[1](
                                    Coord(start_indices.canonicalize())
                                )
                                var si = start_indices
                                comptime for _off in range(1, Self.simd_width):
                                    _advance_indices(si, self.shape)
                                    self.func[1](Coord(si.canonicalize()))
                            else:
                                self.func[Self.simd_width, alignment](
                                    Coord(start_indices.canonicalize())
                                )
                        else:
                            self.func[Self.simd_width, Self.simd_width](
                                Coord(start_indices.canonicalize())
                            )

                # Leader: wait for cancel result, extract values, then
                # immediately issue the next cancel before the barrier.
                # This eliminates one barrier per iteration by letting
                # thread 0 read and reuse the result buffer in the same
                # critical section, publishing via separate shared vars.
                if thread_idx.x == 0:
                    _mbarrier_wait_acquire_cta(mbar, phase)
                    phase ^= 1

                    var is_canceled = (
                        clusterlaunchcontrol_query_cancel_is_canceled(result)
                    )
                    var ctaid = (
                        clusterlaunchcontrol_query_cancel_get_first_ctaid["x"](
                            result
                        )
                    )

                    # Issue next cancel only if this one succeeded.
                    if Bool(is_canceled):
                        cluster_sync_release()
                        cluster_sync_acquire()
                        if elect_one_sync_with_mask(mask=1):
                            clusterlaunchcontrol_try_cancel(result, mbar)
                        _ = mbarrier_arrive_expect_tx_relaxed(mbar, Int32(16))

                    # Publish extracted values for all threads.
                    canceled[unsafe_offset=0] = is_canceled
                    next_tile[unsafe_offset=0] = ctaid

                # Single barrier — all threads see broadcast values.
                barrier()

                if canceled[unsafe_offset=0] == 0:
                    break

                tile_id = Int(next_tile[unsafe_offset=0])

            # Tail: only the first block handles remainder elements.
            if block_idx.x == 0 and thread_idx.x < self.unpacked_tail_length:
                self.func[1](
                    Coord(
                        _start_indices_of_nth_element[Self.shape_types](
                            self.packed_region_length + thread_idx.x, self.shape
                        ).canonicalize()
                    )
                )


@always_inline
def _elementwise_impl_gpu_clc[
    rank: Int,
    //,
    shape_types: TypeList[Trait=CoordLike, ...],
    simd_width: Int,
    block_size: Int,
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    elems_per_thread: Int,
    *,
    trace_description: StaticString = "",
](
    func: FuncType, shape: IndexList[rank, element_type=_], ctx: DeviceContext
) raises:
    """Executes `func` over `shape` on SM100+ GPUs using Cluster Launch Control
    work-stealing.

    Each thread block processes one tile of `block_size * elems_per_thread`
    packed elements, then attempts to cancel and steal the work of a
    not-yet-launched block. This gives the reduced overhead of persistent
    kernels with the preemption and load-balancing benefits of one-block-per-
    tile launches.

    Parameters:
        rank: The rank of the buffer.
        shape_types: The `Coord` element types of the shape, carrying whichever
            extents are known at compile time.
        simd_width: The SIMD vector width to use.
        block_size: The number of threads per block.
        FuncType: The body function type.
        elems_per_thread: Number of packed elements each thread processes per
            tile. Higher values increase instruction-level parallelism and
            reduce CLC cancel frequency.
        trace_description: Description of the trace.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    var length = shape.flattened_length()
    var num_packed_elems, unpacked_tail_length = udivmod(length, simd_width)
    var packed_region_length = length - unpacked_tail_length

    if length == 0:
        return

    var num_tiles = uceildiv(num_packed_elems, block_size * elems_per_thread)
    if num_tiles == 0:
        num_tiles = 1

    @always_inline
    def launch[handle_uneven_simd: Bool]() raises {imm}:
        var k = _ClcKernel[
            shape_types=shape_types,
            handle_uneven_simd=handle_uneven_simd,
            simd_width=simd_width,
            block_size=block_size,
            elems_per_thread=elems_per_thread,
            trace_description=trace_description,
        ](
            func,
            shape,
            num_packed_elems,
            unpacked_tail_length,
            packed_region_length,
        )
        ctx.enqueue_function(
            k,
            grid_dim=num_tiles,
            block_dim=block_size,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )

    unswitch(shape[rank - 1] % simd_width != 0, launch)


# ===-----------------------------------------------------------------------===#
# Grid-stride elementwise (pre-SM100)
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _GridStrideKernel[
    rank: Int,
    shape_dtype: DType,
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    shape_types: TypeList[Trait=CoordLike, ...],
    handle_uneven_simd: Bool,
    simd_width: Int,
    block_size: Int,
    elems_per_thread: Int,
    trace_description: StaticString = "",
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Grid-stride elementwise kernel as a callable struct.

    See `_ClcKernel` for the rationale behind the callable-struct form.
    """

    var func: Self.FuncType
    var shape: IndexList[Self.rank, element_type=Self.shape_dtype]
    var num_packed_elems: Int
    var unpacked_tail_length: Int
    var packed_region_length: Int

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.block_size)
        )
    )
    @__name(
        t"{Self.trace_description}_r{Self.rank}_w{Self.simd_width}_b{Self.block_size}.gs.{Self.handle_uneven_simd}"
    )
    def __call__(self) capturing:
        # process the packed region — each thread handles multiple packed
        # elements at stride block_size for coalesced access and ILP.
        var tid = thread_idx.x + Self.block_size * block_idx.x
        var stride = Self.block_size * grid_dim.x

        with PDL():
            for base_idx in range(
                tid,
                self.num_packed_elems,
                stride * Self.elems_per_thread,
            ):
                comptime for e in range(Self.elems_per_thread):
                    var idx = base_idx + e * stride
                    if idx < self.num_packed_elems:
                        var start_indices = _start_indices_of_nth_element[
                            Self.shape_types
                        ](idx * Self.simd_width, self.shape)
                        comptime if Self.handle_uneven_simd:
                            # Even when simd_width doesn't evenly divide the shape, we can still guarantee alignment when rank == 1.
                            # TODO(GEX-3960): for N-D tensors, `alignment = gcd(simd_width, shape[-1])`.
                            # This needs the static shape propagated via the `Coord` type.
                            comptime alignment = (
                                Self.simd_width if Self.rank == 1 else 1
                            )
                            if (
                                start_indices[Self.rank - 1] + Self.simd_width
                                > self.shape[Self.rank - 1]
                            ):
                                self.func[1](
                                    Coord(start_indices.canonicalize())
                                )
                                var si = start_indices
                                comptime for _off in range(1, Self.simd_width):
                                    _advance_indices(si, self.shape)
                                    self.func[1](Coord(si.canonicalize()))
                            else:
                                self.func[Self.simd_width, alignment](
                                    Coord(start_indices.canonicalize())
                                )
                        else:
                            self.func[Self.simd_width, Self.simd_width](
                                Coord(start_indices.canonicalize())
                            )

            # process the tail region
            if tid < self.unpacked_tail_length:
                self.func[1](
                    Coord(
                        _start_indices_of_nth_element[Self.shape_types](
                            Int(self.packed_region_length + tid), self.shape
                        ).canonicalize()
                    )
                )


@always_inline
def _elementwise_impl_gpu_grid_stride[
    rank: Int,
    //,
    shape_types: TypeList[Trait=CoordLike, ...],
    simd_width: Int,
    block_size: Int,
    num_waves: Int,
    sm_count: Int,
    threads_per_multiprocessor: Int,
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    elems_per_thread: Int,
    *,
    trace_description: StaticString = "",
](
    func: FuncType, shape: IndexList[rank, element_type=_], ctx: DeviceContext
) raises:
    """Executes `func` over `shape` using a grid-stride loop.

    Parameters:
        rank: The rank of the buffer.
        shape_types: The `Coord` element types of the shape, carrying whichever
            extents are known at compile time.
        simd_width: The SIMD vector width to use.
        block_size: The number of threads per block.
        num_waves: The number of waves to saturate SMs.
        sm_count: The number of streaming multiprocessors.
        threads_per_multiprocessor: The number of threads per SM.
        FuncType: The body function type.
        elems_per_thread: Number of packed elements each thread processes per
            stride iteration for instruction-level parallelism.
        trace_description: Description of the trace.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    # optimized implementation inspired by https://archive.md/Tye9y#selection-1101.2-1151.3

    var length = shape.flattened_length()
    var num_packed_elems, unpacked_tail_length = udivmod(length, simd_width)
    var packed_region_length = length - unpacked_tail_length

    if length == 0:
        return

    # Grid is sized to saturate SMs — do NOT divide by elems_per_thread
    # here, as that would reduce occupancy. The unrolling only helps when
    # threads have multiple grid-stride iterations to process.
    var num_blocks: Int = clamp(
        uceildiv(num_packed_elems, block_size),
        1,
        sm_count
        * ufloordiv(threads_per_multiprocessor, block_size)
        * num_waves,
    )

    @always_inline
    def launch[handle_uneven_simd: Bool]() raises {imm}:
        var k = _GridStrideKernel[
            shape_types=shape_types,
            handle_uneven_simd=handle_uneven_simd,
            simd_width=simd_width,
            block_size=block_size,
            elems_per_thread=elems_per_thread,
            trace_description=trace_description,
        ](
            func,
            shape,
            num_packed_elems,
            unpacked_tail_length,
            packed_region_length,
        )
        ctx.enqueue_function(
            k,
            grid_dim=num_blocks,
            block_dim=block_size,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )

    unswitch(shape[rank - 1] % simd_width != 0, launch)


# ===-----------------------------------------------------------------------===#
# Dual-elementwise grid-stride
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _DualGridStrideKernel[
    shape_0_types: TypeList[Trait=CoordLike, ...],
    shape_1_types: TypeList[Trait=CoordLike, ...],
    Func0Type: RegisterPassable
    & ImplicitlyCopyable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    Func1Type: RegisterPassable
    & ImplicitlyCopyable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    handle_uneven_simd: Bool,
    simd_width: Int,
    block_size: Int,
    elems_per_thread: Int,
    trace_description: StaticString = "",
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Dual-elementwise grid-stride kernel as a callable struct.

    See `_ClcKernel` for the rationale behind the callable-struct form.
    """

    comptime rank = Self.shape_0_types.length

    var func_0: Self.Func0Type
    var func_1: Self.Func1Type
    var shape_0: Coord[*Self.shape_0_types]
    var shape_1: Coord[*Self.shape_1_types]
    var num_packed_0: Int
    var unpacked_tail_0: Int
    var packed_region_0: Int
    var num_packed_1: Int
    var unpacked_tail_1: Int
    var packed_region_1: Int
    var max_packed: Int

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.block_size)
        )
    )
    @__name(
        t"{Self.trace_description}_r{Self.rank}_w{Self.simd_width}_b{Self.block_size}.dual_gs.{Self.handle_uneven_simd}"
    )
    def __call__(self) capturing:
        comptime assert Self.shape_0_types.length == Self.shape_1_types.length
        var tid = thread_idx.x + Self.block_size * block_idx.x
        var stride = Self.block_size * grid_dim.x

        with PDL():
            for base_idx in range(
                tid,
                self.max_packed,
                stride * Self.elems_per_thread,
            ):
                comptime for e in range(Self.elems_per_thread):
                    var idx = base_idx + e * stride
                    if idx < self.num_packed_0:
                        var start_indices_0 = (
                            _get_start_indices_of_nth_subvolume[0](
                                idx * Self.simd_width,
                                coord_to_index_list(self.shape_0),
                            )
                        )
                        comptime if Self.handle_uneven_simd:
                            if Int(
                                start_indices_0[Self.rank - 1].value()
                            ) + Self.simd_width > Int(
                                self.shape_0[Self.rank - 1].value()
                            ):
                                self.func_0[1](Coord(start_indices_0))
                                var si = start_indices_0
                                comptime for _off in range(1, Self.simd_width):
                                    _advance_indices(
                                        si, coord_to_index_list(self.shape_0)
                                    )
                                    self.func_0[1](Coord(si))
                            else:
                                self.func_0[Self.simd_width](
                                    Coord(start_indices_0)
                                )
                        else:
                            self.func_0[Self.simd_width, Self.simd_width](
                                Coord(start_indices_0)
                            )
                    if idx < self.num_packed_1:
                        var start_indices_1 = (
                            _get_start_indices_of_nth_subvolume[0](
                                idx * Self.simd_width,
                                coord_to_index_list(self.shape_1),
                            )
                        )
                        comptime if Self.handle_uneven_simd:
                            if Int(
                                start_indices_1[Self.rank - 1].value()
                            ) + Self.simd_width > Int(
                                self.shape_1[Self.rank - 1].value()
                            ):
                                self.func_1[1](Coord(start_indices_1))
                                var si = start_indices_1
                                comptime for _off in range(1, Self.simd_width):
                                    _advance_indices(
                                        si, coord_to_index_list(self.shape_1)
                                    )
                                    self.func_1[1](Coord(si))
                            else:
                                self.func_1[Self.simd_width](
                                    Coord(start_indices_1)
                                )
                        else:
                            self.func_1[Self.simd_width, Self.simd_width](
                                Coord(start_indices_1)
                            )

            if tid < self.unpacked_tail_0:
                self.func_0[1, Self.rank](
                    Coord(
                        _get_start_indices_of_nth_subvolume[0](
                            Int(self.packed_region_0 + tid),
                            coord_to_index_list(self.shape_0),
                        )
                    )
                )
            if tid < self.unpacked_tail_1:
                self.func_1[1, Self.rank](
                    Coord(
                        _get_start_indices_of_nth_subvolume[0](
                            Int(self.packed_region_1 + tid),
                            coord_to_index_list(self.shape_1),
                        )
                    )
                )


@always_inline
def _dual_elementwise_impl_gpu_grid_stride[
    simd_width: Int,
    block_size: Int,
    num_waves: Int,
    sm_count: Int,
    threads_per_multiprocessor: Int,
    Func0Type: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    Func1Type: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    elems_per_thread: Int,
    *,
    trace_description: StaticString = "",
](
    func_0: Func0Type,
    func_1: Func1Type,
    *,
    shape_0: Coord,
    shape_1: Coord,
    ctx: DeviceContext,
) raises:
    """Executes two elementwise functions over their respective shapes in a
    single kernel launch using a grid-stride loop.

    Both shapes must have the same element type (the dispatcher casts them
    before calling this function).

    Parameters:
        simd_width: The SIMD vector width to use.
        block_size: The number of threads per block.
        num_waves: The number of waves to saturate SMs.
        sm_count: The number of streaming multiprocessors.
        threads_per_multiprocessor: The number of threads per SM.
        Func0Type: The first body function type.
        Func1Type: The second body function type.
        elems_per_thread: Number of packed elements each thread processes per
            stride iteration for instruction-level parallelism.
        trace_description: Description of the trace.

    Args:
        func_0: The first body function closure.
        func_1: The second body function closure.
        shape_0: The shape for the first function.
        shape_1: The shape for the second function.
        ctx: The device context.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime assert shape_0.rank == shape_1.rank
    comptime rank = shape_0.rank

    var length_0 = Int(shape_0.product())
    var length_1 = Int(shape_1.product())
    var num_packed_0, unpacked_tail_0 = udivmod(length_0, simd_width)
    var num_packed_1, unpacked_tail_1 = udivmod(length_1, simd_width)
    var packed_region_0 = length_0 - unpacked_tail_0
    var packed_region_1 = length_1 - unpacked_tail_1
    var max_packed = (
        num_packed_0 if num_packed_0 > num_packed_1 else num_packed_1
    )

    if length_0 == 0 and length_1 == 0:
        return

    var num_blocks: Int = clamp(
        uceildiv(max_packed, block_size),
        1,
        sm_count
        * ufloordiv(threads_per_multiprocessor, block_size)
        * num_waves,
    )

    var any_unaligned = (
        Int(shape_0[rank - 1].value()) % simd_width != 0
        or Int(shape_1[rank - 1].value()) % simd_width != 0
    )

    @always_inline
    def launch[handle_uneven_simd: Bool]() raises {imm}:
        var k = _DualGridStrideKernel[
            handle_uneven_simd=handle_uneven_simd,
            simd_width=simd_width,
            block_size=block_size,
            elems_per_thread=elems_per_thread,
            trace_description=trace_description,
        ](
            func_0,
            func_1,
            shape_0,
            shape_1,
            num_packed_0,
            unpacked_tail_0,
            packed_region_0,
            num_packed_1,
            unpacked_tail_1,
            packed_region_1,
            max_packed,
        )
        ctx.enqueue_function(
            k,
            grid_dim=num_blocks,
            block_dim=block_size,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )

    unswitch(any_unaligned, launch)


# ===-----------------------------------------------------------------------===#
# Dispatch
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_gpu[
    simd_width: Int,
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    trace_description: StaticString,
](func: FuncType, *, shape: Coord, ctx: DeviceContext) raises:
    """Executes `func[width](indices)` as sub-tasks for a suitable
    combination of width and indices so as to cover shape on the GPU.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.
        trace_description: Description of the trace.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime rank = shape.rank

    comptime hw_info = ctx.default_device_info

    comptime registers_per_thread = 255
    comptime num_waves = get_defined_int["MOJO_ELEMENTWISE_NUM_WAVES", 32]()
    comptime registers_per_block = hw_info.max_registers_per_block
    comptime sm_count: Int = hw_info.sm_count
    comptime threads_per_multiprocessor = hw_info.threads_per_multiprocessor

    comptime assert (
        sm_count > 0 and threads_per_multiprocessor > 0
    ), "the sm_count and thread_count must be known"

    comptime block_size_unrounded = registers_per_block // registers_per_thread

    # Round down to the warp/wavefront size for correct hardware alignment
    # on both NVIDIA (warp=32) and AMD (wavefront=64).
    comptime warp_size = WARP_SIZE
    comptime default_block_size = (
        128 if ctx.default_device_info
        == B200 else block_size_unrounded - (block_size_unrounded % warp_size)
    )
    comptime block_size = get_defined_int[
        "MOJO_ELEMENTWISE_BLOCK_SIZE", default_block_size
    ]()
    comptime short_row_block_size = get_defined_int[
        "MOJO_ELEMENTWISE_SHORT_ROW_BLOCK_SIZE",
        64 if ctx.default_device_info == B200 else default_block_size,
    ]()
    comptime elems_per_thread = get_defined_int[
        "MOJO_ELEMENTWISE_ELEMS_PER_THREAD",
        4 if has_nvidia_gpu_accelerator() else 1,
    ]()
    comptime clc_min_packed_per_row = get_defined_int[
        "MOJO_ELEMENTWISE_CLC_MIN_PACKED_PER_ROW", 9
    ]()
    var packed_elems_per_row = ufloordiv(
        Int(shape[rank - 1].value()), simd_width
    )
    var shape_idx = coord_to_index_list(shape)

    var length = Int(shape.product())
    var use_32bit = length <= Int(UInt32.MAX)

    if length == 0:
        return

    # Gate on the cheap compile-time flag first so the more expensive
    # accelerator-arch predicate is only evaluated when work-stealing is
    # actually enabled. This avoids per-instantiation comptime cost in the
    # common case where `_USE_CLC_WORK_STEALING` is off.
    comptime if _USE_CLC_WORK_STEALING and _has_sm_100x_or_newer():
        var num_packed = ufloordiv(length, simd_width)
        var num_tiles = uceildiv(num_packed, block_size * elems_per_thread)

        if packed_elems_per_row < clc_min_packed_per_row or num_tiles <= 1:
            # Short rows or single-tile workloads: use grid-stride to avoid
            # CLC synchronization overhead (mbarrier, cancel, cluster fences)
            # when there is nothing to steal.
            if use_32bit:
                _elementwise_impl_gpu_grid_stride[
                    shape_types=type_of(shape).ParamListType,
                    simd_width=simd_width,
                    block_size=short_row_block_size,
                    num_waves=num_waves,
                    sm_count=sm_count,
                    threads_per_multiprocessor=threads_per_multiprocessor,
                    elems_per_thread=elems_per_thread,
                    trace_description=trace_description,
                ](func=func, shape=shape_idx.cast[.uint32](), ctx=ctx)
            else:
                _elementwise_impl_gpu_grid_stride[
                    shape_types=type_of(shape).ParamListType,
                    simd_width=simd_width,
                    block_size=short_row_block_size,
                    num_waves=num_waves,
                    sm_count=sm_count,
                    threads_per_multiprocessor=threads_per_multiprocessor,
                    elems_per_thread=elems_per_thread,
                    trace_description=trace_description,
                ](func=func, shape=shape_idx.cast[.uint64](), ctx=ctx)
        else:
            if use_32bit:
                _elementwise_impl_gpu_clc[
                    shape_types=type_of(shape).ParamListType,
                    simd_width=simd_width,
                    block_size=block_size,
                    elems_per_thread=elems_per_thread,
                    trace_description=trace_description,
                ](func=func, shape=shape_idx.cast[.uint32](), ctx=ctx)
            else:
                _elementwise_impl_gpu_clc[
                    shape_types=type_of(shape).ParamListType,
                    simd_width=simd_width,
                    block_size=block_size,
                    elems_per_thread=elems_per_thread,
                    trace_description=trace_description,
                ](func=func, shape=shape_idx.cast[.uint64](), ctx=ctx)
    else:
        if use_32bit:
            _elementwise_impl_gpu_grid_stride[
                shape_types=type_of(shape).ParamListType,
                simd_width=simd_width,
                block_size=block_size,
                num_waves=num_waves,
                sm_count=sm_count,
                threads_per_multiprocessor=threads_per_multiprocessor,
                elems_per_thread=elems_per_thread,
                trace_description=trace_description,
            ](func=func, shape=shape_idx.cast[.uint32](), ctx=ctx)
        else:
            _elementwise_impl_gpu_grid_stride[
                shape_types=type_of(shape).ParamListType,
                simd_width=simd_width,
                block_size=block_size,
                num_waves=num_waves,
                sm_count=sm_count,
                threads_per_multiprocessor=threads_per_multiprocessor,
                elems_per_thread=elems_per_thread,
                trace_description=trace_description,
            ](func=func, shape=shape_idx.cast[.uint64](), ctx=ctx)


@always_inline
def _dual_elementwise_impl_gpu[
    simd_width: Int,
    Func0Type: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    Func1Type: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    *,
    trace_description: StaticString,
](
    func_0: Func0Type,
    func_1: Func1Type,
    *,
    shape_0: Coord,
    shape_1: Coord,
    ctx: DeviceContext,
) raises:
    """Executes two elementwise functions over their respective shapes in a
    single GPU kernel launch.

    Parameters:
        simd_width: The SIMD vector width to use.
        Func0Type: The first body function type.
        Func1Type: The second body function type.
        trace_description: Description of the trace.

    Args:
        func_0: The first body function closure.
        func_1: The second body function closure.
        shape_0: The shape for the first function.
        shape_1: The shape for the second function.
        ctx: The device context.

    Raises:
        If the GPU kernel launch fails.
    """

    comptime hw_info = ctx.default_device_info

    comptime registers_per_thread = 255
    comptime num_waves = get_defined_int["MOJO_ELEMENTWISE_NUM_WAVES", 32]()
    comptime registers_per_block = hw_info.max_registers_per_block
    comptime sm_count: Int = hw_info.sm_count
    comptime threads_per_multiprocessor = hw_info.threads_per_multiprocessor

    comptime assert (
        sm_count > 0 and threads_per_multiprocessor > 0
    ), "the sm_count and thread_count must be known"

    comptime block_size_unrounded = registers_per_block // registers_per_thread
    comptime warp_size = WARP_SIZE
    comptime default_block_size = (
        128 if ctx.default_device_info
        == B200 else block_size_unrounded - (block_size_unrounded % warp_size)
    )
    comptime block_size = get_defined_int[
        "MOJO_ELEMENTWISE_BLOCK_SIZE", default_block_size
    ]()
    comptime elems_per_thread = get_defined_int[
        "MOJO_ELEMENTWISE_ELEMS_PER_THREAD",
        4 if has_nvidia_gpu_accelerator() else 1,
    ]()

    var max_length = Int(shape_0.product())
    var len_1 = Int(shape_1.product())
    if len_1 > max_length:
        max_length = len_1
    var use_32bit = max_length <= Int(UInt32.MAX)

    if max_length == 0:
        return

    if use_32bit:
        _dual_elementwise_impl_gpu_grid_stride[
            simd_width=simd_width,
            block_size=block_size,
            num_waves=num_waves,
            sm_count=sm_count,
            threads_per_multiprocessor=threads_per_multiprocessor,
            elems_per_thread=elems_per_thread,
            trace_description=trace_description,
        ](
            func_0=func_0,
            func_1=func_1,
            shape_0=shape_0.cast[.uint32](),
            shape_1=shape_1.cast[.uint32](),
            ctx=ctx,
        )
    else:
        _dual_elementwise_impl_gpu_grid_stride[
            simd_width=simd_width,
            block_size=block_size,
            num_waves=num_waves,
            sm_count=sm_count,
            threads_per_multiprocessor=threads_per_multiprocessor,
            elems_per_thread=elems_per_thread,
            trace_description=trace_description,
        ](
            func_0=func_0,
            func_1=func_1,
            shape_0=shape_0.cast[.uint64](),
            shape_1=shape_1.cast[.uint64](),
            ctx=ctx,
        )
