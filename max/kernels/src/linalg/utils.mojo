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

"""Provides shared CPU matmul utilities including tile-consumer traits, kernel shape selection, and partial SIMD load/store helpers."""

from std.math import align_down, align_up, ceildiv, iota
from std.sys._build import is_debug_build
from std.sys.info import CompilationTarget, simd_width_of, size_of
from std.sys.intrinsics import masked_load, masked_store
from std.utils.index import Index, IndexList
from std.algorithm import vectorize
from layout.layout import *
from layout import LayoutTensor, TileTensor, Coord, TensorLayout
from layout.tile_layout import Layout as _NewLayout
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.reflection import reflect


@always_inline
def partial_simd_load[
    dtype: DType, //, width: Int
](
    storage: UnsafePointer[mut=False, Scalar[dtype], ...],
    lbound: Int,
    rbound: Int,
    pad_value: Scalar[dtype],
) -> SIMD[dtype, width]:
    """Loads a vector with dynamic bound.

    Out of bound data will be filled with pad value. Data is valid if
    lbound <= idx < rbound for idx from 0 to (simd_width-1). For example:

        addr 0  1  2  3
        data x 42 43  x

        partial_simd_load[4](addr0, 1, 3) #gives [0 42 43 0]

    Parameters:
        dtype: The DType of storage.
        width: The system simd vector size.

    Args:
        storage: Pointer to the address to perform load.
        lbound: Lower bound of valid index within simd (inclusive).
        rbound: Upper bound of valid index within simd (non-inclusive).
        pad_value: Value to fill for out of bound indices.

    Returns:
        The SIMD vector loaded and zero-filled.
    """
    # Create a mask based on input bounds.
    var effective_lbound = SIMD[.int32, width](max(lbound, 0))
    var effective_rbound = SIMD[.int32, width](min(width, rbound))
    var incr = iota[.int32, width]()
    var mask = incr.ge(effective_lbound) & incr.lt(effective_rbound)

    return masked_load[width](storage, mask, pad_value)


@always_inline
def partial_simd_store[
    dtype: DType, //, width: SIMDLength
](
    storage: UnsafePointer[mut=True, Scalar[dtype], ...],
    lbound: Int,
    rbound: Int,
    data: SIMD[dtype, width],
):
    """Stores a vector with dynamic bound.

    Out of bound data will ignored. Data is valid if lbound <= idx < rbound for
    idx from 0 to (simd_width-1).

    e.g.
        addr 0 1 2  3
        data 0 0 0  0

        partial_simd_load[4](addr0, 1, 3, [-1, 42, 43, -1]) #gives [0 42 43 0]

    Parameters:
        dtype: The DType of storage.
        width: The system simd vector size.

    Args:
        storage: Pointer to the address to perform load.
        lbound: Lower bound of valid index within simd (inclusive).
        rbound: Upper bound of valid index within simd (non-inclusive).
        data: The vector value to store.
    """
    # Create a mask based on input bounds.
    var effective_lbound = SIMD[.int32, width](max(lbound, 0))
    var effective_rbound = SIMD[.int32, width](min(width, rbound))
    var incr = iota[.int32, width]()
    var mask = incr.ge(effective_lbound) & incr.lt(effective_rbound)

    return masked_store(data, storage, mask)


comptime elementwise_epilogue_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](IndexList[2], SIMD[dtype, width]) capturing -> None

comptime elementwise_compute_lambda_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](IndexList[2], SIMD[dtype, width]) capturing -> SIMD[dtype, width]


trait TileConsumer(DevicePassable, TrivialRegisterPassable):
    """Trait for an epilogue operation which consumes a tile of data.

    The kernel feeds the consumer a tile (in `src_address_space`); the
    consumer is terminal (does whatever it does and returns nothing).
    For a non-terminal counterpart, see `TileOperation`; for aux-load
    capabilities, see `AuxLoading` / `AuxLoadingPipelined` (which refine
    `TileOperation`, not this trait).
    """

    comptime src_address_space: AddressSpace
    """AddressSpace of the tile being consumed"""

    # Default `DevicePassable` boilerplate so conformers need not restate it:
    # the device type is the consumer itself, encoding bit-copies it, and the
    # name comes from reflection.
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(reflect[Self].name())

    def __call__[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        ref self,
        tile_coord: Coord,
        tile: TileTensor[
            dtype, LayoutType, ..., address_space=Self.src_address_space
        ],
        thread_layout: _NewLayout,
    ) -> None:
        """Consume `tile`; terminal (returns nothing).

        Parameters:
            dtype: Element type of the tile.
            LayoutType: Layout of the tile.
        Args:
            tile_coord: Absolute element coord of this sub-tile's top-left in the
                global output tensor.
            tile: The tile of data being consumed.
            thread_layout: Logical layout of the threads collaborating on this
                call
        """
        ...


trait TileOperation(DevicePassable, TrivialRegisterPassable):
    """Non-terminal counterpart to `TileConsumer`: mutates a tile in
    place between MMA and store. Composes additively with capability
    subtraits like `AuxLoading` for ops that need pre-loaded data.

    Surface matches `TileConsumer.__call__` modulo `mut=True` on the
    tile (the op writes back) and on `self`. The two traits are role-
    distinct (terminal vs non-terminal); a struct that wants to be
    both would need separate disambiguation, which we'll deal with
    if/when that comes up.
    """

    comptime src_address_space: AddressSpace
    """AddressSpace of the tile being transformed (typically LOCAL)."""

    # Default `DevicePassable` boilerplate so conformers need not restate it:
    # the device type is the op itself, encoding bit-copies it, and the name
    # comes from reflection.
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return String(reflect[Self].name())

    def __call__[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        mut self,
        tile_coord: Coord,
        tile: TileTensor[
            dtype,
            LayoutType,
            MutAnyOrigin,
            ...,
            address_space=Self.src_address_space,
        ],
        thread_layout: _NewLayout,
    ) -> type_of(tile):
        """Transform `tile` and return it.

        Returns a `TileTensor` of the same shape as its tile argument.

        Parameters:
            dtype: Element type of the tile.
            LayoutType: Layout of the tile.
        Args:
            tile_coord: Absolute element coord of this sub-tile's top-left in
                the global output tensor — consistent with `TileConsumer.__call__`
                and the legacy `elementwise_compute_lambda_type`.
            thread_layout: Logical layout of the threads collaborating on this
                call. Passed as a value (its static dims are comptime-readable
                via `type_of(thread_layout)`), so the kernel states "this call
                runs over these threads" the way the `tile` argument states
                "this tile is M x N". Impls that load aux inputs use it (e.g. to
                build a load copier) and should `comptime assert` it is static.
            tile: Per-warp tile (kernel side passes a `TileTensor` view over the
                per-thread register storage; `tile.static_shape` is the warp
                tile shape).
        """
        ...


struct NullTileConsumer(TileConsumer):
    """No-op TileConsumer. Used as the default when no fusion is requested,
    and as the placeholder type when a kernel's `tile_consumer` Optional is
    None.
    """

    comptime src_address_space = AddressSpace.LOCAL

    def __init__(out self):
        comptime assert (
            False
        ), "NullTileConsumer is a null sentinel. Do not use!"

    def __call__[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        ref self,
        tile_coord: Coord,
        tile: TileTensor[
            dtype, LayoutType, ..., address_space=Self.src_address_space
        ],
        thread_layout: _NewLayout,
    ) -> None:
        pass


struct NullTileOperation(TileOperation):
    """No-op `TileOperation` sentinel: parallel to `NullTileConsumer`.
    Used as the default `TileOperationType` so kernels without a fused
    op compile without callers having to spell out a placeholder.
    """

    comptime src_address_space = AddressSpace.LOCAL

    def __init__(out self):
        comptime assert (
            False
        ), "NullTileOperation is a null sentinel. Do not use!"

    def __call__[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        mut self,
        tile_coord: Coord,
        tile: TileTensor[
            dtype,
            LayoutType,
            MutAnyOrigin,
            ...,
            address_space=Self.src_address_space,
        ],
        thread_layout: _NewLayout,
    ) -> type_of(tile):
        return tile


@always_inline
def is_valid_epilogue[T: AnyType]() -> Bool:
    """Whether `T` is a real epilogue rather than a null sentinel.

    A kernel that accepts a `TileOperation`/`TileConsumer` type parameter
    (defaulting to `NullTileOperation`/`NullTileConsumer`) uses this in a
    `comptime if` to elide the epilogue call when no epilogue was bound.

    Parameters:
        T: The epilogue type to test (a `TileOperation` or `TileConsumer`).

    Returns:
        False iff `T` is one of the epilogue null sentinels.
    """
    return not (T == NullTileOperation or T == NullTileConsumer)


@always_inline
def lora_qkv_plane_row_offset[
    splits: IndexList[2]
](out_col: Int, plane_stride: Int) -> Int:
    """Additive row offset into a 3-plane row-stacked activation `[3*M, K]`.

    Lets a grouped matmul read a *different* plane of the activation per
    output-column region within a single launch, without a capturing closure
    (which the warp-specialized kernel's GPU slicer cannot handle in the load
    path). Used by the LoRA-B QKV expand to select the matching plane of the
    planar shrink output `P [3, M, R]` for the Q / K / V output regions.

    For output column `out_col`, the plane is 0 when `out_col < splits[0]`, 1
    when `out_col < splits[1]`, else 2; the returned offset is
    `plane * plane_stride`. `splits == (0, 0)` disables it (returns 0), so the
    default is a no-op for every other caller.
    """
    comptime if splits[0] <= 0:
        return 0
    else:
        var plane = 2
        if out_col < splits[0]:
            plane = 0
        elif out_col < splits[1]:
            plane = 1
        return plane * plane_stride


@fieldwise_init
struct KernelConfig:
    """Static configuration of the matmul inner kernel."""

    # Static number of rows of the micro kernel.
    var kernel_rows: Int

    # Static number of columns of the micro kernel.
    var kernel_cols: Int

    # Static info on simd vector size.
    var simd_size: Int


@fieldwise_init
struct MicroKernelShape(TrivialRegisterPassable):
    """Record describing the inner kernel shape."""

    var simd_rows: Int
    var simd_cols: Int


@fieldwise_init
struct GemmShape(TrivialRegisterPassable):
    """Helper class to unpack gemm dimension and layout."""

    var M: Int
    var N: Int
    var K: Int

    @staticmethod
    def get[
        transpose_b: Bool,
        layout_c: Layout,
        layout_a: Layout,
        layout_b: Layout,
    ](
        c: LayoutTensor[mut=False, _, layout_c, ...],
        a: LayoutTensor[mut=False, _, layout_a, ...],
        b: LayoutTensor[mut=False, _, layout_b, ...],
    ) -> GemmShape:
        """Constructor of a gemm shape record from input buffers.

        M, N, and K are intentionally calculated using `a` and `c` ONLY. This
        is because `b` may be padded to a multiple of the tile size if it has
        been pre-packed.

        Parameters:
            transpose_b: Whether matrix B is stored in transposed form.
            layout_c: The memory layout of the output C tensor.
            layout_a: The memory layout of the input A tensor.
            layout_b: The memory layout of the input B tensor.

        Args:
            c: LayoutTensor with allocated output space.
            a: LayoutTensor containing matrix operand A.
            b: LayoutTensor containing matrix operand B.
        """

        # We only want a 2D tensor for now
        comptime assert c.rank == 2
        comptime assert a.rank == 2
        comptime assert b.rank == 2

        return GemmShape(c.dim[0](), c.dim[1](), a.dim[1]())

    @staticmethod
    def get[
        transpose_b: Bool,
    ](
        c: TileTensor[mut=False, ...],
        a: TileTensor[mut=False, ...],
        b: TileTensor[mut=False, ...],
    ) -> GemmShape:
        """Constructor of a gemm shape record from TileTensor inputs.

        M, N, and K are intentionally calculated using `a` and `c` ONLY. This
        is because `b` may be padded to a multiple of the tile size if it has
        been pre-packed.

        Parameters:
            transpose_b: Whether matrix B is stored in transposed form.

        Args:
            c: TileTensor with allocated output space.
            a: TileTensor containing matrix operand A.
            b: TileTensor containing matrix operand B.
        """

        comptime assert c.rank == 2, "c must be of rank 2"
        comptime assert a.rank == 2, "a must be of rank 2"
        comptime assert b.rank == 2, "b must be of rank 2"

        return GemmShape(Int(c.dim[0]()), Int(c.dim[1]()), Int(a.dim[1]()))

    # TODO: re-enable using IndexList.
    @always_inline
    def __getitem__(self, idx: Int) -> Int:
        if idx == 0:
            return self.M
        if idx == 1:
            return self.N
        return self.K

    def __setitem__(mut self, idx: Int, value: Int):
        if idx == 0:
            self.M = value
            return
        if idx == 1:
            self.N = value
            return
        if idx == 2:
            self.K = value
            return

    def __init__(out self, index: IndexList[3]):
        """Constructor of a gemm shape record from a index tuple.

        Args:
            index: The int tuple containing the index(m,n,k).
        """
        self.M = index[0]
        self.N = index[1]
        self.K = index[2]

    def as_index(self) -> IndexList[3]:
        """Utility to convert the underlying data to an index tuple. So that the
        utilities such as elementwise add can be used.

        Returns:
            The constructed index tuple.
        """
        return Index(self.M, self.N, self.K)

    def __add__(self, rhs: GemmShape) -> GemmShape:
        """Coordinate-wise addition of two gemm shape records.

        Args:
            rhs: Another gemm shape record to add with.
        """
        return GemmShape(self.as_index() + rhs.as_index())

    def __sub__(self, rhs: GemmShape) -> GemmShape:
        """Coordinate-wise subtraction of two gemm shape records.

        Args:
            rhs: Another gemm shape record to subtract with.
        """
        return GemmShape(self.as_index() - rhs.as_index())


# Helper heuristic function to decide on tile size
#  Returns (TileN, TileK)
@always_inline
def calculate_tile_n_k[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    kernel_cols: Int,
](n: Int, k: Int) -> IndexList[2]:
    """Helper heuristic function to decide on tile size to partition the matmul
    given the cache size and desired data layout.

    Parameters:
        a_type: The dtype of the A tensor.
        b_type: The dtype of the B tensor.
        c_type: The dtype of the C tensor.
        kernel_cols: The umber of columns of the micro kernel.

    Args:
        n: The N dimension of the matmul.
        k: The K dimension of the matmul.

    Returns:
        The calculated tile size to partition the matmul as (TileN, TileK).
    """

    comptime pack_cache_size = get_pack_data_size[b_type]()
    comptime use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    comptime use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
    comptime factor = get_matmul_arch_factor[use_vnni, use_i8mm]()

    var least_tile_n: Int = kernel_cols

    # Max tile K size based on smallest Tile N.
    var largest_tile_k = align_down(pack_cache_size // least_tile_n, factor)

    # Prioritize shape on K dimension, so try to fit in the whole
    #  input on the tile.

    var tile_k = min(largest_tile_k, align_up(k, factor))

    # Calculate number of InnerSize to fit in tile_n dimension,
    var max_tile_n_in_inner_size = pack_cache_size // tile_k // kernel_cols
    var full_data_tile_n_in_inner_size = ceildiv(n, kernel_cols)
    var tile_n_in_inner_size = min(
        max_tile_n_in_inner_size, full_data_tile_n_in_inner_size
    )

    # Calculate tile_n size.
    var tile_n = tile_n_in_inner_size * kernel_cols

    return Index(tile_n, tile_k)


def calculate_tile_n_k[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    kernel_cols: Int,
](global_tile_shape: GemmShape) -> IndexList[2]:
    return calculate_tile_n_k[a_type, b_type, c_type, kernel_cols](
        global_tile_shape.N, global_tile_shape.K
    )


@always_inline
def _get_tile_n_k[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    kernel_cols: Int,
    transpose_b: Bool,
](b: TileTensor[mut=False, ...]) -> IndexList[2]:
    comptime assert b.rank == 2
    var tile_n_k: IndexList[2]

    comptime if not transpose_b:
        tile_n_k = calculate_tile_n_k[a_type, b_type, c_type, kernel_cols](
            Int(b.dim[1]()), Int(b.dim[0]())
        )
    else:
        tile_n_k = calculate_tile_n_k[a_type, b_type, c_type, kernel_cols](
            Int(b.dim[0]()), Int(b.dim[1]())
        )

    return tile_n_k


# The number of registers used for the inner kernel is:
#   kernel_rows*kernel_cols + 1*kernel_cols + 1
def get_matmul_kernel_shape_x86[kernel_type: Bool]() -> MicroKernelShape:
    """Returns the micro kernel shape tuned for x86 targets.

    Parameters:
        kernel_type: Selects between the two tuned shapes for the target.
    """
    comptime if CompilationTarget.has_avx512f():
        comptime if kernel_type:
            return MicroKernelShape(8, 3)
        else:
            return MicroKernelShape(6, 4)
    else:
        return MicroKernelShape(4, 3)


def get_matmul_kernel_shape_ARM[
    a_type: DType, b_type: DType, c_type: DType, kernel_type: Bool
]() -> MicroKernelShape:
    """Returns the micro kernel shape tuned for ARM targets.

    Parameters:
        a_type: The dtype of the A tensor.
        b_type: The dtype of the B tensor.
        c_type: The dtype of the C tensor.
        kernel_type: Selects between the two tuned shapes for the target.
    """
    comptime if CompilationTarget.is_neoverse_n1():
        comptime if kernel_type:
            return MicroKernelShape(4, 4)
        else:
            return MicroKernelShape(8, 2)
    else:
        comptime use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

        comptime if use_i8mm:
            return MicroKernelShape(4, 6)
        elif kernel_type:
            return MicroKernelShape(6, 4)
        else:
            return MicroKernelShape(8, 2)


# AVX512 and Neon have 32 registers and AVX has 16.
# The largest kernel for AVX is 4x3 which needs 16 registers and gives the best result.
# For AVX512 a 5x4, 5x5, or 6x4 kernel can be used, 6x4 gives the best result.
# For the Graviton 2 a 8x2 kernel gives the best result in most cases.
# For the Graviton 3 a 6x4 or 4x6 kernel gives the best result.
def get_matmul_kernel_shape[
    a_type: DType, b_type: DType, c_type: DType, kernel_type: Bool
]() -> MicroKernelShape:
    """Returns the micro kernel shape for the current target and dtypes.

    Dispatches to the ARM or x86 variant based on the compilation target.

    Parameters:
        a_type: The dtype of the A tensor.
        b_type: The dtype of the B tensor.
        c_type: The dtype of the C tensor.
        kernel_type: Selects between the two tuned shapes for the target.
    """
    comptime use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

    comptime if CompilationTarget.has_neon():
        return get_matmul_kernel_shape_ARM[
            a_type, b_type, c_type, kernel_type
        ]()
    else:
        return get_matmul_kernel_shape_x86[kernel_type]()


def get_matmul_arch_factor[use_vnni: Bool, use_i8mm: Bool]() -> Int:
    """Returns the architecture-dependent alignment factor for matmul tiling.

    Parameters:
        use_vnni: Whether VNNI is available on the target.
        use_i8mm: Whether the i8mm instruction is available on the target.
    """
    if use_i8mm:
        return 8
    elif use_vnni:
        return 4
    else:
        return 1


# prefetching at least on the Graviton 2 performs worse than without.
def get_matmul_prefetch_b_distance_k() -> Int:
    """Returns the K-dimension prefetch distance for the B matrix.

    Returns zero on NEON targets where prefetching hurts performance.
    """
    comptime if CompilationTarget.has_neon():
        return 0
    return 4


# Min task size. This is copied from MLAS.
# TODO: Replace this magic number with a heuristic based on arch.
def get_min_task_size() -> Int:
    """Returns the minimum task size used to limit parallel matmul task counts.
    """
    return 65536


# Unroll factor in packing B
def get_packB_unroll_factor() -> Int:
    """Returns the unroll factor applied while packing the B matrix."""
    return 8


# ===-----------------------------------------------------------------------===#
# Partition Heuristics
# ===-----------------------------------------------------------------------===#


@always_inline
def get_matmul_num_tasks[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    simd_size: Int,
    kernel_type: Bool,
](m: Int, n: Int, k: Int, max_num_tasks: Int) -> Int:
    """Compute the number of tasks for parallel matmul.
    The max number of tasks is the thread or core count.

    Parameters:
        a_type: The dtype of the A tensor.
        b_type: The dtype of the B tensor.
        c_type: The dtype of the C tensor.
        simd_size: The SIMD vector width for the target and dtype.
        kernel_type: Selects between the two tuned shapes for the target.

    Args:
        m: The M dimension of the matmul.
        n: The N dimension of the matmul.
        k: The K dimension of the matmul.
        max_num_tasks: Upper bound on task count (thread or core count).
    """

    # The min tasks complexity is from MLAS.
    # TODO: We can fine-tune this based on mojo.matmul's scaling.
    var num_tasks = ceildiv(m * n * k, get_min_task_size())
    num_tasks = min(num_tasks, max_num_tasks)

    # Limit num_tasks by row-wise and column-wise partition because we don't
    # support partition in k dim yet. E.x. 32x32x1024 uses 16 threads by min
    # task complexity but we only want it to use <= 4 threads for now since
    # M and N are very small.
    comptime kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, kernel_type
    ]()
    var max_row_tasks = ceildiv(m, 2 * kernel_shape.simd_rows)
    var max_col_tasks = ceildiv(n, kernel_shape.simd_cols * simd_size)
    num_tasks = min(num_tasks, max_row_tasks * max_col_tasks)

    return num_tasks


@fieldwise_init
struct SubMatmulConfig(ImplicitlyCopyable):
    """Static configuration of sub-matrices in parallel matmul."""

    # Starting Indices of sub-matrices.
    var offset: IndexList[3]

    # Dimension of sub-matrices.
    var shape: IndexList[3]

    @always_inline
    def is_valid(self) -> Bool:
        return self.shape > Index(0, 0, 0)


# The work is first grouped into blocks for alignment and load/store efficiency.
# This will partition the work blocks between tasks as even as possible.
@always_inline
def partition_work(
    task_id: Int, num_tasks: Int, work: Int, work_block_size: Int
) -> IndexList[2]:
    """Partitions `work` into blocks distributed across `num_tasks` tasks.

    The work is first grouped into `work_block_size`-sized blocks for alignment
    and load/store efficiency, then the blocks are split between tasks as
    evenly as possible.

    Args:
        task_id: The index of the task to compute the range for.
        num_tasks: The total number of tasks sharing the work.
        work: The total amount of work to partition.
        work_block_size: The block size used to align the partitioning.

    Returns:
        A pair `(offset, length)` describing this task's work range.
    """
    var num_work_blocks = ceildiv(work, work_block_size)
    var blocks_per_task, blocks_per_task_extra = divmod(
        num_work_blocks, num_tasks
    )

    var work_per_task = blocks_per_task * work_block_size
    var work_id = (
        work_per_task * task_id + blocks_per_task_extra * work_block_size
    )

    if task_id < blocks_per_task_extra:
        work_per_task = (blocks_per_task + 1) * work_block_size
        work_id = task_id * work_per_task
        return IndexList[2](work_id, min(work - work_id, work_per_task))

    return IndexList[2](work_id, min(work - work_id, work_per_task))


def get_partitioned_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    kernel_rows: Int,
    kernel_cols: Int,
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    """Returns the sub-matmul config for a given task in a parallel matmul.

    When the i8mm instruction is available the partition is forced to have an
    even number of rows, except possibly for the last range.

    Parameters:
        a_type: The dtype of the A tensor.
        b_type: The dtype of the B tensor.
        c_type: The dtype of the C tensor.
        kernel_rows: The static number of rows of the micro kernel.
        kernel_cols: The static number of columns of the micro kernel.

    Args:
        m: The M dimension of the matmul.
        n: The N dimension of the matmul.
        k: The K dimension of the matmul.
        task_id: The index of the task to compute the partition for.
        num_tasks: The total number of tasks sharing the matmul.

    Returns:
        The sub-matmul offset and shape for this task.
    """
    comptime use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

    comptime if use_i8mm:
        # i8mm needs to have even partitions in m.
        # Only the last range is allowed to be odd.
        var partition = get_partitioned_matmul_mojo[
            b_type, kernel_rows, kernel_cols, use_i8mm
        ](m // 2, n, k, task_id, num_tasks)
        var t0 = 2 * partition.offset[0]
        var t1 = 2 * partition.shape[0]
        if t0 + t1 == m - 1:
            t1 = m - t0
        partition.offset[0] = t0
        partition.shape[0] = t1
        return partition
    else:
        return get_partitioned_matmul_mojo[b_type, kernel_rows, kernel_cols](
            m, n, k, task_id, num_tasks
        )


def get_partitioned_matmul_mojo[
    b_type: DType,
    kernel_rows: Int,
    kernel_cols: Int,
    use_i8mm: Bool = False,
](m: Int, n: Int, k: Int, task_id: Int, num_tasks: Int) -> SubMatmulConfig:
    """Returns the sub-matmul config for a task using the Mojo partitioner.

    Splits the work into row and column tasks via `partition_work` after
    determining the task grid with `get_partitioned_matmul_mojo_shape`.

    Parameters:
        b_type: The dtype of the B tensor.
        kernel_rows: The static number of rows of the micro kernel.
        kernel_cols: The static number of columns of the micro kernel.
        use_i8mm: Whether the i8mm instruction is available on the target.

    Args:
        m: The M dimension of the matmul.
        n: The N dimension of the matmul.
        k: The K dimension of the matmul.
        task_id: The index of the task to compute the partition for.
        num_tasks: The total number of tasks sharing the matmul.

    Returns:
        The sub-matmul offset and shape for this task.
    """
    var shape = get_partitioned_matmul_mojo_shape[
        b_type, kernel_rows, kernel_cols, use_i8mm
    ](m, n, k, num_tasks)
    var num_row_tasks = shape[0]
    var num_col_tasks = shape[1]
    var row_task_id, col_task_id = divmod(task_id, num_col_tasks)

    var row_range = partition_work(row_task_id, num_row_tasks, m, kernel_rows)
    var col_range = partition_work(col_task_id, num_col_tasks, n, kernel_cols)
    return SubMatmulConfig(
        Index(row_range[0], col_range[0], 0),
        Index(row_range[1], col_range[1], k),
    )


def get_partitioned_matmul_mojo_shape[
    b_type: DType,
    kernel_rows: Int,
    kernel_cols: Int,
    use_i8mm: Bool,
](m: Int, n: Int, k: Int, num_tasks: Int) -> IndexList[2]:
    """Returns the row and column task counts that best balance a parallel matmul.

    Searches over feasible task grids and selects the one that minimizes the
    per-task work, with heuristics for small `m` and L2-cache-aware column
    partitioning.

    Parameters:
        b_type: The dtype of the B tensor.
        kernel_rows: The static number of rows of the micro kernel.
        kernel_cols: The static number of columns of the micro kernel.
        use_i8mm: Whether the i8mm instruction is available on the target.

    Args:
        m: The M dimension of the matmul.
        n: The N dimension of the matmul.
        k: The K dimension of the matmul.
        num_tasks: The total number of tasks sharing the matmul.

    Returns:
        A pair `(num_row_tasks, num_col_tasks)` describing the task grid.
    """
    var num_row_tasks = 1
    var num_col_tasks = 1

    var min_work = m * n

    var num_packs_m = ceildiv(m, kernel_rows)
    var num_packs_n = ceildiv(n, kernel_cols)
    var max_num_packs_m = num_packs_m
    var max_num_packs_n = num_packs_n
    if (use_i8mm and 2 * m > n) or m > n:
        var half_l2size = get_pack_data_size[b_type]()
        # Limit the partitions in N if the size is smaller than half the L2 cache size.
        var num_packs_n2 = max(k * n // half_l2size, 1)
        if num_packs_m * num_packs_n2 >= num_tasks:
            max_num_packs_n = min(num_packs_n, num_packs_n2)
    else:
        # get the minimum work in n
        var worki = kernel_cols * max((num_packs_n // num_tasks), 1)
        # ensure the work in m is not much smaller than in n
        var num_packs_m2 = ceildiv(m, align_down(worki, kernel_rows))
        if num_packs_n * num_packs_m2 >= num_tasks:
            max_num_packs_m = min(max_num_packs_m, num_packs_m2)

    max_num_packs_m = min(max_num_packs_m, num_tasks)
    max_num_packs_n = min(max_num_packs_n, num_tasks)
    # Loop over all possible partitions and find the partition that balances the work best.
    for j in range(max_num_packs_m, 0, -1):
        var workj = kernel_rows * ceildiv(num_packs_m, j) if j != 1 else m
        for i in range(min(num_tasks // j, max_num_packs_n), 0, -1):
            var worki = kernel_cols * ceildiv(num_packs_n, i) if i != 1 else n
            var work = workj * worki
            if work <= min_work:
                min_work = work
                num_row_tasks = j
                num_col_tasks = i

    # heuristic for small m
    if m <= 32 and num_packs_n >= num_tasks:
        num_row_tasks = 1
        num_col_tasks = num_tasks

    return Index(num_row_tasks, num_col_tasks)


def get_pack_data_size[dtype: DType]() -> Int:
    """Utility to compute the number of elements to pack in each tile.

    Parameters:
        dtype: Element type whose byte size scales the available cache or
            stack budget into an element count.

    Returns:
        The number of elements to pack.
    """
    comptime KB = 1024

    comptime if is_debug_build():
        # Only use the large cache size for release build as debug build may
        # contain additional data could cause stack overflow.
        # Restrict it to 4K.
        return 4 * KB // size_of[dtype]()

    comptime if CompilationTarget.is_macos():
        # Macos has lower stack limit so lower this allocation too.
        # Restrict it to 64K.
        return 64 * KB // size_of[dtype]()

    comptime if CompilationTarget.has_neon() or CompilationTarget.has_avx512f():
        # TODO: This should be 1/2 of L2 cache size on Intel. Graviton 2 and
        # Skylake server have a 1 MiB L1 cache AMD Rome has a 512 KiB L2 cache
        # return half the cache size as 4 byte elements
        return 512 * KB // size_of[dtype]()

    return 256 * KB // size_of[dtype]()


@always_inline
def get_kernel_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    *,
    kernel_type: Bool = False,
]() -> KernelConfig:
    """Extracts matmul configuration parameters for exported functions.

    Parameters:
        a_type: The dtype of the A tensor.
        b_type: The dtype of the B tensor.
        c_type: The dtype of the C tensor.
        kernel_type: Selects between the two tuned shapes for the target
            (defaults to False).
    """
    comptime simd_size = simd_width_of[c_type]()

    comptime kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, kernel_type
    ]()

    return {
        kernel_rows = kernel_shape.simd_rows,
        kernel_cols = kernel_shape.simd_cols * simd_size,
        simd_size = simd_size,
    }


@always_inline
def use_vnni_fn[a_type: DType, b_type: DType, c_type: DType]() -> Bool:
    """Returns whether the VNNI instruction should be used for the given dtypes.

    Parameters:
        a_type: The dtype of the A tensor.
        b_type: The dtype of the B tensor.
        c_type: The dtype of the C tensor.
    """
    comptime if (
        CompilationTarget.has_neon_int8_dotprod()
        and not CompilationTarget.has_neon_int8_matmul()
    ):
        return (
            (a_type == .int8 and b_type == .int8)
            or (a_type == .uint8 and b_type == .uint8)
        ) and c_type == .int32
    elif CompilationTarget.has_avx2():
        return a_type == .uint8 and b_type == .int8 and c_type == .int32
    else:
        return False


@always_inline
def use_i8mm_fn[a_type: DType, b_type: DType, c_type: DType]() -> Bool:
    """Returns whether the i8mm instruction should be used for the given dtypes.

    Parameters:
        a_type: The dtype of the A tensor.
        b_type: The dtype of the B tensor.
        c_type: The dtype of the C tensor.
    """
    # u8u8, u8s8, s8s8, but not s8u8
    # Output must be 32-bit integer (int32 or uint32) since i8mm produces 4-wide
    # SIMD vectors.
    return (
        CompilationTarget.has_neon_int8_matmul()
        and (c_type == .int32 or c_type == .uint32)
        and (
            (a_type == .uint8 and b_type == .uint8)
            or (a_type == .uint8 and b_type == .int8)
            or (a_type == .int8 and b_type == .int8)
        )
    )


# Determines which kernel shape to use based on the matmul shape MxNxK.
# Currently only allows two shapes.
@always_inline
def get_kernel_type(m: Int, n: Int, k: Int) -> Bool:
    """Returns the kernel shape variant to use based on the matmul dimensions.

    Args:
        m: The M dimension of the matmul.
        n: The N dimension of the matmul.
        k: The K dimension of the matmul.
    """
    comptime if CompilationTarget.has_avx512f():
        return m > 0 and m <= 32
    elif CompilationTarget.has_neon():
        comptime if CompilationTarget.is_neoverse_n1():
            return (k % 4096) == 0
        else:
            return m > 32

    else:
        return False


def dispatch_get_kernel_type[
    FuncType: ImplicitlyCopyable & def[x: Bool]() raises -> None,
](func: FuncType, m: Int, n: Int, k: Int) raises:
    """Invokes `func` with the kernel type selected for the matmul dimensions.

    Parameters:
        FuncType: The comptime-parameterized function type to dispatch,
            specialized on a boolean kernel type.

    Args:
        func: The comptime-parameterized function to dispatch, specialized on
            a boolean kernel type.
        m: The M dimension of the matmul.
        n: The N dimension of the matmul.
        k: The K dimension of the matmul.
    """
    if get_kernel_type(m, n, k):
        func[True]()
    else:
        func[False]()


def dispatch_get_kernel_type[
    FuncType: ImplicitlyCopyable & def[x: Bool]() -> None,
](func: FuncType, m: Int, n: Int, k: Int):
    """Invokes `func` with the kernel type selected for the matmul dimensions.

    Non-raising overload of `dispatch_get_kernel_type`.

    Parameters:
        FuncType: The comptime-parameterized function type to dispatch,
            specialized on a boolean kernel type.

    Args:
        func: The comptime-parameterized function to dispatch, specialized on
            a boolean kernel type.
        m: The M dimension of the matmul.
        n: The N dimension of the matmul.
        k: The K dimension of the matmul.
    """
    if get_kernel_type(m, n, k):
        func[True]()
    else:
        func[False]()


@always_inline
def packA_i8mm[
    a_type: DType
](
    t0: Int,
    t1: Int,
    k: Int,
    a_ptr: UnsafePointer[mut=False, Scalar[a_type], ...],
    a_packed_ptr: UnsafePointer[mut=True, Scalar[a_type], ...],
):
    """Packs a range of rows of matrix A for the i8mm kernel layout.

    Parameters:
        a_type: The dtype of the A tensor.

    Args:
        t0: The starting row index of the range to pack.
        t1: The ending row index of the range to pack (exclusive).
        k: The K dimension of the matmul.
        a_ptr: Pointer to the source A matrix in row-major layout.
        a_packed_ptr: Pointer to the destination packed A buffer.
    """

    @always_inline
    def packA_helper[
        nrow: Int
    ](offset: Int) {var k, var t0, imm a_ptr, imm a_packed_ptr}:
        var kl = align_down(k, 8)
        var kh = align_up(k, 8)
        var j = t0 + offset
        for l in range(0, k, 8):
            comptime for idx in range(nrow):
                var t0 = a_ptr.load[width=8]((j + idx) * k + l)
                a_packed_ptr.store(kh * j + 2 * l + 8 * idx, t0)

        comptime for idx in range(nrow):
            var t0 = partial_simd_load[8](
                a_ptr + ((j + idx) * k + kl), 0, k - kl, 0
            )
            partial_simd_store(
                a_packed_ptr + (kh * j + 2 * kl + 8 * idx),
                0,
                k - kl,
                t0,
            )

    vectorize[2](t1 - t0, packA_helper)


@fieldwise_init
struct InnerKernelID(TrivialRegisterPassable):
    """Identifies the inner matmul kernel variant selected for a target."""

    comptime DEFAULT = InnerKernelID(0)
    comptime VNNI = InnerKernelID(1)
    comptime NEON = InnerKernelID(2)
    comptime I8MM = InnerKernelID(3)

    var value: Int

    @always_inline
    def __eq__(self, rhs: InnerKernelID) -> Bool:
        return self.value == rhs.value


@always_inline
def select_inner_kernel[
    a_type: DType, b_type: DType, c_type: DType
]() -> InnerKernelID:
    """Returns the inner kernel variant to use for the given dtypes and target.

    Parameters:
        a_type: The dtype of the A tensor.
        b_type: The dtype of the B tensor.
        c_type: The dtype of the C tensor.
    """
    comptime use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    comptime use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()

    comptime if use_i8mm:
        return InnerKernelID.I8MM
    elif CompilationTarget.has_neon() and not use_vnni and not use_i8mm:
        return InnerKernelID.NEON
    elif not use_vnni and not CompilationTarget.has_neon():
        return InnerKernelID.DEFAULT
    else:
        return InnerKernelID.VNNI
