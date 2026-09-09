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
"""Implements tiled CPU matrix multiplication kernels and their dispatcher.

Defines the `InnerMatmulKernel` trait and the `TiledMatmul` struct that drive
the outer M/N/K tile loops, plus the `matmul` entry point that selects an
inner microkernel (default, VNNI, NEON, or I8MM) or routes to Apple
Accelerate and GEMV fast paths.
"""
from std.collections import Optional
from std.math import align_up, ceildiv
from std.sys.info import align_of, simd_width_of

from std.algorithm import tile, vectorize

from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    TileTensor,
)
from layout.tile_layout import TensorLayout, row_major
from std.memory import alloc, dealloc, Allocation
from std.memory.alloc import Layout as AllocLayout
from max.runtime.asyncrt import parallelism_level

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type

from ...gemv import gemv
from ...packing import BTileGenerator
from ...utils import (
    GemmShape,
    InnerKernelID,
    KernelConfig,
    calculate_tile_n_k,
    dispatch_get_kernel_type,
    elementwise_epilogue_type,
    get_kernel_config,
    get_min_task_size,
    get_partitioned_matmul,
    packA_i8mm,
    select_inner_kernel,
)
from .apple_accelerate import apple_gemv, apple_matmul, use_apple_accelerate_lib
from .default import Inner_matmul_default
from .i8mm import Inner_matmul_i8mm
from .neon import Inner_matmul_neon
from .vnni import Inner_matmul_vnni

# Define a trait that defines the common functions across all existing
# microkernels:
# - _run_inner_loop_default()
# - _run_inner_loop_vnni()
# - _run_inner_loop_neon()
# - _run_inner_loop_i8mm()


trait InnerMatmulKernel(Deinitable, ImplicitlyCopyable):
    """Trait for CPU matmul microkernels operating on pre-packed tiles.

    Conforming types implement `__inner_matmul__`, which accumulates a
    (kernel_rows × TileN × TileK) block of the output matrix using a
    packed B tile in cache-friendly layout.
    """

    def __inner_matmul__[
        kernel_rows: Int,
        kernel_cols: Int,
        simd_size: Int,
    ](
        self,
        c: TileTensor[mut=True, ...],
        a: TileTensor,
        b_packed: TileTensor,
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: IndexList[2],
        skip_boundary_check: Bool,
    ):
        """Accumulates one packed `B` tile into the corresponding `C` tile.

        Parameters:
            kernel_rows: Number of `C` rows the microkernel accumulates per
                tile.
            kernel_cols: Number of `C` columns the microkernel accumulates per
                tile.
            simd_size: SIMD vector width used by the inner accumulation.

        Args:
            c: Output tile to accumulate into.
            a: Non-transposed left operand tile.
            b_packed: Pre-packed right operand panel (rank 3).
            global_offset: `(M, N, K)` offset of this tile in the full problem.
            global_bound: `(M, N, K)` bounds of the full problem.
            tile_n_k: Dynamic `(N, K)` extent of the tile to process.
            skip_boundary_check: Whether to skip partial-tile boundary handling.
        """
        comptime assert b_packed.flat_rank == 3, "b_packed must be rank 3"
        ...


def elementwise_epilogue_c_tile[
    simd_width: Int,
    c_type: DType,
    func: def[dtype: DType, width: SIMDLength, *, alignment: Int = 1](
        IndexList[2], SIMD[dtype, width]
    ) capturing -> None,
](
    offset: GemmShape,
    tile_len: GemmShape,
    c: TileTensor[mut=False, c_type, address_space=.GENERIC, ...],
):
    """Applies a vectorized epilogue function over a 2D C output tile.

    Iterates column chunks of `tile_len.N` using SIMD width `simd_width`,
    calling `func` with the global (m, n) coordinates and the loaded values.

    Parameters:
        simd_width: SIMD vector width for epilogue loads.
        c_type: Data type of the C matrix.
        func: Epilogue function called with coordinates and a SIMD value chunk.

    Args:
        offset: Starting (M, N, K) offset within the global matmul space.
        tile_len: Number of rows and columns to process.
        c: Read-only view of the C output tile.
    """

    @always_inline
    def activation_on_col_chunk[col_chunk_size: Int](idx_n: Int) {imm}:
        var n_coord = idx_n + offset.N
        for idx_m in range(tile_len.M):
            var m_coord = idx_m + offset.M
            var c_coord = Index(m_coord, n_coord)
            var c_val = c.load_linear[width=col_chunk_size, alignment=1](
                c_coord
            )
            func[c_type, col_chunk_size](c_coord, c_val)

    vectorize[simd_width](tile_len.N, activation_on_col_chunk)


# Interface method
def tiled_matmul_run[
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    simd_size: Int,
    elementwise_epilogue_enabled: Bool,
    kernel_id: InnerKernelID,
    algorithm: InnerMatmulKernel,
    ElementwiseEpilogueFnType: ImplicitlyCopyable
    & def(
        GemmShape,
        GemmShape,
        TileTensor[mut=False, address_space=.GENERIC, ...],
    ) -> None,
](
    alg: algorithm,
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, address_space=.GENERIC, ...],
    elementwise_epilogue_fn: ElementwiseEpilogueFnType,
    global_tile_shape: GemmShape,
    global_tile_offset: GemmShape,
):
    """Interface function to run tiled matmul on a given sub-tile.

    Parameters:
        config: Kernel configuration controlling tile dimensions and SIMD
            width.
        transpose_b: Whether the B operand is transposed.
        b_packed: Whether B was pre-packed offline in the cache-friendly
            layout.
        simd_size: SIMD vector width for the elementwise epilogue.
        elementwise_epilogue_enabled: Whether to apply the elementwise
            epilogue on the last K tile.
        kernel_id: Identifier of the inner microkernel to dispatch.
        algorithm: Microkernel implementing the inner accumulate loop.
        ElementwiseEpilogueFnType: Type of the elementwise epilogue function
            applied to each output tile.

    Args:
        alg: InnerMatmulKernel algorithm for microkernel.
        c: Pre-allocated buffer space for result.
        a: Operand A of the matmul.
        b: Operand B of the mamtul.
        elementwise_epilogue_fn: The elementwise epilogue function.
        global_tile_shape: Tile shape this call will process.
        global_tile_offset: Tile offset on the original buffer.
    """

    var tile_n_k = calculate_tile_n_k[
        a.dtype, b.dtype, c.dtype, config.kernel_cols
    ](global_tile_shape)

    # Construct simple TileTensors to strip any extra type params (e.g.
    # linear_idx_type, element_size) from the existential `...` pattern.
    var c_tt = TileTensor(c.ptr, c.layout)
    var a_tt = TileTensor(a.ptr, a.layout)
    var b_tt = TileTensor(b.ptr, b.layout)
    var matmul = TiledMatmul[
        config,
        transpose_b,
        b_packed,
        elementwise_epilogue_enabled,
        kernel_id,
    ](
        alg,
        c_tt,
        a_tt,
        b_tt,
        tile_n_k,
        global_tile_offset,
        global_tile_shape,
        BTileGenerator[
            config,
            a.dtype,
            b.dtype,
            c.dtype,
            type_of(b_tt).LayoutType,
            transpose_b,
            b_packed,
        ].get(b_tt, tile_n_k),
        elementwise_epilogue_fn,
    )
    matmul._outer_k_loop()


# Tiled Matmul Implementation.
@fieldwise_init
struct TiledMatmul[
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_epilogue_enabled: Bool,
    kernel_id: InnerKernelID,
    a_type: DType,
    a_layout: TensorLayout,
    a_origin: ImmOrigin,
    b_type: DType,
    b_layout: TensorLayout,
    b_origin: ImmOrigin,
    c_type: DType,
    c_layout: TensorLayout,
    c_origin: MutOrigin,
    algorithm: InnerMatmulKernel,
    ElementwiseEpilogueFnType: ImplicitlyCopyable
    & def(
        GemmShape,
        GemmShape,
        TileTensor[mut=False, address_space=.GENERIC, ...],
    ) -> None,
](ImplicitlyCopyable):
    """Tiled matmul implementation integrating packing, inner loop and tile
    partitions.

    Parameters:
        config: Kernel configuration controlling tile dimensions and SIMD
            width.
        transpose_b: Whether the B operand is transposed.
        b_packed: Whether B was pre-packed offline in the cache-friendly
            layout.
        elementwise_epilogue_enabled: Whether to apply the elementwise
            epilogue on the last K tile.
        kernel_id: Identifier of the inner microkernel to dispatch.
        a_type: Element type of the A operand.
        a_layout: Memory layout of the A operand.
        a_origin: Memory origin provenance of the A operand.
        b_type: Element type of the B operand.
        b_layout: Memory layout of the B operand.
        b_origin: Memory origin provenance of the B operand.
        c_type: Element type of the C output.
        c_layout: Memory layout of the C output.
        c_origin: Memory origin provenance of the mutable C output.
        algorithm: Microkernel implementing the inner accumulate loop.
        ElementwiseEpilogueFnType: Type of the elementwise epilogue function
            applied to each output tile.
    """

    var alg: Self.algorithm
    """Inner microkernel conforming to `InnerMatmulKernel`."""
    var c: TileTensor[Self.c_type, Self.c_layout, Self.c_origin]
    """Output `TileTensor` the result accumulates into."""
    var a: TileTensor[Self.a_type, Self.a_layout, Self.a_origin]
    """Left operand `TileTensor`."""
    var b: TileTensor[Self.b_type, Self.b_layout, Self.b_origin]
    """Right operand `TileTensor`."""
    # Dynamic tile parameter.
    var tile_n_k: IndexList[2]
    """Dynamic `(N, K)` tile extents used to partition the problem."""

    # Tile starting points on the (M,N,K) coordinates.
    var global_tile_offset: GemmShape
    """`(M, N, K)` offset of this routine's tile region."""

    # Tile sizes this routine will process on the (M,N,K) coordinates.
    var global_tile_shape: GemmShape
    """`(M, N, K)` extent of this routine's tile region."""

    var b_tile_generator: BTileGenerator[
        Self.config,
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.b_layout,
        Self.transpose_b,
        Self.b_packed,
        Self.b_origin,
    ]
    """Generator that packs `B` sub-tiles for the inner kernel."""

    var elementwise_epilogue_fn: Self.ElementwiseEpilogueFnType
    """Fused elementwise epilogue applied to each output tile."""

    def _outer_m_loop[
        tile_kernel_cols: Int
    ](
        self,
        global_offset: GemmShape,
        sub_tile_n: Int,
        sub_tile_k: Int,
        last_k_tile: Bool,
    ):
        """
        Helper function: Pack a sub-tile of B and iterate through all the rows
            of C.

        Parameters:
            tile_kernel_cols: Inner dimension of the packed data layout.

        Args:
            global_offset: 3D global offset within the whole
                matmul problem space.
            sub_tile_n: Dynamic tile size to use on N dimension.
            sub_tile_k: Dynamic tile size to use on K dimension.
            last_k_tile: The last k tile.
        """
        # valid distance in each dimension from the current offset to the end of the matrix
        var knm_bounds = (
            self.global_tile_shape + self.global_tile_offset - global_offset
        )

        var b_packed_tile = self.b_tile_generator.get_tile[tile_kernel_cols](
            global_offset,
            Index(sub_tile_n, sub_tile_k),
            Index(knm_bounds.N, knm_bounds.K),
        )

        # Launch the MLoop
        # The upper bounds apply to runtime packing. For pre-packing, the
        # tile has been padded to fit (sub_tile_n, sub_tile_k).
        var sub_tile_n_k = Index(
            min(sub_tile_n, knm_bounds.N), min(sub_tile_k, knm_bounds.K)
        )

        @always_inline
        def row_iteration[
            tile_kernel_rows: Int
        ](row_offset: Int) {var sub_tile_n_k, var b_packed_tile, imm}:
            var skip_boundary_check = knm_bounds[1] > sub_tile_n
            self.alg.__inner_matmul__[
                tile_kernel_rows,
                tile_kernel_cols,
                Self.config.simd_size,
            ](
                self.c,
                self.a,
                b_packed_tile,
                global_offset + GemmShape(row_offset, 0, 0),
                self.global_tile_offset + self.global_tile_shape,
                sub_tile_n_k,
                skip_boundary_check,
            )

            if Self.elementwise_epilogue_enabled and last_k_tile:
                self.elementwise_epilogue_fn(
                    global_offset + GemmShape(row_offset, 0, 0),
                    GemmShape(
                        tile_kernel_rows, sub_tile_n_k[0], sub_tile_n_k[1]
                    ),
                    self.c.as_immut(),
                )

        comptime if Self.kernel_id == InnerKernelID.I8MM:
            tile[[2 * Self.config.kernel_rows, 8, 6, 4, 2, 1],](
                0,  # starting row offset
                knm_bounds.M,  # row bound
                row_iteration,
            )
        else:
            tile[[Self.config.kernel_rows, 4, 3, 2, 1],](
                0, knm_bounds.M, row_iteration
            )

    # Iterate on the N dimension of the gemm space.
    def _outer_n_loop(
        self, global_offset: GemmShape, sub_tile_k: Int, last_k_tile: Bool
    ):
        """Iterate on the N dimension of the whole problem space.

        Args:
            global_offset: 3D global offset within the whole matmul problem
                space.
            sub_tile_k: Dynamic tile size to use on K dimension.
            last_k_tile: The last k tile.
        """
        var valid_col_count: Int = (
            self.global_tile_shape.N
            + self.global_tile_offset.N
            - global_offset.N
        )
        var tile_n: Int = self.tile_n_k[0]

        @always_inline
        def m_loop[
            secondary_tile_size: Int
        ](col_idx: Int, tile_size_n: Int) {imm}:
            self._outer_m_loop[secondary_tile_size](
                global_offset + GemmShape(0, col_idx, 0),
                tile_size_n,
                sub_tile_k,
                last_k_tile,
            )

        # if b is packed, the packing was performed offline using a single inner
        # size and tile_n.
        comptime if not Self.b_packed:
            comptime secondary_tiles: List[Int] = [
                Self.config.kernel_cols,
                2 * Self.config.simd_size,
                Self.config.simd_size,
            ]
            tile[
                secondary_tiles,
                Self.config.simd_size,
            ](
                0,
                valid_col_count,
                tile_n,
                2 * Self.config.simd_size,
                Self.config.simd_size,
                primary_cleanup_tile=Self.config.simd_size,
                workgroup_function=m_loop,
            )
        else:
            comptime secondary_tiles_packed_b: List[Int] = [
                Self.config.kernel_cols
            ]
            tile[secondary_tiles_packed_b, Self.config.kernel_cols](
                0,
                valid_col_count,
                tile_n,
                primary_cleanup_tile=tile_n,
                workgroup_function=m_loop,
            )

    # Iterate over the K dimension of the gemm space.
    def _outer_k_loop(
        self,
    ):
        """Iterate on the K dimension of the whole problem space."""

        # Each tiled iteration on the k dimension.
        @always_inline
        def k_iteration(k_offset: Int, k_tile_size: Int) {imm}:
            var last_k_tile = (
                k_offset + k_tile_size + self.global_tile_offset.K
                == self.global_tile_shape.K
            )
            self._outer_n_loop(
                GemmShape(0, 0, k_offset) + self.global_tile_offset,
                k_tile_size,
                last_k_tile,
            )

        tile(
            0,  # k offset
            self.global_tile_shape.K,  # valid K count
            self.tile_n_k[1],  # max tile k size
            workgroup_function=k_iteration,
        )


@always_inline
def _matmul_cpu_impl[
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    kernel_id: InnerKernelID,
    algorithm: InnerMatmulKernel,
](
    alg: algorithm,
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, address_space=.GENERIC, ...],
    num_threads: Int = -1,
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime assert c.rank == 2 and c.flat_rank == 2
    comptime assert a.rank == 2 and a.flat_rank == 2
    comptime assert b.rank == 2 and b.flat_rank == 2

    var shape = GemmShape.get[transpose_b](c, a, b)
    var m = shape.M
    var n = shape.N
    var k = shape.K
    # Matrix by vector pattern -> use gemv
    if n == 1:
        var out = TileTensor(c.ptr, row_major(Coord(Int(c.dim[0]()))))
        var rhs = TileTensor(b.ptr, row_major(Coord(Int(b.dim[0]()))))
        gemv[parallelize=True, elementwise_lambda_fn=elementwise_lambda_fn](
            out, a, rhs
        )
    else:
        # SGEMM calls for MacOS >= 13.0.0 and a, b, c of type Float32 are
        # directed to the special Apple-specific implementations.
        # apple_matmul handles generic matmuls.
        # apple_gemv handles cases with M=1 (where apple_matmul is suboptimal).
        comptime if use_apple_accelerate_lib[c.dtype, a.dtype, b.dtype]():
            if m == 1:
                apple_gemv[
                    b_packed=b_packed,
                    transpose_b=transpose_b,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](c, a, b, ctx=ctx)
            else:
                comptime apple_transpose = True if b_packed else transpose_b
                apple_matmul[
                    transpose_b=apple_transpose,
                    elementwise_lambda_fn=elementwise_lambda_fn,
                ](c, a, b)
            return

        var complexity = m * n * k
        var num_tasks = min(
            ceildiv(complexity, get_min_task_size()),
            num_threads if num_threads > 0 else parallelism_level(ctx),
        )

        comptime use_i8mm = kernel_id == InnerKernelID.I8MM
        comptime simd_size = config.simd_size
        comptime alignment = align_of[SIMD[c.dtype, simd_size]]()
        var kh = align_up(k, 8)
        var mh = align_up(m, 2)

        var a_packed_alloc: Optional[Allocation[Scalar[a.dtype]]] = None
        comptime if use_i8mm:
            # Retire the empty `None` before reassigning: `Optional[Allocation]`
            # is not implicitly deletable, so overwriting it cannot drop the old
            # value implicitly.
            a_packed_alloc^.deinit_with(dealloc[Scalar[a.dtype]])
            a_packed_alloc = alloc(
                AllocLayout[Scalar[a.dtype]](count=mh * kh, alignment=alignment)
            )

        @always_inline
        def pack_task_func(
            task_id: Int,
        ) {mut a_packed_alloc, var m, var k, var num_tasks, imm}:
            var sub_matmul_config = get_partitioned_matmul[
                a.dtype,
                b.dtype,
                c.dtype,
                config.kernel_rows,
                config.kernel_cols,
            ](m, 1, k, task_id, num_tasks)
            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return
            var t0 = sub_matmul_config.offset[0]
            var t1 = t0 + sub_matmul_config.shape[0]
            packA_i8mm[a.dtype](
                t0, t1, k, a.ptr, a_packed_alloc.unsafe_value().unsafe_ptr()
            )

        @always_inline
        def task_func(
            task_id: Int,
        ) {var m, var k, var num_tasks, var n, var mh, var kh, imm}:
            var sub_matmul_config = get_partitioned_matmul[
                a.dtype,
                b.dtype,
                c.dtype,
                config.kernel_rows,
                config.kernel_cols,
            ](m, n, k, task_id, num_tasks)

            if (
                sub_matmul_config.shape[0] <= 0
                or sub_matmul_config.shape[1] <= 0
            ):
                return

            comptime use_i8mm = kernel_id == InnerKernelID.I8MM

            comptime if use_i8mm:
                _submatmul_sequential_sync[
                    config,
                    transpose_b,
                    b_packed,
                    elementwise_lambda_fn,
                    kernel_id,
                ](
                    alg,
                    c,
                    TileTensor(
                        a_packed_alloc.unsafe_value().unsafe_ptr(),
                        row_major(Coord(mh, kh)),
                    ),
                    b,
                    GemmShape(sub_matmul_config.shape),
                    GemmShape(sub_matmul_config.offset),
                )
            else:
                _submatmul_sequential_sync[
                    config,
                    transpose_b,
                    b_packed,
                    elementwise_lambda_fn,
                    kernel_id,
                ](
                    alg,
                    c,
                    TileTensor(
                        a.ptr.unsafe_mut_cast[True]().as_unsafe_any_origin(),
                        a.layout,
                    ),
                    b,
                    GemmShape(sub_matmul_config.shape),
                    GemmShape(sub_matmul_config.offset),
                )

        # i8mm partition needs to be optimized as a function of m, n and k
        # Also parallelize currently is slower than asyn_parallelize which is depreciated now.
        # See issue 27734
        comptime if use_i8mm:
            sync_parallelize(pack_task_func, num_tasks, ctx)

        # TODO (#12624): Closure captures some state on the stack so this needs
        # to be synchronous in order to keep that state alive
        sync_parallelize(task_func, num_tasks, ctx)

        a_packed_alloc^.deinit_with(dealloc[Scalar[a.dtype]])


@always_inline
def matmul[
    *,
    transpose_b: Bool = False,
    b_packed: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    saturated_vnni: Bool = False,
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, address_space=.GENERIC, ...],
    kernel_type_m: Int,
    num_threads: Int = -1,
    ctx: Optional[DeviceContext] = None,
) raises:
    """TileTensor matmul dispatcher. Selects kernel type and delegates to
    `_matmul_cpu_impl`.

    Parameters:
        transpose_b: Whether the B operand is transposed (defaults to
            `False`).
        b_packed: Whether B was pre-packed offline in the cache-friendly
            layout (defaults to `False`).
        elementwise_lambda_fn: Optional elementwise epilogue applied to each
            output tile (defaults to `None`).
        saturated_vnni: When `True`, uses the saturating x86 variant, which
            requires the `a` operand to lie in `[0, 127]`; it saves
            instructions only on the AVX2 emulation path used when the target
            lacks VNNI (defaults to `False`).

    Args:
        c: Output matrix buffer accumulating the matmul result.
        a: Left operand of the matmul.
        b: Right operand of the matmul.
        kernel_type_m: M dimension used to select the kernel variant, or `0`
            if unknown.
        num_threads: Number of worker threads to use (defaults to `-1`,
            which selects automatically).
        ctx: Device context governing parallelism (defaults to `None`).
    """
    comptime assert c.rank == 2, "c must be rank 2"
    comptime assert a.rank == 2, "a must be rank 2"
    comptime assert b.rank == 2, "b must be rank 2"
    comptime assert c.flat_rank == 2
    comptime assert a.flat_rank == 2
    comptime assert b.flat_rank == 2

    # KERN-2790: for low-precision output (f16/bf16), route through an f32
    # scratch buffer so we accumulate in fp32 across all K tiles and cast
    # down in a single pass at the end. Mirrors GPU behavior where the
    # fp32 accumulator is cast to the output dtype right at the epilogue.
    comptime scratch_type = get_accum_type[c.dtype]()
    comptime if scratch_type != c.dtype:
        var scratch_shape = GemmShape.get[transpose_b](c, a, b)
        var scratch_m = scratch_shape.M
        var scratch_n = scratch_shape.N

        comptime scratch_simd = simd_width_of[scratch_type]()
        comptime scratch_align = align_of[SIMD[scratch_type, scratch_simd]]()
        var scratch_alloc = alloc(
            AllocLayout[Scalar[scratch_type]](
                count=scratch_m * scratch_n, alignment=scratch_align
            )
        ).into_managed()
        var scratch_ptr: UnsafePointer[
            Scalar[scratch_type], origin_of(scratch_alloc)
        ] = scratch_alloc.unsafe_ptr()
        var scratch = TileTensor(
            scratch_ptr, row_major(Coord(scratch_m, scratch_n))
        )

        @__parameter
        @always_inline
        def cast_epilogue[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](coord: IndexList[2], val: SIMD[dtype, width]):
            var cast_val = val.cast[c.dtype]()
            comptime if elementwise_lambda_fn:
                comptime user_fn = elementwise_lambda_fn.value()
                user_fn[c.dtype, width, alignment=alignment](coord, cast_val)
            else:
                c.store_linear[width=width, alignment=alignment](
                    coord, cast_val
                )

        matmul[
            transpose_b=transpose_b,
            b_packed=b_packed,
            elementwise_lambda_fn=cast_epilogue,
            saturated_vnni=saturated_vnni,
        ](scratch, a, b, kernel_type_m, num_threads, ctx)

        return

    comptime kernel_id = select_inner_kernel[a.dtype, b.dtype, c.dtype]()

    @always_inline
    def dispatch_on_kernel_type[
        kernel_type: Bool
    ]() raises {c, a, b, num_threads, ctx}:
        comptime config = get_kernel_config[
            a.dtype,
            b.dtype,
            c.dtype,
            kernel_type=kernel_type,
        ]()

        comptime if kernel_id == InnerKernelID.DEFAULT:
            _matmul_cpu_impl[
                config,
                transpose_b,
                b_packed,
                elementwise_lambda_fn,
                kernel_id,
            ](
                Inner_matmul_default(),
                c,
                a,
                b,
                num_threads,
                ctx,
            )
        elif kernel_id == InnerKernelID.VNNI:
            _matmul_cpu_impl[
                config,
                transpose_b,
                b_packed,
                elementwise_lambda_fn,
                kernel_id,
            ](
                Inner_matmul_vnni[saturated_vnni](),
                c,
                a,
                b,
                num_threads,
                ctx,
            )
        elif kernel_id == InnerKernelID.NEON:
            _matmul_cpu_impl[
                config,
                transpose_b,
                b_packed,
                elementwise_lambda_fn,
                kernel_id,
            ](
                Inner_matmul_neon(),
                c,
                a,
                b,
                num_threads,
                ctx,
            )
        elif kernel_id == InnerKernelID.I8MM:
            _matmul_cpu_impl[
                config,
                transpose_b,
                b_packed,
                elementwise_lambda_fn,
                kernel_id,
            ](
                Inner_matmul_i8mm(),
                c,
                a,
                b,
                num_threads,
                ctx,
            )
        else:
            comptime assert False, "no _run_inner_loop implementation"

    var shape = GemmShape.get[transpose_b](c, a, b)
    var n = shape.N
    var k = shape.K
    dispatch_get_kernel_type(dispatch_on_kernel_type, kernel_type_m, n, k)


def _submatmul_sequential_sync[
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    kernel_id: InnerKernelID,
    algorithm: InnerMatmulKernel,
](
    alg: algorithm,
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, address_space=.GENERIC, ...],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
):
    comptime simd_size = config.simd_size

    def elementwise_closure(
        offset: GemmShape,
        shape: GemmShape,
        c_read: TileTensor[mut=False, address_space=.GENERIC, ...],
    ):
        comptime if elementwise_lambda_fn:
            comptime func = elementwise_lambda_fn.value()
            elementwise_epilogue_c_tile[
                simd_size,
                c_read.dtype,
                func,
            ](
                offset,
                shape,
                c_read,
            )
        else:
            pass

    tiled_matmul_run[
        config,
        transpose_b,
        b_packed,
        simd_size,
        elementwise_lambda_fn.__bool__(),
        kernel_id,
    ](
        alg,
        c,
        a,
        b,
        elementwise_closure,
        sub_matrix_shape,
        sub_matrix_offset,
    )


def _submatmul_sequential_sync[
    config: KernelConfig,
    transpose_b: Bool,
    b_packed: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    saturated_vnni: Bool,
](
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, address_space=.GENERIC, ...],
    sub_matrix_shape: GemmShape,
    sub_matrix_offset: GemmShape,
):
    comptime kernel_id = select_inner_kernel[a.dtype, b.dtype, c.dtype]()

    comptime if kernel_id == InnerKernelID.DEFAULT:
        _submatmul_sequential_sync[
            config,
            transpose_b,
            b_packed,
            elementwise_lambda_fn,
            kernel_id,
        ](
            Inner_matmul_default(),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
        )
    elif kernel_id == InnerKernelID.VNNI:
        _submatmul_sequential_sync[
            config,
            transpose_b,
            b_packed,
            elementwise_lambda_fn,
            kernel_id,
        ](
            Inner_matmul_vnni[saturated_vnni](),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
        )
    elif kernel_id == InnerKernelID.NEON:
        _submatmul_sequential_sync[
            config,
            transpose_b,
            b_packed,
            elementwise_lambda_fn,
            kernel_id,
        ](
            Inner_matmul_neon(),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
        )
    elif kernel_id == InnerKernelID.I8MM:
        _submatmul_sequential_sync[
            config,
            transpose_b,
            b_packed,
            elementwise_lambda_fn,
            kernel_id,
        ](
            Inner_matmul_i8mm(),
            c,
            a,
            b,
            sub_matrix_shape,
            sub_matrix_offset,
        )
    else:
        comptime assert False, "no _run_inner_loop implementation"
