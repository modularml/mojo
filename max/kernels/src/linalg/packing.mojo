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

"""Provides matrix packing routines that reorder B tiles into cache-friendly layouts for matmul."""

from std.math import align_down, align_up
from std.sys import align_of, simd_width_of
from std.sys.info import CompilationTarget
from std.sys.intrinsics import PrefetchOptions

from std.algorithm import unswitch
from linalg.utils import partial_simd_load
from layout import Coord, Idx, TileTensor
from layout.coord import DynamicCoord
from layout.tile_layout import RowMajorLayout
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from std.sys import prefetch
from layout.tile_layout import TensorLayout, row_major
from std.memory import (
    unsafe_memcpy,
    unsafe_memset_zero,
    unsafe_stack_allocation,
)

from std.utils.index import Index, IndexList

from .matmul.cpu.apple_accelerate import use_apple_accelerate_lib
from .transpose import transpose, transpose_inplace
from .utils import (
    GemmShape,
    KernelConfig,
    _get_tile_n_k,
    dispatch_get_kernel_type,
    get_kernel_config,
    get_matmul_arch_factor,
    get_pack_data_size,
    get_packB_unroll_factor,
    use_i8mm_fn,
    use_vnni_fn,
)


@fieldwise_init
struct PackMatrixRows[
    original_mut: Bool,
    //,
    dtype: DType,
    simd_size: Int,
    row_inner_size: Int,
    packed_origin: MutOrigin,
    original_origin: Origin[mut=original_mut],
    packed_layout: TensorLayout,
    original_layout: TensorLayout,
](ImplicitlyCopyable):
    """Pack rows from a matrix into the mlas packed layout and
    extract inner vectors of rows into the packed inner dimension,
    e.g. extract tile [X, Y] and pack into [Xo][Y][Xi].

    Parameters:
        original_mut: True if the original matrix buffer is mutable (inferred).
        dtype: Element type of the matrix being packed.
        simd_size: SIMD vector width for `dtype`.
        row_inner_size: Size of the inner dimension along rows in the
            packed layout; must be a multiple of `simd_size`.
        packed_origin: Origin of the packed output `TileTensor`.
        original_origin: Origin of the original input `TileTensor`.
        packed_layout: Layout of the packed output `TileTensor`.
        original_layout: Layout of the original input `TileTensor`.
    """

    # packed matrix (rank 3)
    var packed_matrix: TileTensor[
        Self.dtype, Self.packed_layout, Self.packed_origin
    ]
    # original matrix (rank 2)
    var original_matrix: TileTensor[
        Self.dtype, Self.original_layout, Self.original_origin
    ]
    # offsets in original matrix
    var global_offset: DynamicCoord[.int64, 2]
    # number of Row and Col to pack.
    #  in [Row, Col]
    var pack_tile_dim: DynamicCoord[.int64, 2]
    # valid data bound within the tile.
    var valid_data_dim: DynamicCoord[.int64, 2]
    # valid multiple-of-simd data bound within the tile.
    var valid_simd_dim: DynamicCoord[.int64, 2]

    # Interface method:
    #  run the packing and store to the given buffer.
    @staticmethod
    def run(
        packed_matrix: TileTensor[
            Self.dtype, Self.packed_layout, Self.packed_origin
        ],
        original_matrix: TileTensor[
            Self.dtype, Self.original_layout, Self.original_origin
        ],
        global_offset: IndexList[2],
        pack_tile_dim: IndexList[2],
        valid_data_dim: IndexList[2],
    ):
        """Interface function to run the packing routine.
        Args:
            packed_matrix: Pre-allocated buffer space for packed data.
            original_matrix: Data buffer containing the original matrix
                to pack.
            global_offset: Offset to use when indexing the original matrix.
            pack_tile_dim: 2D dimension tuple describing the size of the
                packed tile.
            valid_data_dim: 2D dimension tuple describing the amount of
                valid data on the global buffer starting from the offset.
        """
        comptime assert Self.row_inner_size % Self.simd_size == 0

        var instance = Self(
            packed_matrix,
            original_matrix,
            Coord(global_offset),
            Coord(pack_tile_dim),
            Coord(valid_data_dim),
            Coord(
                Index(
                    align_down(
                        min(
                            valid_data_dim[0],
                            pack_tile_dim[0],
                        ),
                        Self.simd_size,
                    ),
                    align_down(
                        min(
                            valid_data_dim[1],
                            pack_tile_dim[1],
                        ),
                        Self.simd_size,
                    ),
                )
            ),
        )

        instance._pack()

    def _transpose_pack_helper[
        skip_row_bound: Bool,
        skip_col_bound: Bool,
    ](
        self,
        transpose_buffer: TileTensor[mut=True, Self.dtype, ...],
        local_off_set: IndexList[2],
    ):
        """Helper function: transpose packs a [simd_size, simd_size] sub-tile of
        matrix, with bound checking and zero-filling. Bound checking can be
        statically skipped, based on the parameters.
           Args:
               skip_row_bound: Boundary check on x dimension will be
                   skipped if true.
               skip_col_bound: Boundary check on y dimension will be
                   skpped if true.
               transpose_buffer: Pre-allocated work space to hold
                   transposed temporary data.
               local_off_set: Offset of the sub-tile to work on
                   within the whole tile of data to pack.
        """
        # Calculate the remaining bound from the local offset.
        # Boundaries for readable data.
        var read_bound = Index(
            Int(self.valid_data_dim[0].value()) - local_off_set[0],
            Int(self.valid_data_dim[1].value()) - local_off_set[1],
        )
        # Boundaries for writeable space.
        var write_bound = Index(
            Int(self.pack_tile_dim[0].value()) - local_off_set[0],
            Int(self.pack_tile_dim[1].value()) - local_off_set[1],
        )

        # Global index the packing is starting from.
        var start_idx_global = Index(
            local_off_set[0] + Int(self.global_offset[0].value()),
            local_off_set[1] + Int(self.global_offset[1].value()),
        )

        # Fill the simd_size x simd_size transpose buffer
        #  with un-transposed data.
        comptime for idx in range(Self.simd_size):
            comptime inner_row_idx = idx
            # Check that the current row has valid data.
            if skip_row_bound or (inner_row_idx < read_bound[0]):
                var row_global_index = Index(
                    start_idx_global[0] + inner_row_idx,
                    start_idx_global[1],
                )
                var row_data: SIMD[Self.dtype, Self.simd_size]

                comptime if skip_col_bound:
                    # This is fastest path where both row and col bounds
                    #  are skipped so the code path is simd-in and simd-out
                    #  without any predicate.
                    row_data = self.original_matrix.load_linear[
                        width=Self.simd_size, alignment=1
                    ](row_global_index)
                else:
                    # Not skipping col bound, need to do a partial fill of
                    #  the transpose buffer row.
                    row_data = partial_simd_load[Self.simd_size](
                        self.original_matrix.ptr
                        + self.original_matrix.layout(
                            Coord(
                                row_global_index[0],
                                row_global_index[1],
                            )
                        ),
                        0,  # no left bound.
                        read_bound[1],
                        0,
                    )

                transpose_buffer.store_linear[
                    width=Self.simd_size, alignment=1
                ](Index(inner_row_idx, 0), row_data)
            else:
                # Row out of defined bound, fill the transpose buffer with zero
                transpose_buffer.store_linear[
                    width=Self.simd_size, alignment=1
                ](
                    Index(inner_row_idx, 0),
                    SIMD[Self.dtype, Self.simd_size](0),
                )

        # Transpose the buffered data
        transpose_inplace[Self.simd_size, Self.simd_size, Self.dtype](
            transpose_buffer
        )

        # Write to packed space:
        #  transposed_inner_row_idx now corresponds to the original column idx.
        comptime for idx in range(Self.simd_size):
            var transposed_data = transpose_buffer.load_linear[
                width=Self.simd_size, alignment=1
            ](Index(idx, 0))
            # compute the packed index
            var _row_outer, _row_inner = divmod(
                local_off_set[0], Self.row_inner_size
            )

            if skip_col_bound or (idx < write_bound[1]):
                self.packed_matrix.store_linear[
                    width=Self.simd_size, alignment=1
                ](
                    Index(
                        _row_outer,
                        local_off_set[1] + idx,
                        _row_inner,
                    ),
                    transposed_data,
                )
            # Out of bound columns are discarded as there's no allocation for them
            #  in the packed buffer.

    def _pack(self):
        """Helper function: Allocates transpose workspace and launch the
        transpose helper function until all required data has been packed.
        """

        var transpose_buffer = tt_stack_allocation[dtype=Self.dtype,](
            row_major[Self.simd_size, Self.simd_size]()
        )

        var valid_tile_simd_dim = Index(
            Int(
                min(
                    self.valid_simd_dim[0].value(),
                    self.pack_tile_dim[0].value(),
                )
            ),
            Int(
                min(
                    self.valid_simd_dim[1].value(),
                    self.pack_tile_dim[1].value(),
                )
            ),
        )

        # fill rows with valid data

        var row_idx: Int = 0
        var col_idx: Int = 0

        # An unswitch-able unit function that transpose packs a small tile.
        @always_inline
        def transpose_pack_unit[
            static_switch0: Bool, static_switch1: Bool
        ]() {var transpose_buffer, imm}:
            self._transpose_pack_helper[
                # skip_row_bound, skip_col_bound
                static_switch0,
                static_switch1,
            ](
                transpose_buffer,
                # local offset
                (row_idx, col_idx),
            )

        # Pack the whole matrices with the unit helper.
        var pack_tile_rows = Int(self.pack_tile_dim[0].value())
        var pack_tile_cols = Int(self.pack_tile_dim[1].value())
        while row_idx < pack_tile_rows:
            col_idx = 0
            while col_idx < pack_tile_cols:
                unswitch(
                    row_idx + Self.simd_size <= valid_tile_simd_dim[0],
                    col_idx + Self.simd_size <= valid_tile_simd_dim[1],
                    transpose_pack_unit,
                )
                col_idx += Self.simd_size
            row_idx += Self.simd_size


@fieldwise_init
struct PackMatrixCols[
    original_mut: Bool,
    //,
    dtype: DType,
    simd_size: Int,
    column_inner_size: Int,
    use_vnni: Bool,
    use_i8mm: Bool,
    packed_origin: MutOrigin,
    original_origin: Origin[mut=original_mut],
    packed_layout: TensorLayout,
    original_layout: TensorLayout,
](ImplicitlyCopyable):
    """Pack columns from a matrix into the mlas packed layout and
    extract inner vectors of columns into the packed inner dimension,
    e.g. extracts [X, Y] and packs as [Yo][X][Yi].

    Parameters:
        original_mut: True if the original matrix buffer is mutable (inferred).
        dtype: Element type of the matrix being packed.
        simd_size: SIMD vector width for `dtype`.
        column_inner_size: Size of the inner dimension along columns in the
            packed layout; must be a multiple of `simd_size`.
        use_vnni: True to pack for the VNNI instruction layout.
        use_i8mm: True to pack for the i8mm instruction layout.
        packed_origin: Origin of the packed output `TileTensor`.
        original_origin: Origin of the original input `TileTensor`.
        packed_layout: Layout of the packed output `TileTensor`.
        original_layout: Layout of the original input `TileTensor`.
    """

    # packed matrix (rank 3)
    var packed_matrix: TileTensor[
        Self.dtype, Self.packed_layout, Self.packed_origin
    ]
    # original matrix (rank 2)
    var original_matrix: TileTensor[
        Self.dtype, Self.original_layout, Self.original_origin
    ]
    # offsets in original matrix:
    var global_offset: DynamicCoord[.int64, 2]
    # number of Row and Col to pack.
    #  in [Row, Col]
    var pack_tile_dim: DynamicCoord[.int64, 2]
    # valid data bound within the tile.
    var valid_data_dim: DynamicCoord[.int64, 2]

    # Interface function:
    @staticmethod
    def run(
        packed_matrix: TileTensor[
            Self.dtype, Self.packed_layout, Self.packed_origin
        ],
        original_matrix: TileTensor[
            Self.dtype, Self.original_layout, Self.original_origin
        ],
        global_offset: IndexList[2],
        pack_tile_dim: IndexList[2],
        valid_data_dim: IndexList[2],
    ):
        """Interface function to run the packing routine.
        Args:
            packed_matrix: Pre-allocated buffer space for packed data.
            original_matrix: Data buffer containing the original matrix
                to pack.
            global_offset: Offset to use when indexing the original matrix.
            pack_tile_dim: 2D dimension tuple describing the size of the
                packed tile.
            valid_data_dim: 2D dimension tuple describing the amount of
                valid data on the global buffer starting from the offset.
        """
        comptime assert Self.column_inner_size % Self.simd_size == 0
        assert (
            pack_tile_dim[1] % Self.column_inner_size == 0
        ), "Unimplemented tile pattern."

        var instance = Self(
            packed_matrix,
            original_matrix,
            Coord(global_offset),
            Coord(pack_tile_dim),
            Coord(valid_data_dim),
        )

        instance._pack()

    @always_inline
    def _pack_helper[
        skip_row_bound: Bool, skip_col_bound: Bool
    ](self, row_start: Int, valid_row_count: Int, col_start: Int):
        """Helper function: copy several simd vectors on the column from the
        original matrix to the packed buffer. The copies are unrolled and
        prefetched (if not with neon).
            Args:
                skip_row_bound: Boundary check on row dimension will be skipped
                    if true.
                skip_col_bound: Boundary check on column dimension will be skipped
                    if true.
        """

        comptime unroll_factor = get_packB_unroll_factor()

        @always_inline
        @__parameter
        def pack_vector(row_idx: Int, col_idx: Int):
            var global_idx = Index(
                Int(self.global_offset[0].value()) + row_idx,
                Int(self.global_offset[1].value()) + col_idx,
            )
            var valid_cols = Int(self.valid_data_dim[1].value())
            var data = SIMD[Self.dtype, Self.simd_size](0)
            if skip_col_bound or (col_idx + Self.simd_size <= valid_cols):
                data = self.original_matrix.load_linear[
                    width=Self.simd_size, alignment=1
                ](global_idx)
            elif col_idx < valid_cols:
                # Starting point within bound but cannot load a whole
                #  vector. Do a partial load.
                data = partial_simd_load[Self.simd_size](
                    self.original_matrix.ptr
                    + self.original_matrix.layout(
                        Coord(global_idx[0], global_idx[1])
                    ),
                    0,
                    valid_cols - col_idx,
                    0,
                )

            # map to packed index
            var col_idx_outer, col_idx_inner = divmod(
                col_idx, Self.column_inner_size
            )
            self.packed_matrix.store_linear[width=Self.simd_size, alignment=1](
                Index(
                    col_idx_outer,
                    row_idx,
                    col_idx_inner,
                ),
                data,
            )

        @always_inline
        @__parameter
        def pack_body[idx: Int]():
            pack_vector(row_start + idx, col_start)

        @always_inline
        @__parameter
        def prefetch_body[idx: Int]():
            var global_row_idx = (
                Int(self.global_offset[0].value())
                + row_start
                + unroll_factor
                + idx
            )
            var global_col_idx = Int(self.global_offset[1].value()) + col_start
            prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](
                self.original_matrix.ptr
                + self.original_matrix.layout(
                    Coord(global_row_idx, global_col_idx)
                )
            )

        comptime if skip_row_bound:
            if not CompilationTarget.has_neon():
                comptime for i in range(unroll_factor):
                    prefetch_body[i]()

            comptime for i in range(unroll_factor):
                pack_body[i]()
        else:
            for row_idx in range(row_start, valid_row_count):
                pack_vector(row_idx, col_start)

    def _pack_vnni(self):
        """Copy the B tile from the original matrix to the packed buffer for VNNI.
        """
        comptime assert Self.use_vnni

        comptime vnni_cols = 4

        var kc = Int(self.valid_data_dim[0].value())
        var nc = Int(self.valid_data_dim[1].value())
        var nr = Self.column_inner_size
        for i in range(0, Int(self.pack_tile_dim[0].value()), vnni_cols):
            for j in range(Int(self.pack_tile_dim[1].value()) // nr):
                for p in range(nr):
                    comptime for l in range(vnni_cols):
                        var local_idx = Index(i + l, p + nr * j)
                        var val = 0 if local_idx[0] >= kc or local_idx[
                            1
                        ] >= nc else self.original_matrix.load_linear[width=1](
                            Index(
                                Int(self.global_offset[0].value())
                                + local_idx[0],
                                Int(self.global_offset[1].value())
                                + local_idx[1],
                            )
                        )
                        self.packed_matrix.store_linear[width=1](
                            Index(
                                j,
                                i // vnni_cols,
                                vnni_cols * p + l,
                            ),
                            val,
                        )

    def _pack_i8mm(self):
        comptime i8mm_rows = 2
        comptime i8mm_cols = 8

        comptime assert Self.use_i8mm
        var kc = Int(self.valid_data_dim[0].value())
        var nc = Int(self.valid_data_dim[1].value())
        comptime nr = Self.column_inner_size // 2
        for i in range(0, Int(self.pack_tile_dim[0].value()), i8mm_cols):
            for j in range(Int(self.pack_tile_dim[1].value()) // nr):
                for p in range(0, nr, i8mm_rows):
                    for i2 in range(i8mm_cols):
                        for p2 in range(i8mm_rows):
                            var local_idx = Index(i + i2, nr * j + p + p2)
                            var val = 0 if local_idx[0] >= kc or local_idx[
                                1
                            ] >= nc else self.original_matrix.load_linear[
                                width=1
                            ](
                                Index(
                                    Int(self.global_offset[0].value())
                                    + local_idx[0],
                                    Int(self.global_offset[1].value())
                                    + local_idx[1],
                                )
                            )
                            self.packed_matrix.store_linear[width=1](
                                Index(
                                    j,
                                    i // i8mm_cols,
                                    i8mm_cols * p + i8mm_cols * p2 + i2,
                                ),
                                val,
                            )

    def _pack_default(self):
        """Copy the B tile from the original matrix to the packed buffer.
        Each iteration copies a block of shape (unroll_factor, simd_size)."""
        comptime assert not Self.use_vnni and not Self.use_i8mm
        var valid_row_count = Int(
            min(self.valid_data_dim[0].value(), self.pack_tile_dim[0].value())
        )
        comptime unroll_factor = get_packB_unroll_factor()

        var row_idx: Int = 0
        var col_idx: Int = 0

        @always_inline
        def pack_unit[
            skip_row_bound: Bool, skip_col_bound: Bool
        ]() {var valid_row_count, imm}:
            self._pack_helper[skip_row_bound, skip_col_bound](
                row_idx, valid_row_count, col_idx
            )

        var pack_tile_cols = Int(self.pack_tile_dim[1].value())
        var valid_cols = Int(self.valid_data_dim[1].value())
        while row_idx < valid_row_count:
            col_idx = 0
            while col_idx < pack_tile_cols:
                unswitch(
                    row_idx + unroll_factor < valid_row_count,
                    col_idx + Self.simd_size < valid_cols,
                    pack_unit,
                )
                col_idx += Self.simd_size
            row_idx += unroll_factor

    def _pack(self):
        comptime if Self.use_vnni:
            self._pack_vnni()
        elif Self.use_i8mm:
            self._pack_i8mm()
        else:
            self._pack_default()


@always_inline
def _pack_matmul_b_shape_func_impl[
    a_type: DType,
    c_type: DType,
    transpose_in_0: Bool,
](
    b_input: TileTensor[mut=False, address_space=.GENERIC, ...],
    kernel_type_m: Int = 0,
) -> IndexList[2]:
    """Computes the padded shape required by `pack_b` directly from TileTensor
    dimensions.

    If transpose_b is True, this returns the un-transposed shape, since pack_b
    will un-transpose `b_ref` as part of the packing layout transformation."""
    comptime assert b_input.rank == 2, "b must be rank 2"
    comptime assert b_input.flat_rank == 2, "b must have a non-nested layout"

    var output = IndexList[2]()

    var dim0 = Int(b_input.dim[0]())
    var dim1 = Int(b_input.dim[1]())
    var n = dim0 if transpose_in_0 else dim1
    var k = dim1 if transpose_in_0 else dim0
    var tile_n_k = IndexList[2]()

    @always_inline
    def dispatch_on_kernel_type[kernel_type: Bool]() {mut tile_n_k, b_input}:
        comptime config = get_kernel_config[
            a_type,
            b_input.dtype,
            c_type,
            kernel_type=kernel_type,
        ]()
        tile_n_k = _get_tile_n_k[
            a_type, b_input.dtype, c_type, config.kernel_cols, transpose_in_0
        ](b_input)

    dispatch_get_kernel_type(dispatch_on_kernel_type, kernel_type_m, n, k)

    comptime if transpose_in_0:
        output[0] = dim1
        output[1] = dim0
    else:
        output[0] = dim0
        output[1] = dim1

    comptime if not use_apple_accelerate_lib[c_type, a_type, b_input.dtype]():
        comptime use_vnni = use_vnni_fn[a_type, b_input.dtype, c_type]()
        comptime use_i8mm = use_i8mm_fn[a_type, b_input.dtype, c_type]()
        comptime factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
        output[0] = align_up(output[0], factor)
        output[1] = align_up(output[1], tile_n_k[0])
    else:
        var tmp = output[0]
        output[0] = output[1]
        output[1] = tmp

    return output


def pack_b[
    transpose_b: Bool,
    simd_size: Int,
    inner_size: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
](
    dst: TileTensor[mut=True, b_type, address_space=.GENERIC, ...],
    src: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    tile_n: Int,
    tile_k: Int,
):
    """Utility function to pack the entire B matrix, such that each
    [tile_n // inner_size, tile_k, inner_size] tile of src is contiguous in dst.

    Tiles (not tile contents) are stored in rowmajor order, so tile[i, j] is
    tile_n * tile_k bytes away from tile[i, j+1].

    Parameters:
        transpose_b: True if the B operand is transposed, stored as
            `[N, K]` instead of `[K, N]`.
        simd_size: SIMD vector width for `b_type`.
        inner_size: Size of the inner dimension along N in the packed
            tile; must be a multiple of `simd_size`.
        a_type: Element type of the A operand of the matmul.
        b_type: Element type of the B operand being packed.
        c_type: Element type of the C output of the matmul.

    Args:
        dst: Pre-allocated mutable buffer that receives the packed B
            matrix.
        src: Read-only buffer containing the original B matrix to pack.
        tile_n: Tile size along the N dimension of the matmul.
        tile_k: Tile size along the K dimension of the matmul.
    """
    # Strip extra type params from existential `...` pattern.
    var src_tt = TileTensor(src.ptr, src.layout)
    var dst_tt = TileTensor(dst.ptr, dst.layout)
    unsafe_memset_zero(
        dst_tt.ptr, dst_tt.num_elements()
    )  # zero the padding to be safe
    var dst_flat_ptr = dst_tt.ptr
    var dst_offset: Int = 0

    comptime use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    comptime use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
    comptime factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
    comptime inner_size2 = inner_size // 2 if use_i8mm else inner_size

    comptime if not transpose_b:
        var k_in = Int(src_tt.dim[0]())
        var n_in = Int(src_tt.dim[1]())
        var k_out = Int(dst_tt.dim[0]())
        var n_out = Int(dst_tt.dim[1]())

        assert (
            n_out % tile_n == 0
        ), "N dimension of output must be padded to tile_n"

        for idx_k in range(0, k_out, tile_k):
            var tile_k2 = align_up(min(tile_k, k_out - idx_k), factor)

            for idx_n in range(0, n_out, tile_n):
                var packed_dst_view = TileTensor(
                    dst_flat_ptr + dst_offset,
                    row_major(
                        Coord(
                            tile_n // inner_size2,
                            tile_k2 // factor,
                            inner_size2 * factor,
                        )
                    ),
                )
                var valid_k = min(tile_k2, k_in - idx_k)
                var valid_n = min(tile_n, n_in - idx_n)
                PackMatrixCols[
                    b_type,
                    simd_size,
                    inner_size,
                    use_vnni,
                    use_i8mm,
                    packed_dst_view.origin,
                    src_tt.origin,
                ].run(
                    packed_dst_view,
                    src_tt,
                    # Input is [K, N]:
                    # Starting global offset for packing.
                    Index(idx_k, idx_n),
                    Index(tile_k2, tile_n),
                    # Valid amount of input from the starting offset.
                    Index(valid_k, valid_n),
                )
                dst_offset += tile_n * tile_k2
    else:
        # _t = transpose, annoying WAR since variables can't have same name in if/else
        var k_in_t = Int(src_tt.dim[1]())
        var n_in_t = Int(src_tt.dim[0]())
        var k_out_t = Int(dst_tt.dim[0]())
        var n_out_t = Int(dst_tt.dim[1]())

        assert (
            n_out_t % tile_n == 0
        ), "N dimension of output must be padded to tile_n"

        for idx_k_t in range(0, k_out_t, tile_k):
            for idx_n_t in range(0, n_out_t, tile_n):
                var packed_dst_view_t = TileTensor(
                    dst_flat_ptr + dst_offset,
                    row_major(
                        Coord(
                            tile_n // inner_size,
                            tile_k,
                            inner_size,
                        )
                    ),
                )
                var valid_k_t = min(tile_k, k_in_t - idx_k_t)
                var valid_n_t = min(tile_n, n_in_t - idx_n_t)
                PackMatrixRows[
                    b_type,
                    simd_size,
                    inner_size,
                ].run(
                    packed_dst_view_t,
                    src_tt,
                    # Input is [N, K]:
                    # Starting global offset for packing.
                    Index(idx_n_t, idx_k_t),
                    Index(tile_n, tile_k),
                    # Valid amount of input from the starting offset.
                    Index(valid_n_t, valid_k_t),
                )
                dst_offset += tile_n * tile_k


@always_inline
def _pack_b_ndbuffer_impl[
    b_type: DType,
    //,
    a_type: DType,
    c_type: DType,
    transposed: Bool,
](
    b_input: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    output_buffer: TileTensor[mut=True, b_type, address_space=.GENERIC, ...],
    kernel_type_m: Int,
) raises:
    """TileTensor implementation of `_pack_b_ndbuffer_impl`.

    Performs the layout transformation on `b_input` expected by
    `matmul_dynamic_tile` when `b_packed` is True and stores the result in
    `output_buffer`.

    When transpose_b is True, this also un-transposes b_input as part of the
    layout transformation."""
    comptime assert b_input.rank == 2, "b must be rank 2"
    comptime assert b_input.flat_rank == 2, "b must have a non-nested layout"
    comptime assert output_buffer.rank == 2, "output must be rank 2"
    comptime assert (
        output_buffer.flat_rank == 2
    ), "output must have a non-nested layout"

    var dim0 = Int(b_input.dim[0]())
    var dim1 = Int(b_input.dim[1]())

    # Matrix by vector pattern -> use gemv
    if dim1 == 1:
        # For gemv no packing is necessary
        unsafe_memcpy(dest=output_buffer.ptr, src=b_input.ptr, count=dim0)

    else:
        var n = dim0 if transposed else dim1
        var k = dim1 if transposed else dim0

        comptime if use_apple_accelerate_lib[c_type, a_type, b_type]():
            comptime if not transposed:
                var perm_ptr = unsafe_stack_allocation[2, Int]()
                perm_ptr[0] = 1
                perm_ptr[1] = 0

                transpose(output_buffer, b_input, perm_ptr)

            else:
                unsafe_memcpy(
                    dest=output_buffer.ptr, src=b_input.ptr, count=n * k
                )
            return

        @always_inline
        def dispatch_on_kernel_type[
            kernel_type: Bool
        ]() {output_buffer, b_input}:
            comptime config = get_kernel_config[
                a_type,
                b_type,
                c_type,
                kernel_type=kernel_type,
            ]()
            var tile_n_k = _get_tile_n_k[
                a_type, b_type, c_type, config.kernel_cols, transposed
            ](b_input)
            pack_b[
                transposed,
                config.simd_size,
                config.kernel_cols,
                a_type,
                b_type,
                c_type,
            ](output_buffer, b_input, tile_n_k[0], tile_n_k[1])

        dispatch_get_kernel_type(dispatch_on_kernel_type, kernel_type_m, n, k)


@always_inline
def pack_matmul_b_shape_func[
    a_type: DType,
    c_type: DType,
    transpose_in_0: Bool,
](
    b_input: TileTensor[mut=False, address_space=.GENERIC, ...],
    kernel_type_m: Int = 0,
) -> IndexList[2]:
    """TileTensor primary implementation of `pack_matmul_b_shape_func`.

    Takes `kernel_type_m` directly instead of extracting it from `a_shape`
    static shape params (0 = dynamic M).

    Parameters:
        a_type: Element type of the A operand of the matmul.
        c_type: Element type of the C output of the matmul.
        transpose_in_0: True if the B operand is transposed, stored as
            `[N, K]` instead of `[K, N]`.

    Args:
        b_input: Read-only rank-2 `TileTensor` containing the B matrix
            whose padded shape is computed.
        kernel_type_m: M dimension used to select the matmul kernel variant
            (defaults to 0, meaning dynamic M).
    """
    return _pack_matmul_b_shape_func_impl[a_type, c_type, transpose_in_0](
        b_input, kernel_type_m
    )


@always_inline
def pack_b_ndbuffer[
    b_type: DType,
    //,
    a_type: DType,
    c_type: DType,
](
    b_input: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    output_buffer: TileTensor[mut=True, b_type, address_space=.GENERIC, ...],
    kernel_type_m: Int = 0,
) raises:
    """TileTensor primary implementation of `pack_b_ndbuffer`.

    Takes `kernel_type_m` directly instead of extracting it from `a_shape`
    static shape params (0 = dynamic M).

    Parameters:
        b_type: Element type of the B operand being packed (inferred).
        a_type: Element type of the A operand of the matmul.
        c_type: Element type of the C output of the matmul.

    Args:
        b_input: Read-only rank-2 `TileTensor` containing the original B
            matrix to pack.
        output_buffer: Pre-allocated mutable rank-2 `TileTensor` that
            receives the packed B matrix.
        kernel_type_m: M dimension used to select the matmul kernel variant
            (defaults to 0, meaning dynamic M).
    """
    _pack_b_ndbuffer_impl[a_type, c_type, transposed=False](
        b_input, output_buffer, kernel_type_m
    )


@always_inline
def pack_transposed_b_ndbuffer[
    b_type: DType,
    //,
    a_type: DType,
    c_type: DType,
](
    b_input: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    output_buffer: TileTensor[mut=True, b_type, address_space=.GENERIC, ...],
    kernel_type_m: Int = 0,
) raises:
    """TileTensor primary implementation of `pack_transposed_b_ndbuffer`.

    Takes `kernel_type_m` directly instead of extracting it from `a_shape`
    static shape params (0 = dynamic M).

    Parameters:
        b_type: Element type of the B operand being packed (inferred).
        a_type: Element type of the A operand of the matmul.
        c_type: Element type of the C output of the matmul.

    Args:
        b_input: Read-only rank-2 `TileTensor` containing the original B
            matrix to pack, stored as `[N, K]`.
        output_buffer: Pre-allocated mutable rank-2 `TileTensor` that
            receives the packed B matrix.
        kernel_type_m: M dimension used to select the matmul kernel variant
            (defaults to 0, meaning dynamic M).
    """
    _pack_b_ndbuffer_impl[a_type, c_type, transposed=True](
        b_input, output_buffer, kernel_type_m
    )


@fieldwise_init
struct BTileGenerator[
    config: KernelConfig,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_layout: TensorLayout,
    transpose_b: Bool,
    b_packed: Bool,
    origin: ImmOrigin,
](ImplicitlyCopyable):
    """Struct to encapsulate a tile of B that supports prepacking.

    If b_packed is true, calls to get_tile will return a buffer view from B.
    Otherwise, calls to get_tile will copy a tile from B into a stack allocated
    scratch buffer and return a view of that.

    Parameters:
        config: Kernel configuration supplying `simd_size` and `kernel_cols`
            used by the packing routines.
        a_type: Element type of the A operand of the matmul.
        b_type: Element type of the B operand being packed.
        c_type: Element type of the C output of the matmul.
        b_layout: Layout of the B `TileTensor`.
        transpose_b: True if the B operand is transposed, stored as [N, K].
        b_packed: True if B is already pre-packed into the mlas layout.
        origin: Origin of the B `TileTensor`.
    """

    var b: TileTensor[
        Self.b_type, Self.b_layout, Self.origin
    ]  # packed layout if b_packed is True
    var b_tile_stack_ptr: UnsafePointer[Scalar[Self.b_type], MutUntrackedOrigin]
    var tile_n_k: DynamicCoord[.int64, 2]

    # needs to be always_inline so b_tile_stack_ptr gets allocated on caller's stack
    @always_inline
    @staticmethod
    def get(
        b: TileTensor[Self.b_type, Self.b_layout, Self.origin],
        tile_n_k: IndexList[2],
    ) -> BTileGenerator[
        Self.config,
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.b_layout,
        Self.transpose_b,
        Self.b_packed,
        Self.origin,
    ]:
        var b_tile_stack_ptr = UnsafePointer[
            Scalar[Self.b_type], MutUntrackedOrigin
        ].unsafe_dangling()

        assert not (
            Self.transpose_b and Self.b_packed
        ), "b cannot be both transposed and pre-packed."

        comptime if not Self.b_packed:
            b_tile_stack_ptr = unsafe_stack_allocation[
                get_pack_data_size[Self.b_type](),
                Self.b_type,
                align_of[SIMD[Self.b_type, simd_width_of[Self.b_type]()]](),
            ]()

        return BTileGenerator[
            Self.config,
            Self.a_type,
            Self.b_type,
            Self.c_type,
            Self.b_layout,
            Self.transpose_b,
            Self.b_packed,
        ](b, b_tile_stack_ptr, Coord(tile_n_k))

    comptime PackedTileLayout = RowMajorLayout[Int, Int, Int]

    def get_tile[
        inner_size: Int
    ](
        self,
        global_offset: GemmShape,
        tile_dim_nk: IndexList[2],
        valid_data_dim_nk: IndexList[2],
    ) -> TileTensor[Self.b_type, Self.PackedTileLayout, ImmutAnyOrigin]:
        """Get a packed matrix (B) tile.

        Parameters:
            inner_size: Size of the inner dimension along N in the packed
                tile; must be a multiple of `simd_size`.

        Args:
            global_offset: Offset in the global M, N, K dimensions.
            tile_dim_nk: Tile shape based on cache size and matrix dimensions.
            valid_data_dim_nk: The upper bounds for N and K dimensions.

        valid_data_tile_nk is ignored for pre-packing, where the tile is padded
        to have shape of tile_dim_nk.

        Returns:
            A view of the packed tile.

        """
        comptime use_vnni = use_vnni_fn[Self.a_type, Self.b_type, Self.c_type]()
        comptime use_i8mm = use_i8mm_fn[Self.a_type, Self.b_type, Self.c_type]()

        comptime factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
        comptime inner_size2 = inner_size // 2 if use_i8mm else inner_size

        var k = align_up(tile_dim_nk[1], factor)
        var tile_shape_nopack = IndexList[3](
            tile_dim_nk[0] // inner_size2,
            k // factor,
            factor * inner_size2,
        )

        var packed_b = TileTensor(
            self.b_tile_stack_ptr,
            row_major(
                Coord(
                    Int(tile_shape_nopack[0]),
                    Int(tile_shape_nopack[1]),
                    Int(tile_shape_nopack[2]),
                )
            ),
        )

        comptime if Self.transpose_b and not Self.b_packed:
            PackMatrixRows[
                Self.b_type,
                Self.config.simd_size,
                inner_size,
            ].run(
                packed_b,
                self.b,
                # Input is [N, K]:
                # Starting global offset for packing.
                Index(global_offset.N, global_offset.K),
                Index(tile_dim_nk[0], tile_dim_nk[1]),
                # Valid amount of input from the starting offset.
                Index(valid_data_dim_nk[0], valid_data_dim_nk[1]),
            )
            return packed_b.as_immut().as_unsafe_any_origin()
        elif (not Self.transpose_b) and (not Self.b_packed):
            PackMatrixCols[
                Self.b_type,
                Self.config.simd_size,
                inner_size,
                use_vnni,
                use_i8mm,
                packed_b.origin,
                Self.origin,
            ].run(
                packed_b,
                self.b,
                # Input is [K, N]:
                # Starting global offset for packing.
                Index(global_offset.K, global_offset.N),
                Index(tile_dim_nk[1], tile_dim_nk[0]),
                # Valid amount of input from the starting offset.
                Index(valid_data_dim_nk[1], valid_data_dim_nk[0]),
            )
        elif Self.b_packed and not Self.transpose_b:
            # Need to use tile_k that generator was initialized with.
            # When packing is done online, tile_dim_nk can vary in each call to
            # get_tile (if handling a residual K tile), but packing assumes that
            # tile_k is constant.

            var factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
            comptime inner_size2 = inner_size // 2 if use_i8mm else inner_size

            var tile_k = Int(self.tile_n_k[1].value())
            var tile_k2 = align_up(min(tile_k, valid_data_dim_nk[1]), factor)

            var tile_shape_pack = IndexList[3](
                Int(self.tile_n_k[0].value()) // inner_size2,
                tile_k2 // factor,
                inner_size2 * factor,
            )
            var tile_k_idx = global_offset.K // tile_k
            var n_padded = Int(self.b.dim[1]())
            var b_tile_view = TileTensor(
                # tiles are ordered in row-major order
                # a bit of trickieness going on here, this works because:
                #   1. tile_k is the same for every thread (tile_n is not) since threads
                #       don't currently partition on the K dimension
                #   2. the n dimension of each thread's tile is guaranteed to be an
                #       exact multiple of the inner size
                #   3. each tile has dims [tile_n/inner, tile_k, inner]
                self.b.ptr
                + (tile_k_idx * tile_k * n_padded + global_offset.N * tile_k2),
                row_major(
                    Coord(
                        Int(tile_shape_pack[0]),
                        Int(tile_shape_pack[1]),
                        Int(tile_shape_pack[2]),
                    )
                ),
            )
            return b_tile_view.as_unsafe_any_origin()

        else:
            assert False, "unreachable, b_packed not supported with transpose_b"

        return packed_b.as_immut().as_unsafe_any_origin()
