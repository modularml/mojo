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

"""Provides register-file accumulator structures used by the matmul inner loop."""

from std.collections.optional import Optional
from layout import TileTensor
from std.math import fma
from std.sys import align_of, prefetch
from std.sys.info import CompilationTarget
from std.sys.intrinsics import PrefetchOptions

from std.algorithm.functional import tile
from linalg.utils import (
    partial_simd_load,
    partial_simd_store,
)
from std.memory import unsafe_stack_allocation

from std.utils.index import IndexList


# ===-----------------------------------------------------------------------===#
# Helper Functions
# ===-----------------------------------------------------------------------===#
# TODO: rename to _MatmulAccumulators?
struct _Accumulator[
    dtype: DType,
    num_rows: Int,
    num_cols: Int,
    simd_width: Int,
    row_start: Int = 0,
    row_stop: Int = num_rows,
](Defaultable):
    """
    Parameters:
        dtype: DType of accumulator.
        num_rows: Number of rows in register tiling.
        num_cols: Number of columns in register tiling.
        simd_width: Number of lanes of a SIMD vector.
    """

    comptime tile_columns = Self.num_cols * Self.simd_width
    comptime _size = Self.num_rows * Self.num_cols * Self.simd_width

    # The output buffer, should have num_rows x num_cols x simd_width.
    var _storage: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]

    @always_inline
    def __init__(out self):
        comptime assert (
            (Self.num_cols > 0)
            and (Self.num_rows > 0)
            and (Self.simd_width > 0)
        )
        comptime alignment = align_of[SIMD[Self.dtype, Self.simd_width]]()
        self._storage = unsafe_stack_allocation[
            Self._size, Self.dtype, alignment=alignment
        ]()

    @always_inline
    def __init__(
        out self,
        other_storage: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
    ):
        comptime assert (
            (Self.num_cols > 0)
            and (Self.num_rows > 0)
            and (Self.simd_width > 0)
        )
        self._storage = other_storage.unsafe_origin_cast[MutUntrackedOrigin]()

    # NOTE: This is NOT a deepcopy; self uses the same _storage as copy.
    @always_inline
    def __init__(out self, *, copy: Self):
        comptime assert (
            (Self.num_cols > 0)
            and (Self.num_rows > 0)
            and (Self.simd_width > 0)
        )
        self._storage = copy._storage

    @staticmethod
    @always_inline
    def _storage_index(m: Int, n: Int) -> Int:
        return (m * Self.num_cols + n) * Self.simd_width

    @always_inline
    def __getitem__(self, m: Int, n: Int) -> SIMD[Self.dtype, Self.simd_width]:
        return self._storage.load[width=Self.simd_width](
            self._storage_index(m, n)
        )

    @always_inline
    def __setitem__(
        mut self, m: Int, n: Int, value: SIMD[Self.dtype, Self.simd_width]
    ):
        self._storage.store(self._storage_index(m, n), value)

    @always_inline
    def _partial_set[
        partial_width: SIMDLength
    ](mut self, offset: Int, value: SIMD[Self.dtype, partial_width]):
        self._storage.store[width=partial_width](offset, value)

    @always_inline
    def _partial_get[
        partial_width: Int
    ](mut self, idx: Int) -> SIMD[Self.dtype, partial_width]:
        return self._storage.load[width=partial_width](idx)

    # In c+=(a*b), each of a, b, and c can have different dtypes.
    @always_inline
    def fma[
        a_dtype: DType, b_dtype: DType
    ](
        mut self,
        m: Int,
        n: Int,
        a: SIMD[a_dtype, Self.simd_width],
        b: SIMD[b_dtype, Self.simd_width],
    ):
        # TODO: the order of 'a' and 'b' in the following FMA and its impact on accuracy.
        self[m, n] = (b.cast[Self.dtype]()).fma(
            (a.cast[Self.dtype]()), self[m, n]
        )

    @always_inline
    @staticmethod
    def _transfer[
        FuncType: ImplicitlyCopyable
        & def(
            # TODO: Ideally `ptr` should have same origin as `base_ptr`, but I cannot
            # get it to compile successfully.
            m: Int,
            n: Int,
            ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin],
        ) -> None,
    ](
        func: FuncType,
        base_ptr: UnsafePointer[Scalar[Self.dtype], _],
        stride: Int,
    ):
        var row_ptr = base_ptr

        comptime for m in range(Self.num_rows):
            comptime for n in range(Self.num_cols):
                func(
                    m,
                    n,
                    (row_ptr + n * Self.simd_width)
                    .unsafe_mut_cast[True]()
                    .as_unsafe_any_origin(),
                )
            row_ptr += stride

    # TODO: merge with load
    @always_inline
    def load(
        mut self,
        base_ptr: UnsafePointer[mut=False, Scalar[Self.dtype], _],
        stride: Int,
    ):
        @always_inline
        def do_transfer(
            m: Int, n: Int, ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
        ) {mut self}:
            # TODO: Ideally `ptr` should be immutable, but origins aren't inferring correctly.
            self[m, n] = ptr.load[width=Self.simd_width]()

        Self._transfer(do_transfer, base_ptr, stride)

    @always_inline
    def load(
        mut self,
        c_ptr: UnsafePointer[mut=False, Scalar[Self.dtype], _],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
        skip_boundary_check: Bool = False,
    ):
        self._transfer[True](
            c_ptr, c_stride, tile_n_idx, c_bound, skip_boundary_check
        )

    @always_inline
    def store(
        mut self,
        c_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
        skip_boundary_check: Bool = False,
    ):
        self._transfer[False](
            c_ptr, c_stride, tile_n_idx, c_bound, skip_boundary_check
        )

    @always_inline
    def _transfer[
        is_load: Bool
    ](
        mut self,
        c_ptr: UnsafePointer[Scalar[Self.dtype], _],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
        skip_boundary_check: Bool,
    ):
        comptime assert is_load or c_ptr.mut, "ahhh"
        var c_ptr_loc = c_ptr + tile_n_idx

        if skip_boundary_check:
            comptime if is_load:
                self.load(c_ptr_loc, c_stride)
            else:
                self.store(c_ptr_loc.unsafe_mut_cast[True](), c_stride)
        else:
            var transfer_count = min(
                c_bound[1] - tile_n_idx, Self.num_cols * Self.simd_width
            )
            var row_ptrs = Array[_, Self.num_rows](
                fill_with=lambda (row: Int) -> UnsafePointer[
                    Scalar[Self.dtype], AnyOrigin[mut=c_ptr.mut]
                ]: (c_ptr_loc + row * c_stride).as_unsafe_any_origin()
            )

            self._transfer_loop[0, is_load](
                transfer_count, row_ptrs.unsafe_ptr(), c_stride
            )

    @always_inline
    def _transfer_columns[
        origin: Origin,
        //,
        base_column: Int,
        column_count: Int,
        is_load: Bool,
    ](
        mut self,
        row_ptrs: UnsafePointer[UnsafePointer[Scalar[Self.dtype], origin], _],
        stride: Int,
    ):
        """Loads or stores one or more columns from the base column for each
        row of the tile."""
        comptime assert is_load or origin.mut, "ahhh"
        comptime column_step = min(column_count, Self.simd_width)

        @__parameter
        @always_inline
        def body(row: Int, col: Int):
            comptime if is_load:
                comptime if CompilationTarget.has_neon():
                    var data = row_ptrs[row].load[width=column_step](col)
                    self._partial_set(row * Self.tile_columns + col, data)
                else:
                    var data = row_ptrs[0].load[width=column_step](
                        stride * row + col
                    )
                    self._partial_set(row * Self.tile_columns + col, data)
            else:
                var data = self._partial_get[column_step](
                    row * Self.tile_columns + col
                )

                comptime if CompilationTarget.has_neon():
                    row_ptrs[row].unsafe_mut_cast[True]().store(col, data)
                else:
                    row_ptrs[0].unsafe_mut_cast[True]().store(
                        stride * row + col, data
                    )

        comptime for row in range(Self.num_rows):
            # Iterate twice for a pairwise load/store or once for any other access.

            comptime for col in range(
                base_column, base_column + column_count, column_step
            ):
                body(row, col)

    @always_inline
    def _transfer_loop[
        origin: Origin, //, base_column: Int, is_load: Bool
    ](
        mut self,
        transfer_count: Int,
        row_ptrs: UnsafePointer[UnsafePointer[Scalar[Self.dtype], origin], _],
        stride: Int,
    ):
        """Loads/stores all pairwise vectors of the tile and dispatches the
        remaining non-pairwise elements."""
        comptime assert is_load or origin.mut, "ahhh"
        comptime tile_columns_remaining = Self.tile_columns - base_column
        # Support fusion of LDP/STP instructions by emitting pairs of load/store with neon
        comptime column_groups = 2 if CompilationTarget.has_neon() else 1

        # vector instructions.
        comptime if tile_columns_remaining >= column_groups * Self.simd_width:
            if transfer_count >= base_column + column_groups * Self.simd_width:
                self._transfer_columns[
                    base_column, column_groups * Self.simd_width, is_load
                ](row_ptrs, stride)
                self._transfer_loop[
                    base_column + column_groups * Self.simd_width, is_load
                ](transfer_count, row_ptrs, stride)
                return

        comptime if tile_columns_remaining >= Self.simd_width:
            comptime if CompilationTarget.has_neon():
                self._transfer_tail[base_column, Self.simd_width, is_load](
                    transfer_count, row_ptrs, stride
                )
            else:
                self._transfer_tail_mask[base_column, is_load](
                    transfer_count, row_ptrs, stride
                )

    @always_inline
    def _transfer_tail[
        origin: Origin, //, base_column: Int, tail_size: Int, is_load: Bool
    ](
        mut self,
        transfer_count: Int,
        row_ptrs: UnsafePointer[UnsafePointer[Scalar[Self.dtype], origin], _],
        stride: Int,
    ):
        """Loads/stores the last elements of the tile that cannot be accessed
        pairwise."""
        comptime assert is_load or origin.mut, "ahhh"

        if transfer_count & tail_size:
            self._transfer_columns[base_column, tail_size, is_load](
                row_ptrs, stride
            )
            comptime tile_columns_remaining = Self.tile_columns - base_column - tail_size

            comptime if tile_columns_remaining >= tail_size // 2 and tail_size > 1:
                self._transfer_tail[
                    base_column + tail_size, tail_size // 2, is_load
                ](transfer_count, row_ptrs, stride)
            return

        comptime if tail_size > 1:
            self._transfer_tail[base_column, tail_size // 2, is_load](
                transfer_count, row_ptrs, stride
            )

    @always_inline
    def _transfer_tail_mask[
        origin: Origin, //, base_column: Int, is_load: Bool
    ](
        mut self,
        transfer_count: Int,
        row_ptrs: UnsafePointer[UnsafePointer[Scalar[Self.dtype], origin], _],
        stride: Int,
    ):
        comptime assert is_load or origin.mut, "ahhh"

        var tail_size = transfer_count - base_column

        comptime for row in range(Self.num_rows):
            comptime col = base_column // Self.simd_width

            comptime if is_load:
                self[row, col] = partial_simd_load[Self.simd_width](
                    row_ptrs[0] + (stride * row + base_column),
                    0,
                    tail_size,
                    0,
                )
            else:
                partial_simd_store(
                    (
                        row_ptrs[0] + (stride * row + base_column)
                    ).unsafe_mut_cast[True](),
                    0,
                    tail_size,
                    self[row, col],
                )

    # TODO: merge with store
    @always_inline
    def store(
        mut self,
        base_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], _],
        stride: Int,
    ):
        @always_inline
        def do_transfer(
            m: Int, n: Int, ptr: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]
        ) {mut self}:
            ptr.store(self[m, n])

        Self._transfer(do_transfer, base_ptr, stride)

    # ===-------------------------------------------------------------------===#
    # Init/Load/Store register tiles
    # ===-------------------------------------------------------------------===#

    @always_inline
    def init(mut self):
        comptime if Self.dtype.is_floating_point():
            self.init(0.0)
        else:
            self.init(0)

    @always_inline
    def init(mut self, val: Scalar[Self.dtype]):
        # TODO: refactor with _transfer
        comptime for m in range(Self.num_rows):
            comptime for n in range(Self.num_cols):
                self[m, n] = val

    @always_inline
    def load[
        dt: DType,
        //,
        partial_load: Bool = False,
    ](
        mut self,
        input: UnsafePointer[mut=False, Scalar[dt], ...],
        input_stride: Int,
        partial_load_size: Optional[Int] = None,
    ):
        """Load a register tile from the input buffer.

        Parameters:
            dt: DType of the input.
            partial_load: Whether load input partially.

        Args:
            input: Pointer to input buffer.
            input_stride: Stride between input segments of size `num_cols * simd_width`.
            partial_load_size: Size of partial load for input.
        """

        # TODO: could we lift partial_load_size out of the loop?
        comptime for i in range(Self.num_rows):
            comptime for j in range(Self.num_cols):
                var input_ptr = input + i * input_stride + j * Self.simd_width
                comptime partial_load_last_vec = partial_load and (
                    j == Self.num_cols - 1
                )

                # TODO: check if partial_load_size has value.
                self[i, j] = _simd_load_maybe_partial[
                    Self.simd_width, partial_load_last_vec
                ](input_ptr, 0, partial_load_size).cast[Self.dtype]()

    @always_inline
    def store[
        dt: DType,
        //,
        partial_store: Bool = False,
    ](
        mut self,
        output: UnsafePointer[mut=True, Scalar[dt], ...],
        output_stride: Int,
        partial_store_size: Optional[Int] = None,
    ):
        """Load a register tile from the input buffer.

        Parameters:
            dt: DType of the output.
            partial_store: Whether store output partially.

        Args:
            output: Pointer to output buffer.
            output_stride: Stride between output segments of size `num_cols * simd_width`.
            partial_store_size: Size of partial store to the output.
        """

        # TODO: could we lift partial_store_size out of the loop?
        comptime for i in range(Self.num_rows):
            comptime for j in range(Self.num_cols):
                comptime partial_store_last_vec = partial_store and (
                    j == Self.num_cols - 1
                )
                _simd_store_maybe_partial[
                    Self.simd_width, partial_store_last_vec
                ](
                    output,
                    i * output_stride + j * Self.simd_width,
                    self[i, j].cast[dt](),
                    partial_store_size,
                )

    # ===-------------------------------------------------------------------===#
    # Accumulation entry point.
    # ===-------------------------------------------------------------------===#
    @always_inline
    def accumulate[
        a_type: DType,
        b_type: DType,
        //,
        prefetch_offset: Optional[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[mut=False, Scalar[a_type], ...],
        a_stride: Int,
        b: UnsafePointer[mut=False, Scalar[b_type], ...],
        b_stride: Int,
        partial_load_b_size: Optional[Int] = None,
    ):
        """Compute c += a * b with register tiling on SIMD ISAs.

        Parameters:
            a_type: DType of the a.
            b_type: DType of the b.
            prefetch_offset: The distance to  prefetch ahead.
            partial_load_b: Whether use partial load for B.
        Args:
            length: Number of elements in accumulation.
            a: The input buffer A.
            a_stride: A's stride between each `length` segment.
            b: The input buffer B.
            b_stride: B's stride between each `num_cols x simd_width` segment.
            partial_load_b_size: The partial load B size.
        """

        comptime if CompilationTarget.has_neon():
            self._accumulate_neon[
                prefetch_offset=None,
                partial_load_b=partial_load_b,
            ](
                length,
                a,
                a_stride,
                b,
                b_stride,
                partial_load_b_size,
            )
        else:
            self._accumulate_x86_simd[
                prefetch_offset=prefetch_offset,
                partial_load_b=partial_load_b,
            ](
                length,
                a,
                a_stride,
                b,
                b_stride,
                partial_load_b_size,
            )

    @always_inline
    def accumulate[
        a_type: DType,
        b_type: DType,
        //,
        # TODO: move the following params to accumulate function.
        prefetch_offset: Optional[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[mut=False, Scalar[a_type], ...],
        a_base_offsets: TileTensor[mut=False, .int32, ...],
        a_offset: Int,
        b: UnsafePointer[mut=False, Scalar[b_type], ...],
        b_stride: Int,
        partial_load_b_size: Optional[Int] = None,
    ):
        """Compute c += a * b with register tiling on SIMD ISAs.

        This version applies to the cases where the rows in A are not separated
        evenly by a single stride. E.x. pointwise conv with stride > 1.

        Parameters:
            a_type: DType of the a.
            b_type: DType of the b.
            prefetch_offset: The distance to  prefetch ahead.
            partial_load_b: Whether use partial load for B.

        Args:
            length: Number of elements in accumulation.
            a: The input buffer A.
            a_base_offsets: Base offsets of rows in A.
            a_offset: Offset into A rows.
            b: The input buffer B.
            b_stride: B's stride between each `num_cols x simd_width` segment.
            partial_load_b_size: The partial load B size.


        The A offsets work as follow:

            a_base_offsets[0]: ------------------------------
            a_base_offsets[1]: ------------------------------
            ...
            a_base_offsets[2]: ------------------------------
            ...
            ...
            a_base_offsets[3]: ------------------------------
                                    ^                    ^
                                a_offset        a_offset + length
        """
        comptime assert (
            a_base_offsets.flat_rank == 1
        ), "a_base_offsets must be rank 1"

        comptime if CompilationTarget.has_neon():
            self._accumulate_neon[
                prefetch_offset=None,
                partial_load_b=partial_load_b,
            ](
                length,
                a,
                a_base_offsets,
                a_offset,
                b,
                b_stride,
                partial_load_b_size,
            )
        else:
            self._accumulate_x86_simd[
                prefetch_offset=prefetch_offset,
                partial_load_b=partial_load_b,
            ](
                length,
                a,
                a_base_offsets,
                a_offset,
                b,
                b_stride,
                partial_load_b_size,
            )

    # ===-------------------------------------------------------------------===#
    # Accumulation optimized for AVX2 and AVX512
    # ===-------------------------------------------------------------------===#

    # An example of accumulation with register tiling.
    #
    # B vector 0-3 -->         reg1     reg2     reg3     reg4
    #                       |========|========|========|========|
    # A point  0   ==> reg0 |  reg5  |  reg6  |  reg7  |  reg8  |
    #                       |--------|--------|--------|--------|
    # A point  1   ==> reg0 |  reg9  |  reg10 |  reg11 |  reg12 |
    #                       |--------|--------|--------|--------|
    # A point  2   ==> reg0 |  reg13 |  reg14 |  reg15 |  reg16 |
    #                       |--------|--------|--------|--------|
    # A point  3   ==> reg0 |  reg17 |  reg18 |  reg19 |  reg20 |
    #                       |--------|--------|--------|--------|
    # A point  4   ==> reg0 |  reg21 |  reg22 |  reg23 |  reg24 |
    #                       |--------|--------|--------|--------|
    # A point  5   ==> reg0 |  reg25 |  reg26 |  reg27 |  reg28 |
    #                       |--------|--------|--------|--------|
    #
    #    ==>      :         Broadcast a scalar into a SIMD register.
    #    -->      :         Load a SIMD vector from memory.
    # simd_width  :         |--------|
    # kernel_width:         |-----------------------------------|
    #
    # The accumulation proceeds as:
    #   for l in range(length):
    #       reg5 += reg0 * reg1
    #       reg6 += reg0 * reg2
    #       ...
    #
    # Note that we can reuse reg0 for different A points because of hardware's
    # register renaming.

    @always_inline
    def _accumulate_x86_simd[
        a_type: DType,
        b_type: DType,
        //,
        prefetch_offset: Optional[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[mut=False, Scalar[a_type], ...],
        a_stride: Int,
        b: UnsafePointer[mut=False, Scalar[b_type], ...],
        b_stride: Int,
        partial_load_b_size: Optional[Int] = None,
    ):
        """Accumulation optimized for AVX512 and AVX2."""

        comptime assert not CompilationTarget.has_neon()

        comptime kernel_width = Self.num_cols * Self.simd_width
        var b_ptr = b

        for l in range(length):
            # prefetch
            comptime if prefetch_offset:
                comptime for j in range(Self.num_cols):
                    prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ](
                        b_ptr
                        + prefetch_offset.value() * kernel_width
                        + j * Self.simd_width
                    )

            comptime for i in range(Self.row_start, Self.row_stop):
                # Broadcast an scalar from A to a simd vector.
                var a_splat_vec = SIMD[a_type, Self.simd_width](
                    a[l + i * a_stride]
                )

                comptime for j in range(Self.num_cols):
                    # Load a simd vector from B.
                    var b_vec = _simd_load_maybe_partial[
                        Self.simd_width, partial_load_b
                    ](b_ptr, j * Self.simd_width, partial_load_b_size)

                    # The following should be lifted to registers and show up as
                    # FMA instructions.
                    self[i, j] = fma(
                        a_splat_vec.cast[Self.dtype](),
                        b_vec.cast[Self.dtype](),
                        self[i, j],
                    )

            b_ptr = b_ptr + b_stride

    @always_inline
    def _accumulate_x86_simd[
        a_type: DType,
        b_type: DType,
        //,
        prefetch_offset: Optional[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[mut=False, Scalar[a_type], ...],
        a_base_offsets: TileTensor[mut=False, .int32, ...],
        a_offset: Int,
        b: UnsafePointer[mut=False, Scalar[b_type], ...],
        b_stride: Int,
        partial_load_b_size: Optional[Int] = None,
    ):
        """Accumulation optimized for AVX512 and AVX2."""

        comptime assert not CompilationTarget.has_neon()
        comptime assert (
            a_base_offsets.flat_rank == 1
        ), "a_base_offsets must be rank 1"

        comptime kernel_width = Self.num_cols * Self.simd_width
        var b_ptr = b

        for l in range(length):
            # prefetch
            comptime if prefetch_offset:
                comptime for j in range(Self.num_cols):
                    prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ](
                        b_ptr
                        + prefetch_offset.value() * kernel_width
                        + j * Self.simd_width
                    )

            comptime for i in range(Self.row_start, Self.row_stop):
                # Broadcast an scalar from A to a simd vector.
                var a_idx = Int(a_base_offsets[i]) + a_offset + l
                var a_splat_vec = SIMD[a_type, Self.simd_width](a[a_idx])

                comptime for j in range(Self.num_cols):
                    # Load a simd vector from B.
                    var b_vec = _simd_load_maybe_partial[
                        Self.simd_width, partial_load_b
                    ](b_ptr, j * Self.simd_width, partial_load_b_size)

                    # The following should be lifted to registers and show up as
                    # FMA instructions.
                    self[i, j] = fma(
                        a_splat_vec.cast[Self.dtype](),
                        b_vec.cast[Self.dtype](),
                        self[i, j],
                    )

            b_ptr = b_ptr + b_stride

    # ===-------------------------------------------------------------------===#
    # Accumulation optimized for NEON
    # ===-------------------------------------------------------------------===#

    # An example of accumulation with register tiling.
    #
    # B vector 0-3 -->                  reg6     reg7     reg6     reg7
    #                                |========|========|========|========|
    # A point  0   --> reg0, reg0[i] |  reg8  |  reg9  |  reg10 |  reg11 |
    #                                |--------|--------|--------|--------|
    # A point  1   --> reg1, reg1[i] |  reg12 |  reg13 |  reg14 |  reg15 |
    #                                |--------|--------|--------|--------|
    # A point  2   --> reg2, reg2[i] |  reg16 |  reg17 |  reg18 |  reg19 |
    #                                |--------|--------|--------|--------|
    # A point  3   --> reg3, reg3[i] |  reg20 |  reg21 |  reg22 |  reg23 |
    #                                |--------|--------|--------|--------|
    # A point  4   --> reg4, reg4[i] |  reg24 |  reg25 |  reg26 |  reg27 |
    #                                |--------|--------|--------|--------|
    # A point  5   --> reg5, reg5[i] |  reg28 |  reg29 |  reg30 |  reg31 |
    #                                |--------|--------|--------|--------|
    #
    #    -->      :         Load a SIMD vector from memory.
    # simd_width  :         |--------|
    # kernel_width:         |-----------------------------------|
    #
    #
    # The accumulation proceeds as:
    #   for i in range(lanes):
    #     for l in range(length):
    #         reg5 += reg0[i] * reg1
    #         reg6 += reg0[i] * reg2
    #         ...
    #
    # Neon FMA can take a lane of a register (reg0[i]). It's more efficient to load
    # A vectors first, then perform `num_lanes x num_rows x num_cols` FMA ops.
    #
    # We can reuse reg6, reg7 for different B vectors on Graviton3. This may spill
    # registers on Graviton2.

    @always_inline
    def _accumulate_neon[
        a_type: DType,
        b_type: DType,
        //,
        prefetch_offset: Optional[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[mut=False, Scalar[a_type], ...],
        a_stride: Int,
        b: UnsafePointer[mut=False, Scalar[b_type], ...],
        b_stride: Int,
        partial_load_b_size: Optional[Int] = None,
    ):
        """Accumulation optimized for NEON."""
        comptime assert CompilationTarget.has_neon()

        @always_inline
        def micro_kernel[num_lanes: Int](offset: Int) {mut self, imm}:
            var a_vecs = Array[SIMD[a_type, num_lanes], Self.num_rows](
                uninitialized=True
            )

            # Load vectors of size num_lanes from input.
            comptime for i in range(Self.row_start, Self.row_stop):
                a_vecs[i] = a.load[width=num_lanes](offset + i * a_stride)

            var b_ptr = b + offset * b_stride

            comptime for lane in range(num_lanes):
                comptime for j in range(Self.num_cols):
                    # Load a simd vector from B.
                    var b_vec = _simd_load_maybe_partial[
                        Self.simd_width, partial_load_b
                    ](b_ptr, j * Self.simd_width, partial_load_b_size)

                    comptime for i in range(Self.row_start, Self.row_stop):
                        # The following should be lifted to registers and show up as
                        # FMA instructions.
                        self[i, j] = fma[
                            dtype=Self.dtype, width=Self.simd_width
                        ](
                            a_vecs[i][lane].cast[Self.dtype](),
                            b_vec.cast[Self.dtype](),
                            self[i, j],
                        )

                b_ptr = b_ptr + b_stride

        # Load vectors from A first. The remainder is handled one element at a time.
        tile[[Self.simd_width, 1]](0, length, micro_kernel)

    @always_inline
    def _accumulate_neon[
        a_type: DType,
        b_type: DType,
        //,
        prefetch_offset: Optional[Int] = None,
        partial_load_b: Bool = False,
    ](
        mut self,
        length: Int,
        a: UnsafePointer[mut=False, Scalar[a_type], ...],
        a_base_offsets: TileTensor[mut=False, .int32, ...],
        a_offset: Int,
        b: UnsafePointer[mut=False, Scalar[b_type], ...],
        b_stride: Int,
        partial_load_b_size: Optional[Int] = None,
    ):
        """Accumulation optimized for NEON."""
        comptime assert CompilationTarget.has_neon()
        comptime assert (
            a_base_offsets.flat_rank == 1
        ), "a_base_offsets must be rank 1"

        @always_inline
        def micro_kernel[num_lanes: Int](offset: Int) {mut self, imm}:
            var a_vecs = Array[SIMD[a_type, num_lanes], Self.num_rows](
                uninitialized=True
            )

            # Load vectors of size num_lanes from input.
            comptime for i in range(Self.row_start, Self.row_stop):
                var a_idx = Int(a_base_offsets[i]) + a_offset + offset
                a_vecs[i] = a.load[width=num_lanes](a_idx)

            var b_ptr = b + offset * b_stride

            comptime for lane in range(num_lanes):
                comptime for j in range(Self.num_cols):
                    # Load a simd vector from B.
                    var b_vec = _simd_load_maybe_partial[
                        Self.simd_width, partial_load_b
                    ](b_ptr, j * Self.simd_width, partial_load_b_size)

                    comptime for i in range(Self.row_start, Self.row_stop):
                        # The following should be lifted to registers and show up as
                        # FMA instructions.
                        self[i, j] = fma[
                            dtype=Self.dtype, width=Self.simd_width
                        ](
                            a_vecs[i][lane].cast[Self.dtype](),
                            b_vec.cast[Self.dtype](),
                            self[i, j],
                        )

                b_ptr += b_stride

        # Load vectors from A first. The remainder is handled one element at a time.
        tile[[Self.simd_width, 1]](0, length, micro_kernel)


@always_inline
def _simd_load_maybe_partial[
    dt: DType, //, simd_width: Int, partial_load: Bool
](
    ptr: UnsafePointer[mut=False, Scalar[dt], ...],
    offset: Int,
    partial_load_size: Optional[Int] = None,
) -> SIMD[dt, simd_width]:
    """Load a simd vector. The vector may exceed the data's end, i.e.,
    offset + simd_width > end. In this case, if user specifies partial load, we
    will load partial values of size (end - offset), and fill the rest lanes
    with 0.

    One use case is in convolution when the output channel is NOT multiple of
    simd_width and is NOT padded with zeros at the end. We need to partially load
    the filter near the end.
    """

    comptime if partial_load:
        return partial_simd_load[simd_width](
            ptr + offset, 0, partial_load_size.value(), 0
        )
    else:
        return ptr.load[width=simd_width](offset)


@always_inline
def _simd_store_maybe_partial[
    dt: DType, //, simd_width: SIMDLength, partial_store: Bool
](
    ptr: UnsafePointer[mut=True, Scalar[dt], ...],
    offset: Int,
    vec: SIMD[dt, simd_width],
    partial_store_size: Optional[Int] = None,
):
    """Store a simd vector. The vector may exceed the data's end, i.e.,
    offset + simd_width > end. In this case, if user specifies partial_store, we
    will store `partial_store_size` lanes of input vector.
    """

    comptime if partial_store:
        # TODO: check if partial_store_size is present.
        return partial_simd_store[simd_width](
            ptr + offset, 0, partial_store_size.value(), vec
        )
    else:
        return ptr.store(offset, vec)
