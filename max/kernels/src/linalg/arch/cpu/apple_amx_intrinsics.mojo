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
#
# This file contains wrappers around Apple's AMX assembly instruction set.
# For information on the Apple AMX instruction set, see
# https://www.notion.so/modularai/Apple-AMX-Resources-2cc523b9c851498787dfloat946ebb09930e.
#
# ===-----------------------------------------------------------------------===#

"""Provides low-level wrappers around Apple's AMX assembly instruction set for matrix operations."""

from std.sys._assembly import inlined_assembly

from layout import TileTensor
from layout.tile_layout import row_major
from std.memory import (
    unsafe_memcpy,
    unsafe_memset_zero,
    unsafe_stack_allocation,
)


# All AMX instructions are of the form
# `0x00201000 | ((op & 0x1F) << 5) | (operand & 0x1F)`
# where `op` is the operation and `operand` is the register to operate on.


@always_inline
def _no_op_imms[op: Int32, imm: Int32]():
    # In Apple's Accelerate, instruction 17 is apparently always prefixed by
    # three nops.
    inlined_assembly[
        "nop\nnop\nnop\n.word (0x201000 + ($0 << 5) + $1)",
        NoneType,
        constraints="i,i,~{memory}",
        has_side_effect=True,
    ](op, imm)


@always_inline
def _op_gpr[op: Int32](gpr: Int64):
    inlined_assembly[
        ".word (0x201000 + ($0 << 5) + 0$1 - ((0$1 >> 4) * 6))",
        NoneType,
        constraints="i,r,~{memory}",
        has_side_effect=True,
    ](op, gpr)


# The `set` and `clr` take no non-constant operands, and so we pass them as
# immediate values via meta parameters.
@always_inline
def _set():
    _no_op_imms[17, 0]()


@always_inline
def _clr():
    _no_op_imms[17, 1]()


@always_inline
def ldx(gpr: Int):
    """Loads data from the memory address encoded in gpr into an AMX X register row.

    Args:
        gpr: Encoded operand combining the memory address and X register row index.
    """
    _op_gpr[0](Int64(gpr))


@always_inline
def ldy(gpr: Int):
    """Loads data from the memory address encoded in gpr into an AMX Y register row.

    Args:
        gpr: Encoded operand combining the memory address and Y register row index.
    """
    _op_gpr[1](Int64(gpr))


@always_inline
def stx(gpr: Int):
    """Stores an AMX X register row to the memory address encoded in gpr.

    Args:
        gpr: Encoded operand combining the memory address and X register row index.
    """
    _op_gpr[2](Int64(gpr))


@always_inline
def sty(gpr: Int):
    """Stores an AMX Y register row to the memory address encoded in gpr.

    Args:
        gpr: Encoded operand combining the memory address and Y register row index.
    """
    _op_gpr[3](Int64(gpr))


@always_inline
def ldz(gpr: Int):
    """Loads data from the memory address encoded in gpr into an AMX Z accumulator row.

    Args:
        gpr: Encoded operand combining the memory address and Z accumulator row index.
    """
    _op_gpr[4](Int64(gpr))


@always_inline
def stz(gpr: Int):
    """Stores an AMX Z accumulator row to the memory address encoded in gpr.

    Args:
        gpr: Encoded operand combining the memory address and Z accumulator row index.
    """
    _op_gpr[5](Int64(gpr))


@always_inline
def ldzi(gpr: Int):
    """Loads data from the memory address encoded in gpr into an AMX Z accumulator row using interleaved layout.

    Args:
        gpr: Encoded operand combining the memory address and Z accumulator row
            index for interleaved layout.
    """
    _op_gpr[6](Int64(gpr))


@always_inline
def stzi(gpr: Int):
    """Stores an AMX Z accumulator row to the memory address encoded in gpr using interleaved layout.

    Args:
        gpr: Encoded operand combining the memory address and Z accumulator row
            index for interleaved layout.
    """
    _op_gpr[7](Int64(gpr))


@always_inline
def extrx(gpr: Int):
    """
    Extracts a row or moves it to x, result in amx0.

    Args:
        gpr: Encoded operand selecting the source row and extraction mode.
    """
    _op_gpr[8](Int64(gpr))


@always_inline
def extry(gpr: Int):
    """
    Extracts a row or moves it to y, result in amx0.

    Args:
        gpr: Encoded operand selecting the source row and extraction mode.
    """
    _op_gpr[9](Int64(gpr))


@always_inline
def fma64(gpr: Int):
    """
    Float64 matrix multiply and add.

    Args:
        gpr: Encoded operand combining the X, Y, and Z row indices for the instruction.
    """
    _op_gpr[10](Int64(gpr))


@always_inline
def fsm64(gpr: Int):
    """
    Float64 matrix multiply and subtract.

    Args:
        gpr: Encoded operand combining the X, Y, and Z row indices for the instruction.
    """
    _op_gpr[11](Int64(gpr))


@always_inline
def fma32(gpr: Int):
    """
    Float32 matrix multiply and add.

    Args:
        gpr: Encoded operand combining the X, Y, and Z row indices for the instruction.
    """
    _op_gpr[12](Int64(gpr))


@always_inline
def fsm32(gpr: Int):
    """
    Float32 matrix multiply and subtract.

    Args:
        gpr: Encoded operand combining the X, Y, and Z row indices for the instruction.
    """
    _op_gpr[13](Int64(gpr))


@always_inline
def mac16(gpr: Int):
    """
    SI16 matrix multiply and add.

    Args:
        gpr: Encoded operand combining the X, Y, and Z row indices for the instruction.
    """
    _op_gpr[14](Int64(gpr))


@always_inline
def fma16(gpr: Int):
    """
    Float16 matrix multiply and subtract.

    Args:
        gpr: Encoded operand combining the X, Y, and Z row indices for the instruction.
    """
    _op_gpr[15](Int64(gpr))


@always_inline
def fms16(gpr: Int):
    """
    Float16 matrix multiply and add.

    Args:
        gpr: Encoded operand combining the X, Y, and Z row indices for the instruction.
    """
    _op_gpr[16](Int64(gpr))


@always_inline
def vec_int__(gpr: Int):
    """
    Horizontal ui16 multiply `z0[i] += x0[i] + y0[i]`.
    """
    _op_gpr[18](Int64(gpr))


@always_inline
def vecfp(gpr: Int):
    """
    Horizontal float16 multiply `z0[i] += x0[i] + y0[i]`.
    """
    _op_gpr[19](Int64(gpr))


@always_inline
def max_int__(gpr: Int):
    """
    UI16 matrix multiply.

    Args:
        gpr: Encoded operand combining the X, Y, and Z row indices for the instruction.
    """
    _op_gpr[20](Int64(gpr))


@always_inline
def matfp(gpr: Int):
    """
    Float16 matrix multiply.

    Args:
        gpr: Encoded operand combining the X, Y, and Z row indices for the instruction.
    """
    _op_gpr[21](Int64(gpr))


@always_inline
def genlut(gpr: Int):
    """Issues the Apple AMX genlut instruction using the address encoded in gpr.

    Args:
        gpr: Encoded operand for the lookup table instruction.
    """
    _op_gpr[22](Int64(gpr))


# Apple.amx.LoadStore is a set of utilities that are thin wrappers around
# the inline assembly calls, and they provide an easier interface to use
# the amx registers.
#
# The M1 AMX hardware has 3 dedicated register banks, in fp32 mode they
# can be described as:
#
#     float X[8][16], Y[8][16], Z[64][16];
#
#  All instructions reading and writing these AMX registers are memory
#  instructions. The ops defined here marks the direction into/out of amx
#  registers. e.g. :
#
#       load_store.store_x(ptr, idx),
#
#   will read a row of 16 fp32 elements from memory at `ptr`, and save the
#   data in X[idx][:].
#   while
#
#       load_store.load_x (ptr, idx),
#
#   is the opposite, taking X[idx][:] and write to the memory location `ptr`.


@always_inline
def _encode_load_store[
    row_count: Int, dtype: DType
](src: UnsafePointer[Scalar[dtype], _], start_index: Int) -> Int:
    """
    Utility to do the bit encoding for load and store ops.
    """
    var src_idx = Int(src) | (start_index << 56)

    comptime if row_count == 2:
        src_idx |= 1 << 62
    return src_idx


@always_inline
def store_x[
    row_count: Int, dtype: DType
](src: UnsafePointer[Scalar[dtype], _], start_index: Int):
    """Loads row_count rows from memory at src into AMX X registers starting at start_index.

    Parameters:
        row_count: Number of X register rows to load; must be 1 or 2.
        dtype: Element type of the X register rows.

    Args:
        src: Source memory address to load rows from.
        start_index: Starting X register row index to load into.
    """
    ldx(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def store_y[
    row_count: Int, dtype: DType
](src: UnsafePointer[Scalar[dtype], _], start_index: Int):
    """Loads row_count rows from memory at src into AMX Y registers starting at start_index.

    Parameters:
        row_count: Number of Y register rows to load; must be 1 or 2.
        dtype: Element type of the Y register rows.

    Args:
        src: Source memory address to load rows from.
        start_index: Starting Y register row index to load into.
    """
    ldy(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def store_z[
    row_count: Int, dtype: DType
](src: UnsafePointer[Scalar[dtype], _], start_index: Int):
    """Loads row_count rows from memory at src into AMX Z accumulator registers starting at start_index.

    Parameters:
        row_count: Number of Z accumulator rows to load; must be 1 or 2.
        dtype: Element type of the Z accumulator rows.

    Args:
        src: Source memory address to load rows from.
        start_index: Starting Z accumulator row index to load into.
    """
    ldz(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def read_x[
    row_count: Int, dtype: DType
](src: UnsafePointer[mut=True, Scalar[dtype], _], start_index: Int):
    """Stores row_count rows from AMX X registers starting at start_index to memory at src.

    Parameters:
        row_count: Number of X register rows to store; must be 1 or 2.
        dtype: Element type of the X register rows.

    Args:
        src: Destination memory address for the stored rows.
        start_index: Starting X register row index to store from.
    """
    stx(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def read_y[
    row_count: Int, dtype: DType
](src: UnsafePointer[mut=True, Scalar[dtype], _], start_index: Int):
    """Stores row_count rows from AMX Y registers starting at start_index to memory at src.

    Parameters:
        row_count: Number of Y register rows to store; must be 1 or 2.
        dtype: Element type of the Y register rows.

    Args:
        src: Destination memory address for the stored rows.
        start_index: Starting Y register row index to store from.
    """
    sty(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def load_z[
    row_count: Int, dtype: DType
](src: UnsafePointer[mut=True, Scalar[dtype], _], start_index: Int):
    """Stores row_count rows from AMX Z accumulator registers starting at start_index to memory at src.

    Parameters:
        row_count: Number of Z accumulator rows to store; must be 1 or 2.
        dtype: Element type of the Z accumulator rows.

    Args:
        src: Destination memory address for the stored rows.
        start_index: Starting Z accumulator row index to store from.
    """
    stz(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def transpose_z_to_x_or_y[
    destination: StaticString, dtype: DType
](z_col_index: Int, xy_row_index: Int, z_row_suboffset: Int):
    """Transposes a downsampled column of the AMX Z register into a row of the X or Y register.

    Wraps the fp32 transpose mode of the AMX `extry` instruction. One element in every
    four of the selected Z column (16 elements total) is extracted and placed into a row
    of the destination X or Y register bank.

    Parameters:
        destination: Either `"X"` or `"Y"`, selecting the destination register bank.
        dtype: Must be `DType.float32`.

    Args:
        z_col_index: Column of Z to read from; must be in [0, 15].
        xy_row_index: Row of the destination X or Y register; must be in [0, 7].
        z_row_suboffset: Starting row suboffset within the Z column; must be in [0, 3].
    """
    # transpose_z_to_x_or_y is a thin wrapper around the fp32 transpose mode of
    # the amx instruction `extry`. This instruction takes a (sub) column of
    # register Z (see description above), and transposes it into a row in either
    # register X or register Y.
    #
    # Note that each column of Z has 64 element but each row of X or Y has only
    # 16 elements. The slightly strange part of this instruction is that the
    # value written into X/Y is actually a downsample (i.e. one in every four)
    # result of a column of Z.
    #
    # The instruction takes 1 static parameter dest and 3 dynamic parameters:
    # z_col_index, xy_row_index, and z_row_suboffset.
    # dest can be either `X` or `Y`.
    # With the X,Y,Z data layout described as
    #
    #    float X[8][16], Y[8][16], Z[64][16];
    #
    #  This instruction essentially takes:
    #
    #    extracted_column [16] = Z[z_row_suboffset : 64 : 4][z_col_index]
    #
    # and writes extracted_column[16] to X/Y[xy_row_index][:].
    #  Legal ranges for the parameters:
    #    z_col_index needs to be 0-15,
    #    xy_row_index needs to be 0-7,
    #    z_row_suboffset needs to be 0-4.

    # The destination must be either "X" or "Y".
    comptime assert destination == "X" or destination == "Y"
    # The type must be Float32.
    comptime assert dtype == .float32

    # make the y offset field
    #  shift left by 6 to make this an offset in rows,
    #    in fp32 mode, there are 16 elements / 64 byte per row.
    #  The offset field has to be given in bytes.
    var offset = ((z_col_index << 2) | z_row_suboffset) << 20 | (
        xy_row_index << 6
    )

    comptime is_x_destination = destination == "X"

    var operand = offset | (
        0x8000000004004000 if is_x_destination else 0x8000000010004000
    )

    extry(operand)


@always_inline
def fma[
    mode: StaticString, dtype: DType
](z_row_index: Int, x_row_index: Int, y_row_index: Int, clear_z: Bool):
    """Issues an AMX fused multiply-add of X and Y register rows accumulated into the Z accumulator.

    Supports two modes selected by the `mode` parameter. In `"ROW"` mode the
    instruction performs an elementwise fused multiply-add computing
    `Z[z_row_index][:] += X[x_row_index][:] * Y[y_row_index][:]`. In `"TILE"`
    mode it computes the outer product `Y[y_row_index][:] X X[x_row_index][:]`
    (a 16x16 matrix) and accumulates the result into
    `Z[z_row_index::step 4][:]`. When `clear_z` is true the existing Z values
    are ignored instead of accumulated. Issues the `fma.fp32` AMX instruction.

    Parameters:
        mode: Either `"ROW"` for elementwise fma or `"TILE"` for outer-product fma.
        dtype: Must be `DType.float32`.

    Args:
        z_row_index: Row of Z to write; must be in [0, 8) in row mode or [0, 4) in tile mode.
        x_row_index: Row of X to read; must be in [0, 8).
        y_row_index: Row of Y to read; must be in [0, 8).
        clear_z: When true, clears the target Z rows instead of accumulating into them.
    """
    # Apple.amx.fma abstracts the fma operation on the amx hardware. Two modes of
    #  fma operations are supported in this instruction, referred to here as
    #  `RowMode` and `TileMode`.
    # `RowMode` is elementwise fma, for each set of given indices, the instruction
    #  computes z[z_row_index][:] += X[x_row_index][:] * Y[y_row_index][:].
    # `TileMode` is matrix fma, each op computes an outer product of:
    #   Y[y_row_index][:] X X[x_row_index][:], (generating a 16x16 matrix)
    #   and the resulting matrix is accumulated into Z[z_row_index::step 4][:].
    #  When clear_z is true, the existing value in Z will be ignored instead of
    #   being accumulated.
    #
    # Issues fma.fp32 instruction to AMX.
    #  Required input range (behavior for out of range is undefined):
    #  z_row_index : [0, 8) in row mode, [0, 4) in tile mode.
    #  x_row_index, y_row_index : always in [0, 8).

    # The mode must be either "TILE" or "ROW".
    comptime assert mode == "TILE" or mode == "ROW"
    # The type must be Float32.
    comptime assert dtype == .float32

    comptime is_row_mode = mode == "ROW"

    var operand = (
        y_row_index << 6
        | x_row_index << 16
        | z_row_index << 20
        | ((1 << 27) if clear_z else 0)
        | ((1 << 63) if is_row_mode else 0)
    )

    fma32(operand)


@always_inline
def dot_at_b(
    c: TileTensor[mut=True, address_space=.GENERIC, ...],
    a: type_of(c),
    b: type_of(c),
):
    """Performs a matrix multiply C = A^T * B using Apple AMX instructions.

    Supports f32 (16x16) and f16 (32x32) tiles. All matrices are stored in
    row-major order.

    Args:
        c: Output matrix C in row-major order; rank-2 tile with static 16x16
            shape for f32 or 32x32 shape for f16.
        a: Input matrix A in row-major order; same shape and dtype as c.
        b: Input matrix B in row-major order; same shape and dtype as c.
    """
    comptime assert (
        c.dtype == .float32 or c.dtype == .float16
    ), "the buffer dtype must be float32 or float16"

    var a_pointer = a.ptr
    var b_pointer = b.ptr
    var c_pointer = c.ptr

    comptime assert c.flat_rank == 2, "expected rank 2 tile"
    comptime assert (
        c.static_shape[0] != -1 and c.static_shape[1] != -1
    ), "dot_at_b requires fully static tile dimensions"
    comptime num_elements = c.static_shape[0] * c.static_shape[1]

    # TODO: We can elide the copy if the data is already aligned.
    var a_buffer = unsafe_stack_allocation[
        num_elements, Scalar[c.dtype], alignment=128
    ]()
    var b_buffer = unsafe_stack_allocation[
        num_elements, Scalar[c.dtype], alignment=128
    ]()
    var c_buffer = unsafe_stack_allocation[
        num_elements, Scalar[c.dtype], alignment=128
    ]()

    unsafe_memcpy(dest=a_buffer, src=a_pointer, count=num_elements)
    unsafe_memcpy(dest=b_buffer, src=b_pointer, count=num_elements)
    unsafe_memset_zero(c_buffer, num_elements)

    # _set() has the side effect of clearing the z tile
    _set()

    comptime dim0 = c.static_shape[0]

    comptime if c.dtype == .float32:
        comptime assert (
            c.static_shape[0] == 16 and c.static_shape[1] == 16
        ), "f32 AMX matmul requires 16x16 tiles"
        comptime for j in range(2):
            comptime for i in range(8):
                ldx((i << 56) | Int(b_buffer + (j * 8 + i) * dim0))
                ldy((i << 56) | Int(a_buffer + (j * 8 + i) * dim0))

            comptime for i in range(8):
                fma32((i << 6 << 10) | (i << 6))

        comptime for i in range(0, 64, 4):
            stz((i << 56) | Int(c_buffer + (i >> 2) * dim0))
    elif c.dtype == .float16:
        comptime assert (
            c.static_shape[0] == 32 and c.static_shape[1] == 32
        ), "f16 AMX matmul requires 32x32 tiles"
        comptime for j in range(4):
            comptime for i in range(8):
                ldx((i << 56) | Int(b_buffer + (j * 8 + i) * dim0))
                ldy((i << 56) | Int(a_buffer + (j * 8 + i) * dim0))

            comptime for i in range(8):
                fma16((i << 6 << 10) | (i << 6))

        comptime for i in range(0, 64, 2):
            stz((i << 56) | Int(c_buffer + (i >> 1) * dim0))

    _clr()

    unsafe_memcpy(dest=c_pointer, src=c_buffer, count=num_elements)
