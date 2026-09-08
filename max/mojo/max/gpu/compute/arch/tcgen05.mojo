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

"""This module includes utilities for working with the
tensorcore 5th generation (tcgen05) instructions."""

from std.os import abort
from std.sys import _RegisterPackType, size_of
from std.sys._assembly import inlined_assembly
from std.sys.info import _has_blackwell_tcgen05
from std._gpu.intrinsics import _get_nvtx_register_constraint
from std.memory import bitcast

from max.gpu import external_memory
from max.gpu.compute.mma import _str_iota  # TODO: move to a string module
from max.gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptor


@always_inline("nodebug")
def check_blackwell_constraint():
    """Compile-time constraint ensuring Blackwell hardware is targeted."""
    comptime assert _has_blackwell_tcgen05(), (
        "The tcgen05 instructions are only applicable on nVidia Blackwell"
        " (sm_100a, sm_101a) hardware."
    )


struct TensorMemory(TrivialRegisterPassable):
    """A wrapper around tensor memory allocated for tcgen05 instructions."""

    var ptr: Pointer[UInt32, MutUntrackedOrigin, address_space=.SHARED]
    """Pointer to the tensor memory address."""

    var num_cols: UInt32
    """The number of columns in the tensor memory."""

    @always_inline
    def __init__(out self, num_cols: UInt32):
        """Initialize the TensorMemory struct.

        Args:
            num_cols: The number of columns to allocate.
        """
        # Bitcast to avoid `cannot implicitly convert` error.
        self.ptr = (
            external_memory[UInt32, address_space=.SHARED, alignment=16]()
            .unsafe_bitcast[UInt32]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        self.num_cols = num_cols


@always_inline("nodebug")
def tcgen05_alloc[
    cta_group: Int32
](
    ptr_tmem_addr: Pointer[mut=True, UInt32, _, address_space=.SHARED],
    num_cols: UInt32,
):
    """Allocates tensor memory for use with tcgen05 instructions.

    Parameters:
        cta_group: The cooperative thread array (CTA) group ID.

    Args:
        ptr_tmem_addr: Shared memory pointer to hold tensor memory address.
        num_cols: The number of columns to allocate.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()
    comptime assert cta_group == 1 or cta_group == 2, "cta_group must be 1 or 2"
    inlined_assembly[
        "tcgen05.alloc.cta_group::"
        + String(cta_group)
        + ".sync.aligned.shared::cta.b32 [$0], $1;",
        NoneType,
        constraints="r,r",
        has_side_effect=True,
    ](
        UInt32(Int(ptr_tmem_addr)),
        num_cols,
    )


@always_inline
def tcgen05_dealloc[cta_group: Int32](tmem_addr: UInt32, num_cols: UInt32):
    """Deallocates tensor memory allocated by tcgen05_alloc().

    This function deallocates tensor memory that was previously allocated using
    tcgen05_alloc(). The deallocation must be performed by the same CTA group
    that performed the allocation.

    Parameters:
        cta_group: The cooperative thread array (CTA) group ID.

    Args:
        tmem_addr: Address of the tensor memory to deallocate.
        num_cols: Number of columns in the tensor memory.
    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()
    comptime assert cta_group == 1 or cta_group == 2, "cta_group must be 1 or 2"
    inlined_assembly[
        "tcgen05.dealloc.cta_group::"
        + String(cta_group)
        + ".sync.aligned.b32 $0, $1;",
        NoneType,
        constraints="r,r",
        has_side_effect=True,
    ](tmem_addr, num_cols)


@always_inline
def tcgen05_ld[
    *,
    datapaths: Int,
    bits: Int,
    repeat: Int,
    dtype: DType,
    pack: Bool,
    width: Int = (datapaths * bits * repeat) // (32 * 32),
](tmem_addr: UInt32) -> Array[Scalar[dtype], width]:
    """Loads data from tensor memory into registers.

    Parameters:
        datapaths: The first dimension of the shape.
        bits: The second dimension of the shape.
        repeat: The repeat factor.
        dtype: The data type to load.
        pack: Whether to pack two 16-bit chunks of adjacent columns into a single 32-bit register.
        width: The number elements in the result vector.

    Args:
        tmem_addr: The address of the tensor memory to load from.

    Returns:
        The Array containing the loaded data.
    """
    check_blackwell_constraint()

    comptime assert (
        (datapaths == 16 and bits == 64)
        or (datapaths == 16 and bits == 128)
        or (datapaths == 16 and bits == 256)
        or (datapaths == 32 and bits == 32)
    ), (
        "`datapaths`x`bits`b must be 16x64b, 16x128b, 16x256b or 32x32b, got "
        + String(datapaths)
        + "x"
        + String(bits)
        + "b."
    )

    comptime assert repeat in [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ], String(
        "`repeat` must be a power of 2 in the range [1, 128], got ", repeat, "."
    )

    comptime assert width in [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ], String(
        "`width` must be a power of 2 in the range [1, 128], got ", width, "."
    )

    comptime assert (
        width == (repeat * bits * datapaths) // (32 * 32)
        and size_of[dtype]() == 4
    ), String(
        (
            "Only support 4B data type and width must be equal to (num * n"
            " * m) // (32 * 32). width is "
        ),
        width,
        " and ",
        dtype,
        " but need ",
        (repeat * bits * datapaths) // (32 * 32),
    )

    comptime shape_str = String(datapaths) + "x" + String(bits)
    comptime num_str = String(repeat)
    comptime pack_str = ".pack::16b" if pack else ""
    comptime constraints_str = (
        "=" + _get_nvtx_register_constraint[dtype]() + ","
    ) * width + "r"
    comptime output_args_str = "{" + _str_iota[
        width, prefix="$", sep=","
    ]() + "}"
    comptime addr_str = "[$" + String(width) + "]"

    @__parameter
    @always_inline("nodebug")
    def call_ld_intrinsic[
        pack_type: TrivialRegisterPassable
    ]() -> Array[Scalar[dtype], width]:
        var r = inlined_assembly[
            "tcgen05.ld.sync.aligned."
            + shape_str
            + "b.x"
            + num_str
            + pack_str
            + ".b32 "
            + output_args_str
            + ", "
            + addr_str
            + ";",
            pack_type,
            constraints=constraints_str,
            has_side_effect=True,
        ](tmem_addr)
        var ptr = Pointer(to=r).unsafe_bitcast[Scalar[dtype]]()
        var result = Array[Scalar[dtype], width](uninitialized=True)
        comptime for i in range(width):
            result[i] = ptr[unsafe_offset=i]
        return result^

    # fmt: off
    comptime S = Scalar[dtype]
    comptime if width == 1:
        return call_ld_intrinsic[S]()
    elif width == 2:
        return call_ld_intrinsic[
                _RegisterPackType[S, S]
            ]()
    elif width == 4:
        return  call_ld_intrinsic[
                _RegisterPackType[S, S, S, S]
            ]()
    elif width == 8:
        return call_ld_intrinsic[
                _RegisterPackType[S, S, S, S, S, S, S, S]
             ]()
    elif width == 16:
        return call_ld_intrinsic[
                _RegisterPackType[S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S
            ]
        ]()
    elif width == 32:
        return call_ld_intrinsic[
                _RegisterPackType[S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S
            ]
        ]()
    elif width == 64:
        return call_ld_intrinsic[
                _RegisterPackType[S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S
            ]
        ]()
    elif width == 128:
        return call_ld_intrinsic[
                _RegisterPackType[S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
                                  S, S, S, S, S, S, S, S,
            ]
        ]()
    else:
        comptime assert False, "width must be a power of 2 in the range [1, 128]."
    # fmt: on


def tcgen05_st[
    dtype: DType,
    width: Int,
    //,
    *,
    datapaths: Int,
    bits: Int,
    repeat: Int,
    pack: Bool,
](tmem_addr: UInt32, data: Array[Scalar[dtype], width]):
    """Stores data from registers into tensor memory.

    Parameters:
        dtype: The data type to store.
        width: The number of elements in the data vector.
        datapaths: The first dimension of the shape.
        bits: The second dimension of the shape.
        repeat: The repeat factor.
        pack: Whether to pack two 16-bit chunks of adjacent columns into a single 32-bit register.

    Args:
        tmem_addr: The address of the tensor memory to store to.
        data: The data to store into the tensor memory.
    """
    check_blackwell_constraint()

    comptime assert (
        (datapaths == 16 and bits == 64)
        or (datapaths == 16 and bits == 128)
        or (datapaths == 16 and bits == 256)
        or (datapaths == 32 and bits == 32)
    ), "`datapaths`x`bits`b must be 16x64b, 16x128b, 16x256b or 32x32b."

    comptime assert repeat in [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ], "`repeat` must be a power of 2 in the range [1, 128]."

    comptime assert width in [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    ], "`width` must be a power of 2 in the range [1, 128]."

    comptime assert (
        width == (repeat * bits * datapaths) // (32 * 32)
        and size_of[dtype]() == 4
    ), (
        "Only support 4B data type and width must be equal to (num * n"
        " * m) // (32 * 32)."
    )

    comptime shape_str = String(datapaths) + "x" + String(bits)
    comptime num_str = String(repeat)
    comptime pack_str = ".unpack::16b" if pack else ""
    comptime constraints_str = (
        _get_nvtx_register_constraint[dtype]() + ","
    ) * Int(width) + "r"
    comptime addr_str = "[$" + String(Int(width)) + "]"
    comptime input_args_str = "{" + _str_iota[
        Int(width), prefix="$", sep=","
    ]() + "}"

    comptime asm_str = (
        "tcgen05.st.sync.aligned."
        + shape_str
        + "b.x"
        + num_str
        + pack_str
        + ".b32 "
        + addr_str
        + ", "
        + input_args_str
        + ";"
    )

    # fmt: off
    comptime if width == 1:
        inlined_assembly[asm_str, NoneType, constraints=constraints_str, has_side_effect=True](
            data[0],
            tmem_addr)
    elif width == 2:
        inlined_assembly[asm_str, NoneType, constraints=constraints_str, has_side_effect=True](
            data[0], data[1],
            tmem_addr)
    elif width == 4:
        inlined_assembly[asm_str, NoneType, constraints=constraints_str, has_side_effect=True](
            data[0], data[1], data[2], data[3],
            tmem_addr)
    elif width == 8:
        inlined_assembly[asm_str, NoneType, constraints=constraints_str, has_side_effect=True](
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            tmem_addr)
    elif width == 16:
        inlined_assembly[asm_str, NoneType, constraints=constraints_str, has_side_effect=True](
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
            tmem_addr)
    elif width == 32:
        inlined_assembly[asm_str, NoneType, constraints=constraints_str, has_side_effect=True](
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
            tmem_addr)
    elif width == 64:
        inlined_assembly[asm_str, NoneType, constraints=constraints_str, has_side_effect=True](
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
            data[40], data[41], data[42], data[43], data[44], data[45], data[46], data[47],
            data[48], data[49], data[50], data[51], data[52], data[53], data[54], data[55],
            data[56], data[57], data[58], data[59], data[60], data[61], data[62], data[63],
            tmem_addr)
    else:
        inlined_assembly[asm_str, NoneType, constraints=constraints_str, has_side_effect=True](
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
            data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
            data[40], data[41], data[42], data[43], data[44], data[45], data[46], data[47],
            data[48], data[49], data[50], data[51], data[52], data[53], data[54], data[55],
            data[56], data[57], data[58], data[59], data[60], data[61], data[62], data[63],
            data[64], data[65], data[66], data[67], data[68], data[69], data[70], data[71],
            data[72], data[73], data[74], data[75], data[76], data[77], data[78], data[79],
            data[80], data[81], data[82], data[83], data[84], data[85], data[86], data[87],
            data[88], data[89], data[90], data[91], data[92], data[93], data[94], data[95],
            data[96], data[97], data[98], data[99], data[100], data[101], data[102], data[103],
            data[104], data[105], data[106], data[107], data[108], data[109], data[110], data[111],
            data[112], data[113], data[114], data[115], data[116], data[117], data[118], data[119],
            data[120], data[121], data[122], data[123], data[124], data[125], data[126], data[127],
            tmem_addr)
    # fmt: on


@always_inline
def tcgen05_release_allocation_lock[cta_group: Int32]():
    """Releases the allocation lock for the current CTA group.

    Parameters:
        cta_group: The cooperative thread array (CTA) group ID.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()
    comptime assert cta_group == 1 or cta_group == 2, "cta_group must be 1 or 2"

    inlined_assembly[
        "tcgen05.relinquish_alloc_permit.cta_group::"
        + String(cta_group)
        + ".sync.aligned;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline
def tcgen05_load_wait():
    """Waits for tensor memory loads to complete.


    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()

    inlined_assembly[
        "tcgen05.wait::ld.sync.aligned;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline
def tcgen05_store_wait():
    """Waits for tensor memory stores to complete.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()

    inlined_assembly[
        "tcgen05.wait::st.sync.aligned;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline
def tcgen05_fence_before():
    """Orders all the prior asynchronous `tcgen05` operations.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()

    inlined_assembly[
        "tcgen05.fence::before_thread_sync;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline
def tcgen05_fence_after():
    """Orders all the subsequent asynchronous `tcgen05` operations.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()

    inlined_assembly[
        "tcgen05.fence::after_thread_sync;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline
def tcgen05_cp[
    *,
    cta_group: Int32,
    datapaths: Int,
    bits: Int,
    src_fmt: String = "",
    dst_fmt: String = "",
    multicast: String = "",
](tmem_addr: UInt32, s_desc: MMASmemDescriptor):
    """Copies data from shared memory described by the matrix descriptor `s_desc` to tensor memory `tmem_addr`.

    Parameters:
        cta_group: The cooperative thread array (CTA) group ID.
        datapaths: The first dimension of the shape.
        bits: The second dimension of the shape.
        src_fmt: Source format string.
        dst_fmt: Destination format string.
        multicast: Multicast string.

    Args:
        tmem_addr: Address of the tensor memory.
        s_desc: Matrix descriptor for the copy operation.

    Note:
        This function is only available on NVIDIA Blackwell GPUs (SM 100+).
    """
    check_blackwell_constraint()
    comptime assert cta_group == 1 or cta_group == 2, "cta_group must be 1 or 2"

    comptime assert (
        (datapaths == 128 and bits == 256)
        or (datapaths == 4 and bits == 256)
        or (datapaths == 128 and bits == 128)
        or (datapaths == 64 and bits == 128)
        or (datapaths == 32 and bits == 128)
    ), (
        "`datapaths`x`bits`b must be 128x256b, 4x256b, 128x128b, 64x128b or"
        " 32x128b."
    )

    comptime assert (
        src_fmt == "" or src_fmt == "b6x16_p32" or src_fmt == "b4x16_p64"
    ), "src_fmt must be empty, 'b6x16_p32' or 'b4x16_p64'."

    comptime assert (
        dst_fmt == "" or dst_fmt == "b8x16"
    ), "dst_fmt must be empty or 'b8x16'."

    comptime assert not (
        (dst_fmt.byte_length() == 0) ^ (src_fmt.byte_length() == 0)
    ), "Both or none of dst_fmt and src_fmt must be provided."

    comptime assert (
        multicast == ""
        or multicast == "warpx2::02_13"
        or multicast == "warpx2::01_23"
        or multicast == "warpx4"
    ), "multicast must be empty, 'warpx2::02_13', 'warpx2::01_23' or 'warpx4'."

    comptime assert (
        datapaths != 32 or bits != 128
    ) or multicast == "warpx4", "For 32x128b, multicast must be 'warpx4'"

    comptime asm_str = String(
        "tcgen05.cp.cta_group::",
        String(cta_group),
        ".",
        String(datapaths),
        "x",
        String(bits),
        "b",
        ("" if not multicast else "." + multicast),
        ("" if not dst_fmt else "." + dst_fmt),
        ("" if not src_fmt else "." + src_fmt),
        " [$0], $1;",
    )

    inlined_assembly[
        asm_str,
        NoneType,
        has_side_effect=True,
        constraints="r,l",
    ](tmem_addr, s_desc)
