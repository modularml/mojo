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
warp-matrix-matrix-multiplication (wmma) instructions."""

from std.collections.string.string_span import _get_kgen_string
from std.sys import _RegisterPackType, is_nvidia_gpu, llvm_intrinsic, size_of
from std.sys._assembly import inlined_assembly
from std.sys.info import (
    CompilationTarget,
    _cdna_4_or_newer,
    _is_amd_rdna,
    _is_amd_rdna3,
    _is_amd_rdna4,
    is_amd_gpu,
    is_apple_m5,
)

from std._gpu._utils import (
    _get_llvm_struct_fields,
    array_to_llvm_struct,
    llvm_struct_to_array,
    llvm_struct_to_simd,
    simd_to_llvm_struct,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.mma_operand_descriptor import MMAOperandDescriptor
from std.memory import bitcast

from std.utils import StaticTuple
from std.utils.index import Index

# Import architecture-specific MMA implementations
from .arch.mma_nvidia import _mma_nvidia
from .arch.mma_amd import _mma_amd
from .arch.mma_apple import _mma_apple


def get_amd_fp8_dtype() -> Optional[DType]:
    """Gets the appropriate FP8 dtype for the current AMD GPU architecture.

    Returns:
        - `DType.float8_e4m3fn` for CDNA4+ and RDNA4+ GPUs
        - `DType.float8_e4m3fnuz` for CDNA1-3 GPUs
        - `None` for RDNA3 (no native FP8 support).
    """

    comptime if _is_amd_rdna3():
        return None
    elif _is_amd_rdna4() or _cdna_4_or_newer():
        return DType.float8_e4m3fn
    else:
        return DType.float8_e4m3fnuz


def get_amd_bf8_dtype() -> Optional[DType]:
    """Gets the appropriate BF8 dtype for the current AMD GPU architecture.

    Returns:
        - `DType.float8_e5m2` for CDNA4+ and RDNA4+ GPUs
        - `DType.float8_e5m2fnuz` for CDNA1-3 GPUs
        - `None` for RDNA3 (no native BF8 support).
    """

    comptime if _is_amd_rdna3():
        return None
    elif _is_amd_rdna4() or _cdna_4_or_newer():
        return DType.float8_e5m2
    else:
        return DType.float8_e5m2fnuz


@always_inline
def _unsupported_mma_op(d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    # fmt: off
    comptime assert False, String(
        "no valid implementation of mma for a=",
        Int(a.length), "x",  a.dtype,
        ", b=",  Int(b.length), "x",  b.dtype,
        ", c=",  Int(c.length), "x",  c.dtype,
        ", and d=", Int(d.length), "x", d.dtype,
    )
    # fmt: on


@always_inline
def _has_type[type: DType](a: DType, b: DType, c: DType, d: DType) -> Bool:
    return _has_type[(type, type, type, type)](a, b, c, d)


@always_inline
def _has_type[
    abcd: Tuple[DType, DType, DType, DType]
](a: DType, b: DType, c: DType, d: DType) -> Bool:
    return (a, b, c, d) == abcd


@always_inline
def _has_shape[size: Int](a: Int, b: Int, c: Int, d: Int) -> Bool:
    return _has_shape[(size, size, size, size)](a, b, c, d)


@always_inline
def _has_shape[
    abcd: Tuple[Int, Int, Int, Int]
](a: Int, b: Int, c: Int, d: Int) -> Bool:
    return (a, b, c, d) == abcd


# ===----------------------------------------------------------------------===#
# NVVM Helper Functions (used by WGMMA operations)
# ===----------------------------------------------------------------------===#


def _dtype_to_nvvm_type[
    out_type: DType, in_type: DType = out_type
]() -> __mlir_type.`!kgen.deferred`:
    comptime if out_type == .float16 or out_type == .uint32:
        # Special case when input types are integers, the result has to be integer too.
        if in_type != out_type and in_type.is_integral():
            return __mlir_attr.`si32`
        return __mlir_attr.`f16`
    else:
        return out_type.__mlir_type()


def _dtype_to_nvvm_wgmma_type[
    out_type: DType, in_type: DType = out_type
]() -> __mlir_type.`!kgen.deferred`:
    comptime if out_type == .float8_e4m3fn:
        return __mlir_attr[`#nvvm.wgmma_type<e4m3>`]
    elif out_type == .float8_e5m2:
        return __mlir_attr[`#nvvm.wgmma_type<e5m2>`]
    elif out_type == .float16 or out_type == .uint32:
        # Special case when input types are integers, the result has to be integer too.
        if in_type != out_type and in_type.is_integral():
            return __mlir_attr[`#nvvm.wgmma_type<s32>`]
        return __mlir_attr[`#nvvm.wgmma_type<f16>`]
    elif out_type == .int8:
        return __mlir_attr[`#nvvm.wgmma_type<s8>`]
    elif out_type == .uint8:
        return __mlir_attr[`#nvvm.wgmma_type<u8>`]
    elif out_type == .float32:
        return __mlir_attr[`#nvvm.wgmma_type<tf32>`]
    else:
        return __mlir_deferred_attr[
            `#nvvm.wgmma_type<`, +_dtype_to_nvvm_type[out_type, in_type](), `>`
        ]


def _get_shape[m: Int, n: Int, k: Int]() -> __mlir_type.`!kgen.deferred`:
    return __mlir_deferred_attr[
        `#nvvm.shape<m =`,
        +m.__mlir_index__(),
        `, n =`,
        +n.__mlir_index__(),
        `, k =`,
        +k.__mlir_index__(),
        `>`,
    ]


def _to_nvvm_scale_out[s: Int]() -> __mlir_type.`!kgen.deferred`:
    comptime if s == 0:
        return __mlir_attr.`#nvvm.wgmma_scale_out<zero>`
    else:
        return __mlir_attr.`#nvvm.wgmma_scale_out<one>`


def _to_nvvm_scale_in[s: Int]() -> __mlir_type.`!kgen.deferred`:
    comptime if s == -1:
        return __mlir_attr.`#nvvm.wgmma_scale_in<neg>`
    else:
        return __mlir_attr.`#nvvm.wgmma_scale_in<one>`


def _to_nvvm_layout[s: StaticString]() -> __mlir_type.`!kgen.deferred`:
    return __mlir_deferred_attr[
        `#nvvm.mma_layout<`, +_get_kgen_string[s](), `>`
    ]


@always_inline
def mma[block_size: Int = 1](mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """Performs warp sync Tensor Core based Matrix-multiply and accumulate (MMA) operation.

    This function executes a matrix multiply-accumulate operation using GPU Tensor Cores,
    synchronizing across the warp. It dispatches to architecture-specific implementations
    for NVIDIA and AMD GPUs.

    Parameters:
        block_size: The size of the block of the MMA operation (e.g., 4x4x4_16B). Applies to AMD GPUs only.

    Args:
        d: Output SIMD vector to store the result.
        a: First input matrix as SIMD vector.
        b: Second input matrix as SIMD vector.
        c: Accumulator matrix as SIMD vector.

    The operation performed is: d = (a * b) + c

    Supported configurations depend on the GPU architecture:
    - NVIDIA: Various combinations of FP32, FP16, BF16, and FP8 formats
    - AMD: Limited subset of FP32 and FP16 operations

    Note:
        - All threads in a warp must execute this operation together
        - Input matrices must be properly loaded and formatted for Tensor Core operations
        - Matrix dimensions and data types must match hardware requirements
    """

    comptime if is_nvidia_gpu():
        _mma_nvidia(d, a, b, c)
    elif is_amd_gpu():
        _mma_amd[block_size](d, a, b, c)
    # MSTDL-2556: Compilation Target Check doesn't match above
    elif is_apple_m5() and CompilationTarget._has_feature["metal4_0"]():
        _mma_apple(d, a, b, c)
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


# ===------------------------------------------------------------------===#
# LDMATRIX Instruction
# ===------------------------------------------------------------------===#


@always_inline
def ld_matrix[
    dtype: DType, //, simd_width: Int, *, transpose: Bool = False
](ptr: Pointer[mut=False, Scalar[dtype], ...]) -> SIMD[dtype, simd_width]:
    """Loads a matrix from shared memory into registers in a format suitable for tensor core operations.

    This function performs a warp-synchronized load from shared memory to registers, formatting the data
    to be directly usable by tensor core Matrix Multiply-Accumulate (MMA) instructions.

    Parameters:
        dtype: The data type of the matrix elements (e.g. float16, float32).
        simd_width: The width of the SIMD vector to load.
        transpose: Whether to transpose the matrix during load (only supported for half precision).

    Args:
        ptr: Pointer to shared memory containing the source matrix data.

    Returns:
        SIMD vector containing the loaded matrix data, properly formatted for MMA operations.

    Note:
        - All threads in a warp must execute this operation together.
        - For transposed loads, only half precision (float16) is supported.
        - The register width is fixed at 4 bytes (32 bits).
        - Supported configurations:
            - x1: One 32-bit register per thread.
            - x2: Two 32-bit registers per thread.
            - x4: Four 32-bit registers per thread.

    Example:

        ```mojo
        from max.gpu.compute.mma import ld_matrix
        from std.memory import alloc, dealloc, Layout

        var layout = Layout[Float16](count=8)
        var allocation = alloc(layout)
        var ptr = allocation.unsafe_ptr()

        # Load 8x8 matrix of float16 values
        var data = ld_matrix[simd_width=8](ptr)

        # Load transposed matrix
        var transposed = ld_matrix[simd_width=8, transpose=True](ptr)

        dealloc(allocation^)
        ```
    """

    comptime assert (transpose and dtype.is_half_float()) or (
        not transpose
    ), "Transposed ld_matrix is only for half precision."

    # The register width is fixed at 4 Bytes (32 bits)
    comptime register_btypes = 4
    comptime register_width = register_btypes // size_of[dtype]()
    comptime num_registers = simd_width // register_width

    # Full intrinsic is base + suffix
    comptime base = "llvm.nvvm.ldmatrix.sync.aligned.m8n8"

    @__parameter
    def get_suffix() -> String:
        comptime sfx = ".b16.p3"
        if transpose:
            return ".trans" + sfx
        return sfx

    var d: SIMD[dtype, simd_width]

    # Here .x1 means every thread would use a single register, x2 is 2 while x4 is 4 registers
    # An mma of shape m16n8k8 of type TF32 means for Matrix A every thread would have 4 registers hence .x4
    # and input simd_width being equal to 4
    comptime if num_registers == 1:
        comptime ins = base + ".x1" + get_suffix()
        var r = llvm_intrinsic[ins, UInt32](ptr)
        var r0 = bitcast[dtype, register_width](r[0])

        d = rebind[SIMD[dtype, simd_width]](r0)

    elif num_registers == 2:
        comptime ins = base + ".x2" + get_suffix()
        var r = llvm_intrinsic[ins, _RegisterPackType[UInt32, UInt32]](ptr)
        var r0 = bitcast[dtype, register_width](r[0])
        var r1 = bitcast[dtype, register_width](r[1])

        d = rebind[SIMD[dtype, simd_width]](r0.join(r1))

    else:
        comptime assert (
            num_registers == 4
        ), "no valid implementation of ldmatrix instruction"
        comptime ins = base + ".x4" + get_suffix()
        var r = llvm_intrinsic[
            ins, _RegisterPackType[UInt32, UInt32, UInt32, UInt32]
        ](ptr)

        # Unpack result to 4 vectors (one per register), then concat them to return.
        var r0 = bitcast[dtype, register_width](r[0])
        var r1 = bitcast[dtype, register_width](r[1])
        var r2 = bitcast[dtype, register_width](r[2])
        var r3 = bitcast[dtype, register_width](r[3])
        d = rebind[SIMD[dtype, simd_width]](r0.join(r1).join(r2.join(r3)))

        # The following creates additional copies uint32 <-> 2xbf16 in matmul.
        # comptime
        # for i in range(num_registers):
        #     var vec_per_register = bitcast[dtype, register_width](
        #         rebind[UInt32](r[i])
        #     )

        #     comptime
        #     for j in range(register_width):
        #         d[i * register_width + j] = vec_per_register[j]

    return d


# ===------------------------------------------------------------------===#
# STMATRIX Instruction
# ===------------------------------------------------------------------===#


@always_inline
def st_matrix[
    dtype: DType, //, simd_width: Int, *, transpose: Bool = False
](
    ptr: Pointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    d: SIMD[.float32, simd_width],
):
    """Performs warp-synchronized copy from registers to shared memory.

    This function stores data from registers to shared memory in a format that can be
    directly used by tensor core Matrix Multiply-Accumulate (MMA) instructions. It uses
    the NVIDIA stmatrix instruction to perform an efficient warp-synchronized store.

    Parameters:
        dtype: Data type of elements to store.
        simd_width: Width of the SIMD vector.
        transpose: If True, transposes the matrix during store.

    Args:
        ptr: Pointer to shared memory where data will be stored.
        d: SIMD vector containing the data to store.

    Constraints:
        - Must be used with shared memory pointers.
        - Number of registers must be 1, 2, or 4.
        - Data must be properly aligned for matrix operations.
        - All threads in warp must participate.
        - Only supported on NVIDIA GPUs with tensor core capabilities.

    Note:
        The function performs a warp-synchronized operation - all threads in the warp
        must execute this instruction to avoid deadlock.
    """

    comptime assert dtype in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), ""

    comptime num_matrices = simd_width

    comptime base = "stmatrix.sync.aligned"

    @__parameter
    def get_suffix() -> String:
        comptime sfx = ".m8n8"
        if transpose:
            return ".trans" + sfx
        return sfx

    comptime if num_matrices == 1:
        comptime ins = base + get_suffix() + ".x1.shared.b16 [$0], {$1};\n"
        inlined_assembly[ins, NoneType, constraints="r,r"](
            Int32(Int(ptr)), d[0]
        )

    elif num_matrices == 2:
        comptime ins = base + get_suffix() + ".x2.shared.b16 [$0], {$1, $2};\n"
        inlined_assembly[ins, NoneType, constraints="r,r,r"](
            Int32(Int(ptr)), d[0], d[1]
        )

    else:
        comptime assert (
            num_matrices == 4
        ), "no valid implementation of stmatrix instruction"

        comptime ins = base + get_suffix() + ".x4.shared.b16 [$0], {$1, $2, $3, $4};\n"
        inlined_assembly[ins, NoneType, constraints="r,r,r,r,r"](
            Int32(Int(ptr)), d[0], d[1], d[2], d[3]
        )


# ===------------------------------------------------------------------===#
# Warp group MMA asynchronous compute and synchronization instructions.
# ===------------------------------------------------------------------===#


# Shared memory operand descriptor.
struct WGMMADescriptor[dtype: DType](
    MMAOperandDescriptor, TrivialRegisterPassable
):
    """Descriptor for shared memory operands used in warp group matrix multiply operations.

    This struct represents a descriptor that encodes information about shared memory layout
    and access patterns for warp group matrix multiply operations. The descriptor contains
    the following bit fields:

    | Bit field | Size | Description |
    |-----------|------|-------------|
    | 0-13   |  14  | Base address in shared memory |
    | 16-29   |  14  | LBO: leading dim byte offset |
    | 32-45   |  14  | SBO: stride dim byte offset |
    | 49-51   |   3  | Matrix base offset, 0 for canonical layouts |
    | 62-63   |   2  | Swizzle mode: <br>&nbsp;&nbsp;0: no swizzle, <br>&nbsp;&nbsp;1: 128B swizzle, <br>&nbsp;&nbsp;2: 64B swizzle, <br>&nbsp;&nbsp;3: 32B swizzle |

    Note:

    - Some bits are unused.
    - Base address, LBO, and SBO ignore 4 least significant bits.

    Parameters:
        dtype: The data type of the shared memory operand. This affects memory alignment
               and access patterns for the descriptor.

    See: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
    """

    var desc: Int64
    """The 64-bit descriptor value that encodes shared memory layout information.

    This field stores the complete descriptor with all bit fields packed into a single 64-bit integer:
    - Bits 0-13: Base address in shared memory (14 bits)
    - Bits 16-29: Leading dimension stride in bytes (14 bits)
    - Bits 32-45: Stride dimension offset in bytes (14 bits)
    - Bits 49-51: Base offset (3 bits)
    - Bits 62-63: Swizzle mode for memory access pattern (2 bits)

    The descriptor is used by NVIDIA Hopper architecture's warp group matrix multiply instructions
    to efficiently access shared memory with the appropriate layout and access patterns.
    """

    def __init__(out self, val: Int64):
        """Initialize descriptor with raw 64-bit value.

        This constructor allows creating a descriptor directly from a 64-bit integer
        that already contains the properly formatted bit fields for the descriptor.

        The implicit attribute enables automatic conversion from `Int64` to `WGMMADescriptor`.

        Args:
            val: A 64-bit integer containing the complete descriptor bit layout.
        """
        self.desc = val

    @always_inline
    def _insert_bit[start_bit: Int](self, val: Int64) -> Self:
        """Insert bits at specified position in descriptor.

        Parameters:
            start_bit: Starting bit position.

        Args:
            val: Value to insert.

        Returns:
            Updated descriptor value with inserted bits.
        """
        return Self(self.desc | (val << Int64(start_bit)))

    @staticmethod
    def create[
        stride_byte_offset: Int,
        leading_byte_offset: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    ](
        smem_ptr: Pointer[Scalar[Self.dtype], address_space=.SHARED, ...],
    ) -> Self:
        """Create a descriptor for shared memory operand.

        Parameters:
            stride_byte_offset: Stride dimension offset in bytes.
            leading_byte_offset: Leading dimension stride in bytes.
            swizzle_mode: Memory access pattern mode.

        Args:
            smem_ptr: Pointer to shared memory operand.

        Returns:
            Initialized descriptor for the shared memory operand.
        """

        # TMA enumerates no swizzle, 32, 64, 128B as 0, 1, 2, 3.
        # WGMMA enumerates these as 0, 3, 2, 1.
        @__parameter
        def _convert_swizzle_enum[mode: Int32]() -> Int64:
            comptime if mode == 0:
                return mode.cast[.int64]()
            else:
                return (4 - mode).cast[.int64]()

        comptime swizzle = _convert_swizzle_enum[swizzle_mode._value]()
        var offset = Int64(0)
        var stride_dim = Int64(stride_byte_offset)
        var lead_dim = Int64(leading_byte_offset)

        # Extract 18 bits and ignore 4 LSB.
        var base_ptr = Int(smem_ptr)
        var start_address = (base_ptr & 0x3FFFF) >> 4

        # Start from LSB in case updated higher bits gets overwritten.
        var desc = Self(0)
        # bits [48 .. 32]
        # bits  0:14 address in share memory
        desc = desc._insert_bit[0](Int64(start_address))
        # bits 14:16 unused
        # bits 16:30 leading dim byte offset
        desc = desc._insert_bit[16](lead_dim)
        # bits 30:32 unused
        # bits 32:46 stride dim byte offset
        desc = desc._insert_bit[32](stride_dim)
        # bits 49:52 offset
        desc = desc._insert_bit[49](offset)
        # bits 53:62 unused
        # bits 62:64 swizzle type
        desc = desc._insert_bit[62](swizzle)

        return desc

    @always_inline
    def __iadd__(mut self, offset: Int):
        """Add offset to descriptor's base address in-place.

        Args:
            offset: Byte offset to add to base address.
        """
        self.desc += Int64((offset & 0x3FFFF) >> 4)

    @always_inline
    def __add__(self, offset: Int) -> Self:
        """Add offset to descriptor's base address.

        Args:
            offset: Byte offset to add to base address.

        Returns:
            New descriptor with updated base address.
        """
        return Self(self.desc + Int64((offset & 0x3FFFF) >> 4))


@always_inline
def wgmma_fence_aligned():
    """Inserts a memory fence for warp group matrix multiply operations.

    This ensures all prior shared memory accesses are visible before subsequent WGMMA operations.
    Must be called before starting a new sequence of WGMMA operations.
    """
    __mlir_op.`nvvm.wgmma.fence.aligned`[_type=None]()


@always_inline
def wgmma_commit_group_sync():
    """Commits pending warp group matrix multiply operations.

    This synchronizes the warp group and ensures all WGMMA operations have been committed.
    Must be called after a sequence of WGMMA operations before accessing results.
    """
    __mlir_op.`nvvm.wgmma.commit.group.sync.aligned`[_type=None]()


@always_inline
def wgmma_wait_group_sync[group: Int = 0]():
    """Waits for all pending warp group matrix multiply operations to complete.

    This synchronizes the warp group and ensures all WGMMA operations have finished executing.
    Must be called after commit and before accessing results.

    Parameters:
        group: The number of pending wgmma-groups to wait until.
    """
    inlined_assembly[
        "wgmma.wait_group.sync.aligned $0;", NoneType, constraints="n"
    ](group)


@always_inline
def wgmma_async[
    m: Int,
    n: Int,
    k: Int,
    c_dtype: DType,
    width: Int,
    /,
    *,
    a_type: DType,
    b_type: DType,
    accum_type: DType = c_dtype,
    layout_a: StaticString = "row",
    layout_b: StaticString = "col",
    scale_d: Int = 1,
    scale_a: Int = 1,
    scale_b: Int = 1,
](
    mat_a_desc: WGMMADescriptor,
    mat_b_desc: WGMMADescriptor,
    c_reg: StaticTuple[Scalar[c_dtype], width],
) -> type_of(c_reg):
    """Performs warp group async Matrix-multiply and accumulate (WGMMA) operation.

    This function executes an asynchronous matrix multiplication using warp group MMA instructions.
    It supports various data types including tensor float32, bfloat16, float16, float8, int8, and uint8.

    Parameters:
        m: Number of rows in matrix A and output matrix.
        n: Number of columns in matrix B and output matrix.
        k: Number of columns in matrix A / rows in matrix B.
        c_dtype: Data type of the output matrix C.
        width: Width of the StaticTuple register for matrix C.
        a_type: Data type of matrix A.
        b_type: Data type of matrix B.
        accum_type: Accumulation data type (defaults to c_dtype).
        layout_a: Memory layout for matrix A ("row" or "col").
        layout_b: Memory layout for matrix B ("row" or "col").
        scale_d: Scale factor for matrix C.
        scale_a: Scale factor for matrix A.
        scale_b: Scale factor for matrix B.

    Args:
        mat_a_desc: WGMMA descriptor for matrix A.
        mat_b_desc: WGMMA descriptor for matrix B.
        c_reg: StaticTuple containing matrix C values.

    Returns:
        `StaticTuple` containing the result of the matrix multiplication.

    Constraints:
        - The number of output registers must match the instruction shape:
          `(m * n // 128) * size_of(accum_type) == width * size_of(c_dtype)`.
        - Data type combinations must be compatible with hardware WGMMA instructions.
    """

    comptime assert (m * n // 128) * size_of[accum_type]() == width * size_of[
        c_dtype
    ](), String(
        "Number of output registers ",
        String(width),
        " don't match the instruction shape ",
        String(Index(m, n, k)),
    )

    comptime assert scale_d == 1 or scale_d == 0, (
        "Invalid scale in value of scaled_d '"
        + String(scale_d)
        + "' which is not supported. Only 1 or 0 is supported as the"
        " scale in values."
    )

    comptime assert scale_a == 1 or scale_a == -1, (
        "Invalid scale in value of scaled_a '"
        + String(scale_a)
        + "' which is not supported. Only 1 or -1 is supported as the"
        " scale in values."
    )

    comptime assert scale_b == 1 or scale_b == -1, (
        "Invalid scale in value of scaled_b '"
        + String(scale_b)
        + "' which is not supported. Only 1 or -1 is supported as the"
        " scale in values."
    )

    var desc_a_value = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i64](
        mat_a_desc.desc._mlir_value
    )
    var desc_b_value = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i64](
        mat_b_desc.desc._mlir_value
    )

    comptime layout_a_value = _get_kgen_string[layout_a]()
    comptime layout_b_value = _get_kgen_string[layout_b]()

    comptime type_d_value = __mlir_attr.`#nvvm.wgmma_type<f32>` if c_dtype == DType.float32 else _dtype_to_nvvm_wgmma_type[
        c_dtype, a_type
    ]()

    var llvmst = array_to_llvm_struct[c_dtype, width](c_reg)
    # TODO: Simplify with parametric alias
    var llvmres = __mlir_op.`nvvm.wgmma.mma_async`[
        shape=_get_shape[m, n, k](),
        typeA=_dtype_to_nvvm_wgmma_type[a_type](),
        typeB=_dtype_to_nvvm_wgmma_type[b_type](),
        typeD=type_d_value,
        scaleD=_to_nvvm_scale_out[scale_d](),
        scaleA=_to_nvvm_scale_in[scale_a](),
        scaleB=_to_nvvm_scale_in[scale_b](),
        layoutA=_to_nvvm_layout[layout_a](),
        layoutB=_to_nvvm_layout[layout_b](),
        _type=__mlir_deferred_type[
            `!llvm.struct<(`,
            +_get_kgen_string[_get_llvm_struct_fields[width, c_dtype]()](),
            `)>`,
        ],
    ](llvmst, desc_a_value, desc_b_value)

    return llvm_struct_to_array[c_dtype, width](llvmres)


@always_inline
def wgmma_async[
    m: Int,
    n: Int,
    k: Int,
    c_dtype: DType,
    width: SIMDLength,
    /,
    *,
    a_type: DType,
    b_type: DType,
    accum_type: DType = c_dtype,
    layout_a: StaticString = "row",
    layout_b: StaticString = "col",
    scale_d: Int = 1,
    scale_a: Int = 1,
    scale_b: Int = 1,
](
    mat_a_desc: WGMMADescriptor,
    mat_b_desc: WGMMADescriptor,
    c_reg: SIMD[c_dtype, width],
) -> type_of(c_reg):
    """Performs warp group async Matrix-multiply and accumulate (WGMMA) operation.

    This function executes an asynchronous matrix multiplication using warp group MMA instructions.
    It supports various data types including tensor float32, bfloat16, float16, float8, int8, and uint8.

    Parameters:
        m: Number of rows in matrix A and output matrix.
        n: Number of columns in matrix B and output matrix.
        k: Number of columns in matrix A / rows in matrix B.
        c_dtype: Data type of the output matrix C.
        width: Width of the SIMD register for matrix C.
        a_type: Data type of matrix A.
        b_type: Data type of matrix B.
        accum_type: Accumulation data type (defaults to c_dtype).
        layout_a: Memory layout for matrix A ("row" or "col").
        layout_b: Memory layout for matrix B ("row" or "col").
        scale_d: Scale factor for matrix C.
        scale_a: Scale factor for matrix A.
        scale_b: Scale factor for matrix B.

    Args:
        mat_a_desc: WGMMA descriptor for matrix A.
        mat_b_desc: WGMMA descriptor for matrix B.
        c_reg: SIMD register containing matrix C values.

    Returns:
        SIMD register containing the result of the matrix multiplication.

    Constraints:
        - The number of output registers must match the instruction shape:
          `(m * n // 128) * size_of(accum_type) == width * size_of(c_dtype)`.
        - Data type combinations must be compatible with hardware WGMMA instructions.
    """

    comptime assert (m * n // 128) * size_of[accum_type]() == width * size_of[
        c_dtype
    ](), String(
        "Number of output registers ",
        String(Int(width)),
        " don't match the instruction shape ",
        String(Index(m, n, k)),
    )

    comptime assert scale_d == 1 or scale_d == 0, (
        "Invalid scale in value of scaled_d '"
        + String(scale_d)
        + "' which is not supported. Only 1 or 0 is supported as the"
        " scale in values."
    )

    comptime assert scale_a == 1 or scale_a == -1, (
        "Invalid scale in value of scaled_a '"
        + String(scale_a)
        + "' which is not supported. Only 1 or -1 is supported as the"
        " scale in values."
    )

    comptime assert scale_b == 1 or scale_b == -1, (
        "Invalid scale in value of scaled_b '"
        + String(scale_b)
        + "' which is not supported. Only 1 or -1 is supported as the"
        " scale in values."
    )

    var desc_a_value = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i64](
        mat_a_desc.desc._mlir_value
    )
    var desc_b_value = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i64](
        mat_b_desc.desc._mlir_value
    )

    comptime layout_a_value = _get_kgen_string[layout_a]()
    comptime layout_b_value = _get_kgen_string[layout_b]()
    comptime type_d_value = __mlir_attr.`#nvvm.wgmma_type<f32>` if c_dtype == DType.float32 else _dtype_to_nvvm_wgmma_type[
        c_dtype, a_type
    ]()

    var llvmst = simd_to_llvm_struct[c_dtype, width](c_reg)
    # TODO: Simplify with parametric alias
    var llvmres = __mlir_op.`nvvm.wgmma.mma_async`[
        shape=_get_shape[m, n, k](),
        typeA=_dtype_to_nvvm_wgmma_type[a_type](),
        typeB=_dtype_to_nvvm_wgmma_type[b_type](),
        typeD=type_d_value,
        scaleD=_to_nvvm_scale_out[scale_d](),
        scaleA=_to_nvvm_scale_in[scale_a](),
        scaleB=_to_nvvm_scale_in[scale_b](),
        layoutA=_to_nvvm_layout[layout_a](),
        layoutB=_to_nvvm_layout[layout_b](),
        _type=__mlir_deferred_type[
            `!llvm.struct<(`,
            +_get_kgen_string[_get_llvm_struct_fields[width, c_dtype]()](),
            `)>`,
        ],
    ](llvmst, desc_a_value, desc_b_value)

    return llvm_struct_to_simd[c_dtype, width](llvmres)


@always_inline
def wgmma_async[
    m: Int,
    n: Int,
    k: Int,
    a_dtype: DType,
    c_dtype: DType,
    frag_a_width: SIMDLength,
    frag_c_width: SIMDLength,
    /,
    *,
    a_type: DType,
    b_type: DType,
    accum_type: DType = c_dtype,
    layout_a: StaticString = "row",
    layout_b: StaticString = "col",
    scale_d: Int = 1,
    scale_a: Int = 1,
    scale_b: Int = 1,
](
    mat_a_frag: SIMD[a_dtype, frag_a_width],
    mat_b_desc: WGMMADescriptor,
    c: SIMD[c_dtype, frag_c_width],
) -> type_of(c):
    """Performs warp group async Matrix-multiply and accumulate (WGMMA) operation.

    Parameters:
        m: Number of rows in output matrix.
        n: Number of columns in output matrix.
        k: Inner dimension for matrix multiplication.
        a_dtype: Data type of matrix A fragment.
        c_dtype: Data type of output matrix C.
        frag_a_width: Width of matrix A fragment.
        frag_c_width: Width of output matrix C fragment.
        a_type: Data type of matrix A.
        b_type: Data type of matrix B.
        accum_type: Data type used for accumulation (defaults to c_dtype).
        layout_a: Layout of matrix A ("row" or "col", defaults to "row").
        layout_b: Layout of matrix B ("row" or "col", defaults to "col").
        scale_d: Scale factor for output matrix C (defaults to 1).
        scale_a: Scale factor for matrix A (defaults to 1).
        scale_b: Scale factor for matrix B (defaults to 1).

    Args:
        mat_a_frag: Fragment containing matrix A data.
        mat_b_desc: Descriptor for matrix B data.
        c: Fragment containing matrix C data.

    Returns:
        Updated matrix C fragment after WGMMA operation.

    Currently only supports:
    - m=64, k=16.
    - BF16 input types.
    - FP32 accumulation.
    - Row major matrix A.
    - Column major matrix B (or row major for BF16).
    """
    comptime assert (m * n // 128) * size_of[
        accum_type
    ]() == frag_c_width * size_of[c_dtype](), String(
        "Number of output registers ",
        String(Int(frag_c_width)),
        " don't match the instruction shape ",
        String(Index(m, n, k)),
    )

    comptime assert (m * k // 128) * size_of[
        a_type
    ]() == frag_a_width * size_of[a_dtype](), String(
        "Number of input a registers ",
        String(Int(frag_a_width)),
        " don't match the instruction shape ",
        String(Index(m, n, k)),
    )
    # for now, limited support
    comptime assert m == 64
    comptime assert k == 16
    comptime assert a_type == .bfloat16
    comptime assert b_type == .bfloat16
    comptime assert accum_type == .float32
    comptime assert c_dtype == .float32
    comptime assert layout_a == "row"
    comptime assert layout_b == "col" or (
        layout_b == "row" and b_type == .bfloat16
    )

    var desc_b_value = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i64](
        mat_b_desc.desc._mlir_value
    )
    comptime trans_b = 1 if layout_b == "row" else 0

    comptime assert (
        m == 64
        and k == 16
        and a_type == b_type == .bfloat16
        and accum_type == c_dtype == .float32
    ), "unsupported config"
    var a0 = bitcast[.uint32, 1](
        SIMD[.bfloat16, 2](
            rebind[BFloat16](mat_a_frag[0]),
            rebind[BFloat16](mat_a_frag[1]),
        )
    )
    var a1 = bitcast[.uint32, 1](
        SIMD[.bfloat16, 2](
            rebind[BFloat16](mat_a_frag[2]),
            rebind[BFloat16](mat_a_frag[3]),
        )
    )
    var a2 = bitcast[.uint32, 1](
        SIMD[.bfloat16, 2](
            rebind[BFloat16](mat_a_frag[4]),
            rebind[BFloat16](mat_a_frag[5]),
        )
    )
    var a3 = bitcast[.uint32, 1](
        SIMD[.bfloat16, 2](
            rebind[BFloat16](mat_a_frag[6]),
            rebind[BFloat16](mat_a_frag[7]),
        )
    )

    comptime input_reg_spec = _str_iota[n // 2, prefix="$"]()
    comptime input_constraints_prefix = "=f," * (n // 2)
    comptime input_constraints_suffix = _str_iota[n // 2, sep=","]()
    comptime constraints = input_constraints_prefix + "r,r,r,r,l,n,n,n,n," + input_constraints_suffix

    # fmt: off
    comptime if n == 8:
        var r = inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $9, 0;
                wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16
                {""" + input_reg_spec + """},
                 {$4, $5, $6, $7},
                 $8, p, $10, $11, $12;
            }""",
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints = constraints,
        ](
            a0, a1, a2, a3,
            desc_b_value,
            scale_d, scale_a, scale_b, trans_b,
            c[0], c[1], c[2], c[3],
        )

        return rebind[type_of(c)](
            SIMD[.float32, 4](r[0], r[1], r[2], r[3])
        )
    elif n == 16:
        var r = inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $13, 0;
                wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16
                {""" + input_reg_spec + """},
                 {$8, $9, $10, $11},
                 $12, p, $14, $15, $16;
            }""",
            _RegisterPackType[
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
            ],
            constraints = constraints,
        ](
            a0, a1, a2, a3,
            desc_b_value,
            scale_d, scale_a, scale_b, trans_b,
            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
        )

        return rebind[type_of(c)](
            SIMD[.float32, 8](
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
            )
        )
    elif n == 32:
        var r = inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $21, 0;
                wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16
                {""" + input_reg_spec + """},
                 {$16, $17, $18, $19},
                 $20, p, $22, $23, $24;
            }""",
            _RegisterPackType[
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
            ],
            constraints = constraints,
        ](
            a0, a1, a2, a3,
            desc_b_value,
            scale_d, scale_a, scale_b, trans_b,
            c[0],  c[1],  c[2],  c[3],  c[4],  c[5],  c[6],  c[7],
            c[8],  c[9],  c[10], c[11], c[12], c[13], c[14], c[15],
        )

        return rebind[type_of(c)](
            SIMD[.float32, 16](
                r[0],  r[1],  r[2],  r[3],  r[4],  r[5],  r[6],  r[7],
                r[8],  r[9],  r[10], r[11], r[12], r[13], r[14], r[15],
            )
        )
    elif n == 64:
        var r = inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $37, 0;
                wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
                {""" + input_reg_spec + """},
                 {$32, $33, $34, $35},
                 $36, p, $38, $39, $40;
            }""",
            _RegisterPackType[
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
            ],
            constraints = constraints,
        ](
            a0, a1, a2, a3,
            desc_b_value,
            scale_d, scale_a, scale_b, trans_b,
            c[0],  c[1],  c[2],  c[3],  c[4],  c[5],  c[6],  c[7],
            c[8],  c[9],  c[10], c[11], c[12], c[13], c[14], c[15],
            c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23],
            c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31],
        )

        return rebind[type_of(c)](
            SIMD[.float32, 32](
                r[0],  r[1],  r[2],  r[3],  r[4],  r[5],  r[6],  r[7],
                r[8],  r[9],  r[10], r[11], r[12], r[13], r[14], r[15],
                r[16], r[17], r[18], r[19], r[20], r[21], r[22], r[23],
                r[24], r[25], r[26], r[27], r[28], r[29], r[30], r[31],
            )
        )
    elif n == 128:
        var r = inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $69, 0;
                wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16
                {""" + input_reg_spec + """},
                 {$64, $65, $66, $67},
                 $68, p, $70, $71, $72;
            }""",
            _RegisterPackType[
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
            ],
            constraints = constraints,
        ](
            a0, a1, a2, a3,
            desc_b_value,
            scale_d, scale_a, scale_b, trans_b,
            c[0],  c[1],  c[2],  c[3],  c[4],  c[5],  c[6],  c[7],
            c[8],  c[9],  c[10], c[11], c[12], c[13], c[14], c[15],
            c[16], c[17], c[18], c[19], c[20], c[21], c[22], c[23],
            c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31],
            c[32], c[33], c[34], c[35], c[36], c[37], c[38], c[39],
            c[40], c[41], c[42], c[43], c[44], c[45], c[46], c[47],
            c[48], c[49], c[50], c[51], c[52], c[53], c[54], c[55],
            c[56], c[57], c[58], c[59], c[60], c[61], c[62], c[63],
        )
        return rebind[type_of(c)](
            SIMD[.float32, 64](
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17],
                r[18], r[19], r[20], r[21], r[22], r[23], r[24], r[25],
                r[26], r[27], r[28], r[29], r[30], r[31], r[32], r[33],
                r[34], r[35], r[36], r[37], r[38], r[39], r[40], r[41],
                r[42], r[43], r[44], r[45], r[46], r[47], r[48], r[49],
                r[50], r[51], r[52], r[53], r[54], r[55], r[56], r[57],
                r[58], r[59], r[60], r[61], r[62], r[63],
            )
        )
    elif n == 256:
        var r = inlined_assembly[
            """
            {
                .reg .pred p;
                setp.ne.b32 p, $133, 0;
                wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
                {""" + input_reg_spec + """},
                 {$128, $129, $130, $131},
                 $132, p, $134, $135, $136;
            }""",
            _RegisterPackType[
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
                Float32, Float32, Float32, Float32,
            ],
            constraints = constraints,
        ](
            a0, a1, a2, a3,
            desc_b_value,
            scale_d, scale_a, scale_b, trans_b,
            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
            c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18],
            c[19], c[20], c[21], c[22], c[23], c[24], c[25], c[26], c[27],
            c[28], c[29], c[30], c[31], c[32], c[33], c[34], c[35], c[36],
            c[37], c[38], c[39], c[40], c[41], c[42], c[43], c[44], c[45],
            c[46], c[47], c[48], c[49], c[50], c[51], c[52], c[53], c[54],
            c[55], c[56], c[57], c[58], c[59], c[60], c[61], c[62], c[63],
            c[64], c[65], c[66], c[67], c[68], c[69], c[70], c[71], c[72],
            c[73], c[74], c[75], c[76], c[77], c[78], c[79], c[80], c[81],
            c[82], c[83], c[84], c[85], c[86], c[87], c[88], c[89], c[90],
            c[91], c[92], c[93], c[94], c[95], c[96], c[97], c[98], c[99],
            c[100], c[101], c[102], c[103], c[104], c[105], c[106], c[107],
            c[108], c[109], c[110], c[111], c[112], c[113], c[114], c[115],
            c[116], c[117], c[118], c[119], c[120], c[121], c[122], c[123],
            c[124], c[125], c[126], c[127],
        )

        return rebind[type_of(c)](
            SIMD[.float32, 128](
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
                r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
                r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27],
                r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36],
                r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45],
                r[46], r[47], r[48], r[49], r[50], r[51], r[52], r[53], r[54],
                r[55], r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                r[64], r[65], r[66], r[67], r[68], r[69], r[70], r[71], r[72],
                r[73], r[74], r[75], r[76], r[77], r[78], r[79], r[80], r[81],
                r[82], r[83], r[84], r[85], r[86], r[87], r[88], r[89], r[90],
                r[91], r[92], r[93], r[94], r[95], r[96], r[97], r[98], r[99],
                r[100], r[101], r[102], r[103], r[104], r[105], r[106], r[107],
                r[108], r[109], r[110], r[111], r[112], r[113], r[114], r[115],
                r[116], r[117], r[118], r[119], r[120], r[121], r[122], r[123],
                r[124], r[125], r[126], r[127],
            )
        )
    else:
        comptime assert False, String("the n value '", n, "' is not valid")
    # fmt: on


@always_inline("nodebug")
def _str_iota[
    count: Int, *, prefix: String = String(), sep: String = ", "
]() -> String:
    return _str_iota_impl[count, prefix=prefix, sep=sep]()


@always_inline("nodebug")
def _str_iota_impl[
    count: Int, *, prefix: String = String(), sep: String = ", "
]() -> String:
    var s = String()
    for i in range(count):
        s += prefix + String(i)
        if i < count - 1:
            s += StringSlice(sep)
    return s
