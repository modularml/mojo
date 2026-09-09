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
"""This module includes utilities for working with the SM100 MMA instructions."""

from std.os import abort
from std.sys import size_of
from std.sys._assembly import inlined_assembly
from std.sys.info import _has_blackwell_tcgen05

from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.mma_operand_descriptor import MMAOperandDescriptor

from std.utils.index import IndexList
from std.hashlib.hasher import Hasher

# ===----------------------------------------------------------------------=== #
# MMA Instruction Descriptor
# ===----------------------------------------------------------------------=== #


@fieldwise_init("implicit")
struct UMMAKind(Equatable, Hashable, TrivialRegisterPassable, Writable):
    """Struct for UMMA instruction types.

    This struct defines the different types of UMMA instructions that is supported by BlackWell.
    """

    var _value: Int32

    comptime KIND_TF32 = Self(0)
    """TF32 type."""

    comptime KIND_F16 = Self(2)
    """F16 type."""

    comptime KIND_F8F6F4 = Self(3)
    """F8F6F4 type."""

    comptime KIND_I8 = Self(4)
    """I8 type."""

    comptime KIND_MXF8F6F4 = Self(5)
    """MXF8F6F4 type."""

    comptime KIND_MXF4 = Self(6)
    """MXF4 type."""

    comptime KIND_MXF4NVF4 = Self(7)
    """MXF4NVF4 type."""

    @always_inline("nodebug")
    def __int__(self) -> Int:
        """Convert UMMA kind to an integer value.

        Returns:
            The integer value representing the UMMA instruction type.
        """
        return Int(self._value)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        """Check if two UMMA kinds are equal.

        Args:
            other: The other UMMA kind to compare with.

        Returns:
            True if the UMMA kinds are equal, False otherwise.
        """
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        """Check if two UMMA kinds are not equal.

        Args:
            other: The other UMMA kind to compare with.

        Returns:
            True if the UMMA kinds are not equal, False otherwise.
        """
        return self._value != other._value

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        """Write the UMMA kind to a writer.

        Args:
            writer: The writer to write the UMMA kind to.
        """
        if self == Self.KIND_TF32:
            writer.write("kind::tf32")
        elif self == Self.KIND_F16:
            writer.write("kind::f16")
        elif self == Self.KIND_F8F6F4:
            writer.write("kind::f8f6f4")
        elif self == Self.KIND_MXF8F6F4:
            writer.write("kind::mxf8f6f4")
        elif self == Self.KIND_I8:
            writer.write("kind::i8")
        elif self == Self.KIND_MXF4:
            writer.write("kind::mxf4")
        elif self == Self.KIND_MXF4NVF4:
            writer.write("kind::mxf4nvf4")
        else:
            writer.write("kind::unknown")


@always_inline
def _constrained_mma_m[
    mma_m: Int,
    mma_m_valid: Tuple[Int, Int],
    mma_kind: UMMAKind,
    /,
    *,
    use_cta_pair: Bool = False,
]():
    """Constrain the MMA M value based on the MMA valid range and use peer cta flag.

    This function constrains the MMA M value to be within the valid range and returns the constrained value.
    """

    comptime using_pair_string = (
        " when using pair cta." if use_cta_pair else " when not using pair cta."
    )

    comptime assert mma_m in mma_m_valid, String(
        "Invalid MMA M: ",
        mma_m,
        " , MMA M has to be ",
        mma_m_valid[0],
        " or ",
        mma_m_valid[1],
        " for ",
        mma_kind,
        using_pair_string,
    )


@always_inline
def _constrained_mma_n[
    mma_n: Int,
    mma_n_range: Tuple[Int, Int],
    multiple_of: Int,
    mma_kind: UMMAKind,
    /,
    *,
    use_cta_pair: Bool = False,
]():
    """Constrain the MMA N value based on the MMA valid range and use peer cta flag.

    This function constrains the MMA N value to be within the valid range and returns the constrained value.
    """

    comptime using_pair_string = (
        " when using pair cta." if use_cta_pair else " when not using pair cta."
    )

    comptime lower_bound = mma_n_range[0]
    comptime upper_bound = mma_n_range[1]

    comptime assert (
        mma_n >= lower_bound
        and mma_n <= upper_bound
        and mma_n % multiple_of == 0
    ), String(
        "Invalid MMA N: ",
        mma_n,
        " ,MMA N has to be between ",
        lower_bound,
        " and ",
        upper_bound,
        " and a multiple of ",
        lower_bound % 8,
        " for ",
        mma_kind,
        using_pair_string,
    )


@always_inline
def _get_f16_mma_shape[
    output_shape: IndexList[2, element_type=.uint32],
    /,
    *,
    use_cta_pair: Bool = False,
]() -> IndexList[3, element_type=.uint32]:
    """Get the shape of the MMA instruction for F16 MMA kind.

    This function returns the shape of the MMA instruction for F16 MMA kind.
    """
    comptime mma_m = output_shape[0]
    comptime mma_n = output_shape[1]

    comptime if not use_cta_pair:
        _constrained_mma_m[
            mma_m,
            (64, 128),
            UMMAKind.KIND_F16,
            use_cta_pair=use_cta_pair,
        ]()

        comptime if mma_m == 64:
            _constrained_mma_n[
                mma_n,
                (8, 256),
                8,
                UMMAKind.KIND_F16,
                use_cta_pair=use_cta_pair,
            ]()
            return IndexList[3, element_type=.uint32](mma_m, mma_n, 16)
        elif mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (16, 256),
                16,
                UMMAKind.KIND_F16,
                use_cta_pair=use_cta_pair,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 16)
        else:
            comptime assert False, String("Invalid MMA shape: ", mma_m, mma_n)

    else:
        _constrained_mma_m[
            mma_m,
            (128, 256),
            UMMAKind.KIND_F16,
            use_cta_pair=use_cta_pair,
        ]()

        comptime if mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (32, 256),
                32,
                UMMAKind.KIND_F16,
                use_cta_pair=use_cta_pair,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 16)
        elif mma_m == 256:
            _constrained_mma_n[
                mma_n,
                (32, 256),
                32,
                UMMAKind.KIND_F16,
                use_cta_pair=use_cta_pair,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 16)
        else:
            comptime assert False, String("Invalid MMA shape: ", mma_m, mma_n)


@always_inline
def _get_tf32_mma_shape[
    output_shape: IndexList[2, element_type=.uint32],
    /,
    *,
    use_pair_cta: Bool = False,
]() -> IndexList[3, element_type=.uint32]:
    """Get the shape of the MMA instruction for TF32 MMA kind.

    This function returns the shape of the MMA instruction for TF32 MMA kind.
    """

    comptime mma_m = output_shape[0]
    comptime mma_n = output_shape[1]

    comptime if not use_pair_cta:
        _constrained_mma_m[
            mma_m,
            (64, 128),
            UMMAKind.KIND_TF32,
            use_cta_pair=use_pair_cta,
        ]()

        comptime if mma_m == 64:
            _constrained_mma_n[
                mma_n,
                (8, 256),
                8,
                UMMAKind.KIND_TF32,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 8)
        elif mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (16, 256),
                16,
                UMMAKind.KIND_TF32,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 8)
        else:
            comptime assert False, String("Invalid MMA shape: ", mma_m, mma_n)
    else:
        _constrained_mma_m[
            mma_m,
            (128, 256),
            UMMAKind.KIND_TF32,
            use_cta_pair=use_pair_cta,
        ]()

        comptime if mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (32, 256),
                32,
                UMMAKind.KIND_TF32,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 8)
        elif mma_m == 256:
            _constrained_mma_n[
                mma_n,
                (32, 256),
                32,
                UMMAKind.KIND_TF32,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 8)
        else:
            comptime assert False, String("Invalid MMA shape: ", mma_m, mma_n)


@always_inline
def _get_f8f6f4_mma_shape[
    output_shape: IndexList[2, element_type=.uint32],
    /,
    *,
    use_pair_cta: Bool = False,
]() -> IndexList[3, element_type=.uint32]:
    """Get the shape of the MMA instruction for F8F6F4 MMA kind.

    This function returns the shape of the MMA instruction for F8F6F4 MMA kind.
    """

    comptime mma_m = output_shape[0]
    comptime mma_n = output_shape[1]

    comptime if not use_pair_cta:
        _constrained_mma_m[
            mma_m,
            (64, 128),
            UMMAKind.KIND_F8F6F4,
            use_cta_pair=use_pair_cta,
        ]()

        comptime if mma_m == 64:
            _constrained_mma_n[
                mma_n,
                (8, 256),
                8,
                UMMAKind.KIND_F8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 32)
        elif mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (16, 256),
                16,
                UMMAKind.KIND_F8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 32)

        else:
            comptime assert False, String("Invalid MMA shape: ", mma_m, mma_n)

    else:
        _constrained_mma_m[
            mma_m,
            (128, 256),
            UMMAKind.KIND_F8F6F4,
            use_cta_pair=use_pair_cta,
        ]()

        comptime if mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (32, 256),
                32,
                UMMAKind.KIND_F8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 32)
        elif mma_m == 256:
            _constrained_mma_n[
                mma_n,
                (32, 256),
                32,
                UMMAKind.KIND_F8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 32)
        else:
            comptime assert False, String("Invalid MMA shape: ", mma_m, mma_n)


@always_inline
def _get_mxf8f6f4_mma_shape[
    output_shape: IndexList[2, element_type=.uint32],
    /,
    *,
    use_pair_cta: Bool = False,
]() -> IndexList[3, element_type=.uint32]:
    """Get the shape of the MMA instruction for MXF8F6F4 MMA kind.

    This function returns the shape of the MMA instruction for MXF8F6F4 MMA kind.
    """

    comptime mma_m = output_shape[0]
    comptime mma_n = output_shape[1]

    comptime if not use_pair_cta:
        _constrained_mma_m[
            mma_m,
            (128, -1),
            UMMAKind.KIND_MXF8F6F4,
            use_cta_pair=use_pair_cta,
        ]()

        comptime if mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (8, 256),
                8,
                UMMAKind.KIND_MXF8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 32)

        else:
            comptime assert False, String("Invalid MMA shape: ", mma_m, mma_n)

    else:
        _constrained_mma_m[
            mma_m,
            (128, 256),
            UMMAKind.KIND_MXF8F6F4,
            use_cta_pair=use_pair_cta,
        ]()

        comptime if mma_m == 128:
            _constrained_mma_n[
                mma_n,
                (16, 256),
                16,
                UMMAKind.KIND_MXF8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 32)
        elif mma_m == 256:
            _constrained_mma_n[
                mma_n,
                (16, 256),
                16,
                UMMAKind.KIND_MXF8F6F4,
                use_cta_pair=use_pair_cta,
            ]()

            return IndexList[3, element_type=.uint32](mma_m, mma_n, 32)
        else:
            comptime assert False, String("Invalid MMA shape: ", mma_m, mma_n)


@always_inline
def _mxf8f6f4_operand_format[dtype: DType]() -> UInt32:
    """Encode one operand's format for the `kind::mxf8f6f4` descriptor.

    The A and B format fields are three bits each and independent, which is
    what makes mixed-precision operands (W4A8: E4M3 activations against E2M1
    weights) expressible under this kind. `uint8` stands in for E2M1, matching
    the placeholder used throughout the block-scaled kernels; under this kind
    an E2M1 operand is read from shared memory padded to one byte per element
    (`PACKED_FP4_ALIGN16B`), not from the densely packed layout the FP4-only
    kinds read.
    """
    comptime if dtype == .float8_e4m3fn:
        return 0
    elif dtype == .float8_e5m2:
        return 1
    else:
        comptime assert dtype == .uint8, String(
            "Unsupported operand type for kind::mxf8f6f4: ", dtype
        )
        return 5


struct UMMAInsDescriptor[
    mma_kind: UMMAKind,
](TrivialRegisterPassable):
    """Descriptor for UMMA instructions.

    This struct represents a descriptor that encodes information about UMMA instructions.
    The descriptor contains the following bit fields:

    - Sparsity (2 bits): The sparsity of the input matrices. Currently defaults to dense matrices.
    - Saturate for integer types (1 bits): Whether to saturate the result for integer types. Currently not supported.
    - Matrix D type (2 bits): Data type of matrix D.
    - Matrix A type (3 bits): Data type of matrix A.
    - Matrix B type (3 bits): Data type of matrix B.
    - Negate A matrix (1 bit): Whether to negate matrix A. Currently defaults to False.
    - Negate B matrix (1 bit): Whether to negate matrix B. Currently defaults to False.
    - Transpose A (1 bit): Whether to transpose matrix A.
    - Transpose B (1 bit): Whether to transpose matrix B.
    - N, Dimension of Matrix B (6 bits): Number of columns in matrix B. 3 LSBs are unused.
    - M, Dimension of Matrix A (6 bits): Number of rows in matrix A. 3 LSBs are unused.

    Parameters:
        mma_kind: The kind of UMMA instruction.

    See: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05%2520mma#tcgen05-instuction-desc-kind-tf32-f16-f8f6f4
    """

    var desc: UInt32
    """The 32-bit descriptor value that encodes UMMA instruction information.

    This field stores the complete descriptor with all bit fields packed into a single 32-bit integer:
    - Bits 0-1: Sparsity selector(2 bits)
    - Bits 2: Sparsity enable(1 bit)
    - Bits 3: Saturate for integer types (1 bit)
    - Bits 4-5: Matrix D type (2 bits)
    - Bits 6: Reserved (1 bit)
    - Bits 7-9: Matrix A type (3 bits)
    - Bits 10-12: Matrix B type (3 bits)
    - Bits 13: Negate A matrix (1 bit)
    - Bits 14: Negate B matrix (1 bit)
    - Bits 15: Transpose A (1 bit)
    - Bits 16: Transpose B (1 bit)
    - Bits 17-22: N, Dimension of Matrix B (6 bits)
    - Bits 23: Reserved (1 bit)
    - Bits 24-28: M, Dimension of Matrix A (5 bits)
    - Bits 29: Reserved (1 bit)
    - Bits 30-31: Maximum shift while attempting B matrix (2 bits)
    """

    def __init__(out self, value: UInt32):
        """Initialize descriptor with raw 32-bit value.

        This constructor allows creating a descriptor directly from a 32-bit integer
        that already contains the properly formatted bit fields for the descriptor.

        Args:
            value: A 32-bit integer containing the complete descriptor bit layout.
        """
        self.desc = value

    @staticmethod
    def _insert_bit[start_bit: Int](desc: UInt32, val: UInt32) -> UInt32:
        """Insert bits at specified position in descriptor.

        Parameters:
            start_bit: Starting bit position.

        Args:
            desc: Descriptor value.
            val: Value to insert.

        Returns:
            Updated descriptor value with inserted bits.
        """

        return desc | (val << UInt32(start_bit))

    @staticmethod
    def _create_tf32_desc[
        d_type: DType, a_type: DType, b_type: DType
    ]() -> UInt32:
        """Create a descriptor for TF32 UMMA instructions.

        This function creates a descriptor for TF32 UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.

        Returns:
            A 32-bit integer containing the descriptor bit layout.
        """

        comptime assert (
            d_type == .float32 and a_type == .float32 and b_type == .float32
        ), String(
            "Invalid operand data type for UMMA instruction: ",
            d_type,
            " and ",
            a_type,
            " and ",
            b_type,
        )

        comptime d_type_bit = Self._insert_bit[4](0x0, 0x1)
        comptime a_type_bit = Self._insert_bit[7](d_type_bit, 0x2)
        comptime desc = Self._insert_bit[10](a_type_bit, 0x2)

        return desc

    @staticmethod
    def _create_f16_desc[
        d_type: DType, a_type: DType, b_type: DType
    ]() -> UInt32:
        """Create a descriptor for F16 UMMA instructions.

        This function creates a descriptor for F16 UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.

        Returns:
            A 32-bit integer containing the descriptor bit layout.
        """

        comptime available_d_types = (DType.float32, DType.float16)
        comptime available_operand_types = (DType.bfloat16, DType.float16)

        comptime assert d_type in available_d_types, String(
            "Invalid d data type for UMMA instruction: ", d_type
        )

        comptime assert (
            a_type in available_operand_types
            and b_type in available_operand_types
        ), String(
            "Invalid operand data type for UMMA instruction: ",
            a_type,
            " and ",
            b_type,
        )

        comptime d_type_bit = Self._insert_bit[4](
            0x0, UInt32(1) if d_type == .float32 else UInt32(0)
        )
        comptime a_type_bit = Self._insert_bit[7](
            d_type_bit, UInt32(1) if a_type == .bfloat16 else UInt32(0)
        )
        comptime desc = Self._insert_bit[10](
            a_type_bit, UInt32(1) if b_type == .bfloat16 else UInt32(0)
        )

        return desc

    @staticmethod
    def _create_f8f6f4_desc[
        d_type: DType, a_type: DType, b_type: DType
    ]() -> UInt32:
        """Create a descriptor for F8F6F4 UMMA instructions.

        This function creates a descriptor for F8F6F4 UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.

        Returns:
            A 32-bit integer containing the descriptor bit layout.
        """

        comptime available_d_types = (DType.float16, DType.float32)
        comptime available_operand_types = (
            DType.float8_e4m3fn,
            DType.float8_e5m2,
        )

        comptime assert d_type in available_d_types, String(
            "Invalid d data type for UMMA instruction: ", d_type
        )

        comptime assert (
            a_type in available_operand_types
            and b_type in available_operand_types
        ), String(
            "Currently only support E4M3 and E5M2 for UMMA kind: ",
            Self.mma_kind,
        )

        comptime d_type_bit = Self._insert_bit[4](
            0x0, UInt32(1) if d_type == .float32 else UInt32(0)
        )

        comptime a_type_bit = Self._insert_bit[7](
            d_type_bit, UInt32(1) if a_type == .float8_e5m2 else UInt32(0)
        )
        comptime desc = Self._insert_bit[10](
            a_type_bit, UInt32(1) if b_type == .float8_e5m2 else UInt32(0)
        )

        return desc

    @staticmethod
    def _create_mxf8f6f4_desc[
        d_type: DType, a_type: DType, b_type: DType, scale_type: DType
    ]() -> UInt32:
        """Create a descriptor for MXF8F6F4 UMMA instructions.

        This function creates a descriptor for MXF8F6F4 UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.
            scale_type: The data type of the scale factors.

        Returns:
            A 32-bit integer containing the descriptor bit layout.
        """

        comptime available_d_types = (DType.float32,)
        comptime available_scale_types = (DType.float8_e8m0fnu,)

        comptime assert d_type in available_d_types, String(
            "Invalid d data type for UMMA instruction: ", d_type
        )

        comptime assert scale_type in available_scale_types, String(
            "Invalid scale data type for UMMA instruction: ", scale_type
        )

        comptime a_type_bit = Self._insert_bit[7](
            0x0, _mxf8f6f4_operand_format[a_type]()
        )

        comptime b_type_bit = Self._insert_bit[10](
            a_type_bit, _mxf8f6f4_operand_format[b_type]()
        )

        comptime desc = Self._insert_bit[23](b_type_bit, 1)

        return desc

    @staticmethod
    def _create_mxf4_mxf4nvf4_desc[
        d_type: DType, a_type: DType, b_type: DType, scale_type: DType
    ]() -> UInt32:
        """Create a descriptor for MXF4/MXF4NVF4 UMMA instructions.

        This function creates a descriptor for MXF4/MXF4NVF4 UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.
            scale_type: The data type of the scale factors.

        Returns:
            A 32-bit integer containing the descriptor bit layout.
        """

        comptime available_d_types = (DType.float32,)
        comptime available_operand_types = (
            DType.uint8,  # TODO: (KERN-2238) replace with FP4-E2M1
        )
        comptime available_scale_types = (
            DType.float8_e4m3fn,
            DType.float8_e8m0fnu,
        )

        comptime assert d_type in available_d_types, String(
            "Invalid d data type for UMMA instruction: ", d_type
        )

        comptime assert (
            a_type in available_operand_types
            and b_type in available_operand_types
        ), String("Currently only support E2M1 for UMMA kind: ", Self.mma_kind)

        comptime assert scale_type in available_scale_types, String(
            "Invalid scale data type for UMMA instruction: ", scale_type
        )

        comptime a_type_bit = Self._insert_bit[7](0x0, 1)

        comptime b_type_bit = Self._insert_bit[10](a_type_bit, 1)

        comptime desc = Self._insert_bit[23](
            b_type_bit,
            UInt32(0) if scale_type == .float8_e4m3fn else UInt32(1),
        )

        return desc

    @staticmethod
    def create[
        d_type: DType,
        a_type: DType,
        b_type: DType,
        output_shape: IndexList[2, element_type=.uint32],
        /,
        *,
        transpose_a: Bool = False,
        transpose_b: Bool = True,
    ]() -> Self:
        """Create a descriptor for UMMA instructions.

        This function creates a descriptor for UMMA instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.
            output_shape: The shape of the output matrix.
            transpose_a: Whether to transpose matrix A.
            transpose_b: Whether to transpose matrix B.

        Returns:
            A 32-bit integer containing the complete descriptor bit layout.
        """

        comptime M_bit = Self._insert_bit[17](0x0, UInt32(output_shape[1] >> 3))
        comptime desc = Self._insert_bit[24](
            M_bit, UInt32(output_shape[0] >> 4)
        )

        comptime transpose_a_bit = Self._insert_bit[15](
            0x0, UInt32(1) if transpose_a else UInt32(0)
        )
        comptime transpose_bit = Self._insert_bit[16](
            transpose_a_bit, UInt32(0) if transpose_b else UInt32(1)
        )

        comptime if Self.mma_kind == UMMAKind.KIND_TF32:
            return Self(
                desc
                | Self._create_tf32_desc[d_type, a_type, b_type]()
                | transpose_bit
            )

        elif Self.mma_kind == UMMAKind.KIND_F16:
            return Self(
                desc
                | Self._create_f16_desc[d_type, a_type, b_type]()
                | transpose_bit
            )

        elif Self.mma_kind == UMMAKind.KIND_F8F6F4:
            return Self(
                desc
                | Self._create_f8f6f4_desc[d_type, a_type, b_type]()
                | transpose_bit
            )
        else:
            comptime assert False, String(
                "Unsupported UMMA kind: ", Self.mma_kind
            )

    @staticmethod
    def create[
        d_type: DType,
        a_type: DType,
        b_type: DType,
        scale_type: DType,
        output_shape: IndexList[2, element_type=.uint32],
        /,
        *,
        transpose_a: Bool = False,
        transpose_b: Bool = True,
    ]() -> Self:
        """Create a descriptor for UMMA MXF8F6F4 instructions.

        This function creates a descriptor for UMMA MXF8F6F4 instructions based on the provided parameters.

        Parameters:
            d_type: The data type of matrix D.
            a_type: The data type of matrix A.
            b_type: The data type of matrix B.
            scale_type: The data type of the scale factors (only applicable to MXF8F6F4).
            output_shape: The shape of the output matrix.
            transpose_a: Whether to transpose matrix A.
            transpose_b: Whether to transpose matrix B.

        Returns:
            A 32-bit integer containing the complete descriptor bit layout.
        """

        comptime M_bit = Self._insert_bit[17](0x0, UInt32(output_shape[1] >> 3))
        comptime desc = Self._insert_bit[27](
            M_bit, UInt32(output_shape[0] >> 7)
        )

        comptime transpose_a_bit = Self._insert_bit[15](
            0x0, UInt32(1) if transpose_a else UInt32(0)
        )
        comptime transpose_bit = Self._insert_bit[16](
            transpose_a_bit, UInt32(0) if transpose_b else UInt32(1)
        )

        comptime if Self.mma_kind == UMMAKind.KIND_MXF8F6F4:
            return Self(
                desc
                | Self._create_mxf8f6f4_desc[
                    d_type, a_type, b_type, scale_type
                ]()
                | transpose_bit
            )
        elif (
            Self.mma_kind == UMMAKind.KIND_MXF4NVF4
            or Self.mma_kind == UMMAKind.KIND_MXF4
        ):
            return Self(
                desc
                | Self._create_mxf4_mxf4nvf4_desc[
                    d_type, a_type, b_type, scale_type
                ]()
                | transpose_bit
            )
        else:
            comptime assert False, String(
                "Unsupported UMMA kind: ", Self.mma_kind
            )

    @staticmethod
    def update_desc_with_sf_id[
        sf_id: UInt32
    ](inst_desc: UMMAInsDescriptor[Self.mma_kind],) -> Self:
        """Update the descriptor with the scale factor ID.

        Parameters:
            sf_id: The scale factor ID.

        Args:
            inst_desc: The descriptor to update.

        Returns:
            The updated descriptor.
        """

        comptime sfa_bit = Self._insert_bit[4](0x0, sf_id)
        comptime sfb_bit = Self._insert_bit[29](sfa_bit, sf_id)

        return Self(inst_desc.desc | sfb_bit)


# ===----------------------------------------------------------------------=== #
# MMA Shared Memory Descriptor
# ===----------------------------------------------------------------------=== #


struct MMASmemDescriptor(MMAOperandDescriptor, TrivialRegisterPassable):
    """Descriptor for shared memory operands tcgen05 mma instructions.

    This struct represents a descriptor that encodes information about shared memory layout
    and access patterns for warp group matrix multiply operations. The descriptor contains
    the following bit fields:

    | Bit field | Size | Description |
    |-----------|------|-------------|
    | 0-13   |  14  | Base address in shared memory |
    | 16-29   |  14  | LBO: leading dim byte offset |
    | 32-45   |  14  | SBO: stride dim byte offset |
    | 46-48   |   3  | Fixed constant value: 0b001 |
    | 49-51   |   3  | Matrix base offset, 0 for canonical layouts |
    | 52      |   1  | Leading dimension stride mode:<br>&nbsp;&nbsp;0: byte offset relative<br>&nbsp;&nbsp;1: byte address absolute<br>(only used for 48B K tile) |
    | 53-60   |   8  | Fixed constant value: 0 |
    | 61-63   |   3  | Swizzle mode:<br>&nbsp;&nbsp;0: No swizzling<br>&nbsp;&nbsp;1: 128-Byte with 32B atomic swizzling<br>&nbsp;&nbsp;2: 128-Byte swizzling<br>&nbsp;&nbsp;4: 64-Byte swizzling<br>&nbsp;&nbsp;6: 32-Byte swizzling |

    Note:

    - Some bits are unused.
    - Base address, LBO, and SBO ignore 4 least significant bits.

    See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-desc-layout

    """

    var desc: UInt64
    """The 64-bit descriptor encodes shared memory operand information."""

    comptime mask_14_bits: Int = (1 << 14) - 1
    """Mask with the lower 14 bits set."""

    @always_inline
    def __init__(out self, val: UInt64):
        """Initialize descriptor with raw 64-bit value.

        This constructor allows creating a descriptor directly from a 64-bit integer
        that already contains the properly formatted bit fields for the descriptor.

        The implicit attribute enables automatic conversion from `UInt64` to `MMASmemDescriptor`.

        Args:
            val: A 64-bit integer containing the complete descriptor bit layout.
        """
        self.desc = val

    @always_inline
    def _insert_bit[start_bit: Int](self, val: UInt64) -> Self:
        """Insert bits at specified position in descriptor.

        Parameters:
            start_bit: Starting bit position.

        Args:
            val: Value to insert.

        Returns:
            Updated descriptor value with inserted bits.
        """
        return Self(self.desc | (val << UInt64(start_bit)))

    @staticmethod
    @always_inline
    def create[
        stride_byte_offset: Int,
        leading_byte_offset: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    ](smem_ptr: Pointer[_, _, address_space=.SHARED]) -> Self:
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
        def _convert_swizzle_enum[mode: TensorMapSwizzle]() -> Int64:
            comptime if mode == TensorMapSwizzle.SWIZZLE_NONE:
                return 0
            elif mode == TensorMapSwizzle.SWIZZLE_32B:
                return 6
            elif mode == TensorMapSwizzle.SWIZZLE_64B:
                return 4
            elif mode == TensorMapSwizzle.SWIZZLE_128B:
                return 2
            else:
                comptime assert False, String(
                    "Unsupported swizzle mode: ", mode
                )

        comptime swizzle = _convert_swizzle_enum[swizzle_mode._value]()

        # Extract 18 bits and ignore 4 LSB.
        var base_ptr = UInt32(Int(smem_ptr))
        var start_address = UInt64((base_ptr >> 4) & UInt32(Self.mask_14_bits))

        # Ignore 4 LSB.
        var sbo = UInt64((stride_byte_offset >> 4) & Self.mask_14_bits)
        var lbo = UInt64((leading_byte_offset >> 4) & Self.mask_14_bits)

        # Start from LSB. Mask out higher bits to avoid overwriting.
        var desc = Self(0)
        # bits  0-13 address in share memory
        desc = desc._insert_bit[0](start_address)
        # bits 14-16 unused
        # bits 16-29 leading dim byte offset
        desc = desc._insert_bit[16](lbo)
        # bits 30-32 unused
        # bits 32-45 stride dim byte offset
        desc = desc._insert_bit[32](sbo)
        # bits 46-48 001
        desc = desc._insert_bit[46](1)
        # bits 49-51 matrix base offset, not supported
        # bits 52    LBO mode, only matters for 48B K tile and not supported
        # bits 53-60 fixed, 0
        # bits 61-63 swizzle type
        desc = desc._insert_bit[61](UInt64(swizzle))

        return desc

    @always_inline
    def __iadd__(mut self, offset: Int):
        """Add offset to descriptor's base address in-place.

        Args:
            offset: Byte offset to add to base address.
        """
        self = self + offset

    @always_inline
    def __add__(self, offset: Int) -> Self:
        """Add offset to descriptor's base address.

        Args:
            offset: Byte offset to add to base address.

        Returns:
            New descriptor with updated base address.
        """
        return Self(self.desc + UInt64((offset >> 4) & Self.mask_14_bits))


struct MMASmemDescriptorPair(TrivialRegisterPassable):
    """Descriptor for shared memory operands tcgen05 mma instructions.

    This struct represents a descriptor that encodes information about shared memory layout
    and access patterns for warp group matrix multiply operations. The descriptor contains
    the following bit fields:

    | Bit field | Size | Description |
    |-----------|------|-------------|
    | 0-13   |  14  | Base address in shared memory |
    | 16-29   |  14  | LBO: leading dim byte offset |
    | 32-45   |  14  | SBO: stride dim byte offset |
    | 46-48   |   3  | Fixed constant value: 0b001 |
    | 49-51   |   3  | Matrix base offset, 0 for canonical layouts |
    | 52      |   1  | Leading dimension stride mode:<br>&nbsp;&nbsp;0: byte offset relative<br>&nbsp;&nbsp;1: byte address absolute<br>(only used for 48B K tile) |
    | 53-60   |   8  | Fixed constant value: 0 |
    | 61-63   |   3  | Swizzle mode:<br>&nbsp;&nbsp;0: No swizzling<br>&nbsp;&nbsp;1: 128-Byte with 32B atomic swizzling<br>&nbsp;&nbsp;2: 128-Byte swizzling<br>&nbsp;&nbsp;4: 64-Byte swizzling<br>&nbsp;&nbsp;6: 32-Byte swizzling |

    Note:

    - Some bits are unused.
    - Base address, LBO, and SBO ignore 4 least significant bits.

    See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-desc-layout

    """

    var hi: UInt32
    """The low 32-bits of the descriptor."""
    var lo: UInt32
    """The high 32-bits of the descriptor."""

    comptime mask_14_bits: UInt32 = (1 << 14) - 1
    """Mask with the lower 14 bits set."""

    @always_inline
    def __init__(out self, hi: UInt32, lo: UInt32):
        """Initialize descriptor with raw 64-bit value.

        This constructor allows creating a descriptor directly from a 64-bit integer
        that already contains the properly formatted bit fields for the descriptor.

        The implicit attribute enables automatic conversion from `UInt64` to `MMASmemDescriptor`.

        Args:
            hi: A 32-bit integer containing the upper half of the descriptor layout.
            lo: A 32-bit integer containing the lower half of the descriptor layout.
        """
        self.hi = hi
        self.lo = lo

    @always_inline
    def _insert_bit[start_bit: Int](self, val: UInt32) -> Self:
        """Insert bits at specified position in descriptor.

        Parameters:
            start_bit: Starting bit position.

        Args:
            val: Value to insert.

        Returns:
            Updated descriptor value with inserted bits.
        """

        comptime if start_bit < 32:
            return Self(self.hi, self.lo | (val << UInt32(start_bit)))
        else:
            return Self(self.hi | (val << UInt32(start_bit - 32)), self.lo)

    @staticmethod
    @always_inline
    def create[
        stride_byte_offset: Int,
        leading_byte_offset: Int,
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    ](smem_ptr: Pointer[_, _, address_space=.SHARED],) -> Self:
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
        def _convert_swizzle_enum[mode: TensorMapSwizzle]() -> Int64:
            comptime if mode == TensorMapSwizzle.SWIZZLE_NONE:
                return 0
            elif mode == TensorMapSwizzle.SWIZZLE_32B:
                return 6
            elif mode == TensorMapSwizzle.SWIZZLE_64B:
                return 4
            elif mode == TensorMapSwizzle.SWIZZLE_128B:
                return 2
            else:
                comptime assert False, String(
                    "Unsupported swizzle mode: ", mode
                )

        comptime swizzle = _convert_swizzle_enum[swizzle_mode._value]()

        # Extract 18 bits and ignore 4 LSB.
        var base_ptr = UInt32(Int(smem_ptr))
        var start_address = (base_ptr >> 4) & Self.mask_14_bits

        # Ignore 4 LSB.
        var sbo: UInt32 = UInt32(stride_byte_offset >> 4) & Self.mask_14_bits
        var lbo: UInt32 = UInt32(leading_byte_offset >> 4) & Self.mask_14_bits

        # Start from LSB. Mask out higher bits to avoid overwriting.
        var desc = Self(0, 0)
        # bits  0-13 address in share memory
        desc = desc._insert_bit[0](start_address)
        # bits 14-16 unused
        # bits 16-29 leading dim byte offset
        desc = desc._insert_bit[16](lbo)
        # bits 30-32 unused
        # bits 32-45 stride dim byte offset
        desc = desc._insert_bit[32](sbo)
        # bits 46-48 001
        desc = desc._insert_bit[46](1)
        # bits 49-51 matrix base offset, not supported
        # bits 52    LBO mode, only matters for 48B K tile and not supported
        # bits 53-60 fixed, 0
        # bits 61-63 swizzle type
        desc = desc._insert_bit[61](UInt32(swizzle))

        return desc

    @always_inline
    def __iadd__(mut self, offset: UInt32):
        """Add offset to descriptor's base address in-place.

        Args:
            offset: Byte offset to add to base address.
        """
        self = self + offset

    @always_inline
    def __add__(self, offset: UInt32) -> Self:
        """Add offset to descriptor's base address.

        Args:
            offset: Byte offset to add to base address.

        Returns:
            New descriptor with updated base address.
        """
        return Self(self.hi, self.lo + ((offset >> 4) & Self.mask_14_bits))

    @always_inline
    def descriptor(self) -> MMASmemDescriptor:
        """Get the descriptor for the shared memory operand.

        Returns:
            The descriptor for the shared memory operand as a `MMASmemDescriptor`.
        """
        return MMASmemDescriptor(UInt64(self.hi) << 32 | UInt64(self.lo))


# ===----------------------------------------------------------------------=== #
# UMMA
# ===----------------------------------------------------------------------=== #


@always_inline
def mma[
    kind: UMMAKind,
    //,
    cta_group: Int = 1,
    /,
    *,
    c_scale: UInt32 = 1,
](
    a_desc: MMASmemDescriptor,
    b_desc: MMASmemDescriptor,
    c_tmem: UInt32,
    inst_desc: UMMAInsDescriptor[kind],
):
    """Perform a matrix multiply-accumulate operation using the tcgen05.mma instruction.

    Parameters:
        kind: Data type of the matrices.
        cta_group: Number of ctas used by MMA.
        c_scale: Scale factor for the C matrix, 0 or 1.

    Args:
        a_desc: The descriptor for the A matrix.
        b_desc: The descriptor for the B matrix.
        c_tmem: The address of the C matrix in the tensor memory.
        inst_desc: The descriptor for the MMA instruction.
    """
    comptime assert (
        _has_blackwell_tcgen05()
    ), "tcgen05.mma not supported on this GPU"

    comptime assert c_scale == 0 or c_scale == 1, String(
        "Invalid c_scale: ", c_scale
    )

    comptime if cta_group == 1:
        var masks = IndexList[4, element_type=DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::1."""
            + String(kind)
            + """ [$0], $1, $2, $3, {$5, $6, $7, $8}, p;
            }""",
            NoneType,
            constraints="r,l,l,r,n,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
        )
    elif cta_group == 2:
        var masks = IndexList[8, element_type=DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::2."""
            + String(kind)
            + """ [$0], $1, $2, $3, {$5, $6, $7, $8, $9, $10, $11, $12}, p;
            }""",
            NoneType,
            constraints="r,l,l,r,n,r,r,r,r,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
            masks[4],
            masks[5],
            masks[6],
            masks[7],
        )
    else:
        comptime assert False, String("Unsupported cta group: ", cta_group)


@always_inline
def mma[
    kind: UMMAKind,
    //,
    cta_group: Int = 1,
    /,
](
    a_desc: MMASmemDescriptor,
    b_desc: MMASmemDescriptor,
    c_tmem: UInt32,
    inst_desc: UMMAInsDescriptor[kind],
    sfa_tmem: UInt32,
    sfb_tmem: UInt32,
    c_scale: UInt32,
):
    """Perform a matrix multiply-accumulate operation using the tcgen05.mma instruction.

    Parameters:
        kind: Data type of the matrices.
        cta_group: Number of ctas used by MMA.

    Args:
        a_desc: The descriptor for the A matrix.
        b_desc: The descriptor for the B matrix.
        c_tmem: The address of the C matrix in the tensor memory.
        inst_desc: The descriptor for the MMA instruction.
        sfa_tmem: The address of the block scale factor A in the tensor memory.
        sfb_tmem: The address of the block scale factor B in the tensor memory.
        c_scale: Scale factor for the C matrix, 0 or 1.
    """
    comptime assert (
        _has_blackwell_tcgen05()
    ), "tcgen05.mma not supported on this GPU"

    comptime assert kind in (
        UMMAKind.KIND_MXF8F6F4,
        UMMAKind.KIND_MXF4,
        UMMAKind.KIND_MXF4NVF4,
    ), "Only MXF8F6F4, MXF4, or MXF4NVF4 MMA kind supports block scale factors"

    @always_inline
    def _get_scale_vector_size[kind: UMMAKind]() -> String:
        var scale_vector_size = String(".block_scale")

        comptime if kind == UMMAKind.KIND_MXF4NVF4:
            scale_vector_size = scale_vector_size + String(".block16")
        return scale_vector_size

    comptime scale_vector_size = _get_scale_vector_size[kind]()

    comptime if cta_group == 1:
        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::1."""
            + String(kind)
            + scale_vector_size
            + """ [$0], $1, $2, $3, [$5], [$6], p;
            }""",
            NoneType,
            constraints="r,l,l,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            sfa_tmem,
            sfb_tmem,
        )
    elif cta_group == 2:
        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::2."""
            + String(kind)
            + scale_vector_size
            + """ [$0], $1, $2, $3, [$5], [$6], p;
            }""",
            NoneType,
            constraints="r,l,l,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            sfa_tmem,
            sfb_tmem,
        )
    else:
        comptime assert False, String("Unsupported cta group: ", cta_group)


@always_inline
def mma[
    kind: UMMAKind,
    //,
    cta_group: Int = 1,
    /,
    *,
](
    a_desc: MMASmemDescriptor,
    b_desc: MMASmemDescriptor,
    c_tmem: UInt32,
    inst_desc: UMMAInsDescriptor[kind],
    c_scale: UInt32,
):
    """Perform a matrix multiply-accumulate operation using the tcgen05.mma instruction.

    Parameters:
        kind: Data type of the matrices.
        cta_group: Number of ctas used by MMA.

    Args:
        a_desc: The descriptor for the A matrix.
        b_desc: The descriptor for the B matrix.
        c_tmem: The address of the C matrix in the tensor memory.
        inst_desc: The descriptor for the MMA instruction.
        c_scale: Scale factor for the C matrix. Any non-zero value is translated to `1`.
    """
    comptime assert (
        _has_blackwell_tcgen05()
    ), "tcgen05.mma not supported on this GPU"

    comptime if cta_group == 1:
        var masks = IndexList[4, element_type=DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::1."""
            + String(kind)
            + """ [$0], $1, $2, $3, {$5, $6, $7, $8}, p;
            }""",
            NoneType,
            constraints="r,l,l,r,r,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
        )
    elif cta_group == 2:
        var masks = IndexList[8, element_type=DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::2."""
            + String(kind)
            + """ [$0], $1, $2, $3, {$5, $6, $7, $8, $9, $10, $11, $12}, p;
            }""",
            NoneType,
            constraints="r,l,l,r,r,r,r,r,r,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
            masks[4],
            masks[5],
            masks[6],
            masks[7],
        )
    else:
        comptime assert False, String("Unsupported cta group: ", cta_group)


@always_inline
def mma[
    kind: UMMAKind,
    //,
    cta_group: Int = 1,
    /,
](
    a_desc: UInt32,
    b_desc: MMASmemDescriptor,
    c_tmem: UInt32,
    inst_desc: UMMAInsDescriptor[kind],
    c_scale: UInt32,
):
    """Perform a matrix multiply-accumulate operation using the tcgen05.mma instruction.

    Parameters:
        kind: Data type of the matrices.
        cta_group: Number of ctas used by MMA.

    Args:
        a_desc: The descriptor for the A matrix.
        b_desc: The descriptor for the B matrix.
        c_tmem: The address of the C matrix in the tensor memory.
        inst_desc: The descriptor for the MMA instruction.
        c_scale: Scale factor for the C matrix. Any non-zero value is interpreted as `1`.
    """
    comptime assert (
        _has_blackwell_tcgen05()
    ), "tcgen05.mma not supported on this GPU"

    comptime if cta_group == 1:
        var masks = IndexList[4, element_type=DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::1."""
            + String(kind)
            + """ [$0], [$1], $2, $3, {$5, $6, $7, $8}, p;
            }""",
            NoneType,
            constraints="r,r,l,r,r,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
        )
    elif cta_group == 2:
        var masks = IndexList[8, element_type=DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::2."""
            + String(kind)
            + """ [$0], [$1], $2, $3, {$5, $6, $7, $8, $9, $10, $11, $12}, p;
            }""",
            NoneType,
            constraints="r,r,l,r,r,r,r,r,r,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
            masks[4],
            masks[5],
            masks[6],
            masks[7],
        )
    else:
        comptime assert False, String("Unsupported cta group: ", cta_group)


@always_inline
def mma[
    kind: UMMAKind,
    //,
    cta_group: Int = 1,
    /,
    *,
    c_scale: UInt32 = 1,
](
    a_desc: UInt32,
    b_desc: MMASmemDescriptor,
    c_tmem: UInt32,
    inst_desc: UMMAInsDescriptor[kind],
):
    """Perform a matrix multiply-accumulate operation using the tcgen05.mma instruction.

    Parameters:
        kind: Data type of the matrices.
        cta_group: Number of ctas used by MMA.
        c_scale: Scale factor for the C matrix, 0 or 1.

    Args:
        a_desc: The descriptor for the A matrix.
        b_desc: The descriptor for the B matrix.
        c_tmem: The address of the C matrix in the tensor memory.
        inst_desc: The descriptor for the MMA instruction.
    """
    comptime assert (
        _has_blackwell_tcgen05()
    ), "tcgen05.mma not supported on this GPU"

    comptime assert c_scale == 0 or c_scale == 1, String(
        "Invalid c_scale: ", c_scale
    )

    comptime if cta_group == 1:
        var masks = IndexList[4, element_type=DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::1."""
            + String(kind)
            + """ [$0], [$1], $2, $3, {$5, $6, $7, $8}, p;
            }""",
            NoneType,
            constraints="r,r,l,r,n,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
        )
    elif cta_group == 2:
        var masks = IndexList[8, element_type=DType.uint32](0)

        inlined_assembly[
            """{
                .reg .pred p;
                setp.ne.b32 p, $4, 0;
                tcgen05.mma.cta_group::2."""
            + String(kind)
            + """ [$0], [$1], $2, $3, {$5, $6, $7, $8, $9, $10, $11, $12}, p;
            }""",
            NoneType,
            constraints="r,r,l,r,n,r,r,r,r,r,r,r,r",
        ](
            c_tmem,
            a_desc,
            b_desc,
            inst_desc,
            c_scale,
            masks[0],
            masks[1],
            masks[2],
            masks[3],
            masks[4],
            masks[5],
            masks[6],
            masks[7],
        )
    else:
        comptime assert False, String("Unsupported cta group: ", cta_group)


# ===----------------------------------------------------------------------=== #
# UMMA Sync
# ===----------------------------------------------------------------------=== #


@always_inline
def mma_arrive[
    cta_group: Int = 1,
](mbar_ptr: Pointer[_, _, address_space=.SHARED]):
    """Arrive at the mbar pointer for the MMA instruction.

    Parameters:
        cta_group: Number of ctas used by MMA.

    Args:
        mbar_ptr: Pointer to the mbar.
    """

    comptime assert cta_group in (1, 2), String(
        "Unsupported cta group: ", cta_group
    )

    comptime type = mbar_ptr.T
    comptime assert size_of[type]() == 8, "mbar_ptr must be 8 bytes"

    inlined_assembly[
        "tcgen05.commit.cta_group::"
        + String(cta_group)
        + ".mbarrier::arrive::one.shared::cluster.b64 [$0];",
        NoneType,
        constraints="r",
    ](Int32(Int(mbar_ptr)))


@always_inline
def mma_arrive_multicast[
    cta_group: Int = 1,
](mbar_ptr: Pointer[_, _, address_space=.SHARED], cta_mask: UInt16,):
    """Arrive at the mbar pointer for the MMA instruction for multiple ctas.

    Parameters:
        cta_group: Number of ctas used by MMA.

    Args:
        mbar_ptr: Pointer to the mbar.
        cta_mask: Mask of ctas to signal.
    """

    comptime assert cta_group in (1, 2), String(
        "Unsupported cta group: ", cta_group
    )

    comptime type = mbar_ptr.T
    comptime assert size_of[type]() == 8, "mbar_ptr must be 8 bytes"

    inlined_assembly[
        "tcgen05.commit.cta_group::"
        + String(cta_group)
        + ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [$0], $1;",
        NoneType,
        constraints="r,h",
    ](Int32(Int(mbar_ptr)), cta_mask)
