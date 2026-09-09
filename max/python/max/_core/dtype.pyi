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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""Max DType definitions."""

import enum

import numpy

class DType(enum.Enum):
    """The tensor data type."""

    bool = 1
    """Boolean data type. Stores ``True`` or ``False`` values."""

    int8 = 135
    """8-bit signed integer, range -128 to 127."""

    int16 = 137
    """16-bit signed integer, range -32,768 to 32,767."""

    int32 = 139
    """32-bit signed integer, range -2,147,483,648 to 2,147,483,647."""

    int64 = 141
    """
    64-bit signed integer, range -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807.
    """

    uint8 = 134
    """8-bit unsigned integer, range 0 to 255."""

    uint16 = 136
    """16-bit unsigned integer, range 0 to 65,535."""

    uint32 = 138
    """32-bit unsigned integer, range 0 to 4,294,967,295."""

    uint64 = 140
    """64-bit unsigned integer, range 0 to 18,446,744,073,709,551,615."""

    float16 = 79
    """
    16-bit IEEE 754 half-precision floating-point. 1 sign bit, 5 exponent bits, 10 mantissa bits.
    """

    float32 = 81
    """
    32-bit IEEE 754 single-precision floating-point. 1 sign bit, 8 exponent bits, 23 mantissa bits.
    """

    float64 = 82
    """
    64-bit IEEE 754 double-precision floating-point. 1 sign bit, 11 exponent bits, 52 mantissa bits.
    """

    bfloat16 = 80
    """
    16-bit bfloat16 (Brain Float) format. 1 sign bit, 8 exponent bits, 7 mantissa bits.
    """

    float4_e2m1fn = 64
    """
    4-bit floating-point with 2 exponent bits and 1 mantissa bits, finite values only.
    """

    float6_e2m3fn = 65
    """
    6-bit floating-point with 2 exponent bits and 3 mantissa bits, finite values only.
    """

    float6_e3m2fn = 66
    """
    6-bit floating-point with 3 exponent bits and 2 mantissa bits, finite values only.
    """

    float8_e8m0fnu = 73
    """
    8-bit floating-point with 8 exponent bits and 0 mantissa bits, finite values only.
    """

    float8_e4m3fn = 75
    """
    8-bit floating-point with 4 exponent bits and 3 mantissa bits, finite values only.
    """

    float8_e4m3fnuz = 76
    """
    8-bit floating-point with 4 exponent bits and 3 mantissa bits, finite values only, no negative zero.
    """

    float8_e5m2 = 77
    """8-bit floating-point with 5 exponent bits and 2 mantissa bits."""

    float8_e5m2fnuz = 78
    """
    8-bit floating-point with 5 exponent bits and 2 mantissa bits, finite values only, no negative zero.
    """

    @property
    def align(self) -> int:
        """
        Returns the alignment requirement of the data type in bytes.

        The alignment specifies the memory boundary that values of this data type
        must be aligned to for optimal performance and correctness.
        """

    @property
    def size_in_bits(self) -> int:
        """
        Returns the size of the data type in bits.

        This indicates how many bits are required to store a single value
        of this data type in memory.
        """

    @property
    def size_in_bytes(self) -> int:
        """
        Returns the size of the data type in bytes.

        This indicates how many bytes are required to store a single value
        of this data type in memory.
        """

    def is_integral(self) -> __builtins__.bool:
        """Checks if the data type is an integer type."""

    def is_unsigned_integral(self) -> __builtins__.bool:
        """Checks if the data type is an unsigned integer type."""

    def is_signed_integral(self) -> __builtins__.bool:
        """Checks if the data type is a signed integer type."""

    def is_float(self) -> __builtins__.bool:
        """Checks if the data type is a floating-point type."""

    def is_float8(self) -> __builtins__.bool:
        """Checks if the data type is an 8-bit floating-point type."""

    def is_half(self) -> __builtins__.bool:
        """Checks if the data type is a half-precision floating-point type."""

    def to_numpy(self) -> numpy.dtype:
        """
        Converts this ``DType`` to the corresponding NumPy dtype.

        Returns:
            DType: The corresponding NumPy dtype object.

        Raises:
            ValueError: If the dtype is not supported.
        """

    @classmethod
    def from_numpy(cls, dtype: numpy.dtype) -> DType:
        """
        Converts a NumPy dtype to the corresponding DType.

        Args:
            dtype (np.dtype): The NumPy dtype to convert.

        Returns:
            DType: The corresponding DType enum value.

        Raises:
            ValueError: If the input dtype is not supported.
        """

    @property
    def _mlir(self) -> str: ...
