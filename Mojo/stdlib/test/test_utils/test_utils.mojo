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
"""Provides general test utility functions for the Mojo standard library tests."""

from std.ffi import external_call

from std.builtin.simd import _simd_apply
from std.testing import assert_equal, assert_true


def check_write_to(
    value: Some[Writable], *, expected: String, is_repr: Bool
) raises:
    """Check that the write_to or write_repr_to of the value is equal to the expected string.

    Args:
        value: The Writable value to check.
        expected: The expected string.
        is_repr: Whether to check the repr version of the value.

    Raises:
        Error: if the written string does not equal `expected`.
    """

    var string = String()
    if is_repr:
        value.write_repr_to(string)
    else:
        value.write_to(string)
    assert_equal(string, expected)


def check_write_to(
    value: Some[Writable], *, contains: String, is_repr: Bool
) raises:
    """Check that the write_to or write_repr_to of the value contains the expected string.

    Args:
        value: The Writable value to check.
        contains: The string to check for in the output.
        is_repr: Whether to check the repr version of the value.

    Raises:
        Error: if the written string does not contain `contains`.
    """
    var string = String()
    if is_repr:
        value.write_repr_to(string)
    else:
        value.write_to(string)
    assert_true(contains in string)


@always_inline
def libm_call[
    dtype: DType,
    width: SIMDLength,
    //,
    fn_fp32: StaticString,
    fn_fp64: StaticString,
](arg: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Calls a libm function with the appropriate float32 or float64 version.

    Parameters:
        dtype: The data type (must be float32 or float64).
        width: The SIMD width.
        fn_fp32: Name of the float32 version of the libm function.
        fn_fp64: Name of the float64 version of the libm function.

    Args:
        arg: The input SIMD vector.

    Returns:
        The result of calling the libm function.
    """

    @always_inline("nodebug")
    def _float32_dispatch[
        input_type: DType, result_type: DType
    ](arg: Scalar[input_type]) -> Scalar[result_type]:
        return external_call[fn_fp32, Scalar[result_type]](arg)

    @always_inline("nodebug")
    def _float64_dispatch[
        input_type: DType, result_type: DType
    ](arg: Scalar[input_type]) -> Scalar[result_type]:
        return external_call[fn_fp64, Scalar[result_type]](arg)

    comptime assert dtype in [
        DType.float32,
        DType.float64,
    ], "input dtype must be float32 or float64"

    comptime if dtype == .float32:
        return _simd_apply[_float32_dispatch, result_dtype=dtype](arg)
    else:
        return _simd_apply[_float64_dispatch, result_dtype=dtype](arg)
