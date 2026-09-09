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
"""Provides mathematical helper functions for numerical testing."""


from std.builtin.dtype import _integral_type_of
from std.memory import bitcast
from std.sys import bit_width_of


def ulp_distance[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Int:
    """Computes the distance between two floating-point values in ULPs.

    ULP (Unit in the Last Place) distance measures how many representable
    floating-point values exist between two numbers.

    Parameters:
        dtype: The floating-point data type.

    Args:
        a: First floating-point value.
        b: Second floating-point value.

    Returns:
        The ULP distance between the two values.
    """
    comptime T = _integral_type_of[dtype]()
    # widen to Int first to avoid overflow
    var a_int = Int(bitcast[T](a))
    var b_int = Int(bitcast[T](b))
    # to twos complement
    comptime two_complement_const: Int = 1 << (bit_width_of[dtype]() - 1)
    a_int = two_complement_const - a_int if a_int < 0 else a_int
    b_int = two_complement_const - b_int if b_int < 0 else b_int
    return abs(a_int - b_int)
