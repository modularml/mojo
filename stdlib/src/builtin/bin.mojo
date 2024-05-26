# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements the `bin()` function.

These are Mojo built-ins, so you don't need to import them.
"""


# Need this until we have constraints to stop the compiler from matching this
# directly to bin[type: DType](num: Scalar[type]).
@always_inline("nodebug")
fn bin(b: Scalar[DType.bool], /) -> String:
    """Returns the binary representation of a scalar bool.

    Args:
        b: A scalar bool value.

    Returns:
        The binary string representation of b.
    """
    return bin(int(b))


fn bin[type: DType](num: Scalar[type], /) -> String:
    """Return the binary string representation an integral value.

    ```mojo
    print(bin(123))
    print(bin(-123))
    ```
    ```plaintext
    '0b1111011'
    '-0b1111011'
    ```

    Parameters:
        type: The data type of the integral scalar.

    Args:
        num: An integral scalar value.

    Returns:
        The binary string representation of num.
    """
    constrained[type.is_integral(), "Expected integral value"]()
    alias BIN_PREFIX = "0b"

    if num == 0:
        return BIN_PREFIX + "0"

    # TODD: pre-allocate string size when #2194 is resolved
    var result = String()
    var cpy = abs(num)
    while cpy > 0:
        result += str(cpy & 1)
        cpy = cpy >> 1

    result = BIN_PREFIX + result[::-1]
    return "-" + result if num < 0 else result


@always_inline("nodebug")
fn bin[T: Indexer](num: T, /) -> String:
    """Returns the binary representation of an indexer type.

    Parameters:
        T: The Indexer type.

    Args:
        num: An indexer value.

    Returns:
        The binary string representation of num.
    """
    return bin(Scalar[DType.index](index(num)))
