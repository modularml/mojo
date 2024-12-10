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
"""Implements types that work with unsafe pointers.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import bitcast
```
"""

from sys import bitwidthof

# ===-----------------------------------------------------------------------===#
# bitcast
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn bitcast[
    type: DType,
    width: Int, //,
    new_type: DType,
    new_width: Int = width,
](val: SIMD[type, width]) -> SIMD[new_type, new_width]:
    """Bitcasts a SIMD value to another SIMD value.

    Constraints:
        The bitwidth of the two types must be the same.

    Parameters:
        type: The source type.
        width: The source width.
        new_type: The target type.
        new_width: The target width.

    Args:
        val: The source value.

    Returns:
        A new SIMD value with the specified type and width with a bitcopy of the
        source SIMD value.
    """
    constrained[
        bitwidthof[SIMD[type, width]]()
        == bitwidthof[SIMD[new_type, new_width]](),
        "the source and destination types must have the same bitwidth",
    ]()

    @parameter
    if new_type == type:
        return rebind[SIMD[new_type, new_width]](val)
    return __mlir_op.`pop.bitcast`[
        _type = __mlir_type[
            `!pop.simd<`, new_width.value, `, `, new_type.value, `>`
        ]
    ](val.value)


@always_inline("nodebug")
fn _uint(n: Int) -> DType:
    if n == 8:
        return DType.uint8
    elif n == 16:
        return DType.uint16
    elif n == 32:
        return DType.uint32
    else:
        return DType.uint64


@always_inline("nodebug")
fn pack_bits[
    width: Int, //,
    new_type: DType = _uint(width),
](val: SIMD[DType.bool, width]) -> Scalar[new_type]:
    """Packs a SIMD bool into an integer.

    Constraints:
        The width of the bool vector must be the same as the bitwidth of the
        target type.

    Parameters:
        width: The source width.
        new_type: The target type.

    Args:
        val: The source value.

    Returns:
        A new integer scalar which has the same bitwidth as the bool vector.
    """
    constrained[
        width == bitwidthof[Scalar[new_type]](),
        (
            "the width of the bool vector must be the same as the bitwidth of"
            " the target type"
        ),
    ]()

    return __mlir_op.`pop.bitcast`[
        _type = __mlir_type[`!pop.scalar<`, new_type.value, `>`]
    ](val.value)
