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

# ===----------------------------------------------------------------------===#
# bitcast
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn bitcast[
    new_type: DType, new_width: Int, src_type: DType, src_width: Int
](val: SIMD[src_type, src_width]) -> SIMD[new_type, new_width]:
    """Bitcasts a SIMD value to another SIMD value.

    Constraints:
        The bitwidth of the two types must be the same.

    Parameters:
        new_type: The target type.
        new_width: The target width.
        src_type: The source type.
        src_width: The source width.

    Args:
        val: The source value.

    Returns:
        A new SIMD value with the specified type and width with a bitcopy of the
        source SIMD value.
    """
    constrained[
        bitwidthof[SIMD[src_type, src_width]]()
        == bitwidthof[SIMD[new_type, new_width]](),
        "the source and destination types must have the same bitwidth",
    ]()

    @parameter
    if new_type == src_type:
        return rebind[SIMD[new_type, new_width]](val)
    return __mlir_op.`pop.bitcast`[
        _type = __mlir_type[
            `!pop.simd<`, new_width.value, `, `, new_type.value, `>`
        ]
    ](val.value)


@always_inline("nodebug")
fn bitcast[
    new_type: DType, src_type: DType
](val: SIMD[src_type, 1]) -> SIMD[new_type, 1]:
    """Bitcasts a SIMD value to another SIMD value.

    Constraints:
        The bitwidth of the two types must be the same.

    Parameters:
        new_type: The target type.
        src_type: The source type.

    Args:
        val: The source value.

    Returns:
        A new SIMD value with the specified type and width with a bitcopy of the
        source SIMD value.
    """
    constrained[
        bitwidthof[SIMD[src_type, 1]]() == bitwidthof[SIMD[new_type, 1]](),
        "the source and destination types must have the same bitwidth",
    ]()

    return bitcast[new_type, 1, src_type, 1](val)


@always_inline("nodebug")
fn bitcast[
    new_type: DType, src_width: Int
](val: SIMD[DType.bool, src_width]) -> Scalar[new_type]:
    """Packs a SIMD bool into an integer.

    Constraints:
        The bitwidth of the two types must be the same.

    Parameters:
        new_type: The target type.
        src_width: The source width.

    Args:
        val: The source value.

    Returns:
        A new integer scalar which has the same bitwidth as the bool vector.
    """
    constrained[
        src_width == bitwidthof[Scalar[new_type]](),
        "the source and destination types must have the same bitwidth",
    ]()

    return __mlir_op.`pop.bitcast`[
        _type = __mlir_type[`!pop.scalar<`, new_type.value, `>`]
    ](val.value)
