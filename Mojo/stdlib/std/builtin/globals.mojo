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
"""Utilities for working with global constants.

This module provides helper functions for efficiently creating references to
compile-time constants without materializing entire data structures in memory.
"""

from std.traits import IsTriviallyCopyable, IsTriviallyDeinitable


def global_constant[
    T: Copyable & Deinitable, //, value: T
]() -> ref[ImmStaticOrigin] T:
    """Creates a reference to a compile-time constant value stored in static memory.

    This function stores the compile-time constant in the binary's read-only data
    section and returns a reference to it, avoiding runtime materialization. This
    is useful for large lookup tables where you want to access individual elements
    without copying the entire structure onto the stack.

    Constraints:
        The type `T` must have trivial copy and destroy semantics. Self-contained
        types like `Int`, `SIMD`, and `Array` (with trivial element types)
        work. Types with heap allocations like `Dict`, `List`, or `String` do not
        work because their internal pointers would be invalid at runtime and they
        have non-trivial copy/destroy operations.

    Parameters:
        T: The type of the constant value. Must have trivial copy and destroy
            semantics (`IsTriviallyCopyable[T]` and `IsTriviallyDeinitable[T]` must be `True`).
        value: The compile-time constant value.

    Returns:
        A reference to the global constant with `ImmStaticOrigin`.

    Examples:
    ```mojo
    from std.builtin.globals import global_constant

    # Create a reference to a constant array and access elements
    comptime lookup_table: Array[Int, 4] = [1, 2, 3, 4]
    var element = global_constant[lookup_table]()[2]  # Access without materializing entire array
    print(element)  # Prints: 3

    # Use with more complex compile-time values
    def compute(x: Int) -> Int:
        return x * 2 + 1

    comptime data: Array[Int, 3] = [1, compute(5), 100]
    ref data_ref = global_constant[data]()
    print(data_ref[0], data_ref[1], data_ref[2])  # Prints: 1 11 100
    ```
    """
    comptime assert IsTriviallyCopyable[T] and IsTriviallyDeinitable[T], (
        "global_constant requires a type with trivial copy and destroy"
        " semantics. Types with heap allocations like Dict, List, or String"
        " are not supported because their internal pointers would be"
        " invalid at runtime."
    )
    return ImmPointer[origin=ImmStaticOrigin](
        _mlir_value=__mlir_op.`pop.global_constant`[value=value]()
    )[]
