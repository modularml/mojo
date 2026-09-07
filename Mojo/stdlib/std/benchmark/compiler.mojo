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
"""Provides compiler hints to prevent optimization of benchmark code.

This module includes utilities that prevent the compiler from optimizing away
code being benchmarked.
The `keep()` and `black_box()` functions provide hints to the compiler from
overly optimizing away code being benchmarked.
"""

from std.sys._assembly import inlined_assembly

# ===-----------------------------------------------------------------------===#
# keep
# ===-----------------------------------------------------------------------===#


@always_inline
def keep[T: AnyType, origin: Origin, //](ref[origin] value: T):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
        T: The type of the input.
        origin: The origin of the input.

    Args:
        value: The value to not optimize away.
    """
    var tmp_ptr = Pointer(to=value)
    inlined_assembly[
        "",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](tmp_ptr)


# ===-----------------------------------------------------------------------===#
# black_box
# ===-----------------------------------------------------------------------===#


@always_inline
def black_box[
    T: AnyType, origin: Origin, //
](ref[origin] value: T) -> ref[origin] T:
    """Prevents the compiler from optimizing away computations or values.

    Unlike `black_box(take=...)`, this function takes a borrowed value and
    returns the value by reference.

    This is an identity function that hints to the compiler to be maximally
    pessimistic about what it could do with the value. Unlike normal identity
    functions, `black_box` prevents the compiler from making assumptions about
    the input value or optimizing across the function call boundary.

    The primary use case is in benchmarking, where you want to measure the
    performance of code as it would execute with unknown inputs at runtime,
    rather than with compile-time constants that the compiler can optimize away.

    Parameters:
        T: The type of the value.
        origin: The origin of the value.

    Args:
        value: The value to prevent optimization on.

    Returns:
        The same value, but the compiler cannot make assumptions about it or
        optimize across this function boundary.

    Notes:
        - Input expressions are still optimized before being passed to
          `black_box`. To prevent this, wrap operands individually.
        - If you do not need the return value, use `keep()` instead, which
          prevents optimization without returning the value.

    Examples:

    ```mojo
    def benchmark_contains():
        var haystack = "abcdefghijklmnopqrstuvwxyz"
        var needle = "lmnop"

        for _ in range(100):
            _ = needle in haystack
    ```

    In the above example, the compiler/LLVM may make optimizations like:
        - `needle` and `haystack` are constant, it may move the `in` check
          outside the loop and delete the loop.
        - `needle` and `haystack` have values known at compile time, and the
          result is always `True`, replace the `in` check with a constant
          `True`.
        - Since the result of the `in` check is unused, it may delete the
          operation entirely.

    To prevent said optimizations, you can use `black_box` (and `keep`):

    ```mojo
    from std.benchmark import keep, black_box

    def benchmark_contains():
        var haystack = "abcdefghijklmnopqrstuvwxyz"
        var needle = "lmnop"

        for _ in range(100):
            # black_box:
            # Prevent the compiler from making assumptions about the input
            # values.
            var found = black_box(needle) in black_box(haystack)

            # keep:
            # Prevent the compiler from removing the `in` check even
            # though the result is unused.
            keep(found)
    ```
    """
    keep(value)
    return value


@always_inline
def black_box[T: Movable, //](*, var take: T) -> T:
    """Prevents the compiler from optimizing away computations or values.

    Unlike `black_box(ref T)`, this function takes an owned value and return the
    value by move.

    This is an identity function that hints to the compiler to be maximally
    pessimistic about what it could do with the value. Unlike normal identity
    functions, `black_box` prevents the compiler from making assumptions about
    the input value or optimizing across the function call boundary.

    The primary use case is in benchmarking, where you want to measure the
    performance of code as it would execute with unknown inputs at runtime,
    rather than with compile-time constants that the compiler can optimize away.

    Parameters:
        T: The type of the value.

    Args:
        take: The value to prevent optimization on.

    Returns:
        The same value, but the compiler cannot make assumptions about it or
        optimize across this function boundary.
    """
    keep(take)
    return take^
