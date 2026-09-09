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
"""Implementations of vectorize functions."""

import std.sys
from std.math import align_down


# ===-----------------------------------------------------------------------===#
# Vectorize
# ===-----------------------------------------------------------------------===#


@always_inline
def vectorize[
    func: def[width: Int](idx: Int) -> None,
    //,
    simd_width: Int,
    /,
    *,
    unroll_factor: Int = 1,
](size: Int, closure: func):
    """Simplifies SIMD optimized loops by mapping a function across a range from
    0 to `size`, incrementing by `simd_width` at each step. The remainder of
    `size % simd_width` will run in separate iterations.

    Parameters:
        func: The function that will be called in the loop body.
        simd_width: The SIMD vector width.
        unroll_factor: The unroll factor for the main loop (Default 1).

    Args:
        size: The upper limit for the loop.
        closure: The captured state of the function bound to func.

    The below example demonstrates how you could improve the performance of a
    loop, by setting multiple values at the same time using SIMD registers on
    the machine:

    ```mojo
    from std.algorithm.functional import vectorize
    from std.memory.alloc import alloc, dealloc, Layout
    from std.sys import simd_width_of

    # The amount of elements to loop through
    comptime size = 10
    # How many Dtype.int32 elements fit into the SIMD register (4 on 128bit)
    comptime simd_width = simd_width_of[DType.int32]()  # assumed to be 4 in this example

    def main():
        var allocation = alloc(Layout[Int32](count=size))

        def closure[width: Int](i: Int) {mut}:
            print("storing", width, "els at pos", i)
            allocation.unsafe_ptr().store[width=width](i, Int32(i))

        vectorize[simd_width](size, closure)
        print(allocation.unsafe_ptr().load[width=simd_width]())
        print(allocation.unsafe_ptr().load[width=simd_width](simd_width))

        dealloc(allocation^)
    ```

    On a machine with a SIMD register size of 128, this will set 4xInt32 values
    on each iteration. The remainder of 10 % 4 is 2, so those last two elements
    will be set in two separate iterations:

    ```plaintext
    storing 4 els at pos 0
    storing 4 els at pos 4
    storing 1 els at pos 8
    storing 1 els at pos 9
    [0, 0, 0, 0, 4, 4, 4, 4, 8, 9]
    ```

    You can also unroll the loop to potentially improve performance at the cost
    of binary size:

    ```text
    vectorize[closure, width, unroll_factor=2](size)
    ```

    In the generated assembly the function calls will be repeated, resulting in
    fewer arithmetic, comparison, and conditional jump operations. The assembly
    would look like this in pseudocode:

    ```text
    closure[4](0)
    closure[4](4)
    # Remainder loop won't unroll unless `size` is passed as a parameter
    for i in range(8, 10):
        closure[1](i)
        closure[1](i)
    ```

    You can pass `size` as a parameter if it's compile time known to reduce the
    iterations for the remainder. This only occurs if the remainder is an
    exponent of 2 (2, 4, 8, 16, ...). The remainder loop will still unroll for
    performance improvements if not an exponent of 2.
    """
    comptime assert simd_width > 0, "simd width must be > 0"
    comptime assert unroll_factor > 0, "unroll factor must be > 0"
    assert size >= 0, "size must be >= 0"

    comptime unrolled_simd_width = simd_width * unroll_factor
    var simd_end = align_down(UInt(size), UInt(simd_width))
    var unrolled_end = align_down(UInt(size), UInt(unrolled_simd_width))

    for unrolled_idx in range(0, Int(unrolled_end), unrolled_simd_width):
        comptime for idx in range(unroll_factor):
            closure[simd_width](unrolled_idx + idx * simd_width)

    comptime if unroll_factor > 1:
        for simd_idx in range(Int(unrolled_end), Int(simd_end), simd_width):
            closure[simd_width](simd_idx)

    for i in range(Int(simd_end), size):
        closure[1](i)


@always_inline
def vectorize[
    func: def[width: Int](idx: Int, evl: Int) -> None,
    //,
    simd_width: Int,
    /,
    *,
    unroll_factor: Int = 1,
](size: Int, closure: func):
    """Simplifies SIMD optimized loops by mapping a function across a range from
    0 to `size`, incrementing by `simd_width` at each step. The main loop runs
    with a fixed SIMD width of `simd_width`. Any remainder (`size % simd_width`)
    is executed with a single final call using predication via the `evl`
    (effective vector length) argument.

    Compared to `vectorize` variants that run the remainder as scalar
    iterations (`width=1`), this version keeps the SIMD width fixed and passes
    the number of active lanes in `evl` for the last (partial) vector. The
    closure is responsible for honoring `evl` (e.g. using masked loads/stores)
    to avoid out-of-bounds accesses.

    Parameters:
        func: The function that will be called in the loop body. It must accept
            an `idx` and an `evl` (effective vector length). For all full SIMD
            iterations `evl == simd_width`. For the final partial iteration
            `0 < evl < simd_width`.
        simd_width: The SIMD vector width.
        unroll_factor: The unroll factor for the main loop (Default 1).

    Args:
        size: The upper limit for the loop.
        closure: The captured state of the function bound to func.

    The below example demonstrates how to set multiple values at the same time
    using SIMD registers, while handling the tail with `evl` by generating a mask:

    ```mojo
    from std.algorithm.functional import vectorize
    from std.memory.alloc import alloc, dealloc, Layout
    from std.sys import simd_width_of
    from std.math import iota
    from std.sys.intrinsics import masked_store

    comptime size = 10
    comptime simd_width = simd_width_of[DType.int32]()  # assumed 4 in this example

    def main():
        var allocation = alloc(Layout[Int32](count=size))

        def closure[width: Int](i: Int, evl: Int) {mut}:
            print("storing", evl, "of", width, "els at pos", i)
            var val = SIMD[.int32, width](i)

            # Optimization: Constant propagation eliminates this check in the main loop
            if evl == width:
                allocation.unsafe_ptr().store[width=width](i, val)
            else:
                # Tail loop: Generate mask from EVL to prevent OOB
                var mask = iota[.int32, width]().lt(Int32(evl))
                masked_store[width](val, allocation.unsafe_ptr() + i, mask)

        vectorize[simd_width](size, closure)

        print(allocation.unsafe_ptr().load[width=simd_width]())
        print(allocation.unsafe_ptr().load[width=simd_width](simd_width))
        print(allocation.unsafe_ptr().load[width=2](2 * simd_width))
        dealloc(allocation^)
    ```

    On a machine with a SIMD register size of 128, this will set 4xInt32 values
    on each full iteration. The remainder of 10 % 4 is 2, so the tail will be
    handled by a single call with `evl=2`:

    ```plaintext
    storing 4 of 4 els at pos 0
    storing 4 of 4 els at pos 4
    storing 2 of 4 els at pos 8
    [0, 0, 0, 0]
    [4, 4, 4, 4]
    [8, 8]
    ```

    You can also unroll the main loop to potentially improve performance at the
    cost of binary size:

    ```text
    vectorize[simd_width, unroll_factor=2](size, closure)
    ```

    In the generated assembly the full-width calls will be repeated, resulting
    in fewer arithmetic, comparison, and conditional jump operations. In
    pseudocode:

    ```text
    closure[4](0, 4)
    closure[4](4, 4)
    closure[4](8, 2)  # single predicated tail call
    ```

    Notes:
        - This implementation does not execute the remainder as scalar
          iterations. The closure must correctly handle `evl` to keep memory
          accesses in-bounds.
        - If `size < simd_width`, the loop will consist of a single call:
          `closure[simd_width](0, size)`.
    """
    comptime assert simd_width > 0, "simd width must be > 0"
    comptime assert unroll_factor > 0, "unroll factor must be > 0"
    assert size >= 0, "size must be >= 0"

    comptime unrolled_simd_width = simd_width * unroll_factor
    var simd_end = Int(align_down(UInt(size), UInt(simd_width)))
    var unrolled_end = Int(align_down(UInt(size), UInt(unrolled_simd_width)))

    for unrolled_idx in range(0, unrolled_end, unrolled_simd_width):
        comptime for idx in range(unroll_factor):
            closure[simd_width](unrolled_idx + idx * simd_width, simd_width)

    comptime if unroll_factor > 1:
        for simd_idx in range(unrolled_end, simd_end, simd_width):
            closure[simd_width](simd_idx, simd_width)

    var remainder = size - simd_end
    if remainder > 0:
        closure[simd_width](simd_end, remainder)


@always_inline
def vectorize[
    func: def[width: Int](idx: Int) -> None,
    //,
    simd_width: Int,
    /,
    *,
    size: Int,
    unroll_factor: Int = 1,
](closure: func):
    """Simplifies SIMD optimized loops by mapping a function across a range from
    0 to `size`, incrementing by `simd_width` at each step. The remainder of
    `size % simd_width` will run in a single iteration if it's an exponent of
    2.

    Parameters:
        func: The function that will be called in the loop body.
        simd_width: The SIMD vector width.
        size: The upper limit for the loop.
        unroll_factor: The unroll factor for the main loop (Default 1).

    Args:
        closure: The captured state of the function bound to func.

    The below example demonstrates how you could improve the performance of a
    loop, by setting multiple values at the same time using SIMD registers on
    the machine:

    ```mojo
    from std.algorithm.functional import vectorize
    from std.memory.alloc import alloc, dealloc, Layout
    from std.sys import simd_width_of

    # The amount of elements to loop through
    comptime size = 10
    # How many Dtype.int32 elements fit into the SIMD register (4 on 128bit)
    comptime simd_width = simd_width_of[DType.int32]()  # assumed to be 4 in this example

    def main():
        var allocation = alloc(Layout[Int32](count=size))

        def closure[width: Int](i: Int) {mut}:
            print("storing", width, "els at pos", i)
            allocation.unsafe_ptr().store[width=width](i, Int32(i))

        vectorize[simd_width](size, closure)
        print(allocation.unsafe_ptr().load[width=simd_width]())
        print(allocation.unsafe_ptr().load[width=simd_width](simd_width))

        dealloc(allocation^)
    ```

    On a machine with a SIMD register size of 128, this will set 4xInt32 values
    on each iteration. The remainder of 10 % 4 is 2, so those last two elements
    will be set in a single iteration:

    ```plaintext
    storing 4 els at pos 0
    storing 4 els at pos 4
    storing 2 els at pos 8
    [0, 0, 0, 0, 4, 4, 4, 4, 8, 8]
    ```

    If the remainder is not an exponent of 2 (2, 4, 8, 16 ...) there will be a
    separate iteration for each element. However passing `size` as a parameter
    also allows the loop for the remaining elements to be unrolled.

    You can also unroll the main loop to potentially improve performance at the
    cost of binary size:

    ```text
    vectorize[width, size=size, unroll_factor=2](closure)
    ```

    In the generated assembly the function calls will be repeated, resulting in
    fewer arithmetic, comparison, and conditional jump operations. The assembly
    would look like this in pseudocode:

    ```text
    closure[4](0)
    closure[4](4)
    closure[2](8)
    ```
    """
    comptime assert simd_width > 0, "simd width must be > 0"
    comptime assert unroll_factor > 0, "unroll factor must be > 0"
    comptime assert size >= 0, "size must be >= 0"

    comptime unrolled_simd_width = simd_width * unroll_factor
    comptime simd_end = align_down(size, simd_width)
    comptime unrolled_end = align_down(size, unrolled_simd_width)

    comptime for unrolled_idx in range(0, unrolled_end, unrolled_simd_width):
        comptime for idx in range(unroll_factor):
            closure[simd_width](unrolled_idx + idx * simd_width)

    comptime if unroll_factor > 1:
        for simd_idx in range(unrolled_end, simd_end, simd_width):
            closure[simd_width](simd_idx)

    comptime if size > simd_end:
        comptime if (size - simd_end).is_power_of_two():
            closure[size - simd_end](simd_end)
        else:
            comptime for i in range(simd_end, size):
                closure[1](i)
