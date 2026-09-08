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
"""Provides functions for pseudorandom numbers.

You can import these APIs from the `random` package. For example:

```mojo
from std.random import seed
```

These functions use a shared, global pseudorandom number generator (PRNG)
state. The global random state is shared across threads and concurrent
access can cause race conditions and undefined behavior.

Warning:
    NOT cryptographically secure. This PRNG is suitable for simulations,
    games, and general statistical purposes, but shouldn't be used for
    security-sensitive applications such as generating passwords,
    authentication tokens, or encryption keys.
"""

import std.math
from std.math import floor
from std.time import perf_counter_ns

from ._rng import _get_global_random_state


def seed():
    """Seeds the random number generator using a time-based value.

    Example:

    ```mojo
    from std.random import seed

    seed()
    ```
    """
    seed(perf_counter_ns())


def seed(a: Int):
    """Seeds the random number generator using the value provided.

    Args:
        a: The seed value to initialize the PRNG state.

    Example:

    ```mojo
    from std.random import seed

    seed(123456)
    ```
    """
    _get_global_random_state()[].seed(UInt64(a))


def random_float64(min: Float64 = 0, max: Float64 = 1) -> Float64:
    """Returns a random `Float64` number from the given range [min, max).

    Args:
        min: The minimum number in the range (inclusive, default is 0.0).
        max: The maximum number in the range (exclusive, default is 1.0).

    Returns:
        A random number sampled uniformly from [min, max).

    Example:
    ```mojo
    from std.random import random_float64, seed

    seed()
    var rnd = random_float64(10.0, 20.0)
    print(rnd)  # Random float between 10.0 and 20.0
    ```
    """
    return _get_global_random_state()[].random_float64(min, max)


def random_si64(min: Int64, max: Int64) -> Int64:
    """Returns a random `Int64` number from the given range [min, max].

    Args:
        min: The minimum number in the range (inclusive).
        max: The maximum number in the range (inclusive).

    Returns:
        A random integer sampled uniformly from [min, max].

    Example:

    ```mojo
    from std.random import random_si64, seed

    seed()
    var rnd = random_si64(-100, 100)
    print(rnd)  # Random Int64 between -100 and 100
    ```
    """
    return _get_global_random_state()[].random_int64(min, max)


def random_ui64(min: UInt64, max: UInt64) -> UInt64:
    """Returns a random `UInt64` number from the given range [min, max].

    Args:
        min: The minimum number in the range (inclusive).
        max: The maximum number in the range (inclusive).

    Returns:
        A random unsigned integer sampled uniformly from [min, max].

    Example:

    ```mojo
    from std.random import random_ui64, seed

    seed()
    var rnd = random_ui64(0, 100)
    print(rnd)  # Random UInt64 between 0 and 100
    ```
    """
    return _get_global_random_state()[].random_uint64(min, max)


def randint[
    dtype: DType
](
    span: MutSpan[Scalar[dtype], _],
    low: Int,
    high: Int,
) where dtype.is_integral():
    """Fills memory with uniformly distributed random integers in the range [low, high].

    Constraints:
        The type must be integral.

    Parameters:
        dtype: The dtype of the pointer.

    Args:
        span: The memory area to fill.
        low: The inclusive lower bound.
        high: The inclusive upper bound.
    """
    randint(span.unsafe_ptr(), len(span), low, high)


def randint[
    dtype: DType
](
    ptr: MutPointer[Scalar[dtype], _],
    size: Int,
    low: Int,
    high: Int,
) where dtype.is_integral():
    """Fills memory with uniformly distributed random integers in the range [low, high].

    Constraints:
        The type must be integral.

    Parameters:
        dtype: The dtype of the pointer.

    Args:
        ptr: The pointer to the memory area to fill.
        size: The number of elements to fill.
        low: The inclusive lower bound.
        high: The inclusive upper bound.

    Example:

    ```mojo
    from std.random import randint, seed
    seed()
    var size: Int = 10
    var data = List(length=size, fill=Int32(0))
    randint(data, -50, 50)
    for i in range(size):
        print(data[i])  # Random Int32 between -50 and 50
    ```
    """

    comptime if dtype.is_signed():
        for si in range(size):
            ptr.unsafe_offset(si)[] = random_si64(Int64(low), Int64(high)).cast[
                dtype
            ]()
    else:
        for ui in range(size):
            ptr.unsafe_offset(ui)[] = random_ui64(
                UInt64(low), UInt64(high)
            ).cast[dtype]()


def rand[
    dtype: DType
](
    span: MutSpan[Scalar[dtype], _],
    /,
    *,
    min: Float64 = 0.0,
    max: Float64 = 1.0,
    int_scale: Optional[Int] = None,
):
    """Fills memory with random values from a uniform distribution.

    Behavior depends on the dtype:
    - Floating-point types sample values uniformly from [min, max).
    - Integral types sample values uniformly from [min, max],
    clamped to the representable range of the dtype.

    Parameters:
        dtype: The dtype of the pointer.

    Args:
        span: The memory area to fill.
        min: The lower bound of the range.
        max: The upper bound of the range.
        int_scale: Optional quantization scale for floating-point types.
            When provided, values are quantized to increments of 2^(-int_scale).
    """
    rand(span.unsafe_ptr(), len(span), min=min, max=max, int_scale=int_scale)


def rand[
    dtype: DType
](
    ptr: MutPointer[Scalar[dtype], ...],
    size: Int,
    /,
    *,
    min: Float64 = 0.0,
    max: Float64 = 1.0,
    int_scale: Optional[Int] = None,
):
    """Fills memory with random values from a uniform distribution.

    Behavior depends on the dtype:
    - Floating-point types sample values uniformly from [min, max).
    - Integral types sample values uniformly from [min, max],
      clamped to the representable range of the dtype.

    Parameters:
        dtype: The dtype of the pointer.

    Args:
        ptr: The pointer to the memory area to fill.
        size: The number of elements to fill.
        min: The lower bound of the range.
        max: The upper bound of the range.
        int_scale: Optional quantization scale for floating-point types.
            When provided, values are quantized to increments of 2^(-int_scale).

    Example:

    ```mojo
    from std.random import rand, seed

    seed()
    var size: Int = 10
    var data = List(length=size, fill=Float32(0))
    rand(data, min=0.0, max=1.0, int_scale=16)
    for i in range(size):
        print(data[i])  # Random Float32 between 0.0 and 1.0
    ```
    """
    var scale_val = int_scale.or_else(-1)

    comptime if dtype.is_floating_point():
        if scale_val >= 0:
            var scale_double: Float64 = Float64(1 << scale_val)
            for i in range(size):
                var rnd = random_float64(min, max)
                ptr.unsafe_offset(i)[] = (
                    floor(rnd * scale_double) / scale_double
                ).cast[dtype]()
        else:
            for i in range(size):
                var rnd = random_float64(min, max)
                ptr.unsafe_offset(i)[] = rnd.cast[dtype]()

        return

    comptime if dtype.is_signed():
        var min_ = std.math.max(
            Scalar[dtype].MIN.cast[.int64](), min.cast[.int64]()
        )
        var max_ = std.math.min(
            max.cast[.int64](), Scalar[dtype].MAX.cast[.int64]()
        )
        for i in range(size):
            ptr.unsafe_offset(i)[] = random_si64(min_, max_).cast[dtype]()
        return

    comptime if dtype == .bool or dtype.is_unsigned():
        var min_ = std.math.max(min.cast[.uint64](), 0)
        var max_ = std.math.min(
            max.cast[.uint64](), Scalar[dtype].MAX.cast[.uint64]()
        )
        for i in range(size):
            ptr.unsafe_offset(i)[] = random_ui64(min_, max_).cast[dtype]()
        return


def randn_float64(
    mean: Float64 = 0.0, standard_deviation: Float64 = 1.0
) -> Float64:
    """Returns a random `Float64` sampled from a normal distribution.

    Args:
        mean: The mean of the distribution.
        standard_deviation: The standard deviation of the distribution.

    Returns:
        A random `Float64` sampled from Normal(mean, standard_deviation).

    Example:

    ```mojo
    from std.random import randn_float64, seed
    seed()
    var rnd = randn_float64(0.0, 1.0)
    print(rnd)  # Random Float64 from Normal(0.0, 1.0)
    ```
    """
    return _get_global_random_state()[].normal_float64(mean, standard_deviation)


def randn[
    dtype: DType
](
    span: MutSpan[Scalar[dtype], _],
    mean: Float64 = 0.0,
    standard_deviation: Float64 = 1.0,
):
    """Fills memory with random values from a Normal distribution.

    Constraints:
        The type should be floating point.

    Parameters:
        dtype: The dtype of the pointer.

    Args:
        span: The memory area to fill.
        mean: The mean of the normal distribution.
        standard_deviation: The standard deviation of the normal distribution.
    """
    randn(span.unsafe_ptr(), len(span), mean, standard_deviation)


def randn[
    dtype: DType
](
    ptr: MutPointer[Scalar[dtype], ...],
    size: Int,
    mean: Float64 = 0.0,
    standard_deviation: Float64 = 1.0,
):
    """Fills memory with random values from a Normal distribution.

    Constraints:
        The type should be floating point.

    Parameters:
        dtype: The dtype of the pointer.

    Args:
        ptr: The pointer to the memory area to fill.
        size: The number of elements to fill.
        mean: The mean of the normal distribution.
        standard_deviation: The standard deviation of the normal distribution.

    Example:

    ```mojo
    from std.random import randn, seed

    seed()
    var size: Int = 10
    var data = List(length=size, fill=Float64(0))
    randn(data, mean=0.0, standard_deviation=1.0)
    for i in range(size):
        print(data[i])  # Random Float64 from Normal(0.0, 1.0)
    ```
    """

    for i in range(size):
        ptr.unsafe_offset(i)[] = randn_float64(mean, standard_deviation).cast[
            dtype
        ]()
    return


def shuffle[T: Copyable, //](mut list: List[T]):
    """Shuffles the elements of the list randomly.

    Performs an in-place Fisher-Yates shuffle on the provided list.

    Args:
        list: The list to modify.

    Parameters:
        T: The type of element in the List.

    Example:

    ```mojo
    from std.random import shuffle
    var list: List[Int] = [0, 1, 2, 3, 4, 5]
    shuffle(list)
    print(list)  # The list elements are now in random order
    # There is a very small chance that the shuffled list is
    # the same as the original
    ```
    """
    for i in reversed(range(len(list))):
        var j = Int(random_ui64(0, UInt64(i)))
        list.swap_elements(i, j)
