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
"""Implements the `Hashable` trait and `hash()` built-in function.

There are a few main tools in this module:

- `Hashable` trait for types implementing `__hash__(self) -> Int`
- `hash[T: Hashable](hashable: T) -> Int` built-in function.
- A `hash()` implementation for arbitrary byte strings,
  `hash(data: DTypePointer[DType.int8], n: Int) -> Int`,
  is the workhorse function, which implements efficient hashing via SIMD
  vectors. See the documentation of this function for more details on the hash
  implementation.
- `hash(SIMD)` and `hash(Int8)` implementations
    These are useful helpers to specialize for the general bytes implementation.
"""

import random
from sys.ffi import _get_global

from memory import memcpy, memset_zero, stack_allocation

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _div_ceil_positive(numerator: Int, denominator: Int) -> Int:
    return (numerator + denominator - 1)._positive_div(denominator)


# ===----------------------------------------------------------------------=== #
# Implementation
# ===----------------------------------------------------------------------=== #

# This hash secret is XOR-ed with the final hash value for common hash functions.
# Doing so can help prevent DDOS attacks on data structures relying on these
# hash functions. See `hash(bytes, n)` documentation for more details.
# TODO(27659): This is always 0 right now
# var HASH_SECRET = int(random.random_ui64(0, UInt64.MAX)


fn _HASH_SECRET() -> Int:
    var ptr = _get_global[
        "HASH_SECRET", _initialize_hash_secret, _destroy_hash_secret
    ]()
    return ptr.bitcast[Int]()[0]


fn _initialize_hash_secret(
    payload: UnsafePointer[NoneType],
) -> UnsafePointer[NoneType]:
    var secret = random.random_ui64(0, UInt64.MAX)
    var data = UnsafePointer[Int].alloc(1)
    data[] = int(secret)
    return data.bitcast[NoneType]()


fn _destroy_hash_secret(p: UnsafePointer[NoneType]):
    p.free()


trait Hashable:
    """A trait for types which specify a function to hash their data.

    This hash function will be used for applications like hash maps, and
    don't need to be cryptographically secure. A good hash function will
    hash similar / common types to different values, and in particular
    the _low order bits_ of the hash, which are used in smaller dictionaries,
    should be sensitive to any changes in the data structure. If your type's
    hash function doesn't meet this criteria it will get poor performance in
    common hash map implementations.

    ```mojo
    @value
    struct Foo(Hashable):
        fn __hash__(self) -> Int:
            return 4  # chosen by fair random dice roll

    var foo = Foo()
    print(hash(foo))
    ```
    """

    fn __hash__(self) -> Int:
        """Return a 64-bit hash of the type's data."""
        ...


fn hash[T: Hashable](hashable: T) -> Int:
    """Hash a Hashable type using its underlying hash implementation.

    Parameters:
        T: Any Hashable type.

    Args:
        hashable: The input data to hash.

    Returns:
        A 64-bit integer hash based on the underlying implementation.
    """
    return hashable.__hash__()


fn _djbx33a_init[type: DType, size: Int]() -> SIMD[type, size]:
    return SIMD[type, size](5361)


fn _djbx33a_hash_update[
    type: DType, size: Int
](data: SIMD[type, size], next: SIMD[type, size]) -> SIMD[type, size]:
    return data * 33 + next


alias _HASH_INIT = _djbx33a_init
alias _HASH_UPDATE = _djbx33a_hash_update


@always_inline
fn _hash_simd[type: DType, size: Int](data: SIMD[type, size]) -> Int:
    """Hash a SIMD byte vector using direct DJBX33A hash algorithm.

    See `hash(bytes, n)` documentation for more details.

    Parameters:
        type: The SIMD dtype of the input data.
        size: The SIMD width of the input data.

    Args:
        data: The input data to hash.

    Returns:
        A 64-bit integer hash. This hash is _not_ suitable for
        cryptographic purposes, but will have good low-bit
        hash collision statistical properties for common data structures.
    """
    # Some types will have non-integer ratios, eg. DType.bool
    alias int8_size = _div_ceil_positive(
        type.bitwidth(), DType.uint8.bitwidth()
    ) * size
    # Stack allocate bytes for `data` and load it into that memory.
    # Then reinterpret as int8 and pass to the specialized int8 hash function.
    # - Ensure that the alignment matches both types, otherwise
    #   an aligned load or store will be offset and cause
    #   nondeterminism (read) or memory corruption (write)
    # TODO(#31160): use math.lcm
    # Technically this is LCM, but alignments should always be multiples of 2.
    alias alignment = max(
        alignof[SIMD[type, size]](), alignof[SIMD[DType.uint8, int8_size]]()
    )
    var bytes = stack_allocation[int8_size, DType.uint8, alignment=alignment]()
    memset_zero(bytes, int8_size)
    bytes.bitcast[type]().store[width=size](data)
    return _hash_int8(bytes.load[width=int8_size]())


fn _hash_int8[size: Int](data: SIMD[DType.uint8, size]) -> Int:
    """Hash a SIMD byte vector using direct DJBX33A hash algorithm.

    This naively implements DJBX33A, with a hash secret appended at the end.
    The hash secret is computed randomly at compile time, so different executions
    will use different secrets, and thus have different hash outputs. This is
    useful in preventing DDOS attacks against hash functions using a
    non-cryptographic hash function like DJBX33A.

    See `hash(bytes, n)` documentation for more details.

    Parameters:
        size: The SIMD width of the input data.

    Args:
        data: The input data to hash.

    Returns:
        A 64-bit integer hash. This hash is _not_ suitable for
        cryptographic purposes, but will have good low-bit
        hash collision statistical properties for common data structures.
    """
    var hash_data = _HASH_INIT[DType.int64, 1]()
    for i in range(size):
        hash_data = _HASH_UPDATE(hash_data, data[i].cast[DType.int64]())
    # TODO(27659): 'lit.globalvar.ref' error
    # return int(hash_data) ^ HASH_SECRET
    return int(hash_data) ^ _HASH_SECRET()


fn hash(bytes: DTypePointer[DType.uint8], n: Int) -> Int:
    """Hash a byte array using a SIMD-modified DJBX33A hash algorithm.

    Similar to `hash(bytes: DTypePointer[DType.int8], n: Int) -> Int` but
    takes a `DTypePointer[DType.uint8]` instead of `DTypePointer[DType.int8]`.
    See the overload for a complete description of the algorithm.

    Args:
        bytes: The byte array to hash.
        n: The length of the byte array.

    Returns:
        A 64-bit integer hash. This hash is _not_ suitable for
        cryptographic purposes, but will have good low-bit
        hash collision statistical properties for common data structures.
    """
    return hash(bytes.bitcast[DType.int8](), n)


# TODO: Remove this overload once we have finished the transition to uint8
# for bytes. See https://github.com/modularml/mojo/issues/2317
fn hash(bytes: DTypePointer[DType.int8], n: Int) -> Int:
    """Hash a byte array using a SIMD-modified DJBX33A hash algorithm.

    The DJBX33A algorithm is commonly used for data structures that rely
    on well-distributed hashing for performance. The low order bits of the
    result depend on each byte in the input, meaning that single-byte changes
    will result in a changed hash even when masking out most bits eg. for small
    dictionaries.

    _This hash function is not suitable for cryptographic purposes._ The
    algorithm is easy to reverse and produce deliberate hash collisions.
    We _do_ however initialize a random hash secret which is mixed into
    the final hash output. This can help prevent DDOS attacks on applications
    which make use of this function for dictionary hashing. As a consequence,
    hash values are deterministic within an individual runtime instance ie.
    a value will always hash to the same thing, but in between runs this value
    will change based on the hash secret.

    Standard DJBX33A is:

    - Set _hash_ = 5361
    - For each byte: _hash_ = 33 * _hash_ + _byte_

    Instead, for all bytes except trailing bytes that don't align
    to the max SIMD vector width, we:

    - Interpret those bytes as a SIMD vector.
    - Apply a vectorized hash: _v_ = 33 * _v_ + _bytes_as_simd_value_
    - Call [`reduce_add()`](/mojo/stdlib/builtin/simd/SIMD#reduce_add) on the
      final result to get a single hash value.
    - Use this value in fallback for the remaining suffix bytes
      with standard DJBX33A.

    Python uses DJBX33A with a hash secret for smaller strings, and
    then the SipHash algorithm for longer strings. The arguments and tradeoffs
    are well documented in PEP 456. We should consider this and deeper
    performance/security tradeoffs as Mojo evolves.

    References:

    - [Wikipedia: Non-cryptographic hash function](https://en.wikipedia.org/wiki/Non-cryptographic_hash_function)
    - [Python PEP 456](https://peps.python.org/pep-0456/)
    - [PHP Hash algorithm and collisions](https://www.phpinternalsbook.com/php5/hashtables/hash_algorithm.html)


    ```mojo
    from random import rand
    var n = 64
    var rand_bytes = DTypePointer[DType.int8].alloc(n)
    rand(rand_bytes, n)
    hash(rand_bytes, n)
    ```

    Args:
        bytes: The byte array to hash.
        n: The length of the byte array.

    Returns:
        A 64-bit integer hash. This hash is _not_ suitable for
        cryptographic purposes, but will have good low-bit
        hash collision statistical properties for common data structures.
    """
    alias type = DType.int64
    alias type_width = type.bitwidth() // DType.int8.bitwidth()
    alias simd_width = simdwidthof[type]()
    # stride is the byte length of the whole SIMD vector
    alias stride = type_width * simd_width

    # Compute our SIMD strides and tail length
    # n == k * stride + r
    var k = n // stride
    var r = n % stride
    debug_assert(n == k * stride + r, "wrong hash tail math")

    # 1. Reinterpret the underlying data as a larger int type
    var simd_data = bytes.bitcast[type]()

    # 2. Compute DJBX33A, but strided across the SIMD vector width.
    #    This is almost the same as DBJX33A, except:
    #    - The order in which bytes of data update the hash is permuted
    #    - For larger inputs, a small constant number of bytes from the
    #      beginning of the string (3/4 of the first vector load)
    #      have a slightly different power of 33 as a coefficient.
    var hash_data = _HASH_INIT[type, simd_width]()
    for i in range(k):
        var update = simd_data.load[width=simd_width](i * simd_width)
        hash_data = _HASH_UPDATE(hash_data, update)

    # 3. Copy the tail data (smaller than the SIMD register) into
    #    a final hash state update vector that's stack-allocated.
    if r != 0:
        var remaining = StaticTuple[Int8, stride]()
        var ptr = DTypePointer[DType.int8](
            UnsafePointer.address_of(remaining).bitcast[Int8]()
        )
        memcpy(ptr, bytes + k * stride, r)
        memset_zero(ptr + r, stride - r)  # set the rest to 0
        var last_value = ptr.bitcast[type]().load[width=simd_width]()
        hash_data = _HASH_UPDATE(hash_data, last_value)

    # Now finally, hash the final SIMD vector state. This will also use
    # DJBX33A to make sure that higher-order bits of the vector will
    # mix and impact the low-order bits, and is mathematically necessary
    # for this function to equate to naive DJBX33A.
    return _hash_simd(hash_data)
