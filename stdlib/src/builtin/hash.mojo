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

from builtin.dtype import _uint_type_of_width
import random
from sys.ffi import _get_global

from memory import memcpy, memset_zero, stack_allocation

# TODO remove this import onece InlineArray is moved to collections
from utils import InlineArray


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


# Based on the hash function used by ankerl::unordered_dense::hash
# https://martin.ankerl.com/2022/08/27/hashmap-bench-01/#ankerl__unordered_dense__hash
fn _ankerl_init[type: DType, size: Int]() -> SIMD[type, size]:
    alias int_type = _uint_type_of_width[type.bitwidth()]()
    alias init = Int64(-7046029254386353131).cast[int_type]()
    return SIMD[type, size](bitcast[type, 1](init))


fn _ankerl_hash_update[
    type: DType, size: Int
](data: SIMD[type, size], next: SIMD[type, size]) -> SIMD[type, size]:
    # compute the hash as though the type is uint
    alias int_type = _uint_type_of_width[type.bitwidth()]()
    var data_int = bitcast[int_type, size](data)
    var next_int = bitcast[int_type, size](next)
    var result = (data_int * next_int) ^ next_int
    return bitcast[type, size](result)


alias _HASH_INIT = _djbx33a_init
alias _HASH_UPDATE = _djbx33a_hash_update


# This is incrementally better than DJBX33A, in that it fixes some of the
# performance issue we've been seeing with Dict. It's still not ideal as
# a long-term hash function.
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

    @parameter
    if type == DType.bool:
        return _hash_simd(data.cast[DType.int8]())

    var hash_data = _ankerl_init[type, size]()
    hash_data = _ankerl_hash_update(hash_data, data)

    alias int_type = _uint_type_of_width[type.bitwidth()]()
    var final_data = bitcast[int_type, 1](hash_data[0]).cast[DType.uint64]()

    @parameter
    fn hash_value[i: Int]():
        final_data = _ankerl_hash_update(
            final_data,
            bitcast[int_type, 1](hash_data[i + 1]).cast[DType.uint64](),
        )

    unroll[hash_value, size - 1]()
    return int(final_data)


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
    """Hash a byte array using a SIMD-modified hash algorithm.

    _This hash function is not suitable for cryptographic purposes._ The
    algorithm is easy to reverse and produce deliberate hash collisions.
    The hash function is designed to have relatively good mixing and statistical
    properties for use in hash-based data structures.  We _do_ however initialize
    a random hash secret which is mixed into the final hash output. This can help
    prevent DDOS attacks on applications which make use of this function for
    dictionary hashing. As a consequence, hash values are deterministic within an
    individual runtime instance ie.  a value will always hash to the same thing,
    but in between runs this value will change based on the hash secret.

    We take advantage of Mojo's first-class SIMD support to create a
    SIMD-vectorized hash function, using some simple hash algorithm as a base.

    - Interpret those bytes as a SIMD vector, padded with zeros to align
        to the system SIMD width.
    - Apply the simple hash function parallelized across SIMD vectors.
    - Hash the final SIMD vector state to reduce to a single value.

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
    alias type = DType.uint64
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

    # 2. Compute the hash, but strided across the SIMD vector width.
    var hash_data = _HASH_INIT[type, simd_width]()
    for i in range(k):
        var update = SIMD[size=simd_width].load(simd_data, i * simd_width)
        hash_data = _HASH_UPDATE(hash_data, update)

    # 3. Copy the tail data (smaller than the SIMD register) into
    #    a final hash state update vector that's stack-allocated.
    if r != 0:
        var remaining = InlineArray[Int8, stride](unsafe_uninitialized=True)
        var ptr = DTypePointer[DType.int8](
            UnsafePointer.address_of(remaining).bitcast[Int8]()
        )
        memcpy(ptr, bytes + k * stride, r)
        memset_zero(ptr + r, stride - r)  # set the rest to 0
        var last_value = SIMD[size=simd_width].load(ptr.bitcast[type]())
        hash_data = _HASH_UPDATE(hash_data, last_value)

    # Now finally, hash the final SIMD vector state.
    return _hash_simd(hash_data)
