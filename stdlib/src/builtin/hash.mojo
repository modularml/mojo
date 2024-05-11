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
"""Defines the `Hashable` and `Hasher` traits. Implements DefaultHasher and `hash()` built-in function."""


trait Hasher:
    """Trait which every hash function implementer needs to implement."""

    fn __init__(inout self):
        """Expects a no argument instantiation."""
        ...

    fn _update_with_bytes(
        inout self, data: DTypePointer[DType.uint8], length: Int
    ):
        """Conribute to the hash value based on a sequence of bytes. Use only for complex types which are not just a composition of Hashable types.
        """
        ...

    fn _update_with_simd[
        dt: DType, size: Int
    ](inout self, value: SIMD[dt, size]):
        """Contribute to the hash value with a compile time know fix size value. Used inside of std lib to avoid runtime branching.
        """
        ...

    # fn update[T: Hashable](inout self, value: T):
    #     """Contribute to the hash value with a Hashable value. Should be used by implementors of Hashable types which are a composition of Hashable types.
    #     """
    #     ...

    fn _finish[dt: DType = DType.uint64](owned self) -> Scalar[dt]:
        """Used internally to generate the final hash value, should be simplified to `_finish(owned self) -> Scalar[hash_value_dt]`
        once trait declarations support parameters and we can switch to `trait Hasher[hash_value_dt: DType]`.
        This is beneficial as hash functions could have different implementations based on the type.
        """
        ...


trait Hashable:
    """A trait for types which want to be able to hash their data.

    For example as following:
    ```mojo
    @value
    struct Person(Hashable):
        var name: String
        var age: Int
        fn __hash__[H: Hasher](self, inout hasher: H):
            # hasher.update(self.name)
            self.name.__hash__(hasher)
            # hasher.update(self.age)
            self.age.__hash__(hasher)

    var foo = Person("Alex", 42)
    print(hash(foo))
    ```
    """

    fn __hash__[H: Hasher](self, inout hasher: H):
        """Call the update function on hasher with values which need to be considered for hashing.
        """
        ...


# alias HasherKey = U256(
#     random.random_ui64(0, UInt64.MAX),
#     random.random_ui64(0, UInt64.MAX),
#     random.random_ui64(0, UInt64.MAX),
#     random.random_ui64(0, UInt64.MAX),
# )
alias DefaultHasher = AHasher[U256(0)]


fn hash[
    V: Hashable, hasher_type: Hasher = DefaultHasher, dt: DType = DType.uint64
](hashable: V) -> Scalar[dt]:
    """Hash a Hashable type using provided hasher type.

    Parameters:
        V: Any Hashable type.
        hasher_type: Hasher type defaults to std lib DefaultHasher.
        dt: Hash value dtype.

    Args:
        hashable: The input data to hash.

    Returns:
        A hash value of provided dtype.
    """
    var hasher = hasher_type()
    # hasher.update(hashable)
    hashable.__hash__(hasher)
    return hasher^._finish[dt]()


# ===----------------------------------------------------------------------=== #
# AHasher implementation based on https://github.com/tkaitchuck/aHash
# ===----------------------------------------------------------------------=== #


alias U256 = SIMD[DType.uint64, 4]
alias U128 = SIMD[DType.uint64, 2]
alias MULTIPLE = 6364136223846793005
alias ROT = 23


@always_inline("nodebug")
fn _bswap(val: SIMD) -> __type_of(val):
    return llvm_intrinsic["llvm.bswap", __type_of(val), has_side_effect=False](
        val
    )


fn _rotate_bits_left(value: UInt64, shift: Int) -> UInt64:
    return (value << shift) | (value >> (64 - shift))


@always_inline
fn _folded_multiply(s: UInt64, by: UInt64) -> UInt64:
    var b1 = s * _bswap(by)
    var b2 = _bswap(s) * (~by)
    return b1 ^ _bswap(b2)


@always_inline
fn _read_small(data: DTypePointer[DType.uint8], length: Int) -> U128:
    if length >= 2:
        if length >= 4:
            # len 4-8
            var a = data.bitcast[DType.uint32]().load().cast[DType.uint64]()
            var b = data.offset(length - 4).bitcast[DType.uint32]().load().cast[
                DType.uint64
            ]()
            return U128(a, b)
        else:
            var a = data.bitcast[DType.uint16]().load().cast[DType.uint64]()
            var b = data.offset(length - 1).load().cast[DType.uint64]()
            return U128(a, b)
    else:
        if length > 0:
            var a = data.load().cast[DType.uint64]()
            return U128(a, a)
        else:
            return U128(0, 0)


struct AHasher[secret_key: U256 = U256(0)](Hasher):
    var buffer: UInt64
    var pad: UInt64
    var extra_keys: U128

    fn __init__(inout self):
        var key = U256(0)
        var pi_key = key ^ U256(
            0x243F_6A88_85A3_08D3,
            0x1319_8A2E_0370_7344,
            0xA409_3822_299F_31D0,
            0x082E_FA98_EC4E_6C89,
        )
        self.buffer = pi_key[0]
        self.pad = pi_key[1]
        self.extra_keys = U128(pi_key[2], pi_key[3])

    @always_inline
    fn _update_with_simd[
        dt: DType, size: Int
    ](inout self, value: SIMD[dt, size]):
        # TODO: think about utilizing large_update
        @unroll
        for i in range(size):
            self.buffer = _folded_multiply(
                value[i].cast[DType.uint64]() ^ self.buffer, MULTIPLE
            )

    @always_inline
    fn _large_update(inout self, new_data: U128):
        var combined = _folded_multiply(
            new_data[0] ^ self.extra_keys[0], new_data[1] ^ self.extra_keys[1]
        )
        self.buffer = _rotate_bits_left(
            (self.buffer + self.pad) ^ combined, ROT
        )

    @always_inline
    fn _finish[dt: DType = DType.uint64](owned self) -> Scalar[dt]:
        var rot = self.buffer & 63
        var folded = _folded_multiply(self.buffer, self.pad)
        var result = _rotate_bits_left(folded, int(rot))
        return bitcast[dt, 1](result)

    @always_inline
    fn _update_with_bytes(
        inout self, data: DTypePointer[DType.uint8], length: Int
    ):
        self.buffer = (self.buffer + length) * MULTIPLE
        if length > 8:
            if length > 16:
                var tail = data.offset(length - 16).bitcast[
                    DType.uint64
                ]().load[width=2]()
                self._large_update(tail)
                var offset = 0
                while length - offset > 16:
                    var block = data.offset(offset).bitcast[
                        DType.uint64
                    ]().load[width=2]()
                    self._large_update(block)
                    offset += 16
            else:
                var a = data.bitcast[DType.uint64]().load()
                var b = data.offset(length - 8).bitcast[DType.uint64]().load()
                self._large_update(U128(a, b))
        else:
            var value = _read_small(data, length)
            self._large_update(value)

    # @always_inline
    # fn update[T: Hashable](inout self, value: T):
    #     value.__hash__(self)
