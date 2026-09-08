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

from std.hashlib._ahash import AHasher
from std.hashlib.hasher import Hasher
from std.collections import Span
from std.pathlib import Path

from std.testing import assert_equal
from std.testing import TestSuite


struct DummyHasher(Hasher):
    var _dummy_value: UInt64

    def __init__(out self):
        self._dummy_value = 0

    def _update_with_bytes(mut self, data: Span[Byte, _]):
        for i in range(len(data)):
            self._dummy_value += data[i].cast[.uint64]()

    def _update_with_simd(mut self, value: SIMD[_, _]):
        self._dummy_value += value.cast[.uint64]().reduce_add()

    def update(mut self, value: ImmSpan[Byte, _]):
        value.__hash__(self)

    def finish(var self) -> UInt64:
        return self._dummy_value


@fieldwise_init
struct SomeHashableStruct(Hashable, ImplicitlyCopyable):
    var _value: Int64


def test_hasher() raises:
    var hasher = DummyHasher()
    var hashable = SomeHashableStruct(42)
    hashable.__hash__(hasher)
    assert_equal(hasher^.finish(), 42)


def test_hash_with_hasher() raises:
    var hashable = SomeHashableStruct(10)
    assert_equal(hash[DummyHasher](hashable), 10)


@fieldwise_init
struct ComplexHashableStruct(Hashable):
    var _value1: SomeHashableStruct
    var _value2: SomeHashableStruct

    def __hash__[H: Hasher](self, mut hasher: H):
        self._value1.__hash__(hasher)
        self._value2.__hash__(hasher)


def test_complex_hasher() raises:
    var hasher = DummyHasher()
    var hashable = ComplexHashableStruct(
        SomeHashableStruct(42), SomeHashableStruct(10)
    )
    hashable.__hash__(hasher)
    assert_equal(hasher^.finish(), 52)


def test_complex_hash_with_hasher() raises:
    var hashable = ComplexHashableStruct(
        SomeHashableStruct(42), SomeHashableStruct(10)
    )
    assert_equal(hash[DummyHasher](hashable), 52)


@fieldwise_init
struct ComplexHashableStructWithList(Hashable):
    var _value1: SomeHashableStruct
    var _value2: SomeHashableStruct
    var _value3: List[UInt8]

    def __hash__[H: Hasher](self, mut hasher: H):
        self._value1.__hash__(hasher)
        self._value2.__hash__(hasher)
        # This is okay because self is passed as read-only so the pointer will
        # be valid until at least the end of the function
        hasher._update_with_bytes(
            Span(unsafe_ptr=self._value3.unsafe_ptr(), length=len(self._value3))
        )


@fieldwise_init
struct ComplexHashableStructWithListAndWideSIMD(Hashable):
    var _value1: SomeHashableStruct
    var _value2: SomeHashableStruct
    var _value3: List[UInt8]
    var _value4: SIMD[.uint32, 4]

    def __hash__[H: Hasher](self, mut hasher: H):
        self._value1.__hash__(hasher)
        self._value2.__hash__(hasher)
        # This is okay because self is passed as read-only so the pointer will
        # be valid until at least the end of the function
        hasher._update_with_bytes(
            Span(unsafe_ptr=self._value3.unsafe_ptr(), length=len(self._value3))
        )
        self._value4.__hash__(hasher)


def test_update_with_bytes() raises:
    var hasher = DummyHasher()
    var hashable = ComplexHashableStructWithList(
        SomeHashableStruct(42), SomeHashableStruct(10), [UInt8(1), 2, 3]
    )
    hashable.__hash__(hasher)
    assert_equal(hasher^.finish(), 58)


comptime _hash_with_hasher = hash[
    T=_, HasherType=AHasher[SIMD[.uint64, 4](0, 0, 0, 0)]
]


def test_with_ahasher() raises:
    var hashable1 = ComplexHashableStructWithList(
        SomeHashableStruct(42), SomeHashableStruct(10), [UInt8(1), 2, 3]
    )
    var hash_value = _hash_with_hasher(hashable1)
    assert_equal(hash_value, 7948090191592501094)
    var hashable2 = ComplexHashableStructWithListAndWideSIMD(
        SomeHashableStruct(42),
        SomeHashableStruct(10),
        [UInt8(1), 2, 3],
        SIMD[.uint32, 4](1, 2, 3, 4),
    )
    hash_value = _hash_with_hasher(hashable2)
    assert_equal(hash_value, 1754891767834419861)


def test_hash_hashable_with_hasher_types() raises:
    assert_equal(_hash_with_hasher(DType.uint64), 6529703120343940753)
    assert_equal(_hash_with_hasher(StaticString("")), 11583516797109448887)
    assert_equal(_hash_with_hasher(String()), 11583516797109448887)
    assert_equal(_hash_with_hasher(StringSlice("")), 11583516797109448887)
    assert_equal(_hash_with_hasher(Int(-123)), 4720193641311814362)
    assert_equal(_hash_with_hasher(UInt(123)), 4498397628805512285)
    assert_equal(
        _hash_with_hasher(SIMD[.float16, 4](0.1, -0.1, 12, 0)),
        9316495345323385448,
    )
    assert_equal(_hash_with_hasher(Path("/tmp")), 16491058316913697698)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
