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
# RUN: %mojo %s


from builtin._hasher import _HashableWithHasher, _Hasher, _hash_with_hasher
from testing import assert_equal


struct DummyHasher(_Hasher):
    var _dummy_value: UInt64

    fn __init__(inout self):
        self._dummy_value = 0

    fn _update_with_bytes(
        inout self, data: DTypePointer[DType.uint8], length: Int
    ):
        for i in range(length):
            self._dummy_value += data[i].cast[DType.uint64]()

    fn _update_with_simd(inout self, value: SIMD[_, _]):
        self._dummy_value += value.cast[DType.uint64]().reduce_add()

    fn update[T: _HashableWithHasher](inout self, value: T):
        value.__hash__(self)

    fn finish(owned self) -> UInt64:
        return self._dummy_value


@value
struct SomeHashableStruct(_HashableWithHasher):
    var _value: Int64

    fn __hash__[H: _Hasher](self, inout hasher: H):
        hasher._update_with_simd(self._value)


def test_hasher():
    var hasher = DummyHasher()
    var hashable = SomeHashableStruct(42)
    hasher.update(hashable)
    assert_equal(hasher^.finish(), 42)


def test_hash_with_hasher():
    var hashable = SomeHashableStruct(10)
    assert_equal(_hash_with_hasher[DummyHasher](hashable), 10)


@value
struct ComplexeHashableStruct(_HashableWithHasher):
    var _value1: SomeHashableStruct
    var _value2: SomeHashableStruct

    fn __hash__[H: _Hasher](self, inout hasher: H):
        hasher.update(self._value1)
        hasher.update(self._value2)


def test_complex_hasher():
    var hasher = DummyHasher()
    var hashable = ComplexeHashableStruct(
        SomeHashableStruct(42), SomeHashableStruct(10)
    )
    hasher.update(hashable)
    assert_equal(hasher^.finish(), 52)


def test_complexe_hash_with_hasher():
    var hashable = ComplexeHashableStruct(
        SomeHashableStruct(42), SomeHashableStruct(10)
    )
    assert_equal(_hash_with_hasher[DummyHasher](hashable), 52)


@value
struct ComplexHashableStructWithList(_HashableWithHasher):
    var _value1: SomeHashableStruct
    var _value2: SomeHashableStruct
    var _value3: List[UInt8]

    fn __hash__[H: _Hasher](self, inout hasher: H):
        hasher.update(self._value1)
        hasher.update(self._value2)
        # This is okay because self is passed as borrowed so the pointer will
        # be valid until at least the end of the function
        hasher._update_with_bytes(
            data=DTypePointer(self._value3.unsafe_ptr()),
            length=len(self._value3),
        )
        _ = self._value3


def test_update_with_bytes():
    var hasher = DummyHasher()
    var hashable = ComplexHashableStructWithList(
        SomeHashableStruct(42), SomeHashableStruct(10), List[UInt8](1, 2, 3)
    )
    hasher.update(hashable)
    assert_equal(hasher^.finish(), 58)


def main():
    test_hasher()
    test_hash_with_hasher()
    test_complex_hasher()
    test_complexe_hash_with_hasher()
    test_update_with_bytes()
