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


from hashlib._hasher import _hash_with_hasher, _HashableWithHasher, _Hasher
from hashlib._ahash import AHasher
from memory import UnsafePointer
from pathlib import Path
from python import Python, PythonObject
from testing import assert_equal, assert_true
from utils import StringRef


struct DummyHasher(_Hasher):
    var _dummy_value: UInt64

    fn __init__(inout self):
        self._dummy_value = 0

    fn _update_with_bytes(inout self, data: UnsafePointer[UInt8], length: Int):
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
    assert_equal(_hash_with_hasher[HasherType=DummyHasher](hashable), 10)


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
    assert_equal(_hash_with_hasher[HasherType=DummyHasher](hashable), 52)


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
            data=self._value3.unsafe_ptr(),
            length=len(self._value3),
        )
        _ = self._value3


@value
struct ComplexHashableStructWithListAndWideSIMD(_HashableWithHasher):
    var _value1: SomeHashableStruct
    var _value2: SomeHashableStruct
    var _value3: List[UInt8]
    var _value4: SIMD[DType.uint32, 4]

    fn __hash__[H: _Hasher](self, inout hasher: H):
        hasher.update(self._value1)
        hasher.update(self._value2)
        # This is okay because self is passed as borrowed so the pointer will
        # be valid until at least the end of the function
        hasher._update_with_bytes(
            data=self._value3.unsafe_ptr(),
            length=len(self._value3),
        )
        hasher.update(self._value4)
        _ = self._value3


def test_update_with_bytes():
    var hasher = DummyHasher()
    var hashable = ComplexHashableStructWithList(
        SomeHashableStruct(42), SomeHashableStruct(10), List[UInt8](1, 2, 3)
    )
    hasher.update(hashable)
    assert_equal(hasher^.finish(), 58)


def test_with_ahasher():
    var hashable1 = ComplexHashableStructWithList(
        SomeHashableStruct(42), SomeHashableStruct(10), List[UInt8](1, 2, 3)
    )
    var hash_value = _hash_with_hasher(hashable1)
    assert_equal(hash_value, 12427888534629009331)
    var hashable2 = ComplexHashableStructWithListAndWideSIMD(
        SomeHashableStruct(42),
        SomeHashableStruct(10),
        List[UInt8](1, 2, 3),
        SIMD[DType.uint32, 4](1, 2, 3, 4),
    )
    hash_value = _hash_with_hasher(hashable2)
    assert_equal(hash_value, 9463003097190363949)


def test_hash_hashable_with_hasher_types():
    assert_equal(_hash_with_hasher(DType.uint64), 5919096275431609211)
    assert_equal(_hash_with_hasher(""), 12914568033466041247)
    assert_equal(_hash_with_hasher(str("")), 12914568033466041247)
    assert_equal(_hash_with_hasher(StringRef("")), 12914568033466041247)
    assert_equal(_hash_with_hasher(Int(-123)), 7309790389124252133)
    assert_equal(_hash_with_hasher(UInt(123)), 11416101997646518198)
    assert_equal(
        _hash_with_hasher(SIMD[DType.float16, 4](0.1, -0.1, 12, 0)),
        236488340994185196,
    )
    assert_equal(_hash_with_hasher(Path("/tmp")), 862170317972693446)
    # Hash value of PythonObject is randomized by default
    # can be deterministic if env var PYTHONHASHSEED is set
    assert_true(_hash_with_hasher(PythonObject("hello")) != 0)


def main():
    test_hasher()
    test_hash_with_hasher()
    test_complex_hasher()
    test_complexe_hash_with_hasher()
    test_update_with_bytes()
    test_with_ahasher()
    test_hash_hashable_with_hasher_types()
