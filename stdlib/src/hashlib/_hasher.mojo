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

from ._ahash import AHasher
from memory import UnsafePointer


trait _HashableWithHasher:
    fn __hash__[H: _Hasher](self, inout hasher: H):
        ...


trait _Hasher:
    fn __init__(inout self):
        ...

    fn _update_with_bytes(inout self, data: UnsafePointer[UInt8], length: Int):
        ...

    fn _update_with_simd(inout self, value: SIMD[_, _]):
        ...

    fn update[T: _HashableWithHasher](inout self, value: T):
        ...

    fn finish(owned self) -> UInt64:
        ...


alias default_hasher = AHasher[SIMD[DType.uint64, 4](0, 0, 0, 0)]


fn _hash_with_hasher[
    HashableType: _HashableWithHasher, HasherType: _Hasher = default_hasher
](hashable: HashableType) -> UInt64:
    var hasher = HasherType()
    hasher.update(hashable)
    var value = hasher^.finish()
    return value


fn _hash_with_hasher[
    HasherType: _Hasher = default_hasher
](data: UnsafePointer[UInt8], len: Int) -> UInt64:
    var hasher = HasherType()
    hasher._update_with_bytes(data, len)
    var value = hasher^.finish()
    return value
