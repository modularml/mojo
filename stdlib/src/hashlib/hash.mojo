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
"""Implements the `hash()` built-in function.
"""
from memory import UnsafePointer
from .hasher import Hashable, Hasher, default_hasher

# ===----------------------------------------------------------------------=== #
# Implementation
# ===----------------------------------------------------------------------=== #


fn hash[
    HashableType: Hashable, HasherType: Hasher = default_hasher
](hashable: HashableType) -> UInt64:
    """Hash a Hashable type using provided Hasher type.

    Parameters:
        HashableType: Any Hashable type.
        HasherType: Any Hasher type.

    Args:
        hashable: The input data to hash.

    Returns:
        A 64-bit integer hash based on the underlying implementation of the provided hasher.
    """
    var hasher = HasherType()
    hasher.update(hashable)
    var value = hasher^.finish()
    return value


fn hash[
    HasherType: Hasher = default_hasher
](bytes: UnsafePointer[UInt8], n: Int) -> UInt64:
    """Hash bytes using provided Hasher type.

    Parameters:
        HasherType: Any Hasher type.

    Args:
        bytes: The pointer to input data to hash.
        n: The length of the data.

    Returns:
        A 64-bit integer hash based on the underlying implementation of the provided hasher.
    """
    var hasher = HasherType()
    hasher._update_with_bytes(bytes, n)
    var value = hasher^.finish()
    return value
