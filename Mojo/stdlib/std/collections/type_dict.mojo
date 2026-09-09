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
"""Provides a data structure which acts to map a value of a given type to one
of a set of types."""


struct TypeDict[
    T: Equatable & Movable,
    Trait: type_of(AnyType),
    //,
    keys: List[T],
    *values: Trait,
](TrivialRegisterPassable):
    """A compile-time map from a value of type `T` to a type.

    `TypeDict` pairs a list of compile-time key values with a list of types of
    the same length. Looking up a key value yields the type at the matching
    position, so it acts as a value-keyed switch over types resolved entirely
    at compile time.

    Parameters:
        T: The type of the key values.
        Trait: The trait that every mapped type conforms to.
        keys: The key values, one per mapped type.
        values: The mapped types, parallel to `keys`.

    Examples:

    ```mojo
    from std.collections import TypeDict
    from std.testing import assert_equal

    comptime td = TypeDict[
        T=Int,
        Trait=AnyType,
        [1,2,3],
        Int, String, Float64,
    ]

    def main() raises:
        comptime assert td.get[1] == Int
        comptime assert td.get[2] == String
        comptime assert td.get[3] == Float64
        assert_equal(td.length, 3)
    ```
    """

    comptime length = len(Self.values)
    """The number of entries in the map."""

    comptime _index[key: Self.T] = Self.keys.try_index(key)

    comptime get[key: Self.T] = Self.values[
        Self._index[key].or_else(Self._assert_key_is_present[key]())
    ]
    """Gets the type mapped to `key`.

    Parameters:
        key: The key value to look up.

    Constraints:
        `key` must be present in `keys`.
    """

    @staticmethod
    def _assert_key_is_present[key: Self.T]() -> Int:
        """Checks if `key` is present in `keys`.

        Parameters:
            key: The key value to check.

        Returns:
            `Int.MAX` if `key` is present in `keys`, otherwise raises a compile-time
            error.
        """
        comptime assert Self._index[key], "Key is not present in TypeDict"

        return Int.MAX

    @always_inline("builtin")
    def __init__(out self):
        """Constructs a `TypeDict`.

        Constraints:
            `keys` and `values` must have the same length.
        """
        comptime assert len(Self.keys) == len(
            Self.values
        ), "TypeDict requires one type per key"

    @always_inline
    def __len__(self) -> Int:
        """Gets the number of entries in the map.

        Returns:
            The number of entries in the map (`length`).
        """
        return Self.length
