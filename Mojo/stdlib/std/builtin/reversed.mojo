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
"""Provides the `reversed` function for reverse iteration over collections.

These are Mojo built-ins, so you don't need to import them.
"""

from std.builtin.rebind import downcast
from std.collections import Deque
from std.collections.deque import _DequeIter
from std.collections.dict import _DictEntryIter, _DictKeyIter, _DictValueIter
from std.collections.list import _ListIter
from std.collections.array import _ArrayIter
from std.hashlib import Hasher

from std.collections.span import Span, _SpanIter

# ===----------------------------------------------------------------------=== #
#  Reversible
# ===----------------------------------------------------------------------=== #


trait ReversibleRange:
    """
    The `ReversibleRange` trait describes a range that can be reversed.

    Any type that conforms to `ReversibleRange` works with the builtin
    [`reversed()`](/docs/std/builtin/reversed/reversed/) functions.

    The `ReversibleRange` trait requires the type to define the `__reversed__()`
    method and a `ReversedType` iterator, so each conforming range can return
    its own reversed iterator type instead of a single hard-coded one.

    **Note**: iterators are currently non-raising.
    """

    # TODO: general `Reversible` trait that returns an iterator.
    # iterators currently check __len__() instead of raising an exception
    # so there is no ReversibleRaising trait yet.

    comptime ReversedType: Iterator
    """The iterator type returned by `__reversed__()`."""

    def __reversed__(self) -> Self.ReversedType:
        """Get a reversed iterator for the type.

        **Note**: iterators are currently non-raising.

        Returns:
            The reversed iterator of the type.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  reversed
# ===----------------------------------------------------------------------=== #


def reversed[T: ReversibleRange](value: T) -> T.ReversedType:
    """Get a reversed iterator of the input range.

    **Note**: iterators are currently non-raising.

    Parameters:
        T: The type conforming to ReversibleRange.

    Args:
        value: The range to get the reversed iterator of.

    Returns:
        The reversed iterator of the range.
    """
    return value.__reversed__()


def reversed[T: Copyable](ref value: List[T]) -> type_of(value.__reversed__()):
    """Get a reversed iterator of the input list.

    **Note**: iterators are currently non-raising.

    Parameters:
        T: The type of the elements in the list.

    Args:
        value: The list to get the reversed iterator of.

    Returns:
        The reversed iterator of the list.
    """
    return value.__reversed__()


def reversed[
    T: Copyable, size: Int
](ref value: Array[T, size]) -> _ArrayIter[T, size, origin_of(value), False]:
    """Get a reversed iterator of the input array.

    **Note**: iterators are currently non-raising.

    Parameters:
        T: The type of the elements in the array.
        size: The size of the array.

    Args:
        value: The array to get the reversed iterator of.

    Returns:
        The reversed iterator of the array.
    """
    return value.__reversed__()


def reversed[
    T: Copyable & Deinitable
](ref value: Deque[T]) -> _DequeIter[T, origin_of(value), False]:
    """Get a reversed iterator of the deque.

    **Note**: iterators are currently non-raising.

    Parameters:
        T: The type of the elements in the deque.

    Args:
        value: The deque to get the reversed iterator of.

    Returns:
        The reversed iterator of the deque.
    """
    return value.__reversed__()


def reversed[
    K: KeyElement & Copyable,
    V: Copyable,
    H: Hasher,
](ref value: Dict[K, V, H],) -> _DictKeyIter[K, V, H, origin_of(value), False]:
    """Get a reversed iterator of the input dict.

    **Note**: iterators are currently non-raising.

    Parameters:
        K: The type of the keys in the dict.
        V: The type of the values in the dict.
        H: The type of the hasher in the dict.

    Args:
        value: The dict to get the reversed iterator of.

    Returns:
        The reversed iterator of the dict keys.
    """
    return value.__reversed__()


def reversed[
    dict_mutability: Bool,
    //,
    K: KeyElement & Copyable,
    V: Copyable,
    H: Hasher,
    dict_origin: Origin[mut=dict_mutability],
](ref value: _DictValueIter[K, V, H, dict_origin]) -> _DictValueIter[
    K, V, H, dict_origin, False
]:
    """Get a reversed iterator of the input dict values.

    **Note**: iterators are currently non-raising.

    Parameters:
        dict_mutability: Whether the reference to the dict values is mutable.
        K: The type of the keys in the dict.
        V: The type of the values in the dict.
        H: The type of the hasher in the dict.
        dict_origin: The origin of the dict values.

    Args:
        value: The dict values to get the reversed iterator of.

    Returns:
        The reversed iterator of the dict values.
    """
    return value.__reversed__()


def reversed[
    dict_mutability: Bool,
    //,
    K: KeyElement & Copyable,
    V: Copyable,
    H: Hasher,
    dict_origin: Origin[mut=dict_mutability],
](ref value: _DictEntryIter[K, V, H, dict_origin]) -> _DictEntryIter[
    K, V, H, dict_origin, False
]:
    """Get a reversed iterator of the input dict items.

    **Note**: iterators are currently non-raising.

    Parameters:
        dict_mutability: Whether the reference to the dict items is mutable.
        K: The type of the keys in the dict.
        V: The type of the values in the dict.
        H: The type of the hasher in the dict.
        dict_origin: The origin of the dict items.

    Args:
        value: The dict items to get the reversed iterator of.

    Returns:
        The reversed iterator of the dict items.
    """
    var src = value.src
    return _DictEntryIter[K, V, H, dict_origin, False](
        len(src[]._order) - 1, 0, src
    )


@always_inline
def reversed[
    T: Copyable
](value: Span[T, _]) -> _SpanIter[T, value.origin, forward=False]:
    """Get a reversed iterator of the input Span.

    **Note**: iterators are currently non-raising.

    Parameters:
        T: The type of the elements in the Span.

    Args:
        value: The Span to get the reversed iterator of.

    Returns:
        The reversed iterator of the Span.
    """
    return rebind[_SpanIter[T, value.origin, forward=False]](
        value.__reversed__()
    )
