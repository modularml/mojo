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
"""Provides the `reversed` function for reverse iteration over collections.

These are Mojo built-ins, so you don't need to import them.
"""

from .range import _StridedRange

from collections.list import _ListIter

from collections.dict import _DictKeyIter, _DictValueIter, _DictEntryIter

# ===----------------------------------------------------------------------=== #
#  Reversible
# ===----------------------------------------------------------------------=== #


trait ReversibleRange:
    """
    The `ReversibleRange` trait describes a range that can be reversed.

    Any type that conforms to `ReversibleRange` works with the builtin
    [`reversed()`](/mojo/stdlib/builtin/reversed.html) functions.

    The `ReversibleRange` trait requires the type to define the `__reversed__()`
    method.

    **Note**: iterators are currently non-raising.
    """

    # TODO: general `Reversible` trait that returns an iterator.
    # iterators currently check __len__() instead of raising an exception
    # so there is no ReversibleRaising trait yet.

    fn __reversed__(self) -> _StridedRange:
        """Get a reversed iterator for the type.

        **Note**: iterators are currently non-raising.

        Returns:
            The reversed iterator of the type.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  reversed
# ===----------------------------------------------------------------------=== #


fn reversed[T: ReversibleRange](value: T) -> _StridedRange:
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


fn reversed[
    T: CollectionElement
](ref [_]value: List[T]) -> _ListIter[T, __lifetime_of(value), False]:
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


fn reversed[
    K: KeyElement,
    V: CollectionElement,
](ref [_]value: Dict[K, V],) -> _DictKeyIter[K, V, __lifetime_of(value), False]:
    """Get a reversed iterator of the input dict.

    **Note**: iterators are currently non-raising.

    Parameters:
        K: The type of the keys in the dict.
        V: The type of the values in the dict.

    Args:
        value: The dict to get the reversed iterator of.

    Returns:
        The reversed iterator of the dict keys.
    """
    return value.__reversed__()


fn reversed[
    K: KeyElement,
    V: CollectionElement,
    dict_mutability: Bool,
    dict_lifetime: AnyLifetime[dict_mutability].type,
](ref [_]value: _DictValueIter[K, V, dict_lifetime]) -> _DictValueIter[
    K, V, dict_lifetime, False
]:
    """Get a reversed iterator of the input dict values.

    **Note**: iterators are currently non-raising.

    Parameters:
        K: The type of the keys in the dict.
        V: The type of the values in the dict.
        dict_mutability: Whether the reference to the dict values is mutable.
        dict_lifetime: The lifetime of the dict values.

    Args:
        value: The dict values to get the reversed iterator of.

    Returns:
        The reversed iterator of the dict values.
    """
    return value.__reversed__()


fn reversed[
    K: KeyElement,
    V: CollectionElement,
    dict_mutability: Bool,
    dict_lifetime: AnyLifetime[dict_mutability].type,
](ref [_]value: _DictEntryIter[K, V, dict_lifetime]) -> _DictEntryIter[
    K, V, dict_lifetime, False
]:
    """Get a reversed iterator of the input dict items.

    **Note**: iterators are currently non-raising.

    Parameters:
        K: The type of the keys in the dict.
        V: The type of the values in the dict.
        dict_mutability: Whether the reference to the dict items is mutable.
        dict_lifetime: The lifetime of the dict items.

    Args:
        value: The dict items to get the reversed iterator of.

    Returns:
        The reversed iterator of the dict items.
    """
    var src = value.src
    return _DictEntryIter[K, V, dict_lifetime, False](
        src[]._reserved() - 1, 0, src
    )
