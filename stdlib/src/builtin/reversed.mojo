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

from .range import _StridedRangeIterator

from collections.list import _ListIter

from collections.dict import _DictKeyIter

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

    fn __reversed__(self) -> _StridedRangeIterator:
        """Get a reversed iterator for the type.

        **Note**: iterators are currently non-raising.

        Returns:
            The reversed iterator of the type.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  reversed
# ===----------------------------------------------------------------------=== #


fn reversed[T: ReversibleRange](value: T) -> _StridedRangeIterator:
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
    mutability: __mlir_type.`i1`,
    self_life: AnyLifetime[mutability].type,
    T: CollectionElement,
](
    value: Reference[List[T], mutability, self_life]._mlir_type,
) -> _ListIter[
    T, mutability, self_life, False
]:
    """Get a reversed iterator of the input list.

    **Note**: iterators are currently non-raising.

    Args:
        value: The list to get the reversed iterator of.

    Returns:
        The reversed iterator of the list.
    """
    return Reference(value)[].__reversed__[mutability, self_life]()


fn reversed[
    mutability: __mlir_type.`i1`,
    self_life: AnyLifetime[mutability].type,
    K: KeyElement,
    V: CollectionElement,
](
    value: Reference[Dict[K, V], mutability, self_life]._mlir_type,
) -> _DictKeyIter[K, V, mutability, self_life, False]:
    """Get a reversed iterator of the input list.

    **Note**: iterators are currently non-raising.

    Args:
        value: The list to get the reversed iterator of.

    Returns:
        The reversed iterator of the list.
    """
    return Reference(value)[].__reversed__[mutability, self_life]()
