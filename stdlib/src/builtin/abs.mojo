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
"""Provides the `abs` function.

These are Mojo built-ins, so you don't need to import them.
"""


trait Absable:
    """
    The `Absable` trait describes a type that defines an absolute value
    operation.

    Types that conform to `Absable` will work with the builtin `abs` function.
    The absolute value operation always returns the same type as the input.

    For example:
    ```mojo
    struct Point(Absable):
        var x: Float64
        var y: Float64

        fn __abs__(self) -> Self:
            return sqrt(self.x * self.x + self.y * self.y)
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __abs__(self) -> Self:
        ...


@always_inline
fn abs[T: Absable](value: T) -> T:
    """Get the absolute value of the given object.

    Parameters:
        T: The type conforming to Absable.

    Args:
        value: The object to get the absolute value of.

    Returns:
        The absolute value of the object.
    """
    return value.__abs__()


# TODO: https://github.com/modularml/modular/issues/38694
# TODO: https://github.com/modularml/modular/issues/38695
# TODO: Remove this
@always_inline
fn abs(value: IntLiteral) -> IntLiteral:
    """Get the absolute value of the given IntLiteral.

    Args:
        value: The IntLiteral to get the absolute value of.

    Returns:
        The absolute value of the IntLiteral.
    """
    return value.__abs__()


# TODO: https://github.com/modularml/modular/issues/38694
# TODO: https://github.com/modularml/modular/issues/38695
# TODO: Remove this
@always_inline
fn abs(value: FloatLiteral) -> FloatLiteral:
    """Get the absolute value of the given FloatLiteral.

    Args:
        value: The FloatLiteral to get the absolute value of.

    Returns:
        The absolute value of the FloatLiteral.
    """
    return value.__abs__()
