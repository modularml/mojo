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

"""Implements the `Floatable` and `FloatableRaising` traits.

These are Mojo built-ins, so you don't need to import them.
"""


trait Floatable:
    """The `Floatable` trait describes a type that can be converted to a Float.

    Any type that conforms to `Floatable` works with the built-in `float`
    function.

    This trait requires the type to implement the `__float__()` method.

    For example:

    ```mojo
    @value
    struct Foo(Floatable):
        var i: Float64

        fn __float__(self) -> Float64:
            return self.i
    ```

    A `Foo` can now be converted to a `Float64` using `float`:

    ```mojo
    var f = float(Foo(5.5))
    ```

    **Note:** If the `__float__()` method can raise an error, use
    the [`FloatableRaising`](/mojo/stdlib/builtin/floatable/floatableraising)
    trait instead.
    """

    fn __float__(self) -> Float64:
        """Get the float point representation of the value.

        Returns:
            The float point representation of the value.
        """
        ...


trait FloatableRaising:
    """The `FloatableRaising` trait describes a type that can be converted to a
    Float, but the conversion might raise an error (e.g.: a string).

    Any type that conforms to `FloatableRaising` works with the built-in `float`
    function.

    This trait requires the type to implement the `__float__()` method, which
    can raise an error.

    For example:

    ```mojo
    from utils import Variant

    @value
    struct MaybeFloat(FloatableRaising):
        var value: Variant[Float64, NoneType]

        fn __float__(self) raises -> Float64:
            if self.value.isa[NoneType]():
                raise "Float expected"
            return self.value[Float64]
    ```

    A `MaybeFloat` can now be converted to `Float64` using `float`:

    ```mojo
    try:
        print(float(MaybeFloat(4.6)))
    except:
        print("error occured")
    ```
    """

    fn __float__(self) raises -> Float64:
        """Get the float point representation of the value.

        Returns:
            The float point representation of the value.

        Raises:
            If the type does not have a float point representation.
        """
        ...


@always_inline
fn float[T: Floatable](value: T, /) -> Float64:
    """Get the Float representation of the value.

    Parameters:
        T: The Floatable type.

    Args:
        value: The object to get the float point representation of.

    Returns:
        The float point representation of the value.
    """
    return value.__float__()


@always_inline
fn float[T: FloatableRaising](value: T, /) raises -> Float64:
    """Get the Float representation of the value.

    Parameters:
        T: The Floatable type.

    Args:
        value: The object to get the float point representation of.

    Returns:
        The float point representation of the value.

    Raises:
        If the type does not have a float point representation.
    """
    return value.__float__()


# TODO: Int can't conform to Floatable at the moment due to circular
#       dependency with SIMD.
@always_inline
fn float(value: Int, /) -> Float64:
    """Get the Float representation of the Int.

    Args:
        value: The Int to get the float point representation of.

    Returns:
        The float point representation of the Int.
    """
    return Float64(value)
