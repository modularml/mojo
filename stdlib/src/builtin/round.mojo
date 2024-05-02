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
"""Provides the `round` function.

These are Mojo built-ins, so you don't need to import them.
"""


trait Roundable:
    """
    The `Roundable` trait describes a type that defines an rounded value
    operation.

    Types that conform to `Roundable` will work with the builtin `round`
    function. The round operation always returns the same type as the input.

    For example:
    ```mojo
    @value
    struct Complex(Roundable):
        var re: Float64
        var im: Float64

        fn __round__(self) -> Self:
            return Self(round(re), round(im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __round__(self) -> Self:
        ...


@always_inline
fn round[T: Roundable](value: T) -> T:
    """Get the rounded value of the given object.

    Parameters:
        T: The type conforming to Roundable.

    Args:
        value: The object to get the rounded value of.

    Returns:
        The rounded value of the object.
    """
    return value.__round__()
