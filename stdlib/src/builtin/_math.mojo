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
"""Module to contain some components of the future math module.

This is needed to work around some circular dependencies; all elements of this
module should be exposed by the current `math` module. The contents of this
module should be eventually moved to the `math` module when it's open sourced.
"""

# ===----------------------------------------------------------------------===#
# Ceilable
# ===----------------------------------------------------------------------===#


trait Ceilable:
    """
      The `Ceilable` trait describes a type that defines a ceiling operation.

      Types that conform to `Ceilable` will work with the builtin `ceil`
      function. The ceiling operation always returns the same type as the input.

      For example:
      ```mojo
      from math import Ceilable, ceil

      @value
      struct Complex(Ceilable):
          var re: Float64
          var im: Float64

          fn __ceil__(self) -> Self:
              return Self(ceil(re), ceil(im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __ceil__(self) -> Self:
        ...


# ===----------------------------------------------------------------------===#
# Floorable
# ===----------------------------------------------------------------------===#


trait Floorable:
    """
    The `Floorable` trait describes a type that defines a floor operation.

    Types that conform to `Floorable` will work with the builtin `floor`
    function. The floor operation always returns the same type as the input.

    For example:
    ```mojo
    from math import Floorable, floor

    @value
    struct Complex(Floorable):
        var re: Float64
        var im: Float64

        fn __floor__(self) -> Self:
            return Self(floor(re), floor(im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __floor__(self) -> Self:
        ...
