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
"""Provides the levels the `@inline` decorator accepts.

These are in the prelude, so they need no import.
"""


struct InlineLevel(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Names the levels the `@inline` decorator accepts.

    `@inline` takes one of these, so a misspelling is caught where it is
    written, and a comptime expression can choose a level per instantiation:

    ```mojo
    @inline(.always)
    def doubled(x: Int) -> Int:
        return x * 2

    @inline(policy)
    def scaled[policy: InlineLevel](x: Int) -> Int:
        return x * 3

    def main():
        print(doubled(1) + scaled[.never](2))
    ```
    """

    var _value: Int
    """The compiler's own inline level, which `@inline` reads directly."""

    @doc_hidden
    @always_inline("builtin")
    def __init__(out self, *, value: Int):
        """Construct an `InlineLevel` from the compiler's own level.

        Args:
            value: The compiler's inline level.
        """
        self._value = value

    @always_inline("builtin")
    def __eq__(self, rhs: Self) -> Bool:
        """Compare two levels.

        Args:
            rhs: The level to compare against.

        Returns:
            True if the levels are the same.
        """
        return self._value == rhs._value

    comptime automatic = Self(value=0)
    """Leaves the decision to the compiler's heuristics."""

    comptime always = Self(value=1)
    """Always inlines the function."""

    comptime nodebug = Self(value=2)
    """Always inlines the function, and drops its debug info when inlining."""

    # 3 is AlwaysBuiltin, which `@inline` does not accept: it also needs the
    # foldability checks `@always_inline("builtin")` performs.

    comptime never = Self(value=4)
    """Never inlines the function."""
