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
"""Defines some type aliases.

These are Mojo built-ins, so you don't need to import them.
"""

alias AnyTrivialRegType = __mlir_type.`!kgen.type`
"""Represents any register passable Mojo data type."""

alias ImmutableLifetime = __mlir_type.`!lit.lifetime<0>`
"""Immutable lifetime reference type."""

alias MutableLifetime = __mlir_type.`!lit.lifetime<1>`
"""Mutable lifetime reference type."""

alias ImmutableAnyLifetime = __mlir_attr.`#lit.any.lifetime<0>: !lit.lifetime<0>`
"""The immutable lifetime that might access any memory value."""

alias MutableAnyLifetime = __mlir_attr.`#lit.any.lifetime<1>: !lit.lifetime<1>`
"""The mutable lifetime that might access any memory value."""

# TODO: We don't have a "static" lifetime to use yet, so we use Any.
alias ImmutableStaticLifetime = ImmutableAnyLifetime
"""The immutable lifetime that lasts for the entire duration of program execution."""

alias MutableStaticLifetime = MutableAnyLifetime
"""The mutable lifetime that lasts for the entire duration of program execution."""

alias LifetimeSet = __mlir_type.`!lit.lifetime.set`
"""A set of lifetime parameters."""


# Helper to build a value of !lit.lifetime type.
# TODO: Should be a parametric alias.
struct AnyLifetime[is_mutable: Bool]:
    """This represents a lifetime reference of potentially parametric type.
    TODO: This should be replaced with a parametric type alias.

    Parameters:
        is_mutable: Whether the lifetime reference is mutable.
    """

    alias type = __mlir_type[
        `!lit.lifetime<`,
        is_mutable.value,
        `>`,
    ]
