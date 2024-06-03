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

alias NoneType = __mlir_type.`!kgen.none`
"""Represents the absence of a value."""

alias ImmutableLifetime = __mlir_type.`!lit.lifetime<0>`
"""Immutable lifetime reference type."""

alias MutableLifetime = __mlir_type.`!lit.lifetime<1>`
"""Mutable lifetime reference type."""

alias ImmutableStaticLifetime = __mlir_attr.`#lit.lifetime<0>: !lit.lifetime<0>`
"""The immutable lifetime that lasts for the entire duration of program execution."""

alias MutableStaticLifetime = __mlir_attr.`#lit.lifetime<1>: !lit.lifetime<1>`
"""The mutable lifetime that lasts for the entire duration of program execution."""

alias LifetimeSet = __mlir_type.`!lit.lifetime.set`
"""A set of lifetime parameters."""


# Helper to build !lit.lifetime type.
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
