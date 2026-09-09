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
"""Defines the Identifiable trait for identity-based comparisons.

This module provides the `Identifiable` trait, which enables types to support
identity comparison using the `is` and `is not` operators. Identity comparison
checks if two values refer to the same object, distinct from equality
comparison which checks if two values have the same content.
"""


trait Identifiable:
    """The Identifiable trait denotes a type with an identity
    which can be compared with other instances of itself.
    """

    def __is__(self, rhs: Self) -> Bool:
        """Define whether `self` has the same identity as `rhs`.

        Args:
            rhs: The right hand side of the comparison.

        Returns:
            True if `self` is `rhs`.
        """
        ...

    @always_inline
    def __isnot__(self, rhs: Self) -> Bool:
        """Define whether `self` has a different identity than `rhs`.

        Args:
            rhs: The right hand side of the comparison.

        Returns:
            True if `self` is not `rhs`.
        """
        return not (self is rhs)
