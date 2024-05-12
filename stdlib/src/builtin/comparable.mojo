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


trait Comparable(EqualityComparable):
    """A type which can be compared with other instances of itself."""

    fn __lt__(self, rhs: Self) -> Bool:
        """Define whether `self` is less than `rhs`.

        Args:
            rhs: The right hand side of the comparison.

        Returns:
            True if `self` is less than `rhs`.
        """
        ...

    fn __le__(self, rhs: Self) -> Bool:
        """Define whether `self` is less than or equal to `rhs`.

        Args:
            rhs: The right hand side of the comparison.

        Returns:
            True if `self` is less than or equal to `rhs`.
        """
        ...

    fn __gt__(self, rhs: Self) -> Bool:
        """Define whether `self` is greater than `rhs`.

        Args:
            rhs: The right hand side of the comparison.

        Returns:
            True if `self` is greater than `rhs`.
        """
        ...

    fn __ge__(self, rhs: Self) -> Bool:
        """Define whether `self` is greater than or equal to `rhs`.

        Args:
            rhs: The right hand side of the comparison.

        Returns:
            True if `self` is greater than or equal to `rhs`.
        """
        ...
