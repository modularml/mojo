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


trait EqualityComparable:
    """A type which can be compared for equality with other instances of itself.
    """

    fn __eq__(self, other: Self) -> Bool:
        """Define whether two instances of the object are equal to each other.

        Args:
            other: Another instance of the same type.

        Returns:
            True if the instances are equal according to the type's definition
            of equality, False otherwise.
        """
        pass

    fn __ne__(self, other: Self) -> Bool:
        """Define whether two instances of the object are not equal to each other.

        Args:
            other: Another instance of the same type.

        Returns:
            True if the instances are not equal according to the type's definition
            of equality, False otherwise.
        """
        pass
