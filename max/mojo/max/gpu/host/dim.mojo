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
"""This module implements the dim type."""

from std.utils.index import IndexList


@fieldwise_init("implicit")
struct Dim(TrivialRegisterPassable, Writable):
    """Represents a dimension with up to three components (x, y, z).

    This struct is commonly used to represent grid and block dimensions
    for kernel launches.
    """

    var _value: IndexList[3]
    """Internal storage for the three dimension components (x, y, z).

    This field stores the values for all three dimensions using an IndexList
    with a fixed size of 3. The dimensions are accessed in order: x, y, z.
    """

    @implicit
    def __init__[I: Indexer, //](out self, x: I):
        """Initializes Dim with a single indexable value for x.

        y and z dimensions are set to 1.

        Parameters:
            I: The type of the indexable value.

        Args:
            x: The value for the x dimension.
        """
        self._value = IndexList[3](index(x), 1, 1)

    def __init__[I0: Indexer, I1: Indexer, //](out self, x: I0, y: I1):
        """Initializes Dim with indexable values for x and y.

        z dimension is set to 1.

        Parameters:
            I0: The type of the first indexable value.
            I1: The type of the second indexable value.

        Args:
            x: The value for the x dimension.
            y: The value for the y dimension.
        """
        self._value = IndexList[3](index(x), index(y), 1)

    def __init__[
        I0: Indexer, I1: Indexer, I2: Indexer, //
    ](out self, x: I0, y: I1, z: I2):
        """Initializes Dim with indexable values for x, y, and z.

        Parameters:
            I0: The type of the first indexable value.
            I1: The type of the second indexable value.
            I2: The type of the third indexable value.

        Args:
            x: The value for the x dimension.
            y: The value for the y dimension.
            z: The value for the z dimension.
        """
        self._value = IndexList[3](index(x), index(y), index(z))

    @implicit
    def __init__[I: Indexer & Copyable, //](out self, dims: Tuple[I]):
        """Initializes Dim with a tuple containing a single indexable value.

        y and z dimensions are set to 1.

        Parameters:
            I: The type of the indexable value in the tuple.

        Args:
            dims: A tuple with one element for x dimension.
        """
        self._value = IndexList[3](index(dims[0]), 1, 1)

    @implicit
    def __init__[
        I0: Indexer & Copyable,
        I1: Indexer & Copyable,
        //,
    ](out self, dims: Tuple[I0, I1]):
        """Initializes Dim with a tuple of two indexable values.

        The z dimension is set to 1.

        Parameters:
            I0: The type of the first indexable value in the tuple.
            I1: The type of the second indexable value in the tuple.

        Args:
            dims: A tuple with two elements: x and y dimensions.
        """
        self._value = IndexList[3](index(dims[0]), index(dims[1]), 1)

    @implicit
    def __init__[
        I0: Indexer & Copyable,
        I1: Indexer & Copyable,
        I2: Indexer & Copyable,
        //,
    ](out self, dims: Tuple[I0, I1, I2]):
        """Initializes Dim with a tuple of three indexable values.

        Parameters:
            I0: The type of the first indexable value in the tuple.
            I1: The type of the second indexable value in the tuple.
            I2: The type of the third indexable value in the tuple.

        Args:
            dims: Tuple with three elements: x, y, and z dimensions.
        """
        self._value = IndexList[3](
            index(dims[0]), index(dims[1]), index(dims[2])
        )

    def __getitem__(self, idx: Int) -> Int:
        """Gets the dimension value at the specified index.

        Args:
            idx: The index (0 for x, 1 for y, 2 for z).

        Returns:
            The value of the dimension at the given index.
        """
        return self._value[idx]

    def write_to(self, mut writer: Some[Writer]):
        """Writes a formatted string representation of the Dim.

        Args:
            writer: The Writer to write to.
        """
        writer.write("(x=", self.x(), ", ")
        if self.y() != 1 or self.z() != 1:
            writer.write("y=", self.y())
            if self.z() != 1:
                writer.write(", z=", self.z())
        writer.write(")")

    def z(self) -> Int:
        """Returns the z dimension.

        Returns:
            The value of the z dimension.
        """
        return self[2]

    def y(self) -> Int:
        """Returns the y dimension.

        Returns:
            The value of the y dimension.
        """
        return self[1]

    def x(self) -> Int:
        """Returns the x dimension.

        Returns:
            The value of the x dimension.
        """
        return self[0]
