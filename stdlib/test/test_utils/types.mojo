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


struct MoveOnly[T: Movable](Movable):
    """Utility for testing MoveOnly types.

    Parameters:
        T: Can be any type satisfying the Movable trait.
    """

    var data: T
    """Test data payload."""

    fn __init__(inout self, owned i: T):
        """Construct a MoveOnly providing the payload data.

        Args:
            i: The test data payload.
        """
        self.data = i^

    fn __moveinit__(inout self, owned other: Self):
        """Move construct a MoveOnly from an existing variable.

        Args:
            other: The other instance that we copying the payload from.
        """
        self.data = other.data^


struct ExplicitCopyOnly(ExplicitlyCopyable):
    var value: Int
    var copy_count: Int

    fn __init__(inout self, value: Int):
        self.value = value
        self.copy_count = 0

    fn __init__(inout self, *, other: Self):
        self.value = other.value
        self.copy_count = other.copy_count + 1


struct CopyCounter(CollectionElement):
    """Counts the number of copies performed on a value."""

    var copy_count: Int

    fn __init__(inout self):
        self.copy_count = 0

    fn __moveinit__(inout self, owned existing: Self):
        self.copy_count = existing.copy_count

    fn __copyinit__(inout self, existing: Self):
        self.copy_count = existing.copy_count + 1


struct MoveCounter[T: CollectionElementNew](
    CollectionElement,
    CollectionElementNew,
):
    """Counts the number of moves performed on a value."""

    var value: T
    var move_count: Int

    fn __init__(inout self, owned value: T):
        """Construct a new instance of this type. This initial move is not counted.
        """
        self.value = value^
        self.move_count = 0

    # TODO: This type should not be ExplicitlyCopyable, but has to be to satisfy
    #       CollectionElementNew at the moment.
    fn __init__(inout self, *, other: Self):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """
        self.value = T(other=other.value)
        self.move_count = other.move_count

    fn __moveinit__(inout self, owned existing: Self):
        self.value = existing.value^
        self.move_count = existing.move_count + 1

    # TODO: This type should not be Copyable, but has to be to satisfy
    #       CollectionElement at the moment.
    fn __copyinit__(inout self, existing: Self):
        # print("ERROR: _MoveCounter copy constructor called unexpectedly!")
        self.value = T(other=existing.value)
        self.move_count = existing.move_count
