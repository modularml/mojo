# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct MoveOnly[T: Movable](Movable):
    """Utility for testing MoveOnly types.

    Parameters:
        T: Can be any type satisfying the Movable trait.
    """

    var data: T
    """Test data payload."""

    fn __init__(inout self, i: T):
        """Construct a MoveOnly providing the payload data.

        Args:
            i: The test data payload.
        """
        self.data = i ^

    fn __moveinit__(inout self, owned other: Self):
        """Move construct a MoveOnly from an existing variable.

        Args:
            other: The other instance that we copying the payload from.
        """
        self.data = other.data ^
        other.data = 0
