# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements various testing utils.

You can import these types from the `testing` package. For example:

```mojo
from testing import assert_true
```
"""


# ===----------------------------------------------------------------------=== #
# MoveOnlyInt
# ===----------------------------------------------------------------------=== #
struct MoveOnlyInt(Movable):
    """Utility for testing MoveOnly types."""

    var data: Int
    """Test data payload."""

    fn __init__(inout self, i: Int):
        """Construct a MoveOnly providing the payload data.

        Args:
             i: The test data payload.
        """
        self.data = i

    fn __moveinit__(inout self, owned other: Self):
        """Move construct a MoveOnly from an existing variable.

        Args:
             other: The other instance that we copying the payload from.
        """
        self.data = other.data
        other.data = 0
