# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
