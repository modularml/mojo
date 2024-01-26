# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the StringRef class.

These are Mojo built-ins, so you don't need to import them.
"""

from memory.unsafe import DTypePointer, Pointer

# ===----------------------------------------------------------------------===#
# StringRef
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct StringRef(Sized, CollectionElement, Stringable, Hashable):
    """
    Represent a constant reference to a string, i.e. a sequence of characters
    and a length, which need not be null terminated.
    """

    var data: DTypePointer[DType.int8]
    """A pointer to the beginning of the string data being referenced."""
    var length: Int
    """The length of the string being referenced."""

    @always_inline
    fn __init__(str: StringLiteral) -> StringRef:
        """Construct a StringRef value given a constant string.

        Args:
            str: The input constant string.

        Returns:
            Constructed `StringRef` object.
        """
        return StringRef(str.data(), len(str))

    fn __str__(self) -> String:
        """Convert the string reference to a string.

        Returns:
            A new string.
        """
        return self

    @always_inline
    fn __init__(ptr: Pointer[Int8], len: Int) -> StringRef:
        """Construct a StringRef value given a (potentially non-0 terminated
        string).

        The constructor takes a raw pointer and a length.

        Args:
            ptr: Pointer to the string.
            len: The length of the string.

        Returns:
            Constructed `StringRef` object.
        """

        return Self {data: ptr, length: len}

    @always_inline
    fn __init__(ptr: DTypePointer[DType.int8], len: Int) -> StringRef:
        """Construct a StringRef value given a (potentially non-0 terminated
        string).

        The constructor takes a raw pointer and a length.

        Args:
            ptr: Pointer to the string.
            len: The length of the string.

        Returns:
            Constructed `StringRef` object.
        """

        return Self {data: ptr, length: len}

    @always_inline
    fn __init__(ptr: Pointer[Int8]) -> StringRef:
        """Construct a StringRef value given a null-terminated string.

        Args:
            ptr: Pointer to the string.

        Returns:
            Constructed `StringRef` object.
        """

        return DTypePointer[DType.int8](ptr.address)

    @always_inline
    fn __init__(ptr: DTypePointer[DType.int8]) -> StringRef:
        """Construct a StringRef value given a null-terminated string.

        Args:
            ptr: Pointer to the string.

        Returns:
            Constructed `StringRef` object.
        """

        var len = 0
        while ptr.load(len):
            len += 1

        return StringRef(ptr, len)

    @always_inline
    fn _as_ptr(self) -> DTypePointer[DType.int8]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The pointer to the underlying memory.
        """
        return self.data

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the string is empty or not.

        Returns:
          Returns True if the string is not empty and False otherwise.
        """
        return len(self) != 0

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the string.

        Returns:
          The length of the string.
        """
        return self.length

    @always_inline("nodebug")
    fn __eq__(self, rhs: StringRef) -> Bool:
        """Compares two strings are equal.

        Args:
          rhs: The other string.

        Returns:
          True if the strings match and False otherwise.
        """
        if len(self) != len(rhs):
            return False
        for i in range(len(self)):
            if self.data.load(i) != rhs.data.load(i):
                return False
        return True

    @always_inline("nodebug")
    fn __ne__(self, rhs: StringRef) -> Bool:
        """Compares two strings are not equal.

        Args:
          rhs: The other string.

        Returns:
          True if the strings do not match and False otherwise.
        """
        return not (self == rhs)

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> StringRef:
        """Get the string value at the specified position.

        Args:
          idx: The index position.

        Returns:
          The character at the specified position.
        """
        return StringRef {data: self.data + idx, length: 1}

    fn __hash__(self) -> Int:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self.data, self.length)
