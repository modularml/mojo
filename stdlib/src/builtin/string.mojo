# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements basic object methods for working with strings.

These are Mojo built-ins, so you don't need to import them.
"""

from collections.dict import KeyElement
from collections.vector import DynamicVector
from math import abs as _abs
from math import max as _max
from math import min as _min
from math.bit import ctlz
from sys.info import bitwidthof

from memory.anypointer import AnyPointer
from memory.buffer import Buffer
from memory.memory import memcmp, memcpy
from memory.unsafe import DTypePointer, Pointer

from utils.index import StaticIntTuple
from utils.static_tuple import StaticTuple

from .io import _snprintf, _snprintf_scalar, _StringableTuple
from .range import _StridedRange

# ===----------------------------------------------------------------------===#
# ord
# ===----------------------------------------------------------------------===#


fn ord(s: String) -> Int:
    """Returns an integer that represents the given one-character string.

    Given a string representing one ASCII character, return an integer
    representing the code point of that character. For example, `ord("a")`
    returns the integer `97`. This is the inverse of the `chr()` function.

    Currently, extended ASCII characters are not supported in this function.

    Args:
        s: The input string, which must contain only a single character.

    Returns:
        An integer representing the code point of the given character.
    """
    debug_assert(len(s) == 1, "input string length must be 1")
    return int(s._buffer[0])


# ===----------------------------------------------------------------------===#
# chr
# ===----------------------------------------------------------------------===#


fn chr(c: Int) -> String:
    """Returns a string based on the given Unicode code point.

    Returns the string representing a character whose code point (which must be a
    positive integer between 0 and 255) is the integer `i`. For example,
    `chr(97)` returns the string `"a"`. This is the inverse of the `ord()`
    function.

    Args:
        c: An integer between 0 and 255 that represents a code point.

    Returns:
        A string containing a single character based on the given code point.
    """
    debug_assert(0 <= c <= 255, "input ordinal must be in range")
    var buf = Pointer[Int8].alloc(2)
    buf[0] = c
    buf[1] = 0
    return String(buf, 2)


# ===----------------------------------------------------------------------===#
# strtol
# ===----------------------------------------------------------------------===#


# TODO: this is hard coded for decimal base
fn atol(str: String) raises -> Int:
    """Parses the given string as a base-10 integer and returns that value.

    For example, `atol("19")` returns `19`. If the given string cannot be parsed
    as an integer value, an error is raised. For example, `atol("hi")` raises an
    error.

    Args:
        str: A string to be parsed as a base-10 integer.

    Returns:
        An integer value that represents the string, or otherwise raises.
    """
    if not str:
        raise Error("Empty String cannot be converted to integer.")
    var result = 0
    var is_negative: Bool
    var start: Int
    if str[0] == "-":
        is_negative = True
        start = 1
    else:
        is_negative = False
        start = 0

    alias ord_0 = ord("0")
    alias ord_9 = ord("9")
    for pos in range(start, len(str)):
        var digit = int(str._buffer[pos])
        if ord_0 <= digit <= ord_9:
            result += digit - ord_0
        else:
            raise Error("String is not convertible to integer.")
        if pos + 1 < len(str):
            var nextresult = result * 10
            if nextresult < result:
                raise Error(
                    "String expresses an integer too large to store in Int."
                )
            result = nextresult
    if is_negative:
        result = -result
    return result


# ===----------------------------------------------------------------------===#
# isdigit
# ===----------------------------------------------------------------------===#


fn isdigit(c: Int8) -> Bool:
    """Determines whether the given character is a digit [0-9].

    Args:
        c: The character to check.

    Returns:
        True if the character is a digit.
    """
    alias ord_0 = ord("0")
    alias ord_9 = ord("9")
    return ord_0 <= int(c) <= ord_9


# ===----------------------------------------------------------------------===#
# isupper
# ===----------------------------------------------------------------------===#


fn isupper(c: Int8) -> Bool:
    """Determines whether the given character is an uppercase character.
       This currently only respects the default "C" locale, i.e. returns
       True only if the character specified is one of ABCDEFGHIJKLMNOPQRSTUVWXYZ.

    Args:
        c: The character to check.

    Returns:
        True if the character is uppercase.
    """
    return _is_ascii_uppercase(c)


fn _is_ascii_uppercase(c: Int8) -> Bool:
    alias ord_a = ord("A")
    alias ord_z = ord("Z")
    return ord_a <= int(c) <= ord_z


# ===----------------------------------------------------------------------===#
# islower
# ===----------------------------------------------------------------------===#


fn islower(c: Int8) -> Bool:
    """Determines whether the given character is an lowercase character.
       This currently only respects the default "C" locale, i.e. returns
       True only if the character specified is one of abcdefghijklmnopqrstuvwxyz.

    Args:
        c: The character to check.

    Returns:
        True if the character is lowercase.
    """
    return _is_ascii_lowercase(c)


fn _is_ascii_lowercase(c: Int8) -> Bool:
    alias ord_a = ord("a")
    alias ord_z = ord("z")
    return ord_a <= int(c) <= ord_z


# ===----------------------------------------------------------------------===#
# isspace
# ===----------------------------------------------------------------------===#


fn isspace(c: Int8) -> Bool:
    """Determines whether the given character is a whitespace character.
       This currently only respects the default "C" locale, i.e. returns
       True only if the character specified is one of
       " \n\t\r\f\v".

    Args:
        c: The character to check.

    Returns:
        True if the character is one of the whitespace characters listed above, otherwise False.
    """

    alias ord_space = ord(" ")
    alias ord_tab = ord("\t")
    alias ord_carriage_return = ord("\r")

    return c == ord_space or ord_tab <= int(c) <= ord_carriage_return


# ===----------------------------------------------------------------------===#
# String
# ===----------------------------------------------------------------------===#
struct String(Sized, Stringable, KeyElement, Boolable):
    """Represents a mutable string."""

    alias _buffer_type = DynamicVector[Int8]
    var _buffer: Self._buffer_type
    """The underlying storage for the string."""

    fn __str__(self) -> String:
        return self

    @always_inline
    fn __init__(inout self, owned impl: Self._buffer_type):
        """Construct a string from a buffer of bytes.

        The buffer must be terminated with a null byte:

        ```mojo
        var buf = DynamicVector[Int8]()
        buf.append(ord('H'))
        buf.append(ord('i'))
        buf.append(0)
        var hi = String(buf)
        ```

        Args:
            impl: The buffer.
        """
        self._buffer = impl ^

    @always_inline
    fn __init__(inout self):
        """Construct an uninitialized string."""
        self._buffer = Self._buffer_type()

    @always_inline
    fn __init__(inout self, str: StringRef):
        """Construct a string from a StringRef object.

        Args:
            str: The StringRef from which to construct this string object.
        """
        var length = len(str)
        var buffer = Self._buffer_type()
        buffer.resize(length + 1, 0)
        memcpy(rebind[DTypePointer[DType.int8]](buffer.data), str.data, length)
        buffer[length] = 0
        self._buffer = buffer ^

    @always_inline
    fn __init__(inout self, str: StringLiteral):
        """Constructs a String value given a constant string.

        Args:
            str: The input constant string.
        """

        self = String(StringRef(str))

    fn __init__[stringable: Stringable](inout self, value: stringable):
        """Creates a string from a value that conforms to Stringable trait.

        Parameters:
            stringable: The Stringable type.

        Args:
            value: The value that conforms to Stringable.
        """

        self = str(value)

    @always_inline
    fn __init__(inout self, ptr: Pointer[Int8], len: Int):
        """Creates a string from the buffer. Note that the string now owns
        the buffer.

        The buffer must be terminated with a null byte.

        Args:
            ptr: The pointer to the buffer.
            len: The length of the buffer, including the null terminator.
        """
        self._buffer = Self._buffer_type()
        self._buffer.data = rebind[AnyPointer[Int8]](ptr)
        self._buffer.size = len

    @always_inline
    fn __init__(inout self, ptr: DTypePointer[DType.int8], len: Int):
        """Creates a string from the buffer. Note that the string now owns
        the buffer.

        The buffer must be terminated with a null byte.

        Args:
            ptr: The pointer to the buffer.
            len: The length of the buffer, including the null terminator.
        """
        self = String(ptr.address, len)

    @always_inline
    fn __copyinit__(inout self, existing: Self):
        """Creates a deep copy of an existing string.

        Args:
            existing: The string to copy.
        """
        self._buffer = existing._buffer

    @always_inline
    fn __moveinit__(inout self, owned existing: String):
        """Move the value of a string.

        Args:
            existing: The string to move.
        """
        self._buffer = existing._buffer ^

    @staticmethod
    fn _unchecked_from_bytes(owned impl: Self._buffer_type) -> String:
        """Construct a string from a sequence of bytes.

        This does no validation that the given bytes are valid in any specific
        String encoding.

        Args:
            impl: The buffer. This should not have an existing NUL terminator.
        """

        # Add the NUL terminator.
        impl.push_back(0)

        return String(impl ^)

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the string is not empty.

        Returns:
            True if the string length is greater than zero, and False otherwise.
        """
        return len(self) > 0

    fn __getitem__(self, idx: Int) -> String:
        """Gets the character at the specified position.

        Args:
            idx: The index value.

        Returns:
            A new string containing the character at the specified position.
        """
        if idx < 0:
            return self.__getitem__(len(self) + idx)

        debug_assert(0 <= idx < len(self), "index must be in range")
        var buf = Self._buffer_type(capacity=1)
        buf.append(self._buffer[idx])
        buf.append(0)
        return String(buf ^)

    @always_inline
    fn _adjust_span(self, span: Slice) -> Slice:
        """Adjusts the span based on the string length."""
        var adjusted_span = span

        if adjusted_span.start < 0:
            adjusted_span.start = len(self) + adjusted_span.start

        if not adjusted_span._has_end():
            adjusted_span.end = len(self)
        elif adjusted_span.end < 0:
            adjusted_span.end = len(self) + adjusted_span.end

        return adjusted_span

    @always_inline
    fn __getitem__(self, span: Slice) -> String:
        """Gets the sequence of characters at the specified positions.

        Args:
            span: A slice that specifies positions of the new substring.

        Returns:
            A new string containing the string at the specified positions.
        """

        var adjusted_span = self._adjust_span(span)
        if adjusted_span.step == 1:
            return StringRef(
                (self._buffer.data + span.start).value,
                len(adjusted_span),
            )

        var buffer = Self._buffer_type()
        var adjusted_span_len = len(adjusted_span)
        buffer.resize(adjusted_span_len + 1, 0)
        for i in range(adjusted_span_len):
            buffer[i] = self._as_ptr().offset(adjusted_span[i]).load()
        buffer[adjusted_span_len] = 0
        return Self(buffer ^)

    @always_inline
    fn __len__(self) -> Int:
        """Returns the string length.

        Returns:
            The string length.
        """
        # Avoid returning -1 if the buffer is not initialized
        if not self._as_ptr():
            return 0

        # The negative 1 is to account for the terminator.
        return len(self._buffer) - 1

    @always_inline
    fn __eq__(self, other: String) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        if len(self) != len(other):
            return False

        if int(self._as_ptr()) == int(other._as_ptr()):
            return True

        return memcmp(self._as_ptr(), other._as_ptr(), len(self)) == 0

    @always_inline
    fn __ne__(self, other: String) -> Bool:
        """Compares two Strings if they do not have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are not equal and False otherwise.
        """
        return not (self == other)

    # "str1"+"str2" -> "str1str2"
    @always_inline
    fn __add__(self, other: String) -> String:
        """Creates a string by appending another string at the end.

        Args:
            other: The string to append.

        Returns:
            The new constructed string.
        """
        if not self:
            return other
        if not other:
            return self
        var self_len = len(self)
        var other_len = len(other)
        var total_len = self_len + other_len
        var buffer = Self._buffer_type()
        buffer.resize(total_len + 1, 0)
        memcpy(
            rebind[Pointer[Int8]](buffer.data),
            rebind[Pointer[Int8]](self._buffer.data),
            self_len,
        )
        memcpy(
            rebind[Pointer[Int8]](buffer.data + self_len),
            rebind[Pointer[Int8]](other._buffer.data),
            other_len,
        )
        buffer[total_len] = 0
        return Self(buffer ^)

    @always_inline
    fn __radd__(self, other: String) -> String:
        """Creates a string by prepending another string to the start.

        Args:
            other: The string to prepend.

        Returns:
            The new constructed string.
        """
        return other + self

    @always_inline
    fn __iadd__(inout self, other: String):
        """Appends another string to this string.

        Args:
            other: The string to append.
        """
        self = self + other

    fn join[rank: Int](self, elems: StaticIntTuple[rank]) -> String:
        """Joins the elements from the tuple using the current string as a
        delimiter.

        Parameters:
            rank: The size of the tuple.

        Args:
            elems: The input tuple.

        Returns:
            The joined string.
        """
        if len(elems) == 0:
            return ""
        var curr = String(elems[0])
        for i in range(1, len(elems)):
            curr += self + String(elems[i])
        return curr

    fn join[*Stringables: Stringable](self, *elems: *Stringables) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            Stringables: The Stringable types.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """
        alias types = VariadicList(Stringables)
        alias count = len(types)

        var args = _StringableTuple(elems)

        if count == 0:
            return ""

        var result = args._at[0]()

        @parameter
        fn each[i: Int]():
            result += self + args._at[i + 1]()

        unroll[each, count - 1]()
        return result

    fn _strref_dangerous(self) -> StringRef:
        """
        Returns an inner pointer to the string as a StringRef.
        This functionality is extremely dangerous because Mojo eagerly releases
        strings.  Using this requires the use of the _strref_keepalive() method
        to keep the underlying string alive long enough.
        """
        return StringRef {data: self._as_ptr(), length: len(self)}

    fn _strref_keepalive(self):
        """
        A noop that keeps `self` alive through the call.  This
        can be carefully used with `_strref_dangerous()` to wield inner pointers
        without the string getting deallocated early.
        """
        pass

    fn _as_ptr(self) -> DTypePointer[DType.int8]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The pointer to the underlying memory.
        """
        return rebind[DTypePointer[DType.int8]](self._buffer.data)

    fn as_bytes(self) -> DynamicVector[Int8]:
        """Retrieves the underlying byte sequence encoding the characters in
        this string.

        This does not include the trailing null terminator.

        Returns:
            A sequence containing the encoded characters stored in this string.
        """

        # TODO(lifetimes): Return a reference rather than a copy
        var copy = self._buffer
        var last = copy.pop_back()
        debug_assert(
            last == 0,
            "expected last element of String buffer to be null terminator",
        )

        return copy

    fn _strref_from_start(self, start: Int) -> StringRef:
        """Gets the StringRef pointing to the substring after the specified slice start position.

        If start is negative, it is interpreted as the number of characters
        from the end of the string to start at.

        Warning: This method is as dangerous as `String._strref_dangerous()`.

        Args:
            start: Starting index of the slice.

        Returns:
            A StringRef borrowed from the current string containing the
            characters of the slice starting at start.
        """

        var self_len = len(self)

        var abs_start: Int
        if start < 0:
            # Avoid out of bounds earlier than the start
            # len = 5, start = -3,  then abs_start == 2, i.e. a partial string
            # len = 5, start = -10, then abs_start == 0, i.e. the full string
            abs_start = _max(self_len + start, 0)
        else:
            # Avoid out of bounds past the end
            # len = 5, start = 2,   then abs_start == 2, i.e. a partial string
            # len = 5, start = 8,   then abs_start == 5, i.e. an empty string
            abs_start = _min(start, self_len)

        debug_assert(
            abs_start >= 0, "strref absolute start must be non-negative"
        )
        debug_assert(
            abs_start <= self_len,
            "strref absolute start must be less than source String len",
        )

        var data = self._as_ptr() + abs_start
        var length = self_len - abs_start

        return StringRef(data, length)

    fn _steal_ptr(inout self) -> DTypePointer[DType.int8]:
        """Transfer ownership of pointer to the underlying memory.
        The caller is responsible for freeing up the memory.

        Returns:
            The pointer to the underlying memory.
        """
        var ptr = self._as_ptr()
        self._buffer.data = AnyPointer[Int8]()
        self._buffer.size = 0
        self._buffer.capacity = 0
        return ptr

    fn count(self, substr: String) -> Int:
        """Return the number of non-overlapping occurrences of substring
        `substr` in the string.

        If sub is empty, returns the number of empty strings between characters
        which is the length of the string plus one.

        Args:
          substr: The substring to count.

        Returns:
          The number of occurrences of `substr`.
        """
        if not substr:
            return len(self) + 1

        var res = 0
        var offset = 0

        while True:
            var pos = self.find(substr, offset)
            if pos == -1:
                break
            res += 1

            offset = pos + len(substr)

        return res

    fn __contains__(self, substr: String) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return self.find(substr) != -1

    fn find(self, substr: String, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """
        if not substr:
            return 0

        if len(self) < len(substr) + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var haystack_str = self._strref_from_start(start)

        var loc = _memmem(
            haystack_str._as_ptr(),
            len(haystack_str),
            substr._as_ptr(),
            len(substr),
        )

        if not loc:
            return -1

        return int(loc) - int(self._as_ptr())

    fn rfind(self, substr: String, start: Int = 0) -> Int:
        """Finds the offset of the last occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """
        if not substr:
            return len(self)

        if len(self) < len(substr) + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var haystack_str = self._strref_from_start(start)

        var loc = _memrmem(
            haystack_str._as_ptr(),
            haystack_str.length,
            substr._as_ptr(),
            len(substr),
        )

        if not loc:
            return -1

        return int(loc) - int(self._as_ptr())

    fn split(self, delimiter: String) raises -> DynamicVector[String]:
        """Split the string by a delimiter.

        Args:
          delimiter: The string to split on.

        Returns:
          A DynamicVector of Strings containing the input split by the delimiter.

        Raises:
          Error if an empty delimiter is specified.
        """
        if not delimiter:
            raise Error("empty delimiter not allowed to be passed to split.")

        var output = DynamicVector[String]()

        var current_offset = 0
        while True:
            var loc = self.find(delimiter, current_offset)
            # delimiter not found, so add the search slice from where we're currently at
            if loc == -1:
                output.push_back(self[current_offset:])
                break

            # We found a delimiter, so add the preceding string slice
            output.push_back(self[current_offset:loc])

            # Advance our search offset past the delimiter
            current_offset = loc + len(delimiter)

        return output

    fn replace(self, old: String, new: String) -> String:
        """Return a copy of the string with all occurrences of substring `old`
        if replaced by `new`.

        Args:
          old: The substring to replace.
          new: The substring to replace with.

        Returns:
          The string where all occurences of `old` are replaced with `new`.
        """
        if not old:
            return self._interleave(new)

        var occurrences = self.count(old)
        if occurrences == -1:
            return self

        var self_start = self._as_ptr()
        var self_ptr = self._as_ptr()
        var new_ptr = new._as_ptr()

        var self_len = len(self)
        var old_len = len(old)
        var new_len = len(new)

        var res = DynamicVector[Int8]()
        res.reserve(self_len + (old_len - new_len) * occurrences + 1)

        for _ in range(occurrences):
            var curr_offset = int(self_ptr) - int(self_start)

            var idx = self.find(old, curr_offset)

            debug_assert(idx >= 0, "expected to find occurrence during find")

            # Copy preceding unchanged chars
            for _ in range(curr_offset, idx):
                res.push_back(self_ptr.load())
                self_ptr += 1

            # Insert a copy of the new replacement string
            for i in range(new_len):
                res.push_back(new_ptr.load(i))

            self_ptr += old_len

        while True:
            var val = self_ptr.load()
            if val == 0:
                break
            res.push_back(self_ptr.load())
            self_ptr += 1

        res.push_back(0)
        return String(res ^)

    fn strip(self) -> String:
        """Return a copy of the string with leading and trailing whitespace characters removed.

        See `isspace` for a list of whitespace characters

        Returns:
          A copy of the string with no leading or trailing whitespace characters.
        """

        return self.lstrip().rstrip()

    fn rstrip(self) -> String:
        """Return a copy of the string with trailing whitespace characters removed.

        See `isspace` for a list of whitespace characters

        Returns:
          A copy of the string with no trailing whitespace characters.
        """

        var r_idx = len(self)
        while r_idx > 0 and isspace(ord(self[r_idx - 1])):
            r_idx -= 1

        return self[:r_idx]

    fn lstrip(self) -> String:
        """Return a copy of the string with leading whitespace characters removed.

        See `isspace` for a list of whitespace characters

        Returns:
          A copy of the string with no leading whitespace characters.
        """

        var l_idx = 0
        while l_idx < len(self) and isspace(ord(self[l_idx])):
            l_idx += 1

        return self[l_idx:]

    fn __hash__(self) -> Int:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        var data = DTypePointer[DType.int8](self._buffer.data.value)
        return hash(data, len(self))

    fn _interleave(self, val: String) -> String:
        var res = DynamicVector[Int8]()
        var val_ptr = val._as_ptr()
        var self_ptr = self._as_ptr()
        res.reserve(len(val) * len(self) + 1)
        for i in range(len(self)):
            for j in range(len(val)):
                res.push_back(val_ptr.load(j))
            res.push_back(self_ptr.load(i))
        res.push_back(0)
        return String(res ^)

    fn tolower(self) -> String:
        """Returns a copy of the string with all ASCII cased characters converted to lowercase.

        Returns:
            A new string where cased letters have been convered to lowercase.
        """

        # TODO(#26444):
        # Support the Unicode standard casing behavior to handle cased letters
        # outside of the standard ASCII letters.
        return self._toggle_ascii_case[_is_ascii_uppercase]()

    fn toupper(self) -> String:
        """Returns a copy of the string with all ASCII cased characters converted to uppercase.

        Returns:
            A new string where cased letters have been converted to uppercase.
        """

        # TODO(#26444):
        # Support the Unicode standard casing behavior to handle cased letters
        # outside of the standard ASCII letters.
        return self._toggle_ascii_case[_is_ascii_lowercase]()

    fn _toggle_ascii_case[check_case: fn (Int8) -> Bool](self) -> String:
        var copy: String = self

        var char_ptr = copy._as_ptr()

        for i in range(len(self)):
            var char: Int8 = char_ptr[i]
            if check_case(char):
                var lower = _toggle_ascii_case(char)
                char_ptr[i] = lower

        return copy

    fn startswith(self, prefix: String, start: Int = 0, end: Int = -1) -> Bool:
        """Checks if the string starts with the specified prefix between start
        and end positions. Returns True if found and False otherwise.

        Args:
          prefix: The prefix to check.
          start: The start offset from which to check.
          end: The end offset from which to check.

        Returns:
          True if the self[start:end] is prefixed by the input prefix.
        """
        if end == -1:
            return self.find(prefix, start) == start
        return self[start:end].startswith(prefix)

    fn endswith(self, suffix: String, start: Int = 0, end: Int = -1) -> Bool:
        """Checks if the string end with the specified suffix between start
        and end positions. Returns True if found and False otherwise.

        Args:
          suffix: The suffix to check.
          start: The start offset from which to check.
          end: The end offset from which to check.

        Returns:
          True if the self[start:end] is suffixed by the input suffix.
        """
        if end == -1:
            return self._endswith_impl(suffix, start)
        return self[start:end]._endswith_impl(suffix)

    fn _endswith_impl(self, suffix: String, start: Int = 0) -> Bool:
        return self.rfind(suffix, start) + len(suffix) == len(self)


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _vec_fmt[
    *types: AnyRegType
](
    str: AnyPointer[Int8], size: Int, fmt: StringLiteral, *arguments: *types
) -> Int:
    return _snprintf(rebind[Pointer[Int8]](str), size, fmt, arguments)


fn _toggle_ascii_case(char: Int8) -> Int8:
    """Assuming char is a cased ASCII character, this function will return the opposite-cased letter
    """

    # ASCII defines A-Z and a-z as differing only in their 6th bit,
    # so converting is as easy as a bit flip.
    return char ^ (1 << 5)


fn _memmem[
    type: DType, range_fn: fn (Int, Int) -> _StridedRange
](
    haystack: DTypePointer[type],
    haystack_len: Int,
    needle: DTypePointer[type],
    needle_len: Int,
) -> DTypePointer[type]:
    if not needle_len:
        return haystack
    if needle_len > haystack_len:
        return DTypePointer[type]()
    if needle_len == 1:
        return _memchr[type, range_fn](haystack, needle[0], haystack_len)
    for i in range_fn(haystack_len, needle_len):
        if haystack[i] != needle[0]:
            continue

        if memcmp(haystack + i, needle, needle_len) == 0:
            return haystack + i
    return DTypePointer[type]()


fn _memchr[
    type: DType, range_fn: fn (Int, Int) -> _StridedRange
](
    source: DTypePointer[type],
    char: Scalar[type],
    len: Int,
) -> DTypePointer[
    type
]:
    if not len:
        return DTypePointer[type]()
    for i in range_fn(len, 1):
        if source[i] == char:
            return source + i
    return DTypePointer[type]()


@always_inline
fn _forward_range(haystack_len: Int, needle_len: Int) -> _StridedRange:
    return range(0, haystack_len - needle_len + 1, 1)


@always_inline
fn _reverse_range(haystack_len: Int, needle_len: Int) -> _StridedRange:
    return range(haystack_len - needle_len, -1, -1)


@always_inline
fn _memmem[
    type: DType
](
    haystack: DTypePointer[type],
    haystack_len: Int,
    needle: DTypePointer[type],
    needle_len: Int,
) -> DTypePointer[type]:
    return _memmem[type, _forward_range](
        haystack, haystack_len, needle, needle_len
    )


@always_inline
fn _memrmem[
    type: DType
](
    haystack: DTypePointer[type],
    haystack_len: Int,
    needle: DTypePointer[type],
    needle_len: Int,
) -> DTypePointer[type]:
    return _memmem[type, _reverse_range](
        haystack, haystack_len, needle, needle_len
    )


fn _calc_initial_buffer_size_int32(n0: Int) -> Int:
    # See https://commaok.xyz/post/lookup_tables/ and
    # https://lemire.me/blog/2021/06/03/computing-the-number-of-digits-of-an-integer-even-faster/
    # for a description.
    alias lookup_table = VariadicList[Int](
        4294967296,
        8589934582,
        8589934582,
        8589934582,
        12884901788,
        12884901788,
        12884901788,
        17179868184,
        17179868184,
        17179868184,
        21474826480,
        21474826480,
        21474826480,
        21474826480,
        25769703776,
        25769703776,
        25769703776,
        30063771072,
        30063771072,
        30063771072,
        34349738368,
        34349738368,
        34349738368,
        34349738368,
        38554705664,
        38554705664,
        38554705664,
        41949672960,
        41949672960,
        41949672960,
        42949672960,
        42949672960,
    )
    var n = UInt32(n0)
    var log2 = int((bitwidthof[DType.uint32]() - 1) ^ ctlz(n | 1))
    return (n0 + lookup_table[int(log2)]) >> 32


fn _calc_initial_buffer_size_int64(n0: UInt64) -> Int:
    var result: Int = 1
    var n = n0
    while True:
        if n < 10:
            return result
        if n < 100:
            return result + 1
        if n < 1_000:
            return result + 2
        if n < 10_000:
            return result + 3
        n //= 10_000
        result += 4


fn _calc_initial_buffer_size(n0: Int) -> Int:
    var n = _abs(n0)
    var sign = 0 if n0 > 0 else 1
    alias is_32bit_system = bitwidthof[DType.index]() == 32

    # Add 1 for the terminator
    @parameter
    if is_32bit_system:
        return sign + _calc_initial_buffer_size_int32(n) + 1

    # The value only has low-bits.
    if n >> 32 == 0:
        return sign + _calc_initial_buffer_size_int32(n) + 1
    return sign + _calc_initial_buffer_size_int64(n) + 1


fn _calc_initial_buffer_size(n: FloatLiteral) -> Int:
    return 128 + 1  # Add 1 for the terminator


fn _calc_initial_buffer_size[type: DType](n0: SIMD[type, 1]) -> Int:
    @parameter
    if type.is_integral():
        var n = _abs(n0)
        var sign = 0 if n0 > 0 else 1
        alias is_32bit_system = bitwidthof[DType.index]() == 32

        @parameter
        if is_32bit_system or bitwidthof[type]() <= 32:
            return sign + _calc_initial_buffer_size_int32(int(n)) + 1
        else:
            return (
                sign
                + _calc_initial_buffer_size_int64(n.cast[DType.uint64]())
                + 1
            )

    return 128 + 1  # Add 1 for the terminator
