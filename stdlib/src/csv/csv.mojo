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

from memory import memcmp, memcpy

alias QUOTE_MINIMAL = 0
alias QUOTE_ALL = 1
alias QUOTE_NONNUMERIC = 2
alias QUOTE_NONE = 3
alias QUOTE_STRINGS = 4
alias QUOTE_NOTNULL = 5


struct Dialect:
    """
    Describe a CSV dialect.
    """

    var _valid: Bool
    """Whether the dialect is valid."""
    var delimiter: String
    """The delimiter used to separate fields."""
    var quotechar: String
    """The character used to quote fields containing special characters."""
    var escapechar: String
    """The character used to escape the delimiter or quotechar."""
    var doublequote: Bool
    """Whether quotechar inside a field is doubled."""
    var skipinitialspace: Bool
    """Whether whitespace immediately following the delimiter is ignored."""
    var lineterminator: String
    """The sequence used to terminate lines."""
    var quoting: Int
    """The quoting mode."""

    fn __init__(
        inout self: Self,
        delimiter: String,
        quotechar: String,
        escapechar: String = "",
        doublequote: Bool = False,
        skipinitialspace: Bool = False,
        lineterminator: String = "\r\n",
        quoting: Int = QUOTE_MINIMAL,
    ):
        """
        Initialize a Dialect object.

        Args:
            delimiter: The delimiter used to separate fields.
            quotechar: The character used to quote fields containing special
                characters.
            escapechar: The character used to escape the delimiter or quotechar.
            doublequote: Whether quotechar inside a field is doubled.
            skipinitialspace: Whether whitespace immediately following the
                delimiter is ignored.
            lineterminator: The sequence used to terminate lines.
            quoting: The quoting mode.
        """
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.escapechar = escapechar
        self.doublequote = doublequote
        self.skipinitialspace = skipinitialspace
        self.lineterminator = lineterminator
        self.quoting = quoting
        self._valid = False

    fn validate(inout self: Self) raises:
        """
        Validate the dialect.
        """
        self._valid = _validate_reader_dialect(self)


struct reader:
    """
    CSV reader.

    This struct reads CSV files.

    Example:

        >>> with open("example.csv", "r") as csvfile:
        ...     reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        ...     for row in reader:
        ...         print(row)
        ['a', 'b', 'c']
        ['1', '2', '3']
    """

    var dialect: Dialect
    """The CSV dialect."""
    var content: String
    """The content of the CSV file."""

    fn __init__(
        inout self: Self,
        csvfile: FileHandle,
        delimiter: String,
        quotechar: String = '"',
        escapechar: String = "",
        doublequote: Bool = False,
        skipinitialspace: Bool = False,
        lineterminator: String = "\r\n",
        quoting: Int = QUOTE_MINIMAL,
    ) raises:
        """
        Initialize a Dialect object.

        Args:
            csvfile: The CSV file to read from.
            delimiter: The delimiter used to separate fields.
            quotechar: The character used to quote fields containing special
                characters.
            escapechar: The character used to escape the delimiter or quotechar.
            doublequote: Whether quotechar inside a field is doubled.
            skipinitialspace: Whether whitespace immediately following the
                delimiter is ignored.
            lineterminator: The sequence used to terminate lines.
            quoting: The quoting mode.
        """
        self.dialect = Dialect(
            delimiter=delimiter,
            quotechar=quotechar,
            escapechar=escapechar,
            doublequote=doublequote,
            skipinitialspace=skipinitialspace,
            lineterminator=lineterminator,
            quoting=quoting,
        )
        self.dialect.validate()

        # TODO: Implement streaming to prevent loading the entire file into memory
        self.content = csvfile.read()

    fn __iter__(self: Self) raises -> _ReaderIter[__lifetime_of(self)]:
        """
        Iterate through the CSV lines.

        Returns:
            Iterator.
        """
        return _ReaderIter[__lifetime_of(self)](reader=self)

    fn __len__(self: Self) -> Int:
        """
        Get the number of lines in the CSV file.

        Returns:
            The number of lines in the CSV file.
        """
        return len(self.content)


# ===------------------------------------------------------------------=== #
# Auxiliary structs and functions
# ===------------------------------------------------------------------=== #

alias START_RECORD = 0
alias START_FIELD = 1
alias IN_FIELD = 2
alias IN_QUOTED_FIELD = 3
alias ESCAPED_CHAR = 4
alias ESCAPED_IN_QUOTED_FIELD = 5
alias END_FIELD = 6
alias END_RECORD = 7
alias QUOTE_IN_QUOTED_FIELD = 8


struct _ReaderIter[
    reader_mutability: Bool, //,
    reader_lifetime: AnyLifetime[reader_mutability].type,
](Sized):
    """Iterator for any random-access container"""

    var reader_ref: Reference[reader, reader_lifetime]
    var pos: Int
    var field_pos: Int
    var quoted: Bool
    var quotechar: String
    var delimiter: String
    var doublequote: Bool
    var escapechar: String
    var quoting: Int
    var eat_crnl: Bool
    var content_ptr: UnsafePointer[UInt8]
    var bytes_len: Int

    fn __init__(inout self, ref [reader_lifetime]reader: reader):
        self.reader_ref = reader
        self.pos = 0
        self.field_pos = 0
        self.quoted = False
        self.quotechar = reader.dialect.quotechar
        self.delimiter = reader.dialect.delimiter
        self.doublequote = reader.dialect.doublequote
        self.escapechar = reader.dialect.escapechar
        self.quoting = reader.dialect.quoting
        self.content_ptr = reader.content.unsafe_ptr()
        self.bytes_len = len(reader)
        self.eat_crnl = False

    @always_inline
    fn __next__(inout self: Self) raises -> List[String]:
        return self.next_row()

    fn __len__(self) -> Int:
        # This is the current way to imitate the StopIteration exception
        # TODO: Remove when the iterators are implemented and streaming is done
        return self.bytes_len - self.pos

    fn next_row(inout self) -> List[String]:
        var row = List[String]()

        # TODO: This is spaghetti code mimicing the CPython implementation
        #       We should refactor this to be more readable and maintainable
        #       See parse_process_char() function in cpython/Modules/_csv.c
        var state = START_RECORD

        var content_ptr = self.content_ptr
        var delimiter_ptr = self.delimiter.unsafe_ptr()
        var delimiter_len = self.delimiter.byte_length()
        var quotechar_ptr = self.quotechar.unsafe_ptr()
        var quotechar_len = self.quotechar.byte_length()
        var escapechar_ptr = self.escapechar.unsafe_ptr()
        var escapechar_len = self.escapechar.byte_length()

        @always_inline
        fn _is_delimiter(ptr: UnsafePointer[UInt8]) -> Bool:
            return _is_eq(ptr, delimiter_ptr, delimiter_len)

        @always_inline
        fn _is_quotechar(ptr: UnsafePointer[UInt8]) -> Bool:
            return _is_eq(ptr, quotechar_ptr, quotechar_len)

        @always_inline
        fn _is_escapechar(ptr: UnsafePointer[UInt8]) -> Bool:
            return escapechar_len and _is_eq(
                ptr, escapechar_ptr, escapechar_len
            )

        if _is_eol(content_ptr.offset(self.pos)):
            self.pos += 1

        self.field_pos = self.pos
        self.eat_crnl = False

        while self.pos < self.bytes_len:
            var curr_ptr = content_ptr.offset(self.pos)

            # print(
            #     "CHAR: ", repr(chr(int(curr_ptr[]))), " STATE:", state, " POS: ", self.pos
            # )

            # TODO: Use match statement when supported by Mojo
            if state == START_RECORD:
                if _is_eol(curr_ptr):
                    state = END_RECORD
                else:
                    state = START_FIELD
                continue  # do not consume the character
            elif state == START_FIELD:
                self.field_pos = self.pos
                if _is_delimiter(curr_ptr):
                    # save empty field
                    self._save_field(row)
                elif _is_quotechar(curr_ptr):
                    self._mark_quote()
                    state = IN_QUOTED_FIELD
                else:
                    state = IN_FIELD
                    continue  # do not consume the character
            elif state == IN_FIELD:
                if _is_delimiter(curr_ptr):
                    state = END_FIELD
                    continue
                elif _is_eol(curr_ptr):
                    state = END_RECORD
                elif _is_escapechar(curr_ptr):
                    state = ESCAPED_CHAR
                else:
                    pass
            elif state == IN_QUOTED_FIELD:
                if _is_quotechar(curr_ptr):
                    if self.doublequote:
                        state = QUOTE_IN_QUOTED_FIELD
                    else:  # end of quoted field
                        state = IN_FIELD
                elif _is_escapechar(curr_ptr):
                    state = ESCAPED_IN_QUOTED_FIELD
                else:
                    pass
            elif state == QUOTE_IN_QUOTED_FIELD:
                # double-check with CPython implementation
                if _is_quotechar(curr_ptr):
                    state = IN_QUOTED_FIELD
                elif _is_delimiter(curr_ptr):
                    self._save_field(row)
                    state = START_FIELD
            elif state == ESCAPED_CHAR:
                state = IN_QUOTED_FIELD
            elif state == ESCAPED_IN_QUOTED_FIELD:
                state = IN_QUOTED_FIELD
            elif state == END_FIELD:
                self._save_field(row)
                state = START_FIELD
            elif state == END_RECORD:
                self.eat_crnl = True
                break

            self.pos += 1

        if self.field_pos < self.pos:
            self.eat_crnl = True
            self._save_field(row)

        # TODO: Handle the escapechar and skipinitialspace options
        return row

    @always_inline("nodebug")
    fn _mark_quote(inout self):
        self.quoted = True

    fn _save_field(inout self, inout row: List[String]):
        start_idx, end_idx = (
            self.field_pos,
            self.pos,
        ) if not self.quoted else (self.field_pos + 1, self.pos - 1)
        if self.eat_crnl:
            end_idx -= 1

        # TODO: Not sure if there is a cleaner way to do it performance-wise
        var length = end_idx - start_idx
        var buff = List[UInt8, hint_trivial_type=True]()
        buff.resize(length + 1, 0)
        memcpy(
            dest=buff.data, src=self.content_ptr.offset(start_idx), count=length
        )
        var final_field = String(buff)

        if self.doublequote:
            quotechar = self.quotechar
            final_field = final_field.replace(quotechar * 2, quotechar)
        row.append(final_field)
        # reset values
        self.quoted = False


fn _validate_reader_dialect(dialect: Dialect) raises -> Bool:
    """
    Validate a dialect.

    Args:
        dialect: A Dialect object.

    Returns:
        True if the dialect is valid, False if not.
    """
    if len(dialect.delimiter) != 1:
        raise Error("TypeError: delimiter must be a 1-character string")
    if len(dialect.quotechar) != 1:
        raise Error("TypeError: quotechar must be a 1-character string")
    if dialect.escapechar:
        if len(dialect.escapechar) != 1:
            raise Error("TypeError: escapechar must be a 1-character string")
        if (
            dialect.escapechar == dialect.delimiter
            or dialect.escapechar == dialect.quotechar
        ):
            raise Error(
                "TypeError: escapechar must not be delimiter or quotechar"
            )
    if dialect.quoting not in (
        QUOTE_ALL,
        QUOTE_MINIMAL,
        QUOTE_NONNUMERIC,
        QUOTE_NONE,
        QUOTE_STRINGS,
        QUOTE_NOTNULL,
    ):
        raise Error("TypeError: bad 'quoting' value")
    if dialect.doublequote:
        if dialect.escapechar in (dialect.delimiter, dialect.quotechar):
            raise Error(
                "TypeError: single-character escape sequence must be different"
                " from delimiter and quotechar"
            )
    return True


@always_inline("nodebug")
fn _is_eq(
    ptr1: UnsafePointer[UInt8], ptr2: UnsafePointer[UInt8], len: Int
) -> Bool:
    return memcmp(ptr1, ptr2, len) == 0


@always_inline("nodebug")
fn _is_eol(ptr: UnsafePointer[UInt8]) -> Bool:
    alias nl = ord("\n")
    alias cr = ord("\r")
    var c = ptr[]
    return c == nl or c == cr
