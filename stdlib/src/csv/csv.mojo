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

alias QUOTE_MINIMAL = 0
alias QUOTE_ALL = 1
alias QUOTE_NONNUMERIC = 2
alias QUOTE_NONE = 3
alias QUOTE_STRINGS = 4
alias QUOTE_NOTNULL = 5


@value
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


@value
struct reader:
    """
    CSV reader.

    This class reads CSV files.

    Example:

        >>> with open("example.csv", "r") as csvfile:
        ...     reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        ...     for row in reader:
        ...         print(row)
        ['a', 'b', 'c']
        ['1', '2', '3']
    """

    var _dialect: Dialect
    var _lines: List[String]

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
        self._dialect = Dialect(
            delimiter=delimiter,
            quotechar=quotechar,
            escapechar=escapechar,
            doublequote=doublequote,
            skipinitialspace=skipinitialspace,
            lineterminator=lineterminator,
            quoting=quoting,
        )
        self._dialect.validate()

        # TODO: Implement streaming to prevent loading the entire file into memory
        self._lines = csvfile.read().splitlines()

    fn get_row(self: Self, idx: Int) raises -> List[String]:
        """
        Returns an specific line in the CSV file.

        Args:
            idx: The index of the line to return.

        Returns:
            The line at the given index.
        """
        var line_str = self._lines[idx]

        var line_parser = _CSVLineParser(line_str, self._dialect)
        return line_parser.get_row()

    fn lines_count(self: Self) -> Int:
        """
        Returns the number of lines in the CSV file.

        Returns:
            The number of lines in the CSV file.
        """
        # TODO: This has no sense once we implement streaming
        return len(self._lines)

    fn __iter__(self: Self) raises -> _ReaderIter[__lifetime_of(self)]:
        """
        Iterate through the CSV lines.

        Returns:
            Iterator.
        """
        return _ReaderIter[__lifetime_of(self)](reader_ref=self, idx=0)


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

# Refactor above spaghetti code


struct _CSVLineParser:
    var line: String
    var dialect: Dialect
    var field: String
    var pos: Int
    var quoted: Bool

    def __init__(inout self, line: String, dialect: Dialect):
        self.line = line
        self.dialect = dialect
        self.field = ""
        self.pos = 0
        self.quoted = False

    fn get_row(inout self) -> List[String]:
        var pos = 0

        var row = List[String]()

        # TODO: This is spaghetti code mimicing the CPython implementation
        #       We should refactor this to be more readable and maintainable
        #       See parse_process_char() function in cpython/Modules/_csv.c
        var state = START_RECORD

        while pos < len(self.line):
            var c = self._get_char(pos)
            # print('CHAR: ', c, ' STATE:', state, ' FIELD: ', self.field, ' POS: ', pos)

            # TODO: Use match statement when supported by Mojo
            if state == START_RECORD:
                if c == "\n" or c == "\r":
                    state = END_RECORD
                else:
                    state = START_FIELD
                continue  # do not consume the character
            elif state == START_FIELD:
                if c == self.dialect.delimiter:
                    # save empty field
                    self._save_field(row)
                elif c == self.dialect.quotechar:
                    self._mark_quote()
                    state = IN_QUOTED_FIELD
                else:
                    state = IN_FIELD
                    continue  # do not consume the character
            elif state == IN_FIELD:
                if c == self.dialect.delimiter:
                    state = END_FIELD
                    continue
                elif c == "\n" or c == "\r":
                    state = END_RECORD
                elif self.dialect.escapechar and c == self.dialect.escapechar:
                    state = ESCAPED_CHAR
                else:
                    self._add_to_field(c)
            elif state == IN_QUOTED_FIELD:
                if c == self.dialect.quotechar:
                    if self.dialect.doublequote:
                        state = QUOTE_IN_QUOTED_FIELD
                    else:  # end of quoted field
                        state = IN_FIELD
                elif c == self.dialect.escapechar:
                    state = ESCAPED_IN_QUOTED_FIELD
                else:
                    self._add_to_field(c)
            elif state == QUOTE_IN_QUOTED_FIELD:
                # double-check with CPython implementation
                if c == self.dialect.quotechar:
                    self._add_to_field(c)
                    state = IN_QUOTED_FIELD
                elif c == self.dialect.delimiter:
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
                break
            pos += 1

        if self.field:
            self._save_field(row)

        # TODO: Handle the escapechar and skipinitialspace options
        return row

    @always_inline("nodebug")
    fn _get_char(self, pos: Int) -> String:
        return self.line[pos]

    @always_inline("nodebug")
    fn _add_to_field(inout self, c: String):
        self.field += c

    @always_inline("nodebug")
    fn _mark_quote(inout self):
        self.quoted = True

    fn _save_field(inout self, inout row: List[String]):
        var final_field: String = self.field
        var quotechar = self.dialect.quotechar
        if self.dialect.doublequote:
            final_field = final_field.replace(quotechar * 2, quotechar)
        row.append(final_field)
        # reset values
        self.quoted = False
        self.field = ""


@value
struct _ReaderIter[
    reader_mutability: Bool, //,
    reader_lifetime: AnyLifetime[reader_mutability].type,
](Sized):
    """Iterator for any random-access container"""

    var reader_ref: Reference[reader, reader_lifetime]
    var idx: Int

    @always_inline
    fn __next__(inout self: Self) raises -> List[String]:
        var line = self.reader_ref[].get_row(self.idx)
        self.idx += 1
        return line

    fn __len__(self) -> Int:
        # This is the current way to imitate the StopIteration exception
        # TODO: Remove when the iterators are implemented and streaming is done
        return self.reader_ref[].lines_count() - self.idx


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
