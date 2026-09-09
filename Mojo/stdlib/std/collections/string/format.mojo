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
"""String formatting utilities for Mojo.

This module provides string formatting functionality similar to Python's
`str.format()` method. The `format()` method (available on the
[`String`](/docs/std/collections/string/string/String/#format) and
[`StringSpan`](/docs/std/collections/string/string_span/StringSpan/#format)
types) takes the current string as a template (or "format string"), which can
contain literal text and/or replacement fields delimited by curly braces (`{}`).
The replacement fields are replaced with the values of the arguments.

Replacement fields can mapped to the arguments in one of two ways:

- Automatic indexing by argument position:

  ```mojo
  var s = "{} is {}".format("Mojo", "🔥")
  ```

- Manual indexing by argument position:

  ```mojo
  var s = "{1} is {0}".format("hot", "🔥")
  ```

The replacement fields can also contain the `!r` or `!s` conversion flags, to
indicate whether the argument should be formatted using `repr()` or `String()`,
respectively:

```mojo
var s = "{!r}".format("test")
```

Note that the following features from Python's `str.format()` are
**not yet supported**:

- Named arguments (for example `"{name} is {adjective}"`).
- Accessing the attributes of an argument value (for example, `"{0.name}"`.
- Accessing an indexed value from the argument (for example, `"{1[0]}"`).
- Format specifiers for controlling output format (width, precision, and so on).

Examples:

```mojo
# Basic formatting
var s1 = "Hello {0}!".format("World")  # Hello World!

# Multiple arguments
var s2 = "{0} plus {1} equals {2}".format(1, 2, 3)  # 1 plus 2 equals 3

# Conversion flags
var s4 = "{!r}".format("test")  # "'test'"
```

This module has no public API; its functionality is available through the
[`String.format()`](/docs/std/collections/string/string/String/#format) and
[`StringSpan.format()`](/docs/std/collections/string/string_span/StringSpan/#format)
methods.
"""


from std.builtin.globals import global_constant
from std.collections.string.string_span import get_static_string
from std.utils import Variant

# ===-----------------------------------------------------------------------===#
# Formatter
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _PrecompiledEntries[
    format_origin: ImmOrigin, entry_origin: ImmOrigin, //, *Ts: Writable
](ImplicitlyCopyable):
    """Holds a non-owning view of precompiled format string entries.

    This struct stores the parsed replacement fields from a format string along
    with metadata for efficient formatting. It uses a `Span` to reference the
    entries without owning them, allowing it to work with both static (compile-
    time) and runtime-allocated entries:

    - **Static origins**: When used with `compile_entries_comptime()`, both the
      format string and entries have static storage duration (stored in global
      constants), enabling zero-allocation formatting.
    - **Runtime origins**: When converted from `_PrecompiledEntriesRuntime`, it
      provides a non-owning view of runtime-allocated entries via `Span`.

    Parameters:
        format_origin: The origin of the format string data (can be static or runtime).
        entry_origin: The origin of the entries array (can be static or runtime).
        Ts: The types of the arguments that will be formatted.
    """

    var entries: Span[_FormatCurlyEntry[Self.format_origin], Self.entry_origin]
    var size_hint: Int
    var format: StringSlice[Self.format_origin]


@fieldwise_init
struct _PrecompiledEntriesRuntime[format_origin: ImmOrigin, //, *Ts: Writable](
    Movable
):
    """Holds precompiled format string entries with owned runtime-allocated storage.

    This struct is similar to `_PrecompiledEntries` but uses a `List` to own
    the dynamically-allocated entries. It's used by `compile_entries_runtime()`
    when parsing format strings at runtime. The entries can then be converted
    to a `_PrecompiledEntries` (via `Span`) for use with `format_precompiled()`.

    Parameters:
        format_origin: The origin of the format string data (runtime).
        Ts: The types of the arguments that will be formatted.
    """

    var entries: List[_FormatCurlyEntry[Self.format_origin]]
    var size_hint: Int
    var format: StringSlice[Self.format_origin]


@always_inline
def _comptime_list_to_span[
    T: Deinitable & Copyable, //, list: List[T]
]() -> Span[T, ImmStaticOrigin]:
    """Convert a comptime list to a runtime span of static constant origin."""

    def list_to_array[list: List[T]]() -> Array[T, len(list)]:
        var array = Array[T, len(list)](uninitialized=True)

        comptime for i in range(len(list)):
            Pointer(to=array[i]).unsafe_write(materialize[list[i].copy()]())
        return array^

    comptime array = list_to_array[list]()
    return Span(global_constant[array]())


struct _FormatUtils:
    # TODO: Allow a way to provide a `comptime _PrecompiledEntries` to avoid
    # allocations in the `_PrecompiledEntries` struct.
    @staticmethod
    def format_precompiled[
        *Ts: Writable,
    ](
        mut writer: Some[Writer],
        compiled: _PrecompiledEntries[*Ts],
        *args: *Ts,
    ):
        """Format the arguments using the given format string and precompiled entries.
        """
        var offset = 0
        var ptr = compiled.format.as_bytes().unsafe_ptr()
        var fmt_len = compiled.format.byte_length()

        @always_inline
        def _build_slice(
            p: ImmPointer[UInt8, _], start: Int, end: Int
        ) -> StringSlice[p.origin]:
            return StringSlice(
                unsafe_from_utf8=Span(
                    unsafe_ptr=p.unsafe_offset(start), length=end - start
                )
            )

        var auto_arg_index = 0
        for e in compiled.entries:
            # offset can equal fmt_len when format ends with a replacement field
            assert offset <= fmt_len, "offset > format.byte_length()"
            writer.write(_build_slice(ptr, offset, e.first_curly))
            e._format_entry[*Ts](writer, *args, auto_idx=auto_arg_index)
            offset = e.last_curly + 1

        writer.write(_build_slice(ptr, offset, fmt_len))

    @staticmethod
    def format[*Ts: Writable](format: StringSlice, *args: *Ts) raises -> String:
        """Format the arguments using the given format string.

        Parameters:
            Ts: The types of the format substitution values; each must be
                `Writable`.

        Args:
            format: The format string.
            args: Values substituted into the format string.
        """
        var buffer = String()
        Self.format_to_runtime(buffer, format, *args)
        return buffer^

    @staticmethod
    def format_to_runtime[
        *Ts: Writable,
    ](mut writer: Some[Writer], format: StringSlice, *args: *Ts,) raises:
        """Format arguments into a writer using a runtime format string.

        This function parses and compiles the format string at runtime, then
        writes the formatted output to the provided writer. Use this when the
        format string is not known at compile time.

        For compile-time format strings, prefer `format_to_comptime()` which
        parses the format string at compile time and can catch format errors
        during compilation.

        Parameters:
            Ts: The types of the format substitution values; each must be
                `Writable`.

        Args:
            writer: The writer to write the formatted output to.
            format: The format string to parse.
            args: The arguments to format into the replacement fields.

        Raises:
            An error if the format string is invalid or if replacement fields
            don't match the provided arguments.
        """
        var compiled = Self.compile_entries_runtime[*Ts](format)
        Self.format_precompiled(
            writer,
            _PrecompiledEntries[*Ts](
                Span(compiled.entries), compiled.size_hint, compiled.format
            ),
            *args,
        )

    @staticmethod
    def format_to_comptime[
        format: StaticString,
        *Ts: Writable,
    ](mut writer: Some[Writer], *args: *Ts):
        """Format arguments into a writer using a compile-time format string.

        This function parses and compiles the format string at compile time,
        enabling zero-allocation formatting and catching format errors during
        compilation rather than at runtime.

        This is more efficient than `format_to_runtime()` because:
        - Format string parsing happens once at compile time
        - Format errors are caught during compilation
        - The compiled entries are stored in static memory
        - No runtime allocations for parsing the format string

        Parameters:
            format: The format string to parse at compile time. Must be a
                string literal or StaticString.
            Ts: The types of the format substitution values; each must be
                `Writable`.

        Args:
            writer: The writer to write the formatted output to.
            args: The arguments to format into the replacement fields.
        """
        comptime result = _FormatUtils.compile_entries_runtime_no_raises[*Ts](
            format
        )

        comptime if result.isa[Error]():
            comptime assert not result.isa[Error](), String(result[Error])
        else:
            comptime entries = result[type_of(result).Ts[0]]
            _FormatUtils.format_precompiled[*Ts](
                writer,
                _PrecompiledEntries[*Ts](
                    _comptime_list_to_span[entries.entries](),
                    entries.size_hint,
                    get_static_string[format](),
                ),
                *args,
            )

    @staticmethod
    def compile_entries_runtime_no_raises[
        *Ts: Writable
    ](
        format: StringSlice,
    ) -> Variant[
        _PrecompiledEntriesRuntime[format_origin=ImmOrigin(format.origin), *Ts],
        Error,
    ]:
        """Parses and compiles a format string without raising an error.

        Instead of raising an error, this function returns a `Variant`. This is
        useful if you are trying to call `compile_entries` at comptime which
        does not support raising functions.
        """
        try:
            return Self.compile_entries_runtime[*Ts](format)
        except e:
            return e^

    @staticmethod
    def compile_entries_runtime[
        *Ts: Writable
    ](
        format: StringSlice,
    ) raises -> _PrecompiledEntriesRuntime[
        format_origin=ImmOrigin(format.origin), *Ts
    ]:
        """Parses and compiles a format string at runtime.

        This function analyzes a format string, parses all replacement fields
        (e.g., `{}`, `{0}`, `{!r}`), and validates them against the provided
        argument types. The parsed entries are stored in a dynamically-allocated
        `List` for later use in formatting.

        Parameters:
            Ts: The types of the arguments that will be formatted.

        Args:
            format: The format string to parse.

        Returns:
            A `_PrecompiledEntriesRuntime` struct containing the parsed format
            entries and metadata.

        Raises:
            An error if the format string is invalid or if replacement fields
            don't match the provided argument types.
        """
        comptime FormatOrigin = ImmOrigin(format.origin)
        comptime EntryType = _FormatCurlyEntry[FormatOrigin]

        var manual_indexing_count = 0
        var automatic_indexing_count = 0
        var raised_manual_index = Optional[Int](None)
        var raised_automatic_index = Optional[Int](None)
        var raised_kwarg_field = Optional[StringSlice[FormatOrigin]](None)
        comptime n_args = Ts.length
        comptime `}` = UInt8(ord("}"))
        comptime `{` = UInt8(ord("{"))
        comptime l_err = "there is a single curly { left unclosed or unescaped"
        comptime r_err = "there is a single curly } left unclosed or unescaped"

        var entries = List[EntryType]()
        var start = Optional[Int](None)
        var skip_next = False
        var fmt_bytes = format.as_bytes()
        var fmt_len = len(fmt_bytes)
        var total_estimated_entry_byte_width = 0

        for i in range(fmt_len):
            if skip_next:
                skip_next = False
                continue
            if fmt_bytes.unsafe_get(i) == `{`:
                if not start:
                    start = i
                    continue
                if i - start.value() != 1:
                    raise Error(l_err)
                # python escapes double curlies
                entries.append(EntryType(start.value(), i, field=False))
                start = None
                continue
            elif fmt_bytes.unsafe_get(i) == `}`:
                if not start:
                    # python escapes double curlies
                    if (i + 1) < fmt_len and fmt_bytes.unsafe_get(i + 1) == `}`:
                        entries.append(EntryType(i, i + 1, field=True))
                        total_estimated_entry_byte_width += 2
                        skip_next = True
                        continue
                    # if it is not an escaped one, it is an error
                    raise Error(r_err)

                var start_value = start.value()
                var current_entry = EntryType(start_value, i, field=NoneType())

                if i - start_value != 1:
                    if current_entry._handle_field_and_break(
                        format,
                        n_args,
                        i,
                        start_value,
                        automatic_indexing_count,
                        raised_automatic_index,
                        manual_indexing_count,
                        raised_manual_index,
                        raised_kwarg_field,
                        total_estimated_entry_byte_width,
                    ):
                        break
                else:  # automatic indexing
                    if automatic_indexing_count >= n_args:
                        raised_automatic_index = automatic_indexing_count
                        break
                    automatic_indexing_count += 1
                    total_estimated_entry_byte_width += 8  # guessing
                entries.append(current_entry^)
                start = None

        if raised_automatic_index:
            raise Error("Automatic indexing require more args in *args")
        elif raised_kwarg_field:
            var val = raised_kwarg_field.value()
            raise Error("Index ", val, " not in kwargs")
        elif manual_indexing_count and automatic_indexing_count:
            raise Error("Cannot both use manual and automatic indexing")
        elif raised_manual_index:
            var val = raised_manual_index.value()
            raise Error("Index ", val, " not in *args")
        elif start:
            raise Error(l_err)
        return {entries^, total_estimated_entry_byte_width, format}


# NOTE(#3765): an interesting idea would be to allow custom start and end
# characters for formatting (passed as parameters to Formatter), this would be
# useful for people developing custom templating engines as it would allow
# determining e.g. `<mojo` [...] `>` [...] `</mojo>` html tags.
# And going a step further it might even be worth it adding custom format
# specification start character, and custom format specs themselves (by defining
# a trait that all format specifications conform to)
struct _FormatCurlyEntry[origin: ImmOrigin](ImplicitlyCopyable):
    """The struct that handles string formatting by curly braces entries.
    This is internal for the types: `StringSlice` compatible types.
    """

    var first_curly: Int
    """The index of an opening brace around a substitution field."""
    var last_curly: Int
    """The index of a closing brace around a substitution field."""
    # TODO: ord("a") conversion flag not supported yet
    var conversion_flag: UInt8
    """The type of conversion for the entry: {ord("s"), ord("r")}."""
    # TODO: ord("a") conversion flag not supported yet
    comptime supported_conversion_flags = SIMD[.uint8, 2](
        UInt8(ord("s")), UInt8(ord("r"))
    )
    """Currently supported conversion flags: `__str__` and `__repr__`."""
    comptime _FieldVariantType = Variant[
        StringSlice[Self.origin], Int, NoneType, Bool
    ]
    """Purpose of the `Variant` `Self.field`:

    - `Int` for manual indexing: (value field contains `0`).
    - `NoneType` for automatic indexing: (value field contains `None`).
    - `StringSlice` for **kwargs indexing: (value field contains `foo`).
    - `Bool` for escaped curlies: (value field contains False for `{` or True
        for `}`).
    """
    var field: Self._FieldVariantType
    """Store the substitution field. See `Self._FieldVariantType` docstrings for
    more details."""

    def __init__(
        out self,
        first_curly: Int,
        last_curly: Int,
        field: Self._FieldVariantType,
        conversion_flag: UInt8 = 0,
    ):
        """Construct a format entry.

        Args:
            first_curly: The index of an opening brace around a substitution
                field.
            last_curly: The index of a closing brace around a substitution
                field.
            field: Store the substitution field.
            conversion_flag: The type of conversion for the entry.
        """
        self.first_curly = first_curly
        self.last_curly = last_curly
        self.field = field
        self.conversion_flag = conversion_flag

    @always_inline
    def is_escaped_brace(ref self) -> Bool:
        """Whether the field is escaped_brace.

        Returns:
            The result.
        """
        return self.field.isa[Bool]()

    @always_inline
    def is_kwargs_field(ref self) -> Bool:
        """Whether the field is kwargs_field.

        Returns:
            The result.
        """
        return self.field.isa[String]()

    @always_inline
    def is_automatic_indexing(ref self) -> Bool:
        """Whether the field is automatic_indexing.

        Returns:
            The result.
        """
        return self.field.isa[NoneType]()

    @always_inline
    def is_manual_indexing(ref self) -> Bool:
        """Whether the field is manual_indexing.

        Returns:
            The result.
        """
        return self.field.isa[Int]()

    def _handle_field_and_break(
        mut self,
        fmt_src: StringSlice[Self.origin],
        len_pos_args: Int,
        i: Int,
        start_value: Int,
        mut automatic_indexing_count: Int,
        mut raised_automatic_index: Optional[Int],
        mut manual_indexing_count: Int,
        mut raised_manual_index: Optional[Int],
        mut raised_kwarg_field: Optional[StringSlice[Self.origin]],
        mut total_estimated_entry_byte_width: Int,
    ) raises -> Bool:
        @always_inline("nodebug")
        def _build_slice(
            p: ImmPointer[UInt8, _], start: Int, end: Int
        ) -> StringSlice[p.origin]:
            return StringSlice(
                unsafe_from_utf8=Span[UInt8, p.origin](
                    unsafe_ptr=p.unsafe_offset(start), length=end - start
                )
            )

        var field = _build_slice(
            fmt_src.as_bytes().unsafe_ptr(), start_value + 1, i
        )
        var field_ptr = field.as_bytes().unsafe_ptr()
        var field_len = i - (start_value + 1)
        var exclamation_index = -1
        var idx = 0
        while idx < field_len:
            if field_ptr[unsafe_offset=idx] == UInt8(ord("!")):
                exclamation_index = idx
                break
            idx += 1
        var new_idx = exclamation_index + 1
        if exclamation_index != -1:
            if new_idx == field_len:
                raise Error("Empty conversion flag.")
            var conversion_flag = field_ptr[unsafe_offset=new_idx]
            if field_len - new_idx > 1 or (
                conversion_flag not in Self.supported_conversion_flags
            ):
                var f = _build_slice(field_ptr, new_idx, field_len)
                raise Error('Conversion flag "', f, '" not recognized.')
            self.conversion_flag = conversion_flag
            field = _build_slice(field_ptr, 0, exclamation_index)
        else:
            new_idx += 1

        # TODO(MSTDL-2243): Add format spec parsing

        if field.byte_length() == 0:
            # an empty field, so it's automatic indexing
            if automatic_indexing_count >= len_pos_args:
                raised_automatic_index = automatic_indexing_count
                return True
            automatic_indexing_count += 1
        else:
            try:
                # field is a number for manual indexing:
                # TODO: add support for "My name is {0.name}".format(Person(name="Fred"))
                # TODO: add support for "My name is {0[name]}".format({"name": "Fred"})
                var number = Int(field)
                self.field = number
                if number >= len_pos_args or number < 0:
                    raised_manual_index = number
                    return True
                manual_indexing_count += 1
            except e:

                def check_string() {e} -> Bool:
                    return "not convertible to integer" in String(e)

                debug_assert(check_string, "Not the expected error from atol")
                # field is a keyword for **kwargs:
                # TODO: add support for "My name is {person.name}".format(person=Person(name="Fred"))
                # TODO: add support for "My name is {person[name]}".format(person={"name": "Fred"})
                var f = field
                self.field = f
                raised_kwarg_field = f
                return True
        return False

    def _format_entry[
        *Ts: Writable,
    ](self, mut writer: Some[Writer], *args: *Ts, mut auto_idx: Int):
        # TODO(#3403 and/or #3252): this function should be able to use
        # Writer syntax when the type implements it, since it will give great
        # performance benefits. This also needs to be able to check if the given
        # args[i] conforms to the trait needed by the conversion_flag to avoid
        # needing to constraint that every type needs to conform to every trait.
        comptime r_value = UInt8(ord("r"))
        comptime s_value = UInt8(ord("s"))
        # alias a_value = UInt8(ord("a")) # TODO

        def _format(idx: Int) {imm self, imm args, mut writer}:
            comptime for i in range(Ts.length):
                if i == idx:
                    var flag = self.conversion_flag
                    var empty = flag == 0

                    ref arg = args[i]
                    if empty or flag == s_value:
                        arg.write_to(writer)
                    elif flag == r_value:
                        arg.write_repr_to(writer)

        if self.is_escaped_brace():
            writer.write("}" if self.field[Bool] else "{")
        elif self.is_manual_indexing():
            _format(self.field[Int])
        elif self.is_automatic_indexing():
            _format(auto_idx)
            auto_idx += 1
