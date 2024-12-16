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
"""Implements Formatting utilities."""

from collections import Optional

from memory import UnsafePointer

# TODO: _FormatCurlyEntry and _FormatSpec should be public in the future for
# people who want to write their own templating engines. This is not yet done
# because the implementation is incomplete and we are missing crucial features.

# ===-----------------------------------------------------------------------===#
# Formatter
# ===-----------------------------------------------------------------------===#


# NOTE(#3765): an interesting idea would be to allow custom start and end
# characters for formatting (passed as parameters to Formatter), this would be
# useful for people developing custom templating engines as it would allow
# detemining e.g. `<mojo` [...] `>` [...] `</mojo>` html tags.
# And going a step further it might even be worth it adding custom format
# specification start character, and custom format specs themselves (by defining
# a trait that all format specifications conform to)
@value
struct _FormatCurlyEntry(CollectionElement, CollectionElementNew):
    """The struct that handles string formatting by curly braces entries.
    This is internal for the types: `String`, `StringLiteral` and `StringSlice`.
    """

    var first_curly: Int
    """The index of an opening brace around a substitution field."""
    var last_curly: Int
    """The index of a closing brace around a substitution field."""
    # TODO: ord("a") conversion flag not supported yet
    var conversion_flag: UInt8
    """The type of conversion for the entry: {ord("s"), ord("r")}."""
    var format_spec: Optional[_FormatSpec]
    """The format specifier."""
    # TODO: ord("a") conversion flag not supported yet
    alias supported_conversion_flags = SIMD[DType.uint8, 2](ord("s"), ord("r"))
    """Currently supported conversion flags: `__str__` and `__repr__`."""
    alias _FieldVariantType = Variant[String, Int, NoneType, Bool]
    """Purpose of the `Variant` `Self.field`:

    - `Int` for manual indexing: (value field contains `0`).
    - `NoneType` for automatic indexing: (value field contains `None`).
    - `String` for **kwargs indexing: (value field contains `foo`).
    - `Bool` for escaped curlies: (value field contains False for `{` or True
        for `}`).
    """
    var field: Self._FieldVariantType
    """Store the substitution field. See `Self._FieldVariantType` docstrings for
    more details."""
    alias _args_t = VariadicPack[element_trait=_CurlyEntryFormattable, *_]
    """Args types that are formattable by curly entry."""

    fn __init__(out self, *, other: Self):
        """Construct a format entry by copying another.

        Args:
            other: The other format entry.
        """
        self.first_curly = other.first_curly
        self.last_curly = other.last_curly
        self.conversion_flag = other.conversion_flag
        self.field = Self._FieldVariantType(other=other.field)
        self.format_spec = other.format_spec

    fn __init__(
        mut self,
        first_curly: Int,
        last_curly: Int,
        field: Self._FieldVariantType,
        conversion_flag: UInt8 = 0,
        format_spec: Optional[_FormatSpec] = None,
    ):
        """Construct a format entry.

        Args:
            first_curly: The index of an opening brace around a substitution
                field.
            last_curly: The index of a closing brace around a substitution
                field.
            field: Store the substitution field.
            conversion_flag: The type of conversion for the entry.
            format_spec: The format specifier.
        """
        self.first_curly = first_curly
        self.last_curly = last_curly
        self.field = field
        self.conversion_flag = conversion_flag
        self.format_spec = format_spec

    @always_inline
    fn is_escaped_brace(ref self) -> Bool:
        """Whether the field is escaped_brace.

        Returns:
            The result.
        """
        return self.field.isa[Bool]()

    @always_inline
    fn is_kwargs_field(ref self) -> Bool:
        """Whether the field is kwargs_field.

        Returns:
            The result.
        """
        return self.field.isa[String]()

    @always_inline
    fn is_automatic_indexing(ref self) -> Bool:
        """Whether the field is automatic_indexing.

        Returns:
            The result.
        """
        return self.field.isa[NoneType]()

    @always_inline
    fn is_manual_indexing(ref self) -> Bool:
        """Whether the field is manual_indexing.

        Returns:
            The result.
        """
        return self.field.isa[Int]()

    @staticmethod
    fn format(fmt_src: StringSlice, args: Self._args_t) raises -> String:
        """Format the entries.

        Args:
            fmt_src: The format source.
            args: The arguments.

        Returns:
            The result.
        """
        alias len_pos_args = __type_of(args).__len__()
        entries, size_estimation = Self._create_entries(fmt_src, len_pos_args)
        var fmt_len = fmt_src.byte_length()
        var buf = String._buffer_type(capacity=fmt_len + size_estimation)
        buf.size = 1
        buf.unsafe_set(0, 0)
        var res = String(buf^)
        var offset = 0
        var ptr = fmt_src.unsafe_ptr()
        alias S = StringSlice[StaticConstantOrigin]

        @always_inline("nodebug")
        fn _build_slice(p: UnsafePointer[UInt8], start: Int, end: Int) -> S:
            return S(ptr=p + start, length=end - start)

        var auto_arg_index = 0
        for e in entries:
            debug_assert(offset < fmt_len, "offset >= fmt_src.byte_length()")
            res += _build_slice(ptr, offset, e[].first_curly)
            e[]._format_entry[len_pos_args](res, args, auto_arg_index)
            offset = e[].last_curly + 1

        res += _build_slice(ptr, offset, fmt_len)
        return res^

    @staticmethod
    fn _create_entries(
        fmt_src: StringSlice, len_pos_args: Int
    ) raises -> (List[Self], Int):
        """Returns a list of entries and its total estimated entry byte width.
        """
        var manual_indexing_count = 0
        var automatic_indexing_count = 0
        var raised_manual_index = Optional[Int](None)
        var raised_automatic_index = Optional[Int](None)
        var raised_kwarg_field = Optional[String](None)
        alias `}` = UInt8(ord("}"))
        alias `{` = UInt8(ord("{"))
        alias l_err = "there is a single curly { left unclosed or unescaped"
        alias r_err = "there is a single curly } left unclosed or unescaped"

        var entries = List[Self]()
        var start = Optional[Int](None)
        var skip_next = False
        var fmt_ptr = fmt_src.unsafe_ptr()
        var fmt_len = fmt_src.byte_length()
        var total_estimated_entry_byte_width = 0

        for i in range(fmt_len):
            if skip_next:
                skip_next = False
                continue
            if fmt_ptr[i] == `{`:
                if not start:
                    start = i
                    continue
                if i - start.value() != 1:
                    raise Error(l_err)
                # python escapes double curlies
                entries.append(Self(start.value(), i, field=False))
                start = None
                continue
            elif fmt_ptr[i] == `}`:
                if not start and (i + 1) < fmt_len:
                    # python escapes double curlies
                    if fmt_ptr[i + 1] == `}`:
                        entries.append(Self(i, i + 1, field=True))
                        total_estimated_entry_byte_width += 2
                        skip_next = True
                        continue
                elif not start:  # if it is not an escaped one, it is an error
                    raise Error(r_err)

                var start_value = start.value()
                var current_entry = Self(start_value, i, field=NoneType())

                if i - start_value != 1:
                    if current_entry._handle_field_and_break(
                        fmt_src,
                        len_pos_args,
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
                    if automatic_indexing_count >= len_pos_args:
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
            raise Error("Index " + val + " not in kwargs")
        elif manual_indexing_count and automatic_indexing_count:
            raise Error("Cannot both use manual and automatic indexing")
        elif raised_manual_index:
            var val = str(raised_manual_index.value())
            raise Error("Index " + val + " not in *args")
        elif start:
            raise Error(l_err)
        return entries^, total_estimated_entry_byte_width

    fn _handle_field_and_break(
        mut self,
        fmt_src: StringSlice,
        len_pos_args: Int,
        i: Int,
        start_value: Int,
        mut automatic_indexing_count: Int,
        mut raised_automatic_index: Optional[Int],
        mut manual_indexing_count: Int,
        mut raised_manual_index: Optional[Int],
        mut raised_kwarg_field: Optional[String],
        mut total_estimated_entry_byte_width: Int,
    ) raises -> Bool:
        alias S = StringSlice[StaticConstantOrigin]

        @always_inline("nodebug")
        fn _build_slice(p: UnsafePointer[UInt8], start: Int, end: Int) -> S:
            return S(ptr=p + start, length=end - start)

        var field = _build_slice(fmt_src.unsafe_ptr(), start_value + 1, i)
        var field_ptr = field.unsafe_ptr()
        var field_len = i - (start_value + 1)
        var exclamation_index = -1
        var idx = 0
        while idx < field_len:
            if field_ptr[idx] == ord("!"):
                exclamation_index = idx
                break
            idx += 1
        var new_idx = exclamation_index + 1
        if exclamation_index != -1:
            if new_idx == field_len:
                raise Error("Empty conversion flag.")
            var conversion_flag = field_ptr[new_idx]
            if field_len - new_idx > 1 or (
                conversion_flag not in Self.supported_conversion_flags
            ):
                var f = String(_build_slice(field_ptr, new_idx, field_len))
                _ = field
                raise Error('Conversion flag "' + f + '" not recognised.')
            self.conversion_flag = conversion_flag
            field = _build_slice(field_ptr, 0, exclamation_index)
        else:
            new_idx += 1

        var extra = int(new_idx < field_len)
        var fmt_field = _build_slice(field_ptr, new_idx + extra, field_len)
        self.format_spec = _FormatSpec.parse(fmt_field)
        var w = int(self.format_spec.value().width) if self.format_spec else 0
        # fully guessing the byte width here to be at least 8 bytes per entry
        # minus the length of the whole format specification
        total_estimated_entry_byte_width += 8 * int(w > 0) + w - (field_len + 2)

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
                var number = int(field)
                self.field = number
                if number >= len_pos_args or number < 0:
                    raised_manual_index = number
                    return True
                manual_indexing_count += 1
            except e:
                alias unexp = "Not the expected error from atol"
                debug_assert("not convertible to integer" in str(e), unexp)
                # field is a keyword for **kwargs:
                # TODO: add support for "My name is {person.name}".format(person=Person(name="Fred"))
                # TODO: add support for "My name is {person[name]}".format(person={"name": "Fred"})
                var f = str(field)
                self.field = f
                raised_kwarg_field = f
                return True
        return False

    fn _format_entry[
        len_pos_args: Int
    ](self, mut res: String, args: Self._args_t, mut auto_idx: Int) raises:
        # TODO(#3403 and/or #3252): this function should be able to use
        # Writer syntax when the type implements it, since it will give great
        # performance benefits. This also needs to be able to check if the given
        # args[i] conforms to the trait needed by the conversion_flag to avoid
        # needing to constraint that every type needs to conform to every trait.
        alias `r` = UInt8(ord("r"))
        alias `s` = UInt8(ord("s"))
        # alias `a` = UInt8(ord("a")) # TODO

        @parameter
        fn _format(idx: Int) raises:
            @parameter
            for i in range(len_pos_args):
                if i == idx:
                    var type_impls_repr = True  # TODO
                    var type_impls_str = True  # TODO
                    var type_impls_write_repr = True  # TODO
                    var type_impls_write_str = True  # TODO
                    var flag = self.conversion_flag
                    var empty = flag == 0 and not self.format_spec

                    var data: String
                    if empty and type_impls_write_str:
                        data = str(args[i])  # TODO: use writer and return
                    elif empty and type_impls_str:
                        data = str(args[i])
                    elif flag == `s` and type_impls_write_str:
                        if empty:
                            # TODO: use writer and return
                            pass
                        data = str(args[i])
                    elif flag == `s` and type_impls_str:
                        data = str(args[i])
                    elif flag == `r` and type_impls_write_repr:
                        if empty:
                            # TODO: use writer and return
                            pass
                        data = repr(args[i])
                    elif flag == `r` and type_impls_repr:
                        data = repr(args[i])
                    elif self.format_spec:
                        self.format_spec.value().format(res, args[i])
                        return
                    else:
                        alias argnum = "Argument number: "
                        alias does_not = " does not implement the trait "
                        alias needed = "needed for conversion_flag: "
                        var flg = String(List[UInt8](flag, 0))
                        raise Error(argnum + str(i) + does_not + needed + flg)

                    if self.format_spec:
                        self.format_spec.value().format(
                            res, data.as_string_slice()
                        )
                    else:
                        res += data

        if self.is_escaped_brace():
            res += "}" if self.field[Bool] else "{"
        elif self.is_manual_indexing():
            _format(self.field[Int])
        elif self.is_automatic_indexing():
            _format(auto_idx)
            auto_idx += 1


# ===-----------------------------------------------------------------------===#
# Format Specification
# ===-----------------------------------------------------------------------===#


trait _CurlyEntryFormattable(Stringable, Representable):
    """This trait is used by the `format()` method to support format specifiers.
    Currently, it is a composition of both `Stringable` and `Representable`
    traits i.e. a type to be formatted must implement both. In the future this
    will be less constrained.
    """

    ...


# TODO: trait _FormattableStr: fn __format__(self, spec: FormatSpec) -> String:
# TODO: trait _FormattableWrite: fn __format__(self, spec: FormatSpec, *, writer: Writer):
# TODO: add usage of these traits before trying to coerce to repr/str/int/float


@value
@register_passable("trivial")
struct _FormatSpec:
    """Store every field of the format specifier in a byte (e.g., ord("+") for
    sign). It is stored in a byte because every [format specifier](
    https://docs.python.org/3/library/string.html#formatspec) is an ASCII
    character.
    """

    var fill: UInt8
    """If a valid align value is specified, it can be preceded by a fill
    character that can be any character and defaults to a space if omitted.
    """
    var align: UInt8
    """The meaning of the various alignment options is as follows:

    | Option | Meaning|
    |:------:|:-------|
    |'<' | Forces the field to be left-aligned within the available space \
    (this is the default for most objects).|
    |'>' | Forces the field to be right-aligned within the available space \
    (this is the default for numbers).|
    |'=' | Forces the padding to be placed after the sign (if any) but before \
    the digits. This is used for printing fields in the form `+000000120`. This\
    alignment option is only valid for numeric types. It becomes the default\
    for numbers when `0` immediately precedes the field width.|
    |'^' | Forces the field to be centered within the available space.|
    """
    var sign: UInt8
    """The sign option is only valid for number types, and can be one of the
    following:

    | Option | Meaning|
    |:------:|:-------|
    |'+' | indicates that a sign should be used for both positive as well as\
    negative numbers.|
    |'-' | indicates that a sign should be used only for negative numbers (this\
    is the default behavior).|
    |space | indicates that a leading space should be used on positive numbers,\
    and a minus sign on negative numbers.|
    """
    var coerce_z: Bool
    """The 'z' option coerces negative zero floating-point values to positive
    zero after rounding to the format precision. This option is only valid for
    floating-point presentation types.
    """
    var alternate_form: Bool
    """The alternate form is defined differently for different types. This
    option is only valid for types that implement the trait `# TODO: define
    trait`. For integers, when binary, octal, or hexadecimal output is used,
    this option adds the respective prefix '0b', '0o', '0x', or '0X' to the
    output value. For float and complex the alternate form causes the result of
    the conversion to always contain a decimal-point character, even if no
    digits follow it.
    """
    var width: UInt8
    """A decimal integer defining the minimum total field width, including any
    prefixes, separators, and other formatting characters. If not specified,
    then the field width will be determined by the content. When no explicit
    alignment is given, preceding the width field by a zero ('0') character
    enables sign-aware zero-padding for numeric types. This is equivalent to a
    fill character of '0' with an alignment type of '='.
    """
    var grouping_option: UInt8
    """The ',' option signals the use of a comma for a thousands separator. For
    a locale aware separator, use the 'n' integer presentation type instead. The
    '_' option signals the use of an underscore for a thousands separator for
    floating-point presentation types and for integer presentation type 'd'. For
    integer presentation types 'b', 'o', 'x', and 'X', underscores will be
    inserted every 4 digits. For other presentation types, specifying this
    option is an error.
    """
    var precision: UInt8
    """The precision is a decimal integer indicating how many digits should be
    displayed after the decimal point for presentation types 'f' and 'F', or
    before and after the decimal point for presentation types 'g' or 'G'. For
    string presentation types the field indicates the maximum field size - in
    other words, how many characters will be used from the field content. The
    precision is not allowed for integer presentation types.
    """
    var type: UInt8
    """Determines how the data should be presented.

    The available integer presentation types are:

    | Option | Meaning|
    |:------:|:-------|
    |'b' |Binary format. Outputs the number in base 2.|
    |'c' |Character. Converts the integer to the corresponding unicode\
    character before printing.|
    |'d' |Decimal Integer. Outputs the number in base 10.|
    |'o' |Octal format. Outputs the number in base 8.|
    |'x' |Hex format. Outputs the number in base 16, using lower-case letters\
    for the digits above 9.|
    |'X' |Hex format. Outputs the number in base 16, using upper-case letters\
    for the digits above 9. In case '#' is specified, the prefix '0x' will be\
    upper-cased to '0X' as well.|
    |'n' |Number. This is the same as 'd', except that it uses the current\
    locale setting to insert the appropriate number separator characters.|
    |None | The same as 'd'.|

    In addition to the above presentation types, integers can be formatted with
    the floating-point presentation types listed below (except 'n' and None).
    When doing so, float() is used to convert the integer to a floating-point
    number before formatting.

    The available presentation types for float and Decimal values are:

    | Option | Meaning|
    |:------:|:-------|
    |'e' |Scientific notation. For a given precision p, formats the number in\
    scientific notation with the letter `e` separating the coefficient from the\
    exponent. The coefficient has one digit before and p digits after the\
    decimal point, for a total of p + 1 significant digits. With no precision\
    given, uses a precision of 6 digits after the decimal point for float, and\
    shows all coefficient digits for Decimal. If no digits follow the decimal\
    point, the decimal point is also removed unless the # option is used.|
    |'E' |Scientific notation. Same as 'e' except it uses an upper case `E` as\
    the separator character.|
    |'f' |Fixed-point notation. For a given precision p, formats the number as\
    a decimal number with exactly p digits following the decimal point. With no\
    precision given, uses a precision of 6 digits after the decimal point for\
    float, and uses a precision large enough to show all coefficient digits for\
    Decimal. If no digits follow the decimal point, the decimal point is also\
    removed unless the '#' option is used.|
    |'F' |Fixed-point notation. Same as 'f', but converts nan to NAN and inf to\
    INF.|
    |'g' |General format. For a given precision p >= 1, this rounds the number\
    to p significant digits and then formats the result in either fixed-point\
    format or in scientific notation, depending on its magnitude. A precision\
    of 0 is treated as equivalent to a precision of 1.\
    The precise rules are as follows: suppose that the result formatted with\
    presentation type 'e' and precision p-1 would have exponent exp. Then, if\
    m <= exp < p, where m is -4 for floats and -6 for Decimals, the number is\
    formatted with presentation type 'f' and precision p-1-exp. Otherwise, the\
    number is formatted with presentation type 'e' and precision p-1. In both\
    cases insignificant trailing zeros are removed from the significand, and\
    the decimal point is also removed if there are no remaining digits\
    following it, unless the '#' option is used.\
    With no precision given, uses a precision of 6 significant digits for\
    float. For Decimal, the coefficient of the result is formed from the\
    coefficient digits of the value; scientific notation is used for values\
    smaller than 1e-6 in absolute value and values where the place value of the\
    least significant digit is larger than 1, and fixed-point notation is used\
    otherwise.\
    Positive and negative infinity, positive and negative zero, and nans, are\
    formatted as inf, -inf, 0, -0 and nan respectively, regardless of the\
    precision.|
    |'G' |General format. Same as 'g' except switches to 'E' if the number gets\
    too large. The representations of infinity and NaN are uppercased, too.|
    |'n' |Number. This is the same as 'g', except that it uses the current\
    locale setting to insert the appropriate number separator characters.|
    |'%' |Percentage. Multiplies the number by 100 and displays in fixed ('f')\
    format, followed by a percent sign.|
    |None |For float this is like the 'g' type, except that when fixed-point\
    notation is used to format the result, it always includes at least one\
    digit past the decimal point, and switches to the scientific notation when\
    exp >= p - 1. When the precision is not specified, the latter will be as\
    large as needed to represent the given value faithfully.\
    For Decimal, this is the same as either 'g' or 'G' depending on the value\
    of context.capitals for the current decimal context.\
    The overall effect is to match the output of str() as altered by the other\
    format modifiers.|
    """

    fn __init__(
        mut self,
        fill: UInt8 = ord(" "),
        align: UInt8 = 0,
        sign: UInt8 = ord("-"),
        coerce_z: Bool = False,
        alternate_form: Bool = False,
        width: UInt8 = 0,
        grouping_option: UInt8 = 0,
        precision: UInt8 = 0,
        type: UInt8 = 0,
    ):
        """Construct a FormatSpec instance.

        Args:
            fill: Defaults to space.
            align: Defaults to `0` which is adjusted to the default for the arg
                type.
            sign: Defaults to `-`.
            coerce_z: Defaults to False.
            alternate_form: Defaults to False.
            width: Defaults to `0` which is adjusted to the default for the arg
                type.
            grouping_option: Defaults to `0` which is adjusted to the default for
                the arg type.
            precision: Defaults to `0` which is adjusted to the default for the
                arg type.
            type: Defaults to `0` which is adjusted to the default for the arg
                type.
        """
        self.fill = fill
        self.align = align
        self.sign = sign
        self.coerce_z = coerce_z
        self.alternate_form = alternate_form
        self.width = width
        self.grouping_option = grouping_option
        self.precision = precision
        self.type = type

    @staticmethod
    fn parse(fmt_str: StringSlice) -> Optional[Self]:
        """Parses the format spec string.

        Args:
            fmt_str: The StringSlice with the format spec.

        Returns:
            An instance of FormatSpec.
        """

        # FIXME: the need for the following dynamic characteristics will
        # probably mean the parse method will have to be called at the
        # formatting stage in cases where it's dynamic.
        # TODO: add support for "{0:{1}}".format(123, "10")
        # TODO: add support for more complex cases as well
        # >>> width = 10
        # >>> precision = 4
        # >>> value = decimal.Decimal('12.34567')
        # >>> 'result: {value:{width}.{precision}}'.format(...)
        alias `:` = UInt8(ord(":"))
        var f_len = fmt_str.byte_length()
        var f_ptr = fmt_str.unsafe_ptr()
        var colon_idx = -1
        var idx = 0
        while idx < f_len:
            if f_ptr[idx] == `:`:
                exclamation_index = idx
                break
            idx += 1

        if colon_idx == -1:
            return None

        # TODO: Future implementation of format specifiers
        return None

    # TODO: this should be in StringSlice.__format__(self, spec: FormatSpec, *, writer: Writer):
    fn format(self, mut res: String, item: StringSlice) raises:
        """Transform a String according to its format specification.

        Args:
            res: The resulting String.
            item: The item to format.
        """

        # TODO: align, fill, etc.
        res += item

    fn format[T: _CurlyEntryFormattable](self, mut res: String, item: T) raises:
        """Stringify a type according to its format specification.

        Args:
            res: The resulting String.
            item: The item to stringify.
        """
        var type_implements_format_write = True  # TODO
        var type_implements_format_write_raising = True  # TODO
        var type_implements_format = True  # TODO
        var type_implements_format_raising = True  # TODO
        var type_implements_float = True  # TODO
        var type_implements_float_raising = True  # TODO
        var type_implements_int = True  # TODO
        var type_implements_int_raising = True  # TODO

        # TODO: send to the type's  __format__ method if it has one
        # TODO: transform to int/float depending on format spec
        # TODO: send to float/int 's  __format__ method
        # their methods should stringify as hex/bin/oct etc.
        res += str(item)


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#
