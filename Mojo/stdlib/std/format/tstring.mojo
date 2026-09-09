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
"""Implements `TString`, a template string that captures interpolated values at compile-time."""
import std.format._utils as fmt
from std.builtin.globals import global_constant
from std.os import abort
from std.utils import StaticTuple


@always_inline
def _strlen(ptr: ImmPointer[Byte, _]) -> Int:
    var offset = 0
    while ptr[unsafe_offset=offset]:
        offset += 1
    return offset


struct TString[origins: ImmOrigin, //, *Ts: Writable](Movable, Writable):
    """A template string that captures interpolated values at compile-time.

    TString is a zero-cost abstraction for string interpolation that preserves
    type information and defers formatting until explicitly requested. Unlike
    regular strings or f-strings, TString retains the original format template
    and typed values, enabling efficient lazy formatting and type-safe string
    composition.

    TString instances are created by the compiler when using t-string literal
    syntax: `t"Hello {name}!"`.

    Parameters:
        origins: The origin of the interpolated values.
        Ts: The types of the interpolated values.
    """

    comptime _InjectedValues = VariadicPack[
        origin=Self.origins, element_trait=Writable, False, *Self.Ts
    ]
    var _values: Self._InjectedValues
    var _encoded: ImmPointer[Byte, ImmStaticOrigin]
    """The template's NUL-separated literal parts, encoded at construction."""

    @doc_hidden
    @always_inline
    def __init__(
        out self,
        *,
        var pack: Self._InjectedValues,
        encoded: ImmPointer[Byte, ImmStaticOrigin],
    ):
        self._values = pack^
        self._encoded = encoded

    @always_inline
    def _write_to_impl(
        self, mut writer: Some[Writer], encoded_bytes: ImmPointer[Byte, _]
    ):
        var offset = 0

        @always_inline
        def write_string() {mut writer, imm} -> Int:
            var literal_start = encoded_bytes.unsafe_offset(offset)
            var literal_length = _strlen(literal_start)
            var string_literal = StringSlice(
                unsafe_from_utf8=Span(
                    unsafe_ptr=literal_start, length=literal_length
                )
            )
            writer.write_string(string_literal)
            return literal_length

        # Alternate writing NUL terminated string-literal part, followed
        # by the interpolated replacement field.
        comptime for i in range(Self.Ts.length):
            var length = write_string()
            offset += length + 1
            self._values[i].write_to(writer)

        # Write the final string literal part.
        _ = write_string()

    def write_to(self, mut writer: Some[Writer]):
        """Write the formatted string to a writer.

        This method implements the `Writable` trait by formatting the TString's
        template with its interpolated values and writing the result to the
        provided writer.

        Args:
            writer: The writer to output the formatted string to.
        """
        self._write_to_impl(writer, self._encoded)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write a debug representation of the TString to a writer.

        This method provides a detailed view of the TString's internal structure,
        showing the format template, type parameters, and the actual interpolated
        values. This is useful for debugging and understanding the TString's
        composition.

        Args:
            writer: The writer to output the debug representation to.
        """

        comptime assert Self.Ts.all_conforms_to[
            Writable
        ]()  # satisfy where clause.

        var self_ptr = Pointer(to=self)

        def fields(mut writer: Some[Writer]) {self_ptr}:
            self_ptr[]._values._write_to[is_repr=True](writer, start="", end="")

        fmt.FormatStruct(writer, "TString").params(
            fmt.TypeNames[*Self.Ts](),
        ).fields(fields)


@always_inline
def __make_tstring[
    format_string: __mlir_type.`!kgen.string`, *Ts: Writable
](
    *args: *Ts,
    out tstring: TString[origins=ImmOrigin(type_of(args).origin), *Ts],
):
    """Compiler entry point for creating TStrings from t-string expressions.

    This function is called by the compiler when it encounters a t-string
    literal expression like `t"Hello {name}!"`. The compiler extracts the
    format string and argument expressions, then generates a call to this
    function to construct the corresponding TString object.

    Parameters:
        format_string: The compile-time string literal containing the template.
        Ts: The types of the interpolated values.

    Args:
        args: The values to interpolate into the template string.

    Returns:
        The constructed TString object.
    """
    comptime fmt_str = StaticString(format_string)
    comptime length = _count_encoded_bytes(fmt_str)
    comptime bytes = _encoded_bytes[length](fmt_str)

    ref global_bytes = global_constant[bytes]()
    tstring = {
        pack = rebind_var[type_of(tstring)._InjectedValues](args.copy()),
        encoded = Pointer(to=global_bytes).unsafe_bitcast[Byte](),
    }


def _count_encoded_bytes(format: StaticString) -> Int:
    var count = 0

    def addone(_byte: Byte) {mut}:
        count += 1

    _encode_format_string(format, addone)
    return count


def _encoded_bytes[
    length: Int
](format: StaticString) -> StaticTuple[Byte, length]:
    var bytes = StaticTuple[Byte, length](fill=0)
    var index = 0

    def append(byte: Byte) {mut}:
        bytes[index] = byte
        index += 1

    _encode_format_string(format, append)
    return bytes


def _encode_format_string(format: StaticString, f: Some[def(Byte)]):
    """Encode a format string into a flat byte sequence.

    The output is an alternating sequence of NUL-terminated literal segments
    and replacement field boundaries. For N replacement fields, there are
    always N+1 literal segments.

    The replacement fields themselves are not stored — their positions are
    implied by the NUL boundaries. Escaped braces (`{{`/`}}`) are resolved
    to `{`/`}` in the literal text.

    If the format string starts with `{}`, the first literal segment is
    empty (a bare NUL byte). Likewise if the format string ends with `{}`,
    the last literal segment is empty. This means the output always begins
    and ends with a (possibly empty) NUL-terminated literal segment.

    For example, `"result: {} + {} = {}"` encodes as
    `"result: \0 + \0 = \0\0"`, which we walks through as:

        1. literal: "result: \0"
        2. arg: 0
        3. literal: " + \0"
        4. arg: 1
        5. literal: " = \0"
        6. arg: 2
        7. literal: "\0"      (empty — format ends with {})

    At runtime, the we write bytes until NUL to get a literal
    segment, writes the next interpolated argument, and repeats until
    the final literal segment (which has no argument after it).

    Args:
        format: The format string to encode.
        f: Called once per encoded byte, in order, instead of returning a
            buffer directly. This lets callers collect the bytes however
            they need to, for example counting them or writing them into a
            fixed-size buffer.
    """
    comptime LBRACE = Byte(123)  # '{'
    comptime RBRACE = Byte(125)  # '}'
    comptime NUL = Byte(0)

    var bytes = format.as_bytes()
    var i = 0

    # Note: using `bytes.unsafe_ptr()[unsafe_offset=i]` is intentional over
    # using `bytes[i]` as it puts less stress on the comptime interpreter
    # resulting in better compile times.

    @always_inline
    def peek_next_is(byte: Byte) {imm} -> Bool:
        return (
            i + 1 < len(bytes)
            and bytes.unsafe_ptr()[unsafe_offset=i + 1] == byte
        )

    while i < len(bytes):
        var byte = bytes.unsafe_ptr()[unsafe_offset=i]
        if byte == LBRACE:
            if peek_next_is(LBRACE):
                # Escaped brace {{ -> {
                f(LBRACE)
            elif peek_next_is(RBRACE):
                # Empty replacement field {} -> NUL separator.
                f(NUL)
            else:
                abort()

            # skip past escaped brace or replacement field
            i += 2
        elif byte == RBRACE:
            if not peek_next_is(RBRACE):
                abort()

            # Escaped brace }} -> }
            f(RBRACE)
            i += 2
        else:
            f(byte)
            i += 1

    # Terminate the final literal segment with NUL.
    f(NUL)
