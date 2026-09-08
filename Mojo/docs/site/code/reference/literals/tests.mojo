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
# tests.mojo
# Tests for literals.mdx code examples.
# Skip: leading-zero decimal error, empty base-prefix errors,
#        bad-exponent errors, duplicate `greet()` (signature
#        collision), Drawable.draw error pair only, discard
#        pattern `_, y = ...` (tuple-destructuring syntax in
#        flux).
from std.testing import assert_equal
from std.math import sqrt


# --- Integer bases ---


def test_integer_bases() raises:
    assert_equal(0xFF, 255)
    assert_equal(0o52, 42)
    assert_equal(0b101010, 42)
    assert_equal(0xFF, 255)  # lowercase prefix
    assert_equal(0o52, 42)  # uppercase O
    assert_equal(0b101010, 42)  # uppercase B


# --- Integer underscores ---


def test_integer_underscores() raises:
    assert_equal(1_000_000, 1000000)
    var x = 1__000  # cannot test with trailing underscore, kills formatter
    assert_equal(x, 1000)


# --- Float forms ---


def test_float_forms() raises:
    assert_equal(1.0, 1.0)
    assert_equal(0.5, 0.5)
    assert_equal(2.0, 2.0)
    assert_equal(2.5e-3, 0.0025)
    assert_equal(1e10, 1.0e10)
    assert_equal(1_000.000_5, 1000.0005)


# --- String quote styles ---


def test_string_quotes() raises:
    assert_equal("Hello", "Hello")
    var triple = """Multi-line
string
"""
    assert_equal(triple, "Multi-line\nstring\n")
    var triple_single = """Also
    multi-line"""
    assert_equal(triple_single, "Also\n    multi-line")


# --- Adjacent string concatenation ---


def test_string_concat() raises:
    var x = "Hello, World"
    assert_equal(x, "Hello, World")


# --- Raw strings ---


def test_raw_string() raises:
    var p = r"C:\path\to\file"
    assert_equal(p, "C:\\path\\to\\file")


# --- Unicode in source ---


def test_unicode_source() raises:
    var wave = "👋"
    assert_equal(wave.byte_length(), 4)


# --- Unicode escapes ---


def test_unicode_escapes() raises:
    assert_equal("\U0001F44B", "👋")
    assert_equal("\u20AC", "€")
    assert_equal("\u00A9", "©")


# --- Escape sequences ---


def test_escape_sequences() raises:
    assert_equal("\n", "\x0A")
    assert_equal("\t", "\x09")
    assert_equal("\r", "\x0D")
    assert_equal("\\", "\x5C")
    assert_equal('"', "\x22")
    assert_equal("\0", "\x00")
    assert_equal("\101", "A")  # octal 101 = 65 = 'A'


# --- Triple-quoted line continuation ---


def test_triple_line_continuation() raises:
    var x = """\
    No leading newline.\
    """
    assert_equal(x.find("\n"), -1)  # no newline survives


# --- T-string interpolation ---


def test_t_string_basic() raises:
    var name = "World"
    var greeting = String(t"Hello, {name}!")
    assert_equal(greeting, "Hello, World!")
    assert_equal(String(t"1 + 1 = {1 + 1}"), "1 + 1 = 2")


# --- T-string literal braces ---


def test_t_string_braces() raises:
    assert_equal(
        String(t"Use {{braces}} in t-strings"),
        "Use {braces} in t-strings",
    )


# --- Nested t-strings ---


def test_t_string_nested() raises:
    var name = "world"
    var greeting = String(t"Hello, {t"dear {name}"}!")
    assert_equal(greeting, "Hello, dear world!")


# --- Boolean literals ---


def test_boolean() raises:
    var x = True
    var y = False
    assert_equal(x, True)
    assert_equal(y, False)
    assert_equal(x and not y, True)


# --- None literal ---

# Not testable
# def test_none() raises:
#    var x: NoneType = None
#    assert_equal(Bool(x), False)


# --- Self literal ---


@fieldwise_init
struct Point:
    var x: Float64
    var y: Float64

    @staticmethod
    def create() -> Self:
        return Self(0.0, 0.0)

    def distance(self) -> Float64:
        return sqrt(self.x**2 + self.y**2)


def test_self_literal() raises:
    var p = Point.create()
    assert_equal(p.x, 0.0)
    assert_equal(p.y, 0.0)
    var q = Point(3.0, 4.0)
    assert_equal(q.distance(), 5.0)


# --- Ellipsis as required trait method ---


trait Drawable:
    def draw(self) -> None:
        ...


@fieldwise_init
struct Circle(Drawable):
    var radius: Float64

    def draw(self) -> None:
        pass


def test_ellipsis_trait() raises:
    var c = Circle(1.0)
    c.draw()
    assert_equal(c.radius, 1.0)


def main() raises:
    test_integer_bases()
    test_integer_underscores()
    test_float_forms()
    test_string_quotes()
    test_string_concat()
    test_raw_string()
    test_unicode_source()
    test_unicode_escapes()
    test_escape_sequences()
    test_triple_line_continuation()
    test_t_string_basic()
    test_t_string_braces()
    test_t_string_nested()
    test_boolean()
    # test_none()
    test_self_literal()
    test_ellipsis_trait()
