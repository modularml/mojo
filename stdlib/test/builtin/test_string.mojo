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
# RUN: %mojo -debug-level full %s

from builtin.string import (
    _calc_initial_buffer_size_int32,
    _calc_initial_buffer_size_int64,
)
from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from utils import StringRef


@value
struct AString(Stringable):
    fn __str__(borrowed self: Self) -> String:
        return "a string"


fn test_stringable() raises:
    assert_equal("hello", str("hello"))
    assert_equal("0", str(String(0)))
    assert_equal("AAA", str(StringRef("AAA")))
    assert_equal("a string", str(String(AString())))


fn test_constructors() raises:
    # Default construction
    assert_equal(0, len(String()))
    assert_true(not String())

    # Construction from Int
    var s0 = String(0)
    assert_equal("0", str(String(0)))
    assert_equal(1, len(s0))

    var s1 = String(123)
    assert_equal("123", str(String(123)))
    assert_equal(3, len(s1))

    # Construction from StringLiteral
    var s2 = String("abc")
    assert_equal("abc", str(s2))
    assert_equal(3, len(s2))


fn test_copy() raises:
    var s0 = String("find")
    var s1 = String(s0)
    s1._buffer[3] = ord("e")
    assert_equal("find", s0)
    assert_equal("fine", s1)


fn test_equality_operators() raises:
    var s0 = String("abc")
    var s1 = String("def")
    assert_equal(s0, s0)
    assert_not_equal(s0, s1)

    var s2 = String("abc")
    assert_equal(s0, s2)
    # Explicitly invoke eq and ne operators
    assert_true(s0 == s2)
    assert_false(s0 != s2)

    # Is case sensitive
    var s3 = "ABC"
    assert_not_equal(s0, s3)

    # Implicit conversion can promote for eq and ne
    assert_equal(s0, "abc")
    assert_not_equal(s0, "notabc")


fn test_add() raises:
    var s1 = String("123")
    var s2 = String("abc")
    var s3 = s1 + s2
    assert_equal("123abc", s3)

    var s4 = String("x")
    var s5 = s4.join(1, 2, 3)
    assert_equal("1x2x3", s5)

    var s6 = s4.join(s1, s2)
    assert_equal("123xabc", s6)

    var s7 = String()
    assert_equal("abc", s2 + s7)

    assert_equal("abcdef", s2 + "def")
    assert_equal("123abc", "123" + s2)

    var s8 = String("abc is ")
    var s9 = AString()
    assert_equal("abc is a string", s8 + s9)


fn test_string_join() raises:
    var sep = String(",")
    var s0 = String("abc")
    var s1 = sep.join(s0, s0, s0, s0)
    assert_equal("abc,abc,abc,abc", s1)

    assert_equal(sep.join(1, 2, 3), "1,2,3")

    assert_equal(sep.join(1, "abc", 3), "1,abc,3")


fn test_stringref() raises:
    var a = StringRef("AAA")
    var b = StringRef("BBB")
    var c = StringRef("AAA")

    assert_equal(3, len(a))
    assert_equal(3, len(b))
    assert_equal(3, len(c))
    assert_equal(4, len("ABBA"))

    # Equality operators
    assert_not_equal(a, b)
    assert_not_equal(b, a)

    # Self equality
    assert_equal(a, a)

    # Value equality
    assert_equal(a, c)


fn test_stringref_from_dtypepointer() raises:
    var a = StringRef("AAA")
    var b = StringRef(a.data)
    assert_equal(3, len(a))
    assert_equal(3, len(b))
    assert_equal(a, b)


fn test_ord() raises:
    # Regular ASCII
    assert_equal(ord("A"), 65)
    assert_equal(ord("Z"), 90)
    assert_equal(ord("0"), 48)
    assert_equal(ord("9"), 57)
    assert_equal(ord("a"), 97)
    assert_equal(ord("z"), 122)
    assert_equal(ord("!"), 33)

    # FIXME(#26881): Extended ASCII is not yet supported
    # This should be `assert_equal` when extended ASCII is supported
    assert_not_equal(ord("α"), 224)


fn test_chr() raises:
    assert_equal("A", chr(65))
    assert_equal("a", chr(97))
    assert_equal("!", chr(33))


fn test_string_indexing() raises:
    var str = String("Hello Mojo!!")

    assert_equal("H", str[0])
    assert_equal("!", str[-1])
    assert_equal("H", str[-len(str)])
    assert_equal("llo Mojo!!", str[2:])
    assert_equal("lo Mojo!", str[3:-1:1])
    assert_equal("lo Moj", str[3:-3])

    assert_equal("!!ojoM olleH", str[::-1])

    assert_equal("!!ojoM oll", str[2::-1])

    assert_equal("!oo le", str[::-2])

    assert_equal("!jMolH", str[:-1:-2])


fn test_atol() raises:
    assert_equal(375, atol(String("375")))
    assert_equal(1, atol(String("001")))
    assert_equal(-89, atol(String("-89")))

    # Negative cases
    try:
        _ = atol(String("9.03"))
        raise Error("Failed to raise when converting string to integer.")
    except e:
        assert_equal(str(e), "String is not convertible to integer.")

    try:
        _ = atol(String(""))
        raise Error("Failed to raise when converting empty string to integer.")
    except e:
        assert_equal(str(e), "Empty String cannot be converted to integer.")

    try:
        _ = atol(String("9223372036854775832"))
        raise Error(
            "Failed to raise when converting an integer too large to store in"
            " Int."
        )
    except e:
        assert_equal(
            str(e), "String expresses an integer too large to store in Int."
        )


fn test_calc_initial_buffer_size_int32() raises:
    assert_equal(1, _calc_initial_buffer_size_int32(0))
    assert_equal(1, _calc_initial_buffer_size_int32(9))
    assert_equal(2, _calc_initial_buffer_size_int32(10))
    assert_equal(2, _calc_initial_buffer_size_int32(99))
    assert_equal(8, _calc_initial_buffer_size_int32(99999999))
    assert_equal(9, _calc_initial_buffer_size_int32(100000000))
    assert_equal(9, _calc_initial_buffer_size_int32(999999999))
    assert_equal(10, _calc_initial_buffer_size_int32(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int32(4294967295))


fn test_calc_initial_buffer_size_int64() raises:
    assert_equal(1, _calc_initial_buffer_size_int64(0))
    assert_equal(1, _calc_initial_buffer_size_int64(9))
    assert_equal(2, _calc_initial_buffer_size_int64(10))
    assert_equal(2, _calc_initial_buffer_size_int64(99))
    assert_equal(9, _calc_initial_buffer_size_int64(999999999))
    assert_equal(10, _calc_initial_buffer_size_int64(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int64(9999999999))
    assert_equal(11, _calc_initial_buffer_size_int64(10000000000))
    assert_equal(20, _calc_initial_buffer_size_int64(18446744073709551615))


fn test_contains() raises:
    var str = String("Hello world")

    assert_true(str.__contains__(""))
    assert_true(str.__contains__("He"))
    assert_true("lo" in str)
    assert_true(str.__contains__(" "))
    assert_true(str.__contains__("ld"))

    assert_false(str.__contains__("bellow"))
    assert_true("bellow" not in str)


fn test_find() raises:
    var str = String("Hello world")

    assert_equal(0, str.find(""))
    assert_equal(0, str.find("Hello"))
    assert_equal(2, str.find("llo"))
    assert_equal(6, str.find("world"))
    assert_equal(-1, str.find("universe"))

    # Test find() offset is absolute, not relative (issue mojo/#1355)
    var str2 = String("...a")
    assert_equal(3, str2.find("a", 0))
    assert_equal(3, str2.find("a", 1))
    assert_equal(3, str2.find("a", 2))
    assert_equal(3, str2.find("a", 3))

    # Test find() support for negative start positions
    assert_equal(4, str.find("o", -10))
    assert_equal(7, str.find("o", -5))

    assert_equal(-1, String("abc").find("abcd"))


fn test_count() raises:
    var str = String("Hello world")

    assert_equal(12, str.count(""))
    assert_equal(1, str.count("Hell"))
    assert_equal(3, str.count("l"))
    assert_equal(1, str.count("ll"))
    assert_equal(1, str.count("ld"))
    assert_equal(0, str.count("universe"))

    assert_equal(String("aaaaa").count("a"), 5)
    assert_equal(String("aaaaaa").count("aa"), 3)


fn test_replace() raises:
    # Replace empty
    var s1 = String("abc")
    assert_equal("xaxbxc", s1.replace("", "x"))
    assert_equal("->a->b->c", s1.replace("", "->"))

    var s2 = String("Hello Python")
    assert_equal("Hello Mojo", s2.replace("Python", "Mojo"))
    assert_equal("HELLo Python", s2.replace("Hell", "HELL"))
    assert_equal("Hello Python", s2.replace("HELL", "xxx"))
    assert_equal("HellP oython", s2.replace("o P", "P o"))
    assert_equal("Hello Pything", s2.replace("thon", "thing"))
    assert_equal("He||o Python", s2.replace("ll", "||"))
    assert_equal("He--o Python", s2.replace("l", "-"))
    assert_equal("He-x--x-o Python", s2.replace("l", "-x-"))

    var s3 = String("a   complex  test case  with some  spaces")
    assert_equal("a  complex test case with some spaces", s3.replace("  ", " "))


fn test_rfind() raises:
    # Basic usage.
    assert_equal(String("hello world").rfind("world"), 6)
    assert_equal(String("hello world").rfind("bye"), -1)

    # Repeated substrings.
    assert_equal(String("ababab").rfind("ab"), 4)

    # Empty string and substring.
    assert_equal(String("").rfind("ab"), -1)
    assert_equal(String("foo").rfind(""), 3)

    # Test that rfind(start) returned pos is absolute, not relative to specifed
    # start. Also tests positive and negative start offsets.
    assert_equal(String("hello world").rfind("l", 5), 9)
    assert_equal(String("hello world").rfind("l", -5), 9)
    assert_equal(String("hello world").rfind("w", -3), -1)
    assert_equal(String("hello world").rfind("w", -5), 6)

    assert_equal(-1, String("abc").rfind("abcd"))

    # Special characters.
    # TODO(#26444): Support unicode strings.
    # assert_equal(String("こんにちは").rfind("にち"), 2)
    # assert_equal(String("🔥🔥").rfind("🔥"), 1)


fn test_split() raises:
    # Reject empty delimiters
    try:
        _ = String("hello").split("")
        raise Error("failed to reject empty delimiter")
    except e:
        assert_equal(
            "empty delimiter not allowed to be passed to split.", str(e)
        )

    # Split in middle
    var d1 = String("n")
    var in1 = String("faang")
    var res1 = in1.split(d1)
    assert_equal(len(res1), 2)
    assert_equal(res1[0], "faa")
    assert_equal(res1[1], "g")

    # Matches should be properly split in multiple case
    var d2 = String(" ")
    var in2 = String("modcon is coming soon")
    var res2 = in2.split(d2)
    assert_equal(len(res2), 4)
    assert_equal(res2[0], "modcon")
    assert_equal(res2[1], "is")
    assert_equal(res2[2], "coming")
    assert_equal(res2[3], "soon")

    # No match from the delimiter
    var d3 = String("x")
    var in3 = String("hello world")
    var res3 = in3.split(d3)
    assert_equal(len(res3), 1)
    assert_equal(res3[0], "hello world")

    # Multiple character delimiter
    var d4 = String("ll")
    var in4 = String("hello")
    var res4 = in4.split(d4)
    assert_equal(len(res4), 2)
    assert_equal(res4[0], "he")
    assert_equal(res4[1], "o")


fn test_isupper() raises:
    assert_true(isupper(ord("A")))
    assert_true(isupper(ord("B")))
    assert_true(isupper(ord("Y")))
    assert_true(isupper(ord("Z")))

    assert_false(isupper(ord("A") - 1))
    assert_false(isupper(ord("Z") + 1))

    assert_false(isupper(ord("!")))
    assert_false(isupper(ord("0")))


fn test_islower() raises:
    assert_true(islower(ord("a")))
    assert_true(islower(ord("b")))
    assert_true(islower(ord("y")))
    assert_true(islower(ord("z")))

    assert_false(islower(ord("a") - 1))
    assert_false(islower(ord("z") + 1))

    assert_false(islower(ord("!")))
    assert_false(islower(ord("0")))


fn test_lower() raises:
    assert_equal(String("HELLO").lower(), "hello")
    assert_equal(String("hello").lower(), "hello")
    assert_equal(String("FoOBaR").lower(), "foobar")

    assert_equal(String("MOJO🔥").lower(), "mojo🔥")

    # TODO(#26444): Non-ASCII not supported yet
    assert_equal(String("É").lower(), "É")


fn test_upper() raises:
    assert_equal(String("hello").upper(), "HELLO")
    assert_equal(String("HELLO").upper(), "HELLO")
    assert_equal(String("FoOBaR").upper(), "FOOBAR")

    assert_equal(String("mojo🔥").upper(), "MOJO🔥")

    # TODO(#26444): Non-ASCII not supported yet
    assert_equal(String("É").upper(), "É")


fn test_isspace() raises:
    print("checking true cases")
    assert_true(isspace(ord(" ")))
    assert_true(isspace(ord("\n")))
    assert_true(isspace(ord("\t")))
    assert_true(isspace(ord("\r")))
    assert_true(isspace(ord("\v")))
    assert_true(isspace(ord("\f")))

    print("Checking false cases")
    assert_false(isspace(ord("a")))
    assert_false(isspace(ord("u")))
    assert_false(isspace(ord("s")))
    assert_false(isspace(ord("t")))
    assert_false(isspace(ord("i")))
    assert_false(isspace(ord("n")))
    assert_false(isspace(ord("z")))
    assert_false(isspace(ord(".")))


fn test_rstrip() raises:
    var empty_string = String("")
    assert_true(empty_string.rstrip() == "")

    var space_string = String(" \t\n\r\v\f  ")
    assert_true(space_string.rstrip() == "")

    var str0 = String("     n ")
    assert_true(str0.rstrip() == "     n")

    var str1 = String("string")
    assert_true(str1.rstrip() == "string")

    var str2 = String("something \t\n\t\v\f")
    assert_true(str2.rstrip() == "something")


fn test_lstrip() raises:
    var empty_string = String("")
    assert_true(empty_string.lstrip() == "")

    var space_string = String(" \t\n\r\v\f  ")
    assert_true(space_string.lstrip() == "")

    var str0 = String("     n ")
    assert_true(str0.lstrip() == "n ")

    var str1 = String("string")
    assert_true(str1.lstrip() == "string")

    var str2 = String(" \t\n\t\v\fsomething")
    assert_true(str2.lstrip() == "something")


fn test_strip() raises:
    print("start strip")
    var empty_string = String("")
    assert_true(empty_string.strip() == "")

    var space_string = String(" \t\n\r\v\f  ")
    assert_true(space_string.strip() == "")

    var str0 = String("     n ")
    assert_true(str0.strip() == "n")

    var str1 = String("string")
    assert_true(str1.strip() == "string")

    var str2 = String(" \t\n\t\v\fsomething \t\n\t\v\f")
    assert_true(str2.strip() == "something")


fn test_hash() raises:
    fn assert_hash_equals_literal_hash(s: StringLiteral) raises:
        assert_equal(hash(s), hash(String(s)))

    assert_hash_equals_literal_hash("a")
    assert_hash_equals_literal_hash("b")
    assert_hash_equals_literal_hash("c")
    assert_hash_equals_literal_hash("d")
    assert_hash_equals_literal_hash("this is a longer string")
    assert_hash_equals_literal_hash(
        """
Blue: We have to take the amulet to the Banana King.
Charlie: Oh, yes, The Banana King, of course. ABSOLUTELY NOT!
Pink: He, he's counting on us, Charlie! (Pink starts floating) ah...
Blue: If we don't give the amulet to the Banana King, the vortex will open and let out a thousand years of darkness.
Pink: No! Darkness! (Pink is floating in the air)"""
    )


fn test_startswith() raises:
    var str = String("Hello world")

    assert_true(str.startswith("Hello"))
    assert_false(str.startswith("Bye"))

    assert_true(str.startswith("llo", 2))
    assert_true(str.startswith("llo", 2, -1))
    assert_false(str.startswith("llo", 2, 3))


fn test_endswith() raises:
    var str = String("Hello world")

    assert_true(str.endswith(""))
    assert_true(str.endswith("world"))
    assert_true(str.endswith("ld"))
    assert_false(str.endswith("universe"))

    assert_true(str.endswith("ld", 2))
    assert_true(str.endswith("llo", 2, 5))
    assert_false(str.endswith("llo", 2, 3))


def test_removeprefix():
    assert_equal(String("hello world").removeprefix("hello"), " world")
    assert_equal(String("hello world").removeprefix("world"), "hello world")
    assert_equal(String("hello world").removeprefix("hello world"), "")
    assert_equal(String("hello world").removeprefix("llo wor"), "hello world")


def test_removesuffix():
    assert_equal(String("hello world").removesuffix("world"), "hello ")
    assert_equal(String("hello world").removesuffix("hello"), "hello world")
    assert_equal(String("hello world").removesuffix("hello world"), "")
    assert_equal(String("hello world").removesuffix("llo wor"), "hello world")


def test_intable():
    assert_equal(int(String("123")), 123)

    with assert_raises():
        _ = int(String("hi"))


def test_string_mul():
    assert_equal(String("*") * 0, "")
    assert_equal(String("!") * 10, String("!!!!!!!!!!"))
    assert_equal(String("ab") * 5, "ababababab")


def main():
    test_constructors()
    test_copy()
    test_equality_operators()
    test_add()
    test_stringable()
    test_string_join()
    test_stringref()
    test_stringref_from_dtypepointer()
    test_ord()
    test_chr()
    test_string_indexing()
    test_atol()
    test_calc_initial_buffer_size_int32()
    test_calc_initial_buffer_size_int64()
    test_contains()
    test_find()
    test_count()
    test_replace()
    test_rfind()
    test_split()
    test_isupper()
    test_islower()
    test_lower()
    test_upper()
    test_isspace()
    test_rstrip()
    test_lstrip()
    test_strip()
    test_hash()
    test_startswith()
    test_endswith()
    test_removeprefix()
    test_removesuffix()
    test_intable()
    test_string_mul()
