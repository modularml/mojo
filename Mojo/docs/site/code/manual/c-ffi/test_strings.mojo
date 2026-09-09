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
# test_strings.mojo
# Tests for c-ffi.mdx: "Passing strings", covering Mojo->C conversion,
# C->Mojo conversion via strdup + Span, and string literals as C strings.
from std.ffi import c_char, c_int, c_size_t, external_call
from std.testing import assert_equal, assert_true


# --- Mojo String -> C char* (verified by measuring it with strlen) ---


def test_mojo_string_to_c() raises:
    var name: String = "hello"
    var length = external_call["strlen", c_size_t](name.as_c_string_slice())
    assert_equal(Int(length), 5)
    _ = name


# --- Mojo String -> C char* consumed by a real C function (atoi) ---


def test_atoi() raises:
    var value: String = "42"  # Well-formed ASCII number
    var s_ptr = value.as_c_string_slice().ptr()  # null-terminated char*
    var result: c_int = external_call["atoi", c_int](s_ptr)
    assert_equal(result, c_int(42))
    _ = value


# --- C char* -> Mojo String via strdup + Span ---


def test_c_string_to_mojo() raises:
    var name: String = "Echo"
    var cptr = external_call[
        "strdup", Optional[Pointer[c_char, MutUntrackedOrigin]]
    ](name.as_c_string_slice().ptr())
    assert_true(Bool(cptr))
    var ptr = cptr.value()
    # Measure the C buffer with strlen, not the Mojo string's byte_length.
    var length = external_call["strlen", c_size_t](ptr)
    var span = Span(unsafe_ptr=ptr.unsafe_bitcast[Byte](), length=Int(length))
    assert_equal(String(from_utf8=span), String("Echo"))
    external_call["free", NoneType](ptr.unsafe_bitcast[NoneType]())
    _ = name


# --- String literal -> C string (verified by measuring it with strlen) ---


def test_literal_to_c() raises:
    var length = external_call["strlen", c_size_t](
        "libm.so.6".as_c_string_slice().ptr()
    )
    assert_equal(Int(length), 9)  # "libm.so.6" is 9 bytes


def main() raises:
    test_mojo_string_to_c()
    test_atoi()
    test_c_string_to_mojo()
    test_literal_to_c()
