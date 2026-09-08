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
# Known bug: StringLiteral and String passed to external_call expecting
# const char* receive garbage instead of the string data pointer.
# Uses strlen() from libc — no C reference file needed.
#
# XFAIL: *
# RUN: %mojo %s | FileCheck %s
#
# Bug: The C ABI lowering in LowerPOPToLLVMExternalCalls applies AAPCS
# indirect-passing to 24-byte memoryOnly structs (StringLiteral and String
# both lower to !kgen.struct<(pointer<none>, index, index) memoryOnly>).
# strlen() receives a pointer to the alloca holding the struct instead of the
# string data pointer, returning a garbage value instead of the correct length.
#
# TODO: Remove XFAIL when fixed in MOCO-3375.

from std.ffi import external_call


def test_string_literal():
    # StringLiteral is always null-terminated; its data pointer is a valid const char*.
    var n = external_call["strlen", Int]("hello")
    print("string_literal:", n)


# CHECK: string_literal: 5


def test_string_literal_empty():
    var n = external_call["strlen", Int]("")
    print("string_literal_empty:", n)


# CHECK: string_literal_empty: 0


def test_string():
    # String constructed from a literal has FLAG_HAS_NUL_TERMINATOR set.
    var s = String("hello")
    var n = external_call["strlen", Int](s)
    print("string:", n)


# CHECK: string: 5


def test_string_empty():
    var s = String("")
    var n = external_call["strlen", Int](s)
    print("string_empty:", n)


# CHECK: string_empty: 0


def main():
    test_string_literal()
    test_string_literal_empty()
    test_string()
    test_string_empty()
