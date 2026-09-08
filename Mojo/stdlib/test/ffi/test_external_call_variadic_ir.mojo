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

# Checks that `external_call`'s `num_fixed_args` declares the callee variadic
# with exactly the requested fixed arguments, and that omitting it still
# declares a non-variadic callee. The declaration is what the ABI hangs off, so
# checking it here catches a regression on every target rather than only on the
# ones that pass variadic arguments differently.

# RUN: mkdir -p %t
# RUN: %mojo-build %s -o %t/test_external_call_variadic_ir.ll --emit llvm
# RUN: FileCheck %s --input-file=%t/test_external_call_variadic_ir.ll

from std.ffi import c_char, c_int, external_call
from std.memory import alloc


# char*, size_t and const char* are the fixed arguments; the two arguments
# after them are variadic.
# CHECK-DAG: declare i32 @snprintf(ptr, i64, ptr, ...)
def format_into(buf: Pointer[c_char, _], size: Int) -> c_int:
    var fmt = "%d/%d".as_c_string_slice()
    return external_call["snprintf", c_int, num_fixed_args=3](
        buf, size, fmt.ptr(), c_int(6), c_int(7)
    )


# A `None` count leaves the callee non-variadic.
# CHECK-DAG: declare i64 @strlen(ptr)
def length_of(buf: Pointer[c_char, _]) -> Int:
    return external_call["strlen", Int](buf)


# A count of `0` declares a callee whose every argument is variadic, which a
# `None` count cannot express.
# CHECK-DAG: declare void @mojo_test_all_variadic(...)
def all_variadic(value: c_int):
    external_call["mojo_test_all_variadic", NoneType, num_fixed_args=0](value)


def main():
    var allocation = alloc[c_char]({count = 16}).into_managed()
    var buf = allocation.unsafe_ptr()
    print(Int(format_into(buf, 16)), length_of(buf))
    # This test stops at LLVM IR, so `mojo_test_all_variadic` needs no
    # definition; the call exists only to force its declaration into the IR.
    all_variadic(c_int(0))
