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

from std.sys import size_of
from std.utils import Variant


# Compile-time guard: the LLDB Variant formatter
# (Mojo/lib/MojoLLDB/Language/Formatters/MojoVariantTypeFormatter.cpp) reads
# `String`'s three header words positionally when `String` is the active
# arm, because the lowered `!kgen.struct<(pointer, index, index) memoryOnly>`
# shape loses LIT field names. If the stdlib `String` layout changes in a
# way that breaks those assumptions, this test stops building and the
# formatter needs to be updated in lockstep. See MOCO-3787 for the
# principled fix that removes the positional dependency.
def _check_string_layout():
    comptime assert (
        reflect[String].field_count() == 3
    ), "Variant LLDB formatter expects String to have 3 fields"
    comptime names = reflect[String].field_names()
    comptime assert (
        names[0] == "_ptr_or_data"
    ), "Variant LLDB formatter expects String field 0 to be `_ptr_or_data`"
    comptime assert (
        names[1] == "_len_or_data"
    ), "Variant LLDB formatter expects String field 1 to be `_len_or_data`"
    comptime assert (
        names[2] == "_capacity_or_data"
    ), "Variant LLDB formatter expects String field 2 to be `_capacity_or_data`"
    comptime assert (
        size_of[String]() == 24
    ), "Variant LLDB formatter assumes `size_of[String]() == 24`"
    # Offsets pin field ordering: a future reordering pass that kept the
    # names/count/size but shuffled fields would otherwise silently feed the
    # formatter the wrong words.
    comptime assert (
        reflect[String].field_offset[name="_ptr_or_data"]() == 0
    ), "Variant LLDB formatter expects `_ptr_or_data` at offset 0"
    comptime assert (
        reflect[String].field_offset[name="_len_or_data"]() == 8
    ), "Variant LLDB formatter expects `_len_or_data` at offset 8"
    comptime assert (
        reflect[String].field_offset[name="_capacity_or_data"]() == 16
    ), "Variant LLDB formatter expects `_capacity_or_data` at offset 16"


def keep_alive[*Ts: AnyType](*args: *Ts):
    pass


def main():
    _check_string_layout()

    var v = Variant[Int, String](42)

    print("breakpoint1")  # breakpoint

    v.set[String]("hello, world")

    print("breakpoint2")  # breakpoint

    # Exercise the small-string (inline) path. Short strings fit inline in
    # the String header rather than being heap-allocated.
    v.set[String]("hi")

    print("breakpoint3")  # breakpoint

    # Empty string — exercises the heap path's `size == 0` short-circuit.
    v.set[String](String(""))

    print("breakpoint4")  # breakpoint

    # A 3-way variant verifies discriminant > 1 indexing into the union.
    var w = Variant[Int, Bool, String](True)

    print("breakpoint5")  # breakpoint

    # Switch `w` to its last arm (String) — confirms the boundary case
    # `discriminant == numArms - 1` resolves to the correct union child.
    w.set[String]("last arm")

    print("breakpoint6")  # breakpoint

    keep_alive(v, w)
