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

# Regression test for MOCO-3765: calling the same external function more than
# once through `external_call` must not be rejected with "existing function with
# conflicting attributes". Each `external_call` site ends up lowering into the
# same `pop.external_call`, and the lowering pass must treat identical calls as
# matching the already-declared LLVM function.
#
# Scenarios covered:
#   - sret + byval MEMORY-class struct (Struct17/31/33 on x86-64)
#   - two-register coerced struct (Struct12, Struct16)
#   - single-register coerced small struct (Struct4)
#
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_duplicate_call
# RUN: %t.dir/test_duplicate_call | FileCheck %s

from std.ffi import external_call


@fieldwise_init
struct Struct4(TrivialRegisterPassable):
    var a: UInt8
    var b: UInt8
    var c: UInt8
    var d: UInt8


@fieldwise_init
struct Struct12(TrivialRegisterPassable):
    var a: UInt32
    var b: UInt32
    var c: UInt32


@fieldwise_init
struct Struct16(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt64


@fieldwise_init
struct Struct17(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt64
    var c: UInt8


def main():
    # Small single-register struct, called twice.
    var s4a = external_call["c_func_4byte", Struct4](Struct4(0, 1, 2, 3))
    var s4b = external_call["c_func_4byte", Struct4](Struct4(10, 20, 30, 40))
    print("4byte:", Int(s4a.a), Int(s4a.b), Int(s4a.c), Int(s4a.d))
    print("4byte:", Int(s4b.a), Int(s4b.b), Int(s4b.c), Int(s4b.d))
    # CHECK: 4byte: 1 2 3 4
    # CHECK: 4byte: 11 21 31 41

    # Two-register coerced struct, called twice.
    var s12a = external_call["c_func_12byte", Struct12](Struct12(100, 200, 300))
    var s12b = external_call["c_func_12byte", Struct12](Struct12(7, 8, 9))
    print("12byte:", Int(s12a.a), Int(s12a.b), Int(s12a.c))
    print("12byte:", Int(s12b.a), Int(s12b.b), Int(s12b.c))
    # CHECK: 12byte: 101 201 301
    # CHECK: 12byte: 8 9 10

    # Two-register (16-byte) struct, called twice.
    var s16a = external_call["c_func_16byte", Struct16](Struct16(1000, 2000))
    var s16b = external_call["c_func_16byte", Struct16](Struct16(1, 2))
    print("16byte:", Int(s16a.a), Int(s16a.b))
    print("16byte:", Int(s16b.a), Int(s16b.b))
    # CHECK: 16byte: 1001 2001
    # CHECK: 16byte: 2 3

    # MEMORY-class struct: sret return + byval argument (x86-64).
    var s17a = external_call["c_func_17byte", Struct17](Struct17(10, 20, 30))
    var s17b = external_call["c_func_17byte", Struct17](Struct17(100, 200, 50))
    print("17byte:", Int(s17a.a), Int(s17a.b), Int(s17a.c))
    print("17byte:", Int(s17b.a), Int(s17b.b), Int(s17b.c))
    # CHECK: 17byte: 11 21 31
    # CHECK: 17byte: 101 201 51
