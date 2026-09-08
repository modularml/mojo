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
# Tests that a Mojo function declared with the abi("C") effect is compiled
# with C ABI at its definition site.
#
# Two struct types exercise different ABI classification classes:
#
#   FloatPair  {float×2}  8 B   x86-64 SysV: SSE, returned packed in XMM0
#                              ARM64 AAPCS: HFA-2, V0/V1
#   BigStruct  {int64×3} 24 B   x86-64 SysV: Memory -> byval arg + sret return
#                              ARM64 AAPCS: indirect arg + sret return
#
# The test verifies three scenarios for each struct:
#
#   1. Direct Mojo call: ConvertKGENCall applies C ABI coercion at the call
#      site; processCABIFunctionDefinition applies it at the definition entry
#      and exit.  Both sides must agree for the result to be correct.
#
#   2. Call through abi("C") pointer: taking &mojo_add_one yields the address
#      of the C-ABI function (modified in place, not a Mojo-ABI impl), so the
#      existing call_indirect coercion path also works correctly.
#
#   3. C invokes Mojo callback: the function pointer is passed to C land via
#      external_call; C calls through it using C ABI, which must match the
#      Mojo-defined function's compiled ABI.
#
# BigStruct's "C invokes Mojo callback" case (test_c_calls_mojo_big) is the
# regression test for MOCO-3939: the non-external branch of
# processCABIFunctionDefinition was missing the llvm.byval argument attribute
# (and matching sret on the return) that the external branch already set,
# so C callers passed a byval pointer that the Mojo callee interpreted as a
# direct value, corrupting the call.
#
# C reference: c_reference.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_effect_reference.lo %s -o %t.dir/test_mojo_defined_extern_c
# RUN: %t.dir/test_mojo_defined_extern_c | FileCheck %s

from std.ffi import external_call


@fieldwise_init
struct FloatPair(TrivialRegisterPassable):
    var x: Float32
    var y: Float32


def mojo_add_one(p: FloatPair) abi("C") -> FloatPair:
    return FloatPair(p.x + 1.0, p.y + 1.0)


def test_direct():
    # Direct Mojo-to-Mojo call through a abi("C") function definition.
    # ConvertKGENCall inserts C ABI coercion at the call site;
    # processCABIFunctionDefinition inserts it at the function entry/exit.
    var r = mojo_add_one(FloatPair(1.0, 2.0))
    print(r.x, r.y)


# CHECK: 2.0 3.0


def test_through_pointer():
    # Taking the address of a abi("C") function gives the C-ABI function's
    # address (in-place rewrite, no rename).  The existing call_indirect
    # coercion path then applies C ABI coercion at the call site.
    var fp: def(FloatPair) thin abi("C") -> FloatPair = mojo_add_one
    var r = fp(FloatPair(3.0, 4.0))
    print(r.x, r.y)


# CHECK: 4.0 5.0


def test_c_calls_mojo():
    # Pass the Mojo-defined abi("C") function to C as a callback.
    # C invokes it using C ABI; the function must have been compiled with C ABI
    # at its definition site for the result to be correct.
    var r = external_call["c_apply_float_pair_fn", FloatPair](
        mojo_add_one, FloatPair(5.0, 6.0)
    )
    print(r.x, r.y)


# CHECK: 6.0 7.0


# ============================================================================
# BigStruct: {int64_t, int64_t, int64_t} (24 bytes, Memory class).
# x86-64 SysV: arg passed indirectly with llvm.byval, return via sret pointer.
# ARM64 AAPCS: arg passed indirectly, return via sret pointer.
# ============================================================================


@fieldwise_init
struct BigStruct(TrivialRegisterPassable):
    var a: Int64
    var b: Int64
    var c: Int64


def mojo_big_add_one(p: BigStruct) abi("C") -> BigStruct:
    return BigStruct(p.a + 1, p.b + 1, p.c + 1)


def test_direct_big():
    # Direct Mojo-to-Mojo call through a abi("C") function definition whose
    # arg and return are Memory-class.  Both the call site and the definition
    # must coerce to byval/sret consistently.
    var r = mojo_big_add_one(BigStruct(1, 2, 3))
    print(r.a, r.b, r.c)


# CHECK: 2 3 4


def test_through_pointer_big():
    # Taking the address of a abi("C") function gives the C-ABI function's
    # address; calling through the pointer goes through the call_indirect
    # coercion path with byval/sret.
    var fp: def(BigStruct) thin abi("C") -> BigStruct = mojo_big_add_one
    var r = fp(BigStruct(10, 20, 30))
    print(r.a, r.b, r.c)


# CHECK: 11 21 31


def test_c_calls_mojo_big():
    # Regression test for MOCO-3939.  Pass the Mojo-defined abi("C") function
    # to C as a callback; C invokes it with a 24-byte struct that must travel
    # via llvm.byval pointer (caller writes, callee reloads).  Before the fix,
    # processCABIFunctionDefinition's non-external branch did not set byval on
    # the block argument, so the callee read garbage instead of the struct.
    var r = external_call["c_apply_big_struct_fn", BigStruct](
        mojo_big_add_one, BigStruct(100, 200, 300)
    )
    print(r.a, r.b, r.c)


# CHECK: 101 201 301


def main():
    test_direct()
    test_through_pointer()
    test_c_calls_mojo()
    test_direct_big()
    test_through_pointer_big()
    test_c_calls_mojo_big()
