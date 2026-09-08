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
# Tests that abi("C") function pointers obtained from C at runtime are
# called with the correct C ABI (CallIndirectOpConversion path).
#
# Two struct types are used:
#
#   FloatPair  {float×2}  8B   x86: SSE/f64        arm: HFA-2/Identity
#   BigStruct  {int64×3} 24B   x86: sret/indirect  arm: sret/indirect
#
# FloatPair exercises register coercion (x86-64) / identity (ARM64) and tests
# the conditional function-pointer selection pattern.
# BigStruct exercises the sret+indirect-argument path on both architectures,
# which is the most complex and historically bug-prone code path.
#
# Exhaustive struct-type ABI classification coverage lives in
# KGEN/test/mojo-integration/extern-c-abi, which tests external_call
# (ConvertPOPExternalCall).  This file focuses only on what is unique to the
# abi("C") function-pointer mechanism.
#
# C reference: c_reference.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_effect_reference.lo %s -o %t.dir/test_extern_c_indirect
# RUN: %t.dir/test_extern_c_indirect | FileCheck %s

from std.ffi import external_call


# ============================================================================
# FloatPair: {float, float} (8 bytes)
# x86-64: SSE -> f64   |   ARM64: HFA-2, Identity
# ============================================================================


@fieldwise_init
struct FloatPair(TrivialRegisterPassable):
    var x: Float32
    var y: Float32


def pick_fn(use_add_one: Bool) -> def(FloatPair) thin abi("C") -> FloatPair:
    if use_add_one:
        return external_call[
            "c_get_add_one", def(FloatPair) thin abi("C") -> FloatPair
        ]()
    return external_call[
        "c_get_add_ten", def(FloatPair) thin abi("C") -> FloatPair
    ]()


def test_float_pair_conditional():
    var input = FloatPair(1.5, 2.5)

    var f1 = pick_fn(True)
    var r1 = f1(input)
    print("test_float_pair_conditional -", r1.x, r1.y)

    var f2 = pick_fn(False)
    var r2 = f2(input)
    print("test_float_pair_conditional -", r2.x, r2.y)


# CHECK: test_float_pair_conditional - 2.5 3.5
# CHECK: test_float_pair_conditional - 11.5 12.5


def test_float_pair_direct():
    # Representative of: lib.get_function[def(FloatPair) abi("C") -> FloatPair](...)
    var f = external_call[
        "c_get_add_one", def(FloatPair) thin abi("C") -> FloatPair
    ]()
    var r = f(FloatPair(3.0, 4.0))
    print("test_float_pair_direct -", r.x, r.y)


# CHECK: test_float_pair_direct - 4.0 5.0


# ============================================================================
# BigStruct: {int64_t, int64_t, int64_t} (24 bytes)
# x86-64: size > 16 -> Memory: sret for return, indirect for arg
# ARM64:  size > 16 -> sret for return, indirect for arg
# ============================================================================


@fieldwise_init
struct BigStruct(TrivialRegisterPassable):
    var a: Int64
    var b: Int64
    var c: Int64


def test_big_struct():
    var f = external_call[
        "c_get_big_struct_add_one", def(BigStruct) thin abi("C") -> BigStruct
    ]()
    var r = f(BigStruct(1, 2, 3))
    print("test_big_struct -", r.a, r.b, r.c)


# CHECK: test_big_struct - 2 3 4


def main():
    test_float_pair_conditional()
    test_float_pair_direct()
    test_big_struct()
