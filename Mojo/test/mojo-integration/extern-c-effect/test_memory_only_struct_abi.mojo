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
# Tests that a memory-only struct crosses an abi("C") boundary under the
# platform C ABI, as both argument and return value, identically to a
# register-passable struct of the same layout.
#
# The struct layouts match c_reference.c:
#
#   Pair  {float×2}   8 B  x86-64 SysV: SSE      ARM64 AAPCS: HFA-2
#   Big   {int64×3}  24 B  x86-64 SysV: Memory   ARM64 AAPCS: indirect
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_effect_reference.lo %s -o %t.dir/test_memory_only_struct_abi
# RUN: %t.dir/test_memory_only_struct_abi | FileCheck %s


# Neither struct conforms to TrivialRegisterPassable, so both are memory-only.
@fieldwise_init
struct Pair:
    var x: Float32
    var y: Float32


@fieldwise_init
struct Big:
    var a: Int64
    var b: Int64
    var c: Int64


def mojo_add_one(p: Pair) abi("C") -> Pair:
    return Pair(p.x + 1.0, p.y + 1.0)


def mojo_big_add_one(p: Big) abi("C") -> Big:
    return Big(p.a + 1, p.b + 1, p.c + 1)


# Declared rather than called through external_call[], whose return_type
# parameter is bounded RegisterPassable and so cannot name a memory-only struct.
@extern("c_add_one")
def c_add_one(p: Pair) abi("C") -> Pair:
    ...


@extern("c_big_struct_add_one")
def c_big_struct_add_one(p: Big) abi("C") -> Big:
    ...


@extern("c_apply_float_pair_fn")
def c_apply_float_pair_fn(
    fp: def(Pair) thin abi("C") -> Pair, p: Pair
) abi("C") -> Pair:
    ...


@extern("c_apply_big_struct_fn")
def c_apply_big_struct_fn(
    fp: def(Big) thin abi("C") -> Big, p: Big
) abi("C") -> Big:
    ...


# ============================================================================
# Pair: 8 B, classified into registers on both platforms.
# ============================================================================


def test_direct():
    # Both sides are Mojo, so the call site and the definition must agree on
    # the coerced form.
    var r = mojo_add_one(Pair(1.0, 2.0))
    print(r.x, r.y)


# CHECK: 2.0 3.0


def test_through_pointer():
    var fp: def(Pair) thin abi("C") -> Pair = mojo_add_one
    var r = fp(Pair(3.0, 4.0))
    print(r.x, r.y)


# CHECK: 4.0 5.0


def test_mojo_calls_c():
    # Call-site coercion against a real C definition.
    var r = c_add_one(Pair(5.0, 6.0))
    print(r.x, r.y)


# CHECK: 6.0 7.0


def test_c_calls_mojo():
    # Definition-side coercion: C invokes the Mojo-defined abi("C") function,
    # so the Mojo entry block must reconstruct the argument from the C ABI form
    # and return in the class the C ABI assigns.
    var r = c_apply_float_pair_fn(mojo_add_one, Pair(7.0, 8.0))
    print(r.x, r.y)


# CHECK: 8.0 9.0


# ============================================================================
# Big: 24 B, MEMORY class — indirect argument and sret return.
# ============================================================================


def test_direct_big():
    var r = mojo_big_add_one(Big(1, 2, 3))
    print(r.a, r.b, r.c)


# CHECK: 2 3 4


def test_through_pointer_big():
    var fp: def(Big) thin abi("C") -> Big = mojo_big_add_one
    var r = fp(Big(10, 20, 30))
    print(r.a, r.b, r.c)


# CHECK: 11 21 31


def test_mojo_calls_c_big():
    var r = c_big_struct_add_one(Big(40, 50, 60))
    print(r.a, r.b, r.c)


# CHECK: 41 51 61


def test_c_calls_mojo_big():
    # The argument must travel as a byval pointer the callee reloads, and the
    # result through the caller-provided sret slot.
    var r = c_apply_big_struct_fn(mojo_big_add_one, Big(100, 200, 300))
    print(r.a, r.b, r.c)


# CHECK: 101 201 301


def main():
    test_direct()
    test_through_pointer()
    test_mojo_calls_c()
    test_c_calls_mojo()
    test_direct_big()
    test_through_pointer_big()
    test_mojo_calls_c_big()
    test_c_calls_mojo_big()
