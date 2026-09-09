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
# Tests that @extern with abi("C") calls C library functions with the correct
# C ABI at the call site (ConvertKGENCall path, direct named-symbol calls).
#
# Unlike external_call[], @extern binds a symbol name to a Mojo declaration
# and lets users call it with ordinary Mojo call syntax.  This test verifies
# that the C ABI coercion is applied correctly at the call site when a
# function is declared with both @extern and abi("C").
#
# Two struct types are used:
#
#   FloatPair  {float×2}  8B   x86: SSE/f64        arm: HFA-2/Identity
#   BigStruct  {int64×3} 24B   x86: sret/indirect  arm: sret/indirect
#
# C reference: c_reference.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_effect_reference.lo %s -o %t.dir/test_extern_c_direct
# RUN: %t.dir/test_extern_c_direct | FileCheck %s


@fieldwise_init
struct FloatPair(TrivialRegisterPassable):
    var x: Float32
    var y: Float32


@fieldwise_init
struct BigStruct(TrivialRegisterPassable):
    var a: Int64
    var b: Int64
    var c: Int64


@extern("c_add_one")
def c_add_one(p: FloatPair) abi("C") -> FloatPair:
    ...


@extern("c_add_ten")
def c_add_ten(p: FloatPair) abi("C") -> FloatPair:
    ...


@extern("c_big_struct_add_one")
def c_big_struct_add_one(p: BigStruct) abi("C") -> BigStruct:
    ...


def test_float_pair():
    var r1 = c_add_one(FloatPair(1.0, 2.0))
    print("float_pair add_one -", r1.x, r1.y)

    var r2 = c_add_ten(FloatPair(1.0, 2.0))
    print("float_pair add_ten -", r2.x, r2.y)


# CHECK: float_pair add_one - 2.0 3.0
# CHECK: float_pair add_ten - 11.0 12.0


def test_big_struct():
    var r = c_big_struct_add_one(BigStruct(10, 20, 30))
    print("big_struct -", r.a, r.b, r.c)


# CHECK: big_struct - 11 21 31


def main():
    test_float_pair()
    test_big_struct()
