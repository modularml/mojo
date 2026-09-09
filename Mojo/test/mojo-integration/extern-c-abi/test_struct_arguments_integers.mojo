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
# C reference: c_abi_test_int_structs.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_integers
# RUN: %t.dir/test_integers | FileCheck %s

from std.ffi import external_call


# ============================================================================
# 1-byte struct (single UInt8)
# ============================================================================
@fieldwise_init
struct Struct1(TrivialRegisterPassable):
    var a: UInt8


def test_1byte():
    var s = Struct1(10)
    var result = external_call["c_func_1byte", Struct1](s)
    print("1byte:", Int(result.a))


# CHECK: 1byte: 11


# ============================================================================
# 2-byte struct (two UInt8)
# ============================================================================
@fieldwise_init
struct Struct2(TrivialRegisterPassable):
    var a: UInt8
    var b: UInt8


def test_2byte():
    var s = Struct2(10, 20)
    var result = external_call["c_func_2byte", Struct2](s)
    print("2byte:", Int(result.a), Int(result.b))


# CHECK: 2byte: 11 21


# ============================================================================
# 4-byte struct (four UInt8)
# ============================================================================
@fieldwise_init
struct Struct4(TrivialRegisterPassable):
    var a: UInt8
    var b: UInt8
    var c: UInt8
    var d: UInt8


def test_4byte():
    var s = Struct4(18, 52, 86, 120)
    var result = external_call["c_func_4byte", Struct4](s)
    print(
        "4byte:",
        Int(result.a),
        Int(result.b),
        Int(result.c),
        Int(result.d),
    )


# CHECK: 4byte: 19 53 87 121


# ============================================================================
# 8-byte struct (single UInt64)
# ============================================================================
@fieldwise_init
struct Struct8(TrivialRegisterPassable):
    var a: UInt64


def test_8byte():
    var s = Struct8(1000)
    var result = external_call["c_func_8byte", Struct8](s)
    print("8byte:", Int(result.a))


# CHECK: 8byte: 1001


# ============================================================================
# 9-byte struct (UInt64 + UInt8)
# ============================================================================
@fieldwise_init
struct Struct9(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt8


def test_9byte():
    var s = Struct9(100, 200)
    var result = external_call["c_func_9byte", Struct9](s)
    print("9byte:", Int(result.a), Int(result.b))


# CHECK: 9byte: 101 201


# ============================================================================
# 12-byte struct (three UInt32)
# ============================================================================
@fieldwise_init
struct Struct12(TrivialRegisterPassable):
    var a: UInt32
    var b: UInt32
    var c: UInt32


def test_12byte():
    var s = Struct12(100, 200, 300)
    var result = external_call["c_func_12byte", Struct12](s)
    print("12byte:", Int(result.a), Int(result.b), Int(result.c))


# CHECK: 12byte: 101 201 301


# ============================================================================
# 16-byte struct (two UInt64)
# ============================================================================
@fieldwise_init
struct Struct16(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt64


def test_16byte():
    var s = Struct16(1000, 2000)
    var result = external_call["c_func_16byte", Struct16](s)
    print("16byte:", Int(result.a), Int(result.b))


# CHECK: 16byte: 1001 2001


# ============================================================================
# 17-byte struct (two UInt64 + UInt8) - MEMORY class
# ============================================================================
@fieldwise_init
struct Struct17(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt64
    var c: UInt8


def test_17byte():
    var s = Struct17(10, 20, 30)
    var result = external_call["c_func_17byte", Struct17](s)
    print("17byte:", Int(result.a), Int(result.b), Int(result.c))


# CHECK: 17byte: 11 21 31


# ============================================================================
# 31-byte struct - MEMORY class
# ============================================================================
@fieldwise_init
struct Struct31(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt64
    var c: UInt64
    var d: UInt32
    var e: UInt16
    var f: UInt8


def test_31byte():
    var s = Struct31(10, 20, 30, 40, 50, 60)
    var result = external_call["c_func_31byte", Struct31](s)
    print(
        "31byte:",
        Int(result.a),
        Int(result.b),
        Int(result.c),
        Int(result.d),
        Int(result.e),
        Int(result.f),
    )


# CHECK: 31byte: 11 21 31 41 51 61


# ============================================================================
# 33-byte struct (four UInt64 + UInt8) - MEMORY class
# ============================================================================
@fieldwise_init
struct Struct33(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt64
    var c: UInt64
    var d: UInt64
    var e: UInt8


def test_33byte():
    var s = Struct33(100, 200, 300, 400, 5)
    var result = external_call["c_func_33byte", Struct33](s)
    print(
        "33byte:",
        Int(result.a),
        Int(result.b),
        Int(result.c),
        Int(result.d),
        Int(result.e),
    )


# CHECK: 33byte: 101 201 301 401 6


# ============================================================================
# Main - run all tests
# ============================================================================
def main():
    # Passing tests
    test_1byte()
    test_2byte()  # INTEGER class coercion - Phase 1
    test_4byte()  # INTEGER class coercion - Phase 1
    test_8byte()  # INTEGER class coercion - Phase 1
    test_9byte()
    test_12byte()  # INTEGER class coercion - Phase 2
    test_16byte()

    test_17byte()
    test_31byte()
    test_33byte()
