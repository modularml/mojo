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
# C reference: c_abi_variadic_prototype.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_variadic
# RUN: %t.dir/test_variadic | FileCheck %s

# Note: Variadic C functions require the numFixedArgs attribute on
# pop.external_call to generate correct LLVM IR with isVarArg=true.


# CHECK: variadic_sum: 63
# Test 1: Simple variadic sum
def test_variadic_ints():
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_sum_ints".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Int,
    ](Int(3), Int(10), Int(20), Int(30))
    print("variadic_sum:", result)


# CHECK: variadic_struct4: 2 3 4 5
# Test 2: Variadic with 4-byte struct
@fieldwise_init
struct Struct4(TrivialRegisterPassable):
    var a: UInt8
    var b: UInt8
    var c: UInt8
    var d: UInt8


def test_variadic_struct4():
    var s = Struct4(1, 2, 3, 4)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_struct4".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Struct4,
    ](Int(999), s)
    print(
        "variadic_struct4:",
        Int(result.a),
        Int(result.b),
        Int(result.c),
        Int(result.d),
    )


# CHECK: variadic_struct16: 101 201
# Test 3: Variadic with 16-byte struct
@fieldwise_init
struct Struct16(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt64


def test_variadic_struct16():
    var s = Struct16(100, 200)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_struct16".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Struct16,
    ](Int(999), s)
    print("variadic_struct16:", Int(result.a), Int(result.b))


# CHECK: variadic_struct17: 11 21 31
# Test 4: Variadic with 17-byte struct
@fieldwise_init
struct Struct17(TrivialRegisterPassable):
    var a: UInt64
    var b: UInt64
    var c: UInt8


def test_variadic_struct17():
    var s = Struct17(10, 20, 30)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_struct17".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Struct17,
    ](Int(999), s)
    print("variadic_struct17:", Int(result.a), Int(result.b), Int(result.c))


# CHECK: variadic_mixed: 32.5
# Test 5: Variadic mixed int and float
def test_variadic_mixed():
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_mixed".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Float64,
    ](Int(2), Int(10), Float64(20.5))
    print("variadic_mixed:", result)


def main():
    test_variadic_ints()
    test_variadic_struct4()
    test_variadic_struct16()
    test_variadic_struct17()
    test_variadic_mixed()
