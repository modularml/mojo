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
# C reference: c_abi_variadic_floats.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_variadic_mixed_structs
# RUN: %t.dir/test_variadic_mixed_structs | FileCheck %s

# Note: Variadic C functions require the numFixedArgs attribute on
# pop.external_call to generate correct LLVM IR with isVarArg=true.


# Test 1: 8-byte mixed int/float struct
@fieldwise_init
struct MixedIntFloat8(TrivialRegisterPassable):
    var i: Int32
    var f: Float32


# CHECK: variadic_mixed_if_8byte: 11 11.5
def test_variadic_mixed_if_8byte():
    var s = MixedIntFloat8(10, 10.5)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_mixed_if_8byte".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=MixedIntFloat8,
    ](Int(999), s)
    print("variadic_mixed_if_8byte:", result.i, result.f)


# Test 2: 16-byte mixed double/int struct
@fieldwise_init
struct MixedDoubleInt16(TrivialRegisterPassable):
    var d: Float64
    var i: Int64


# CHECK: variadic_mixed_di_16byte: 101.5 11
def test_variadic_mixed_di_16byte():
    var s = MixedDoubleInt16(100.5, 10)
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_mixed_di_16byte".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=MixedDoubleInt16,
    ](Int(999), s)
    print("variadic_mixed_di_16byte:", result.d, result.i)


def main():
    test_variadic_mixed_if_8byte()
    test_variadic_mixed_di_16byte()
