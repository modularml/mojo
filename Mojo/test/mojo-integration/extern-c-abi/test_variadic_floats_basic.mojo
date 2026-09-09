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
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_variadic_floats_basic
# RUN: %t.dir/test_variadic_floats_basic | FileCheck %s

# Note: Variadic C functions require the numFixedArgs attribute on
# pop.external_call to generate correct LLVM IR with isVarArg=true.
# Without it, float/double args would go in SIMD registers instead of
# on the stack where va_arg reads them (on ARM64).


# CHECK: variadic_floats: 64.5
def test_variadic_floats():
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_floats".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Float64,
    ](Int(3), Float64(10.5), Float64(20.5), Float64(30.5))
    print("variadic_floats:", result)


# CHECK: variadic_doubles: 604.5
def test_variadic_doubles():
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_doubles".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Float64,
    ](Int(3), Float64(100.5), Float64(200.5), Float64(300.5))
    print("variadic_doubles:", result)


# CHECK: variadic_int_float: 32.5
def test_variadic_int_float():
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_int_float".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Float64,
    ](Int(999), Int(10), Float64(20.5))
    print("variadic_int_float:", result)


def main():
    test_variadic_floats()
    test_variadic_doubles()
    test_variadic_int_float()
