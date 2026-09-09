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
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_variadic_mixed_int_struct_and_float
# RUN: %t.dir/test_variadic_mixed_int_struct_and_float | FileCheck %s

# Regression test: when a multi-field struct forces the ABI coercion path,
# a single-field RegisterPassable float struct (flattened to bare f32 at
# the POP level) must still be passed correctly as a variadic argument.
# On ARM64, this requires float→int bitcast to prevent LLVM's float→double
# promotion.


@fieldwise_init
struct IntStruct8(TrivialRegisterPassable):
    var a: Int32
    var b: Int32


@fieldwise_init
struct FloatStruct4(TrivialRegisterPassable):
    var a: Float32


def test_variadic_mixed_int_struct_and_float():
    var s_int = IntStruct8(10, 20)
    var s_float = FloatStruct4(5.5)
    # IntStruct8 forces ABI coercion; FloatStruct4 is flattened to f32
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_mixed_int_struct_and_float".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=FloatStruct4,
    ](Int(999), s_int, s_float)
    print("variadic_mixed_int_struct_and_float:", result.a)


# CHECK: variadic_mixed_int_struct_and_float: 36.5


def main():
    test_variadic_mixed_int_struct_and_float()
