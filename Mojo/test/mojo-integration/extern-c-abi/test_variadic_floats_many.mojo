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
# RUN: mojo build -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_variadic_floats_many
# RUN: %t.dir/test_variadic_floats_many | FileCheck %s
# CHECK: 90.0

# Note: Variadic C functions require the numFixedArgs attribute on
# pop.external_call to generate correct LLVM IR with isVarArg=true.


def main():
    # Test: Pass 12 float values to stress SSE register exhaustion
    # On x86_64: first 8 in xmm0-xmm7, remaining 4 on stack
    # C function: sum = sum_of(val + count) for each val, returns sum + count
    # sum = (1+2+...+12) = 78, plus count (12) = 90
    var result = __mlir_op.`pop.external_call`[
        func="c_func_variadic_many_floats".value,
        numFixedArgs=__mlir_attr[`1 : index`],
        _type=Float64,
    ](
        Int(12),
        Float64(1.0),
        Float64(2.0),
        Float64(3.0),
        Float64(4.0),
        Float64(5.0),
        Float64(6.0),
        Float64(7.0),
        Float64(8.0),
        Float64(9.0),
        Float64(10.0),
        Float64(11.0),
        Float64(12.0),
    )
    print(result)
