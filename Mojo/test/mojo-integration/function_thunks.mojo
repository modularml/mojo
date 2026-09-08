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

# RUN: rm -rf %t.function-thunks
# RUN: mkdir -p %t.function-thunks
# RUN: mojo precompile %S/inputs/func_package_foo -o %t.function-thunks/func_package_foo.mojoc
# RUN: mojo precompile %S/inputs/func_package_bar -o %t.function-thunks/func_package_bar.mojoc
# RUN: kgen-opt %t.function-thunks/func_package_foo.mojoc | FileCheck %s --check-prefix=THUNK
# RUN: kgen-opt %t.function-thunks/func_package_bar.mojoc | FileCheck %s --check-prefix=THUNK
# RUN: kgen-translate -import-mojo %s --mojo-enable-prebuilt-packages -I %t.function-thunks | FileCheck %s
# RUN: mojo doc -Werror %s -o /dev/null -I %t.function-thunks

# THUNK-COUNT-1: lit.fn @"def

from func_package_foo.module import foo
from func_package_bar.module import bar

# CHECK-LABEL: lit.file_module @function_thunks


def thunk[T: AnyType](x: T):
    pass


# CHECK-LABEL: lit.fn @"test_fn
def test_fn():
    # CHECK: lit.call {{.*}}foo
    _ = foo()
    # CHECK: lit.call {{.*}}bar
    _ = bar()

    # CHECK: kgen.create_closure{{.*}}@"def(::SIMD[DType.int, 1]) thin -> None|def(x: ::SIMD[DType.int, 1]) thin -> None|{{.*}}[def(x: ::SIMD[DType.int, 1]) thin -> None](::SIMD[DType.int, 1])"
    var f: def(Int) thin -> None = thunk[Int]


# The imported `@std` package is loaded from precompiled bytecode.

# CHECK-LABEL: lit.package @std

# CHECK-NOT: lit.fn @"def(::SIMD[DType.int, 1]) thin -> None|def(y: ::SIMD[DType.int, 1]) thin -> None|{{.*}}[def(y: ::SIMD[DType.int, 1]) thin -> None](::SIMD[DType.int, 1])"

# CHECK-LABEL: lit.package @func_package_foo
# CHECK: lit.fn @"foo
# CHECK: kgen.create_closure{{.*}}@"def(::SIMD[DType.int, 1]) thin -> None|def(y: ::SIMD[DType.int, 1]) thin -> None|{{.*}}[def(y: ::SIMD[DType.int, 1]) thin -> None](::SIMD[DType.int, 1])"

# CHECK-COUNT-1: lit.fn @"def(::SIMD[DType.int, 1]) thin -> None|def(y: ::SIMD[DType.int, 1]) thin -> None|{{.*}}[def(y: ::SIMD[DType.int, 1]) thin -> None](::SIMD[DType.int, 1])"

# CHECK-LABEL: lit.package @func_package_bar
# CHECK: lit.fn @"bar
# CHECK: kgen.create_closure{{.*}}@"def(::SIMD[DType.int, 1]) thin -> None|def(y: ::SIMD[DType.int, 1]) thin -> None|{{.*}}[def(y: ::SIMD[DType.int, 1]) thin -> None](::SIMD[DType.int, 1])"

# CHECK-NOT: lit.fn @"def(::Int) thin -> None|def(::Int) thin -> None|{{.*}}[def(::Int) thin -> None](::Int)"
