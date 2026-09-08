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

# RUN: %parse-mojo-isolated -debug-level full -mlir-print-debuginfo %s | kgen-opt -lower-semantic-cf -check-lifetimes | FileCheck %s

# CheckLifetimes types each debug-info deref from the IR pointee. `var y` is
# ref<Int>, so one deref yields Int. `ref z` is ref<ref<Int>>, so the inner
# deref yields ref<Int> and the outer yields Int.

# CHECK-DAG: #[[Z_INNER:.*]] = #debuginfo.expr.deref<{{.*}}> : !lit.ref<!Int,
# CHECK-DAG: #[[Z_OUTER:.*]] = #debuginfo.expr.deref<#[[Z_INNER]]> : !Int

# CHECK-LABEL: lit.fn @"foo(
# CHECK: lit.var.decl "y" var : !lit.ref<!Int,
# CHECK: debuginfo.value {{.*}} = {{.*}} : !lit.ref<!Int,
# CHECK: lit.var.decl "z" ref : !lit.ref<!lit.ref<!Int,
# CHECK: debuginfo.value {{.*}} #[[Z_OUTER]] = {{.*}} : !lit.ref<!lit.ref<!Int,


def foo(mut x: Int):
    var y = x
    ref z = x
