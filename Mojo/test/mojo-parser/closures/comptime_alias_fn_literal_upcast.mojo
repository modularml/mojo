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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

# When a parameter slot is named via a comptime alias of a closure trait,
# passing a function literal should still inflate to the closure-wrapper struct
# on the caller side, while the parameter declaration retains the alias sugar.


comptime CallbackType = def(Int) -> Int


# The parameter slot keeps the alias sugar - it is not unwrapped to the bare
# closure trait at the declaration site.

# CHECK: lit.alias.decl *"CallbackType`0x":
# CHECK-SAME: meta<!{{[A-Za-z0-9_]+}}> = <!{{[A-Za-z0-9_]+}}>

# CHECK: lit.fn @"repro[def(Int) -> Int & ::AnyType & ::Deinitable & ::Movable]($0)"<F: !alias_CallbackType{{[0-9]+}}>
def repro[F: CallbackType](callback: F) -> Int:
    return callback(0)


def wrap_cb(a: Int) -> Int:
    return a + 1


# On the caller side the function literal is inflated into its closure-
# wrapper struct: __init__() constructs the wrapper, and F is bound to
# the PtrWrapper struct parametrised by the wrap_cb symbol.

# CHECK: lit.fn @"driver()"()
# CHECK: %__call_result_tmp__ = lit.var.decl "__call_result_tmp__" synth
# CHECK-SAME: !lit.ref<!lit.struct<#PtrWrapper
# CHECK: lit.call {{.*}}@"def(a: Int) thin -> Int_PtrWrapper"::@"__init__()"
# CHECK: lit.call {{.*}}@"repro[def(Int) -> Int & ::AnyType & ::Deinitable & ::Movable]($0)"
# CHECK-SAME: <:!alias_CallbackType{{[0-9]+}} #kgen.type<!lit.struct<#PtrWrapper
def driver() -> Int:
    return repro(wrap_cb)
