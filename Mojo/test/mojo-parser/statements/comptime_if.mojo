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
# RUN: %parse-mojo-isolated %s --verify-diagnostics | FileCheck %s

# ===----------------------------------------------------------------------=== #
# comptime if
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"comptime_if_basic{{.*}}"<a: scalar<bool>>()
def comptime_if_basic[a: __mlir_type.`!kgen.scalar<bool>`]():
    # CHECK: kgen.param.if <a> {
    comptime if a:
        # CHECK: lit.var.decl "inside" var
        var inside: Int
    # CHECK: kgen.param.yield
    # CHECK: }


# CHECK-LABEL: lit.fn @"comptime_if_elif{{.*}}"<a: scalar<bool>, b: !Bool>()
def comptime_if_elif[a: __mlir_type.`!kgen.scalar<bool>`, b: Bool]():
    # CHECK: kgen.param.if <a> {
    comptime if a:
        # CHECK: lit.var.decl "inside_1" var
        var inside_1: Int
    # CHECK: } else {
    # CHECK:     kgen.param.if <#lit.struct.extract<:!Bool b, "_mlir_value">> {
    elif b:
        # CHECK:     lit.var.decl "inside_2" var
        var inside_2: Int
    # CHECK:     kgen.param.yield
    # CHECK:   }
    # CHECK:   kgen.param.yield
    # CHECK: }


# CHECK-LABEL: lit.fn @"comptime_if_else{{.*}}"<a: scalar<bool>>()
def comptime_if_else[a: __mlir_type.`!kgen.scalar<bool>`]():
    # CHECK: kgen.param.if <a> {
    comptime if a:
        # CHECK: lit.var.decl "inside_then" var
        var inside_then: Int
    # CHECK: } else {
    else:
        # CHECK: lit.var.decl "inside_else" var
        var inside_else: Int
    # CHECK: }


# ===----------------------------------------------------------------------=== #
# @parameter if (legacy syntax - same IR as comptime if)
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"param_if{{.*}}"<a: scalar<bool>, b: !Bool>()
def param_if[a: __mlir_type.`!kgen.scalar<bool>`, b: Bool]():
    # CHECK: kgen.param.if <a> {
    comptime if a:
        # CHECK: lit.var.decl "inside_1" var
        var inside_1: Int
    # CHECK: } else {
    # CHECK:     kgen.param.if <#lit.struct.extract<:!Bool b, "_mlir_value">> {
    elif b:
        # CHECK:     lit.var.decl "inside_2" var
        var inside_2: Int
    # CHECK:     kgen.param.yield
    # CHECK:   }
    # CHECK:   kgen.param.yield
    # CHECK: }


# CHECK-LABEL: lit.fn @"param_if_andor_i1{{.*}}"<a: scalar<bool>, b: scalar<bool>>()
def param_if_andor_i1[a: __mlir_type.`!kgen.scalar<bool>`, b: __mlir_type.`!kgen.scalar<bool>`]():
    # CHECK: kgen.param.if <cond(a, b, a)>
    comptime if a and b:
        # CHECK:   lit.var.decl "v" var
        var v: Int
    # CHECK:   kgen.param.yield
    # CHECK: } else {
    # CHECK: kgen.param.if <cond(a, a, b)>
    elif a or b:
        # CHECK:   lit.var.decl "w" var
        var w: Int


# CHECK-LABEL: lit.fn @"param_if_and{{.*}}"<a: !Bool, b: !Bool>()
def param_if_and[a: Bool, b: Bool]():
    # CHECK: kgen.param.if <#lit.struct.extract<:!Bool cond(#lit.struct.extract<:!Bool a, "_mlir_value">, b, a), "_mlir_value">> {
    comptime if a and b:
        # CHECK:   lit.var.decl "v" var
        var v: Int
    # CHECK:   kgen.param.yield
    # CHECK: }
