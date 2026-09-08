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

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

# CHECK-LABEL: lit.alias.decl *"T`0x": meta<!lit.struct<#Tuple <:param_list<!AnyType_Movable>
# CHECK-SAME: [!Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int, !Int]
comptime T = Tuple[*TypeList.splat[Trait=Movable, 10, Int]()]


comptime VA_SIZE[*Ts: AnyType] = Ts.length
# CHECK: lit.alias.decl *"Folded`{{.*}}": !alias_Int1 = <sugar_member_alias{{.*}}rebind(:!Int {:scalar<index> 3})))>
comptime Folded = VA_SIZE[Int, Int, Int]

comptime AddOne[i: Int]: Int = i + 1

# Tabulate: [0, 1, 2, 3, 4] from index identity
comptime TabulateIndices = ParameterList.tabulate[5, AddOne]
# CHECK: lit.alias.decl *"TabulateIndices`{{.*}}:param_list<!Int> [{:scalar<index> 1}, {:scalar<index> 2}, {:scalar<index> 3}, {:scalar<index> 4}, {:scalar<index> 5}]>>

# `TypeList.of` infers the element trait from its arguments, so no explicit
# `Trait=` keyword argument is required.
# CHECK-LABEL: lit.alias.decl *"TLOf{{[^"]*}}": meta<!lit.struct<#TypeList <:meta<!AnyType> !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable
# CHECK-SAME: :param_list<!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable> [!Int, !Bool]
comptime TLOf = TypeList.of[Int, Bool]


# Make sure sugar is elided for `TypeList` values These must bind `<{}>` and not
# a `sugar_builtin(apply(...))` wrapper.
# CHECK-LABEL: lit.alias.decl *"TLOfValue{{[^"]*}}": !lit.struct<#TypeList
# CHECK-SAME: [!Int, !Bool]>> = <{}>
comptime TLOfValue = TypeList.of[Int, Bool]()

# CHECK-LABEL: lit.alias.decl *"TLSplatValue{{[^"]*}}": !lit.struct<#TypeList
# CHECK-SAME: [!Int, !Int, !Int]>> = <{}>
comptime TLSplatValue = TypeList.splat[Trait=Movable, 3, Int]()
