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

# RUN: %parse-mojo-isolated %s | FileCheck %s

##===----------------------------------------------------------------------===##
# Building TargetParamAttr
##===----------------------------------------------------------------------===##

from std.builtin.stubs import _MLIR

comptime _TargetType = __mlir_type.`!kgen.target`

# CHECK: lit.alias.decl *"ParametricTargetWithIntPlugin
# CHECK-SAME: opaque_plugin = #kgen.param.index.ref<0, 0> : !Int
comptime ParametricTargetWithIntPlugin[T: Int]: _TargetType = __mlir_attr[
    `#kgen.target<triple = "", `,
    `arch = "", `,
    `simd_bit_width = 0, `,
    `opaque_plugin = `, T,
    `> : !kgen.target`,
]

# CHECK: lit.alias.decl *"ParametricTargetWithTypeListPlugin
# CHECK-SAME: opaque_plugin = #kgen.param.index.ref<0, 0> : !kgen.param_list<!AnyType>
comptime ParametricTargetWithTypeListPlugin[Ts: TypeList[Trait=AnyType, ...]]: _TargetType = __mlir_attr[
    `#kgen.target<triple = "", `,
    `arch = "", `,
    `simd_bit_width = 0, `,
    `opaque_plugin = `, Ts.values,
    `> : !kgen.target`,
]

# CHECK: lit.alias.decl *"GetTargetPlugin
# CHECK-SAME: !lit.generator<<"T": !AnyType, "Target": target>!kgen.param<:!AnyType *(0,0)>>
# CHECK-SAME: = <#kgen.gen<#kgen.get_target_plugin<*(0,1)>>>
comptime GetTargetPlugin[T: AnyType, Target: _TargetType]: T = __mlir_attr[
    `#kgen.get_target_plugin<`, Target, `> : `, +T
]

# CHECK: lit.alias.decl *"IntPlugin
# CHECK-SAME: !Int = <{:scalar<index> 56}>
comptime IntPlugin = GetTargetPlugin[Int, ParametricTargetWithIntPlugin[56]]

# CHECK: lit.alias.decl *"ListPlugin
# CHECK-SAME: param_list<!AnyType> = <[!Scalar_si32, !Scalar_ui32, !Bool]>
comptime ListPlugin = GetTargetPlugin[
    _MLIR.KGENParamListType[AnyType],
    ParametricTargetWithTypeListPlugin[TypeList.of[Int32, UInt32, Bool]()]
]
