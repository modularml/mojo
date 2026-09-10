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

comptime _TargetType = __mlir_type.`!kgen.target`

# CHECK: lit.alias.decl *"ParametricTargetWithPlugin
# CHECK-SAME: opaque_plugin = #kgen.param_list<*(0,0)> : !kgen.param_list<!AnyType>
comptime ParametricTargetWithPlugin[T: AnyType]: _TargetType = __mlir_attr[
    `#kgen.target<triple = "", `,
    `arch = "", `,
    `simd_bit_width = 0, `,
    `opaque_plugin = `, TypeList.of[T]().values,
    `> : !kgen.target`,
]
