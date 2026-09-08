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

# Regression test for MOCO-4304.


# RUN: %parse-mojo-isolated %s -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -lower-lit -verify-parameters | FileCheck %s

from std.reflection import reflect


# Downcast used as the mapper, can not eager lower to an upcast to type.type during lower-lit.
# Otherwise we will fail the verification.


# CHECK: #[[FIELDS:[a-zA-Z0-9_]+]] = #kgen.type<typevalue<#kgen.genref<@"{{.*}}::Tuple"<:param_list<type> #kgen.param_list.tabulate<#kgen.param_list.size<:param_list<type> #kgen.struct_field_types<T>> : index, #kgen.gen<#kgen.param_list.get<:param_list<type> #kgen.struct_field_types<T>, *(0,0)>> : !kgen.generator<<index>type>>
# CHECK: kgen.struct.generator @"{{.*}}::MetaBuilder"<T: type> = struct_inst<"{{.*}}::MetaBuilder"[T]<:type T>(fields: #[[FIELDS]]) memoryOnly>
struct MetaBuilder[T: AnyType](Movable where False):
    var fields: Tuple[
        *reflect[Self.T].field_types().map[downcast[_, Movable]]()
    ]
