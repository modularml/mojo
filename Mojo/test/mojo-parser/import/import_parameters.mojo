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

# RUN: %parse-mojo-isolated %s -I=%S/inputs | FileCheck %s

from test_package.module import ParameterizedType


# CHECK-LABEL: lit.fn @"reference_params_through_imported_struct
def reference_params_through_imported_struct():
    # CHECK: kgen.param.constant: !Int = <{:scalar<index> 10}>
    var cached_type: ParameterizedType[10]
    var value = cached_type.value


# CHECK-LABEL: lit.fn @"ref_param_in_arg
# CHECK-SAME: <?, [[X:.*]]: !Int>[
# CHECK-SAME: lit.ref<!lit.struct<#ParameterizedType <:!Int [[X]]>>{{.*}}> byref_result
def ref_param_in_arg(x: ParameterizedType) -> ParameterizedType[x.value]:
    def nested(x: ParameterizedType, y: ParameterizedType[x.value]):
        pass

    # CHECK: lit.alias.decl *"def_type`3":
    # CHECK-SAME: generator<<?, "x.value`2x": !Int>[2]("x":
    # CHECK-SAME: "y": !lit.ref<{{.*}}#ParameterizedType <:!Int *(0,0)>
    comptime def_type: def(
        x: ParameterizedType, y: ParameterizedType[x.value]
    ) thin -> None = nested
    return x
