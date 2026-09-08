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
# parameter-if
##===----------------------------------------------------------------------===##


struct PStruct[*a: Int](Movable where False):
    @always_inline("builtin")
    @staticmethod
    def predicate() -> Bool:
        comptime size = Self.a.size
        comptime result = size == 2
        return result


# CHECK-LABEL: lit.fn @"double_where_clause
# CHECK-SAME: {<
# CHECK-SAME: ::@PStruct::@"predicate()"
# CHECK-SAME: eq(:scalar<index> from_builtin(#kgen.param_list.size<:param_list<!Int> *"x.a.values``">), 2)), #{{[[:alnum:]]+}}>, <
# CHECK-SAME: ::@PStruct::@"predicate()"
# CHECK-SAME: eq(:scalar<index> from_builtin(#kgen.param_list.size<:param_list<!Int> *"y.a.values``3">), 2)), #{{[[:alnum:]]+}}>}
def double_where_clause(
    x: PStruct[...], y: PStruct[...]
) where type_of(x).predicate() where type_of(y).predicate():
    pass


# CHECK-LABEL: lit.fn @"test_nested_double_where_clause
def test_nested_double_where_clause(x: PStruct[...], y: PStruct[...]):
    comptime if type_of(x).predicate():
        comptime if type_of(y).predicate():
            # CHECK: lit.call {{.*}}@"double_where_clause
            double_where_clause(x, y)
