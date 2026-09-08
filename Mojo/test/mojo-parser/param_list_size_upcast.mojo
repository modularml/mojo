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


struct InlineArr[length: Int](Movable):
    def __init__[*Ts: Movable](out self: InlineArr[Ts.length]):
        pass


struct ListOf[length: Int](Movable where False):
    var storage: InlineArr[Self.length]

    # CHECK-LABEL: lit.fn @"__init__[KGENParamList[::AnyType & ::Copyable
    #
    # The `arr` local has type `InlineArr[Ts.length]`. The size looks through the
    # upcast, so it prints on the original Copyable pack with no `upcast` wrapper
    # (rather than `param_list.size<:param_list<!AnyType_Movable> upcast(...)>`).
    #
    # CHECK: %arr = lit.var.decl "arr" {{.*}}#InlineArr <:!Int {{.*}}#kgen.param_list.size<:param_list<!AnyType_Copyable_Movable> {{[^>]*}}Ts.values
    def __init__[*Ts: Copyable](var *elts: *Ts, out self: ListOf[Ts.length]):
        var arr = InlineArr.__init__[*Ts]()
        self.storage = arr^
