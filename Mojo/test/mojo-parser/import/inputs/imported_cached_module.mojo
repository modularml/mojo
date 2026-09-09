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

# COM: This file is used to test imported from cached bytecode modules.

comptime StringLiteralAlias = __mlir_attr.`"foobar" : !kgen.string`


trait Trait:
    pass


struct FuncRef[def_type: __mlir_type.`!kgen.non_struct_type`, f: def_type](Movable where False):
    pass


struct FuncRefField(Movable where False):
    var func_ref: FuncRef[def() thin -> None, FuncRefField.foo]

    @staticmethod
    def foo():
        pass
