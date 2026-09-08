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

# COM: Run parsing twice to ensure the cache is populated.
# RUN: %parse-mojo-isolated -I=%S/inputs %s -o /dev/null

# CHECK: !Trait = !lit.trait<@imported_cached_module::@Trait>

from imported_cached_module import (
    StringLiteralAlias,
    Trait,
    FuncRefField,
)


# CHECK-LABEL: lit.fn @"assign_from()"
def assign_from():
    # CHECK: string = <"foobar">
    var foo = StringLiteralAlias


# CHECK-LABEL: lit.struct.decl @Struct(!Trait, !AnyType[!Trait])
struct Struct(Trait):
    pass


# CHECK-LABEL: lit.file_module @imported_cached_module
# CHECK: lit.struct.field func_ref : {{.*}}@FuncRefField::@"foo()"
def pull_symbol(x: FuncRefField):
    pass
