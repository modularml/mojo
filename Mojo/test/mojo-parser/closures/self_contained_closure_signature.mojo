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
# RUN: %parse-mojo-isolated %s -mlir-print-op-generic | FileCheck %s

# COM: A closure that captures a value whose type depends on a compile-time
# COM: (origin) parameter of the enclosing function embeds that parameter on
# COM: its closure-storage struct. The storage `__call__` method is nested
# COM: under that struct, so its signature references the parent origin param
# COM: by name (`*"x`"`). The bound `__call__` witness specializes the nested
# COM: method through the storage struct's parameters (`<:origin … *"x`">>`).
# COM: The generic op printer is used because the pretty printer renders
# COM: `functionType`, not `funcTypeGenerator`.


struct Thing[a: Origin[mut=True]](TrivialRegisterPassable):
    pass


struct Foo(Movable where False):
    pass


def use(y: Thing):
    pass


def capture_implicit_origin(var x: Foo, y: Thing[origin_of(x)]):
    # The `capture_it` storage struct is parameterized by the hoisted origin
    # `x`. The nested `__call__` witness binds that origin into the method
    # symbol; the method body type keeps the named parent-param ref.
    #
    # CHECK: @"capture_it()`"<:origin<true> *"x`">> : !kgen.generator<!lit.generator<[1](!lit.ref<!lit.struct<{{#[A-Za-z0-9_]+}} <:origin<true> *"x`">>
    def capture_it() {imm y}:
        use(y)
