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


struct Parametric[a: Int](Movable where False):
    pass


struct StructWithDefault[a: Int, b: Int, c: Int = 1, d: Int = 2](Movable where False):
    pass


struct StructWithDefaultKwOnly[a: Int, b: Int, c: Int = 1, *, d: Int = 2](Movable where False):
    pass


struct StructWithVariadic[a: Int = 1, *b: Int](Movable where False):
    pass


struct DefaultPosOnly[a: Int = 1, /, b: Int = 2, *, c: Int = 3](Movable where False):
    pass


def variadic_params[*a: Int]():
    pass


# CHECK-LABEL: lit.fn @"test_unbound_pack
def test_unbound_pack():
    # CHECK: lit.alias.decl *"all_unbound`": meta<!lit.struct<#StructWithDefault <:!Int ?, :!Int ?, :!Int ?, :!Int ?>, <"a": !Int, "b": !Int, "c": !Int = {{.*}}1{{.*}}, "d": !Int = {{.*}}2{{.*}}>>>
    comptime all_unbound = StructWithDefault[...]

    # CHECK: lit.alias.decl *"first_bound`{{.*}}": meta<!lit.struct<#StructWithDefault <:!Int {:scalar<index> 5}, :!Int ?, :!Int ?, :!Int ?>, <"b": !Int, "c": !Int = {{.*}}1{{.*}}, "d": !Int = {{.*}}2{{.*}}>>>
    comptime first_bound = StructWithDefault[5, ...]

    # CHECK: lit.alias.decl *"last_bound_with_kw`{{.*}}": meta<!lit.struct<#StructWithDefaultKwOnly <:!Int {:scalar<index> 8}, :!Int ?, :!Int ?, :!Int ?>, <"b": !Int, "c": !Int = {{.*}}1{{.*}}, *, "d": !Int = {{.*}}2{{.*}}>>>
    comptime last_bound_with_kw = StructWithDefaultKwOnly[8, d=...]

    # CHECK: lit.alias.decl *"prev_bound_with_kw`{{.*}}: meta<!lit.struct<#StructWithDefaultKwOnly <:!Int {:scalar<index> 8}, :!Int ?, :!Int ?, :!Int ?>, <"b": !Int, "c": !Int = {{.*}}1{{.*}}, *, "d": !Int = {{.*}}2{{.*}}>>>
    comptime prev_bound_with_kw = StructWithDefaultKwOnly[8, ..., d=_]

    # CHECK: lit.alias.decl *"kw_unpacked`{{.*}}: meta<!lit.struct<#StructWithDefaultKwOnly <:!Int ?, :!Int ?, :!Int ?, :!Int ?>, <"a": !Int, "b": !Int, "c": !Int = {{.*}}1{{.*}}, *, "d": !Int = {{.*}}2{{.*}}>>>
    comptime kw_unpacked = StructWithDefaultKwOnly[...]

    # CHECK: lit.alias.decl *"unpack_both{{.*}}: meta<!lit.struct<#DefaultPosOnly <:!Int ?, :!Int ?, :!Int ?>, <"a": !Int = {{.*}}1{{.*}}, |, "b": !Int = {{.*}}2{{.*}}, *, "c": !Int = {{.*}}3{{.*}}>>>
    comptime unpack_both = DefaultPosOnly[...]

    # CHECK: lit.alias.decl *"unpack_ellipsis{{.*}}: meta<!lit.struct<#DefaultPosOnly <:!Int ?, :!Int ?, :!Int ?>, <"a": !Int = {{.*}}1{{.*}}, |, "b": !Int = {{.*}}2{{.*}}, *, "c": !Int = {{.*}}3{{.*}}>>>
    comptime unpack_ellipsis = DefaultPosOnly[...]

    # CHECK: lit.alias.decl *"unbound_variadic`{{.*}}": meta<!lit.struct<#StructWithVariadic <:param_list<!Int> ?, :!Int ?, :!lit.struct<#ParameterList
    comptime unbound_variadic = StructWithVariadic[...]

    # CHECK: lit.alias.decl *"unpack_variadic`{{.*}}": !lit.generator<<"a.values`": param_list<!Int>, {{.*}}pos_vararg>!kgen.func.literal<{{.*}}() -> !kgen.none>
    comptime unpack_variadic = variadic_params[...]

    # CHECK: lit.call {{.*}}variadic_params{{.*}}<:param_list<!Int> []
    unpack_variadic()

    # C_HECK: call {{.*}}variadic_params{{.*}}<:param_list<!Int> []
    # The following code is arguably wrong, it leads to
    # '...' is not allowed in concrete parameter bindings
    #
    # which I think is the right error message that should be reported.
    # variadic_params[...]()


def take_var_pack[*Ts: AnyType](var *values: *Ts):
    pass


# Make sure we can transfer an owned pack.
def pass_var_pack[*Ts: AnyType](var *values: *Ts):
    take_var_pack(*values^)
