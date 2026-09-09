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

# RUN: %parse-mojo-isolated %s -verify-diagnostics | FileCheck %s


struct MemExample(Movable where False):
    pass


def return_generic_memory_only[T: AnyType]() -> T:
    pass


def fudge_int(x: Int) -> Int:
    return x


# CHECK-LABEL: lit.fn @"var_decls()
def var_decls():
    # CHECK: %y = lit.var.decl "y" var
    var y: Int

    # CHECK: %[[Y:.*]] = lit.ref.load %{{.*}}
    # CHECK: %[[F:.*]] = lit.call {{.*}}::@"fudge_int{{.*}}(%[[Y]])
    # CHECK: lit.ref.store %[[F]], %y
    y = fudge_int(y)

    # CHECK: %z = lit.var.decl {{.*}} : !lit.ref<:meta<!Int> #alias_Int,
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %y
    # CHECK-NEXT: lit.ref.store [[TMP]], %z
    var z = y
    z = 42
    # CHECK-NEXT: [[TMP:%.*]] = kgen.param.constant: !alias_Int1 = <rebind(:!Int {:scalar<index> 42})>
    # CHECK-NEXT: lit.ref.store [[TMP]], %z


# CHECK-LABEL: lit.fn @"test_var_let_scopes
def test_var_let_scopes(cond: Bool):
    # CHECK: lit.var.decl "c"
    # CHECK: hlcf.elif
    var c = 10
    if cond:
        # CHECK: lit.var.decl "c"
        var c = 42
    # CHECK: else
    else:
        # CHECK: lit.var.decl "c"
        var c = 123


# CHECK-LABEL: lit.fn @"test_var_origin_mangling
def test_var_origin_mangling[x: Int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.var.decl "y" var : !lit.ref<!Int, mut *"y`">
        var y = x
    # CHECK: } else {
    else:
        # CHECK: lit.var.decl "y" var : !lit.ref<!Int, mut *"y`1">
        var y = x


# CHECK-LABEL: lit.fn @"test_nested_var_origin_mangling
def test_nested_var_origin_mangling[x: Int](c: Bool):
    # CHECK: hlcf.elif
    if c:
        # CHECK: lit.var.decl "y" var : !lit.ref<!Int, mut *"y`">
        var y = x

    # CHECK: lit.fn *"nested()"
    def nested() capturing:
        # CHECK: lit.var.decl "y" var : !lit.ref<!Int, mut *"y`2x">
        var y = x


# Issue #18157 and issue #18158, shadowing variables should be able to reference
# the shadowed variable on the RHS.
def test_shadowing_reference_shadowed(cond: Bool):
    var num: Int = 10
    if cond:
        var num = fudge_int(42)


# ===----------------------------------------------------------------------=== #
# Explicitly declared mutable variables used across control flow.
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"var_decls_mutable()
def var_decls_mutable() raises -> None:
    # CHECK: %x = lit.var.decl "x" var
    var x = 123

    # CHECK: [[TMP:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 42}>
    # CHECK: [[F:%.*]] = lit.call {{.*}}::@"fudge_int{{.*}}([[TMP]])
    # CHECK: lit.ref.store [[F]], %x
    x = fudge_int(42)


def use_int(x: Int):
    pass


# Check values are declared at top level where they belong.
# https://github.com/modularml/modular/issues/34368


# CHECK-LABEL: lit.fn @"walrus_control_flow
def walrus_control_flow(a: Int) raises:
    # CHECK: %curr = lit.var.decl "curr"
    # CHECK: %b = lit.var.decl "b"
    var curr = a
    var b: Int

    # CHECK: lit.loop {
    # CHECK-NEXT: lit.ref.load %curr
    while b := curr + 1:
        # lit.loop.break.else
        # CHECK: lit.ref.load
        # CHECK: use_int
        use_int(b)
        curr = b


# CHECK-LABEL: lit.fn @"addrSpaces
def addrSpaces[lt1: MutOrigin, lt2: ImmOrigin, as1: AddressSpace]():
    # CHECK: lit.var.decl "ref1" {{.*}}!lit.ref<!MemExample, mut {{.*}}lt1{{.*}}, #lit.struct.extract<:!SIMDLength #lit.struct.extract<:!AddressSpace as1, "_value">, "_mlir_value">>
    var ref1: Pointer[MemExample, lt1, address_space=as1]._mlir_lit_ref

    # CHECK: lit.alias.decl [[AS2:.*]]: !AddressSpace = {{.*}} {42}
    comptime as2: AddressSpace = AddressSpace(42)

    # CHECK: lit.var.decl "ref2" {{.*}}!lit.ref<!MemExample, imm *"lt2._mlir_origin`1", 42>, mut
    var ref2: __mlir_type[
        `!lit.ref<`,
        MemExample,
        `, `,
        lt2._mlir_origin,
        `, `,
        +as2._value._mlir_value,
        `>`,
    ]


# https://github.com/modular/modular/issues/4765
def redundant_var_ref():
    var n = 1
    # expected-warning @+1 {{nested 'var' or 'ref' patterns are redundant, remove the outer pattern}}
    var ref a = n
    # https://github.com/modular/modular/issues/6700
    # expected-warning @+1 {{nested 'var' or 'ref' patterns are redundant, remove the outer pattern}}
    ref var y : Int = n
