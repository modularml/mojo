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

comptime one = __mlir_attr.`1 : index`


# CHECK: lit.fn @"mlirMagicTest{{.*}}(%x: bf16, %y: f8E5M2)
def mlirMagicTest(
    x: __mlir_type.bf16, y: __mlir_type.f8E5M2
) -> __mlir_type.index:
    # CHECK: lit.alias.decl [[A:.*]] = <#alias_one>
    comptime a: __mlir_type.index = one
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<f64, mut
    var b: __mlir_type.f64
    # CHECK: %c = lit.var.decl "c" var : !lit.ref<pointer<pointer<f32>>, mut
    var c: __mlir_type.`!kgen.pointer<!kgen.pointer<f32>>`

    # CHECK: %d = lit.var.decl
    # CHECK: [[TMP:%.*]] = kgen.param.constant: i17 = <4>
    # CHECK: lit.ref.store [[TMP]], %d
    var d = __mlir_attr.`4: i17`

    # CHECK: %dt = lit.var.decl
    # CHECK: [[TMP:%.*]] = kgen.param.constant: dtype = <f32>
    # CHECK: lit.ref.store [[TMP]], %dt
    var dt = __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype `

    # CHECK-NEXT: %idxConstant = lit.var.decl
    # CHECK: kgen.param.constant = <42>
    var idxConstant = __mlir_op.`index.constant`[value=__mlir_attr.`42 : index`]()

    # CHECK: [[TMP:%.*]] = lit.ref.load %idxConstant
    # CHECK: [[TMP2:%.*]] = index.castu [[TMP:%.*]] : index to i1
    var i1Cast = __mlir_op.`index.castu`[_type=__mlir_type.i1](idxConstant)

    # CHECK: lit.alias.decl *"new_lower{{.*}} = <42>
    comptime new_lower = __mlir_attr[
        `#kgen.param.expr<max, `, a, `, `, __mlir_attr.`42 : index`, `> : index`
    ]

    # CHECK: kgen.param.constant = <#alias_new_lower>
    # CHECK-NEXT: kgen.param.constant = <#alias_one>
    # CHECK-NEXT: index.shru
    # CHECK-NEXT: lit.return
    return __mlir_op.`index.shru`(new_lower, one)


# CHECK-LABEL: lit.fn @"mlirTypesAndAttrs{{.*}}()"<dtype: dtype>()
def mlirTypesAndAttrs[dtype: __mlir_type.`!kgen.dtype`]():
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<scalar<dtype>, mut
    var a: __mlir_type[`!kgen.scalar<`, dtype, `>`]
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<simd<4, dtype>,
    var b: __mlir_type[`!kgen.simd<4, `, dtype, `>`]


# Issue #6282: [Lit] Placeholder substitution does not work on nested types
# CHECK-LABEL: lit.struct.decl @ComplexSubstitution<T: dtype>
struct ComplexSubstitution[T: __mlir_type.`!kgen.dtype`](Movable where False):
    # CHECK: lit.struct.field pointer : !kgen.pointer<scalar<T>>
    var pointer: __mlir_type[`!kgen.pointer<!kgen.scalar<`, Self.T, `>>`]


# Issue #6374: [Lit] Add support for type placeholder
# CHECK-LABEL: typePlaceholder
def typePlaceholder():
    # CHECK: %x = lit.var.decl {{.*}} : !lit.ref<param_list<i32>,
    var x: __mlir_type[`!kgen.param_list<`, __mlir_type.i32, `>`]


# CHECK-LABEL: lit.fn @"fancierSubstitutions
def fancierSubstitutions():
    # CHECK: = lit.var.decl {{.*}} : !lit.ref<complex<i32>,
    var complexInt: __mlir_type[`complex<`, __mlir_type.i32, `>`]

    # CHECK: lit.alias.decl [[A:.*]] = <#alias_one>
    comptime a: __mlir_type.index = one
    # CHECK: lit.alias.decl *"new_lower{{.*}}" = <42>
    comptime new_lower = __mlir_attr[
        `#kgen.param.expr<max,`, a, `, `, __mlir_attr.`42 : index`, `> : index`
    ]


# This shows that we can use unary+ to make the printer avoid printing types.
# See Issue #6468: [Lit] __mlir_attr construction fails for !kgen.list
# CHECK-LABEL: @"testAttrConcatWithoutType{{.*}}()"<length>() ->
def testAttrConcatWithoutType[
    length: __mlir_type.index,
]():
    # CHECK: lit.alias.decl *"x{{.*}}": param_list<index> = <[1, length]>
    comptime x = __mlir_attr[
        `#kgen.param_list<`, +one, `,`, length, `> : !kgen.param_list<index>`
    ]


# Show conversion of lvalue address into a pointer.
# Issue #6825: Expose a way to get the address of an lvalue


# CHECK-LABEL: lit.struct.decl @MyPointer<elType: non_struct_type>
struct MyPointer[elType: __mlir_type.`!kgen.non_struct_type`](RegisterPassable):
    comptime StorageTy = __mlir_type[`!kgen.pointer<`, Self.elType, `>`]
    # CHECK: lit.struct.field value : !kgen.param<:non_struct_type sugar_member_alias(!lit.struct<#MyPointer <:non_struct_type elType>>, "StorageTy", pointer<:non_struct_type elType>)>
    var value: Self.StorageTy

    @implicit
    def __init__(out self, value: Self.StorageTy):
        self.value = value


# CHECK-LABEL: lit.fn @"structured_for_loop()"
def structured_for_loop() -> __mlir_type.index:
    # CHECK: %0 = hlcf.loop (%arg0 = %index0 : index) -> index {
    __mlir_region loop_body(i: __mlir_type.index):
        # CHECK-NEXT: %index = kgen.param.constant = <#alias_one>
        # CHECK-NEXT: %1 = index.add %arg0, %index
        # CHECK-NEXT: hlcf.continue %1 : index
        __mlir_op.`hlcf.continue`(__mlir_op.`index.add`(i, one))

    # CHECK: lit.return %0 : index
    return __mlir_op.`hlcf.loop`[
        _type=__mlir_type.index, _region=__mlir_attr.`"loop_body"`
    ](__mlir_attr.`0 : index`)


# CHECK-LABEL: lit.fn @"mlir_properties
def mlir_properties(arg0: __mlir_type.i64, arg1: __mlir_type.i64):
    # CHECK: llvm.add %arg0, %arg1 overflow<nsw> : i64
    _ = __mlir_op.`llvm.add`[
        _type=__mlir_type.i64,
        _properties=__mlir_attr.`{overflowFlags = #llvm.overflow<nsw>}`,
    ](arg0, arg1)
