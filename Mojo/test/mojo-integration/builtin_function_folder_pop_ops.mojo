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

# This test depends on real stdlib types (DType, SIMD, IntLiteral methods) to
# test pop.* MLIR op folding integration. It should remain as an integration
# test.

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s \
# RUN:   | kgen-opt --kgen-print-inline-type-values | FileCheck %s


struct BoolT[x: Bool](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


struct BuiltinBoolT[x: __mlir_type.`!kgen.scalar<bool>`](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


struct DTypeT[x: DType](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


struct BuiltinSI32T[x: __mlir_type.`!kgen.scalar<si32>`](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


##===----------------------------------------------------------------------===##
# Fold pop.cast_from_builtin
##===----------------------------------------------------------------------===##


# 139 = DType.int32
comptime MLIR_UI8_139 = __mlir_attr.`139 : ui8`
comptime POP_UI8_139 = __mlir_attr.`#kgen.simd<139> : !kgen.scalar<ui8>`


# 77 = DType.f8e5m2
comptime MLIR_UI8_77 = __mlir_attr.`77 : ui8`
comptime POP_UI8_77 = __mlir_attr.`#kgen.simd<77> : !kgen.scalar<ui8>`


@always_inline("builtin")
def pop_cast_from_builtin_bool(
    x: __mlir_type.i1,
) -> __mlir_type.`!kgen.scalar<bool>`:
    return __mlir_op.`pop.cast_from_builtin`[
        _type=__mlir_type.`!kgen.scalar<bool>`
    ](x)


# CHECK-LABEL: lit.fn @"fold_pop_cast_from_builtin_bool
def fold_pop_cast_from_builtin_bool() -> (
    BuiltinBoolT[__mlir_attr.`#kgen.simd<true> : !kgen.scalar<bool>`]
):
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#BuiltinBoolT <:scalar<bool>
    var a = BuiltinBoolT[pop_cast_from_builtin_bool(__mlir_attr.`true : i1`)]()
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<!lit.struct<#BuiltinBoolT <:scalar<bool>
    var b = BuiltinBoolT[pop_cast_from_builtin_bool(__mlir_attr.`false : i1`)]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.cast_to_builtin
##===----------------------------------------------------------------------===##


struct UInt8T[x: __mlir_type.ui8](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


@always_inline("builtin")
def pop_cast_to_builtin_ui8(x: UInt8._mlir_type) -> __mlir_type.ui8:
    return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.ui8](x)


# CHECK-LABEL: lit.fn @"fold_pop_cast_to_builtin_ui8
def fold_pop_cast_to_builtin_ui8() -> UInt8T[MLIR_UI8_139]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}}<:ui8 139>>
    var a = UInt8T[pop_cast_to_builtin_ui8(POP_UI8_139)]()
    return a


@always_inline("builtin")
def pop_cast_to_builtin_bool(
    x: __mlir_type.`!kgen.scalar<bool>`,
) -> __mlir_type.i1:
    return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i1](x)


# CHECK-LABEL: lit.fn @"fold_pop_cast_to_builtin_bool
def fold_pop_cast_to_builtin_bool() -> BoolT[True]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}}:scalar<bool> true}>>,
    var a = BoolT[
        pop_cast_to_builtin_bool(
            __mlir_attr.`#kgen.simd<true> : !kgen.scalar<bool>`
        )
    ]()
    # CHECK:  %b = lit.var.decl "b" var : {{.*}}:scalar<bool> false}>>,
    var b = BoolT[
        pop_cast_to_builtin_bool(
            __mlir_attr.`#kgen.simd<false> : !kgen.scalar<bool>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.dtype.from_ui8
##===----------------------------------------------------------------------===##


@always_inline("builtin")
def pop_dtype_from_ui8(ui8: __mlir_type.ui8) -> DType:
    return DType(mlir_value=__mlir_op.`pop.dtype.from_ui8`(ui8))


# CHECK-LABEL: lit.fn @"fold_pop_dtype_from_ui8
def fold_pop_dtype_from_ui8() -> DTypeT[.int32]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} si32}>>,
    var a = DTypeT[pop_dtype_from_ui8(MLIR_UI8_139)]()
    # CHECK: %b = lit.var.decl "b" var : {{.*}} f8e5m2}>>,
    var b = DTypeT[pop_dtype_from_ui8(MLIR_UI8_77)]()


##===----------------------------------------------------------------------===##
# Fold pop.cast
##===----------------------------------------------------------------------===##


@always_inline("builtin")
def pop_cast(
    x: __mlir_type.`!kgen.scalar<si8>`,
) -> __mlir_type.`!kgen.scalar<ui8>`:
    return __mlir_op.`pop.cast`[_type=__mlir_type.`!kgen.scalar<ui8>`](x)


struct POPUInt8T[x: __mlir_type.`!kgen.scalar<ui8>`](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


comptime POP_SI8_N1 = __mlir_attr.`#kgen.simd<-1> : !kgen.scalar<si8>`
comptime POP_UI8_N1 = __mlir_attr.`#kgen.simd<255> : !kgen.scalar<ui8>`


# CHECK-LABEL: lit.fn @"fold_pop_cast
def fold_pop_cast() -> POPUInt8T[POP_UI8_N1]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} 255)>>,
    var a = POPUInt8T[pop_cast(POP_SI8_N1)]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_splat
##===----------------------------------------------------------------------===##


struct POPUInt8x4T[x: __mlir_type.`!kgen.simd<4, ui8>`](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


comptime POP_UI8x4_N1 = __mlir_attr.`#kgen.simd<255, 255, 255, 255> : !kgen.simd<4, ui8>`


@always_inline("builtin")
def pop_simd_splat(
    x: __mlir_type.`!kgen.scalar<ui8>`,
) -> __mlir_type.`!kgen.simd<4, ui8>`:
    return __mlir_op.`pop.simd.splat`[_type=__mlir_type.`!kgen.simd<4, ui8>`](x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_splat
def fold_pop_simd_splat() -> POPUInt8x4T[POP_UI8x4_N1]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPUInt8x4T <:simd<4, ui8> {{.*}} #alias_POP_UI8_N1), 255)>>,
    var a = POPUInt8x4T[pop_simd_splat(POP_UI8_N1)]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_and
##===----------------------------------------------------------------------===##


@always_inline("builtin")
def pop_simd_and(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, ui8>`:
    return __mlir_op.`pop.simd.and`(x, y)


comptime POP_UI8x4_Fold = __mlir_attr.`#kgen.simd<42, 255, 1, 0> : !kgen.simd<4, ui8>`


# CHECK-LABEL: lit.fn @"fold_pop_simd_and
def fold_pop_simd_and() -> POPUInt8x4T[POP_UI8x4_Fold]:
    # CHECK: %a = lit.var.decl "a" {{.*}} <42, 255, 1, 0>
    var a = POPUInt8x4T[
        pop_simd_and(
            __mlir_attr.`#kgen.simd<46, 255, 1, 0> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<43, 255, 1, 1> : !kgen.simd<4, ui8>`,
        )
    ]()
    return a


# CHECK-LABEL: lit.fn @"pop_unresolved_simd_and
@always_inline("builtin")
def pop_unresolved_simd_and[
    dt: DType, n: Int
](x: SIMD[dt, n], y: SIMD[dt, n]) -> SIMD[dt, n]._mlir_type:
    return __mlir_op.`pop.simd.and`(x._mlir_value, y._mlir_value)


##===----------------------------------------------------------------------===##
# Fold pop.simd_xor
##===----------------------------------------------------------------------===##


@always_inline("builtin")
def pop_simd_xor(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, ui8>`:
    return __mlir_op.`pop.simd.xor`(x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_xor
def fold_pop_simd_xor() -> POPUInt8x4T[POP_UI8x4_Fold]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPUInt8x4T <:simd<4, ui8> {{.*}} <9, 15, 1, 1>, <35, 240, 0, 1>), <42, 255, 1, 0>)>>
    var a = POPUInt8x4T[
        pop_simd_xor(
            __mlir_attr.`#kgen.simd<9, 15, 1, 1> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<35, 240, 0, 1> : !kgen.simd<4, ui8>`,
        )
    ]()
    return a


# CHECK-LABEL: lit.fn @"pop_unresolved_simd_xor
@always_inline("builtin")
def pop_unresolved_simd_xor[
    dt: DType, n: Int
](x: SIMD[dt, n], y: SIMD[dt, n]) -> SIMD[dt, n]._mlir_type:
    return __mlir_op.`pop.simd.xor`(x._mlir_value, y._mlir_value)


##===----------------------------------------------------------------------===##
# Fold pop.simd_or
##===----------------------------------------------------------------------===##


@always_inline("builtin")
def pop_simd_or(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, ui8>`:
    return __mlir_op.`pop.simd.or`(x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_or
def fold_pop_simd_or() -> POPUInt8x4T[POP_UI8x4_Fold]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPUInt8x4T <:simd<4, ui8> {{.*}} <8, 15, 1, 0>, <34, 240, 1, 0>), <42, 255, 1, 0>)>>
    var a = POPUInt8x4T[
        pop_simd_or(
            __mlir_attr.`#kgen.simd<8, 15, 1, 0> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<34, 240, 1, 0> : !kgen.simd<4, ui8>`,
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_cmp
##===----------------------------------------------------------------------===##


struct POPBoolx4T[x: __mlir_type.`!kgen.simd<4, bool>`](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


comptime POP_Boolx4_EQ_Fold = __mlir_attr.`#kgen.simd<false, true, true, false> : !kgen.simd<4, bool>`


@always_inline("builtin")
def pop_simd_cmp_eq(
    x: __mlir_type.`!kgen.simd<4, si8>`, y: __mlir_type.`!kgen.simd<4, si8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<eq>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_eq(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<eq>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_ne(
    x: __mlir_type.`!kgen.simd<4, si8>`, y: __mlir_type.`!kgen.simd<4, si8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<ne>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_ne(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<ne>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_ult(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<lt>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_slt(
    x: __mlir_type.`!kgen.simd<4, si8>`, y: __mlir_type.`!kgen.simd<4, si8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<lt>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_ule(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<le>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_sle(
    x: __mlir_type.`!kgen.simd<4, si8>`, y: __mlir_type.`!kgen.simd<4, si8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<le>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_ugt(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<gt>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_sgt(
    x: __mlir_type.`!kgen.simd<4, si8>`, y: __mlir_type.`!kgen.simd<4, si8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<gt>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_uge(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<ge>`](x, y)


@always_inline("builtin")
def pop_simd_cmp_sge(
    x: __mlir_type.`!kgen.simd<4, si8>`, y: __mlir_type.`!kgen.simd<4, si8>`
) -> __mlir_type.`!kgen.simd<4, bool>`:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<ge>`](x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_cmp
def fold_pop_simd_cmp() -> POPBoolx4T[POP_Boolx4_EQ_Fold]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} <true, false, false, false>)>>
    var a = POPBoolx4T[
        pop_simd_cmp_eq(
            __mlir_attr.`#kgen.simd<46, -128, 1, 0> : !kgen.simd<4, si8>`,
            __mlir_attr.`#kgen.simd<46, 127, -1, 1> : !kgen.simd<4, si8>`,
        )
    ]()
    # CHECK: %b = lit.var.decl "b" var : {{.*}} <false, true, true, false>)>>
    var b = POPBoolx4T[
        pop_simd_cmp_eq(
            __mlir_attr.`#kgen.simd<46, 255, 1, 0> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<43, 255, 1, 1> : !kgen.simd<4, ui8>`,
        )
    ]()
    # CHECK: %c = lit.var.decl "c" var : {{.*}} <false, true, true, true>)>>
    var c = POPBoolx4T[
        pop_simd_cmp_ne(
            __mlir_attr.`#kgen.simd<46, -128, 1, 0> : !kgen.simd<4, si8>`,
            __mlir_attr.`#kgen.simd<46, 127, -1, 1> : !kgen.simd<4, si8>`,
        )
    ]()
    # CHECK: %d = lit.var.decl "d" var : {{.*}} <true, false, false, true>)>>
    var d = POPBoolx4T[
        pop_simd_cmp_ne(
            __mlir_attr.`#kgen.simd<46, 255, 1, 0> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<43, 255, 1, 1> : !kgen.simd<4, ui8>`,
        )
    ]()
    # CHECK: %e = lit.var.decl "e" var : {{.*}} <false, true, false, true>)>>
    var e = POPBoolx4T[
        pop_simd_cmp_slt(
            __mlir_attr.`#kgen.simd<46, -128, 1, 0> : !kgen.simd<4, si8>`,
            __mlir_attr.`#kgen.simd<46, 127, -1, 1> : !kgen.simd<4, si8>`,
        )
    ]()
    # CHECK: %f = lit.var.decl "f" var : {{.*}} <false, false, false, true>)>>
    var f = POPBoolx4T[
        pop_simd_cmp_ult(
            __mlir_attr.`#kgen.simd<46, 255, 1, 0> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<43, 255, 1, 1> : !kgen.simd<4, ui8>`,
        )
    ]()
    # CHECK: %g = lit.var.decl "g" var : {{.*}} <true, true, false, true>)>>
    var g = POPBoolx4T[
        pop_simd_cmp_sle(
            __mlir_attr.`#kgen.simd<46, -128, 1, 0> : !kgen.simd<4, si8>`,
            __mlir_attr.`#kgen.simd<46, 127, -1, 1> : !kgen.simd<4, si8>`,
        )
    ]()
    # CHECK: %h = lit.var.decl "h" var : {{.*}} <false, true, true, true>)>>
    var h = POPBoolx4T[
        pop_simd_cmp_ule(
            __mlir_attr.`#kgen.simd<46, 255, 1, 0> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<43, 255, 1, 1> : !kgen.simd<4, ui8>`,
        )
    ]()
    # CHECK: %i = lit.var.decl "i" var : {{.*}} <false, false, true, false>)>>
    var i = POPBoolx4T[
        pop_simd_cmp_sgt(
            __mlir_attr.`#kgen.simd<46, -128, 1, 0> : !kgen.simd<4, si8>`,
            __mlir_attr.`#kgen.simd<46, 127, -1, 1> : !kgen.simd<4, si8>`,
        )
    ]()
    # CHECK: %j = lit.var.decl "j" var : {{.*}} <true, false, false, false>)>>
    var j = POPBoolx4T[
        pop_simd_cmp_ugt(
            __mlir_attr.`#kgen.simd<46, 255, 1, 0> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<43, 255, 1, 1> : !kgen.simd<4, ui8>`,
        )
    ]()
    # CHECK: %k = lit.var.decl "k" var : {{.*}} <true, false, true, false>)>>
    var k = POPBoolx4T[
        pop_simd_cmp_sge(
            __mlir_attr.`#kgen.simd<46, -128, 1, 0> : !kgen.simd<4, si8>`,
            __mlir_attr.`#kgen.simd<46, 127, -1, 1> : !kgen.simd<4, si8>`,
        )
    ]()
    # CHECK: %l = lit.var.decl "l" var : {{.*}} <true, true, true, false>)>>
    var l = POPBoolx4T[
        pop_simd_cmp_uge(
            __mlir_attr.`#kgen.simd<46, 255, 1, 0> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<43, 255, 1, 1> : !kgen.simd<4, ui8>`,
        )
    ]()
    return b


# CHECK-LABEL: lit.fn @"pop_unresolved_simd_cmp_sge
@always_inline("builtin")
def pop_unresolved_simd_cmp_sge[
    dt: DType, n: Int, m: Int
](x: SIMD[dt, n + m], y: SIMD[dt, n + m]) -> SIMD[.bool, n + m]._mlir_type:
    return __mlir_op.`pop.cmp`[pred=__mlir_attr.`#kgen.cmp_pred<ge>`](
        x._mlir_value, y._mlir_value
    )


@always_inline("builtin")
def var_decls[dtype: DType](value: IntLiteral) -> Scalar[dtype]._mlir_type:
    # Convert the IntLiteral to !kgen.simd<si32>
    var si32 = __mlir_attr[
        `#pop.int_literal_convert<`, value.value, `> : !kgen.scalar<si32>`
    ]
    # Convert !kgen.simd<si32> to !kgen.simd<X>
    var s = __mlir_op.`pop.cast`[_type=Scalar[dtype]._mlir_type](si32)
    # Convert !kgen.simd<X> to !kgen.simd<ui8>
    var pop_ui8 = __mlir_op.`pop.cast`[_type=UInt8._mlir_type](si32)
    # Convert !kgen.simd<ui8> to ui8
    var ui8 = __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.ui8](pop_ui8)
    # Convert ui8 to dtype
    var dt = __mlir_op.`pop.dtype.from_ui8`(ui8)
    # Convert dtype to ui8
    var dt_ui8 = __mlir_op.`pop.dtype.to_ui8`(dt)
    # Convert the ui8 back to !kgen.simd<ui8>
    var pop_ui8_2 = __mlir_op.`pop.cast_from_builtin`[_type=UInt8._mlir_type](
        dt_ui8
    )
    # Convert !kgen.simd<ui8> back to !kgen.simd<X>
    var t = __mlir_op.`pop.cast`[_type=Scalar[dtype]._mlir_type](pop_ui8_2)
    # Combine the two
    var u = __mlir_op.`pop.simd.xor`(s, t)
    return u


# CHECK-LABEL: lit.fn @"fold_var_decls
def fold_var_decls() -> (
    BuiltinSI32T[__mlir_attr.`#kgen.simd<0> : !kgen.scalar<si32>`]
):
    # CHECK: %a = lit.var.decl "a" var : {{.*}} 0)>>
    var a = BuiltinSI32T[var_decls[.int32](42)]()
    # CHECK: %b = lit.var.decl "b" var : {{.*}} -256)>>
    var b = BuiltinSI32T[var_decls[.int32](-1)]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_reduce_or
##===----------------------------------------------------------------------===##


@always_inline("builtin")
def pop_simd_reduce_or(
    x: __mlir_type.`!kgen.simd<4, ui8>`,
) -> __mlir_type.`!kgen.scalar<ui8>`:
    return __mlir_op.`pop.simd.reduce_or`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_reduce_or
def fold_pop_simd_reduce_or() -> POPUInt8T[POP_UI8_77]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} <1, 8, 68, 0>), 77)>>
    var a = POPUInt8T[
        pop_simd_reduce_or(
            __mlir_attr.`#kgen.simd<1, 8, 68, 0> : !kgen.simd<4, ui8>`
        )
    ]()
    return a


# CHECK-LABEL: lit.fn @"pop_unresolved_simd_reduce_or
@always_inline("builtin")
def pop_unresolved_simd_reduce_or[
    dt: DType, n: Int
](x: SIMD[dt, n]) -> SIMD[dt, 1]._mlir_type:
    return __mlir_op.`pop.simd.reduce_or`(x._mlir_value)


##===----------------------------------------------------------------------===##
# Fold pop.simd_reduce_and
##===----------------------------------------------------------------------===##


@always_inline("builtin")
def pop_simd_reduce_and(
    x: __mlir_type.`!kgen.simd<4, ui8>`,
) -> __mlir_type.`!kgen.scalar<ui8>`:
    return __mlir_op.`pop.simd.reduce_and`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_reduce_and
def fold_pop_simd_reduce_and() -> POPUInt8T[POP_UI8_77]:
    # CHECK: %a = lit.var.decl "a" var : {{.*}} <79, 93, 207, 221>), 77)>>
    var a = POPUInt8T[
        pop_simd_reduce_and(
            __mlir_attr.`#kgen.simd<79, 93, 207, 221> : !kgen.simd<4, ui8>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold kgen.param.assert
##===----------------------------------------------------------------------===##


def unfoldable_function() -> Bool:
    return False


@always_inline("builtin")
def kgen_assert() -> Bool:
    comptime assert unfoldable_function(), "Ignore this"
    return True


# CHECK-LABEL: lit.fn @"fold_kgen_assert
def fold_kgen_assert() -> BoolT[True]:
    var a = BoolT[kgen_assert()]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_neg
##===----------------------------------------------------------------------===##


struct POPSInt8x4T[x: __mlir_type.`!kgen.simd<4, si8>`](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


comptime POP_SI8x4_Fold = __mlir_attr.`#kgen.simd<42, -120, 1, 0> : !kgen.simd<4, si8>`


@always_inline("builtin")
def pop_simd_neg(
    x: __mlir_type.`!kgen.simd<4, si8>`,
) -> __mlir_type.`!kgen.simd<4, si8>`:
    return __mlir_op.`pop.neg`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_neg
def fold_pop_simd_neg() -> POPSInt8x4T[POP_SI8x4_Fold]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPSInt8x4T <:simd<4, si8> {{.*}} <-42, 120, -1, 0>), <42, -120, 1, 0>)>>
    var a = POPSInt8x4T[
        pop_simd_neg(
            __mlir_attr.`#kgen.simd<-42, 120, -1, 0> : !kgen.simd<4, si8>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_floor
##===----------------------------------------------------------------------===##


struct POPF32x4T[x: __mlir_type.`!kgen.simd<4, f32>`](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


comptime POP_F32x4_Floor = __mlir_attr.`#kgen.simd<"1.0", "-3.0", "0.0", "4.0"> : !kgen.simd<4, f32>`


@always_inline("builtin")
def pop_simd_floor(
    x: __mlir_type.`!kgen.simd<4, f32>`,
) -> __mlir_type.`!kgen.simd<4, f32>`:
    return __mlir_op.`pop.floor`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_floor
def fold_pop_simd_floor() -> POPF32x4T[POP_F32x4_Floor]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPF32x4T <:simd<4, f32> {{.*}} <"1.5", "-2.29999995", "0", "4.9000001">), <"1", "-3", "0", "4">)>>
    var a = POPF32x4T[
        pop_simd_floor(
            __mlir_attr.`#kgen.simd<"1.5", "-2.3", "0.0", "4.9"> : !kgen.simd<4, f32>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_floor (non-float)
##===----------------------------------------------------------------------===##


comptime POP_SI8x4_Floor = __mlir_attr.`#kgen.simd<7, -42, 1, 0> : !kgen.simd<4, si8>`


@always_inline("builtin")
def pop_simd_floor_si8(
    x: __mlir_type.`!kgen.simd<4, si8>`,
) -> __mlir_type.`!kgen.simd<4, si8>`:
    return __mlir_op.`pop.floor`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_floor_si8
def fold_pop_simd_floor_si8() -> POPSInt8x4T[POP_SI8x4_Floor]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPSInt8x4T <:simd<4, si8> {{.*}} <7, -42, 1, 0>), <7, -42, 1, 0>)>>
    var a = POPSInt8x4T[
        pop_simd_floor_si8(
            __mlir_attr.`#kgen.simd<7, -42, 1, 0> : !kgen.simd<4, si8>`
        )
    ]()
    return a


struct POPBool2T[x: __mlir_type.`!kgen.simd<2, bool>`](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


comptime POP_BOOL2_Floor = __mlir_attr.`#kgen.simd<true, false> : !kgen.simd<2, bool>`


@always_inline("builtin")
def pop_simd_floor_bool(
    x: __mlir_type.`!kgen.simd<2, bool>`,
) -> __mlir_type.`!kgen.simd<2, bool>`:
    return __mlir_op.`pop.floor`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_floor_bool
def fold_pop_simd_floor_bool() -> POPBool2T[POP_BOOL2_Floor]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPBool2T <:simd<2, bool> {{.*}} <true, false>), <true, false>)>>
    var a = POPBool2T[
        pop_simd_floor_bool(
            __mlir_attr.`#kgen.simd<true, false> : !kgen.simd<2, bool>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_ceil
##===----------------------------------------------------------------------===##


comptime POP_F32x4_Ceil = __mlir_attr.`#kgen.simd<"2.0", "-2.0", "0.0", "5.0"> : !kgen.simd<4, f32>`


@always_inline("builtin")
def pop_simd_ceil(
    x: __mlir_type.`!kgen.simd<4, f32>`,
) -> __mlir_type.`!kgen.simd<4, f32>`:
    return __mlir_op.`pop.ceil`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_ceil
def fold_pop_simd_ceil() -> POPF32x4T[POP_F32x4_Ceil]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPF32x4T <:simd<4, f32> {{.*}} <"1.5", "-2.29999995", "0", "4.9000001">), <"2", "-2", "0", "5">)>>
    var a = POPF32x4T[
        pop_simd_ceil(
            __mlir_attr.`#kgen.simd<"1.5", "-2.3", "0.0", "4.9"> : !kgen.simd<4, f32>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_ceil (non-float)
##===----------------------------------------------------------------------===##


comptime POP_SI8x4_Ceil = __mlir_attr.`#kgen.simd<7, -42, 1, 0> : !kgen.simd<4, si8>`


@always_inline("builtin")
def pop_simd_ceil_si8(
    x: __mlir_type.`!kgen.simd<4, si8>`,
) -> __mlir_type.`!kgen.simd<4, si8>`:
    return __mlir_op.`pop.ceil`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_ceil_si8
def fold_pop_simd_ceil_si8() -> POPSInt8x4T[POP_SI8x4_Ceil]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPSInt8x4T <:simd<4, si8> {{.*}} <7, -42, 1, 0>), <7, -42, 1, 0>)>>
    var a = POPSInt8x4T[
        pop_simd_ceil_si8(
            __mlir_attr.`#kgen.simd<7, -42, 1, 0> : !kgen.simd<4, si8>`
        )
    ]()
    return a


comptime POP_BOOL2_Ceil = __mlir_attr.`#kgen.simd<true, false> : !kgen.simd<2, bool>`


@always_inline("builtin")
def pop_simd_ceil_bool(
    x: __mlir_type.`!kgen.simd<2, bool>`,
) -> __mlir_type.`!kgen.simd<2, bool>`:
    return __mlir_op.`pop.ceil`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_ceil_bool
def fold_pop_simd_ceil_bool() -> POPBool2T[POP_BOOL2_Ceil]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPBool2T <:simd<2, bool> {{.*}} <true, false>), <true, false>)>>
    var a = POPBool2T[
        pop_simd_ceil_bool(
            __mlir_attr.`#kgen.simd<true, false> : !kgen.simd<2, bool>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_trunc
##===----------------------------------------------------------------------===##


comptime POP_F32x4_Trunc = __mlir_attr.`#kgen.simd<"1.0", "-2.0", "0.0", "4.0"> : !kgen.simd<4, f32>`


@always_inline("builtin")
def pop_simd_trunc(
    x: __mlir_type.`!kgen.simd<4, f32>`,
) -> __mlir_type.`!kgen.simd<4, f32>`:
    return __mlir_op.`pop.trunc`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_trunc
def fold_pop_simd_trunc() -> POPF32x4T[POP_F32x4_Trunc]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPF32x4T <:simd<4, f32> {{.*}} <"1.5", "-2.29999995", "0", "4.9000001">), <"1", "-2", "0", "4">)>>
    var a = POPF32x4T[
        pop_simd_trunc(
            __mlir_attr.`#kgen.simd<"1.5", "-2.3", "0.0", "4.9"> : !kgen.simd<4, f32>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_trunc (non-float)
##===----------------------------------------------------------------------===##


comptime POP_SI8x4_Trunc = __mlir_attr.`#kgen.simd<7, -42, 1, 0> : !kgen.simd<4, si8>`


@always_inline("builtin")
def pop_simd_trunc_si8(
    x: __mlir_type.`!kgen.simd<4, si8>`,
) -> __mlir_type.`!kgen.simd<4, si8>`:
    return __mlir_op.`pop.trunc`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_trunc_si8
def fold_pop_simd_trunc_si8() -> POPSInt8x4T[POP_SI8x4_Trunc]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPSInt8x4T <:simd<4, si8> {{.*}} <7, -42, 1, 0>), <7, -42, 1, 0>)>>
    var a = POPSInt8x4T[
        pop_simd_trunc_si8(
            __mlir_attr.`#kgen.simd<7, -42, 1, 0> : !kgen.simd<4, si8>`
        )
    ]()
    return a


comptime POP_BOOL2_Trunc = __mlir_attr.`#kgen.simd<true, false> : !kgen.simd<2, bool>`


@always_inline("builtin")
def pop_simd_trunc_bool(
    x: __mlir_type.`!kgen.simd<2, bool>`,
) -> __mlir_type.`!kgen.simd<2, bool>`:
    return __mlir_op.`pop.trunc`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_trunc_bool
def fold_pop_simd_trunc_bool() -> POPBool2T[POP_BOOL2_Trunc]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPBool2T <:simd<2, bool> {{.*}} <true, false>), <true, false>)>>
    var a = POPBool2T[
        pop_simd_trunc_bool(
            __mlir_attr.`#kgen.simd<true, false> : !kgen.simd<2, bool>`
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_div
##===----------------------------------------------------------------------===##


comptime POP_F32x4_Div = __mlir_attr.`#kgen.simd<"2.5", "-1.5", "0.0", "2.0"> : !kgen.simd<4, f32>`


@always_inline("builtin")
def pop_simd_div(
    x: __mlir_type.`!kgen.simd<4, f32>`,
    y: __mlir_type.`!kgen.simd<4, f32>`,
) -> __mlir_type.`!kgen.simd<4, f32>`:
    return __mlir_op.`pop.div`(x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_div
def fold_pop_simd_div() -> POPF32x4T[POP_F32x4_Div]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPF32x4T <:simd<4, f32> {{.*}} <"2.5", "-1.5", "0", "2">)>>
    var a = POPF32x4T[
        pop_simd_div(
            __mlir_attr.`#kgen.simd<"5.0", "-3.0", "0.0", "4.0"> : !kgen.simd<4, f32>`,
            __mlir_attr.`#kgen.simd<"2.0", "2.0", "1.0", "2.0"> : !kgen.simd<4, f32>`,
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_floordiv
##===----------------------------------------------------------------------===##


comptime POP_F32x4_FloorDiv = __mlir_attr.`#kgen.simd<"2.0", "-3.0", "0.0", "2.0"> : !kgen.simd<4, f32>`


@always_inline("builtin")
def pop_simd_floordiv(
    x: __mlir_type.`!kgen.simd<4, f32>`,
    y: __mlir_type.`!kgen.simd<4, f32>`,
) -> __mlir_type.`!kgen.simd<4, f32>`:
    return __mlir_op.`pop.floordiv`(x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_floordiv
def fold_pop_simd_floordiv() -> POPF32x4T[POP_F32x4_FloorDiv]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPF32x4T <:simd<4, f32> {{.*}} <"2", "-3", "0", "2">)>>
    var a = POPF32x4T[
        pop_simd_floordiv(
            __mlir_attr.`#kgen.simd<"7.0", "7.0", "0.0", "4.0"> : !kgen.simd<4, f32>`,
            __mlir_attr.`#kgen.simd<"3.0", "-3.0", "1.0", "2.0"> : !kgen.simd<4, f32>`,
        )
    ]()
    return a


##===----------------------------------------------------------------------===##
# Fold pop.simd_shl
##===----------------------------------------------------------------------===##


@always_inline("builtin")
def pop_simd_shl(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, ui8>`:
    return __mlir_op.`pop.shl`(x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_shl
def fold_pop_simd_shl() -> POPUInt8x4T[POP_UI8x4_Fold]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPUInt8x4T <:simd<4, ui8> {{.*}} <21, 255, 1, 192>, <1, 0, 0, 4>), <42, 255, 1, 0>)>>
    var a = POPUInt8x4T[
        pop_simd_shl(
            __mlir_attr.`#kgen.simd<21, 255, 1, 192> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<1, 0, 0, 4> : !kgen.simd<4, ui8>`,
        )
    ]()
    return a


@always_inline("builtin")
def pop_simd_shr(
    x: __mlir_type.`!kgen.simd<4, ui8>`, y: __mlir_type.`!kgen.simd<4, ui8>`
) -> __mlir_type.`!kgen.simd<4, ui8>`:
    return __mlir_op.`pop.shr`(x, y)


# CHECK-LABEL: lit.fn @"fold_pop_simd_shr
def fold_pop_simd_shr() -> POPUInt8x4T[POP_UI8x4_Fold]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPUInt8x4T <:simd<4, ui8> {{.*}} <168, 255, 187, 16>, <2, 0, 7, 5>), <42, 255, 1, 0>)>>
    var a = POPUInt8x4T[
        pop_simd_shr(
            __mlir_attr.`#kgen.simd<168, 255, 187, 16> : !kgen.simd<4, ui8>`,
            __mlir_attr.`#kgen.simd<2, 0, 7, 5> : !kgen.simd<4, ui8>`,
        )
    ]()
    return a


# ===----------------------------------------------------------------------===##
# Fold pop.simd_abs
##===----------------------------------------------------------------------===##
comptime POP_F32x4_Abs = __mlir_attr.`#kgen.simd<"7.0", "7.0", "NaN", "inf"> : !kgen.simd<4, f32>`


@always_inline("builtin")
def pop_simd_abs(
    x: __mlir_type.`!kgen.simd<4, f32>`,
) -> __mlir_type.`!kgen.simd<4, f32>`:
    return __mlir_op.`pop.abs`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_abs
def fold_pop_simd_abs() -> POPF32x4T[POP_F32x4_Abs]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPF32x4T <:simd<4, f32> {{.*}} <"7", "7", "NaN", "+Inf">)>>
    var a = POPF32x4T[
        pop_simd_abs(
            __mlir_attr.`#kgen.simd<"7.0", "7.0", "-NaN", "-Inf"> : !kgen.simd<4, f32>`,
        )
    ]()
    return a


# ===----------------------------------------------------------------------===##
# Fold pop.simd_round
##===----------------------------------------------------------------------===##


struct POPF32x8T[x: __mlir_type.`!kgen.simd<8, f32>`](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


comptime POP_F32x8_Round = __mlir_attr.`#kgen.simd<"1.0", "2.0", "2.0", "-1.0", "-2.0", "-2.0", "4.0", "-4.0"> : !kgen.simd<8, f32>`


@always_inline("builtin")
def pop_simd_round(
    x: __mlir_type.`!kgen.simd<8, f32>`,
) -> __mlir_type.`!kgen.simd<8, f32>`:
    return __mlir_op.`pop.round`(x)


# CHECK-LABEL: lit.fn @"fold_pop_simd_round
def fold_pop_simd_round() -> POPF32x8T[POP_F32x8_Round]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#POPF32x8T <:simd<8, f32> {{.*}} <"1", "2", "2", "-1", "-2", "-2", "4", "-4">)>>
    var a = POPF32x8T[
        pop_simd_round(
            __mlir_attr.`#kgen.simd<"1.1", "1.5", "1.7", "-1.1", "-1.5", "-1.7", "4.5", "-4.5"> : !kgen.simd<8, f32>`,
        )
    ]()
    return a
