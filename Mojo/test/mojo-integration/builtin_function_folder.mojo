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

# This test depends on real stdlib types and functions (utils._select, DType
# methods, Int.__ceildiv__, StaticString) to test builtin function folding
# integration. It should remain as an integration test.

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s


from std.utils._select import _select_register_value as select


struct IntT[x: Int](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


struct BoolT[x: Bool](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


##===----------------------------------------------------------------------===##
# Fold select op
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"fold_select_op
def fold_select_op[B: Int = 4, C: Int = 3]() -> IntT[B]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#IntT <:!Int B>>, mut *"a`1">
    var a = IntT[select(True, B, C)]()
    return a


##===----------------------------------------------------------------------===##
# Fold Bool ops
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"fold_bool_init
def fold_bool_init() -> BoolT[True]:
    comptime T = Bool(
        mlir_value=__mlir_attr.`#kgen.simd<true> : !kgen.scalar<bool>`
    )
    comptime F = Bool(
        mlir_value=__mlir_attr.`#kgen.simd<false> : !kgen.scalar<bool>`
    )
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#BoolT <:!Bool {:scalar<bool> true}>>, mut *"a`3">
    var a = BoolT[T]()
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<!lit.struct<#BoolT <:!Bool {:scalar<bool> false}>>, mut *"b`4">
    var b = BoolT[F]()
    return a


##===----------------------------------------------------------------------===##
# Fold DType ops
##===----------------------------------------------------------------------===##


struct UInt8T[x: UInt8._mlir_type](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


comptime UI8_139 = __mlir_attr.`#kgen.simd<139> : !kgen.scalar<ui8>`


# CHECK-LABEL: lit.fn @"fold_dtype_as_ui8
def fold_dtype_as_ui8() -> UInt8T[UI8_139]:
    comptime A = DType.int32
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#UInt8T <:scalar<ui8> {{.*}}, 139)>>, mut *"a
    var a = UInt8T[A._as_ui8()]()
    return a


struct DTypeT[x: DType](ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


# CHECK-LABEL: lit.fn @"fold_dtype_from_ui8
def fold_dtype_from_ui8() -> DTypeT[.int32]:
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<!lit.struct<#DTypeT <:!DType {:dtype si32}>>, mut *"a
    var a = DTypeT[._from_ui8(UI8_139)]()
    return a


##===----------------------------------------------------------------------===##
# Ignore 'comptime assert' (MOCO-2839)
##===----------------------------------------------------------------------===##


def fold_constrained_func[x: Int]() -> IntT[x]:
    comptime assert False, "this is being ignored anyway"
    return IntT[x]()


# CHECK-LABEL: lit.fn @"fold_constrained()
def fold_constrained() -> IntT[1]:
    return fold_constrained_func[1]()


##===----------------------------------------------------------------------===##
# Handle comptime statements
##===----------------------------------------------------------------------===##


@always_inline("builtin")
def comptime_statements() -> Bool:
    comptime t = True
    comptime s = not t
    return s


def fold_comptime_statements() -> BoolT[False]:
    var b = BoolT[comptime_statements()]()
    return b
